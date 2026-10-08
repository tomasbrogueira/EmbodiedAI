"""Mission-bound research routes with conservative continuous footprint checks.

Coordinates here are explicitly assumed metres. This does not certify free
space or gravity and must not replace the calibrated navigation input gate.
"""
from __future__ import annotations

import heapq
import math
import numpy as np
from .planning import TRAVERSABLE


def _local_point(cell, arrays):
    y, x = cell
    return np.r_[arrays['origin'] + (np.array([x, y]) + .5) * arrays['resolution'][0],
                 arrays['support_height'][y, x]]


def _segment_cells(a, b, arrays, radius):
    """Cells touched by the exact horizontal capsule, including out of grid.

    The distance between a segment and a rectangle is zero on intersection;
    otherwise it occurs at a segment endpoint or a rectangle corner.
    """
    res, origin = arrays['resolution'][0], arrays['origin']
    low = np.floor((np.minimum(a[:2], b[:2])-radius-origin)/res).astype(int)
    high = np.floor((np.maximum(a[:2], b[:2])+radius-origin)/res).astype(int)
    if np.prod(high-low+1) > 250_000:
        raise ValueError('Research capsule exceeds bounded cell test count')
    delta = b[:2]-a[:2]
    norm = float(delta @ delta)
    for y in range(low[1], high[1]+1):
        for x in range(low[0], high[0]+1):
            lo = origin + np.array([x,y])*res; hi = lo+res
            t0, t1 = 0., 1.
            for axis in range(2):
                if abs(delta[axis]) < 1e-14:
                    if not lo[axis] <= a[axis] <= hi[axis]: t0, t1 = 1., 0.; break
                else:
                    v = sorted(((lo[axis]-a[axis])/delta[axis], (hi[axis]-a[axis])/delta[axis]))
                    t0, t1 = max(t0,v[0]), min(t1,v[1])
            if t0 <= t1:
                distance = 0.
            else:
                distance = min(np.linalg.norm(p[:2]-np.clip(p[:2],lo,hi)) for p in (a,b))
                for corner in (lo,hi,np.array([lo[0],hi[1]]),np.array([hi[0],lo[1]])):
                    t = 0. if norm == 0 else np.clip((corner-a[:2])@delta/norm,0.,1.)
                    distance = min(distance, np.linalg.norm(corner-a[:2]-t*delta))
            if distance <= radius+1e-12:
                yield int(y),int(x)


def segment_supported(a, b, arrays, metadata):
    """Full swept circle intersects only supported cells and compatible heights."""
    radius = metadata['footprint_radius_with_clearance']
    profile = metadata['profile']; shape = arrays['geometry_state'].shape
    length = np.linalg.norm(b[:2]-a[:2])
    if abs(b[2]-a[2]) > length*math.tan(math.radians(profile['max_slope_degrees']))+1e-8:
        return False
    delta=b[:2]-a[:2]; norm=float(delta@delta)
    for y,x in _segment_cells(a,b,arrays,radius):
        if not (0<=y<shape[0] and 0<=x<shape[1]): return False
        if arrays['geometry_state'][y,x] != TRAVERSABLE or arrays['policy_blocked_mask'][y,x]: return False
        center=arrays['origin']+(np.array([x,y])+.5)*arrays['resolution'][0]
        t=0. if norm==0 else np.clip((center-a[:2])@delta/norm,0.,1.)
        allowance=profile['max_step']+(radius+arrays['resolution'][0]/math.sqrt(2))*math.tan(math.radians(profile['max_slope_degrees']))
        if abs(arrays['support_height'][y,x]-(a[2]+t*(b[2]-a[2]))) > allowance+1e-8: return False
    return True


def forward_corridor_route(arrays, metadata, cameras, rotations, settings, target_points=None):
    """Plan first-to-last forward-view floor targets; never change components.

    Camera poses must be W2C. Target lookahead and maximum endpoint adjustment
    are declared mission assumptions. Failed missions keep their no-path result.
    """
    basis=arrays['projection_basis']; projected=cameras@basis.T
    directions=rotations[:,2]@basis.T
    directions[:,2]=0.; lengths=np.linalg.norm(directions[:,:2],axis=1)
    result={'status':'no_path','reason':None,'path':[],'path_cells':[],
            'mission':{'policy':'recorded_forward_corridor',**settings}}
    targets_local=None if target_points is None else np.asarray(target_points,float)@basis.T
    if targets_local is not None and (targets_local.shape!=(2,3) or not np.isfinite(targets_local).all()):
        raise ValueError('Mission targets must be two finite observed native-map floor points')
    result['mission']['target_source']='first_last_observed_forward_floor_samples' if targets_local is not None else 'assumed_camera_lookahead'
    if np.any(lengths[[0,-1]]<1e-8):
        result['reason']='Camera forward axis has no horizontal direction'; return result
    directions/=np.maximum(lengths[:,None],1e-8)
    progress=float((projected[-1,:2]-projected[0,:2])@directions[0,:2])
    result['mission']['camera_forward_progress_m']=progress
    if progress < settings['minimum_progress_m']:
        result['status']='blocked_inputs';result['reason']='Recorded camera motion does not agree with the forward mission; inspect pose convention';return result
    cells=[tuple(map(int,cell)) for cell in np.argwhere(arrays['decision_state']==TRAVERSABLE)]
    if not cells:
        result['reason']='No observed traversable floor for the requested mission'; return result
    cloud=np.array([_local_point(cell,arrays) for cell in cells])
    chosen=[];targets=[];adjustments=[]
    for number,i in enumerate((0,-1)):
        target=projected[i,:2]+settings['camera_lookahead_m']*directions[i,:2] if targets_local is None else targets_local[number,:2]
        if (target-projected[i,:2])@directions[i,:2]<=0:
            result['reason']='Observed mission target is not ahead of its camera';return result
        distance=np.linalg.norm(cloud[:,:2]-target,axis=1)
        candidates=sorted(range(len(cells)),key=lambda j:(distance[j],cells[j]))
        selected=next((j for j in candidates if distance[j]<=settings['max_endpoint_adjustment_m'] and
                       (targets_local is None or abs(cloud[j,2]-targets_local[number,2])<=metadata['profile']['max_step']+3*metadata['profile']['max_roughness']) and
                       segment_supported(cloud[j],cloud[j],arrays,metadata)),None)
        targets.append(target.tolist())
        if selected is None:
            result['reason']='Requested forward floor endpoint lacks support within the declared adjustment bound'
            result['mission']['target_xy_assumed_metres']=targets;return result
        chosen.append(cells[selected]);adjustments.append(float(distance[selected]))
    start,goal=chosen
    result['mission'].update(target_xy_assumed_metres=targets,endpoint_adjustment_m=adjustments)
    if np.linalg.norm(_local_point(goal,arrays)[:2]-_local_point(start,arrays)[:2])<settings['minimum_progress_m']:
        result['reason']='Forward mission endpoints are too close';return result
    distances={start:0.};parents={};pending=[(0.,*start)];valid=set(cells)
    edge_cache={}
    while pending:
        cost,y,x=heapq.heappop(pending);cell=(y,x)
        if cost>distances[cell]:continue
        if cell==goal:break
        for dy,dx in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
            nxt=(y+dy,x+dx)
            if nxt not in valid:continue
            key=tuple(sorted((cell,nxt)))
            if key not in edge_cache: edge_cache[key]=segment_supported(_local_point(cell,arrays),_local_point(nxt,arrays),arrays,metadata)
            if not edge_cache[key]:continue
            step=np.linalg.norm(_local_point(cell,arrays)-_local_point(nxt,arrays))
            score=cost+step*.5*(arrays['costs'][cell]+arrays['costs'][nxt])
            if score<distances.get(nxt,np.inf):
                distances[nxt]=float(score);parents[nxt]=cell;heapq.heappush(pending,(float(score),*nxt))
    if goal not in distances:
        result['reason']='No continuous supported footprint route connects the requested forward endpoints';return result
    route=[goal]
    while route[-1]!=start:route.append(parents[route[-1]])
    route.reverse()
    # Only remove bends when the entire connecting capsule and heights pass.
    short=[route[0]];i=0
    while i<len(route)-1:
        j=len(route)-1
        while j>i+1 and not segment_supported(_local_point(route[i],arrays),_local_point(route[j],arrays),arrays,metadata):j-=1
        short.append(route[j]);i=j
    path=np.array([_local_point(cell,arrays)@basis for cell in short])
    result.update(status='ok',reason='Requested forward corridor research mission; continuous observed support, unverified free space',
                  path=path.tolist(),path_cells=[list(cell) for cell in route],waypoint_cells=[list(cell) for cell in short],
                  start=path[0].tolist(),goal=path[-1].tolist(),path_cost=distances[goal],
                  swept_footprint_checked=True)
    return result
