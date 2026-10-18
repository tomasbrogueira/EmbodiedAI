"""Reachable camera-relative goals and explicitly connected suffix proposals."""
import numpy as np
import time
from scipy.ndimage import label
from scipy.spatial import cKDTree

def start_node(grid,camera,policy):
    points=grid['points']; basis=grid['basis']; radius=grid.get('radius',grid['cell']*8)
    observed=(grid['point_rows']>=0)&np.isfinite(points).all(1)
    delta=points-camera
    observed &= delta@basis[:,2]<0
    if policy=='visible_ground':
        observed &= np.isfinite(grid['cost'].ravel())
    ids=np.flatnonzero(observed)
    if not len(ids): return None,'no_observed_ground'
    horizontal=np.linalg.norm(delta[ids]@basis[:,:2],axis=1)
    nearest=int(ids[np.argmin(horizontal)])
    if policy=='strict' and horizontal.min()>radius/2: return None,'no_current_ground'
    if not np.isfinite(grid['cost'].ravel()[nearest]): return None,'current_ground_blocked'
    return nearest,'ok'

def propose(grid,camera,forward,search,state,enforce,*,policy='strict',lookahead=12.,budget_ms=40.,exclude_anchors=()):
    """Try useful observed components and validate each proposed raw connector later.

    A demo start may move between observed candidates. Strict ground association
    never skips a blocked nearest sample. All searches share one wall budget.
    """
    empty=lambda reason,**details:dict(status=reason,positions=np.empty((0,3)),ids=np.empty(0,np.int64),anchor=None,diagnostics=details)
    normal=grid['basis'][:,2]; heading=forward-(forward@normal)*normal
    length=np.linalg.norm(heading)
    if length<1e-8: return empty('no_valid_heading')
    heading/=length
    radius=grid.get('radius',8*grid['cell'])
    start,reason=start_node(grid,camera,policy)
    if start is None: return empty(reason)
    starts=[start]
    if policy=='visible_ground':
        groups,_=label(np.isfinite(grid['cost']))  # Four-way; never joins corners.
        ids=np.flatnonzero(np.isfinite(grid['cost'].ravel()))
        delta=grid['points'][ids]-camera
        ahead=delta@heading; lateral=np.linalg.norm(delta@grid['basis'][:,:2],axis=1)
        visible=((delta@normal)<0)&(ahead<=lookahead*radius)&(ahead>=-radius)
        # A video demo starts on ground observed in this frame. Old map islands
        # must not monopolize the start while current traversable ground lies
        # farther ahead. Retained path anchors are considered separately below.
        if 'goal_observed' in grid:visible &= grid['goal_observed'][ids]
        ids,ahead,lateral=ids[visible],ahead[visible],lateral[visible]
        group_ids=groups.ravel()[ids]; seeds=[]
        for group in np.unique(group_ids):
            rows=np.flatnonzero(group_ids==group)
            if len(rows)<6: continue
            nearest=rows[np.argmin(lateral[rows])]
            # Tiny islands must not establish a persistent four-point path.
            if ahead[rows].max()-ahead[nearest]<radius/2: continue
            seeds.append((float(lateral[nearest]),int(ids[nearest])))
        seeds=sorted(seeds)
        starts=[s for distance,s in seeds if distance<=seeds[0][0]+radius][:4] if seeds else []
        # In a video demo the current visible patch is not execution feedback.
        # Keep searching from reachable retained anchors, including just behind
        # the camera, rather than insisting on a new isolated visible start.
        if state.path is not None:
            finite=np.flatnonzero(np.isfinite(grid['cost'].ravel()))
            tree=cKDTree(grid['points'][finite]); old=state.path[state.committed:]
            distances,nearest=tree.query(old,distance_upper_bound=radius/2)
            retained=[]
            for point,distance,index in zip(old,distances,nearest):
                if not np.isfinite(distance): continue
                xy=np.floor((point@grid['basis'])[:2]/grid['cell']).astype(int)-grid['origin']
                w=grid['cost'].shape[0]
                if ((xy<0)|(xy>=w)).any(): continue
                if grid.get('hard_veto') is not None and grid['hard_veto'][xy[1],xy[0]]: continue
                row=int(finite[index]); ahead=(grid['points'][row]-camera)@heading
                if not -4*radius<=ahead<=lookahead*radius: continue
                retained.append((float(np.linalg.norm((point-camera)@grid['basis'][:,:2])),row))
            # Different anchor cells can share the same component; deduplicate.
            anchored=[]; seen=set()
            for _,row in sorted(retained):
                component=int(groups.ravel()[row])
                if component not in seen: seen.add(component); anchored.append(row)
            starts=list(dict.fromkeys(starts+anchored[:2]))[:4]
        if not starts: return empty('no_useful_forward_component')
    began=time.perf_counter(); best=None; best_score=-np.inf; reason='no_goal_ahead'
    diagnostics=dict(start_candidates=len(starts),reachable_nodes=0,anchor_candidates=0)
    for start in starts:
        remaining=budget_ms-(time.perf_counter()-began)*1000
        if remaining<=0: break
        status,distance,previous,settled,expanded=search(grid['cost'],grid['height'],start,grid['cell'],enforce,remaining,max_stride=grid.get('search_stride',1))
        if status<0: continue
        ids=np.flatnonzero(settled)
        diagnostics['reachable_nodes']=max(diagnostics['reachable_nodes'],len(ids))
        delta=grid['points'][ids]-camera; ahead=delta@heading
        lateral=np.linalg.norm(delta-(delta@normal)[:,None]*normal-ahead[:,None]*heading,axis=1)
        start_ahead=(grid['points'][start]-camera)@heading
        take=(ahead>=.75*radius)&(ahead<=lookahead*radius)&(lateral<=ahead)
        take &= ahead-start_ahead>=radius/2
        if 'goal_observed' in grid:take &= grid['goal_observed'][ids]
        if not take.any(): continue
        candidates=ids[take]; scores=ahead[take]-.5*lateral[take]-.05*distance[candidates]
        # Reach the furthest observed frontier first. Clearance/path costs
        # choose the detour; they must not favour a tiny nearby goal instead.
        furthest=ahead[take].max();frontier=np.flatnonzero(ahead[take]>=furthest-grid['cell'])
        pick=int(frontier[np.argmax(scores[frontier])]); goal=int(candidates[pick])
        def chain(end):
            rows=[int(end)]
            while rows[-1]!=start:
                parent=int(previous[rows[-1]])
                if parent<0 or len(rows)>grid['cost'].size: raise RuntimeError('Broken local predecessor chain')
                rows.append(parent)
            return np.array(rows[::-1],np.int64)
        route_ids=chain(goal); anchor=None; positions=grid['points'][route_ids]
        if state.path is not None:
            old=state.path[state.committed:]
            xy=np.floor((old@grid['basis'])[:,:2]/grid['cell']).astype(int)-grid['origin']
            w=grid['cost'].shape[0]; inside=((xy>=0)&(xy<w)).all(1)
            hard=grid.get('hard_veto')
            if hard is not None:
                offsets=np.flatnonzero(inside)
                inside[offsets]&=~hard[xy[offsets,1],xy[offsets,0]]
            offsets=np.flatnonzero(inside)
            if exclude_anchors:
                offsets=offsets[~np.isin(offsets+state.committed,list(exclude_anchors))]
            matches=cKDTree(grid['points'][ids]).query_ball_point(old[offsets],radius/2) if len(offsets) else []
            choices=[]
            for offset,neighbors in zip(offsets,matches):
                for neighbor in neighbors:
                    row=int(ids[neighbor]); bridge=np.linalg.norm(old[offset]-grid['points'][row])
                    choices.append((float(distance[row]+bridge),int(offset),row))
            diagnostics['anchor_candidates']=max(diagnostics['anchor_candidates'],len(choices))
            if not choices:
                reason='connection_lost'; diagnostics['retained_points_inside_window']=int(inside.sum()); continue
            _,offset,row=min(choices); anchor=state.committed+offset
            connector=chain(row);common=0
            while common<min(len(connector),len(route_ids)) and connector[common]==route_ids[common]:common+=1
            route_ids=np.r_[connector[common-1:][::-1],route_ids[common:]]
            positions=grid['points'][route_ids]
            if not np.array_equal(state.path[anchor],positions[0]): positions=np.vstack((state.path[anchor],positions))
            # A near association only proposes a connector. Raw support, slope
            # and swept obstacles must validate it before any splice is committed.
        score=float(scores[pick]-.1*np.linalg.norm((grid['points'][start]-camera)@grid['basis'][:,:2]))
        rank=(float(ahead[take][pick]),score)
        if best is None or rank>best_score:
            best_score=rank
            best=dict(status='budget_limited' if status else 'candidate',positions=positions,
                ids=route_ids,anchor=anchor,goal=grid['points'][goal],expanded=expanded,diagnostics=dict(diagnostics))
    return best if best is not None else empty(reason,**diagnostics)
