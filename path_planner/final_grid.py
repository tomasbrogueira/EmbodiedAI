"""Bounded approximate final-map route between observed recording endpoints."""
import numpy as np
from scipy.ndimage import label
from .local_route import start_node

def final_proposal(grid,cameras,search,enforce,*,policy='strict',budget_ms=200.):
    empty=lambda reason:dict(status=reason,positions=np.empty((0,3)),ids=np.empty(0,np.int64),anchor=None)
    points=grid['points']; finite=np.isfinite(grid['cost'].ravel()); basis=grid['basis']
    groups,_=label(finite.reshape(grid['cost'].shape)); counts=np.bincount(groups.ravel())
    def endpoints(camera,toward=None):
        if policy=='strict':
            row,_=start_node(grid,camera,policy); return [] if row is None else [row]
        ids=np.flatnonzero(finite&((points-camera)@basis[:,2]<0))
        if not len(ids): return []
        distance=np.linalg.norm((points[ids]-camera)@basis[:,:2],axis=1)
        choices=[]
        for group in np.unique(groups.ravel()[ids]):
            if counts[group]<6: continue
            rows=np.flatnonzero(groups.ravel()[ids]==group); pick=rows[np.argmin(distance[rows])]
            useful=True
            if toward is not None:
                target_distance=np.linalg.norm((points[ids[rows]]-toward)@basis[:,:2],axis=1)
                useful=target_distance[np.argmin(distance[rows])]-target_distance.min()>=grid['radius']/2
            choices.append((float(distance[pick]),int(ids[pick]),useful))
        choices=sorted(choices)
        if toward is not None and choices:
            # Demo initialization can skip an unproductive island only within
            # the nearest observed-ground neighborhood. This is not a claim of
            # connection from the camera/robot or permission to jump a barrier.
            viable=[entry for entry in choices if entry[2] and entry[0]<=choices[0][0]+grid['radius']]
            if viable:choices=viable
        return [row for _,row,_ in choices[:4]]
    start_frame=next((i for i,c in enumerate(cameras) if endpoints(c,cameras[-1])),None)
    goal_frame=next((i for i in range(len(cameras)-1,-1,-1) if endpoints(cameras[i])),None)
    if start_frame is None or goal_frame is None: return empty('no_observed_recording_endpoints')
    starts=endpoints(cameras[start_frame],cameras[goal_frame])[:1]; goals=endpoints(cameras[goal_frame])[:1]; best=None; best_score=np.inf
    for start in starts:
        status,distance,previous,settled,expanded=search(grid['cost'],grid['height'],start,grid['cell'],enforce,budget_ms/len(starts),max_stride=grid.get('search_stride',1))
        if status<0: continue
        reachable=np.flatnonzero(settled)
        if not len(reachable): continue
        complete=next((g for g in goals if settled[g] and g!=start),None)
        if complete is None:
            target=points[goals[0]]
            goal=int(reachable[np.argmin(np.linalg.norm(points[reachable]-target,axis=1))])
            if np.linalg.norm(points[goal]-points[start])<grid['radius']/2: continue
            score=1e3+np.linalg.norm(points[goal]-target)
        else:
            goal=complete
            score=np.linalg.norm((points[start]-cameras[start_frame])@basis[:,:2])+np.linalg.norm((points[goal]-cameras[goal_frame])@basis[:,:2])
        chain=[goal]
        while chain[-1]!=start:
            row=int(previous[chain[-1]])
            if row<0 or len(chain)>len(points): raise RuntimeError('Broken final-grid predecessor chain')
            chain.append(row)
        ids=np.asarray(chain[::-1],np.int64)
        if score<best_score:
            best_score=score; best=dict(status='candidate' if complete is not None else 'partial_candidate',
                positions=points[ids],ids=ids,anchor=None,goal=points[goal],expanded=expanded,
                goal_reached=complete is not None,start_frame=start_frame,goal_frame=goal_frame,
                demo_start_offset=float(np.linalg.norm((points[start]-cameras[start_frame])@basis[:,:2])),
                optimality_scope='bounded approximate terrain grid; no raw-point optimality claim')
    return best if best is not None else empty('no_connected_final_route')
