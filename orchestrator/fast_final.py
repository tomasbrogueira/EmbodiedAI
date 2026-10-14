"""Direct packet archive and bounded final-map planning; no dense exact stages."""
from collections import deque
from pathlib import Path
import time
import numpy as np
from orchestrator.artifacts import write_json
from orchestrator.local_evidence import PacketIndex,StableTerrain
from orchestrator.live_planning import LocalEngine
from path_planner.local_grid import fresh_ground
from path_planner.fast_slope import screen_sources
from path_planner.terrain_reasons import counts,RETRY
from path_planner.surface_layers import overlay_surface
from path_planner.final_grid import final_proposal

class FastFinal:
    def __init__(self,args,directory):
        self.args=args; self.root=directory/'fast_packets'; self.root.mkdir()
        self.directory=directory/'local_planning'; self.directory.mkdir()
        self.paths=[]

    def append(self,p):
        path=self.root/f'frame_{p["source_index"]:06d}.npz'
        # Canonical projected coordinates and source-separated masks are saved
        # once. No full-resolution prediction hashing or finalized exact costs.
        np.savez(path,**{k:v for k,v in p.items() if k not in ('source_rgb','revision')})
        self.paths.append(path)

    def finish(self,progress=None):
        begin=time.perf_counter(); engine=LocalEngine(self.args,self.directory)
        try:
            packets=[]
            for path in self.paths:
                with np.load(path,allow_pickle=False) as data: p={k:data[k].copy() for k in data.files}
                p['shape']=tuple(p['shape']); p['source_index']=int(p['source_index']);p['source_time']=float(p['source_time'])
                packets.append(p)
            if not packets: return self.unavailable('no_projected_packets')
            if any(not bool(p.get('semantic_valid',True)) for p in packets): return self.unavailable('incomplete_required_semantics')
            for p in packets:
                if engine.calibrate(p): break
            if engine.reference is None: return self.unavailable('waiting_for_reference')
            basis=engine.basis; radius=engine.radius; height=engine.height
            cameras=np.asarray([p['camera'] for p in packets]); xy=cameras@basis[:,:2]
            lower=xy.min(0)-16*radius; upper=xy.max(0)+16*radius
            span=float((upper-lower).max()); cell=max(radius/8,span/508)
            width=min(512,max(32,int(np.ceil(span/cell))+4))
            centre=np.r_[(lower+upper)/2,float(np.mean(cameras@basis[:,2]))]@basis.T
            memory=StableTerrain(capacity=len(packets)+1,layer_filter=getattr(self.args,'filter_lower_layers',True)); engine.history=[]
            for tick,p in enumerate(packets):
                if progress is not None: progress('fast terrain grid',tick,len(packets))
                fresh=fresh_ground(p,centre,basis,radius,height,width=width,divisor=radius/cell,layer_filter=memory.layer_filter)
                grid=memory.update(fresh,p,tick,radius,height)
                engine.history.append(dict(tick=tick,p=p,index=PacketIndex(p,basis,radius) if self.args.plan else None,ray=memory.ray,ground_mask=grid.get('selected_ground')))
            if progress is not None:progress('sparse slope exclusion',0,1)
            full=screen_sources(grid,engine.history,engine.reference,radius,self.args.steepness)
            original_cost=grid['cost'].copy() if self.args.plan else None
            plan=None; valid=False; reason='costs_ready' if self.args.cost_map else 'slopes_ready';metrics={};fallback=None
            if self.args.plan:
                for _ in range(4):
                    if progress is not None: progress('fast final path',0,1)
                    plan=final_proposal(grid,cameras,engine.search,self.args.steepness,policy=self.args.start_policy)
                    if len(plan['positions'])<2: reason=plan['status'];break
                    valid,reason,metrics=engine.validate(grid,full,plan)
                    if valid: break
                    if reason=='geometry_rejected':
                        partial=engine.validated_prefix(grid,full,plan,metrics)
                        if partial is not None:
                            if fallback is None or np.linalg.norm(partial[0]['positions'][-1]-partial[0]['positions'][0])>np.linalg.norm(fallback[0]['positions'][-1]-fallback[0]['positions'][0]):fallback=partial
                        engine.reject_geometry(grid,plan,metrics)
                    elif reason!='slope_rejected': break
                if not valid and fallback is not None:
                    plan,metrics=fallback
                    # Edge-retry exclusions are search hints, not permanent
                    # point hazards. Restore only the proven prefix's hints;
                    # semantic/geometry/slope exclusions can never be restored.
                    rows=plan['ids'];flags=grid['reasons'].ravel()[rows]
                    restore=rows[(flags&np.uint8(255^RETRY))==0]
                    grid['cost'].ravel()[restore]=original_cost.ravel()[restore]
                    grid['reasons'].ravel()[restore]&=np.uint8(255^RETRY)
                    valid,reason,metrics=engine.validate(grid,full,plan)
            ids=np.flatnonzero((grid['point_rows']>=0)&np.isfinite(grid['points']).all(1)); revision=len(packets)
            overlay=dict(points=grid['points'][ids],costs=grid['cost'].ravel()[ids],slopes=full[ids],
                **overlay_surface(grid),
                reasons=grid['reasons'].ravel()[ids],source_frames=grid['source_frames'][ids],
                route=plan['positions'] if valid else np.empty((0,3)),history=np.empty((0,3)),
                goal=plan['goal'] if valid else np.empty((0,3)),revision=np.array(revision),
                costs_enabled=np.array(self.args.cost_map),slopes_enabled=np.array(self.args.steepness))
            temporary=self.directory/'display.tmp.npz'; np.savez(temporary,**overlay);temporary.replace(self.directory/'display.npz')
            selected_goal_reached=bool(valid and plan.get('goal_reached',False))
            last_goal_reached=selected_goal_reached and plan.get('goal_frame')==len(packets)-1
            meta=dict(status=('ready' if last_goal_reached else 'partial') if valid else reason,
                active_valid=bool(valid),revision=revision,latest_revision=revision,phase='final_fast',
                grid_capacity=width*width,cell_size=cell,radius=radius,clearance_height=height,
                reference=engine.reference.tolist(),validation_backend=engine.backend,backend_note=engine.gpu_error,
                processing_ms=(time.perf_counter()-begin)*1000,eligible_cells=int(np.isfinite(grid['cost']).sum()),
                slope_mode='sparse source-frame exclusion' if self.args.steepness else 'disabled',terrain_exclusions=counts(grid),
                layer_filter=memory.layer_filter,
                start_policy=self.args.start_policy,units='model units',calibration=engine.calibration,
                geometry_validation=metrics,optimality_scope='bounded approximate terrain grid',
                source_packets=len(packets),goal_reached=bool(last_goal_reached),selected_goal_reached=selected_goal_reached,
                start_source_index=packets[plan['start_frame']]['source_index'] if valid else None,
                goal_source_index=packets[plan['goal_frame']]['source_index'] if valid else None)
            if valid:meta['demo_start_offset']=plan.get('demo_start_offset')
            write_json(self.directory/'status.json',meta);return meta
        finally: engine.close()

    def unavailable(self,reason):
        meta=dict(status=reason,active_valid=False,phase='final_fast',revision=len(self.paths),latest_revision=len(self.paths))
        write_json(self.directory/'status.json',meta);return meta
