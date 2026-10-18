"""Aligned frame packets -> bounded terrain -> asynchronous local planning."""
from collections import deque
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import numpy as np
from orchestrator.artifacts import write_json
from orchestrator.local_evidence import PacketIndex, StableTerrain
from orchestrator.paths import ROOT
from path_planner.floor_reference import estimate_floor_reference,estimate_local_floor_reference
from path_planner.local_grid import (basis_for, fresh_ground,
    raw_node_slopes, Search, PathState, certify)
from path_planner.fast_slope import prepare_slopes,screen_grid
from path_planner.terrain_reasons import RETRY,counts
from orchestrator.visibility import RayEvidence
from path_planner.surface_layers import obsolete_lower,overlay_surface,GroundAtlas
from path_planner.local_route import propose


class GPUValidator:
    """Own one optional CUDA worker; serialized requests own their NPZ slot."""
    def __init__(self,python,directory):
        self.directory=directory; self.process=None; self.messages=queue.Queue()
        self.index_token=None
        self.log=(directory/'gpu-validation.log').open('w')
        # Venv symlinks resolve to the base Python, so inspect the original path.
        prefix=Path(python).absolute().parent.parent
        site=prefix/f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/nvidia'
        libraries=[str(site/name/'lib') for name in ('cuda_nvrtc','cuda_runtime')]
        env=os.environ|{'PYTHONPATH':str(ROOT),'LD_LIBRARY_PATH':os.pathsep.join(libraries+['/usr/local/cuda/lib64',os.environ.get('LD_LIBRARY_PATH','')]),
                        'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONUNBUFFERED':'1'}
        self.process=subprocess.Popen([python,'-m','path_planner.gpu_worker'],cwd=ROOT,
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1,env=env)
        def read():
            try:
                for line in self.process.stdout: self.messages.put(json.loads(line))
            except Exception as exc: self.messages.put(dict(status='error',error=str(exc)))
            finally: self.messages.put(dict(status='error',error='GPU validator exited'))
        threading.Thread(target=read,daemon=True).start()
        try: self.receive(30)
        except BaseException: self.close(); raise

    def receive(self,timeout=30):
        result=self.messages.get(timeout=timeout)
        if result['status']=='error': raise RuntimeError(result['error'])
        return result

    def check(self,points,positive,negative,slopes,route,normal,radius,height,enforce,cache_token=None):
        reuse=cache_token is not None and cache_token==self.index_token
        path=self.directory/('validation-route.npz' if reuse else 'validation-input.npz')
        payload=dict(route=route,normal=normal,height=height,enforce=enforce)
        if not reuse:payload.update(points=points,positive=positive,negative=negative,slopes=slopes,radius=radius)
        np.savez(path,**payload)
        self.process.stdin.write(json.dumps(dict(input=str(path),op='route' if reuse else 'check'))+'\n'); self.process.stdin.flush()
        result=self.receive()
        self.index_token=cache_token
        return bool(result['valid']),result

    def close(self):
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.stdin.write('{"op":"close"}\n'); self.process.stdin.flush(); self.process.wait(timeout=5)
            except (OSError,subprocess.TimeoutExpired):
                self.process.terminate()
                try: self.process.wait(timeout=5)
                except subprocess.TimeoutExpired: self.process.kill(); self.process.wait()
        self.log.close()


class LocalEngine:
    def __init__(self,args,directory):
        self.args=args; self.directory=directory
        self.reference=None; self.radius=None; self.height=None; self.basis=None
        self.bootstrap=deque(maxlen=5); self.bootstrap_packets=deque(maxlen=5); self.history=deque(maxlen=20)
        self.memory=StableTerrain(layer_filter=getattr(args,'filter_lower_layers',True)); self.path=PathState(); self.tick=0;self.path_epoch=0
        self.search=Search() if args.plan else None
        self.gpu=None; self.gpu_error=None
        self.backend='cpu'; self.calibration='waiting_for_reference'
        self.reference_report={};self.last_validated=None
        self.ground_atlas=GroundAtlas();self.disconnected_frames=set()
        self.validation_cache=None;self.validation_generation=0

    def retire_path(self,p,reason):
        archive=self.directory/'retired_paths';archive.mkdir(exist_ok=True)
        np.savez(archive/f'epoch_{self.path_epoch:04d}.npz',points=self.path.path,committed=self.path.committed,
            revision=p['revision'],reason=reason)
        self.path=PathState();self.last_validated=None;self.path_epoch+=1;self.disconnected_frames.clear()

    def calibrate(self,p):
        self.bootstrap_packets.append(p)
        shape=p['shape']; image=np.full((*shape,3),np.nan,dtype=np.float32)
        selected=np.zeros(shape,bool)
        image.reshape(-1,3)[p['pixels']]=p['points']
        support=p.get('base_support_mask',np.zeros(len(p['points']),bool)) if self.args.reference_source=='base_support' else p['positive']
        selected.ravel()[p['pixels']]=support&~p['negative']
        self.bootstrap.append(dict(id=p['source_index'],points=image,floor_mask=selected,camera=p['camera']))
        normal,report=estimate_floor_reference(self.bootstrap)
        report['method']='whole observed ground'
        if normal is None:normal,report=estimate_local_floor_reference(self.bootstrap,report)
        self.reference_report=report
        if normal is None:
            self.calibration=report['reason']; return False
        self.reference=np.asarray(self.args.reference,dtype=float) if self.args.reference is not None else normal
        self.reference/=np.linalg.norm(self.reference)
        spacing=[f.get('scene_spacing',f['spacing']) for f in report['frames'] if f['reason']=='accepted']
        self.radius=float(self.args.radius or 20*np.median(spacing))
        if self.args.clearance_height is not None:
            self.height=max(self.radius,float(self.args.clearance_height)); self.calibration='supplied_relative_height'
        else:
            # An observed plane supplies a camera-height proxy, not a robot body.
            drops=[]
            for f in self.bootstrap:
                points=f['points'][f['floor_mask']]
                if len(points):
                    drop=float(np.median((f['camera']-points)@self.reference))
                    if np.isfinite(drop) and drop>0: drops.append(drop)
            if not drops: self.reference=None; self.calibration='waiting_for_clearance'; return False
            self.height=max(self.radius,float(np.median(drops))); self.calibration='observed_camera_ground_height_proxy'
        self.basis=basis_for(self.reference)
        self.bootstrap.clear()
        return True

    def collect_evidence(self,grid,full,plan):
        radius,height,normal=self.radius,self.height,self.reference
        # Source-frame screening already ran only for representative rows.
        # Fast modes never build the full per-point slope/refiner KD tree.
        points=[]; positives=[]; negatives=[]
        latest=list(self.history)[-2:];cleared_count=0;lower_count=0
        for slot in self.history:
            ids=slot['index'].query(plan['positions'],normal,radius,height)
            samples=slot['p']['points'][ids]; pos=slot['p']['positive'][ids].copy(); neg=slot['p']['negative'][ids]
            xy=np.floor((samples@grid['basis'])[:,:2]/grid['cell']).astype(int)-grid['origin']
            w=grid['cost'].shape[0]; inside=((xy>=0)&(xy<w)).all(1); rows=np.flatnonzero(inside)
            cells=xy[rows,1]*w+xy[rows,0]
            lower=obsolete_lower(samples,grid)&~neg
            lower_count+=int(lower.sum())
            if grid.get('layer_filter'):
                # Other selected-positive layers cannot lend their support to
                # this surface. Higher unselected surfaces remain obstacles.
                band=abs(samples[rows]@normal-grid['height'].ravel()[cells])<=grid['layer_tolerance']
                pos[rows]&=band
            # Newer confirmed support can correct an old untyped floor label at
            # the same surface. Never reinterpret an explicit negative mask.
            floor=(~pos[rows])&~neg[rows]&np.isfinite(grid['cost'].ravel()[cells])
            floor &= grid.get('confirmed_ticks',grid['source_ticks'])[cells]>slot['tick']
            floor &= abs(samples[rows]@normal-grid['height'].ravel()[cells])<=radius/20
            pos[rows[floor]]=True
            # Clear an old untyped sample only if two later observations show
            # depth rays passing its location. Explicit negatives never clear.
            if 'cleared' not in slot:slot['cleared']=np.zeros(len(slot['p']['points']),bool)
            candidates=rows[(~pos[rows])&~neg[rows]&~lower[rows]&np.isfinite(grid['cost'].ravel()[cells])]
            distinct=len({int(w['p'].get('source_index',w['tick'])) for w in latest})==2
            if candidates.size and len(latest)==2 and distinct and latest[0]['tick']>slot['tick']:
                epoch=tuple(w['tick'] for w in latest)
                if slot.get('visibility_epoch')!=epoch:
                    slot['visibility_epoch']=epoch;slot['visibility_checked']=np.zeros(len(slot['p']['points']),bool)
                candidates=candidates[~slot['visibility_checked'][ids[candidates]]]
                clear=np.ones(len(candidates),bool)
                for witness in latest:
                    if witness.get('ray') is None:witness['ray']=RayEvidence(witness['p'],witness.get('ground_mask'))
                    clear &= witness['ray'].clears(samples[candidates],radius/20)
                slot['cleared'][ids[candidates[clear]]]=True
                slot['visibility_checked'][ids[candidates]]=True
            keep=(~slot['cleared'][ids]&~lower)|neg
            cleared_count+=int((slot['cleared'][ids]&~neg).sum())
            points.append(samples[keep]); positives.append(pos[keep]); negatives.append(neg[keep])
        points=np.concatenate(points); positive=np.concatenate(positives); negative=np.concatenate(negatives)
        values=raw_node_slopes(points,grid,full) if self.args.steepness else np.zeros(len(points))
        return points,positive,negative,values,cleared_count,lower_count

    def validate(self,grid,full,plan):
        radius,height,normal=self.radius,self.height,self.reference
        sweep=np.vstack((plan['positions'],plan['positions']+height*normal))
        lower=sweep.min(0)-radius;upper=sweep.max(0)+radius
        cache=self.validation_cache
        reuse=cache is not None and cache['grid'] is grid and cache['full'] is full and (lower>=cache['lower']).all() and (upper<=cache['upper']).all()
        if not reuse:
            self.validation_generation+=1
            cache=dict(lower=lower,upper=upper,grid=grid,full=full,evidence=self.collect_evidence(grid,full,plan),token=self.validation_generation)
            self.validation_cache=cache
        points,positive,negative,values,cleared_count,lower_count=cache['evidence']
        if self.args.validation_backend!='cpu' and self.gpu_error is None:
            try:
                if self.gpu is None: self.gpu=GPUValidator(self.args.lingbot_python,self.directory)
                valid,metrics=self.gpu.check(points,positive,negative,values,plan['positions'],normal,radius,height,self.args.steepness,cache_token=cache['token'])
                metrics['stale_geometry_cleared']=cleared_count
                metrics['lower_layer_points_removed']=lower_count
                metrics['evidence_cache_hit']=bool(reuse)
                self.backend='gpu'; return valid,'validated' if valid else 'geometry_rejected',metrics
            except Exception as exc:
                self.gpu_error=str(exc)
                if self.gpu is not None: self.gpu.close(); self.gpu=None
        self.backend='cpu'
        diagnostics={}
        valid,ms=certify(points,positive,negative,values,plan['positions'],normal,radius,height,self.args.steepness,diagnostics=diagnostics)
        return valid,'validated' if valid else 'geometry_rejected',dict(cpu_ms=ms,gpu_fallback=self.gpu_error,stale_geometry_cleared=cleared_count,lower_layer_points_removed=lower_count,evidence_cache_hit=bool(reuse),**diagnostics)

    def validated_prefix(self,grid,full,plan,metrics):
        """Retain useful valid progress before a failed edge; never cross it."""
        failures=metrics.get('metrics',metrics)
        limits=[int(i)+1 for i in failures.get('failed_edges',[])]
        limits.extend(int(i) for i in failures.get('failed_nodes',[]))
        if not limits:return None
        count=min(limits)
        if count<2:return None
        positions=plan['positions'][:count]
        if np.linalg.norm(positions[-1]-positions[0])<self.radius/2:return None
        shift=len(plan['positions'])-len(plan['ids'])
        partial=dict(plan,positions=positions,ids=plan['ids'][:count-shift],goal=positions[-1],goal_reached=False,status='partial_candidate')
        valid,reason,checks=self.validate(grid,full,partial)
        return (partial,checks) if valid else None

    def reject_geometry(self,grid,plan,metrics):
        """Block failed witnesses for this revision and reject an invalid join."""
        failures=metrics.get('metrics',metrics)  # GPU worker wraps device metrics.
        shift=len(plan['positions'])-len(plan['ids'])
        bad_points=set(failures.get('failed_nodes',[]))
        for edge in failures.get('failed_edges',[]): bad_points.add(int(edge)+1)
        rejected_anchor=False
        if shift and (0 in failures.get('failed_nodes',[]) or 0 in failures.get('failed_edges',[])):
            rejected_anchor=True;bad_points.discard(1)
        ids=[plan['ids'][i-shift] for i in bad_points if 0<=i-shift<len(plan['ids'])]
        if ids:
            grid['cost'].ravel()[ids]=np.inf
            if 'reasons' in grid:grid['reasons'].ravel()[ids]|=RETRY
        return rejected_anchor

    def process(self,p):
        begin=time.perf_counter()
        # Sources and visibility witnesses change between packets. Reuse is
        # restricted to checks within this update and the same evidence box.
        self.validation_cache=None
        meta=dict(status='waiting_for_reference',revision=p['revision'],source_index=p['source_index'],
            source_time=p['source_time'],active_valid=False,units='model units',start_policy=self.args.start_policy,
            validation_backend=self.backend,calibration=self.calibration)
        if not p.get('semantic_valid',True):
            return meta|dict(status='incomplete_required_semantics'),None,None
        if self.reference is None and not self.calibrate(p):
            return meta|dict(calibration=self.calibration,reference_diagnostics=self.reference_report),None,None
        meta.update(reference=self.reference.tolist(),radius=self.radius,clearance_height=self.height,calibration=self.calibration,
            reference_method=self.reference_report.get('method','supplied / fixed'))
        # Current camera plus a small forward bias extends the local window ahead.
        heading=p['forward']-(p['forward']@self.reference)*self.reference
        heading/=max(np.linalg.norm(heading),1e-12)
        lookahead=float(self.args.lookahead) if self.args.plan else 12.
        auto_range=self.args.plan and self.args.start_policy=='visible_ground' and getattr(self.args,'auto_lookahead',True)
        if auto_range:
            delta=p['points']-p['camera'];ahead=delta@heading
            lateral=np.linalg.norm(delta-(delta@self.reference)[:,None]*self.reference-ahead[:,None]*heading,axis=1)
            observed=p['positive']&~p['negative']&((delta@self.reference)<0)&(ahead>0)&(lateral<=ahead)
            if observed.any():lookahead=min(48.,max(lookahead,float(np.quantile(ahead[observed],.995))/self.radius))
        meta['effective_lookahead']=lookahead
        # Stable capacity keeps temporal arrays aligned even as range changes.
        # Include near ground and the visible frontier together, rather than
        # shifting a short window far ahead and losing every retained anchor.
        divisor=4 if auto_range else 8
        width=224 if auto_range else max(200,int(np.ceil((lookahead+8)*divisor)))
        centre=p['camera']+heading*self.radius*lookahead/2
        sources=list(self.bootstrap_packets) if self.bootstrap_packets else [p]
        self.bootstrap_packets.clear()
        for source in sources:
            fresh=fresh_ground(source,centre,self.basis,self.radius,self.height,width=width,divisor=divisor,layer_filter=self.memory.layer_filter)
            prepare_slopes(fresh,source,self.reference,self.radius,self.args.steepness)
            grid=self.memory.update(fresh,source,self.tick,self.radius,self.height)
            self.history.append(dict(tick=self.tick,p=source,index=PacketIndex(source,self.basis,self.radius) if self.args.plan else None,ray=self.memory.ray,ground_mask=grid.get('selected_ground')))
            self.tick+=1
        full=screen_grid(grid,self.args.steepness)
        if self.memory.layer_filter:self.ground_atlas.update(grid,p['camera'])
        corrected=False
        if self.args.plan and self.path.path is not None and self.args.start_policy=='visible_ground' and getattr(self.args,'replan_layer_changes',True):
            tail=self.path.path[self.path.committed:];obsolete=obsolete_lower(tail,grid)
            if int(obsolete.sum())>=3 and obsolete.mean()>=.5:
                self.retire_path(p,'confirmed ground-layer correction');corrected=True
        plan=None; reason='costs_ready' if self.args.cost_map else 'slopes_ready'; valid=False; metrics={};fallback=None
        original_cost=grid['cost'].copy() if self.args.plan else None
        if self.args.plan:
            excluded_anchors=set()
            proposal_state=self.path;restart_reason=None
            for _ in range(3):
                plan=propose(grid,p['camera'],p['forward'],self.search,proposal_state,self.args.steepness,
                    policy=self.args.start_policy,lookahead=lookahead,budget_ms=self.args.planning_budget_ms,exclude_anchors=excluded_anchors)
                lost=plan['status']=='connection_lost'
                if lost:self.disconnected_frames.add(int(p['source_index']))
                if (lost and len(self.disconnected_frames)>=2 and self.args.start_policy=='visible_ground'
                        and getattr(self.args,'replan_disconnected',True)):
                    # A video camera is not execution feedback. Continue the
                    # local demo only after raw validation, with an explicit
                    # new segment; never fabricate a bridge to the old path.
                    proposal_state=PathState();restart_reason='visible ground disconnected from previous inferred path'
                    plan=propose(grid,p['camera'],p['forward'],self.search,proposal_state,self.args.steepness,
                        policy=self.args.start_policy,lookahead=lookahead,budget_ms=self.args.planning_budget_ms)
                if len(plan['positions'])<2: reason=plan['status']; break
                valid,reason,metrics=self.validate(grid,full,plan)
                if valid:
                    if restart_reason is None:self.disconnected_frames.clear()
                    break
                if reason=='geometry_rejected':
                    partial=self.validated_prefix(grid,full,plan,metrics)
                    if partial is not None:
                        ahead=(partial[0]['goal']-p['camera'])@heading
                        if .75*self.radius<=ahead<=lookahead*self.radius:
                            if fallback is None or ahead>(fallback[0]['goal']-p['camera'])@heading:fallback=partial
                    failures=metrics.get('metrics',metrics)
                    join_failed=plan['anchor'] is not None and (0 in failures.get('failed_nodes',[]) or 0 in failures.get('failed_edges',[]))
                    if join_failed:self.disconnected_frames.add(int(p['source_index']))
                    if (self.reject_geometry(grid,plan,metrics) or join_failed) and plan['anchor'] is not None: excluded_anchors.add(plan['anchor'])
                elif reason!='slope_rejected': break
            if not valid and fallback is not None:
                plan,metrics=fallback;rows=plan['ids'];flags=grid['reasons'].ravel()[rows]
                restore=rows[(flags&np.uint8(255^RETRY))==0]
                grid['cost'].ravel()[restore]=original_cost.ravel()[restore]
                grid['reasons'].ravel()[restore]&=np.uint8(255^RETRY)
                valid,reason,metrics=self.validate(grid,full,plan)
            if (not valid and self.path.path is not None and len(self.disconnected_frames)>=2
                    and self.args.start_policy=='visible_ground' and getattr(self.args,'replan_disconnected',True)):
                # A rejected old anchor is another loss of continuity. Search
                # hints from that failed join must not suppress fresh ground.
                restore=(grid['reasons']&np.uint8(255^RETRY))==0
                grid['cost'][restore]=original_cost[restore];grid['reasons'][restore]&=np.uint8(255^RETRY)
                fresh_plan=propose(grid,p['camera'],p['forward'],self.search,PathState(),self.args.steepness,
                    policy=self.args.start_policy,lookahead=lookahead,budget_ms=self.args.planning_budget_ms)
                if len(fresh_plan['positions'])>=2:
                    fresh_valid,fresh_reason,fresh_metrics=self.validate(grid,full,fresh_plan)
                    if not fresh_valid:
                        prefix=self.validated_prefix(grid,full,fresh_plan,fresh_metrics)
                        if prefix is not None:
                            fresh_plan,fresh_metrics=prefix
                            fresh_valid=(fresh_plan['goal']-p['camera'])@heading>=.75*self.radius
                    if fresh_valid:
                        plan,metrics,valid=fresh_plan,fresh_metrics,True
                        restart_reason='previous inferred path cannot connect to current observed ground'
            if valid and restart_reason is not None:self.retire_path(p,restart_reason)
        partial=bool(valid and plan.get('status')=='partial_candidate')
        meta.update(status=('partial_ready' if partial else 'ready') if valid else reason,active_valid=bool(valid),validation_backend=self.backend,
            layer_filter=self.memory.layer_filter,path_epoch=self.path_epoch,layer_corrected=corrected,
            continuity_restart=bool(valid and restart_reason is not None) if self.args.plan else False,
            grid_capacity=width*width,
            lower_layer_updates_ignored=grid['lower_layer_updates_ignored'],
            backend_note=self.gpu_error,processing_ms=(time.perf_counter()-begin)*1000,
            eligible_cells=int(np.isfinite(grid['cost']).sum()),history_frames=len(self.history),
            slope_mode='sparse source-frame exclusion' if self.args.steepness else 'disabled',
            terrain_exclusions=counts(grid),
            geometry_validation=metrics,progress_kind='inferred camera / proposed history; no execution feedback',
            route_vertices=len(plan['positions']) if valid else 0,
            route_diagnostics={} if plan is None else plan.get('diagnostics',{}))
        ids=np.flatnonzero(np.isfinite(grid['points']).all(1)&(grid['point_rows']>=0))
        overlay=dict(points=grid['points'][ids],costs=grid['cost'].ravel()[ids],slopes=full[ids],
            **overlay_surface(grid),
            **self.ground_atlas.overlay(),
            reasons=grid['reasons'].ravel()[ids],source_frames=grid['source_frames'][ids],
            route=plan['positions'] if valid else np.empty((0,3)),
            last_route=self.last_validated if self.last_validated is not None else np.empty((0,3)),
            history=np.empty((0,3)) if self.path.path is None else self.path.path[:self.path.committed+1],
            goal=plan['goal'] if valid else np.empty((0,3)),revision=np.array(p['revision']),
            costs_enabled=np.array(self.args.cost_map),slopes_enabled=np.array(self.args.steepness))
        return meta,overlay,plan if valid else None

    def commit(self,plan,overlay):
        if plan is not None:
            self.path.splice(plan['anchor'],plan['positions'])
            overlay['history']=self.path.path[:self.path.committed+1]
            self.last_validated=plan['positions'].copy();overlay['last_route']=self.last_validated

    def close(self):
        if self.gpu is not None: self.gpu.close()


class LivePlanning:
    """Capacity-one pending packet; superseded results never become active routes."""
    def __init__(self,args,directory):
        self.directory=directory/'local_planning'; self.directory.mkdir()
        self.engine=LocalEngine(args,self.directory)
        self.condition=threading.Condition(); self.pending=None; self.closing=False; self.revision=0
        self.meta=dict(status='waiting_for_reference',active_valid=False,revision=0)
        self.thread=threading.Thread(target=self.loop,daemon=True); self.thread.start()

    def submit(self,p):
        with self.condition:
            self.revision+=1; p=dict(p,revision=self.revision)
            self.pending=p
            self.meta=self.meta|dict(status='updating',active_valid=False,latest_revision=self.revision)
            write_json(self.directory/'status.json',self.meta)
            self.condition.notify()

    def snapshot(self):
        with self.condition: return dict(self.meta)

    def loop(self):
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda:self.pending is not None or self.closing)
                    if self.pending is None: break
                    p,self.pending=self.pending,None
                try:
                    meta,overlay,plan=self.engine.process(p)
                except Exception as exc:
                    meta=dict(status='failed',error=str(exc),active_valid=False,revision=p['revision']); overlay=plan=None
                with self.condition:
                    delayed=p['revision']!=self.revision
                    if overlay is not None:
                        # Processing is ordered even when capture runs ahead.
                        # Preserve a completed validated trace and its shared
                        # prefix instead of starving the viewer until EOF.
                        # Delayed results are historical, never current.
                        self.engine.commit(plan,overlay)
                        temporary=self.directory/'display.tmp.npz'; np.savez(temporary,**overlay)
                        temporary.replace(self.directory/'display.npz')
                    self.meta=meta|dict(latest_revision=self.revision)
                    if delayed:
                        self.meta.update(status='updating',active_valid=False,
                            last_completed_status=meta['status'],last_completed_revision=p['revision'],
                            planning_lag_frames=self.revision-p['revision'])
                    write_json(self.directory/'status.json',self.meta)
                    with (self.directory/'events.jsonl').open('a') as log:
                        log.write(json.dumps(self.meta)+'\n')
        finally: self.engine.close()

    def close(self):
        with self.condition: self.closing=True; self.condition.notify()
        self.thread.join()
