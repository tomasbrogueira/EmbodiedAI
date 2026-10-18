"""Forward range, obstacle detours and explicit demo segment recovery."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from tests.test_local_planning import packet
from orchestrator.live_planning import LocalEngine
from orchestrator.local_evidence import StableTerrain
from path_planner.local_grid import basis_for,fresh_ground,Search,PathState
from path_planner.local_route import propose
from path_planner.surface_layers import GroundAtlas,overlay_surface,surface_from_overlay,obsolete_lower
from visualizer.display import planning_view

class ForwardTests(unittest.TestCase):
    def engine(self,directory,**overrides):
        args=dict(plan=True,steepness=False,cost_map=True,validation_backend='cpu',start_policy='visible_ground',
            lookahead=6.,auto_lookahead=True,planning_budget_ms=40.,filter_lower_layers=True,
            replan_layer_changes=True,replan_disconnected=True)
        args.update(overrides);engine=LocalEngine(SimpleNamespace(**args),Path(directory))
        engine.reference=np.array([0.,0.,1.]);engine.radius=.08;engine.height=.15;engine.basis=basis_for(engine.reference)
        return engine

    def p(self,index=0):return packet()|dict(revision=index+1,source_index=index,source_time=float(index))

    def test_forward_frontier_and_obstacle_detour_are_raw_validated(self):
        p=self.p();p['negative']=(abs(p['points'][:,0]-.205)<.015)&(abs(p['points'][:,1])<.12)
        with tempfile.TemporaryDirectory() as directory:
            engine=self.engine(directory);meta,overlay,plan=engine.process(p)
            self.assertTrue(meta['active_valid'],meta)
            self.assertGreater(plan['goal'][0],1.3)
            a,b=plan['positions'][:-1],plan['positions'][1:]
            cross=((a[:,0]<.205)&(b[:,0]>=.205))|((b[:,0]<.205)&(a[:,0]>=.205))
            a,b=a[cross],b[cross];wall=a+((.205-a[:,0])/(b[:,0]-a[:,0]))[:,None]*(b-a)
            self.assertGreater(len(wall),0);self.assertTrue((abs(wall[:,1])>.12).all())
            self.assertGreater(meta['effective_lookahead'],6.)
            self.assertLessEqual(meta['grid_capacity'],512**2)

    def test_disconnected_demo_recovers_only_after_distinct_frames_and_keeps_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            engine=self.engine(directory);old=np.array([[20.,20.,0.],[21.,20.,0.]])
            engine.path.path=old.copy()
            for index in (0,0):
                meta,_,plan=engine.process(self.p(index));self.assertFalse(meta['active_valid']);self.assertIsNone(plan)
            meta,overlay,plan=engine.process(self.p(1))
            self.assertTrue(meta['active_valid']);self.assertTrue(meta['continuity_restart']);self.assertEqual(meta['path_epoch'],1)
            engine.commit(plan,overlay);self.assertLess(engine.path.path[:,0].max(),2.)
            with np.load(Path(directory)/'retired_paths/epoch_0000.npz') as d:np.testing.assert_array_equal(d['points'],old)

    def test_goal_must_be_observed_in_current_frame_not_only_old_map(self):
        with tempfile.TemporaryDirectory() as directory:
            engine=self.engine(directory);p=self.p();meta,overlay,plan=engine.process(p);engine.commit(plan,overlay)
            self.assertGreater(plan['goal'][0],1.3)
            current=self.p(1);take=current['points'][:,0]<.8
            current.update(points=current['points'][take],positive=current['positive'][take],negative=current['negative'][take])
            meta,_,plan=engine.process(current)
            self.assertTrue(meta['active_valid'],meta);self.assertLess(plan['goal'][0],.8);self.assertGreater(plan['goal'][0],.7)

    def test_demo_start_uses_current_ground_instead_of_stale_nearer_island(self):
        p=self.p();basis=basis_for(np.array([0.,0.,1.]));memory=StableTerrain()
        grid=memory.update(fresh_ground(p,p['camera'],basis,.08,.15),p,0,.08,.15)
        # Old near-camera ground is disconnected from the current forward
        # patch. Both remain eligible in the accumulated map.
        wall=int(np.floor(.3/grid['cell'])-grid['origin'][0]);grid['cost'][:,wall]=np.inf
        grid['goal_observed']=np.isfinite(grid['points']).all(1)&(grid['points'][:,0]>.6)
        plan=propose(grid,p['camera'],p['forward'],Search(),PathState(),False,policy='visible_ground',lookahead=16.)
        self.assertGreater(len(plan['positions']),2,plan)
        self.assertGreater(plan['positions'][0,0],.6);self.assertGreater(plan['goal'][0],.9)

    def test_demo_start_uses_current_part_of_connected_component(self):
        p=self.p();basis=basis_for(np.array([0.,0.,1.]));memory=StableTerrain()
        grid=memory.update(fresh_ground(p,p['camera'],basis,.08,.15),p,0,.08,.15)
        grid['goal_observed']=np.isfinite(grid['points']).all(1)&(grid['points'][:,0]>.6)
        plan=propose(grid,p['camera'],p['forward'],Search(),PathState(),False,policy='visible_ground',lookahead=16.)
        self.assertGreater(plan['positions'][0,0],.6);self.assertGreater(plan['goal'][0],.9)

    def test_strict_or_disabled_recovery_keeps_connection_failure(self):
        for settings in (dict(start_policy='strict'),dict(replan_disconnected=False)):
            with tempfile.TemporaryDirectory() as directory:
                engine=self.engine(directory,**settings);engine.path.path=np.array([[20.,20.,0.],[21.,20.,0.]])
                for index in (0,1,2):
                    meta,_,plan=engine.process(self.p(index));self.assertEqual(meta['status'],'connection_lost');self.assertIsNone(plan)
                self.assertEqual(engine.path_epoch,0)

    def test_rejected_fresh_segment_does_not_retire_existing_path(self):
        with tempfile.TemporaryDirectory() as directory:
            engine=self.engine(directory);old=np.array([[20.,20.,0.],[21.,20.,0.]])
            engine.path.path=old.copy();engine.validate=lambda *args:(False,'geometry_rejected',{})
            for index in (0,1):meta,_,plan=engine.process(self.p(index))
            self.assertFalse(meta['active_valid']);self.assertIsNone(plan);self.assertEqual(engine.path_epoch,0)
            np.testing.assert_array_equal(engine.path.path,old)

    def test_raw_rejected_anchor_can_recover_in_demo_but_strict_keeps_it(self):
        for policy in ['visible_ground','strict']:
            with tempfile.TemporaryDirectory() as directory:
                engine=self.engine(directory,start_policy=policy)
                meta,overlay,plan=engine.process(self.p());self.assertTrue(meta['active_valid']);engine.commit(plan,overlay)
                old=engine.path.path.copy();real_validate=engine.validate
                def check(grid,full,plan):
                    if plan['anchor'] is not None:return False,'geometry_rejected',dict(failed_edges=[0])
                    return real_validate(grid,full,plan)
                engine.validate=check
                for index in (1,2):meta,overlay,plan=engine.process(self.p(index))
                if policy=='visible_ground':
                    self.assertTrue(meta['active_valid']);self.assertTrue(meta['continuity_restart']);self.assertEqual(meta['path_epoch'],1)
                    with np.load(Path(directory)/'retired_paths/epoch_0000.npz') as d:np.testing.assert_array_equal(d['points'],old)
                else:
                    self.assertFalse(meta['active_valid']);self.assertEqual(meta['path_epoch'],0);np.testing.assert_array_equal(engine.path.path,old)

    def test_evidence_cache_rechecks_a_route_outside_its_original_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            engine=self.engine(directory,auto_lookahead=False)
            context=[];real_validate=engine.validate
            def capture(grid,full,plan):
                context[:]=[grid,full];return real_validate(grid,full,plan)
            engine.validate=capture
            p=self.p();p.update(points=np.vstack((p['points'],[.805,.005,.05])),positive=np.r_[p['positive'],False],negative=np.r_[p['negative'],False])
            meta,overlay,plan=engine.process(p);self.assertTrue(meta['active_valid'])
            # The validated prefix stays inside the same evidence box.
            grid,full=context
            prefix=dict(plan,positions=plan['positions'][:4])
            valid,_,checks=engine.validate(grid,full,prefix)
            self.assertTrue(valid);self.assertTrue(checks['evidence_cache_hit'])
            far=dict(plan,positions=np.array([[.45,.005,0.],[.85,.005,0.]]))
            valid,_,checks=engine.validate(grid,full,far)
            self.assertFalse(valid);self.assertFalse(checks['evidence_cache_hit'])

    def test_new_frame_invalidates_cached_free_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            engine=self.engine(directory,auto_lookahead=False);context=[];real_validate=engine.validate
            def capture(grid,full,plan):
                context[:]=[grid,full];return real_validate(grid,full,plan)
            engine.validate=capture
            meta,_,_=engine.process(self.p());self.assertTrue(meta['active_valid']);old_token=engine.validation_cache['token']
            p=self.p(1);p.update(points=np.vstack((p['points'],[.205,.005,.05])),positive=np.r_[p['positive'],False],negative=np.r_[p['negative'],True])
            engine.process(p);self.assertGreater(engine.validation_cache['token'],old_token)
            grid,full=context
            blocked=dict(positions=np.array([[.105,.005,0.],[.305,.005,0.]]))
            valid,_,_=engine.validate(grid,full,blocked);self.assertFalse(valid)

    def test_atlas_filters_old_window_and_display_preserves_obstacles_and_raw(self):
        p=self.p();p['points']=p['points']+[0.,0.,.06];basis=basis_for(np.array([0.,0.,1.]))
        memory=StableTerrain(layer_filter=True);atlas=GroundAtlas()
        for index in (0,1):
            p['source_index']=index;g=memory.update(fresh_ground(p,p['camera'],basis,.08,.15,layer_filter=True),p,index,.08,.15)
        atlas.update(g,p['camera'])
        far=p|dict(camera=p['camera']+[4.,0.,0.],source_index=2)
        g=memory.update(fresh_ground(far,far['camera'],basis,.08,.15,layer_filter=True),far,2,.08,.15)
        surface=surface_from_overlay(overlay_surface(g)|atlas.overlay())
        points=np.array([[.205,.005,0.],[.205,.005,.06],[.205,.005,.1],[.205,.005,0.]])
        self.assertTrue(obsolete_lower(points,surface)[0])
        raw=dict(points=points,colors=np.array([[30,210,90],[30,210,90],[90,90,90],[255,0,0]],np.uint8),
            current_points=points[1:3],current_colors=np.array([[30,210,90],[90,90,90]],np.uint8))
        clean,colors=planning_view(raw,surface)
        self.assertFalse(((clean==points[0]).all(1)&(colors==[30,210,90]).all(1)).any())
        self.assertTrue((clean==points[2]).all(1).any());self.assertTrue((colors==[255,0,0]).all(1).any())
        original,_=planning_view(raw,surface,True);np.testing.assert_array_equal(original,points)

if __name__=='__main__':unittest.main()
