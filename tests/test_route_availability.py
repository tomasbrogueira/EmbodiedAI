"""Calibration, partial-route and viewer regressions without model inference."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from path_planner.floor_reference import estimate_floor_reference,estimate_local_floor_reference
from path_planner.local_grid import basis_for
from orchestrator.live_planning import LocalEngine
from visualizer.local_planning import LocalOverlay,route_message
from tests.test_local_planning import packet

class AvailabilityTests(unittest.TestCase):
    def frames(self):
        y,x=np.mgrid[:256,:256];x=(x-128)*.002;y=(y-128)*.002
        points=np.stack((x,y,.8*(x*x+y*y)),axis=-1)
        return [dict(id=i,points=points,floor_mask=np.ones(x.shape,bool),camera=np.array([0.,0.,.5])) for i in range(3)]

    def test_curved_whole_ground_can_bootstrap_from_consistent_local_patches(self):
        frames=self.frames();normal,report=estimate_floor_reference(frames)
        self.assertIsNone(normal)
        local,details=estimate_local_floor_reference(frames,report)
        self.assertIsNotNone(local);self.assertEqual(details['reason'],'ok')
        self.assertEqual(len([f for f in details['frames'] if f['reason']=='accepted']),3)
        self.assertEqual(details['method'],'local ground patches')

    def test_one_frame_cannot_supply_three_local_reference_observations(self):
        frame=self.frames()[0];normal,report=estimate_local_floor_reference([frame,frame,frame])
        self.assertIsNone(normal);self.assertEqual(report['reason'],'insufficient_reference_frames')

    def test_auto_window_reaches_observed_ground_without_faking_camera_support(self):
        p=packet();take=(p['points'][:,0]>.9)&(p['points'][:,0]<1.3)
        p=p|dict(points=p['points'][take],positive=p['positive'][take],negative=p['negative'][take],revision=1,source_index=0,source_time=0.)
        base=dict(plan=True,steepness=False,cost_map=True,validation_backend='cpu',start_policy='visible_ground',lookahead=6.,planning_budget_ms=40.)
        with tempfile.TemporaryDirectory() as directory:
            fixed=LocalEngine(SimpleNamespace(**base,auto_lookahead=False),Path(directory))
            auto=LocalEngine(SimpleNamespace(**base,auto_lookahead=True),Path(directory))
            for engine in (fixed,auto):
                engine.reference=np.array([0.,0.,1.]);engine.radius=.08;engine.height=.15;engine.basis=basis_for(engine.reference)
            rejected,_,_=fixed.process(p);self.assertFalse(rejected['active_valid'])
            ready,overlay,plan=auto.process(p);self.assertTrue(ready['active_valid'])
            self.assertGreater(ready['effective_lookahead'],6.)
            self.assertGreater(plan['positions'][0,0],.9)
            self.assertGreater((plan['goal']-p['camera'])@p['forward'],6*.08)

    def test_live_partial_route_is_committed_only_after_validation(self):
        p=packet()|dict(revision=1,source_index=0,source_time=0.)
        args=SimpleNamespace(plan=True,steepness=False,cost_map=True,validation_backend='cpu',start_policy='strict',lookahead=6.,planning_budget_ms=40.)
        with tempfile.TemporaryDirectory() as directory:
            engine=LocalEngine(args,Path(directory));engine.reference=np.array([0.,0.,1.]);engine.radius=.08;engine.height=.15;engine.basis=basis_for(engine.reference)
            real_validate=engine.validate
            def candidate_check(grid,full,plan):
                if len(plan['positions'])>6:return False,'geometry_rejected',dict(failed_edges=[5],failed_nodes=[])
                return real_validate(grid,full,plan)
            engine.validate=candidate_check
            meta,overlay,plan=engine.process(p)
            self.assertEqual(meta['status'],'partial_ready');self.assertTrue(meta['active_valid'])
            self.assertIsNone(engine.last_validated)
            engine.commit(plan,overlay)
            np.testing.assert_array_equal(overlay['last_route'],overlay['route'])
            self.assertEqual(len(plan['positions']),6)

    def test_status_explains_missing_reference_and_rejected_routes(self):
        text=route_message(dict(status='waiting_for_reference',reference_diagnostics=dict(frames=[dict(reason='accepted')])))
        self.assertIn('1/3 suitable frames',text)
        self.assertIn('support, clearance or grade',route_message(dict(status='geometry_rejected')))
        self.assertIn('previous validated',route_message(dict(status='updating'),has_history=True))

    def test_inactive_current_route_is_hidden_while_history_remains_visible(self):
        overlay=LocalOverlay(SimpleNamespace());overlay.handles={name:SimpleNamespace(visible=True) for name in ['grid','route','goal','history','last_route']}
        overlay.visibility(True,True,False)
        self.assertFalse(overlay.handles['route'].visible);self.assertFalse(overlay.handles['goal'].visible)
        self.assertTrue(overlay.handles['history'].visible);self.assertTrue(overlay.handles['last_route'].visible)
        overlay.visibility(True,True,True);self.assertTrue(overlay.handles['route'].visible);self.assertFalse(overlay.handles['last_route'].visible)

if __name__=='__main__':unittest.main()
