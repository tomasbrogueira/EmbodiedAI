"""Focused numerical/scheduling checks. No video, model loading or full pipeline."""
import tempfile
import json
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from orchestrator.local_evidence import StableTerrain
from orchestrator.live_planning import LivePlanning, LocalEngine
from path_planner.local_grid import basis_for, fresh_ground, PathState, Search, certify
from path_planner.local_route import propose, start_node
from path_planner.final_grid import final_proposal
from orchestrator.fast_final import FastFinal


def packet():
    cell=.01
    x,y=np.meshgrid(np.arange(-40,141)*cell+.005,np.arange(-60,61)*cell+.005)
    points=np.concatenate([np.c_[x.ravel()+dx,y.ravel()+dy,np.zeros(x.size)]
        for dx,dy in [(0.,0.),(.001,.001),(-.001,-.001)]]).astype(np.float32)
    return dict(points=points,positive=np.ones(len(points),bool),negative=np.zeros(len(points),bool),
                camera=np.array([.005,.005,.3]),forward=np.array([1.,0.,0.]))


class LocalPlanningTests(unittest.TestCase):
    def setUp(self):
        self.p=packet(); self.normal=np.array([0.,0.,1.]); self.basis=basis_for(self.normal)
        self.memory=StableTerrain(); self.search=Search()

    def grid(self,p,tick):
        fresh=fresh_ground(p,p['camera'],self.basis,.08,.15)
        return self.memory.update(fresh,p,tick,.08,.15)

    def test_current_ground_never_skips_a_blocked_nearest_sample(self):
        grid=self.grid(self.p,0)
        start,reason=start_node(grid,self.p['camera'],'strict'); self.assertEqual(reason,'ok')
        grid['cost'].ravel()[start]=np.inf
        self.assertEqual(start_node(grid,self.p['camera'],'strict')[1],'current_ground_blocked')

    def test_goal_is_camera_relative_and_suffix_preserves_shared_prefix(self):
        state=PathState(); grid=self.grid(self.p,0)
        plan=propose(grid,self.p['camera'],self.p['forward'],self.search,state,False,lookahead=6.)
        self.assertGreater(len(plan['positions']),2)
        self.assertLessEqual((plan['goal']-self.p['camera'])@self.p['forward'],6*.08)
        state.splice(None,plan['positions']); original=state.path.copy()
        camera=plan['positions'][5]+[0.,0.,.3]
        next_p=self.p|dict(camera=camera); grid=self.grid(next_p,1)
        next_plan=propose(grid,camera,self.p['forward'],self.search,state,False,lookahead=6.)
        self.assertIsNotNone(next_plan['anchor'])
        np.testing.assert_array_equal(next_plan['positions'][0],original[next_plan['anchor']])
        state.splice(next_plan['anchor'],next_plan['positions'])
        np.testing.assert_array_equal(state.path[:state.committed+1],original[:state.committed+1])
        valid,_=certify(self.p['points'],self.p['positive'],self.p['negative'],np.zeros(len(self.p['points'])),
                        next_plan['positions'],self.normal,.08,.15,False)
        self.assertTrue(valid)

    def test_negative_veto_survives_witness_expiry_and_new_positives(self):
        p=self.p.copy(); p['negative']=(abs(p['points'][:,0]-.205)<.006)&(abs(p['points'][:,1])<.05)
        self.grid(p,0); grid=self.grid(self.p,21)
        veto=self.memory.state['veto']>=0
        self.assertTrue(veto.any()); self.assertFalse(np.isfinite(grid['cost'][veto]).any())

    def test_height_disagreement_is_not_a_permanent_semantic_veto(self):
        self.grid(self.p,0)
        raised=self.p|dict(points=self.p['points']+[0.,0.,.05])
        conflict=self.grid(raised,1)
        cells=self.memory.state['conflict']==1
        self.assertTrue(cells.any());self.assertFalse(np.isfinite(conflict['cost'][cells]).any())
        confirmed=self.grid(raised,2)
        self.assertTrue(np.isfinite(confirmed['cost'][cells]).any())
        self.assertFalse((self.memory.state['veto'][cells]>=0).any())

    def test_final_grid_does_not_cross_disconnected_components(self):
        grid=self.grid(self.p,0)
        wall=int(np.floor(.2/.01)-grid['origin'][0]);grid['cost'][:,wall]=np.inf
        cameras=np.array([self.p['camera'],self.p['camera']+[.4,0.,0.]])
        plan=final_proposal(grid,cameras,self.search,False,policy='strict')
        self.assertEqual(plan['status'],'partial_candidate');self.assertFalse(plan['goal_reached'])
        self.assertTrue((plan['positions'][:,0]<.2).all())

    def test_fast_final_uses_bounded_grid_without_exact_analysis(self):
        args=SimpleNamespace(plan=True,steepness=True,cost_map=True,reference_source='base_support',
            reference=None,radius=.08,clearance_height=.15,validation_backend='cpu',start_policy='strict')
        with tempfile.TemporaryDirectory() as directory:
            final=FastFinal(args,Path(directory))
            for i in range(3):
                p=self.p|dict(camera=self.p['camera']+[i*.06,0.,0.],source_index=i,source_time=float(i),
                    pixels=np.arange(len(self.p['points'])),shape=(363,181),base_support_mask=self.p['positive'])
                final.append(p)
            with patch('orchestrator.analysis.costs',side_effect=AssertionError('Dense costs called')),patch('orchestrator.analysis.route',side_effect=AssertionError('Exact search called')),patch('path_planner.slope.estimate_slope',side_effect=AssertionError('Full slope calculation called')),patch('path_planner.local_grid.Refiner.slopes',side_effect=AssertionError('Robust KD refiner called')):
                result=final.finish()
            self.assertTrue(result['active_valid']);self.assertTrue(result['goal_reached'])
            self.assertLessEqual(result['grid_capacity'],512*512)

    def test_missing_connection_returns_unavailable(self):
        state=PathState(); state.path=np.array([[20.,20.,0.],[21.,20.,0.]])
        plan=propose(self.grid(self.p,0),self.p['camera'],self.p['forward'],self.search,state,False)
        self.assertEqual(plan['status'],'connection_lost'); self.assertEqual(len(plan['positions']),0)

    def test_visible_demo_skips_an_isolated_nearest_island(self):
        grid=self.grid(self.p,0); start,_=start_node(grid,self.p['camera'],'visible_ground')
        y,x=divmod(start,200)
        grid['cost'][y-2:y+3,x-2:x+3]=np.inf; grid['cost'].ravel()[start]=1
        result=self.search(grid['cost'],grid['height'],start,.01,False)
        self.assertEqual(int(result[3].sum()),1)
        plan=propose(grid,self.p['camera'],self.p['forward'],self.search,PathState(),False,
                     policy='visible_ground',lookahead=6.)
        self.assertGreater(len(plan['positions']),2)
        self.assertGreaterEqual((plan['goal']-plan['positions'][0])@self.p['forward'],.08/2)

    def test_anchor_can_join_a_nearby_cell_only_through_validated_geometry(self):
        grid=self.grid(self.p,0); state=PathState()
        initial=propose(grid,self.p['camera'],self.p['forward'],self.search,state,False,lookahead=6.)
        state.splice(None,initial['positions']); old=state.path.copy()
        xy=np.floor((old@self.basis)[:,:2]/.01).astype(int)-grid['origin']
        grid['cost'][xy[:,1],xy[:,0]]=np.inf
        plan=propose(grid,self.p['camera'],self.p['forward'],self.search,state,False,
                     policy='visible_ground',lookahead=6.)
        self.assertIsNotNone(plan['anchor'])
        np.testing.assert_array_equal(plan['positions'][0],old[plan['anchor']])
        self.assertGreater(np.linalg.norm(plan['positions'][1]-plan['positions'][0]),0)
        valid,_=certify(self.p['points'],self.p['positive'],self.p['negative'],np.zeros(len(self.p['points'])),
                        plan['positions'],self.normal,.08,.15,False)
        self.assertTrue(valid)
        grid['hard_veto'][xy[:,1],xy[:,0]]=True
        blocked=propose(grid,self.p['camera'],self.p['forward'],self.search,state,False,
                        policy='visible_ground',lookahead=6.)
        self.assertEqual(blocked['status'],'connection_lost')

    def test_search_does_not_cut_blocked_diagonal_corners(self):
        cost=np.full((5,5),np.inf); cost[1,1]=cost[2,2]=1
        result=self.search(cost,np.zeros_like(cost),6,.01,False)
        self.assertFalse(result[3][12])

    def test_cost_only_does_not_create_a_route_or_cuda_worker(self):
        args=SimpleNamespace(plan=False,steepness=False,cost_map=True,reference_source='base_support',
                             validation_backend='auto',start_policy='strict')
        with tempfile.TemporaryDirectory() as directory:
            engine=LocalEngine(args,Path(directory)); engine.reference=self.normal; engine.radius=.08;engine.height=.15;engine.basis=self.basis
            p=self.p|dict(revision=1,source_index=0,source_time=0.)
            meta,overlay,plan=engine.process(p)
            self.assertEqual(meta['status'],'costs_ready'); self.assertIsNone(plan);self.assertIsNone(engine.gpu)
            self.assertEqual(len(overlay['route']),0); self.assertTrue(np.isfinite(overlay['costs']).any())

    def test_latest_packet_supersedes_an_old_result(self):
        started=threading.Event(); release=threading.Event(); visited=[]
        class FakeEngine:
            def __init__(self,*args): pass
            def process(self,p):
                visited.append(p['revision'])
                if p['revision']==1: started.set(); release.wait(2)
                return dict(status='costs_ready',revision=p['revision'],active_valid=False),None,None
            def commit(self,*args): pass
            def close(self): pass
        with tempfile.TemporaryDirectory() as directory,patch('orchestrator.live_planning.LocalEngine',FakeEngine):
            runner=LivePlanning(None,Path(directory)); runner.submit({})
            self.assertTrue(started.wait(1));runner.submit({});runner.submit({})
            self.assertFalse(runner.snapshot()['active_valid']); release.set(); runner.close()
            self.assertEqual(visited,[1,3]);self.assertEqual(runner.snapshot()['revision'],3)

    def test_ready_status_is_published_between_mapping_packets(self):
        started=threading.Event(); release=threading.Event()
        class FakeEngine:
            def __init__(self,*args): pass
            def process(self,p):
                started.set(); release.wait(2)
                return dict(status='ready',revision=p['revision'],active_valid=True),None,None
            def commit(self,*args): pass
            def close(self): pass
        with tempfile.TemporaryDirectory() as directory,patch('orchestrator.live_planning.LocalEngine',FakeEngine):
            runner=LivePlanning(None,Path(directory)); runner.submit({})
            self.assertTrue(started.wait(1))
            status=Path(directory)/'local_planning/status.json'
            self.assertFalse(json.loads(status.read_text())['active_valid'])
            release.set(); runner.close()
            # No subsequent perception packet was needed to publish readiness.
            self.assertTrue(json.loads(status.read_text())['active_valid'])

    def test_delayed_validated_path_is_published_as_history_while_capture_continues(self):
        first_started=threading.Event();second_started=threading.Event()
        release_first=threading.Event();release_second=threading.Event();committed=[]
        route=np.array([[0.,0.,0.],[.3,0.,0.]])
        class FakeEngine:
            def __init__(self,*args):pass
            def process(self,p):
                index=p['revision']
                if index==1:first_started.set();release_first.wait(2)
                else:second_started.set();release_second.wait(2)
                return (dict(status='ready',revision=index,active_valid=True),
                    dict(route=route,revision=np.array(index)),dict(revision=index,positions=route))
            def commit(self,plan,overlay):
                committed.append(plan['revision']);overlay['last_route']=plan['positions'].copy()
            def close(self):pass
        with tempfile.TemporaryDirectory() as directory,patch('orchestrator.live_planning.LocalEngine',FakeEngine):
            runner=LivePlanning(None,Path(directory))
            try:
                runner.submit({});self.assertTrue(first_started.wait(1));runner.submit({});release_first.set()
                self.assertTrue(second_started.wait(1))
                metadata=runner.snapshot();self.assertEqual(metadata['status'],'updating');self.assertFalse(metadata['active_valid'])
                self.assertEqual(metadata['latest_revision'],2);self.assertEqual(metadata['last_completed_revision'],1)
                self.assertEqual(committed,[1])
                with np.load(Path(directory)/'local_planning/display.npz') as d:np.testing.assert_array_equal(d['last_route'],route)
            finally:release_first.set();release_second.set();runner.close()

if __name__=='__main__': unittest.main()
