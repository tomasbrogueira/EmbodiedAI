"""Focused exclusion, visibility and connectivity regressions; no models."""
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from tests.test_local_planning import packet
from path_planner.fast_slope import SourceSlope,prepare_slopes,screen_grid
from path_planner.local_grid import basis_for,fresh_ground,Search,PathState
from path_planner.local_route import propose
from path_planner.terrain_reasons import SEMANTIC,GEOMETRY,UNKNOWN_SLOPE,STEEP
from orchestrator.local_evidence import StableTerrain,PacketIndex
from orchestrator.live_planning import LocalEngine
from orchestrator.visibility import RayEvidence
from visualizer.local_planning import terrain_colors

def image_packet(angle=0.):
    y,x=np.mgrid[:41,:81];points=np.stack((x*.002,y*.002,x*.002*np.tan(angle)),axis=-1)
    points[:,:,2]+=np.random.default_rng(1).normal(0,.0002,x.shape)
    return dict(points=points.reshape(-1,3),pixels=np.arange(x.size),shape=x.shape,
        positive=np.ones(x.size,bool),negative=np.zeros(x.size,bool))

class FastTerrainTests(unittest.TestCase):
    def test_lightweight_source_screen_accepts_noisy_flat_and_rejects_steep(self):
        for angle in [0.,np.pi/4]:
            p=image_packet(angle);screen=SourceSlope(p,np.array([0.,0.,1.]),.08)
            rows=np.arange(len(p['points']));slopes=screen.slopes(rows)
            self.assertTrue(np.isfinite(slopes).all())
            if angle:self.assertGreater(np.median(slopes),np.pi/6)
            else:self.assertLess(np.median(slopes),np.radians(5))

    def test_missing_source_neighbors_stays_unknown(self):
        p=image_packet();p['positive'][:]=False;p['positive'][100]=True
        self.assertTrue(np.isnan(SourceSlope(p,np.array([0.,0.,1.]),.08).slopes([100])[0]))

    def test_compatible_ground_refreshes_source_coordinates_and_slopes(self):
        p=packet();basis=basis_for(np.array([0.,0.,1.]));memory=StableTerrain()
        first=fresh_ground(p,p['camera'],basis,.08,.15);memory.update(first,p,0,.08,.15)
        later=p|dict(points=p['points']+[0.,0.,.003],source_index=7)
        fresh=fresh_ground(later,p['camera'],basis,.08,.15);fresh['slopes']=np.full(fresh['cost'].size,.1)
        grid=memory.update(fresh,later,1,.08,.15);observed=grid['point_rows']>=0
        self.assertTrue((grid['source_ticks'][observed]==1).all())
        self.assertTrue((grid['source_frames'][observed]==7).all())
        np.testing.assert_allclose(grid['points'][observed,2],.003)
        np.testing.assert_allclose(grid['slopes'][observed],.1)

    def occupied(self,negative=False):
        p=packet();basis=basis_for(np.array([0.,0.,1.]));memory=StableTerrain()
        points=np.vstack((p['points'],[.005,.005,.04]));pos=np.r_[p['positive'],False];neg=np.r_[p['negative'],negative]
        old=p|dict(points=points,positive=pos,negative=neg)
        def update(p,tick):return memory.update(fresh_ground(p,p['camera'],basis,.08,.15),p,tick,.08,.15)
        update(old,0);return p,memory,update

    def test_two_observed_free_rays_recover_geometry(self):
        p,memory,update=self.occupied();occupied=memory.state['occupied']>=0;self.assertTrue(occupied.any())
        update(p,1);self.assertTrue((memory.state['occupied'][occupied]>=0).all())
        grid=update(p,2);self.assertTrue((memory.state['occupied'][occupied]<0).all())
        self.assertFalse((grid['reasons'][occupied]&GEOMETRY).any())

    def test_positive_ground_does_not_clear_semantic_negative(self):
        p,memory,update=self.occupied(True);update(p,1);grid=update(p,2)
        veto=memory.state['veto']>=0;self.assertTrue(veto.any())
        self.assertFalse(np.isfinite(grid['cost'][veto]).any());self.assertTrue((grid['reasons'][veto]&SEMANTIC).all())

    def test_repeated_frame_is_not_two_visibility_witnesses(self):
        p,memory,update=self.occupied();update(p|dict(source_index=7),1);update(p|dict(source_index=7),2)
        self.assertTrue((memory.state['occupied']>=0).any())
        update(p|dict(source_index=14),3);self.assertFalse((memory.state['occupied']>=0).any())

    def test_ground_without_a_ray_through_obstacle_is_not_free_volume(self):
        p,memory,update=self.occupied();take=np.linalg.norm(p['points'][:,:2]-[.005,.005],axis=1)<.02
        limited=p|dict(points=p['points'][take],positive=p['positive'][take],negative=p['negative'][take],camera=p['camera']+[.3,0.,0.])
        self.assertFalse(RayEvidence(limited).clears(np.array([[.005,.005,.04]]),.004)[0])
        update(limited,1);update(limited,2);self.assertTrue((memory.state['occupied']>=0).any())

    def test_longer_grid_connections_never_skip_unobserved_strip(self):
        cost=np.ones((12,12));cost[:,5]=np.inf
        result=Search()(cost,np.zeros_like(cost),4*12+3,.01,False,max_stride=4)
        self.assertFalse(result[3].reshape(12,12)[:,6:].any())

    def test_longer_grid_connections_respect_steep_height_barrier(self):
        cost=np.ones((12,12));height=np.zeros_like(cost);height[:,5:]=1.
        result=Search()(cost,height,4*12+3,.01,True,max_stride=4)
        self.assertFalse(result[3].reshape(12,12)[:,5:].any())

    def test_longer_grid_connections_do_not_cut_diagonal_corners(self):
        cost=np.full((12,12),np.inf);cost[1,1]=cost[2,2]=cost[3,3]=1.
        result=Search()(cost,np.zeros_like(cost),13,.01,False,max_stride=4)
        self.assertEqual(result[3].sum(),1)

    def test_unknown_steep_and_semantic_obstacle_have_distinct_colors(self):
        d=dict(costs=np.array([1.,np.inf,np.inf,np.inf]),slopes=np.array([0.,np.nan,.8,np.nan]),
            slopes_enabled=np.array(True),reasons=np.array([0,UNKNOWN_SLOPE,STEEP,SEMANTIC|UNKNOWN_SLOPE],np.uint8))
        colours=terrain_colors(d,'Traversability')
        np.testing.assert_array_equal(colours,[[30,210,90],[135,145,155],[235,155,35],[230,50,50]])

    def test_partial_fallback_validates_and_stops_before_failed_edge(self):
        p=packet();normal=np.array([0.,0.,1.]);basis=basis_for(normal);radius=.08
        grid=StableTerrain().update(fresh_ground(p,p['camera'],basis,radius,.15),p,0,radius,.15)
        plan=propose(grid,p['camera'],p['forward'],Search(),PathState(),False)
        original=plan['positions'].copy()
        with tempfile.TemporaryDirectory() as directory:
            engine=LocalEngine(SimpleNamespace(plan=True,steepness=False,validation_backend='cpu'),Path(directory))
            engine.reference=normal;engine.radius=radius;engine.height=.15;engine.basis=basis
            engine.history=[dict(tick=0,p=p,index=PacketIndex(p,basis,radius))]
            result=engine.validated_prefix(grid,np.zeros(grid['cost'].size),plan,dict(failed_edges=[5]))
            self.assertIsNotNone(result);partial,checks=result
            self.assertEqual(len(partial['positions']),6);self.assertFalse(partial['goal_reached'])
            self.assertEqual(checks['failed_edges'],[])
            np.testing.assert_array_equal(plan['positions'],original)
            self.assertIsNone(engine.validated_prefix(grid,np.zeros(grid['cost'].size),plan,dict(failed_edges=[0])))

if __name__=='__main__':unittest.main()
