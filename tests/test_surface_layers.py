"""Confirmed surface-layer filtering and inferred-path epoch regressions."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from tests.test_local_planning import packet
from path_planner.local_grid import basis_for,fresh_ground
from path_planner.surface_layers import obsolete_lower
from orchestrator.local_evidence import StableTerrain
from orchestrator.live_planning import LocalEngine

class LayerTests(unittest.TestCase):
    def setUp(self):
        self.p=packet();self.basis=basis_for(np.array([0.,0.,1.]));self.radius=.08

    def fresh(self,p):return fresh_ground(p,p['camera'],self.basis,.08,.15,layer_filter=True)

    def stacked(self,camera_height=.3):
        points=np.vstack((self.p['points'],self.p['points']+[0.,0.,.06]));n=len(points)
        return self.p|dict(points=points,positive=np.ones(n,bool),negative=np.zeros(n,bool),camera=np.array([.005,.005,camera_height]))

    def test_overlapping_supported_layers_use_actual_upper_points_not_mean(self):
        p=self.stacked();grid=self.fresh(p);observed=grid['point_rows']>=0
        self.assertTrue(observed.any());np.testing.assert_allclose(grid['points'][observed,2],.06)
        np.testing.assert_array_equal(grid['points'][observed],p['points'][grid['point_rows'][observed]])
        self.assertTrue((grid['point_rows'][observed]>=len(self.p['points'])).all())

    def test_surface_above_camera_cannot_replace_ground_under_an_overpass(self):
        grid=self.fresh(self.stacked(.03));observed=grid['point_rows']>=0
        np.testing.assert_allclose(grid['points'][observed,2],0.)

    def test_changed_layer_needs_two_distinct_frames(self):
        memory=StableTerrain(layer_filter=True)
        memory.update(self.fresh(self.p),self.p,0,.08,.15)
        upper=self.p|dict(points=self.p['points']+[0.,0.,.06],source_index=7)
        pending=memory.update(self.fresh(upper),upper,1,.08,.15)
        self.assertFalse(np.isfinite(pending['cost']).any())
        repeated=memory.update(self.fresh(upper),upper,2,.08,.15)
        self.assertFalse(np.isfinite(repeated['cost']).any())
        upper=upper|dict(source_index=14);confirmed=memory.update(self.fresh(upper),upper,3,.08,.15)
        self.assertTrue(np.isfinite(confirmed['cost']).any());self.assertTrue(confirmed['layer_confirmed'].any())

    def test_lower_copy_does_not_overwrite_confirmed_upper_surface(self):
        memory=StableTerrain(layer_filter=True);upper=self.p|dict(points=self.p['points']+[0.,0.,.06])
        for tick in [0,1]:grid=memory.update(self.fresh(upper),upper,tick,.08,.15)
        lower=memory.update(self.fresh(self.p),self.p,2,.08,.15);observed=lower['point_rows']>=0
        np.testing.assert_allclose(lower['points'][observed,2],.06)
        self.assertGreater(lower['lower_layer_updates_ignored'],0)
        self.assertTrue(obsolete_lower(self.p['points'],lower).any())

    def test_lower_ground_is_allowed_when_camera_moves_under_previous_surface(self):
        memory=StableTerrain(layer_filter=True);upper=self.p|dict(points=self.p['points']+[0.,0.,.06])
        for tick in [0,1]:memory.update(self.fresh(upper),upper,tick,.08,.15)
        below=self.p|dict(camera=np.array([.005,.005,.03]))
        for tick in [2,3]:grid=memory.update(self.fresh(below),below,tick,.08,.15)
        observed=grid['point_rows']>=0;np.testing.assert_allclose(grid['points'][observed,2],0.)

    def test_negative_below_corrected_layer_still_vetoes_column(self):
        memory=StableTerrain(layer_filter=True);p=self.p.copy();p['negative']=(abs(p['points'][:,0]-.205)<.006)&(abs(p['points'][:,1])<.05)
        memory.update(self.fresh(p),p,0,.08,.15)
        upper=self.p|dict(points=self.p['points']+[0.,0.,.06])
        for tick in [1,2]:grid=memory.update(self.fresh(upper),upper,tick,.08,.15)
        veto=memory.state['veto']>=0;self.assertTrue(veto.any());self.assertFalse(np.isfinite(grid['cost'][veto]).any())

    def test_layer_correction_replans_in_new_epoch_without_rewriting_old_path(self):
        args=SimpleNamespace(plan=True,steepness=False,cost_map=True,validation_backend='cpu',start_policy='visible_ground',
            lookahead=6.,auto_lookahead=False,planning_budget_ms=40.,filter_lower_layers=True,replan_layer_changes=True)
        with tempfile.TemporaryDirectory() as directory:
            engine=LocalEngine(args,Path(directory));engine.reference=np.array([0.,0.,1.]);engine.radius=.08;engine.height=.15;engine.basis=self.basis
            original_points=self.p['points'].copy()
            p=self.p|dict(revision=1,source_index=0,source_time=0.)
            meta,overlay,plan=engine.process(p);self.assertTrue(meta['active_valid']);engine.commit(plan,overlay);old_path=engine.path.path.copy()
            for i in [1,2]:
                p=self.p|dict(points=self.p['points']+[0.,0.,.06],revision=i+1,source_index=i,source_time=float(i))
                meta,overlay,plan=engine.process(p)
                engine.commit(plan,overlay)
            self.assertTrue(meta['active_valid']);self.assertTrue(meta['layer_corrected']);self.assertEqual(meta['path_epoch'],1)
            self.assertGreater(meta['geometry_validation']['lower_layer_points_removed'],0)
            np.testing.assert_allclose(plan['positions'][:,2],.06)
            np.testing.assert_array_equal(self.p['points'],original_points)
            with np.load(Path(directory)/'retired_paths/epoch_0000.npz') as data:np.testing.assert_array_equal(data['points'],old_path)

if __name__=='__main__':unittest.main()
