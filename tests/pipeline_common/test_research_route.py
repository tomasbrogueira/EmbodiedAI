import unittest
from copy import deepcopy
import numpy as np
from pipeline_common.research_route import forward_corridor_route,segment_supported
from pipeline_common.research_route_visuals import route_arrow_segments
from derive_pose_corrected_run import corrected_numeric


class ResearchMissionTests(unittest.TestCase):
    def fixture(self):
        shape=(70,30)
        a={'origin':np.array([-1.5,0.]),'resolution':np.array([.1]),'projection_basis':np.eye(3),
           'decision_state':np.full(shape,2,np.uint8),'geometry_state':np.full(shape,2,np.uint8),
           'policy_blocked_mask':np.zeros(shape,bool),'support_height':np.zeros(shape),'costs':np.ones(shape)}
        m={'footprint_radius_with_clearance':.12,'profile':{'max_slope_degrees':25.,'max_step':.08}}
        cameras=np.array([[0.,0.,1.5],[0.,4.5,1.5]])
        rotations=np.tile(np.array([[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]]),(2,1,1))
        mission={'camera_lookahead_m':1.5,'max_endpoint_adjustment_m':.5,'minimum_progress_m':.25}
        return a,m,cameras,rotations,mission

    def test_forward_objective_has_continuous_route_and_goal(self):
        a,m,cameras,rotations,settings=self.fixture()
        r=forward_corridor_route(a,m,cameras,rotations,settings)
        self.assertEqual(r['status'],'ok');self.assertEqual(len(r['path']),2)
        self.assertGreater(r['goal'][1],r['start'][1])
        self.assertTrue(segment_supported(*np.array(r['path']),a,m))
        arrows=route_arrow_segments(r['path'],[0.,0.,1.],1.)
        self.assertTrue(np.all(arrows[:,1,1]>arrows[:,0,1]))

    def test_unknown_band_does_not_switch_to_another_component(self):
        a,m,cameras,rotations,settings=self.fixture()
        a['geometry_state'][30:33]=0;a['decision_state'][28:35]=0
        r=forward_corridor_route(a,m,cameras,rotations,settings)
        self.assertEqual(r['status'],'no_path');self.assertEqual(r['path'],[])

    def test_shortcut_cannot_clip_an_obstacle_or_unknown_with_footprint(self):
        a,m,*_=self.fixture();a['geometry_state'][30,15]=1
        self.assertFalse(segment_supported(np.array([-.1,1.5,0.]),np.array([-.1,6.,0.]),a,m))
        a['geometry_state'][30,15]=0
        self.assertFalse(segment_supported(np.array([-.1,1.5,0.]),np.array([-.1,6.,0.]),a,m))

    def test_backwards_pose_track_is_blocked_not_reversed(self):
        a,m,cameras,rotations,settings=self.fixture();cameras[1,1]=-4.5
        r=forward_corridor_route(a,m,cameras,rotations,settings)
        self.assertEqual(r['status'],'blocked_inputs');self.assertEqual(r['path'],[])

    def test_corrected_geometry_tracks_static_landmark_across_camera_motion(self):
        # C2W camera advances +Z; unchanged depth of one static landmark shrinks.
        e=np.tile(np.eye(4)[:3],(2,1,1));e[1,2,3]=1.
        g={'images':np.zeros((2,2,2,3),np.uint8),'extrinsic':e.astype(np.float32),
           'intrinsic':np.tile(np.eye(3,dtype=np.float32),(2,1,1)),
           'depth':np.stack([np.full((2,2,1),4.),np.full((2,2,1),3.)]).astype(np.float32),
           'world_points_conf':np.full((2,2,2),2.,np.float32),'world_points':np.zeros((2,2,2,3),np.float32)}
        original=deepcopy(g);out=corrected_numeric(g)
        np.testing.assert_array_equal(out['world_points'][:,0,0],[[0,0,4],[0,0,4]])
        np.testing.assert_array_equal(out['extrinsic'][1,:,3],[0,0,-1])
        for key in g:np.testing.assert_array_equal(g[key],original[key])
        for key in ('depth','intrinsic','images','world_points_conf'):np.testing.assert_array_equal(out[key],g[key])
