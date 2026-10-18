from contextlib import nullcontext
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from plugins.small_obstacles.scene_pose import CameraFollower, camera_view
from plugins.small_obstacles.viewer_overview import overview_html, previews_html


class Camera:
    update_timestamp = 1.
    def __init__(self):
        self._position = np.array([3.,3.,3.])
        self.look_at = np.zeros(3)
    @property
    def position(self):
        return self._position
    @position.setter
    def position(self, value):
        self.look_at = self.look_at+np.asarray(value)-self._position
        self._position = np.asarray(value)


class ViewerCameraTests(unittest.TestCase):
    def pose(self, index=0):
        rotation = Rotation.from_euler('xyz', [.3, -.6, .8])
        return {'source_index':index, 'source_time':index*.25, 'position_world':[.1*index,.2,.3],
                'quaternion_wxyz':rotation.as_quat()[[3,0,1,2]].tolist(), 'vertical_fov':.8}

    def test_third_person_view_is_behind_above_and_wider_with_roll(self):
        pose = self.pose(7)
        view = camera_view(pose, 2.)
        rotation = Rotation.from_euler('xyz', [.3,-.6,.8]).as_matrix()
        expected_position = np.asarray(pose['position_world'])-rotation[:,2]*2*1.2-rotation[:,1]*2*.6
        np.testing.assert_allclose(view['position'], expected_position)
        np.testing.assert_allclose(view['look_at'], np.asarray(pose['position_world'])+rotation[:,2]*2*.3)
        np.testing.assert_allclose(view['up'], -rotation[:,1])
        self.assertAlmostEqual(view['fov'], np.deg2rad(80.))
        self.assertGreater(view['fov'], pose['vertical_fov'])
        self.assertGreater(view['near'], 0)

    def test_follow_pause_resume_and_new_video_are_independent_of_viewer_target(self):
        camera = Camera()
        client = SimpleNamespace(camera=camera, atomic=nullcontext)
        clients = {1: client}
        server = SimpleNamespace(initial_camera=SimpleNamespace(), get_clients=lambda:clients)
        follower = CameraFollower(server)
        follower.update(self.pose(), 2.)
        first = camera_view(self.pose(), 2.)
        np.testing.assert_allclose(camera.position, first['position'])
        np.testing.assert_allclose(camera.look_at, first['look_at'])
        camera.position = [7.,8.,9.]
        follower.update(self.pose(1), 2., follow=False)
        np.testing.assert_allclose(camera.position, [7,8,9])
        follower.update(self.pose(1), 2., follow=True)
        np.testing.assert_allclose(camera.position, camera_view(self.pose(1), 2.)['position'])
        new_camera = Camera()
        clients[2] = SimpleNamespace(camera=new_camera, atomic=nullcontext)
        follower.update(self.pose(1), 2., follow=True)
        np.testing.assert_allclose(new_camera.position, camera.position)
        follower.reset()
        follower.update(self.pose(), 1., follow=False)
        np.testing.assert_allclose(camera.position, camera_view(self.pose(), 1.)['position'])
        # Resume at the same frame after free navigation must also reposition.
        camera.position = [4.,5.,6.]
        follower.update(self.pose(), 1., follow=True)
        np.testing.assert_allclose(camera.position, camera_view(self.pose(), 1.)['position'])

    def test_distance_height_and_fov_adjustments_apply_at_same_final_frame(self):
        camera = Camera()
        client = SimpleNamespace(camera=camera, atomic=nullcontext)
        server = SimpleNamespace(initial_camera=SimpleNamespace(), get_clients=lambda:{1:client})
        follower = CameraFollower(server)
        pose = self.pose(2)
        follower.update(pose, 2.)
        old_position = camera.position.copy()
        follower.update(pose, 2., distance=2., height=.8, fov_degrees=95.)
        view = camera_view(pose, 2., distance=2., height=.8, fov_degrees=95.)
        self.assertGreater(np.linalg.norm(camera.position-np.asarray(pose['position_world'])),
                           np.linalg.norm(old_position-np.asarray(pose['position_world'])))
        np.testing.assert_allclose(camera.position, view['position'])
        np.testing.assert_allclose(camera.look_at, view['look_at'])
        self.assertAlmostEqual(camera.fov, np.deg2rad(95.))

    def test_overview_escapes_editable_values_and_encodes_two_frame_previews(self):
        html = overview_html({'vlm_prompt_name':'<script>bad()</script>',
            'prompts':['<img src=x onerror=bad()>']}, directory='<unsafe>', roles=['lingbot','sam'])
        self.assertNotIn('<script>', html)
        self.assertNotIn('<img src=x', html)
        self.assertIn('&lt;script&gt;', html)
        rgb = np.full((28,28,3), 100, np.uint8)
        mask = np.zeros((28,28), bool); mask[0,0] = True
        preview = previews_html(rgb, mask, 2.)
        self.assertEqual(preview.count('data:image/jpeg;base64,'), 2)
        self.assertIn('Mapped frame at 2.00 s', preview)
        self.assertTrue((rgb==100).all())


if __name__=='__main__':
    unittest.main()
