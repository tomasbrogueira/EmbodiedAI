"""A camera-height sweep is much taller than the local surface radius."""

import os
import unittest

import numpy as np

from cost_map import CostMap, _settings
from planner import find_path
import dataset_path as stage


def mapped(obstacle_height):
    angle = .6
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]])
    points = np.array([[0., 0., 0.], [.6, 0., 0.], [.3, 0., obstacle_height]])
    points = points @ rotation.T + [2., -3., .7]
    mask = np.array([True, True, False])
    settings = _settings(dict(radius=1., connection_radius=.61, max_gap=2.,
                              obstacle_margin=.05, clearance_height=15., support_mask=mask))
    return stage.restore_cost_map(points, np.array([1., 1., np.inf]), mask,
                                  rotation[:, 2], settings, CostMap)


@unittest.skipUnless(os.environ.get("DATASET_NATIVE_LIBRARY"), "Native library must be compiled on KTH")
class CameraClearanceNativeTests(unittest.TestCase):
    def test_tall_sweep_blocks_high_obstacle_but_preserves_ceiling_clearance(self):
        for obstacle_height, expected in ((14.9, "partial"), (15.06, "ok")):
            with self.subTest(obstacle_height=obstacle_height):
                cm = mapped(obstacle_height)
                self.assertTrue(cm.obstacle_free(cm.points[0], cm.points[0]))
                self.assertTrue(cm.obstacle_free(cm.points[1], cm.points[1]))
                self.assertEqual(cm._connected(0, 1), expected == "ok")
                native = stage.NativeSearch(os.environ["DATASET_NATIVE_LIBRARY"])
                self.assertTrue(native.supports_clearance)
                result = stage.exact_route(cm, 0, 1, np.array([0, 1]), native, find_path,
                                           lambda *a, **k: None, allow_partial=True)
                self.assertEqual(result.status, expected)
                np.testing.assert_array_equal(result.ids, [0] if expected == "partial" else [0, 1])
                self.assertAlmostEqual(result.total_cost, 0. if expected == "partial" else .6)


if __name__ == "__main__":
    unittest.main()
