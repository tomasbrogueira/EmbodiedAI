"""Run with DATASET_NATIVE_LIBRARY set to the staged compiled library."""
import os
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(os.environ.get('DATASET_REPO', Path.cwd())) / 'path-planning'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cost_map import CostMap, _settings, obstacle_distance_squared
from planner import find_path
import dataset_path as stage


@unittest.skipUnless(os.environ.get('DATASET_NATIVE_LIBRARY'), 'Native library must be compiled on KTH')
class NativeClearanceTests(unittest.TestCase):
    def native(self):
        native = stage.NativeSearch(os.environ['DATASET_NATIVE_LIBRARY'])
        self.assertTrue(native.supports_clearance, 'Tests require the new clearance export')
        return native

    def mapped(self, points, up=(0., 0., 1.), margin=.05):
        points = np.asarray(points, dtype=float)
        mask = np.array([True, True, False])
        settings = _settings(dict(radius=1., connection_radius=.61, max_gap=2.,
                                  obstacle_margin=margin, support_mask=mask))
        return stage.restore_cost_map(points, np.array([1., 1., np.inf]), mask,
                                      np.asarray(up), settings, CostMap)

    def search(self, cm):
        native = self.native()
        native.prepare_map(cm, cm.support_rows)
        c = cm.config
        return native(cm.points[:2], cm.cost[:2], cm.reference, c['connection_radius'],
                      c['max_slope'], 1., 10., 0, 1, set(), max_gap=c['max_gap'],
                      obstacle_margin=c['obstacle_margin'], clearance_height=c['clearance_height'])

    def test_overhead_blocks_continuous_edge_between_clear_endpoints(self):
        cm = self.mapped([[0, 0, 0], [.6, 0, 0], [.3, 0, .9]])
        self.assertTrue(cm.obstacle_free(cm.points[0], cm.points[0]))
        self.assertTrue(cm.obstacle_free(cm.points[1], cm.points[1]))
        self.assertFalse(cm._connected(0, 1))
        self.assertEqual(self.search(cm)[0], 0)
        result = stage.exact_route(cm, 0, 1, np.array([0, 1]), self.native(), find_path,
                                   lambda *a, **k: None, allow_partial=True)
        self.assertEqual((result.status, result.total_cost), ('partial', 0))
        np.testing.assert_array_equal(result.ids, [0])

    def test_safe_native_relaxation_at_closed_boundaries_and_height_limit(self):
        for point in ([.3, .05, .5], [.3, .05 + 1e-12, .5], [.3, .05 - 1e-12, .5],
                      [.3, .06, .5], [.3, 0, 1.06], [.3, 0, -.06], [.3, 0, .5]):
            with self.subTest(point=point):
                cm = self.mapped([[0, 0, 0], [.6, 0, 0], point])
                status, distance, _ = self.search(cm)
                if cm._connected(0, 1):
                    self.assertEqual(status, 1, 'Native removed an original permitted edge')
                    self.assertAlmostEqual(distance[1], .6)
                if point == [.3, 0, .5]:
                    self.assertEqual(status, 0)

    def test_rotated_sweeps_match_exact_python_predicate_away_from_boundary(self):
        rng = np.random.default_rng(129)
        for _ in range(30):
            rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            origin = rng.normal(size=3) * 3
            point = np.array([rng.uniform(-.1, .7), rng.uniform(-.1, .1), rng.uniform(-.1, 1.1)])
            original = np.array([[0., 0., 0.], [.6, 0., 0.], point])
            cm = self.mapped(original @ rotation.T + origin, rotation[:, 2])
            squared = obstacle_distance_squared(cm.points[2:], cm.points[0], cm.points[1], cm.reference, 1.)[0]
            status = self.search(cm)[0]
            if squared < (.05 - 1e-7) ** 2:
                self.assertEqual(status, 0)
            elif squared > (.05 + 1e-7) ** 2:
                self.assertEqual(status, 1)

    def test_margin_zero_and_coincident_edges_remain_safe_relaxations(self):
        for point in ([.3, 0, .5], [.3, .01, .5]):
            cm = self.mapped([[0, 0, 0], [.6, 0, 0], point], margin=0.)
            if cm._connected(0, 1):
                self.assertEqual(self.search(cm)[0], 1)
        cm = self.mapped([[0, 0, 0], [0, 0, 0], [2, 0, .5]])
        status, distance, previous = self.search(cm)
        self.assertEqual((status, distance[1], previous[1]), (1, 0, 0))


if __name__ == '__main__':
    unittest.main()
