"""Acceleration bounds must retain every blocker accepted by the exact predicate."""
import unittest

import numpy as np
from scipy.spatial import cKDTree

from cost_map import CostMap, obstacle_distance_squared


class OverheadIndexRoundoffTests(unittest.TestCase):
    def index(self, obstacles, up, height, margin):
        # Isolate candidate retrieval from unrelated floor support and cost terms.
        costs = object.__new__(CostMap)
        costs.points = np.asarray(obstacles, dtype=float)
        costs.obstacle_rows = np.arange(len(obstacles))
        costs.obstacles = cKDTree(costs.points)
        costs.reference = np.asarray(up, dtype=float)
        costs.config = dict(clearance_height=height, obstacle_margin=margin, max_neighbors=8)
        costs._index_overhead()
        return costs

    def assert_matches_exact(self, costs, a, b=None):
        b = a if b is None else b
        exact = np.any(obstacle_distance_squared(costs.points, a, b, costs.reference,
                          costs.config["clearance_height"]) <= costs.config["obstacle_margin"] ** 2)
        self.assertEqual(not costs.obstacle_free(a, b), exact)
        if np.array_equal(a, b):
            self.assertEqual(bool(costs._blocked_nodes(np.array([a]))[0]), exact)
        return exact

    def test_rotated_closed_margin_boundary(self):
        a = np.array([-1.1670657358473249, -.78850583537539, .6454153185782223])
        p = [-1.1505001019379664, -.7663087082532827, .6752401374495267]
        up = [.31408863123090003, .566829682561601, .7616117401265649]
        self.assertTrue(self.assert_matches_exact(self.index([p], up, .081, .00405), a))

    def test_rotated_zero_margin_exact_blocker(self):
        up = np.array([.6, .8, 0])
        self.assertTrue(self.assert_matches_exact(
            self.index([.025 * up], up, .1, 0.), np.zeros(3)))

    def test_transformed_boundary_candidates_match_exact_predicate(self):
        rng = np.random.default_rng(147)
        for _ in range(40):
            up = rng.normal(size=3)
            up /= np.linalg.norm(up)
            tangent = np.eye(3)[np.argmin(np.abs(up))]
            tangent -= tangent @ up * up
            tangent /= np.linalg.norm(tangent)
            a = rng.uniform(-3, 3, 3)
            for scale in (.001, 1., 37.):
                h, m = .081 * scale, .00405 * scale
                start = a * scale
                obstacles = [start + up * h / 2 + tangent * m]
                costs = self.index(obstacles, up, h, m)
                self.assert_matches_exact(costs, start)
                self.assert_matches_exact(costs, start, start + np.cross(up, tangent) * scale * .04)

    def test_batched_node_shortcut_matches_exact_boundary_checks(self):
        rng = np.random.default_rng(718)
        for _ in range(20):
            up = rng.normal(size=3)
            up /= np.linalg.norm(up)
            tangent = np.eye(3)[np.argmin(np.abs(up))]
            tangent -= tangent @ up * up
            tangent /= np.linalg.norm(tangent)
            positions = rng.uniform(-3, 3, (40, 3))
            h, m = .081, .00405
            obstacles = positions + up * h / 2 + tangent * m
            costs = self.index(obstacles, up, h, m)
            expected = [np.any(obstacle_distance_squared(obstacles, position, position, up, h) <= m ** 2)
                        for position in positions]
            np.testing.assert_array_equal(costs._blocked_nodes(positions), expected)


if __name__ == "__main__":
    unittest.main()
