"""Shortest paths and failure behavior on original point-row graphs."""

from types import SimpleNamespace
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from cost_map import build_cost_map
from planner import find_path


def graph(points, costs, edges):
    points = np.asarray(points, dtype=float)
    adjacency = [[] for _ in points]
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    return points, SimpleNamespace(points=points.copy(), cost=np.asarray(costs, dtype=float),
                                   neighbours=lambda row: adjacency[row])


class PlannerTests(unittest.TestCase):
    def assert_failure(self, result, status):
        self.assertEqual(result.status, status)
        self.assertEqual(result.ids.shape, (0,))
        self.assertEqual(result.positions.shape, (0, 3))
        self.assertEqual(result.total_cost, np.inf)

    def test_minimum_total_cost_and_original_ids(self):
        points, costs = graph([[9, 9, 9], [2, 0, 0], [0, 0, 0], [1, 0, 0]],
                              [np.inf, 3, 1, 2], [(2, 1), (2, 3), (3, 1)])
        result = find_path(points, costs, np.int64(2), 1)
        self.assertEqual(result.status, "ok")
        self.assertAlmostEqual(result.total_cost, 4)
        assert_array_equal(result.ids, [2, 1])
        assert_array_equal(result.positions, points[result.ids])

    def test_longer_low_cost_route_beats_short_expensive_route(self):
        points, costs = graph([[0, 0, 0], [1, 0, 0], [2, 0, 0],
                               [0, 1, 0], [2, 1, 0]],
                              [1, 10, 1, 1, 1], [(0, 1), (1, 2), (0, 3), (3, 4), (4, 2)])
        result = find_path(points, costs, 0, 2)
        assert_array_equal(result.ids, [0, 3, 4, 2])
        self.assertEqual(result.total_cost, 4)

    def test_disconnected_and_invalid_endpoints(self):
        points, costs = graph([[0, 0, 0], [1, 0, 0], [np.nan, 0, 0], [2, 0, 0]],
                              [1, 1, 1, np.inf], [])
        self.assert_failure(find_path(points, costs, 0, 1), "no_path")
        for row in (-1, 4, 0.0, True, np.bool_(False), None, 2, 3):
            with self.subTest(row=row):
                self.assert_failure(find_path(points, costs, row, 1), "invalid_start")
                self.assert_failure(find_path(points, costs, 0, row), "invalid_goal")
        self.assert_failure(find_path(points, costs, 3, 3), "invalid_start")

    def test_valid_equal_endpoints(self):
        points, costs = graph([[1, 2, 3]], [5], [])
        result = find_path(points, costs, 0, 0)
        self.assertEqual(result.status, "ok")
        assert_array_equal(result.ids, [0])
        assert_array_equal(result.positions, points)
        self.assertEqual(result.total_cost, 0)

    def test_coincident_zero_cost_cycle_preserves_distinct_rows(self):
        points, costs = graph([[0, 0, 0]] * 3 + [[1, 0, 0]], [1, 2, 3, 1],
                              [(0, 1), (1, 2), (2, 0), (2, 3)])
        result = find_path(points, costs, 0, 3)
        assert_array_equal(result.ids, [0, 2, 3])
        self.assertEqual(result.total_cost, 2)
        coincident = find_path(points, costs, 1, 2)
        assert_array_equal(coincident.ids, [1, 2])
        self.assertEqual(coincident.total_cost, 0)

    def test_array_correspondence_and_malformed_costs(self):
        points, costs = graph([[0, 0, 0], [1, 0, 0]], [1, 2], [(0, 1)])
        for bad_points in (points[::-1], points[:1], points[:, :2]):
            with self.subTest(points=bad_points), self.assertRaises(ValueError):
                find_path(bad_points, costs, 0, 1)
        for bad_costs in ([1], [[1], [2]], [1, np.nan], [1, -np.inf], [1, .5],
                          [1, 2j], [True, False]):
            with self.subTest(costs=bad_costs), self.assertRaises(ValueError):
                costs.cost = np.asarray(bad_costs)
                find_path(points, costs, 0, 1)

    def test_actual_cost_map_respects_floor_gap(self):
        x, y = np.meshgrid(np.arange(-6, 7) * .01, np.arange(-3, 4) * .01)
        floor = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        for gap in (False, True):
            points = floor[np.abs(floor[:, 0]) >= .03] if gap else floor
            costs = build_cost_map(points, np.zeros(len(points)), [0, 0, 1],
                                   dict(radius=.081, connection_radius=.07,
                                        support_mask=np.ones(len(points), dtype=bool)))
            start = int(np.argmin(np.linalg.norm(points - [-.04, 0, 0], axis=1)))
            goal = int(np.argmin(np.linalg.norm(points - [.04, 0, 0], axis=1)))
            result = find_path(points, costs, start, goal)
            if gap:
                self.assert_failure(result, "no_path")
            else:
                self.assertEqual(result.status, "ok")
                assert_allclose(result.total_cost, .08 * (1 + np.exp(-1)))
                for a, b in zip(result.ids[:-1], result.ids[1:]):
                    self.assertIn(b, costs.neighbours(a))


if __name__ == "__main__":
    unittest.main()
