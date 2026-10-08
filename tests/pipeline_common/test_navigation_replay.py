import json
import unittest

import numpy as np

from pipeline_common.navigation_replay import clip_route_horizon, plan_next_steps
from pipeline_common.research_route import segment_supported


class NavigationReplayTests(unittest.TestCase):
    def fixture(self):
        shape = (50, 30)
        arrays = {
            "origin": np.array([-1.5, 0.]), "resolution": np.array([.1]),
            "projection_basis": np.eye(3), "geometry_state": np.full(shape, 2, np.uint8),
            "decision_state": np.full(shape, 2, np.uint8), "support_height": np.zeros(shape),
            "policy_blocked_mask": np.zeros(shape, bool), "costs": np.ones(shape),
        }
        metadata = {"availability": "available", "footprint_radius_with_clearance": .12,
                    "profile": {"max_slope_degrees": 25., "max_step": .08}}
        return arrays, metadata

    def plan(self, arrays, metadata, agent=(.05, .55), origin=(.05, .55), forward=(0., 1.), **kwargs):
        return plan_next_steps(arrays, metadata, agent, origin, forward, **kwargs)

    def test_deviation_and_shortcuts_keep_whole_footprint_supported(self):
        a, m = self.fixture()
        # A wall in the direct forward line requires a supported lateral detour.
        a["geometry_state"][20:25, 13:17] = 1
        a["decision_state"][18:27, 11:19] = 1
        r = self.plan(a, m)
        self.assertEqual(r["status"], "ok")
        path = np.asarray(r["path_points_assumed_m"])
        self.assertGreater(len(path), 2)
        self.assertGreater(np.max(abs(path[:, 0] - .05)), .3)
        self.assertTrue(all(segment_supported(x, y, a, m) for x, y in zip(path[:-1], path[1:])))
        self.assertFalse(segment_supported(path[0], path[-1], a, m))
        self.assertAlmostEqual(r["display_length_assumed_m"], 2.)

    def test_unknown_band_cannot_jump_to_forward_component(self):
        a, m = self.fixture()
        a["geometry_state"][25:28] = 0
        a["decision_state"][23:30] = 0
        r = self.plan(a, m)
        self.assertEqual(r["status"], "ok")
        self.assertLess(r["goal_point_assumed_m"][1], 2.5)
        self.assertTrue(all(cell[0] < 25 for cell in r["path_cells"]))
        self.assertFalse(r["mission_complete"])

    def test_fixed_mission_is_not_recentered_on_current_agent(self):
        a, m = self.fixture()
        r = self.plan(a, m, agent=(.45, 1.55), max_lateral_m=.06)
        self.assertEqual(r["status"], "ok")
        self.assertAlmostEqual(r["goal_point_assumed_m"][0], .05)
        np.testing.assert_allclose(r["path_points_assumed_m"][0], [.45, 1.55, 0.])
        self.assertEqual(r["mission_origin_xy"], [.05, .55])
        self.assertEqual(r["mission_forward_xy"], [0., 1.])

    def test_actual_noncentre_start_is_connected_without_teleport(self):
        a, m = self.fixture()
        r = self.plan(a, m, agent=(.023, .523))
        self.assertEqual(r["status"], "ok")
        np.testing.assert_allclose(r["path_points_assumed_m"][0], [.023, .523, 0.], atol=0, rtol=0)
        np.testing.assert_allclose(r["agent_ground_point_assumed_m"], [.023, .523, 0.], atol=0, rtol=0)
        self.assertGreater(r["start_adjustment_m"], 0.)
        self.assertLessEqual(r["start_adjustment_m"], .15)

    def test_unknown_agent_ground_or_footprint_never_snaps_to_support(self):
        for failure in ("underfoot", "footprint", "outside"):
            with self.subTest(failure=failure):
                a, m = self.fixture()
                agent = (.05, .55)
                if failure == "underfoot":
                    a["geometry_state"][5, 15] = 0
                    a["decision_state"][5, 15] = 0
                elif failure == "footprint":
                    a["geometry_state"][5, 16] = 0
                    a["decision_state"][5, 16] = 0
                else:
                    agent = (.05, -.1)
                r = self.plan(a, m, agent=agent)
                self.assertEqual(r["status"], "awaiting_support")
                self.assertEqual(r["path_points_assumed_m"], [])
                self.assertIsNone(r["agent_ground_point_assumed_m"])
                self.assertTrue(r["mission_active"])
                self.assertFalse(r["mission_complete"])

    def test_start_adjustment_bound_is_respected(self):
        a, m = self.fixture()
        r = self.plan(a, m, agent=(.023, .523), max_start_adjustment_m=.01)
        self.assertEqual(r["status"], "awaiting_support")
        self.assertIn("adjustment", r["reason"])

    def test_horizon_interpolates_exact_3d_arc_length_across_bend(self):
        route = np.array([[0., 0., 0.], [3., 0., 4.], [3., 4., 4.]])
        clipped = clip_route_horizon(route, 7.)
        np.testing.assert_allclose(clipped, [[0, 0, 0], [3, 0, 4], [3, 2, 4]])
        self.assertAlmostEqual(np.linalg.norm(np.diff(clipped, axis=0), axis=1).sum(), 7.)
        np.testing.assert_allclose(clip_route_horizon(route, 2.5), [[0, 0, 0], [1.5, 0, 2]])
        np.testing.assert_allclose(clip_route_horizon(route, 20), route)

    def test_horizon_empty_zero_and_duplicate_vertices(self):
        self.assertEqual(clip_route_horizon([], 2.).shape, (0, 3))
        np.testing.assert_allclose(clip_route_horizon([[1., 2., 3.], [1., 2., 3.], [1., 5., 3.]], 1.),
                                   [[1., 2., 3.], [1., 3., 3.]])
        np.testing.assert_allclose(clip_route_horizon([[1., 2.], [4., 2.]], 0), [[1., 2.]])
        with self.assertRaises(ValueError):
            clip_route_horizon([[0., 0., 0.]], -1.)

    def test_reaching_known_frontier_waits_without_mission_completion(self):
        a, m = self.fixture()
        r = self.plan(a, m, agent=(.05, 4.75))
        self.assertEqual(r["status"], "awaiting_observation")
        self.assertEqual(r["path_points_assumed_m"], [])
        self.assertTrue(r["mission_active"])
        self.assertFalse(r["mission_complete"])

    def test_bad_inputs_and_unavailable_map_are_explicit(self):
        a, m = self.fixture()
        with self.assertRaises(ValueError):
            self.plan(a, m, forward=(0., 0.))
        a["support_height"][5, 15] = np.nan
        with self.assertRaises(ValueError):
            self.plan(a, m)
        m["availability"] = "blocked_inputs"
        m["reason"] = "prefix calibration unavailable"
        r = self.plan({}, m)
        self.assertEqual(r["status"], "blocked_inputs")
        self.assertEqual(r["reason"], "prefix calibration unavailable")
        self.assertFalse(r["mission_complete"])

    def test_assumed_map_basis_is_applied_once(self):
        a, m = self.fixture()
        a["projection_basis"] = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
        r = self.plan(a, m)
        self.assertEqual(r["status"], "ok")
        np.testing.assert_allclose(r["agent_ground_point_assumed_m"], [-.55, .05, 0.])
        display = np.asarray(r["display_path_points_assumed_m"])
        self.assertAlmostEqual(np.linalg.norm(np.diff(display, axis=0), axis=1).sum(), 2.)

    def test_computed_route_serializes_without_numpy_scalars_or_nonfinite_values(self):
        a, m = self.fixture()
        # Dijkstra obtains its cell indices from NumPy's argwhere. Exercise the
        # actual computed result, including a checked non-centre connector.
        r = self.plan(a, m, agent=(.023, .523))
        self.assertEqual(r["status"], "ok")
        self.assertGreater(len(r["path_cells"]), 2)
        for key in ("path_cells", "waypoint_cells"):
            self.assertTrue(all(type(value) is int for cell in r[key] for value in cell))
        decoded = json.loads(json.dumps(r, allow_nan=False))
        self.assertEqual(decoded, r)
        self.assertEqual(decoded["path_points_assumed_m"][0], [.023, .523, 0.])
        self.assertFalse(decoded["mission_complete"])


if __name__ == "__main__":
    unittest.main()
