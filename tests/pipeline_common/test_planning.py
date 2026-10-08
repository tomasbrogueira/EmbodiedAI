"""CPU geometry/navigation conformance with independent synthetic scenes."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline_common.planning import BLOCKED, TRAVERSABLE, UNKNOWN, build_costmap, plan_requests


def settings(shape=(9, 9), resolution=1.):
    return {
        "planning": {"resolution": resolution, "origin": [0., 0.], "shape": list(shape), "support_height": 0.},
        "robot": {"version": "synthetic_robot_v1", "fixture": True,
                  "footprint_radius": .1, "height": 1., "clearance": .05,
                  "max_slope_degrees": 25., "max_step": .2, "min_support_points": 9,
                  "max_roughness": .03, "unknown_rule": "blocked", "semantic_costs": {}},
    }


def terrain(shape=(9, 9), resolution=1., surface=None, omit=()):
    points = []
    for y in range(shape[0]):
        for x in range(shape[1]):
            if (y, x) in omit:
                continue
            for offset_y in (.15, .5, .85):
                for offset_x in (.15, .5, .85):
                    xx, yy = (x + offset_x) * resolution, (y + offset_y) * resolution
                    zz = 0. if surface is None else surface(xx, yy, x, y)
                    points.append([xx, yy, zz])
    return np.asarray(points, dtype=float).reshape((-1, 3))


def build(points, config=None, **kwargs):
    return build_costmap(points, settings() if config is None else config,
                         up=kwargs.pop("up", [0., 0., 1.]),
                         scale=kwargs.pop("scale", {"verified": True, "source": "independent synthetic metre coordinates"}),
                         **kwargs)


class PlanningTests(unittest.TestCase):
    def test_missing_inputs_preserve_empty_numeric_schema_and_every_request(self):
        missing = [
            {"units": "reconstruction_units"}, {"up": None}, {"scale": None},
            {"scale": {"verified": False}},
        ]
        for extra in missing:
            with self.subTest(extra=extra):
                arrays, metadata = build(terrain(), **extra)
                self.assertEqual(metadata["availability"], "blocked_inputs")
                self.assertEqual(arrays["decision_state"].shape, (0, 0))
                self.assertEqual(arrays["policy_blocked_mask"].shape, (0, 0))
                self.assertEqual(arrays["policy_blocked_mask"].dtype, np.bool_)
                self.assertTrue(all(array.dtype != object for array in arrays.values()))
                requests = [{"request_id": "a", "start": [1, 1, 0], "goal": [2, 2, 0]},
                            {"request_id": "b", "start": [2, 2, 0], "goal": [3, 3, 0]}]
                plans = plan_requests(arrays, metadata, requests, "world")
                self.assertEqual([plan["request_id"] for plan in plans], ["a", "b"])
                self.assertTrue(all(plan["status"] == "blocked_inputs" and plan["path"] == [] for plan in plans))
        for field in settings()["robot"]:
            config = settings()
            del config["robot"][field]
            with self.subTest(robot_field=field):
                _, metadata = build(terrain(), config)
                self.assertEqual(metadata["availability"], "blocked_inputs")

    def test_flat_support_and_arbitrary_gravity_return_map_frame_route(self):
        # Rotation sends map +Z up to +Y, and projected Y to map -Z.
        rotation = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
        rotated = terrain() @ rotation.T
        arrays, metadata = build(rotated, up=[0., 1., 0.])
        np.testing.assert_allclose(arrays["projection_basis"] @ arrays["projection_basis"].T, np.eye(3))
        self.assertTrue(np.all(arrays["decision_state"] == TRAVERSABLE))
        self.assertFalse(metadata["clearance_certified"])
        start, goal = np.array([1.2, 1.7, 0.]) @ rotation.T, np.array([7.8, 6.2, 0.]) @ rotation.T
        requests = [{"request_id": "rotated", "start": start.tolist(), "goal": goal.tolist(), "frame": "rotated_world"}]
        result = plan_requests(arrays, metadata, requests, "rotated_world")[0]
        self.assertEqual(result["status"], "ok")
        np.testing.assert_array_equal(result["path"][0], start)
        np.testing.assert_array_equal(result["path"][-1], goal)
        np.testing.assert_allclose(np.asarray(result["path"])[:, 1], 0., atol=1e-10)
        self.assertTrue(result["diagnostic_only"])

    def test_missing_depth_hole_cannot_be_repaired_by_surface_semantics(self):
        config = settings()
        config["robot"]["semantic_costs"] = {"floor": {"cost": -.9}}
        points = terrain(omit={(4, 4)})
        arrays, metadata = build(points, config, semantic_points={"floor": {"points": [[4.5, 4.5, 0.]], "role": "candidate_surface"}})
        self.assertEqual(arrays["decision_state"][4, 4], UNKNOWN)
        self.assertFalse(arrays["observed_mask"][4, 4])
        self.assertTrue(np.isinf(arrays["costs"][4, 4]))
        plan = plan_requests(arrays, metadata, [{"start": [4.5, 4.5, 0.], "goal": [6.5, 6.5, 0.]}], "world")[0]
        self.assertEqual(plan["status"], "no_path")
        self.assertIn("no snapping", plan["reason"])

    def test_known_step_and_dropoff_block_the_observed_edge(self):
        for offset in (-.8, .8):
            with self.subTest(offset=offset):
                points = terrain(surface=lambda xx, yy, x, y: offset if x >= 5 else 0.)
                arrays, _ = build(points)
                self.assertTrue(np.all(arrays["geometry_state"][:, 4:6] == BLOCKED))
                self.assertEqual(arrays["decision_state"][4, 2], TRAVERSABLE)
                self.assertNotEqual(arrays["decision_state"][4, 7], TRAVERSABLE)

    def test_step_just_beyond_robot_limit_blocks_without_roughness_allowance(self):
        arrays, _ = build(terrain(surface=lambda xx, yy, x, y: .21 if x >= 5 else 0.))
        self.assertTrue(np.all(arrays["geometry_state"][:, 4:6] == BLOCKED))

    def test_disconnected_elevated_shell_cannot_seed_floor_within_step_limit(self):
        arrays, metadata = build(terrain(surface=lambda xx, yy, x, y: .15))
        self.assertFalse(np.any(arrays["decision_state"] == TRAVERSABLE))
        self.assertEqual(metadata["support_cells"], 0)

    def test_reachable_low_step_is_supported_from_ground_anchor(self):
        arrays, _ = build(terrain(surface=lambda xx, yy, x, y: .15 if x >= 5 else 0.))
        self.assertTrue(np.all(arrays["decision_state"] == TRAVERSABLE))

    def test_continuous_slope_is_not_a_step_but_steep_support_blocks(self):
        config = settings()
        config["robot"]["max_step"] = .05
        arrays, _ = build(terrain(surface=lambda xx, yy, x, y: .1 * xx), config)
        self.assertTrue(np.all(arrays["decision_state"] == TRAVERSABLE))
        np.testing.assert_allclose(arrays["slope_degrees"], np.degrees(np.arctan(.1)), atol=1e-8)
        arrays, _ = build(terrain(surface=lambda xx, yy, x, y: 1.0 * xx))
        self.assertTrue(np.all(arrays["decision_state"] == BLOCKED))

    def test_rough_support_blocks_even_with_favorable_surface_prior(self):
        config = settings()
        config["robot"]["semantic_costs"] = {"floor": {"cost": -.99}}
        points = terrain(surface=lambda xx, yy, x, y: .10 if (abs(xx - 4.5) < .01 and abs(yy - 4.5) < .01) else 0.)
        arrays, _ = build(points, config, semantic_points={"floor": {"points": [[4.5, 4.5, .1]], "role": "candidate_surface"}})
        self.assertGreater(arrays["roughness"][4, 4], config["robot"]["max_roughness"])
        self.assertEqual(arrays["decision_state"][4, 4], BLOCKED)

    def test_wall_and_low_overhang_block_and_high_overhang_never_becomes_floor(self):
        floor = terrain()
        wall = np.array([[4.5, yy, zz] for yy in (4.15, 4.5, 4.85) for zz in np.linspace(0., 1.5, 25)])
        arrays, _ = build(np.concatenate((floor, wall)))
        self.assertEqual(arrays["decision_state"][4, 4], BLOCKED)
        lower_shell = terrain(shape=(1, 1)) + [4., 4., .7]
        arrays, _ = build(np.concatenate((floor, lower_shell)))
        self.assertEqual(arrays["decision_state"][4, 4], BLOCKED)
        self.assertLess(arrays["clearance_observed"][4, 4], 1.)
        arrays, metadata = build(terrain(surface=lambda xx, yy, x, y: 2.))
        self.assertFalse(np.any(arrays["decision_state"] == TRAVERSABLE))
        self.assertEqual(metadata["support_cells"], 0)
        arrays, _ = build(wall)
        self.assertNotEqual(arrays["decision_state"][4, 4], TRAVERSABLE)

    def test_close_stacked_floor_and_ceiling_do_not_blend_into_support(self):
        config = settings(shape=(1, 1))
        config["robot"]["footprint_radius"] = 0.
        config["robot"]["clearance"] = 0.
        floor = terrain(shape=(1, 1))
        ceiling = floor + [0., 0., .04]
        arrays, _ = build(np.r_[floor, ceiling], config)
        self.assertEqual(arrays["support_count"][0, 0], 9)
        self.assertAlmostEqual(arrays["support_height"][0, 0], 0.)
        self.assertAlmostEqual(arrays["clearance_observed"][0, 0], .04)
        self.assertEqual(arrays["decision_state"][0, 0], BLOCKED)

    def test_stacked_clearance_uses_lower_surface_gap_on_sloped_floor(self):
        config = settings()
        floor = terrain(surface=lambda xx, yy, x, y: .1 * xx)
        ceiling = floor + [0., 0., 1.04]
        arrays, _ = build(np.r_[floor, ceiling], config)
        np.testing.assert_allclose(arrays["clearance_observed"], 1.04, atol=1e-8)
        self.assertTrue(np.all(arrays["decision_state"] == BLOCKED))

    def test_footprint_gaps_and_boundaries_remain_unknown(self):
        config = settings()
        config["robot"]["footprint_radius"] = .6
        config["robot"]["clearance"] = 0.
        arrays, _ = build(terrain(omit={(4, 4)}), config)
        for cell in [(4, 4), (4, 3), (4, 5), (3, 4), (5, 4), (0, 4)]:
            self.assertEqual(arrays["decision_state"][cell], UNKNOWN, cell)
        self.assertEqual(arrays["decision_state"][2, 2], TRAVERSABLE)

    def test_invalid_points_do_not_create_coverage_and_npz_has_no_pickle(self):
        points = np.array([[np.nan, 0., 0.], [1., np.inf, 0.], [-1., -1., 0.]])
        arrays, metadata = build(points)
        self.assertFalse(np.any(arrays["observed_mask"]))
        self.assertTrue(np.all(arrays["decision_state"] == UNKNOWN))
        self.assertEqual(metadata["invalid_or_outside_points"], 3)
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "costmap.npz"
            np.savez(file, **arrays)
            with np.load(file, allow_pickle=False) as saved:
                self.assertTrue(all(saved[key].dtype != object for key in saved.files))

    def test_hazard_policy_is_shared_conflicts_and_unmapped_ids_are_explicit(self):
        config = settings()
        config["robot"]["semantic_costs"] = {"hazard": {"blocked": True}, "floor": {"cost": -.8}}
        semantics = {
            "hazard": {"points": [[4.5, 4.5, 0.]], "role": "hazard"},
            "floor": {"points": [[4.5, 4.5, 0.]], "role": "candidate_surface"},
            "unmapped:test": {"points": [[4.5, 4.5, 0.]], "role": "hazard"},
        }
        arrays, metadata = build(terrain(), config, semantic_points=semantics)
        self.assertEqual(arrays["decision_state"][4, 4], BLOCKED)
        self.assertEqual(arrays["geometry_state"][4, 4], TRAVERSABLE)
        self.assertEqual(metadata["unmapped_concepts"], ["unmapped:test"])
        self.assertEqual(metadata["semantic_conflict_cells"], 1)
        plan = plan_requests(arrays, metadata, [{"start": [2.5, 4.5, 0.], "goal": [6.5, 4.5, 0.]}], "world")[0]
        self.assertEqual(plan["status"], "ok")
        self.assertNotIn([4, 4], plan["path_cells"])

    def test_soft_hazard_cost_changes_route_without_changing_ground_support(self):
        config = settings()
        config["robot"]["semantic_costs"] = {"hazard": 20.}
        arrays, metadata = build(terrain(), config, semantic_points={"hazard": [[4.5, 4.5, 0.]]})
        self.assertEqual(arrays["decision_state"][4, 4], TRAVERSABLE)
        result = plan_requests(arrays, metadata, [{"start": [2.5, 4.5, 0.], "goal": [6.5, 4.5, 0.]}], "world")[0]
        self.assertEqual(result["status"], "ok")
        self.assertNotIn([4, 4], result["path_cells"])
        self.assertGreater(len(result["path_cells"]), 5)

    def test_no_path_frame_endpoint_height_and_reproducibility(self):
        points = terrain(omit={(y, 4) for y in range(9)})
        arrays, metadata = build(points)
        requests = [
            {"request_id": "hole_barrier", "start": [2.5, 4.5, 0.], "goal": [6.5, 4.5, 0.]},
            {"request_id": "wrong_frame", "frame": "camera", "start": [2.5, 2.5, 0.], "goal": [3.5, 3.5, 0.]},
            {"request_id": "wrong_height", "start": [2.5, 2.5, 10.], "goal": [3.5, 3.5, 0.]},
            {"request_id": "outside", "start": [-.01, 2.5, 0.], "goal": [3.5, 3.5, 0.]},
            {"request_id": "valid", "start": [1.5, 1.5, 0.], "goal": [3.5, 6.5, 0.]},
        ]
        result = plan_requests(arrays, metadata, requests, "world")
        self.assertEqual([item["status"] for item in result], ["no_path", "blocked_inputs", "no_path", "no_path", "ok"])
        self.assertEqual(result, plan_requests(arrays, metadata, requests, "world"))
        self.assertTrue(all(result[index]["path"] == [] for index in range(4)))

    def test_endpoint_actual_footprint_cannot_snap_from_boundary_to_cell_center(self):
        arrays, metadata = build(terrain())
        result = plan_requests(arrays, metadata, [{"start": [.01, 2.5, 0.], "goal": [3.5, 3.5, 0.]}], "world")[0]
        self.assertEqual(result["status"], "no_path")
        self.assertIn("footprint", result["reason"])

    def test_endpoint_footprint_does_not_inflate_nearby_policy_or_geometry_twice(self):
        config = settings(resolution=.1)
        config["robot"].update(footprint_radius=.12, clearance=0.)
        config["robot"]["semantic_costs"] = {"excluded": {"blocked": True}}
        floor = terrain(resolution=.1)
        # Cell(4,6) begins at x=.6: its nearest edge is .15m from the
        # endpoint(.45,.45), outside the .12m disk. Its once-padded neighbor
        # cell(4,5) intersects that disk and must not trigger a second padding.
        for source in ("hazard", "candidate_surface", "geometry"):
            with self.subTest(source=source):
                points, semantics = floor, None
                if source == "geometry":
                    overhead = terrain(shape=(1, 1), resolution=.1) + [.6, .4, .7]
                    points = np.concatenate((floor, overhead))
                else:
                    semantics = {"excluded": {"points": [[.65, .45, 0.]], "role": source}}
                arrays, metadata = build(points, config, semantic_points=semantics)
                self.assertEqual(arrays["decision_state"][4, 4], TRAVERSABLE)
                self.assertEqual(arrays["decision_state"][4, 5],
                                 TRAVERSABLE if source == "candidate_surface" else BLOCKED)
                start, goal = [.45, .45, 0.], [.25, .45, 0.]
                result = plan_requests(arrays, metadata, [{"start": start, "goal": goal}], "world")[0]
                self.assertEqual(result["status"], "ok", result["reason"])
                self.assertEqual(result["path"][0], start)
                self.assertEqual(result["path"][-1], goal)
                self.assertEqual(np.count_nonzero(arrays["policy_blocked_mask"]),
                                 0 if source == "geometry" else 1)
                self.assertFalse(arrays["policy_blocked_mask"][4, 5])

    def test_endpoint_exact_footprint_still_rejects_raw_policy_geometry_and_unknown(self):
        config = settings(resolution=.1)
        config["robot"].update(footprint_radius=.12, clearance=0.)
        config["robot"]["semantic_costs"] = {"excluded": {"blocked": True}}
        # The cell center remains usable, but the exact x=.49 endpoint's disk
        # reaches x=.61 and therefore intersects the raw cell(4,6).
        for source in ("hazard", "candidate_surface", "geometry", "unknown"):
            with self.subTest(source=source):
                floor = terrain(resolution=.1, omit={(4, 6)} if source == "unknown" else ())
                points, semantics = floor, None
                if source == "geometry":
                    overhead = terrain(shape=(1, 1), resolution=.1) + [.6, .4, .7]
                    points = np.concatenate((floor, overhead))
                elif source != "unknown":
                    semantics = {"excluded": {"points": [[.65, .45, 0.]], "role": source}}
                arrays, metadata = build(points, config, semantic_points=semantics)
                self.assertEqual(arrays["decision_state"][4, 4], TRAVERSABLE)
                result = plan_requests(arrays, metadata, [
                    {"start": [.49, .45, 0.], "goal": [.25, .45, 0.]}
                ], "world")[0]
                self.assertEqual(result["status"], "no_path")
                self.assertEqual(result["path"], [])
                self.assertIn("blocked policy" if source in ("hazard", "candidate_surface")
                              else "unknown or blocked geometry", result["reason"])

    def test_endpoints_are_uniformly_map_frame_xyz_and_invalid_formats_are_errors(self):
        arrays, metadata = build(terrain())
        for point in ([1.5, 1.5], [1.5], [1.5, 1.5, np.nan], None, "invalid"):
            with self.subTest(point=point):
                result = plan_requests(arrays, metadata, [{"start": point, "goal": [3.5, 3.5, 0.]}], "world")[0]
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["path"], [])
                self.assertIn("three finite map-frame XYZ", result["reason"])

    def test_verified_metric_points_are_not_scaled_twice(self):
        arrays, _ = build(terrain(), scale={"verified": True, "factor": 5., "source": "already applied by geometry provider"})
        self.assertTrue(np.all(arrays["decision_state"] == TRAVERSABLE))
        self.assertTrue(np.all(arrays["observed_mask"]))


if __name__ == "__main__":
    unittest.main()
