"""CPU integration checks for voxel steepness and robot feasibility.

Surfaces have independently specified angles before fusion quantization. These
checks exercise planner decisions rather than reproducing the terrain fitter.
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pipeline_common.planning import BLOCKED, TRAVERSABLE, UNKNOWN, build_costmap
from pipeline_research_plan import DEFAULT_ASSUMPTIONS, plan_illustration


VOXEL_SIZE = .05
RESOLUTION = .1
SHAPE = (12, 12)
INTERIOR = np.s_[3:9, 3:9]
ANGLE_TOLERANCE = 4.  # Declared tolerance for .05 m occupancy quantization.
PATCH_SETTINGS = {"neighbor_radius": .25, "second_radius": .4}


def surface(angle=25.):
    # Ten independent raw samples along each cell axis, away from cell edges.
    axis = (np.arange(SHAPE[0] * 10) + .5) * .01
    x, y = np.meshgrid(axis, axis)
    return np.column_stack((x.ravel(), y.ravel(),
                            np.tan(np.radians(angle)) * x.ravel()))


def fused_centers(points):
    # Occupancy quantization is the public fusion contract, not a fitted plane.
    indices = np.unique(np.floor(points / VOXEL_SIZE).astype(np.int64), axis=0)
    return (indices + .5) * VOXEL_SIZE


def settings(limit=35.):
    return {
        "planning": {"resolution": RESOLUTION, "origin": [0., 0.],
                     "shape": list(SHAPE), "support_height": 0.,
                     "terrain_source": "voxels", "terrain": dict(PATCH_SETTINGS)},
        "robot": {"version": "independent_voxel_fixture_v1", "fixture": True,
                  "footprint_radius": 0., "clearance": 0., "height": .35,
                  "max_slope_degrees": limit, "max_step": .08,
                  "max_roughness": .05, "min_support_points": 9,
                  "unknown_rule": "blocked", "semantic_costs": {}},
    }


def build(points, centers, config=None, *, up=(0., 0., 1.), **kwargs):
    return build_costmap(
        points, settings() if config is None else config,
        units=kwargs.pop("units", "metres"), up=up,
        scale={"verified": True, "source": "independent synthetic metre surface"},
        terrain_voxels=None if centers is None else {"centers": centers},
        voxel_size=VOXEL_SIZE, **kwargs)


class VoxelPlanningTests(unittest.TestCase):
    def test_limit_changes_feasibility_without_changing_measured_ramp(self):
        points = surface()
        centers = fused_centers(points)
        blocked, blocked_meta = build(points, centers, settings(15.))
        permitted, permitted_meta = build(points, centers, settings(35.))

        self.assertTrue(permitted["terrain_slope_valid"][INTERIOR].all())
        np.testing.assert_allclose(permitted["slope_degrees"][INTERIOR],
                                   25., atol=ANGLE_TOLERANCE)
        self.assertTrue(np.all(blocked["decision_state"][INTERIOR] == BLOCKED))
        self.assertTrue(np.all(permitted["decision_state"][INTERIOR] == TRAVERSABLE))
        for key in ("slope_degrees", "terrain_normal", "terrain_slope_valid",
                    "terrain_fit_residual", "terrain_support_count"):
            np.testing.assert_array_equal(blocked[key], permitted[key])
        self.assertEqual(blocked_meta["terrain_estimation"],
                         permitted_meta["terrain_estimation"])
        self.assertFalse(permitted_meta["terrain_estimation"]["robot_capability_used"])
        self.assertEqual(blocked_meta["max_slope_degrees"], 15.)
        self.assertEqual(permitted_meta["max_slope_degrees"], 35.)

    def test_reliable_steep_voxels_stay_blocked_despite_floor_reward(self):
        points = surface(40.)
        centers = fused_centers(points)
        config = settings(25.)
        config["robot"]["semantic_costs"] = {"floor": {"cost": -.99}}
        plain, _ = build(points, centers, config)
        rewarded, metadata = build(points, centers, config, semantic_points={
            "floor": {"points": points, "role": "candidate_surface"}})

        self.assertTrue(rewarded["terrain_slope_valid"][INTERIOR].all())
        self.assertTrue(np.all(rewarded["slope_degrees"][INTERIOR] > 25.))
        self.assertTrue(rewarded["slope_blocked_mask"][INTERIOR].all())
        self.assertTrue(np.all(rewarded["decision_state"][INTERIOR] == BLOCKED))
        self.assertTrue(np.isinf(rewarded["costs"][INTERIOR]).all())
        np.testing.assert_array_equal(rewarded["geometry_state"], plain["geometry_state"])
        np.testing.assert_array_equal(rewarded["decision_state"], plain["decision_state"])
        self.assertIn("floor", metadata["applied_concepts"])

    def test_collinear_voxels_cannot_borrow_raw_floor_or_semantic_support(self):
        points = surface(0.)
        centers = np.column_stack(((np.arange(24) + .5) * VOXEL_SIZE,
                                   np.full(24, .075), np.full(24, .025)))
        config = settings()
        config["robot"]["semantic_costs"] = {"floor": {"cost": -.99}}
        arrays, metadata = build(points, centers, config, semantic_points={
            "floor": {"points": points, "role": "candidate_surface"}})

        self.assertTrue(arrays["observed_mask"].all())
        self.assertFalse(arrays["terrain_slope_valid"].any())
        self.assertTrue(np.isnan(arrays["slope_degrees"]).all())
        self.assertTrue(np.all(arrays["decision_state"] == UNKNOWN))
        self.assertTrue(np.isinf(arrays["costs"]).all())
        self.assertEqual(metadata["support_cells"], 0)
        self.assertEqual(metadata["visibility_model"], "none")
        self.assertFalse(metadata["clearance_certified"])

    def test_rotated_map_and_up_preserve_slope_and_map_frame_normals(self):
        points = surface()
        centers = fused_centers(points)
        # A proper rotation moves +Z up to +Y while preserving the voxel lattice.
        rotation = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
        original, _ = build(points, centers)
        rotated, metadata = build(points @ rotation.T, centers @ rotation.T,
                                  up=np.array([0., 0., 1.]) @ rotation.T)

        self.assertTrue(rotated["terrain_slope_valid"][INTERIOR].all())
        np.testing.assert_allclose(rotated["slope_degrees"][INTERIOR],
                                   25., atol=ANGLE_TOLERANCE)
        np.testing.assert_allclose(rotated["slope_degrees"], original["slope_degrees"],
                                   atol=1e-8, equal_nan=True)
        np.testing.assert_allclose(rotated["terrain_normal"],
                                   original["terrain_normal"] @ rotation.T,
                                   atol=1e-8, equal_nan=True)
        np.testing.assert_array_equal(rotated["decision_state"], original["decision_state"])
        np.testing.assert_allclose(metadata["slope_reference_up"], [0., 1., 0.])

    def test_research_ramp_uses_camera_up_independently_of_fitted_ground(self):
        points = surface()
        centers = fused_centers(points)
        cameras = np.array([[.2, .6, 1.], [1., .6, 1.5]])
        rotation = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        config = deepcopy(DEFAULT_ASSUMPTIONS)
        config.update(terrain=dict(PATCH_SETTINGS), require_voxel_terrain=True)
        config["robot"].update(footprint_radius=0., clearance=0.,
                               max_slope_degrees=15., min_support_points=9)
        originals = [value.copy() for value in (points, centers, cameras)]
        result, arrays = plan_illustration(
            points, cameras, np.tile(rotation, (len(cameras), 1, 1)), config,
            {"map_frame": "independent_native_fixture", "original_planning_status": "blocked_inputs"},
            voxels_native={"centers": centers}, voxel_size_native=VOXEL_SIZE)

        self.assertIsNotNone(arrays, result["reason"])
        np.testing.assert_allclose(result["assumed_up_vector"], [0., 0., 1.], atol=1e-10)
        ground_normal = np.asarray(result["ground_plane"]["normal_native"])
        ground_angle = np.degrees(np.arccos(np.clip(abs(ground_normal[2]), 0., 1.)))
        self.assertAlmostEqual(ground_angle, 25., places=6)
        self.assertEqual(result["slope_reference"]["source"], "assumed_upright_first_camera")
        self.assertFalse(result["slope_reference"]["gravity_measured"])
        valid = arrays["terrain_slope_valid"]
        self.assertGreater(np.count_nonzero(valid), 25)
        self.assertAlmostEqual(float(np.median(arrays["slope_degrees"][valid])),
                               25., delta=ANGLE_TOLERANCE)
        self.assertTrue(np.all(arrays["decision_state"][valid] == BLOCKED))
        self.assertEqual(result["status"], "no_path")
        self.assertEqual(result["path_points"], [])
        self.assertFalse(result["calibration_verified"])
        self.assertFalse(result["safety_validated"])
        metadata = result["diagnostics"]["planning_check_metadata"]
        self.assertEqual(metadata["units"], "assumed_metres")
        self.assertEqual(metadata["terrain_estimation"]["voxel_size"], VOXEL_SIZE)
        for actual, original in zip((points, centers, cameras), originals):
            np.testing.assert_array_equal(actual, original)

    def test_numeric_schema_records_units_and_round_trips_without_pickle(self):
        points = surface()
        arrays, metadata = build(points, fused_centers(points))
        self.assertEqual(metadata["shape"], list(SHAPE))
        self.assertEqual(metadata["units"], "metres")
        self.assertEqual(arrays["terrain_normal"].shape, (*SHAPE, 3))
        for key in ("slope_degrees", "terrain_slope_valid", "terrain_fit_residual",
                    "terrain_support_count", "slope_blocked_mask"):
            self.assertEqual(arrays[key].shape, SHAPE)
        self.assertEqual(arrays["terrain_slope_valid"].dtype, np.bool_)
        self.assertEqual(arrays["terrain_support_count"].dtype, np.int64)
        self.assertTrue(all(value.dtype != object for value in arrays.values()))
        json.dumps(metadata, allow_nan=False)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "voxel_costmap.npz"
            np.savez_compressed(target, **arrays)
            with np.load(target, allow_pickle=False) as saved:
                self.assertEqual(set(saved.files), set(arrays))
                for key in arrays:
                    np.testing.assert_array_equal(saved[key], arrays[key])

    def test_missing_voxels_or_unverified_units_fail_closed(self):
        points = surface()
        for centers, extra in ((None, {}), (fused_centers(points), {"units": "reconstruction_units"})):
            with self.subTest(extra=extra, missing=centers is None):
                arrays, metadata = build(points, centers, **extra)
                self.assertEqual(metadata["availability"], "blocked_inputs")
                self.assertEqual(arrays["decision_state"].shape, (0, 0))
                self.assertTrue(all(value.dtype != object for value in arrays.values()))


if __name__ == "__main__":
    unittest.main()
