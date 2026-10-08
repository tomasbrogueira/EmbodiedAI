"""Independent voxel fixtures for terrain measurements, without models."""
import json
import math
import unittest
from unittest import mock

import numpy as np

from pipeline_common.terrain import _voxel_xy_coverage, estimate_voxel_terrain


def ramp(angle=0., voxel=.02, span=.8, origin=(0., 0., 0.)):
    axis = (np.arange(round(span / voxel)) + .5) * voxel
    x, y = np.meshgrid(axis, axis)
    points = np.column_stack((x.ravel(), y.ravel(), np.tan(np.radians(angle)) * x.ravel()))
    # Quantize exactly as a fused occupancy map, with one center per voxel.
    origin = np.asarray(origin)
    return np.unique(origin + (np.floor((points - origin) / voxel) + .5) * voxel, axis=0)


def estimate(points, **kwargs):
    arguments = dict(basis=np.eye(3), origin=[0., 0.], resolution=.1,
                     shape=(8, 8), voxel_size=.02,
                     settings={"neighbor_radius": .16, "second_radius": .24})
    arguments.update(kwargs)
    return estimate_voxel_terrain(points, **arguments)


def quantized_reference_plane(angle, basis, voxel, phase=0., span=.8, spacing=.005):
    """Rotate known raw plane before native-axis occupancy quantization."""
    axis = (np.arange(round(span / spacing)) + .5) * spacing
    x, y = np.meshgrid(axis, axis)
    local = np.column_stack((x.ravel(), y.ravel(), np.tan(np.radians(angle)) * x.ravel()))
    native = local @ basis
    return np.unique(phase + (np.floor((native - phase) / voxel) + .5) * voxel, axis=0)


def reference_basis():
    up = np.array([.03556209671530983, -.8963417047833864, -.44193538616320777])
    up /= np.linalg.norm(up)
    x = np.cross([0., 0., 1.], up)
    x /= np.linalg.norm(x)
    return np.array([x, np.cross(up, x), up])


class TerrainTests(unittest.TestCase):
    def test_flat_voxels_and_numeric_unknown_schema(self):
        arrays, metadata = estimate(ramp())
        self.assertTrue(arrays["valid_mask"].all())
        np.testing.assert_allclose(arrays["slope_degrees"], 0., atol=1e-10)
        np.testing.assert_allclose(arrays["normal"], np.broadcast_to([0., 0., 1.], (8, 8, 3)), atol=1e-10)
        self.assertFalse(metadata["robot_capability_used"])
        json.dumps(metadata, allow_nan=False)
        empty, meta = estimate(np.empty((0, 3)))
        self.assertFalse(empty["valid_mask"].any())
        self.assertTrue(np.isnan(empty["normal"]).all())
        self.assertTrue(np.isnan(empty["slope_degrees"]).all())
        self.assertTrue(np.isnan(empty["residual"]).all())
        self.assertEqual(np.sum(empty["support_count"]), 0)
        self.assertEqual(meta["occupied_columns"], 0)

    def test_voxelized_ramps_below_and_above_robot_angle(self):
        for angle in (20., 30.):
            with self.subTest(angle=angle):
                arrays, _ = estimate(ramp(angle))
                center = arrays["slope_degrees"][2:6, 2:6]
                self.assertTrue(np.isfinite(center).all())
                np.testing.assert_allclose(center, angle, atol=2.)
                self.assertTrue(np.all(center < 25.) if angle < 25 else np.all(center > 25.))

    def test_rotated_basis_returns_map_normals_and_same_angles(self):
        rotation = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        source = ramp(20.)
        original, _ = estimate(source)
        rotated, _ = estimate(source @ rotation.T, basis=rotation.T)
        np.testing.assert_array_equal(rotated["valid_mask"], original["valid_mask"])
        np.testing.assert_allclose(rotated["slope_degrees"], original["slope_degrees"], atol=1e-10)
        np.testing.assert_allclose(rotated["normal"], original["normal"] @ rotation.T, atol=1e-10)

    def test_unit_scaling_with_explicit_settings_preserves_measurements(self):
        source = ramp(20.)
        original, metadata = estimate(source)
        factor = 1000.
        settings = dict(metadata["settings"])
        for key in ("lower_layer_height", "neighbor_radius", "second_radius", "min_baseline", "max_residual", "huber_delta"):
            settings[key] *= factor
        scaled, _ = estimate(source * factor, resolution=.1 * factor,
                             voxel_size=.02 * factor, settings=settings)
        np.testing.assert_array_equal(scaled["valid_mask"], original["valid_mask"])
        np.testing.assert_allclose(scaled["slope_degrees"], original["slope_degrees"], atol=1e-9)
        np.testing.assert_allclose(scaled["residual"] / factor, original["residual"], atol=1e-10)

    def test_sparse_and_collinear_centers_are_unknown(self):
        for source in (ramp()[:5], np.array([[x, .15, .01] for x in np.linspace(.01, .79, 50)])):
            with self.subTest(count=len(source)):
                arrays, metadata = estimate(source)
                self.assertFalse(arrays["valid_mask"].any())
                self.assertGreater(sum(metadata["small_scale_failures"].values()), 0)

    def test_large_neighborhood_cannot_rescue_sparse_small_patch_or_hole(self):
        source = ramp()
        distance = np.linalg.norm(source[:, :2] - [.35, .35], axis=1)
        source = np.r_[source[distance > .14], [[.35, .35, .01]]]
        arrays, metadata = estimate(source, settings={"neighbor_radius": .08, "second_radius": .3})
        self.assertFalse(arrays["valid_mask"][3, 3])
        self.assertTrue(np.isnan(arrays["slope_degrees"][3, 3]))
        self.assertGreater(metadata["small_scale_failures"]["insufficient_neighbors"], 0)
        source = ramp()
        source = source[~((source[:, 0] >= .3) & (source[:, 0] < .4) & (source[:, 1] >= .3) & (source[:, 1] < .4))]
        hole, _ = estimate(source)
        self.assertFalse(hole["valid_mask"][3, 3])

    def test_upper_level_is_excluded_by_fixed_lower_layer(self):
        floor = ramp(20.)
        floor_only, _ = estimate(floor)
        stacked, metadata = estimate(np.r_[floor, floor + [0., 0., .5]])
        np.testing.assert_array_equal(stacked["valid_mask"], floor_only["valid_mask"])
        np.testing.assert_allclose(stacked["slope_degrees"], floor_only["slope_degrees"], atol=1e-10)
        self.assertEqual(metadata["lower_layer_center_count"], len(floor))

    def test_duplicates_and_input_order_do_not_change_one_voxel_votes(self):
        source = ramp(20.)
        original = source.copy()
        arrays, metadata = estimate(source)
        repeated, repeated_meta = estimate(np.r_[source[::-1], np.repeat(source[10:20], 40, axis=0)])
        for key in arrays:
            np.testing.assert_allclose(repeated[key], arrays[key], equal_nan=True, atol=1e-10)
        self.assertEqual(repeated_meta["unique_center_count"], metadata["unique_center_count"])
        np.testing.assert_array_equal(source, original)

    def test_voxel_origin_and_resolution_changes_respect_quantization_tolerance(self):
        for size, origin in ((.02, [.009, -.007, .003]), (.04, [-.01, .01, -.015])):
            with self.subTest(size=size):
                arrays, _ = estimate(ramp(20., voxel=size, origin=origin), voxel_size=size,
                                     settings={"neighbor_radius": .2, "second_radius": .3})
                self.assertTrue(arrays["valid_mask"][2:6, 2:6].all())
                np.testing.assert_allclose(arrays["slope_degrees"][2:6, 2:6], 20., atol=4.)

    def test_two_valid_scales_with_disagreeing_normals_remain_unknown(self):
        source = ramp()
        target = np.array([.35, .35])
        distance = np.linalg.norm(source[:, :2] - target, axis=1)
        source[:, 2] = np.where(distance <= .10, .01, .01 + 1.2 * (source[:, 0] - target[0]))
        arrays, metadata = estimate(source, settings={"neighbor_radius": .08, "second_radius": .3,
                                                     "max_residual": .2, "huber_delta": .2,
                                                     "lower_layer_height": .3})
        self.assertFalse(arrays["valid_mask"][3, 3])
        self.assertGreater(metadata["scale_disagreement_cells"], 0)

    def test_valid_agreeing_scales_select_the_steeper_normal(self):
        source = ramp()
        target = np.array([.35, .35])
        distance = np.linalg.norm(source[:, :2] - target, axis=1)
        source[:, 2] = np.where(distance <= .10, .01,
                               .01 + np.tan(np.radians(8.)) * (source[:, 0] - target[0]))
        arrays, _ = estimate(source, settings={"neighbor_radius": .08, "second_radius": .3})
        self.assertTrue(arrays["valid_mask"][3, 3])
        self.assertGreater(arrays["slope_degrees"][3, 3], 3.)
        self.assertLess(arrays["slope_degrees"][3, 3], 10.)
        angle = np.degrees(np.arctan2(np.linalg.norm(arrays["normal"][3, 3, :2]),
                                     arrays["normal"][3, 3, 2]))
        self.assertAlmostEqual(angle, arrays["slope_degrees"][3, 3])

    def test_huber_patch_tolerates_an_isolated_quantized_height_outlier(self):
        source = ramp()
        nearest = np.argmin(np.linalg.norm(source[:, :2] - [.37, .37], axis=1))
        source[nearest, 2] += .045
        arrays, _ = estimate(source)
        self.assertTrue(arrays["valid_mask"][3, 3])
        self.assertLess(arrays["slope_degrees"][3, 3], 1.)
        self.assertLess(arrays["residual"][3, 3], .01)

    def test_coarse_occupied_cubes_cover_finer_grid_without_extra_votes(self):
        voxel = .214
        for angle in (0., 30.):
            with self.subTest(angle=angle):
                source = quantized_reference_plane(angle, np.eye(3), voxel,
                                                  span=2., spacing=.01)
                arrays, metadata = estimate(source, shape=(20, 20), voxel_size=voxel, settings=None)
                self.assertTrue(arrays["valid_mask"].all())
                self.assertEqual(metadata["occupied_columns"], 400)
                self.assertGreater(metadata["query_columns_without_center"], 300)
                self.assertGreater(metadata["outside_grid_center_count"], 0)
                self.assertEqual(metadata["lower_layer_center_count"], len(source))
                self.assertLessEqual(int(arrays["support_count"].max()), len(source))
                self.assertFalse(metadata["query_coverage_is_support_or_free_space"])
                self.assertEqual(metadata["method"], "observed_lower_voxel_two_scale_huber_heightfield_v2_box_coverage")
                if angle == 0:
                    np.testing.assert_allclose(arrays["slope_degrees"], 0., atol=1e-10)
                else:
                    self.assertTrue(np.all(arrays["slope_degrees"] > 25.))
                    np.testing.assert_allclose(arrays["slope_degrees"][3:17, 3:17], angle, atol=4.)
                repeated, _ = estimate(np.repeat(source, 3, axis=0), shape=(20, 20),
                                       voxel_size=voxel, settings=None)
                for key in arrays:
                    np.testing.assert_allclose(repeated[key], arrays[key], equal_nan=True, atol=1e-10)

    def test_rotated_cube_coverage_is_exact_not_bounding_box_fill(self):
        angle = np.pi / 4
        basis = np.array([[np.cos(angle), np.sin(angle), 0.],
                          [-np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
        mask, _ = _voxel_xy_coverage(np.array([[0., 0., 0.]]), basis,
                                     np.array([-1., -1.]), .1, (20, 20), 1.)
        # The projected cube is the diamond |x|+|y| <= sqrt(.5).
        # A square has positive overlap iff its closest diamond-distance is
        # strictly below that bound; corner-only contact has zero area.
        center = -1. + (np.arange(20) + .5) * .1
        nearest = np.maximum(np.abs(center) - .05, 0.)
        expected = nearest[:, None] + nearest[None, :] < np.sqrt(.5) - 1e-12
        np.testing.assert_array_equal(mask, expected)
        self.assertTrue(mask[10, 10])
        self.assertFalse(mask[16, 16])
        aligned, _ = _voxel_xy_coverage(np.array([[.5, .5, .5]]), np.eye(3),
                                        np.array([0., 0.]), .1, (12, 12), 1.)
        self.assertTrue(aligned[:10, :10].all())
        self.assertFalse(aligned[10:, :].any())
        self.assertFalse(aligned[:, 10:].any())

    def test_coarse_cube_coverage_does_not_interpolate_unobserved_holes(self):
        voxel = .214
        source = quantized_reference_plane(0., np.eye(3), voxel, span=2., spacing=.01)
        source = source[(source[:, 0] < .65) | (source[:, 0] > 1.5)]
        arrays, metadata = estimate(source, shape=(20, 20), voxel_size=voxel, settings=None)
        self.assertFalse(arrays["valid_mask"][:, 8:14].any())
        self.assertTrue(np.isnan(arrays["slope_degrees"][:, 8:14]).all())
        self.assertLess(metadata["occupied_columns"], 400)
        self.assertTrue(arrays["valid_mask"][3:17, :3].all())

    def test_coarse_coverage_keeps_sparse_and_small_patch_gates(self):
        sources = (np.array([[.1, .1, .1]]),
                   np.array([[x, .15, .1] for x in np.arange(.107, 2., .214)]))
        for source in sources:
            with self.subTest(count=len(source)):
                arrays, metadata = estimate(source, shape=(20, 20), voxel_size=.214, settings=None)
                self.assertFalse(arrays["valid_mask"].any())
                self.assertGreater(metadata["occupied_columns"], 0)
                self.assertEqual(int(arrays["support_count"].sum()), 0)
        dense = quantized_reference_plane(0., np.eye(3), .214, span=2., spacing=.01)
        arrays, metadata = estimate(dense, shape=(20, 20), voxel_size=.214,
                                   settings={"neighbor_radius": .05, "second_radius": 1.})
        self.assertEqual(metadata["occupied_columns"], 400)
        self.assertFalse(arrays["valid_mask"].any())
        self.assertEqual(metadata["small_scale_failures"]["insufficient_neighbors"], 400)

    def test_boundary_center_columns_do_not_alias_at_grid_row_edges(self):
        source = np.array([[.5, .1, 0.], [.1, .3, 1.]])
        arrays, metadata = estimate(source, shape=(2, 2), resolution=.2, voxel_size=.4,
                                   settings={"lower_layer_height": .05})
        self.assertEqual(metadata["center_columns"], 2)
        self.assertEqual(metadata["lower_layer_center_count"], 2)
        self.assertEqual(metadata["retained_grid_bbox_intersecting_voxels"], 2)
        self.assertFalse(arrays["valid_mask"].any())

    def test_arbitrary_axis_zero_reference_has_quantization_resolution_metadata(self):
        basis = reference_basis()
        for phase in (0., .017, .041):
            with self.subTest(phase=phase):
                source = quantized_reference_plane(0., basis, .05, phase)
                original = source.copy()
                arrays, metadata = estimate(source, basis=basis, voxel_size=.05,
                                           settings={"neighbor_radius": .35, "second_radius": .55})
                slope = arrays["slope_degrees"][2:6, 2:6]
                self.assertTrue(np.isfinite(slope).all())
                # The known source plane is exactly level. Quantized native
                # centers need not define a zero-slope plane after rotation.
                self.assertGreater(float(np.median(slope)), .05)
                self.assertLess(float(np.max(slope)), 3.)
                height_bound = .025 * np.sum(np.abs(basis[2]))
                self.assertAlmostEqual(metadata["reference_up_height_quantization_bound"], height_bound)
                self.assertAlmostEqual(metadata["angular_quantization_resolution_proxy_degrees"]["small"],
                                       math.degrees(math.atan(height_bound / .35)))
                self.assertFalse(metadata["quantization_uncertainty_changes_measurement_or_policy"])
                self.assertIn("not a confidence interval", metadata["angular_proxy_definition"])
                json.dumps(metadata, allow_nan=False)
                np.testing.assert_array_equal(source, original)

    def test_arbitrary_axis_reference_preserves_gentle_and_threshold_ramps(self):
        basis = reference_basis()
        for phase in (0., .017, .041):
            for angle in (5., 24., 26., 30.):
                with self.subTest(angle=angle, phase=phase):
                    source = quantized_reference_plane(angle, basis, .05, phase)
                    arrays, _ = estimate(source, basis=basis, voxel_size=.05,
                                         settings={"neighbor_radius": .35, "second_radius": .55})
                    slope = arrays["slope_degrees"][2:6, 2:6]
                    self.assertTrue(np.isfinite(slope).all())
                    np.testing.assert_allclose(slope, angle, atol=2.)
                    self.assertTrue(np.all(slope < 25.) if angle < 25 else np.all(slope > 25.))
                    if angle == 5.:
                        self.assertTrue(np.all(slope > 2.))

    def test_projected_coverage_work_is_bounded_before_pair_allocation(self):
        with mock.patch("pipeline_common.terrain.MAX_COVERAGE_PAIR_TESTS", 4):
            with self.assertRaisesRegex(ValueError, "bounded pair tests"):
                _voxel_xy_coverage(np.array([[.5, .5, .5]]), np.eye(3),
                                   np.array([0., 0.]), .1, (10, 10), 1.)

    def test_invalid_inputs_and_robot_settings_are_rejected(self):
        bad = [dict(centers=[[0, 0, np.nan]]), dict(centers=[[True, True, True]]),
               dict(basis=np.zeros((3, 3))), dict(origin=[0]), dict(resolution=0),
               dict(voxel_size=True), dict(shape=(True, 8)), dict(shape=(0, 8)), dict(shape=np.array(8)),
               dict(settings={"max_slope_degrees": 25}), dict(settings={"min_neighbors": 5}),
               dict(settings={"huber_iterations": False}), dict(settings={"neighbor_radius": -1}),
               dict(settings={"second_radius": .01}), dict(settings={"max_residual": float("inf")}),
               dict(settings={"max_scale_disagreement_degrees": 91})]
        for arguments in bad:
            with self.subTest(arguments=arguments):
                points = arguments.pop("centers", ramp())
                with self.assertRaises(ValueError):
                    estimate(points, **arguments)


if __name__ == "__main__":
    unittest.main()
