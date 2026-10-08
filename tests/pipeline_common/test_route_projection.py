"""Independent directed pinhole fixtures; no renderer, models or saved jobs."""
from copy import deepcopy
import json
import unittest
from unittest import mock

import numpy as np

from pipeline_common.route_projection import project_route_to_source


def transform():
    return {"matrix": np.eye(3).tolist(), "source_shape": [100, 100],
            "processed_shape": [100, 100], "pad_ltrb": [0, 0, 0, 0]}


def project(route=None, **arguments):
    values = dict(intrinsic=np.array([[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]]),
                  world_to_camera=np.column_stack((np.eye(3), np.zeros(3))),
                  source_to_processed=transform(), arrowhead_length_pixels=8., arrow_spacing_pixels=20.)
    values.update(arguments)
    return project_route_to_source([[-.6, 0., 2.], [.6, 0., 2.]] if route is None else route, **values)


class RouteProjectionTests(unittest.TestCase):
    def test_exact_source_pixels_and_start_to_goal_arrow_order(self):
        result = project()
        np.testing.assert_allclose(result["source_segments"], [[[20., 50.], [80., 50.]]], atol=1e-10)
        self.assertEqual(result["report"]["status"], "visible")
        self.assertEqual(result["report"]["visible_segments"], 1)
        self.assertEqual(result["report"]["arrow_count"], 3)
        for triangle in result["source_arrowheads"]:
            delta = triangle[0] - np.mean(triangle[1:], axis=0)
            np.testing.assert_allclose(delta, [8., 0.], atol=1e-10)
        reverse = project([[.6, 0., 2.], [-.6, 0., 2.]])
        np.testing.assert_allclose(reverse["source_segments"], result["source_segments"][:, ::-1], atol=1e-10)
        self.assertTrue(np.all(reverse["source_arrowheads"][:, 0, 0]
                               < np.mean(reverse["source_arrowheads"][:, 1:, 0], axis=1)))

    def test_world_to_camera_translation_and_rotation_are_used_exactly(self):
        camera_path = np.array([[-.6, 0., 2.], [.6, 0., 2.]])
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        translation = np.array([.3, -.5, .2])
        world = (camera_path - translation) @ rotation
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = rotation, translation
        actual = project(world, world_to_camera=pose)
        expected = project(camera_path)
        np.testing.assert_allclose(actual["source_segments"], expected["source_segments"], atol=1e-10)
        np.testing.assert_allclose(actual["source_arrowheads"], expected["source_arrowheads"], atol=1e-10)

    def test_far_to_near_saved_route_is_not_reversed_to_suit_camera(self):
        result = project([[-.4, 0., 4.], [.4, 0., 2.]])
        np.testing.assert_allclose(result["source_segments"], [[[40., 50.], [70., 50.]]], atol=1e-10)
        self.assertTrue(np.all(result["source_arrowheads"][:, 0, 0]
                               > np.mean(result["source_arrowheads"][:, 1:, 0], axis=1)))

    def test_inverse_saved_crop_affine_uses_centers_without_extra_half_pixel(self):
        saved = transform()
        saved.update(matrix=[[.5, 0., -25.25], [0., .5, -.25], [0., 0., 1.]], source_shape=[200, 300])
        result = project(source_to_processed=saved)
        np.testing.assert_allclose(result["source_segments"], [[[90.5, 100.5], [210.5, 100.5]]], atol=1e-10)
        self.assertEqual(result["report"]["pixel_coordinates"], "source_rgb_saved_pixel_centers")
        wide = project([[-4., 0., 2.], [4., 0., 2.]], source_to_processed=saved)
        np.testing.assert_allclose(wide["source_segments"], [[[50.5, 100.5], [248.5, 100.5]]], atol=1e-9)

    def test_exif_rotation_affine_keeps_directed_source_coordinates(self):
        saved = transform()
        saved["matrix"] = [[0., -1., 99.], [1., 0., 0.], [0., 0., 1.]]
        result = project(source_to_processed=saved)
        np.testing.assert_allclose(result["source_segments"], [[[50., 79.], [50., 19.]]], atol=1e-10)
        self.assertTrue(np.all(result["source_arrowheads"][:, 0, 1]
                               < np.mean(result["source_arrowheads"][:, 1:, 1], axis=1)))

    def test_source_roi_and_processed_padding_clip_without_extrapolation(self):
        saved = transform()
        saved["pad_ltrb"] = [10, 5, 10, 5]
        result = project([[-2., 0., 2.], [2., 0., 2.]], source_to_processed=saved)
        np.testing.assert_allclose(result["source_segments"], [[[10., 50.], [89., 50.]]], atol=1e-9)
        self.assertTrue(np.all(result["source_arrowheads"][..., 0] >= 10.))
        self.assertTrue(np.all(result["source_arrowheads"][..., 0] <= 89.))
        outside = project([[0., -2., 2.], [.1, -2., 2.]], source_to_processed=saved)
        self.assertEqual(outside["report"]["status"], "offscreen")
        self.assertEqual(outside["report"]["arrow_count"], 0)
        self.assertEqual(outside["source_segments"].shape, (0, 2, 2))

    def test_head_vertices_outside_crop_are_suppressed(self):
        saved = transform()
        saved["pad_ltrb"] = [0, 10, 0, 0]
        result = project([[-.6, -.8, 2.], [.6, -.8, 2.]], source_to_processed=saved)
        self.assertEqual(result["report"]["status"], "visible")
        self.assertEqual(result["report"]["arrow_count"], 0)
        self.assertGreater(result["report"]["arrowhead_candidates"], 0)

    def test_near_plane_and_image_frustum_clip_before_division(self):
        behind = project([[-.2, 0., -1.], [.2, 0., -.5]])
        self.assertEqual(behind["report"]["status"], "behind")
        self.assertEqual(behind["report"]["visible_segments"], 0)
        crossing = project([[-1., 0., -1.], [.2, 0., 2.]], near_depth=.01)
        self.assertEqual(crossing["report"]["status"], "visible")
        self.assertTrue(np.isfinite(crossing["source_segments"]).all())
        self.assertTrue(np.all(crossing["source_segments"] >= -1e-7))
        self.assertTrue(np.all(crossing["source_segments"] <= 99. + 1e-7))

    def test_optical_depth_occlusion_never_bridges_saved_wall(self):
        depth = np.full((100, 100), 3.)
        depth[:, 40:61] = 1.
        result = project(depth=depth, depth_valid=np.ones((100, 100), bool))
        self.assertEqual(result["report"]["status"], "visible")
        self.assertEqual(result["report"]["visible_segments"], 2)
        self.assertGreater(result["report"]["occluded_samples"], 0)
        for segment in result["source_segments"]:
            self.assertTrue(segment[:, 0].max() < 40. or segment[:, 0].min() > 60.)
        hidden = project(depth=np.ones((100, 100)), depth_valid=np.ones((100, 100), bool))
        self.assertEqual(hidden["report"]["status"], "occluded")
        self.assertEqual(hidden["report"]["arrow_count"], 0)

    def test_depth_interpolation_is_perspective_correct_not_linear_z(self):
        result = project([[-.3, 0., 1.], [1.2, 0., 4.]], depth=np.full((100, 100), 2.2),
                         depth_valid=np.ones((100, 100), bool), depth_relative_tolerance=0.)
        self.assertEqual(result["report"]["visible_segments"], 1)
        end_x = result["source_segments"][0, 1, 0]
        self.assertGreater(end_x, 60.)
        self.assertLess(end_x, 65.)

    def test_missing_or_nonfinite_provided_depth_is_unknown_not_clear(self):
        for depth, valid in ((np.full((100, 100), 3.), np.zeros((100, 100), bool)),
                             (np.full((100, 100), np.nan), np.ones((100, 100), bool))):
            with self.subTest(nonfinite=bool(np.isnan(depth).any())):
                result = project(depth=depth, depth_valid=valid)
                self.assertEqual(result["report"]["status"], "depth_unknown")
                self.assertGreater(result["report"]["unknown_depth_samples"], 0)
                self.assertEqual(result["report"]["visible_segments"], 0)
                self.assertEqual(result["report"]["arrow_count"], 0)
        unchecked = project()
        self.assertEqual(unchecked["report"]["depth_occlusion"], "unchecked_no_depth")

    def test_untrusted_depth_gap_splits_route_and_arrow_pieces(self):
        valid = np.ones((100, 100), bool)
        valid[:, 45:56] = False
        result = project(depth=np.full((100, 100), 3.), depth_valid=valid)
        self.assertEqual(result["report"]["visible_segments"], 2)
        for segment in result["source_segments"]:
            self.assertTrue(segment[:, 0].max() < 45. or segment[:, 0].min() > 55.)

    def test_short_ordered_segments_join_for_display_sized_heads(self):
        route = [[x, 0., 2.] for x in np.linspace(-.6, .6, 61)]
        result = project(route, arrowhead_length_pixels=18., arrow_spacing_pixels=45.)
        self.assertEqual(result["report"]["arrow_count"], 1)
        triangle = result["source_arrowheads"][0]
        np.testing.assert_allclose(triangle[0] - np.mean(triangle[1:], axis=0), [18., 0.], atol=1e-9)

    def test_absolute_relative_depth_tolerance_is_explicit(self):
        depth = np.full((100, 100), 1.95)
        mask = np.ones((100, 100), bool)
        hidden = project(depth=depth, depth_valid=mask, depth_relative_tolerance=0.)
        self.assertEqual(hidden["report"]["status"], "occluded")
        shown = project(depth=depth, depth_valid=mask, depth_relative_tolerance=.03)
        self.assertEqual(shown["report"]["status"], "visible")
        absolute = project(depth=depth, depth_valid=mask, depth_relative_tolerance=0., depth_absolute_tolerance=.1)
        self.assertEqual(absolute["report"]["status"], "visible")

    def test_no_route_or_degenerate_route_produces_no_generic_arrow(self):
        for route in ([], [[0., 0., 2.]], [[0., 0., 2.], [0., 0., 2.]], [[0., 0., 1.], [0., 0., 2.]]):
            with self.subTest(route=route):
                result = project(route)
                self.assertEqual(result["report"]["status"], "no_route")
                self.assertEqual(result["source_arrowheads"].shape, (0, 3, 2))

    def test_arrowhead_display_budget_is_recorded(self):
        result = project(max_arrowheads=1)
        self.assertEqual(result["report"]["arrow_count"], 1)
        self.assertTrue(result["report"]["arrowheads_display_sampled"])
        self.assertEqual(result["report"]["arrowhead_candidates"], 3)

    def test_filled_head_interior_depth_is_checked_not_only_vertices(self):
        depth = np.full((100, 100), 3.)
        valid = np.ones((100, 100), bool)
        baseline = project(depth=depth, depth_valid=valid, arrowhead_length_pixels=16.)
        for missing in (False, True):
            with self.subTest(missing=missing):
                changed_depth, changed_valid = depth.copy(), valid.copy()
                if missing:
                    changed_valid[52, 42] = False
                else:
                    changed_depth[52, 42] = 1.
                actual = project(depth=changed_depth, depth_valid=changed_valid, arrowhead_length_pixels=16.)
                np.testing.assert_allclose(actual["source_segments"], baseline["source_segments"], atol=1e-10)
                self.assertEqual(actual["report"]["arrow_count"], baseline["report"]["arrow_count"] - 1)
                self.assertGreater(actual["report"]["arrowhead_depth_grid_samples"], 0)

    def test_filled_head_depth_budget_omits_heads_without_changing_route(self):
        with mock.patch("pipeline_common.route_projection.MAX_ARROW_DEPTH_PAIR_TESTS", 4):
            result = project(depth=np.full((100, 100), 3.), depth_valid=np.ones((100, 100), bool))
        self.assertEqual(result["report"]["status"], "visible")
        self.assertEqual(result["report"]["visible_segments"], 1)
        self.assertEqual(result["report"]["arrow_count"], 0)
        self.assertGreater(result["report"]["arrowheads_omitted_depth_budget"], 0)
        self.assertLessEqual(result["report"]["arrowhead_depth_pair_tests"], 4)

    def test_inputs_unchanged_and_report_is_finite_json(self):
        route = np.array([[-.6, 0., 2.], [.6, 0., 2.]])
        depth = np.full((100, 100), 3.)
        mask = np.ones((100, 100), bool)
        saved = transform()
        before = (route.copy(), depth.copy(), mask.copy(), deepcopy(saved))
        result = project(route, depth=depth, depth_valid=mask, source_to_processed=saved)
        np.testing.assert_array_equal(route, before[0])
        np.testing.assert_array_equal(depth, before[1])
        np.testing.assert_array_equal(mask, before[2])
        self.assertEqual(saved, before[3])
        json.dumps(result["report"], allow_nan=False)

    def test_invalid_calibration_depth_and_budgets_fail_closed(self):
        cases = [dict(intrinsic=np.zeros((3, 3))), dict(intrinsic=np.eye(3, dtype=bool)),
                 dict(world_to_camera=np.zeros((3, 4))), dict(world_to_camera=np.ones((4, 4))),
                 dict(source_to_processed={}), dict(near_depth=0), dict(depth_relative_tolerance=-1),
                 dict(sample_spacing_pixels=3), dict(arrowhead_length_pixels=.1), dict(arrow_spacing_pixels=1),
                 dict(max_samples=True), dict(max_samples=20001), dict(max_arrowheads=513),
                 dict(depth=np.ones((99, 100)), depth_valid=np.ones((99, 100), bool)),
                 dict(depth=np.ones((100, 100))), dict(depth_valid=np.ones((100, 100), bool)),
                 dict(depth=np.ones((100, 100)), depth_valid=np.ones((100, 100), int))]
        for case in cases:
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    project(**case)
        for route in ([[0., 0., np.nan], [1., 0., 2.]], [[True, True, True]], [[1., 2.]], np.zeros((4097, 3))):
            with self.subTest(route_shape=np.shape(route)):
                with self.assertRaises(ValueError):
                    project(route)
        with self.assertRaisesRegex(ValueError, "sample budget"):
            project(max_samples=1)
        bad_transform = transform()
        bad_transform["pad_ltrb"] = [50, 0, 50, 0]
        with self.assertRaises(ValueError):
            project(source_to_processed=bad_transform)


if __name__ == "__main__":
    unittest.main()
