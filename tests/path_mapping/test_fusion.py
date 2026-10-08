"""CPU checks for evidence weighting, pixel alignment and temporal support."""

import unittest

import numpy as np

from path_mapping.fusion import fuse_frame_sequence


def inputs(sequence=2, height=1, width=2):
    shape = (sequence, height, width)
    return {
        "world_points": np.full((*shape, 3), 0.01),
        "point_confidence": np.full(shape, 2.0),
        "depth": np.ones(shape),
        "path_scores": np.ones(shape),
        "rgb": np.full((*shape, 3), 120, dtype=np.uint8),
    }


class FusionTests(unittest.TestCase):
    def test_negative_evidence_uses_all_pixels_and_raw_confidence(self):
        arrays = inputs()
        arrays["point_confidence"][1] = 4
        arrays["path_scores"][0] = 0.9
        arrays["path_scores"][1] = 0
        result = fuse_frame_sequence(**arrays)
        np.testing.assert_allclose(result.path_weights, [3.6])
        np.testing.assert_allclose(result.total_weights, [12])
        np.testing.assert_allclose(result.path_probabilities, [0.3])
        np.testing.assert_array_equal(result.observations, [2])
        np.testing.assert_array_equal(result.point_counts, [4])
        self.assertFalse(result.path_flags[0])
        self.assertEqual(result.path_points.shape, (2, 3))
        np.testing.assert_array_equal(result.path_frame_indices, [0, 0])
        self.assertEqual(result.context_points.shape, (4, 3))

    def test_pixels_do_not_substitute_for_distinct_frames(self):
        result = fuse_frame_sequence(**inputs(sequence=1, width=100))
        self.assertEqual(result.observations.tolist(), [1])
        self.assertEqual(result.point_counts.tolist(), [100])
        self.assertFalse(result.path_flags[0])
        result = fuse_frame_sequence(**inputs(), path_probability_threshold=1)
        self.assertTrue(result.path_flags[0])

    def test_floor_negative_coordinates_and_shifted_origin(self):
        arrays = inputs(sequence=1, width=4)
        arrays["world_points"][0, 0] = [[-0.001, 0, 0], [-0.05, 0, 0], [0, 0, 0], [0.05, 0, 0]]
        result = fuse_frame_sequence(**arrays, min_observations=1)
        np.testing.assert_array_equal(result.voxel_indices, [[-1, 0, 0], [0, 0, 0], [1, 0, 0]])
        np.testing.assert_allclose(result.voxel_centers[:, 0], [-0.025, 0.025, 0.075])
        np.testing.assert_array_equal(result.point_counts, [2, 1, 1])
        shifted = fuse_frame_sequence(**arrays, map_origin=(0.05, 0, 0), min_observations=1)
        np.testing.assert_array_equal(shifted.voxel_indices, [[-2, 0, 0], [-1, 0, 0], [0, 0, 0]])
        np.testing.assert_allclose(shifted.voxel_centers, result.voxel_centers)

    def test_invalid_geometry_is_filtered_before_fusion(self):
        arrays = inputs(sequence=1, width=7)
        arrays["world_points"][0, 0, 0, 0] = np.nan
        arrays["point_confidence"][0, 0, 1] = np.inf
        arrays["depth"][0, 0, 2] = -1
        arrays["depth"][0, 0, 3] = np.inf
        arrays["point_confidence"][0, 0, 4] = 0
        arrays["point_confidence"][0, 0, 5] = 1.49
        arrays["point_confidence"][0, 0, 6] = 100
        result = fuse_frame_sequence(**arrays, min_observations=1)
        self.assertEqual(len(result.context_points), 1)
        self.assertEqual(result.total_weights.tolist(), [100])
        self.assertTrue(result.path_flags[0])

    def test_empty_valid_evidence_retains_output_shapes(self):
        arrays = inputs()
        arrays["depth"][:] = 0
        result = fuse_frame_sequence(**arrays)
        for name in ("path_points", "path_colors", "context_points", "context_colors", "voxel_indices", "voxel_centers", "voxel_colors"):
            self.assertEqual(getattr(result, name).shape, (0, 3), name)
        for name in ("path_scores", "path_confidence", "path_frame_indices", "context_frame_indices", "path_probabilities", "observations", "path_flags", "path_weights", "total_weights", "point_counts"):
            self.assertEqual(getattr(result, name).shape, (0,), name)

    def test_depth_trailing_singleton_and_inclusive_threshold(self):
        arrays = inputs()
        arrays["depth"] = arrays["depth"][..., None]
        arrays["path_scores"][:] = 0.5
        arrays["point_confidence"][:] = 1.5
        result = fuse_frame_sequence(**arrays)
        self.assertTrue(result.path_flags[0])
        self.assertEqual(result.observations.tolist(), [2])

    def test_weighted_voxel_color_and_input_ownership(self):
        arrays = inputs(sequence=1)
        arrays["rgb"][0, 0] = [[0, 0, 0], [200, 100, 50]]
        arrays["point_confidence"][0, 0] = [2, 6]
        original_points = arrays["world_points"].copy()
        result = fuse_frame_sequence(**arrays, min_observations=1)
        np.testing.assert_array_equal(result.voxel_colors, [[150, 75, 38]])
        result.context_points[:] = 42
        np.testing.assert_array_equal(arrays["world_points"], original_points)

    def test_grid_mismatches_and_invalid_scores_are_rejected(self):
        cases = {
            "point_confidence": np.ones((1, 1, 2)),
            "depth": np.ones((2, 1, 2, 2)),
            "path_scores": np.ones((2, 2, 1)),
            "rgb": np.ones((2, 1, 2, 3), dtype=np.float32),
            "world_points": np.ones((2, 1, 2, 4)),
        }
        for key, bad in cases.items():
            with self.subTest(key=key):
                arrays = inputs()
                arrays[key] = bad
                with self.assertRaises(ValueError):
                    fuse_frame_sequence(**arrays)
        for score in (np.nan, np.inf, -0.1, 1.1):
            with self.subTest(score=score):
                arrays = inputs()
                arrays["path_scores"][0, 0, 0] = score
                with self.assertRaisesRegex(ValueError, "finite probabilities"):
                    fuse_frame_sequence(**arrays)

    def test_invalid_settings_are_rejected(self):
        cases = [
            {"voxel_size": 0}, {"voxel_size": -1}, {"voxel_size": np.nan},
            {"voxel_size": True}, {"min_point_confidence": -1},
            {"min_point_confidence": np.inf}, {"path_probability_threshold": 1.1},
            {"path_probability_threshold": np.nan}, {"min_observations": 0},
            {"min_observations": 2.0}, {"min_observations": True},
            {"map_origin": [0, 0]}, {"map_origin": [0, np.inf, 0]},
        ]
        for settings in cases:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                fuse_frame_sequence(**inputs(), **settings)

    def test_nonpositive_confidence_is_filtered_even_with_zero_minimum(self):
        arrays = inputs(sequence=1)
        arrays["point_confidence"][0, 0] = [0, -1]
        result = fuse_frame_sequence(**arrays, min_point_confidence=0)
        self.assertEqual(len(result.voxel_indices), 0)

    def test_extreme_quantization_and_weight_overflow_are_explicit(self):
        arrays = inputs()
        arrays["world_points"][:] = 1e30
        with self.assertRaisesRegex(ValueError, "int64 range"):
            fuse_frame_sequence(**arrays)
        arrays = inputs()
        arrays["point_confidence"][:] = np.finfo(np.float64).max
        with self.assertRaisesRegex(ValueError, "weights overflowed"):
            fuse_frame_sequence(**arrays)


if __name__ == "__main__":
    unittest.main()
