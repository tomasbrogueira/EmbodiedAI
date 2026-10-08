"""Model-free checks of the lazy adapters and SAM instance reduction."""

from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from path_mapping.models import merge_instance_scores


class InstanceScoreTests(unittest.TestCase):
    def test_overlapping_instances_keep_highest_confidence(self):
        masks = np.array(
            [[[True, True], [False, False]], [[False, True], [True, False]]]
        )
        actual = merge_instance_scores(masks[:, None], np.array([0.6, 0.9]), (2, 2))
        np.testing.assert_allclose(actual, [[0.6, 0.9], [0.9, 0.0]])
        self.assertEqual(actual.dtype, np.float32)

    def test_empty_instances_leave_background_zero(self):
        actual = merge_instance_scores(
            np.zeros((0, 1, 2, 3), dtype=bool), np.zeros(0), (2, 3)
        )
        np.testing.assert_array_equal(actual, np.zeros((2, 3), dtype=np.float32))

    def test_below_threshold_instance_is_excluded(self):
        masks = np.array([[[True, False]], [[False, True]]])
        actual = merge_instance_scores(masks, np.array([0.4, 0.7]), (1, 2), 0.5)
        np.testing.assert_allclose(actual, [[0.0, 0.7]])

    def test_mask_probability_threshold(self):
        actual = merge_instance_scores(
            np.array([[[0.2, 0.8]]]), np.array([0.9]), (1, 2)
        )
        np.testing.assert_allclose(actual, [[0.0, 0.9]])

    def test_mismatched_score_count_is_rejected(self):
        with self.assertRaises(ValueError):
            merge_instance_scores(np.zeros((2, 2, 3), bool), np.ones(1), (2, 3))

    def test_misaligned_mask_grid_is_rejected(self):
        with self.assertRaises(ValueError):
            merge_instance_scores(np.zeros((1, 2, 3), bool), np.ones(1), (3, 2))

    def test_nonfinite_and_out_of_range_scores_are_rejected(self):
        for score in (np.nan, np.inf, -np.inf, -0.1, 1.1):
            with self.subTest(score=score), self.assertRaises(ValueError):
                merge_instance_scores(np.ones((1, 1, 1), bool), np.array([score]), (1, 1))

    def test_nonfinite_masks_are_rejected(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                merge_instance_scores(np.array([[[value]]]), np.array([0.8]), (1, 1))

    def test_invalid_thresholds_are_rejected(self):
        for threshold in (np.nan, -0.1, 1.1):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                merge_instance_scores(
                    np.ones((1, 1, 1), bool), np.array([0.8]), (1, 1), threshold
                )


class LazyImportTests(unittest.TestCase):
    def test_import_does_not_load_models_or_torch(self):
        script = (
            f"import sys; sys.path.insert(0, {str(REPO_ROOT / 'src')!r}); "
            "import path_mapping.models; "
            "assert 'torch' not in sys.modules; "
            "assert 'sam3' not in sys.modules; "
            "assert 'lingbot_map' not in sys.modules"
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=20
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
