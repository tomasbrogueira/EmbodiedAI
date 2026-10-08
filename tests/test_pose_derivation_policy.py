import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from derive_pose_corrected_run import (
    CORRECTED_METHOD, UNVERIFIED_DISPLAY_METHOD, derivation_policy, numeric_for_derivation,
)


class PoseDerivationPolicyTests(unittest.TestCase):
    def test_missing_evidence_still_rejected_by_default(self):
        with self.assertRaisesRegex(ValueError, 'independent cross-frame evidence'):
            derivation_policy({'correction_supported': False, 'contradictory_pairs': 2})

    def test_display_fallback_preserves_original_numeric_and_blocks_planning(self):
        policy = derivation_policy({'correction_supported': False}, allow_unverified_display_copy=True)
        self.assertEqual(policy['method'], UNVERIFIED_DISPLAY_METHOD)
        self.assertFalse(policy['planning_admitted'])
        self.assertFalse(policy['pose_correction_verified'])
        original = {'extrinsic': np.arange(12, dtype=np.float32).reshape(1, 3, 4),
                    'world_points': np.arange(24, dtype=np.float32).reshape(1, 2, 4, 3)}
        derived = numeric_for_derivation(original, policy)
        for key in original:
            np.testing.assert_array_equal(original[key], derived[key])
            self.assertFalse(np.shares_memory(original[key], derived[key]))

    def test_contradictory_provenance_cannot_enable_correction(self):
        with self.assertRaisesRegex(ValueError, 'contradictory'):
            numeric_for_derivation({}, {'method': CORRECTED_METHOD, 'pose_correction_verified': False})
        with self.assertRaisesRegex(ValueError, 'contradictory'):
            numeric_for_derivation({}, {'method': UNVERIFIED_DISPLAY_METHOD, 'pose_correction_verified': True})

    def test_verified_evidence_uses_the_original_correction_method(self):
        policy = derivation_policy({'correction_supported': True})
        self.assertEqual(policy['method'], CORRECTED_METHOD)
        self.assertTrue(policy['pose_correction_verified'])
        self.assertTrue(policy['planning_admitted'])


if __name__ == '__main__':
    unittest.main()
