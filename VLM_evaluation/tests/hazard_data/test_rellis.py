"""Original semantic-ID behavior, with no model or real-dataset dependency."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from common import COMPONENT, read_json
from traversability_hazard_data.rellis import derive_rellis_reference


class RellisReferencesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = read_json(COMPONENT / "configs/hazards/policy.json")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "ids.png"

    def tearDown(self):
        self.temporary.cleanup()

    def derive(self, labels):
        Image.fromarray(np.asarray(labels, dtype=np.uint8)).save(self.path)
        return derive_rellis_reference(self.path, self.policy)

    def test_water_puddle_union_keeps_every_pixel_and_original_shape(self):
        labels = np.full((37, 61), 7, dtype=np.uint8)
        labels[0, 0] = 6
        labels[1, 2] = 31
        labels[-1, -1] = 4
        masks, valid, details = self.derive(labels)
        self.assertEqual(set(masks), {"water", "tree"})
        np.testing.assert_array_equal(masks["water"], np.isin(labels, [6, 31]))
        self.assertEqual(int(masks["water"].sum()), 2)
        self.assertEqual(int(masks["tree"].sum()), 1)
        self.assertEqual(valid.shape, labels.shape)
        self.assertTrue(valid.all())
        self.assertEqual(details["concept_area_fractions"]["tree"], 1 / labels.size)
        self.assertIn("tree", details["tiny_concepts"])

    def test_sky_is_valid_background_and_ignored_ids_are_not_hazards(self):
        labels = np.array([[0, 3, 9], [7, 1, 10], [23, 7, 7]], dtype=np.uint8)
        masks, valid, details = self.derive(labels)
        self.assertEqual(masks, {})
        np.testing.assert_array_equal(valid, ~np.isin(labels, [0, 3, 9]))
        self.assertEqual(details["ignored_pixel_count"], 3)
        self.assertAlmostEqual(details["ignored_fraction"], 1 / 3)

    def test_original_ids_are_used_without_learning_id_remapping(self):
        policy_ids = self.policy["datasets"]["rellis"]["hazard_label_ids"]
        labels = np.full((10, 10), 7, dtype=np.uint8)
        for position, values in enumerate(policy_ids.values()):
            labels.flat[position] = values[0]
        masks, _, _ = self.derive(labels)
        self.assertEqual(set(masks), set(policy_ids))
        self.assertTrue(all(mask.sum() == 1 for mask in masks.values()))
        self.assertNotIn("sky", masks)
        self.assertNotIn("grass", masks)
        self.assertNotIn("object", masks)

    def test_ignored_fraction_on_either_side_of_five_percent(self):
        for ignored in (4, 5, 6):
            with self.subTest(ignored=ignored):
                labels = np.full((10, 10), 7, dtype=np.uint8)
                labels.flat[:ignored] = 0
                labels.flat[-1] = 17
                masks, valid, details = self.derive(labels)
                self.assertEqual(masks["person"].sum(), 1)
                self.assertEqual(valid.sum(), 100 - ignored)
                self.assertEqual(details["ignored_fraction"], ignored / 100)
                self.assertEqual(details["ignored_pixel_count"], ignored)

    def test_tiny_marker_is_relative_and_excludes_equal_threshold(self):
        for size, expected_tiny in ((1000, False), (2000, True)):
            with self.subTest(size=size):
                labels = np.full((10, size // 10), 7, dtype=np.uint8)
                labels[0, 0] = 17
                masks, _, details = self.derive(labels)
                self.assertEqual(masks["person"].sum(), 1)
                self.assertEqual("person" in details["tiny_concepts"], expected_tiny)

    def test_unexpected_semantic_ids_are_errors(self):
        for value in (2, 11, 255):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "[Ii][Dd]|unexpected"):
                self.derive([[7, value]])

    def test_rgb_color_labels_cannot_masquerade_as_original_ids(self):
        Image.fromarray(np.full((3, 4, 3), 7, dtype=np.uint8)).save(self.path)
        with self.assertRaises(ValueError):
            derive_rellis_reference(self.path, self.policy)


if __name__ == "__main__":
    unittest.main()
