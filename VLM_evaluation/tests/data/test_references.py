"""Original-ID reference transfer fixtures; these are not benchmark results."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from traversability_data.annotations import save_annotation
from traversability_data.references import AVOIDED_IDS, PERMITTED_IDS, derive_references


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.run = self.root / "run"
        self.data.mkdir()
        self.run.mkdir()
        Image.new("RGB", (10, 10), "gray").save(self.data / "rgb.png")
        Image.new("L", (10, 10), 255).save(self.data / "mask.png")
        self.frame_id = "fixture:s:f"
        self.region_id = "fixture:s:f:r1"
        self.frame = {"frame_id": self.frame_id, "source": "fixture", "scene_id": "fixture_site",
                      "sequence_id": "s", "timestamp_s": None, "image_path": "rgb.png", "split": "development"}
        self.region = {"region_id": self.region_id, "frame_id": self.frame_id, "mask_path": "mask.png",
                       "planning_relevant": True, "selected_for_classification": False}
        write_rows(self.run / "frames.jsonl", [self.frame])
        write_rows(self.run / "regions.jsonl", [self.region])
        write_rows(self.run / "annotations.jsonl", [{"region_id": self.region_id, "reference_label": None,
                    "semantic_class": None, "hazard_type": None, "mask_quality": "unchecked",
                    "annotation_status": "pending", "robot_profile_id": "rellis_material_v1"}])
        metadata = self.run / "metadata" / "data.json"
        metadata.parent.mkdir()
        metadata.write_text(json.dumps({"is_fixture": True, "robot_profile_id": "rellis_material_v1",
                        "robot_policy": "fixture material policy", "recipe": "fixture", "expected_frames": 1,
                        "semantic_aids": {self.frame_id: "semantic.png"}}), encoding="utf-8")
        self.set_aid(np.ones((10, 10), dtype=np.uint8))

    def set_aid(self, values):
        Image.fromarray(values).save(self.data / "semantic.png")

    def transfer(self):
        return json.loads((self.run / "metadata" / "reference_transfer.jsonl").read_text().splitlines()[0])

    def test_pure_permitted_reference_and_idempotent_resume(self):
        result = derive_references(self.run, self.data)
        self.assertEqual(result[0]["reference_label"], "traversable")
        self.assertEqual(result[0]["semantic_class"], "dirt")
        self.assertEqual(result[0]["mask_quality"], "valid")
        self.assertEqual(result[0]["annotation_source"], "dataset_policy")
        self.assertEqual(derive_references(self.run, self.data), result)
        transfer = self.transfer()
        self.assertEqual(transfer["mask_pixels"], 100)
        self.assertEqual(transfer["nonvoid_coverage"], 1.0)
        self.assertEqual(transfer["class_counts"], {"1": 100})
        self.assertEqual(transfer["class_fractions"], {"1": 1.0})
        self.assertTrue(transfer["is_fixture"])
        metadata = json.loads((self.run / "metadata" / "data.json").read_text())
        self.assertEqual(metadata["recipe"], "fixture")
        self.assertEqual(metadata["semantic_aids"], {self.frame_id: "semantic.png"})

    def test_every_documented_original_class_maps(self):
        for class_id in sorted(PERMITTED_IDS | AVOIDED_IDS):
            with self.subTest(class_id=class_id):
                # New run each time: completed labels and frozen provenance cannot be replaced.
                run = self.root / f"class_{class_id}"
                write_rows(run / "frames.jsonl", [self.frame])
                write_rows(run / "regions.jsonl", [self.region])
                self.set_aid(np.full((10, 10), class_id, dtype=np.uint8))
                row = derive_references(run, self.data, {self.frame_id: "semantic.png"})[0]
                expected = "traversable" if class_id in PERMITTED_IDS else "non_traversable"
                self.assertEqual(row["reference_label"], expected)
                self.assertEqual(row["annotation_status"], "complete")

    def test_one_avoided_pixel_is_never_majority_voted_away(self):
        aid = np.ones((10, 10), dtype=np.uint8)
        aid[0, 0] = 17
        self.set_aid(aid)
        row = derive_references(self.run, self.data)[0]
        self.assertIsNone(row["reference_label"])
        self.assertEqual(row["annotation_status"], "pending")
        self.assertEqual(row["mask_quality"], "mixed")
        self.assertEqual(row["reference_exclusion_reason"], "annotated_hazard_overlap")
        self.assertEqual(self.transfer()["avoided_pixels"], 1)
        self.assertEqual(self.transfer()["dominant_class_fraction"], .99)

    def test_missing_aid_can_resume_to_reference_without_inventing_unknown(self):
        row = derive_references(self.run, self.data, {})[0]
        self.assertIsNone(row["reference_label"])
        self.assertEqual(row["reference_exclusion_reason"], "missing_semantic_aid")
        self.assertIsNone(self.transfer()["avoided_pixels"])
        row = derive_references(self.run, self.data)[0]
        self.assertEqual(row["reference_label"], "traversable")

    def test_grass_and_void_remain_pending(self):
        for class_id in (0, 3):
            with self.subTest(class_id=class_id):
                self.set_aid(np.full((10, 10), class_id, dtype=np.uint8))
                row = derive_references(self.run, self.data)[0]
                self.assertIsNone(row["reference_label"])
                self.assertEqual(row["annotation_status"], "pending")
                self.assertEqual(self.transfer()["avoided_pixels"], 0)

    def test_coverage_and_purity_thresholds_are_applied_to_nonvoid_pixels(self):
        aid = np.ones((10, 10), dtype=np.uint8)
        aid.flat[:6] = 0
        self.set_aid(aid)
        row = derive_references(self.run, self.data)[0]
        self.assertEqual(row["reference_exclusion_reason"], "insufficient_nonvoid_coverage")
        self.assertEqual(self.transfer()["dominant_class_fraction"], 1.0)
        aid.fill(6)
        aid.flat[:6] = 3
        self.set_aid(aid)
        row = derive_references(self.run, self.data)[0]
        self.assertEqual(row["reference_exclusion_reason"], "insufficient_class_purity")
        self.assertEqual(self.transfer()["nonvoid_coverage"], 1.0)

    def test_exact_threshold_is_eligible_when_no_avoided_pixel_is_present(self):
        aid = np.ones((10, 10), dtype=np.uint8)
        aid.flat[:5] = 0
        self.set_aid(aid)
        self.assertEqual(derive_references(self.run, self.data)[0]["reference_label"], "traversable")

    def test_whole_semantic_image_is_validated_outside_region(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[0, 0] = 255
        Image.fromarray(mask).save(self.data / "mask.png")
        aid = np.ones((10, 10), dtype=np.uint8)
        aid[-1, -1] = 2
        self.set_aid(aid)
        before = (self.run / "annotations.jsonl").read_bytes()
        with self.assertRaisesRegex(ValueError, "Unexpected original RELLIS IDs"):
            derive_references(self.run, self.data)
        self.assertEqual((self.run / "annotations.jsonl").read_bytes(), before)

    def test_ground_truth_dimensions_are_checked(self):
        self.set_aid(np.ones((9, 10), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "dimensions"):
            derive_references(self.run, self.data)

    def test_manual_completed_annotation_survives_automatic_derivation(self):
        manual = save_annotation(self.run, self.region_id, "unknown", "valid", "unclear")
        self.assertEqual(derive_references(self.run, self.data)[0], manual)
        self.assertEqual(self.transfer()["dominant_class_id"], 1)
        self.assertEqual(self.transfer()["reference_annotation_source"], "manual")

    def test_optional_manual_annotation_synchronizes_transfer_provenance(self):
        self.set_aid(np.zeros((10, 10), dtype=np.uint8))
        derive_references(self.run, self.data)
        self.assertIsNone(self.transfer()["reference_label"])
        save_annotation(self.run, self.region_id, "unknown", "valid", "unclear")
        self.assertEqual(self.transfer()["reference_label"], "unknown")
        self.assertEqual(self.transfer()["reference_annotation_source"], "manual")
        self.assertIsNone(self.transfer()["reference_exclusion_reason"])
        self.assertEqual(self.transfer()["nonvoid_coverage"], 0)

    def test_different_completed_automatic_reference_rejected_before_any_writes(self):
        derive_references(self.run, self.data)
        paths = [self.run / "annotations.jsonl", self.run / "metadata" / "data.json",
                 self.run / "metadata" / "reference_transfer.jsonl"]
        before = [path.read_bytes() for path in paths]
        self.set_aid(np.full((10, 10), 6, dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "Completed automatic reference differs"):
            derive_references(self.run, self.data)
        self.assertEqual([path.read_bytes() for path in paths], before)

    def test_changed_completed_provenance_is_rejected_even_when_label_matches(self):
        derive_references(self.run, self.data)
        with self.assertRaisesRegex(ValueError, "Frozen reference"):
            derive_references(self.run, self.data, coverage_threshold=.99)

    def test_unknown_aid_frame_and_duplicate_transfer_ids_reject(self):
        with self.assertRaisesRegex(ValueError, "unknown frame"):
            derive_references(self.run, self.data, {"not_a_frame": "semantic.png"})
        derive_references(self.run, self.data)
        transfer = self.transfer()
        write_rows(self.run / "metadata" / "reference_transfer.jsonl", [transfer, transfer])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            derive_references(self.run, self.data)

    def test_threshold_validation(self):
        for value in (0, -1, .90, .94, 1.1, float("nan"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                derive_references(self.run, self.data, coverage_threshold=value)

    def test_thresholds_freeze_even_when_every_reference_is_pending(self):
        self.set_aid(np.zeros((10, 10), dtype=np.uint8))
        derive_references(self.run, self.data)
        original = (self.run / "annotations.jsonl").read_bytes()
        with self.assertRaisesRegex(ValueError, "Frozen reference thresholds"):
            derive_references(self.run, self.data, purity_threshold=.99)
        self.assertEqual(original, (self.run / "annotations.jsonl").read_bytes())


if __name__ == "__main__":
    unittest.main()
