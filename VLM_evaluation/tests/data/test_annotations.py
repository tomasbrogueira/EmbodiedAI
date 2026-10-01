"""CPU-only fixtures for optional manual annotation persistence."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from traversability_data.annotations import annotation_viewer, save_annotation, show_region


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class AnnotationTests(unittest.TestCase):
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
        self.frames = [{"frame_id": "fixture:s:f", "source": "fixture", "scene_id": "fixture_site",
                        "sequence_id": "s", "timestamp_s": None, "image_path": "rgb.png",
                        "split": "development"}]
        self.regions = [{"region_id": f"fixture:s:f:r{index}", "frame_id": "fixture:s:f",
                         "mask_path": "mask.png", "planning_relevant": True,
                         "selected_for_classification": True} for index in (1, 2)]
        self.pending = [{"region_id": row["region_id"], "reference_label": None,
                         "semantic_class": None, "hazard_type": None, "mask_quality": "unchecked",
                         "annotation_status": "pending", "robot_profile_id": "rellis_material_v1"}
                        for row in reversed(self.regions)]
        write_rows(self.run / "frames.jsonl", self.frames)
        write_rows(self.run / "regions.jsonl", self.regions)
        write_rows(self.run / "annotations.jsonl", self.pending)
        metadata = self.run / "metadata" / "data.json"
        metadata.parent.mkdir()
        metadata.write_text(json.dumps({"is_fixture": True, "robot_profile_id": "rellis_material_v1",
                                        "robot_policy": "fixture policy"}), encoding="utf-8")

    def rows(self):
        return [json.loads(line) for line in (self.run / "annotations.jsonl").read_text().splitlines()]

    def test_save_joins_by_id_and_preserves_other_pending_rows(self):
        result = save_annotation(self.run, self.regions[0]["region_id"], "traversable", "valid", "dirt")
        self.assertEqual(result["annotation_status"], "complete")
        indexed = {row["region_id"]: row for row in self.rows()}
        self.assertEqual(indexed[self.regions[1]["region_id"]]["annotation_status"], "pending")
        self.assertIsNone(indexed[self.regions[1]["region_id"]]["reference_label"])
        self.assertEqual(result["robot_profile_id"], "rellis_material_v1")

    def test_completed_annotation_is_immutable_and_identical_save_is_noop(self):
        region_id = self.regions[0]["region_id"]
        saved = save_annotation(self.run, region_id, "unknown", "mixed")
        before = (self.run / "annotations.jsonl").read_bytes()
        self.assertEqual(save_annotation(self.run, region_id, "unknown", "mixed"), saved)
        self.assertEqual((self.run / "annotations.jsonl").read_bytes(), before)

    def test_identical_save_repairs_interrupted_transfer_publication(self):
        region_id = self.regions[0]["region_id"]
        transfer = self.run / "metadata/reference_transfer.jsonl"
        write_rows(transfer, [{"region_id": region_id, "reference_label": None,
                               "reference_annotation_source": "dataset_policy",
                               "reference_exclusion_reason": "unresolved_class"}])
        with patch("traversability_data.annotations._publish_transfer", side_effect=RuntimeError("interrupted transfer")):
            with self.assertRaises(RuntimeError):
                save_annotation(self.run, region_id, "unknown", "valid")
        saved = next(row for row in self.rows() if row["region_id"] == region_id)
        self.assertEqual(saved["annotation_status"], "complete")
        before = (self.run / "annotations.jsonl").read_bytes()
        self.assertEqual(save_annotation(self.run, region_id, "unknown", "valid"), saved)
        self.assertEqual((self.run / "annotations.jsonl").read_bytes(), before)
        row = json.loads(transfer.read_text())
        self.assertEqual(row["reference_label"], "unknown")
        self.assertEqual(row["reference_annotation_source"], "manual")
        with self.assertRaisesRegex(ValueError, "immutable"):
            save_annotation(self.run, region_id, "traversable", "valid")
        self.assertEqual((self.run / "annotations.jsonl").read_bytes(), before)

    def test_missing_label_stays_pending_not_unknown(self):
        result = save_annotation(self.run, self.regions[0]["region_id"], None, "broken")
        self.assertIsNone(result["reference_label"])
        self.assertEqual(result["annotation_status"], "pending")

    def test_function_fallback_resumes_by_skipping_completed(self):
        save_annotation(self.run, self.regions[0]["region_id"], "traversable", "valid")
        resumed = annotation_viewer(self.run, self.data, use_widgets=False)
        self.assertEqual(resumed.pending_region_ids, [self.regions[1]["region_id"]])
        resumed.save("non_traversable", "valid", "object")
        self.assertEqual(resumed.pending_region_ids, [])
        with self.assertRaisesRegex(ValueError, "No pending"):
            resumed.save("unknown")

    def test_duplicate_and_unknown_ids_reject_without_replacing_file(self):
        before = (self.run / "annotations.jsonl").read_bytes()
        with self.assertRaises(ValueError):
            save_annotation(self.run, "not_a_region", "unknown")
        self.assertEqual((self.run / "annotations.jsonl").read_bytes(), before)
        write_rows(self.run / "annotations.jsonl", self.pending + [self.pending[0]])
        duplicated = (self.run / "annotations.jsonl").read_bytes()
        with self.assertRaises(ValueError):
            save_annotation(self.run, self.regions[0]["region_id"], "unknown")
        self.assertEqual((self.run / "annotations.jsonl").read_bytes(), duplicated)

    def test_failed_persistence_preserves_prior_annotations(self):
        before = (self.run / "annotations.jsonl").read_bytes()
        with patch("traversability_data.annotations.atomic_jsonl", side_effect=OSError("fixture write failure")):
            with self.assertRaises(OSError):
                save_annotation(self.run, self.regions[0]["region_id"], "unknown")
        self.assertEqual((self.run / "annotations.jsonl").read_bytes(), before)

    def test_invalid_label_and_mask_quality_fail(self):
        with self.assertRaises(ValueError):
            save_annotation(self.run, self.regions[0]["region_id"], "missing")
        with self.assertRaises(ValueError):
            save_annotation(self.run, self.regions[0]["region_id"], "unknown", "excellent")

    def test_viewer_preserves_original_display_alignment(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure = show_region(self.run, self.data, self.regions[0]["region_id"])
        self.addCleanup(plt.close, figure)
        self.assertEqual(figure.axes[0].images[0].get_array().shape, (10, 10, 3))
        self.assertEqual(figure.axes[1].images[1].get_array().shape, (10, 10, 4))
        self.assertEqual(figure.axes[2].images[0].get_array().shape, (10, 10))

    def test_viewer_rejects_unaligned_mask_without_resizing(self):
        import matplotlib
        matplotlib.use("Agg")
        Image.new("L", (5, 10), 255).save(self.data / "mask.png")
        with self.assertRaisesRegex(ValueError, "dimensions"):
            show_region(self.run, self.data, self.regions[0]["region_id"])


if __name__ == "__main__":
    unittest.main()
