"""End-to-end CPU coverage, resumability and fixed-contract validation."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from traversability_data import (create_cpu_fixture, coverage_summary, export_run,
                                prepare_run, read_jsonl, save_annotation, validate_records)
from traversability_data.storage import atomic_jsonl, data_path, read_json, resolve_roots


class RunTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.data = self.base / "inputs"
        self.run = self.base / "runs" / "fixture"
        self.fixture = create_cpu_fixture(self.data, self.run)
        self.frames = read_jsonl(self.run / "frames.jsonl")
        self.regions = read_jsonl(self.run / "regions.jsonl")
        self.annotations = read_jsonl(self.run / "annotations.jsonl")

    def tearDown(self):
        self.temporary.cleanup()

    def test_fixture_complete_workflow_and_resumption(self):
        first = (self.run / "annotations.jsonl").read_bytes()
        self.assertEqual(len(self.frames), 2)
        self.assertEqual(len(self.regions), 18)
        self.assertEqual(sum(row["selected_for_classification"] for row in self.regions), 10)
        create_cpu_fixture(self.data, self.run)
        self.assertEqual(first, (self.run / "annotations.jsonl").read_bytes())
        summary = coverage_summary(self.run)
        self.assertTrue(summary["is_fixture"])
        self.assertEqual(summary["real_frames_by_split"], {})
        self.assertFalse(summary["complete_planned_frame_coverage"])
        exported = export_run(self.run, self.data, self.base / "export")
        self.assertEqual(first, (exported / "annotations.jsonl").read_bytes())

    def test_preparation_preserves_completed_annotations(self):
        first = (self.run / "annotations.jsonl").read_bytes()
        prepare_run(self.run, self.frames, self.data)
        self.assertEqual(first, (self.run / "annotations.jsonl").read_bytes())
        self.assertEqual(sum(row["annotation_status"] == "complete" for row in self.annotations), 1)
        self.assertTrue(all(row["reference_label"] is None for row in self.annotations if row["annotation_status"] == "pending"))

    def test_changed_policy_and_rgb_rejected(self):
        with self.assertRaisesRegex(ValueError, "metadata"):
            prepare_run(self.run, self.frames, self.data, metadata={"robot_profile_id": "different"})
        path = data_path(self.data, self.frames[0]["image_path"])
        Image.fromarray(np.zeros((24, 32, 3), dtype=np.uint8)).save(path)
        with self.assertRaisesRegex(ValueError, "RGB changed"):
            prepare_run(self.run, self.frames, self.data)

    def test_prepared_recording_sample_is_frozen(self):
        extra = {**self.frames[0], "frame_id": "fixture:development:001"}
        with self.assertRaisesRegex(ValueError, "keyframe sample"):
            prepare_run(self.run, self.frames + [extra], self.data)

    def test_export_rejects_stale_transfer_labels(self):
        atomic_jsonl(self.run / "metadata/reference_transfer.jsonl", [
            {"region_id": self.regions[0]["region_id"], "reference_label": "unknown"}])
        with self.assertRaisesRegex(ValueError, "transfer is stale"):
            export_run(self.run, self.data, self.base / "export")
        self.assertFalse((self.base / "export/annotations.jsonl").exists())

    def test_duplicates_and_missing_joins(self):
        with self.assertRaisesRegex(ValueError, "Duplicate frame_id"):
            validate_records(self.frames + [self.frames[0]], self.regions)
        with self.assertRaisesRegex(ValueError, "Duplicate region_id"):
            validate_records(self.frames, self.regions + [self.regions[0]])
        with self.assertRaisesRegex(ValueError, "Duplicate region_id"):
            validate_records(self.frames, self.regions, self.annotations + [self.annotations[0]])
        invalid = {**self.regions[0], "frame_id": "absent"}
        with self.assertRaisesRegex(ValueError, "missing frame"):
            validate_records(self.frames, [invalid])

    def test_rellis_and_phone_split_leakage(self):
        frame = {**self.frames[0], "source": "rellis", "sequence_id": "00000", "split": "test"}
        with self.assertRaisesRegex(ValueError, "split leakage"):
            validate_records([frame], [])
        frame = {**self.frames[0], "source": "phone", "sequence_id": "A_one", "scene_id": "phone:A"}
        other = {**frame, "frame_id": "phone:A_two:1", "sequence_id": "A_two", "image_path": "other.png", "split": "test"}
        with self.assertRaisesRegex(ValueError, "split leakage"):
            validate_records([frame, other], [])
        # Shared campus identity is allowed for held-out RELLIS recordings.
        validate_records([{**frame, "source": "rellis", "sequence_id": "00000", "scene_id": "campus"},
                          {**other, "source": "rellis", "sequence_id": "00001", "scene_id": "campus"}], [])

    def test_dimensions_channels_and_unsafe_paths(self):
        mask = data_path(self.data, self.regions[0]["mask_path"])
        Image.fromarray(np.ones((3, 4), dtype=np.uint8)).save(mask)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            validate_records(self.frames, self.regions, data_root=self.data)
        for relative in ("../escape.png", "C:/escape.png", "images\\file.png", "/escape.png"):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                data_path(self.data, relative, must_exist=False)

    def test_copied_rgb_pixels_cannot_leak_across_splits(self):
        first = data_path(self.data, self.frames[0]["image_path"])
        second = data_path(self.data, self.frames[1]["image_path"])
        with Image.open(first) as image:
            image.save(second, format="PNG")
        with self.assertRaisesRegex(ValueError, "pixel content split leakage"):
            validate_records(self.frames, [], data_root=self.data)

    def test_label_unknown_is_explicit_and_complete(self):
        pending = next(row for row in self.annotations if row["annotation_status"] == "pending")
        self.assertIsNone(pending["reference_label"])
        completed = save_annotation(self.run, pending["region_id"], "unknown", mask_quality="valid")
        self.assertEqual(completed["annotation_status"], "complete")
        with self.assertRaisesRegex(ValueError, "immutable"):
            save_annotation(self.run, pending["region_id"], "traversable", mask_quality="valid")

    def test_atomic_failure_preserves_previous_file(self):
        original = (self.run / "annotations.jsonl").read_bytes()
        with patch("traversability_data.storage.os.replace", side_effect=OSError("simulated interruption")):
            with self.assertRaises(OSError):
                atomic_jsonl(self.run / "annotations.jsonl", [])
        self.assertEqual(original, (self.run / "annotations.jsonl").read_bytes())
        self.assertFalse(list(self.run.glob("*.tmp")))

    def test_tum_does_not_fill_public_recipe_coverage(self):
        separate = self.base / "supplementary"
        frames = [{**frame, "source": "tum", "frame_id": f"tum:{i}", "sequence_id": f"tum_{i}"}
                  for i, frame in enumerate(self.frames)]
        prepare_run(separate, frames, self.data)
        summary = coverage_summary(separate)
        self.assertEqual(summary["real_frames_by_split"], {})
        self.assertEqual(summary["stored_frames_by_source"], {"tum": 2})

    def test_external_roots_and_precedence(self):
        with patch.dict("os.environ", {"TRAVERSABILITY_DATA_ROOT": str(self.data), "TRAVERSABILITY_RUN_ROOT": "generated"}):
            roots = resolve_roots(repo_root=self.base)
            self.assertEqual(roots.data_root, self.data)
            self.assertEqual(roots.run_root, self.base / "generated")
            explicit = resolve_roots(repo_root=self.base, data_root="explicit")
            self.assertEqual(explicit.data_root, self.base / "explicit")


if __name__ == "__main__":
    unittest.main()
