"""Hand-checked CPU cases; every generated input is explicitly a fixture."""

from __future__ import annotations

import copy
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest import mock

from PIL import Image

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from traversability_evaluation import evaluate, export_report, render_review_gallery
from traversability_evaluation.__main__ import main as cli_main


MODEL = "clip_vit_b32"
POLICY = "rellis_material_v1"


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def canonical_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


class Fixture:
    """Tiny aligned images and ID-keyed records, never experiment evidence."""

    def __init__(self, root: Path, name: str = "fixture"):
        self.data_root = root / "data"
        self.run_root = root / "runs"
        self.name = name
        self.run = self.run_root / name
        self.data_root.mkdir(exist_ok=True)
        (self.run / "metadata").mkdir(parents=True, exist_ok=True)
        self.frames: list[dict] = []
        self.regions: list[dict] = []
        self.annotations: list[dict] = []
        self.predictions: list[dict] = []
        self.transfer: list[dict] = []
        self.metadata = {
            "fixture": True,
            "model_key": MODEL,
            "configuration": {
                "settings": {"robot_profile_id": POLICY},
                "prompt": "fixture",
                "model": {"checkpoint": "fixture", "revision": "fixture"},
            },
        }
        self.config = {
            "data_root": str(self.data_root),
            "run_root": str(self.run_root),
            "run_name": name,
            "model_keys": [MODEL],
            "reference_policy_id": POLICY,
            "fixture": True,
            "expected_metadata": {"/configuration/prompt": "fixture"},
        }

    def frame(self, split: str = "test", source: str = "rellis", index: int = 0) -> dict:
        sequence = "00000" if split == "development" else "00001"
        frame_id = f"{source}:{sequence}:{index:03d}"
        image_path = f"{self.name}/{source}_{split}_{index:03d}.png"
        (self.data_root / image_path).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4), (30, 40, 50)).save(self.data_root / image_path)
        frame = {
            "frame_id": frame_id,
            "source": source,
            "scene_id": "one-fixture-campus",
            "sequence_id": sequence,
            "timestamp_s": float(index),
            "image_path": image_path,
            "split": split,
            "fixture": True,
        }
        self.frames.append(frame)
        return frame

    def region(self, frame: dict, reference: str | None, prediction: str = "unknown", *,
               quality: str = "valid", status: str = "ok", selected: bool = True,
               semantic_class: str | None = None, prediction_class: str | None = None,
               avoided_pixels: int | None = None) -> str:
        region_id = f"{frame['frame_id']}:r{len(self.regions):02d}"
        mask_path = f"{self.name}/mask_{len(self.regions):02d}.png"
        Image.new("L", (4, 4), 255).save(self.data_root / mask_path)
        self.regions.append({
            "region_id": region_id, "frame_id": frame["frame_id"], "mask_path": mask_path,
            "planning_relevant": True, "selected_for_classification": selected, "fixture": True,
        })
        self.annotations.append({
            "region_id": region_id, "reference_label": reference,
            "semantic_class": semantic_class, "hazard_type": None, "mask_quality": quality,
            "annotation_status": "pending" if reference is None else "complete",
            "robot_profile_id": POLICY, "fixture": True,
        })
        self.predictions.append({
            "region_id": region_id, "frame_id": frame["frame_id"], "model_key": MODEL,
            "label": prediction, "semantic_class": prediction_class, "reason": "fixture",
            "raw_response": "fixture", "status": status,
            "error_code": "fixture_failure" if status == "error" else None, "fixture": True,
        })
        if avoided_pixels is not None:
            self.transfer.append({
                "region_id": region_id, "frame_id": frame["frame_id"],
                "avoided_pixels": avoided_pixels, "mask_pixels": 16,
                "mask_sha256": hashlib.sha256((self.data_root / mask_path).read_bytes()).hexdigest(),
                "policy_id": POLICY, "reference_label": reference, "is_fixture": True,
            })
        return region_id

    def inference_metadata(self) -> None:
        """Construct the documented metadata shape by hand, with no sibling import."""
        paths = {frame["image_path"] for frame in self.frames}
        paths.update(region["mask_path"] for region in self.regions)
        contents = {
            path: {"status": "present", "sha256": hashlib.sha256((self.data_root / path).read_bytes()).hexdigest()}
            for path in sorted(paths)
        }
        base = {
            "metadata_version": 1, "model_key": MODEL,
            "configuration": {
                "configuration_version": 1, "model_key": MODEL,
                "model": {"checkpoint": "fixture", "revision": "fixture", "family": "clip"},
                "settings": {
                    "robot_profile_id": POLICY, "robot_policy": "Fixture: permit concrete; avoid water.",
                    "language_quantization_bits": 4, "scene_visual_token_budget": 384,
                    "crop_visual_token_budget": 256, "context_token_limit": 2048,
                    "max_new_tokens": 128, "request_concurrency": 1, "seed": 0,
                    "enable_thinking": False, "device": "cuda:0",
                },
                "preprocessing": {"version": 1, "mask": "nonzero foreground", "crop": "fixture"},
                "prompt": "Fixture instruction.",
                "labels": ["traversable", "non_traversable", "unknown"],
                "software_versions": {"python": "fixture", "Pillow": "fixture"},
            },
            "execution": {"kind": "injected_fixture"},
            "inputs": {
                "frames": sorted(self.frames, key=lambda row: row["frame_id"]),
                "regions": sorted(self.regions, key=lambda row: row["region_id"]), "contents": contents,
            },
        }
        self.metadata = copy.deepcopy(base)
        self.metadata["actual_backend"] = {"implementation": "handwritten_fixture"}
        self.metadata["configuration_fingerprint"] = canonical_hash(base)
        self.metadata["compatibility_fingerprint"] = canonical_hash(self.metadata)
        self.config.pop("expected_metadata")

    def emulate_real_provenance_for_coverage_check(self) -> None:
        """Only exercise coverage branches; these temporary inputs remain synthetic tests."""
        for records in (self.frames, self.regions, self.annotations, self.predictions, self.transfer):
            for record in records:
                record.pop("fixture", None)
                record.pop("is_fixture", None)
        self.metadata.pop("fixture", None)
        self.config["fixture"] = False

    def save(self) -> None:
        for name in ("frames", "regions", "annotations"):
            write_jsonl(self.run / f"{name}.jsonl", getattr(self, name))
        write_jsonl(self.run / "predictions" / f"{MODEL}.jsonl", self.predictions)
        (self.run / "metadata" / f"{MODEL}.json").write_text(
            json.dumps(self.metadata), encoding="utf-8")
        if self.transfer:
            write_jsonl(self.run / "metadata" / "reference_transfer.jsonl", self.transfer)

    def evaluate(self) -> dict:
        self.save()
        return evaluate(self.config)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = Fixture(Path(self.temporary.name))

    def comparison(self, report: dict, split: str = "test", run_name: str | None = None) -> dict:
        rows = [row for row in report["comparison"] if row["split"] == split]
        if run_name is not None:
            rows = [row for row in rows if row["run_name"] == run_name]
        self.assertEqual(len(rows), 1, rows)
        return rows[0]

    def assert_rate(self, actual: dict, numerator: int, denominator: int):
        self.assertEqual(actual["numerator"], numerator)
        self.assertEqual(actual["denominator"], denominator)
        if denominator:
            self.assertAlmostEqual(actual["rate"], numerator / denominator)
        else:
            self.assertIsNone(actual["rate"])

    def assert_invalid(self, report: dict):
        self.assertTrue(report["validation"]["errors"], report)
        self.assertTrue(all(row["status"] == "invalid" for row in report["comparison"]))
        self.assertTrue(all(row["metrics"] is None for row in report["comparison"]))

    def test_manually_checked_nine_region_metrics_and_matrix(self):
        fixture = self.fixture
        frame = fixture.frame()
        pairs = [
            ("traversable", "traversable", "ok"),
            ("traversable", "non_traversable", "ok"),
            ("traversable", "unknown", "ok"),
            ("non_traversable", "traversable", "ok"),
            ("non_traversable", "traversable", "ok"),
            ("non_traversable", "non_traversable", "ok"),
            ("non_traversable", "unknown", "error"),
            ("unknown", "traversable", "ok"),
            ("unknown", "unknown", "ok"),
        ]
        for reference, prediction, status in pairs:
            fixture.region(frame, reference, prediction, status=status)
        row = self.comparison(fixture.evaluate())
        self.assertEqual(row["status"], "scored")
        self.assertEqual(row["eligible_regions"], 9)
        self.assertEqual(row["prediction_failures"], 1)
        self.assertEqual(row["confusion_matrix"], [[1, 1, 1], [2, 1, 1], [1, 0, 1]])
        self.assert_rate(row["metrics"]["unsafe_acceptance"], 2, 4)
        self.assert_rate(row["metrics"]["useful_acceptance"], 1, 3)
        self.assert_rate(row["metrics"]["predicted_unknown_rate"], 3, 9)
        self.assert_rate(row["metrics"]["reference_unknown_acceptance"], 1, 2)

    def test_each_artifact_can_be_shuffled_independently(self):
        fixture = self.fixture
        for split in ("development", "test"):
            frame = fixture.frame(split)
            for reference, prediction in [("traversable", "traversable"),
                                          ("non_traversable", "unknown"),
                                          ("unknown", "non_traversable")]:
                fixture.region(frame, reference, prediction, avoided_pixels=1)
        original = fixture.evaluate()
        for index, records in enumerate((fixture.frames, fixture.regions, fixture.annotations,
                                         fixture.predictions, fixture.transfer)):
            random.Random(index + 10).shuffle(records)
        shuffled = fixture.evaluate()
        for split in ("development", "test"):
            first, second = self.comparison(original, split), self.comparison(shuffled, split)
            self.assertEqual(first["metrics"], second["metrics"])
            self.assertEqual(first["confusion_matrix"], second["confusion_matrix"])

    def test_completed_unknown_pending_and_missing_annotations_are_distinct(self):
        fixture = self.fixture
        frame = fixture.frame()
        fixture.region(frame, "unknown", "traversable")
        fixture.region(frame, None, "traversable")
        missing_annotation = fixture.region(frame, "non_traversable", "traversable")
        fixture.annotations = [row for row in fixture.annotations if row["region_id"] != missing_annotation]
        report = fixture.evaluate()
        row = self.comparison(report)
        self.assertEqual(row["eligible_regions"], 1)
        self.assert_rate(row["metrics"]["reference_unknown_acceptance"], 1, 1)
        self.assert_rate(row["metrics"]["unsafe_acceptance"], 0, 0)
        self.assertEqual(row["confusion_matrix"], [[0, 0, 0], [0, 0, 0], [1, 0, 0]])
        coverage = report["coverage"]["annotations"]
        self.assertEqual(coverage["expected"], 3)
        self.assertEqual(coverage["available"], 2)
        self.assertEqual(coverage["missing"], 1)
        self.assertEqual(coverage["pending"], 1)
        self.assertEqual(coverage["complete"], 1)
        self.assertEqual(coverage["complete_unknown"], 1)
        self.assertEqual(coverage["primary_eligible"], 1)

    def test_error_status_is_unknown_even_when_label_says_traversable(self):
        fixture = self.fixture
        fixture.region(fixture.frame(), "non_traversable", "traversable", status="error")
        report = fixture.evaluate()
        row = self.comparison(report)
        self.assert_rate(row["metrics"]["unsafe_acceptance"], 0, 1)
        self.assert_rate(row["metrics"]["predicted_unknown_rate"], 1, 1)
        self.assertEqual(row["prediction_failures"], 1)
        self.assertEqual(row["confusion_matrix"], [[0, 0, 0], [0, 0, 1], [0, 0, 0]])
        self.assertTrue(report["validation"]["warnings"])

    def test_missing_primary_prediction_blocks_every_split(self):
        fixture = self.fixture
        fixture.region(fixture.frame("development"), "traversable", "traversable")
        missing = fixture.region(fixture.frame(), "non_traversable", "traversable")
        fixture.predictions = [row for row in fixture.predictions if row["region_id"] != missing]
        report = fixture.evaluate()
        for row in report["comparison"]:
            self.assertEqual(row["status"], "incomplete")
            self.assertIsNone(row["metrics"])
            self.assertIsNone(row["confusion_matrix"])
        row = self.comparison(report)
        self.assertEqual(row["missing_predictions"], 1)
        self.assertEqual(row["missing_prediction_ids"], [missing])

    def test_missing_prediction_for_excluded_region_is_still_explicit(self):
        fixture = self.fixture
        frame = fixture.frame()
        fixture.region(frame, "traversable", "traversable")
        missing = fixture.region(frame, None, "traversable", quality="mixed")
        fixture.predictions.pop()
        row = self.comparison(fixture.evaluate())
        self.assertEqual(row["status"], "scored")
        self.assertEqual(row["missing_predictions"], 1)
        self.assertEqual(row["missing_prediction_ids"], [missing])
        self.assert_rate(row["metrics"]["useful_acceptance"], 1, 1)

    def test_empty_denominators_and_no_eligible_regions_are_null(self):
        fixture = self.fixture
        fixture.region(fixture.frame(), None)
        row = self.comparison(fixture.evaluate())
        self.assertEqual(row["status"], "no_eligible_regions")
        for metric in row["metrics"].values():
            self.assert_rate(metric, 0, 0)
        self.assertEqual(row["confusion_matrix"], [[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    def test_unselected_regions_do_not_enter_metrics(self):
        fixture = self.fixture
        frame = fixture.frame()
        fixture.region(frame, "non_traversable", "non_traversable")
        fixture.region(frame, "non_traversable", "traversable", selected=False)
        row = self.comparison(fixture.evaluate())
        self.assertEqual(row["eligible_regions"], 1)
        self.assert_rate(row["metrics"]["unsafe_acceptance"], 0, 1)

    def test_duplicate_records_are_invalid_in_every_joined_artifact(self):
        for artifact in ("frames", "regions", "annotations", "predictions", "transfer"):
            with self.subTest(artifact=artifact):
                fixture = Fixture(Path(self.temporary.name), artifact)
                fixture.region(fixture.frame(), "traversable", "traversable", avoided_pixels=0)
                records = getattr(fixture, artifact)
                records.append(copy.deepcopy(records[0]))
                self.assert_invalid(fixture.evaluate())

    def test_orphan_records_do_not_silently_disappear(self):
        for artifact, key in (("regions", "frame_id"), ("annotations", "region_id"),
                              ("predictions", "region_id"), ("transfer", "region_id")):
            with self.subTest(artifact=artifact):
                fixture = Fixture(Path(self.temporary.name), f"orphan_{artifact}")
                fixture.region(fixture.frame(), "traversable", "traversable", avoided_pixels=0)
                getattr(fixture, artifact)[0][key] = "nonexistent-id"
                self.assert_invalid(fixture.evaluate())

    def test_wrong_model_frame_reference_policy_or_configuration_is_invalid(self):
        changes = [
            ("prediction_model", lambda fixture: fixture.predictions[0].update(model_key="qwen3_vl_4b")),
            ("prediction_frame", lambda fixture: fixture.predictions[0].update(frame_id="wrong-frame")),
            ("policy", lambda fixture: fixture.annotations[0].update(robot_profile_id="wrong-policy")),
            ("metadata_model", lambda fixture: fixture.metadata.update(model_key="qwen3_vl_4b")),
            ("metadata_configuration", lambda fixture: fixture.metadata["configuration"].update(prompt="unexpected")),
        ]
        for name, change in changes:
            with self.subTest(change=name):
                fixture = Fixture(Path(self.temporary.name), name)
                fixture.region(fixture.frame(), "traversable", "traversable")
                change(fixture)
                self.assert_invalid(fixture.evaluate())

    def test_pinned_metadata_fingerprint_detects_configuration_change(self):
        fixture = self.fixture
        fixture.region(fixture.frame(), "traversable", "traversable")
        canonical = json.dumps(fixture.metadata, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, allow_nan=False).encode("utf-8")
        fixture.config["comparisons"] = [{
            "run_name": fixture.name, "model_key": MODEL,
            "expected_metadata_sha256": hashlib.sha256(canonical).hexdigest(),
        }]
        self.assertEqual(self.comparison(fixture.evaluate())["status"], "scored")
        fixture.metadata["configuration"]["model"]["revision"] = "different-revision"
        self.assert_invalid(fixture.evaluate())

    def test_invalid_labels_statuses_and_required_fields_are_rejected(self):
        changes = [
            ("reference_label", lambda fixture: fixture.annotations[0].update(reference_label="grass")),
            ("prediction_label", lambda fixture: fixture.predictions[0].update(label="accept")),
            ("prediction_status", lambda fixture: fixture.predictions[0].update(status="timeout")),
            ("complete_null", lambda fixture: fixture.annotations[0].update(reference_label=None)),
            ("missing_frame_field", lambda fixture: fixture.frames[0].pop("scene_id")),
        ]
        for name, change in changes:
            with self.subTest(case=name):
                fixture = Fixture(Path(self.temporary.name), name)
                fixture.region(fixture.frame(), "traversable", "traversable")
                change(fixture)
                self.assert_invalid(fixture.evaluate())

    def test_malformed_jsonl_cannot_yield_a_partial_score(self):
        fixture = self.fixture
        fixture.region(fixture.frame(), "traversable", "traversable")
        fixture.save()
        path = fixture.run / "predictions" / f"{MODEL}.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write("{malformed fixture record}\n")
        self.assert_invalid(evaluate(fixture.config))

    def test_duplicate_json_keys_are_not_silently_last_wins(self):
        fixture = self.fixture
        fixture.region(fixture.frame(), "traversable", "traversable")
        fixture.save()
        path = fixture.run / "annotations.jsonl"
        text = path.read_text(encoding="utf-8")
        text = text.replace('"reference_label": "traversable"',
                            '"reference_label": "non_traversable", "reference_label": "traversable"')
        path.write_text(text, encoding="utf-8")
        self.assert_invalid(evaluate(fixture.config))

    def test_invalid_split_and_recording_leakage_are_rejected(self):
        for name in ("split", "recording_leak"):
            with self.subTest(case=name):
                fixture = Fixture(Path(self.temporary.name), name)
                fixture.region(fixture.frame("development"), "traversable", "traversable")
                fixture.region(fixture.frame("test"), "traversable", "traversable")
                if name == "split":
                    fixture.frames[0]["split"] = "validation"
                else:
                    fixture.frames[1]["sequence_id"] = fixture.frames[0]["sequence_id"]
                self.assert_invalid(fixture.evaluate())

    def test_phone_location_leakage_is_rejected_despite_distinct_recordings(self):
        fixture = self.fixture
        fixture.region(fixture.frame("development", source="phone"), "traversable", "traversable")
        fixture.region(fixture.frame("test", source="phone"), "traversable", "traversable")
        self.assertNotEqual(fixture.frames[0]["sequence_id"], fixture.frames[1]["sequence_id"])
        self.assertEqual(fixture.frames[0]["scene_id"], fixture.frames[1]["scene_id"])
        self.assert_invalid(fixture.evaluate())

    def test_annotation_quality_excludes_mixed_broken_and_unchecked(self):
        fixture = self.fixture
        frame = fixture.frame()
        for quality in ("valid", "mixed", "broken", "unchecked"):
            fixture.region(frame, "non_traversable", "traversable", quality=quality)
        report = fixture.evaluate()
        row = self.comparison(report)
        self.assertEqual(row["eligible_regions"], 1)
        self.assert_rate(row["metrics"]["unsafe_acceptance"], 1, 1)
        quality_rows = {row["mask_quality"]: row for row in report["mask_breakdown"]
                        if row["split"] == "test" and row["source"] == "rellis"}
        for quality in ("valid", "mixed", "broken", "unchecked"):
            self.assertEqual(quality_rows[quality]["regions"], 1)
            self.assertEqual(quality_rows[quality]["complete"], 1)

    def test_physically_defective_masks_are_excluded_from_primary_metrics(self):
        defects = ("empty", "wrong_dimensions", "rgb", "unreadable", "missing")
        for defect in defects:
            with self.subTest(defect=defect):
                fixture = Fixture(Path(self.temporary.name), defect)
                frame = fixture.frame()
                fixture.region(frame, "traversable", "traversable")
                fixture.region(frame, "non_traversable", "traversable")
                mask = fixture.data_root / fixture.regions[-1]["mask_path"]
                if defect == "empty":
                    Image.new("L", (4, 4), 0).save(mask)
                elif defect == "wrong_dimensions":
                    Image.new("L", (3, 4), 255).save(mask)
                elif defect == "rgb":
                    Image.new("RGB", (4, 4), "white").save(mask)
                elif defect == "unreadable":
                    mask.write_text("fixture: this is not an image", encoding="utf-8")
                else:
                    mask.unlink()
                report = fixture.evaluate()
                row = self.comparison(report)
                self.assertEqual(row["eligible_regions"], 1)
                self.assertEqual(row["selected_regions"], 2)
                self.assert_rate(row["metrics"]["unsafe_acceptance"], 0, 0)
                self.assert_rate(row["metrics"]["useful_acceptance"], 1, 1)
                self.assertEqual(sum(row["physical_defects"] for row in report["mask_breakdown"]
                                     if row["split"] == "test" and row["source"] == "rellis"), 1)

    def test_single_channel_png_modes_use_every_nonzero_value_as_foreground(self):
        fixture = self.fixture
        frame = fixture.frame()
        for mode, value in (("1", 1), ("L", 1), ("I;16", 512)):
            fixture.region(frame, "non_traversable", "traversable")
            Image.new(mode, (4, 4), value).save(fixture.data_root / fixture.regions[-1]["mask_path"])
        report = fixture.evaluate()
        row = self.comparison(report)
        self.assertEqual(row["eligible_regions"], 3)
        self.assert_rate(row["metrics"]["unsafe_acceptance"], 3, 3)
        self.assertTrue(all(case["foreground_pixels"] == 16 for case in report["error_cases"]))
        self.assertTrue(all(case["mask_defect"] is None for case in report["error_cases"]))

    def test_empty_single_channel_modes_are_all_excluded(self):
        fixture = self.fixture
        frame = fixture.frame()
        for mode in ("1", "L", "I;16"):
            fixture.region(frame, "non_traversable", "traversable")
            Image.new(mode, (4, 4), 0).save(fixture.data_root / fixture.regions[-1]["mask_path"])
        report = fixture.evaluate()
        self.assertEqual(self.comparison(report)["eligible_regions"], 0)
        self.assertTrue(all(case["foreground_pixels"] == 0 for case in report["error_cases"]))
        self.assertTrue(all(case["mask_defect"] == "mask_empty" for case in report["error_cases"]))

    def test_path_escape_is_rejected_without_opening_external_input(self):
        for field in ("image_path", "mask_path"):
            with self.subTest(field=field):
                fixture = Fixture(Path(self.temporary.name), field)
                fixture.region(fixture.frame(), "traversable", "traversable")
                target = fixture.frames if field == "image_path" else fixture.regions
                target[0][field] = "../outside-fixture.png"
                self.assert_invalid(fixture.evaluate())

    def test_hazard_overlap_includes_mixed_pending_and_error_as_unknown(self):
        fixture = self.fixture
        frame = fixture.frame()
        fixture.region(frame, "non_traversable", "traversable", avoided_pixels=2)
        fixture.region(frame, None, "traversable", quality="mixed", avoided_pixels=1)
        fixture.region(frame, "non_traversable", "traversable", status="error", avoided_pixels=1)
        fixture.region(frame, "traversable", "traversable", avoided_pixels=0)
        report = fixture.evaluate()
        primary = self.comparison(report)
        self.assert_rate(primary["metrics"]["unsafe_acceptance"], 1, 2)
        rows = [row for row in report["hazard_overlap"] if row["split"] == "test" and row["source"] == "all"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "scored")
        self.assert_rate(rows[0]["metric"], 2, 3)

    def test_hazard_overlap_missing_predictions_and_missing_evidence_are_explicit(self):
        fixture = self.fixture
        fixture.region(fixture.frame(), None, "traversable", quality="mixed", avoided_pixels=1)
        fixture.predictions.clear()
        report = fixture.evaluate()
        rows = [row for row in report["hazard_overlap"] if row["split"] == "test" and row["source"] == "all"]
        self.assertEqual(rows[0]["status"], "incomplete")
        self.assertIsNone(rows[0]["metric"])
        fixture.transfer.clear()
        (fixture.run / "metadata" / "reference_transfer.jsonl").unlink()
        report = fixture.evaluate()
        rows = [row for row in report["hazard_overlap"] if row["split"] == "test" and row["source"] == "all"]
        self.assertEqual(rows[0]["status"], "unavailable")
        self.assertIsNone(rows[0]["metric"])

    def test_partial_transfer_evidence_cannot_improve_hazard_diagnostic(self):
        fixture = self.fixture
        frame = fixture.frame()
        fixture.region(frame, "non_traversable", "non_traversable", avoided_pixels=1)
        absent_evidence = fixture.region(frame, "non_traversable", "traversable")
        report = fixture.evaluate()
        rows = [row for row in report["hazard_overlap"] if row["split"] == "test" and row["source"] == "all"]
        self.assertEqual(rows[0]["status"], "unavailable")
        self.assertIsNone(rows[0]["metric"])
        self.assertEqual(rows[0]["missing_transfer_ids"], [absent_evidence])

    def test_actual_nullable_transfer_sidecar_represents_missing_semantic_aid(self):
        fixture = self.fixture
        frame = fixture.frame()
        fixture.region(frame, "non_traversable", "non_traversable", avoided_pixels=1)
        missing_aid = fixture.region(frame, None, "traversable")
        mask_path = fixture.data_root / fixture.regions[-1]["mask_path"]
        fixture.transfer.append({
            "region_id": missing_aid, "frame_id": frame["frame_id"], "mask_pixels": 16,
            "mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
            "avoided_pixels": None, "semantic_aid_path": None, "semantic_aid_sha256": None,
            "policy_id": POLICY, "reference_label": None, "is_fixture": True,
            "reference_exclusion_reason": "missing_semantic_aid",
        })
        report = fixture.evaluate()
        self.assertFalse(report["validation"]["errors"])
        self.assertEqual(self.comparison(report)["status"], "scored")
        row = next(row for row in report["hazard_overlap"] if row["split"] == "test" and row["source"] == "all")
        self.assertEqual(row["status"], "unavailable")
        self.assertIsNone(row["metric"])

    def test_actual_transfer_provenance_mismatches_are_rejected(self):
        changes = {
            "policy_id": "wrong-policy", "frame_id": "wrong-frame", "mask_sha256": "0" * 64,
            "mask_pixels": 15, "reference_label": "traversable", "avoided_pixels": 17,
        }
        for field, changed in changes.items():
            with self.subTest(field=field):
                fixture = Fixture(Path(self.temporary.name), f"transfer_{field}")
                fixture.region(fixture.frame(), "non_traversable", "traversable", avoided_pixels=1)
                fixture.transfer[0][field] = changed
                self.assert_invalid(fixture.evaluate())

    def test_recognized_inference_metadata_checks_inputs_and_both_fingerprints(self):
        fixture = self.fixture
        fixture.region(fixture.frame(), "non_traversable", "traversable")
        fixture.inference_metadata()
        original = copy.deepcopy(fixture.metadata)
        self.assertEqual(self.comparison(fixture.evaluate())["status"], "scored")
        Image.new("RGB", (4, 4), (200, 40, 50)).save(fixture.data_root / fixture.frames[0]["image_path"])
        self.assert_invalid(fixture.evaluate())
        Image.new("RGB", (4, 4), (30, 40, 50)).save(fixture.data_root / fixture.frames[0]["image_path"])
        fixture.metadata = copy.deepcopy(original)
        fixture.metadata["actual_backend"]["implementation"] = "mutated-fixture"
        self.assert_invalid(fixture.evaluate())
        fixture.metadata = copy.deepcopy(original)
        fixture.metadata["configuration"]["settings"]["robot_profile_id"] = "wrong-policy"
        self.assert_invalid(fixture.evaluate())
        fixture.metadata = copy.deepcopy(original)
        fixture.metadata["configuration_fingerprint"] = "0" * 64
        signed = {key: value for key, value in fixture.metadata.items() if key != "compatibility_fingerprint"}
        fixture.metadata["compatibility_fingerprint"] = canonical_hash(signed)
        self.assert_invalid(fixture.evaluate())

    def test_injected_inference_metadata_cannot_run_as_real(self):
        fixture = self.fixture
        fixture.region(fixture.frame(), "traversable", "traversable")
        fixture.emulate_real_provenance_for_coverage_check()
        fixture.inference_metadata()
        self.assert_invalid(fixture.evaluate())

    def test_fixture_is_not_usable_as_real_experiment_results(self):
        fixture = self.fixture
        fixture.region(fixture.frame(), "traversable", "traversable")
        report = fixture.evaluate()
        frame_coverage = report["coverage"]["frames"]
        self.assertTrue(frame_coverage)
        self.assertTrue(all(row["real_available"] == 0 for row in frame_coverage))
        self.assertTrue(all(row["real_readable"] == 0 for row in frame_coverage))
        self.assertEqual(report["coverage"]["status"], "fixture_only")
        phone = report["coverage"]["phone_hazard_clips"]
        self.assertEqual(phone["expected"], 6)
        self.assertEqual(phone["available_recordings"], 0)
        self.assertEqual(phone["not_available"], 6)
        self.assertEqual(phone["small_hazard_coverage"], "not_established")
        fixture.config["fixture"] = False
        self.assert_invalid(fixture.evaluate())

    def test_metadata_fixture_provenance_alone_prevents_real_scoring(self):
        fixture = self.fixture
        fixture.region(fixture.frame(), "traversable", "traversable")
        for records in (fixture.frames, fixture.regions, fixture.annotations, fixture.predictions):
            for record in records:
                record.pop("fixture")
        fixture.config["fixture"] = False
        self.assertTrue(fixture.metadata["fixture"])
        report = fixture.evaluate()
        self.assert_invalid(report)
        for row in report["coverage"].get("frames", []):
            self.assertEqual(row["real_available"], 0)
            self.assertEqual(row["real_readable"], 0)

    def test_per_source_rows_do_not_mix_development_and_held_out(self):
        fixture = self.fixture
        fixture.region(fixture.frame("development"), "non_traversable", "traversable")
        fixture.region(fixture.frame("test"), "non_traversable", "non_traversable")
        fixture.region(fixture.frame("test", source="tum"), "traversable", "unknown")
        report = fixture.evaluate()
        by_key = {(row["split"], row["source"]): row for row in report["per_source"]}
        self.assert_rate(by_key[("development", "rellis")]["metrics"]["unsafe_acceptance"], 1, 1)
        self.assert_rate(by_key[("test", "rellis")]["metrics"]["unsafe_acceptance"], 0, 1)
        self.assert_rate(by_key[("test", "tum")]["metrics"]["predicted_unknown_rate"], 1, 1)
        self.assert_rate(self.comparison(report, "development")["metrics"]["unsafe_acceptance"], 1, 1)
        self.assert_rate(self.comparison(report)["metrics"]["unsafe_acceptance"], 0, 1)

    def test_frame_totals_do_not_hide_missing_expected_recordings(self):
        fixture = self.fixture
        fixture.region(fixture.frame("development"), "traversable", "traversable")
        for index in range(4):
            fixture.region(fixture.frame("test", index=index), "traversable", "traversable")
        fixture.emulate_real_provenance_for_coverage_check()
        fixture.config["coverage"] = {
            "expected_frames": {"rellis": {"development": 1, "test": 4}},
            "expected_recordings": {"rellis": {
                "00000": 1, "00001": 1, "00002": 1, "00003": 1, "00004": 1,
            }},
        }
        report = fixture.evaluate()
        self.assertFalse(report["validation"]["errors"])
        rows = report["coverage"]["frames"]
        self.assertTrue(all(row["available"] >= row["expected"] for row in rows))
        self.assertEqual(report["coverage"]["status"], "public_subset_provisional")

    def test_zero_selected_regions_cannot_claim_complete_public_coverage(self):
        fixture = self.fixture
        fixture.region(fixture.frame("development"), "traversable", "traversable", selected=False)
        fixture.region(fixture.frame("test"), "traversable", "traversable", selected=False)
        fixture.emulate_real_provenance_for_coverage_check()
        fixture.config["coverage"] = {
            "expected_frames": {"rellis": {"development": 1, "test": 1}},
            "expected_recordings": {"rellis": {"00000": 1, "00001": 1}},
        }
        report = fixture.evaluate()
        self.assertFalse(report["validation"]["errors"])
        self.assertEqual(report["coverage"]["annotations"]["expected"], 0)
        self.assertEqual(report["coverage"]["status"], "public_subset_provisional")
        self.assertTrue(all(row["status"] == "no_eligible_regions" for row in report["comparison"]))

    def test_missing_semantic_aid_prevents_complete_reference_coverage(self):
        fixture = self.fixture
        fixture.region(fixture.frame("development"), "traversable", "traversable")
        fixture.region(fixture.frame("test"), None, "unknown")
        fixture.annotations[-1].update(annotation_source="dataset_policy",
                                       reference_exclusion_reason="missing_semantic_aid")
        fixture.emulate_real_provenance_for_coverage_check()
        fixture.config["coverage"] = {
            "expected_frames": {"rellis": {"development": 1, "test": 1}},
            "expected_recordings": {"rellis": {"00000": 1, "00001": 1}},
        }
        report = fixture.evaluate()
        self.assertFalse(report["validation"]["errors"])
        self.assertTrue(all(row["readable"] == row["expected"] for row in report["coverage"]["frames"]))
        self.assertEqual(report["coverage"]["annotations"]["pending"], 1)
        self.assertEqual(report["coverage"]["status"], "public_subset_provisional")

    def test_missing_image_is_counted_and_excludes_all_its_regions(self):
        fixture = self.fixture
        frame = fixture.frame()
        fixture.region(frame, "traversable", "traversable")
        fixture.region(frame, "non_traversable", "traversable")
        (fixture.data_root / frame["image_path"]).unlink()
        report = fixture.evaluate()
        coverage = next(row for row in report["coverage"]["frames"]
                        if row["source"] == "rellis" and row["split"] == "test")
        self.assertEqual(coverage["available"], 1)
        self.assertEqual(coverage["readable"], 0)
        self.assertEqual(self.comparison(report)["eligible_regions"], 0)
        self.assertEqual(sum(row["physical_defects"] for row in report["mask_breakdown"]
                             if row["source"] == "rellis" and row["split"] == "test"), 2)

    def test_multiple_configurations_of_one_model_join_identical_inputs(self):
        fixture = self.fixture
        fixture.region(fixture.frame(), "non_traversable", "traversable")
        fixture.save()
        second = Fixture(Path(self.temporary.name), "second")
        for name in ("frames", "regions", "annotations", "predictions"):
            setattr(second, name, copy.deepcopy(getattr(fixture, name)))
        second.predictions[0]["label"] = "unknown"
        second.metadata["configuration"]["prompt"] = "second prompt"
        second.save()
        config = copy.deepcopy(fixture.config)
        config["comparisons"] = [
            {"run_name": fixture.name, "model_key": MODEL, "configuration_label": "first",
             "expected_metadata": {"/configuration/prompt": "fixture"}},
            {"run_name": second.name, "model_key": MODEL, "configuration_label": "second",
             "expected_metadata": {"/configuration/prompt": "second prompt"}},
        ]
        report = evaluate(config)
        self.assert_rate(self.comparison(report, run_name=fixture.name)["metrics"]["unsafe_acceptance"], 1, 1)
        self.assert_rate(self.comparison(report, run_name=second.name)["metrics"]["unsafe_acceptance"], 0, 1)
        second.annotations[0]["reference_label"] = "traversable"
        second.save()
        report = evaluate(config)
        self.assertTrue(report["validation"]["errors"])
        self.assertEqual(self.comparison(report, run_name=fixture.name)["status"], "scored")
        self.assertEqual(self.comparison(report, run_name=second.name)["status"], "invalid")
        self.assertIsNone(self.comparison(report, run_name=second.name)["metrics"])

    def test_json_and_csv_export_preserve_counts_and_nulls(self):
        fixture = self.fixture
        fixture.region(fixture.frame(), "traversable", "unknown")
        report = fixture.evaluate()
        destination = fixture.run / "evaluation"
        exported = export_report(report, destination)
        self.assertIsInstance(exported, dict)
        json_paths = list(destination.glob("*.json"))
        csv_paths = list(destination.glob("*.csv"))
        self.assertTrue(json_paths)
        self.assertTrue(csv_paths)
        for path in json_paths:
            json.loads(path.read_text(encoding="utf-8"))
        text = "\n".join(path.read_text(encoding="utf-8") for path in json_paths)
        self.assertIn('"numerator"', text)
        self.assertIn('"denominator"', text)
        self.assertIn("null", text)

    def test_review_suggestions_are_tentative_and_unsupported_cases_unreviewed(self):
        fixture = self.fixture
        frame = fixture.frame()
        recognition = fixture.region(frame, "non_traversable", "traversable",
                                     semantic_class="water", prediction_class="concrete")
        segmentation = fixture.region(frame, "non_traversable", "traversable", quality="mixed")
        unsupported = fixture.region(frame, "non_traversable", "traversable")
        report = fixture.evaluate()
        cases = {case["region_id"]: case for case in report["error_cases"]}
        self.assertEqual(cases[recognition]["suggested_category"], "recognition")
        self.assertEqual(cases[segmentation]["suggested_category"], "segmentation")
        self.assertIn(cases[unsupported].get("suggested_category"), (None, "unreviewed"))
        self.assertTrue(all(case.get("confirmed_category") is None for case in cases.values()))
        html = render_review_gallery(report)
        self.assertIn("tentative suggestion", html)
        self.assertNotIn("Confirmed:", html)
        for group in ("Recognition", "Target Grounding", "Policy", "Segmentation", "Unreviewed"):
            self.assertIn(group, html)

    def test_policy_suggestion_requires_matching_canonical_class_and_reference_policy(self):
        fixture = self.fixture
        frame = fixture.frame()
        supported = fixture.region(frame, "traversable", "non_traversable",
                                   semantic_class="concrete", prediction_class="concrete")
        unsupported = fixture.region(frame, "unknown", "traversable",
                                     semantic_class="water", prediction_class="water")
        report = fixture.evaluate()
        cases = {case["region_id"]: case for case in report["error_cases"]}
        self.assertEqual(cases[supported]["suggested_category"], "policy")
        self.assertIn(cases[unsupported].get("suggested_category"), (None, "unreviewed"))
        self.assertIsNone(cases[supported]["confirmed_category"])

    def test_gallery_caps_review_at_twenty_and_prioritizes_unsafe_cases(self):
        fixture = self.fixture
        frame = fixture.frame()
        for _ in range(24):
            fixture.region(frame, "traversable", "non_traversable")
        unsafe_id = fixture.region(frame, "non_traversable", "traversable")
        fixture.predictions[-1]["raw_response"] = "<script>fixture</script>"
        report = fixture.evaluate()
        html = render_review_gallery(report, limit=100)
        self.assertEqual(html.count("<article "), 20)
        self.assertIn(unsafe_id, html)
        self.assertNotIn("<script>fixture</script>", html)
        self.assertIn("&lt;script&gt;fixture&lt;/script&gt;", html)
        self.assertEqual(render_review_gallery(report, limit=5).count("<article "), 5)

    def test_grounding_confirmation_requires_explicit_reviewer_evidence(self):
        fixture = self.fixture
        fixture.region(fixture.frame(), "non_traversable", "traversable")
        report = fixture.evaluate()
        case_id = report["error_cases"][0]["case_id"]
        review_path = Path(self.temporary.name) / "checked-review.jsonl"
        fixture.config["review_labels_path"] = str(review_path)
        write_jsonl(review_path, [{"case_id": case_id, "category": "target_grounding", "note": ""}])
        unchecked = fixture.evaluate()
        self.assertIsNone(unchecked["error_cases"][0]["confirmed_category"])
        self.assertTrue(unchecked["validation"]["warnings"])
        write_jsonl(review_path, [{
            "case_id": case_id, "category": "target_grounding",
            "note": "Checked fixture: outlined region is water; response identifies adjacent concrete.",
        }])
        checked = fixture.evaluate()
        self.assertEqual(checked["error_cases"][0]["confirmed_category"], "target_grounding")
        self.assertIn("Confirmed: target_grounding", render_review_gallery(checked))

    def test_exports_stay_under_evaluation_and_preserve_review_annotations(self):
        fixture = self.fixture
        fixture.region(fixture.frame(), "traversable", "unknown")
        report = fixture.evaluate()
        outside = Path(self.temporary.name) / "outside-evaluation"
        with self.assertRaises(ValueError):
            export_report(report, outside)
        self.assertFalse(outside.exists())
        destination = fixture.run / "evaluation"
        destination.mkdir()
        review = destination / "review_labels.jsonl"
        reviewed = '{"fixture":true,"note":"existing optional reviewer work"}\n'
        review.write_text(reviewed, encoding="utf-8")
        export_report(report, destination)
        self.assertEqual(review.read_text(encoding="utf-8"), reviewed)

    def test_notebook_is_thin_and_output_free(self):
        notebook = REPOSITORY / "notebooks" / "03_classification_evaluation.ipynb"
        document = json.loads(notebook.read_text(encoding="utf-8"))
        code = [cell for cell in document["cells"] if cell["cell_type"] == "code"]
        self.assertTrue(code)
        for cell in code:
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])
        source = "\n".join("".join(cell["source"]) for cell in code)
        self.assertIn("traversability_evaluation", source)
        self.assertNotIn("traversability_inference", source)
        self.assertNotIn("traversability_data", source)

    @unittest.skipUnless(importlib.util.find_spec("IPython"), "IPython is optional for notebook smoke execution")
    def test_notebook_cells_execute_and_export_with_fixture_inputs(self):
        fixture = self.fixture
        fixture.region(fixture.frame(), "non_traversable", "traversable", avoided_pixels=1)
        fixture.save()
        defaults = json.loads((REPOSITORY / "configs/evaluation/default.json").read_text(encoding="utf-8"))
        defaults.update(fixture.config)
        notebook = json.loads((REPOSITORY / "notebooks/03_classification_evaluation.ipynb").read_text(encoding="utf-8"))
        code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
        namespace = {"__name__": "evaluation_notebook_fixture"}
        previous_modules = set(sys.modules)
        with redirect_stdout(io.StringIO()), mock.patch("IPython.display.display") as displayed:
            exec(compile(code[0], "classification-notebook-setup", "exec"), namespace)
            namespace.update(config=defaults, EXPORT=True, PLOTS=False)
            for index, source in enumerate(code[1:], 1):
                exec(compile(source, f"classification-notebook-cell-{index}", "exec"), namespace)
        self.assertGreater(displayed.call_count, 3)
        self.assertFalse(namespace["report"]["validation"]["errors"])
        self.assertEqual(self.comparison(namespace["report"])["status"], "scored")
        hazard = next(row for row in namespace["report"]["hazard_overlap"]
                      if row["split"] == "test" and row["source"] == "all")
        self.assert_rate(hazard["metric"], 1, 1)
        self.assertTrue(namespace["exported_paths"])
        for path in namespace["exported_paths"].values():
            self.assertTrue(Path(path).is_relative_to(fixture.run / "evaluation"))
            self.assertTrue(Path(path).is_file())
        for module in set(sys.modules) - previous_modules:
            self.assertFalse(module.startswith(("traversability_inference", "traversability_data")), module)

    def test_cli_exports_and_preserves_coverage_on_missing_primary_prediction(self):
        fixture = self.fixture
        fixture.region(fixture.frame(), "non_traversable", "traversable", avoided_pixels=1)
        fixture.save()
        defaults = json.loads((REPOSITORY / "configs/evaluation/default.json").read_text(encoding="utf-8"))
        defaults.update(fixture.config)
        config_path = Path(self.temporary.name) / "cli-fixture-config.json"
        config_path.write_text(json.dumps(defaults), encoding="utf-8")
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(cli_main(["--config", str(config_path), "--export"]), 0)
        self.assertEqual(json.loads(stdout.getvalue())["coverage"]["annotations"]["expected"], 1)
        self.assertTrue((fixture.run / "evaluation" / "report.json").is_file())
        self.assertTrue((fixture.run / "evaluation" / "comparison.csv").is_file())
        self.assertTrue((fixture.run / "evaluation" / "review_gallery.html").is_file())
        fixture.predictions.clear()
        fixture.save()
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(cli_main(["--config", str(config_path), "--export"]), 2)
        output = json.loads(stdout.getvalue())
        self.assertEqual(output["coverage"]["annotations"]["expected"], 1)
        self.assertEqual(output["coverage"]["annotations"]["available"], 1)
        self.assertEqual(self.comparison(output)["status"], "incomplete")
        self.assertIsNone(self.comparison(output)["metrics"])


if __name__ == "__main__":
    unittest.main()
