"""Contract/failure tests use independently marked CPU inputs and fake models."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from traversability_hazard_segmentation import run_conditions, freeze_selection
from traversability_hazard_segmentation import controller
from traversability_hazard_segmentation.common import (atomic_json, atomic_jsonl, binary_mask,
    fingerprint, hash_file, read_inputs, read_json, read_jsonl, safe_path, validate_frame)
from traversability_hazard_segmentation.fixtures import prepare_fixture, FakeSegmenter
from traversability_hazard_segmentation.selection import freeze_with_roots, validate_selection


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = prepare_fixture(self.root / "run", self.root / "data")
        self.run = self.root / "run"
        self.data = self.root / "data"
        self.selection = read_json(self.run / "segmentation/selection.json")
        self.sam = FakeSegmenter()

    def tearDown(self):
        self.temp.cleanup()

    def run_one(self, condition="vlm__qwen3_vl_4b"):
        return run_conditions({**self.config, "conditions": [condition]}, segmenter=self.sam)

    def test_source_counts_warmup_hash_and_order_independence(self):
        self.assertEqual(len(self.selection["test_frame_ids"]), 20)
        self.assertEqual(len(self.selection["warmup_frame_ids"]), 5)
        self.assertEqual(self.selection["coverage"]["selected_test_per_source"], {"rellis": 10, "coco": 10})
        for name in ("frames.jsonl", "references.jsonl"):
            rows = read_jsonl(self.run / name)
            atomic_jsonl(self.run / name, list(reversed(rows)))
        again = freeze_with_roots(self.run, self.data, fixture=True)
        self.assertEqual(again, self.selection)
        with self.assertRaisesRegex(ValueError, "changed"):
            freeze_with_roots(self.run, self.data, fixture=True, seed=1)

    def test_public_selection_resolves_environment_root(self):
        with patch.dict("os.environ", {"TRAVERSABILITY_DATA_ROOT": str(self.data)}):
            self.assertEqual(freeze_selection(self.run), self.selection)

    def test_public_selection_explicit_roots_override_environment(self):
        with patch.dict("os.environ", {"TRAVERSABILITY_DATA_ROOT": str(self.root / "wrong-data"),
                                       "TRAVERSABILITY_POLICY_PATH": str(self.root / "wrong-policy.json")}):
            self.assertEqual(freeze_selection(self.run, data_root=self.data,
                                             policy_path=self.config["policy_path"]), self.selection)

    def test_insufficient_inputs_incomplete_without_model_calls(self):
        (self.run / "segmentation/selection.json").unlink()
        for path in (self.run / "predictions").glob("*.jsonl"):
            path.unlink()
        for path in (self.run / "metadata").glob("*.json"):
            if path.name != "dataset.json":
                path.unlink()
        frames = read_jsonl(self.run / "frames.jsonl")
        removed = next(row["frame_id"] for row in frames if row["source"] == "coco" and row["split"] == "test")
        frames = [row for row in frames if row["frame_id"] != removed]
        refs = [row for row in read_jsonl(self.run / "references.jsonl") if row["frame_id"] != removed]
        atomic_jsonl(self.run / "frames.jsonl", frames)
        atomic_jsonl(self.run / "references.jsonl", refs)
        dataset = read_json(self.run / "metadata/dataset.json")
        dataset["selected_frame_ids"] = [row["frame_id"] for row in frames]
        atomic_json(self.run / "metadata/dataset.json", dataset)
        result = self.run_one()
        self.assertEqual(result["status"], "incomplete_selection")
        self.assertFalse(result["complete"])
        self.assertEqual(self.sam.calls, [])

    def test_new_selection_cannot_be_created_after_predictions(self):
        (self.run / "segmentation/selection.json").unlink()
        with self.assertRaisesRegex(ValueError, "Freeze selection before predictions"):
            freeze_with_roots(self.run, self.data, fixture=True)
        self.assertFalse((self.run / "segmentation/selection.json").exists())

    def test_failed_raw_unicode_is_preserved_through_sam_handoff_and_resume(self):
        path = self.run / "predictions/qwen3_vl_4b.jsonl"
        rows = read_jsonl(path)
        fid = self.selection["test_frame_ids"][0]
        prediction = next(row for row in rows if row["frame_id"] == fid)
        prediction.update(prompts=[], status="error", error_code="invalid_response_unicode",
                          raw_response="\ud800")
        atomic_jsonl(path, rows)
        result = self.run_one()
        self.assertTrue(result["complete"])
        self.assertEqual(self.run_one(), result)
        transaction = next(tx for tx in (self.run / "segmentation/vlm__qwen3_vl_4b/.transactions").glob("*.json")
                           if read_json(tx)["frame"]["frame_id"] == fid)
        self.assertEqual(read_json(transaction)["upstream"]["prediction"]["raw_response"], "\ud800")
        self.assertIn(b"\\ud800", transaction.read_bytes())

    def test_native_inference_audit_blocks_changed_predictions_before_sam(self):
        prediction_path = self.run / "predictions/qwen3_vl_4b.jsonl"
        metadata_path = self.run / "metadata/qwen3_vl_4b.json"
        rows = read_jsonl(prediction_path)
        metadata = read_json(metadata_path)
        metadata["prediction_digests"] = {}
        metadata["attempts"] = {}
        for row in rows:
            digest = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":"),
                                              ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()
            metadata["prediction_digests"][row["frame_id"]] = digest
            metadata["attempts"][row["frame_id"]] = [{"record": row, "record_sha256": digest,
                                                     "status": row["status"], "error_code": row["error_code"]}]
        atomic_json(metadata_path, metadata)
        altered = copy.deepcopy(rows)
        altered[0]["raw_response"] += "changed"
        atomic_jsonl(prediction_path, altered)
        with self.assertRaisesRegex(ValueError, "prediction digest"):
            self.run_one()
        self.assertEqual(self.sam.calls, [])
        atomic_jsonl(prediction_path, rows)
        metadata["attempts"][rows[0]["frame_id"]][-1]["status"] = "corrupt"
        atomic_json(metadata_path, metadata)
        with self.assertRaisesRegex(ValueError, "latest inference attempt"):
            self.run_one()
        self.assertEqual(self.sam.calls, [])

    def test_duplicate_joins_and_metadata_identity_fail(self):
        rows = read_jsonl(self.run / "references.jsonl")
        atomic_jsonl(self.run / "references.jsonl", rows + [rows[0]])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            read_inputs(self.run, self.data)
        atomic_jsonl(self.run / "references.jsonl", rows[1:])
        with self.assertRaisesRegex(ValueError, "join"):
            read_inputs(self.run, self.data)

    def test_portable_paths_dimensions_and_hashes(self):
        frame = read_jsonl(self.run / "frames.jsonl")[0]
        for relative in ("../escape.png", "C:/escape.png", "a\\b.png", "/absolute.png", "a//b", "a/./b", "CON.png", "trailing./mask.png", "bad?/mask.png"):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                safe_path(self.data, relative)
        altered = {**frame, "width": frame["width"] + 1}
        with self.assertRaisesRegex(ValueError, "dimensions"):
            validate_frame(altered, self.data)
        with self.assertRaisesRegex(ValueError, "hash"):
            validate_frame({**frame, "image_sha256": "0" * 64}, self.data)

    def test_changed_reference_asset_and_policy_identity(self):
        from PIL import Image
        ref = read_jsonl(self.run / "references.jsonl")[0]
        Image.new("L", (9, 5), 0).save(self.data / ref["valid_mask_path"])
        with self.assertRaisesRegex(ValueError, "changed|evaluable pixels"):
            validate_selection(self.run, self.data)
        dataset = read_json(self.run / "metadata/dataset.json")
        dataset["alias_sha256"] = "0" * 64
        atomic_json(self.run / "metadata/dataset.json", dataset)
        with self.assertRaisesRegex(ValueError, "alias_sha256"):
            read_inputs(self.run, self.data)

    def test_rellis_annotation_coverage_and_ignored_positive_masks_are_rejected(self):
        from PIL import Image
        rows = read_jsonl(self.run / "references.jsonl")
        row = next(item for item in rows if ":rellis:" in item["frame_id"] and item["present_concepts"])
        Image.new("L", (9, 5), 0).save(self.data / row["valid_mask_path"])
        with self.assertRaisesRegex(ValueError, "absence eligibility"):
            read_inputs(self.run, self.data)
        row["absence_scoring_eligible"] = False
        atomic_jsonl(self.run / "references.jsonl", rows)
        with self.assertRaisesRegex(ValueError, "within evaluable pixels"):
            read_inputs(self.run, self.data)

    def test_exact_five_percent_ignored_pixels_remains_absence_eligible(self):
        from PIL import Image
        frames = read_jsonl(self.run / "frames.jsonl")
        frame = next(item for item in frames if ":rellis:" in item["frame_id"] and item["frame_id"].endswith(":000"))
        references = read_jsonl(self.run / "references.jsonl")
        reference = next(item for item in references if item["frame_id"] == frame["frame_id"])
        rgb = self.data / frame["image_path"]
        Image.new("RGB", (20, 1), (10, 20, 30)).save(rgb)
        frame.update(width=20, height=1, image_sha256=hash_file(rgb))
        valid = Image.new("L", (20, 1), 255)
        valid.putpixel((0, 0), 0)
        valid.save(self.data / reference["valid_mask_path"])
        atomic_jsonl(self.run / "frames.jsonl", frames)
        self.assertTrue(read_inputs(self.run, self.data)["references"][frame["frame_id"]]["absence_scoring_eligible"])

    def test_all_conditions_prompts_and_exact_artifacts(self):
        result = run_conditions(self.config, segmenter=self.sam)
        self.assertTrue(result["complete"])
        self.assertFalse(result["comparison_ready"])
        refs = {row["frame_id"]: row for row in read_jsonl(self.run / "references.jsonl")}
        policy = read_json(self.config["policy_path"])
        for condition in result["conditions"]:
            frames = read_jsonl(self.run / f"segmentation/{condition}/frames.jsonl")
            queries = read_jsonl(self.run / f"segmentation/{condition}/queries.jsonl")
            self.assertEqual(len(frames), 20)
            for row in frames:
                self.assertEqual(set(row), controller.FRAME_KEYS)
                source = next(frame for frame in read_jsonl(self.run / "frames.jsonl") if frame["frame_id"] == row["frame_id"])
                self.assertEqual(binary_mask(self.run, row["union_mask_path"], source).shape, (5, 9))
                phrases = [query["phrase"] for query in queries if query["frame_id"] == row["frame_id"]]
                if condition == "fixed_policy":
                    self.assertEqual(phrases, policy["canonical_prompts"])
                if condition == "reference_present":
                    self.assertEqual(phrases, refs[row["frame_id"]]["present_concepts"])
            for row in queries:
                self.assertEqual(set(row), controller.QUERY_KEYS)
                self.assertEqual(row["returned_instance_count"], len(row["mask_paths"]))
        self.assertFalse(self.sam.closed)

    def test_original_aliases_unknown_vague_empty_and_upstream_error(self):
        self.run_one()
        base = self.run / "segmentation/vlm__qwen3_vl_4b"
        rows = read_jsonl(base / "queries.jsonl")
        self.assertEqual({row["phrase"] for row in rows if row["canonical_concept"] == "water"}, {"water", "puddle"})
        self.assertTrue(all(row["canonical_concept"] is None for row in rows if row["phrase"] == "traffic cone"))
        self.assertTrue(all(row["status"] == "error" for row in rows if row["phrase"] == "obstacle"))
        frames = read_jsonl(base / "frames.jsonl")
        self.assertTrue(all(row["status"] == "ok" and row["requested_queries"] == 0 for row in frames if row["frame_id"].endswith(":000")))
        self.assertTrue(all(row["status"] == "error" and row["upstream_status"] == "error" for row in frames if row["frame_id"].endswith(":001")))

    def test_partial_query_outputs_preserved_as_errors(self):
        self.run_one("vlm__qwen3_5_4b")
        rows = read_jsonl(self.run / "segmentation/vlm__qwen3_5_4b/queries.jsonl")
        bottle = [row for row in rows if row["phrase"] == "bottle"]
        self.assertTrue(bottle)
        self.assertTrue(all(row["status"] == "error" and len(row["mask_paths"]) == 1 for row in bottle))

    def test_compatible_terminal_errors_resume_without_calls(self):
        first = self.run_one()
        count = len(self.sam.calls)
        frames = (self.run / "segmentation/vlm__qwen3_vl_4b/frames.jsonl").read_bytes()
        self.assertEqual(self.run_one(), first)
        self.assertEqual(len(self.sam.calls), count)
        self.assertEqual((self.run / "segmentation/vlm__qwen3_vl_4b/frames.jsonl").read_bytes(), frames)
        self.sam.settings["resolution"] = 999
        with self.assertRaisesRegex(ValueError, "identity changed"):
            self.run_one()

    def test_separately_executed_conditions_share_sam_settings_and_actual_identity(self):
        self.run_one("reference_present")
        changed = FakeSegmenter()
        changed.settings["confidence_threshold"] = 0.9
        with self.assertRaisesRegex(ValueError, "identity changed across conditions"):
            run_conditions({**self.config, "conditions": ["fixed_policy"]}, segmenter=changed)
        self.assertEqual(changed.calls, [])
        with self.assertRaisesRegex(ValueError, "settings differ across conditions"):
            run_conditions({**self.config, "conditions": ["fixed_policy"],
                            "sam_settings": {**self.config["sam_settings"],
                                             "mask_probability_threshold": 0.9}},
                           segmenter=self.sam)

    def test_missing_prediction_remains_incomplete_and_can_arrive(self):
        path = self.run / "predictions/qwen3_vl_4b.jsonl"
        rows = read_jsonl(path)
        fid = self.selection["test_frame_ids"][0]
        atomic_jsonl(path, [row for row in rows if row["frame_id"] != fid])
        first = self.run_one()
        self.assertFalse(first["complete"])
        atomic_jsonl(path, rows)
        self.assertTrue(self.run_one()["complete"])

    def test_changed_upstream_and_missing_completed_mask_rejected(self):
        self.run_one()
        path = self.run / "predictions/qwen3_vl_4b.jsonl"
        rows = read_jsonl(path)
        fid = self.selection["test_frame_ids"][0]
        next(row for row in rows if row["frame_id"] == fid)["raw_response"] += "changed"
        atomic_jsonl(path, rows)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            self.run_one()

    def test_missing_mask_is_corruption(self):
        self.run_one("fixed_policy")
        frames = read_jsonl(self.run / "segmentation/fixed_policy/frames.jsonl")
        (self.run / frames[0]["union_mask_path"]).unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_one("fixed_policy")

    def test_atomic_transaction_recovers_without_repeat(self):
        original = controller.atomic_jsonl
        fired = [False]
        def crash(path, rows):
            if Path(path).name == "frames.jsonl" and not fired[0]:
                fired[0] = True
                raise OSError("simulated crash after query publication")
            return original(path, rows)
        with patch.object(controller, "atomic_jsonl", crash), self.assertRaises(OSError):
            self.run_one("reference_present")
        count = len(self.sam.calls)
        self.assertEqual(count, 1)
        self.assertTrue(self.run_one("reference_present")["complete"])
        self.assertEqual(len(self.sam.calls), 20)

    def test_injection_requires_fixture_even_for_completed_run(self):
        with self.assertRaisesRegex(ValueError, "fixture"):
            run_conditions({**self.config, "fixture": False}, segmenter=self.sam)
        fake = FakeSegmenter()
        fake.metadata["fixture"] = False
        fake.settings["fixture"] = False
        with self.assertRaisesRegex(ValueError, "provenance"):
            run_conditions(self.config, segmenter=fake)

    def test_mutation_during_call_detected(self):
        original = self.sam.segment_image
        def mutate(frame, prompts, root):
            result = original(frame, prompts, root)
            self.sam.settings["resolution"] = 999
            return result
        self.sam.segment_image = mutate
        with self.assertRaisesRegex(ValueError, "mutated"):
            self.run_one()

    def test_owned_close_and_unavailable_are_explicit(self):
        with patch("traversability_hazard_segmentation.load_segmenter", return_value=self.sam):
            result = run_conditions({**self.config, "conditions": ["reference_present"]})
        self.assertTrue(result["complete"])
        self.assertTrue(self.sam.closed)
        with patch("traversability_hazard_segmentation.load_segmenter", side_effect=RuntimeError("missing dependency")):
            result = run_conditions({**self.config, "conditions": ["fixed_policy"]})
        self.assertFalse(result["complete"])
        self.assertEqual(result["conditions"]["fixed_policy"]["completed_frames"], 0)

    def test_config_identity_flags_and_fixture_mismatch(self):
        for override in ({"task_id": "legacy"}, {"schema_version": True}, {"enable_real_run": "false"}, {"sam_settings": {"fixture": False}}, {"selection": {"test_frames_per_source": 1}}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                run_conditions({**self.config, **override}, segmenter=self.sam)

    def test_cleanup_errors_are_saved(self):
        self.sam.close = lambda: (_ for _ in ()).throw(RuntimeError("close failure"))
        with patch("traversability_hazard_segmentation.load_segmenter", return_value=self.sam):
            result = run_conditions({**self.config, "conditions": ["reference_present"]})
        self.assertEqual(result["cleanup_error"]["error_code"], "sam_close_error")
        self.assertIn("cleanup_error", read_json(self.run / "segmentation/reference_present/metadata.json"))

    def test_stale_vlm_frame_identity_is_rejected(self):
        path = self.run / "metadata/qwen3_vl_4b.json"
        meta = read_json(path)
        frames = sorted(read_jsonl(self.run / "frames.jsonl"), key=lambda row: row["frame_id"])
        frames[0]["image_sha256"] = "0" * 64
        meta["inputs"] = {"dataset": read_json(self.run / "metadata/dataset.json"), "frames": frames}
        atomic_json(path, meta)
        with self.assertRaisesRegex(ValueError, "frame/image identity"):
            self.run_one()

    def test_resealed_legacy_transaction_cannot_resume(self):
        self.run_one("reference_present")
        path = next((self.run / "segmentation/reference_present/.transactions").glob("*.json"))
        tx = read_json(path)
        tx["frame"]["task_id"] = "legacy_region"
        tx["fingerprint"] = fingerprint({key: value for key, value in tx.items() if key != "fingerprint"})
        atomic_json(path, tx)
        with self.assertRaisesRegex(ValueError, "task/schema"):
            self.run_one("reference_present")

    def test_unavailable_load_can_be_resolved_without_stale_current_status(self):
        config = {**self.config, "conditions": ["reference_present"]}
        with patch("traversability_hazard_segmentation.load_segmenter", side_effect=RuntimeError("checkpoint unavailable")):
            failed = run_conditions(config)
        self.assertFalse(failed["complete"])
        ready = run_conditions(config, segmenter=self.sam)
        self.assertTrue(ready["complete"])
        self.assertEqual(ready["conditions"]["reference_present"]["availability"]["status"], "available")
        self.assertTrue(read_json(self.run / "segmentation/reference_present/metadata.json")["load_attempts"])


if __name__ == "__main__":
    unittest.main()
