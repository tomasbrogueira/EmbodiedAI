"""Independent CPU checks for the integrated producers' exact metadata layouts."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(DIRECTORY.parents[1] / "src"))
sys.path.insert(0, str(DIRECTORY))

from fixture_inputs import COMPONENT, HazardFixture, MODEL, canonical_hash
from traversability_hazard_evaluation import evaluate, export_report
from traversability_hazard_evaluation.artifacts import fingerprint


class NativeContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = HazardFixture(Path(self.temporary.name))

    def native_layout(self):
        fixture = self.fixture
        fixture.save()
        signature = evaluate(fixture.config)["identity"]["observed_input_signature"]
        data = fixture.dataset_metadata
        data["policy_sha256"] = hashlib.sha256((COMPONENT / "configs/hazards/policy.json").read_bytes()).hexdigest()
        data["alias_sha256"] = data.pop("alias_hash")
        data.pop("policy_hash")
        data.pop("source_annotations")
        data["coverage"] = {"expected_frames_by_source_split": fixture.config["coverage"]["expected_frames"]}
        data["annotations"] = {frame["frame_id"]: {"kind": "marked_synthetic_annotation",
            "path": ref["valid_mask_path"], "sha256": hashlib.sha256((fixture.data_root / ref["valid_mask_path"]).read_bytes()).hexdigest()}
            for frame, ref in zip(fixture.frames, fixture.references)}
        paths = {frame["image_path"] for frame in fixture.frames}
        for ref in fixture.references:
            paths.add(ref["valid_mask_path"])
            paths.update(ref["concept_masks"].values())
        data["assets"] = {relative: hashlib.sha256((fixture.data_root / relative).read_bytes()).hexdigest() for relative in sorted(paths)}
        data["observed_input_signature"] = signature
        data["dataset_fingerprint"] = canonical_hash({"metadata": {key: value for key, value in data.items() if key != "dataset_fingerprint"},
            "frames": sorted(fixture.frames, key=lambda row: row["frame_id"]),
            "references": sorted(fixture.references, key=lambda row: row["frame_id"])})
        fixture.model_metadata = {"task_id": "hazard_prompt_v1", "schema_version": 1, "model_key": MODEL,
            "configuration": {"policy_sha256": data["policy_sha256"], "alias_sha256": data["alias_sha256"],
                              "model": {"revision": "fixture"}, "settings": {"device": "cpu"}},
            "execution": {"kind": "injected_fixture"}, "inputs": {"dataset": copy.deepcopy(data), "frames": copy.deepcopy(fixture.frames),
                "rgb_contents": {frame["image_path"]: {"status": "present", "sha256": frame["image_sha256"]} for frame in fixture.frames}}}
        active = json.loads((COMPONENT / "configs/hazards/evaluation.json").read_text("utf-8"))
        active.update({key: fixture.config[key] for key in ("component_root", "data_root", "run_root", "run_name", "model_keys", "fixture", "coverage")})
        active["segmentation"]["enabled"] = False
        fixture.config = active
        fixture.save()
        return data

    def test_native_data_inference_pointers_and_prepared_signature_are_verified(self):
        self.fixture.discovery_case()
        self.native_layout()
        report = evaluate(self.fixture.config)
        self.assertEqual(report["validation"]["errors"], [])
        self.assertTrue(report["identity"]["input_identity_verified"])
        self.assertTrue(report["comparison"]["discovery_complete"])
        self.assertFalse(report["comparison"]["comparison_ready"])
        self.assertEqual(report["coverage"]["real_manifest_frames"], 0)

    def test_native_changed_annotation_and_same_count_reference_mask_are_rejected(self):
        for asset in ("annotation", "concept"):
            with self.subTest(asset=asset):
                self.fixture = HazardFixture(Path(self.temporary.name) / asset)
                self.fixture.discovery_case()
                data = self.native_layout()
                relative = (next(iter(data["annotations"].values()))["path"] if asset == "annotation"
                            else self.fixture.references[0]["concept_masks"]["water"])
                self.fixture.mask(relative, [(12, 12)])
                report = evaluate(self.fixture.config)
                self.assertTrue(report["validation"]["errors"])
                self.assertFalse(report["comparison"]["discovery_complete"])
                self.assertFalse(report["identity"]["input_identity_verified"])

    def test_missing_native_frozen_asset_is_validation_failure_not_reader_crash(self):
        self.fixture.discovery_case()
        self.native_layout()
        relative = self.fixture.references[0]["concept_masks"]["water"]
        (self.fixture.data_root / relative).unlink()
        report = evaluate(self.fixture.config)
        self.assertTrue(report["validation"]["errors"])
        self.assertFalse(report["comparison"]["discovery_complete"])
        self.assertFalse(report["identity"]["input_identity_verified"])

    def test_unicode_inference_transaction_digests_use_ascii_escape_recipe(self):
        frame = self.fixture.frame(concepts={"water": [(8, 8)]})
        row = self.fixture.predict(frame, ["water", "café"])
        encoded = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        self.fixture.model_metadata.update(prediction_digests={frame["frame_id"]: digest},
            attempts={frame["frame_id"]: [{"record": copy.deepcopy(row), "record_sha256": digest, "diagnostics": {}}]})
        self.fixture.save()
        report = evaluate(self.fixture.config)
        self.assertEqual(report["validation"]["errors"], [])
        self.assertEqual(report["concepts"]["phrase_audit"][0]["unscored_concrete_phrases"][0]["raw_phrase"], "café")
        self.fixture.model_metadata["prediction_digests"][frame["frame_id"]] = canonical_hash(row)
        self.fixture.save()
        self.assertFalse(evaluate(self.fixture.config)["comparison"]["discovery_complete"])

    def test_failed_raw_unicode_is_preserved_as_escaped_json_in_report_export(self):
        frame = self.fixture.frame(concepts={"water": [(8, 8)]})
        row = self.fixture.predict(frame, [], status="error", error_code="invalid_json")
        self.fixture.save()
        row["raw_response"] = "malformed\ud800generation"
        # The inference writer uses escaped audit JSON for malformed Unicode.
        prediction_path = self.fixture.run / "predictions" / f"{MODEL}.jsonl"
        prediction_path.write_text(json.dumps(row, ensure_ascii=True) + "\n", encoding="utf-8")
        report = evaluate(self.fixture.config)
        self.assertEqual(report["validation"]["errors"], [])
        paths = export_report(report, self.fixture.run / "evaluation")
        saved = json.loads(Path(paths["hazard_report.json"]).read_text("utf-8"))
        self.assertEqual(saved, report)
        self.assertEqual(saved["concepts"]["phrase_audit"][0]["raw_response"], row["raw_response"])

    def test_native_profile_hashes_preserve_failed_raw_nonscalar_unicode(self):
        fixture = self.fixture
        fixture.add_segmentation_case()
        profile = fixture.add_profile("sam")
        fixture.save()
        profile["samples"][0].update(status="error", error_code="invalid_raw_text",
            raw_response="bad\ud800generation", call_error={"message": "bad\ud800generation"})
        for sample in profile["samples"]:
            encoded = json.dumps(sample, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                 allow_nan=False).encode("utf-8", errors="backslashreplace")
            sample["record_sha256"] = hashlib.sha256(encoded).hexdigest()
        profile["summary"]["failed_calls"] = 1
        profile["metadata"]["samples_fingerprint"] = fingerprint(profile["samples"])
        profile["metadata"]["summary_fingerprint"] = fingerprint(profile["summary"])
        directory = next(iter(fixture.profiles))
        for filename, value in (("metadata.json", profile["metadata"]), ("summary.json", profile["summary"])):
            (fixture.run / directory / filename).write_text(json.dumps(value, ensure_ascii=True), encoding="utf-8")
        (fixture.run / directory / "samples.jsonl").write_text(
            "".join(json.dumps(sample, ensure_ascii=True) + "\n" for sample in profile["samples"]), encoding="utf-8")
        report = evaluate(fixture.config)
        self.assertEqual(report["validation"]["errors"], [])
        self.assertTrue(report["deployment"]["rows"][0]["valid"])
        self.assertEqual(report["deployment"]["rows"][0]["failed_calls"], 1)
        self.assertFalse(report["deployment"]["rows"][0]["resource_measurements_available"])

    def test_terminal_eos_requires_exact_committed_parse_audit_and_preserves_duplicates(self):
        frame = self.fixture.frame(concepts={"water": [(8, 8)]})
        row = self.fixture.predict(frame, [" PUDDLE ", "water"])
        response = '{"prompts":[" PUDDLE ","puddle","water"]}'
        row["raw_response"] = response + "<|im_end|>"
        digest = canonical_hash(row)
        self.fixture.model_metadata.update(prediction_digests={frame["frame_id"]: digest}, attempts={frame["frame_id"]: [{
            "record": copy.deepcopy(row), "record_sha256": digest,
            "diagnostics": {"parse_response_text": response,
                "generation": {"terminal_eos": True, "stop_reason": "eos", "eos_token_id": 151645, "eos_text": "<|im_end|>"}}}]})
        self.fixture.save()
        report = evaluate(self.fixture.config)
        self.assertEqual(report["validation"]["errors"], [])
        self.assertEqual(report["concepts"]["phrase_audit"][0]["raw_phrases"], [" PUDDLE ", "puddle", "water"])
        audit = self.fixture.model_metadata["attempts"][frame["frame_id"]][-1]
        audit["diagnostics"]["generation"]["eos_text"] = "invented"
        self.fixture.save()
        self.assertFalse(evaluate(self.fixture.config)["comparison"]["discovery_complete"])

    def test_original_phrase_rewriting_is_rejected_even_when_alias_scores_agree(self):
        frame = self.fixture.frame(concepts={"water": [(8, 8)]})
        row = self.fixture.predict(frame, ["puddle"])
        row["raw_response"] = '{"prompts":[" Puddle "]}'
        self.fixture.save()
        report = evaluate(self.fixture.config)
        self.assertFalse(report["comparison"]["discovery_complete"])
        self.assertTrue(report["validation"]["errors"])

    def test_native_dual_selection_hash_binds_full_original_inputs(self):
        self.fixture.add_segmentation_case()
        data = self.native_layout()
        fixture = self.fixture
        original = fixture.selection
        original.update(policy_sha256=data["policy_sha256"], dataset_fingerprint=data["dataset_fingerprint"],
            identity={"task_id": "hazard_prompt_v1", "schema_version": 1, "fixture": True,
                      "policy_sha256": data["policy_sha256"], "policy_hash": canonical_hash(fixture.policy),
                      "alias_sha256": data["alias_sha256"], "dataset": copy.deepcopy(data),
                      "frames": copy.deepcopy(fixture.frames), "references": copy.deepcopy(fixture.references), "assets": copy.deepcopy(data["assets"])},
            selected_inputs={frame["frame_id"]: {"image_sha256": frame["image_sha256"], "reference_sha256": canonical_hash(ref)}
                             for frame, ref in zip(fixture.frames, fixture.references)})
        original.pop("selection_hash")
        original["fingerprint"] = canonical_hash(original)
        original["selection_hash"] = original["fingerprint"]
        condition = f"vlm__{MODEL}"
        metadata = fixture.segmentation[condition]["metadata"]
        metadata.update(policy_sha256=data["policy_sha256"], dataset_fingerprint=data["dataset_fingerprint"], selection_hash=original["fingerprint"])
        settings = metadata["sam_settings"]
        settings["checkpoint_repository"] = settings.pop("checkpoint")
        settings["checkpoint_revision"] = settings.pop("revision")
        fixture.config["segmentation"].update(enabled=True, conditions=[condition], expected_test_per_source={"rellis": 1}, expected_warmup_frames=0)
        fixture.save()
        report = evaluate(fixture.config)
        self.assertEqual(report["validation"]["errors"], [])
        self.assertTrue(report["segmentation"]["coverage"]["identity_verified"])
        original["identity"]["assets"][fixture.frames[0]["image_path"]] = "f" * 64
        sealed = {key: value for key, value in original.items() if key not in ("fingerprint", "selection_hash")}
        original["fingerprint"] = original["selection_hash"] = canonical_hash(sealed)
        fixture.save()
        self.assertFalse(evaluate(fixture.config)["segmentation"]["complete"])

    def test_resumed_profile_warmups_are_per_session_and_null_blocked_timings_remain_visible(self):
        fixture = self.fixture
        fixture.add_segmentation_case()
        dev = fixture.frame(split="development")
        fixture.predict(dev, [])
        fixture.selection["warmup_frame_ids"] = [dev["frame_id"]]
        fixture.config["segmentation"]["expected_warmup_frames"] = 1
        sealed = {key: value for key, value in fixture.selection.items() if key != "selection_hash"}
        fixture.selection["selection_hash"] = canonical_hash(sealed)
        fixture.segmentation[f"vlm__{MODEL}"]["metadata"]["selection_hash"] = fixture.selection["selection_hash"]
        profile = fixture.add_profile("sam")
        for index, sample in enumerate(profile["samples"]):
            sample["session_id"] = index
            sample.update(elapsed_s=None, timing_valid=False, call_attempted=False, status="error", error_code="upstream_error", sam_query_count=0)
            warmup = {**sample, "sample_id": f"warmup:{index}", "phase": "warmup", "frame_id": dev["frame_id"], "repeat": None,
                      "elapsed_s": 1.0, "status": "ok", "error_code": None, "timing_valid": True, "call_attempted": True}
            profile["samples"].append(warmup)
            if index == 1:
                break
        profile["metadata"]["sessions"] = [{"session_id": 0}, {"session_id": 1}]
        profile["summary"].update(failed_calls=2, sam_query_count=0, valid_measured_calls=0, invalid_timing_calls=0,
            unattempted_calls=2, unavailable_sam_query_counts=0, latency={"median_s": None, "p95_s": None})
        fixture.save()
        report = evaluate(fixture.config)
        self.assertEqual(report["validation"]["errors"], [])
        row = report["deployment"]["rows"][0]
        self.assertTrue(row["valid"])
        self.assertTrue(row["complete"])
        self.assertEqual(row["warmup_sessions"], 2)
        self.assertFalse(row["latency_measured"])
        self.assertFalse(row["resource_measurements_available"])


if __name__ == "__main__":
    unittest.main()
