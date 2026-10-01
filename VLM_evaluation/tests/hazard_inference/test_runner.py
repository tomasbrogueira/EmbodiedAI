"""Marked synthetic CPU fixtures: no model, image library, downloads or GPU."""

import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from traversability_hazard_inference.configuration import (configuration_snapshot,
                                                          normalize_settings)
from traversability_hazard_inference.records import InferenceError, prediction
from traversability_hazard_inference.runner import run_inference
from traversability_hazard_inference.storage import (atomic_json, atomic_jsonl,
                                                    read_json, read_jsonl)


MODEL = "qwen3_vl_4b"
# A tiny independent synthetic PNG; fixture backends operate only on these bytes.
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jJ1kAAAAASUVORK5CYII=")


class CPUFixtureBackend:
    def __init__(self, transform=None):
        self.model_key = MODEL
        self.metadata = {"fixture": True, "fixture_id": "independent_hazard_runner_cpu_v1",
                         "actual_dtype": "synthetic_cpu_bytes"}
        self.last_diagnostics = {}
        self.calls = []
        self.closed = 0
        self.transform = transform

    def predict_image(self, frame, data_root):
        self.calls.append(deepcopy(frame))
        assert (Path(data_root) / frame["image_path"]).read_bytes() == PNG
        self.last_diagnostics = {"fixture": True, "original_dimensions": [1, 1],
                                 "generation_complete": True, "vague_unusable": []}
        result = prediction(frame, MODEL, prompts=["Person", "puddle"],
                            raw_response='{"prompts":["Person","puddle"]}')
        return self.transform(frame, result) if self.transform else result

    def close(self):
        self.closed += 1


class RunnerCPUFixtures(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hazard-runner-cpu-fixture-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "external_data"
        self.run = self.root / "external_runs" / "hazard_prompt_v1"
        self.data.mkdir()
        self.run.mkdir(parents=True)
        self.frames = []
        for index, frame_id in enumerate(("fixture:b", "fixture:a")):
            relative = f"fixture_{index}.png"
            (self.data / relative).write_bytes(PNG)
            self.frames.append({"frame_id": frame_id, "source": "fixture",
                                "scene_id": "synthetic", "sequence_id": "fixture",
                                "timestamp_s": 7.25 if index == 0 else None,
                                "image_path": relative, "split": "test",
                                "width": 1, "height": 1,
                                "image_sha256": hashlib.sha256(PNG).hexdigest()})
        self.dataset = {"task_id": "hazard_prompt_v1", "schema_version": 1,
                        "fixture": True, "dataset_fingerprint": "synthetic_cpu_fixture_identity",
                        "source_annotation_provenance": {"fixture": "synthetic_no_annotations"}}
        self.write_inputs()
        # These deliberately invalid files must never be opened by inference.
        (self.run / "references.jsonl").write_text("POISON REFERENCE CONTENT", encoding="utf-8")

    def write_inputs(self):
        atomic_jsonl(self.run / "frames.jsonl", self.frames)
        atomic_json(self.run / "metadata/dataset.json", self.dataset)

    @property
    def output(self):
        return self.run / f"predictions/{MODEL}.jsonl"

    @property
    def metadata(self):
        return self.run / f"metadata/{MODEL}.json"

    def infer(self, backend=None, settings=None):
        options = {"fixture": True}
        options.update(settings or {})
        return run_inference(self.run, MODEL, options, self.data,
                             backend=backend if backend is not None else CPUFixtureBackend())

    def test_complete_ids_timestamps_original_input_projection_and_raw(self):
        self.frames[0]["image_specific_reference_names"] = ["POISON TARGET LABEL"]
        self.write_inputs()
        frame_bytes = (self.run / "frames.jsonl").read_bytes()
        backend = CPUFixtureBackend()
        original = backend.predict_image

        def observe(frame, data_root):
            self.assertTrue(self.metadata.exists())
            self.assertNotIn("image_specific_reference_names", frame)
            self.assertEqual(set(frame), {"frame_id", "source", "scene_id", "sequence_id",
                                         "timestamp_s", "image_path", "split", "width",
                                         "height", "image_sha256"})
            return original(frame, data_root)

        backend.predict_image = observe
        summary = self.infer(backend)
        self.assertTrue(summary["fixture"])
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["ok_frames"], 2)
        records = read_jsonl(self.output)
        self.assertEqual([record["frame_id"] for record in records], ["fixture:b", "fixture:a"])
        self.assertEqual(backend.calls[0]["timestamp_s"], 7.25)
        self.assertIsNone(backend.calls[1]["timestamp_s"])
        self.assertEqual((self.run / "frames.jsonl").read_bytes(), frame_bytes)
        self.assertEqual(records[0]["prompts"], ["Person", "puddle"])
        self.assertEqual(records[0]["raw_response"], '{"prompts":["Person","puddle"]}')
        self.assertEqual(backend.closed, 0)
        metadata = read_json(self.metadata)
        self.assertEqual(metadata["execution"]["kind"], "injected_fixture")
        self.assertEqual(len(metadata["attempts"]), 2)
        self.assertIsNone(metadata["pending_prediction"])
        self.assertEqual(summary["predictions_path"], f"predictions/{MODEL}.jsonl")

    def test_no_work_resume_never_loads_and_preserves_errors(self):
        failing = CPUFixtureBackend(lambda frame, record: prediction(
            frame, MODEL, raw_response="unfinished response", error_code="generation_truncated"))
        self.infer(failing)
        before = self.metadata.read_bytes()
        backend = CPUFixtureBackend()
        with patch("traversability_hazard_inference.load_backend") as loader:
            summary = self.infer(backend)
        loader.assert_not_called()
        self.assertEqual(backend.calls, [])
        self.assertEqual(summary["error_frames"], 2)
        self.assertEqual(self.metadata.read_bytes(), before)
        self.assertTrue(all(record["raw_response"] == "unfinished response" for record in read_jsonl(self.output)))

    def test_explicit_retry_changes_only_errors_and_retains_attempt_history(self):
        backend = CPUFixtureBackend(lambda frame, record: prediction(
            frame, MODEL, raw_response="bad", error_code="invalid_json")
            if frame["frame_id"] == "fixture:a" else record)
        self.infer(backend)
        first = read_jsonl(self.output)[0]
        retry = CPUFixtureBackend()
        self.infer(retry, {"retry_errors": True})
        self.assertEqual([frame["frame_id"] for frame in retry.calls], ["fixture:a"])
        self.assertEqual(read_jsonl(self.output)[0], first)
        attempts = read_json(self.metadata)["attempts"]
        self.assertEqual(len(attempts["fixture:b"]), 1)
        self.assertEqual(len(attempts["fixture:a"]), 2)
        self.assertEqual(attempts["fixture:a"][0]["error_code"], "invalid_json")
        self.assertEqual(attempts["fixture:a"][0]["record"]["raw_response"], "bad")

    def test_invalid_outputs_and_exceptions_publish_complete_errors(self):
        transforms = (lambda frame, row: None,
                      lambda frame, row: "unstructured raw text",
                      lambda frame, row: dict(row, frame_id="other"),
                      lambda frame, row: dict(row, prompts=["person", "PERSON"]),
                      lambda frame, row: dict(row, confidence=1),
                      lambda frame, row: dict(row, status="error", error_code=None),
                      lambda frame, row: dict(row, prompts=["\ud800"],
                                               raw_response='{"prompts":["\ud800"]}'))
        for index, transform in enumerate(transforms):
            with self.subTest(index=index):
                child = self.root / f"invalid_{index}"
                child.mkdir()
                shutil.copy2(self.run / "frames.jsonl", child / "frames.jsonl")
                (child / "metadata").mkdir()
                shutil.copy2(self.run / "metadata/dataset.json", child / "metadata/dataset.json")
                summary = run_inference(child, MODEL, {"fixture": True}, self.data,
                                        backend=CPUFixtureBackend(transform))
                self.assertEqual(summary["error_frames"], 2)
                self.assertTrue(summary["complete"])
                rows = read_jsonl(child / f"predictions/{MODEL}.jsonl")
                self.assertTrue(all(row["error_code"] == "backend_output_invalid_record" for row in rows))
                if index == 1:
                    self.assertEqual(rows[0]["raw_response"], "unstructured raw text")
                if index == 6:
                    self.assertEqual(rows[0]["raw_response"], '{"prompts":["\ud800"]}')

        backend = CPUFixtureBackend()

        def fail(*args):
            error = InferenceError("fixture_error", "Synthetic fixture failure")
            error.raw_response = "preserved raw error"
            raise error

        backend.predict_image = fail
        self.infer(backend)
        self.assertTrue(all(row["error_code"] == "fixture_error" for row in read_jsonl(self.output)))
        self.assertEqual(read_jsonl(self.output)[0]["raw_response"], "preserved raw error")
        self.assertIn("Synthetic fixture failure", read_json(self.metadata)["attempts"]["fixture:b"][0]["diagnostics"]["error_message"])

    def test_missing_rgb_is_an_error_and_later_presence_is_incompatible(self):
        image = self.data / self.frames[0]["image_path"]
        image.unlink()
        backend = CPUFixtureBackend()
        self.infer(backend)
        self.assertEqual([frame["frame_id"] for frame in backend.calls], ["fixture:a"])
        self.assertEqual(read_jsonl(self.output)[0]["error_code"], "image_missing")
        image.write_bytes(PNG)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            self.infer(settings={"retry_errors": True})

    def test_fixture_injection_requires_all_explicit_markers(self):
        for location in ("settings", "dataset", "backend", "fixture_id", "model_key"):
            with self.subTest(location=location):
                backend = CPUFixtureBackend()
                settings = {"fixture": True}
                saved_dataset = deepcopy(self.dataset)
                if location == "settings":
                    settings["fixture"] = False
                elif location == "dataset":
                    self.dataset["fixture"] = False
                    self.write_inputs()
                elif location == "backend":
                    backend.metadata["fixture"] = False
                elif location == "fixture_id":
                    del backend.metadata["fixture_id"]
                else:
                    del backend.model_key
                with self.assertRaises(ValueError):
                    run_inference(self.run, MODEL, settings, self.data, backend=backend)
                self.assertFalse(self.metadata.exists())
                self.dataset = saved_dataset
                self.write_inputs()
        with self.assertRaisesRegex(ValueError, "explicitly supplied"):
            run_inference(self.run, MODEL, {"fixture": True}, self.data)

    def test_configuration_dataset_frames_and_actual_audit_reject_resume(self):
        self.infer()
        with self.assertRaisesRegex(ValueError, "incompatible"):
            self.infer(settings={"visual_token_budget": 512})
        for change in ("dataset", "frame", "actual", "fixture_identity"):
            with self.subTest(change=change):
                backend = CPUFixtureBackend()
                old_frames, old_dataset = deepcopy(self.frames), deepcopy(self.dataset)
                if change == "dataset":
                    self.dataset["dataset_fingerprint"] = "changed"
                    self.write_inputs()
                elif change == "frame":
                    self.frames[0]["timestamp_s"] = 8.5
                    self.write_inputs()
                elif change == "actual":
                    backend.metadata["actual_dtype"] = "changed"
                else:
                    backend.metadata["fixture_id"] = "changed"
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    self.infer(backend)
                self.frames, self.dataset = old_frames, old_dataset
                self.write_inputs()

    def test_changed_reference_contents_do_not_change_inference_identity(self):
        self.infer()
        before = self.metadata.read_bytes()
        (self.run / "references.jsonl").write_text("DIFFERENT INVALID REFERENCES", encoding="utf-8")
        backend = CPUFixtureBackend()
        self.infer(backend)
        self.assertEqual(backend.calls, [])
        self.assertEqual(self.metadata.read_bytes(), before)

    def test_declared_policy_identity_fields_match_actual_frozen_configuration(self):
        configuration = configuration_snapshot(MODEL, {"fixture": True})
        keys = ("policy_sha256", "policy_fingerprint", "alias_sha256")
        self.dataset.update({key: configuration[key] for key in keys})
        self.write_inputs()
        self.assertTrue(self.infer()["complete"])
        for key in keys:
            with self.subTest(key=key):
                self.dataset[key] = "mismatched_declared_identity"
                self.write_inputs()
                backend = CPUFixtureBackend()
                with self.assertRaisesRegex(ValueError, f"Dataset policy identity mismatch: {key}"):
                    self.infer(backend)
                self.assertEqual(backend.calls, [])
                self.dataset[key] = configuration[key]
        self.write_inputs()

    def test_paths_duplicate_ids_legacy_artifacts_and_hash_mismatch_reject(self):
        for relative in ("../outside.png", "/absolute.png", "C:/outside.png", "bad\\image.png", "image.png:stream"):
            with self.subTest(relative=relative):
                self.frames[0]["image_path"] = relative
                self.write_inputs()
                with self.assertRaises(ValueError):
                    self.infer()
                self.assertFalse(self.metadata.exists())
        self.frames[0]["image_path"] = "fixture_0.png"
        self.frames.append(deepcopy(self.frames[0]))
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "Duplicate frame_id"):
            self.infer()
        self.frames.pop()
        self.write_inputs()
        for name in ("regions.jsonl", "annotations.jsonl"):
            marker = self.run / name
            marker.write_text("contents never read", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Legacy"):
                self.infer()
            marker.unlink()
        (self.data / "fixture_0.png").write_bytes(PNG + b"changed")
        with self.assertRaisesRegex(ValueError, "RGB identity mismatch"):
            self.infer()

    def test_orphans_duplicates_unknown_rows_and_metadata_corruption_reject(self):
        self.output.parent.mkdir()
        self.output.write_text("", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Orphaned"):
            self.infer()
        self.output.unlink()
        self.infer()
        rows = read_jsonl(self.output)
        for bad in (rows + [rows[0]], [dict(rows[0], frame_id="unknown")],
                    [dict(rows[0], region_id="legacy")]):
            atomic_jsonl(self.output, bad)
            with self.assertRaises(ValueError):
                self.infer()
        atomic_jsonl(self.output, rows)
        metadata = read_json(self.metadata)
        metadata["attempts"]["fixture:b"][0]["status"] = "corrupt"
        atomic_json(self.metadata, metadata)
        with self.assertRaisesRegex(ValueError, "corrupt"):
            self.infer()

    def test_interruption_recovers_staged_prediction_without_repeat(self):
        writes = []

        def interrupted(path, rows):
            writes.append(deepcopy(rows))
            if len(writes) == 2:
                raise OSError("Synthetic CPU publication interruption")
            atomic_jsonl(path, rows)

        with patch("traversability_hazard_inference.runner.atomic_jsonl", side_effect=interrupted):
            with self.assertRaisesRegex(OSError, "interruption"):
                self.infer()
        self.assertEqual(len(read_jsonl(self.output)), 1)
        self.assertIsNotNone(read_json(self.metadata)["pending_prediction"])
        backend = CPUFixtureBackend()
        summary = self.infer(backend)
        self.assertTrue(summary["complete"])
        self.assertEqual(backend.calls, [])
        self.assertEqual(len(read_jsonl(self.output)), 2)
        self.assertIsNone(read_json(self.metadata)["pending_prediction"])

    def test_interrupted_retry_recovery_is_not_retried_twice(self):
        failing = CPUFixtureBackend(lambda frame, row: prediction(frame, MODEL, error_code="fixture_error"))
        self.infer(failing)
        with patch("traversability_hazard_inference.runner.atomic_jsonl", side_effect=OSError("Interrupted retry")):
            with self.assertRaises(OSError):
                self.infer(failing, {"retry_errors": True})
        retry = CPUFixtureBackend()
        self.infer(retry, {"retry_errors": True})
        self.assertEqual([frame["frame_id"] for frame in retry.calls], ["fixture:a"])
        rows = read_jsonl(self.output)
        self.assertEqual(rows[0]["status"], "error")
        self.assertEqual(rows[1]["status"], "ok")
        self.assertEqual(len(read_json(self.metadata)["attempts"]["fixture:b"]), 2)

    def test_interruption_after_prediction_write_recovers_without_repeat(self):
        original = atomic_json

        def interrupted(path, metadata):
            if metadata.get("prediction_digests") and metadata.get("pending_prediction") is None:
                raise OSError("Synthetic commit interruption")
            original(path, metadata)

        with patch("traversability_hazard_inference.runner.atomic_json", side_effect=interrupted):
            with self.assertRaises(OSError):
                self.infer()
        self.assertEqual(len(read_jsonl(self.output)), 1)
        backend = CPUFixtureBackend()
        self.infer(backend)
        self.assertEqual([frame["frame_id"] for frame in backend.calls], ["fixture:a"])

    def test_staged_resume_never_overwrites_unrelated_committed_rows(self):
        writes = []

        def interrupted(path, rows):
            writes.append(rows)
            if len(writes) == 2:
                raise OSError("Interrupted")
            atomic_jsonl(path, rows)

        with patch("traversability_hazard_inference.runner.atomic_jsonl", side_effect=interrupted):
            with self.assertRaises(OSError):
                self.infer()
        rows = read_jsonl(self.output)
        rows[0]["raw_response"] = "UNRELATED COMMITTED EDIT"
        atomic_jsonl(self.output, rows)
        with self.assertRaisesRegex(ValueError, "committed audit"):
            self.infer()
        self.assertEqual(read_jsonl(self.output)[0]["raw_response"], "UNRELATED COMMITTED EDIT")

    def test_mutated_inputs_or_actual_audit_are_not_published(self):
        for target in ("rgb", "frames", "dataset", "backend"):
            with self.subTest(target=target):
                child = self.root / f"mutation_{target}"
                shutil.copytree(self.run, child)
                child_data = self.root / f"mutation_data_{target}"
                shutil.copytree(self.data, child_data)
                backend = CPUFixtureBackend()
                original = backend.predict_image

                def mutate(frame, data_root):
                    row = original(frame, data_root)
                    if target == "rgb":
                        (Path(data_root) / frame["image_path"]).write_bytes(PNG + b"mutation")
                    elif target == "frames":
                        rows = read_jsonl(child / "frames.jsonl")
                        rows[0]["timestamp_s"] = 99.0
                        atomic_jsonl(child / "frames.jsonl", rows)
                    elif target == "dataset":
                        metadata = read_json(child / "metadata/dataset.json")
                        metadata["dataset_fingerprint"] = "mutated"
                        atomic_json(child / "metadata/dataset.json", metadata)
                    else:
                        backend.metadata["actual_dtype"] = "mutated"
                    return row

                backend.predict_image = mutate
                with self.assertRaises(RuntimeError):
                    run_inference(child, MODEL, {"fixture": True}, child_data, backend=backend)
                self.assertFalse((child / f"predictions/{MODEL}.jsonl").exists())
                self.assertEqual(backend.closed, 0)

    def test_backend_load_failure_completes_errors_and_no_work_does_not_reload(self):
        self.dataset["fixture"] = False
        self.write_inputs()
        with patch("traversability_hazard_inference.load_backend", side_effect=RuntimeError("offline missing weights")) as loader:
            summary = run_inference(self.run, MODEL, {}, self.data)
        loader.assert_called_once()
        self.assertEqual(summary["error_frames"], 2)
        self.assertTrue(summary["complete"])
        self.assertTrue(all(row["error_code"] == "backend_load_failed" for row in read_jsonl(self.output)))
        self.assertEqual(read_json(self.metadata)["actual_backend"]["load_status"], "error")
        with patch("traversability_hazard_inference.load_backend") as loader:
            run_inference(self.run, MODEL, {}, self.data)
        loader.assert_not_called()

    def official_cpu_stub(self, settings=None):
        """Exact-class API stub: never invokes the real Qwen constructor or model."""
        from traversability_hazard_inference.qwen import QwenBackend
        stub = object.__new__(QwenBackend)
        stub.model_key = MODEL
        stub.settings = normalize_settings(MODEL, settings or {})
        stub.metadata = {"fixture": False, "cpu_test_stub": True,
                         "configuration_snapshot": configuration_snapshot(MODEL, stub.settings)}
        fixture = CPUFixtureBackend()
        stub.predict_image = fixture.predict_image
        stub.close = fixture.close
        stub.last_diagnostics = {"fixture": True, "cpu_test_stub": True}
        return stub, fixture

    def test_official_class_cpu_stub_caller_and_internal_ownership(self):
        stub, fixture = self.official_cpu_stub()
        summary = run_inference(self.run, MODEL, {}, self.data, backend=stub)
        self.assertTrue(summary["fixture"])
        self.assertEqual(fixture.closed, 0)
        other_run = self.root / "owned_stub"
        shutil.copytree(self.run, other_run)
        (other_run / f"predictions/{MODEL}.jsonl").unlink()
        (other_run / f"metadata/{MODEL}.json").unlink()
        owned, owned_fixture = self.official_cpu_stub()
        with patch("traversability_hazard_inference.load_backend", return_value=owned):
            run_inference(other_run, MODEL, {}, self.data)
        self.assertEqual(owned_fixture.closed, 1)

    def test_official_subclass_cannot_bypass_fixture_guard(self):
        from traversability_hazard_inference.qwen import QwenBackend

        class UnmarkedSubclass(QwenBackend):
            pass

        stub = object.__new__(UnmarkedSubclass)
        stub.model_key = MODEL
        stub.metadata = {"fixture": False}
        with self.assertRaisesRegex(ValueError, "injected CPU backend"):
            run_inference(self.run, MODEL, {}, self.data, backend=stub)

    def test_explicit_retry_can_load_after_unavailable_backend(self):
        with patch("traversability_hazard_inference.load_backend", side_effect=RuntimeError("CPU fixture unavailable weights")):
            run_inference(self.run, MODEL, {}, self.data)
        old = read_jsonl(self.output)
        stub, fixture = self.official_cpu_stub({"retry_errors": True})
        with patch("traversability_hazard_inference.load_backend", return_value=stub):
            summary = run_inference(self.run, MODEL, {"retry_errors": True}, self.data)
        self.assertEqual(summary["ok_frames"], 2)
        self.assertEqual(fixture.closed, 1)
        metadata = read_json(self.metadata)
        self.assertTrue(any(audit.get("load_status") == "error" for audit in metadata["load_history"]))
        for frame_id, history in metadata["attempts"].items():
            self.assertEqual(len(history), 2)
            self.assertEqual(history[0]["record"], next(row for row in old if row["frame_id"] == frame_id))

    def test_changed_policy_during_call_is_not_published(self):
        policy = self.root / "fixture_policy.json"
        source = Path(normalize_settings(MODEL, {})["policy_path"])
        shutil.copy2(source, policy)
        backend = CPUFixtureBackend()
        original = backend.predict_image

        def mutate(frame, data_root):
            row = original(frame, data_root)
            value = read_json(policy)
            value["scope"] += " Synthetic mutation."
            atomic_json(policy, value)
            return row

        backend.predict_image = mutate
        with self.assertRaisesRegex(RuntimeError, "Frozen policy"):
            self.infer(backend, {"policy_path": str(policy)})
        self.assertFalse(self.output.exists())

    def test_rgb_and_artifact_symlink_escapes_reject_before_publication(self):
        outside = self.root / "outside.png"
        outside.write_bytes(PNG)
        image = self.data / "fixture_0.png"
        image.unlink()
        try:
            image.symlink_to(outside)
        except OSError:
            self.skipTest("This Windows account cannot create symlinks")
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.infer()
        image.unlink()
        image.write_bytes(PNG)
        outside_dir = self.root / "outside_predictions"
        outside_dir.mkdir()
        (self.run / "predictions").symlink_to(outside_dir, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.infer()
        self.assertEqual(list(outside_dir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
