"""CPU-only fixture tests; these are never measured experiment results."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from traversability_inference.records import InferenceError, prediction
from traversability_inference.runner import run_inference
from traversability_inference.storage import (atomic_json, atomic_jsonl,
                                               exclusive_lock, input_identity,
                                               read_json, read_jsonl,
                                               read_manifests,
                                               stable_fingerprint)


MODEL = "qwen3_vl_4b"


class FixtureBackend:
    fixture_identity = "synthetic_runner_fixture_v1"

    def __init__(self, transform=None):
        self.metadata = {"fixture": True, "device": "cpu", "actual_dtype": "float32"}
        self.calls = []
        self.closed = 0
        self.transform = transform

    def predict_frame(self, frame, regions, data_root):
        self.calls.append((deepcopy(frame), deepcopy(regions)))
        records = [prediction(region, MODEL, parsed={"label": "traversable",
                             "semantic_class": "concrete", "reason": "Dry concrete"},
                              raw_response='{"label":"traversable","semantic_class":"concrete","reason":"Dry concrete"}')
                   for region in regions]
        return self.transform(records) if self.transform else records

    def close(self):
        self.closed += 1


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "run"
        self.data = self.root / "data"
        self.run.mkdir()
        self.data.mkdir()
        self.frames = [{"frame_id": frame_id, "source": "synthetic_fixture",
                        "scene_id": "fixture_scene", "sequence_id": "fixture_seq",
                        "timestamp_s": timestamp, "image_path": f"{frame_id}.png",
                        "split": split}
                       for frame_id, timestamp, split in (("b", 7.25, "test"),
                                                         ("a", None, "development"))]
        self.regions = [{"region_id": region_id, "frame_id": frame_id,
                         "mask_path": f"{region_id}.png", "planning_relevant": True,
                         "selected_for_classification": selected}
                        for region_id, frame_id, selected in (("ra2", "a", False),
                                                               ("rb1", "b", True),
                                                               ("ra1", "a", True))]
        for frame in self.frames:
            Image.new("RGB", (20, 20), (100, 120, 150)).save(self.data / frame["image_path"])
        for region in self.regions:
            mask = Image.new("L", (20, 20), 0)
            mask.paste(255, (4, 4, 12, 14))
            mask.save(self.data / region["mask_path"])
        self.write_manifests()
        # Deliberately unreadable annotations prove the prediction runner has no
        # dependency on reference labels, completion, purity or reference masks.
        (self.run / "annotations.jsonl").write_text("this is not JSON", encoding="utf-8")

    def write_manifests(self):
        atomic_jsonl(self.run / "frames.jsonl", self.frames)
        atomic_jsonl(self.run / "regions.jsonl", self.regions)

    @property
    def output(self):
        return self.run / "predictions" / f"{MODEL}.jsonl"

    @property
    def metadata(self):
        return self.run / "metadata" / f"{MODEL}.json"

    def infer(self, backend=None, region_ids=None, settings=None):
        return run_inference(self.run, MODEL, settings or {}, self.data,
                             backend=backend or FixtureBackend(), region_ids=region_ids)

    def test_id_join_completeness_raw_and_timestamp_preservation(self):
        frozen_frames = (self.run / "frames.jsonl").read_bytes()
        backend = FixtureBackend()
        results = self.infer(backend)
        self.assertEqual([record["region_id"] for record in results], ["rb1", "ra1"])
        self.assertEqual([frame["frame_id"] for frame, _ in backend.calls], ["b", "a"])
        self.assertEqual(backend.calls[0][0]["timestamp_s"], 7.25)
        self.assertIsNone(backend.calls[1][0]["timestamp_s"])
        self.assertTrue(all(record["raw_response"].startswith('{"label"') for record in results))
        self.assertEqual((self.run / "frames.jsonl").read_bytes(), frozen_frames)
        self.assertEqual(read_jsonl(self.output), results)
        self.assertEqual(backend.closed, 0)
        self.assertEqual(read_json(self.metadata)["execution"]["kind"], "injected_fixture")

    def test_subset_resume_preserves_errors_and_previously_completed_subsets(self):
        failing = FixtureBackend(lambda records: [prediction(
            self.regions[2], MODEL, raw_response="bad response", error_code="parse_failed", reason="Bad JSON")])
        self.infer(failing, ["ra1"])
        metadata_before = self.metadata.read_bytes()
        backend = FixtureBackend()
        result = self.infer(backend, ["rb1"])
        self.assertEqual([row["region_id"] for row in result], ["rb1", "ra1"])
        self.assertEqual(result[1]["status"], "error")
        self.assertEqual(result[1]["raw_response"], "bad response")
        self.assertEqual([region["region_id"] for _, batch in backend.calls for region in batch], ["rb1"])
        self.assertEqual(self.metadata.read_bytes(), metadata_before)
        backend.calls.clear()
        result = self.infer(backend, ["ra2", "ra1"])
        self.assertEqual(len(result), 3)
        self.assertEqual([region["region_id"] for _, batch in backend.calls for region in batch], ["ra2"])

    def test_annotation_changes_do_not_change_identity(self):
        self.infer()
        metadata_before = self.metadata.read_bytes()
        (self.run / "annotations.jsonl").write_text('{"reference_label":"non_traversable"}', encoding="utf-8")
        (self.run / "metadata" / "reference_transfer.jsonl").write_text("bad reference bytes", encoding="utf-8")
        backend = FixtureBackend()
        self.infer(backend)
        self.assertEqual(backend.calls, [])
        self.assertEqual(self.metadata.read_bytes(), metadata_before)

    def test_invalid_ids_joins_and_paths_fail_before_publication(self):
        for selected in (["missing"], ["ra1", "ra1"], "ra1", [42]):
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                self.infer(region_ids=selected)
        self.assertFalse((self.run / "metadata").exists())
        self.regions[0]["frame_id"] = "missing"
        self.write_manifests()
        with self.assertRaises(ValueError):
            self.infer()
        self.assertFalse((self.run / "metadata").exists())
        self.regions[0]["frame_id"] = "a"
        for path in ("../outside.png", "/absolute.png", "C:/absolute.png", "bad\\mask.png", "mask.png:stream"):
            self.regions[0]["mask_path"] = path
            self.write_manifests()
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.infer()

    def test_duplicate_manifest_ids_are_rejected(self):
        self.regions.append(deepcopy(self.regions[0]))
        self.write_manifests()
        with self.assertRaisesRegex(ValueError, "Duplicate region_id"):
            self.infer()

    def test_malformed_backend_outputs_become_complete_error_predictions(self):
        transforms = {
            "backend_output_type": lambda rows: None,
            "backend_output_missing_id": lambda rows: [],
            "backend_output_duplicate_id": lambda rows: rows + rows,
            "backend_output_unexpected_id": lambda rows: [dict(rows[0], region_id="other")],
            "backend_output_invalid_record": lambda rows: [dict(rows[0], label="yes")],
        }
        for code, transform in transforms.items():
            with self.subTest(code=code):
                child_run = self.root / code
                child_run.mkdir()
                shutil.copy2(self.run / "frames.jsonl", child_run / "frames.jsonl")
                shutil.copy2(self.run / "regions.jsonl", child_run / "regions.jsonl")
                result = run_inference(child_run, MODEL, {}, self.data, backend=FixtureBackend(transform))
                self.assertEqual(len(result), 2)
                self.assertTrue(all(row["label"] == "unknown" and row["status"] == "error" for row in result))
                self.assertTrue(all(row["error_code"] == code for row in result))
                if code == "backend_output_invalid_record":
                    self.assertTrue(all(row["raw_response"] for row in result))

    def test_backend_exception_preserves_diagnostic_and_raw_response(self):
        backend = FixtureBackend()
        error = InferenceError("synthetic_failure", "Fixture failed")
        error.raw_response = "unfinished response"
        backend.predict_frame = lambda *args: (_ for _ in ()).throw(error)
        result = self.infer(backend)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(row["error_code"] == "synthetic_failure" for row in result))
        self.assertTrue(all(row["raw_response"] == "unfinished response" for row in result))

    def test_malformed_backend_text_preserves_raw_response(self):
        for raw_result in ("surrounding prose", ["unmapped generated text"]):
            child_run = self.root / ("text" if isinstance(raw_result, str) else "text_list")
            child_run.mkdir()
            shutil.copy2(self.run / "frames.jsonl", child_run / "frames.jsonl")
            shutil.copy2(self.run / "regions.jsonl", child_run / "regions.jsonl")
            result = run_inference(child_run, MODEL, {}, self.data,
                                   backend=FixtureBackend(lambda rows: raw_result))
            expected = raw_result if isinstance(raw_result, str) else raw_result[0]
            self.assertTrue(all(row["raw_response"] == expected for row in result))
            self.assertTrue(all(row["status"] == "error" for row in result))

    def test_invalid_exception_diagnostic_is_normalized(self):
        backend = FixtureBackend()
        backend.predict_frame = lambda *args: (_ for _ in ()).throw(InferenceError(None, "bad diagnostic"))
        self.assertTrue(all(row["error_code"] == "inference_failed" for row in self.infer(backend)))

    def test_actual_metadata_and_fixture_identity_changes_reject_resume(self):
        self.infer()
        backend = FixtureBackend()
        backend.metadata["actual_dtype"] = "float16"
        with self.assertRaisesRegex(ValueError, "Actual backend"):
            self.infer(backend)
        backend = FixtureBackend()
        backend.fixture_identity = "different_fixture"
        with self.assertRaisesRegex(ValueError, "incompatible"):
            self.infer(backend)
        with patch("traversability_inference.load_backend") as loader:
            with self.assertRaisesRegex(ValueError, "incompatible"):
                run_inference(self.run, MODEL, {}, self.data)
            loader.assert_not_called()

    def test_configuration_and_input_content_changes_reject_resume(self):
        self.infer()
        with self.assertRaises(ValueError):
            self.infer(settings={"scene_visual_token_budget": 128})
        image = self.data / self.frames[0]["image_path"]
        Image.new("RGB", (20, 20), (5, 5, 5)).save(image)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            self.infer()

    def test_input_roots_cache_roots_subsets_and_row_order_do_not_affect_identity(self):
        self.infer(region_ids=["ra1"], settings={"cache_dir": str(self.root / "cache_a")})
        copied_run = self.root / "copied_run"
        copied_data = self.root / "copied_data"
        shutil.copytree(self.run, copied_run)
        shutil.copytree(self.data, copied_data)
        atomic_jsonl(copied_run / "frames.jsonl", list(reversed(self.frames)))
        atomic_jsonl(copied_run / "regions.jsonl", list(reversed(self.regions)))
        backend = FixtureBackend()
        result = run_inference(copied_run, MODEL, {"cache_dir": str(self.root / "cache_b")},
                               copied_data, backend=backend, region_ids=["rb1"])
        self.assertEqual(len(result), 2)
        self.assertEqual([region["region_id"] for _, batch in backend.calls for region in batch], ["rb1"])
        self.assertEqual(read_json(copied_run / "metadata" / f"{MODEL}.json"), read_json(self.metadata))

    def test_missing_inputs_have_deterministic_fingerprint_sentinels(self):
        (self.data / self.regions[0]["mask_path"]).unlink()
        first = input_identity(self.frames, self.regions, self.data)
        self.assertEqual(first["contents"]["ra2.png"], {"status": "missing"})
        copied_data = self.root / "alternate_data"
        shutil.copytree(self.data, copied_data)
        second = input_identity(list(reversed(self.frames)), list(reversed(self.regions)), copied_data)
        self.assertEqual(stable_fingerprint(first), stable_fingerprint(second))

    def test_orphan_duplicates_corruption_and_metadata_tampering_are_rejected(self):
        self.output.parent.mkdir()
        self.output.write_text("", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Orphaned"):
            self.infer()
        self.output.unlink()
        records = self.infer()
        for corrupted in (json.dumps(records[0]) + "\n" + json.dumps(records[0]) + "\n",
                          json.dumps(records[0]) + "\n{broken",
                          json.dumps(records[0]) + "\n\n",
                          '{"region_id":"rb1","region_id":"ra1"}\n'):
            self.output.write_text(corrupted, encoding="utf-8")
            with self.subTest(corrupted=corrupted), self.assertRaises(ValueError):
                self.infer()
        atomic_jsonl(self.output, records)
        metadata = read_json(self.metadata)
        metadata["actual_backend"]["actual_dtype"] = "changed"
        atomic_json(self.metadata, metadata)
        with self.assertRaisesRegex(ValueError, "fingerprint is corrupt"):
            self.infer()

    def test_interrupted_second_frame_write_resumes_only_unpublished_frame(self):
        calls = []

        def interrupted(path, records):
            calls.append(list(records))
            if len(calls) == 2:
                raise OSError("Synthetic interrupted write")
            atomic_jsonl(path, records)

        with patch("traversability_inference.runner.atomic_jsonl", side_effect=interrupted):
            with self.assertRaises(OSError):
                self.infer()
        self.assertTrue(self.metadata.exists())
        self.assertEqual([row["region_id"] for row in read_jsonl(self.output)], ["rb1"])
        (self.output.parent / f".{MODEL}.jsonl.abandoned.tmp").write_text("broken", encoding="utf-8")
        backend = FixtureBackend()
        result = self.infer(backend)
        self.assertEqual(len(result), 2)
        self.assertEqual([frame["frame_id"] for frame, _ in backend.calls], ["a"])

    def test_metadata_is_published_before_prediction_and_owned_backend_closes(self):
        backend = FixtureBackend()
        original = backend.predict_frame

        def observe(*args):
            self.assertTrue(self.metadata.exists())
            return original(*args)

        backend.predict_frame = observe
        with patch("traversability_inference.load_backend", return_value=backend):
            run_inference(self.run, MODEL, {}, self.data)
        self.assertEqual(backend.closed, 1)
        self.assertEqual(read_json(self.metadata)["execution"]["kind"], "real_model")

    def test_explicit_official_backend_retains_real_identity_and_caller_ownership(self):
        from traversability_inference.configuration import normalize_settings
        from traversability_inference.qwen import QwenBackend

        official = object.__new__(QwenBackend)
        official.model_key = MODEL
        official.settings = normalize_settings(MODEL, {})
        fixture = FixtureBackend()
        official.metadata = fixture.metadata
        official.predict_frame = fixture.predict_frame
        official.close = fixture.close
        self.infer(official, ["ra1"])
        self.assertEqual(read_json(self.metadata)["execution"]["kind"], "real_model")
        self.assertEqual(fixture.closed, 0)
        with patch("traversability_inference.load_backend", return_value=official):
            result = run_inference(self.run, MODEL, {}, self.data, region_ids=["rb1"])
        self.assertEqual(len(result), 2)
        self.assertEqual(fixture.closed, 1)
        official.settings["scene_visual_token_budget"] = 128
        with self.assertRaisesRegex(ValueError, "Loaded backend settings"):
            self.infer(official)

    def test_inputs_changed_during_prediction_are_not_published(self):
        def mutate(records):
            Image.new("RGB", (20, 20), (5, 5, 5)).save(self.data / "b.png")
            return records

        with self.assertRaisesRegex(RuntimeError, "Frozen inference input changed"):
            self.infer(FixtureBackend(mutate))
        self.assertTrue(self.metadata.exists())
        self.assertFalse(self.output.exists())
        with self.assertRaisesRegex(ValueError, "incompatible"):
            self.infer()

    def test_actual_metadata_changed_during_prediction_is_not_published(self):
        backend = FixtureBackend()

        def mutate(records):
            backend.metadata["actual_dtype"] = "float16"
            return records

        backend.transform = mutate
        with self.assertRaisesRegex(RuntimeError, "Frozen backend configuration changed"):
            self.infer(backend)
        self.assertTrue(self.metadata.exists())
        self.assertEqual(read_json(self.metadata)["actual_backend"]["actual_dtype"], "float32")
        self.assertFalse(self.output.exists())
        self.assertEqual(backend.closed, 0)

    def test_official_settings_changed_during_prediction_are_not_published(self):
        from traversability_inference.configuration import normalize_settings
        from traversability_inference.qwen import QwenBackend

        official = object.__new__(QwenBackend)
        official.model_key = MODEL
        official.settings = normalize_settings(MODEL, {})
        fixture = FixtureBackend()
        official.metadata = fixture.metadata
        official.close = fixture.close

        def mutate(frame, regions, data_root):
            records = fixture.predict_frame(frame, regions, data_root)
            official.settings["scene_visual_token_budget"] = 128
            return records

        official.predict_frame = mutate
        with self.assertRaisesRegex(RuntimeError, "Official backend settings changed"):
            self.infer(official)
        self.assertTrue(self.metadata.exists())
        self.assertEqual(read_json(self.metadata)["configuration"]["settings"]["scene_visual_token_budget"], 384)
        self.assertFalse(self.output.exists())
        self.assertEqual(fixture.closed, 0)

    def test_manifests_changed_during_prediction_are_not_published(self):
        def mutate(records):
            self.frames[0]["timestamp_s"] = 8.0
            self.write_manifests()
            return records

        with self.assertRaisesRegex(RuntimeError, "Frozen inference manifest changed"):
            self.infer(FixtureBackend(mutate))
        self.assertTrue(self.metadata.exists())
        self.assertFalse(self.output.exists())

    def test_symlink_escape_is_rejected_before_hashing_external_inputs(self):
        external = self.root / "outside.png"
        Image.new("RGB", (20, 20)).save(external)
        image = self.data / "b.png"
        image.unlink()
        try:
            image.symlink_to(external)
        except OSError:
            self.skipTest("Creating symlinks is unavailable to this account")
        with self.assertRaisesRegex(InferenceError, "escapes data root"):
            self.infer()
        self.assertFalse(self.metadata.exists())

    def test_atomic_replace_failure_leaves_previous_published_file(self):
        destination = self.root / "artifact.json"
        atomic_json(destination, {"value": "before"})
        with patch("traversability_inference.storage.os.replace", side_effect=OSError("Interrupted")):
            with self.assertRaises(OSError):
                atomic_json(destination, {"value": "after"})
        self.assertEqual(read_json(destination), {"value": "before"})
        self.assertEqual(list(self.root.glob(".artifact.json.*.tmp")), [])

    def test_exclusive_lock_is_released_after_exception(self):
        lock = self.root / "fixture.lock"
        with self.assertRaisesRegex(RuntimeError, "Fixture aborted"):
            with exclusive_lock(lock):
                with self.assertRaisesRegex(RuntimeError, "locked"):
                    with exclusive_lock(lock):
                        pass
                raise RuntimeError("Fixture aborted")
        with exclusive_lock(lock):
            pass


if __name__ == "__main__":
    unittest.main()
