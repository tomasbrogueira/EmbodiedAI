"""Public preparation/validation contracts on explicitly synthetic CPU inputs."""

from collections import Counter
import copy
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from common import (COMPONENT, FRAME_KEYS, REFERENCE_KEYS, config, file_snapshot,
                    read_json, read_jsonl, sha256, write_json, write_jsonl)
from traversability_hazard_data import prepare_run, validate_run


class FixtureRunTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.config = config(self.base)
        self.data = Path(self.config["data_root"])
        self.run = prepare_run(self.config, fixture=True)
        self.frames = read_jsonl(self.run / "frames.jsonl")
        self.references = read_jsonl(self.run / "references.jsonl")
        self.metadata = read_json(self.run / "metadata/dataset.json")

    def tearDown(self):
        self.temporary.cleanup()

    def test_fixture_is_explicitly_synthetic_and_covers_both_sources(self):
        summary = validate_run(self.run, self.data)
        self.assertIsInstance(self.run, Path)
        self.assertTrue(summary["valid"])
        self.assertEqual(summary["task_id"], "hazard_prompt_v1")
        self.assertEqual(summary["schema_version"], 1)
        self.assertTrue(summary["fixture"])
        self.assertEqual(summary["frames"], 11)
        self.assertEqual(summary["references"], 11)
        self.assertTrue(self.metadata["fixture"])
        self.assertEqual(Counter(row["source"] for row in self.frames), {"rellis": 5, "coco": 6})
        self.assertEqual(len({row["frame_id"] for row in self.frames}), 11)
        self.assertEqual(len({row["image_path"] for row in self.frames}), 11)
        self.assertEqual(summary["dataset_fingerprint"], self.metadata["dataset_fingerprint"])
        self.assertFalse((self.run / "regions.jsonl").exists())
        self.assertFalse((self.run / "annotations.jsonl").exists())

    def test_exact_frame_and_reference_schemas_original_dimensions_and_hashes(self):
        policy = read_json(COMPONENT / "configs/hazards/policy.json")
        references = {row["frame_id"]: row for row in self.references}
        for frame in self.frames:
            self.assertEqual(set(frame), FRAME_KEYS)
            self.assertIsNone(frame["timestamp_s"])
            image = self.data / frame["image_path"]
            self.assertEqual(sha256(image), frame["image_sha256"])
            with Image.open(image) as rgb:
                self.assertEqual(rgb.size, (frame["width"], frame["height"]))
            reference = references[frame["frame_id"]]
            self.assertEqual(set(reference), REFERENCE_KEYS)
            expected = (list(policy["datasets"]["rellis"]["hazard_label_ids"])
                        if frame["source"] == "rellis" else policy["datasets"]["coco"]["scored_category_names"])
            self.assertEqual(reference["scored_concepts"], expected)
            self.assertEqual(reference["status"], "complete")
            self.assertEqual(reference["reference_scope"], "annotated_hazard_concepts")
            self.assertEqual(reference["annotation_source"], "dataset_policy")
            self.assertEqual(set(reference["concept_masks"]), set(reference["present_concepts"]))
            self.assertEqual(set(reference["concept_pixel_counts"]), set(reference["present_concepts"]))
            for concept, relative in reference["concept_masks"].items():
                with Image.open(self.data / relative) as mask:
                    self.assertEqual(mask.mode, "L")
                    self.assertEqual(mask.size, (frame["width"], frame["height"]))
                    values = np.asarray(mask)
                self.assertTrue(set(np.unique(values)).issubset({0, 255}))
                self.assertEqual(int(np.count_nonzero(values)), reference["concept_pixel_counts"][concept])
                self.assertGreater(reference["concept_pixel_counts"][concept], 0)
            with Image.open(self.data / reference["valid_mask_path"]) as valid:
                self.assertEqual(valid.mode, "L")
                self.assertEqual(valid.size, (frame["width"], frame["height"]))
                self.assertTrue(set(np.unique(np.asarray(valid))).issubset({0, 255}))
            if frame["source"] == "coco":
                self.assertTrue(reference["absence_scoring_eligible"])

    def test_portable_paths_and_safe_asset_names(self):
        paths = [row["image_path"] for row in self.frames]
        for reference in self.references:
            paths.extend(reference["concept_masks"].values())
            paths.append(reference["valid_mask_path"])
        for relative in paths:
            self.assertFalse(PurePosixPath(relative).is_absolute())
            self.assertNotIn("\\", relative)
            self.assertNotIn(":", relative)
            self.assertNotIn("..", PurePosixPath(relative).parts)
            self.assertNotIn("//", relative)
            self.assertTrue((self.data / relative).is_file())
        self.assertEqual(self.metadata["policy_sha256"], sha256(COMPONENT / "configs/hazards/policy.json"))
        self.assertEqual(len(self.metadata["alias_sha256"]), 64)
        self.assertTrue(self.metadata["annotations"])
        self.assertTrue(self.metadata["assets"])
        self.assertTrue(self.metadata["selection"])
        self.assertTrue(self.metadata["coverage"])
        self.assertTrue(self.metadata["reference_details"])
        for relative, digest in self.metadata["assets"].items():
            self.assertEqual(sha256(self.data / relative), digest)

    def test_preparation_is_idempotent_in_content_and_modification_times(self):
        before = file_snapshot(self.base)
        second = prepare_run(self.config, fixture=True)
        self.assertEqual(second, self.run)
        self.assertEqual(file_snapshot(self.base), before)
        validate_run(second, self.data)

    def test_fixture_fingerprint_is_portable_between_roots(self):
        other_config = config(self.base / "relocated")
        other = prepare_run(other_config, fixture=True)
        other_metadata = read_json(other / "metadata/dataset.json")
        self.assertEqual(self.metadata["dataset_fingerprint"], other_metadata["dataset_fingerprint"])
        self.assertEqual(self.frames, read_jsonl(other / "frames.jsonl"))
        self.assertEqual(self.references, read_jsonl(other / "references.jsonl"))

    def test_changed_original_rgb_is_rejected_without_replacement(self):
        path = self.data / self.frames[0]["image_path"]
        with Image.open(path) as image:
            values = np.asarray(image).copy()
        values[0, 0, 0] ^= 255
        Image.fromarray(values).save(path)
        changed = path.read_bytes()
        with self.assertRaises((ValueError, OSError)):
            validate_run(self.run, self.data)
        with self.assertRaises((ValueError, OSError)):
            prepare_run(self.config, fixture=True)
        self.assertEqual(path.read_bytes(), changed)

    def test_rgb_identity_hashes_original_bytes_even_when_decoded_pixels_match(self):
        path = self.data / self.frames[0]["image_path"]
        with Image.open(path) as image:
            original_pixels = np.asarray(image).copy()
        path.write_bytes(path.read_bytes() + b"original-byte-identity-test")
        with Image.open(path) as image:
            np.testing.assert_array_equal(np.asarray(image), original_pixels)
        with self.assertRaises((ValueError, OSError)):
            validate_run(self.run, self.data)

    def test_changed_semantic_annotation_is_rejected(self):
        rellis_id = next(frame["frame_id"] for frame in self.frames if frame["source"] == "rellis")
        annotation = self.metadata["annotations"][rellis_id]
        path = self.data / annotation["path"]
        with Image.open(path) as image:
            values = np.asarray(image).copy()
        values[0, 0] = 34 if values[0, 0] != 34 else 7
        Image.fromarray(values).save(path)
        with self.assertRaises((ValueError, OSError)):
            validate_run(self.run, self.data)
        with self.assertRaises((ValueError, OSError)):
            prepare_run(self.config, fixture=True)

    def test_changed_policy_hash_is_rejected(self):
        self.metadata["policy_sha256"] = "0" * 64
        write_json(self.run / "metadata/dataset.json", self.metadata)
        with self.assertRaises((ValueError, OSError)):
            validate_run(self.run, self.data)
        with self.assertRaises((ValueError, OSError)):
            prepare_run(self.config, fixture=True)

    def test_nonbinary_wrong_dimension_and_changed_mask_assets_are_rejected(self):
        reference = next(row for row in self.references if row["concept_masks"])
        path = self.data / next(iter(reference["concept_masks"].values()))
        original = path.read_bytes()
        frame = next(row for row in self.frames if row["frame_id"] == reference["frame_id"])
        for values in (np.full((frame["height"], frame["width"]), 17, dtype=np.uint8),
                       np.full((2, 3), 255, dtype=np.uint8),
                       np.zeros((frame["height"], frame["width"], 3), dtype=np.uint8)):
            with self.subTest(shape=values.shape):
                Image.fromarray(values).save(path)
                with self.assertRaises((ValueError, OSError)):
                    validate_run(self.run, self.data)
                path.write_bytes(original)
        validate_run(self.run, self.data)

    def test_join_duplicates_counts_category_coverage_and_schema_are_rejected(self):
        path = self.run / "references.jsonl"
        original = path.read_bytes()
        changes = []
        changes.append(self.references + [self.references[0]])
        changes.append(self.references[:-1])
        for field, value in (("frame_id", "absent:frame"), ("scored_concepts", []),
                             ("present_concepts", ["sky"]), ("status", "pending"),
                             ("concept_pixel_counts", {"person": -1}), ("region_id", "legacy")):
            rows = copy.deepcopy(self.references)
            rows[0][field] = value
            changes.append(rows)
        for index, rows in enumerate(changes):
            with self.subTest(mutation=index):
                write_jsonl(path, rows)
                with self.assertRaises((ValueError, OSError)):
                    validate_run(self.run, self.data)
                path.write_bytes(original)

    def test_frame_paths_identity_dimensions_and_split_leakage_are_rejected(self):
        path = self.run / "frames.jsonl"
        original = path.read_bytes()
        changes = [self.frames + [self.frames[0]], self.frames[:-1]]
        rellis_index = next(index for index, row in enumerate(self.frames) if row["sequence_id"] == "00000")
        for field, value in (("split", "test"), ("width", 0), ("height", True),
                             ("image_sha256", "0" * 64), ("image_path", "../outside.png"),
                             ("image_path", "C:/outside.png"), ("image_path", "images\\frame.png"),
                             ("region_id", "legacy")):
            rows = copy.deepcopy(self.frames)
            rows[rellis_index][field] = value
            changes.append(rows)
        for index, rows in enumerate(changes):
            with self.subTest(mutation=index):
                write_jsonl(path, rows)
                with self.assertRaises((ValueError, OSError)):
                    validate_run(self.run, self.data)
                path.write_bytes(original)


class PreparationBoundaryTests(unittest.TestCase):
    def test_missing_real_sources_produce_pending_coverage_and_no_invented_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = config(temporary, "real")
            with patch("socket.create_connection", side_effect=AssertionError("network prohibited")), \
                 patch("urllib.request.urlopen", side_effect=AssertionError("download prohibited")):
                run = prepare_run(cfg)
                summary = validate_run(run, cfg["data_root"])
            self.assertFalse(summary["fixture"])
            self.assertEqual(summary["frames"], 0)
            self.assertEqual(summary["references"], 0)
            self.assertEqual(read_jsonl(run / "frames.jsonl"), [])
            self.assertEqual(read_jsonl(run / "references.jsonl"), [])
            metadata = read_json(run / "metadata/dataset.json")
            coverage_text = json.dumps(metadata["coverage"]).lower()
            for text in ("rellis", "coco", "pending"):
                self.assertIn(text, coverage_text)
            self.assertFalse(any(Path(cfg["data_root"]).rglob("*.png")))
            first = file_snapshot(temporary)
            prepare_run(cfg)
            self.assertEqual(file_snapshot(temporary), first)

    def test_legacy_run_refusal_preserves_all_legacy_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = config(temporary, "legacy")
            run = Path(cfg["run_root"]) / cfg["run_name"]
            (run / "metadata").mkdir(parents=True)
            (run / "regions.jsonl").write_text('{"region_id":"legacy"}\n', encoding="utf-8")
            (run / "frames.jsonl").write_text('{"frame_id":"legacy"}\n', encoding="utf-8")
            write_json(run / "metadata/dataset.json", {"task_id": "region_classification_v1"})
            first = file_snapshot(run)
            with self.assertRaises((ValueError, OSError)):
                prepare_run(cfg, fixture=True)
            self.assertEqual(file_snapshot(run), first)
            self.assertFalse(Path(cfg["data_root"]).exists())

    def test_unidentified_nonempty_run_is_refused_before_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = config(temporary, "unknown")
            run = Path(cfg["run_root"]) / cfg["run_name"]
            run.mkdir(parents=True)
            (run / "notes.txt").write_text("existing result", encoding="utf-8")
            first = file_snapshot(run)
            with self.assertRaises((ValueError, OSError)):
                prepare_run(cfg, fixture=True)
            self.assertEqual(file_snapshot(run), first)

    def test_safe_config_file_mapping_and_explicit_root_precedence(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            cfg = config(base / "explicit")
            filename = base / "config.json"
            write_json(filename, cfg)
            wrong = base / "environment"
            with patch.dict(os.environ, {"TRAVERSABILITY_DATA_ROOT": str(wrong / "data"),
                                         "TRAVERSABILITY_RUN_ROOT": str(wrong / "runs"),
                                         "TRAVERSABILITY_CACHE_ROOT": str(wrong / "cache")}):
                run = prepare_run(filename, fixture=True)
            self.assertEqual(run, Path(cfg["run_root"]) / cfg["run_name"])
            self.assertTrue(validate_run(run, cfg["data_root"])["valid"])
            self.assertFalse(wrong.exists())

    def test_unsafe_run_names_are_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            for name in ("../escape", "nested/run", "C:/run", "a\\run", ".", ""):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    prepare_run(config(temporary, name), fixture=True)

    def test_task_schema_fixture_flag_and_frozen_selection_parameters_are_strict(self):
        changes = [
            {"task_id": "legacy"}, {"schema_version": True}, {"schema_version": 2},
            {"seed": 1}, {"rellis": {"frames_per_sequence": 1}},
            {"rellis": {"sequences": ["00000"]}},
            {"coco": {"positive_counts": {"development": 1, "test": 2}}},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            for change in changes:
                with self.subTest(change=change), self.assertRaises(ValueError):
                    prepare_run({**config(temporary), **change}, fixture=True)
            with self.assertRaises(ValueError):
                prepare_run(config(temporary), fixture=1)

    def test_changed_original_policy_bytes_require_a_new_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            cfg = config(base)
            policy = base / "policy.json"
            policy.write_bytes((COMPONENT / "configs/hazards/policy.json").read_bytes())
            cfg["policy_path"] = str(policy)
            run = prepare_run(cfg, fixture=True)
            before = file_snapshot(run)
            policy.write_bytes(policy.read_bytes() + b"\n")
            with self.assertRaises((ValueError, OSError)):
                prepare_run(cfg, fixture=True)
            self.assertEqual(file_snapshot(run), before)

    def test_public_apis_need_no_other_new_package_models_gpu_or_network(self):
        script = r'''
import importlib.abc
import json
from pathlib import Path
import socket
import sys
import tempfile
import urllib.request

sys.path.insert(0, sys.argv[1])
class Blocked(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        base = fullname.split(".")[0]
        if base in {"torch", "transformers", "sam2", "sam3", "huggingface_hub", "cv2"}:
            raise AssertionError("model/GPU dependency: " + fullname)
        if base.startswith("traversability_hazard_") and base != "traversability_hazard_data":
            raise AssertionError("other hazard package: " + fullname)
sys.meta_path.insert(0, Blocked())
def prohibited(*args, **kwargs):
    raise AssertionError("network is prohibited")
socket.create_connection = prohibited
urllib.request.urlopen = prohibited
from traversability_hazard_data import prepare_run, validate_run
with tempfile.TemporaryDirectory() as temp:
    cfg = {"data_root": str(Path(temp)/"data"), "run_root": str(Path(temp)/"runs"),
           "cache_root": str(Path(temp)/"cache"), "run_name": "independent"}
    run = prepare_run(cfg, fixture=True)
    report = validate_run(run, cfg["data_root"])
    assert report["fixture"] and report["frames"] == 11 and report["valid"]
    print(json.dumps({"fixture": report["fixture"], "frames": report["frames"]}))
'''
        result = subprocess.run([sys.executable, "-c", script, str(COMPONENT / "src")],
                                capture_output=True, text=True, check=False, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('"frames": 11', result.stdout)


if __name__ == "__main__":
    unittest.main()
