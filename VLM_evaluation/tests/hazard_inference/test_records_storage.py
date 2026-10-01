"""CPU contract tests; only independent stdlib fixtures and local frozen policy."""

from copy import deepcopy
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from traversability_hazard_inference.configuration import (
    configuration_snapshot, configured_roots, component_root,
    load_policy, normalize_settings, prompt_text,
)
from traversability_hazard_inference.records import (
    InferenceError, parse_response, prediction, validate_prediction,
)
from traversability_hazard_inference import storage


def frame():
    return {
        "frame_id": "fixture:frame", "source": "fixture", "scene_id": "scene",
        "sequence_id": "seq", "timestamp_s": None, "image_path": "rgb/frame.png",
        "split": "test", "width": 8, "height": 9, "image_sha256": "a" * 64,
    }


class ParsingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = load_policy(normalize_settings("qwen3_vl_4b", {}))

    def test_duplicates_preserve_first_original_and_alias_distinct_phrases(self):
        result = parse_response(json.dumps({"prompts": [" Person ", "PERSON", "water", "puddle", " USB  cable "]}), self.policy)
        self.assertEqual(result["prompts"], [" Person ", "water", "puddle", " USB  cable "])
        self.assertEqual(result["diagnostics"]["duplicate_count"], 1)
        self.assertEqual(result["diagnostics"]["unmapped"], [" USB  cable "])

    def test_vague_terms_remain_unusable_without_alias_expansion(self):
        result = parse_response('{"prompts":[" Unsafe  Area ","obstacles","cable"]}', self.policy)
        self.assertEqual(result["prompts"], [" Unsafe  Area ", "obstacles", "cable"])
        self.assertEqual(result["diagnostics"]["vague_unusable"], [" Unsafe  Area ", "obstacles"])
        self.assertEqual(result["diagnostics"]["unmapped"], ["cable"])

    def test_valid_empty_vs_parse_failure(self):
        result = parse_response(' {"prompts": []} \n', self.policy)
        record = prediction(frame(), "qwen3_vl_4b", prompts=result["prompts"], raw_response='{"prompts":[]}')
        self.assertEqual(record["status"], "ok")
        self.assertIsNone(record["error_code"])
        failed = prediction(frame(), "qwen3_vl_4b", raw_response="bad", error_code="invalid_json")
        self.assertEqual(failed["status"], "error")
        self.assertEqual(failed["raw_response"], "bad")
        self.assertEqual(failed["prompts"], [])

    def test_strict_json_schema_and_phrase_limits(self):
        cases = [
            ("", "invalid_json"), ("null", "invalid_response_schema"),
            ('[{"prompts":[]}]', "invalid_response_schema"),
            ('{"prompts":[],"confidence":1}', "invalid_response_schema"),
            ('{"prompts":[],"prompts":["person"]}', "invalid_json"),
            ('{"prompts":NaN}', "invalid_json"),
            ('{"prompts":Infinity}', "invalid_json"),
            ('{"prompts":1e999}', "invalid_json"),
            (r'{"prompts":["\ud800"]}', "invalid_phrase_unicode"),
            ('{"prompts":["' + chr(0xD800) + '"]}', "invalid_response_unicode"),
            ('{"prompts":"person"}', "invalid_prompts_type"),
            ('{"prompts":null}', "invalid_prompts_type"),
            ('{"prompts":[null]}', "invalid_phrase_type"),
            ('{"prompts":[true]}', "invalid_phrase_type"),
            ('{"prompts":[1]}', "invalid_phrase_type"),
            ('{"prompts":[""]}', "empty_phrase"),
            ('{"prompts":[" \\t\\n"]}', "empty_phrase"),
            ('```json\n{"prompts":[]}\n```', "invalid_json"),
            ('{"prompts":[]} explanation', "invalid_json"),
            ('<think></think>{"prompts":[]}', "invalid_json"),
            (json.dumps({"prompts": ["x" * 81]}), "phrase_too_long"),
            (json.dumps({"prompts": ["person"] * 33}), "too_many_prompts"),
        ]
        for raw, code in cases:
            with self.subTest(raw=raw):
                with self.assertRaises(InferenceError) as caught:
                    parse_response(raw, self.policy)
                self.assertEqual(caught.exception.code, code)
        self.assertEqual(len(parse_response(json.dumps({"prompts": ["x" * 80]}), self.policy)["prompts"][0]), 80)
        self.assertEqual(len(parse_response(json.dumps({"prompts": [str(i) for i in range(32)]}), self.policy)["prompts"]), 32)

    def test_saved_record_rejects_invalid_identity_status_and_duplicates(self):
        base = prediction(frame(), "qwen3_vl_4b", prompts=["person"], raw_response='{"prompts":["person"]}<|im_end|>')
        self.assertIs(validate_prediction(base), base)  # raw retains generation controls
        mutations = [
            {"task_id": "legacy"}, {"schema_version": True}, {"model_key": "clip_vit_b32"},
            {"status": "error", "error_code": "failure"}, {"error_code": "failure"},
            {"prompts": ["person", " Person "]}, {"raw_response": None}, {"region_id": "r"},
        ]
        for change in mutations:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_prediction({**base, **change})
        with self.assertRaises(ValueError):
            validate_prediction(base, frame_id="foreign")
        with self.assertRaises(ValueError):
            validate_prediction(base, model_key="qwen3_5_4b")


class StorageTests(unittest.TestCase):
    def test_paths_reject_lexical_and_symlink_escapes(self):
        for value in ("../x", "/x", "C:/x", "a\\b", "a/../b", "a//b", "a/./b", "x:stream", "", "a\x00b"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                storage.validate_relative_path(value)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            safe = root / "safe"
            safe.mkdir()
            self.assertEqual(storage.safe_path(safe, "rgb/image.png"), safe / "rgb/image.png")
            try:
                (safe / "link").symlink_to(root, target_is_directory=True)
            except OSError:
                self.skipTest("Host does not permit directory symlink creation")
            with self.assertRaises(ValueError):
                storage.safe_path(safe, "link/image.png")

    def test_frame_validation_duplicate_types_and_preservation(self):
        f = frame()
        f["extra_identity"] = {"opaque": 1}
        frozen = deepcopy(f)
        storage.validate_frames([f])
        self.assertEqual(f, frozen)
        for change in ({"width": True}, {"height": 0}, {"timestamp_s": float("nan")},
                       {"timestamp_s": True}, {"image_sha256": "A" * 64}, {"split": "train"},
                       {"region_id": "r"}, {"source": ""}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                storage.validate_frames([{**f, **change}])
        with self.assertRaises(ValueError):
            storage.validate_frames([f, f])

    def test_strict_reads_canonical_hash_and_content_identity(self):
        self.assertEqual(storage.stable_fingerprint({"a": 1, "b": 2}), storage.stable_fingerprint({"b": 2, "a": 1}))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset = root / "asset"
            self.assertEqual(storage.content_identity(asset), {"status": "missing"})
            asset.write_bytes(b"fixture RGB bytes")
            self.assertEqual(storage.content_identity(asset), {"status": "present", "sha256": hashlib.sha256(asset.read_bytes()).hexdigest()})
            for content in ('{"a":1,"a":2}\n', '\n', '[]\n', '{"a":NaN}\n'):
                asset.write_text(content, encoding="utf-8")
                with self.subTest(content=content), self.assertRaises(ValueError):
                    storage.read_jsonl(asset)

    def test_atomic_failure_preserves_prior_file_and_removes_temporary(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "predictions.jsonl"
            storage.atomic_jsonl(destination, [{"frame_id": "first"}])
            initial = destination.read_bytes()
            with patch.object(storage.os, "replace", side_effect=OSError("fixture interruption")):
                with self.assertRaises(OSError):
                    storage.atomic_jsonl(destination, [{"frame_id": "second"}])
            self.assertEqual(destination.read_bytes(), initial)
            self.assertEqual(list(destination.parent.glob("*.tmp")), [])

    def test_raw_surrogates_remain_auditable_without_publication_failure(self):
        # Preserve even a malformed injected decoder result in diagnostic JSON;
        # unusable Unicode phrases are rejected separately by the parser.
        raw = 'malformed decoder ' + chr(0xD800)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prediction.json"
            storage.atomic_json(path, {"raw_response": raw})
            self.assertEqual(storage.read_json(path)["raw_response"], raw)

    def test_lock_exclusivity_and_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "output.lock"
            with storage.exclusive_lock(path):
                with self.assertRaises(RuntimeError):
                    with storage.exclusive_lock(path):
                        self.fail("Concurrent lock acquired")
            with storage.exclusive_lock(path):
                self.assertTrue(path.exists())


class ConfigurationTests(unittest.TestCase):
    def test_default_cache_matches_notebook_setup_and_explicit_override_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with patch.dict(os.environ, {"TRAVERSABILITY_CACHE_ROOT": str(root)}):
                for model in ("qwen3_vl_4b", "qwen3_5_4b"):
                    with self.subTest(model=model):
                        expected = str(root / "huggingface/hub")
                        self.assertEqual(normalize_settings(model, {})["cache_dir"], expected)
                        self.assertEqual(normalize_settings(model, {"cache_dir": None})["cache_dir"], expected)
                        explicit = root / "explicit-cache"
                        self.assertEqual(normalize_settings(model, {"cache_dir": str(explicit)})["cache_dir"], str(explicit))
                self.assertEqual(list(root.iterdir()), [])

    def test_prompt_is_common_and_omits_source_policy(self):
        settings = normalize_settings("qwen3_vl_4b", {})
        policy = load_policy(settings)
        text = prompt_text(policy)
        for phrase in policy["canonical_prompts"]:
            self.assertIn(phrase, text)
        for forbidden in ("rellis", "coco", "present_concepts", "scored_category_names", "ignore_label_ids"):
            self.assertNotIn(forbidden, text.lower())
        self.assertEqual(configuration_snapshot("qwen3_vl_4b", {})["prompt"], configuration_snapshot("qwen3_5_4b", {})["prompt"])

    def test_operational_options_and_external_roots_are_portable(self):
        original = configuration_snapshot("qwen3_vl_4b", {})
        changed = configuration_snapshot("qwen3_vl_4b", {"fixture": True, "retry_errors": True, "cache_dir": "other-cache", "local_files_only": False})
        self.assertEqual(original, changed)
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(os.environ, {"TRAVERSABILITY_REPO_ROOT": str(ROOT.parent), "TRAVERSABILITY_DATA_ROOT": temporary}):
                self.assertEqual(component_root(), ROOT)
                self.assertEqual(configured_roots()["data_root"], Path(temporary).resolve())

    def test_settings_reject_legacy_and_unsupported_values(self):
        for model, values in (("clip_vit_b32", {}), ("qwen3_vl_4b", {"crop_visual_token_budget": 256}),
                              ("qwen3_5_4b", {"device": "auto"}), ("qwen3_5_4b", {"device": "cpu"}),
                              ("qwen3_5_4b", {"max_new_tokens": 128}), ("qwen3_5_4b", {"visual_token_budget": 1025}),
                              ("qwen3_5_4b", {"context_token_limit": 256}), ("qwen3_5_4b", {"enable_thinking": True}),
                              ("qwen3_5_4b", {"fixture": 1})):
            with self.subTest(model=model, values=values), self.assertRaises(ValueError):
                normalize_settings(model, values)

    def test_imports_work_with_model_and_other_traversability_modules_blocked(self):
        script = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split('.')[0]
        if root in {'torch','transformers','bitsandbytes','huggingface_hub','PIL','numpy'} or (root.startswith('traversability_') and root != 'traversability_hazard_inference'):
            raise AssertionError('Forbidden import: '+fullname)
sys.meta_path.insert(0, Block())
import traversability_hazard_inference
import traversability_hazard_inference.qwen
import traversability_hazard_inference.runner
import traversability_hazard_inference.download
'''
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, "-B", "-c", script], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
