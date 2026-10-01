"""Explicit CPU mocks of official SAM3 image state/tensor APIs."""

import hashlib
import importlib
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from traversability_hazard_segmentation import sam3_adapter as adapter


class MockTensor:
    def __init__(self, values, log):
        self.values, self.log = values, log

    def detach(self):
        self.log.append("detach")
        return self

    def cpu(self):
        self.log.append("cpu")
        return self

    def numpy(self):
        return self.values


class MockProcessor:
    def __init__(self, *, reset_error=False):
        self.images, self.prompts, self.resets, self.tensor_log = [], [], 0, []
        self.reset_error = reset_error
        self.image_features = object()
        self.probabilities = np.zeros((1, 1, 3, 5), dtype=np.float32)
        self.score_buffer = np.array([0.9], dtype=np.float32)
        self.close_calls = 0

    def set_image(self, image):
        self.images.append((image.mode, image.size))
        return {"backbone_out": {"image_features": self.image_features}}

    def reset_all_prompts(self, state):
        self.resets += 1
        if self.reset_error:
            raise RuntimeError("reset failed")
        assert state["backbone_out"]["image_features"] is self.image_features
        for key in ("language_features", "language_mask", "language_embeds"):
            state["backbone_out"].pop(key, None)
        for key in ("geometric_prompt", "boxes", "masks", "masks_logits", "scores"):
            state.pop(key, None)

    def set_text_prompt(self, *, state, prompt):
        assert "geometric_prompt" not in state
        assert "language_features" not in state["backbone_out"]
        assert "masks" not in state
        self.prompts.append(prompt)
        state["backbone_out"]["language_features"] = prompt
        state["geometric_prompt"] = object()
        self.probabilities.fill(0.2)
        self.probabilities[0, 0, 1, (len(self.prompts) - 1) % 5] = 0.9
        self.score_buffer[0] = 0.9 if len(self.prompts) == 1 else 0.7
        state["masks_logits"] = MockTensor(self.probabilities, self.tensor_log)
        state["masks"] = MockTensor(self.probabilities > 0.5, self.tensor_log)
        state["scores"] = MockTensor(self.score_buffer, self.tensor_log)
        if prompt == "boom":
            # A partially populated, reset state remains available after failure.
            raise RuntimeError("grounding failed after mask output")
        if prompt == "none":
            state["masks_logits"] = MockTensor(np.empty((0, 1, 3, 5)), self.tensor_log)
            state["scores"] = MockTensor(np.empty((0,)), self.tensor_log)
        if prompt == "bad-size":
            state["masks_logits"] = np.ones((1, 1, 5, 3))
        if prompt == "multiple":
            probabilities = np.concatenate((self.probabilities, self.probabilities), axis=0)
            probabilities[1, 0].fill(0.2)
            probabilities[1, 0, 2, 4] = 0.9
            state["masks_logits"] = MockTensor(probabilities, self.tensor_log)
            state["scores"] = MockTensor(np.array([0.9, 0.8]), self.tensor_log)
        return state

    def close(self):
        self.close_calls += 1


class Sam3AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "frame.png"
        Image.new("RGB", (5, 3), (17, 23, 91)).save(self.path)
        self.frame = {"frame_id": "coco:fixture:1", "source": "coco", "scene_id": "fixture",
            "sequence_id": "fixture", "timestamp_s": None, "image_path": "frame.png",
            "split": "test", "width": 5, "height": 3,
            "image_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(), "fixture": True}

    def make_segmenter(self, processor=None, **settings):
        processor = processor or MockProcessor()
        return adapter.load_segmenter({"fixture": True, "processor": processor, **settings}), processor

    def test_image_reuse_reset_and_cpu_copies_preserve_original_phrase(self):
        segmenter, processor = self.make_segmenter()
        result = segmenter.segment_image(self.frame, ["water", "puddle"], self.root)
        self.assertEqual(processor.images, [("RGB", (5, 3))])
        self.assertEqual(processor.resets, 2)
        self.assertEqual(processor.prompts, ["water", "puddle"])
        self.assertEqual(result["frame"]["sam_query_count"], 2)
        first, second = result["queries"]
        self.assertTrue(first["masks"][0][1, 0])
        self.assertFalse(first["masks"][0][1, 1])
        self.assertTrue(second["masks"][0][1, 1])
        self.assertEqual(first["masks"][0].shape, (3, 5))
        self.assertAlmostEqual(first["scores"][0], 0.9, places=6)
        self.assertAlmostEqual(second["scores"][0], 0.7, places=6)
        self.assertGreaterEqual(processor.tensor_log.count("cpu"), 4)
        processor.probabilities.fill(0)
        self.assertEqual(int(result["frame"]["union_mask"].sum()), 2)
        self.assertEqual(result["settings"]["execution_kind"], "fixture")
        self.assertTrue(result["settings"]["fixture_identity"])

    def test_empty_list_and_empty_detections_are_successful_empties(self):
        segmenter, processor = self.make_segmenter()
        empty = segmenter.segment_image(self.frame, [], self.root)
        self.assertEqual(empty["frame"]["status"], "ok")
        self.assertEqual(empty["frame"]["union_mask"].shape, (3, 5))
        self.assertFalse(empty["frame"]["union_mask"].any())
        self.assertEqual(empty["frame"]["sam_query_count"], 0)
        self.assertEqual(len(processor.images), 1)
        no_detection = segmenter.segment_image(self.frame, ["none"], self.root)
        self.assertEqual(no_detection["queries"][0]["masks"], [])
        self.assertEqual(no_detection["queries"][0]["status"], "ok")
        self.assertEqual(no_detection["frame"]["sam_query_count"], 1)

    def test_partial_query_error_is_preserved_and_following_query_continues(self):
        segmenter, processor = self.make_segmenter()
        result = segmenter.segment_image(self.frame, ["boom", "cup"], self.root)
        self.assertEqual(result["frame"]["status"], "error")
        self.assertIn("grounding failed", result["queries"][0]["error_code"])
        self.assertEqual(len(result["queries"][0]["masks"]), 1)
        self.assertEqual(result["queries"][1]["status"], "ok")
        self.assertEqual(int(result["frame"]["union_mask"].sum()), 2)
        self.assertEqual(processor.resets, 2)

    def test_image_encoder_failure_preserves_requested_query_joins(self):
        segmenter, processor = self.make_segmenter()
        with patch.object(processor, "set_image", side_effect=RuntimeError("image encoding failed")):
            result = segmenter.segment_image(self.frame, ["water", "cup"], self.root)
        self.assertEqual([query["phrase"] for query in result["queries"]], ["water", "cup"])
        self.assertTrue(all(query["status"] == "error" for query in result["queries"]))
        self.assertIn("image encoding failed", result["frame"]["error_code"])
        self.assertEqual(result["frame"]["union_mask"].shape, (3, 5))
        self.assertEqual(result["frame"]["sam_query_count"], 0)

    def test_all_instances_and_probability_threshold_are_preserved(self):
        segmenter, _ = self.make_segmenter()
        result = segmenter.segment_image(self.frame, ["multiple"], self.root)
        query = result["queries"][0]
        self.assertEqual(len(query["masks"]), 2)
        self.assertEqual(query["scores"], [0.9, 0.8])
        self.assertEqual(int(query["union_mask"].sum()), 2)
        self.assertEqual(query["status"], "ok")

    def test_vague_and_reset_errors_do_not_count_as_sam_queries(self):
        segmenter, processor = self.make_segmenter(MockProcessor(reset_error=True))
        result = segmenter.segment_image(self.frame, ["Obstacle", "cup"], self.root)
        self.assertEqual([q["sam_query_count"] for q in result["queries"]], [0, 0])
        self.assertEqual(result["queries"][0]["error_code"], "vague_unusable_phrase")
        self.assertEqual(processor.prompts, [])
        self.assertEqual(result["frame"]["status"], "error")

    def test_image_hash_paths_sizes_prompts_and_fixture_are_validated_before_encoding(self):
        changes = [{"image_sha256": "0" * 64}, {"image_path": "../frame.png"},
                   {"height": 5}, {"fixture": False}, {"schema_version": 0}, {"schema_version": True}]
        for change in changes:
            with self.subTest(change=change):
                segmenter, processor = self.make_segmenter()
                result = segmenter.segment_image({**self.frame, **change}, ["cup"], self.root)
                self.assertEqual(result["frame"]["status"], "error")
                self.assertEqual(processor.images, [])
        segmenter, processor = self.make_segmenter()
        result = segmenter.segment_image(self.frame, ["cup", " CUP "], self.root)
        self.assertEqual(result["frame"]["status"], "error")
        self.assertEqual(processor.images, [])

    def test_bad_output_dimensions_are_errors_and_mask_threshold_is_explicit(self):
        segmenter, _ = self.make_segmenter()
        result = segmenter.segment_image(self.frame, ["bad-size"], self.root)
        self.assertEqual(result["queries"][0]["status"], "error")
        threshold_segmenter, processor = self.make_segmenter(mask_probability_threshold=0.9)
        threshold_result = threshold_segmenter.segment_image(self.frame, ["cup"], self.root)
        # float32 0.9 is below decimal 0.9; strict comparison must remain empty.
        self.assertFalse(threshold_result["frame"]["union_mask"].any())
        self.assertEqual(threshold_result["settings"]["sam3_existing_mask_threshold"], 0.5)

    def test_injection_loading_and_identity_guards(self):
        with self.assertRaises(adapter.SegmenterUnavailable) as injected:
            adapter.load_segmenter({"processor": MockProcessor()})
        self.assertEqual(injected.exception.error_code, "fixture_mode_required")
        with self.assertRaises(adapter.SegmenterUnavailable) as disabled:
            adapter.load_segmenter({})
        self.assertEqual(disabled.exception.error_code, "model_loading_disabled")
        for setting in ({"device": "cuda"}, {"policy_hash": "0" * 64}, {"schema_version": True}):
            with self.assertRaises(ValueError):
                self.make_segmenter(**setting)
        with self.assertRaises(adapter.SegmenterUnavailable):
            self.make_segmenter(checkpoint_revision="main")
        frozen_policy = json.loads((adapter._component_root() / "configs/hazards/policy.json").read_text(encoding="utf-8"))
        invalid_policy = self.root / "invalid_policy.json"
        invalid_documents = [json.dumps({**frozen_policy, "schema_version": True}),
                             json.dumps(frozen_policy).replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1'),
                             json.dumps({**frozen_policy, "presence_min_pixels": float("nan")})]
        for document in invalid_documents:
            invalid_policy.write_text(document, encoding="utf-8")
            with self.subTest(policy=document[:70]), self.assertRaises(ValueError):
                self.make_segmenter(policy_path=str(invalid_policy))

    def test_absolute_runtime_policy_path_is_supported_and_not_persisted(self):
        policy_path = adapter._component_root() / "configs/hazards/policy.json"
        segmenter, _ = self.make_segmenter(policy_path=str(policy_path))
        result = segmenter.segment_image(self.frame, ["cup"], self.root)
        self.assertEqual(result["frame"]["status"], "ok")
        self.assertNotIn("policy_path", segmenter.settings)
        self.assertNotIn("policy_path", result["settings"])

    def test_package_and_adapter_imports_load_no_model_libraries(self):
        script = "import sys; import traversability_hazard_segmentation; import traversability_hazard_segmentation.sam3_adapter; assert not any(name in sys.modules for name in ('torch','torchvision','sam3','transformers','huggingface_hub'))"
        environment = dict(os.environ)
        source_root = str(Path(__file__).resolve().parents[2] / "src")
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = source_root + (os.pathsep + existing if existing else "")
        result = subprocess.run([sys.executable, "-c", script], env=environment,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_close_is_idempotent_and_does_not_close_injected_processor(self):
        segmenter, processor = self.make_segmenter()
        segmenter.close()
        segmenter.close()
        self.assertEqual(processor.close_calls, 0)
        self.assertIsNone(segmenter.processor)
        result = segmenter.segment_image(self.frame, [], self.root)
        self.assertEqual(result["frame"]["status"], "error")

    def test_cached_and_local_checkpoint_resolution_checks_integrity_without_download(self):
        checkpoint = self.root / "sam3.pt"
        checkpoint.write_bytes(b"explicit checkpoint fixture")
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        settings = {"cache_root": str(self.root), "checkpoint_path": "sam3.pt",
                    "allow_downloads": False, "local_files_only": True}
        with patch.object(adapter, "CHECKPOINT_SIZE", checkpoint.stat().st_size), patch.object(adapter, "CHECKPOINT_SHA256", digest):
            resolved, identity = adapter._resolve_checkpoint(settings)
            self.assertEqual(resolved, checkpoint)
            self.assertEqual(identity["method"], "local_file_sha256")
            with patch.object(adapter, "_component_root", return_value=self.root):
                relative_resolved, _ = adapter._resolve_checkpoint({**settings, "cache_root": "."})
                self.assertEqual(relative_resolved, checkpoint)
                with patch.dict(os.environ, {"TRAVERSABILITY_CACHE_ROOT": "."}):
                    environment_resolved, _ = adapter._resolve_checkpoint({**settings, "cache_root": None})
                    self.assertEqual(environment_resolved, checkpoint)
            calls = []
            hub = SimpleNamespace(hf_hub_download=lambda **kwargs: calls.append(kwargs) or str(checkpoint))
            real_import = importlib.import_module
            with patch.object(adapter.importlib, "import_module", side_effect=lambda name: hub if name == "huggingface_hub" else real_import(name)):
                resolved, identity = adapter._resolve_checkpoint({**settings, "checkpoint_path": None})
            self.assertEqual(calls[0]["revision"], adapter.CHECKPOINT_REVISION)
            self.assertEqual(calls[0]["cache_dir"], str(self.root / "huggingface/hub"))
            self.assertTrue(calls[0]["local_files_only"])
            self.assertEqual(identity["method"], "pinned_hf_cache_sha256")
            checkpoint.write_bytes(b"wrong content")
            with self.assertRaises(adapter.SegmenterUnavailable) as mismatch:
                adapter._resolve_checkpoint(settings)
            self.assertEqual(mismatch.exception.error_code, "checkpoint_integrity_mismatch")
        for invalid in ("../sam3.pt", "CON.pt", "bad?/sam3.pt", "trailing./sam3.pt", "nul\0.pt", "a\\sam3.pt"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                adapter._resolve_checkpoint({**settings, "checkpoint_path": invalid})

    def test_code_revision_provenance_and_source_blob_integrity_fail_closed(self):
        package = self.root / "sam3"
        package.mkdir()
        source = package / "model_builder.py"
        data = b"# explicitly mocked official source bytes\n"
        source.write_bytes(data)
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        spec = SimpleNamespace(submodule_search_locations=[str(package)])
        provenance = {"url": "https://github.com/facebookresearch/sam3.git",
                      "vcs_info": {"commit_id": adapter.CODE_REVISION}}
        distribution = SimpleNamespace(read_text=lambda filename: json.dumps(provenance))
        with patch.object(adapter.importlib.util, "find_spec", return_value=spec), \
             patch.object(adapter.importlib.metadata, "distribution", return_value=distribution), \
             patch.object(adapter, "_SOURCE_BLOBS", {"model_builder.py": blob}):
            self.assertEqual(adapter._verify_code({})["status"], "verified")
            git_calls = []
            def git_read(arguments, **kwargs):
                git_calls.append(arguments)
                return SimpleNamespace(stdout=adapter.CODE_REVISION if "rev-parse" in arguments else "")
            with patch.object(adapter, "_component_root", return_value=self.root), \
                 patch.object(adapter.subprocess, "run", side_effect=git_read):
                self.assertEqual(adapter._verify_code({"code_source_root": "."})["method"], "clean_official_revision_checkout")
            self.assertEqual(git_calls[0][2], str(self.root.resolve()))
            source.write_bytes(b"modified source")
            with self.assertRaises(adapter.SegmenterUnavailable) as changed:
                adapter._verify_code({})
            self.assertEqual(changed.exception.error_code, "code_source_mismatch")
            provenance["vcs_info"]["commit_id"] = "main"
            with self.assertRaises(adapter.SegmenterUnavailable) as unpinned:
                adapter._verify_code({})
            self.assertEqual(unpinned.exception.error_code, "code_provenance_required")

    def test_real_loader_uses_explicit_indexed_context_and_disables_upstream_download(self):
        context_log, builder_calls, processor_calls = [], [], []
        class DeviceContext:
            def __enter__(self):
                context_log.append("enter")
            def __exit__(self, *args):
                context_log.append("exit")
        cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 2,
                               device=lambda index: context_log.append(index) or DeviceContext())
        torch = SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda="12.8"))
        model = object()
        builder = SimpleNamespace(build_sam3_image_model=lambda **kwargs: builder_calls.append(kwargs) or model)
        processor_module = SimpleNamespace(Sam3Processor=lambda loaded, **kwargs: processor_calls.append((loaded, kwargs)) or MockProcessor())
        modules = {"torch": torch, "sam3.model_builder": builder, "sam3.model.sam3_image_processor": processor_module}
        real_import = importlib.import_module
        versions = {"torch": "2.10.0", "numpy": "1.26.4"}
        with patch.object(adapter, "_verify_code", return_value={"status": "verified"}), \
             patch.object(adapter, "_resolve_checkpoint", return_value=(self.path, {"status": "verified"})), \
             patch.object(adapter, "_software", return_value=versions), \
             patch.object(adapter.importlib, "import_module", side_effect=lambda name: modules[name] if name in modules else real_import(name)):
            segmenter = adapter.load_segmenter({"enable_model_loading": True, "device": "cuda:1"})
        self.assertEqual(context_log, [1, "enter", "exit"])
        self.assertEqual(builder_calls[0]["device"], "cuda")
        self.assertFalse(builder_calls[0]["load_from_HF"])
        self.assertFalse(builder_calls[0]["enable_inst_interactivity"])
        self.assertEqual(processor_calls[0][1]["device"], "cuda:1")
        self.assertEqual(processor_calls[0][1]["resolution"], 1008)
        self.assertFalse(segmenter.metadata["fixture"])
        segmenter.close()


if __name__ == "__main__":
    unittest.main()
