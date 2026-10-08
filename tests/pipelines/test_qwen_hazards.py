"""Model-free conformance against actual common records and the lossless bridge."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
for source in (ROOT / "src", ROOT / "VLM_evaluation/src"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from pipeline_common.contracts import FramePacket, GeometryFrame, validate_semantic
from pipeline_common.io import file_sha256, rgb_sha256
from pipelines.qwen_hazards import create_adapter
from traversability_hazard_inference.configuration import MODEL_SPECS
from traversability_hazard_inference.records import prediction


class FakeQwen:
    def __init__(self, model_key="qwen3_5_4b", prompts=None, error_code=None):
        self.model_key = model_key
        self.prompts = list(["puddle", "person", "fallen log"] if prompts is None else prompts)
        self.error_code = error_code
        self.calls, self.closed = [], 0
        self.fixture_identity = "qwen-discovery-fixture-v1"
        self.metadata = {"fixture": True, "fixture_identity": self.fixture_identity,
                         "model_key": model_key, **MODEL_SPECS[model_key]}
        self.last_diagnostics = {}
        self.after_call = None

    def predict_image(self, frame, data_root):
        from pipeline_common.input_bridge import validate_model_input
        validate_model_input(frame, data_root)
        self.calls.append((deepcopy(frame), Path(data_root)))
        text = json.dumps({"prompts": self.prompts})
        self.last_diagnostics = {
            "frame_id": frame["frame_id"], "model_key": self.model_key,
            "preprocessing": {"original_width": frame["width"], "original_height": frame["height"],
                "processed_width": 28, "processed_height": 28, "alignment_factor": 28},
            "tokens": {"input_tokens": 137, "visual_tokens": 1, "output_tokens": 17},
            "generation": {"terminal_eos": True, "stop_reason": "eos", "eos_token_id": 248046},
            "parse_response_text": text,
            "status": "error" if self.error_code else "ok", "error_code": self.error_code,
        }
        returned = prediction(frame, self.model_key, prompts=self.prompts, raw_response=text + "<|im_end|>", error_code=self.error_code)
        if self.after_call:
            self.after_call(frame, returned)
        return returned

    def close(self):
        self.closed += 1


class FakeSam:
    def __init__(self):
        self.fixture_identity = "sam-instance-fixture-v1"
        self.metadata = {"fixture": True, "fixture_identity": self.fixture_identity}
        self.calls, self.closed = [], 0
        self.errors = {}
        self.image_error = False
        self.after_call = None
        self.returned = None

    def segment_image(self, frame, prompts, data_root):
        from pipeline_common.input_bridge import validate_model_input
        validate_model_input(frame, data_root)
        self.calls.append((deepcopy(frame), list(prompts), Path(data_root)))
        queries = []
        for index, phrase in enumerate(prompts):
            mask = np.zeros((frame["height"], frame["width"]), dtype=np.bool_)
            mask[:, index % frame["width"]] = True
            # Deliberate overlap: all concepts share one pixel.
            mask[0, 0] = True
            error = self.errors.get(phrase)
            queries.append({"phrase": phrase, "masks": [mask], "scores": [0.8],
                            "status": "error" if error else "ok", "error_code": error,
                            "sam_query_count": 1})
        failed = any(query["status"] == "error" for query in queries)
        self.returned = {"queries": queries, "frame": {"frame_id": frame["frame_id"],
            "status": "error" if failed or self.image_error else "ok",
            "error_code": "sam_image_error:RuntimeError:encode failed" if self.image_error else "sam_query_failure" if failed else None,
            "sam_query_count": len(prompts), "union_mask": np.ones((frame["height"], frame["width"]), dtype=np.bool_)},
            "settings": deepcopy(self.metadata)}
        if self.after_call:
            self.after_call(self.returned)
        return self.returned

    def close(self):
        self.closed += 1


def create_cli_evidence(output, *, crash_one_frame=False):
    """Real shared CLI/map/evaluator with marked injected CPU model providers.

The normal shared --fixture providers bypass production adapters. This explicit
test injection exercises the production adapter implementation without a model.
The analytic common reference is created before any model-provider call.
"""
    from pipeline_common.fixture import create_fixture
    from pipeline_common.evaluation import fingerprint
    from pipeline_common.io import read_json, write_json
    from run_pipeline import main as run_main
    from evaluate_pipeline import main as evaluate_main
    from compare_pipelines import main as compare_main
    from pipelines.fixed_hazards import create_adapter as create_fixed
    output = create_fixture(output)
    base_config = read_json(output / "run.json")
    # Canonical semantic IDs in both production hazard adapters are bare nouns.
    original_cost = base_config["robot"]["semantic_costs"].pop("hazard.water.v1")
    base_config["robot"]["semantic_costs"]["water"] = original_cost
    reference = read_json(output / "reference/reference.json")
    reference["taxonomy"]["concept_ids"] = ["ground_surface", "water", "person", "log"]
    reference["kinds"]["robot_decision"]["robot_reference_policy"]["robot_fingerprint"] = fingerprint(base_config["robot"])
    write_json(output / "reference/reference.json", reference)
    variants = ["qwen3_5_4b", "qwen3_vl_4b", "fixed_hazards"]
    evaluations, runs = [], {}
    for key in variants:
        pipeline = "fixed_hazards" if key == "fixed_hazards" else "qwen_hazards"
        path = ROOT / "configs/pipelines" / ("fixed_hazards.json" if pipeline == "fixed_hazards" else f"qwen_hazards_{key}.json")
        options = read_json(path)["pipeline"]
        sam = FakeSam()
        options.update(fixture=True, _segmenter=sam)
        if pipeline == "qwen_hazards":
            qwen = FakeQwen(key)
            if crash_one_frame:
                def crash(returned):
                    if returned["frame"]["frame_id"] == "frame_000001":
                        raise RuntimeError("fixture SAM crash after possible execution")
                sam.after_call = crash
            options["_backend"] = qwen
            adapter = create_adapter(options)
        else:
            adapter = create_fixed(options)
        config = deepcopy(base_config)
        config.update(pipeline_id=pipeline, pipeline_config={k: v for k, v in options.items() if not k.startswith("_")})
        config_path = output / "configs" / f"{key}.json"
        write_json(config_path, config)
        run_path, eval_path = output / "runs" / key, output / "evaluations" / key
        with patch("pipeline_common.fixture_adapters.create_fixture_adapter", return_value=adapter):
            result = run_main(["--pipeline", pipeline, "--sequence", str(output / "sequence/sequence.json"),
                "--config", str(config_path), "--geometry-cache", str(output / "geometry_cache"),
                "--output", str(run_path), "--fixture"])
        if result != 0:
            raise AssertionError(read_json(run_path / "run.json"))
        evaluate_main(["--run", str(run_path), "--reference", str(output / "reference/reference.json"), "--output", str(eval_path)])
        evaluate_main(["--run", str(run_path), "--output", str(output / "evaluations_no_reference" / key)])
        evaluations.append(eval_path)
        runs[key] = {"run": read_json(run_path / "run.json"), "semantic_frames": [json.loads(line) for line in (run_path / "semantics/frames.jsonl").read_text().splitlines() if line.strip()],
                     "concepts": read_json(run_path / "map/concepts.json"), "report": read_json(eval_path / "report.json")}
    arguments = [arg for directory in evaluations for arg in ("--evaluation", str(directory))]
    if compare_main(arguments + ["--output", str(output / "comparison")]) != 0:
        raise AssertionError(read_json(output / "comparison/comparison.json"))
    return output, runs


class QwenHazardsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        rgb = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
        rgb.setflags(write=False)
        image_path = self.root / "processed.png"
        Image.fromarray(rgb).save(image_path)
        points = np.zeros((4, 5, 3), dtype=np.float64)
        points[..., 2] = 1
        pose = np.eye(4)[:3]
        pose[:, 3] = [2, -3, 4]
        geometry = GeometryFrame(points=points, depth=np.ones((4, 5)), validity=np.ones((4, 5), dtype=np.bool_),
            confidence=np.ones((4, 5)), intrinsics=np.eye(3), world_to_camera=pose,
            processed_grid_id="crop-resize-grid-v1", geometry_fingerprint="geometry-with-captured-pose-v1",
            pose_revision="pose-captured-frame-7", up=(0, 0, 1))
        self.frame = FramePacket(sequence_id="generic-phone-sequence", frame_id="frame-7", timestamp_ns=7_000_000_000,
            timestamp_provenance={"clock": "fixture", "synthetic": True}, rgb=rgb, image_path=image_path,
            encoded_file_sha256=file_sha256(image_path), decoded_rgb_sha256=rgb_sha256(rgb),
            processed_grid_id="crop-resize-grid-v1", source_rgb_identity={"decoded_rgb_sha256": "a" * 64},
            source_to_processed={"crop": [5, 7, 10, 8], "resize": [5, 4]}, geometry=geometry)
        self.qwen, self.sam = FakeQwen(), FakeSam()

    def make_adapter(self, **updates):
        config = {"fixture": True, "model_key": self.qwen.model_key, "_backend": self.qwen, "_segmenter": self.sam}
        config.update(updates)
        adapter = create_adapter(config)
        self.addCleanup(adapter.close)
        return adapter

    def assert_valid(self, result):
        validate_semantic(result, self.frame)
        self.assertEqual(result.geometry_fingerprint, self.frame.geometry.geometry_fingerprint)
        self.assertEqual(result.timestamp_ns, self.frame.timestamp_ns)
        self.assertLessEqual(result.started_monotonic_ns, result.completed_monotonic_ns)

    def test_both_variants_select_one_qwen_and_preserve_hazard_meanings(self):
        for key in sorted(MODEL_SPECS.keys() & {"qwen3_5_4b", "qwen3_vl_4b"}):
            with self.subTest(key=key):
                self.qwen = FakeQwen(key)
                result = self.make_adapter().observe(self.frame)
                self.assert_valid(result)
                self.assertEqual(result.status, "ok")
                self.assertEqual([query.concept_id for query in result.queries], ["water", "person", "log"])
                self.assertEqual(result.model_calls, {"qwen_predict_image": 1, "sam3_segment_image": 1})
                self.assertEqual(result.query_count, 3)
                self.assertEqual(len(self.qwen.calls), 1)

    def test_original_phrase_order_aliases_unknown_nouns_and_overlap(self):
        self.qwen.prompts = [" Water puddle ", "water", "Cable", "person"]
        adapter = self.make_adapter()
        result = adapter.observe(self.frame)
        self.assert_valid(result)
        self.assertEqual(self.sam.calls[0][1], self.qwen.prompts)
        self.assertEqual([query.original_phrase for query in result.queries], self.qwen.prompts)
        self.assertEqual([query.concept_id for query in result.queries[:2]], ["water", "water"])
        self.assertTrue(result.queries[2].concept_id.startswith("unmapped:"))
        self.assertEqual(result.queries[2].concept_id, adapter._concept(" cable "))
        self.assertTrue(all(query.instances[0].mask[0, 0] for query in result.queries))
        self.assertTrue(all(query.role == "hazard" for query in result.queries))
        self.assertTrue(all(query.mapping_version == "visible_avoid_concepts_v1" for query in result.queries))

    def test_empty_discovery_skips_sam_encoding_by_default(self):
        self.qwen.prompts = []
        result = self.make_adapter().observe(self.frame)
        self.assert_valid(result)
        self.assertEqual((result.status, result.queries, result.query_count), ("ok", [], 0))
        self.assertEqual(result.model_calls["sam3_segment_image"], 0)
        self.assertEqual(self.sam.calls, [])

    def test_legacy_empty_discovery_encodes_and_preserves_image_error(self):
        self.qwen.prompts = []
        for image_error in (False, True):
            with self.subTest(image_error=image_error):
                self.sam.image_error = image_error
                result = self.make_adapter(empty_discovery_policy="encode_image").observe(self.frame)
                self.assert_valid(result)
                self.assertEqual(result.status, "error" if image_error else "ok")
                self.assertEqual(result.query_count, 0)
                self.assertEqual(result.model_calls["sam3_segment_image"], 1)
                self.assertEqual(self.sam.calls[-1][1], [])

    def test_upstream_failures_never_call_sam_or_become_successful_empty(self):
        for error in ("generation_truncated", "invalid_json", "generation_unverified_stop", "inference_failure"):
            with self.subTest(error=error):
                self.qwen.error_code = error
                result = self.make_adapter().observe(self.frame)
                self.assert_valid(result)
                self.assertEqual(result.status, "error")
                self.assertEqual((result.query_count, result.model_calls["sam3_segment_image"]), (0, 0))
                self.assertEqual(result.error["code"], error)
                self.assertIn("<|im_end|>", result.adapter_provenance["qwen_prediction"]["raw_response"])
        self.assertEqual(self.sam.calls, [])

    def test_unverified_success_and_invalid_strict_parser_are_failures(self):
        for kind in ("unverified", "invalid_json", "phrase_mismatch"):
            with self.subTest(kind=kind):
                def alter(frame, returned):
                    if kind == "unverified":
                        self.qwen.last_diagnostics["generation"].update(terminal_eos=False, stop_reason="unverified_stop")
                    elif kind == "invalid_json":
                        self.qwen.last_diagnostics["parse_response_text"] = '{"prompts":[],"extra":1}'
                    else:
                        self.qwen.last_diagnostics["parse_response_text"] = '{"prompts":["dog"]}'
                self.qwen.after_call = alter
                result = self.make_adapter().observe(self.frame)
                self.assertEqual(result.status, "error")
                self.assertEqual(result.adapter_provenance["strict_parser_outcome"]["status"], "error")
        self.assertEqual(self.sam.calls, [])

    def test_raw_diagnostics_tokens_and_wrapped_preprocessing_are_snapshots(self):
        result = self.make_adapter().observe(self.frame)
        provenance = result.adapter_provenance
        self.qwen.last_diagnostics["tokens"]["output_tokens"] = 999
        self.qwen.metadata["new_mutation"] = True
        self.assertEqual(provenance["actual_output_tokens"], 17)
        self.assertEqual(provenance["qwen_diagnostics"]["tokens"]["output_tokens"], 17)
        wrapper = provenance["preprocessing"]
        self.assertEqual(wrapper["input_space"], "lingbot_processed_rgb")
        self.assertEqual(wrapper["actual_qwen_resize"]["processed_width"], 28)
        self.assertEqual(wrapper["source_to_processed"], self.frame.source_to_processed)
        self.assertNotIn("new_mutation", wrapper["backend_metadata_snapshot"])
        self.assertEqual(provenance["configuration_snapshot"]["preprocessing"]["input"], "original whole RGB image")

    def test_hash_grid_geometry_pose_and_input_restrictions(self):
        result = self.make_adapter().observe(self.frame)
        self.assert_valid(result)
        qframe = self.qwen.calls[0][0]
        self.assertEqual(qframe, self.sam.calls[0][0])
        self.assertEqual(qframe["input_contract_id"], "semantic_mapping_v1")
        self.assertNotIn("source", qframe)  # Never invent RELLIS/COCO membership.
        self.assertFalse({"depth", "geometry", "masks", "references", "concepts"} & qframe.keys())
        self.assertNotEqual(qframe["encoded_file_sha256"], qframe["decoded_rgb_sha256"])
        self.assertEqual(result.adapter_provenance["input"]["pose_revision"], "pose-captured-frame-7")

    def test_pixel_and_file_tampering_prevent_all_model_calls(self):
        changed = np.array(self.frame.rgb, copy=True)
        changed[0, 0] = [255, 0, 0]
        packets = [replace(self.frame, rgb=changed), replace(self.frame, decoded_rgb_sha256="b" * 64),
                   replace(self.frame, encoded_file_sha256="b" * 64)]
        for packet in packets:
            result = self.make_adapter().observe(packet)
            self.assertEqual(result.status, "error")
            self.assertEqual(sum(result.model_calls.values()), 0)
        self.assertFalse(self.qwen.calls or self.sam.calls)

    def test_qwen_input_dict_is_isolated_and_file_mutation_blocks_sam(self):
        self.qwen.after_call = lambda frame, returned: frame.update(frame_id="mutated-local-record")
        result = self.make_adapter().observe(self.frame)
        self.assertEqual(result.status, "ok")
        self.assertEqual(self.sam.calls[-1][0]["frame_id"], self.frame.frame_id)
        def mutate_file(frame, returned):
            Image.fromarray(np.zeros_like(self.frame.rgb)).save(self.frame.image_path)
        self.qwen.after_call = mutate_file
        previous_calls = len(self.sam.calls)
        result = self.make_adapter().observe(self.frame)
        self.assertEqual(result.status, "error")
        self.assertEqual(len(self.sam.calls), previous_calls)

    def test_partial_masks_remain_diagnostic_and_are_copied(self):
        self.sam.errors["person"] = "sam_query_error:partial"
        result = self.make_adapter().observe(self.frame)
        self.assert_valid(result)
        self.assertEqual(result.status, "partial")
        self.assertEqual(result.queries[1].status, "error")
        self.assertTrue(result.queries[1].instances[0].metadata["diagnostic_only"])
        before = result.queries[0].instances[0].mask.copy()
        self.sam.returned["queries"][0]["masks"][0][:] = False
        np.testing.assert_array_equal(result.queries[0].instances[0].mask, before)
        self.assertFalse(result.queries[0].instances[0].mask.flags.writeable)
        self.assertEqual(result.query_count, 3)

    def test_all_query_failures_and_no_detection_success_are_distinct(self):
        self.sam.errors = {phrase: "sam_query_error:failed" for phrase in self.qwen.prompts}
        result = self.make_adapter().observe(self.frame)
        self.assert_valid(result)
        self.assertEqual(result.status, "error")
        self.sam.errors = {}
        def empty_masks(returned):
            for query in returned["queries"]:
                query.update(masks=[], scores=[])
        self.sam.after_call = empty_masks
        result = self.make_adapter().observe(self.frame)
        self.assertEqual(result.status, "ok")
        self.assertTrue(result.queries and all(query.instances == [] for query in result.queries))

    def test_reset_failure_and_text_failure_use_actual_counts(self):
        def alter(returned):
            returned["queries"][0].update(status="error", error_code="reset_failed", sam_query_count=0)
            returned["queries"][1].update(status="error", error_code="text_failed", sam_query_count=1)
            returned["frame"].update(status="error", error_code="sam_query_failure", sam_query_count=2)
        self.sam.after_call = alter
        result = self.make_adapter().observe(self.frame)
        self.assert_valid(result)
        self.assertEqual(result.status, "partial")
        self.assertEqual(result.query_count, 2)

    def test_global_sam_errors_are_preserved(self):
        self.sam.after_call = lambda returned: returned["frame"].update(status="error", error_code="cuda_failure")
        result = self.make_adapter().observe(self.frame)
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error["code"], "cuda_failure")

    def test_sam_crash_and_malformed_output_do_not_fabricate_zero_queries(self):
        def crash(returned):
            raise RuntimeError("after a text execution")
        self.sam.after_call = crash
        result = self.make_adapter().observe(self.frame)
        self.assertEqual(result.status, "error")
        self.assertIsNone(result.query_count)
        self.assertEqual(result.model_calls["sam3_segment_image"], 1)
        self.assertEqual(result.adapter_provenance["query_counts"]["availability"], "unavailable")
        self.assert_valid(result)
        self.assertTrue(result.query_count_reason)
        self.sam.after_call = lambda returned: returned["queries"].reverse()
        result = self.make_adapter().observe(self.frame)
        self.assertEqual(result.status, "error")
        self.assertIsNone(result.query_count)

    def test_bad_masks_keep_already_audited_execution_counts(self):
        self.sam.after_call = lambda returned: returned["queries"][0].update(masks=[np.ones((1, 1), dtype=np.bool_)])
        result = self.make_adapter().observe(self.frame)
        self.assert_valid(result)
        self.assertEqual(result.status, "error")
        self.assertEqual(result.query_count, 3)
        self.assertIsNone(result.query_count_reason)

    def test_real_settings_metadata_policy_mismatch_is_rejected(self):
        for key in ("policy_sha256", "alias_sha256", "prompt_sha256"):
            with self.subTest(key=key):
                self.qwen = FakeQwen()
                self.qwen.metadata[key] = "0" * 64
                result = self.make_adapter().observe(self.frame)
                self.assertEqual(result.status, "error")
                self.assertEqual(sum(result.model_calls.values()), 0)
        self.qwen = FakeQwen()
        self.sam.metadata["policy_hash"] = "0" * 64
        result = self.make_adapter().observe(self.frame)
        self.assertEqual(result.status, "error")
        self.assertEqual(result.model_calls["sam3_segment_image"], 0)

    def test_injected_ownership_and_fixture_identity(self):
        adapter = self.make_adapter()
        adapter.observe(self.frame)
        adapter.close()
        adapter.close()
        self.assertEqual((self.qwen.closed, self.sam.closed), (0, 0))
        self.assertEqual(adapter.observe(self.frame).status, "error")
        with self.assertRaises(ValueError):
            create_adapter({"model_key": self.qwen.model_key, "_backend": self.qwen})
        self.qwen.metadata["fixture"] = False
        with self.assertRaises(ValueError):
            self.make_adapter()

    def test_forbidden_variant_switch_download_or_device_is_rejected(self):
        for updates in ({"model_key": "qwen3_5_2b"}, {"model_keys": ["qwen3_5_4b", "qwen3_vl_4b"]},
                        {"fallback_model_key": "qwen3_vl_4b"}, {"qwen_settings": {"local_files_only": False}},
                        {"sam_settings": {"allow_downloads": True}}, {"sam_settings": {"device": "cuda:1"}}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                self.make_adapter(**updates)

    def test_import_and_factory_do_not_import_model_libraries(self):
        code = "from pipelines.qwen_hazards import create_adapter; import sys; create_adapter({'model_key':'qwen3_5_4b'}); assert not ({'torch','transformers','bitsandbytes','sam3'} & sys.modules.keys())"
        import os
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT / "src")
        completed = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_configs_have_one_pinned_variant_and_identical_fixed_sam_settings(self):
        fixed = json.loads((ROOT / "configs/pipelines/fixed_hazards.json").read_text())["pipeline"]
        for key in ("qwen3_5_4b", "qwen3_vl_4b"):
            config = json.loads((ROOT / f"configs/pipelines/qwen_hazards_{key}.json").read_text())
            options = config["pipeline"]
            self.assertEqual(options["sam_settings"], fixed["sam_settings"])
            self.assertEqual(options["model_key"], key)
            self.assertEqual(options["model"]["revision"], MODEL_SPECS[key]["revision"])
            self.assertIs(options["enable_model_loading"], False)
            self.assertEqual(options["empty_discovery_policy"], "skip_sam")
            self.assertFalse({"model_keys", "fallback_model_key"} & options.keys())
            adapter = create_adapter(options)
            adapter.close()

    def test_loading_reuses_one_selected_qwen_and_owned_cleanup(self):
        adapter = create_adapter({"model_key": self.qwen.model_key, "enable_model_loading": True})
        self.addCleanup(adapter.close)
        self.qwen.metadata.update(fixture=False, quantization={"compute_dtype": "torch.bfloat16"})
        self.sam.metadata.update(fixture=False)
        with patch("traversability_hazard_inference.load_backend", return_value=self.qwen) as loader, \
             patch("traversability_hazard_segmentation.load_segmenter", return_value=self.sam) as sam_loader, \
             patch.object(adapter, "_synchronize"):
            first = adapter.observe(self.frame)
            second = adapter.observe(self.frame)
        self.assertEqual((first.status, second.status), ("ok", "ok"))
        loader.assert_called_once()
        self.assertEqual(loader.call_args.args[0], self.qwen.model_key)
        sam_loader.assert_called_once()
        self.assertTrue(first.adapter_provenance["stage_timestamps"]["qwen_loading"]["loaded_this_call"])
        self.assertFalse(second.adapter_provenance["stage_timestamps"]["qwen_loading"]["loaded_this_call"])
        self.assertTrue(second.adapter_provenance["stage_timestamps"]["sam3"]["synchronized"])
        adapter.close()
        adapter.close()
        self.assertEqual((self.qwen.closed, self.sam.closed), (1, 1))

    def test_sam_loading_failure_records_attempt_and_cleanup_does_not_retry(self):
        adapter = create_adapter({"model_key": self.qwen.model_key, "enable_model_loading": True})
        self.addCleanup(adapter.close)
        self.qwen.metadata.update(fixture=False, quantization={"compute_dtype": "torch.bfloat16"})
        with patch("traversability_hazard_inference.load_backend", return_value=self.qwen) as loader, \
             patch("traversability_hazard_segmentation.load_segmenter", side_effect=RuntimeError("sam unavailable")) as sam_loader, \
             patch.object(adapter, "_synchronize"):
            result = adapter.observe(self.frame)
        self.assert_valid(result)
        self.assertEqual(result.status, "error")
        self.assertEqual(result.model_calls["sam3_segment_image"], 0)
        self.assertEqual(result.query_count, 0)
        stage = result.adapter_provenance["stage_timestamps"]["sam3_loading"]
        self.assertIsNotNone(stage["completion_monotonic_ns"])
        self.assertIn("error", stage)
        loader.assert_called_once()
        sam_loader.assert_called_once()
        adapter.close()
        self.assertEqual(self.qwen.closed, 1)

    def test_cleanup_failure_still_closes_both_owned_models(self):
        adapter = self.make_adapter()
        adapter._owned_backend = adapter._owned_segmenter = True
        with patch.object(self.sam, "close", side_effect=RuntimeError("sam cleanup failed")) as sam_close, \
             patch.object(self.qwen, "close") as qwen_close:
            with self.assertRaises(RuntimeError):
                adapter.close()
            qwen_close.assert_called_once()
            sam_close.assert_called_once()
        adapter.close()

    def test_actual_legacy_qwen_reader_and_sam_generic_input_path(self):
        from traversability_hazard_inference.preparation import load_rgb
        from traversability_hazard_segmentation.sam3_adapter import Sam3Segmenter
        from pipeline_common.input_bridge import to_model_input
        record, data_root = to_model_input(self.frame)
        with load_rgb(record, data_root) as rgb:
            np.testing.assert_array_equal(np.asarray(rgb), self.frame.rgb)
        images, prompts = [], []
        class Processor:
            def set_image(inner, image):
                images.append(np.array(image))
                return {}
            def reset_all_prompts(inner, state):
                pass
            def set_text_prompt(inner, *, state, prompt):
                prompts.append(prompt)
                return {"masks": np.ones((1, 4, 5), dtype=np.bool_), "scores": np.array([0.75])}
        self.sam = Sam3Segmenter({"fixture": True}, processor=Processor())
        result = self.make_adapter().observe(self.frame)
        self.assert_valid(result)
        self.assertEqual(result.status, "ok")
        np.testing.assert_array_equal(images[0], self.frame.rgb)
        self.assertEqual(prompts, self.qwen.prompts)
        self.assertEqual(result.query_count, 3)

    def test_shared_fusion_keeps_overlap_aliases_partial_and_unqueried_concepts(self):
        from pipeline_common.fusion import VoxelFuser
        self.qwen.prompts = ["puddle", "water", "person", "fallen log", "cable"]
        self.sam.errors["person"] = "sam_query_error:partial"
        result = self.make_adapter().observe(self.frame)
        fuser = VoxelFuser({"voxel_size": 0.25})
        fuser.add_semantics(result, self.frame)
        fuser.add_semantics(result, self.frame)
        voxels, evidence, concepts, journal, metadata = fuser.export()
        registry = {concept["concept_id"]: index for index, concept in enumerate(concepts)}
        observed = {concepts[index]["concept_id"] for index in evidence["concept_row"]}
        self.assertEqual(observed, {"water", "log", result.queries[-1].concept_id})
        self.assertIn("person", registry)  # Diagnostic registry, no accepted evidence.
        self.assertNotIn("dog", registry)  # Qwen omission never becomes an observed negative.
        self.assertEqual(int(evidence["frame_support"].max()), 1)  # Alias/replay cap.
        self.assertEqual(metadata["semantic_frame_count"], 1)
        self.assertFalse(any(row.get("fused") for row in journal if row.get("concept_id") == "person"))
        self.assertEqual(len(voxels["centers"]), 1)

    def test_shared_scheduler_drops_pending_frames_and_retains_captured_geometry_age(self):
        from pipeline_common.scheduling import LatestPendingWorker, mapped_capture
        from pipeline_common.fusion import VoxelFuser
        entered, release = threading.Event(), threading.Event()
        def delay(frame, returned):
            if frame["frame_id"] == "frame-7":
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("test release timed out")
        self.qwen.after_call = delay
        adapter = self.make_adapter()
        worker = LatestPendingWorker(adapter)
        self.addCleanup(worker.close)
        capture = time.monotonic_ns() - 10_000_000
        worker.submit(self.frame, capture)
        self.assertTrue(entered.wait(5))
        frame8 = replace(self.frame, frame_id="frame-8", timestamp_ns=8_000_000_000)
        new_points = self.frame.geometry.points.copy()
        new_points[..., 2] = 9
        frame9 = replace(self.frame, frame_id="frame-9", timestamp_ns=9_000_000_000,
                         geometry=replace(self.frame.geometry, points=new_points))
        worker.submit(frame8, capture + 1_000_000)
        worker.submit(frame9, capture + 2_000_000)
        self.assertEqual(worker.pending[0].frame_id, "frame-9")
        self.assertEqual([event["frame_id"] for event in worker.events], ["frame-8"])
        self.frame.geometry.world_to_camera[:, 3] = 100  # Newest caller pose cannot change in-flight pose.
        fuser = VoxelFuser({"voxel_size": 0.25})
        fuser.add_geometry(frame9)  # Geometry advances before old semantic completion.
        release.set()
        rows = list(worker.drain())
        self.assertEqual([row["frame"].frame_id for row in rows], ["frame-7", "frame-9"])
        np.testing.assert_array_equal(rows[0]["frame"].geometry.world_to_camera[:, 3], [2, -3, 4])
        self.assertEqual(rows[0]["mapped_capture_monotonic_ns"], capture)
        self.assertGreaterEqual(rows[0]["completed_monotonic_ns"] - capture, 10_000_000)
        self.assertFalse(rows[0]["frame"].rgb.flags.writeable)
        self.assertFalse(rows[0]["frame"].geometry.points.flags.writeable)
        fuser.add_semantics(rows[0]["result"], rows[0]["frame"])
        voxels, evidence, concepts, journal, _ = fuser.export()
        np.testing.assert_array_equal(voxels["voxel_indices"][evidence["voxel_row"], 2], np.full(len(evidence["voxel_row"]), 4))
        with self.assertRaises(ValueError):
            validate_semantic(rows[0]["result"], frame9)
        unknown = replace(self.frame, timestamp_ns=None, timestamp_provenance={"kind": "unknown"})
        self.assertIsNone(mapped_capture(unknown, 0, capture, 1.0))

    def test_both_variants_and_fixed_hazards_use_real_shared_cli_artifacts_and_evaluator(self):
        from pipeline_common.io import read_json
        output, runs = create_cli_evidence(self.root / "cli_evidence")
        for key in ("qwen3_5_4b", "qwen3_vl_4b"):
            result = runs[key]
            self.assertEqual({row["concept_id"] for row in result["concepts"]}, {"water", "person", "log"})
            self.assertEqual(len(result["semantic_frames"]), 3)
            self.assertTrue(all(frame["status"] == "ok" for frame in result["semantic_frames"]))
            self.assertTrue(all(frame["adapter_provenance"]["model_key"] == key for frame in result["semantic_frames"]))
            self.assertEqual(result["report"]["metrics"]["semantic_query_count"]["value"], 9)
            self.assertEqual(result["report"]["metrics"]["pipeline_failure_rate"]["value"], 0)
            summary = read_json(output / "runs" / key / "summary.json")
            self.assertEqual(summary["timing_samples"]["semantic_call_ms"], [frame["adapter_provenance"]["timing"]["semantic_call_ms"] for frame in result["semantic_frames"]])
            self.assertTrue(result["run"]["fixture"])
            missing = read_json(output / "evaluations_no_reference" / key / "report.json")
            self.assertIsNone(missing["metrics"]["safe_precision"]["value"])
            self.assertEqual(missing["metrics"]["safe_precision"]["status"], "unavailable")
            with np.load(output / "runs" / key / "map/voxels.npz", allow_pickle=False) as actual, \
                 np.load(output / "runs/fixed_hazards/map/voxels.npz", allow_pickle=False) as fixed:
                for name in fixed.files:
                    np.testing.assert_array_equal(actual[name], fixed[name])
        self.assertEqual(runs["fixed_hazards"]["report"]["metrics"]["semantic_query_count"]["value"], 48)
        self.assertTrue(read_json(output / "comparison/comparison.json")["compatible"])

    def test_unknown_query_count_is_serialized_and_evaluated_as_unavailable(self):
        output, runs = create_cli_evidence(self.root / "cli_crash", crash_one_frame=True)
        for key in ("qwen3_5_4b", "qwen3_vl_4b"):
            row = runs[key]["semantic_frames"][1]
            self.assertEqual(row["status"], "error")
            self.assertIsNone(row["query_count"])
            self.assertTrue(row["query_count_reason"])
            metric = runs[key]["report"]["metrics"]["semantic_query_count"]
            self.assertIsNone(metric["value"])
            self.assertEqual(metric["status"], "unavailable")
            self.assertAlmostEqual(runs[key]["report"]["metrics"]["pipeline_failure_rate"]["value"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
