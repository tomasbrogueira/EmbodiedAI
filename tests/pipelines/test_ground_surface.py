"""CPU conformance through the existing SAM3 fixture processor and shared bridge."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "VLM_evaluation" / "src"))

from pipeline_common.contracts import FramePacket, GeometryFrame, validate_semantic
from pipeline_common.io import file_sha256, rgb_sha256
from pipelines.ground_surface import (
    PROMPT_CONFIG_IDS, create_adapter, development_prompt_variants,
    inspect_legacy_path_archive,
)
from traversability_hazard_segmentation.sam3_adapter import Sam3Segmenter, SegmenterUnavailable


class Processor:
    """Explicit fixture: real reader/processor control flow without Torch/models."""
    def __init__(self, output=None, *, image_error=False, reset_error=False, query_error=False):
        self.output = output or {"masks": np.zeros((0, 1, 3, 4), bool), "scores": np.zeros(0)}
        self.image_error, self.reset_error, self.query_error = image_error, reset_error, query_error
        self.prompts = []
        self.images = []

    def set_image(self, image):
        self.images.append(np.array(image))
        if self.image_error:
            raise RuntimeError("fixture_image_failure")
        return {}

    def reset_all_prompts(self, state):
        state.clear()
        if self.reset_error:
            raise RuntimeError("fixture_reset_failure")

    def set_text_prompt(self, *, state, prompt):
        self.prompts.append(prompt)
        if self.query_error:
            error = RuntimeError("fixture_partial_query_failure")
            error.partial_output = self.output
            raise error
        return self.output


class GroundSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rgb = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
        self.path = self.root / "aligned.png"
        Image.fromarray(self.rgb).save(self.path)
        self.frame = FramePacket(
            sequence_id="generic_phone_sequence", frame_id="frame_2", timestamp_ns=None,
            timestamp_provenance={"clock": "unknown", "kind": "unknown"}, rgb=self.rgb,
            image_path=self.path, encoded_file_sha256=file_sha256(self.path),
            decoded_rgb_sha256=rgb_sha256(self.rgb), processed_grid_id="lingbot_crop_grid",
            source_rgb_identity={"sha256": "source_original_identity"},
            source_to_processed={"crop_xywh": [12, 4, 8, 6], "output_hw": [3, 4]},
        )

    def adapter(self, processor=None, prompt="ground"):
        processor = processor or Processor()
        segmenter = Sam3Segmenter({"fixture": True}, processor=processor)
        adapter = create_adapter({"fixture": True, "prompt": prompt, "_segmenter": segmenter})
        self.addCleanup(adapter.close)
        return adapter, processor, segmenter

    def test_three_named_configs_preserve_exact_phrase_role_and_identity(self):
        digests = set()
        for name, phrase in (("outdoor", "ground"), ("indoor", "floor"), ("path_legacy", "Path")):
            with self.subTest(name=name):
                config = json.loads((ROOT / "configs" / "pipelines" / f"ground_surface_{name}.json").read_text())
                processor = Processor()
                segmenter = Sam3Segmenter({"fixture": True}, processor=processor)
                section = {**config["pipeline"], "fixture": True, "_segmenter": segmenter}
                adapter = create_adapter(section)
                self.addCleanup(adapter.close)
                self.assertEqual(config["pipeline"]["config_id"], PROMPT_CONFIG_IDS[phrase])
                result = adapter.observe(self.frame)
                validate_semantic(result, self.frame)
                query = result.queries[0]
                self.assertEqual((query.original_phrase, query.concept_id, query.role),
                                 (phrase, "ground_surface", "candidate_surface"))
                self.assertEqual(processor.prompts, [phrase])
                self.assertEqual(result.adapter_provenance["config_id"], config["pipeline"]["config_id"])
                self.assertEqual(result.adapter_provenance["legacy_compatibility"]["ablation"], phrase == "Path")
                digests.add(result.adapter_provenance["config_digest"])
        self.assertEqual(len(digests), 3)

    def test_real_generic_reader_receives_exact_aligned_pixels_and_two_hashes(self):
        mask = np.ones((1, 1, 3, 4), dtype=bool)
        adapter, processor, _ = self.adapter(Processor({"masks": mask, "scores": [0.83]}))
        result = adapter.observe(self.frame)
        self.assertEqual(result.status, "ok", result.error)
        np.testing.assert_array_equal(processor.images[0], self.rgb)
        self.assertEqual(result.queries[0].instances[0].mask.shape, (3, 4))
        provenance = result.adapter_provenance["input"]
        self.assertEqual(provenance["encoded_file_sha256"], file_sha256(self.path))
        self.assertEqual(provenance["decoded_rgb_sha256"], rgb_sha256(self.rgb))
        self.assertNotEqual(provenance["encoded_file_sha256"], provenance["decoded_rgb_sha256"])
        self.assertEqual(provenance["source_to_processed"], self.frame.source_to_processed)
        self.assertIn("uncalibrated", result.queries[0].instances[0].score_meaning)
        self.assertEqual(result.model_calls, {"sam3": 1})
        self.assertEqual(result.query_count, 1)
        self.assertIsNone(result.timestamp_ns)
        self.assertLessEqual(result.started_monotonic_ns, result.completed_monotonic_ns)

    def test_empty_success_differs_from_image_query_and_reset_failure(self):
        cases = [(Processor(), "ok", 1), (Processor(image_error=True), "error", 0),
                 (Processor(reset_error=True), "error", 0), (Processor(query_error=True), "error", 1)]
        for processor, expected, count in cases:
            with self.subTest(expected=expected, count=count):
                adapter, _, _ = self.adapter(processor)
                result = adapter.observe(self.frame)
                self.assertEqual(result.status, expected, result.error)
                self.assertEqual(result.queries[0].status, expected)
                self.assertEqual(result.queries[0].instances, [])
                self.assertEqual(result.query_count, count)
                self.assertEqual(result.model_calls["sam3"], 1)

    def test_partial_failed_masks_are_diagnostic_with_original_score(self):
        mask = np.ones((1, 1, 3, 4), bool)
        adapter, _, _ = self.adapter(Processor({"masks": mask, "scores": [0.91]}, query_error=True))
        result = adapter.observe(self.frame)
        self.assertEqual(result.status, "error")
        query = result.queries[0]
        self.assertEqual(query.status, "error")
        self.assertEqual(len(query.instances), 1)
        self.assertTrue(query.instances[0].metadata["diagnostic_only"])
        self.assertEqual(query.instances[0].score, 0.91)
        self.assertEqual(result.query_count, 1)

    def test_capture_failure_keeps_earlier_valid_instance_diagnostic(self):
        output = {"masks": [np.ones((3, 4), bool), np.ones((4, 3), bool)], "scores": [0.6, 0.9]}
        adapter, _, _ = self.adapter(Processor(output))
        result = adapter.observe(self.frame)
        self.assertEqual(result.status, "error")
        self.assertEqual(len(result.queries[0].instances), 1)
        self.assertTrue(result.queries[0].instances[0].metadata["diagnostic_only"])

    def test_bridge_rejects_tampered_bytes_pixels_dimensions_and_jpeg_before_model(self):
        altered = self.rgb.copy()
        altered[0, 0, 0] ^= 1
        cases = [replace(self.frame, encoded_file_sha256="0" * 64),
                 replace(self.frame, decoded_rgb_sha256="0" * 64),
                 replace(self.frame, rgb=altered), replace(self.frame, rgb=self.rgb[:, :3])]
        jpeg = self.root / "lossy.jpg"
        Image.fromarray(self.rgb).save(jpeg)
        with Image.open(jpeg) as image:
            jpeg_rgb = np.array(image.convert("RGB"))
        cases.append(replace(self.frame, image_path=jpeg, rgb=jpeg_rgb,
                             encoded_file_sha256=file_sha256(jpeg), decoded_rgb_sha256=rgb_sha256(jpeg_rgb)))
        for frame in cases:
            with self.subTest(path=str(frame.image_path), shape=frame.rgb.shape):
                adapter, processor, _ = self.adapter()
                result = adapter.observe(frame)
                self.assertEqual(result.status, "error")
                self.assertEqual(result.model_calls["sam3"], 0)
                self.assertEqual(processor.images, [])
                self.assertEqual(result.query_count, 0)

    def test_overlaps_remain_individual_and_mask_ownership_is_copied(self):
        masks = np.ones((2, 1, 3, 4), bool)
        processor = Processor({"masks": masks, "scores": [0.6, 0.9]})
        adapter, _, _ = self.adapter(processor)
        result = adapter.observe(self.frame)
        self.assertEqual(len(result.queries[0].instances), 2)
        self.assertNotEqual(result.queries[0].instances[0].observation_id,
                            result.queries[0].instances[1].observation_id)
        masks[:] = False
        self.assertTrue(result.queries[0].instances[0].mask.all())
        self.assertTrue(result.queries[0].instances[1].mask.all())

    def test_cleanup_is_idempotent_and_closed_observe_is_explicit_error(self):
        adapter, processor, segmenter = self.adapter()
        adapter.close()
        adapter.close()
        self.assertIsNone(segmenter.processor)
        result = adapter.observe(self.frame)
        self.assertEqual(result.status, "error")
        self.assertEqual(result.model_calls["sam3"], 0)
        self.assertEqual(processor.prompts, [])
        self.assertIn("closed", result.error["message"])

    def test_config_rejects_test_selection_dynamic_prompt_download_and_real_fixture(self):
        bad = [{"prompt": "path"}, {"prompt": ["ground", "floor"]},
               {"prompt": "floor", "config_id": PROMPT_CONFIG_IDS["ground"]},
               {"prompt_selection": {"frozen": False}},
               {"prompt_selection": {"frozen": True, "selection_split": "test"}},
               {"sam3": {"allow_downloads": True}},
               {"sam3": {"local_files_only": False}}, {"_segmenter": object()},
               {"schema_version": True}, {"sam3": {"fixture": 0}}, {"role": "hazard"}]
        for config in bad:
            with self.subTest(config=config), self.assertRaises(ValueError):
                create_adapter(config)

    def test_development_comparison_freezes_same_frames_and_sam_settings(self):
        sequence = {"contract_id": "semantic_mapping_v1", "schema_version": 1,
                    "split": "development", "frames": [{"frame_id": "frozen-1"}]}
        variants = development_prompt_variants({"sam3": {"confidence_threshold": 0.63}}, sequence)
        self.assertEqual([v["prompt"] for v in variants], ["ground", "floor", "Path"])
        self.assertEqual(len({v["prompt_selection"]["comparison_frame_digest"] for v in variants}), 1)
        self.assertEqual([v["sam3"]["confidence_threshold"] for v in variants], [0.63] * 3)
        sequence["split"] = "test"
        with self.assertRaisesRegex(ValueError, "development-only"):
            development_prompt_variants({}, sequence)

    def test_lazy_import_and_factory_do_not_import_models_or_torch(self):
        script = (f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r}); "
                  "from pipelines.ground_surface import create_adapter; "
                  "adapter=create_adapter({}); assert 'torch' not in sys.modules; "
                  "assert 'sam3' not in sys.modules; adapter.close()")
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_malformed_backend_counts_and_frame_failure_remain_explicit(self):
        class Backend:
            metadata = {"fixture": True, "fixture_identity": "ground_surface_test_stub_v1"}
            def __init__(self, raw): self.raw = raw
            def segment_image(self, *args): return self.raw
            def close(self): pass
        mask = np.ones((3, 4), bool)
        valid = {"phrase": "ground", "status": "ok", "error_code": None,
                 "masks": [mask], "scores": [0.9], "sam_query_count": 1}
        for frame_count, query_count in ((None, 1), (1, 0), (0, 0), (True, 1)):
            raw = {"frame": {"status": "ok", "sam_query_count": frame_count},
                   "queries": [{**valid, "sam_query_count": query_count}]}
            adapter = create_adapter({"fixture": True, "_segmenter": Backend(raw)})
            with self.subTest(frame_count=frame_count, query_count=query_count):
                result = adapter.observe(self.frame)
                self.assertEqual(result.status, "error")
                self.assertEqual(result.queries[0].status, "error")
                self.assertIsNone(result.query_count)
                self.assertIsNotNone(result.query_count_reason)
            adapter.close()
        raw = {"frame": {"status": "error", "error_code": "image_failed", "sam_query_count": 1},
               "queries": [valid]}
        adapter = create_adapter({"fixture": True, "_segmenter": Backend(raw)})
        result = adapter.observe(self.frame)
        self.assertEqual(result.status, "error")
        self.assertTrue(result.queries[0].instances[0].metadata["diagnostic_only"])
        self.assertEqual(result.adapter_provenance["timing"]["loading_ms"], 0.0)
        self.assertIsNotNone(result.adapter_provenance["timing"]["sam_call_ms"])
        adapter.close()

    def test_backend_crash_preserves_unknown_count_and_no_usable_evidence(self):
        class CrashBackend:
            metadata = {"fixture": True, "fixture_identity": "ground_surface_crash_fixture_v1"}
            def segment_image(self, *args): raise RuntimeError("crash_after_unknown_work")
            def close(self): pass
        adapter = create_adapter({"fixture": True, "_segmenter": CrashBackend()})
        self.addCleanup(adapter.close)
        result = adapter.observe(self.frame)
        self.assertEqual(result.status, "error")
        self.assertEqual(result.model_calls["sam3"], 1)
        self.assertIsNone(result.query_count)
        self.assertIn("trustworthy", result.query_count_reason)
        self.assertEqual(result.queries[0].instances, [])

    def test_unavailable_checkpoint_is_explicit_without_model_or_fixture_fallback(self):
        from unittest.mock import Mock
        module = Mock()
        module.load_segmenter.side_effect = SegmenterUnavailable("checkpoint_unavailable", "fixture test missing local weight")
        adapter = create_adapter({"sam3": {"enable_model_loading": True}})
        self.addCleanup(adapter.close)
        with patch("pipelines.ground_surface._sam_module", return_value=module):
            result = adapter.observe(self.frame)
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error["code"], "checkpoint_unavailable")
        self.assertEqual(result.model_calls["sam3"], 0)
        self.assertEqual(result.query_count, 0)
        self.assertFalse(result.adapter_provenance["fixture"])
        self.assertEqual(result.queries[0].instances, [])

    def geometry_frame(self):
        y, x = np.mgrid[:3, :4]
        points = np.stack(((x - 1.5) / 5., (y - 1.) / 5., np.full((3, 4), 2.)), axis=-1)
        geometry = GeometryFrame(points=points, depth=np.full((3, 4), 2.),
            validity=np.ones((3, 4), bool), confidence=np.full((3, 4), 2.),
            intrinsics=np.array([[10., 0., 1.5], [0., 10., 1.], [0., 0., 1.]]),
            world_to_camera=np.eye(4)[:3], processed_grid_id=self.frame.processed_grid_id,
            geometry_fingerprint="fixture_geometry_v1", units="metres", up=(0., 0., 1.),
            scale={"verified": True, "fixture": True, "source": "synthetic"})
        return replace(self.frame, geometry=geometry)

    def test_common_fuser_caps_overlaps_pixels_replay_and_distinct_frames(self):
        from pipeline_common.fusion import VoxelFuser
        frame = self.geometry_frame()
        masks = np.ones((2, 1, 3, 4), bool)
        adapter, _, _ = self.adapter(Processor({"masks": masks, "scores": [0.6, 0.9]}))
        observation = adapter.observe(frame)
        fuser = VoxelFuser({"voxel_size": 10., "origin": [-5., -5., 0.]})
        fuser.add_semantics(observation, frame)
        fuser.add_semantics(observation, frame)
        voxels, evidence, concepts, journal, metadata = fuser.export()
        np.testing.assert_allclose(evidence["positive_weight"], [1.8])
        np.testing.assert_allclose(evidence["observed_weight"], [2.])
        np.testing.assert_allclose(evidence["evidence_score"], [0.9])
        np.testing.assert_array_equal(evidence["frame_support"], [1])
        np.testing.assert_array_equal(voxels["frame_support"], [1])
        self.assertEqual(concepts[0]["role"], "candidate_surface")
        self.assertEqual(concepts[0]["original_phrases"], ["ground"])
        self.assertEqual(metadata["valid_surface_pixels"], 12)
        second = replace(frame, frame_id="second_independent_frame")
        fuser.add_semantics(adapter.observe(second), second)
        self.assertEqual(fuser.export()[1]["frame_support"].tolist(), [2])
        self.assertEqual(sum(row["kind"] == "semantic" for row in journal), 1)

    def test_common_fusion_distinguishes_empty_coverage_and_failed_partial_masks(self):
        from pipeline_common.fusion import VoxelFuser
        frame = self.geometry_frame()
        for processor, expected_rows in ((Processor(), 1),
            (Processor({"masks": np.ones((1, 1, 3, 4), bool), "scores": [.99]}, query_error=True), 0)):
            adapter, _, _ = self.adapter(processor)
            observation = adapter.observe(frame)
            fuser = VoxelFuser({"voxel_size": 10., "origin": [-5., -5., 0.]})
            fuser.add_semantics(observation, frame)
            voxels, evidence, _, journal, _ = fuser.export()
            self.assertEqual(len(voxels["voxel_indices"]), 1)
            self.assertEqual(len(evidence["voxel_row"]), expected_rows)
            self.assertEqual(evidence["positive_weight"].sum(), 0.)
            if expected_rows:
                self.assertEqual(evidence["observed_weight"].tolist(), [2.])
            else:
                self.assertFalse(next(row for row in journal if row["kind"] == "semantic_query")["fused"])

    def test_surface_prior_cannot_override_common_steep_slope_rejection(self):
        from pipeline_common.planning import BLOCKED, build_costmap
        x, y = np.meshgrid(np.linspace(.05, .95, 12), np.linspace(.05, .95, 12))
        points = np.column_stack((x.ravel(), y.ravel(), (0.8 * x).ravel()))
        config = {"planning": {"origin": [0., 0.], "resolution": 1., "shape": [1, 1], "support_height": 0.},
            "robot": {"version": "synthetic_ground_surface_robot_v1", "fixture": True,
                "footprint_radius": 0., "height": .5, "clearance": 0., "max_slope_degrees": 15.,
                "max_step": .1, "max_roughness": .01, "min_support_points": 3,
                "unknown_rule": "blocked", "semantic_costs": {"ground_surface": {"cost": -100., "blocked": False}}}}
        arrays, meta = build_costmap(points, config, units="metres", up=[0., 0., 1.],
            scale={"verified": True, "fixture": True},
            semantic_points={"ground_surface": {"role": "candidate_surface", "points": points,
                                                 "evidence_score": np.ones(len(points))}})
        self.assertEqual(meta["availability"], "available", meta["reason"])
        self.assertEqual(meta["physical_violation_cells"], 1)
        self.assertEqual(arrays["decision_state"].tolist(), [[BLOCKED]])
        self.assertTrue(np.isinf(arrays["costs"][0, 0]))

    def test_actual_runner_and_evaluator_with_named_adapter_fixture(self):
        from pipeline_common.sequence import prepare
        from pipeline_common.io import write_json
        from run_pipeline import main as run_main
        from evaluate_pipeline import main as evaluate_main
        sources = self.root / "source_frames"
        sources.mkdir()
        for i in range(2):
            Image.fromarray(self.rgb).save(sources / f"frame_{i}.png")
        sequence_dir = self.root / "sequence"
        prepare(sources, {"contract_id": "semantic_mapping_v1", "schema_version": 1,
            "sequence_id": "surface_adapter_integration_fixture", "split": "development",
            "fixture": True, "max_frames": 2}, sequence_dir)
        evaluations = []
        for name, phrase in (("outdoor", "ground"), ("indoor", "floor"), ("path_legacy", "Path")):
            config = json.loads((ROOT / "configs" / "pipelines" / f"ground_surface_{name}.json").read_text())
            config["fixture"] = True
            config["goals"] = [{"request_id": "missing_physical_profile", "start": [0., 0., 0.], "goal": [1., 1., 0.]}]
            config_path = self.root / f"{name}.json"
            write_json(config_path, config)
            processor = Processor({"masks": np.ones((2, 1, 3, 4), bool), "scores": [.6, .9]})
            segmenter = Sam3Segmenter({"fixture": True}, processor=processor)
            def factory(pipeline, section, sequence):
                self.assertEqual(pipeline, "ground_surface")
                self.assertTrue(sequence["fixture"])
                from pipelines import create_adapter as registry_factory
                return registry_factory(pipeline, {**section, "_segmenter": segmenter})
            output = self.root / f"{name}_run"
            with self.subTest(name=name), patch("pipeline_common.fixture_adapters.create_fixture_adapter", side_effect=factory):
                code = run_main(["--pipeline", "ground_surface", "--sequence", str(sequence_dir / "sequence.json"),
                    "--config", str(config_path), "--output", str(output), "--fixture"])
                self.assertEqual(code, 0)
            run_identity = json.loads((output / "run.json").read_text())
            self.assertEqual(run_identity["status"], "complete")
            self.assertTrue(run_identity["fixture"])
            concepts = json.loads((output / "map/concepts.json").read_text())
            self.assertEqual(concepts[0]["concept_id"], "ground_surface")
            self.assertEqual(concepts[0]["original_phrases"], [phrase])
            rows = [json.loads(line) for line in (output / "semantics/frames.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual([row["queries"][0]["original_phrase"] for row in rows], [phrase] * 2)
            self.assertIsNone(segmenter.processor)
            for required in ("frames.jsonl", "config.resolved.json", "geometry/manifest.json",
                "map/manifest.json", "map/voxels.npz", "map/semantic_evidence.npz",
                "map/contributions.jsonl", "planning/costmap.npz", "planning/plans.jsonl", "events.jsonl", "summary.json"):
                self.assertTrue((output / required).is_file(), required)
            with np.load(output / "map/semantic_evidence.npz", allow_pickle=False) as saved:
                self.assertFalse(any(saved[key].dtype.hasobject for key in saved.files))
                self.assertTrue(np.all(saved["frame_support"] == 2))
                self.assertTrue(np.allclose(saved["evidence_score"], .9))
            plans = [json.loads(line) for line in (output / "planning/plans.jsonl").read_text().splitlines()]
            self.assertEqual(plans[0]["status"], "blocked_inputs")
            evaluation = self.root / f"{name}_evaluation"
            self.assertEqual(evaluate_main(["--run", str(output), "--output", str(evaluation)]), 0)
            self.assertTrue((evaluation / "report.json").is_file())
            self.assertTrue((evaluation / "metrics.csv").is_file())
            evaluations.append(evaluation)
        from pipeline_common.evaluation import compare, fingerprint
        comparison = compare(evaluations, self.root / "comparison")
        self.assertTrue(comparison["compatible"])
        self.assertIsNone(comparison["ranking"])
        altered_report = json.loads((evaluations[0] / "report.json").read_text())
        altered_report["comparability"]["identities"]["robot"] = {"version": "other_robot", "fixture": True}
        identities = altered_report["comparability"]["identities"]
        fingerprints = {key: fingerprint(value) for key, value in identities.items()}
        altered_report["comparability"].update(fingerprints=fingerprints, combined=fingerprint(fingerprints))
        altered = self.root / "incompatible_evaluation"
        write_json(altered / "report.json", altered_report)
        rejected = compare([evaluations[0], altered], self.root / "comparison_rejected")
        self.assertFalse(rejected["compatible"])
        self.assertTrue(any(diff["identity"] == "robot" for diff in rejected["differences"]))


class LegacyCompatibilityTests(unittest.TestCase):
    def test_legacy_path_readability_and_fingerprint_prompt_rejection(self):
        from path_mapping import runner
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            geometry, sources, scores = runner.fixture()
            gp, sp = root / "geometry.npz", root / "sam_scores.npz"
            runner.save_geometry(gp, geometry, sources, {"fixture": True})
            runner.save_scores(sp, scores, geometry, "Path", {"fixture": True})
            original = (gp.read_bytes(), sp.read_bytes())
            audit = inspect_legacy_path_archive(gp, sp)
            self.assertEqual(audit["legacy_pipeline_id"], "sam3_lingbot_path_v1")
            self.assertFalse(audit["usable_for_new_pipeline_fusion"])
            self.assertFalse(audit["original_instances_available"])
            self.assertEqual(audit["scores_archive_sha256"], hashlib.sha256(original[1]).hexdigest())
            self.assertEqual((gp.read_bytes(), sp.read_bytes()), original)
            with self.assertRaisesRegex(ValueError, "requested geometry"):
                inspect_legacy_path_archive(gp, sp, expected_geometry_fingerprint="0" * 64)
            for phrase in ("ground", "floor", "path"):
                runner.save_scores(sp, scores, geometry, phrase, {})
                with self.assertRaisesRegex(ValueError, "prompt"):
                    inspect_legacy_path_archive(gp, sp)
            runner.save_scores(sp, scores, geometry, "Path", {})
            for key in runner.GEOMETRY_KEYS:
                changed = {k: v.copy() for k, v in geometry.items()}
                changed[key].reshape(-1)[0] += 1
                runner.save_geometry(gp, changed, sources, {})
                with self.subTest(key=key), self.assertRaises(ValueError):
                    inspect_legacy_path_archive(gp, sp)


if __name__ == "__main__":
    unittest.main()
