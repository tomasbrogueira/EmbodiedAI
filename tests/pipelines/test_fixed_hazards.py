"""CPU conformance through real contracts/bridge and a mocked SAM processor."""
import copy
from dataclasses import replace
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from pipeline_common.contracts import FramePacket, GeometryFrame, validate_semantic
from pipeline_common.io import file_sha256, rgb_sha256
from pipelines.fixed_hazards import CANONICAL_PROMPTS, create_adapter
from traversability_hazard_segmentation import load_segmenter
from traversability_hazard_segmentation.common import validate_frame


class Processor:
    """Mimic output state mutation, overlapping concepts and partial failures."""
    def __init__(self, *, failed=(), resets_failed=(), image_failed=False, empty=False):
        self.failed, self.resets_failed = set(failed), set(resets_failed)
        self.image_failed, self.empty = image_failed, empty
        self.images, self.prompts, self.reset_count = [], [], 0
        self.close_calls = 0

    def set_image(self, image):
        self.images.append(np.array(image))
        if self.image_failed:
            raise RuntimeError("fixture image encoding failed")
        self.shape = (image.height, image.width)
        return {"image_features": object()}

    def reset_all_prompts(self, state):
        phrase = CANONICAL_PROMPTS[self.reset_count % len(CANONICAL_PROMPTS)]
        self.reset_count += 1
        for key in ("masks", "scores"):
            state.pop(key, None)
        if phrase in self.resets_failed:
            raise RuntimeError("fixture reset failed")

    def set_text_prompt(self, *, state, prompt):
        self.prompts.append(prompt)
        masks = []
        if not self.empty and prompt in {"water", "person", "log"}:
            mask = np.zeros(self.shape, dtype=bool)
            mask[1, 1] = True  # overlapping concepts must survive separately
            mask[1, {"water": 2, "person": 3, "log": 4}[prompt]] = True
            masks.append(mask)
            if prompt == "person":
                second = np.zeros(self.shape, dtype=bool)
                second[2, 3] = True
                masks.append(second)
        state.update(masks=np.asarray(masks, dtype=bool).reshape((-1, *self.shape)),
            scores=np.asarray([0.8] * len(masks)))
        if prompt in self.failed:
            raise RuntimeError("fixture failed after valid partial mask")
        return state

    def close(self):
        self.close_calls += 1


class FixedHazardsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        rgb = np.arange(4 * 6 * 3, dtype=np.uint8).reshape((4, 6, 3))
        image_path = self.root / "processed.png"
        Image.fromarray(rgb).save(image_path)
        self.frame = FramePacket(sequence_id="generic-phone-video", frame_id="frame-0042",
            timestamp_ns=None, timestamp_provenance={"status": "unknown", "clock": None},
            rgb=rgb, image_path=image_path, encoded_file_sha256=file_sha256(image_path),
            decoded_rgb_sha256=rgb_sha256(rgb), processed_grid_id="lingbot-grid-42",
            source_rgb_identity={"frame_id": "original-0042", "source": "generic_video"},
            source_to_processed={"kind": "crop_resize", "source_width": 12,
                "source_height": 10, "crop_xyxy": [0, 1, 12, 9], "processed_width": 6,
                "processed_height": 4})

    def make_adapter(self, processor=None):
        processor = processor or Processor()
        segmenter = load_segmenter({"fixture": True, "processor": processor})
        adapter = create_adapter({"fixture": True, "_segmenter": segmenter})
        self.addCleanup(adapter.close)
        return adapter, segmenter, processor

    def geometry_packet(self, *, invalid=None):
        y, x = np.mgrid[:4, :6]
        points = np.stack((x * .2 + .01, y * .2 + .01, np.ones((4, 6))), axis=-1)
        valid = np.ones((4, 6), dtype=bool)
        if invalid is not None: valid[invalid] = False
        geometry = GeometryFrame(points=points, depth=np.ones((4, 6)),
            validity=valid, confidence=np.ones((4, 6)), intrinsics=np.eye(3),
            world_to_camera=np.eye(4), processed_grid_id=self.frame.processed_grid_id,
            geometry_fingerprint="synthetic-geometry-42", units="metres", up=(0., 0., 1.),
            scale={"verified": True, "fixture": True, "source": "independent_synthetic_geometry"})
        return replace(self.frame, geometry=geometry)

    def test_shared_fuser_keeps_sixteen_concepts_and_source_traceability(self):
        from pipeline_common.fusion import VoxelFuser
        adapter, _, _ = self.make_adapter()
        frame = self.geometry_packet()
        observation = adapter.observe(frame)
        fuser = VoxelFuser({"voxel_size": .1})
        fuser.add_semantics(observation, frame)
        voxels, evidence, concepts, journal, _ = fuser.export()
        concept_rows = {row["concept_id"]: i for i, row in enumerate(concepts)}
        self.assertEqual(set(concept_rows), set(CANONICAL_PROMPTS))
        overlap = np.floor(frame.geometry.points[1, 1] / .1).astype(np.int64)
        overlap_row = np.flatnonzero(np.all(voxels["voxel_indices"] == overlap, axis=1))[0]
        for concept in ("water", "person", "log"):
            selected = ((evidence["concept_row"] == concept_rows[concept])
                & (evidence["voxel_row"] == overlap_row))
            self.assertEqual(selected.sum(), 1)
            self.assertGreater(evidence["positive_weight"][selected][0], 0)
            contribution = next(row for row in journal if row["kind"] == "semantic" and row["concept_id"] == concept)
            self.assertEqual(contribution["frame_id"], frame.frame_id)
            self.assertEqual(contribution["original_phrases"], [concept])
            query = next(q for q in observation.queries if q.concept_id == concept)
            self.assertEqual(contribution["query_ids"], [query.query_id])
            self.assertEqual(contribution["pose_revision"], frame.geometry.pose_revision)
        # Replaying exactly the same observations cannot add support or weight.
        fuser.add_semantics(observation, frame)
        again = fuser.export()
        for key in evidence: np.testing.assert_array_equal(evidence[key], again[1][key])

    def test_shared_fuser_excludes_failed_partial_masks_and_invalid_geometry(self):
        from pipeline_common.fusion import VoxelFuser
        adapter, _, _ = self.make_adapter(Processor(failed={"person"}))
        frame = self.geometry_packet(invalid=(1, 2))
        observation = adapter.observe(frame)
        fuser = VoxelFuser({"voxel_size": .1})
        fuser.add_semantics(observation, frame)
        voxels, evidence, concepts, journal, _ = fuser.export()
        rows = {row["concept_id"]: i for i, row in enumerate(concepts)}
        self.assertFalse(np.any(evidence["concept_row"] == rows["person"]))
        person = next(row for row in journal if row["kind"] == "semantic_query" and row["concept_id"] == "person")
        self.assertFalse(person["fused"])
        self.assertEqual(len(person["observation_ids"]), 2)
        invalid_index = np.floor(frame.geometry.points[1, 2] / .1).astype(np.int64)
        self.assertFalse(np.any(np.all(voxels["voxel_indices"] == invalid_index, axis=1)))
        self.assertGreater(np.count_nonzero(evidence["positive_weight"] > 0), 0)
        no_geometry = VoxelFuser()
        raw = adapter.observe(self.frame)
        no_geometry.add_semantics(raw, self.frame)
        self.assertEqual(len(no_geometry.export()[1]["positive_weight"]), 0)
        self.assertEqual(len(raw.queries), 16)

    def test_shared_worker_consumes_actual_adapter_on_immutable_inflight_grid(self):
        from pipeline_common.scheduling import LatestPendingWorker
        adapter, _, _ = self.make_adapter()
        frame = self.geometry_packet()
        worker = LatestPendingWorker(adapter)
        self.addCleanup(worker.close)
        worker.submit(frame)
        records = list(worker.drain())
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertNotIn("error", record)
        self.assertFalse(record["frame"].rgb.flags.writeable)
        self.assertFalse(record["frame"].geometry.points.flags.writeable)
        self.assertEqual(record["result"].query_count, 16)
        self.assertEqual(record["result"].geometry_fingerprint, frame.geometry.geometry_fingerprint)
        self.assertEqual(record["result"].processed_grid_id, frame.processed_grid_id)
        self.assertIsNone(record["mapped_capture_monotonic_ns"])

    def test_shared_geometry_overrides_semantic_surface_reward_and_hazard_prior(self):
        from pipeline_common.planning import BLOCKED, build_costmap
        # Independent metric synthetic floor, plus an observed low shell in the
        # robot's body envelope. These values are never a real robot profile.
        cloud = np.array([[x, y, 0.] for y in (.2, .5, .8) for x in (.2, .5, .8)]
            + [[.5, .5, .1]])
        config = {"robot": {"version": "independent_fixture_robot_v1", "fixture": True,
            "unknown_rule": "blocked", "semantic_costs": {"ground_surface": -100., "water": 0.},
            "footprint_radius": 0., "height": .4, "clearance": 0., "max_slope_degrees": 20.,
            "max_step": .05, "max_roughness": .001, "min_support_points": 3},
            "planning": {"origin": [0., 0.], "shape": [1, 1], "resolution": 1., "support_height": 0.}}
        arrays, metadata = build_costmap(cloud, config, units="metres", up=(0., 0., 1.),
            scale={"verified": True, "fixture": True}, semantic_points={
                "ground_surface": {"points": cloud[:9], "role": "candidate_surface"},
                "water": {"points": cloud[:9], "role": "hazard"}})
        self.assertEqual(metadata["availability"], "available")
        self.assertEqual(arrays["decision_state"][0, 0], BLOCKED)
        self.assertEqual(arrays["geometry_state"][0, 0], BLOCKED)
        self.assertTrue(np.isinf(arrays["costs"][0, 0]))
        self.assertGreater(metadata["physical_violation_cells"], 0)

    def test_inventory_one_call_and_independent_overlapping_instances(self):
        adapter, segmenter, processor = self.make_adapter()
        with patch.object(segmenter, "segment_image", wraps=segmenter.segment_image) as call:
            observation = adapter.observe(self.frame)
        self.assertEqual(call.call_count, 1)
        model_frame, prompts, root = call.call_args.args
        self.assertEqual(prompts, list(CANONICAL_PROMPTS))
        self.assertEqual(processor.prompts, list(CANONICAL_PROMPTS))
        self.assertEqual(model_frame["input_contract_id"], "semantic_mapping_v1")
        self.assertNotIn("source", model_frame)
        self.assertNotIn("reference", model_frame)
        self.assertEqual(root, self.root)
        self.assertEqual(len(processor.images), 1)
        self.assertEqual(processor.reset_count, 16)
        np.testing.assert_array_equal(processor.images[0], self.frame.rgb)
        self.assertEqual([q.original_phrase for q in observation.queries], list(CANONICAL_PROMPTS))
        self.assertEqual([q.concept_id for q in observation.queries], list(CANONICAL_PROMPTS))
        self.assertTrue(all(q.role == "hazard" for q in observation.queries))
        selected = {q.concept_id: q for q in observation.queries}
        self.assertEqual(len(selected["person"].instances), 2)
        for concept in ("water", "person", "log"):
            self.assertTrue(selected[concept].instances[0].mask[1, 1])
            self.assertEqual(selected[concept].instances[0].mask.shape, (4, 6))
        selected["water"].instances[0].mask[:] = False
        self.assertTrue(selected["person"].instances[0].mask[1, 1])
        self.assertEqual(observation.status, "ok")
        self.assertEqual(observation.query_count, 16)
        self.assertEqual(sum(observation.model_calls.values()), 1)
        validate_semantic(observation, self.frame)

    def test_partial_failure_keeps_successful_queries_and_diagnostic_masks(self):
        adapter, _, processor = self.make_adapter(Processor(failed={"person"}, resets_failed={"tree"}))
        observation = adapter.observe(self.frame)
        selected = {q.concept_id: q for q in observation.queries}
        self.assertEqual(observation.status, "partial")
        self.assertEqual(observation.query_count, 15)
        self.assertEqual(selected["tree"].score_metadata["sam_query_count"], 0)
        self.assertEqual(selected["person"].score_metadata["sam_query_count"], 1)
        self.assertEqual(selected["person"].status, "error")
        self.assertEqual(len(selected["person"].instances), 2)
        self.assertTrue(all(not i.metadata["fusion_eligible"] for i in selected["person"].instances))
        self.assertEqual(selected["water"].status, "ok")
        self.assertEqual(selected["log"].status, "ok")
        self.assertIn("dog", processor.prompts)
        self.assertEqual(observation.adapter_provenance["counts"]["failed_queries"], 2)

    def test_successful_empties_and_image_failure_are_distinct(self):
        adapter, _, _ = self.make_adapter(Processor(empty=True))
        empty = adapter.observe(self.frame)
        self.assertEqual(empty.status, "ok")
        self.assertEqual(empty.query_count, 16)
        self.assertTrue(all(q.status == "ok" and not q.instances for q in empty.queries))
        failed_adapter, _, processor = self.make_adapter(Processor(image_failed=True))
        failed = failed_adapter.observe(self.frame)
        self.assertEqual(failed.status, "error")
        self.assertEqual(failed.query_count, 0)
        self.assertEqual(len(failed.queries), 16)
        self.assertTrue(all(q.status == "error" and not q.instances for q in failed.queries))
        self.assertEqual(processor.prompts, [])
        self.assertIn("image encoding failed", failed.error)

    def test_all_failed_queries_are_error(self):
        adapter, _, _ = self.make_adapter(Processor(failed=CANONICAL_PROMPTS))
        failed = adapter.observe(self.frame)
        self.assertEqual(failed.status, "error")
        self.assertEqual(failed.query_count, 16)

    def test_bridge_rejects_hash_pixels_and_file_format_before_model(self):
        adapter, segmenter, processor = self.make_adapter()
        changed_rgb = self.frame.rgb.copy()
        changed_rgb[0, 0, 0] ^= 1
        bad_packets = [replace(self.frame, encoded_file_sha256="0" * 64),
            replace(self.frame, decoded_rgb_sha256="0" * 64),
            replace(self.frame, rgb=changed_rgb)]
        with patch.object(segmenter, "segment_image", wraps=segmenter.segment_image) as call:
            for packet in bad_packets:
                with self.subTest(packet=packet.encoded_file_sha256):
                    with self.assertRaises(ValueError):
                        adapter.observe(packet)
            self.assertEqual(call.call_count, 0)
        jpg = self.root / "processed.jpg"
        Image.fromarray(self.frame.rgb).save(jpg)
        with self.assertRaisesRegex(ValueError, "lossless"):
            adapter.observe(replace(self.frame, image_path=jpg, encoded_file_sha256=file_sha256(jpg)))
        self.assertEqual(processor.images, [])

    def test_post_call_packet_mutation_is_rejected(self):
        adapter, segmenter, _ = self.make_adapter()
        original = segmenter.segment_image
        def mutate(*args):
            result = original(*args)
            self.frame.rgb[0, 0, 0] ^= 1
            return result
        with patch.object(segmenter, "segment_image", side_effect=mutate):
            with self.assertRaises(ValueError):
                adapter.observe(self.frame)

    def test_backend_metadata_mutation_does_not_modify_packet(self):
        adapter, segmenter, _ = self.make_adapter()
        original = segmenter.segment_image
        before = copy.deepcopy(self.frame.source_rgb_identity)
        def mutate(*args):
            result = original(*args)
            args[0]["source_rgb_identity"]["source"] = "forged"
            return result
        with patch.object(segmenter, "segment_image", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "provenance changed"):
                adapter.observe(self.frame)
        self.assertEqual(self.frame.source_rgb_identity, before)

    def test_partial_status_uses_query_outcomes_and_provenance_cannot_contradict(self):
        adapter, segmenter, _ = self.make_adapter(Processor(failed={"person"}))
        original = segmenter.segment_image
        def different_aggregate(*args):
            result = original(*args)
            result["frame"]["error_code"] = "opaque_aggregate_query_failure"
            return result
        with patch.object(segmenter, "segment_image", side_effect=different_aggregate):
            self.assertEqual(adapter.observe(self.frame).status, "partial")
        for key, value in (("fixture", False), ("policy_hash", "0" * 64),
            ("checkpoint_sha256", "0" * 64)):
            def contradict(*args):
                result = original(*args)
                result["settings"][key] = value
                return result
            with self.subTest(key=key):
                with patch.object(segmenter, "segment_image", side_effect=contradict):
                    with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                        adapter.observe(self.frame)

    def test_actual_checkpoint_policy_grid_hashes_and_clocks_are_retained(self):
        adapter, segmenter, _ = self.make_adapter()
        observation = adapter.observe(self.frame)
        provenance = observation.adapter_provenance
        inputs, actual, policy = provenance["input"], provenance["sam_settings_actual"], provenance["policy"]
        self.assertEqual(inputs["encoded_file_sha256"], self.frame.encoded_file_sha256)
        self.assertEqual(inputs["decoded_rgb_sha256"], self.frame.decoded_rgb_sha256)
        self.assertNotEqual(inputs["encoded_file_sha256"], inputs["decoded_rgb_sha256"])
        self.assertEqual(inputs["source_to_processed"], self.frame.source_to_processed)
        self.assertEqual(actual["checkpoint_sha256"], segmenter.metadata["checkpoint_sha256"])
        self.assertEqual(actual["policy_hash"], policy["policy_hash"])
        self.assertEqual(actual["policy_file_sha256"], policy["policy_file_sha256"])
        self.assertEqual(actual["fixture_identity"], segmenter.fixture_identity)
        self.assertEqual(actual["mask_comparison"], "strictly_greater_than")
        self.assertIsNone(observation.timestamp_ns)
        self.assertGreaterEqual(observation.completed_monotonic_ns, observation.started_monotonic_ns)

    def test_wrong_grid_mask_and_actual_count_are_rejected(self):
        adapter, segmenter, _ = self.make_adapter()
        result = segmenter.segment_image(
            importlib.import_module("pipeline_common.input_bridge").to_model_input(self.frame)[0],
            list(CANONICAL_PROMPTS), self.root)
        for kind in ("mask", "counts", "phrase", "frame_error", "query_zero"):
            malformed = copy.deepcopy(result)
            if kind == "mask": malformed["queries"][2]["masks"] = [np.ones((6, 4), dtype=bool)]
            if kind == "counts": malformed["frame"]["sam_query_count"] = 99
            if kind == "phrase": malformed["queries"][2]["phrase"] = "puddle"
            if kind == "frame_error": malformed["frame"]["error_code"] = "unexpected_frame_error"
            if kind == "query_zero":
                malformed["queries"][2]["sam_query_count"] = 0
                malformed["frame"]["sam_query_count"] = 15
            with self.subTest(kind=kind):
                with patch.object(segmenter, "segment_image", return_value=malformed):
                    with self.assertRaises(ValueError): adapter.observe(self.frame)

    def test_generic_runtime_does_not_relax_legacy_benchmark_validation(self):
        adapter, _, _ = self.make_adapter()
        self.assertEqual(adapter.observe(self.frame).status, "ok")
        legacy = {"frame_id": "phone-42", "source": "generic_video", "scene_id": "phone",
            "sequence_id": self.frame.sequence_id, "timestamp_s": None, "split": "test",
            "image_path": "processed.png", "width": 6, "height": 4,
            "image_sha256": self.frame.encoded_file_sha256}
        with self.assertRaisesRegex(ValueError, "source"):
            validate_frame(legacy, self.root)

    def test_policy_inventory_and_fixture_injection_are_strict(self):
        policy_path = Path(__file__).resolve().parents[2] / "VLM_evaluation/configs/hazards/policy.json"
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        policy["canonical_prompts"][0], policy["canonical_prompts"][1] = policy["canonical_prompts"][1], policy["canonical_prompts"][0]
        changed = self.root / "changed_policy.json"
        changed.write_text(json.dumps(policy), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "ordered"):
            create_adapter({"policy_path": str(changed)})
        with self.assertRaisesRegex(ValueError, "fixture"):
            create_adapter({"_segmenter": object()})
        with self.assertRaisesRegex(ValueError, "provenance"):
            create_adapter({"fixture": True, "_segmenter": object()})

    def explicit_policy(self):
        specs = [
            {"phrase": "tiled walkway", "concept_id": "ground_surface", "role": "candidate_surface"},
            {"phrase": "water", "concept_id": "water", "role": "hazard"},
            {"phrase": "garden hose", "concept_id": "hose", "role": "hazard"},
        ]
        return {"task_id": "hazard_prompt_v1", "schema_version": 1,
            "policy_id": "test_vocabulary_v1", "vocabulary_mode": "explicit_concepts_v1",
            "canonical_prompts": [spec["phrase"] for spec in specs], "concepts": specs,
            "aliases": {spec["phrase"]: [spec["phrase"]] for spec in specs},
            "vague_unusable_phrases": ["obstacle"]}

    def write_explicit_policy(self, policy):
        path = self.root / "explicit_policy.json"
        path.write_text(json.dumps(policy), encoding="utf-8")
        return {"vocabulary_mode": "explicit_concepts_v1", "policy_path": str(path)}

    def test_generic_fixed_vocabulary_preserves_phrases_roles_and_partial_failures(self):
        policy = self.explicit_policy()
        options = self.write_explicit_policy(policy)
        processor = Processor(failed={"water"})
        segmenter = load_segmenter({"fixture": True, "processor": processor,
            "policy_path": options["policy_path"], "policy_id": policy["policy_id"]})
        adapter = create_adapter({**options, "fixture": True, "_segmenter": segmenter})
        self.addCleanup(adapter.close)
        observation = adapter.observe(self.frame)
        validate_semantic(observation, self.frame)
        self.assertEqual(observation.status, "partial")
        self.assertEqual(processor.prompts, policy["canonical_prompts"])
        self.assertEqual(observation.query_count, 3)
        self.assertEqual(observation.adapter_provenance["counts"]["requested_queries"], 3)
        self.assertEqual(observation.adapter_provenance["policy"]["concepts"], policy["concepts"])
        self.assertEqual([(q.original_phrase, q.concept_id, q.role) for q in observation.queries],
            [(s["phrase"], s["concept_id"], s["role"]) for s in policy["concepts"]])
        failed = observation.queries[1]
        self.assertEqual(failed.status, "error")
        self.assertTrue(failed.instances)
        self.assertTrue(all(i.metadata["diagnostic_partial"] and not i.metadata["fusion_eligible"]
                            for i in failed.instances))
        # A crash has unknown executions and retains every configured role.
        with patch.object(segmenter, "segment_image", side_effect=RuntimeError("fixture crash")):
            crash = adapter.observe(self.frame)
        self.assertIsNone(crash.query_count)
        self.assertTrue(crash.query_count_reason)
        self.assertEqual([(q.concept_id, q.role) for q in crash.queries],
                         [(q.concept_id, q.role) for q in observation.queries])
        self.assertEqual(crash.adapter_provenance["counts"]["failed_queries"], 3)

    def test_custom_vocabulary_requires_opt_in_roles_unique_phrases_and_pinned_hashes(self):
        policy = self.explicit_policy()
        options = self.write_explicit_policy(policy)
        with self.assertRaisesRegex(ValueError, "ordered"):
            create_adapter({"policy_path": options["policy_path"]})
        with self.assertRaisesRegex(ValueError, "policy_path"):
            create_adapter({"vocabulary_mode": "explicit_concepts_v1"})
        with self.assertRaisesRegex(ValueError, "prefilter|reorder"):
            create_adapter({**options, "prompts": list(reversed(policy["canonical_prompts"]))})
        with self.assertRaisesRegex(ValueError, "policy_hash mismatch"):
            create_adapter({**options, "policy_hash": "0" * 64})
        variants = []
        no_role = copy.deepcopy(policy)
        del no_role["concepts"][0]["role"]
        variants.append(no_role)
        hazard_as_surface = copy.deepcopy(policy)
        hazard_as_surface["concepts"][1]["role"] = "candidate_surface"
        variants.append(hazard_as_surface)
        conflicting_roles = copy.deepcopy(policy)
        conflicting_roles["concepts"][2]["concept_id"] = "ground_surface"
        variants.append(conflicting_roles)
        duplicate = copy.deepcopy(policy)
        duplicate["concepts"][2]["phrase"] = duplicate["concepts"][0]["phrase"]
        duplicate["canonical_prompts"][2] = duplicate["canonical_prompts"][0]
        variants.append(duplicate)
        wrong_order = copy.deepcopy(policy)
        wrong_order["canonical_prompts"].reverse()
        variants.append(wrong_order)
        reused_identity = copy.deepcopy(policy)
        reused_identity["policy_id"] = "visible_avoid_concepts_v1"
        variants.append(reused_identity)
        excessive = copy.deepcopy(policy)
        excessive["concepts"] *= 11
        variants.append(excessive)
        for index, malformed in enumerate(variants):
            with self.subTest(index=index):
                with self.assertRaises(ValueError):
                    create_adapter(self.write_explicit_policy(malformed))

    def test_cleanup_borrowed_and_owned_resources(self):
        borrowed, segmenter, processor = self.make_adapter()
        borrowed.observe(self.frame)
        borrowed.close()
        borrowed.close()
        self.assertIsNotNone(segmenter.processor)
        self.assertEqual(processor.close_calls, 0)
        owned = create_adapter({"fixture": True})
        with patch("traversability_hazard_segmentation.load_segmenter", return_value=segmenter) as load:
            with patch.object(segmenter, "close", wraps=segmenter.close) as close:
                owned.observe(self.frame)
                owned.observe(self.frame)
                self.assertEqual(load.call_count, 1)
                owned.close()
                owned.close()
                self.assertEqual(close.call_count, 1)
        self.assertEqual(processor.close_calls, 0)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            owned.observe(self.frame)

    def test_missing_model_is_unavailable_without_empty_success(self):
        adapter = create_adapter({})
        self.addCleanup(adapter.close)
        for attempt in range(2):
            failed = adapter.observe(self.frame)
            self.assertEqual(failed.status, "error")
            self.assertEqual(failed.adapter_provenance["availability"], "unavailable")
            self.assertIn("model_loading_disabled", failed.error["message"])
            self.assertEqual(sum(failed.model_calls.values()), 0)
            self.assertEqual(failed.query_count, 0)
            self.assertEqual(len(failed.queries), 16)
            self.assertTrue(all(q.status == "error" and not q.instances for q in failed.queries))

    def test_unexpected_backend_crash_has_unknown_counts_and_no_fake_empty_success(self):
        adapter, segmenter, _ = self.make_adapter()
        with patch.object(segmenter, "segment_image", side_effect=RuntimeError("crash after text execution")):
            observation = adapter.observe(self.frame)
        self.assertEqual(observation.status, "error")
        self.assertEqual(sum(observation.model_calls.values()), 1)
        self.assertIsNone(observation.query_count)
        self.assertTrue(observation.query_count_reason)
        self.assertIsNone(observation.adapter_provenance["counts"]["executed_queries"])
        self.assertTrue(all(q.status == "error" for q in observation.queries))
        validate_semantic(observation, self.frame)

    def test_synchronization_boundary_and_failure_do_not_invent_cheap_calls(self):
        adapter, _, _ = self.make_adapter()
        with patch.object(adapter, "_synchronize") as synchronize:
            successful = adapter.observe(self.frame)
        self.assertEqual(synchronize.call_count, 2)
        self.assertTrue(successful.adapter_provenance["timing"]["synchronized"])
        with patch.object(adapter, "_synchronize", side_effect=[None, RuntimeError("fixture synchronization failure")]):
            failed = adapter.observe(self.frame)
        self.assertEqual(failed.status, "error")
        self.assertEqual(sum(failed.model_calls.values()), 1)
        self.assertIsNone(failed.query_count)
        self.assertIsNone(failed.adapter_provenance["timing"]["sam_call_ms"])
        self.assertFalse(failed.adapter_provenance["timing"]["synchronized"])

    def test_cleanup_error_releases_wrapper_reference_and_is_reported(self):
        _, segmenter, _ = self.make_adapter()
        owned = create_adapter({"fixture": True})
        with patch("traversability_hazard_segmentation.load_segmenter", return_value=segmenter):
            owned.observe(self.frame)
        with patch.object(segmenter, "close", side_effect=RuntimeError("fixture cleanup failed")):
            with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                owned.close()
        self.assertIsNone(owned._segmenter)
        owned.close()

    def test_import_does_not_load_model_dependencies(self):
        process = subprocess.run([sys.executable, "-c",
            "import sys; import pipelines.fixed_hazards; "
            "assert not any(m in sys.modules for m in ['torch','sam3','transformers'])"],
            capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_common_cli_layout_evaluator_and_baseline_with_actual_fixed_adapter(self):
        from pipeline_common.sequence import prepare
        from pipeline_common.io import read_json
        from run_pipeline import main as run_main
        from evaluate_pipeline import main as evaluate_main
        from compare_pipelines import main as compare_main
        source = self.root / "source"
        source.mkdir()
        Image.fromarray(self.frame.rgb).save(source / "input.png")
        prepared = self.root / "sequence"
        prepare(source, {"contract_id": "semantic_mapping_v1", "schema_version": 1,
            "sequence_id": "fixed-hazards-cli-fixture", "split": "development", "fixture": True,
            "max_frames": 1}, prepared)
        config = json.loads((Path(__file__).resolve().parents[2] / "configs/pipelines/fixed_hazards.json").read_text())
        config["fixture"] = True
        adapter, _, _ = self.make_adapter()
        required = ("run.json", "config.resolved.json", "frames.jsonl", "geometry/manifest.json",
            "semantics/frames.jsonl", "map/manifest.json", "map/voxels.npz", "map/concepts.json",
            "map/semantic_evidence.npz", "map/contributions.jsonl", "planning/costmap.npz",
            "planning/plans.jsonl", "events.jsonl", "summary.json")
        geometry_cache = None
        for pipeline in ("geometry_only", "ground_surface", "fixed_hazards"):
            run_config = copy.deepcopy(config)
            run_config["pipeline_id"] = pipeline
            run_config["pipeline"]["pipeline_id"] = pipeline
            path = self.root / f"{pipeline}.json"
            path.write_text(json.dumps(run_config), encoding="utf-8")
            output = self.root / pipeline
            args = ["--pipeline", pipeline, "--sequence", str(prepared / "sequence.json"),
                "--config", str(path), "--output", str(output), "--fixture"]
            if geometry_cache is not None: args += ["--geometry-cache", str(geometry_cache)]
            if pipeline == "fixed_hazards":
                with patch("pipeline_common.fixture_adapters.create_fixture_adapter", return_value=adapter):
                    self.assertEqual(run_main(args), 0)
            else:
                self.assertEqual(run_main(args), 0)
            if pipeline == "geometry_only": geometry_cache = output / "geometry/cache"
            for relative in required:
                self.assertTrue((output / relative).is_file(), relative)
            report_dir = self.root / f"{pipeline}_evaluation"
            self.assertEqual(evaluate_main(["--run", str(output), "--output", str(report_dir)]), 0)
            report = read_json(report_dir / "report.json")
            self.assertIsNone(report["metrics"]["safe_precision"]["value"])
            self.assertEqual(report["metrics"]["safe_precision"]["status"], "unavailable")
            if pipeline == "fixed_hazards":
                concepts = {row["concept_id"] for row in json.loads((output / "map/concepts.json").read_text())}
                self.assertEqual(concepts, set(CANONICAL_PROMPTS))
                self.assertEqual(report["metrics"]["semantic_query_count"]["value"], 16)
                semantics = json.loads((output / "semantics/frames.jsonl").read_text())
                self.assertEqual(len(semantics["queries"]), 16)
                self.assertIsNotNone(semantics["adapter_provenance"]["sam_adapter_identity"]["fixture_identity"])
                for query in semantics["queries"]:
                    for instance in query["instances"]:
                        with np.load(output / instance["mask_path"], allow_pickle=False) as masks:
                            self.assertEqual(masks["mask"].dtype, np.bool_)
            self.assertFalse(read_json(output / "summary.json")["models_executed"])
        compare_dir = self.root / "comparison"
        args = ["--output", str(compare_dir)]
        for pipeline in ("geometry_only", "ground_surface", "fixed_hazards"):
            args += ["--evaluation", str(self.root / f"{pipeline}_evaluation")]
        self.assertEqual(compare_main(args), 0)
        comparison = read_json(compare_dir / "comparison.json")
        self.assertTrue(comparison["compatible"])
        self.assertIsNone(comparison["ranking"])

    def test_unknown_counts_remain_null_in_common_evaluator_and_failures_denominator(self):
        from pipeline_common.sequence import prepare
        from pipeline_common.io import read_json
        from run_pipeline import main as run_main
        from evaluate_pipeline import main as evaluate_main
        source = self.root / "source"
        source.mkdir()
        for index in range(2): Image.fromarray(self.frame.rgb).save(source / f"input_{index}.png")
        prepared = self.root / "sequence"
        prepare(source, {"contract_id": "semantic_mapping_v1", "schema_version": 1,
            "sequence_id": "fixed-hazards-crash-fixture", "split": "development", "fixture": True,
            "max_frames": 2}, prepared)
        config = json.loads((Path(__file__).resolve().parents[2] / "configs/pipelines/fixed_hazards.json").read_text())
        config["fixture"] = True
        path = self.root / "config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        adapter, segmenter, _ = self.make_adapter()
        original = segmenter.segment_image
        calls = 0
        def crash_second(*args):
            nonlocal calls
            calls += 1
            if calls == 2: raise RuntimeError("fixture crash after unreported text executions")
            return original(*args)
        output = self.root / "run"
        with patch("pipeline_common.fixture_adapters.create_fixture_adapter", return_value=adapter):
            with patch.object(segmenter, "segment_image", side_effect=crash_second):
                self.assertEqual(run_main(["--pipeline", "fixed_hazards", "--sequence", str(prepared / "sequence.json"),
                    "--config", str(path), "--output", str(output), "--fixture"]), 0)
        evaluation = self.root / "evaluation"
        self.assertEqual(evaluate_main(["--run", str(output), "--output", str(evaluation)]), 0)
        report = read_json(evaluation / "report.json")
        counts = report["metrics"]["semantic_query_count"]
        self.assertIsNone(counts["value"])
        self.assertEqual(counts["status"], "unavailable")
        self.assertEqual(counts["known_query_count"], 16)
        failure = report["metrics"]["pipeline_failure_rate"]
        self.assertEqual(failure["numerator"], 1)
        self.assertEqual(failure["denominator"], 2)
        records = [json.loads(line) for line in (output / "semantics/frames.jsonl").read_text().splitlines()]
        self.assertIsNone(records[1]["query_count"])
        self.assertTrue(records[1]["query_count_reason"])
        self.assertEqual(len(records[1]["queries"]), 16)


if __name__ == "__main__":
    unittest.main()
