"""Conformance checks using independently constructed CPU reference assets."""

from __future__ import annotations

import copy

import hashlib

import json

from pathlib import Path

import subprocess

import sys

import numpy as np

import unittest
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pipeline_common.evaluation import BASE_METRIC_IDS, compare, evaluate, fingerprint

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")

def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

def grid(shape=(2, 3)):
    return {"shape": list(shape), "origin": [0., 0., 0.], "resolution": 1.,
            "projection_basis": [[1., 0., 0.], [0., 1., 0.]], "map_frame": "fixture_world"}

def make_run(root, pipeline="geometry_only", states=None, mode="quality_replay"):
    root.mkdir()
    if states is None:
        states = np.array([[2, 1, 0], [2, 2, 0]], dtype=np.uint8)
    config = {"contract_id": "semantic_mapping_v1", "schema_version": 1, "pipeline_id": pipeline,
        "pipeline": {"checkpoint": "fixture"}, "mode": mode, "protocol_id": "fixture_v1",
        "sequence": {"sequence_id": "fixture", "split": "test", "manifest_digest": "abc", "frame_count": 3},
        "geometry_identity": {"fingerprint": "frozen_geometry"},
        "geometry": {"up": {"vector": [0, 0, 1], "verified": True, "source": "analytic fixture"},
                     "scale": {"verified": True, "meters_per_unit": 1., "source": "analytic fixture"}},
        "fusion": {"voxel_origin": [0, 0, 0], "voxel_resolution": 1., "policy": "correlated_frame_cap_v1"},
        "robot": {"version": "synthetic_robot_v1", "fixture": True, "footprint_radius": .1},
        "goals": [{"request_id": "route", "start": [.5, .5, 0], "goal": [.5, 1.5, 0]}],
        "planning": {"unknown_space": "blocked"}, "evaluation_grid": grid(),
        "hardware_budget": {"visible_devices": "none", "host_ram_bytes": 1024},
        "semantic_keyframe_ids": ["a", "b", "c"],
        "runtime": {"scheduler": "latest_pending_v1", "cadence": 1, "capture_schedule": "test_source"}}
    run = {"contract_id": "semantic_mapping_v1", "schema_version": 1, "pipeline_id": pipeline,
           "status": "complete", "fixture": True, "capabilities": {"semantic": pipeline != "geometry_only"}}
    summary = {"required_frame_attempts": 3, "eligible_replay_duration_s": 2.,
        "timing_samples": {"semantic_call_ms": [1., 2., 3.], "geometry_stage_ms": [5., 10., 20.],
                           "end_to_end_latency_ms": [6., 12., 24.], "semantic_evidence_age_ms": [5., 8., 14.]},
        "timing_metadata": {mid: {"clock": "monotonic", "timing_boundaries": "measured stage start/end", "gpu_work": False}
            for mid in ("semantic_call_ms", "geometry_stage_ms", "end_to_end_latency_ms", "semantic_evidence_age_ms")}}
    write_json(root / "run.json", run)
    write_json(root / "config.resolved.json", config)
    write_json(root / "summary.json", summary)
    write_json(root / "geometry/manifest.json", {"geometry_fingerprint": "frozen_geometry", "processed_grid_id": "grid1",
                                               "input_fingerprint": "abc", "units": "metres", "up": [0, 0, 1]})
    write_rows(root / "frames.jsonl", [{"frame_id": "a", "status": "ok"}, {"frame_id": "b", "status": "error"}])
    write_rows(root / "semantics/frames.jsonl", [{"frame_id": "a", "status": "not_applicable" if pipeline == "geometry_only" else "ok",
        "processed_grid_id": "grid1", "decoded_rgb_sha256": "pixels_a", "query_count": 0, "queries": [], "model_calls": {}}])
    write_json(root / "map/concepts.json", {"concepts": []})
    write_rows(root / "planning/plans.jsonl", [{"request_id": "route", "status": "ok", "map_frame": "fixture_world",
                                             "path": [[.5, .5, 0], [.5, 1.5, 0]]}])
    (root / "planning").mkdir(exist_ok=True)
    np.savez(root / "planning/costmap.npz", decision_state=states, origin=np.array([0., 0.]),
             resolution=np.array([1.]), projection_basis=np.eye(3), costs=np.full(states.shape, np.inf),
             support_height=np.full(states.shape, np.nan))
    return config

def add_asset(reference, directory, name, **arrays):
    path = directory / (name + ".npz")
    np.savez(path, **arrays)
    reference["assets"].append({"asset_id": name, "path": path.name,
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

def make_reference(directory, config, truth=None):
    directory.mkdir()
    if truth is None:
        truth = np.array([[2, 1, 2], [1, 2, 1]], dtype=np.uint8)
    reference = {"contract_id": "semantic_mapping_v1", "schema_version": 1, "reference_id": "independent_fixture_v1",
        "reference_protocol_id": "analytic_fixture_v1", "sequence": config["sequence"], "fixture": True,
        "provenance": {"independent": True, "source": "analytically constructed fixture"},
        "taxonomy": {"version": "fixture_v1", "concept_ids": ["hazard_v1", "surface_v1"]},
        "coordinates": {"map_frame": "fixture_world", "units": "metres", "axes": ["x", "y", "z"], "up": [0, 0, 1],
            "alignment": {"verified": True, "source": "fixture"},
            "scale_provenance": {"verified": True, "source": "fixture"}, "pose_provenance": {"verified": True, "source": "fixture"}},
        "assets": [], "kinds": {"robot_decision": {"asset_id": "decisions", "grid": grid(),
            "coverage": {"description": "all independent fixture cells"},
            "robot_reference_policy": {"independent": True, "version": "fixture_v1", "robot_fingerprint": fingerprint(config["robot"])}}}}
    add_asset(reference, directory, "decisions", decision_state=truth, valid_mask=np.ones(truth.shape, bool), ignore_mask=np.zeros(truth.shape, bool))
    path = directory / "reference.json"
    write_json(path, reference)
    return path, reference

def metric(report, name):
    return report["metrics"][name]

def semantic_fixture(tmp_path):
    run = tmp_path / "run"
    config = make_run(run, "fixed_hazards")
    path, reference = make_reference(tmp_path / "ref", config)
    shape = (2, 3)
    truth = np.array([[1, 1, 0], [0, 0, 0]], bool)
    add_asset(reference, path.parent, "semantic_a", valid_mask=np.ones(shape, bool),
              ignore_mask=np.array([[0, 0, 1], [0, 0, 0]], bool), hazard=truth)
    reference["kinds"]["semantic_2d"] = {"coverage": {"description": "independent masks"}, "frames": [
        {"frame_id": "a", "processed_grid_id": "grid1", "decoded_rgb_sha256": "pixels_a", "asset_id": "semantic_a",
         "concept_masks": {"hazard_v1": "hazard"}}]}
    write_json(path, reference)
    (run / "semantics/masks").mkdir()
    np.savez(run / "semantics/masks/a.npz", accepted=np.array([[1, 0, 1], [1, 0, 0]], bool), failed=np.ones(shape, bool))
    write_rows(run / "semantics/frames.jsonl", [{"frame_id": "a", "status": "partial", "processed_grid_id": "grid1",
        "decoded_rgb_sha256": "pixels_a", "query_count": 2, "model_calls": {"sam3": 1}, "queries": [
            {"query_id": "q1", "concept_id": "hazard_v1", "status": "ok", "instances": [
                {"mask_path": "semantics/masks/a.npz", "mask_key": "accepted"}]},
            {"query_id": "q2", "concept_id": "hazard_v1", "status": "error", "instances": [
                {"mask_path": "semantics/masks/a.npz", "mask_key": "failed"}]}]}])
    return run, path, reference

class EvaluationTests(unittest.TestCase):
    def test_producer_frames_cannot_be_relabelled_by_roi_or_reference(self):
        for producer in ("geometry", "geometry_identity", "geometry/manifest.json", "map/manifest.json"):
            with self.subTest(producer=producer), tempfile.TemporaryDirectory() as temporary:
                tmp_path = Path(temporary)
                config = make_run(tmp_path / "run")
                if producer.endswith(".json"):
                    write_json(tmp_path / "run" / producer, {"map_frame": "producer_world_A"})
                else:
                    config[producer]["map_frame"] = "producer_world_A"
                    write_json(tmp_path / "run/config.resolved.json", config)
                path, _ = make_reference(tmp_path / "ref", config)
                with self.assertRaisesRegex(ValueError, "Producer.*map frame"):
                    evaluate(tmp_path / "run", path, tmp_path / "eval")

    def test_unauditable_qwen_count_remains_null_with_known_counts_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            make_run(tmp_path / "run", "qwen_hazards")
            write_rows(tmp_path / "run/semantics/frames.jsonl", [
                {"frame_id": "a", "status": "ok", "query_count": 2, "model_calls": {"sam3": 1}, "queries": []},
                {"frame_id": "b", "status": "error", "query_count": None, "model_calls": {}, "queries": [],
                 "adapter_provenance": {"query_counts": {"unavailable_reason": "SAM adapter crashed after call submission"}}}])
            report = evaluate(tmp_path / "run", None, tmp_path / "eval")
            count = metric(report, "semantic_query_count")
            self.assertEqual(count["status"], "unavailable")
            self.assertIsNone(count["value"])
            self.assertEqual(count["known_query_count"], 2)
            self.assertIsNone(report["operational_counts"]["query_count"])
            self.assertEqual(report["operational_counts"]["known_query_count"], 2)

    def test_voxel_semantics_uses_canonical_size_and_sparse_unknowns(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            run = tmp_path / "run"
            config = make_run(run, "fixed_hazards")
            config["fusion"] = {"origin": [0, 0, 0], "voxel_size": 1.}
            config["evaluation"] = {"semantic_evidence_threshold": .5}
            write_json(run / "config.resolved.json", config)
            write_json(run / "map/manifest.json", {"origin": [0, 0, 0], "voxel_size": 1.})
            write_json(run / "map/concepts.json", [{"concept_id": "hazard_v1"}])
            np.savez(run / "map/voxels.npz", voxel_indices=np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int64))
            np.savez(run / "map/semantic_evidence.npz", voxel_row=np.array([0, 1], dtype=np.int64),
                     concept_row=np.array([0, 0], dtype=np.int64), evidence_score=np.array([1., 1.]))
            path, reference = make_reference(tmp_path / "ref", config)
            add_asset(reference, path.parent, "voxel_truth", voxel_indices=np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.int64),
                      valid_mask=np.ones(3, bool), ignore_mask=np.zeros(3, bool), hazard=np.array([1, 0, 1], bool))
            reference["kinds"]["semantic_3d"] = {"asset_id": "voxel_truth", "voxel_origin": [0, 0, 0],
                "voxel_resolution": 1., "prediction_threshold": .5, "coverage": {"description": "independent voxel samples"},
                "concept_masks": {"hazard_v1": "hazard"}}
            write_json(path, reference)
            report = evaluate(run, path, tmp_path / "eval")
            self.assertEqual(metric(report, "semantic_iou.hazard_v1")["value"], 1 / 3)
            self.assertEqual(metric(report, "semantic_recall.hazard_v1")["value"], .5)
            self.assertEqual(metric(report, "semantic_iou.hazard_v1")["reference_coverage"]["kind"], "semantic_3d")
            # The common fusion threshold is the canonical default when no
            # separate frozen evaluation threshold was requested.
            config.pop("evaluation")
            config["fusion"]["semantic_threshold"] = .5
            write_json(run / "config.resolved.json", config)
            fallback = evaluate(run, path, tmp_path / "eval_fusion_threshold")
            self.assertEqual(metric(fallback, "semantic_iou.hazard_v1")["value"], 1 / 3)
            self.assertEqual(fallback["comparability"]["identities"]["semantic_evaluation_threshold"], .5)
            # An explicit evaluation override must remain a comparison gate.
            config["evaluation"] = {"semantic_evidence_threshold": .75}
            write_json(run / "config.resolved.json", config)
            reference["kinds"]["semantic_3d"]["prediction_threshold"] = .75
            write_json(path, reference)
            explicit = evaluate(run, path, tmp_path / "eval_explicit_threshold")
            self.assertEqual(explicit["comparability"]["identities"]["semantic_evaluation_threshold"], .75)
            result = compare([tmp_path / "eval_fusion_threshold", tmp_path / "eval_explicit_threshold"], tmp_path / "compare_thresholds")
            self.assertFalse(result["compatible"])
            self.assertIn("evaluation_policy", [d["identity"] for d in result["differences"]])
            self.assertIn("semantic_evaluation_threshold", [d["identity"] for d in result["differences"]])

    def test_unsynchronized_gpu_timings_and_component_memory_are_unavailable(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            make_run(tmp_path / "run", "fixed_hazards")
            summary = json.loads((tmp_path / "run/summary.json").read_text())
            summary["timing_metadata"]["semantic_call_ms"].update(gpu_work=True, synchronized=False)
            summary["operational_metrics"] = {"joint_gpu_allocated_peak_bytes": {
                "value": 1000, "collection_scope": "sum_of_separate_component_peaks"}}
            write_json(tmp_path / "run/summary.json", summary)
            report = evaluate(tmp_path / "run", None, tmp_path / "eval")
            self.assertIsNone(metric(report, "semantic_call_ms.median")["value"])
            self.assertIsNone(metric(report, "joint_gpu_allocated_peak_bytes")["value"])

    def test_frozen_roi_missing_predictions_and_safety_denominators(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            config = make_run(tmp_path / "run", states=np.array([[2, 1]], dtype=np.uint8))
            ref, _ = make_reference(tmp_path / "ref", config)
            report = evaluate(tmp_path / "run", ref, tmp_path / "eval")
            assert metric(report, "decision_coverage")["numerator"] == 2
            assert metric(report, "decision_coverage")["denominator"] == 6
            assert metric(report, "unknown_rate")["value"] == 4 / 6
            assert metric(report, "safe_precision")["value"] == 1.
            assert metric(report, "safe_recall")["value"] == 1 / 3
            assert metric(report, "unsafe_as_traversable_rate")["value"] == 0.
            assert metric(report, "pipeline_failure_rate")["value"] == 2 / 3
            assert set(BASE_METRIC_IDS) <= report["metrics"].keys()
            assert (tmp_path / "eval/metrics.csv").exists()

    def test_unknown_map_cannot_fake_precision_or_recall(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            config = make_run(tmp_path / "run", states=np.empty((0, 0), dtype=np.uint8))
            ref, _ = make_reference(tmp_path / "ref", config)
            report = evaluate(tmp_path / "run", ref, tmp_path / "eval")
            assert metric(report, "unknown_rate")["value"] == 1.
            assert metric(report, "safe_precision")["value"] is None
            assert metric(report, "safe_precision")["status"] == "unavailable"
            assert metric(report, "safe_recall")["value"] == 0.
            assert metric(report, "unsafe_as_traversable_rate")["value"] == 0.

    def test_missing_reference_and_no_semantic_capability(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            make_run(tmp_path / "run")
            report = evaluate(tmp_path / "run", None, tmp_path / "eval")
            assert metric(report, "safe_recall")["value"] is None
            assert metric(report, "reference_valid_path_rate")["value"] is None
            assert metric(report, "semantic_call_ms.p95")["status"] == "not_applicable"
            assert metric(report, "semantic_query_count")["value"] == 0
            assert "NaN" not in (tmp_path / "eval/report.json").read_text()

    def test_geometry_semantic_taxonomy_is_not_applicable(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            config = make_run(tmp_path / "run")
            ref, _ = make_reference(tmp_path / "ref", config)
            report = evaluate(tmp_path / "run", ref, tmp_path / "eval")
            assert metric(report, "semantic_iou.hazard_v1")["status"] == "not_applicable"
            assert metric(report, "semantic_iou.hazard_v1")["value"] is None

    def test_invalid_supplied_reference_is_rejected(self):
        for case_index, case in enumerate([
    (lambda r: r["sequence"].update(manifest_digest="bad"), "sequence"),
    (lambda r: r.update(fixture=False), "fixture"),
    (lambda r: r["provenance"].update(independent=False), "independent"),
    (lambda r: r["assets"][0].update(sha256="bad"), "hash"),
    (lambda r: r["assets"][0].update(path="../outside.npz"), "asset"),
]):
            with self.subTest(case_index=case_index):
                mutation,error = case
                with tempfile.TemporaryDirectory() as temporary:
                    tmp_path = Path(temporary)
                    config = make_run(tmp_path / "run")
                    path, reference = make_reference(tmp_path / "ref", config)
                    mutation(reference)
                    write_json(path, reference)
                    with self.assertRaisesRegex(ValueError, expected_regex=error):
                        evaluate(tmp_path / "run", path, tmp_path / "eval")
                    assert not (tmp_path / "eval/report.json").exists()

    def test_reference_unknown_ignore_and_empty_denominators(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            config = make_run(tmp_path / "run")
            path, reference = make_reference(tmp_path / "ref", config, truth=np.zeros((2, 3), np.uint8))
            report = evaluate(tmp_path / "run", path, tmp_path / "eval")
            assert metric(report, "safe_recall")["denominator"] == 0
            assert metric(report, "safe_recall")["value"] is None
            assert metric(report, "unsafe_as_traversable_rate")["value"] is None

    def test_unverified_reference_physical_scores_are_unavailable(self):
        for case_index, case in enumerate(["alignment", "scale_provenance", "pose_provenance"]):
            with self.subTest(case_index=case_index):
                kind = case
                with tempfile.TemporaryDirectory() as temporary:
                    tmp_path = Path(temporary)
                    config = make_run(tmp_path / "run")
                    path, reference = make_reference(tmp_path / "ref", config)
                    reference["coordinates"][kind]["verified"] = False
                    write_json(path, reference)
                    report = evaluate(tmp_path / "run", path, tmp_path / "eval")
                    assert metric(report, "safe_precision")["status"] == "unavailable"
                    assert metric(report, "decision_coverage")["status"] == "available"

    def test_other_robot_policy_cannot_certify_safety(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            config = make_run(tmp_path / "run")
            path, reference = make_reference(tmp_path / "ref", config)
            reference["kinds"]["robot_decision"]["robot_reference_policy"]["robot_fingerprint"] = "other_robot"
            write_json(path, reference)
            report = evaluate(tmp_path / "run", path, tmp_path / "eval")
            assert metric(report, "safe_recall")["value"] is None

    def test_actual_masks_partial_failures_and_ignore_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            run, path, _ = semantic_fixture(tmp_path)
            report = evaluate(run, path, tmp_path / "eval")
            assert metric(report, "semantic_precision.hazard_v1")["value"] == .5
            assert metric(report, "semantic_recall.hazard_v1")["value"] == .5
            assert metric(report, "semantic_iou.hazard_v1")["value"] == 1 / 3
            assert metric(report, "semantic_iou.hazard_v1")["reference_coverage"]["valid_pixels"] == 5
            assert metric(report, "semantic_iou.surface_v1")["value"] is None
            assert report["operational_counts"]["model_calls"] == {"sam3": 1}
            assert report["operational_counts"]["successful_queries"] == 1
            assert metric(report, "semantic_query_count")["value"] == 2

    def test_omitted_queries_do_not_become_successful_negatives(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            run, path, _ = semantic_fixture(tmp_path)
            rows = [json.loads(line) for line in (run / "semantics/frames.jsonl").read_text().splitlines()]
            rows[0].update(queries=[], status="ok", query_count=0)
            write_rows(run / "semantics/frames.jsonl", rows)
            report = evaluate(run, path, tmp_path / "eval")
            assert report["operational_counts"]["successful_queries"] == 0
            assert metric(report, "semantic_recall.hazard_v1")["value"] == 0.
            assert metric(report, "semantic_precision.hazard_v1")["value"] is None

    def test_semantic_grid_or_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            run, path, reference = semantic_fixture(tmp_path)
            reference["kinds"]["semantic_2d"]["frames"][0]["processed_grid_id"] = "cropped_other_grid"
            write_json(path, reference)
            with self.assertRaisesRegex(ValueError, expected_regex="grid"):
                evaluate(run, path, tmp_path / "eval")

    def test_independent_navigation_rejects_path_shortcut_and_missing_requests(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            config = make_run(tmp_path / "run")
            path, reference = make_reference(tmp_path / "ref", config)
            add_asset(reference, path.parent, "navigation", safe_for_robot=np.ones((2, 3), bool),
                      valid_mask=np.ones((2, 3), bool), ignore_mask=np.zeros((2, 3), bool))
            reference["kinds"]["navigation"] = {"asset_id": "navigation", "grid": grid(), "coverage": {"description": "analytic robot clearance"},
                "robot_fingerprint": fingerprint(config["robot"]),
                "robot_reference_policy": {"independent": True, "footprint_and_clearance_certified": True},
                "requests": [{"request_id": "route", "valid_reference": True}]}
            write_json(path, reference)
            report = evaluate(tmp_path / "run", path, tmp_path / "eval")
            assert metric(report, "reference_valid_path_rate")["value"] == 1.
            # Returned internal planner path with wrong goal is not reference valid.
            write_rows(tmp_path / "run/planning/plans.jsonl", [{"request_id": "route", "status": "ok", "map_frame": "fixture_world",
                                                             "path": [[.5, .5, 0]]}])
            report = evaluate(tmp_path / "run", path, tmp_path / "eval2")
            assert metric(report, "reference_valid_path_rate")["value"] == 0.

    def test_quality_timing_is_not_live_and_paced_cache_is_not_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            make_run(tmp_path / "run", "ground_surface")
            report = evaluate(tmp_path / "run", None, tmp_path / "eval")
            assert metric(report, "semantic_call_ms.median")["value"] == 2.
            assert metric(report, "geometry_stage_ms.p95")["value"] == 19.
            for mid in ("semantic_update_rate_hz", "semantic_evidence_age_ms.p95", "end_to_end_latency_ms.p95"):
                assert metric(report, mid)["value"] is None
            config = json.loads((tmp_path / "run/config.resolved.json").read_text())
            config["mode"] = "paced_runtime"
            write_json(tmp_path / "run/config.resolved.json", config)
            summary = json.loads((tmp_path / "run/summary.json").read_text())
            summary["measurement_scope"] = "cached_geometry_scheduler_replay"
            write_json(tmp_path / "run/summary.json", summary)
            report = evaluate(tmp_path / "run", None, tmp_path / "eval2")
            assert metric(report, "semantic_update_rate_hz")["value"] is None
            assert metric(report, "end_to_end_latency_ms.p95")["value"] is None

    def test_missing_frozen_roi_raw_reconstruction_operational_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            config = make_run(tmp_path / "run")
            config.pop("evaluation_grid")
            write_json(tmp_path / "run/config.resolved.json", config)
            report = evaluate(tmp_path / "run", None, tmp_path / "eval")
            assert metric(report, "decision_coverage")["value"] is None
            assert metric(report, "pipeline_failure_rate")["value"] == 2 / 3

    def test_comparison_refuses_every_required_identity_mismatch(self):
        for case_index, case in enumerate(["mode", "geometry_identity", "fusion", "robot", "goals", "planning", "evaluation_grid",
                                  "hardware_budget", "semantic_keyframe_ids", "geometry", "protocol_id", "sequence"]):
            with self.subTest(case_index=case_index):
                key = case
                with tempfile.TemporaryDirectory() as temporary:
                    tmp_path = Path(temporary)
                    make_run(tmp_path / "a")
                    config = make_run(tmp_path / "b", "ground_surface")
                    changed = copy.deepcopy(config)
                    if key == "mode":
                        changed[key] = "paced_runtime"
                    elif isinstance(changed[key], dict):
                        changed[key]["comparison_test_variant"] = True
                    elif isinstance(changed[key], list):
                        changed[key] = changed[key][:-1]
                    else:
                        changed[key] += "_variant"
                    write_json(tmp_path / "b/config.resolved.json", changed)
                    evaluate(tmp_path / "a", None, tmp_path / "eval_a")
                    evaluate(tmp_path / "b", None, tmp_path / "eval_b")
                    report = compare([tmp_path / "eval_a", tmp_path / "eval_b"], tmp_path / "comparison")
                    assert report["compatible"] is False
                    assert report["ranking"] is None
                    assert report["differences"]

    def test_matching_four_pipelines_are_comparable_without_ranking(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            evaluations = []
            for name in ("geometry_only", "ground_surface", "fixed_hazards", "qwen_hazards"):
                make_run(tmp_path / name, name)
                evaluate(tmp_path / name, None, tmp_path / (name + "_eval"))
                evaluations.append(tmp_path / (name + "_eval"))
            report = compare(evaluations, tmp_path / "comparison")
            assert report["compatible"] is True
            assert report["ranking"] is None
            assert len(report["evaluations"]) == 4

    def test_incomplete_run_and_modified_report_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            make_run(tmp_path / "run")
            evaluate(tmp_path / "run", None, tmp_path / "eval")
            report = json.loads((tmp_path / "eval/report.json").read_text())
            report["comparability"]["identities"]["robot"] = {"modified": True}
            write_json(tmp_path / "eval/report.json", report)
            with self.assertRaisesRegex(ValueError, expected_regex="corruption"):
                compare([tmp_path / "eval", tmp_path / "eval"], tmp_path / "comparison")
            record = json.loads((tmp_path / "run/run.json").read_text())
            record["status"] = "running"
            write_json(tmp_path / "run/run.json", record)
            with self.assertRaisesRegex(ValueError, expected_regex="completed"):
                evaluate(tmp_path / "run", None, tmp_path / "eval2")

    def test_cli_exact_interfaces(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            make_run(tmp_path / "run")
            root = Path(__file__).resolve().parents[2]
            completed = subprocess.run([sys.executable, str(root / "src/evaluate_pipeline.py"), "--run", str(tmp_path / "run"),
                                        "--output", str(tmp_path / "eval")], capture_output=True, text=True)
            assert completed.returncode == 0, completed.stderr
            completed = subprocess.run([sys.executable, str(root / "src/compare_pipelines.py"),
                                       "--evaluation", str(tmp_path / "eval"), "--evaluation", str(tmp_path / "eval"),
                                       "--output", str(tmp_path / "comparison")], capture_output=True, text=True)
            assert completed.returncode == 0, completed.stderr

if __name__ == "__main__":
    unittest.main()
