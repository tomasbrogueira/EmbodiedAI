"""Deterministic CPU-only hazard profiling tests; no models or GPU jobs."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

COMPONENT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(COMPONENT / "src"))

from traversability_data.storage import atomic_json, atomic_jsonl, read_json, read_jsonl
from traversability_hazard_benchmark import aggregate_samples, profile_backend
from traversability_hazard_benchmark.profiling import _timed_call, _vlm_result
from traversability_hazard_segmentation.fixtures import FakeBackend, FakeSegmenter, prepare_fixture


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class Probe:
    """Explicit synthetic telemetry, never real GPU evidence."""
    available = False
    unavailable_reason = "synthetic probe"

    def __init__(self, *, fail_reset=False, fail_sync=False, fail_snapshot=False):
        self.resets = self.synchronizations = 0
        self.fail_reset, self.fail_sync, self.fail_snapshot = fail_reset, fail_sync, fail_snapshot

    def synchronize(self):
        self.synchronizations += 1
        if self.fail_sync:
            raise RuntimeError("synthetic synchronization failure")

    def reset_peak(self):
        self.resets += 1
        if self.fail_reset:
            raise RuntimeError("synthetic peak-reset failure")

    def snapshot(self):
        if self.fail_snapshot:
            raise RuntimeError("synthetic snapshot failure")
        return {"allocated_bytes": 100, "reserved_bytes": 200,
                "device_used_bytes": 1000, "device_total_bytes": 20000}

    def peaks(self):
        return {"allocated_peak_bytes": 500, "reserved_peak_bytes": 700}


class TimedBackend(FakeBackend):
    def __init__(self, clock, *, mode="ok"):
        super().__init__()
        self.clock, self.mode = clock, mode
        self.fail_once = True
        self.received_frames = []

    def predict_image(self, frame, data_root):
        self.received_frames.append(deepcopy(frame))
        result = super().predict_image(frame, data_root)
        self.clock.advance(0.2)
        if self.mode == "oom_once" and len(self.calls) == 6 and self.fail_once:
            self.fail_once = False
            raise RuntimeError("CUDA out of memory: explicit CPU fixture")
        if self.mode == "ordinary_exception":
            raise RuntimeError("synthetic ordinary call failure")
        if self.mode == "mutate_settings":
            self.settings["resolution"] = 999
        if self.mode == "mutate_image":
            (Path(data_root) / frame["image_path"]).write_bytes(b"changed frozen bytes")
        if self.mode == "invalid_join":
            result["frame_id"] = "wrong-frame"
        if self.mode == "invalid_unicode":
            result.update(prompts=[], raw_response="\ud800", status="error",
                          error_code="invalid_response_unicode")
        return result


class TimedSegmenter(FakeSegmenter):
    def __init__(self, clock, *, mode="ok"):
        super().__init__()
        self.clock, self.mode = clock, mode
        self.received_frames = []

    def segment_image(self, frame, prompts, data_root):
        self.received_frames.append(deepcopy(frame))
        result = super().segment_image(frame, prompts, data_root)
        self.clock.advance(0.3)
        if self.mode == "exception":
            raise RuntimeError("synthetic unexpected SAM failure")
        if self.mode == "partial":
            if result["queries"]:
                result["queries"][0].update(status="error", error_code="partial_fixture_error")
                result["frame"].update(status="error", error_code="partial_fixture_error")
        return result


class ArithmeticTests(unittest.TestCase):
    def test_failed_vlm_response_cannot_report_usable_hazards(self):
        frame = {"frame_id": "synthetic"}
        response = {"task_id": "hazard_prompt_v1", "schema_version": 1,
                    "frame_id": "synthetic", "model_key": "qwen3_vl_4b",
                    "prompts": ["water"], "raw_response": "truncated",
                    "status": "error", "error_code": "generation_truncated"}
        with self.assertRaisesRegex(ValueError, "cannot expose usable prompts"):
            _vlm_result(response, frame, "qwen3_vl_4b")
        del response["prompts"]
        response.update(status="ok", error_code=None)
        with self.assertRaisesRegex(ValueError, "eight hazard schema fields"):
            _vlm_result(response, frame, "qwen3_vl_4b")

    def test_attempt_quantiles_include_failures_and_exclude_warmups(self):
        rows = [{"phase": "warmup", "frame_id": "warmup", "elapsed_s": 100, "status": "ok"}]
        rows += [{"phase": "measured", "frame_id": str(index), "repeat": 0, "elapsed_s": duration,
                  "status": "error" if index == 2 else "ok", "sam_query_count": index}
                 for index, duration in enumerate((1.0, 2.0, 5.0))]
        summary = aggregate_samples(rows, expected_measured_frames=3, expected_unique_frames=3)
        self.assertEqual(summary["latency"]["median_s"], 2.0)
        self.assertAlmostEqual(summary["latency"]["p95_s"], 4.7)
        self.assertAlmostEqual(summary["successful_latency"]["median_s"], 1.5)
        self.assertEqual(summary["failed_attempt_latency"]["median_s"], 5.0)
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["failed_calls"], 1)
        self.assertEqual(summary["sam_query_count"], 3)

    def test_no_success_or_invalid_timing_has_null_quantiles(self):
        rows = [{"phase": "measured", "frame_id": "x", "repeat": 0, "elapsed_s": None,
                 "timing_valid": False, "status": "error", "sam_query_count": None}]
        summary = aggregate_samples(rows)
        self.assertEqual(summary["latency"], {"median_s": None, "p95_s": None})
        self.assertEqual(summary["successful_latency"], {"median_s": None, "p95_s": None})
        self.assertEqual(summary["unavailable_sam_query_counts"], 1)
        self.assertFalse(summary["complete"])

    def test_duplicate_slots_invalid_duration_and_query_counts_rejected(self):
        row = {"phase": "measured", "frame_id": "x", "repeat": 0, "elapsed_s": 1.0,
               "status": "ok", "sam_query_count": 1}
        with self.assertRaises(ValueError):
            aggregate_samples([row, row])
        for change in ({"elapsed_s": math.nan}, {"elapsed_s": -1}, {"sam_query_count": True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                aggregate_samples([{**row, **change}])

    def test_whole_interval_timing_and_memory_attribution(self):
        clock, probe = Clock(), Probe()
        value, elapsed, error, memory, valid = _timed_call(lambda: clock.advance(0.8), probe, probe.snapshot(), clock)
        self.assertIsNone(value)
        self.assertIsNone(error)
        self.assertTrue(valid)
        self.assertEqual(elapsed, 0.8)
        self.assertEqual(probe.resets, 1)
        self.assertEqual(probe.synchronizations, 2)
        self.assertEqual(memory["incremental_allocated_peak_bytes"], 400)
        self.assertEqual(memory["device_used_peak_bytes"], 1000)
        self.assertEqual(memory["incremental_device_used_peak_bytes"], 0)

    def test_telemetry_failures_preserve_nulls_without_inventing_zero_timing(self):
        for probe in (Probe(fail_reset=True), Probe(fail_sync=True), Probe(fail_snapshot=True)):
            with self.subTest(probe=probe):
                clock = Clock()
                _, elapsed, error, memory, valid = _timed_call(lambda: clock.advance(0.1), probe, {}, clock)
                if probe.fail_sync:
                    self.assertIsNone(elapsed)
                    self.assertFalse(valid)
                    self.assertTrue(error["timing_invalid"])
                else:
                    self.assertTrue(valid)
                if probe.fail_reset:
                    self.assertIsNone(memory["allocated_peak_bytes"])
                if probe.fail_snapshot:
                    self.assertIsNone(memory["device_used_peak_bytes"])
                self.assertIsNone(memory["incremental_allocated_peak_bytes"])


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.config = prepare_fixture(root / "run", root / "data")
        self.clock, self.probe = Clock(), Probe()
        self.config.update(_clock=self.clock, _memory_probe=self.probe,
                           component="vlm", condition_key="vlm__qwen3_vl_4b", profile_id="test_vlm")

    def test_sam_profile_uses_same_actual_adapter_as_segmentation_conditions(self):
        from traversability_hazard_segmentation import run_conditions
        run_conditions({**self.config, "conditions": ["reference_present"]},
                       segmenter=FakeSegmenter())
        segmenter = TimedSegmenter(self.clock)
        segmenter.settings["mask_probability_threshold"] = 0.9
        result = profile_backend({**self.config, "component": "sam", "condition_key": "fixed_policy",
                                  "profile_id": "wrong_threshold"}, segmenter=segmenter)
        self.assertFalse(result["summary"]["complete"])
        self.assertIn("identity changed across conditions",
                      result["metadata"]["sessions"][-1]["fatal_error"]["message"])
        self.assertEqual(segmenter.calls, [])

    def test_failed_raw_unicode_is_preserved_without_losing_measured_errors(self):
        report = profile_backend(self.config, backend=TimedBackend(self.clock, mode="invalid_unicode"))
        self.assertTrue(report["summary"]["complete"])
        self.assertEqual(report["summary"]["failed_calls"], 40)
        self.assertTrue(all(sample["vlm"]["raw_response"] == "\ud800" for sample in report["samples"]))
        self.assertIn(b"\\ud800", (Path(report["output_dir"]) / "samples.jsonl").read_bytes())

    def test_exact_schedule_raw_errors_and_fixture_provenance(self):
        backend = TimedBackend(self.clock)
        report = profile_backend(self.config, backend=backend)
        self.assertEqual(len(backend.calls), 45)
        self.assertEqual(report["summary"]["measured_frames"], 40)
        self.assertEqual(report["summary"]["unique_test_frames"], 20)
        self.assertEqual(report["summary"]["failed_calls"], 4)
        self.assertTrue(report["summary"]["complete"])
        self.assertFalse(report["summary"]["comparison_ready"])
        self.assertTrue(all(abs(sample["elapsed_s"] - 0.2) < 1e-10 for sample in report["samples"]))
        self.assertFalse(backend.closed)
        failed = [sample for sample in report["samples"] if sample.get("vlm", {}).get("status") == "error"]
        self.assertTrue(failed)
        self.assertTrue(all(sample["vlm"]["raw_response"] == "{bad JSON" for sample in failed))
        self.assertEqual(report["summary"]["budget"]["status"], "unmeasured")
        self.assertIsNone(report["summary"]["budget"]["allocated_within_proposed_vlm_budget"])
        self.assertEqual(report["metadata"]["execution_kind"], "injected_fixture")
        self.assertEqual(report["metadata"]["selection_hash"], report["summary"]["selection_hash"])

    def test_fixed_policy_calls_and_partial_query_outputs_remain_failures(self):
        self.config.update(component="sam", condition_key="fixed_policy", profile_id="test_sam")
        segmenter = TimedSegmenter(self.clock, mode="partial")
        report = profile_backend(self.config, segmenter=segmenter)
        policy = read_json(COMPONENT / "configs/hazards/policy.json")
        self.assertTrue(all(prompts == policy["canonical_prompts"] for _, prompts in segmenter.calls))
        self.assertEqual(report["summary"]["sam_query_count"], 40 * 16)
        self.assertEqual(report["summary"]["failed_calls"], 40)
        self.assertTrue(report["summary"]["complete"])
        query = report["samples"][0]["sam"]["queries"][0]
        self.assertEqual(query["status"], "error")
        self.assertEqual(query["returned_instance_count"], 1)
        self.assertEqual(query["scores"], [0.75])
        self.assertFalse(segmenter.closed)

    def test_condition_alias_phrases_empties_and_upstream_errors_preserved(self):
        self.config.update(component="sam", profile_id="test_handoff")
        segmenter = TimedSegmenter(self.clock)
        report = profile_backend(self.config, segmenter=segmenter)
        aliased = [prompts for _, prompts in segmenter.calls if "water" in prompts]
        self.assertTrue(aliased)
        self.assertTrue(all(prompts == ["water", "puddle"] for prompts in aliased))
        empty = [sample for sample in report["samples"] if sample["prompts"] == [] and sample["status"] == "ok"]
        upstream = [sample for sample in report["samples"] if sample["error_code"] == "upstream_error"]
        self.assertTrue(empty)
        self.assertTrue(upstream)
        self.assertTrue(all(sample["sam_segment_image_calls"] == 0 and sample["sam_query_count"] == 0 for sample in upstream))
        self.assertTrue(all(sample["vlm"]["raw_response"] == "{bad JSON" for sample in upstream))

    def test_reference_present_uses_each_frame_reference_names_only(self):
        self.config.update(component="sam", condition_key="reference_present", profile_id="test_reference")
        segmenter = TimedSegmenter(self.clock)
        report = profile_backend(self.config, segmenter=segmenter)
        references = {row["frame_id"]: row for row in read_jsonl(Path(self.config["run_dir"]) / "references.jsonl")}
        self.assertTrue(all(prompts == references[frame_id]["present_concepts"] for frame_id, prompts in segmenter.calls))
        self.assertTrue(report["summary"]["complete"])
        self.assertFalse(any("present_concepts" in frame or "concept_masks" in frame for frame in segmenter.received_frames))

    def test_combined_requires_opt_in_and_owns_one_peak_reset_per_frame(self):
        self.config.update(component="combined", profile_id="test_combined")
        backend, segmenter = TimedBackend(self.clock), TimedSegmenter(self.clock)
        with self.assertRaisesRegex(ValueError, "enable_combined"):
            profile_backend(self.config, backend=backend, segmenter=segmenter)
        self.config["enable_combined"] = True
        report = profile_backend(self.config, backend=backend, segmenter=segmenter)
        self.assertEqual(self.probe.resets, 45)
        good = [sample for sample in report["samples"] if sample["vlm"]["status"] == "ok"]
        self.assertTrue(all(abs(sample["elapsed_s"] - 0.5) < 1e-10 for sample in good))
        self.assertTrue(all(abs(sample["stage_times_s"]["vlm"] - 0.2) < 1e-10 for sample in good))
        self.assertTrue(all(abs(sample["stage_times_s"]["sam"] - 0.3) < 1e-10 for sample in good))
        self.assertEqual(report["summary"]["combined_latency"], report["summary"]["latency"])
        self.assertFalse(report["summary"]["joint_memory_measured"])

    def test_missing_telemetry_is_null_and_isolated_combined_fields_are_null(self):
        self.config.pop("_memory_probe")
        report = profile_backend(self.config, backend=TimedBackend(self.clock))
        self.assertIsNone(report["summary"]["memory"]["allocated_peak_bytes"])
        self.assertTrue(report["summary"]["memory"]["unavailable_reasons"])
        self.assertIsNone(report["summary"]["combined_latency"])
        self.assertFalse(report["summary"]["joint_memory_measured"])

    def test_completed_compatible_resume_preserves_errors_without_calls(self):
        backend = TimedBackend(self.clock)
        first = profile_backend(self.config, backend=backend)
        second = profile_backend(self.config, backend=backend)
        self.assertEqual(first["samples"], second["samples"])
        self.assertEqual(len(backend.calls), 45)
        self.assertEqual(len(second["metadata"]["sessions"]), 1)
        self.config["vlm_settings"] = {"visual_token_limit": 700}
        with self.assertRaisesRegex(ValueError, "identity differs"):
            profile_backend(self.config, backend=backend)

    def test_interrupted_resume_skips_terminal_failed_slot_and_records_new_warmups(self):
        backend = TimedBackend(self.clock, mode="oom_once")
        first = profile_backend(self.config, backend=backend)
        self.assertFalse(first["summary"]["complete"])
        self.assertEqual(first["summary"]["measured_frames"], 1)
        preserved = deepcopy(first["samples"][-1])
        second = profile_backend(self.config, backend=backend)
        self.assertTrue(second["summary"]["complete"])
        self.assertEqual(second["summary"]["measured_frames"], 40)
        self.assertEqual(second["summary"]["warmup"]["attempted_calls"], 10)
        self.assertEqual(second["samples"][5], preserved)
        self.assertEqual(len(second["metadata"]["sessions"]), 2)
        self.assertFalse(backend.closed)

    def test_ordinary_exceptions_continue_with_valid_failed_attempt_latency(self):
        report = profile_backend(self.config, backend=TimedBackend(self.clock, mode="ordinary_exception"))
        self.assertTrue(report["summary"]["complete"])
        self.assertEqual(report["summary"]["failed_calls"], 40)
        self.assertIsNone(report["summary"]["successful_latency"]["median_s"])
        self.assertAlmostEqual(report["summary"]["failed_attempt_latency"]["median_s"], 0.2)

    def test_unexpected_sam_failure_reports_unknown_text_call_count(self):
        self.config.update(component="sam", condition_key="fixed_policy", profile_id="sam_exception")
        report = profile_backend(self.config, segmenter=TimedSegmenter(self.clock, mode="exception"))
        self.assertTrue(report["summary"]["complete"])
        self.assertEqual(report["summary"]["unavailable_sam_query_counts"], 40)
        self.assertTrue(all(sample["requested_prompt_count"] == 16 and sample["sam_segment_image_calls"] == 1 for sample in report["samples"]))

    def test_changed_image_rejected_before_calls_and_after_call_published_as_error(self):
        backend = TimedBackend(self.clock, mode="mutate_image")
        report = profile_backend(self.config, backend=backend)
        self.assertEqual(len(report["samples"]), 1)
        self.assertEqual(report["samples"][0]["error_code"], "input_or_identity_changed")
        self.assertFalse(report["summary"]["complete"])
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            profile_backend(self.config, backend=backend)
        self.assertEqual(len(backend.calls), 1)

    def test_settings_mutation_and_invalid_frame_join_are_explicit_errors(self):
        for mode in ("mutate_settings", "invalid_join"):
            with self.subTest(mode=mode):
                self.config["profile_id"] = mode
                backend = TimedBackend(self.clock, mode=mode)
                report = profile_backend(self.config, backend=backend)
                self.assertTrue(report["samples"])
                self.assertEqual(report["samples"][0]["status"], "error")
                if mode == "mutate_settings":
                    self.assertFalse(report["summary"]["complete"])
                else:
                    self.assertEqual(report["summary"]["failed_calls"], 40)
                    self.assertIsInstance(report["samples"][0]["vlm"]["raw_response"], str)

    def test_injection_fixture_flags_identity_and_portable_profile_paths_rejected(self):
        for name in ("../escape", "C:bad", "a/b", "CON", "bad."):
            with self.subTest(name=name), self.assertRaises(ValueError):
                profile_backend({**self.config, "profile_id": name}, backend=TimedBackend(self.clock))
        backend = TimedBackend(self.clock)
        backend.fixture_identity = None
        backend.settings.pop("fixture_identity")
        backend.metadata.pop("fixture_id")
        with self.assertRaisesRegex(ValueError, "fixture_identity"):
            profile_backend(self.config, backend=backend)
        with self.assertRaises(ValueError):
            profile_backend({**self.config, "fixture": False}, backend=TimedBackend(self.clock))
        for owner, marker in (("settings", False), ("metadata", False), ("settings", 1)):
            conflicting = TimedBackend(self.clock)
            getattr(conflicting, owner)["fixture"] = marker
            with self.subTest(owner=owner, marker=marker), self.assertRaisesRegex(ValueError, "fixture marker"):
                profile_backend(self.config, backend=conflicting)

    def test_isolated_output_audit_runs_outside_timed_api_call(self):
        from traversability_hazard_benchmark import profiling
        validate = profiling._vlm_result
        def slower_validation(*arguments, **keywords):
            self.clock.advance(10.0)
            return validate(*arguments, **keywords)
        with patch.object(profiling, "_vlm_result", side_effect=slower_validation):
            report = profile_backend(self.config, backend=TimedBackend(self.clock))
        self.assertTrue(all(abs(sample["elapsed_s"] - 0.2) < 1e-10 for sample in report["samples"]))

    def test_upstream_blocked_sam_has_null_unattempted_time(self):
        self.config.update(component="sam", profile_id="blocked_timing")
        report = profile_backend(self.config, segmenter=TimedSegmenter(self.clock))
        blocked = [sample for sample in report["samples"] if sample["call_attempted"] is False]
        self.assertTrue(blocked)
        self.assertTrue(all(sample["elapsed_s"] is None and sample["sam_query_count"] == 0 for sample in blocked))
        self.assertEqual(report["summary"]["invalid_timing_calls"], 0)
        self.assertTrue(report["summary"]["complete"])

    def test_real_public_profile_uses_fresh_worker_and_rejects_test_hooks(self):
        from traversability_hazard_benchmark import profiling
        prepared = {"fixture": False, "config": {"enable_real_run": True, "data_root": Path("portable-root")}, "output": Path("unused")}
        with patch.object(profiling, "_prepare", return_value=prepared), patch.object(profiling.sys, "version_info", (3, 12)), \
                patch.object(profiling.subprocess, "run", return_value=type("Result", (), {"returncode": 0})()) as launch, \
                patch.object(profiling, "_read_profile", return_value={"summary": "saved"}):
            self.assertEqual(profile_backend({}), {"summary": "saved"})
            arguments = launch.call_args.args[0]
            self.assertEqual(arguments[1:], ["-m", "traversability_hazard_benchmark.worker"])
            self.assertTrue(launch.call_args.kwargs["capture_output"])
            self.assertNotIn("shell", launch.call_args.kwargs)
            payload = __import__("json").loads(launch.call_args.kwargs["input"])
            self.assertIs(payload["fixture"], False)
            self.assertEqual(payload["task_id"], "hazard_prompt_v1")
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["data_root"], "portable-root")
            prepared["config"]["unsupported_setting"] = object()
            with self.assertRaisesRegex(TypeError, "JSON values"):
                profile_backend({})
            prepared["config"].pop("unsupported_setting")
            prepared["config"]["_clock"] = self.clock
            with self.assertRaisesRegex(ValueError, "fixture mode"):
                profile_backend({})

    def test_saved_sample_identity_and_duplicate_slot_rejected_even_when_complete(self):
        backend = TimedBackend(self.clock)
        report = profile_backend(self.config, backend=backend)
        path = Path(report["output_dir"]) / "samples.jsonl"
        changed = deepcopy(report["samples"])
        changed[-1]["fixture"] = False
        atomic_jsonl(path, changed)
        with self.assertRaisesRegex(ValueError, "identity/fixture"):
            profile_backend(self.config, backend=backend)
        changed = deepcopy(report["samples"])
        changed[-1]["frame_id"], changed[-1]["repeat"] = changed[-2]["frame_id"], changed[-2]["repeat"]
        atomic_jsonl(path, changed)
        with self.assertRaisesRegex(ValueError, "Duplicate|hash mismatch"):
            profile_backend(self.config, backend=backend)

    def test_completed_resume_checks_actual_adapter_settings_and_record_content(self):
        backend = TimedBackend(self.clock)
        report = profile_backend(self.config, backend=backend)
        backend.settings["visual_token_budget"] = 123
        with self.assertRaisesRegex(ValueError, "actual settings/model"):
            profile_backend(self.config, backend=backend)
        backend.settings.pop("visual_token_budget")
        altered = deepcopy(report["samples"])
        altered[-1]["elapsed_s"] = 999.0
        atomic_jsonl(Path(report["output_dir"]) / "samples.jsonl", altered)
        with self.assertRaisesRegex(ValueError, "content hash mismatch"):
            profile_backend(self.config, backend=backend)

    def test_mocked_owned_loader_closes_only_its_adapter_and_loads_separately(self):
        from traversability_hazard_benchmark.profiling import _execute, _prepare
        self.config.update(component="sam", condition_key="fixed_policy", profile_id="owned")
        segmenter = TimedSegmenter(self.clock)
        def factory(settings):
            self.clock.advance(0.7)
            return segmenter
        with patch("traversability_hazard_segmentation.load_segmenter", side_effect=factory):
            report = _execute(_prepare(self.config))
        self.assertTrue(segmenter.closed)
        load = report["metadata"]["sessions"][0]["load"]["sam"]
        self.assertAlmostEqual(load["elapsed_s"], 0.7)
        self.assertAlmostEqual(report["summary"]["latency"]["median_s"], 0.3)

    def test_owned_vlm_loader_uses_shared_hub_cache_and_preserves_explicit_cache(self):
        from traversability_hazard_benchmark.profiling import _execute, _prepare
        cache_root = Path(self.temporary.name) / "runtime-cache"
        override = str(Path(self.temporary.name) / "explicit-model-cache")
        for index, cache_dir in enumerate((None, override)):
            with self.subTest(cache_dir=cache_dir):
                config = deepcopy(self.config)
                settings = {} if cache_dir is None else {"cache_dir": cache_dir}
                config.update(cache_root=str(cache_root), vlm_settings=settings,
                              profile_id=f"owned_vlm_cache_{index}")
                backend = TimedBackend(self.clock)
                with patch("traversability_hazard_inference.load_backend", return_value=backend) as load:
                    report = _execute(_prepare(config))
                load.assert_called_once()
                self.assertEqual(load.call_args.args[0], "qwen3_vl_4b")
                expected = str(cache_root / "huggingface/hub") if cache_dir is None else override
                self.assertEqual(load.call_args.args[1]["cache_dir"], expected)
                self.assertIs(load.call_args.args[1]["local_files_only"], True)
                self.assertTrue(report["summary"]["complete"])
                self.assertTrue(backend.closed)
                self.assertFalse(cache_root.exists())
                self.assertFalse(Path(override).exists())

    def test_mocked_missing_checkpoint_is_unavailable_with_no_frame_samples(self):
        from traversability_hazard_benchmark.profiling import _execute, _prepare
        self.config.update(component="sam", condition_key="fixed_policy", profile_id="unavailable")
        with patch("traversability_hazard_segmentation.load_segmenter", side_effect=RuntimeError("checkpoint access unavailable")):
            report = _execute(_prepare(self.config))
        self.assertEqual(report["samples"], [])
        self.assertFalse(report["summary"]["complete"])
        self.assertIsNone(report["summary"]["latency"]["median_s"])
        self.assertIn("checkpoint", report["metadata"]["sessions"][0]["load"]["sam"]["error"]["message"])

    def test_atomic_attempt_publication_keeps_readable_records(self):
        from traversability_hazard_benchmark import profiling
        actual = profiling.atomic_jsonl
        written = []
        def checked(path, rows):
            actual(path, rows)
            written.append(len(read_jsonl(path)))
        with patch.object(profiling, "atomic_jsonl", side_effect=checked):
            report = profile_backend(self.config, backend=TimedBackend(self.clock))
        self.assertEqual(written, list(range(46)))
        self.assertEqual(len(report["samples"]), 45)


class ImportTests(unittest.TestCase):
    def test_imports_load_no_model_libraries(self):
        code = "import sys; import traversability_hazard_benchmark; assert not any(x in sys.modules for x in ('torch','transformers','sam3')); print('ok')"
        environment = __import__("os").environ.copy()
        environment["PYTHONPATH"] = str(COMPONENT / "src")
        result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, env=environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "ok")


if __name__ == "__main__":
    unittest.main()
