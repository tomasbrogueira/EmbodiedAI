"""CPU-only checks of deployment profiling with explicitly synthetic inputs."""

from __future__ import annotations

import copy
import json
import math
from collections import Counter
from pathlib import Path
import struct
import tempfile
import threading
import unittest
import zlib

from traversability_benchmark import (
    FakeBackend,
    TorchMemoryProbe,
    aggregate_samples,
    freeze_selection,
    prepare_fixture,
    profile_backend,
)


SETTINGS = {
    "robot_profile_id": "rellis_material_v1",
    "robot_policy": "Permit dirt, asphalt and concrete conditional on geometry.",
    "language_quantization_bits": 4,
    "scene_visual_token_budget": 384,
    "crop_visual_token_budget": 256,
    "context_token_limit": 2048,
    "max_new_tokens": 128,
    "request_concurrency": 1,
    "seed": 0,
    "enable_thinking": False,
}


def _write_jsonl(path, records):
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")


def _png_bytes(value=255):
    """Build a valid 4x4 single-channel PNG without image dependencies."""
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 4, 4, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress((b"\x00" + bytes([value]) * 4) * 4))
        + chunk(b"IEND", b"")
    )


def _prepare_records(root, development=6, test=21):
    """Write synthetic manifests with two planning targets per frame."""
    run_dir, data_root = root / "run", root / "data"
    run_dir.mkdir()
    data_root.mkdir()
    frames, regions = [], []
    for split, count in (("development", development), ("test", test)):
        for index in range(count):
            frame_id = f"fixture:{split}:{index:03d}"
            image_path = f"{split}_{index:03d}.png"
            (data_root / image_path).write_bytes(_png_bytes(index + 1))
            frames.append({
                "frame_id": frame_id,
                "source": "fixture",
                "scene_id": f"fixture-{split}",
                "sequence_id": split,
                "timestamp_s": index / 2,
                "image_path": image_path,
                "split": split,
                "fixture": True,
            })
            for region_index in range(3):
                mask_path = f"{split}_{index:03d}_{region_index}.png"
                (data_root / mask_path).write_bytes(_png_bytes())
                regions.append({
                    "region_id": f"{frame_id}:{region_index}",
                    "frame_id": frame_id,
                    "mask_path": mask_path,
                    "planning_relevant": region_index < 2,
                    "selected_for_classification": region_index != 1,
                    "fixture": True,
                })
    _write_jsonl(run_dir / "frames.jsonl", frames)
    _write_jsonl(run_dir / "regions.jsonl", regions)
    return run_dir, data_root, frames, regions


class ManualClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, duration):
        self.now += duration


class Probe:
    """An allocator/device double whose device baseline includes other modules."""
    def __init__(self, available=False):
        self.available = available
        self.synchronizations = 0
        self.resets = 0
        self.snapshots = 0

    def synchronize(self):
        self.synchronizations += 1

    def snapshot(self):
        self.snapshots += 1
        return {
            "allocated_bytes": 100 if self.available else None,
            "reserved_bytes": 200 if self.available else None,
            "device_used_bytes": 1000 if self.available else None,
            "device_total_bytes": 20000 if self.available else None,
        }

    def reset_peak(self):
        self.resets += 1

    def peaks(self):
        return {
            "allocated_peak_bytes": 500 if self.available else None,
            "reserved_peak_bytes": 600 if self.available else None,
        }


class SnapshotFailureProbe(Probe):
    def __init__(self):
        super().__init__(available=True)

    def snapshot(self):
        self.snapshots += 1
        raise RuntimeError("Synthetic device-wide telemetry failure")


class PartialDeviceProbe(Probe):
    def __init__(self):
        super().__init__(available=True)

    def snapshot(self):
        result = super().snapshot()
        result.update(device_used_bytes=None, device_total_bytes=None)
        return result


class BaselineOnlyFailureProbe(Probe):
    def __init__(self):
        super().__init__(available=True)

    def snapshot(self):
        if self.snapshots == 0:
            self.snapshots += 1
            raise RuntimeError("Synthetic initial baseline failure")
        return super().snapshot()


class ResetFailureProbe(Probe):
    def __init__(self):
        super().__init__(available=True)

    def reset_peak(self):
        self.resets += 1
        raise RuntimeError("Synthetic allocator reset failure")

    def peaks(self):
        # These counters predate profiling and must never become measured peaks.
        return {"allocated_peak_bytes": 9_000_000_000, "reserved_peak_bytes": 10_000_000_000}


class SynchronizationFailureProbe(ResetFailureProbe):
    def synchronize(self):
        super().synchronize()
        if self.synchronizations > 1:
            raise RuntimeError("Synthetic pre-call synchronization failure")


class RecordingBackend:
    def __init__(self, model_key, settings, clock, mode="ok"):
        self.model_key = model_key
        self.settings = copy.deepcopy(settings)
        self.clock = clock
        self.mode = mode
        self.calls = []
        self.close_count = 0
        self.metadata = {
            "revision": "synthetic-fixture",
            "processor": {"fixture": True},
            "quantization": {"fixture": True},
            "vocabulary": [{"semantic_class": "synthetic_fixture", "fixture": True}],
            "calibration": {"fixture": True, "threshold": 0.5},
        }

    def predict_frame(self, frame, regions, data_root):
        self.calls.append((copy.deepcopy(frame), copy.deepcopy(regions), Path(data_root)))
        # Preparation, every target request, and parsing all occur inside the call.
        self.clock.advance(0.1)
        predictions = []
        for region in regions:
            self.clock.advance(0.2)
            predictions.append({
                "region_id": region["region_id"],
                "frame_id": frame["frame_id"],
                "model_key": self.model_key,
                "label": "traversable",
                "semantic_class": "concrete",
                "reason": "Synthetic fixture response",
                "raw_response": "Synthetic fixture response",
                "status": "ok",
                "error_code": None,
            })
        self.clock.advance(0.3)
        if frame["split"] == "development":
            self.clock.advance(10)
            return predictions
        if self.mode == "missing":
            return predictions[:-1]
        if self.mode == "duplicate":
            return predictions + predictions[:1]
        if self.mode == "wrong_frame":
            predictions[0]["frame_id"] = "fixture:wrong:frame"
        elif self.mode == "wrong_model":
            predictions[0]["model_key"] = "qwen3_5_2b"
        elif self.mode == "invalid_label":
            predictions[0]["label"] = "safe"
        elif self.mode == "parse_error":
            predictions[0].update(label="unknown", status="error", error_code="parse_error")
        elif self.mode == "ordinary_exception":
            raise RuntimeError("Synthetic request failure")
        elif self.mode == "fatal_exception":
            raise RuntimeError("CUDA out of memory: synthetic fixture")
        elif self.mode == "returned_oom":
            predictions[0].update(label="unknown", status="error", error_code="inference_error", reason="CUDA out of memory: synthetic fixture")
        elif self.mode == "keyboard_interrupt":
            measured_calls = sum(call[0]["split"] == "test" for call in self.calls)
            if measured_calls == 2:
                raise KeyboardInterrupt("Synthetic user interruption")
        elif self.mode == "extra_prediction":
            extra = copy.deepcopy(predictions[0])
            extra["region_id"] = f"{frame['frame_id']}:not-requested"
            predictions.append(extra)
        elif self.mode == "unhashable_label":
            predictions[0]["label"] = ["unknown"]
        elif self.mode == "unhashable_status":
            predictions[0]["status"] = {"invalid": "status"}
        return predictions

    def close(self):
        self.close_count += 1
        if self.mode == "cleanup_exception":
            raise RuntimeError("Synthetic cleanup failure")


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run_dir, self.data_root, self.frames, self.regions = _prepare_records(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def freeze(self):
        return freeze_selection(self.run_dir, self.data_root, seed=0, fixture=True)

    def test_fixed_split_and_all_planning_regions(self):
        selection = self.freeze()
        self.assertEqual(len(selection["warmup_frames"]), 5)
        self.assertEqual(len(selection["test_frames"]), 20)
        self.assertTrue(selection["fixture"])
        development_ids, test_ids = set(), set()
        for key, split, ids in (
            ("warmup_frames", "development", development_ids),
            ("test_frames", "test", test_ids),
        ):
            for entry in selection[key]:
                self.assertEqual(entry["frame"]["split"], split)
                ids.add(entry["frame"]["frame_id"])
                self.assertEqual(len(entry["regions"]), 2)
                self.assertTrue(all(region["planning_relevant"] for region in entry["regions"]))
                self.assertTrue(any(not region["selected_for_classification"] for region in entry["regions"]))
        self.assertFalse(development_ids & test_ids)

    def test_selection_is_deterministic_and_rejects_changed_frozen_content(self):
        first = self.freeze()
        self.assertEqual(first, self.freeze())
        image = self.data_root / first["test_frames"][0]["frame"]["image_path"]
        image.write_bytes(_png_bytes(77))
        with self.assertRaises(ValueError):
            self.freeze()

    def test_selected_mask_changes_are_rejected(self):
        first = self.freeze()
        mask = self.data_root / first["test_frames"][0]["regions"][0]["mask_path"]
        mask.write_bytes(_png_bytes(1))
        with self.assertRaises(ValueError):
            self.freeze()

    def test_fixture_cannot_be_counted_as_real_coverage(self):
        with self.assertRaises(ValueError):
            freeze_selection(self.run_dir, self.data_root, fixture=False)

    def test_recording_split_leakage_is_rejected(self):
        bad = copy.deepcopy(self.frames)
        first_test = next(row for row in bad if row["split"] == "test")
        first_test.update(scene_id=bad[0]["scene_id"], sequence_id=bad[0]["sequence_id"])
        _write_jsonl(self.run_dir / "frames.jsonl", bad)
        with self.assertRaises(ValueError):
            self.freeze()

    def test_annotations_do_not_affect_selection(self):
        first = self.freeze()
        (self.run_dir / "annotations.jsonl").write_text("intentionally invalid fixture references\n", encoding="utf-8")
        self.assertEqual(first, self.freeze())

    def test_manifest_row_order_does_not_change_frozen_id_joins(self):
        first = self.freeze()
        _write_jsonl(self.run_dir / "frames.jsonl", list(reversed(self.frames)))
        _write_jsonl(self.run_dir / "regions.jsonl", list(reversed(self.regions)))
        self.assertEqual(first, self.freeze())

    def test_test_selection_balances_recordings(self):
        records = copy.deepcopy(self.frames)
        index = 0
        for frame in records:
            if frame["split"] == "test":
                frame["sequence_id"] = f"held-out-{index % 4}"
                index += 1
        _write_jsonl(self.run_dir / "frames.jsonl", records)
        selection = self.freeze()
        counts = Counter(entry["frame"]["sequence_id"] for entry in selection["test_frames"])
        self.assertEqual(counts, {f"held-out-{index}": 5 for index in range(4)})

    def test_shortage_fails_before_profiling(self):
        for split, minimum in (("development", 5), ("test", 20)):
            with self.subTest(split=split):
                kept = [row for row in self.frames if row["split"] != split]
                kept += [row for row in self.frames if row["split"] == split][: minimum - 1]
                _write_jsonl(self.run_dir / "frames.jsonl", kept)
                kept_ids = {row["frame_id"] for row in kept}
                _write_jsonl(self.run_dir / "regions.jsonl", [row for row in self.regions if row["frame_id"] in kept_ids])
                with self.assertRaises(ValueError):
                    self.freeze()
        _write_jsonl(self.run_dir / "frames.jsonl", self.frames)
        _write_jsonl(self.run_dir / "regions.jsonl", self.regions)

    def test_duplicate_frame_and_region_ids_are_rejected(self):
        for filename, rows in (("frames.jsonl", self.frames), ("regions.jsonl", self.regions)):
            with self.subTest(filename=filename):
                _write_jsonl(self.run_dir / filename, rows + rows[:1])
                with self.assertRaises(ValueError):
                    self.freeze()
                _write_jsonl(self.run_dir / filename, rows)

    def test_orphan_regions_are_rejected(self):
        bad = copy.deepcopy(self.regions)
        bad[0]["frame_id"] = "fixture:absent:000"
        _write_jsonl(self.run_dir / "regions.jsonl", bad)
        with self.assertRaises(ValueError):
            self.freeze()

    def test_invalid_split_and_escaping_paths_are_rejected(self):
        for field, value in (("split", "validation"), ("image_path", "../outside.png"), ("image_path", str(self.root / "outside.png"))):
            with self.subTest(field=field, value=value):
                bad = copy.deepcopy(self.frames)
                for frame in bad:
                    frame[field] = value
                _write_jsonl(self.run_dir / "frames.jsonl", bad)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    self.freeze()
        _write_jsonl(self.run_dir / "frames.jsonl", self.frames)


class AggregationTests(unittest.TestCase):
    def test_linear_quantiles_counts_and_warmup_exclusion(self):
        samples = [
            {"phase": "warmup", "frame_id": "development", "elapsed_s": 1000, "region_count": 20, "failed_regions": 20},
            *[
                {"phase": "measured", "frame_id": "one", "elapsed_s": value, "region_count": 2, "failed_regions": int(value == 4), "call_error": None}
                for value in (1, 2, 3, 4)
            ],
        ]
        summary = aggregate_samples(samples)
        self.assertEqual(summary["measured_frames"], 4)
        self.assertEqual(summary["requested_regions"], 8)
        self.assertEqual(summary["failed_regions"], 1)
        self.assertEqual(summary["latency"]["median_s"], 2.5)
        self.assertAlmostEqual(summary["latency"]["p95_s"], 3.85)
        self.assertEqual(summary["successful_frames"], 3)
        self.assertEqual(summary["successful_latency"]["median_s"], 2)

    def test_empty_measurements_have_null_latency(self):
        summary = aggregate_samples([])
        self.assertEqual(summary["measured_frames"], 0)
        self.assertIsNone(summary["latency"]["median_s"])
        self.assertIsNone(summary["latency"]["p95_s"])

    def test_call_failures_remain_in_latency_and_region_denominators(self):
        summary = aggregate_samples([
            {"frame_id": "ok", "elapsed_s": 2, "region_count": 3, "failed_regions": 0},
            {"frame_id": "failed", "elapsed_s": 10, "region_count": 4, "failed_regions": 4, "call_error": {"type": "RuntimeError", "message": "fixture"}},
        ])
        self.assertEqual(summary["requested_regions"], 7)
        self.assertEqual(summary["failed_regions"], 4)
        self.assertEqual(summary["failed_calls"], 1)
        self.assertEqual(summary["latency"]["median_s"], 6)
        self.assertEqual(summary["successful_latency"]["median_s"], 2)


class TorchProbeTests(unittest.TestCase):
    def test_failed_device_query_preserves_known_allocator_counters(self):
        class FakeCuda:
            def memory_allocated(self, device):
                return 100

            def memory_reserved(self, device):
                return 200

            def mem_get_info(self, device):
                raise RuntimeError("Synthetic CUDA device telemetry failure")

        class FakeTorch:
            cuda = FakeCuda()

        # Inject the CUDA API directly, avoiding a Torch import or any GPU action.
        probe = TorchMemoryProbe.__new__(TorchMemoryProbe)
        probe.torch, probe.device = FakeTorch(), 0
        snapshot = probe.snapshot()
        self.assertEqual(snapshot["allocated_bytes"], 100)
        self.assertEqual(snapshot["reserved_bytes"], 200)
        self.assertIsNone(snapshot["device_used_bytes"])
        self.assertIsNone(snapshot["device_total_bytes"])
        self.assertIn("device", snapshot["unavailable_reason"])


class ProfilingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run_dir, self.data_root, _, _ = _prepare_records(self.root)
        self.selection = freeze_selection(self.run_dir, self.data_root, fixture=True)
        self.clock = ManualClock()
        self.probe = Probe()
        self.backends = []
        self.factory_calls = []

    def tearDown(self):
        self.temp.cleanup()

    def run_profile(self, mode="ok", output_name="output", mutate_settings=False):
        def factory(model_key, settings):
            self.factory_calls.append((model_key, copy.deepcopy(settings)))
            self.clock.advance(2)
            backend = RecordingBackend(model_key, settings, self.clock, mode)
            self.backends.append(backend)
            if mutate_settings:
                settings["seed"] = 999
            return backend

        return profile_backend(
            "clip_vit_b32", SETTINGS, self.selection,
            data_root=self.data_root, output_dir=self.root / output_name,
            backend_factory=factory, memory_probe=self.probe, clock=self.clock,
            inference_metadata={"fixture": True, "checkpoint_revision": "synthetic-fixture"},
            fixture=True,
        )

    def test_whole_frame_timing_five_warmups_two_repeats(self):
        report = self.run_profile()
        self.assertEqual(len(self.backends), 1)
        backend = self.backends[0]
        self.assertEqual(len(backend.calls), 45)
        self.assertEqual(backend.close_count, 1)
        self.assertEqual(report["load"]["elapsed_s"], 2)
        warmup = [row for row in report["samples"] if row["phase"] == "warmup"]
        measured = [row for row in report["samples"] if row["phase"] == "measured"]
        self.assertEqual(len(warmup), 5)
        self.assertEqual(len(measured), 40)
        self.assertEqual({row["repeat"] for row in measured}, {0, 1})
        self.assertEqual({row["frame_id"] for row in measured}, {entry["frame"]["frame_id"] for entry in self.selection["test_frames"]})
        self.assertTrue(all(row["region_count"] == 2 for row in measured))
        self.assertTrue(all(math.isclose(row["elapsed_s"], 0.8, abs_tol=1e-10) for row in measured))
        self.assertAlmostEqual(report["summary"]["latency"]["median_s"], 0.8)
        self.assertEqual(report["summary"]["requested_regions"], 80)
        self.assertEqual(report["summary"]["warmup"]["attempted_frames"], 5)
        self.assertTrue(report["summary"]["complete"])
        self.assertGreaterEqual(self.probe.synchronizations, 2 * 45)

    def test_settings_are_copied_and_no_automatic_fallback_occurs(self):
        original = copy.deepcopy(SETTINGS)
        self.run_profile(mutate_settings=True)
        self.assertEqual(SETTINGS, original)
        self.assertEqual(self.factory_calls, [("clip_vit_b32", original)])
        self.assertEqual(self.backends[0].close_count, 1)

    def test_ids_timestamps_and_unselected_planning_targets_are_preserved(self):
        report = self.run_profile()
        expected = {entry["frame"]["frame_id"]: entry for entry in self.selection["test_frames"]}
        for sample in report["samples"]:
            if sample["phase"] != "measured":
                continue
            entry = expected[sample["frame_id"]]
            self.assertEqual(sample["timestamp_s"], entry["frame"]["timestamp_s"])
            self.assertEqual(sample["requested_region_ids"], [row["region_id"] for row in entry["regions"]])
        self.assertTrue(all(len(regions) == 2 for _, regions, _ in self.backends[0].calls))

    def test_missing_gpu_statistics_are_null_and_fixture_outputs_are_marked(self):
        report = self.run_profile()
        self.assertTrue(report["metadata"]["fixture"])
        for sample in report["samples"]:
            for key, value in sample["memory"].items():
                if key.endswith("_bytes"):
                    self.assertIsNone(value, key)
        output = self.root / "output"
        for name in ("selection.json", "samples.jsonl", "summary.json", "metadata.json"):
            self.assertTrue((output / name).is_file(), name)
        self.assertTrue(json.loads((output / "metadata.json").read_text(encoding="utf-8"))["fixture"])
        rows = [json.loads(line) for line in (output / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 45)

    def test_allocator_peaks_are_distinct_from_shared_device_baseline(self):
        self.probe = Probe(available=True)
        report = self.run_profile()
        baseline = report["metadata"]["memory_baseline"]
        self.assertEqual(baseline["allocated_bytes"], 100)
        self.assertEqual(baseline["reserved_bytes"], 200)
        self.assertEqual(baseline["device_used_bytes"], 1000)
        memory = report["summary"]["memory"]
        self.assertEqual(memory["allocated_peak_bytes"], 500)
        self.assertEqual(memory["reserved_peak_bytes"], 600)
        self.assertEqual(memory["incremental_allocated_peak_bytes"], 400)
        self.assertEqual(memory["incremental_reserved_peak_bytes"], 400)
        self.assertEqual(memory["device_used_peak_bytes"], 1000)
        self.assertEqual(memory["incremental_device_used_peak_bytes"], 0)
        self.assertEqual(memory["device_total_bytes"], 20000)
        self.assertTrue(memory["device_peak_is_sampled"])
        self.assertGreaterEqual(self.probe.resets, 1)

    def test_unavailable_gpu_cannot_pass_the_budget(self):
        report = self.run_profile()
        memory = report["summary"]["memory"]
        for key in (
            "allocated_peak_bytes", "reserved_peak_bytes", "device_used_peak_bytes",
            "device_total_bytes", "incremental_allocated_peak_bytes",
            "incremental_reserved_peak_bytes", "incremental_device_used_peak_bytes",
        ):
            self.assertIsNone(memory[key], key)
        for key in ("allocated_within_target", "reserved_within_target", "sampled_device_within_target"):
            self.assertIsNone(report["summary"]["budget"][key], key)
        self.assertEqual(report["summary"]["full_pipeline_validation"], "pending")

    def test_snapshot_failure_preserves_total_allocator_peaks_but_not_incrementals(self):
        self.probe = SnapshotFailureProbe()
        report = self.run_profile()
        memory = report["summary"]["memory"]
        self.assertEqual(report["summary"]["measured_frames"], 40)
        self.assertEqual(report["summary"]["failed_regions"], 0)
        self.assertEqual(memory["allocated_peak_bytes"], 500)
        self.assertEqual(memory["reserved_peak_bytes"], 600)
        for key in ("device_used_peak_bytes", "device_total_bytes", "incremental_allocated_peak_bytes", "incremental_reserved_peak_bytes", "incremental_device_used_peak_bytes"):
            self.assertIsNone(memory[key], key)
        self.assertTrue(report["metadata"]["memory_baseline_errors"])
        self.assertTrue(any("snapshot" in reason for reason in memory["unavailable_reasons"]))
        self.assertIsNone(report["summary"]["budget"]["allocated_within_target"])
        self.assertEqual(self.backends[0].close_count, 1)

    def test_partial_device_statistics_do_not_hide_valid_allocator_measurements(self):
        self.probe = PartialDeviceProbe()
        report = self.run_profile()
        memory = report["summary"]["memory"]
        self.assertEqual(memory["allocated_peak_bytes"], 500)
        self.assertEqual(memory["reserved_peak_bytes"], 600)
        self.assertEqual(memory["incremental_allocated_peak_bytes"], 400)
        self.assertEqual(memory["incremental_reserved_peak_bytes"], 400)
        for key in ("device_used_peak_bytes", "device_total_bytes", "incremental_device_used_peak_bytes"):
            self.assertIsNone(memory[key], key)
        self.assertTrue(any("device" in reason for reason in memory["unavailable_reasons"]))
        self.assertIsNone(report["summary"]["budget"]["sampled_device_within_target"])
        self.assertIsNone(report["summary"]["budget"]["allocated_within_target"])

    def test_recovered_telemetry_cannot_invent_a_missing_initial_baseline(self):
        self.probe = BaselineOnlyFailureProbe()
        report = self.run_profile()
        baseline = report["metadata"]["memory_baseline"]
        self.assertTrue(all(value is None for value in baseline.values()))
        self.assertTrue(report["metadata"]["memory_baseline_errors"])
        memory = report["summary"]["memory"]
        self.assertEqual(memory["allocated_peak_bytes"], 500)
        self.assertEqual(memory["reserved_peak_bytes"], 600)
        self.assertEqual(memory["device_used_peak_bytes"], 1000)
        for key in ("incremental_allocated_peak_bytes", "incremental_reserved_peak_bytes", "incremental_device_used_peak_bytes"):
            self.assertIsNone(memory[key], key)
        self.assertEqual(report["summary"]["measured_frames"], 40)

    def test_fixture_mock_gpu_cannot_validate_a_real_component_budget(self):
        self.probe = Probe(available=True)
        report = self.run_profile()
        self.assertEqual(report["summary"]["memory"]["allocated_peak_bytes"], 500)
        self.assertTrue(report["metadata"]["comparison_metadata_complete"])
        self.assertFalse(report["summary"]["comparison_ready"])
        for flag in report["summary"]["budget"].values():
            self.assertIsNone(flag)

    def test_failed_peak_resets_cannot_report_old_allocator_counters(self):
        self.probe = ResetFailureProbe()
        report = self.run_profile()
        self.assertEqual(report["summary"]["measured_frames"], 40)
        for memory in [report["load"]["memory"], report["summary"]["memory"], *[row["memory"] for row in report["samples"]]]:
            for key in ("allocated_peak_bytes", "reserved_peak_bytes", "incremental_allocated_peak_bytes", "incremental_reserved_peak_bytes"):
                self.assertIsNone(memory[key], key)
            self.assertTrue(any("reset_peak" in reason for reason in memory["unavailable_reasons"]))
        self.assertEqual(report["summary"]["memory"]["device_used_peak_bytes"], 1000)
        self.assertIsNone(report["summary"]["budget"]["allocated_within_target"])
        self.assertIsNone(report["summary"]["budget"]["reserved_within_target"])

    def test_failed_pre_call_synchronization_cannot_report_unreset_allocator_peaks(self):
        self.probe = SynchronizationFailureProbe()
        report = self.run_profile()
        self.assertEqual(self.factory_calls, [])
        self.assertEqual(self.probe.resets, 0)
        self.assertIsNotNone(report["load"]["error"])
        self.assertFalse(report["summary"]["complete"])
        for key in ("allocated_peak_bytes", "reserved_peak_bytes", "incremental_allocated_peak_bytes", "incremental_reserved_peak_bytes"):
            self.assertIsNone(report["load"]["memory"][key], key)

    def test_prediction_errors_and_protocol_violations_are_failures(self):
        for mode in ("missing", "duplicate", "wrong_frame", "wrong_model", "invalid_label", "parse_error"):
            with self.subTest(mode=mode):
                report = self.run_profile(mode, output_name=mode)
                self.assertEqual(report["summary"]["measured_frames"], 40)
                self.assertGreaterEqual(report["summary"]["failed_regions"], 40)
                self.assertEqual(report["summary"]["successful_frames"], 0)
                measured = [row for row in report["samples"] if row["phase"] == "measured"]
                self.assertTrue(all(not row["successful"] for row in measured))
                self.assertEqual(self.backends[-1].close_count, 1)

    def test_ordinary_call_failure_continues_and_counts_every_region(self):
        report = self.run_profile("ordinary_exception")
        self.assertEqual(report["summary"]["measured_frames"], 40)
        self.assertEqual(report["summary"]["failed_calls"], 40)
        self.assertEqual(report["summary"]["failed_regions"], 80)
        self.assertEqual(self.backends[0].close_count, 1)

    def test_fatal_gpu_failure_retains_partial_result_and_closes_once(self):
        report = self.run_profile("fatal_exception")
        self.assertFalse(report["summary"]["complete"])
        self.assertEqual(report["summary"]["measured_frames"], 1)
        self.assertEqual(report["summary"]["failed_calls"], 1)
        self.assertEqual(self.backends[0].close_count, 1)
        output = self.root / "output"
        self.assertTrue((output / "summary.json").exists())
        self.assertTrue((output / "samples.jsonl").exists())

    def test_returned_cuda_out_of_memory_stops_without_retry_or_fallback(self):
        report = self.run_profile("returned_oom")
        self.assertEqual(report["summary"]["measured_frames"], 1)
        self.assertEqual(report["summary"]["failed_calls"], 0)
        self.assertEqual(report["summary"]["failed_regions"], 1)
        self.assertFalse(report["summary"]["complete"])
        self.assertIsNotNone(report["metadata"]["fatal_error"])
        self.assertEqual(len(self.factory_calls), 1)
        self.assertEqual(self.backends[0].close_count, 1)

    def test_cleanup_failure_is_recorded_in_readable_partial_artifacts(self):
        report = self.run_profile("cleanup_exception")
        self.assertEqual(report["summary"]["measured_frames"], 40)
        self.assertEqual(self.backends[0].close_count, 1)
        self.assertFalse(report["summary"]["complete"])
        self.assertFalse(report["summary"]["comparison_ready"])
        self.assertEqual(report["metadata"]["cleanup_error"]["type"], "RuntimeError")
        metadata = json.loads((self.root / "output" / "metadata.json").read_text(encoding="utf-8"))
        summary = json.loads((self.root / "output" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["cleanup_error"], report["metadata"]["cleanup_error"])
        self.assertFalse(summary["complete"])

    def test_keyboard_interrupt_preserves_completed_calls_and_releases_own_resources(self):
        self.probe = Probe(available=True)
        report = self.run_profile("keyboard_interrupt")
        self.assertTrue(report["metadata"]["interrupted"])
        self.assertEqual(report["metadata"]["fatal_error"]["type"], "KeyboardInterrupt")
        self.assertEqual(report["summary"]["measured_frames"], 1)
        self.assertFalse(report["summary"]["complete"])
        self.assertEqual(self.backends[0].close_count, 1)
        self.assertFalse(any(thread.name == "traversability-memory" for thread in threading.enumerate()))
        rows = [json.loads(line) for line in (self.root / "output" / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 6)
        self.assertEqual(self.run_profile(output_name="after_interruption")["summary"]["measured_frames"], 40)

    def test_unhashable_prediction_values_are_contract_failures_and_do_not_abort(self):
        for mode in ("unhashable_label", "unhashable_status"):
            with self.subTest(mode=mode):
                report = self.run_profile(mode, output_name=mode)
                self.assertEqual(report["summary"]["measured_frames"], 40)
                self.assertEqual(report["summary"]["failed_regions"], 40)
                self.assertEqual(report["summary"]["failed_calls"], 0)
                self.assertEqual(report["summary"]["successful_frames"], 0)
                self.assertIsNone(report["metadata"]["fatal_error"])
                self.assertEqual(self.backends[-1].close_count, 1)

    def test_unrequested_extra_predictions_invalidate_the_frame_contract(self):
        report = self.run_profile("extra_prediction")
        self.assertEqual(report["summary"]["measured_frames"], 40)
        self.assertEqual(report["summary"]["failed_regions"], 80)
        self.assertEqual(report["summary"]["failed_calls"], 0)
        self.assertEqual(report["summary"]["successful_frames"], 0)
        measured = [row for row in report["samples"] if row["phase"] == "measured"]
        self.assertTrue(all(row["diagnostics"] for row in measured))

    def test_loading_failure_is_separate_from_frame_samples(self):
        def factory(model_key, settings):
            self.clock.advance(3)
            raise RuntimeError("Synthetic loading failure")

        report = profile_backend(
            "clip_vit_b32", SETTINGS, self.selection,
            data_root=self.data_root, output_dir=self.root / "load_failure",
            backend_factory=factory, memory_probe=self.probe, clock=self.clock, fixture=True,
        )
        self.assertEqual(report["samples"], [])
        self.assertEqual(report["load"]["elapsed_s"], 3)
        self.assertIsNotNone(report["load"]["error"])
        self.assertFalse(report["summary"]["complete"])

    def test_changed_inputs_are_rejected_before_backend_loading(self):
        image = self.data_root / self.selection["test_frames"][0]["frame"]["image_path"]
        image.write_bytes(_png_bytes(99))
        with self.assertRaises(ValueError):
            self.run_profile()
        self.assertEqual(self.factory_calls, [])

    def test_existing_output_directory_is_preserved(self):
        output = self.root / "output"
        output.mkdir()
        sentinel = output / "existing-result.json"
        sentinel.write_text('{"preserve": true}\n', encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.run_profile()
        self.assertEqual(sentinel.read_text(encoding="utf-8"), '{"preserve": true}\n')
        self.assertEqual(self.factory_calls, [])

    def test_fixture_requires_explicit_profile_flag(self):
        with self.assertRaises(ValueError):
            profile_backend(
                "clip_vit_b32", SETTINGS, self.selection,
                data_root=self.data_root, output_dir=self.root / "unmarked",
                backend_factory=FakeBackend, memory_probe=Probe(), fixture=False,
            )

    def test_injected_factory_cannot_enter_real_model_profiling(self):
        calls = []

        def factory(model_key, settings):
            calls.append(model_key)
            return FakeBackend(model_key, settings)

        # No real manifests are needed: reject the execution mode before loading.
        with self.assertRaisesRegex(ValueError, "[Ff]ixture"):
            profile_backend(
                "clip_vit_b32", SETTINGS, {"fixture": False},
                data_root=self.data_root, output_dir=self.root / "fake_as_real",
                backend_factory=factory, memory_probe=Probe(), fixture=False,
            )
        self.assertEqual(calls, [])
        self.assertFalse((self.root / "fake_as_real").exists())

    def test_prepared_fixture_and_fake_backend_can_profile_without_inference(self):
        run_dir, data_root = self.root / "demo_run", self.root / "demo_data"
        selection = prepare_fixture(run_dir, data_root)
        self.assertTrue(selection["fixture"])
        report = profile_backend(
            "clip_vit_b32", SETTINGS, selection,
            data_root=data_root, output_dir=self.root / "demo_output",
            backend_factory=FakeBackend, memory_probe=Probe(), fixture=True,
        )
        self.assertEqual(report["summary"]["measured_frames"], 40)
        self.assertEqual(report["summary"]["failed_regions"], 0)
        self.assertTrue(report["metadata"]["fixture"])


if __name__ == "__main__":
    unittest.main()
