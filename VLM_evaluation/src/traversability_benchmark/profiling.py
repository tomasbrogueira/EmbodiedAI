"""Explicit, one-backend-at-a-time whole-frame measurements."""

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from importlib import metadata as package_metadata
import json
import math
from pathlib import Path
import platform
import threading
import time

from .memory import NullMemoryProbe, TorchMemoryProbe, SNAPSHOT_KEYS, _DeviceSampler, _maximum, memory_report
from .selection import _fingerprint

_PROFILE_LOCK = threading.Lock()
_MODELS = {"clip_vit_b32", "qwen3_vl_4b", "qwen3_5_4b", "qwen3_5_2b"}
_BUDGET_BYTES = 6_000_000_000


def _quantile(values, quantile):
    ordered = sorted(values)
    if not ordered:
        return None
    index = (len(ordered) - 1) * quantile
    lower, upper = math.floor(index), math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _latency(samples):
    values = [sample["elapsed_s"] for sample in samples]
    return {"median_s": _quantile(values, 0.5), "p95_s": _quantile(values, 0.95)}


def _successful(sample):
    return bool(sample.get("successful", not sample.get("call_error") and sample.get("failed_regions", 0) == 0 and not sample.get("diagnostics")))


def aggregate_samples(samples):
    """Aggregate measured attempts with linear quantiles, retaining failed calls."""
    measured = [s for s in samples if s.get("phase", "measured") == "measured"]
    for sample in measured:
        duration = sample.get("elapsed_s")
        if not isinstance(duration, (int, float)) or isinstance(duration, bool) or not math.isfinite(duration) or duration < 0:
            raise ValueError("Sample elapsed_s must be finite and nonnegative")
        count, failures = sample.get("region_count", 0), sample.get("failed_regions", 0)
        if type(count) is not int or type(failures) is not int or not 0 <= failures <= count:
            raise ValueError("Sample region/failure counts must be integers with 0 <= failures <= count")
    succeeded = [s for s in measured if _successful(s)]
    by_frame = defaultdict(list)
    for sample in measured:
        by_frame[sample["frame_id"]].append(sample)
    return {
        "measured_frames": len(measured), "expected_measured_frames": 40,
        "unique_test_frames": len(by_frame),
        "requested_regions": sum(s.get("region_count", 0) for s in measured),
        "failed_regions": sum(s.get("failed_regions", 0) for s in measured),
        "failed_calls": sum(bool(s.get("call_error")) for s in measured),
        "failed_frames": len(measured) - len(succeeded), "successful_frames": len(succeeded),
        "latency": _latency(measured), "successful_latency": _latency(succeeded),
        "quantile_method": "linear interpolation at (n-1)*q",
        "per_frame": [{"frame_id": identifier, "region_count": rows[0].get("region_count", 0), **_latency(rows), "failures": sum(not _successful(s) for s in rows)} for identifier, rows in sorted(by_frame.items())],
        "warmup": {
            "attempted_frames": sum(s.get("phase") == "warmup" for s in samples),
            "failed_frames": sum(s.get("phase") == "warmup" and not _successful(s) for s in samples),
        },
        "complete": len(measured) == 40 and len(by_frame) == 20,
    }


def _validate_selection(selection, data_root, fixture):
    if selection.get("fixture") is not fixture:
        raise ValueError("Selection fixture flag must match explicit profiling mode")
    entries = []
    region_ids = set()
    for key, count, split in (("warmup_frames", 5, "development"), ("test_frames", 20, "test")):
        group = selection.get(key, [])
        if len(group) != count:
            raise ValueError(f"Profiling requires exactly {count} {key}")
        for entry in group:
            frame, regions = entry["frame"], entry["regions"]
            if frame.get("split") != split or not regions:
                raise ValueError("Invalid selected split or no planning-relevant regions")
            if not fixture and (frame.get("fixture") or str(frame.get("source", "")).lower().startswith(("fixture", "synthetic", "fake"))):
                raise ValueError("Synthetic frames cannot be presented as real profiling results")
            for region in regions:
                identifier = region["region_id"]
                if identifier in region_ids or region.get("frame_id") != frame["frame_id"] or region.get("planning_relevant") is not True:
                    raise ValueError("Duplicate, misjoined or non-planning region in selection")
                region_ids.add(identifier)
            entries.append(entry)
    if len({e["frame"]["frame_id"] for e in entries}) != 25:
        raise ValueError("Selected frame IDs must be unique and splits disjoint")
    if _fingerprint(entries, data_root) != selection.get("fingerprint"):
        raise ValueError("Selected records or image/mask files changed since freezing")


def _error(error):
    return {"type": type(error).__name__, "message": str(error)}


def _fatal(error):
    text = f"{error.get('type', '')}: {error.get('message', '')}".lower()
    return any(marker in text for marker in ("outofmemoryerror", "cuda out of memory", "cuda error", "device-side assert", "illegal memory access", "device lost", "cublas_status_alloc_failed"))


def _predictions_diagnostics(predictions, frame, regions, model_key):
    expected = {r["region_id"] for r in regions}
    seen = Counter()
    failed = set()
    diagnostics, returned_errors = [], []
    if not isinstance(predictions, list):
        return len(expected), ["Backend must return a list of prediction records"], []
    for prediction in predictions:
        if not isinstance(prediction, dict):
            diagnostics.append("Non-object prediction")
            failed.update(expected)
            continue
        identifier = prediction.get("region_id")
        if not isinstance(identifier, str) or identifier not in expected:
            diagnostics.append(f"Unexpected region_id: {identifier!r}")
            # An invalid extra result still invalidates this frame's output contract.
            failed.update(expected)
            continue
        seen[identifier] += 1
        invalid = (
            prediction.get("frame_id") != frame["frame_id"]
            or prediction.get("model_key") != model_key
            or prediction.get("label") not in ("traversable", "non_traversable", "unknown")
            or prediction.get("status") not in ("ok", "error")
            or (prediction.get("status") == "error" and prediction.get("label") != "unknown")
            or (prediction.get("status") == "error" and not prediction.get("error_code"))
            or (prediction.get("status") == "ok" and prediction.get("error_code") is not None)
        )
        if invalid:
            failed.add(identifier)
            diagnostics.append(f"Invalid prediction contract: {identifier}")
        if prediction.get("status") == "error":
            failed.add(identifier)
            returned_errors.append({"region_id": identifier, "error_code": prediction.get("error_code"), "reason": prediction.get("reason", "")})
    for identifier in expected:
        if seen[identifier] != 1:
            failed.add(identifier)
            diagnostics.append(f"Expected one prediction for {identifier}; received {seen[identifier]}")
    return len(failed), diagnostics, returned_errors


def _timed_call(function, probe, baseline, clock):
    errors = []
    sampler = _DeviceSampler(probe)
    value, error, started = None, None, None
    allocator_peaks_valid = False
    try:
        probe.synchronize()
        try:
            probe.reset_peak()
            allocator_peaks_valid = True
        except Exception as problem:
            errors.append(f"reset_peak: {type(problem).__name__}: {problem}")
        sampler.start()
        started = clock()
        try:
            value = function()
        except Exception as problem:
            error = _error(problem)
        finally:
            try:
                probe.synchronize()
            except Exception as problem:
                error = _error(problem)
                error["timing_invalid"] = True
            duration = clock() - started
    except Exception as problem:
        error = _error(problem)
        error["timing_invalid"] = True
        duration = 0.0 if started is None else clock() - started
    finally:
        sampler.stop()
    return value, duration, error, memory_report(probe, sampler, baseline, errors, allocator_peaks_valid)


def _merge_memory(reports):
    keys = ("allocated_peak_bytes", "reserved_peak_bytes", "device_used_peak_bytes", "device_total_bytes", "incremental_allocated_peak_bytes", "incremental_reserved_peak_bytes", "incremental_device_used_peak_bytes")
    return {
        **{key: _maximum(r.get(key) for r in reports) for key in keys},
        "device_peak_is_sampled": True, "sampling_interval_s": 0.02,
        "unavailable_reasons": list(dict.fromkeys(reason for report in reports for reason in report.get("unavailable_reasons", []))),
    }


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _versions():
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("torch", "torchvision", "transformers", "accelerate", "bitsandbytes", "causal-conv1d", "flash-linear-attention", "flash-attn"):
        try:
            versions[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _default_probe(settings, fixture):
    if fixture:
        return NullMemoryProbe()
    try:
        return TorchMemoryProbe(settings.get("device", "cuda:0"))
    except (ImportError, RuntimeError) as error:
        probe = NullMemoryProbe()
        probe.unavailable_reason = f"{type(error).__name__}: {error}"
        return probe


def _require_idle_kernel(baseline):
    # Current peer adapters empty the process-wide CUDA cache on close. Refuse
    # resident Torch resources before loading rather than altering their cleanup.
    if any(baseline.get(key) is not None and baseline[key] > 0 for key in ("allocated_bytes", "reserved_bytes")):
        raise RuntimeError("Real profiling requires a dedicated idle kernel with no resident Torch CUDA allocations or cached blocks. Other modules may remain resident in separate processes on the shared device. Restart a dedicated kernel; shared-kernel profiling awaits the inference cleanup fix.")


def _metadata_details(backend, supplied, model_key, requested, factory_settings):
    actual = deepcopy(getattr(backend, "metadata", None)) if backend else None
    resolved = deepcopy(getattr(backend, "settings", None)) if backend else None
    actual = actual if isinstance(actual, dict) else None
    resolved = resolved if isinstance(resolved, dict) else None
    missing = []
    if not actual or not actual.get("revision"):
        missing.append("actual_checkpoint_revision")
    if not actual or not (actual.get("processor") or actual.get("preprocessing")):
        missing.append("actual_processor_settings")
    if not actual or "quantization" not in actual:
        missing.append("actual_quantization")
    if resolved is None:
        missing.append("actual_resolved_settings")
    published_configuration = (supplied or {}).get("configuration", {})
    if model_key.startswith("qwen") and not ((actual or {}).get("prompt") or published_configuration.get("prompt")):
        missing.append("actual_prompt")
    if model_key == "clip_vit_b32" and not ((actual or {}).get("vocabulary") and (actual or {}).get("calibration")):
        missing.append("actual_clip_vocabulary_and_calibration")
    sidecar_mismatches = []
    if supplied:
        if supplied.get("model_key", model_key) != model_key:
            sidecar_mismatches.append("model_key")
        published_actual = supplied.get("actual_backend")
        if isinstance(published_actual, dict) and actual is not None and published_actual != actual:
            sidecar_mismatches.append("actual_backend")
        published_settings = supplied.get("configuration", {}).get("settings", supplied.get("settings", {}))
        if isinstance(published_settings, dict):
            for key, value in published_settings.items():
                if key in (resolved or requested) and (resolved or requested)[key] != value:
                    sidecar_mismatches.append(f"settings.{key}")
    if factory_settings != requested:
        missing.append("loader_mutated_requested_settings")
    return {"actual_backend": actual, "actual_settings": resolved, "comparison_metadata_complete": not missing and not sidecar_mismatches, "missing_comparison_metadata": missing, "inference_metadata_mismatches": sidecar_mismatches}


def profile_backend(model_key, settings, selection, *, data_root, output_dir,
                    backend_factory=None, memory_probe=None, clock=None,
                    inference_metadata=None, fixture=False):
    """Load one owned backend and measure five warmups plus 40 whole-frame calls."""
    if not isinstance(settings, dict):
        raise TypeError("settings must be a dict")
    if not fixture and model_key not in _MODELS:
        raise ValueError(f"Unknown real model key: {model_key}")
    if backend_factory is not None and not fixture:
        raise ValueError("Injected backend factories require fixture=True; synthetic responses cannot count as model measurements")
    if fixture and backend_factory is None:
        raise ValueError("Fixture profiling requires an explicitly injected backend factory")
    _validate_selection(selection, data_root, fixture)
    if not _PROFILE_LOCK.acquire(blocking=False):
        raise RuntimeError("Another benchmark is active in this process; profile one backend at a time")
    try:
        return _profile(model_key, settings, selection, Path(data_root), Path(output_dir), backend_factory, memory_probe, clock or time.perf_counter, inference_metadata, fixture)
    finally:
        _PROFILE_LOCK.release()


def _profile(model_key, settings, selection, data_root, output_dir, factory, probe, clock, supplied_metadata, fixture):
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "selection.json", selection)
    requested = deepcopy(settings)
    factory_settings = deepcopy(settings)
    probe = probe if probe is not None else _default_probe(requested, fixture)
    baseline_errors = []
    try:
        probe.synchronize()
        baseline = probe.snapshot()
    except Exception as problem:
        baseline = dict.fromkeys(SNAPSHOT_KEYS)
        baseline_errors.append(f"baseline: {type(problem).__name__}: {problem}")
    metadata = {
        "schema_version": 1, "fixture": fixture, "execution_kind": "injected_fixture" if fixture else "real_model",
        "model_key": model_key, "requested_settings": requested,
        "selection_fingerprint": selection["fingerprint"], "coverage": selection.get("coverage", {}),
        "software_versions": _versions(), "started_at": datetime.now(timezone.utc).isoformat(),
        "memory_baseline": baseline, "memory_baseline_errors": baseline_errors,
        "memory_attribution": "Allocator counters are process-wide; sampled device usage includes other processes. Baseline precedes backend loading.",
        "inference_metadata": deepcopy(supplied_metadata), "sampling_interval_s": 0.02,
        "settings_changed_during_profile": False, "cleanup_error": None, "fatal_error": None,
        "resource_isolation": "fixture backend owns its resources" if fixture else "dedicated idle Torch kernel required; other device processes remain untouched",
        "baseline_scope": "after telemetry/CUDA context initialization, before backend loading; initialization overhead is not attributed by this baseline",
    }
    try:
        describe = getattr(probe, "describe", None)
        metadata["gpu_identity"] = describe() if callable(describe) else None
        metadata["gpu_identity_error"] = None
    except Exception as problem:
        metadata["gpu_identity"] = None
        metadata["gpu_identity_error"] = _error(problem)
    backend = None
    samples = []
    load = {"elapsed_s": None, "error": None, "memory": {}}
    _write_json(output_dir / "metadata.json", metadata)

    def load_one():
        nonlocal backend
        if not fixture:
            _require_idle_kernel(baseline)
        if factory is None:
            from traversability_inference import load_backend
            selected_factory = load_backend
        else:
            selected_factory = factory
        backend = selected_factory(model_key, factory_settings)
        if not callable(getattr(backend, "predict_frame", None)) or not callable(getattr(backend, "close", None)):
            raise TypeError("Loaded backend must provide predict_frame and close")
        return backend

    try:
        _, duration, problem, measured_memory = _timed_call(load_one, probe, baseline, clock)
        load.update(elapsed_s=duration, error=problem, memory=measured_memory)
        metadata.update(_metadata_details(backend, supplied_metadata, model_key, requested, factory_settings))
        if problem is not None:
            metadata["fatal_error"] = problem
        else:
            schedule = [("warmup", None, index, entry) for index, entry in enumerate(selection["warmup_frames"])]
            schedule += [("measured", repeat, None, entry) for repeat in range(2) for entry in selection["test_frames"]]
            initial_actual_settings = deepcopy(getattr(backend, "settings", None))
            initial_actual_metadata = deepcopy(getattr(backend, "metadata", None))
            with (output_dir / "samples.jsonl").open("x", encoding="utf-8") as stream:
                for phase, repeat, warmup_index, entry in schedule:
                    frame, regions = entry["frame"], entry["regions"]
                    frame_input, region_inputs = deepcopy(frame), deepcopy(regions)
                    predictions, duration, problem, measured_memory = _timed_call(
                        lambda: backend.predict_frame(frame_input, region_inputs, data_root), probe, baseline, clock,
                    )
                    failed, diagnostics, prediction_errors = (len(regions), [], []) if problem else _predictions_diagnostics(predictions, frame, regions, model_key)
                    changed = getattr(backend, "settings", None) != initial_actual_settings or getattr(backend, "metadata", None) != initial_actual_metadata
                    if changed:
                        metadata["settings_changed_during_profile"] = True
                        diagnostics.append("Backend configuration changed during profiling")
                    sample = {
                        "phase": phase, "repeat": repeat, "warmup_index": warmup_index,
                        **{key: frame.get(key) for key in ("frame_id", "source", "scene_id", "sequence_id", "timestamp_s")},
                        "region_count": len(regions), "requested_region_ids": [r["region_id"] for r in regions],
                        "elapsed_s": duration, "failed_regions": failed, "call_error": problem,
                        "diagnostics": diagnostics, "prediction_errors": prediction_errors,
                        "successful": not problem and not failed and not diagnostics, "memory": measured_memory,
                    }
                    samples.append(sample)
                    stream.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    returned_fatal = next((e for e in prediction_errors if _fatal({"type": str(e["error_code"]), "message": e["reason"]})), None)
                    if changed or (problem and (_fatal(problem) or problem.get("timing_invalid"))) or returned_fatal:
                        metadata["fatal_error"] = problem or returned_fatal or {"type": "ConfigurationChanged", "message": "Backend settings changed"}
                        break
    except BaseException as problem:
        metadata["fatal_error"] = _error(problem)
        if isinstance(problem, (KeyboardInterrupt, SystemExit)):
            metadata["interrupted"] = True
    finally:
        if backend is not None:
            try:
                backend.close()
            except BaseException as problem:
                metadata["cleanup_error"] = _error(problem)
        if not (output_dir / "samples.jsonl").exists():
            (output_dir / "samples.jsonl").touch(exist_ok=False)
        summary = aggregate_samples(samples)
        reports = ([load["memory"]] if load["memory"] else []) + [s["memory"] for s in samples]
        summary["memory"] = _merge_memory(reports)
        summary["phase_memory"] = {
            "loading": load["memory"],
            "cold_start": next((s["memory"] for s in samples if s.get("warmup_index") == 0), None),
            "warmup": _merge_memory([s["memory"] for s in samples if s["phase"] == "warmup"]),
            "warmed_inference": _merge_memory([s["memory"] for s in samples if s["phase"] == "measured"]),
        }
        summary["cold_start_elapsed_s"] = next((s["elapsed_s"] for s in samples if s.get("warmup_index") == 0), None)
        summary["target_budget_bytes"] = _BUDGET_BYTES
        summary["budget"] = {name: None if fixture or summary["memory"][key] is None else summary["memory"][key] <= _BUDGET_BYTES for name, key in (
            ("allocated_within_target", "incremental_allocated_peak_bytes"),
            ("reserved_within_target", "incremental_reserved_peak_bytes"),
            ("sampled_device_within_target", "incremental_device_used_peak_bytes"),
        )}
        summary["full_pipeline_validation"] = "pending"
        summary["complete"] = summary["complete"] and metadata["fatal_error"] is None and metadata["cleanup_error"] is None
        summary["comparison_ready"] = summary["complete"] and summary["failed_frames"] == 0 and summary["warmup"]["failed_frames"] == 0 and metadata.get("comparison_metadata_complete", False) and not fixture
        metadata.update(finished_at=datetime.now(timezone.utc).isoformat(), load=load)
        _write_json(output_dir / "summary.json", summary)
        _write_json(output_dir / "metadata.json", metadata)
    return {"metadata": metadata, "summary": summary, "samples": samples, "load": load}
