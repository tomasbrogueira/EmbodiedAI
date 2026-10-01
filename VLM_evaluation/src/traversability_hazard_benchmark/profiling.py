"""Atomic hazard profiling, with real models isolated in a fresh worker process."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from importlib import metadata as distribution_metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import threading
import time

from traversability_data.storage import exclusive_lock, read_json, read_jsonl
from traversability_hazard_segmentation.common import atomic_json, atomic_jsonl
from traversability_benchmark.memory import (
    NullMemoryProbe, TorchMemoryProbe, SNAPSHOT_KEYS, _DeviceSampler, memory_report,
)

TASK_ID = "hazard_prompt_v1"
SCHEMA_VERSION = 1
_LOCK = threading.Lock()
_MODEL_KEYS = {"qwen3_vl_4b", "qwen3_5_4b"}
_MEMORY_KEYS = (
    "allocated_peak_bytes", "reserved_peak_bytes", "device_used_peak_bytes", "device_total_bytes",
    "incremental_allocated_peak_bytes", "incremental_reserved_peak_bytes", "incremental_device_used_peak_bytes",
)


def _error(error):
    return {"type": type(error).__name__, "message": str(error)}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _maximum(values):
    values = [value for value in values if value is not None]
    return max(values) if values else None


def _quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * q
    lower, upper = math.floor(index), math.ceil(index)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def _latency(rows, key="elapsed_s"):
    values = [row[key] for row in rows if row.get("timing_valid", True) and row.get(key) is not None]
    return {"median_s": _quantile(values, 0.5), "p95_s": _quantile(values, 0.95)}


def _merge_memory(reports):
    reports = [report for report in reports if report]
    return {
        **{key: _maximum(report.get(key) for report in reports) for key in _MEMORY_KEYS},
        "device_peak_is_sampled": True,
        "sampling_interval_s": 0.02,
        "unavailable_reasons": list(dict.fromkeys(
            reason for report in reports for reason in report.get("unavailable_reasons", [])
        )) or (["No memory measurements recorded"] if not reports else []),
    }


def aggregate_samples(samples, *, expected_measured_frames=40, expected_unique_frames=20):
    """Keep failures in attempt latency, with separate successful/failed summaries."""
    measured = [sample for sample in samples if sample.get("phase") == "measured"]
    identifiers = set()
    for sample in measured:
        identifier = (sample.get("frame_id"), sample.get("repeat"))
        if identifier in identifiers:
            raise ValueError("Duplicate completed measured frame/repeat")
        identifiers.add(identifier)
        elapsed = sample.get("elapsed_s")
        if elapsed is not None and (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
                                    or not math.isfinite(elapsed) or elapsed < 0):
            raise ValueError("elapsed_s must be null or finite and nonnegative")
        if elapsed is None and sample.get("timing_valid", True):
            raise ValueError("Unavailable elapsed time must have timing_valid=false")
        count = sample.get("sam_query_count", 0)
        if count is not None and (type(count) is not int or count < 0):
            raise ValueError("sam_query_count must be a nonnegative integer or explicitly unavailable")
        if sample.get("status") not in ("ok", "error"):
            raise ValueError("A completed attempt requires ok/error status")
    successful = [sample for sample in measured if sample["status"] == "ok"]
    failed = [sample for sample in measured if sample["status"] == "error"]
    frame_ids = {sample["frame_id"] for sample in measured}
    return {
        "measured_frames": len(measured), "expected_measured_frames": expected_measured_frames,
        "unique_test_frames": len(frame_ids), "failed_calls": len(failed),
        "successful_calls": len(successful),
        "valid_measured_calls": sum(sample.get("timing_valid", True) and sample.get("elapsed_s") is not None for sample in measured),
        "invalid_timing_calls": sum(not sample.get("timing_valid", True) and sample.get("call_attempted", True) for sample in measured),
        "unattempted_calls": sum(sample.get("call_attempted") is False for sample in measured),
        "sam_query_count": sum(sample.get("sam_query_count") or 0 for sample in measured),
        "unavailable_sam_query_counts": sum(sample.get("sam_query_count") is None for sample in measured),
        "requested_prompt_count": sum(sample.get("requested_prompt_count", 0) for sample in measured),
        "latency": _latency(measured), "successful_latency": _latency(successful),
        "failed_attempt_latency": _latency(failed),
        "quantile_method": "linear interpolation at (n-1)*q; all valid measured attempts",
        "stage_latency": {
            stage: _latency([{**sample, "stage": sample.get("stage_times_s", {}).get(stage)} for sample in measured], "stage")
            for stage in ("vlm", "sam")
        },
        "warmup": {"attempted_calls": sum(sample.get("phase") == "warmup" for sample in samples),
                   "failed_calls": sum(sample.get("phase") == "warmup" and sample.get("status") == "error" for sample in samples)},
        "memory": _merge_memory(sample.get("memory") for sample in measured),
        "complete": len(measured) == expected_measured_frames and len(frame_ids) == expected_unique_frames,
    }


def _snapshot(probe):
    try:
        return probe.snapshot(), []
    except Exception as error:
        return dict.fromkeys(SNAPSHOT_KEYS), [f"snapshot: {type(error).__name__}: {error}"]


def _timed_call(function, probe, baseline, clock):
    """Synchronize a whole call, retaining null timing if synchronization fails."""
    sampler = _DeviceSampler(probe)
    value = problem = elapsed = None
    peak_valid = False
    telemetry_errors = []
    timing_valid = False
    try:
        probe.synchronize()
        try:
            probe.reset_peak()
            peak_valid = True
        except Exception as error:
            telemetry_errors.append(f"reset_peak: {type(error).__name__}: {error}")
        sampler.start()
        started = clock()
        try:
            value = function()
        except BaseException as error:
            problem = _error(error)
        try:
            probe.synchronize()
            elapsed = clock() - started
            if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
                raise RuntimeError("Invalid elapsed wall time")
            timing_valid = True
        except Exception as error:
            problem = {**_error(error), "timing_invalid": True, "call_error": problem}
            elapsed = None
    except BaseException as error:
        problem = {**_error(error), "timing_invalid": True}
    finally:
        sampler.stop()
    memory = memory_report(probe, sampler, baseline, telemetry_errors, peak_valid)
    return value, elapsed, problem, memory, timing_valid


def _stage_call(function, probe, clock, times, stage):
    # Never reset peaks here: the outer combined interval owns its counters.
    probe.synchronize()
    started = clock()
    try:
        return function()
    finally:
        probe.synchronize()
        times[stage] = clock() - started


def _versions():
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("torch", "torchvision", "sam3", "transformers", "accelerate", "bitsandbytes", "numpy", "Pillow"):
        try:
            versions[name] = distribution_metadata.version(name)
        except distribution_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _safe_name(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) or value in (".", ".."):
        raise ValueError(f"{label} must be a portable single path component")
    if value.endswith(".") or value.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
        raise ValueError(f"{label} is a reserved Windows path component")
    return value


def _json_path(value):
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    raise TypeError(f"Real profile configuration must contain JSON values, not {type(value).__name__}")


def _fixture_identity(adapter):
    settings, metadata = getattr(adapter, "settings", {}), getattr(adapter, "metadata", {})
    value = getattr(adapter, "fixture_identity", None) or settings.get("fixture_identity") or metadata.get("fixture_id")
    markers = [owner["fixture"] for owner in (settings, metadata) if "fixture" in owner]
    if not isinstance(value, str) or not value or not markers or any(marker is not True for marker in markers):
        raise ValueError("Injected adapters require an explicit fixture marker and fixture_identity")
    return value


def _adapter_identity(adapter):
    from traversability_hazard_segmentation.common import portable_settings
    return portable_settings({"settings": getattr(adapter, "settings", None), "metadata": getattr(adapter, "metadata", None)})


def _prepare(config):
    from traversability_hazard_segmentation.common import load_config, resolve_config, fingerprint, portable_settings
    from traversability_hazard_segmentation.selection import validate_selection
    from traversability_hazard_segmentation.controller import condition_input

    config = load_config(config)
    if config.get("task_id", TASK_ID) != TASK_ID or type(config.get("schema_version", 1)) is not int or config.get("schema_version", 1) != 1:
        raise ValueError("Benchmark task/schema must be hazard_prompt_v1 version 1")
    fixture = config.get("fixture", False)
    if type(fixture) is not bool:
        raise ValueError("fixture must be an explicit boolean")
    component = config.get("component", "sam")
    if component not in ("vlm", "sam", "combined"):
        raise ValueError("component must be vlm, sam or combined")
    condition = config.get("condition_key", "fixed_policy")
    conditions = {"fixed_policy", "reference_present", *(f"vlm__{key}" for key in _MODEL_KEYS)}
    if condition not in conditions:
        raise ValueError("Unknown hazard condition_key")
    if component in ("vlm", "combined") and not condition.startswith("vlm__"):
        raise ValueError("VLM/combined profiling requires a VLM condition")
    if component == "combined" and config.get("enable_combined") is not True:
        raise ValueError("Combined profiling requires enable_combined=true")
    if type(config.get("warmup_frames", 5)) is not int or type(config.get("measured_repeats", 2)) is not int or config.get("warmup_frames", 5) != 5 or config.get("measured_repeats", 2) != 2:
        raise ValueError("The frozen protocol requires five warmups and two test repeats")
    profile_id = _safe_name(config.get("profile_id", f"{component}_v1"), "profile_id")
    roots = resolve_config(config)
    run_dir, data_root, policy_path = roots["run_dir"], roots["data_root"], roots["policy_path"]
    selection = read_json(run_dir / "segmentation" / "selection.json")
    inputs = validate_selection(run_dir, data_root, selection, policy_path=policy_path, fixture=fixture)
    test_ids, warmup_ids = selection["test_frame_ids"], selection["warmup_frame_ids"]
    if selection.get("complete") is not True or len(test_ids) != 20 or len(warmup_ids) != 5:
        raise ValueError("Profiling requires a complete frozen 20-test/5-development selection")
    if len(set(test_ids + warmup_ids)) != 25:
        raise ValueError("Selection IDs must be unique and splits disjoint")
    condition_rows = {}
    if component == "sam":
        for frame_id in test_ids + warmup_ids:
            condition_rows[frame_id] = condition_input(inputs, run_dir, condition, inputs["frames"][frame_id])
    settings = {"vlm": config.get("vlm_settings", {}), "sam": config.get("sam_settings", {})}
    for stage in (("vlm", "sam") if component == "combined" else (component,)):
        if not isinstance(settings[stage], dict):
            raise ValueError(f"{stage}_settings must be a dictionary")
        if not fixture and not re.fullmatch(r"cuda:[0-9]+", settings[stage].get("device", "")):
            raise ValueError("Real profiling requires one explicit cuda:N device")
    if component == "combined" and not fixture and settings["vlm"].get("device") != settings["sam"].get("device"):
        raise ValueError("Combined models must use the same explicit device")
    clean = {key: value for key, value in config.items() if not key.startswith("_")}
    identity = {
        "task_id": TASK_ID, "schema_version": 1, "fixture": fixture,
        "component": component, "condition_key": condition,
        "selection_fingerprint": selection["fingerprint"], "input_identity": inputs["identity"],
        "policy_sha256": inputs["policy_sha256"], "alias_sha256": inputs["alias_sha256"],
        "policy_hash": inputs["policy_hash"], "alias_hash": inputs["alias_sha256"],
        "dataset_fingerprint": inputs["dataset"]["dataset_fingerprint"],
        "condition_inputs": portable_settings(condition_rows), "settings": portable_settings(settings),
        "configuration": portable_settings({key: value for key, value in clean.items() if key not in ("repo_root", "component_root", "data_root", "run_root", "cache_root", "run_dir", "policy_path", "profile_id", "resume")}),
    }
    return {
        "config": config, "roots": roots, "selection": selection, "inputs": inputs,
        "condition_rows": condition_rows, "identity": identity, "identity_sha256": fingerprint(identity),
        "output": run_dir / "benchmark" / condition / profile_id,
        "component": component, "condition_key": condition, "fixture": fixture, "settings": settings,
        "model_key": condition.removeprefix("vlm__") if condition.startswith("vlm__") else None,
    }


def _assert_frozen(prepared):
    from traversability_hazard_segmentation.common import fingerprint, safe_path
    from traversability_hazard_segmentation.controller import condition_input
    from traversability_data.storage import hash_file
    roots, inputs = prepared["roots"], prepared["inputs"]
    # Initial validation decoded original dimensions and binary references. Stable
    # records plus equal bytes prove those dimensions remain unchanged without
    # repeatedly decoding every frozen asset outside each measured interval.
    if read_json(roots["run_dir"] / "segmentation/selection.json") != prepared["selection"]:
        raise ValueError("Frozen selection changed")
    if hash_file(roots["policy_path"]) != inputs["policy_sha256"]:
        raise ValueError("Frozen policy hash mismatch")
    if read_json(roots["run_dir"] / "metadata/dataset.json") != inputs["dataset"]:
        raise ValueError("Frozen dataset metadata changed")
    for filename, expected in (("frames.jsonl", inputs["identity"]["frames"]), ("references.jsonl", inputs["identity"]["references"])):
        rows = read_jsonl(roots["run_dir"] / filename)
        if fingerprint(sorted(rows, key=lambda row: row["frame_id"])) != fingerprint(expected):
            raise ValueError("Frozen frame/reference joins changed")
    for relative, expected_sha in inputs["identity"]["assets"].items():
        if hash_file(safe_path(roots["data_root"], relative, True)) != expected_sha:
            raise ValueError("Frozen image/reference asset hash mismatch")
    for frame_id, frozen in prepared["condition_rows"].items():
        current = condition_input(inputs, roots["run_dir"], prepared["condition_key"], inputs["frames"][frame_id])
        if current != frozen:
            raise ValueError("Frozen condition predictions/model identity changed")


def _vlm_result(result, frame, model_key):
    from traversability_hazard_segmentation.common import validate_prompts
    fields = {"task_id", "schema_version", "frame_id", "model_key", "prompts",
              "raw_response", "status", "error_code"}
    if not isinstance(result, dict) or set(result) != fields:
        raise ValueError("VLM prediction must have exactly the eight hazard schema fields")
    if not isinstance(result, dict) or type(result.get("schema_version")) is not int or any(result.get(key) != value for key, value in (
        ("task_id", TASK_ID), ("schema_version", 1), ("frame_id", frame["frame_id"]), ("model_key", model_key),
    )):
        raise ValueError("VLM returned a mismatched whole-image prediction identity")
    if result.get("status") not in ("ok", "error"):
        raise ValueError("VLM returned no terminal status")
    if (result["status"] == "ok" and result.get("error_code") is not None) or (result["status"] == "error" and (not isinstance(result.get("error_code"), str) or not result["error_code"])):
        raise ValueError("VLM status/error_code mismatch")
    if not isinstance(result.get("raw_response"), str):
        raise ValueError("VLM raw_response must be preserved as a string")
    prompts = validate_prompts(result["prompts"])
    if result["status"] == "error" and prompts:
        raise ValueError("Failed VLM response cannot expose usable prompts")
    return {"prompts": prompts, "raw_response": result["raw_response"], "status": result["status"],
            "error_code": result.get("error_code"), "diagnostic": result.get("diagnostic")}


def _sam_result(result, frame, prompts, *, fixture):
    if not isinstance(result, dict) or not isinstance(result.get("frame"), dict) or not isinstance(result.get("queries"), list):
        raise ValueError("SAM must return queries and frame records")
    row, queries = result["frame"], result["queries"]
    if row.get("frame_id") != frame["frame_id"] or row.get("status") not in ("ok", "error"):
        raise ValueError("SAM returned mismatched frame identity/status")
    if (row["status"] == "ok" and row.get("error_code") is not None) or (row["status"] == "error" and (not isinstance(row.get("error_code"), str) or not row["error_code"])):
        raise ValueError("SAM frame status/error_code mismatch")
    if tuple(getattr(row.get("union_mask"), "shape", ())) != (frame["height"], frame["width"]):
        raise ValueError("SAM frame union must have original RGB dimensions")
    if len(queries) != len(prompts) or any(query.get("phrase") != phrase for query, phrase in zip(queries, prompts)):
        raise ValueError("SAM returned missing, reordered or rewritten phrases")
    count = row.get("sam_query_count", result.get("sam_query_count"))
    if type(count) is not int or not 0 <= count <= len(prompts):
        raise ValueError("SAM actual sam_query_count must be audited")
    audit = []
    for query in queries:
        if query.get("status") not in ("ok", "error"):
            raise ValueError("SAM query has no terminal status")
        if (query["status"] == "ok" and query.get("error_code") is not None) or (query["status"] == "error" and (not isinstance(query.get("error_code"), str) or not query["error_code"])):
            raise ValueError("SAM query status/error_code mismatch")
        masks = query.get("masks", [])
        scores = query.get("scores", [])
        if len(masks) != len(scores):
            raise ValueError("SAM masks and scores must align")
        if any(not math.isfinite(float(score)) or not 0 <= float(score) <= 1 for score in scores):
            raise ValueError("SAM scores must be finite probabilities")
        if any(tuple(getattr(mask, "shape", ())) != (frame["height"], frame["width"]) for mask in masks):
            raise ValueError("SAM instance masks must have original RGB dimensions")
        if any(str(getattr(mask, "device", "cpu")) != "cpu" for mask in masks):
            raise ValueError("SAM masks must be copied to CPU")
        audit.append({"phrase": query["phrase"], "status": query["status"], "error_code": query.get("error_code"),
                      "diagnostic": query.get("diagnostic"), "returned_instance_count": len(masks),
                      "scores": [float(score) for score in scores]})
    failed = sum(query["status"] == "error" for query in queries)
    if failed and row["status"] == "ok":
        raise ValueError("Partial SAM query failures cannot be a successful frame")
    return {"status": row["status"], "error_code": row.get("error_code"), "queries": audit,
            "failed_queries": failed, "sam_query_count": count}


def _capture_vlm(prediction, result):
    if isinstance(prediction, dict):
        result["vlm"] = {key: prediction.get(key) for key in ("raw_response", "status", "error_code", "diagnostic")
                         if prediction.get(key) is None or isinstance(prediction.get(key), (str, int, float, bool))}
        original = prediction.get("prompts")
        if isinstance(original, list) and all(isinstance(phrase, str) for phrase in original):
            result["vlm"]["prompts"] = list(original)


def _call(prepared, frame, backend, segmenter, probe, clock, result):
    from traversability_hazard_segmentation.common import adapter_frame
    component = prepared["component"]
    result.update(stage_times_s={}, prompts=[], sam_query_count=0, requested_prompt_count=0,
                  sam_segment_image_calls=0, status="ok", error_code=None)
    if component in ("vlm", "combined"):
        prediction = _stage_call(lambda: backend.predict_image(adapter_frame(frame), prepared["roots"]["data_root"]), probe, clock, result["stage_times_s"], "vlm")
        _capture_vlm(prediction, result)
        vlm = _vlm_result(prediction, frame, prepared["model_key"])
        result.update(vlm=vlm, prompts=vlm["prompts"], requested_prompt_count=len(vlm["prompts"]))
        if vlm["status"] != "ok":
            result.update(status="error", error_code=vlm.get("error_code") or "upstream_vlm_error")
            return result
    else:
        condition = prepared["condition_rows"][frame["frame_id"]]
        result.update(prompts=condition["prompts"], requested_prompt_count=len(condition["prompts"]))
        if condition.get("prediction") is not None:
            result["vlm"] = deepcopy(condition["prediction"])
        if condition["upstream_status"] != "ok":
            result.update(status="error", error_code=f"upstream_{condition['upstream_status']}")
            return result
    if component in ("sam", "combined"):
        result.update(sam_segment_image_calls=1, sam_query_count=None)
        segmented = _stage_call(lambda: segmenter.segment_image(adapter_frame(frame), list(result["prompts"]), prepared["roots"]["data_root"]), probe, clock, result["stage_times_s"], "sam")
        result["_sam_output"] = segmented
    return result


def _budget(prepared, summary):
    target = 6_000_000_000
    reasons = []
    if prepared["fixture"]:
        reasons.append("Synthetic fixture measurements are not hardware evidence")
    if prepared["component"] != "vlm":
        reasons.append("The proposed 6 decimal GB target applies only to VLM; SAM is additional")
    if not summary["complete"]:
        reasons.append("Protocol incomplete")
    if summary["invalid_timing_calls"] or not summary["successful_calls"]:
        reasons.append("No complete valid successful deployment timing evidence")
    if summary["failed_calls"] or summary["warmup"]["failed_calls"]:
        reasons.append("A budget pass requires a fully successful genuine warmup/measured execution path")
    memory = summary["memory"]
    if memory.get("incremental_allocated_peak_bytes") is None or memory.get("incremental_reserved_peak_bytes") is None:
        reasons.append("Required allocator telemetry unavailable")
    measured = not reasons
    return {"proposed_vlm_budget_bytes": target, "shared_gpu_capacity_bytes": 20_000_000_000,
            "status": "measured" if measured else "unmeasured", "unmeasured_reasons": reasons,
            "allocated_within_proposed_vlm_budget": memory["incremental_allocated_peak_bytes"] <= target if measured else None,
            "reserved_within_proposed_vlm_budget": memory["incremental_reserved_peak_bytes"] <= target if measured else None,
            "sampled_device_usage_establishes_budget": False,
            "joint_fit": "unmeasured", "joint_fit_reason": "A complete actual combined replay and free shared-device headroom assessment are required"}


def profile_backend(config, *, backend=None, segmenter=None):
    """Profile frozen whole-image calls; genuine execution uses a fresh process."""
    prepared = _prepare(config)
    if not prepared["fixture"]:
        if backend is not None or segmenter is not None or any(key in prepared["config"] for key in ("_memory_probe", "_clock")):
            raise ValueError("Injected adapters, probes and clocks require explicit fixture mode")
        if prepared["config"].get("enable_real_run") is not True:
            raise ValueError("Real model loading/GPU work is disabled; enable_real_run must be true")
        if sys.version_info < (3, 12):
            raise RuntimeError("The isolated hazard GPU environment requires Python 3.12+")
        payload = {key: value for key, value in prepared["config"].items() if not key.startswith("_")}
        payload.update(fixture=False, task_id=TASK_ID, schema_version=SCHEMA_VERSION)
        environment = os.environ.copy()
        source = str(Path(__file__).resolve().parents[1])
        environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
        kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        process = subprocess.run([sys.executable, "-m", "traversability_hazard_benchmark.worker"],
                                 input=json.dumps(payload, default=_json_path), capture_output=True, text=True, env=environment, **kwargs)
        if process.returncode:
            raise RuntimeError(f"Hazard profile worker unavailable/failed (exit {process.returncode}): {process.stderr[-2000:]}")
        return _read_profile(prepared["output"])
    for stage, adapter in (("vlm", backend), ("sam", segmenter)):
        if prepared["component"] in (stage, "combined"):
            if adapter is None:
                raise ValueError(f"Fixture {stage} profiling requires an explicitly injected adapter")
            _fixture_identity(adapter)
        elif adapter is not None:
            raise ValueError(f"Unused injected {stage} adapter")
    return _execute(prepared, backend=backend, segmenter=segmenter)


def _read_profile(output):
    return {"summary": read_json(output / "summary.json"), "metadata": read_json(output / "metadata.json"),
            "samples": read_jsonl(output / "samples.jsonl"), "output_dir": str(output)}


def _execute(prepared, *, backend=None, segmenter=None):
    from traversability_hazard_segmentation.common import fingerprint
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError("Another hazard profile is active in this process")
    try:
        with exclusive_lock(prepared["output"] / ".profile.lock"):
            return _execute_locked(prepared, backend, segmenter, fingerprint)
    finally:
        _LOCK.release()


def _execute_locked(prepared, backend, segmenter, fingerprint):
    from traversability_hazard_segmentation.common import adapter_frame, validate_prompts
    output, config = prepared["output"], prepared["config"]
    injected = {"vlm": backend, "sam": segmenter}
    expected_identity = {**prepared["identity"], "fixture_adapters": {
        stage: _fixture_identity(adapter) for stage, adapter in injected.items() if adapter is not None
    }}
    identity_sha = fingerprint(expected_identity)
    metadata_path, samples_path = output / "metadata.json", output / "samples.jsonl"
    metadata = read_json(metadata_path) if metadata_path.exists() else None
    samples = read_jsonl(samples_path) if samples_path.exists() else []
    if metadata is not None:
        if metadata.get("identity_sha256") != identity_sha:
            raise ValueError("Existing profile identity differs; choose a new profile_id")
        headers = {"task_id": TASK_ID, "schema_version": 1, "fixture": prepared["fixture"],
                   "component": prepared["component"], "condition_key": prepared["condition_key"],
                   "execution_kind": "injected_fixture" if prepared["fixture"] else "real_model",
                   "selection_hash": prepared["selection"]["fingerprint"]}
        if any(metadata.get(key) != value or type(metadata.get(key)) is not type(value) for key, value in headers.items()):
            raise ValueError("Saved profile metadata identity/fixture mismatch")
        if config.get("resume", True) is not True:
            raise ValueError("Existing profile requires explicit compatible resume or a new profile_id")
        current_injected = {stage: _adapter_identity(adapter) for stage, adapter in injected.items() if adapter is not None}
        actual_saved = metadata.get("actual_adapter_identity", {})
        if any(actual_saved.get(stage) != identity for stage, identity in current_injected.items()):
            raise ValueError("Injected adapter actual settings/model identity differs from saved profile")
    elif samples:
        raise ValueError("Samples without compatible metadata cannot be resumed")
    else:
        metadata = {"task_id": TASK_ID, "schema_version": 1, "fixture": prepared["fixture"],
                    "execution_kind": "injected_fixture" if prepared["fixture"] else "real_model",
                    "component": prepared["component"], "condition_key": prepared["condition_key"],
                    "identity": expected_identity, "identity_sha256": identity_sha,
                    "selection_fingerprint": prepared["selection"]["fingerprint"],
                    "selection_hash": prepared["selection"]["fingerprint"],
                    "policy_hash": prepared["inputs"]["policy_hash"],
                    "alias_hash": prepared["inputs"]["alias_sha256"],
                    "policy_sha256": prepared["inputs"]["policy_sha256"],
                    "alias_sha256": prepared["inputs"]["alias_sha256"],
                    "dataset_fingerprint": prepared["inputs"]["dataset"]["dataset_fingerprint"], "sessions": [],
                    "software_versions": _versions(), "started_at": _now(),
                    "isolation": "caller-owned fixture adapters" if prepared["fixture"] else "fresh subprocess; only owned adapters closed",
                    "isolation_mode": "in_process_fixture" if prepared["fixture"] else "fresh_process",
                    "memory_limitations": ["Allocator counters are process-wide on the explicit device",
                                           "Sampled device usage includes other users/processes and can miss transient peaks",
                                           "Pre-load baseline follows CUDA telemetry context initialization",
                                           "Separate isolated profiles cannot establish joint residency or combined p95"],
                    "combined_enabled": config.get("enable_combined") is True}
    # Validate resumed slots rather than trusting a matching metadata header alone.
    test_ids = set(prepared["selection"]["test_frame_ids"])
    warmup_ids = prepared["selection"]["warmup_frame_ids"]
    sample_ids = set()
    for sample in samples:
        if sample.get("identity_sha256") != identity_sha or sample.get("task_id") != TASK_ID or type(sample.get("schema_version")) is not int or sample["schema_version"] != 1 or sample.get("fixture") is not prepared["fixture"]:
            raise ValueError("Saved sample identity/fixture mismatch")
        if sample.get("phase") not in ("warmup", "measured") or sample.get("status") not in ("ok", "error"):
            raise ValueError("Saved completed sample phase/status mismatch")
        if sample.get("sample_id") in sample_ids or not isinstance(sample.get("sample_id"), str):
            raise ValueError("Saved sample IDs must be unique strings")
        sample_ids.add(sample["sample_id"])
        if sample.get("phase") == "measured" and (sample.get("frame_id") not in test_ids or type(sample.get("repeat")) is not int or sample["repeat"] not in (0, 1)):
            raise ValueError("Saved measured sample is not a frozen test slot")
        if sample["phase"] == "warmup" and (sample.get("frame_id") not in warmup_ids or sample.get("repeat") is not None or sample.get("warmup_index") != warmup_ids.index(sample["frame_id"])):
            raise ValueError("Saved warmup sample is not a frozen development slot")
        validate_prompts(sample.get("prompts"))
        if sample.get("requested_prompt_count") != len(sample["prompts"]):
            raise ValueError("Saved sample prompt count mismatch")
        if sample.get("source") != prepared["inputs"]["frames"][sample["frame_id"]]["source"]:
            raise ValueError("Saved sample source join mismatch")
        if sample.get("record_sha256") != fingerprint({key: value for key, value in sample.items() if key != "record_sha256"}):
            raise ValueError("Saved completed attempt content hash mismatch")
    if metadata.get("samples_fingerprint", fingerprint([])) != fingerprint(samples):
        raise ValueError("Saved sample journal differs from the atomically committed metadata")
    existing = aggregate_samples(samples)
    if existing["complete"] and (output / "summary.json").exists() and read_json(output / "summary.json").get("complete") is True:
        if metadata.get("summary_fingerprint") != fingerprint(read_json(output / "summary.json")):
            raise ValueError("Saved completed summary content hash mismatch")
        return _read_profile(output)
    session_id = len(metadata["sessions"])
    session = {"session_id": session_id, "started_at": _now(), "load": {}, "fatal_error": None, "cleanup_errors": []}
    metadata["sessions"].append(session)
    metadata["samples_fingerprint"] = fingerprint(samples)
    atomic_json(metadata_path, metadata)
    atomic_jsonl(samples_path, samples)
    fixture = prepared["fixture"]
    clock = config.get("_clock", time.perf_counter)
    probe = config.get("_memory_probe")
    owned = []
    adapters = {"vlm": backend, "sam": segmenter}
    if probe is None:
        if fixture:
            probe = NullMemoryProbe()
        else:
            stage = "vlm" if prepared["component"] in ("vlm", "combined") else "sam"
            try:
                probe = TorchMemoryProbe(prepared["settings"][stage]["device"])
            except Exception as error:
                probe = NullMemoryProbe()
                probe.unavailable_reason = f"{type(error).__name__}: {error}"
    baseline, baseline_errors = _snapshot(probe)
    session.update(memory_baseline=baseline, memory_baseline_errors=baseline_errors)
    try:
        _assert_frozen(prepared)
        for stage in (("vlm", "sam") if prepared["component"] == "combined" else (prepared["component"],)):
            if adapters[stage] is not None:
                session["load"][stage] = {"elapsed_s": None, "status": "not_measured", "reason": "Caller-owned injected adapter", "memory": None}
                continue
            def load_one(stage=stage):
                requested_settings = deepcopy(prepared["settings"][stage])
                if stage == "vlm":
                    requested_settings.setdefault("cache_dir", str(prepared["roots"]["cache_root"] / "huggingface/hub"))
                    requested_settings.setdefault("policy_path", str(prepared["roots"]["policy_path"]))
                    if "allow_downloads" in requested_settings:
                        permitted = requested_settings.pop("allow_downloads")
                        if type(permitted) is not bool:
                            raise ValueError("allow_downloads must be boolean")
                        requested_settings["local_files_only"] = not permitted
                    requested_settings.setdefault("local_files_only", True)
                    if "visual_token_limit" in requested_settings:
                        raise ValueError("Use the official hazard VLM visual_token_budget setting")
                    from traversability_hazard_inference import load_backend
                    adapter = load_backend(prepared["model_key"], requested_settings)
                else:
                    requested_settings.setdefault("cache_root", str(prepared["roots"]["cache_root"]))
                    requested_settings.setdefault("allow_downloads", False)
                    requested_settings["enable_model_loading"] = config.get("enable_real_run") is True
                    from traversability_hazard_segmentation import load_segmenter
                    adapter = load_segmenter(requested_settings)
                adapters[stage] = adapter
                owned.append(adapter)
                return adapter
            _, elapsed, problem, memory, valid = _timed_call(load_one, probe, baseline, clock)
            session["load"][stage] = {"elapsed_s": elapsed, "error": problem, "status": "error" if problem else "ok", "timing_valid": valid, "memory": memory}
            atomic_json(metadata_path, metadata)
            if problem:
                raise RuntimeError(f"{stage} unavailable: {problem['type']}: {problem['message']}")
        adapter_identity = {stage: _adapter_identity(adapter) for stage, adapter in adapters.items() if adapter is not None}
        previous_actual = metadata.get("actual_adapter_identity")
        if previous_actual is not None and previous_actual != adapter_identity:
            raise ValueError("Loaded model/checkpoint/settings identity differs from saved profile")
        metadata["actual_adapter_identity"] = adapter_identity
        if not fixture and any((actual.get("settings") or {}).get("fixture") is True or (actual.get("metadata") or {}).get("fixture") is True
                               for actual in adapter_identity.values()):
            raise ValueError("A fixture model cannot be reported as a genuine loaded adapter")
        if "vlm" in adapter_identity:
            actual = adapter_identity["vlm"]
            declared = {**(actual.get("settings") or {}), **(actual.get("metadata") or {})}
            metadata.update(model_key=prepared["model_key"], checkpoint_revision=declared.get("checkpoint_revision", declared.get("revision")),
                            settings=(actual.get("metadata") or {}).get("configuration_snapshot", {}).get("settings", actual.get("settings") or {}))
        if "sam" in adapter_identity:
            metadata["sam_settings"] = adapter_identity["sam"].get("settings") or {}
            from traversability_hazard_segmentation.controller import (
                _adapter_identity as condition_adapter_identity, _shared_sam_identity,
            )
            _shared_sam_identity(prepared["roots"]["run_dir"], None,
                                 condition_adapter_identity(adapters["sam"]))
        session["resident_baseline"], session["resident_baseline_errors"] = _snapshot(probe)
        atomic_json(metadata_path, metadata)
        completed = {(sample["frame_id"], sample["repeat"]) for sample in samples if sample["phase"] == "measured"}
        schedule = [("warmup", frame_id, None, index) for index, frame_id in enumerate(prepared["selection"]["warmup_frame_ids"])]
        schedule += [("measured", frame_id, repeat, None) for repeat in range(2)
                     for frame_id in prepared["selection"]["test_frame_ids"] if (frame_id, repeat) not in completed]
        for phase, frame_id, repeat, warmup_index in schedule:
            _assert_frozen(prepared)
            frame = prepared["inputs"]["frames"][frame_id]
            call_baseline, call_baseline_errors = _snapshot(probe)
            component = prepared["component"]
            captured = {"stage_times_s": {}, "prompts": [], "sam_query_count": 0, "requested_prompt_count": 0,
                        "sam_segment_image_calls": 0, "status": "ok", "error_code": None}
            call_attempted = True
            # Prepare adapter-only inputs outside isolated timing intervals.
            rgb_frame = adapter_frame(frame)
            if component == "sam":
                condition = prepared["condition_rows"][frame_id]
                captured.update(prompts=list(condition["prompts"]), requested_prompt_count=len(condition["prompts"]))
                if condition.get("prediction") is not None:
                    captured["vlm"] = deepcopy(condition["prediction"])
                if condition["upstream_status"] != "ok":
                    call_attempted = False
                    captured.update(status="error", error_code=f"upstream_{condition['upstream_status']}")
                    returned, elapsed, problem, valid = None, None, None, False
                    memory = {**dict.fromkeys(_MEMORY_KEYS), "unavailable_reasons": ["SAM not called because upstream prediction failed/is missing"]}
                else:
                    captured.update(sam_segment_image_calls=1, sam_query_count=None)
                    returned, elapsed, problem, memory, valid = _timed_call(
                        lambda: adapters["sam"].segment_image(rgb_frame, list(captured["prompts"]), prepared["roots"]["data_root"]), probe, baseline, clock)
                    captured["stage_times_s"]["sam"] = elapsed
                    captured["_sam_output"] = returned
            elif component == "vlm":
                returned, elapsed, problem, memory, valid = _timed_call(
                    lambda: adapters["vlm"].predict_image(rgb_frame, prepared["roots"]["data_root"]), probe, baseline, clock)
                captured["stage_times_s"]["vlm"] = elapsed
                _capture_vlm(returned, captured)
            else:
                returned, elapsed, problem, memory, valid = _timed_call(
                    lambda: _call(prepared, frame, adapters["vlm"], adapters["sam"], probe, clock, captured), probe, baseline, clock)
            # Output contract checks and CPU mask audit are outside the measured call.
            if problem is None and call_attempted:
                try:
                    if component == "vlm":
                        vlm = _vlm_result(returned, frame, prepared["model_key"])
                        captured.update(vlm=vlm, prompts=vlm["prompts"], requested_prompt_count=len(vlm["prompts"]),
                                        status=vlm["status"], error_code=vlm["error_code"])
                    if "_sam_output" in captured:
                        sam = _sam_result(captured.pop("_sam_output"), frame, captured["prompts"], fixture=fixture)
                        captured.update(sam=sam, sam_query_count=sam["sam_query_count"], status=sam["status"], error_code=sam["error_code"])
                except Exception as error:
                    problem = {**_error(error), "output_validation_failed": True}
            captured.pop("_sam_output", None)
            post_error = None
            try:
                _assert_frozen(prepared)
                if {stage: _adapter_identity(adapter) for stage, adapter in adapters.items() if adapter is not None} != adapter_identity:
                    raise ValueError("Adapter settings/model identity changed during profiling")
            except BaseException as error:
                post_error = _error(error)
            result = captured
            sample = {"task_id": TASK_ID, "schema_version": 1, "fixture": fixture, "identity_sha256": identity_sha,
                      "execution_kind": "injected_fixture" if fixture else "real_model",
                      "component": component, "condition_key": prepared["condition_key"],
                      "sample_id": f"s{session_id}:{phase}:{frame_id}:{repeat if repeat is not None else warmup_index}",
                      "session_id": session_id, "phase": phase, "frame_id": frame_id, "repeat": repeat,
                      "warmup_index": warmup_index, "source": frame["source"], "split": frame["split"], "elapsed_s": elapsed,
                      "timing_valid": valid, "status": "error" if problem or post_error else result.get("status", "error"),
                      "call_attempted": call_attempted,
                      "error_code": "input_or_identity_changed" if post_error else ("call_error" if problem else result.get("error_code")),
                      "call_error": problem, "post_call_validation_error": post_error,
                      "stage_times_s": result.get("stage_times_s", {}), "prompts": result.get("prompts", []),
                      "requested_prompt_count": result.get("requested_prompt_count", 0), "sam_query_count": result.get("sam_query_count", 0),
                      "sam_segment_image_calls": result.get("sam_segment_image_calls", 0),
                      "vlm": result.get("vlm"), "sam": result.get("sam"), "memory": memory,
                      "call_baseline": call_baseline, "call_baseline_errors": call_baseline_errors}
            for name, base_key in (("allocated", "allocated_bytes"), ("reserved", "reserved_bytes"), ("device_used", "device_used_bytes")):
                peak, start = memory.get(f"{name}_peak_bytes"), call_baseline.get(base_key)
                memory[f"call_incremental_{name}_peak_bytes"] = None if peak is None or start is None else max(0, peak - start)
            sample["record_sha256"] = fingerprint(sample)
            samples.append(sample)
            atomic_jsonl(samples_path, samples)
            metadata["samples_fingerprint"] = fingerprint(samples)
            atomic_json(metadata_path, metadata)
            lower_error = str(problem or result.get("error_code") or "").lower()
            if post_error or (not valid and call_attempted) or any(word in lower_error for word in ("outofmemory", "out of memory", "keyboardinterrupt", "systemexit", "device-side assert")):
                session["fatal_error"] = post_error or problem or {"type": "FatalModelError", "message": lower_error}
                break
    except BaseException as error:
        session["fatal_error"] = _error(error)
    finally:
        for adapter in reversed(owned):
            try:
                adapter.close()
            except BaseException as error:
                session["cleanup_errors"].append(_error(error))
        session["finished_at"] = _now()
        summary = aggregate_samples(samples)
        summary.update(task_id=TASK_ID, schema_version=1, fixture=fixture,
                       execution_kind="injected_fixture" if fixture else "real_model", component=prepared["component"],
                       condition_key=prepared["condition_key"])
        summary.update({key: metadata[key] for key in ("selection_hash", "policy_hash", "alias_hash", "policy_sha256", "alias_sha256", "dataset_fingerprint")})
        summary["memory"] = _merge_memory([summary["memory"]] + [load.get("memory") for item in metadata["sessions"] for load in item["load"].values()])
        summary["loading"] = [item["load"] for item in metadata["sessions"]]
        summary["phase_memory"] = {
            "loading": _merge_memory(load.get("memory") for item in metadata["sessions"] for load in item["load"].values()),
            "warmup": _merge_memory(sample.get("memory") for sample in samples if sample["phase"] == "warmup"),
            "warmed_inference": _merge_memory(sample.get("memory") for sample in samples if sample["phase"] == "measured"),
        }
        summary["cold_start_elapsed_s"] = next((sample["elapsed_s"] for sample in samples if sample.get("warmup_index") == 0), None)
        summary["complete"] = summary["complete"] and session["fatal_error"] is None and not session["cleanup_errors"]
        missing_identity = []
        for stage, actual in metadata.get("actual_adapter_identity", {}).items():
            actual_meta = actual.get("metadata") or {}
            nested_model = actual_meta.get("configuration_snapshot", {}).get("model", {})
            declared = {**nested_model, **(actual.get("settings") or {}), **actual_meta}
            if not declared.get("checkpoint_revision", declared.get("revision")):
                missing_identity.append(f"{stage} checkpoint revision unavailable")
            if stage == "sam" and not declared.get("code_revision"):
                missing_identity.append("SAM code revision unavailable")
            if not fixture and stage == "sam" and any(actual_meta.get(key, {}).get("status") != "verified" for key in ("code_verification", "checkpoint_verification")):
                missing_identity.append("SAM verified official source/checkpoint provenance unavailable")
        summary["identity_limitations"] = missing_identity
        summary["comparison_ready"] = summary["complete"] and not fixture and summary["invalid_timing_calls"] == 0 and summary["valid_measured_calls"] > 0 and not missing_identity
        summary["budget"] = _budget(prepared, summary)
        summary["combined_latency"] = summary["latency"] if prepared["component"] == "combined" else None
        summary["joint_memory_measured"] = (
            prepared["component"] == "combined" and not fixture and summary["complete"]
            and summary["invalid_timing_calls"] == 0 and summary["failed_calls"] == 0
            and {"vlm", "sam"} <= metadata.get("actual_adapter_identity", {}).keys()
            and summary["phase_memory"]["warmed_inference"]["allocated_peak_bytes"] is not None
            and all(sample.get("sam_segment_image_calls") == 1 for sample in samples if sample["phase"] == "measured")
        )
        metadata["finished_at"] = _now()
        metadata["samples_fingerprint"] = fingerprint(samples)
        metadata["summary_fingerprint"] = fingerprint(summary)
        atomic_json(metadata_path, metadata)
        atomic_json(output / "summary.json", summary)
    return _read_profile(output)
