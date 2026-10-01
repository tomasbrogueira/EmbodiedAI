"""Read saved measured deployment profiles; never synthesize joint costs."""

import math
from pathlib import PurePosixPath

from .artifacts import ArtifactError, binding, fingerprint, issue, parse_json, pointer, rate, read_json, safe_path, validate_identity
from .segmentation import load_selection


def _field(obj, ctx, kind, name, default):
    return pointer(obj, binding(ctx["config"], kind, name, default))


def _count(summary, name):
    value = summary.get(name)
    if type(value) is not int or value < 0:
        raise ArtifactError(f"Deployment {name} must be a nonnegative integer")
    return value


def _number(value, name):
    if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
        raise ArtifactError(f"Deployment {name} must be nonnegative or null")


def _summary_identity(summary, ctx):
    if summary.get("task_id") != "hazard_prompt_v1" or type(summary.get("schema_version")) is not int or summary["schema_version"] != 1:
        raise ArtifactError("Deployment summary task/schema mismatch")
    if type(summary.get("fixture")) is not bool or summary["fixture"] != ctx["config"]["fixture"]:
        raise ArtifactError("Deployment summary fixture mismatch")
    expected_execution = "injected_fixture" if summary["fixture"] else "real_model"
    if summary.get("execution_kind") != expected_execution:
        raise ArtifactError("Deployment execution kind disagrees with fixture marker")
    if summary.get("component") not in ("vlm", "sam", "combined"):
        raise ArtifactError("Deployment component must be vlm, sam or combined")
    condition = summary.get("condition_key")
    if not isinstance(condition, str) or not condition:
        raise ArtifactError("Deployment summary requires condition_key")
    if condition not in ("fixed_policy", "reference_present") and not condition.startswith("vlm__"):
        raise ArtifactError("Unknown deployment condition")
    if summary["component"] in ("vlm", "combined") and not condition.startswith("vlm__"):
        raise ArtifactError("VLM/combined cost must identify an actual VLM condition")
    for key in ("complete", "comparison_ready"):
        if type(summary.get(key)) is not bool:
            raise ArtifactError(f"Deployment {key} must be boolean")
    if summary["fixture"] and summary["comparison_ready"]:
        raise ArtifactError("Synthetic deployment fixture cannot be comparison-ready")


def _validate_memory(memory):
    if not isinstance(memory, dict):
        raise ArtifactError("Deployment memory must be an object")
    # Preserve unavailable statistics and all stage-specific details verbatim.
    for name, value in memory.items():
        if isinstance(value, dict):
            _validate_memory(value)
        if name.endswith("_bytes"):
            if value is not None and (type(value) is not int or value < 0):
                raise ArtifactError(f"Deployment memory {name} must be nonnegative bytes or null")
    reasons = memory.get("unavailable_reasons", [])
    if not isinstance(reasons, list) or any(not isinstance(value, str) for value in reasons):
        raise ArtifactError("Deployment memory unavailable_reasons must be strings")


def _has_peak(memory):
    total_peaks = {"allocated_peak_bytes", "reserved_peak_bytes", "device_used_peak_bytes"}
    return any((key in total_peaks and value is not None) or (isinstance(value, dict) and _has_peak(value)) for key, value in memory.items())


def _pins(metadata, ctx, summary, selection, sam_settings):
    component, condition = summary["component"], summary["condition_key"]
    result = {}
    if component in ("vlm", "combined"):
        model = condition[len("vlm__"):]
        if _field(metadata, ctx, "deployment_metadata", "model_key", "/model_key") != model or model not in ctx["config"]["model_keys"]:
            raise ArtifactError("Deployment actual model identity mismatch")
        revision = _field(metadata, ctx, "deployment_metadata", "checkpoint_revision", "/checkpoint_revision")
        settings = _field(metadata, ctx, "deployment_metadata", "settings", "/settings")
        if not isinstance(revision, str) or not revision or not isinstance(settings, dict) or not settings:
            raise ArtifactError("Deployment requires actual model revision/settings pins")
        inference_metadata = ctx.get("model_metadata", {}).get(model)
        if inference_metadata is None and not summary["fixture"]:
            raise ArtifactError("Real deployment requires validated saved inference metadata")
        if inference_metadata is not None:
            expected_revision = _field(inference_metadata, ctx, "model", "checkpoint_revision", "/checkpoint_revision")
            if revision != expected_revision:
                raise ArtifactError("Deployment checkpoint revision differs from saved inference")
            # Settings may be unavailable in early fixture metadata; an explicit
            # binding makes their presence mandatory rather than guessing nesting.
            if not summary["fixture"] or "settings" in inference_metadata or "settings" in ctx["config"].get("metadata_bindings", {}).get("model", {}):
                if settings != _field(inference_metadata, ctx, "model", "settings", "/settings"):
                    raise ArtifactError("Deployment settings differ from saved inference")
        result.update(model_key=model, checkpoint_revision=revision, settings=settings)
    if component in ("sam", "combined"):
        actual = _field(metadata, ctx, "deployment_metadata", "sam_settings", "/sam_settings")
        if not isinstance(actual, dict) or not actual:
            raise ArtifactError("Deployment requires actual SAM revision/settings pins")
        for name in ("checkpoint", "revision"):
            pin = _field(actual, ctx, "sam_settings", name, "/" + name)
            if not isinstance(pin, str) or not pin:
                raise ArtifactError(f"Deployment SAM {name} pin must be nonempty")
        expected = selection.get("sam_settings", sam_settings)
        if expected is not None and actual != expected:
            raise ArtifactError("Deployment SAM checkpoint/settings differ across stages/conditions")
        result["sam_settings"] = actual
    return result


def _latency(rows):
    values = sorted(row["elapsed_s"] for row in rows if row["timing_valid"] and row["elapsed_s"] is not None)
    def quantile(fraction):
        if not values:
            return None
        position = (len(values) - 1) * fraction
        lower, upper = math.floor(position), math.ceil(position)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)
    return {"median_s": quantile(0.5), "p95_s": quantile(0.95)}


def _samples(path, ctx, selection, repeats, summary, metadata):
    """Validate explicit saved calls by their frame/phase/repeat identity."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ArtifactError(f"Cannot read deployment samples: {error}") from error
    rows, seen, sample_ids, measured, warmup, original = [], set(), set(), [], [], []
    for line_number, line in enumerate(lines, 1):
        try:
            sample = parse_json(line)
            kind = "deployment_samples"
            phase = _field(sample, ctx, kind, "phase", "/phase")
            identifier = _field(sample, ctx, kind, "frame_id", "/frame_id")
            repeat = _field(sample, ctx, kind, "repeat", "/repeat")
            elapsed = _field(sample, ctx, kind, "elapsed_s", "/elapsed_s")
            status = _field(sample, ctx, kind, "status", "/status")
            count = _field(sample, ctx, kind, "sam_query_count", "/sam_query_count")
            session = sample.get("session_id")
            if session is not None and (type(session) is not int or session < 0):
                raise ArtifactError("Deployment session ID must be a nonnegative integer")
            if phase not in ("measured", "warmup") or status not in ("ok", "error"):
                raise ArtifactError("Deployment sample phase/status mismatch")
            if not isinstance(identifier, str) or identifier not in ctx["frames"]:
                raise ArtifactError("Deployment sample unknown frame ID")
            frame = ctx["frames"][identifier]
            if phase == "measured":
                if identifier not in selection["test_frame_ids"] or frame["split"] != "test" or type(repeat) is not int or not 0 <= repeat < repeats:
                    raise ArtifactError("Deployment measured sample outside frozen test/repeat selection")
            elif identifier not in selection["warmup_frame_ids"] or frame["split"] != "development" or repeat is not None:
                raise ArtifactError("Deployment warmup sample outside frozen development selection")
            identity = (phase, identifier, repeat) if phase == "measured" else (phase, session, identifier)
            if identity in seen:
                raise ArtifactError("Duplicate deployment frame/phase/repeat sample")
            seen.add(identity)
            _number(elapsed, "sample elapsed_s")
            timing_valid = sample.get("timing_valid", True)
            attempted = sample.get("call_attempted", True)
            if type(timing_valid) is not bool or type(attempted) is not bool:
                raise ArtifactError("Deployment timing/call-attempt markers must be boolean")
            if elapsed is None:
                if timing_valid or status != "error" or not sample.get("error_code"):
                    raise ArtifactError("Unavailable timing requires an explicit failed/blocked call")
                if attempted and not (sample.get("call_error") or sample.get("post_call_validation_error")):
                    raise ArtifactError("Unavailable attempted timing requires a recorded measurement error")
            elif not attempted:
                raise ArtifactError("Unattempted calls cannot carry measured elapsed time")
            if count is not None and (type(count) is not int or count < 0):
                raise ArtifactError("Actual SAM call count must be nonnegative or explicitly unavailable")
            if count is None and (status != "error" or not (sample.get("call_error") or sample.get("post_call_validation_error"))):
                raise ArtifactError("Unavailable SAM count requires recorded call/output failure")
            sample_id = sample.get("sample_id")
            if sample_id is not None:
                if not isinstance(sample_id, str) or not sample_id or sample_id in sample_ids:
                    raise ArtifactError("Deployment sample IDs must be unique strings")
                sample_ids.add(sample_id)
            if "record_sha256" in sample and sample["record_sha256"] != fingerprint({key: value for key, value in sample.items() if key != "record_sha256"}):
                raise ArtifactError("Deployment sample committed content hash mismatch")
            if "identity_sha256" in sample and sample["identity_sha256"] != metadata.get("identity_sha256"):
                raise ArtifactError("Deployment sample profile identity changed")
            if "source" in sample and sample["source"] != frame["source"] or "split" in sample and sample["split"] != frame["split"]:
                raise ArtifactError("Deployment sample source/split join mismatch")
            if "condition_key" in sample and sample["condition_key"] != summary["condition_key"]:
                raise ArtifactError("Deployment sample condition mismatch")
            if "component" in sample and sample["component"] != summary["component"]:
                raise ArtifactError("Deployment sample stage mismatch")
            for name in ("task_id", "schema_version", "fixture", "execution_kind"):
                if name in sample and (sample[name] != summary[name] or (name in ("schema_version", "fixture") and type(sample[name]) is not type(summary[name]))):
                    raise ArtifactError(f"Deployment sample {name} identity mismatch")
            normalized = {"phase": phase, "session_id": session, "frame_id": identifier, "repeat": repeat,
                          "elapsed_s": elapsed, "status": status, "sam_query_count": count,
                          "timing_valid": timing_valid, "call_attempted": attempted}
            rows.append(normalized)
            original.append(sample)
            (measured if phase == "measured" else warmup).append(normalized)
        except (ArtifactError, ValueError, TypeError, KeyError) as error:
            raise ArtifactError(f"Deployment sample line {line_number}: {error}") from error
    if len(measured) != summary["measured_frames"] or len({row["frame_id"] for row in measured}) != summary["unique_test_frames"]:
        raise ArtifactError("Deployment sample measured-frame counters differ from summary")
    if sum(row["status"] == "error" for row in measured) != summary["failed_calls"]:
        raise ArtifactError("Deployment sample failure count differs from summary")
    if sum(row["sam_query_count"] or 0 for row in measured) != summary["sam_query_count"]:
        raise ArtifactError("Deployment sample actual SAM count differs from summary")
    counters = {"valid_measured_calls": sum(row["timing_valid"] and row["elapsed_s"] is not None for row in measured),
                "invalid_timing_calls": sum(not row["timing_valid"] and row["call_attempted"] for row in measured),
                "unattempted_calls": sum(not row["call_attempted"] for row in measured),
                "unavailable_sam_query_counts": sum(row["sam_query_count"] is None for row in measured)}
    for name, value in counters.items():
        if name in summary and summary[name] != value:
            raise ArtifactError(f"Deployment sample {name} differs from summary")
    warmup_summary = summary.get("warmup")
    if isinstance(warmup_summary, dict):
        for name, value in (("attempted_calls", len(warmup)), ("failed_calls", sum(row["status"] == "error" for row in warmup))):
            if name in warmup_summary and warmup_summary[name] != value:
                raise ArtifactError(f"Deployment warmup {name} differs from summary")
    for name, value in _latency(measured).items():
        declared = summary["latency"][name]
        if declared != value and not (declared is not None and value is not None and math.isclose(declared, value, rel_tol=1e-12, abs_tol=1e-12)):
            raise ArtifactError("Deployment sample latency differs from summary")
    if "samples_fingerprint" in metadata and metadata["samples_fingerprint"] != fingerprint(original):
        raise ArtifactError("Deployment sample journal fingerprint changed")
    expected = {(identifier, repeat) for identifier in selection["test_frame_ids"] for repeat in range(repeats)}
    actual = {(row["frame_id"], row["repeat"]) for row in measured}
    by_session = {}
    for row in warmup:
        by_session.setdefault(row["session_id"], set()).add(row["frame_id"])
    required_warmup = set(selection["warmup_frame_ids"])
    # Every session that publishes measured calls must have its own fixed warmups.
    measured_sessions = {row["session_id"] for row in measured}
    warmup_complete = all(by_session.get(session, set()) == required_warmup for session in measured_sessions)
    if not measured_sessions:
        warmup_complete = not required_warmup or any(ids == required_warmup for ids in by_session.values())
    if isinstance(metadata.get("sessions"), list):
        valid_sessions = {row.get("session_id") for row in metadata["sessions"] if isinstance(row, dict)}
        if (set(by_session) | measured_sessions) - valid_sessions:
            raise ArtifactError("Deployment samples refer to an undeclared profile session")
    return {"saved_samples": len(rows), "measured_samples": len(measured), "warmup_samples": len(warmup),
            "warmup_sessions": len(by_session), **counters,
            "missing_measured_calls": [{"frame_id": identifier, "repeat": repeat} for identifier, repeat in sorted(expected - actual)],
            "sample_coverage_complete": actual == expected and warmup_complete}


def evaluate_deployment(ctx, selection=None):
    """Return independent saved stage costs, preserving measurement limitations."""
    config = ctx["config"].get("deployment", {})
    profiles = config.get("profiles", [])
    result = {"rows": [], "complete": True, "limitations": [
        "Separate stage p95 values and GPU peaks are not summed.",
        "Combined latency and joint residency require an actual combined measured profile.",
        "Loading and cold-start costs are separate from warm per-image processing.",
        "Fixture profiles are software checks, not measured model/resource evidence.",
    ]}
    if not profiles:
        return result
    issues = ctx["issues"]
    if selection is None:
        try:
            selection = load_selection(ctx)
        except (ArtifactError, OSError, ValueError, TypeError, KeyError) as error:
            result["complete"] = False
            issue(issues, "invalid_deployment_selection", str(error), stage="deployment")
    seen, sam_settings = set(), None
    for entry in profiles:
        if isinstance(entry, str):
            entry = {"summary_path": entry}
        if not isinstance(entry, dict):
            result["complete"] = False
            issue(issues, "invalid_deployment_profile", "Profile must be a relative summary path or object", stage="deployment")
            continue
        relative = entry.get("summary_path")
        row = {"summary_path": relative, "valid": False, "complete": False, "comparison_ready": False}
        try:
            summary_path = safe_path(ctx["run_dir"], relative)
            if relative in seen:
                raise ArtifactError("Duplicate deployment profile summary")
            seen.add(relative)
            metadata_relative = entry.get("metadata_path", str(PurePosixPath(relative).with_name("metadata.json")))
            metadata = read_json(safe_path(ctx["run_dir"], metadata_relative))
            summary = read_json(summary_path)
            if "summary_fingerprint" in metadata and metadata["summary_fingerprint"] != fingerprint(summary):
                raise ArtifactError("Deployment summary committed fingerprint changed")
            _summary_identity(summary, ctx)
            condition = summary["condition_key"]
            validate_identity(metadata, ctx, "deployment_metadata", condition_key=condition)
            if _field(metadata, ctx, "deployment_metadata", "component", "/component") != summary["component"]:
                raise ArtifactError("Deployment stage metadata/summary mismatch")
            if _field(metadata, ctx, "deployment_metadata", "execution_kind", "/execution_kind") != summary["execution_kind"]:
                raise ArtifactError("Deployment execution metadata/summary mismatch")
            if selection is None:
                raise ArtifactError("Cannot validate deployment coverage without frozen selection")
            if _field(metadata, ctx, "deployment_metadata", "selection_hash", "/selection_hash") != selection["selection_hash"]:
                raise ArtifactError("Deployment frozen selection hash mismatch")
            if "identity" in metadata:
                identity = metadata["identity"]
                if metadata.get("identity_sha256") != fingerprint(identity):
                    raise ArtifactError("Deployment profile identity fingerprint changed")
                if identity.get("input_identity") != selection["artifact"].get("identity"):
                    raise ArtifactError("Deployment frozen input identity differs from selection")
                if identity.get("selection_fingerprint") != selection["selection_hash"]:
                    raise ArtifactError("Deployment profile selection identity changed")
            pins = _pins(metadata, ctx, summary, selection, sam_settings)
            if "sam_settings" in pins:
                sam_settings = pins["sam_settings"]
            repeats = config.get("expected_measured_repeats", 2)
            if type(repeats) is not int or repeats <= 0:
                raise ArtifactError("expected_measured_repeats must be a positive integer")
            measured = _count(summary, "measured_frames")
            expected = _count(summary, "expected_measured_frames")
            unique = _count(summary, "unique_test_frames")
            failures = _count(summary, "failed_calls")
            queries = _count(summary, "sam_query_count")
            if expected != repeats * len(selection["test_frame_ids"]):
                raise ArtifactError("Deployment declared measured coverage differs from frozen selection/repeats")
            if measured > expected or unique > len(selection["test_frame_ids"]) or unique > measured or failures > measured:
                raise ArtifactError("Deployment measured/failure coverage counters are inconsistent")
            coverage_complete = measured == expected and unique == len(selection["test_frame_ids"])
            if summary["complete"] and not coverage_complete:
                raise ArtifactError("Deployment summary claims complete with missing measured coverage")
            if summary["comparison_ready"] and (not summary["complete"] or summary["fixture"]):
                raise ArtifactError("Deployment comparison-ready claim conflicts with incomplete/fixture profile")
            latency = summary.get("latency")
            if not isinstance(latency, dict) or not {"median_s", "p95_s"} <= latency.keys():
                raise ArtifactError("Deployment latency must contain median_s and p95_s")
            for name in ("median_s", "p95_s"):
                _number(latency[name], name)
            valid_timing = summary.get("valid_measured_calls", measured)
            if type(valid_timing) is not int or not 0 <= valid_timing <= measured:
                raise ArtifactError("Deployment valid timing count is inconsistent")
            if valid_timing and (latency["median_s"] is None or latency["p95_s"] is None):
                raise ArtifactError("Valid measured deployment timings require latency statistics")
            if not valid_timing and any(latency[name] is not None for name in ("median_s", "p95_s")):
                raise ArtifactError("Unavailable measured timings require null latency statistics")
            if summary["comparison_ready"] and (not valid_timing or summary.get("invalid_timing_calls", 0)):
                raise ArtifactError("Deployment readiness cannot use unavailable/invalid measured timings")
            _validate_memory(summary.get("memory"))
            phase_memory = summary.get("phase_memory")
            if phase_memory is not None:
                if not isinstance(phase_memory, dict):
                    raise ArtifactError("Deployment phase_memory must be an object")
                _validate_memory(phase_memory)
            row.update({key: summary[key] for key in ("task_id", "schema_version", "fixture", "execution_kind", "component", "condition_key",
                "measured_frames", "expected_measured_frames", "unique_test_frames", "failed_calls", "sam_query_count", "latency", "memory")})
            row.update(valid=True, complete=summary["complete"] and coverage_complete,
                comparison_ready=summary["comparison_ready"], metadata_path=metadata_relative,
                selection_hash=selection["selection_hash"], call_failure_rate=rate(failures, measured),
                result_kind="actual_combined_profile" if summary["component"] == "combined" else "independent_stage_profile")
            row["selection_identity_verified"] = selection.get("identity_verified", False)
            row.update(pins)
            # Preserve measured phase fields without adding loading to each frame.
            row["loading"] = summary.get("loading", summary.get("load", metadata.get("load")))
            row["cold_start"] = summary.get("cold_start", {"elapsed_s": summary.get("cold_start_elapsed_s")})
            row["warmup"] = summary.get("warmup")
            row["phase_memory"] = summary.get("phase_memory")
            row["warm_per_image"] = {"latency": summary["latency"], "memory": (summary.get("phase_memory") or {}).get("warmed_inference")}
            row["limitations"] = summary.get("limitations", metadata.get("measurement_limitations", []))
            row["memory_attribution"] = metadata.get("memory_attribution")
            row["baseline_scope"] = metadata.get("baseline_scope")
            row["target_budget_bytes"] = summary.get("target_budget_bytes")
            row["budget"] = summary.get("budget")
            row["latency_measured"] = bool(valid_timing) and not summary["fixture"]
            warm_memory = (phase_memory or {}).get("warmed_inference") or {}
            row["resource_measurements_available"] = bool(measured) and not summary["fixture"] and (_has_peak(summary["memory"]) or _has_peak(warm_memory))
            for name in ("prompt_count", "prompt_counts", "query_counts", "successful_latency", "measurement_limitations"):
                if name in summary:
                    row[name] = summary[name]
            if summary["fixture"]:
                row["budget"] = None
            samples_relative = entry.get("samples_path")
            if samples_relative is None and "samples_fingerprint" in metadata:
                samples_relative = str(PurePosixPath(relative).with_name("samples.jsonl"))
            if samples_relative is not None:
                row.update(_samples(safe_path(ctx["run_dir"], samples_relative), ctx, selection, repeats, summary, metadata))
                row["samples_path"] = samples_relative
                if not row["sample_coverage_complete"]:
                    row["complete"] = False
                    row["comparison_ready"] = False
        except (ArtifactError, OSError, ValueError, TypeError, KeyError) as error:
            issue(issues, "invalid_deployment_profile", str(error), stage="deployment", summary_path=relative)
            row.update(error=str(error), valid=False, complete=False, comparison_ready=False)
        result["complete"] &= row["complete"]
        result["rows"].append(row)
    return result
