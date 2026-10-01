"""Standalone, transactional whole-image inference over frozen frame records."""

from copy import deepcopy
from pathlib import Path

from .configuration import configuration_snapshot, normalize_settings
from .records import prediction, validate_prediction
from .storage import (atomic_json, atomic_jsonl, content_identity, exclusive_lock,
                      read_json, read_jsonl, safe_path, stable_fingerprint,
                      validate_frames)


TASK_ID = "hazard_prompt_v1"
SCHEMA_VERSION = 1
_FRAME_KEYS = ("frame_id", "source", "scene_id", "sequence_id", "timestamp_s",
               "image_path", "split", "width", "height", "image_sha256")
_LOCAL_KEYS = frozenset({"data_root", "run_root", "run_dir", "cache_root",
                         "cache_dir", "policy_path", "repository_root"})
_BASE_KEYS = frozenset({"task_id", "schema_version", "model_key", "configuration",
                        "execution", "inputs", "actual_backend"})
_AUDIT_KEYS = ("attempts", "load_history", "prediction_digests", "pending_prediction")
_METADATA_KEYS = _BASE_KEYS | frozenset(_AUDIT_KEYS) | {
    "configuration_fingerprint", "compatibility_fingerprint", "audit_fingerprint"}


def _portable(value):
    if isinstance(value, dict):
        return {key: _portable(item) for key, item in value.items() if key not in _LOCAL_KEYS}
    if isinstance(value, (tuple, list)):
        return [_portable(item) for item in value]
    return value


def _hash(value):
    return stable_fingerprint(value)


def _base(metadata):
    return {key: metadata[key] for key in _BASE_KEYS}


def _configuration_base(metadata):
    return {key: metadata[key] for key in _BASE_KEYS if key != "actual_backend"}


def _seal(metadata):
    metadata["configuration_fingerprint"] = _hash(_configuration_base(metadata))
    metadata["compatibility_fingerprint"] = _hash(_base(metadata))
    metadata["audit_fingerprint"] = _hash({key: metadata[key] for key in _AUDIT_KEYS})
    return metadata


def _write_metadata(path, metadata):
    safe_path(path.parents[1], path.relative_to(path.parents[1]).as_posix())
    atomic_json(path, _seal(metadata))


def _new_metadata(base, actual):
    return _seal(dict(base, actual_backend=_portable(deepcopy(actual)), attempts={},
                      load_history=[], prediction_digests={}, pending_prediction=None))


def _validate_metadata(saved, base):
    if not isinstance(saved, dict) or set(saved) != _METADATA_KEYS:
        raise ValueError("Published hazard inference metadata has an invalid schema")
    for key, value in (("configuration_fingerprint", _configuration_base(saved)),
                       ("compatibility_fingerprint", _base(saved)),
                       ("audit_fingerprint", {name: saved[name] for name in _AUDIT_KEYS})):
        if saved[key] != _hash(value):
            raise ValueError(f"Published inference {key} is corrupt")
    if _hash(base) != saved["configuration_fingerprint"]:
        raise ValueError("Inference configuration or frozen inputs are incompatible with saved outputs")
    if not isinstance(saved["actual_backend"], dict):
        raise ValueError("Published actual backend metadata must be an object")
    if (not isinstance(saved["attempts"], dict) or not isinstance(saved["load_history"], list)
            or not isinstance(saved["prediction_digests"], dict)):
        raise ValueError("Published inference audit sections have invalid types")


def _read_inputs(run_dir, data_root):
    for name in ("regions.jsonl", "annotations.jsonl"):
        if safe_path(run_dir, name).exists():
            raise ValueError(f"Legacy region-classification artifact is not a hazard input: {name}")
    frames_path = safe_path(run_dir, "frames.jsonl")
    dataset_path = safe_path(run_dir, "metadata/dataset.json")
    frames = read_jsonl(frames_path)
    validate_frames(frames)
    dataset = read_json(dataset_path)
    if (dataset.get("task_id") != TASK_ID or type(dataset.get("schema_version")) is not int
            or dataset["schema_version"] != SCHEMA_VERSION):
        raise ValueError("Dataset identity does not describe hazard_prompt_v1 schema version 1")
    if type(dataset.get("fixture")) is not bool:
        raise ValueError("Dataset identity requires an explicit fixture boolean")
    if (not isinstance(dataset.get("dataset_fingerprint"), str)
            or not dataset["dataset_fingerprint"].strip()):
        raise ValueError("Dataset identity requires a nonempty dataset_fingerprint")
    contents = {}
    for frame in frames:
        relative = frame["image_path"]
        identity = content_identity(safe_path(data_root, relative))
        if identity.get("status") == "present" and identity.get("sha256") != frame["image_sha256"]:
            raise ValueError(f"RGB identity mismatch for frame: {frame['frame_id']}")
        contents[relative] = identity
    identity = {"dataset": dataset, "frames": sorted(frames, key=lambda row: row["frame_id"]),
                "rgb_contents": contents}
    frozen = {frames_path: content_identity(frames_path), dataset_path: content_identity(dataset_path)}
    if read_jsonl(frames_path) != frames or read_json(dataset_path) != dataset:
        raise RuntimeError("Frozen hazard inputs changed while being read")
    return frames, dataset, identity, frozen


def _assert_inputs(frozen, contents, data_root, frame=None):
    for path, expected in frozen.items():
        if content_identity(path) != expected:
            raise RuntimeError(f"Frozen inference identity file changed: {path.name}")
    if frame is not None:
        relative = frame["image_path"]
        if content_identity(safe_path(data_root, relative)) != contents[relative]:
            raise RuntimeError(f"Frozen RGB input changed: {relative}")


def _assert_all_inputs(frozen, contents, data_root):
    _assert_inputs(frozen, contents, data_root)
    for relative, expected in contents.items():
        if content_identity(safe_path(data_root, relative)) != expected:
            raise RuntimeError(f"Frozen RGB input changed: {relative}")


def _official(backend):
    from .qwen import QwenBackend
    return type(backend) is QwenBackend


def _execution(backend, model_key, settings, dataset):
    if backend is None or _official(backend):
        return {"kind": "real_model"}
    actual = getattr(backend, "metadata", None)
    if (not settings.get("fixture") or not dataset["fixture"] or not isinstance(actual, dict)
            or actual.get("fixture") is not True):
        raise ValueError("An injected CPU backend requires settings, dataset and backend fixture=True")
    fixture_id = actual.get("fixture_id")
    if not isinstance(fixture_id, str) or not fixture_id.strip():
        raise ValueError("An injected CPU backend requires a stable nonempty fixture_id")
    if getattr(backend, "model_key", None) != model_key:
        raise ValueError("An injected CPU backend requires the requested explicit model_key")
    return {"kind": "injected_fixture", "fixture_id": fixture_id,
            "backend_class": f"{type(backend).__module__}.{type(backend).__qualname__}"}


def _actual(backend, model_key, configuration):
    actual = getattr(backend, "metadata", None)
    if not isinstance(actual, dict):
        raise ValueError("Backend metadata must be an immutable JSON object")
    actual = _portable(deepcopy(actual))
    _hash(actual)
    if getattr(backend, "model_key", model_key) != model_key:
        raise ValueError("Loaded backend model_key does not match the requested model")
    if _official(backend):
        backend_configuration = configuration_snapshot(model_key, backend.settings)
        if backend_configuration != configuration:
            raise ValueError("Loaded backend settings do not match the requested configuration")
        if actual.get("configuration_snapshot") != configuration:
            raise ValueError("Loaded backend audit configuration does not match the requested configuration")
    return actual


def _assert_backend(backend, model_key, configuration, actual):
    try:
        if _actual(backend, model_key, configuration) != actual:
            raise ValueError("Actual backend metadata changed")
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError(f"Frozen actual backend configuration changed during inference: {error}") from error


def _assert_configuration(model_key, settings, configuration):
    try:
        if configuration_snapshot(model_key, settings) != configuration:
            raise ValueError("Policy or settings snapshot changed")
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError(f"Frozen policy or inference settings changed during inference: {error}") from error


def _predictions(path, frames, model_key):
    if not path.exists():
        return {}
    known = {frame["frame_id"] for frame in frames}
    result = {}
    for record in read_jsonl(path):
        frame_id = record.get("frame_id")
        if not isinstance(frame_id, str) or frame_id not in known:
            raise ValueError("Published prediction refers to an unknown frame")
        if frame_id in result:
            raise ValueError(f"Duplicate published prediction: {frame_id}")
        result[frame_id] = validate_prediction(record, model_key=model_key, frame_id=frame_id)
    return result


def _ordered(completed, frames):
    return [completed[frame["frame_id"]] for frame in frames if frame["frame_id"] in completed]


def _validate_transaction(metadata, completed, frames, model_key):
    known = {frame["frame_id"] for frame in frames}
    digests = metadata["prediction_digests"]
    if any(frame_id not in known or not isinstance(value, str) for frame_id, value in digests.items()):
        raise ValueError("Published prediction digest refers to an invalid frame")
    attempts = metadata["attempts"]
    if any(frame_id not in known or not isinstance(history, list) or not history
           for frame_id, history in attempts.items()):
        raise ValueError("Published attempt history refers to an invalid frame")
    attempt_digests = {}
    for frame_id, history in attempts.items():
        attempt_digests[frame_id] = set()
        for index, attempt in enumerate(history, 1):
            if (not isinstance(attempt, dict) or set(attempt) !=
                    {"attempt", "record_sha256", "status", "error_code", "diagnostics", "record"}
                    or type(attempt["attempt"]) is not int or attempt["attempt"] != index
                    or not isinstance(attempt["diagnostics"], dict)):
                raise ValueError("Published attempt history has an invalid schema")
            record = validate_prediction(attempt["record"], model_key=model_key, frame_id=frame_id)
            if (attempt["record_sha256"] != _hash(record) or attempt["status"] != record["status"]
                    or attempt["error_code"] != record["error_code"]):
                raise ValueError("Published attempt record identity is corrupt")
            attempt_digests[frame_id].add(attempt["record_sha256"])
    pending = metadata["pending_prediction"]
    pending_id = None
    if pending is not None:
        if (not isinstance(pending, dict)
                or set(pending) != {"record", "record_sha256", "previous_sha256"}):
            raise ValueError("Published pending prediction has an invalid schema")
        pending_id = pending["record"].get("frame_id") if isinstance(pending["record"], dict) else None
        if pending_id not in known:
            raise ValueError("Published pending prediction refers to an unknown frame")
        validate_prediction(pending["record"], model_key=model_key, frame_id=pending_id)
        if _hash(pending["record"]) != pending["record_sha256"]:
            raise ValueError("Published pending prediction digest is corrupt")
        if pending["previous_sha256"] != digests.get(pending_id):
            raise ValueError("Published pending prediction would replace an unrelated record")
        if (pending_id not in attempts
                or attempts[pending_id][-1]["record_sha256"] != pending["record_sha256"]):
            raise ValueError("Published pending prediction has no matching latest attempt")
    if set(attempts) != set(digests) | ({pending_id} if pending_id else set()):
        raise ValueError("Published attempt history does not match committed predictions")
    if set(completed) - set(digests) - ({pending_id} if pending_id else set()):
        raise ValueError("Published predictions have no committed audit identity")
    for frame_id, digest in digests.items():
        if digest not in attempt_digests.get(frame_id, set()):
            raise ValueError("Committed prediction has no matching original attempt record")
        if frame_id not in completed:
            raise ValueError("Committed prediction is missing from the published output")
        allowed = {digest}
        if frame_id == pending_id:
            allowed.add(pending["record_sha256"])
        if _hash(completed[frame_id]) not in allowed:
            raise ValueError("Published prediction differs from its committed audit identity")
    if pending_id in completed and pending_id not in digests:
        if _hash(completed[pending_id]) != pending["record_sha256"]:
            raise ValueError("Pending prediction conflicts with an unrelated published row")
    return pending_id


def _recover(metadata, completed, frames, model_key, metadata_path, predictions_path):
    frame_id = _validate_transaction(metadata, completed, frames, model_key)
    if frame_id is None:
        return set()
    pending = metadata["pending_prediction"]
    completed[frame_id] = pending["record"]
    safe_path(predictions_path.parents[1], predictions_path.relative_to(predictions_path.parents[1]).as_posix())
    atomic_jsonl(predictions_path, _ordered(completed, frames))
    metadata["prediction_digests"][frame_id] = pending["record_sha256"]
    metadata["pending_prediction"] = None
    _write_metadata(metadata_path, metadata)
    return {frame_id}


def _raw(result):
    if isinstance(result, str):
        return result
    if isinstance(result, dict) and isinstance(result.get("raw_response"), str):
        return result["raw_response"]
    return ""


def _diagnostics(backend):
    value = getattr(backend, "last_diagnostics", {})
    if not isinstance(value, dict):
        return {"diagnostic_error": "Backend last_diagnostics was not an object"}
    try:
        result = _portable(deepcopy(value))
        _hash(result)
        return result
    except (TypeError, ValueError) as error:
        return {"diagnostic_error": f"Backend last_diagnostics was not JSON: {error}"}


def _call(backend, frame, data_root, model_key, content):
    if content.get("status") != "present":
        code = "image_missing" if content.get("status") == "missing" else "image_unreadable"
        return prediction(frame, model_key, error_code=code), {"error_message": "RGB file is unavailable", "rgb_identity": content}
    try:
        result = backend.predict_image({key: deepcopy(frame[key]) for key in _FRAME_KEYS}, data_root)
    except Exception as error:
        code = getattr(error, "code", None)
        if not isinstance(code, str) or not code:
            code = "inference_failed"
        raw = getattr(error, "raw_response", "")
        record = prediction(frame, model_key, raw_response=raw if isinstance(raw, str) else "", error_code=code)
        return record, dict(_diagnostics(backend), error_message=f"{type(error).__name__}: {error}")
    diagnostics = _diagnostics(backend)
    try:
        result = validate_prediction(result, model_key=model_key, frame_id=frame["frame_id"])
    except (TypeError, ValueError, KeyError) as error:
        result = prediction(frame, model_key, raw_response=_raw(result), error_code="backend_output_invalid_record")
        diagnostics["error_message"] = f"{type(error).__name__}: {error}"
    return result, diagnostics


def _publish(record, diagnostics, metadata, completed, frames, metadata_path, predictions_path):
    frame_id = record["frame_id"]
    digest = _hash(record)
    history = metadata["attempts"].setdefault(frame_id, [])
    history.append({"attempt": len(history) + 1, "record_sha256": digest,
                    "status": record["status"], "error_code": record["error_code"],
                    "diagnostics": deepcopy(diagnostics), "record": deepcopy(record)})
    metadata["pending_prediction"] = {"record": deepcopy(record), "record_sha256": digest,
                                        "previous_sha256": metadata["prediction_digests"].get(frame_id)}
    _write_metadata(metadata_path, metadata)
    completed[frame_id] = record
    safe_path(predictions_path.parents[1], predictions_path.relative_to(predictions_path.parents[1]).as_posix())
    atomic_jsonl(predictions_path, _ordered(completed, frames))
    metadata["prediction_digests"][frame_id] = digest
    metadata["pending_prediction"] = None
    _write_metadata(metadata_path, metadata)


def _summary(frames, completed, dataset, execution, run_dir, metadata_path, predictions_path):
    errors = sum(record["status"] == "error" for record in completed.values())
    return {"task_id": TASK_ID, "schema_version": SCHEMA_VERSION,
            "fixture": dataset["fixture"], "execution_kind": execution["kind"],
            "expected_frames": len(frames), "requested_frames": len(frames),
            "completed_frames": len(completed), "ok_frames": len(completed) - errors,
            "error_frames": errors, "pending_frames": len(frames) - len(completed),
            "complete": len(completed) == len(frames),
            "metadata_path": metadata_path.relative_to(run_dir).as_posix(),
            "predictions_path": predictions_path.relative_to(run_dir).as_posix()}


def run_inference(run_dir, model_key, settings, data_root, *, backend=None):
    """Publish one whole-image prediction per frame, resuming only compatible runs.

    Errors are completed records. ``retry_errors=True`` explicitly requests one
    new attempt for existing errors; interrupted staged attempts are recovered
    without calling the backend again. Supplied backends remain caller-owned.
    """
    run_dir, data_root = Path(run_dir).resolve(), Path(data_root).resolve()
    settings = normalize_settings(model_key, settings)
    configuration = configuration_snapshot(model_key, settings)
    frames, dataset, inputs, frozen = _read_inputs(run_dir, data_root)
    for key in ("policy_sha256", "policy_fingerprint", "alias_sha256"):
        if key in dataset and dataset[key] != configuration[key]:
            raise ValueError(f"Dataset policy identity mismatch: {key}")
    _assert_configuration(model_key, settings, configuration)
    if settings.get("fixture") and not dataset["fixture"]:
        raise ValueError("Fixture inference requires a fixture dataset identity")
    if settings.get("fixture") and backend is None:
        raise ValueError("Fixture inference requires an explicitly supplied CPU backend")
    execution = _execution(backend, model_key, settings, dataset)
    base = {"task_id": TASK_ID, "schema_version": SCHEMA_VERSION, "model_key": model_key,
            "configuration": configuration, "execution": execution, "inputs": inputs}
    metadata_path = safe_path(run_dir, f"metadata/{model_key}.json")
    predictions_path = safe_path(run_dir, f"predictions/{model_key}.jsonl")
    lock_path = safe_path(run_dir, f"metadata/.{model_key}.lock")
    owned = backend is None
    with exclusive_lock(lock_path):
        metadata = read_json(metadata_path) if metadata_path.exists() else None
        if metadata is None and predictions_path.exists():
            raise ValueError("Orphaned predictions have no hazard inference metadata")
        if metadata is not None:
            _validate_metadata(metadata, base)
        completed = _predictions(predictions_path, frames, model_key)
        if metadata is not None and backend is not None:
            actual = _actual(backend, model_key, configuration)
            unavailable_retry = (metadata["actual_backend"].get("load_status") == "error"
                                 and settings.get("retry_errors")
                                 and all(row["status"] == "error" and row["error_code"] == "backend_load_failed"
                                         for row in completed.values()))
            if actual != metadata["actual_backend"] and not unavailable_retry:
                raise ValueError("Actual backend configuration is incompatible with saved outputs")
        _assert_configuration(model_key, settings, configuration)
        _assert_all_inputs(frozen, inputs["rgb_contents"], data_root)
        recovered = _recover(metadata, completed, frames, model_key, metadata_path, predictions_path) if metadata else set()
        pending = [frame for frame in frames if frame["frame_id"] not in recovered and
                   (frame["frame_id"] not in completed or
                    (settings.get("retry_errors") and completed[frame["frame_id"]]["status"] == "error"))]
        if not pending:
            if backend is not None and metadata is not None:
                actual = _actual(backend, model_key, configuration)
                if actual != metadata["actual_backend"]:
                    raise ValueError("Actual backend configuration is incompatible with saved outputs")
            if metadata is None:
                metadata = _new_metadata(base, {"load_status": "not_needed"})
                _write_metadata(metadata_path, metadata)
            return _summary(frames, completed, dataset, execution, run_dir, metadata_path, predictions_path)
        try:
            load_error = None
            if owned:
                try:
                    from . import load_backend
                    backend = load_backend(model_key, settings)
                except Exception as error:
                    load_error = f"{type(error).__name__}: {error}"
            if load_error is None:
                actual = _actual(backend, model_key, configuration)
                if metadata is not None and actual != metadata["actual_backend"]:
                    unavailable = metadata["actual_backend"].get("load_status") == "error"
                    eligible = not completed or all(row["status"] == "error" and row["error_code"] == "backend_load_failed" for row in completed.values())
                    if not (unavailable and eligible and (settings.get("retry_errors") or not completed)):
                        raise ValueError("Actual backend configuration is incompatible with saved outputs")
                    metadata["load_history"].append(deepcopy(metadata["actual_backend"]))
                    metadata["actual_backend"] = actual
                    _write_metadata(metadata_path, metadata)
            else:
                actual = {"load_status": "error", "error_code": "backend_load_failed",
                          "error_message": load_error, "fixture": dataset["fixture"]}
                if metadata is not None:
                    metadata["load_history"].append(deepcopy(actual))
                    # Existing actual audit remains frozen if a retry load fails.
                    actual = metadata["actual_backend"]
            if metadata is None:
                metadata = _new_metadata(base, actual)
                if load_error is not None:
                    metadata["load_history"].append(deepcopy(actual))
                _write_metadata(metadata_path, metadata)
            _assert_inputs(frozen, inputs["rgb_contents"], data_root)
            for frame in pending:
                _assert_configuration(model_key, settings, configuration)
                _assert_inputs(frozen, inputs["rgb_contents"], data_root, frame)
                if load_error is None:
                    _assert_backend(backend, model_key, configuration, actual)
                    record, diagnostics = _call(backend, frame, data_root, model_key, inputs["rgb_contents"][frame["image_path"]])
                    _assert_backend(backend, model_key, configuration, actual)
                else:
                    record = prediction(frame, model_key, error_code="backend_load_failed")
                    diagnostics = {"error_message": load_error, "stage": "load_backend"}
                _assert_inputs(frozen, inputs["rgb_contents"], data_root, frame)
                _assert_configuration(model_key, settings, configuration)
                _publish(record, diagnostics, metadata, completed, frames, metadata_path, predictions_path)
            return _summary(frames, completed, dataset, execution, run_dir, metadata_path, predictions_path)
        finally:
            if owned and backend is not None:
                backend.close()
