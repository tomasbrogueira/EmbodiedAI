"""Resumable raw current-frame inference over frozen manifests."""

from copy import deepcopy
from pathlib import Path

from .configuration import configuration_metadata, normalize_settings
from .records import InferenceError, prediction, validate_prediction
from .storage import (atomic_json, atomic_jsonl, content_identity, exclusive_lock, input_identity,
                      read_json, read_jsonl, read_manifests, stable_fingerprint)


_LOCAL_ROOT_KEYS = frozenset({"run_dir", "run_root", "data_root", "cache_dir",
                              "cache_root", "repository_root", "calibration_path"})


def _portable(value):
    if isinstance(value, dict):
        return {key: _portable(item) for key, item in value.items()
                if key not in _LOCAL_ROOT_KEYS}
    if isinstance(value, (list, tuple)):
        return [_portable(item) for item in value]
    return value


def _official_backend(backend, model_key):
    if model_key == "clip_vit_b32":
        from .clip import CLIPBackend
        official_type = CLIPBackend
    else:
        from .qwen import QwenBackend
        official_type = QwenBackend
    if type(backend) is not official_type:
        return False
    if backend.model_key != model_key:
        raise ValueError("Loaded backend model_key does not match the requested model")
    return True


def _metadata(base, actual_backend):
    result = dict(base, actual_backend=_portable(deepcopy(actual_backend)))
    # Validate that the actual runtime configuration is fully serializable before
    # any predictions can be published.
    result["configuration_fingerprint"] = stable_fingerprint(base)
    result["compatibility_fingerprint"] = stable_fingerprint(result)
    return result


def _validate_saved_metadata(saved, base):
    expected_keys = set(base) | {"actual_backend", "configuration_fingerprint",
                                 "compatibility_fingerprint"}
    if set(saved) != expected_keys:
        raise ValueError("Published inference metadata has an invalid schema")
    full = {key: value for key, value in saved.items()
            if key != "compatibility_fingerprint"}
    if stable_fingerprint(full) != saved["compatibility_fingerprint"]:
        raise ValueError("Published inference metadata fingerprint is corrupt")
    old_base = {key: saved[key] for key in base}
    if stable_fingerprint(old_base) != saved["configuration_fingerprint"]:
        raise ValueError("Published inference configuration fingerprint is corrupt")
    if stable_fingerprint(base) != saved["configuration_fingerprint"]:
        raise ValueError("Inference configuration or frozen inputs are incompatible with saved outputs")


def _read_predictions(path, regions, model_key):
    if not path.exists():
        return {}
    by_id = {region["region_id"]: region for region in regions}
    result = {}
    for record in read_jsonl(path):
        region_id = record.get("region_id")
        if not isinstance(region_id, str) or region_id not in by_id:
            raise ValueError("Published prediction refers to an unknown region")
        if region_id in result:
            raise ValueError(f"Duplicate published prediction: {region_id}")
        result[region_id] = validate_prediction(record, region=by_id[region_id], model_key=model_key)
    return result


def _raw(record):
    if isinstance(record, str):
        return record
    return record.get("raw_response", "") if isinstance(record, dict) and isinstance(record.get("raw_response", ""), str) else ""


def _error(region, model_key, code, message, raw=""):
    return prediction(region, model_key, raw_response=raw,
                      error_code=code, reason=message)


def _frame_predictions(backend, frame, regions, data_root, model_key):
    try:
        result = backend.predict_frame(deepcopy(frame), deepcopy(regions), data_root)
    except Exception as error:
        code = error.code if isinstance(error, InferenceError) else "inference_failed"
        if not isinstance(code, str) or not code:
            code = "inference_failed"
        raw = getattr(error, "raw_response", "")
        raw = raw if isinstance(raw, str) else ""
        message = f"{type(error).__name__}: {error}"
        return [_error(region, model_key, code, message, raw) for region in regions]
    if not isinstance(result, list):
        return [_error(region, model_key, "backend_output_type", "Backend must return a list of prediction records", _raw(result)) for region in regions]
    expected = {region["region_id"] for region in regions}
    returned = {}
    batch_error = None
    batch_raw = ""
    for record in result:
        if not isinstance(record, dict) or not isinstance(record.get("region_id"), str):
            batch_error = ("backend_output_invalid_record", "Backend returned a record without a valid region ID")
            batch_raw = _raw(record)
            break
        region_id = record["region_id"]
        if region_id not in expected:
            batch_error = ("backend_output_unexpected_id", f"Backend returned unrequested region: {region_id}")
            batch_raw = _raw(record)
            break
        if region_id in returned:
            batch_error = ("backend_output_duplicate_id", f"Backend returned duplicate region: {region_id}")
            break
        returned[region_id] = record
    if batch_error:
        return [_error(region, model_key, *batch_error, _raw(returned.get(region["region_id"])) or batch_raw) for region in regions]
    validated = []
    for region in regions:
        record = returned.get(region["region_id"])
        if record is None:
            validated.append(_error(region, model_key, "backend_output_missing_id", "Backend omitted the requested region"))
            continue
        try:
            validated.append(validate_prediction(record, region=region, model_key=model_key))
        except (ValueError, TypeError) as error:
            validated.append(_error(region, model_key, "backend_output_invalid_record", str(error), _raw(record)))
    return validated


def _assert_frozen(manifest_contents, input_contents, data_root, frame=None, regions=()):
    for path, expected in manifest_contents.items():
        if content_identity(path) != expected:
            raise RuntimeError(f"Frozen inference manifest changed during the run: {path.name}")
    if frame is not None:
        from .preparation import input_path
        paths = {frame["image_path"]}
        paths.update(region["mask_path"] for region in regions)
        for relative in sorted(paths):
            if content_identity(input_path(data_root, relative)) != input_contents[relative]:
                raise RuntimeError(f"Frozen inference input changed during the run: {relative}")


def _assert_backend_frozen(backend, model_key, metadata, official):
    try:
        actual = getattr(backend, "metadata", {})
        if not isinstance(actual, dict):
            raise ValueError("Backend metadata must remain a JSON object")
        if stable_fingerprint(_portable(actual)) != stable_fingerprint(metadata["actual_backend"]):
            raise ValueError("Actual backend metadata changed")
        if official:
            if backend.model_key != model_key:
                raise ValueError("Official backend model key changed")
            current = _portable(configuration_metadata(model_key, backend.settings))
            if stable_fingerprint(current) != stable_fingerprint(metadata["configuration"]):
                raise ValueError("Official backend settings changed")
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError(f"Frozen backend configuration changed during the run: {error}") from error


def run_inference(run_dir, model_key, settings, data_root, *, backend=None, region_ids=None):
    """Classify requested frozen regions and atomically resume compatible outputs.

    The returned list includes earlier completed subsets. Explicitly loaded
    official adapters retain real-model identity; other injected backends are
    fixtures. A supplied backend remains owned by its caller; internally loaded
    model resources are closed even when inference or publication fails.
    """
    run_dir, data_root = Path(run_dir), Path(data_root)
    settings = normalize_settings(model_key, settings)
    config = configuration_metadata(model_key, settings)
    frames, regions = read_manifests(run_dir)
    manifest_contents = {run_dir / name: content_identity(run_dir / name)
                         for name in ("frames.jsonl", "regions.jsonl")}
    if read_manifests(run_dir) != (frames, regions):
        raise RuntimeError("Frozen inference manifests changed while being read")
    known_ids = {region["region_id"] for region in regions}
    if region_ids is None:
        requested = {region["region_id"] for region in regions if region["selected_for_classification"]}
    else:
        if isinstance(region_ids, (str, bytes)):
            raise ValueError("region_ids must be an iterable of unique region ID strings")
        selected_ids = list(region_ids)
        if any(not isinstance(value, str) or value not in known_ids for value in selected_ids):
            raise ValueError("region_ids contains an unknown or invalid region ID")
        if len(set(selected_ids)) != len(selected_ids):
            raise ValueError("region_ids contains duplicate region IDs")
        requested = set(selected_ids)
    execution = {"kind": "real_model"}
    if backend is not None and _official_backend(backend, model_key):
        if _portable(configuration_metadata(model_key, backend.settings)) != _portable(config):
            raise ValueError("Loaded backend settings do not match the requested configuration")
    elif backend is not None:
        execution = {"kind": "injected_fixture", "backend_class": f"{type(backend).__module__}.{type(backend).__qualname__}",
                     "fixture_identity": _portable(getattr(backend, "fixture_identity", None))}
    base = {"metadata_version": 1, "model_key": model_key,
            "configuration": _portable(config), "execution": execution,
            "inputs": input_identity(frames, regions, data_root)}
    _assert_frozen(manifest_contents, base["inputs"]["contents"], data_root)
    metadata_path = run_dir / "metadata" / f"{model_key}.json"
    predictions_path = run_dir / "predictions" / f"{model_key}.jsonl"
    owned = backend is None
    with exclusive_lock(run_dir / "metadata" / f".{model_key}.lock"):
        saved = read_json(metadata_path) if metadata_path.exists() else None
        if saved is None and predictions_path.exists():
            raise ValueError("Orphaned predictions have no immutable inference metadata")
        if saved is not None:
            _validate_saved_metadata(saved, base)
        completed = _read_predictions(predictions_path, regions, model_key)
        try:
            if owned:
                # The package imports no model libraries until loading is requested.
                from . import load_backend
                backend = load_backend(model_key, settings)
            actual = getattr(backend, "metadata", {})
            if not isinstance(actual, dict):
                raise ValueError("Backend metadata must be a JSON object")
            if model_key == "clip_vit_b32" and "calibration" in actual:
                from .calibration import validate_calibration_inputs
                validate_calibration_inputs(actual["calibration"], frames, regions, data_root)
            metadata = _metadata(base, actual)
            if saved is not None and metadata != saved:
                raise ValueError("Actual backend configuration is incompatible with saved outputs")
            official = _official_backend(backend, model_key)
            _assert_backend_frozen(backend, model_key, metadata, official)
            _assert_frozen(manifest_contents, base["inputs"]["contents"], data_root)
            if saved is None:
                atomic_json(metadata_path, metadata)
            by_frame = {}
            for region in regions:
                if region["region_id"] in requested and region["region_id"] not in completed:
                    by_frame.setdefault(region["frame_id"], []).append(region)
            for frame in frames:
                pending = by_frame.get(frame["frame_id"], [])
                if not pending:
                    continue
                _assert_backend_frozen(backend, model_key, metadata, official)
                _assert_frozen(manifest_contents, base["inputs"]["contents"], data_root, frame, pending)
                new_records = _frame_predictions(backend, frame, pending, data_root, model_key)
                _assert_backend_frozen(backend, model_key, metadata, official)
                _assert_frozen(manifest_contents, base["inputs"]["contents"], data_root, frame, pending)
                completed.update({record["region_id"]: record for record in new_records})
                accumulated = [completed[region["region_id"]] for region in regions
                               if region["region_id"] in completed]
                atomic_jsonl(predictions_path, accumulated)
            return [completed[region["region_id"]] for region in regions if region["region_id"] in completed]
        finally:
            if owned and backend is not None:
                backend.close()
