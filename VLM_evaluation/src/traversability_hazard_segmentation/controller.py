"""Diagnostic condition controller and recoverable atomic SAM handoff."""

from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
from uuid import uuid4

from traversability_data.storage import _atomic as atomic_bytes
from .common import (CONDITIONS, TASK_ID, adapter_frame, atomic_json, atomic_jsonl,
    binary_mask, exclusive_lock, fingerprint, hash_file, image_modules, load_config,
    normalize, portable_settings, read_json, read_jsonl, resolve_config, safe_path,
    validate_prompts, validate_task, verify_policy_metadata)
from .selection import freeze_with_roots, validate_selection

QUERY_KEYS = {"task_id", "schema_version", "frame_id", "condition_key", "query_id", "phrase",
    "canonical_concept", "mask_paths", "scores", "returned_instance_count", "union_mask_path", "status", "error_code"}
FRAME_KEYS = {"task_id", "schema_version", "frame_id", "condition_key", "query_ids", "requested_queries",
    "completed_queries", "failed_queries", "sam_query_count", "upstream_status", "union_mask_path", "status", "error_code"}


def _prediction_metadata(meta, inputs, model):
    validate_task(meta, "model metadata")
    if meta.get("model_key") != model:
        raise ValueError("VLM metadata model identity mismatch")
    actual_fixture = meta.get("fixture")
    if actual_fixture is None:
        execution = meta.get("execution", {})
        actual_fixture = execution.get("kind") == "injected_fixture" if execution.get("kind") in ("injected_fixture", "real_model") else None
    if actual_fixture is not inputs["fixture"]:
        raise ValueError("VLM metadata fixture provenance mismatch")
    verify_policy_metadata(meta, inputs, "VLM metadata")
    declared = meta.get("dataset_fingerprint")
    if declared is not None and declared != inputs["dataset"]["dataset_fingerprint"]:
        raise ValueError("VLM dataset identity mismatch")
    frozen_inputs = meta.get("inputs", {})
    if not isinstance(frozen_inputs, dict):
        raise ValueError("VLM inputs identity must be an object")
    declared_dataset = frozen_inputs.get("dataset")
    if declared is None and declared_dataset is None:
        raise ValueError("VLM metadata must bind dataset identity")
    if declared_dataset is not None and declared_dataset != inputs["dataset"]:
        raise ValueError("VLM input dataset identity mismatch")
    if "frames" in frozen_inputs and frozen_inputs["frames"] != [inputs["frames"][key] for key in sorted(inputs["frames"])]:
        raise ValueError("VLM frozen frame/image identity mismatch")
    if "rgb_contents" in frozen_inputs:
        actual = {row["image_path"]: {"status": "present", "sha256": row["image_sha256"]} for row in inputs["frames"].values()}
        if frozen_inputs["rgb_contents"] != actual:
            raise ValueError("VLM frozen RGB identity mismatch")


def condition_input(inputs, run_dir, condition_key, frame):
    if condition_key not in CONDITIONS:
        raise ValueError("Unknown hazard condition")
    result = {"prompts": [], "upstream_status": "ok", "prediction": None,
              "prediction_identity": None, "model_metadata": None}
    if condition_key == "fixed_policy":
        result["prompts"] = list(inputs["policy"]["canonical_prompts"])
    elif condition_key == "reference_present":
        result["prompts"] = list(inputs["references"][frame["frame_id"]]["present_concepts"])
    else:
        model = condition_key[len("vlm__"):]
        prediction_path = safe_path(run_dir, f"predictions/{model}.jsonl")
        metadata_path = safe_path(run_dir, f"metadata/{model}.json")
        if not prediction_path.exists():
            result["upstream_status"] = "missing"
            return result
        if not metadata_path.is_file():
            raise ValueError("Predictions lack model provenance metadata")
        meta = read_json(metadata_path)
        _prediction_metadata(meta, inputs, model)
        if meta.get("pending_prediction") is not None:
            raise ValueError("Inference publication is incomplete; resume inference before SAM")
        # Freeze immutable model/settings provenance, excluding ongoing attempt journals.
        model_identity = {key: meta[key] for key in ("task_id", "schema_version", "model_key", "configuration", "execution", "actual_backend", "settings", "checkpoint_revision", "inputs", "dataset_fingerprint") if key in meta}
        result["model_metadata"] = model_identity
        rows, seen = read_jsonl(prediction_path), set()
        selected = None
        for row in rows:
            validate_task(row, "prediction")
            fields = {"task_id", "schema_version", "frame_id", "model_key", "prompts", "raw_response", "status", "error_code"}
            if set(row) not in (fields, fields | {"fixture"}):
                raise ValueError("Prediction must have the frozen hazard fields")
            identifier = row.get("frame_id")
            if identifier not in inputs["frames"] or identifier in seen or row.get("model_key") != model:
                raise ValueError("Duplicate/missing prediction join or model identity mismatch")
            seen.add(identifier)
            if "fixture" in row and row["fixture"] is not inputs["fixture"]:
                raise ValueError("Prediction fixture provenance mismatch")
            if row.get("status") not in ("ok", "error") or not isinstance(row.get("raw_response"), str):
                raise ValueError("Invalid prediction status/raw response")
            validate_prompts(row.get("prompts"))
            if row["status"] == "ok" and row.get("error_code") is not None:
                raise ValueError("Successful prediction has error code")
            if row["status"] == "error" and (not isinstance(row.get("error_code"), str) or not row["error_code"]):
                raise ValueError("Failed prediction lacks diagnostic")
            if row["status"] == "error" and row["prompts"]:
                raise ValueError("Failed upstream prediction cannot expose usable prompts")
            if "prediction_digests" in meta:
                digests = meta["prediction_digests"]
                digest = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":"),
                                                  ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()
                if not isinstance(digests, dict) or digests.get(identifier) != digest:
                    raise ValueError("Saved prediction digest disagrees with inference audit")
                if "attempts" in meta:
                    attempts = meta["attempts"]
                    history = attempts.get(identifier) if isinstance(attempts, dict) else None
                    if (not isinstance(history, list) or not history or not isinstance(history[-1], dict)
                            or history[-1].get("record") != row or history[-1].get("record_sha256") != digest
                            or history[-1].get("status") != row["status"]
                            or history[-1].get("error_code") != row["error_code"]):
                        raise ValueError("Saved prediction disagrees with latest inference attempt")
            if identifier == frame["frame_id"]:
                selected = row
        if selected is None:
            result["upstream_status"] = "missing"
        else:
            result.update(prediction=selected, prediction_identity=fingerprint(selected), upstream_status=selected["status"])
            result["prompts"] = list(selected["prompts"]) if selected["status"] == "ok" else []
    validate_prompts(result["prompts"])
    return result


def _mask(array, frame):
    np, _ = image_modules()
    array = np.asarray(array)
    if array.shape != (frame["height"], frame["width"]):
        raise ValueError("Segmenter mask must have original RGB dimensions")
    if not set(np.unique(array).tolist()) <= {False, True, 0, 1, 255}:
        raise ValueError("Segmenter must return binary CPU masks")
    return (array != 0).copy()


def _publish_mask(run_dir, relative, array):
    np, Image = image_modules()
    content = io.BytesIO()
    Image.fromarray(np.asarray(array, dtype=np.uint8) * 255).save(content, format="PNG")
    payload = content.getvalue()
    path = safe_path(run_dir, relative)
    if path.exists():
        if hash_file(path) != hashlib.sha256(payload).hexdigest():
            raise ValueError("Immutable published mask conflict")
    else:
        atomic_bytes(path, payload)
    return hashlib.sha256(payload).hexdigest()


def fixture_provenance(segmenter):
    meta = getattr(segmenter, "metadata", {})
    settings = getattr(segmenter, "settings", {})
    identity = getattr(segmenter, "fixture_identity", None) or meta.get("fixture_id") or settings.get("fixture_identity")
    markers = [obj["fixture"] for obj in (meta, settings) if "fixture" in obj]
    if not isinstance(identity, str) or not identity or not markers or any(marker is not True for marker in markers):
        raise ValueError("Injected segmenter requires explicit stable fixture provenance")
    return identity


def _adapter_identity(segmenter):
    return {"settings": portable_settings(getattr(segmenter, "settings", {})),
            "metadata": portable_settings(getattr(segmenter, "metadata", {})),
            "fixture_identity": getattr(segmenter, "fixture_identity", None)}


def _shared_sam_identity(run_dir, settings, adapter=None):
    """Bind separately executed conditions to the same SAM experiment settings."""
    expected_settings = portable_settings(settings)
    for condition in CONDITIONS:
        path = safe_path(run_dir, f"segmentation/{condition}/metadata.json")
        if not path.exists():
            continue
        metadata = read_json(path)
        validate_task(metadata, "condition metadata")
        if metadata.get("condition_key") != condition:
            raise ValueError("Condition metadata identity mismatch")
        if settings is not None and metadata.get("identity", {}).get("settings") != expected_settings:
            raise ValueError("SAM settings differ across conditions; use a new run")
        previous = metadata.get("adapter")
        if adapter is not None and previous is not None and previous != adapter:
            raise ValueError("Actual SAM identity changed across conditions; use a new run")


def _upstream_failure(frame, code):
    np, _ = image_modules()
    return {"queries": [], "frame": {"frame_id": frame["frame_id"],
        "union_mask": np.zeros((frame["height"], frame["width"]), dtype=bool),
        "status": "error", "error_code": code, "sam_query_count": 0}}


def _transaction(run_dir, condition, frame, upstream, result, policy, identity):
    np, _ = image_modules()
    fid = frame["frame_id"]
    raw_queries = result.get("queries")
    returned_frame = result.get("frame", {})
    if not isinstance(raw_queries, list) or len(raw_queries) != len(upstream["prompts"]) or returned_frame.get("frame_id") != fid:
        raise ValueError("Segmenter returned incomplete query/frame joins")
    aliases = {normalize(alias): canonical for canonical, entries in policy["aliases"].items() for alias in entries}
    vague = set(map(normalize, policy["vague_unusable_phrases"]))
    invocation = hashlib.sha256(fid.encode()).hexdigest()[:24] + "_" + uuid4().hex
    base = f"segmentation/{condition}/masks/{invocation}"
    queries, assets = [], {}
    union = np.zeros((frame["height"], frame["width"]), dtype=bool)
    sam_count = 0
    for index, (phrase, raw) in enumerate(zip(upstream["prompts"], raw_queries)):
        if raw.get("phrase") != phrase or raw.get("status") not in ("ok", "error"):
            raise ValueError("Segmenter phrase order/status mismatch")
        status, error = raw["status"], raw.get("error_code")
        if (status == "ok" and error is not None) or (status == "error" and (not isinstance(error, str) or not error)):
            raise ValueError("Invalid segmenter query error semantics")
        if normalize(phrase) in vague and status != "error":
            raise ValueError("Vague query must be an explicit failure")
        masks, scores = raw.get("masks"), raw.get("scores")
        if not isinstance(masks, (list, tuple)) or not isinstance(scores, (list, tuple)) or len(masks) != len(scores):
            raise ValueError("Returned masks/scores must align")
        if any(type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
            raise ValueError("Invalid instance score")
        q_union = np.zeros_like(union)
        paths = []
        for instance, mask in enumerate(masks):
            mask = _mask(mask, frame)
            q_union |= mask
            relative = f"{base}/q{index:03d}_i{instance:03d}.png"
            assets[relative] = _publish_mask(run_dir, relative, mask)
            paths.append(relative)
        if "union_mask" in raw and not np.array_equal(q_union, _mask(raw["union_mask"], frame)):
            raise ValueError("Query union differs from returned instances")
        relative = f"{base}/q{index:03d}_union.png"
        assets[relative] = _publish_mask(run_dir, relative, q_union)
        union |= q_union
        count = raw.get("sam_query_count")
        if type(count) is not int or count not in (0, 1):
            raise ValueError("Each query needs its actual SAM call count")
        sam_count += count
        queries.append({"task_id": TASK_ID, "schema_version": 1, "frame_id": fid, "condition_key": condition,
            "query_id": f"{condition}:{fid}:q{index:03d}", "phrase": phrase,
            "canonical_concept": aliases.get(normalize(phrase)), "mask_paths": paths,
            "scores": list(scores), "returned_instance_count": len(paths), "union_mask_path": relative,
            "status": status, "error_code": error})
    if not np.array_equal(union, _mask(returned_frame.get("union_mask"), frame)):
        raise ValueError("Frame union differs from all returned instances")
    if returned_frame.get("sam_query_count") != sam_count:
        raise ValueError("Frame SAM call count mismatch")
    failures = sum(query["status"] == "error" for query in queries)
    status = "error" if failures or upstream["upstream_status"] == "error" or returned_frame.get("status") == "error" else "ok"
    if returned_frame.get("status") not in ("ok", "error") or (status == "error" and returned_frame.get("status") != "error"):
        raise ValueError("Frame status does not preserve query/upstream errors")
    code = returned_frame.get("error_code")
    if status == "error" and (not isinstance(code, str) or not code):
        raise ValueError("Failed frame requires an error code")
    if status == "ok" and code is not None:
        raise ValueError("Successful frame has error code")
    relative = f"{base}/frame_union.png"
    assets[relative] = _publish_mask(run_dir, relative, union)
    record = {"task_id": TASK_ID, "schema_version": 1, "frame_id": fid, "condition_key": condition,
        "query_ids": [query["query_id"] for query in queries], "requested_queries": len(queries),
        "completed_queries": len(queries) - failures, "failed_queries": failures, "sam_query_count": sam_count,
        "upstream_status": upstream["upstream_status"], "union_mask_path": relative, "status": status, "error_code": code}
    tx = {"identity": identity, "frame": record, "queries": queries, "asset_hashes": assets,
          "upstream": upstream, "adapter_settings": portable_settings(result.get("settings", {}))}
    tx["fingerprint"] = fingerprint(tx)
    return tx


def _validate_transaction(tx, expected, frame, run_dir, upstream, policy):
    sealed = {key: value for key, value in tx.items() if key != "fingerprint"}
    if tx.get("fingerprint") != fingerprint(sealed) or tx.get("identity") != expected:
        raise ValueError("Corrupt/incompatible frame transaction")
    record = tx["frame"]
    if set(record) != FRAME_KEYS or any(set(row) != QUERY_KEYS for row in tx["queries"]):
        raise ValueError("Published artifact key set differs from frozen contract")
    if record["frame_id"] != frame["frame_id"]:
        raise ValueError("Transaction frame mismatch")
    if tx["upstream"] != upstream:
        raise ValueError("Transaction upstream identity mismatch")
    validate_task(record, "completed frame")
    condition = expected["condition_key"]
    if record["condition_key"] != condition or record["upstream_status"] != upstream["upstream_status"]:
        raise ValueError("Transaction condition/upstream mismatch")
    rows = tx["queries"]
    if len(rows) != len(upstream["prompts"]):
        raise ValueError("Completed query count mismatch")
    aliases = {normalize(alias): canonical for canonical, names in policy["aliases"].items() for alias in names}
    vague = set(map(normalize, policy["vague_unusable_phrases"]))
    for index, (row, phrase) in enumerate(zip(rows, upstream["prompts"])):
        validate_task(row, "completed query")
        if row["frame_id"] != frame["frame_id"] or row["condition_key"] != condition or row["query_id"] != f"{condition}:{frame['frame_id']}:q{index:03d}" or row["phrase"] != phrase:
            raise ValueError("Completed query identity/phrase mismatch")
        if row["canonical_concept"] != aliases.get(normalize(phrase)):
            raise ValueError("Completed canonical bookkeeping mismatch")
        if row["status"] not in ("ok", "error") or (row["status"] == "ok" and row["error_code"] is not None) or (row["status"] == "error" and (not isinstance(row["error_code"], str) or not row["error_code"])):
            raise ValueError("Completed query status/error mismatch")
        if normalize(phrase) in vague and row["status"] != "error":
            raise ValueError("Completed vague phrase cannot succeed")
        if not isinstance(row["mask_paths"], list) or not isinstance(row["scores"], list) or len(row["mask_paths"]) != len(row["scores"]) or len(set(row["mask_paths"])) != len(row["mask_paths"]):
            raise ValueError("Completed instance/score joins mismatch")
        if type(row["returned_instance_count"]) is not int or row["returned_instance_count"] != len(row["mask_paths"]) or any(type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1 for score in row["scores"]):
            raise ValueError("Completed instance count/score invalid")
    failures = sum(row["status"] == "error" for row in rows)
    for key, value in (("requested_queries", len(rows)), ("completed_queries", len(rows) - failures), ("failed_queries", failures)):
        if type(record[key]) is not int or record[key] != value:
            raise ValueError("Completed frame query counts mismatch")
    if record["query_ids"] != [row["query_id"] for row in rows] or type(record["sam_query_count"]) is not int or not 0 <= record["sam_query_count"] <= len(rows):
        raise ValueError("Completed frame query IDs/SAM count mismatch")
    if record["status"] not in ("ok", "error") or (record["status"] == "ok" and (failures or upstream["upstream_status"] != "ok" or record["error_code"] is not None)) or (record["status"] == "error" and (not isinstance(record["error_code"], str) or not record["error_code"])):
        raise ValueError("Completed frame status/error mismatch")
    loaded = {}
    for relative, digest in tx["asset_hashes"].items():
        loaded[relative] = binary_mask(run_dir, relative, frame)
        if hash_file(safe_path(run_dir, relative, True)) != digest:
            raise ValueError("Completed mask asset changed")
    referenced = {record["union_mask_path"]}
    for query in tx["queries"]:
        referenced.update(query["mask_paths"])
        referenced.add(query["union_mask_path"])
    if referenced != set(tx["asset_hashes"]):
        raise ValueError("Transaction mask asset joins mismatch")
    np, _ = image_modules()
    frame_union = np.zeros((frame["height"], frame["width"]), dtype=bool)
    for row in rows:
        query_union = np.zeros_like(frame_union)
        for relative in row["mask_paths"]:
            query_union |= loaded[relative]
        if not np.array_equal(query_union, loaded[row["union_mask_path"]]):
            raise ValueError("Completed query mask union mismatch")
        frame_union |= query_union
    if not np.array_equal(frame_union, loaded[record["union_mask_path"]]):
        raise ValueError("Completed frame mask union mismatch")


def _merge(existing, incoming, key):
    merged = {}
    for group in (existing, incoming):
        seen = set()
        for row in group:
            identifier = row[key]
            if identifier in seen or (identifier in merged and merged[identifier] != row):
                raise ValueError("Duplicate/conflicting completed record")
            seen.add(identifier)
            merged[identifier] = row
    return [merged[key] for key in sorted(merged)]


def run_conditions(config, *, segmenter=None):
    config = load_config(config)
    validate_task({"task_id": config.get("task_id", TASK_ID), "schema_version": config.get("schema_version", 1)}, "config")
    roots = resolve_config(config)
    fixture = config.get("fixture", False)
    if type(fixture) is not bool:
        raise ValueError("fixture must be boolean")
    for flag in ("enable_real_run", "retry_errors"):
        if flag in config and type(config[flag]) is not bool:
            raise ValueError(f"{flag} must be boolean")
    if segmenter is not None:
        if not fixture:
            raise ValueError("Injected segmenter requires explicit fixture mode")
        fixture_provenance(segmenter)
    if config.get("retry_errors", False):
        raise ValueError("Terminal errors are immutable; use a new run for deliberate retries")
    options = config.get("selection", {})
    if not isinstance(options, dict):
        raise ValueError("selection must be an object")
    per_source = config.get("test_frames_per_source", options.get("test_frames_per_source", 10))
    warmups = config.get("warmup_frames", options.get("warmup_frames", 5))
    if per_source != 10 or warmups != 5:
        raise ValueError("Conditions require ten test frames per source and five development frames")
    selection = freeze_with_roots(roots["run_dir"], roots["data_root"], policy_path=roots["policy_path"], fixture=fixture,
        test_frames_per_source=per_source, warmup_frames=warmups, seed=config.get("seed", options.get("seed", 0)))
    conditions = config.get("conditions", list(CONDITIONS))
    if not isinstance(conditions, list) or len(set(conditions)) != len(conditions) or any(key not in CONDITIONS for key in conditions):
        raise ValueError("Invalid/duplicate conditions")
    summary = {"task_id": TASK_ID, "schema_version": 1, "fixture": fixture, "selection_hash": selection["fingerprint"],
               "complete": False, "comparison_ready": False, "conditions": {}}
    if not selection["complete"]:
        summary["status"] = "incomplete_selection"
        summary["coverage"] = selection["coverage"]
        return summary
    inputs = validate_selection(roots["run_dir"], roots["data_root"], selection, policy_path=roots["policy_path"], fixture=fixture)
    settings = deepcopy(config.get("sam_settings", config.get("settings", {})))
    if "fixture" in settings and settings["fixture"] is not fixture:
        raise ValueError("SAM settings fixture provenance mismatch")
    settings.update(fixture=fixture, policy_path=str(roots["policy_path"]), cache_root=str(roots["cache_root"]))
    settings["enable_model_loading"] = bool(config.get("enable_real_run", False))
    _shared_sam_identity(roots["run_dir"], settings,
                         _adapter_identity(segmenter) if segmenter is not None else None)
    owned = False
    try:
        for condition in conditions:
            base = f"segmentation/{condition}"
            paths = {name: safe_path(roots["run_dir"], f"{base}/{name}") for name in ("metadata.json", "frames.jsonl", "queries.jsonl")}
            with exclusive_lock(safe_path(roots["run_dir"], f"{base}/.writer.lock")):
                identity = {"selection_hash": selection["fingerprint"], "condition_key": condition, "settings": portable_settings(settings)}
                expected_meta = {"task_id": TASK_ID, "schema_version": 1, "fixture": fixture,
                    "execution_kind": "injected_fixture" if fixture else "real_model", "condition_key": condition,
                    "policy_hash": inputs["policy_hash"], "policy_sha256": inputs["policy_sha256"],
                    "alias_hash": inputs["alias_sha256"], "alias_sha256": inputs["alias_sha256"],
                    "dataset_fingerprint": inputs["dataset"]["dataset_fingerprint"], "selection_hash": selection["fingerprint"],
                    "identity": identity, "adapter": None, "transactions": {}, "load_attempts": [], "complete": False}
                meta = read_json(paths["metadata.json"]) if paths["metadata.json"].exists() else expected_meta
                for key in ("task_id", "schema_version", "fixture", "condition_key", "identity", "policy_hash", "alias_hash", "dataset_fingerprint"):
                    if meta.get(key) != expected_meta[key]:
                        raise ValueError("Condition identity mismatch; use a new run")
                if segmenter is not None and meta["adapter"] is not None and meta["adapter"] != _adapter_identity(segmenter):
                    raise ValueError("Actual segmenter identity changed on resume")
                frames = read_jsonl(paths["frames.jsonl"], missing_ok=True)
                queries = read_jsonl(paths["queries.jsonl"], missing_ok=True)
                # Every published terminal record needs its immutable transaction.
                transactions = {}
                tx_dir = safe_path(roots["run_dir"], f"{base}/.transactions")
                for tx_path in sorted(tx_dir.glob("*.json")) if tx_dir.exists() else []:
                    tx = read_json(tx_path)
                    fid = tx["frame"]["frame_id"]
                    if fid not in selection["test_frame_ids"] or fid in transactions:
                        raise ValueError("Unexpected/duplicate frame transaction")
                    upstream = condition_input(inputs, roots["run_dir"], condition, inputs["frames"][fid])
                    expected = {**identity, "frame_id": fid, "upstream_sha256": fingerprint(upstream)}
                    _validate_transaction(tx, expected, inputs["frames"][fid], roots["run_dir"], upstream, inputs["policy"])
                    if fid in meta["transactions"] and meta["transactions"][fid] != tx["fingerprint"]:
                        raise ValueError("Committed transaction fingerprint changed")
                    transactions[fid] = tx
                allowed_frames = [tx["frame"] for tx in transactions.values()]
                allowed_queries = [row for tx in transactions.values() for row in tx["queries"]]
                if any(row not in allowed_frames for row in frames) or any(row not in allowed_queries for row in queries):
                    raise ValueError("Published records lack committed transactions")
                frames, queries = _merge(frames, allowed_frames, "frame_id"), _merge(queries, allowed_queries, "query_id")
                if transactions:
                    atomic_jsonl(paths["queries.jsonl"], queries)
                    atomic_jsonl(paths["frames.jsonl"], frames)
                meta["transactions"] = {fid: tx["fingerprint"] for fid, tx in transactions.items()}
                atomic_json(paths["metadata.json"], meta)
                missing = []
                for fid in selection["test_frame_ids"]:
                    if fid in transactions:
                        continue
                    frame = inputs["frames"][fid]
                    upstream = condition_input(inputs, roots["run_dir"], condition, frame)
                    if upstream["upstream_status"] == "missing":
                        missing.append(fid)
                        continue
                    if upstream["upstream_status"] == "error":
                        result = _upstream_failure(frame, "upstream_vlm_error")
                    else:
                        if segmenter is None:
                            if not fixture and not config.get("enable_real_run", False):
                                missing.append(fid)
                                meta["availability"] = {"status": "unavailable", "error_code": "real_run_disabled"}
                                continue
                            from . import load_segmenter
                            try:
                                segmenter = load_segmenter(settings)
                                owned = True
                            except Exception as error:
                                diagnostic = {"status": "unavailable", "error_code": getattr(error, "error_code", "sam_load_unavailable"), "message": str(error)}
                                meta["load_attempts"].append(diagnostic)
                                meta["availability"] = diagnostic
                                missing.extend(key for key in selection["test_frame_ids"] if key not in transactions and key not in missing)
                                break
                        actual = _adapter_identity(segmenter)
                        _shared_sam_identity(roots["run_dir"], settings, actual)
                        if meta["adapter"] is not None and meta["adapter"] != actual:
                            raise ValueError("Actual segmenter identity changed")
                        meta["adapter"] = actual
                        meta["sam_settings"] = actual["settings"]
                        meta["availability"] = {"status": "available", "fixture": fixture}
                        atomic_json(paths["metadata.json"], meta)
                        validate_selection(roots["run_dir"], roots["data_root"], selection, policy_path=roots["policy_path"], fixture=fixture)
                        try:
                            result = segmenter.segment_image(adapter_frame(frame), list(upstream["prompts"]), roots["data_root"])
                        except Exception as error:
                            # A throwing adapter is a failure, with one explicit query error per requested phrase.
                            result = _upstream_failure(frame, "segmenter_exception")
                            np, _ = image_modules()
                            result["queries"] = [{"phrase": phrase, "masks": [], "scores": [], "union_mask": np.zeros((frame["height"], frame["width"]), dtype=bool), "status": "error", "error_code": "segmenter_exception", "sam_query_count": 0} for phrase in upstream["prompts"]]
                            meta.setdefault("diagnostics", {})[fid] = str(error)
                        validate_selection(roots["run_dir"], roots["data_root"], selection, policy_path=roots["policy_path"], fixture=fixture)
                        if _adapter_identity(segmenter) != actual:
                            raise ValueError("Segmenter settings mutated during call")
                        if condition_input(inputs, roots["run_dir"], condition, frame) != upstream:
                            raise ValueError("Upstream prediction changed during call")
                    tx_identity = {**identity, "frame_id": fid, "upstream_sha256": fingerprint(upstream)}
                    tx = _transaction(roots["run_dir"], condition, frame, upstream, result, inputs["policy"], tx_identity)
                    tx_path = safe_path(roots["run_dir"], f"{base}/.transactions/{hashlib.sha256(fid.encode()).hexdigest()}.json")
                    atomic_json(tx_path, tx)
                    transactions[fid] = tx
                    queries = _merge(queries, tx["queries"], "query_id")
                    frames = _merge(frames, [tx["frame"]], "frame_id")
                    atomic_jsonl(paths["queries.jsonl"], queries)
                    atomic_jsonl(paths["frames.jsonl"], frames)
                    meta["transactions"][fid] = tx["fingerprint"]
                    atomic_json(paths["metadata.json"], meta)
                meta["complete"] = len(frames) == len(selection["test_frame_ids"])
                meta["missing_frame_ids"] = missing
                atomic_json(paths["metadata.json"], meta)
                summary["conditions"][condition] = {"complete": meta["complete"], "completed_frames": len(frames),
                    "failed_frames": sum(row["status"] == "error" for row in frames), "sam_query_count": sum(row["sam_query_count"] for row in frames),
                    "missing_frame_ids": missing, "availability": meta.get("availability")}
    finally:
        if owned and segmenter is not None:
            try:
                segmenter.close()
            except Exception as error:
                summary["cleanup_error"] = {"error_code": "sam_close_error", "message": str(error)}
                if "paths" in locals():
                    meta["cleanup_error"] = summary["cleanup_error"]
                    atomic_json(paths["metadata.json"], meta)
    summary["complete"] = all(row["complete"] for row in summary["conditions"].values()) and bool(conditions)
    summary["comparison_ready"] = (summary["complete"] and not fixture
                                   and set(conditions) == set(CONDITIONS)
                                   and "cleanup_error" not in summary)
    return summary
