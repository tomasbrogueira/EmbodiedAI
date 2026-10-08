"""Independent-reference evaluation for semantic_mapping_v1 (CPU-only).

No annotation is read by the runtime. This module evaluates complete artifacts,
keeps the frozen ROI denominator, and never turns internal planner success into
navigation accuracy. See REFERENCE_SCHEMA below for the evaluator-only format.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import CONTRACT_ID, PIPELINE_IDS, SCHEMA_VERSION

METRIC_DEFINITION_VERSION = "semantic_mapping_metrics_v1"
BASE_METRIC_IDS = (
    "decision_coverage", "unknown_rate", "safe_precision", "safe_recall",
    "unsafe_as_traversable_rate", "reference_valid_path_rate",
    "pipeline_failure_rate", "semantic_query_count", "semantic_update_rate_hz",
    "geometry_stage_ms.median", "geometry_stage_ms.p95",
    "semantic_call_ms.median", "semantic_call_ms.p95",
    "end_to_end_latency_ms.median", "end_to_end_latency_ms.p95",
    "semantic_evidence_age_ms.median", "semantic_evidence_age_ms.p95",
    "joint_gpu_allocated_peak_bytes", "joint_gpu_reserved_peak_bytes",
    "host_rss_peak_bytes",
)

# This is a public schema description for independently authored fixture/data
# builders, not inferred reference labels or a hidden evaluator input to runtime.
REFERENCE_SCHEMA = {
    "identity": ["contract_id", "schema_version", "reference_id", "reference_protocol_id",
                 "sequence", "fixture", "provenance", "taxonomy", "coordinates", "assets", "kinds"],
    "asset": {"asset_id": "unique string", "path": "relative contained path", "sha256": "file bytes SHA256"},
    "grid": {"shape": "[rows, cols]", "origin": "[x,y,z]", "resolution": "positive scalar",
             "projection_basis": "orthonormal [2,3]", "map_frame": "declared string"},
    "robot_decision": {"asset_id": "NPZ asset", "decision_key": "decision_state",
                       "valid_key": "valid_mask", "ignore_key": "ignore_mask", "grid": "grid",
                       "coverage": "explicit description", "robot_reference_policy": "independent declared policy"},
    "semantic_2d": {"coverage": "explicit description", "frames": [
        {"frame_id": "input ID", "processed_grid_id": "exact evaluated grid",
         "decoded_rgb_sha256": "exact evaluated pixels", "asset_id": "NPZ asset",
         "valid_key": "valid_mask", "ignore_key": "ignore_mask", "concept_masks": {"concept_id": "bool NPZ key"}}]},
    "semantic_3d": {"asset_id": "NPZ asset", "voxel_indices_key": "voxel_indices int64[N,3]",
                    "valid_key": "valid_mask bool[N]", "ignore_key": "ignore_mask bool[N]",
                    "concept_masks": {"concept_id": "bool NPZ key[N]"},
                    "voxel_origin": "same run voxel origin", "voxel_resolution": "same run resolution",
                    "coverage": "explicit description", "prediction_threshold": "frozen run threshold"},
    "navigation": {"asset_id": "NPZ asset", "safe_key": "safe_for_robot bool[rows,cols]",
                   "valid_key": "valid_mask", "ignore_key": "ignore_mask", "grid": "grid",
                   "requests": [{"request_id": "declared request", "valid_reference": True}],
                   "coverage": "explicit description", "robot_fingerprint": "sha256 canonical run robot policy",
                   "robot_reference_policy": {"independent": True, "footprint_and_clearance_certified": True}},
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value: Any) -> str:
    """SHA256 of UTF-8 canonical JSON (sorted keys, compact separators)."""
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    _canonical(value)  # Reject nonfinite JSON constants, including Python's NaN.
    return value


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{number}")
            _canonical(row)
            rows.append(row)
    return rows


def _contained(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or not relative or "\\" in relative:
        raise ValueError("Asset paths must be portable relative paths")
    path = (root / candidate).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing/outside asset: {relative}")
    return path


def _npz(path: Path, *, allow_nonfinite=False) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        result = {name: arrays[name].copy() for name in arrays.files}
    for name, array in result.items():
        if array.dtype.kind not in "biuf":
            raise ValueError(f"Non-numeric/object array {name} in {path}")
        if not allow_nonfinite and array.dtype.kind in "f" and not np.isfinite(array).all():
            raise ValueError(f"Nonfinite array {name} in {path}")
    return result


def _metric(value=None, *, numerator=None, denominator=None, unit="fraction",
            status="unavailable", reason="not measured", coverage=None, **metadata) -> dict:
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value)):
        raise ValueError("Metric values must be finite numeric or null")
    if status not in {"available", "unavailable", "not_applicable"}:
        raise ValueError("Unknown metric status")
    if status != "available" and value is not None:
        raise ValueError("Unavailable metrics must be null")
    if status == "available" and value is None:
        raise ValueError("Available metrics must have a measured finite value")
    return dict(value=value, numerator=numerator, denominator=denominator, unit=unit,
                status=status, reason=reason, reference_coverage=coverage, **metadata)


def _ratio(numerator: int, denominator: int, *, coverage=None, **metadata) -> dict:
    return _metric(numerator / denominator if denominator else None,
                   numerator=int(numerator), denominator=int(denominator),
                   status="available" if denominator else "unavailable",
                   reason=None if denominator else "empty denominator", coverage=coverage, **metadata)


def _grid(spec: dict) -> dict:
    shape = np.asarray(spec.get("shape"))
    origin = np.asarray(spec.get("origin"), dtype=float)
    basis = np.asarray(spec.get("projection_basis"), dtype=float)
    resolution = spec.get("resolution")
    if shape.shape != (2,) or shape.dtype.kind not in "iu" or (shape <= 0).any():
        raise ValueError("Frozen evaluation/reference grid requires positive integer shape [rows,cols]")
    if basis.shape == (3, 3):
        if not np.isfinite(basis).all() or not np.allclose(basis @ basis.T, np.eye(3), atol=1e-6):
            raise ValueError("Planner projection basis must be orthonormal [3,3]")
        basis = basis[:2]
    if origin.shape == (2,) and basis.shape == (2, 3):
        origin = origin @ basis
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("Grid origin must be finite XYZ [3] or projected XY [2]")
    if basis.shape != (2, 3) or not np.isfinite(basis).all() or not np.allclose(basis @ basis.T, np.eye(2), atol=1e-6):
        raise ValueError("Grid projection_basis must be orthonormal [2,3]")
    if isinstance(resolution, bool) or not isinstance(resolution, (int, float)) or not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("Grid resolution must be positive")
    if not isinstance(spec.get("map_frame"), str) or not spec["map_frame"]:
        raise ValueError("Grid map_frame is required")
    return {"shape": tuple(int(x) for x in shape), "origin": origin,
            "basis": basis, "resolution": float(resolution), "map_frame": spec["map_frame"]}


def _centers(grid: dict) -> np.ndarray:
    rows, cols = np.indices(grid["shape"])
    return (grid["origin"] + (cols[..., None] + .5) * grid["resolution"] * grid["basis"][0]
            + (rows[..., None] + .5) * grid["resolution"] * grid["basis"][1])


def _sample(array: np.ndarray, source: dict, target: dict) -> np.ndarray:
    if source["map_frame"] != target["map_frame"] or not np.allclose(source["basis"], target["basis"], atol=1e-6):
        raise ValueError("Grid map frame/projection basis mismatch")
    points = _centers(target)
    xy = (points - source["origin"]) @ source["basis"].T / source["resolution"]
    # Floor is intentional: endpoints/cells are never snapped into coverage.
    col, row = np.floor(xy[..., 0]).astype(int), np.floor(xy[..., 1]).astype(int)
    inside = (row >= 0) & (col >= 0) & (row < array.shape[0]) & (col < array.shape[1])
    output = np.zeros(target["shape"], dtype=array.dtype)
    output[inside] = array[row[inside], col[inside]]
    return output


def _costmap(run_dir: Path, target: dict) -> np.ndarray:
    # Unknown support diagnostics legitimately contain NaN and blocked costs
    # infinity. The consumed decision/grid fields are checked explicitly.
    data = _npz(run_dir / "planning/costmap.npz", allow_nonfinite=True)
    state = data.get("decision_state")
    if state is None or state.dtype != np.uint8 or state.ndim != 2 or not np.isin(state, (0, 1, 2)).all():
        raise ValueError("costmap decision_state must be uint8[rows,cols], codes 0/1/2")
    if state.size == 0:
        return np.zeros(target["shape"], dtype=np.uint8)
    spec = {"shape": list(state.shape), "origin": data["origin"].tolist(),
            "projection_basis": data["projection_basis"].tolist(),
            "resolution": float(data["resolution"].item()), "map_frame": target["map_frame"]}
    return _sample(state, _grid(spec), target)


def _validate_producer_frame(run_dir: Path, config: dict, grid: dict) -> None:
    """No implicit frame transform/relabel is allowed at the evaluator boundary."""
    declared = []
    for label in ("geometry", "geometry_identity", "planning"):
        section = config.get(label)
        if isinstance(section, dict) and section.get("map_frame"):
            declared.append((label, section["map_frame"]))
    for relative in ("geometry/manifest.json", "map/manifest.json", "planning/manifest.json"):
        path = run_dir / relative
        if path.exists():
            manifest = _json(path)
            if manifest.get("map_frame"):
                declared.append((relative, manifest["map_frame"]))
            coordinates = manifest.get("coordinates", {})
            if isinstance(coordinates, dict) and coordinates.get("map_frame"):
                declared.append((relative + ".coordinates", coordinates["map_frame"]))
    for source, frame in declared:
        if frame != grid["map_frame"]:
            raise ValueError(f"Producer {source} map frame {frame!r} differs from frozen evaluation frame {grid['map_frame']!r}; verified transforms are unsupported")


def _validate_reference(path: Path, config: dict, fixture: bool) -> tuple[dict, dict]:
    reference = _json(path)
    for key in REFERENCE_SCHEMA["identity"]:
        if key not in reference:
            raise ValueError(f"Reference missing {key}")
    if reference["contract_id"] != CONTRACT_ID or reference["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Reference contract/version mismatch")
    if reference["sequence"] != config["sequence"]:
        # Ancillary run frame_count is allowed; identities remain exact.
        keys = ("sequence_id", "split", "manifest_digest")
        if any(reference["sequence"].get(k) != config["sequence"].get(k) for k in keys):
            raise ValueError("Reference sequence/split/digest mismatch")
    if type(reference["fixture"]) is not bool or reference["fixture"] != fixture:
        raise ValueError("Reference fixture provenance mismatch")
    provenance = reference["provenance"]
    if provenance.get("independent") is not True or provenance.get("derived_from_evaluated_models", False):
        raise ValueError("References must be independent of evaluated models")
    if not provenance.get("source") or not reference["reference_id"] or not reference["reference_protocol_id"]:
        raise ValueError("Reference IDs and independent source are required")
    if not reference["taxonomy"].get("version") or not isinstance(reference["taxonomy"].get("concept_ids"), list):
        raise ValueError("Reference taxonomy/version is required")
    concepts = reference["taxonomy"]["concept_ids"]
    if any(not isinstance(c, str) or not c for c in concepts) or len(set(concepts)) != len(concepts):
        raise ValueError("Reference concept IDs must be unique strings")
    assets = {}
    for asset in reference["assets"]:
        identity = asset["asset_id"]
        if not identity or identity in assets:
            raise ValueError("Duplicate/missing reference asset identity")
        actual_path = _contained(path.parent, asset["path"])
        actual_hash = hashlib.sha256(actual_path.read_bytes()).hexdigest()
        if actual_hash != asset["sha256"]:
            raise ValueError(f"Reference asset hash mismatch: {identity}")
        assets[identity] = _npz(actual_path)
    if not isinstance(reference["kinds"], dict):
        raise ValueError("Reference kinds must be an object")
    return reference, assets


def _physical_reference_reason(reference: dict, config: dict) -> str | None:
    coordinates = reference["coordinates"]
    for key in ("alignment", "scale_provenance", "pose_provenance"):
        if not isinstance(coordinates.get(key), dict) or coordinates[key].get("verified") is not True or not coordinates[key].get("source"):
            return f"reference {key} is unverified"
    if coordinates.get("units") not in {"metres", "meters", "m"}:
        return "reference metric scale is unavailable"
    if coordinates.get("map_frame") != (config.get("evaluation_grid") or {}).get("map_frame"):
        return "reference map frame is not aligned to the frozen evaluation grid"
    up = np.asarray(coordinates.get("up"), dtype=float)
    axes = coordinates.get("axes")
    if up.shape != (3,) or not np.isfinite(up).all() or not np.isclose(np.linalg.norm(up), 1) or not isinstance(axes, list) or len(axes) != 3:
        return "reference axes/up are unavailable"
    geometry = config.get("geometry", {})
    if not isinstance(config.get("robot"), dict) or not config["robot"].get("version"):
        return "configured robot profile/version unavailable"
    run_up = geometry.get("up")
    if isinstance(run_up, dict):
        if run_up.get("verified") is not True or not run_up.get("source"):
            return "reconstruction up is unverified"
        run_up = run_up.get("vector")
    if run_up is None:
        return "reconstruction up is unavailable"
    run_up = np.asarray(run_up, dtype=float)
    if run_up.shape != (3,) or not np.isfinite(run_up).all() or np.linalg.norm(run_up) <= 0:
        return "reconstruction up is invalid"
    if not np.allclose(up, run_up / np.linalg.norm(run_up)):
        return "reference and reconstruction up differ"
    scale = geometry.get("scale")
    if (not isinstance(scale, dict) or scale.get("verified") is not True or not scale.get("source")
            or not isinstance(scale.get("meters_per_unit"), (int, float)) or scale["meters_per_unit"] <= 0):
        return "reconstruction metric scale is unverified"
    return None


def _kind_masks(kind: dict, assets: dict, shape: tuple) -> tuple[dict, np.ndarray]:
    if not isinstance(kind.get("coverage"), dict) or not kind["coverage"]:
        raise ValueError("Reference kinds must declare coverage")
    data = assets[kind["asset_id"]]
    valid = data[kind.get("valid_key", "valid_mask")]
    ignore = data[kind.get("ignore_key", "ignore_mask")]
    if valid.dtype != np.bool_ or ignore.dtype != np.bool_ or valid.shape != shape or ignore.shape != shape:
        raise ValueError("Reference valid/ignore masks must be boolean on the declared grid")
    return data, valid & ~ignore


def _decision_metrics(metrics: dict, reference: dict, assets: dict, config: dict, predicted: np.ndarray, grid: dict) -> None:
    ids = ("safe_precision", "safe_recall", "unsafe_as_traversable_rate")
    kind = reference["kinds"].get("robot_decision")
    reason = _physical_reference_reason(reference, config)
    if kind is None:
        reason = "independent robot-decision reference kind unavailable"
    elif kind.get("robot_reference_policy", {}).get("independent") is not True:
        reason = "independent robot-reference policy unavailable; material labels cannot imply safety"
    elif (kind["robot_reference_policy"].get("robot_fingerprint") != fingerprint(config.get("robot"))
          and kind["robot_reference_policy"].get("robot") != config.get("robot")):
        reason = "robot reference policy does not identify the exact configured robot"
    if grid is None:
        reason = "frozen evaluation ROI/grid unavailable"
    if reason:
        for name in ids:
            metrics[name] = _metric(reason=reason)
        return
    reference_grid = _grid(kind["grid"])
    data, valid = _kind_masks(kind, assets, reference_grid["shape"])
    truth = data[kind.get("decision_key", "decision_state")]
    if truth.dtype != np.uint8 or truth.shape != reference_grid["shape"] or not np.isin(truth, (0, 1, 2)).all():
        raise ValueError("Robot reference decision_state must be uint8 codes 0/1/2")
    valid &= truth != 0
    # Score C only where it intersects D; no prediction-dependent denominator.
    domain = _sample(np.ones(grid["shape"], dtype=bool), grid, reference_grid)
    valid &= domain
    pred_c = _sample(predicted, grid, reference_grid)
    tp = int(np.count_nonzero(valid & (truth == 2) & (pred_c == 2)))
    fp = int(np.count_nonzero(valid & (truth == 1) & (pred_c == 2)))
    safe = int(np.count_nonzero(valid & (truth == 2)))
    unsafe = int(np.count_nonzero(valid & (truth == 1)))
    coverage = dict(kind["coverage"], valid_cells=int(valid.sum()), frozen_roi_cells=int(predicted.size))
    metrics["safe_precision"] = _ratio(tp, tp + fp, coverage=coverage)
    metrics["safe_recall"] = _ratio(tp, safe, coverage=coverage)
    metrics["unsafe_as_traversable_rate"] = _ratio(fp, unsafe, coverage=coverage)


def _concepts(run_dir: Path, reference: dict | None) -> list[str]:
    concepts = set(reference["taxonomy"]["concept_ids"] if reference else [])
    path = run_dir / "map/concepts.json"
    if path.exists():
        obj = json.loads(path.read_text(encoding="utf-8"))
        rows = obj if isinstance(obj, list) else obj.get("concepts", obj.get("registry", []))
        for row in rows:
            concepts.add(row if isinstance(row, str) else row["concept_id"])
    return sorted(concepts)


def _semantic_scores(tp: int, fp: int, fn: int, coverage: dict) -> dict:
    return {"precision": _ratio(tp, tp + fp, coverage=coverage),
            "recall": _ratio(tp, tp + fn, coverage=coverage),
            "iou": _ratio(tp, tp + fp + fn, coverage=coverage)}


def _semantic_2d(run_dir: Path, rows: list[dict], kind: dict, assets: dict, concepts: list[str]) -> dict:
    if not kind.get("coverage"):
        raise ValueError("Semantic 2D reference coverage is required")
    lookup = {}
    for row in rows:
        if row["frame_id"] in lookup:
            raise ValueError("Duplicate semantic frame identity")
        lookup[row["frame_id"]] = row
    totals = {c: np.zeros(3, dtype=np.int64) for c in concepts}
    supported = {c: 0 for c in concepts}
    reference_ids = set()
    for frame in kind["frames"]:
        if frame["frame_id"] in reference_ids:
            raise ValueError("Duplicate semantic reference frame identity")
        reference_ids.add(frame["frame_id"])
        data = assets[frame["asset_id"]]
        valid = data[frame.get("valid_key", "valid_mask")]
        ignore = data[frame.get("ignore_key", "ignore_mask")]
        if valid.dtype != np.bool_ or valid.ndim != 2 or ignore.dtype != np.bool_ or ignore.shape != valid.shape:
            raise ValueError("Semantic reference valid/ignore masks must be bool[H,W]")
        coverage = valid & ~ignore
        row = lookup.get(frame["frame_id"])
        if row is not None:
            for key in ("processed_grid_id", "decoded_rgb_sha256"):
                if not frame.get(key) or row.get(key) != frame[key]:
                    raise ValueError(f"Semantic reference {key} mismatch")
        predictions = {c: np.zeros(valid.shape, dtype=bool) for c in frame["concept_masks"]}
        if row is not None and row.get("status") in {"ok", "partial"}:
            for query in row.get("queries", []):
                cid = query["concept_id"]
                if query.get("status") != "ok" or cid not in predictions:
                    continue
                for instance in query.get("instances", []):
                    relative = instance.get("mask_path", instance.get("mask_file"))
                    key = instance.get("mask_key", "mask")
                    if not relative:
                        raise ValueError("Semantic instance requires mask_path and mask_key")
                    mask = _npz(_contained(run_dir, relative))[key]
                    if mask.dtype != np.bool_ or mask.shape != valid.shape:
                        raise ValueError("Predicted semantic mask must match the exact reference grid")
                    if instance.get("processed_grid_id", row["processed_grid_id"]) != row["processed_grid_id"]:
                        raise ValueError("Semantic mask grid mismatch")
                    predictions[cid] |= mask
        for cid, mask_key in frame["concept_masks"].items():
            if cid not in totals:
                raise ValueError("Semantic reference concept absent from declared taxonomy")
            truth = data[mask_key]
            if truth.dtype != np.bool_ or truth.shape != valid.shape:
                raise ValueError("Semantic reference masks must be boolean on reference grid")
            pred = predictions[cid]
            totals[cid] += (np.count_nonzero(coverage & pred & truth),
                            np.count_nonzero(coverage & pred & ~truth),
                            np.count_nonzero(coverage & ~pred & truth))
            supported[cid] += int(coverage.sum())
    return {cid: _semantic_scores(*map(int, totals[cid]),
              dict(kind["coverage"], kind="semantic_2d", valid_pixels=supported[cid], frames=len(reference_ids)))
            for cid in concepts if supported[cid]}


def _semantic_3d(run_dir: Path, kind: dict, assets: dict, concepts: list[str], config: dict) -> dict:
    geometry = config.get("fusion", {})
    origin = geometry.get("origin", geometry.get("voxel_origin", [0, 0, 0]))
    size = geometry.get("voxel_size", geometry.get("voxel_resolution", geometry.get("resolution", .25)))
    map_manifest_path = run_dir / "map/manifest.json"
    if map_manifest_path.exists():
        map_manifest = _json(map_manifest_path)
        if map_manifest.get("origin", origin) != origin or map_manifest.get("voxel_size", size) != size:
            raise ValueError("Map manifest and frozen fusion voxel grid mismatch")
        origin = map_manifest.get("origin", origin)
        size = map_manifest.get("voxel_size", size)
    if kind.get("voxel_origin") != origin or kind.get("voxel_resolution") != size:
        raise ValueError("Semantic 3D reference voxel grid mismatch")
    threshold = config.get("evaluation", {}).get("semantic_evidence_threshold", geometry.get("semantic_threshold"))
    if threshold is None or kind.get("prediction_threshold") != threshold:
        raise ValueError("Semantic 3D prediction threshold must be frozen in run config")
    data = assets[kind["asset_id"]]
    indices = data[kind.get("voxel_indices_key", "voxel_indices")]
    if indices.dtype != np.int64 or indices.ndim != 2 or indices.shape[1] != 3 or len(set(map(tuple, indices))) != len(indices):
        raise ValueError("Reference voxel indices must be unique int64[N,3]")
    _, valid = _kind_masks(kind, assets, (len(indices),))
    voxel_data = _npz(run_dir / "map/voxels.npz")
    evidence = _npz(run_dir / "map/semantic_evidence.npz")
    registry = json.loads((run_dir / "map/concepts.json").read_text(encoding="utf-8"))
    registry = registry if isinstance(registry, list) else registry.get("concepts", registry.get("registry", []))
    registry = [row if isinstance(row, str) else row["concept_id"] for row in registry]
    prediction_sets = {cid: set() for cid in concepts}
    for vr, cr, score in zip(evidence["voxel_row"], evidence["concept_row"], evidence["evidence_score"]):
        if vr < 0 or vr >= len(voxel_data["voxel_indices"]) or cr < 0 or cr >= len(registry):
            raise ValueError("Semantic evidence references invalid registry/voxel row")
        cid = registry[int(cr)]
        if cid in prediction_sets and score >= threshold:
            prediction_sets[cid].add(tuple(voxel_data["voxel_indices"][int(vr)]))
    scores = {}
    for cid, key in kind["concept_masks"].items():
        if cid not in prediction_sets:
            raise ValueError("Semantic reference concept absent from taxonomy")
        truth = data[key]
        if truth.dtype != np.bool_ or truth.shape != valid.shape:
            raise ValueError("Reference voxel masks must be bool[N]")
        pred = np.array([tuple(index) in prediction_sets[cid] for index in indices], dtype=bool)
        scores[cid] = _semantic_scores(int((valid & pred & truth).sum()), int((valid & pred & ~truth).sum()),
                                      int((valid & ~pred & truth).sum()),
                                      dict(kind["coverage"], kind="semantic_3d", valid_voxels=int(valid.sum())))
    return scores


def _navigation(metrics: dict, run_dir: Path, reference: dict, assets: dict, config: dict) -> None:
    kind = reference["kinds"].get("navigation")
    reason = _physical_reference_reason(reference, config)
    if kind is None:
        reason = "independent navigation reference kind unavailable"
    elif (kind.get("robot_reference_policy", {}).get("independent") is not True
          or kind["robot_reference_policy"].get("footprint_and_clearance_certified") is not True
          or kind.get("robot_fingerprint") != fingerprint(config.get("robot"))):
        reason = "independent footprint/clearance certification for the exact robot policy unavailable"
    if reason:
        metrics["reference_valid_path_rate"] = _metric(reason=reason)
        return
    grid = _grid(kind["grid"])
    data, valid = _kind_masks(kind, assets, grid["shape"])
    safe = data[kind.get("safe_key", "safe_for_robot")]
    if safe.dtype != np.bool_ or safe.shape != grid["shape"]:
        raise ValueError("Navigation safe_for_robot must be bool on the reference grid")
    safe &= valid
    plans = _jsonl(run_dir / "planning/plans.jsonl")
    lookup = {p["request_id"]: p for p in plans}
    if len(lookup) != len(plans):
        raise ValueError("Duplicate plan request identity")
    success = denominator = 0
    seen = set()
    goals = config.get("goals", [])
    goals = goals.get("requests", []) if isinstance(goals, dict) else goals
    requested = {g["request_id"]: g for g in goals}
    for request in kind["requests"]:
        rid = request["request_id"]
        if rid in seen:
            raise ValueError("Duplicate navigation reference request")
        seen.add(rid)
        if request.get("valid_reference") is not True:
            continue
        if rid not in requested:
            raise ValueError("Navigation reference request was not part of the frozen experiment")
        denominator += 1
        plan = lookup.get(rid, {})
        if plan.get("status") != "ok" or plan.get("map_frame", plan.get("frame")) != grid["map_frame"]:
            continue
        path = np.asarray(plan.get("path", plan.get("path_coordinates", [])), dtype=float)
        if path.ndim != 2 or path.shape[1] != 3 or len(path) < 1 or not np.isfinite(path).all():
            continue
        goal = requested[rid]
        if not np.allclose(path[0], goal["start"], atol=1e-8, rtol=0) or not np.allclose(path[-1], goal["goal"], atol=1e-8, rtol=0):
            continue
        # Conservative raster validation at <=1/4 cell, against independent
        # footprint/clearance-certified reference cells. This is static reference
        # path validity, not measured robot execution/controller success.
        samples = [path[0]]
        for start, end in zip(path[:-1], path[1:]):
            count = max(1, int(np.ceil(np.linalg.norm(end - start) / (grid["resolution"] / 4))))
            samples.extend(start + t * (end - start) for t in np.linspace(0, 1, count + 1)[1:])
        xy = (np.asarray(samples) - grid["origin"]) @ grid["basis"].T / grid["resolution"]
        col, row = np.floor(xy[:, 0]).astype(int), np.floor(xy[:, 1]).astype(int)
        inside = (row >= 0) & (col >= 0) & (row < safe.shape[0]) & (col < safe.shape[1])
        if inside.all() and safe[row, col].all():
            success += 1
    metrics["reference_valid_path_rate"] = _ratio(success, denominator, coverage=kind["coverage"],
        validation_scope="static raster reference certified for configured footprint/clearance; no robot execution claim")


def _operational(metrics: dict, summary: dict, frame_rows: list[dict], semantic_rows: list[dict], config: dict, semantic_capable: bool) -> dict:
    required = summary.get("required_frame_attempts", config["sequence"].get("frame_count", len(frame_rows)))
    if type(required) is not int or required < len(frame_rows):
        raise ValueError("Invalid required frame-attempt denominator")
    failed = required - len(frame_rows)
    ids = set()
    for row in frame_rows:
        frame_id = row.get("frame_id")
        if not frame_id or frame_id in ids:
            raise ValueError("Duplicate/missing frame disposition identity")
        ids.add(frame_id)
        status = row.get("status", row.get("disposition"))
        if status in {"error", "failed", "failure", "missing", "absent", "dropped", "skipped"} or row.get("required_failed") is True:
            failed += 1
    metrics["pipeline_failure_rate"] = _ratio(failed, required, coverage={"required_frame_attempts": required},
                                               definition="required failed/absent frame attempts; queue outcomes retained separately")
    counts = {"model_calls": {}, "query_count": 0, "attempted_queries": 0, "successful_queries": 0,
              "semantic_observations_completed": 0, "semantic_records": len(semantic_rows),
              "query_count_unknown_frames": [], "known_query_count": 0}
    semantic_ids = set()
    for row in semantic_rows:
        if not row.get("frame_id") or row["frame_id"] in semantic_ids:
            raise ValueError("Duplicate/missing semantic observation frame identity")
        semantic_ids.add(row["frame_id"])
        count = row.get("query_count")
        reason = (row.get("query_count_reason") or row.get("adapter_provenance", {}).get("query_counts", {}).get("unavailable_reason")
                  or row.get("error"))
        if count is None and reason:
            counts["query_count_unknown_frames"].append({"frame_id": row.get("frame_id"), "reason": reason})
        elif type(count) is not int or count < 0:
            raise ValueError("Semantic query_count must be a nonnegative integer or explicitly explained null")
        else:
            counts["known_query_count"] += count
        queries = row.get("queries", [])
        counts["attempted_queries"] += len(queries)
        counts["successful_queries"] += sum(q.get("status") == "ok" for q in queries)
        counts["semantic_observations_completed"] += row.get("status") in {"ok", "partial"}
        for model, number in row.get("model_calls", {}).items():
            if type(number) is not int or number < 0:
                raise ValueError("Actual model-call counts must be nonnegative integers")
            counts["model_calls"][model] = counts["model_calls"].get(model, 0) + number
    if not semantic_capable:
        if counts["known_query_count"] or counts["query_count_unknown_frames"] or sum(counts["model_calls"].values()):
            raise ValueError("Geometry-only semantic records must have zero model/query executions")
        metrics["semantic_query_count"] = _metric(0, unit="queries", status="available", reason=None,
                                                   collection_scope="geometry-only: zero actual SAM text executions")
    elif counts["query_count_unknown_frames"]:
        metrics["semantic_query_count"] = _metric(unit="queries", reason="one or more actual query execution counts unauditable",
            known_query_count=counts["known_query_count"], unknown_frames=counts["query_count_unknown_frames"])
    elif semantic_rows:
        metrics["semantic_query_count"] = _metric(counts["known_query_count"], unit="queries", status="available", reason=None)
    else:
        metrics["semantic_query_count"] = _metric(unit="queries", reason="semantic execution records unavailable")
    counts["query_count"] = None if counts["query_count_unknown_frames"] else counts["known_query_count"]
    duration = summary.get("eligible_replay_duration_s")
    if not semantic_capable:
        metrics["semantic_update_rate_hz"] = _metric(unit="Hz", status="not_applicable", reason="pipeline has no semantic capability")
    elif config["mode"] != "paced_runtime":
        metrics["semantic_update_rate_hz"] = _metric(unit="Hz", reason="quality replay cannot establish runtime semantic update rate")
    elif "cached" in str(summary.get("measurement_scope", summary.get("timing_scope", ""))):
        metrics["semantic_update_rate_hz"] = _metric(unit="Hz", reason="cached geometry scheduler replay is not actual end-to-end runtime")
    elif isinstance(duration, (int, float)) and not isinstance(duration, bool) and np.isfinite(duration) and duration > 0:
        metrics["semantic_update_rate_hz"] = _metric(counts["semantic_observations_completed"] / duration,
            numerator=counts["semantic_observations_completed"], denominator=duration, unit="Hz", status="available",
            reason=None, clock="replay monotonic", timing_boundaries="eligible replay start/end")
    else:
        metrics["semantic_update_rate_hz"] = _metric(unit="Hz", reason="measured eligible replay duration unavailable")
    provided = summary.get("operational_metrics", summary.get("metrics", {}))
    for base in ("geometry_stage_ms", "semantic_call_ms", "end_to_end_latency_ms", "semantic_evidence_age_ms"):
        samples = summary.get("timing_samples", {}).get(base)
        metadata = summary.get("timing_metadata", {}).get(base, {})
        unavailable = None
        if base.startswith("semantic_") and not semantic_capable:
            unavailable = ("not_applicable", "pipeline has no semantic capability")
        elif base in {"end_to_end_latency_ms", "semantic_evidence_age_ms"} and (config["mode"] != "paced_runtime"
              or "cached" in str(summary.get("measurement_scope", summary.get("timing_scope", "")))):
            unavailable = ("unavailable", "cached/staged quality replay cannot establish live end-to-end latency/evidence age")
        for suffix, percentile in (("median", 50), ("p95", 95)):
            mid = base + "." + suffix
            if unavailable:
                metrics[mid] = _metric(unit="ms", status=unavailable[0], reason=unavailable[1])
                continue
            direct = provided.get(mid)
            if direct is not None:
                if isinstance(direct, dict) and direct.get("status", "available") == "available":
                    meta = {k: v for k, v in direct.items() if k not in {"value", "numerator", "denominator", "unit", "status", "reason", "reference_coverage"}}
                    if not meta.get("clock") or not meta.get("timing_boundaries") or not meta.get("count") or direct.get("value") is None:
                        metrics[mid] = _metric(unit="ms", reason="timing clock/boundaries unavailable")
                    elif meta.get("gpu_work") and meta.get("synchronized") is not True:
                        metrics[mid] = _metric(unit="ms", reason="asynchronous GPU timing synchronization unavailable")
                    else:
                        if direct["value"] < 0:
                            raise ValueError("Timing measurements must be nonnegative")
                        metrics[mid] = _metric(direct.get("value"), unit="ms", status="available", reason=None, **meta)
                else:
                    metrics[mid] = _metric(unit="ms", reason=direct.get("reason", "timing unavailable") if isinstance(direct, dict) else "timing metadata unavailable")
                continue
            if samples is not None:
                arr = np.asarray(samples, dtype=float)
                if arr.ndim != 1 or not np.isfinite(arr).all() or (arr < 0).any():
                    raise ValueError("Timing samples must be finite nonnegative milliseconds")
                if metadata.get("gpu_work") and metadata.get("synchronized") is not True:
                    metrics[mid] = _metric(unit="ms", reason="asynchronous GPU timing synchronization unavailable")
                elif arr.size and metadata.get("clock") and metadata.get("timing_boundaries"):
                    metrics[mid] = _metric(float(np.percentile(arr, percentile)), unit="ms", status="available", reason=None,
                                           count=int(arr.size), mode=config["mode"], **metadata)
                else:
                    metrics[mid] = _metric(unit="ms", reason="timing samples/clock/boundaries unavailable")
            else:
                metrics[mid] = _metric(unit="ms", reason="measured timing samples unavailable")
    for mid in ("joint_gpu_allocated_peak_bytes", "joint_gpu_reserved_peak_bytes", "host_rss_peak_bytes"):
        measurement = provided.get(mid)
        scope = str(measurement.get("collection_scope", "")).lower() if isinstance(measurement, dict) else ""
        invalid_scope = any(word in scope for word in ("component", "separate", "summed", "sum_of"))
        if mid.startswith("joint_gpu_") and "joint" not in scope:
            invalid_scope = True
        if (isinstance(measurement, dict) and scope and not invalid_scope and measurement.get("status", "available") == "available"
                and measurement.get("value") is not None):
            meta = {k: v for k, v in measurement.items() if k not in {"value", "unit", "status", "reason", "numerator", "denominator", "reference_coverage"}}
            value = measurement["value"]
            if value < 0:
                raise ValueError("Memory peaks cannot be negative")
            metrics[mid] = _metric(value, unit="bytes", status="available", reason=None, **meta)
        else:
            metrics[mid] = _metric(unit="bytes", reason="actual joint measurement with collection scope unavailable")
    return counts


def _comparability(config: dict, run: dict, run_dir: Path) -> tuple[dict, dict]:
    required = ("contract_id", "schema_version", "protocol_id", "mode", "sequence", "geometry_identity",
                "geometry", "fusion", "robot", "goals", "planning", "hardware_budget")
    parts = {}
    for key in required:
        if key not in config:
            raise ValueError(f"Resolved config missing comparison identity: {key}")
        parts[key] = config[key]
    parts["evaluation_grid"] = config.get("evaluation_grid")
    parts["evaluation_policy"] = config.get("evaluation", {})
    parts["semantic_evaluation_threshold"] = config.get("evaluation", {}).get(
        "semantic_evidence_threshold", config.get("fusion", {}).get("semantic_threshold"))
    if config["mode"] == "quality_replay":
        if "semantic_keyframe_ids" not in config:
            raise ValueError("Quality replay requires frozen semantic keyframes")
        parts["semantic_keyframe_ids"] = config["semantic_keyframe_ids"]
    else:
        if "runtime" not in config:
            raise ValueError("Paced runtime requires cadence/scheduler/capture schedule policy")
        parts["runtime"] = config["runtime"]
        parts["eligible_semantic_keyframe_ids"] = config.get("semantic_keyframe_ids")
    parts["fixture"] = run["fixture"]
    geometry_manifest = run_dir / "geometry/manifest.json"
    if geometry_manifest.exists():
        manifest = _json(geometry_manifest)
        # Locations are intentionally excluded: content identity and calibration
        # determine equivalence, not a machine-specific cache directory.
        keys = ("geometry_fingerprint", "fingerprint", "input_fingerprint", "archive_sha256", "pose_revision", "units", "map_frame", "axes", "up", "scale",
                "processed_grid_id", "processed_grid_ids", "frame_grid_ids", "source_to_processed", "depth_kind", "calibration")
        parts["geometry_manifest"] = {k: manifest[k] for k in keys if k in manifest}
    return parts, {key: fingerprint(value) for key, value in parts.items()}


def _write_report(output_dir: Path, report: dict, csv_name="metrics.csv") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    temp = output_dir / ".report.json.tmp"
    temp.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(output_dir / "report.json")
    with (output_dir / csv_name).open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("metric_id", "value", "numerator", "denominator", "unit", "status", "reason", "reference_coverage"))
        for mid, metric in sorted(report.get("metrics", {}).items()):
            writer.writerow((mid, *[metric.get(key) for key in ("value", "numerator", "denominator", "unit", "status", "reason")],
                             _canonical(metric.get("reference_coverage"))))


def evaluate(run_dir: str | Path, reference_path: str | Path | None, output_dir: str | Path) -> dict:
    """Evaluate a complete common run; missing references yield null quality metrics.

    Invalid supplied references (identity, independence, asset hashes or grids)
    raise ValueError. No report is produced from an invalid reference.
    """
    run_dir, output_dir = Path(run_dir).resolve(), Path(output_dir).resolve()
    if output_dir == run_dir or output_dir.is_relative_to(run_dir):
        raise ValueError("Evaluation output must be separate from immutable run artifacts")
    run, config, summary = (_json(run_dir / name) for name in ("run.json", "config.resolved.json", "summary.json"))
    if run.get("status") != "complete":
        raise ValueError("Only explicit completed runs can be evaluated")
    for record in (run, config):
        if record.get("contract_id") != CONTRACT_ID or record.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Run/config contract version mismatch")
    pipeline = config.get("pipeline_id", config.get("pipeline"))
    if pipeline not in PIPELINE_IDS or run.get("pipeline_id", run.get("pipeline", pipeline)) != pipeline:
        raise ValueError("Run/config pipeline mismatch")
    if config.get("mode") not in {"quality_replay", "paced_runtime"} or type(run.get("fixture")) is not bool:
        raise ValueError("Run requires explicit mode and fixture provenance")
    if not isinstance(run.get("capabilities", {}).get("semantic"), bool):
        raise ValueError("Run requires explicit semantic capability")
    semantic_capable = run["capabilities"]["semantic"]
    if semantic_capable == (pipeline == "geometry_only"):
        raise ValueError("Semantic capability contradicts pipeline identity")
    grid = _grid(config["evaluation_grid"]) if config.get("evaluation_grid") else None
    if grid is not None:
        _validate_producer_frame(run_dir, config, grid)
    predicted = _costmap(run_dir, grid) if grid is not None else None
    reference, assets = (None, {}) if reference_path is None else _validate_reference(Path(reference_path).resolve(), config, run["fixture"])
    frame_rows = _jsonl(run_dir / "frames.jsonl")
    semantic_rows = _jsonl(run_dir / "semantics/frames.jsonl")
    metrics = {mid: _metric(reason="independent reference unavailable") for mid in BASE_METRIC_IDS}
    if predicted is not None:
        roi_coverage = {"kind": "frozen_evaluation_grid", "cells": int(predicted.size), "fingerprint": fingerprint(config["evaluation_grid"])}
        metrics["decision_coverage"] = _ratio(int((predicted != 0).sum()), int(predicted.size), coverage=roi_coverage)
        metrics["unknown_rate"] = _ratio(int((predicted == 0).sum()), int(predicted.size), coverage=roi_coverage)
    else:
        for mid in ("decision_coverage", "unknown_rate"):
            metrics[mid] = _metric(reason="frozen evaluation ROI/grid unavailable")
    counts = _operational(metrics, summary, frame_rows, semantic_rows, config, semantic_capable)
    concepts = _concepts(run_dir, reference)
    for cid in concepts:
        for name in ("precision", "recall", "iou"):
            metrics[f"semantic_{name}.{cid}"] = _metric(status="unavailable" if semantic_capable else "not_applicable",
                reason="independent semantic reference kind/coverage unavailable" if semantic_capable else "pipeline has no semantic capability")
    semantic_diagnostics = {}
    if reference is not None:
        _decision_metrics(metrics, reference, assets, config, predicted, grid)
        _navigation(metrics, run_dir, reference, assets, config)
        if semantic_capable:
            kinds = reference["kinds"]
            if "semantic_2d" in kinds:
                semantic_diagnostics["semantic_2d"] = _semantic_2d(run_dir, semantic_rows, kinds["semantic_2d"], assets, concepts)
            if "semantic_3d" in kinds and _physical_reference_reason(reference, config) is None:
                semantic_diagnostics["semantic_3d"] = _semantic_3d(run_dir, kinds["semantic_3d"], assets, concepts, config)
            # When both kinds exist, keep voxel diagnostics separate rather
            # than mixing pixel/voxel denominators into a fabricated mean.
            primary = semantic_diagnostics.get("semantic_2d", semantic_diagnostics.get("semantic_3d", {}))
            for cid, scores in primary.items():
                for name, metric in scores.items():
                    metrics[f"semantic_{name}.{cid}"] = metric
    identities, fingerprints = _comparability(config, run, run_dir)
    reference_identity = None if reference is None else {
        "reference_id": reference["reference_id"], "reference_protocol_id": reference["reference_protocol_id"],
        "manifest_sha256": hashlib.sha256(Path(reference_path).read_bytes()).hexdigest(),
        "assets": [{"asset_id": a["asset_id"], "sha256": a["sha256"]} for a in reference["assets"]],
        "taxonomy": reference["taxonomy"], "coverage": {k: v.get("coverage") for k, v in reference["kinds"].items()}}
    report = {"contract_id": CONTRACT_ID, "schema_version": SCHEMA_VERSION,
        "metric_definition_version": METRIC_DEFINITION_VERSION, "status": "complete", "pipeline": pipeline,
        "mode": config["mode"], "fixture": run["fixture"], "run_dir": str(run_dir), "sequence": config["sequence"],
        "run_identity": fingerprint(run), "reference": reference_identity,
        "comparability": {"identities": identities, "fingerprints": fingerprints, "combined": fingerprint(fingerprints)},
        "metrics": metrics, "semantic_diagnostics": semantic_diagnostics, "operational_counts": counts,
        "diagnostics": {"summary": summary, "plan_status_counts": {status: sum(p.get("status") == status for p in _jsonl(run_dir / "planning/plans.jsonl"))
             for status in ("ok", "no_path", "blocked_inputs", "error")}},
        "limitations": ["2.5D static/kinematic mapping/planning; planner output alone is not navigation validation",
                         "Safety errors must be read together with decision coverage and safe recall",
                         "Quality replay timing is cached/staged; it does not establish live speed or joint fit"]}
    _write_report(output_dir, report)
    return report


def compare(evaluation_dirs: list[str | Path], output_dir: str | Path) -> dict:
    """Explain comparison identity mismatches; never fabricate a combined rank."""
    if len(evaluation_dirs) < 2:
        raise ValueError("Comparison requires at least two evaluation directories")
    reports = [_json(Path(directory) / "report.json") for directory in evaluation_dirs]
    for report in reports:
        if (report.get("contract_id") != CONTRACT_ID or report.get("schema_version") != SCHEMA_VERSION
                or report.get("metric_definition_version") != METRIC_DEFINITION_VERSION or report.get("status") != "complete"):
            raise ValueError("Evaluation contract/definition/completion mismatch")
        actual = {key: fingerprint(value) for key, value in report["comparability"]["identities"].items()}
        if report["comparability"]["fingerprints"] != actual or report["comparability"].get("combined") != fingerprint(actual):
            raise ValueError("Evaluation comparability fingerprint corruption")
    baseline = reports[0]
    differences = []
    for index, report in enumerate(reports[1:], 1):
        first = baseline["comparability"]["fingerprints"]
        other = report["comparability"]["fingerprints"]
        keys = sorted(set(first) | set(other))
        for key in keys:
            if first.get(key) != other.get(key):
                differences.append({"evaluation_index": index, "pipeline": report["pipeline"], "identity": key,
                                    "baseline": baseline["comparability"]["identities"].get(key),
                                    "value": report["comparability"]["identities"].get(key)})
        if baseline.get("reference") != report.get("reference"):
            differences.append({"evaluation_index": index, "pipeline": report["pipeline"], "identity": "reference",
                                "reason": "independent reference identity/assets/taxonomy/coverage differ"})
    result = {"contract_id": CONTRACT_ID, "schema_version": SCHEMA_VERSION,
        "metric_definition_version": METRIC_DEFINITION_VERSION,
        "status": "compatible" if not differences else "incompatible", "compatible": not differences,
        "ranking": None, "reason": "No automatic winner: assess safety, recall and usable coverage together" if not differences else
        "Combined ranking refused: common experiment/reference identities differ",
        "differences": differences, "evaluations": [{"pipeline": r["pipeline"], "run_dir": r["run_dir"], "fixture": r["fixture"],
                                                       "metrics": r["metrics"]} for r in reports]}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp = output_dir / ".comparison.json.tmp"
    temp.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(output_dir / "comparison.json")
    metric_ids = sorted(set().union(*(r["metrics"] for r in reports)))
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("evaluation_index", "pipeline", "metric_id", "value", "status", "reason"))
        for index, report in enumerate(reports):
            for mid in metric_ids:
                metric = report["metrics"].get(mid, _metric(status="not_applicable", reason="concept outside this report taxonomy"))
                writer.writerow((index, report["pipeline"], mid, metric["value"], metric["status"], metric["reason"]))
    return result
