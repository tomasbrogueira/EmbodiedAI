"""Deterministic, surface-only 2.5D planning diagnostics.

Points have already been converted to metres by the geometry provider. ``scale``
is verified provenance, never an instruction to scale them a second time. The
explicit ``up`` argument is the provider's verified map-frame up vector. A tuple
is accepted; a dict must contain ``vector`` and ``verified=True``.

The horizontal grid is expressed in a deterministic orthonormal projection
basis, whose rows are horizontal x, horizontal y, and up. Unknown support stays
unknown. Absence of observed shell points is *not* a free-space observation or a
clearance certificate. Returned paths are static, kinematic diagnostics only.

``semantic_points`` optionally maps concept IDs to Nx3 metric surface points,
or to dictionaries with ``points``, ``role`` and optional ``evidence_score``.
Costs come exclusively from the versioned robot's ``semantic_costs`` policy.
No semantic evidence can repair a geometric violation or missing support.
"""
from __future__ import annotations

from collections import deque
import heapq
import math
from typing import Any

import numpy as np

UNKNOWN, BLOCKED, TRAVERSABLE = 0, 1, 2
DECISION_CODES = {"unknown": UNKNOWN, "blocked": BLOCKED, "traversable": TRAVERSABLE}
DIAGNOSTIC_MASK_NAMES = (
    "raw_candidate_mask", "candidate_mask", "eligible_support_mask",
    "supported_mask", "wall_blocked_mask", "roughness_blocked_mask",
    "clearance_blocked_mask", "step_blocked_mask", "physical_blocked_mask",
    "footprint_unknown_mask", "footprint_blocked_mask",
)
LIMITATIONS = [
    "Static 2.5D grid and circular footprint; no dynamics or robot controller.",
    "Surface-only geometry: unobserved space is unknown, never observed free space.",
    "Height clearance checks observed shells only; unseen overhangs are not certified clear.",
    "Lower support must connect to a declared support-height anchor; multiple floors are not resolved.",
    "Grid support/plane sampling and raster footprint conservatism limit spatial resolution.",
    "Multiple observed heights at the same horizontal sample conservatively form separate shells, including reconstruction noise.",
]


class _EndpointFormatError(ValueError):
    """An invalid endpoint record, as distinct from a valid blocked position."""


def _basis(up: Any) -> np.ndarray:
    if isinstance(up, dict):
        if up.get("verified") is not True:
            raise ValueError("up is not verified")
        up = up.get("vector")
    direction = np.asarray(up, dtype=np.float64)
    if direction.shape != (3,) or not np.isfinite(direction).all():
        raise ValueError("up must be a finite three-vector")
    length = np.linalg.norm(direction)
    if length < 1e-12:
        raise ValueError("up must be nonzero")
    direction = direction / length
    # Prefer map x when it is not parallel to up, then map y. This keeps the
    # familiar XY grid when up=+Z while also handling arbitrary gravity frames.
    axis = np.array([1., 0., 0.])
    horizontal = axis - direction * np.dot(axis, direction)
    if np.linalg.norm(horizontal) < 1e-8:
        axis = np.array([0., 1., 0.])
        horizontal = axis - direction * np.dot(axis, direction)
    horizontal /= np.linalg.norm(horizontal)
    return np.stack((horizontal, np.cross(direction, horizontal), direction))


def _empty(reason: str) -> tuple[dict, dict]:
    arrays = {
        "costs": np.empty((0, 0), dtype=np.float64),
        "observed_mask": np.empty((0, 0), dtype=np.bool_),
        "unknown_mask": np.empty((0, 0), dtype=np.bool_),
        "decision_state": np.empty((0, 0), dtype=np.uint8),
        "geometry_state": np.empty((0, 0), dtype=np.uint8),
        "policy_blocked_mask": np.empty((0, 0), dtype=np.bool_),
        "origin": np.empty((0,), dtype=np.float64),
        "resolution": np.empty((0,), dtype=np.float64),
        "up": np.empty((0,), dtype=np.float64),
        "projection_basis": np.empty((0, 3), dtype=np.float64),
        "support_height": np.empty((0, 0), dtype=np.float64),
        "support_count": np.empty((0, 0), dtype=np.int64),
        "slope_degrees": np.empty((0, 0), dtype=np.float64),
        "roughness": np.empty((0, 0), dtype=np.float64),
        "clearance_observed": np.empty((0, 0), dtype=np.float64),
    }
    for name in DIAGNOSTIC_MASK_NAMES:
        arrays[name] = np.empty((0, 0), dtype=np.bool_)
    return arrays, {
        "schema_version": 1, "availability": "blocked_inputs", "reason": reason,
        "shape": [0, 0], "decision_codes": DECISION_CODES.copy(),
        "projection_basis": [], "visibility_model": "none",
        "diagnostic_counts": {**dict.fromkeys(DIAGNOSTIC_MASK_NAMES, 0),
                              "supported_cells_rejected_by_footprint": 0,
                              "supported_cells_touching_both_unknown_and_blocked": 0},
        "clearance_certified": False, "limitations": LIMITATIONS.copy(),
    }


def _positive_number(record: dict, key: str, *, zero: bool = False) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"missing or nonnumeric {key}")
    value = float(value)
    if not np.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"invalid {key}")
    return value


def _settings(config: dict, units: str, up: Any, scale: Any) -> tuple:
    if units != "metres":
        raise ValueError("metric scale unavailable: geometry units are not metres")
    if not isinstance(scale, dict) or scale.get("verified") is not True:
        raise ValueError("metric scale provenance is not verified")
    if up is None:
        raise ValueError("verified up direction is missing")
    basis = _basis(up)
    planning, robot = config.get("planning"), config.get("robot")
    if not isinstance(planning, dict) or not isinstance(robot, dict):
        raise ValueError("planning grid or robot profile is missing")
    if not robot.get("version") or type(robot.get("fixture")) is not bool:
        raise ValueError("robot profile needs version and explicit fixture provenance")
    if robot.get("unknown_rule") != "blocked":
        raise ValueError("robot unknown_rule must be blocked")
    if not isinstance(robot.get("semantic_costs"), dict):
        raise ValueError("robot semantic concept cost policy is missing")
    values = {key: _positive_number(robot, key, zero=key in {
        "footprint_radius", "clearance", "max_step", "max_roughness", "max_slope_degrees"
    }) for key in ("footprint_radius", "height", "clearance", "max_slope_degrees", "max_step", "max_roughness")}
    if values["max_slope_degrees"] >= 90:
        raise ValueError("maximum slope must be below 90 degrees")
    count = robot.get("min_support_points")
    if type(count) is not int or count < 3:
        raise ValueError("min_support_points must be an integer of at least three")
    values["min_support_points"] = count
    resolution = _positive_number(planning, "resolution")
    origin = np.asarray(planning.get("origin"), dtype=np.float64)
    shape = planning.get("shape")
    if origin.shape != (2,) or not np.isfinite(origin).all():
        raise ValueError("frozen planning origin must be a finite two-vector")
    if not isinstance(shape, (list, tuple)) or len(shape) != 2 or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError("frozen planning shape must be two positive integers [height,width]")
    if shape[0] * shape[1] > 4_000_000:
        raise ValueError("planning grid exceeds declared 4 million cell CPU safety limit")
    anchor = planning.get("support_height")
    if isinstance(anchor, bool) or not isinstance(anchor, (int, float)) or not np.isfinite(anchor):
        raise ValueError("declared support_height anchor is missing")
    return planning, robot, values, basis, origin, resolution, tuple(shape), float(anchor)


def _project_cells(points: np.ndarray, basis: np.ndarray, origin: np.ndarray, resolution: float, shape: tuple) -> tuple:
    # Filter before projection or integer conversion: invalid geometry never
    # creates a cell and infinities do not generate invalid matmul warnings.
    valid = np.isfinite(points).all(axis=1)
    projected = np.full(points.shape, np.nan, dtype=np.float64)
    projected[valid] = points[valid] @ basis.T
    indices = np.zeros((len(points), 2), dtype=np.int64)
    bounded = valid & (projected[:, 0] >= origin[0]) & (projected[:, 1] >= origin[1])
    bounded &= (projected[:, 0] < origin[0] + shape[1] * resolution)
    bounded &= (projected[:, 1] < origin[1] + shape[0] * resolution)
    indices[bounded] = np.floor((projected[bounded, :2] - origin) / resolution).astype(np.int64)
    return projected[bounded], indices[bounded]


def _footprint_offsets(radius: float, resolution: float) -> list[tuple[int, int]]:
    reach = int(math.ceil(radius / resolution + .5))
    result = []
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            distance_x = max(abs(dx) - .5, 0.) * resolution
            distance_y = max(abs(dy) - .5, 0.) * resolution
            if math.hypot(distance_x, distance_y) <= radius + 1e-12:
                result.append((dy, dx))
    return result


def _shift(values: np.ndarray, dy: int, dx: int, fill: Any) -> np.ndarray:
    result = np.full(values.shape, fill, dtype=values.dtype)
    height, width = values.shape
    if abs(dy) >= height or abs(dx) >= width:
        return result
    destination_y = slice(max(0, -dy), min(height, height - dy))
    destination_x = slice(max(0, -dx), min(width, width - dx))
    source_y = slice(max(0, dy), min(height, height + dy))
    source_x = slice(max(0, dx), min(width, width + dx))
    result[destination_y, destination_x] = values[source_y, source_x]
    return result


def build_costmap(points: np.ndarray, config: dict, *, units: str = "metres",
                  up: Any = None, scale: dict | None = None,
                  semantic_points: dict | None = None,
                  terrain_voxels: dict | None = None,
                  voxel_size: float | None = None) -> tuple[dict, dict]:
    """Build the frozen grid, or schema-valid empty arrays for blocked inputs.

    Grid origin is the lower XY boundary in the recorded projection basis.
    Unknown/nontraversable costs are +inf. Unsupported heights, slopes and
    roughness use NaN sentinels in NPZ; metadata never contains JSON NaNs.
    Semantics may add costs/block a geometrically supported cell, never make an
    unknown/blocked cell traversable. Already scaled metric points are required.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("planning points must be numeric [N,3]")
    try:
        planning, robot, profile, basis, origin, resolution, shape, anchor = _settings(config, units, up, scale)
    except (ValueError, TypeError) as error:
        return _empty(str(error))
    if planning.get("terrain_source") == "voxels" and terrain_voxels is None:
        return _empty("voxel terrain requested but the fused voxel map is missing")
    voxel_terrain = terrain_voxels is not None
    projected, cells = _project_cells(points, basis, origin, resolution, shape)
    observed = np.zeros(shape, dtype=np.bool_)
    height = np.full(shape, np.nan, dtype=np.float64)
    count = np.zeros(shape, dtype=np.int64)
    slope = np.full(shape, np.nan, dtype=np.float64)
    roughness = np.full(shape, np.nan, dtype=np.float64)
    clearance = np.full(shape, np.inf, dtype=np.float64)
    seed_height = np.full(shape, np.nan, dtype=np.float64)
    gradients = np.zeros((*shape, 2), dtype=np.float64)
    candidate = np.zeros(shape, dtype=np.bool_)
    physical_blocked = np.zeros(shape, dtype=np.bool_)
    wall_blocked = np.zeros(shape, dtype=np.bool_)
    # Include a traversable plane's entire vertical span while isolating upper
    # shells. Excess rough points outside the lower layer become body obstacles.
    # Surface measurement must not change when the robot's slope limit changes.
    # This extraction angle is a fixed measurement setting, not a capability.
    layer_span = max(4 * profile["max_roughness"],
                     1.5 * resolution * math.tan(math.radians(60.)), 1e-6)
    ground_tolerance = max(profile["max_roughness"] * 3, 1e-5)
    if len(cells):
        flat = cells[:, 1] * shape[1] + cells[:, 0]
        order = np.argsort(flat, kind="stable")
        split = np.flatnonzero(np.diff(flat[order])) + 1
        for group in np.split(order, split):
            x, y = cells[group[0]]
            cloud = projected[group]
            observed[y, x] = True
            # Fit only the lower envelope at each horizontal sample. A ceiling
            # 4 cm above floor can have a small pooled RMS and must not be blended
            # into an apparently traversable mean plane. Quantization is local
            # and microscopic relative to this cell; it does not merge distinct
            # horizontal terrain samples at the planning resolution.
            local_xy = cloud[:, :2] - np.min(cloud[:, :2], axis=0)
            keys = np.rint(local_xy / (resolution * 1e-7)).astype(np.int64)
            _, inverse = np.unique(keys, axis=0, return_inverse=True)
            ordered = np.lexsort((cloud[:, 2], inverse))
            _, first = np.unique(inverse[ordered], return_index=True)
            envelope_indices = ordered[first]
            envelope = cloud[envelope_indices]
            minima = np.empty(len(envelope), dtype=np.float64)
            minima[inverse[envelope_indices]] = envelope[:, 2]
            shell_gap = cloud[:, 2] - minima[inverse]
            stacked = shell_gap > 1e-6
            lower = envelope[envelope[:, 2] <= np.min(envelope[:, 2]) + layer_span + 1e-12]
            count[y, x] = len(lower)
            if len(lower) < 3:
                continue
            centered_xy = lower[:, :2] - np.mean(lower[:, :2], axis=0)
            spread = np.linalg.svd(centered_xy, compute_uv=False)
            if len(spread) < 2 or spread[1] <= 1e-6 * resolution:
                # A vertical wall cannot masquerade as floor merely because its
                # lowest sample happens to have the configured anchor height.
                if np.ptp(cloud[:, 2]) > profile["max_step"] + ground_tolerance:
                    physical_blocked[y, x] = True
                    wall_blocked[y, x] = True
                continue
            center = origin + np.array([x + .5, y + .5]) * resolution
            design = np.column_stack((lower[:, :2] - center, np.ones(len(lower))))
            coefficients, _, _, _ = np.linalg.lstsq(design, lower[:, 2], rcond=None)
            predicted = design @ coefficients
            residual = lower[:, 2] - predicted
            height[y, x] = coefficients[2]
            seed_height[y, x] = float(np.min(predicted))
            gradients[y, x] = coefficients[:2]
            slope[y, x] = math.degrees(math.atan(np.linalg.norm(coefficients[:2])))
            roughness[y, x] = math.sqrt(float(np.mean(residual ** 2)))
            candidate[y, x] = len(lower) >= profile["min_support_points"]
            if (not voxel_terrain and slope[y, x] > profile["max_slope_degrees"] + 1e-8) or roughness[y, x] > profile["max_roughness"] + 1e-8:
                physical_blocked[y, x] = True
            # All shell samples are tested against the fitted ground at their
            # actual XY position. Sloped floor height is not an overhead hazard.
            above = cloud[:, 2] - ((cloud[:, :2] - center) @ coefficients[:2] + coefficients[2])
            overhead = above[above > ground_tolerance]
            # Repeated XY samples disclose distinct surfaces even when their
            # separation is below the allowed *ground* roughness tolerance.
            # Their lower-envelope gap directly bounds observed clearance.
            if np.any(stacked):
                overhead = np.r_[overhead, shell_gap[stacked]]
            if len(overhead):
                clearance[y, x] = max(0., float(np.min(overhead)))
                if clearance[y, x] <= profile["height"] + profile["clearance"]:
                    physical_blocked[y, x] = True

    raw_candidate = candidate.copy()
    terrain_arrays, terrain_metadata = {}, {"source": "raw_observed_surface_points", "method": "cell_lower_envelope_least_squares"}
    if voxel_terrain:
        from .terrain import estimate_voxel_terrain
        if not isinstance(terrain_voxels, dict) or "centers" not in terrain_voxels:
            raise ValueError("terrain_voxels must contain fused voxel centers")
        measured, terrain_metadata = estimate_voxel_terrain(
            terrain_voxels["centers"], basis=basis, origin=origin,
            resolution=resolution, shape=shape, voxel_size=voxel_size,
            settings=planning.get("terrain"))
        terrain_arrays = {"terrain_normal": measured["normal"],
                          "terrain_slope_valid": measured["valid_mask"],
                          "terrain_fit_residual": measured["residual"],
                          "terrain_support_count": measured["support_count"]}
        slope = measured["slope_degrees"]
        # Unreliable estimates do not create traversable support. Raw wall,
        # shell, step and roughness evidence remains independently blocking.
        candidate &= measured["valid_mask"]
        physical_blocked |= measured["valid_mask"] & (slope > profile["max_slope_degrees"] + 1e-8)
    slope_blocked = np.isfinite(slope) & (slope > profile["max_slope_degrees"] + 1e-8)

    # Residual height jump removes the expected elevation change of both local
    # planes. A continuous sloped plane is not incorrectly called a step.
    step_blocked = np.zeros(shape, dtype=np.bool_)
    for dy, dx in ((0, 1), (1, 0)):
        neighbor = _shift(candidate, dy, dx, False)
        neighbor_height = _shift(height, dy, dx, np.nan)
        delta = neighbor_height - height
        expected_here = (gradients[..., 0] * dx + gradients[..., 1] * dy) * resolution
        expected_there = (_shift(gradients[..., 0], dy, dx, 0.) * dx + _shift(gradients[..., 1], dy, dx, 0.) * dy) * resolution
        jump = np.maximum(np.abs(delta - expected_here), np.abs(delta - expected_there))
        violation = candidate & neighbor & (jump > profile["max_step"] + 1e-8)
        step_blocked |= violation | _shift(violation, -dy, -dx, False)
    physical_blocked |= step_blocked

    eligible = candidate & ~physical_blocked
    supported = np.zeros(shape, dtype=np.bool_)
    # Seeds must actually meet the declared ground anchor, allowing only the
    # declared roughness. A disconnected elevated shell is not supporting ground
    # merely because its height is within the robot's stepping capability.
    seeds = np.argwhere(eligible & (np.abs(seed_height - anchor) <= profile["max_roughness"] + 1e-8))
    queue = deque((int(y), int(x)) for y, x in seeds)
    for y, x in queue:
        supported[y, x] = True
    while queue:
        y, x = queue.popleft()
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            yy, xx = y + dy, x + dx
            if 0 <= yy < shape[0] and 0 <= xx < shape[1] and eligible[yy, xx] and not supported[yy, xx]:
                supported[yy, xx] = True
                queue.append((yy, xx))

    base = np.full(shape, UNKNOWN, dtype=np.uint8)
    base[supported] = TRAVERSABLE
    base[physical_blocked] = BLOCKED
    footprint = profile["footprint_radius"] + profile["clearance"]
    offsets = _footprint_offsets(footprint, resolution)
    footprint_unknown = np.zeros(shape, dtype=np.bool_)
    footprint_blocked = np.zeros(shape, dtype=np.bool_)
    for dy, dx in offsets:
        footprint_unknown |= _shift(base == UNKNOWN, dy, dx, True)
        footprint_blocked |= _shift(base == BLOCKED, dy, dx, False)
    state = base.copy()
    state[footprint_unknown] = UNKNOWN
    state[footprint_blocked] = BLOCKED
    state[physical_blocked] = BLOCKED

    semantic_cost = np.zeros(shape, dtype=np.float64)
    semantic_blocked = np.zeros(shape, dtype=np.bool_)
    policy_blocked = np.zeros(shape, dtype=np.bool_)
    hazard_cells = np.zeros(shape, dtype=np.bool_)
    surface_cells = np.zeros(shape, dtype=np.bool_)
    unmapped = []
    applied = []
    if semantic_points is not None and not isinstance(semantic_points, dict):
        raise ValueError("semantic_points must map concept IDs to numeric metric surfaces")
    for concept_id, evidence in sorted((semantic_points or {}).items()):
        role = evidence.get("role", "hazard") if isinstance(evidence, dict) else "hazard"
        if role not in {"hazard", "candidate_surface"}:
            raise ValueError(f"invalid semantic role for {concept_id}")
        cloud = np.asarray(evidence.get("points") if isinstance(evidence, dict) else evidence, dtype=np.float64)
        if cloud.ndim != 2 or cloud.shape[1] != 3:
            raise ValueError(f"semantic points for {concept_id} must be [N,3]")
        if isinstance(evidence, dict) and "evidence_score" in evidence:
            weights = np.asarray(evidence["evidence_score"], dtype=np.float64)
            if weights.shape != (len(cloud),) or not np.isfinite(weights).all() or np.any(weights < 0):
                raise ValueError("semantic evidence scores must be finite nonnegative [N] weights")
            cloud = cloud[weights > planning.get("semantic_min_evidence", 0.)]
        _, locations = _project_cells(cloud, basis, origin, resolution, shape)
        mask = np.zeros(shape, dtype=np.bool_)
        if len(locations):
            mask[locations[:, 1], locations[:, 0]] = True
        if role == "hazard":
            hazard_cells |= mask
        else:
            surface_cells |= mask
        policy = robot["semantic_costs"].get(concept_id)
        if policy is None:
            unmapped.append(str(concept_id))
            continue
        if isinstance(policy, dict):
            cost, blocked = policy.get("cost", 0.), policy.get("blocked", False)
            if type(blocked) is not bool:
                raise ValueError(f"semantic blocked policy must be boolean: {concept_id}")
        else:
            cost, blocked = policy, False
        if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not np.isfinite(cost):
            raise ValueError(f"semantic cost must be finite: {concept_id}")
        if role == "hazard" and cost < 0:
            raise ValueError(f"hazard cost cannot reward traversal: {concept_id}")
        affected = mask.copy()
        if role == "hazard":
            for dy, dx in offsets:
                affected |= _shift(mask, dy, dx, False)
        semantic_cost[affected & (state == TRAVERSABLE)] += float(cost)
        if blocked:
            # Retain the raw exclusion separately from padded grid decisions.
            # Exact endpoint footprints must apply their radius only once.
            policy_blocked |= mask
            semantic_blocked |= affected & (state == TRAVERSABLE)
        applied.append(str(concept_id))
    state[semantic_blocked] = BLOCKED
    costs = np.full(shape, np.inf, dtype=np.float64)
    costs[state == TRAVERSABLE] = np.maximum(1e-6, 1. + semantic_cost[state == TRAVERSABLE])
    diagnostic_masks = {
        "raw_candidate_mask": raw_candidate, "candidate_mask": candidate,
        "eligible_support_mask": eligible, "supported_mask": supported,
        "wall_blocked_mask": wall_blocked,
        "roughness_blocked_mask": np.isfinite(roughness) & (roughness > profile["max_roughness"] + 1e-8),
        "clearance_blocked_mask": np.isfinite(clearance) & (clearance <= profile["height"] + profile["clearance"]),
        "step_blocked_mask": step_blocked,
        "physical_blocked_mask": physical_blocked,
        "footprint_unknown_mask": footprint_unknown,
        "footprint_blocked_mask": footprint_blocked,
    }
    arrays = {
        "costs": costs, "observed_mask": observed, "unknown_mask": state == UNKNOWN,
        "decision_state": state, "geometry_state": base, "origin": origin,
        "policy_blocked_mask": policy_blocked,
        "resolution": np.array([resolution]), "up": basis[2].copy(),
        "projection_basis": basis, "support_height": height, "support_count": count,
        "slope_degrees": slope, "roughness": roughness, "clearance_observed": clearance,
        "slope_blocked_mask": slope_blocked, **terrain_arrays, **diagnostic_masks,
    }
    metadata = {
        "schema_version": 1, "availability": "available", "reason": None,
        "shape": list(shape), "origin": origin.tolist(), "resolution": resolution,
        "up": basis[2].tolist(), "projection_basis": basis.tolist(),
        "projection_convention": "basis rows x,y,up; projected=map_points@basis.T",
        "grid_convention": "origin is lower XY boundary; arrays index [y,x]",
        "units": "metres", "scale": scale, "robot_version": robot["version"],
        "fixture_robot": robot["fixture"], "decision_codes": DECISION_CODES.copy(),
        "visibility_model": "none", "clearance_certified": False,
        "clearance_model": "observed_shell_only", "support_height_anchor": anchor,
        "profile": profile, "footprint_radius_with_clearance": footprint,
        "terrain_estimation": terrain_metadata,
        "slope_reference_up": basis[2].tolist(),
        "max_slope_degrees": profile["max_slope_degrees"],
        "slope_violation_cells": int(np.count_nonzero(slope_blocked)),
        "unknown_rule": "blocked", "endpoint_coordinates": "3D map-frame ground positions; two-dimensional endpoints are rejected",
        "unmapped_concepts": unmapped, "applied_concepts": applied,
        "semantic_conflict_cells": int(np.count_nonzero(hazard_cells & surface_cells)),
        "physical_violation_cells": int(np.count_nonzero(physical_blocked)),
        "support_cells": int(np.count_nonzero(supported)),
        "diagnostic_counts": {
            **{name: int(np.count_nonzero(mask)) for name, mask in diagnostic_masks.items()},
            "supported_cells_rejected_by_footprint": int(np.count_nonzero(supported & (footprint_unknown | footprint_blocked))),
            "supported_cells_touching_both_unknown_and_blocked": int(np.count_nonzero(supported & footprint_unknown & footprint_blocked)),
        },
        "diagnostic_mask_convention": "Independent overlapping reasons before semantic policy; footprint masks test raw geometry cells, including unknown map boundaries",
        "observed_surface_cells": int(np.count_nonzero(observed)),
        "invalid_or_outside_points": int(len(points) - len(projected)),
        "numeric_sentinels": {"costs": "+inf for nontraversable", "unsupported_geometry": "NaN", "clearance_observed": "+inf means no observed shell, not verified clearance"},
        "limitations": LIMITATIONS.copy(),
    }
    return arrays, metadata


def _endpoint(value: Any, arrays: dict, metadata: dict) -> tuple[tuple[int, int], np.ndarray]:
    try:
        point = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError) as error:
        raise _EndpointFormatError("endpoint must have three finite map-frame XYZ coordinates") from error
    if point.shape != (3,) or not np.isfinite(point).all():
        raise _EndpointFormatError("endpoint must have three finite map-frame XYZ coordinates")
    basis = arrays["projection_basis"]
    projected = point @ basis.T
    xy = np.floor((projected[:2] - arrays["origin"]) / arrays["resolution"][0]).astype(np.int64)
    x, y = map(int, xy)
    height, width = arrays["decision_state"].shape
    if not (0 <= y < height and 0 <= x < width):
        raise ValueError("endpoint outside frozen planning grid; no snapping")
    if arrays["decision_state"][y, x] != TRAVERSABLE:
        raise ValueError("endpoint is unknown or blocked; no snapping")
    ground = arrays["support_height"][y, x]
    tolerance = metadata["profile"]["max_step"] + 3 * metadata["profile"]["max_roughness"] + 1e-5
    if abs(projected[2] - ground) > tolerance:
        raise ValueError("endpoint height differs from supporting ground")
    # Endpoint positions are preserved exactly. Check their actual horizontal
    # footprint against all touched raw geometry cells, including map boundaries.
    radius = metadata["footprint_radius_with_clearance"]
    resolution = arrays["resolution"][0]
    origin = arrays["origin"]
    low = np.floor((projected[:2] - radius - origin) / resolution).astype(int)
    high = np.floor((projected[:2] + radius - origin) / resolution).astype(int)
    for yy in range(low[1], high[1] + 1):
        for xx in range(low[0], high[0] + 1):
            near = np.maximum(origin + np.array([xx, yy]) * resolution, np.minimum(projected[:2], origin + np.array([xx + 1, yy + 1]) * resolution))
            if np.linalg.norm(near - projected[:2]) > radius + 1e-12:
                continue
            if not (0 <= yy < height and 0 <= xx < width) or arrays["geometry_state"][yy, xx] != TRAVERSABLE:
                raise ValueError("endpoint footprint meets unknown or blocked geometry")
            if arrays["policy_blocked_mask"][yy, xx]:
                raise ValueError("endpoint footprint meets blocked policy")
    return (y, x), point


def plan_requests(arrays: dict, metadata: dict, requests: list[dict], map_frame: str) -> list[dict]:
    """Four-neighbour Dijkstra paths with exact valid endpoints, no snapping.

    A result is produced for every request, including unavailable inputs and
    malformed requests. Paths use original map-frame XYZ, not projected XY.
    """
    results = []
    for number, request in enumerate(requests):
        if not isinstance(request, dict):
            request = {"request_id": f"request_{number}", "start": None, "goal": None}
        result = {
            "request_id": request.get("request_id", f"request_{number}"),
            "start": request.get("start"), "goal": request.get("goal"),
            "frame": request.get("frame", request.get("map_frame", map_frame)),
            "map_frame": map_frame, "path": [], "status": "blocked_inputs", "reason": None,
            "diagnostic_only": True, "clearance_certified": False,
        }
        if metadata.get("availability") != "available":
            result["reason"] = metadata.get("reason", "planning inputs unavailable")
            results.append(result)
            continue
        if result["frame"] != map_frame:
            result.update(status="blocked_inputs", reason="request frame differs from declared map frame")
            results.append(result)
            continue
        try:
            start, start_xyz = _endpoint(result["start"], arrays, metadata)
            goal, goal_xyz = _endpoint(result["goal"], arrays, metadata)
        except _EndpointFormatError as error:
            result.update(status="error", reason=str(error))
            results.append(result)
            continue
        except (ValueError, TypeError, OverflowError) as error:
            result.update(status="no_path", reason=str(error))
            results.append(result)
            continue
        states, costs = arrays["decision_state"], arrays["costs"]
        distances = {start: 0.}
        parents = {}
        pending = [(0., start[0], start[1])]
        while pending:
            distance, y, x = heapq.heappop(pending)
            current = (y, x)
            if distance > distances[current]:
                continue
            if current == goal:
                break
            for dy, dx in ((-1, 0), (0, -1), (0, 1), (1, 0)):
                yy, xx = y + dy, x + dx
                if not (0 <= yy < states.shape[0] and 0 <= xx < states.shape[1]) or states[yy, xx] != TRAVERSABLE:
                    continue
                cost = distance + .5 * (costs[y, x] + costs[yy, xx]) * arrays["resolution"][0]
                neighbor = (yy, xx)
                if cost < distances.get(neighbor, np.inf):
                    distances[neighbor] = float(cost)
                    parents[neighbor] = current
                    heapq.heappush(pending, (float(cost), yy, xx))
        if goal not in distances:
            result.update(status="no_path", reason="no connected traversable grid route")
        else:
            route, cell = [goal], goal
            while cell != start:
                cell = parents[cell]
                route.append(cell)
            route.reverse()
            basis, origin = arrays["projection_basis"], arrays["origin"]
            path = []
            for y, x in route:
                projected = np.r_[origin + np.array([x + .5, y + .5]) * arrays["resolution"][0], arrays["support_height"][y, x]]
                path.append((projected @ basis).tolist())
            path[0] = start_xyz.tolist()
            if len(path) == 1 and not np.allclose(start_xyz, goal_xyz, atol=0., rtol=0.):
                path.append(goal_xyz.tolist())
            else:
                path[-1] = goal_xyz.tolist()
            result.update(status="ok", reason="internal static 2.5D route; observed-shell clearance only",
                          path=path, path_cost=distances[goal], path_cells=[[y, x] for y, x in route])
        results.append(result)
    return results
