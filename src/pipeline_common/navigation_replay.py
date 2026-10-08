"""Stateful-mission planning for an observation-prefix research replay.

The caller owns observation admission, simulated motion and mission lifetime.
This module plans only through the supplied supported surface grid. Coordinates
and robot capability are explicit research assumptions, not calibration or a
free-space certificate. A temporary frontier never completes the mission.
"""
from __future__ import annotations

import heapq
import math
import numbers

import numpy as np

from .planning import TRAVERSABLE
from .research_route import _local_point, segment_supported


def _number(value, name, *, zero=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(name + " must be a finite real number")
    value = float(value)
    if not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(name + " must be " + ("nonnegative" if zero else "positive"))
    return value


def _xy(value, name):
    array = np.asarray(value)
    if array.shape != (2,) or array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(name + " must contain two finite local XY coordinates")
    return array.astype(float)


def clip_route_horizon(points, horizon_m):
    """Clip an ordered 2D/3D polyline at exact Euclidean arc length.

    Return a copied float array, retaining its first point and interpolating the
    final point inside a segment when necessary. Zero horizon retains only the
    first point; empty routes remain empty. Duplicate consecutive vertices do
    not consume distance. This does not infer support or extend a short route.
    """
    horizon = _number(horizon_m, "horizon_m", zero=True)
    array = np.asarray(points)
    if array.size == 0:
        if array.ndim == 2 and array.shape[1] in (2, 3):
            return array.astype(float).copy()
        if array.ndim == 1:
            return np.empty((0, 3), float)
        raise ValueError("points must contain an ordered [N,2] or [N,3] route")
    if (array.ndim != 2 or array.shape[1] not in (2, 3)
            or array.dtype.kind not in "iuf" or not np.isfinite(array).all()):
        raise ValueError("points must contain a finite ordered [N,2] or [N,3] route")
    array = array.astype(float)
    if len(array) == 1 or horizon == 0:
        return array[:1].copy()
    clipped = [array[0].copy()]
    travelled = 0.
    for a, b in zip(array[:-1], array[1:]):
        delta = b - a
        length = float(np.linalg.norm(delta))
        if not math.isfinite(length):
            raise ValueError("route segment length overflowed")
        if length == 0:
            continue
        remaining = horizon - travelled
        if length >= remaining:
            clipped.append(a + np.clip(remaining / length, 0., 1.) * delta)
            return np.asarray(clipped)
        clipped.append(b.copy())
        travelled += length
    return np.asarray(clipped)


def _validate_map(arrays, metadata):
    shape = np.asarray(arrays["geometry_state"]).shape
    if len(shape) != 2 or not all(shape) or math.prod(shape) > 4_000_000:
        raise ValueError("geometry_state must be a bounded nonempty 2D grid")
    for key in ("geometry_state", "decision_state", "policy_blocked_mask", "support_height", "costs"):
        value = np.asarray(arrays[key])
        if value.shape != shape:
            raise ValueError(key + " must match the geometry grid")
    for key in ("geometry_state", "decision_state"):
        state = np.asarray(arrays[key])
        if state.dtype.kind not in "biuf" or not np.isin(state, (0, 1, 2)).all():
            raise ValueError(key + " contains invalid decision codes")
    if np.asarray(arrays["policy_blocked_mask"]).dtype.kind != "b":
        raise ValueError("policy_blocked_mask must be boolean")
    heights, costs = np.asarray(arrays["support_height"]), np.asarray(arrays["costs"])
    if heights.dtype.kind not in "iuf" or costs.dtype.kind not in "iuf":
        raise ValueError("support heights and costs must be real numeric arrays")
    supported = np.asarray(arrays["geometry_state"]) == TRAVERSABLE
    traversable = np.asarray(arrays["decision_state"]) == TRAVERSABLE
    if np.any(supported & ~np.isfinite(heights)):
        raise ValueError("supported geometry requires finite support heights")
    if np.any(traversable & (~supported | ~np.isfinite(costs) | (costs <= 0))):
        raise ValueError("traversable decisions require supported geometry and finite positive costs")
    _xy(arrays["origin"], "grid origin")
    resolution = np.asarray(arrays["resolution"])
    if resolution.shape != (1,):
        raise ValueError("resolution must contain one positive grid spacing")
    _number(resolution[0], "resolution")
    basis = np.asarray(arrays["projection_basis"])
    if (basis.shape != (3, 3) or basis.dtype.kind not in "iuf" or not np.isfinite(basis).all()
            or not np.allclose(basis @ basis.T, np.eye(3), rtol=0, atol=1e-8)
            or not math.isclose(float(np.linalg.det(basis)), 1., abs_tol=1e-8)):
        raise ValueError("projection_basis must be a proper orthonormal frame")
    _number(metadata["footprint_radius_with_clearance"], "footprint radius", zero=True)
    slope = _number(metadata["profile"]["max_slope_degrees"], "max slope", zero=True)
    if slope >= 90:
        raise ValueError("max slope must be below 90 degrees")
    _number(metadata["profile"]["max_step"], "max step", zero=True)
    return shape, basis.astype(float)


def plan_next_steps(arrays, metadata, agent_xy, mission_origin_xy, mission_forward_xy, *,
                    horizon_m=2., max_start_adjustment_m=.15, max_lateral_m=.75,
                    minimum_progress_m=.25):
    """Plan from the actual supported agent XY toward a temporary frontier.

    Input XY uses the costmap's local horizontal frame in assumed metres; XYZ
    outputs use its assumed map frame. Mission origin and forward direction are
    fixed across replay ticks, independent of camera pitch and current heading.

    The actual agent ground point must have observed raw support and a valid
    complete circle. A graph centre within ``max_start_adjustment_m`` is joined
    with a checked capsule and retained in the route, never used to teleport the
    agent. Eight-neighbour edges and simplification use the same capsule test.

    The goal is the furthest connected forward cell in the fixed mission band;
    ties prefer less lateral deviation then less path cost. The lateral limit
    constrains goal selection, while supported detours can leave that band. Only
    the display route is clipped at ``horizon_m``. A caller must admit only the
    current observation prefix and keep the mission active after video EOF.
    """
    agent = _xy(agent_xy, "agent_xy")
    mission_origin = _xy(mission_origin_xy, "mission_origin_xy")
    forward = _xy(mission_forward_xy, "mission_forward_xy")
    norm = float(np.linalg.norm(forward))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("mission_forward_xy must have nonzero finite length")
    forward /= norm
    lateral = np.array([-forward[1], forward[0]])
    horizon = _number(horizon_m, "horizon_m")
    adjustment = _number(max_start_adjustment_m, "max_start_adjustment_m", zero=True)
    band = _number(max_lateral_m, "max_lateral_m", zero=True)
    progress_min = _number(minimum_progress_m, "minimum_progress_m", zero=True)
    result = {
        "schema_version": 1, "status": "awaiting_support", "reason": None,
        "path_points_assumed_m": [], "display_path_points_assumed_m": [],
        "agent_ground_point_assumed_m": None, "goal_point_assumed_m": None,
        "path_cells": [], "waypoints": [], "waypoint_cells": [],
        "length_assumed_m": 0., "display_length_assumed_m": 0.,
        "start_adjustment_m": None, "horizon_m": horizon, "display_clipped": False,
        "mission_origin_xy": mission_origin.tolist(), "mission_forward_xy": forward.tolist(),
        "mission_active": True, "mission_complete": False, "temporary_frontier": True,
        "swept_footprint_checked": False, "safety_validated": False,
    }
    if metadata.get("availability", "available") != "available":
        result.update(status="blocked_inputs", reason=metadata.get("reason") or "Planning inputs unavailable")
        return result
    shape, basis = _validate_map(arrays, metadata)
    origin, resolution = np.asarray(arrays["origin"], float), float(arrays["resolution"][0])
    grid_xy = (agent - origin) / resolution
    if not np.isfinite(grid_xy).all() or np.any(grid_xy < 0) or grid_xy[0] >= shape[1] or grid_xy[1] >= shape[0]:
        result["reason"] = "Current agent is outside observed support; no relocation"
        return result
    x, y = np.floor(grid_xy).astype(int)
    if arrays["geometry_state"][y, x] != TRAVERSABLE:
        result["reason"] = "Current agent ground is unknown or blocked; no relocation"
        return result
    actual = np.r_[agent, arrays["support_height"][y, x]]
    if not segment_supported(actual, actual, arrays, metadata):
        result["reason"] = "Current agent's full footprint lacks compatible observed support"
        return result
    result["agent_ground_point_assumed_m"] = (actual @ basis).tolist()

    valid = set(map(tuple, np.argwhere(arrays["decision_state"] == TRAVERSABLE)))
    local = {cell: _local_point(cell, arrays) for cell in valid}
    starts = [(float(np.linalg.norm(point[:2] - agent)), cell) for cell, point in local.items()
              if np.linalg.norm(point[:2] - agent) <= adjustment + 1e-12]
    starts.sort()
    start = next((cell for _, cell in starts if segment_supported(actual, local[cell], arrays, metadata)), None)
    if start is None:
        result["reason"] = "No supported graph centre has a checked connection within the start adjustment bound"
        return result
    result["start_adjustment_m"] = float(np.linalg.norm(local[start][:2] - agent))
    distances = {start: float(np.linalg.norm(local[start] - actual) * arrays["costs"][start])}
    parents, edge_cache = {}, {}
    pending = [(distances[start], *start)]
    while pending:
        cost, y, x = heapq.heappop(pending)
        cell = (y, x)
        if cost > distances[cell]:
            continue
        for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
            nxt = (y + dy, x + dx)
            if nxt not in valid:
                continue
            edge = tuple(sorted((cell, nxt)))
            if edge not in edge_cache:
                edge_cache[edge] = segment_supported(local[cell], local[nxt], arrays, metadata)
            if not edge_cache[edge]:
                continue
            step = float(np.linalg.norm(local[nxt] - local[cell]))
            score = cost + step * .5 * (arrays["costs"][cell] + arrays["costs"][nxt])
            if not math.isfinite(score):
                raise ValueError("route cost overflowed")
            if score < distances.get(nxt, math.inf):
                distances[nxt] = float(score)
                parents[nxt] = cell
                heapq.heappush(pending, (float(score), *nxt))

    candidates = []
    for cell, cost in distances.items():
        delta = local[cell][:2] - mission_origin
        forward_progress = float(delta @ forward)
        sideways = abs(float(delta @ lateral))
        if float((local[cell][:2] - agent) @ forward) > progress_min + 1e-12 and sideways <= band + 1e-12:
            candidates.append((forward_progress, sideways, cost, cell))
    if not candidates:
        result.update(status="awaiting_observation", reason="No connected supported forward frontier meets the mission band and progress bound")
        return result
    # Numerical ties are resolved consistently without changing meaningful map progress.
    furthest = max(row[0] for row in candidates)
    candidates = [row for row in candidates if row[0] >= furthest - 1e-9]
    smallest_lateral = min(row[1] for row in candidates)
    candidates = [row for row in candidates if row[1] <= smallest_lateral + 1e-9]
    goal = min(candidates, key=lambda row: (row[2], row[3]))[3]
    route = [goal]
    while route[-1] != start:
        route.append(parents[route[-1]])
    route.reverse()
    raw = [actual] + [local[cell] for cell in route]
    raw_cells = [(int(np.floor(grid_xy[1])), int(np.floor(grid_xy[0])))] + route
    if np.linalg.norm(raw[1] - actual) <= 1e-12:
        raw.pop(1)
        raw_cells.pop(1)
    chosen = [0]
    i = 0
    while i < len(raw) - 1:
        j = len(raw) - 1
        while j > i + 1 and not segment_supported(raw[i], raw[j], arrays, metadata):
            j -= 1
        chosen.append(j)
        i = j
    path = np.array([raw[index] @ basis for index in chosen])
    display = clip_route_horizon(path, horizon)
    length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    display_length = float(np.linalg.norm(np.diff(display, axis=0), axis=1).sum())
    result.update(
        status="ok", reason="Supported temporary forward frontier; mission remains active",
        path_points_assumed_m=path.tolist(), display_path_points_assumed_m=display.tolist(),
        goal_point_assumed_m=path[-1].tolist(), path_cells=[[int(v) for v in cell] for cell in route],
        waypoints=path.tolist(), waypoint_cells=[[int(v) for v in raw_cells[index]] for index in chosen],
        length_assumed_m=length, display_length_assumed_m=display_length,
        display_clipped=length > horizon + 1e-12, swept_footprint_checked=True,
        goal_forward_progress_m=float((local[goal][:2] - mission_origin) @ forward),
        goal_lateral_offset_m=float((local[goal][:2] - mission_origin) @ lateral),
        reachable_cells=len(distances),
    )
    return result
