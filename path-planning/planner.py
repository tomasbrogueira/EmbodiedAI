"""Dijkstra on the existing cost map, retaining original point-row IDs."""

from dataclasses import dataclass
from heapq import heappop, heappush
from numbers import Integral

import numpy as np


@dataclass
class PathResult:
    status: str
    ids: np.ndarray
    positions: np.ndarray
    total_cost: float


def _aligned_costs(points, cost_map):
    points, mapped = np.asarray(points), np.asarray(cost_map.points)
    costs = np.asarray(cost_map.cost)
    if (points.ndim != 2 or points.shape[1] != 3
            or points.dtype.kind not in "iuf" or mapped.dtype.kind not in "iuf"
            or not np.array_equal(points, mapped, equal_nan=True)):
        raise ValueError("points must match cost_map.points in original row order")
    if (costs.shape != (len(points),) or costs.dtype.kind not in "iuf"
            or np.isnan(costs).any() or (costs < 1).any()):
        raise ValueError("costs must be aligned, at least 1, or +inf")
    return points, costs


def _failure(status):
    return PathResult(status, np.empty(0, dtype=np.intp), np.empty((0, 3)), np.inf)


def find_path(points, cost_map, start_id, goal_id):
    """Minimize distance times average endpoint cost over permitted connections.

    Inputs must remain unchanged during the search. Invalid endpoints and
    disconnected regions return empty routes; mismatched arrays raise ValueError.
    """
    points, costs = _aligned_costs(points, cost_map)
    for row, status in ((start_id, "invalid_start"), (goal_id, "invalid_goal")):
        if (isinstance(row, (bool, np.bool_)) or not isinstance(row, Integral)
                or not 0 <= row < len(points) or not np.isfinite(costs[row])
                or not np.isfinite(points[row]).all()):
            return _failure(status)

    distance, previous = {start_id: 0.0}, {}
    queue = [(0.0, start_id)]
    while queue:
        total, row = heappop(queue)
        if total != distance[row]:
            continue
        if row == goal_id:
            route = [row]
            while row != start_id:
                row = previous[row]
                route.append(row)
            ids = np.asarray(route[::-1], dtype=np.intp)
            return PathResult("ok", ids, points[ids], total)
        for other in cost_map.neighbours(row):
            length = np.linalg.norm(np.subtract(points[other], points[row], dtype=float))
            candidate = total + length * (costs[row] / 2 + costs[other] / 2)
            # Strict improvement prevents predecessor cycles on zero-cost edges.
            if candidate < distance.get(other, np.inf):
                distance[other], previous[other] = candidate, row
                heappush(queue, (candidate, other))
    return _failure("no_path")
