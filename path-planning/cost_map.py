"""Relative surface costs and on-demand connections on original point rows."""

from numbers import Integral, Real

import numpy as np
from scipy.spatial import cKDTree


def obstacle_distance_squared(points, a, b, reference, height):
    """Distance to the closed surface edge swept upward by ``height``."""
    delta, up = b - a, height * reference
    result = np.full(len(points), np.inf)
    for start, direction in ((a, delta), (a + up, delta), (a, up), (b, up)):
        offset = points - start
        length2 = direction @ direction
        t = np.clip(offset @ direction / length2, 0, 1) if length2 else np.zeros(len(points))
        residual = offset - t[:, None] * direction
        result = np.minimum(result, np.einsum("ij,ij->i", residual, residual))
    vertical = delta @ reference
    horizontal = delta - vertical * reference
    length2 = horizontal @ horizontal
    if length2:
        offset = points - a
        t = offset @ horizontal / length2
        u = offset @ reference - t * vertical
        inside = (t >= 0) & (t <= 1) & (u >= 0) & (u <= height)
        residual = offset[inside] - t[inside, None] * delta - u[inside, None] * reference
        result[inside] = np.minimum(result[inside], np.einsum("ij,ij->i", residual, residual))
    return result


def _settings(config):
    if not isinstance(config, dict) or "radius" not in config:
        raise ValueError("config must be a dictionary containing radius")
    radius = config["radius"]
    if (isinstance(radius, (bool, np.bool_)) or not isinstance(radius, Real)
            or not np.isfinite(radius) or radius <= 0):
        raise ValueError("radius must be finite and positive")
    defaults = dict(radius=radius, max_slope=np.pi / 6, w_slope=1.0,
                    w_clearance=1.0, clearance_scale=radius,
                    connection_radius=radius / 2, max_gap=radius / 4,
                    obstacle_margin=radius / 20, clearance_height=radius, min_support=6,
                    max_neighbors=65536)
    unknown = config.keys() - (defaults.keys() | {"support_mask"})
    if unknown:
        raise ValueError(f"Unknown cost settings: {sorted(unknown, key=str)}")
    settings = defaults | config
    for key, value in settings.items():
        if key == "support_mask":
            continue
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or value < 0):
            raise ValueError(f"{key} must be finite and nonnegative")
        if key not in ("w_slope", "w_clearance", "obstacle_margin") and value == 0:
            raise ValueError(f"{key} must be positive")
    if not 0 < settings["max_slope"] < np.pi / 2:
        raise ValueError("max_slope must be between 0 and pi/2")
    for key, minimum in (("min_support", 6), ("max_neighbors", 8)):
        if not isinstance(settings[key], Integral) or settings[key] < minimum:
            raise ValueError(f"{key} must be an integer of at least {minimum}")
    return settings


class CostMap:
    """Aligned costs; inputs must remain unchanged while this object is used."""

    def __init__(self, points, slope, reference, settings):
        self.points, self.slope, self.reference, self.config = points, slope, reference, settings
        self.cost = np.full(len(points), np.inf)
        finite = np.isfinite(points).all(axis=1)
        self.rows = np.flatnonzero(finite)
        self.tree = cKDTree(points[self.rows])
        self.eligible = (finite & settings["support_mask"] & np.isfinite(slope)
                         & (slope >= 0) & (slope < settings["max_slope"]))
        self.obstacle_rows = np.flatnonzero(finite & ~settings["support_mask"])
        self.obstacles = cKDTree(points[self.obstacle_rows])
        if reference is None:
            return
        self._index_overhead()
        support = cKDTree(np.unique(points[self.eligible], axis=0))
        eligible_rows = np.flatnonzero(self.eligible)
        batch_size = settings["max_neighbors"]
        for start in range(0, len(eligible_rows), batch_size):
            rows = eligible_rows[start:start + batch_size]
            distance = support.query(points[rows], k=[settings["min_support"]],
                                     distance_upper_bound=np.nextafter(settings["radius"], np.inf))[0][:, 0]
            rows = rows[distance <= settings["radius"]]
            clearance = self.obstacles.query(points[rows])[0]
            keep = clearance > settings["obstacle_margin"]
            rows, clearance = rows[keep], clearance[keep]
            keep = ~self._blocked_nodes(points[rows])
            rows, clearance = rows[keep], clearance[keep]
            self.cost[rows] = (1 + settings["w_slope"]
                               * (slope[rows] / settings["max_slope"]) ** 2
                               + settings["w_clearance"] * np.exp(
                                   -np.minimum(clearance, settings["clearance_scale"])
                                   / settings["clearance_scale"]))
        self.support = cKDTree(np.unique(points[np.isfinite(self.cost)], axis=0))

    def _index_overhead(self):
        """Compress height only in the search index; original coordinates stay intact."""
        reference, c = self.reference, self.config
        tangent = np.eye(3)[np.argmin(np.abs(reference))]
        tangent -= (tangent @ reference) * reference
        tangent /= np.linalg.norm(tangent)
        self.obstacle_frame = np.column_stack((tangent, np.cross(reference, tangent),
                       reference * c["obstacle_margin"] / (c["clearance_height"] / 2 + c["obstacle_margin"])))
        self.overhead = cKDTree(self.points[self.obstacle_rows] @ self.obstacle_frame)
        self._overhead_scale = (max(np.max(np.abs(self.obstacles.mins)), np.max(np.abs(self.obstacles.maxes)))
                                if len(self.obstacle_rows) else 0.)

    def _overhead_padding(self, positions):
        """Account for projection roundoff only in candidate lookup, never collision distance."""
        return 32 * np.finfo(float).eps * (self._overhead_scale + np.max(np.abs(positions), initial=0.)
                                          + self.config["clearance_height"] + self.config["obstacle_margin"])

    def _near(self, position, radius, tree=None, rows=None, box=False):
        """Yield every original row in a closed ball/box in bounded batches."""
        tree, rows = (self.tree, self.rows) if tree is None else (tree, rows)
        if not len(rows):
            return
        budget = self.config["max_neighbors"]
        radius2 = radius * radius
        stack = [(tree.tree, tree.mins, tree.maxes)]
        while stack:
            node, lower, upper = stack.pop()
            distance = np.maximum(np.maximum(lower - position, position - upper), 0)
            if (np.max(distance) if box else distance @ distance) > np.nextafter(radius if box else radius2, np.inf):
                continue
            if node.children <= (min(budget, 128) if box else budget) or node.split_dim < 0:
                for start in range(node.start_idx, node.end_idx, budget):
                    indices = tree.indices[start:min(start + budget, node.end_idx)]
                    delta = tree.data[indices] - position
                    keep = (np.max(np.abs(delta), axis=1) <= radius if box else
                            np.einsum("ij,ij->i", delta, delta) <= radius2)
                    if keep.any():
                        yield rows[indices[keep]]
            else:
                left_upper, right_lower = upper.copy(), lower.copy()
                left_upper[node.split_dim] = right_lower[node.split_dim] = node.split
                stack.append((node.greater, right_lower, upper))
                stack.append((node.lesser, lower, left_upper))

    def _blocked_nodes(self, positions):
        """Batch the common empty-box/hit cases; enumerate only ambiguous boxes."""
        c = self.config
        margin, height = c["obstacle_margin"], c["clearance_height"]
        padding = self._overhead_padding(positions)
        radius = margin + padding
        centres = (positions + height / 2 * self.reference) @ self.obstacle_frame
        distance, nearest = self.overhead.query(centres, p=np.inf,
                                                distance_upper_bound=np.nextafter(radius, np.inf))
        blocked = np.zeros(len(positions), dtype=bool)
        found = np.flatnonzero((nearest < len(self.obstacle_rows)) & (distance <= radius))
        if len(found):
            offset = self.points[self.obstacle_rows[nearest[found]]] - positions[found]
            up = height * self.reference
            t = np.clip(offset @ up / (up @ up), 0, 1)
            residual = offset - t[:, None] * up
            # Boundary and zero-margin cases use the exact predicate below.
            blocked[found] = np.einsum("ij,ij->i", residual, residual) < max(0., margin - padding) ** 2
        for index in found[~blocked[found]]:
            for rows in self._near(centres[index], radius, self.overhead, self.obstacle_rows, box=True):
                if np.any(obstacle_distance_squared(self.points[rows], positions[index], positions[index],
                                                    self.reference, height) <= margin * margin):
                    blocked[index] = True
                    break
        return blocked

    def obstacle_free(self, a, b):
        """Check the entire upward sweep, including endpoints and its spherical margin."""
        c = self.config
        height, margin = c["clearance_height"], c["obstacle_margin"]
        centre = ((a + b) / 2 + height / 2 * self.reference) @ self.obstacle_frame
        radius = (np.max(np.abs((b - a) @ self.obstacle_frame)) / 2 + margin
                  + self._overhead_padding(np.stack((a, b))))
        for rows in self._near(centre, radius, self.overhead, self.obstacle_rows, box=True):
            if np.any(obstacle_distance_squared(self.points[rows], a, b, self.reference, height)
                      <= margin * margin):
                return False
        return True

    def _connected(self, first, second):
        a, b = self.points[min(first, second)], self.points[max(first, second)]
        delta = b - a
        length = np.linalg.norm(delta)
        if length == 0:
            return self.obstacle_free(a, b)
        c = self.config
        height = delta @ self.reference
        horizontal = np.linalg.norm(delta - height * self.reference)
        if abs(height) >= horizontal * np.tan(c["max_slope"]):
            return False
        # Nearest-support distance is 1-Lipschitz: adding half the sample
        # interval guarantees coverage between samples as well as at them.
        steps = max(1, int(np.ceil(length / (c["max_gap"] / 4))))
        padding = length / (2 * steps)
        for start in range(0, steps + 1, c["max_neighbors"]):
            t = np.arange(start, min(start + c["max_neighbors"], steps + 1)) / steps
            distance = self.support.query(a + t[:, None] * delta)[0]
            if np.any(distance + padding > c["max_gap"] / 2):
                return False
        return self.obstacle_free(a, b)

    def neighbours(self, row):
        """Return reachable original IDs, using identical rules both ways."""
        if (isinstance(row, (bool, np.bool_)) or not isinstance(row, Integral)
                or not 0 <= row < len(self.points)):
            raise ValueError("row must be an original point-row ID")
        if not np.isfinite(self.cost[row]):
            return np.empty(0, dtype=np.intp)
        result = []
        for batch in self._near(self.points[row], self.config["connection_radius"]):
            for other in batch:
                if other != row and np.isfinite(self.cost[other]) and self._connected(row, other):
                    result.append(other)
        return np.asarray(sorted(result), dtype=np.intp)


def build_cost_map(points, slope, n_ref, config):
    """Build a relative surface map using the existing slope radius and reference.

    ``radius`` is required in cloud units; ``support_mask`` labels original
    floor rows. Unknown or infeasible rows retain infinite cost. Clearance is
    distance to observed non-floor points, capped at ``clearance_scale``.
    Observed non-floor geometry blocks the edge swept upward by
    ``clearance_height`` (default radius), with the existing obstacle margin.
    This model checks relative surface support and finite upward clearance; it does
    not establish human body fit or observed free volume. No metric scale or
    reference direction is inferred here.
    """
    settings = _settings(config)
    points, slope = np.asarray(points), np.asarray(slope)
    if points.ndim != 2 or points.shape[1] != 3 or points.dtype.kind not in "iuf":
        raise ValueError("points must be a real array with shape (N, 3)")
    if slope.shape != (len(points),) or slope.dtype.kind not in "iuf":
        raise ValueError("slope must be a real array aligned with points")
    support = np.asarray(settings.get("support_mask", np.zeros(len(points), dtype=bool)))
    if support.shape != (len(points),) or support.dtype.kind != "b":
        raise ValueError("support_mask must be a boolean array aligned with points")
    settings["support_mask"] = support
    if n_ref is None:
        reference = None
    else:
        reference = np.asarray(n_ref)
        if (reference.shape != (3,) or reference.dtype.kind not in "iuf"
                or not np.isfinite(reference).all() or not np.any(reference)):
            raise ValueError("n_ref must be a finite nonzero direction, or None")
        reference = reference.astype(float)
        reference /= np.max(np.abs(reference))
        reference /= np.linalg.norm(reference)
    return CostMap(points.astype(float, copy=False), slope, reference, settings)
