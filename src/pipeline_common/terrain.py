"""Deterministic local steepness from observed voxel centers, without a robot.

The caller supplies an orthonormal map-frame basis (rows x, y, up) and its
verified or explicitly assumed provenance. A fitted floor normal must not be
substituted for gravity here. Distances and settings use the centers' units.
Missing voxel-box coverage remains unknown; this estimator does not establish support,
free space, clearance, or traversability.
"""
from __future__ import annotations

import math
import numbers

import numpy as np


METHOD = "observed_lower_voxel_two_scale_huber_heightfield_v2_box_coverage"
MAX_COVERAGE_PAIR_TESTS = 16_000_000


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(name + " must be a positive finite number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be a positive finite number")
    return value


def _real_array(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "iuf":
        raise ValueError(name + " has an invalid numeric shape")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(name + " must be finite")
    return array


def _settings(settings, voxel_size, resolution):
    defaults = {
        "lower_layer_height": max(1.5 * voxel_size, .05),
        "neighbor_radius": max(3 * voxel_size, 1.5 * resolution),
        "min_neighbors": 6,
        "min_baseline": max(voxel_size, .5 * resolution),
        "max_residual": max(voxel_size, .03),
        "huber_delta": max(.5 * voxel_size, .01),
        "huber_iterations": 6,
        "max_scale_disagreement_degrees": 10.,
    }
    if settings is not None and not isinstance(settings, dict):
        raise ValueError("terrain settings must be a dictionary")
    supplied = dict(settings or {})
    if set(supplied) - set(defaults) - {"second_radius"}:
        raise ValueError("unknown terrain setting: " + ", ".join(sorted(set(supplied) - set(defaults) - {"second_radius"})))
    result = {**defaults, **supplied}
    for key in ("lower_layer_height", "neighbor_radius", "min_baseline", "max_residual", "huber_delta"):
        result[key] = _positive(result[key], key)
    result["second_radius"] = _positive(supplied.get("second_radius", 2 * result["neighbor_radius"]), "second_radius")
    if result["second_radius"] < result["neighbor_radius"]:
        raise ValueError("second_radius must be at least neighbor_radius")
    for key, low, high in (("min_neighbors", 6, 1_000_000), ("huber_iterations", 1, 50)):
        value = result[key]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or not low <= value <= high:
            raise ValueError(key + " must be a bounded integer")
        result[key] = int(value)
    angle = result["max_scale_disagreement_degrees"]
    if isinstance(angle, (bool, np.bool_)) or not isinstance(angle, numbers.Real) or not math.isfinite(angle) or not 0 <= angle <= 90:
        raise ValueError("max_scale_disagreement_degrees must be finite in [0,90]")
    result["max_scale_disagreement_degrees"] = float(angle)
    return result


def _span_ok(xy, minimum):
    if len(xy) < 3:
        return False
    # Twice the principal-axis standard deviation measures spread in both
    # horizontal directions. A line or a tiny cluster cannot identify a plane.
    spread = np.linalg.svd(xy - np.mean(xy, axis=0), compute_uv=False)
    return len(spread) == 2 and 2 * spread[1] / math.sqrt(len(xy)) >= minimum - 1e-12 * minimum


def _voxel_xy_coverage(projected, basis, origin, resolution, shape, voxel_size):
    """Exact positive-area projected native-cube/grid-cell intersections.

    The cube remains aligned with native XYZ. Its XY projection is a zonotope;
    separating axes comprise grid X/Y and normals to its projected cube edges.
    The resulting cells are measurement queries, never free-space evidence.
    """
    half_xy = .5 * voxel_size * np.sum(np.abs(basis[:2]), axis=1)
    axes = [np.array([1., 0.]), np.array([0., 1.])]
    for edge in basis[:2].T:
        length = np.linalg.norm(edge)
        if length > 1e-12:
            axes.append(np.array([-edge[1], edge[0]]) / length)
    axes = np.asarray(axes)
    cube_half = .5 * voxel_size * np.sum(np.abs(basis[:2].T @ axes.T), axis=0)
    cell_half = .5 * resolution * np.sum(np.abs(axes), axis=1)
    extent = cube_half + cell_half
    tolerance = 64 * np.finfo(float).eps * max(voxel_size, resolution)
    coverage = np.zeros(shape, dtype=np.bool_)
    tests = 0
    for center in projected[:, :2]:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            low = (center - half_xy - origin) / resolution
            high = (center + half_xy - origin) / resolution
        if not np.isfinite(low).all() or not np.isfinite(high).all():
            raise ValueError("voxel box projection or grid coordinates overflowed")
        if np.any(high <= 0) or low[0] >= shape[1] or low[1] >= shape[0]:
            continue
        first = np.floor(np.clip(low, [0, 0], [shape[1], shape[0]])).astype(np.int64)
        last = np.ceil(np.clip(high, [0, 0], [shape[1], shape[0]])).astype(np.int64) - 1
        if np.any(first > last):
            continue
        count = int((last[0] - first[0] + 1) * (last[1] - first[1] + 1))
        tests += count
        if tests > MAX_COVERAGE_PAIR_TESTS:
            raise ValueError("voxel/grid coverage exceeds sixteen million bounded pair tests")
        xx, yy = np.meshgrid(np.arange(first[0], last[0] + 1), np.arange(first[1], last[1] + 1))
        indices = np.column_stack((xx.ravel(), yy.ravel()))
        targets = origin + (indices + .5) * resolution
        delta = (targets - center) @ axes.T
        overlap = np.all(np.abs(delta) < extent - tolerance, axis=1)
        selected = indices[overlap]
        coverage[selected[:, 1], selected[:, 0]] = True
    return coverage, tests


def _fit(points, target, radius, basis, settings):
    if len(points) < settings["min_neighbors"]:
        return None, "insufficient_neighbors"
    if not _span_ok(points[:, :2], settings["min_baseline"]):
        return None, "insufficient_horizontal_span"
    # Scaling XY keeps the solve conditioned across metre/mm/native units;
    # gradients are converted back afterwards. Every unique voxel starts with
    # one vote; neither pixel density nor repeated frames increase its weight.
    design = np.column_stack(((points[:, :2] - target) / radius, np.ones(len(points))))
    weights = np.ones(len(points))
    coefficients = None
    for _ in range(settings["huber_iterations"]):
        root = np.sqrt(weights)
        coefficients, _, rank, _ = np.linalg.lstsq(design * root[:, None], points[:, 2] * root, rcond=None)
        if rank < 3 or not np.isfinite(coefficients).all():
            return None, "degenerate_fit"
        error = points[:, 2] - design @ coefficients
        weights = np.minimum(1., settings["huber_delta"] / np.maximum(np.abs(error), np.finfo(float).tiny))
    # Refit with the final robust weights so recorded residuals and the normal
    # correspond to the same coefficients.
    root = np.sqrt(weights)
    coefficients, _, rank, _ = np.linalg.lstsq(design * root[:, None], points[:, 2] * root, rcond=None)
    if rank < 3 or not np.isfinite(coefficients).all():
        return None, "degenerate_fit"
    error = points[:, 2] - design @ coefficients
    inliers = np.abs(error) <= settings["max_residual"] + 1e-12 * settings["max_residual"]
    if np.count_nonzero(inliers) < settings["min_neighbors"] or not _span_ok(points[inliers, :2], settings["min_baseline"]):
        return None, "insufficient_robust_support"
    rms = math.sqrt(float(np.sum(weights * error ** 2) / np.sum(weights)))
    if not math.isfinite(rms) or rms > settings["max_residual"] + 1e-12 * settings["max_residual"]:
        return None, "residual_exceeds_limit"
    gradient = coefficients[:2] / radius
    normal_local = np.r_[-gradient, 1.]
    normal_local /= np.linalg.norm(normal_local)
    return {"slope": math.degrees(math.atan(float(np.linalg.norm(gradient)))),
            "normal": normal_local @ basis, "residual": rms,
            "support": int(np.count_nonzero(inliers))}, None


def estimate_voxel_terrain(centers, *, basis, origin, resolution, shape, voxel_size, settings=None):
    """Return fixed geometric slope/normal diagnostics and finite JSON metadata.

    Arrays index [y,x]; ``normal`` contains map-frame XYZ unit vectors. Unknown
    measurements use NaN, valid_mask=False and support_count=0. A small-scale
    fit is mandatory. An invalid larger fit never rescues it; two valid scales
    whose normals differ beyond the configured angle remain unknown. Otherwise
    the steeper valid fit supplies slope, normal, residual, and support_count.
    """
    points = np.asarray(centers)
    if points.ndim != 2 or points.shape[1:] != (3,) or points.dtype.kind not in "iuf":
        raise ValueError("centers must contain real numeric [N,3] voxel centers")
    points = _real_array(points, points.shape, "centers")
    basis = _real_array(basis, (3, 3), "basis")
    if not np.allclose(basis @ basis.T, np.eye(3), rtol=0, atol=1e-8) or not math.isclose(float(np.linalg.det(basis)), 1., abs_tol=1e-8):
        raise ValueError("basis must be a proper orthonormal frame with rows x,y,up")
    origin = _real_array(origin, (2,), "origin")
    resolution, voxel_size = _positive(resolution, "resolution"), _positive(voxel_size, "voxel_size")
    if not isinstance(shape, (list, tuple, np.ndarray)) or (isinstance(shape, np.ndarray) and shape.ndim != 1) or len(shape) != 2 or any(
        isinstance(n, (bool, np.bool_)) or not isinstance(n, numbers.Integral) or n <= 0 for n in shape
    ):
        raise ValueError("shape must contain two positive integers [height,width]")
    shape = tuple(int(n) for n in shape)
    if shape[0] * shape[1] > 4_000_000:
        raise ValueError("terrain grid exceeds four million cells")
    settings = _settings(settings, voxel_size, resolution)
    arrays = {"slope_degrees": np.full(shape, np.nan), "normal": np.full((*shape, 3), np.nan),
              "valid_mask": np.zeros(shape, dtype=np.bool_), "residual": np.full(shape, np.nan),
              "support_count": np.zeros(shape, dtype=np.int64)}
    unique = np.unique(points, axis=0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        projected = unique @ basis.T
        low = (projected[:, :2] - origin) / resolution
    if not np.isfinite(projected).all() or not np.isfinite(low).all():
        raise ValueError("voxel projection or grid coordinates overflowed")
    inside = (low[:, 0] >= 0) & (low[:, 0] < shape[1]) & (low[:, 1] >= 0) & (low[:, 1] < shape[0])
    # A voxel's center may lie outside a cell, or just outside the grid, while
    # its occupied cube intersects observed raw support inside that cell.
    half_xy = .5 * voxel_size * np.sum(np.abs(basis[:2]), axis=1)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        box_low = projected[:, :2] - half_xy
        box_high = projected[:, :2] + half_xy
        grid_high = origin + np.array([shape[1], shape[0]]) * resolution
    if not np.isfinite(box_low).all() or not np.isfinite(box_high).all() or not np.isfinite(grid_high).all():
        raise ValueError("voxel box projection or grid bounds overflowed")
    retained = np.all(box_high > origin, axis=1) & np.all(box_low < grid_high, axis=1)
    if np.any(np.abs(low[retained]) >= 2**62):
        raise ValueError("retained voxel center grid index exceeds integer bounds")
    projected = projected[retained]
    cells = np.floor(low[retained]).astype(np.int64)
    buckets = {}
    if len(cells):
        # Retained centers may lie outside the grid, so a flattened grid key
        # could alias distinct (y,x) columns across a row boundary.
        order = np.lexsort((cells[:, 0], cells[:, 1]))
        changed = np.any(np.diff(cells[order], axis=0) != 0, axis=1)
        for group in np.split(order, np.flatnonzero(changed) + 1):
            x, y = map(int, cells[group[0]])
            column = projected[group]
            keep = column[:, 2] <= np.min(column[:, 2]) + settings["lower_layer_height"] + 1e-12 * settings["lower_layer_height"]
            buckets[(y, x)] = column[keep]

    lower_points = np.concatenate(list(buckets.values()), axis=0) if buckets else np.empty((0, 3))
    coverage, pair_tests = _voxel_xy_coverage(lower_points, basis, origin, resolution, shape, voxel_size)
    queries = np.argwhere(coverage)

    def neighbors(y, x, target, radius):
        reach = int(math.ceil(radius / resolution)) + 1
        # Retained boundary voxels can have center buckets outside the grid.
        y0, y1 = y - reach, y + reach
        x0, x1 = x - reach, x + reach
        if (y1 - y0 + 1) * (x1 - x0 + 1) > 2 * len(buckets):
            parts = [cloud for (yy, xx), cloud in buckets.items() if y0 <= yy <= y1 and x0 <= xx <= x1]
        else:
            parts = [buckets[(yy, xx)] for yy in range(y0, y1 + 1) for xx in range(x0, x1 + 1) if (yy, xx) in buckets]
        cloud = np.concatenate(parts, axis=0) if parts else np.empty((0, 3))
        return cloud[np.linalg.norm(cloud[:, :2] - target, axis=1) <= radius + 1e-12 * radius]

    failures, large_invalid, disagreements = {}, 0, 0
    for y, x in queries:
        y, x = int(y), int(x)
        target = origin + (np.array([x, y]) + .5) * resolution
        small, reason = _fit(neighbors(y, x, target, settings["neighbor_radius"]), target,
                             settings["neighbor_radius"], basis, settings)
        if small is None:
            failures[reason] = failures.get(reason, 0) + 1
            continue
        large, _ = _fit(neighbors(y, x, target, settings["second_radius"]), target,
                        settings["second_radius"], basis, settings)
        selected = small
        if large is None:
            large_invalid += 1
        else:
            angle = math.degrees(math.acos(float(np.clip(small["normal"] @ large["normal"], -1., 1.))))
            if angle > settings["max_scale_disagreement_degrees"] + 1e-8:
                disagreements += 1
                continue
            if large["slope"] > small["slope"]:
                selected = large
        arrays["valid_mask"][y, x] = True
        arrays["slope_degrees"][y, x] = selected["slope"]
        arrays["normal"][y, x] = selected["normal"]
        arrays["residual"][y, x] = selected["residual"]
        arrays["support_count"][y, x] = selected["support"]
    valid_count = int(np.count_nonzero(arrays["valid_mask"]))
    height_quantization_bound = .5 * voxel_size * float(np.sum(np.abs(basis[2])))
    metadata = {"schema_version": 1, "method": METHOD, "settings": settings,
                "shape": list(shape), "origin": origin.tolist(), "resolution": resolution,
                "voxel_size": voxel_size, "basis": basis.tolist(),
                "basis_convention": "rows x,y,up; projected=map_centers@basis.T",
                "normal_frame": "input_map_frame", "distance_units": "same_as_input_centers",
                "robot_capability_used": False, "input_center_count": len(points),
                "unique_center_count": len(unique), "outside_grid_center_count": int(np.count_nonzero(~inside)),
                "lower_layer_center_count": len(lower_points),
                "center_columns": len(buckets), "occupied_columns": len(queries), "valid_cells": valid_count,
                "query_coverage_model": "positive_area_projected_native_voxel_cube_grid_cell_intersection",
                "query_coverage_is_support_or_free_space": False,
                "query_columns_without_center": sum((int(y), int(x)) not in buckets for y, x in queries),
                "coverage_pair_tests": pair_tests, "coverage_pair_test_cap": MAX_COVERAGE_PAIR_TESTS,
                "retained_grid_bbox_intersecting_voxels": int(np.count_nonzero(retained)),
                "center_to_sample_distance_bound": math.sqrt(3.) * .5 * voxel_size,
                "reference_up_height_quantization_bound": height_quantization_bound,
                "angular_quantization_resolution_proxy_degrees": {
                    "small": math.degrees(math.atan(height_quantization_bound / settings["neighbor_radius"])),
                    "large": math.degrees(math.atan(height_quantization_bound / settings["second_radius"]))},
                "angular_proxy_definition": "atan(2*reference_up_height_quantization_bound/nominal_patch_diameter); resolution indicator, not a confidence interval",
                "quantization_uncertainty_changes_measurement_or_policy": False,
                "unknown_cells": shape[0] * shape[1] - valid_count,
                "small_scale_failures": failures, "large_scale_invalid_with_valid_small": large_invalid,
                "scale_disagreement_cells": disagreements,
                "residual_definition": "Huber-weighted height RMS of the selected patch",
                "support_count_definition": "distinct lower-layer voxel centers within max_residual of the selected plane",
                "unknown_numeric_value": "NaN slope, normal, residual; zero support_count",
                "limitations": ["Voxel quantization and declared neighborhood scale limit angular resolution.",
                                "No queries outside occupied voxel-box coverage; no free-space or traversability decision.",
                                "Angular quantization proxies omit reconstruction/gravity errors and are not certified slope bounds.",
                                "Up/gravity and unit provenance are supplied by the caller, not estimated here."]}
    return arrays, metadata
