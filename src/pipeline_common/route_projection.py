"""Bounded source-pixel annotations of an ordered saved native-world route.

This is camera projection, not a planner. Callers must join the route, W2C, K,
pixel-center transform and optional optical-axis depth to the same saved frame.
No metric scale is applied: path and camera translations already share units.
"""
from __future__ import annotations

import math
import numbers

import numpy as np


METHOD = "saved_w2c_intrinsics_inverse_center_affine_directed_route_v1"
MAX_ROUTE_POINTS = 4096
MAX_SAMPLES = 20000
MAX_ARROWHEADS = 512
MAX_ARROW_DEPTH_PAIR_TESTS = 20000
MAX_IMAGE_PIXELS = 16_000_000


def _array(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "iuf":
        raise ValueError(name + " must have a real numeric shape " + str(shape))
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(name + " must be finite")
    return array


def _number(value, name, *, zero=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(name + " must be a finite real number")
    value = float(value)
    if not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(name + " is outside its finite positive range")
    return value


def _shape(value, name):
    if (not isinstance(value, (list, tuple, np.ndarray)) or np.ndim(value) != 1 or len(value) != 2
            or any(isinstance(n, (bool, np.bool_)) or not isinstance(n, numbers.Integral) or n <= 0 for n in value)):
        raise ValueError(name + " must contain positive integer [height,width]")
    shape = tuple(int(n) for n in value)
    if shape[0] * shape[1] > MAX_IMAGE_PIXELS:
        raise ValueError(name + " exceeds the sixteen million pixel bound")
    return shape


def _camera(intrinsic, world_to_camera, transform):
    k = _array(intrinsic, (3, 3), "intrinsic")
    if (k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1], rtol=0, atol=1e-8)
            or abs(np.linalg.det(k)) < 1e-12):
        raise ValueError("intrinsic must be an invertible saved optical-axis camera matrix")
    pose = np.asarray(world_to_camera)
    if pose.shape == (4, 4):
        pose = _array(pose, (4, 4), "world_to_camera")
        if not np.allclose(pose[3], [0, 0, 0, 1], rtol=0, atol=1e-8):
            raise ValueError("world_to_camera has an invalid homogeneous row")
        pose = pose[:3]
    else:
        pose = _array(pose, (3, 4), "world_to_camera")
    rotation = pose[:, :3]
    if (not np.allclose(rotation @ rotation.T, np.eye(3), rtol=0, atol=1e-3)
            or not math.isclose(float(np.linalg.det(rotation)), 1., rel_tol=0, abs_tol=1e-3)):
        raise ValueError("world_to_camera must contain a saved rigid proper rotation")
    if not isinstance(transform, dict):
        raise ValueError("source_to_processed must be the complete saved transform dictionary")
    source_shape = _shape(transform.get("source_shape"), "source_shape")
    processed_shape = _shape(transform.get("processed_shape"), "processed_shape")
    affine = _array(transform.get("matrix"), (3, 3), "source_to_processed matrix")
    if not np.allclose(affine[2], [0, 0, 1], rtol=0, atol=1e-8) or abs(np.linalg.det(affine)) < 1e-12:
        raise ValueError("source_to_processed must be an invertible pixel-center affine")
    pad = transform.get("pad_ltrb")
    if (not isinstance(pad, (list, tuple)) or len(pad) != 4
            or any(isinstance(n, (bool, np.bool_)) or not isinstance(n, numbers.Integral) or n < 0 for n in pad)):
        raise ValueError("Saved pad_ltrb must contain four nonnegative integers")
    left, top, right, bottom = map(int, pad)
    if left + right >= processed_shape[1] or top + bottom >= processed_shape[0]:
        raise ValueError("Saved padding removes the whole processed image")
    source_roi = np.array([0., 0., source_shape[1] - 1., source_shape[0] - 1.])
    processed_roi = np.array([float(left), float(top), processed_shape[1] - right - 1., processed_shape[0] - bottom - 1.])
    source_camera = np.linalg.solve(affine, k)
    if not np.isfinite(source_camera).all():
        raise ValueError("Inverse pixel-center affine projection overflowed")
    return k, pose, affine, source_camera, source_shape, processed_shape, source_roi, processed_roi


def _inside(pixels, roi):
    tolerance = 1e-7
    return ((pixels[:, 0] >= roi[0] - tolerance) & (pixels[:, 1] >= roi[1] - tolerance)
            & (pixels[:, 0] <= roi[2] + tolerance) & (pixels[:, 1] <= roi[3] + tolerance))


def _clip(camera0, camera1, planes):
    """Intersect an ordered camera-space segment with linear halfspaces."""
    lo, hi = 0., 1.
    for normal, offset in planes:
        start, end = float(normal @ camera0 + offset), float(normal @ camera1 + offset)
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError("Camera frustum arithmetic overflowed")
        if start < 0 and end < 0:
            return None
        if start < 0 <= end:
            lo = max(lo, start / (start - end))
        elif end < 0 <= start:
            hi = min(hi, start / (start - end))
        if hi <= lo:
            return None
    return lo, hi


def _pixels(camera, projection):
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        homogeneous = camera @ projection.T
        result = homogeneous[:, :2] / homogeneous[:, 2:3]
    if not np.isfinite(result).all():
        raise ValueError("Route camera projection overflowed")
    return result


def _depth_visibility(source_pixels, route_depth, affine, depth, valid, absolute, relative):
    if depth is None:
        return np.ones(len(route_depth), bool), np.zeros(len(route_depth), bool), np.zeros(len(route_depth), bool)
    processed = np.column_stack((source_pixels, np.ones(len(source_pixels)))) @ affine.T
    xy = processed[:, :2]
    height, width = depth.shape
    nearest = np.floor(xy + .5).astype(np.int64)
    nearest[:, 0] = np.clip(nearest[:, 0], 0, width - 1)
    nearest[:, 1] = np.clip(nearest[:, 1], 0, height - 1)
    known = valid[nearest[:, 1], nearest[:, 0]]
    # Conservatively use the nearest accepted depth and any nearer accepted
    # surface at the four surrounding saved grid centers; never interpolate
    # missing depth or invent an occluder from padding.
    lower = np.floor(xy).astype(np.int64)
    minimum = np.full(len(route_depth), np.inf)
    for dx, dy in ((0, 0), (0, 1), (1, 0), (1, 1)):
        x = np.clip(lower[:, 0] + dx, 0, width - 1)
        y = np.clip(lower[:, 1] + dy, 0, height - 1)
        minimum = np.minimum(minimum, np.where(valid[y, x], depth[y, x], np.inf))
    occluded = known & (route_depth > minimum + absolute + relative * minimum)
    return known & ~occluded, occluded, ~known


def _point_on_piece(points, depths, cumulative, distance):
    index = min(int(np.searchsorted(cumulative, distance, side="right") - 1), len(points) - 2)
    index = max(index, 0)
    fraction = (distance - cumulative[index]) / (cumulative[index + 1] - cumulative[index])
    pixel = points[index] + fraction * (points[index + 1] - points[index])
    z = 1. / ((1. - fraction) / depths[index] + fraction / depths[index + 1])
    return pixel, z


def _head_depth_visible(processed, depths, affine, depth, valid, absolute, relative, remaining):
    """Check the filled head's covered saved grid centers before drawing it."""
    if depth is None:
        return True, 0, 0, False
    first = np.floor(np.min(processed, axis=0)).astype(np.int64)
    last = np.ceil(np.max(processed, axis=0)).astype(np.int64)
    first = np.maximum(first, 0)
    last = np.minimum(last, [depth.shape[1] - 1, depth.shape[0] - 1])
    pairs = int(np.prod(last - first + 1))
    if pairs > remaining:
        return False, 0, 0, True
    xx, yy = np.meshgrid(np.arange(first[0], last[0] + 1), np.arange(first[1], last[1] + 1))
    pixels = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float64)
    edges = np.column_stack((processed[1] - processed[0], processed[2] - processed[0]))
    if abs(np.linalg.det(edges)) < 1e-12:
        return False, pairs, 0, False
    weights = np.linalg.solve(edges, (pixels - processed[0]).T).T
    inside = (weights >= -1e-9).all(axis=1) & (np.sum(weights, axis=1) <= 1. + 1e-9)
    pixels, weights = pixels[inside], weights[inside]
    if not len(pixels):
        return True, pairs, 0, False
    barycentric = np.column_stack((1. - np.sum(weights, axis=1), weights))
    route_z = 1. / np.sum(barycentric / depths, axis=1)
    source = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(affine).T
    visible, _, _ = _depth_visibility(source[:, :2], route_z, affine, depth, valid, absolute, relative)
    return bool(visible.all()), pairs, len(pixels), False


def project_route_to_source(route_world, intrinsic, world_to_camera, source_to_processed, *,
                            depth=None, depth_valid=None, near_depth=1e-6,
                            depth_absolute_tolerance=0., depth_relative_tolerance=.03,
                            sample_spacing_pixels=2., max_samples=MAX_SAMPLES,
                            arrowhead_length_pixels=24., arrow_spacing_pixels=90.,
                            max_arrowheads=MAX_ARROWHEADS):
    """Return source_segments[N,2,2], source_arrowheads[M,3,2], and JSON report.

    Camera axes are OpenCV: x right, y down, z forward. Pixel coordinates are
    centers, exactly matching the saved affine; there is no added half pixel.
    Both the nonpadding processed ROI and source-image center ROI are clipped.
    Original path order is retained. Subdivision samples at most half a saved
    processed pixel apart, using perspective-correct optical-axis route depth.

    If depth is supplied, its exact processed shape and an explicit accepted
    same-frame boolean mask are required. Missing/untrusted nearest depth hides
    the route. Absolute tolerances use native camera/path units, relative ones
    are fractions of saved depth. Occlusion remains a sampled display test,
    not certified clearance. Screen-space triangles follow visible route order;
    heads extending outside image/crop bounds or hidden/untrusted covered saved
    depth grid centers are omitted. Depth-head work is bounded independently.
    """
    k, pose, affine, source_camera, source_shape, processed_shape, source_roi, processed_roi = _camera(
        intrinsic, world_to_camera, source_to_processed)
    near = _number(near_depth, "near_depth")
    absolute = _number(depth_absolute_tolerance, "depth_absolute_tolerance", zero=True)
    relative = _number(depth_relative_tolerance, "depth_relative_tolerance", zero=True)
    spacing = _number(sample_spacing_pixels, "sample_spacing_pixels")
    head_length = _number(arrowhead_length_pixels, "arrowhead_length_pixels")
    arrow_spacing = _number(arrow_spacing_pixels, "arrow_spacing_pixels")
    if not 1 <= head_length or arrow_spacing < head_length or spacing > 2.:
        raise ValueError("Arrow length must be >=1 pixel; arrow spacing >=length; sample spacing <=2 pixels")
    for value, bound, name in ((max_samples, MAX_SAMPLES, "max_samples"), (max_arrowheads, MAX_ARROWHEADS, "max_arrowheads")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or not 1 <= value <= bound:
            raise ValueError(name + " exceeds its positive integer bound")
    path = np.asarray(route_world)
    if path.size == 0:
        path = np.empty((0, 3))
    if path.ndim != 2 or path.shape[1:] != (3,) or len(path) > MAX_ROUTE_POINTS:
        raise ValueError("route_world must contain at most 4096 ordered native XYZ points")
    path = _array(path, path.shape, "route_world")
    if depth is not None:
        depth = np.asarray(depth)
        if depth.shape == (*processed_shape, 1):
            depth = depth[..., 0]
        if depth.shape != processed_shape or depth.dtype.kind not in "iuf":
            raise ValueError("depth must match the exact processed optical-axis grid")
        if depth_valid is None:
            raise ValueError("Provided depth requires an explicit same-frame accepted depth_valid mask")
        valid = np.asarray(depth_valid)
        if valid.shape != processed_shape or valid.dtype.kind != "b":
            raise ValueError("depth_valid must be boolean on the exact processed grid")
        valid = valid & np.isfinite(depth) & (depth > 0)
        yy, xx = np.ogrid[:processed_shape[0], :processed_shape[1]]
        valid &= ((xx >= processed_roi[0]) & (xx <= processed_roi[2])
                  & (yy >= processed_roi[1]) & (yy <= processed_roi[3]))
    else:
        if depth_valid is not None:
            raise ValueError("depth_valid cannot be supplied without optical-axis depth")
        valid = None
    report = {
        "schema_version": 1, "method": METHOD, "status": "no_route", "route_point_count": len(path),
        "direction": "saved_order_start_to_goal", "pixel_coordinates": "source_rgb_saved_pixel_centers",
        "source_shape": list(source_shape), "processed_shape": list(processed_shape),
        "source_roi_xyxy": source_roi.tolist(), "processed_roi_xyxy": processed_roi.tolist(),
        "source_to_processed_matrix": affine.tolist(), "near_depth_native_units": near,
        "input_segments": max(0, len(path) - 1), "behind_segments": 0, "offscreen_segments": 0,
        "projected_segments": 0, "visible_segments": 0, "arrow_count": 0,
        "sample_count": 0, "visible_samples": 0, "occluded_samples": 0, "unknown_depth_samples": 0,
        "degenerate_segments": 0, "sample_budget": int(max_samples),
        "sample_spacing_source_pixels": spacing, "maximum_processed_sample_spacing_pixels": .5,
        "depth_occlusion": "saved_processed_optical_axis_depth" if depth is not None else "unchecked_no_depth",
        "depth_sampling": "nearest_accepted_and_minimum_four_adjacent_saved_centers" if depth is not None else "none",
        "depth_absolute_tolerance_native_units": absolute, "depth_relative_tolerance": relative,
        "arrowhead_length_source_pixels": head_length, "arrow_spacing_source_pixels": arrow_spacing,
        "arrowhead_candidates": 0, "arrowheads_display_sampled": False, "arrowhead_cap": int(max_arrowheads),
        "arrowhead_depth_pair_test_cap": MAX_ARROW_DEPTH_PAIR_TESTS,
        "arrowhead_depth_pair_tests": 0, "arrowhead_depth_grid_samples": 0,
        "arrowheads_omitted_depth_budget": 0,
        "saved_geometry_changed": False, "route_replanned": False, "occlusion_is_safety_certificate": False,
        "limitations": ["Same-frame camera/grid/source identity must be joined by the caller.",
                        "Saved depth has reconstruction uncertainty; occlusion is a sampled display approximation.",
                        "Arrows are screen annotations of the saved research route, not a robot command."]}
    empty = np.empty((0, 2, 2), np.float64)
    empty_heads = np.empty((0, 3, 2), np.float64)
    if len(path) < 2:
        return {"source_segments": empty, "source_arrowheads": empty_heads, "report": report}
    with np.errstate(over="ignore", invalid="ignore"):
        camera = path @ pose[:, :3].T + pose[:, 3]
    if not np.isfinite(camera).all():
        raise ValueError("Native route camera transform overflowed")
    planes = [(np.array([0., 0., 1.]), -near)]
    for projection, roi in ((k, processed_roi), (source_camera, source_roi)):
        planes.extend(((projection[0] - roi[0] * projection[2], 0.),
                       (roi[2] * projection[2] - projection[0], 0.),
                       (projection[1] - roi[1] * projection[2], 0.),
                       (roi[3] * projection[2] - projection[1], 0.)))
    segments, pieces = [], []
    last_route_position = None
    for index, (c0, c1) in enumerate(zip(camera[:-1], camera[1:])):
        if np.array_equal(c0, c1):
            report["degenerate_segments"] += 1
            continue
        if c0[2] < near and c1[2] < near:
            report["behind_segments"] += 1
            continue
        clipped = _clip(c0, c1, planes)
        if clipped is None:
            report["offscreen_segments"] += 1
            continue
        lo, hi = clipped
        endpoints = np.array([c0 + lo * (c1 - c0), c0 + hi * (c1 - c0)])
        source = _pixels(endpoints, source_camera)
        processed = _pixels(endpoints, k)
        length = float(np.linalg.norm(source[1] - source[0]))
        if length <= 1e-9:
            report["degenerate_segments"] += 1
            continue
        report["projected_segments"] += 1
        count = max(1, math.ceil(length / spacing), math.ceil(float(np.linalg.norm(processed[1] - processed[0])) / .5))
        if report["sample_count"] + count + 1 > max_samples:
            raise ValueError("Route projection exceeds its bounded sample budget")
        fraction = np.linspace(0., 1., count + 1)
        pixels = source[0] + fraction[:, None] * (source[1] - source[0])
        inverse_depth = (1. - fraction) / endpoints[0, 2] + fraction / endpoints[1, 2]
        z = 1. / inverse_depth
        visible, occluded, unknown = _depth_visibility(pixels, z, affine, depth, valid, absolute, relative)
        report["sample_count"] += len(pixels)
        report["visible_samples"] += int(visible.sum())
        report["occluded_samples"] += int(occluded.sum())
        report["unknown_depth_samples"] += int(unknown.sum())
        edges = visible[:-1] & visible[1:]
        changes = np.diff(np.r_[False, edges, False].astype(np.int8))
        for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
            segment = np.array([pixels[start], pixels[end]])
            if np.linalg.norm(segment[1] - segment[0]) <= 1e-9:
                continue
            segments.append(segment)
            position = index + lo + (hi - lo) * ((fraction[[start, end]] / endpoints[1, 2]) / inverse_depth[[start, end]])
            depths = z[[start, end]]
            if (pieces and last_route_position is not None and abs(last_route_position - position[0]) < 1e-8
                    and np.linalg.norm(pieces[-1]["points"][-1] - segment[0]) < 1e-7):
                pieces[-1]["points"].append(segment[1])
                pieces[-1]["depths"].append(depths[1])
            else:
                pieces.append({"points": [segment[0], segment[1]], "depths": [depths[0], depths[1]]})
            last_route_position = float(position[1])
    candidates = []
    for piece in pieces:
        points = np.asarray(piece["points"])
        cumulative = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        piece.update(points=points, depths=np.asarray(piece["depths"]), cumulative=cumulative)
        total = float(cumulative[-1])
        if total < 2 * head_length:
            continue
        tip_distances = list(np.arange(max(arrow_spacing, 2 * head_length), total, arrow_spacing))
        if not tip_distances or total - tip_distances[-1] >= .5 * arrow_spacing:
            tip_distances.append(total)
        candidates.extend((piece, float(distance)) for distance in tip_distances)
    report["arrowhead_candidates"] = len(candidates)
    report["arrowheads_display_sampled"] = len(candidates) > max_arrowheads
    selected = np.linspace(0, len(candidates) - 1, min(len(candidates), max_arrowheads), dtype=int) if candidates else []
    heads = []
    for chosen in selected:
        piece, distance = candidates[chosen]
        tip, tip_z = _point_on_piece(piece["points"], piece["depths"], piece["cumulative"], distance)
        tail, tail_z = _point_on_piece(piece["points"], piece["depths"], piece["cumulative"], distance - head_length)
        direction = tip - tail
        norm = np.linalg.norm(direction)
        if norm <= 1e-9:
            continue
        perpendicular = np.array([-direction[1], direction[0]]) / norm * .3 * head_length
        triangle = np.array([tip, tail + perpendicular, tail - perpendicular])
        processed = np.column_stack((triangle, np.ones(3))) @ affine.T
        if not (_inside(triangle, source_roi).all() and _inside(processed[:, :2], processed_roi).all()):
            continue
        visible, _, _ = _depth_visibility(triangle, np.array([tip_z, tail_z, tail_z]), affine, depth, valid, absolute, relative)
        if visible.all():
            admitted, pairs, samples, exhausted = _head_depth_visible(
                processed[:, :2], np.array([tip_z, tail_z, tail_z]), affine, depth, valid, absolute, relative,
                MAX_ARROW_DEPTH_PAIR_TESTS - report["arrowhead_depth_pair_tests"])
            report["arrowhead_depth_pair_tests"] += pairs
            report["arrowhead_depth_grid_samples"] += samples
            report["arrowheads_omitted_depth_budget"] += int(exhausted)
            if admitted:
                heads.append(triangle)
    report["visible_segments"] = len(segments)
    report["arrow_count"] = len(heads)
    if segments:
        report["status"] = "visible"
    elif report["projected_segments"]:
        report["status"] = "depth_unknown" if report["unknown_depth_samples"] else "occluded"
    elif report["behind_segments"] and report["behind_segments"] + report["degenerate_segments"] == report["input_segments"]:
        report["status"] = "behind"
    elif report["degenerate_segments"] == report["input_segments"]:
        report["status"] = "no_route"
    else:
        report["status"] = "offscreen"
    return {"source_segments": np.asarray(segments, dtype=np.float64).reshape(-1, 2, 2),
            "source_arrowheads": np.asarray(heads, dtype=np.float64).reshape(-1, 3, 2), "report": report}
