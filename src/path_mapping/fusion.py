"""Model-free, pixel-aligned fusion of SAM path scores and LingBot geometry."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numbers

import numpy as np


@dataclass(frozen=True)
class FusionResult:
    """Point evidence and lexicographically ordered voxel statistics.

    ``observations`` counts distinct frames with valid geometry in each voxel,
    while ``point_counts`` counts their pixels. A zero SAM score is negative
    evidence, not an unobserved pixel. Coordinates retain the input map's units.
    """

    path_points: np.ndarray
    path_colors: np.ndarray
    path_scores: np.ndarray
    path_confidence: np.ndarray
    path_frame_indices: np.ndarray
    context_points: np.ndarray
    context_colors: np.ndarray
    context_frame_indices: np.ndarray
    voxel_indices: np.ndarray
    voxel_centers: np.ndarray
    path_probabilities: np.ndarray
    observations: np.ndarray
    path_flags: np.ndarray
    path_weights: np.ndarray
    total_weights: np.ndarray
    point_counts: np.ndarray
    voxel_colors: np.ndarray


def _real_array(value, label: str, *, as_float64=True) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{label} must contain real numbers")
    return array.astype(np.float64, copy=False) if as_float64 else array


def _finite_scalar(value, label: str, minimum: float, maximum=None) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{label} must be a finite real number")
    scalar = float(value)
    if not math.isfinite(scalar) or scalar < minimum or (maximum is not None and scalar > maximum):
        bounds = f"in [{minimum}, {maximum}]" if maximum is not None else f"at least {minimum}"
        raise ValueError(f"{label} must be finite and {bounds}")
    return scalar


def fuse_frame_sequence(
    world_points,
    point_confidence,
    depth,
    path_scores,
    rgb,
    *,
    voxel_size=0.05,
    map_origin=(0, 0, 0),
    min_point_confidence=1.5,
    path_probability_threshold=0.5,
    min_observations=2,
) -> FusionResult:
    """Fuse a sequence whose geometry, RGB and SAM scores share one pixel grid.

    Inputs have shapes ``[S,H,W,3]`` for points/RGB and ``[S,H,W]`` for
    confidence/scores. Depth additionally accepts ``[S,H,W,1]``. RGB must be
    uint8. SAM scores must be finite probabilities; zero means no returned path
    instance. LingBot confidence is a positive weight and may exceed one.

    Nonfinite geometry/confidence/depth, nonpositive depth/confidence, and
    confidence below the configured minimum are excluded. All remaining pixels
    contribute to the denominator, including pixels with zero path score.
    Voxels pass inclusive probability and distinct-frame observation thresholds.
    """
    voxel_size = _finite_scalar(voxel_size, "voxel_size", 0)
    if voxel_size == 0:
        raise ValueError("voxel_size must be greater than zero")
    min_point_confidence = _finite_scalar(min_point_confidence, "min_point_confidence", 0)
    probability_threshold = _finite_scalar(path_probability_threshold, "path_probability_threshold", 0, 1)
    if isinstance(min_observations, (bool, np.bool_)) or not isinstance(min_observations, numbers.Integral) or min_observations < 1:
        raise ValueError("min_observations must be a positive integer")
    origin = _real_array(map_origin, "map_origin")
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("map_origin must have three finite coordinates")

    points = _real_array(world_points, "world_points", as_float64=False)
    if points.ndim != 4 or points.shape[-1] != 3 or any(size < 1 for size in points.shape[:3]):
        raise ValueError("world_points must have nonempty shape [S,H,W,3]")
    grid_shape = points.shape[:3]
    confidence = _real_array(point_confidence, "point_confidence")
    depths = _real_array(depth, "depth", as_float64=False)
    scores = _real_array(path_scores, "path_scores")
    colors = np.asarray(rgb)
    if confidence.shape != grid_shape:
        raise ValueError("point_confidence must share the [S,H,W] point grid")
    if depths.shape == (*grid_shape, 1):
        depths = depths[..., 0]
    if depths.shape != grid_shape:
        raise ValueError("depth must share the [S,H,W] point grid, optionally with a trailing singleton")
    if scores.shape != grid_shape:
        raise ValueError("path_scores must share the [S,H,W] point grid")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("path_scores must contain finite probabilities in [0,1]")
    if colors.shape != points.shape or colors.dtype != np.uint8:
        raise ValueError("rgb must be uint8 and share the [S,H,W,3] point grid")

    valid = (
        np.isfinite(points).all(axis=-1)
        & np.isfinite(confidence)
        & np.isfinite(depths)
        & (depths > 0)
        & (confidence > 0)
        & (confidence >= min_point_confidence)
    )
    context_points = points[valid].copy()
    context_colors = colors[valid].copy()
    valid_confidence = confidence[valid]
    valid_scores = scores[valid]
    frame_grid = np.broadcast_to(np.arange(grid_shape[0], dtype=np.int64)[:, None, None], grid_shape)
    context_frames = frame_grid[valid].copy()
    positive = valid_scores > 0

    # floor, rather than truncation, keeps negative-coordinate voxels correct.
    with np.errstate(over="ignore", invalid="ignore"):
        quantized = np.floor((context_points - origin) / voxel_size)
    if not np.isfinite(quantized).all() or (quantized < -(2**63)).any() or (quantized >= 2**63).any():
        raise ValueError("Coordinates and voxel_size produce voxel indices outside int64 range")
    voxel_indices, inverse = np.unique(quantized.astype(np.int64), axis=0, return_inverse=True)
    count = len(voxel_indices)
    with np.errstate(over="ignore", invalid="ignore"):
        total_weights = np.bincount(inverse, weights=valid_confidence, minlength=count)
        path_weights = np.bincount(inverse, weights=valid_confidence * valid_scores, minlength=count)
    if not np.isfinite(total_weights).all() or not np.isfinite(path_weights).all():
        raise ValueError("Accumulated confidence weights overflowed")
    probabilities = np.divide(path_weights, total_weights, out=np.zeros(count, dtype=np.float64), where=total_weights > 0)
    probabilities = np.clip(probabilities, 0, 1)
    point_counts = np.bincount(inverse, minlength=count).astype(np.int64)
    distinct_pairs = np.unique(np.column_stack((inverse, context_frames)), axis=0)
    observations = np.bincount(distinct_pairs[:, 0], minlength=count).astype(np.int64)
    flags = (probabilities >= probability_threshold) & (observations >= min_observations)
    # Normalize weights first to avoid multiplying large raw confidence by 255.
    normalized_weights = valid_confidence / total_weights[inverse]
    voxel_colors = np.empty((count, 3), dtype=np.uint8)
    for channel in range(3):
        sums = np.bincount(inverse, weights=normalized_weights * context_colors[:, channel], minlength=count)
        voxel_colors[:, channel] = np.rint(np.clip(sums, 0, 255)).astype(np.uint8)
    voxel_centers = origin + (voxel_indices.astype(np.float64) + 0.5) * voxel_size
    if not np.isfinite(voxel_centers).all():
        raise ValueError("Voxel centers overflowed")

    return FusionResult(
        path_points=context_points[positive].copy(),
        path_colors=context_colors[positive].copy(),
        path_scores=valid_scores[positive].copy(),
        path_confidence=valid_confidence[positive].copy(),
        path_frame_indices=context_frames[positive].copy(),
        context_points=context_points,
        context_colors=context_colors,
        context_frame_indices=context_frames,
        voxel_indices=voxel_indices,
        voxel_centers=voxel_centers,
        path_probabilities=probabilities,
        observations=observations,
        path_flags=flags,
        path_weights=path_weights,
        total_weights=total_weights,
        point_counts=point_counts,
        voxel_colors=voxel_colors,
    )
