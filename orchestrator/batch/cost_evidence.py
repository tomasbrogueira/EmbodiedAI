"""Aligned floor labels from existing predictions."""

import numpy as np


def load_cost_evidence(points, lingbot_dir, sam_dir, video_name, prompt="floor", *,
                       frame_step, pixel_step, max_depth=25.0,
                       min_depth_conf=2.5, patch_size=14, include_frame_ids=False):
    """Recover labels and optional frame IDs; refuse mismatched XYZ rows."""
    from pathlib import Path

    from io_utils import frame_number, load_depth_map, load_extrinsic, load_intrinsic
    from lingbot_geometry import depth_to_world_points
    from sam_masks import build_sam_index, load_sam_mask, preprocess_mask

    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    for name, value in (("frame_step", frame_step), ("pixel_step", pixel_step),
                        ("patch_size", patch_size)):
        if not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if max_depth is not None and (not np.isfinite(max_depth) or max_depth <= 0):
        raise ValueError("max_depth must be positive or None")
    if min_depth_conf is not None and (not np.isfinite(min_depth_conf)
                                      or min_depth_conf < 0):
        raise ValueError("min_depth_conf must be nonnegative or None")
    paths = sorted(Path(lingbot_dir).glob("frame_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No saved frames in {lingbot_dir}")
    try:
        masks = build_sam_index(Path(sam_dir), video_name, prompt)
    except FileNotFoundError:
        masks = {}

    support = np.zeros(len(points), dtype=bool)
    frame_ids = np.empty(len(points), dtype=np.int64) if include_frame_ids else None
    offset = 0
    for path in paths:
        frame_id = frame_number(path)
        if frame_id % frame_step:
            continue
        with np.load(path, allow_pickle=False) as data:
            depth = load_depth_map(data["depth"], "depth")
            confidence = load_depth_map(data["depth_conf"], "depth_conf")
            intrinsic = load_intrinsic(data["intrinsic"])
            extrinsic = load_extrinsic(data["extrinsic"])
        if confidence.shape != depth.shape:
            raise ValueError(f"Depth/confidence shapes differ in {path}")
        world = depth_to_world_points(depth, intrinsic, extrinsic)
        valid = (np.isfinite(world).all(axis=-1) & np.isfinite(depth)
                 & (depth > 0) & np.isfinite(confidence))
        if max_depth is not None:
            valid &= depth < max_depth
        if min_depth_conf is not None:
            valid &= confidence > min_depth_conf
        sampled = valid[::pixel_step, ::pixel_step]
        chunk = world[::pixel_step, ::pixel_step][sampled].astype(np.float32)
        end = offset + len(chunk)
        if end > len(points) or not np.array_equal(points[offset:end], chunk):
            raise ValueError(f"Cloud rows do not match {path.name}; use the original "
                             "cloud and its sampling/filter settings")
        if frame_ids is not None:
            frame_ids[offset:end] = frame_id
        if len(chunk):
            mask_path = masks.get(frame_id)
            if mask_path is not None:
                mask = preprocess_mask(load_sam_mask(mask_path),
                                       target_shape=depth.shape, patch_size=patch_size)
                support[offset:end] = mask[::pixel_step, ::pixel_step][sampled]
        offset = end
    if offset != len(points):
        raise ValueError("Cloud row count differs from saved frame observations")
    evidence = {"support_mask": support}
    if frame_ids is not None:
        evidence["frame_ids"] = frame_ids
    return evidence
