"""Read existing predictions; callers need PYTHONPATH=path-planning:orchestrator."""


def reference_from_video(lingbot_dir, sam_dir, video_name, prompt="floor", *,
                         max_depth=25.0, min_depth_conf=2.5, patch_size=14):
    """Return an automatic floor normal and evidence, or None when unavailable."""
    from pathlib import Path

    import numpy as np

    from floor_reference import estimate_floor_reference
    from io_utils import frame_number, load_depth_map, load_extrinsic, load_intrinsic
    from sam_masks import build_sam_index, load_sam_mask, preprocess_mask

    try:
        masks = build_sam_index(Path(sam_dir), video_name, prompt)
    except FileNotFoundError:
        masks = {}
    if not masks:
        return None, {"reason": "missing_floor_masks", "frames": [],
                      "max_disagreement_degrees": None}

    from lingbot_geometry import depth_to_world_points

    frames = []
    paths = sorted(Path(lingbot_dir).glob("frame_*.npz"), key=frame_number)[:5]
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            depth = load_depth_map(data["depth"], "depth")
            confidence = load_depth_map(data["depth_conf"], "depth_conf")
            intrinsic = load_intrinsic(data["intrinsic"])
            extrinsic = load_extrinsic(data["extrinsic"])
        if confidence.shape != depth.shape:
            raise ValueError(f"Depth/confidence shapes differ in {path}")
        points = depth_to_world_points(depth, intrinsic, extrinsic)
        valid = (np.isfinite(points).all(axis=-1) & np.isfinite(depth)
                 & (depth > 0) & (depth < max_depth)
                 & np.isfinite(confidence) & (confidence > min_depth_conf))
        floor_mask = np.zeros(depth.shape, dtype=bool)
        mask_path = masks.get(frame_number(path))
        if mask_path is not None:
            floor_mask = preprocess_mask(load_sam_mask(mask_path),
                                         target_shape=depth.shape, patch_size=patch_size)
        frames.append({"id": path.name, "points": points,
                       "floor_mask": valid & floor_mask,
                       "camera": -extrinsic[:, :3].T @ extrinsic[:, 3]})
    normal, report = estimate_floor_reference(frames)
    if normal is not None:
        spacing = np.median([f["spacing"] for f in report["frames"]
                             if f["reason"] == "accepted"])
        report["slope_config"] = {"radius": float(20 * spacing),
                                  "neighbors": 128, "min_samples": 6}
    return normal, report
