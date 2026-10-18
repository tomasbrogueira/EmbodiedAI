"""Find the first supported reference window using the existing plane checks."""

from collections import deque
from pathlib import Path

import numpy as np

SCAN_VERSION = 1


def scan_reference(frames):
    from path_planner.floor_reference import estimate_floor_reference

    window = deque(maxlen=5)
    inspected = checked = 0
    normal, report = None, None
    for frame in frames:
        window.append(frame)
        inspected += 1
        if len(window) < 5:
            continue
        normal, report = estimate_floor_reference(list(window))
        checked += 1
        if normal is not None:
            break
    if report is None:
        normal, report = estimate_floor_reference(list(window))
        checked += 1
    report = dict(report, reference_scan_version=SCAN_VERSION, inspected_frame_count=inspected,
                  windows_checked=checked,
                  selected_window=[frame["id"] for frame in window] if normal is not None else None)
    if normal is not None:
        spacing = np.median([frame["spacing"] for frame in report["frames"] if frame["reason"] == "accepted"])
        report["slope_config"] = dict(radius=float(20 * spacing), neighbors=128, min_samples=6)
    return normal, report


def reference_from_video(lingbot_dir, sam_dir, video_name, prompt="floor", *,
                         max_depth=25., min_depth_conf=2.5, patch_size=14):
    from io_utils import frame_number, load_depth_map, load_extrinsic, load_intrinsic
    from lingbot_geometry import depth_to_world_points
    from sam_masks import build_sam_index, load_sam_mask, preprocess_mask

    try:
        masks = build_sam_index(Path(sam_dir), video_name, prompt)
    except FileNotFoundError:
        masks = {}
    if not masks:
        return None, dict(reason="missing_floor_masks", frames=[], max_disagreement_degrees=None,
                          reference_scan_version=SCAN_VERSION, inspected_frame_count=0,
                          windows_checked=0, selected_window=None)

    def frames():
        for path in sorted(Path(lingbot_dir).glob("frame_*.npz"), key=frame_number):
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
                floor_mask = preprocess_mask(load_sam_mask(mask_path), target_shape=depth.shape, patch_size=patch_size)
            yield dict(id=path.name, points=points, floor_mask=valid & floor_mask,
                       camera=-extrinsic[:, :3].T @ extrinsic[:, 3])

    return scan_reference(frames())
