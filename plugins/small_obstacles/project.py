"""Color the existing LingBot geometry using the plugin's binary masks."""

import json

import numpy as np

from orchestrator.io_utils import load_depth_map, load_extrinsic, load_intrinsic, load_rgb
from orchestrator.pointcloud import save_binary_ply

from .common import rgb_digest


def project(args, unproject=None):
    if unproject is None:
        from orchestrator.lingbot_geometry import depth_to_world_points
        unproject = depth_to_world_points
    manifest = json.loads((args.masks_dir / "manifest.json").read_text())
    if manifest["schema_version"] != 1 or not manifest["frames"]:
        raise ValueError("Unsupported or empty mask manifest")
    all_points, all_colors, total_marked = [], [], 0
    for name in manifest["frames"]:
        # Manifest filenames must stay inside the selected input directories.
        if "/" in name or "\\" in name or not name.startswith("frame_") or not name.endswith(".npz"):
            raise ValueError(f"Invalid frame name: {name}")
        with np.load(args.lingbot_dir / name, allow_pickle=False) as data:
            depth = load_depth_map(data["depth"])
            confidence = load_depth_map(data["depth_conf"], "depth confidence")
            rgb = load_rgb(data["images"])
            intrinsic = load_intrinsic(data["intrinsic"])
            extrinsic = load_extrinsic(data["extrinsic"])
        with np.load(args.masks_dir / name, allow_pickle=False) as data:
            mask = np.asarray(data["out_binary_masks"], dtype=bool)
            if str(data["rgb_sha256"].item()) != rgb_digest(rgb):
                raise ValueError(f"{name}: mask was generated from different RGB predictions")
        if mask.shape != depth.shape or confidence.shape != depth.shape or rgb.shape[:2] != depth.shape:
            raise ValueError(f"{name}: mask, RGB, confidence and depth must align exactly")
        points = np.asarray(unproject(depth, intrinsic, extrinsic), dtype=np.float32)
        if points.shape != (*depth.shape, 3):
            raise ValueError(f"Unexpected world point shape: {points.shape}")
        step = args.pixel_step
        points, rgb, mask = points[::step, ::step], rgb[::step, ::step].copy(), mask[::step, ::step]
        depth, confidence = depth[::step, ::step], confidence[::step, ::step]
        valid = np.isfinite(points).all(axis=-1) & np.isfinite(depth) & (depth > 0) & np.isfinite(confidence)
        if args.min_depth_conf is not None:
            valid &= confidence > args.min_depth_conf
        if args.max_depth is not None:
            valid &= depth < args.max_depth
        rgb[mask] = args.color
        marked = int((valid & mask).sum())
        total_marked += marked
        all_points.append(points[valid])
        all_colors.append(rgb[valid])
        print(f"[Projection] {name}: {marked} marked points", flush=True)
    points, colors = np.concatenate(all_points), np.concatenate(all_colors)
    if not len(points):
        raise ValueError("No valid geometry survives the depth/confidence filters")
    save_binary_ply(args.output, points, colors)
    print(f"Saved {args.output}: {len(points)} points, {total_marked} obstacle points", flush=True)
