"""Color the existing LingBot geometry using the plugin's binary masks."""

import json

import numpy as np

from orchestrator.io_utils import load_depth_map, load_extrinsic, load_intrinsic, load_rgb
from orchestrator.pointcloud import save_binary_ply

from orchestrator.shared import rgb_digest


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
            if str(data["rgb_sha256"].item()) != rgb_digest(rgb):
                raise ValueError(f"{name}: mask was generated from different RGB predictions")
            from orchestrator.semantics import CHANNELS
            typed = {key: data[key].copy() for key in CHANNELS if key in data}
            if not typed:
                typed = {'vlm_non_traversability_mask': np.asarray(data['out_binary_masks'], dtype=bool)}
        from orchestrator.projection import frame_arrays
        from orchestrator.semantics import colors as semantic_colors
        rows, _, _ = frame_arrays(dict(depth=depth, depth_conf=confidence, images=rgb,
            intrinsic=intrinsic, extrinsic=extrinsic), typed, pixel_step=args.pixel_step,
            min_depth_conf=args.min_depth_conf, max_depth=args.max_depth, unproject=unproject)
        points = rows['points']
        colors = semantic_colors(rows['source_rgb'], rows['support_mask'], rows['obstacle_mask'])
        colors[rows['obstacle_mask']] = args.color
        marked = int(rows['obstacle_mask'].sum())
        total_marked += marked
        all_points.append(points)
        all_colors.append(colors)
        print(f"[Projection] {name}: {marked} marked points", flush=True)
    points, colors = np.concatenate(all_points), np.concatenate(all_colors)
    if not len(points):
        raise ValueError("No valid geometry survives the depth/confidence filters")
    save_binary_ply(args.output, points, colors)
    print(f"Saved {args.output}: {len(points)} points, {total_marked} obstacle points", flush=True)
