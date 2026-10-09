"""Orchestration for combining LingBot-MAP geometry with SAM3 semantics."""

import numpy as np

from io_utils import (
    frame_number,
    load_depth_map,
    load_extrinsic,
    load_intrinsic,
    load_rgb,
)
from lingbot_geometry import depth_to_world_points
from pointcloud import save_binary_ply
from sam_masks import (
    build_sam_index,
    load_sam_mask,
    preprocess_mask,
)


def build_semantic_point_cloud(config):
    """Combine LingBot-MAP 3D geometry with SAM3 segmentation masks.

    Background points retain their original RGB colors.
    Segmented points receive config.segment_color.

    Frame observations are concatenated without voxel fusion,
    point averaging, or additional pose transformations.
    """

    # ------------------------------------------------------------
    # 1. Validate configuration and input directories.
    # ------------------------------------------------------------

    if config.frame_step <= 0:
        raise ValueError(
            f"frame_step must be positive, got {config.frame_step}"
        )

    if config.pixel_step <= 0:
        raise ValueError(
            f"pixel_step must be positive, got {config.pixel_step}"
        )

    if not config.lingbot_dir.is_dir():
        raise FileNotFoundError(
            f"LingBot prediction directory not found:\n"
            f"{config.lingbot_dir}"
        )

    lingbot_files = sorted(
        config.lingbot_dir.glob("frame_*.npz")
    )

    if not lingbot_files:
        raise RuntimeError(
            f"No LingBot frame files found in:\n"
            f"{config.lingbot_dir}"
        )

    # Index the SAM3 masks using the video name and prompt.
    sam_by_frame = build_sam_index(
        config.sam_dir,
        config.video_name,
        config.prompt,
    )

    # Select frames according to the requested temporal sampling.
    sampled_files = [
        path
        for path in lingbot_files
        if frame_number(path) % config.frame_step == 0
    ]

    # ------------------------------------------------------------
    # 2. Print configuration and processing information.
    # ------------------------------------------------------------

    print("=" * 70)
    print("LINGBOT-MAP + SAM3 SEMANTIC POINT CLOUD")
    print("=" * 70)
    print(f"Input video: {config.video_path}")
    print(f"Video name: {config.video_name}")
    print(f"SAM3 prompt: {config.prompt}")
    print(f"LingBot directory: {config.lingbot_dir}")
    print(f"SAM3 mask directory: {config.sam_dir}")
    print(f"Available LingBot frames: {len(lingbot_files)}")
    print(f"Available SAM3 masks: {len(sam_by_frame)}")
    print(f"Frames sampled: {len(sampled_files)}")
    print(f"Frame step: {config.frame_step}")
    print(f"Pixel step: {config.pixel_step}")
    print(f"Minimum depth confidence: {config.min_depth_conf}")
    print(f"Maximum depth: {config.max_depth}")
    print(f"Segmentation color (RGB): {config.segment_color.tolist()}")
    print(f"Mask patch size: {config.patch_size}")
    print("Unprojection: official LingBot-MAP helper")
    print("Voxel fusion: disabled")
    print(f"Output: {config.out_file}")
    print("=" * 70)

    all_points = []
    all_colors = []

    total_points = 0
    total_segmented = 0
    frames_processed = 0

    # ------------------------------------------------------------
    # 3. Process each sampled frame.
    # ------------------------------------------------------------

    for lingbot_path in sampled_files:
        frame_idx = frame_number(lingbot_path)

        sam_path = sam_by_frame.get(frame_idx)

        if sam_path is None:
            raise FileNotFoundError(
                f"No matching SAM3 mask for frame {frame_idx}. "
                f"Expected a filename such as "
                f"{config.video_name}_{config.prompt}_frame"
                f"{frame_idx:06d}.npz under {config.sam_dir}."
            )

        # --------------------------------------------------------
        # Load LingBot-MAP frame prediction.
        # --------------------------------------------------------

        with np.load(lingbot_path) as data:
            required_fields = {
                "depth",
                "depth_conf",
                "intrinsic",
                "extrinsic",
                "images",
            }

            missing_fields = required_fields.difference(data.files)

            if missing_fields:
                raise KeyError(
                    f"Missing fields in {lingbot_path}: "
                    f"{sorted(missing_fields)}. "
                    f"Available fields: {data.files}"
                )

            depth = load_depth_map(
                data["depth"],
                "depth",
            )

            depth_conf = load_depth_map(
                data["depth_conf"],
                "depth_conf",
            )

            intrinsic = load_intrinsic(
                data["intrinsic"]
            )

            extrinsic = load_extrinsic(
                data["extrinsic"]
            )

            rgb = load_rgb(
                data["images"]
            )

        if depth.shape != depth_conf.shape:
            raise ValueError(
                f"Frame {frame_idx}: depth shape {depth.shape} does not "
                f"match confidence shape {depth_conf.shape}."
            )

        if depth.shape != rgb.shape[:2]:
            raise ValueError(
                f"Frame {frame_idx}: depth shape {depth.shape} does not "
                f"match RGB shape {rgb.shape[:2]}."
            )

        # --------------------------------------------------------
        # Load and align the corresponding SAM3 mask.
        # --------------------------------------------------------

        sam_mask_original = load_sam_mask(sam_path)

        sam_mask = preprocess_mask(
            sam_mask_original,
            target_shape=depth.shape,
            patch_size=config.patch_size,
        )

        # --------------------------------------------------------
        # Project depth using the official LingBot-MAP geometry.
        # Do not introduce additional pose transformations.
        # --------------------------------------------------------

        points_map = depth_to_world_points(
            depth,
            intrinsic,
            extrinsic,
        )

        # --------------------------------------------------------
        # Apply the same pixel sampling to every aligned array.
        # --------------------------------------------------------

        step = config.pixel_step

        points_map = points_map[::step, ::step]
        depth_sampled = depth[::step, ::step]
        confidence_sampled = depth_conf[::step, ::step]
        rgb_sampled = rgb[::step, ::step]
        mask_sampled = sam_mask[::step, ::step]

        # --------------------------------------------------------
        # Filter invalid geometry and apply optional thresholds.
        # --------------------------------------------------------

        valid = (
            np.isfinite(points_map).all(axis=-1)
            & np.isfinite(depth_sampled)
            & (depth_sampled > 0)
            & np.isfinite(confidence_sampled)
        )

        if config.max_depth is not None:
            valid &= depth_sampled < config.max_depth

        if config.min_depth_conf is not None:
            valid &= confidence_sampled > config.min_depth_conf

        points = points_map[valid]
        colors = rgb_sampled[valid].copy()
        labels = mask_sampled[valid]

        if len(points) == 0:
            print(
                f"[Frame {frame_idx}] No valid points; skipping."
            )
            continue

        # --------------------------------------------------------
        # Apply semantic coloring.
        #
        # Background: original RGB.
        # Segmented points: configured semantic color.
        # --------------------------------------------------------

        colors[labels] = config.segment_color

        all_points.append(
            points.astype(np.float32)
        )

        all_colors.append(
            colors.astype(np.uint8)
        )

        frame_point_count = len(points)
        frame_segmented_count = int(labels.sum())

        frames_processed += 1
        total_points += frame_point_count
        total_segmented += frame_segmented_count

        print(
            f"[Frame {frame_idx:04d}] "
            f"points={frame_point_count:,}, "
            f"segmented={frame_segmented_count:,}, "
            f"mask={sam_path.name}"
        )

    # ------------------------------------------------------------
    # 4. Concatenate observations and export the point cloud.
    # ------------------------------------------------------------

    if not all_points:
        raise RuntimeError(
            "No valid points were collected. Check the input paths, "
            "SAM3 mask filenames, sampling settings, and depth filters."
        )

    points = np.concatenate(
        all_points,
        axis=0,
    )

    colors = np.concatenate(
        all_colors,
        axis=0,
    )

    # Ensure the output directory exists.
    config.out_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_binary_ply(
        config.out_file,
        points,
        colors,
    )

    # ------------------------------------------------------------
    # 5. Print the final summary.
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("SEMANTIC POINT CLOUD COMPLETE")
    print("=" * 70)
    print(f"Frames processed: {frames_processed}")
    print(f"Total points: {total_points:,}")
    print(f"Segmented points: {total_segmented:,}")

    if total_points > 0:
        segmented_fraction = (
            100.0 * total_segmented / total_points
        )
        print(f"Segmented fraction: {segmented_fraction:.2f}%")

    print(f"Output: {config.out_file}")
    print(
        f"File size: "
        f"{config.out_file.stat().st_size / (1024 ** 2):.2f} MiB"
    )
    print("Geometry: official LingBot-MAP unprojection")
    print("Voxel fusion: disabled")
    print("=" * 70)