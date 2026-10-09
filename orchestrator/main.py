"""CLI entry point for the SAM3 + LingBot-MAP orchestrator."""

import argparse
from pathlib import Path

import numpy as np

from semantic_map import build_semantic_point_cloud


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_optional_float(value):
    """Accept a numeric value or 'none' to disable a filter."""
    if value.strip().lower() in {"none", "null", "off"}:
        return None

    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected a number or 'none', got {value!r}"
        ) from exc


def parse_rgb(value):
    """Parse RGB as '255,0,0' or hexadecimal '#FF0000'."""
    try:
        if value.startswith("#"):
            if len(value) != 7:
                raise ValueError("Expected hexadecimal format #RRGGBB")

            channels = [
                int(value[i:i + 2], 16)
                for i in (1, 3, 5)
            ]
        else:
            channels = [
                int(channel.strip())
                for channel in value.split(",")
            ]

            if len(channels) != 3:
                raise ValueError("Expected three RGB values")

        if any(c < 0 or c > 255 for c in channels):
            raise ValueError("RGB channels must be between 0 and 255")

        return np.asarray(channels, dtype=np.uint8)

    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Combine existing LingBot-MAP 3D predictions "
            "with SAM3 segmentation masks."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ------------------------------------------------------------
    # Required input paths
    # ------------------------------------------------------------

    parser.add_argument(
        "--video-path",
        type=Path,
        required=True,
        help="Input video path, used to identify the video name.",
    )

    parser.add_argument(
        "--lingbot-dir",
        type=Path,
        required=True,
        help="Exact directory containing LingBot-MAP frame_*.npz files.",
    )

    parser.add_argument(
        "--sam-dir",
        type=Path,
        required=True,
        help="Exact directory containing the SAM3 mask NPZ files.",
    )

    # ------------------------------------------------------------
    # Semantic configuration
    # ------------------------------------------------------------

    parser.add_argument(
        "--prompt",
        default="floor",
        help="SAM3 prompt used to generate the masks.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PLY path. Defaults to orchestrator/<video>_<prompt>_semantic_map.ply.",
    )

    # ------------------------------------------------------------
    # Sampling and geometry filters
    # ------------------------------------------------------------

    parser.add_argument(
        "--frame-step",
        type=int,
        default=2,
        help="Process every Nth LingBot frame.",
    )

    parser.add_argument(
        "--pixel-step",
        type=int,
        default=1,
        help="Sample every Nth pixel in both image dimensions.",
    )

    parser.add_argument(
        "--min-depth-conf",
        type=parse_optional_float,
        default=2.5,
        help="Minimum depth confidence; use 'none' to disable.",
    )

    parser.add_argument(
        "--max-depth",
        type=parse_optional_float,
        default=25.0,
        help="Maximum depth; use 'none' to disable.",
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        default=14,
        help="Patch size used by the existing mask-alignment procedure.",
    )

    parser.add_argument(
        "--segment-color",
        type=parse_rgb,
        default=np.array([255, 0, 0], dtype=np.uint8),
        help="Segmented-point RGB color, e.g. '255,0,0' or '#FF0000'.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Validate arguments
    # ------------------------------------------------------------

    if args.frame_step <= 0:
        parser.error("--frame-step must be positive")

    if args.pixel_step <= 0:
        parser.error("--pixel-step must be positive")

    if args.patch_size <= 0:
        parser.error("--patch-size must be positive")

    if args.min_depth_conf is not None and args.min_depth_conf < 0:
        parser.error("--min-depth-conf must be non-negative or none")

    if args.max_depth is not None and args.max_depth <= 0:
        parser.error("--max-depth must be positive or none")

    # Resolve input paths relative to the working directory.
    args.video_path = args.video_path.expanduser().resolve()
    args.lingbot_dir = args.lingbot_dir.expanduser().resolve()
    args.sam_dir = args.sam_dir.expanduser().resolve()

    if not args.video_path.is_file():
        parser.error(f"Video does not exist: {args.video_path}")

    if not args.lingbot_dir.is_dir():
        parser.error(
            f"LingBot directory does not exist: {args.lingbot_dir}"
        )

    if not args.sam_dir.is_dir():
        parser.error(f"SAM3 directory does not exist: {args.sam_dir}")

    # Derive the shared video identifier from the input video.
    args.video_name = args.video_path.stem

    # Resolve the output path.
    if args.output is None:
        args.out_file = (
            PROJECT_ROOT
            / "orchestrator"
            / f"{args.video_name}_{args.prompt}_semantic_map.ply"
        )
    else:
        args.out_file = args.output.expanduser().resolve()

    return args


def main():
    args = parse_args()

    print(f"Input video:  {args.video_path}")
    print(f"Video name:   {args.video_name}")
    print(f"LingBot data: {args.lingbot_dir}")
    print(f"SAM3 masks:   {args.sam_dir}")
    print(f"Output PLY:   {args.out_file}")

    build_semantic_point_cloud(args)


if __name__ == "__main__":
    main()