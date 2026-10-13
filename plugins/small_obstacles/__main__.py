"""Independent CLI stages support the project's separate model environments."""

import argparse
import math
from pathlib import Path

from .propose import DEFAULT_MODEL


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("Must be positive")
    return result


def optional_float(value):
    if value.lower() in {"none", "off"}:
        return None
    result = float(value)
    if result < 0 or not math.isfinite(result):
        raise argparse.ArgumentTypeError("Must be finite and non-negative, or none")
    return result


def color(value):
    try:
        channels = [int(c) for c in value.split(",")]
        if len(channels) != 3 or any(c < 0 or c > 255 for c in channels):
            raise ValueError
        return channels
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected RGB such as 255,0,0") from exc


def main():
    parser = argparse.ArgumentParser(description="Optional VLM → SAM3 union mask → LingBot map plugin")
    stages = parser.add_subparsers(dest="stage", required=True)
    propose = stages.add_parser("propose", help="Discover small-obstacle text prompts using Qwen3-VL")
    segment = stages.add_parser("segment", help="Combine all SAM3 prompts into one mask per frame")
    project = stages.add_parser("project", help="Mark the union mask on LingBot's 3D geometry")
    for stage in (propose, segment, project):
        stage.add_argument("--lingbot-dir", type=Path, required=True)
    for stage in (propose, segment):
        stage.add_argument("--max-frames", type=positive_int, help="Limit this stage for a smoke run")
    propose.add_argument("--output", type=Path, required=True)
    propose.add_argument("--model", default=DEFAULT_MODEL, help="HF model ID or local Qwen3-VL checkpoint")
    propose.add_argument("--device", default="auto", help="Transformers device_map: auto, cpu, cuda:0, …")
    propose.add_argument("--keyframe-step", type=positive_int, default=30, help="Sample every Nth saved prediction")
    propose.add_argument("--max-new-tokens", type=positive_int, default=256)
    propose.add_argument("--ground-crop", action="store_true", help="Also inspect a lower-half crop")
    propose.add_argument("--robot", default="a small wheeled ground robot")
    propose.add_argument("--target", choices=['traversable', 'non_traversable'], default='non_traversable')
    segment.add_argument("--prompts", type=Path, help="VLM output or a JSON list of manual prompts")
    segment.add_argument("--target", choices=['traversable', 'non_traversable'])
    segment.add_argument("--extra-traversability-prompts", nargs='+')
    segment.add_argument("--output-dir", type=Path, required=True)
    segment.add_argument("--device", default="cuda")
    segment.add_argument("--checkpoint", help="Local SAM3 checkpoint; otherwise use Hugging Face")
    segment.add_argument("--sam-threshold", type=float, default=0.5)
    segment.add_argument("--frame-step", type=positive_int, default=2, help="Segment every Nth saved prediction")
    project.add_argument("--masks-dir", type=Path, required=True)
    project.add_argument("--output", type=Path, required=True)
    project.add_argument("--pixel-step", type=positive_int, default=1)
    project.add_argument("--min-depth-conf", type=optional_float, default=2.5)
    project.add_argument("--max-depth", type=optional_float, default=25.0)
    project.add_argument("--color", type=color, default=[255, 0, 0])
    args = parser.parse_args()
    if args.stage == "propose":
        from .propose import propose as execute
    elif args.stage == "segment":
        if not 0 <= args.sam_threshold <= 1:
            parser.error("--sam-threshold must lie between 0 and 1")
        from .segment import segment as execute
    else:
        from .project import project as execute
    execute(args)


if __name__ == "__main__":
    main()
