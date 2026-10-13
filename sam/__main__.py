"""Standalone support-video and typed image-query segmentation."""
import argparse
from pathlib import Path
import sys


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'support':
        from sam.support import main as support
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        return support()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['segment'])
    parser.add_argument('--lingbot-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--prompts', type=Path)
    parser.add_argument('--target', choices=['traversable', 'non_traversable'])
    parser.add_argument('--extra-traversability-prompts', nargs='+')
    parser.add_argument('--manual-non-traversability-prompts', nargs='+')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--checkpoint')
    parser.add_argument('--sam-threshold', type=float, default=.5)
    parser.add_argument('--frame-step', type=int, default=2)
    parser.add_argument('--max-frames', type=int)
    args = parser.parse_args()
    if not args.prompts and not args.extra_traversability_prompts and not args.manual_non_traversability_prompts:
        parser.error('Provide typed proposals or independent manual SAM queries')
    if args.frame_step < 1 or not 0 <= args.sam_threshold <= 1:
        parser.error('Use a positive frame step and threshold in [0, 1]')
    from sam.segment import segment
    segment(args)


if __name__ == '__main__':
    main()
