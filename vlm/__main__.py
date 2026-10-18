"""Standalone VLM discovery; never starts geometry or segmentation."""
import argparse
from pathlib import Path
from vlm.propose import DEFAULT_MODEL, propose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['propose'])
    parser.add_argument('--lingbot-dir', type=Path, required=True, help='Saved RGB predictions; depth is not required')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target', choices=['traversable', 'non_traversable'], default='non_traversable')
    parser.add_argument('--instruction-file', type=Path)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--keyframe-step', type=int, default=30)
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--ground-crop', action='store_true')
    parser.add_argument('--robot', default='a small wheeled ground robot')
    args = parser.parse_args()
    if args.keyframe_step < 1 or args.max_new_tokens < 1 or (args.max_frames is not None and args.max_frames < 1):
        parser.error('Frame/token limits must be positive')
    args.instruction = args.instruction_file.read_text() if args.instruction_file else None
    propose(args)


if __name__ == '__main__':
    main()
