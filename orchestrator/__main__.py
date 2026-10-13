"""Compose independent stages, or project existing geometry and semantic masks."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest='stage', required=True)
    run = stages.add_parser('run')
    run.add_argument('--config', type=Path, required=True)
    run.add_argument('--check', action='store_true', help='Validate stage dependencies without creating output or loading models')
    project = stages.add_parser('project')
    project.add_argument('--predictions', type=Path, required=True)
    project.add_argument('--output', type=Path, required=True)
    project.add_argument('--support-dir', type=Path)
    project.add_argument('--semantic-dir', type=Path)
    project.add_argument('--video-name')
    project.add_argument('--support-prompt', default='floor')
    project.add_argument('--frame-step', type=int, default=2)
    project.add_argument('--pixel-step', type=int, default=1)
    project.add_argument('--source-fps', type=float, required=True)
    args = parser.parse_args()
    if args.stage == 'run':
        config = json.loads(args.config.read_text())
        from orchestrator.config import resolve
        if args.check:
            print(json.dumps(resolve(config), indent=2))
        else:
            from orchestrator.pipeline import run_pipeline
            print(json.dumps(run_pipeline(config), indent=2))
    else:
        from orchestrator.projection import build_snapshot
        values = vars(args)
        values.pop('stage')
        print(json.dumps(build_snapshot(**values)['metadata'], indent=2))


if __name__ == '__main__':
    main()
