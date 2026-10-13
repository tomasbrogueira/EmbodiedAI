"""Standalone steepness, costs and certified routes from saved artifacts."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest='stage', required=True)
    for name in ('steepness', 'costs'):
        stage = stages.add_parser(name)
        stage.add_argument('--snapshot', type=Path, required=True)
        stage.add_argument('--output', type=Path, required=True)
        stage.add_argument('--reference', type=float, nargs=3)
        stage.add_argument('--radius', type=float)
        stage.add_argument('--reference-source', choices=['base_support', 'traversability'], default='base_support')
        if name == 'costs':
            stage.add_argument('--slope-policy', choices=['enforce', 'disabled'], default='enforce')
            stage.add_argument('--slopes', type=Path)
            stage.add_argument('--clearance-height', type=float)
            stage.add_argument('--calibration-source', choices=['base_support', 'traversability'], default='base_support')
    stage = stages.add_parser('route')
    stage.add_argument('--costs', type=Path, required=True)
    stage.add_argument('--output', type=Path, required=True)
    stage.add_argument('--start', type=int)
    stage.add_argument('--goal', type=int)
    stage.add_argument('--library', type=Path)
    args = vars(parser.parse_args())
    name = args.pop('stage')
    from orchestrator.analysis import steepness, costs, route
    function = dict(steepness=steepness, costs=costs, route=route)[name]
    if name == 'route':
        args['cost_directory'] = args.pop('costs')
    result = function(**args)
    print(json.dumps(result['metadata'], indent=2))


if __name__ == '__main__':
    main()
