"""Prepare an honest, verified generic RGB sequence (no models)."""
import argparse
import sys
from pipeline_common.sequence import prepare
from pipeline_common.io import read_json

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True); parser.add_argument("--config",required=True); parser.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    try:
        result=prepare(args.input,read_json(args.config),args.output)
        print(f"Prepared {len(result['frames'])} frames: {args.output}"); return 0
    except Exception as error:
        print(f"Preparation failed: {error}",file=sys.stderr); return 1
if __name__=="__main__": raise SystemExit(main())
