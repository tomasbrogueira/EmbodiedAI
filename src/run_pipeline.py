"""Run one of exactly four pretrained mapping architectures."""
import argparse
import sys
from pipeline_common.contracts import PIPELINE_IDS
from pipeline_common.runtime import new_attempt_path,run

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline",required=True,choices=PIPELINE_IDS)
    parser.add_argument("--sequence",required=True);parser.add_argument("--config",required=True)
    outputs=parser.add_mutually_exclusive_group(required=True)
    outputs.add_argument("--output",help="Exact new attempt directory; existing paths are refused")
    outputs.add_argument("--output-root",help="Parent directory for a fresh UTC/pipeline/UUID attempt")
    parser.add_argument("--fixture",action="store_true");parser.add_argument("--geometry-cache")
    args=parser.parse_args(argv)
    output=args.output or new_attempt_path(args.output_root,args.pipeline)
    try:
        result=run(args.pipeline,args.sequence,args.config,output,fixture=args.fixture,cache=args.geometry_cache)
        print(f"{result['pipeline_id']}: {result['status']} ({output})")
        return 0 if result["status"]=="complete" else 2
    except Exception as error:
        print(f"Pipeline failed: {error}",file=sys.stderr);return 1
if __name__=="__main__": raise SystemExit(main())
