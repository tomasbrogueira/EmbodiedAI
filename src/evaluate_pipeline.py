"""Evaluate a complete semantic_mapping_v1 run against optional references."""
from __future__ import annotations

import argparse
import json

from pipeline_common.evaluation import evaluate


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--reference")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        report = evaluate(args.run, args.reference, args.output)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f"Evaluation blocked: {error}\n")
    print(json.dumps({"status": report["status"], "pipeline": report["pipeline"], "reference": report["reference"]}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
