"""Explicit CPU evaluation/export entry point."""

import argparse
import json
from pathlib import Path

from . import evaluate, export_report
from .artifacts import read_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="Opt in to JSON/CSV/JSONL export")
    args = parser.parse_args(argv)
    try:
        report = evaluate(read_json(args.config))
        if args.output_dir is not None:
            paths = export_report(report, args.output_dir)
            print(json.dumps({"exports": paths}, indent=2))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({"fixture": report["fixture"], "comparison": report["comparison"],
                      "validation": report["validation"], "coverage": report["coverage"]}, indent=2, allow_nan=False))
    return 2 if report["validation"]["errors"] or not report["comparison"]["discovery_complete"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
