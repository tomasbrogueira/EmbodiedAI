"""Explicit validation/evaluation entry point; no model loading or downloads."""

import argparse
import json
from pathlib import Path

from . import evaluate, export_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--export", action="store_true", help="Save tables/gallery under the run's evaluation directory")
    parser.add_argument("--plots", action="store_true", help="Export optional confusion plots (requires matplotlib)")
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        report = evaluate(config)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({"coverage": report["coverage"], "comparison": report["comparison"],
                      "validation": report["validation"]}, indent=2, allow_nan=False))
    if args.export:
        export_report(report, plots=args.plots)
    blocked = any(row["status"] in ("invalid", "incomplete") for row in report["comparison"])
    return 2 if report["validation"]["errors"] or blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
