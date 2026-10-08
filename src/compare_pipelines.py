"""Compare semantic_mapping_v1 evaluations with strict experiment gates."""
from __future__ import annotations

import argparse
import json

from pipeline_common.evaluation import compare


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        report = compare(args.evaluation, args.output)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f"Comparison blocked: {error}\n")
    print(json.dumps({"status": report["status"], "differences": report["differences"]}, allow_nan=False))
    return 0 if report["compatible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
