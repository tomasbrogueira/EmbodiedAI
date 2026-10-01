"""Private fresh-process worker; not a legacy region profiling entry point."""

import json
import sys


def main():
    from .profiling import _execute, _prepare
    config = json.load(sys.stdin)
    if config.get("fixture") is not False or config.get("enable_real_run") is not True:
        raise ValueError("Worker requires explicitly authorized genuine execution")
    if sys.version_info < (3, 12):
        raise RuntimeError("SAM3 hazard profiling requires Python 3.12+")
    result = _execute(_prepare(config))
    print(json.dumps({"output_dir": result["output_dir"], "complete": result["summary"]["complete"]}))


if __name__ == "__main__":
    main()
