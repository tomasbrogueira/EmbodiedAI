#!/usr/bin/env python3
"""Explicit local JupyterLab launch with localhost binding and existing authentication."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "traversability-environment.json"


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--env-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_launch_plan(args: argparse.Namespace) -> dict:
    """Prepare a local-only launch without starting the notebook server."""
    repository = args.repo_root.expanduser().resolve()
    expanded = args.env_dir.expanduser()
    env_dir = (expanded if expanded.is_absolute() else repository / expanded).resolve()
    if not 1 <= args.port <= 65535:
        raise ValueError("Port must be in 1..65535")
    python = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    manifest_path = env_dir / MANIFEST_NAME
    environment = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if Path(manifest.get("env_dir", "")).resolve() != env_dir:
            raise ValueError("Setup manifest belongs to another environment")
        if not manifest.get("notebooks") or manifest.get("status") != "installed":
            raise ValueError("Rerun setup with --notebooks before launching JupyterLab")
        environment = manifest["environment"]
    elif not args.dry_run:
        raise ValueError("Run scripts/setup_env.py --notebooks for this environment before launching")
    if not args.dry_run and (not python.is_file() or not (env_dir / "pyvenv.cfg").is_file()):
        raise ValueError("Notebook Python is missing; rerun isolated environment setup")
    return {
        "argv": [str(python), "-m", "jupyterlab", "--ServerApp.ip=127.0.0.1",
                 "--ServerApp.open_browser=False", f"--ServerApp.port={args.port}", "--ServerApp.port_retries=0",
                 f"--ServerApp.root_dir={repository}"],
        "cwd": str(repository),
        "environment": environment,
    }


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    try:
        plan = build_launch_plan(args)
        if args.dry_run:
            print(json.dumps(plan, indent=2))
            return 0
        environment = dict(os.environ)
        for key in ("PIP_TARGET", "PIP_PREFIX", "PIP_USER", "PYTHONHOME", "PYTHONPATH"):
            environment.pop(key, None)
        environment.update(plan["environment"])
        return subprocess.call(plan["argv"], cwd=plan["cwd"], env=environment)
    except (ValueError, OSError) as error:
        print(f"Notebook launch failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
