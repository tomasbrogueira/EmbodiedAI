#!/usr/bin/env python3
"""Execute safe hazard notebook copies; downloads and real models stay disabled."""

import argparse
from contextlib import contextmanager, redirect_stdout
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import sys

HAZARD_NOTEBOOKS = (
    "00_setup_and_checks.ipynb", "10_hazard_data.ipynb",
    "11_hazard_inference.ipynb", "12_hazard_evaluation.ipynb",
    "13_hazard_segmentation_and_cost.ipynb",
    "14_hazard_visualization.ipynb",
)
CONFIG_IDS = {
    "00_setup_and_checks.ipynb": "setup-config",
    "04_deployment_benchmark.ipynb": "benchmark-config",
    "10_hazard_data.ipynb": "hazard-data-config",
    "11_hazard_inference.ipynb": "hazard-configuration",
    "12_hazard_evaluation.ipynb": "hazard-evaluation-config",
    "13_hazard_segmentation_and_cost.ipynb": "configuration",
    "14_hazard_visualization.ipynb": "hazard-visualization-config",
}


@contextmanager
def _kernel_environment(values):
    previous = {key: os.environ.get(key) for key in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def safe_overrides(name, *, fixture=False):
    """Explicit switches applied after configuration and before action cells."""
    if name.startswith("00"):
        return {"CHECK_GPU": False}
    if name.startswith("04"):
        return {"RUN_BENCHMARK": False, "RUN_FAKE_BENCHMARK": fixture}
    if name.startswith("10"):
        return {"RUN_FIXTURE": fixture, "RUN_REAL_PREPARATION": False, "DOWNLOAD_COCO": False}
    if name.startswith("11"):
        return {"DOWNLOAD_WEIGHTS": False, "LOAD_MODEL": False, "RUN_INFERENCE": False, "RETRY_ERRORS": False}
    if name.startswith("12"):
        return {"EXPORT": False}
    if name.startswith("13"):
        return {"PREPARE_SELECTION": False, "RUN_FIXTURE": fixture,
                "RUN_SEGMENTATION": False, "RUN_PROFILES": False,
                "ALLOW_DOWNLOADS": False, "ENABLE_COMBINED": False}
    if name.startswith("14"):
        return {"SHOW_WIDGETS": False, "EXPORT": False}
    raise ValueError(f"No safe notebook configuration registered for {name}")


def prepare_notebook(notebook, name, *, fixture=False):
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if any(cell.get("outputs") or cell.get("execution_count") is not None for cell in code):
        raise ValueError(f"Source notebook {name} must have cleared outputs before checking")
    matches = [cell for cell in code if cell.get("id") == CONFIG_IDS[name]]
    if not matches and name.startswith(("00", "04")):
        matches = [cell for cell in code if cell.get("id", "").endswith("config")]
    if len(matches) != 1:
        raise ValueError(f"Expected one registered configuration cell in {name}")
    values = safe_overrides(name, fixture=fixture)
    matches[0].source += "\n# Offline checker: actions are deliberately constrained.\n" + "\n".join(
        f"{key} = {value!r}" for key, value in values.items()
    )
    return notebook


def _execute_python(notebook, name, output_root):
    """Headless code-cell mode for isolated CPU environments without a kernel."""
    namespace = {"__name__": "__main__"}
    output = io.StringIO()
    with redirect_stdout(output):
        for cell in notebook.cells:
            if cell.cell_type == "code":
                exec(compile(cell.source, f"{name}:{cell.id}", "exec"), namespace)
    (output_root / (name + ".log")).write_text(output.getvalue(), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True, help="New directory for checked copies and optional marked fixture artifacts")
    parser.add_argument("--with-fixture", action="store_true", help="Enable only synthetic CPU fixtures in notebooks 10/13 (and optional legacy 04)")
    parser.add_argument("--execution", choices=("kernel", "python"), default="kernel",
                        help="kernel uses setup's registered kernel; python executes code cells headlessly in this isolated interpreter")
    parser.add_argument("--include-legacy", action="store_true", help="Also check legacy benchmark 04 without enabling real inference")
    parser.add_argument("--visualization-run-root", type=Path, help="Explicit existing saved runs for notebook 14 only; never generates a replay")
    parser.add_argument("--visualization-data-root", type=Path, help="Explicit existing RGB/reference data for notebook 14 only")
    parser.add_argument("--visualization-run-name", default="hazard_prompt_v1", help="Exact saved run for notebook 14")
    args = parser.parse_args(argv)
    if bool(args.visualization_run_root) != bool(args.visualization_data_root):
        parser.error("Supply both visualization saved run and data roots")
    repo = Path(__file__).resolve().parents[1]
    env_dir = Path(sys.prefix).resolve()
    if sys.prefix == sys.base_prefix:
        parser.error("Run with an isolated environment's interpreter; global/shared environments are unsupported")
    manifest = {}
    if args.execution == "kernel":
        manifest_path = env_dir / "traversability-environment.json"
        if not manifest_path.is_file():
            parser.error("Kernel mode requires scripts/setup_env.py --notebooks; use --execution python for an existing isolated CPU environment")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("notebooks") or manifest.get("status") != "installed":
            parser.error("Complete setup with --notebooks before executing kernel checks")
        if Path(manifest.get("env_dir", "")).resolve() != env_dir:
            parser.error("Notebook manifest belongs to a different environment")
    try:
        import nbformat
        if args.execution == "kernel":
            from nbclient import NotebookClient
    except ImportError:
        parser.error("Notebook dependencies are missing; install requirements/notebooks.txt only in the selected isolated environment")
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    results = []
    cache = output_root / "cache"
    environment = dict(manifest.get("environment", {}),
                       TRAVERSABILITY_REPO_ROOT=str(repo),
                       TRAVERSABILITY_DATA_ROOT=str(output_root / "data"),
                       TRAVERSABILITY_RUN_ROOT=str(output_root / "runs"),
                       TRAVERSABILITY_CACHE_ROOT=str(cache),
                       HF_HOME=str(cache / "huggingface"), HF_HUB_CACHE=str(cache / "huggingface/hub"),
                       TORCH_HOME=str(cache / "torch"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    names = (*HAZARD_NOTEBOOKS, *(("04_deployment_benchmark.ipynb",) if args.include_legacy else ()))
    with _kernel_environment(environment):
        for name in names:
            notebook = nbformat.read(repo / "notebooks" / name, as_version=4)
            nbformat.validate(notebook)
            prepare_notebook(notebook, name, fixture=args.with_fixture)
            if name.startswith("14") and args.visualization_run_root:
                cell = next(c for c in notebook.cells if c.get("id") == CONFIG_IDS[name])
                cell.source += "\nconfig.update(" + repr({
                    "run_root": str(args.visualization_run_root.expanduser().resolve()),
                    "data_root": str(args.visualization_data_root.expanduser().resolve()),
                    "run_name": args.visualization_run_name,
                }) + ")\n"
            # Kernel specs can override the launching process's environment.
            # Apply sandbox roots inside the kernel before any notebook imports
            # or root discovery, so a relocated manifest cannot redirect writes.
            notebook.cells.insert(0, nbformat.v4.new_code_cell(
                "import os\nos.environ.update(" + repr(environment) + ")",
                id="offline-checker-environment",
            ))
            if args.execution == "kernel":
                NotebookClient(notebook, kernel_name=manifest["kernel_name"], timeout=300,
                               resources={"metadata": {"path": str(repo / "notebooks")}}).execute()
            else:
                _execute_python(notebook, name, output_root)
            nbformat.write(notebook, output_root / name)
            results.append({"notebook": name, "status": "passed"})
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "fixture": args.with_fixture,
              "evidence_kind": "synthetic_software_check" if args.with_fixture else "safe_notebook_check",
              "real_inference_enabled": False, "downloads_enabled": False,
              "execution": args.execution, "kernel_name": manifest.get("kernel_name"), "notebooks": results}
    (output_root / "check_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
