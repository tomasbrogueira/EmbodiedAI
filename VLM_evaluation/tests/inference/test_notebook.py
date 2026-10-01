"""Portable notebook checks that never enable downloading or GPU execution."""

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks/02_vlm_inference.ipynb"


def test_notebook_is_output_free_and_every_code_cell_compiles():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            compile("".join(cell["source"]), f"notebook_cell_{index}", "exec")


def test_default_notebook_execution_is_cpu_only_and_rerun_retains_backend(tmp_path):
    environment = os.environ.copy()
    environment.update({
        "TRAVERSABILITY_REPO_ROOT": str(ROOT),
        "TRAVERSABILITY_DATA_ROOT": str(tmp_path / "data"),
        "TRAVERSABILITY_RUN_ROOT": str(tmp_path / "runs"),
        "TRAVERSABILITY_CACHE_ROOT": str(tmp_path / "cache"),
    })
    script = (
        "import json, sys; from pathlib import Path; "
        "notebook = json.loads(Path(sys.argv[1]).read_text()); namespace = {}; "
        "cells = [c for c in notebook['cells'] if c['cell_type'] == 'code']; "
        "[exec(''.join(c['source']), namespace) for c in cells]; "
        "assert not {'torch','transformers','bitsandbytes'} & set(sys.modules); "
        "owned = object(); namespace['backend'] = owned; "
        "exec(''.join(cells[0]['source']), namespace); "
        "assert namespace['backend'] is owned; "
        "assert not namespace['RUN_DIR'].exists(); "
        "assert not namespace['CACHE_ROOT'].exists()"
    )
    subprocess.run([sys.executable, "-c", script, str(NOTEBOOK)], env=environment, check=True)
