"""Execute the safe notebook paths without Jupyter or model dependencies."""

import ast
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

REPOSITORY = Path(__file__).resolve().parents[2]
NOTEBOOKS = ("00_setup_and_checks.ipynb", "04_deployment_benchmark.ipynb")


def _execute(name, root, *, fake=False):
    notebook = json.loads((REPOSITORY / "notebooks" / name).read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    with redirect_stdout(io.StringIO()):
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            exec(compile("".join(cell["source"]), f"{name}:{cell['id']}", "exec"), namespace)
            if cell["id"].endswith("config"):
                namespace.update(REPO_ROOT_OVERRIDE=str(REPOSITORY), DATA_ROOT=str(root / "data"), RUN_ROOT=str(root / "runs"), CACHE_ROOT=str(root / "cache"))
                if name.startswith("00"):
                    namespace["CHECK_GPU"] = False
                elif fake:
                    namespace["RUN_FAKE_BENCHMARK"] = True
    return namespace


class NotebookTests(unittest.TestCase):
    def test_notebook_structure_syntax_and_cleared_outputs(self):
        for name in NOTEBOOKS:
            notebook = json.loads((REPOSITORY / "notebooks" / name).read_text(encoding="utf-8"))
            self.assertEqual(notebook["nbformat"], 4)
            self.assertEqual(len({c["id"] for c in notebook["cells"]}), len(notebook["cells"]))
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    self.assertEqual(cell["outputs"], [])
                    self.assertIsNone(cell["execution_count"])
                    ast.parse("".join(cell["source"]))

    def test_default_notebooks_are_read_only_and_benchmark_actions_disabled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.dict(os.environ, {"TRAVERSABILITY_REPO_ROOT": str(REPOSITORY)}):
                setup = _execute(NOTEBOOKS[0], root)
                benchmark = _execute(NOTEBOOKS[1], root)
            self.assertTrue(setup["report"]["python"]["supported"])
            self.assertFalse(benchmark["RUN_BENCHMARK"])
            self.assertFalse(benchmark["RUN_FAKE_BENCHMARK"])
            self.assertEqual(list(root.iterdir()), [])

    def test_explicit_cpu_fixture_runs_without_inference_or_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            namespace = _execute(NOTEBOOKS[1], root, fake=True)
            profiles = list((root / "runs").rglob("summary.json"))
            self.assertEqual(len(profiles), 1)
            summary = json.loads(profiles[0].read_text(encoding="utf-8"))
            metadata = json.loads(profiles[0].with_name("metadata.json").read_text(encoding="utf-8"))
            self.assertTrue(metadata["fixture"])
            self.assertEqual(summary["measured_frames"], 40)
            self.assertEqual(summary["warmup"]["attempted_frames"], 5)
            self.assertEqual(summary["failed_regions"], 0)
            self.assertIsNone(summary["memory"]["allocated_peak_bytes"])
            self.assertFalse(namespace["RUN_BENCHMARK"])


if __name__ == "__main__":
    unittest.main()
