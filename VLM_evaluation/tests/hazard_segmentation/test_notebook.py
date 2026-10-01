"""Notebook-only CPU checks; model loading, downloads, and GPU work stay disabled."""

import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


COMPONENT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = COMPONENT_ROOT / "notebooks/13_hazard_segmentation_and_cost.ipynb"
SWITCHES = ("PREPARE_SELECTION", "RUN_FIXTURE", "RUN_SEGMENTATION", "RUN_PROFILES",
            "ALLOW_DOWNLOADS", "ENABLE_COMBINED")


def read_notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def execute_notebook(root, *, fixture=False, combined=False, repo_parent=False):
    """Run code cells in a clean subprocess without requiring a Jupyter kernel."""
    environment = os.environ.copy()
    environment.update({
        "TRAVERSABILITY_REPO_ROOT": str(COMPONENT_ROOT.parent if repo_parent else COMPONENT_ROOT),
        "TRAVERSABILITY_DATA_ROOT": str(root / "data"),
        "TRAVERSABILITY_RUN_ROOT": str(root / "runs"),
        "TRAVERSABILITY_CACHE_ROOT": str(root / "cache"),
    })
    script = """
import json, sys
from pathlib import Path
notebook = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
namespace = {"__name__": "__main__"}
for cell in notebook["cells"]:
    if cell["cell_type"] != "code":
        continue
    exec(compile("".join(cell["source"]), "notebook:" + cell["id"], "exec"), namespace)
    if cell["id"] == "configuration":
        if sys.argv[2] == "fixture":
            namespace["RUN_FIXTURE"] = True
            namespace["ENABLE_COMBINED"] = sys.argv[3] == "combined"
assert not {"torch", "torchvision", "transformers", "bitsandbytes", "sam3"} & set(sys.modules)
assert namespace["RUN_SEGMENTATION"] is False
assert namespace["RUN_PROFILES"] is False
assert namespace["ALLOW_DOWNLOADS"] is False
if sys.argv[2] == "default":
    assert not namespace["RUN_DIR"].exists()
    assert not namespace["DATA_ROOT"].exists()
    assert not namespace["CACHE_ROOT"].exists()
    existing_summary = object()
    namespace["condition_summary"] = existing_summary
    config_cell = next(cell for cell in notebook["cells"] if cell["id"] == "configuration")
    exec("".join(config_cell["source"]), namespace)
    assert namespace["condition_summary"] is existing_summary
else:
    assert namespace["fixture_run_dir"] != namespace["RUN_DIR"]
    assert namespace["fixture_run_dir"].is_dir()
    assert not namespace["RUN_DIR"].exists()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(NOTEBOOK), "fixture" if fixture else "default",
         "combined" if combined else "isolated"],
        cwd=COMPONENT_ROOT.parent, env=environment, check=False, capture_output=True, text=True,
    )
    if result.returncode:
        raise AssertionError(result.stderr + "\n" + result.stdout[-2000:])
    return result


class HazardNotebookTests(unittest.TestCase):
    def test_valid_cleared_notebook_and_disabled_configuration(self):
        notebook = read_notebook()
        self.assertEqual(notebook["nbformat"], 4)
        identifiers = [cell["id"] for cell in notebook["cells"]]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                ast.parse("".join(cell["source"]))
        configuration = next(cell for cell in notebook["cells"] if cell["id"] == "configuration")
        statements = ast.parse("".join(configuration["source"])).body
        assignments = {target.id: statement.value for statement in statements
                       if isinstance(statement, ast.Assign) for target in statement.targets
                       if isinstance(target, ast.Name)}
        for switch in SWITCHES:
            self.assertIsInstance(assignments[switch], ast.Constant)
            self.assertIs(assignments[switch].value, False)
        if importlib.util.find_spec("nbformat") is not None:
            import nbformat
            nbformat.validate(nbformat.read(NOTEBOOK, as_version=4))

    def test_default_execution_is_read_only_and_portable(self):
        for repo_parent in (False, True):
            with self.subTest(repo_parent=repo_parent), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                execute_notebook(root, repo_parent=repo_parent)
                self.assertEqual(list(root.iterdir()), [])

    @unittest.skipUnless(importlib.util.find_spec("numpy") is not None
                         and importlib.util.find_spec("PIL") is not None,
                         "Fixture smoke requires the explicitly installed CPU image dependencies")
    def test_explicit_fixture_runs_four_conditions_and_two_isolated_profiles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            execute_notebook(root, fixture=True)
            runs = list((root / "runs").iterdir())
            self.assertEqual(len(runs), 1)
            self.assertTrue(runs[0].name.startswith("hazard_segmentation_fixture_"))
            for condition in ("vlm__qwen3_vl_4b", "vlm__qwen3_5_4b", "reference_present", "fixed_policy"):
                path = runs[0] / "segmentation" / condition / "frames.jsonl"
                rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
                self.assertEqual(len(rows), 20)
                self.assertTrue(all(row["status"] in ("ok", "error") for row in rows))
                if condition == "reference_present":
                    self.assertTrue(all(row["status"] == "ok" for row in rows))
                else:
                    self.assertTrue(any(row["status"] == "error" for row in rows))
                if condition == "fixed_policy":
                    self.assertTrue(all(row["requested_queries"] == 16 for row in rows))
            summaries = list(runs[0].glob("benchmark/*/*/summary.json"))
            self.assertEqual(len(summaries), 2)
            components = set()
            for path in summaries:
                summary = json.loads(path.read_text(encoding="utf-8"))
                components.add(summary["component"])
                self.assertTrue(summary["fixture"])
                self.assertTrue(summary["complete"])
                self.assertFalse(summary["comparison_ready"])
                self.assertEqual(summary["measured_frames"], 40)
                self.assertEqual(summary["expected_measured_frames"], 40)
                self.assertGreater(summary["failed_calls"], 0)
                self.assertIsNone(summary["memory"]["allocated_peak_bytes"])
            self.assertEqual(components, {"vlm", "sam"})

    @unittest.skipUnless(importlib.util.find_spec("numpy") is not None
                         and importlib.util.find_spec("PIL") is not None,
                         "Fixture smoke requires the explicitly installed CPU image dependencies")
    def test_combined_fixture_requires_explicit_switch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            execute_notebook(root, fixture=True, combined=True)
            summaries = list((root / "runs").glob("*/benchmark/*/*/summary.json"))
            self.assertEqual(len(summaries), 3)
            combined = [json.loads(path.read_text(encoding="utf-8")) for path in summaries
                        if json.loads(path.read_text(encoding="utf-8"))["component"] == "combined"]
            self.assertEqual(len(combined), 1)
            self.assertTrue(combined[0]["fixture"])
            self.assertFalse(combined[0]["comparison_ready"])


if __name__ == "__main__":
    unittest.main()
