"""Check active notebook registration and constrained offline action execution."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
import io
from unittest import mock

COMPONENT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("hazard_notebook_checker", COMPONENT / "scripts/check_notebooks.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class HazardNotebookCheckerTests(unittest.TestCase):
    def test_every_active_notebook_has_one_safe_registered_configuration(self):
        import nbformat
        for name in checker.HAZARD_NOTEBOOKS:
            with self.subTest(name=name):
                notebook = nbformat.read(COMPONENT / "notebooks" / name, as_version=4)
                original = json.loads((COMPONENT / "notebooks" / name).read_text(encoding="utf-8"))
                checker.prepare_notebook(notebook, name)
                changed = [cell for cell in notebook.cells if "Offline checker:" in cell.source]
                self.assertEqual(len(changed), 1)
                for action, enabled in checker.safe_overrides(name).items():
                    self.assertFalse(enabled, action)
                self.assertTrue(all(not cell.get("outputs") and cell.get("execution_count") is None
                                    for cell in original["cells"] if cell["cell_type"] == "code"))

    def test_fixture_switches_cannot_enable_downloads_real_models_or_combined_work(self):
        for name in checker.HAZARD_NOTEBOOKS:
            for action, enabled in checker.safe_overrides(name, fixture=True).items():
                self.assertEqual(enabled, action == "RUN_FIXTURE", (name, action))

    def test_python_mode_executes_safe_copies_under_new_external_roots(self):
        import nbformat
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            values = {"TRAVERSABILITY_REPO_ROOT": str(COMPONENT),
                      "TRAVERSABILITY_DATA_ROOT": str(root / "data"),
                      "TRAVERSABILITY_RUN_ROOT": str(root / "runs"),
                      "TRAVERSABILITY_CACHE_ROOT": str(root / "cache"),
                      "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
            with checker._kernel_environment(values):
                for name in checker.HAZARD_NOTEBOOKS:
                    notebook = nbformat.read(COMPONENT / "notebooks" / name, as_version=4)
                    checker.prepare_notebook(notebook, name)
                    checker._execute_python(notebook, name, root)
            self.assertFalse((root / "data").exists())
            self.assertFalse((root / "runs").exists())
            self.assertEqual(len(list(root.glob("*.log"))), len(checker.HAZARD_NOTEBOOKS))

    def test_unknown_notebooks_and_uncleared_source_outputs_are_rejected(self):
        import nbformat
        with self.assertRaisesRegex(ValueError, "registered"):
            checker.safe_overrides("99_unknown.ipynb")
        notebook = nbformat.read(COMPONENT / "notebooks/11_hazard_inference.ipynb", as_version=4)
        next(cell for cell in notebook.cells if cell.cell_type == "code").execution_count = 1
        with self.assertRaisesRegex(ValueError, "cleared outputs"):
            checker.prepare_notebook(notebook, "11_hazard_inference.ipynb")

    def test_kernel_spec_environment_cannot_redirect_notebook_roots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env_dir = root / "isolated"
            env_dir.mkdir()
            manifest = {"env_dir": str(env_dir), "status": "installed", "notebooks": True,
                        "kernel_name": "fixture-kernel", "environment": {}}
            (env_dir / "traversability-environment.json").write_text(json.dumps(manifest), encoding="utf-8")
            destination = root / "copies"
            poisoned = {"TRAVERSABILITY_REPO_ROOT": str(root / "stale-component"),
                        "TRAVERSABILITY_DATA_ROOT": str(root / "stale-data"),
                        "TRAVERSABILITY_RUN_ROOT": str(root / "stale-runs")}

            def client(notebook, **kwargs):
                def execute():
                    # Mimic kernel.json env taking precedence over launcher env.
                    with checker._kernel_environment(poisoned):
                        checker._execute_python(notebook, "mocked-kernel.ipynb", destination)
                    return notebook
                return mock.Mock(execute=execute)

            with mock.patch.object(checker.sys, "prefix", str(env_dir)), mock.patch("nbclient.NotebookClient", side_effect=client), redirect_stdout(io.StringIO()):
                self.assertEqual(checker.main(["--output-root", str(destination)]), 0)
            self.assertFalse((root / "stale-data").exists())
            self.assertFalse((root / "stale-runs").exists())
            report = json.loads((destination / "check_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(report["notebooks"]), len(checker.HAZARD_NOTEBOOKS))


if __name__ == "__main__":
    unittest.main()
