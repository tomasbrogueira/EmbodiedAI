"""Check notebook discovery without importing packages or running notebook actions."""

import ast
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


COMPONENT = Path(__file__).resolve().parents[2]
NOTEBOOK_ROOTS = {
    "00_setup_and_checks.ipynb": "REPO_ROOT",
    "01_data_and_annotations.ipynb": "REPO",
    "02_vlm_inference.ipynb": "REPOSITORY_ROOT",
    "03_classification_evaluation.ipynb": "REPOSITORY",
    "04_deployment_benchmark.ipynb": "REPO_ROOT",
}
CONFIGURED_NOTEBOOKS = (
    "00_setup_and_checks.ipynb",
    "02_vlm_inference.ipynb",
    "04_deployment_benchmark.ipynb",
)


def discovery_code(name):
    """Extract the real discovery statements, stopping before package imports."""
    notebook = json.loads((COMPONENT / "notebooks" / name).read_text(encoding="utf-8"))
    variable = NOTEBOOK_ROOTS[name]
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        for index, node in enumerate(tree.body):
            if (
                isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == variable for target in node.targets)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "next"
            ):
                if not isinstance(tree.body[index + 1], ast.If):
                    raise AssertionError(f"Missing discovery failure check in {name}")
                discovery = ast.Module(body=tree.body[:index + 2], type_ignores=[])
                return compile(discovery, name, "exec")
    raise AssertionError(f"Missing root discovery in {name}")


def discover(name, cwd, *, configured=None, override=None):
    namespace = {"REPO_ROOT_OVERRIDE": override}
    environment = {} if configured is None else {"TRAVERSABILITY_REPO_ROOT": str(configured)}
    with patch.dict(os.environ, environment, clear=True), patch("pathlib.Path.cwd", return_value=cwd):
        exec(discovery_code(name), namespace)
    return namespace[NOTEBOOK_ROOTS[name]]


class NotebookLocationTests(unittest.TestCase):
    def test_all_notebooks_find_component_from_three_launch_locations(self):
        for name in NOTEBOOK_ROOTS:
            for location in (COMPONENT.parent, COMPONENT, COMPONENT / "notebooks"):
                with self.subTest(notebook=name, location=str(location)):
                    self.assertEqual(discover(name, location), COMPONENT)

    def test_explicit_environment_root_supports_component_and_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            unrelated = Path(temporary)
            for name in CONFIGURED_NOTEBOOKS:
                for configured in (COMPONENT, COMPONENT.parent):
                    with self.subTest(notebook=name, configured=str(configured)):
                        self.assertEqual(discover(name, unrelated, configured=configured), COMPONENT)

    def test_notebook_override_takes_precedence_over_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            unrelated = Path(temporary)
            for name in (CONFIGURED_NOTEBOOKS[0], CONFIGURED_NOTEBOOKS[2]):
                for override in (COMPONENT, COMPONENT.parent):
                    with self.subTest(notebook=name, override=str(override)):
                        self.assertEqual(
                            discover(name, unrelated, configured=unrelated, override=override), COMPONENT
                        )

    def test_invalid_configured_root_does_not_silently_use_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            invalid = Path(temporary)
            for name in CONFIGURED_NOTEBOOKS:
                with self.subTest(notebook=name), self.assertRaises(RuntimeError):
                    discover(name, COMPONENT, configured=invalid)


if __name__ == "__main__":
    unittest.main()
