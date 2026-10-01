"""Execute the thin notebook's safe defaults from portable launch locations."""

from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from common import COMPONENT, read_json


class NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = read_json(COMPONENT / "notebooks/10_hazard_data.ipynb")

    def test_notebook_has_unique_cells_and_no_saved_outputs(self):
        cells = self.notebook["cells"]
        self.assertEqual(len({cell["id"] for cell in cells}), len(cells))
        for cell in cells:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
        first_code = next(cell for cell in cells if cell["cell_type"] == "code")
        text = "".join(first_code["source"])
        for variable in ("DATA_ROOT", "RUN_ROOT", "CACHE_ROOT", "RUN_FIXTURE",
                         "RUN_REAL_PREPARATION", "DOWNLOAD_COCO"):
            self.assertIn(variable, text)

    def execute(self, cwd, *, override=None):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            environment = {
                "TRAVERSABILITY_DATA_ROOT": str(base / "data"),
                "TRAVERSABILITY_RUN_ROOT": str(base / "runs"),
                "TRAVERSABILITY_CACHE_ROOT": str(base / "cache"),
                "TRAVERSABILITY_REPO_ROOT": str(override) if override else "",
            }
            previous = Path.cwd()
            try:
                os.chdir(cwd)
                with patch.dict(os.environ, environment), \
                     patch("socket.create_connection", side_effect=AssertionError("network prohibited")), \
                     patch("urllib.request.urlopen", side_effect=AssertionError("downloads prohibited")), \
                     redirect_stdout(io.StringIO()):
                    namespace = {}
                    for index, cell in enumerate(self.notebook["cells"]):
                        if cell["cell_type"] == "code":
                            exec(compile("".join(cell["source"]), f"hazard-notebook-cell-{index}", "exec"), namespace)
                self.assertEqual(namespace["REPO"], COMPONENT)
                self.assertTrue(namespace["RUN_FIXTURE"])
                self.assertFalse(namespace["RUN_REAL_PREPARATION"])
                self.assertFalse(namespace["DOWNLOAD_COCO"])
                self.assertTrue(namespace["FIXTURE_CHECK"]["fixture"])
                self.assertEqual(namespace["FIXTURE_CHECK"]["frames"], 11)
                self.assertTrue(namespace["FIXTURE_CHECK"]["valid"])
                self.assertIsNone(namespace["REAL_RUN"])
                self.assertTrue((base / "runs/hazard_prompt_v1_fixture/metadata/dataset.json").is_file())
                self.assertFalse((base / "runs/hazard_prompt_v1").exists())
                self.assertFalse((base / "data/archives").exists())
                self.assertFalse((base / "data/checkpoints").exists())
                return namespace["locate_component"]
            finally:
                os.chdir(previous)

    def test_default_execution_from_checkout_component_and_notebooks(self):
        for location in (COMPONENT.parent, COMPONENT, COMPONENT / "notebooks"):
            with self.subTest(location=location):
                self.execute(location)

    def test_explicit_component_override_from_unrelated_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.execute(Path(temporary), override=COMPONENT)

    def test_explicit_checkout_override_from_unrelated_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.execute(Path(temporary), override=COMPONENT.parent)


if __name__ == "__main__":
    unittest.main()
