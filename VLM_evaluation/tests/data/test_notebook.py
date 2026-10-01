"""Execute default notebook cells without downloads, model loading or widgets."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class NotebookTests(unittest.TestCase):
    def test_default_notebook_cpu_execution_and_resume(self):
        repo = Path(__file__).resolve().parents[2]
        notebook = json.loads((repo / "notebooks/01_data_and_annotations.ipynb").read_text(encoding="utf-8"))
        identifiers = [cell["id"] for cell in notebook["cells"]]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary) / "external-inputs"
            run_root = Path(temporary) / "external-runs"
            with patch.dict("os.environ", {"TRAVERSABILITY_DATA_ROOT": str(data_root), "TRAVERSABILITY_RUN_ROOT": str(run_root)}):
                with redirect_stdout(io.StringIO()):
                    for _ in range(2):
                        namespace = {}
                        for index, cell in enumerate(notebook["cells"]):
                            if cell["cell_type"] == "code":
                                exec(compile("".join(cell["source"]), f"data-notebook-cell-{index}", "exec"), namespace)
            self.assertTrue((run_root / "cpu-data-fixture/annotations.jsonl").is_file())
            self.assertEqual((run_root / "public-rellis-v1/frames.jsonl").read_text(), "")
            self.assertFalse((data_root / "archives").exists())
            self.assertFalse((data_root / "checkpoints").exists())


if __name__ == "__main__":
    unittest.main()
