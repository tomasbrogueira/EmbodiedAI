"""Notebook gating checks with explicitly injected synthetic CPU callbacks.

These fixtures do not run a checkpoint or the production inference runner and
are never experimental model evidence. All subprocesses block model imports.
"""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks/11_hazard_inference.ipynb"

PRELUDE = r'''
import importlib.abc
import json
from pathlib import Path
import sys

blocked = {'torch', 'torchvision', 'transformers', 'bitsandbytes', 'accelerate',
           'huggingface_hub', 'numpy', 'PIL', 'traversability_inference',
           'traversability_hazard_data', 'traversability_hazard_evaluation',
           'traversability_hazard_segmentation', 'traversability_hazard_benchmark'}
class NoModelImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError('Forbidden notebook import: ' + fullname)
sys.meta_path.insert(0, NoModelImports())
notebook = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
code_cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
cells = {cell['metadata']['hazard_step']: ''.join(cell['source']) for cell in code_cells}
namespace = {}
def execute(step):
    exec(compile(cells[step], 'hazard_notebook_' + step, 'exec'), namespace)
'''


class HazardNotebookTests(unittest.TestCase):
    def run_notebook_script(self, script, *, repo_root=ROOT):
        with tempfile.TemporaryDirectory(prefix="hazard-notebook-cpu-fixture-") as temporary:
            root = Path(temporary)
            environment = os.environ.copy()
            environment.update({
                "TRAVERSABILITY_REPO_ROOT": str(repo_root),
                "TRAVERSABILITY_DATA_ROOT": str(root / "data"),
                "TRAVERSABILITY_RUN_ROOT": str(root / "runs"),
                "TRAVERSABILITY_CACHE_ROOT": str(root / "cache"),
                "PYTHONDONTWRITEBYTECODE": "1",
            })
            result = subprocess.run(
                [sys.executable, "-I", "-B", "-c", PRELUDE + textwrap.dedent(script), str(NOTEBOOK)],
                cwd=root, env=environment, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertFalse((root / "data").exists())
            self.assertFalse((root / "runs").exists())
            self.assertFalse((root / "cache").exists())

    def test_output_cleared_and_code_cells_compile(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertEqual(notebook["nbformat_minor"], 5)
        cell_ids = [cell.get("id") for cell in notebook["cells"]]
        self.assertTrue(all(isinstance(cell_id, str) and
                            re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", cell_id)
                            for cell_id in cell_ids))
        self.assertEqual(len(cell_ids), len(set(cell_ids)))
        steps = []
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                compile("".join(cell["source"]), f"hazard_cell_{index}", "exec")
                steps.append(cell["metadata"]["hazard_step"])
        self.assertEqual(steps, ["configuration", "inspection", "download", "load", "inference", "close"])

    def test_default_execution_is_read_only_and_configuration_rerun_retains_ownership(self):
        for repo_root in (ROOT, ROOT.parent):
            with self.subTest(repo_root=repo_root):
                self.run_notebook_script('''
                    for cell in code_cells:
                        exec(''.join(cell['source']), namespace)
                    assert namespace['backend'] is None
                    assert namespace['DOWNLOAD_WEIGHTS'] is False
                    assert namespace['LOAD_MODEL'] is False
                    assert namespace['RUN_INFERENCE'] is False
                    assert namespace['RETRY_ERRORS'] is False
                    assert namespace['settings']['local_files_only'] is True
                    assert not blocked & set(sys.modules)
                    owned = object()
                    namespace['backend'] = owned
                    execute('configuration')
                    assert namespace['backend'] is owned
                    assert not namespace['RUN_DIR'].exists()
                    assert not namespace['CACHE_ROOT'].exists()
                ''', repo_root=repo_root)

    def test_configuration_rejects_unsafe_run_names_without_creating_roots(self):
        self.run_notebook_script(r'''
            original = cells['configuration']
            unsafe_names = ['', '.', '..', '../escape', 'nested/../escape',
                            '/absolute', 'C:/absolute', 'nested\\escape']
            for run_name in unsafe_names:
                cells['configuration'] = original.replace(
                    'RUN_NAME_OVERRIDE = None', 'RUN_NAME_OVERRIDE = ' + repr(run_name))
                try:
                    execute('configuration')
                    raise AssertionError('Unsafe run name should fail: ' + repr(run_name))
                except ValueError:
                    pass
            cells['configuration'] = original.replace(
                'RUN_NAME_OVERRIDE = None', "RUN_NAME_OVERRIDE = 'nested/safe-run'")
            execute('configuration')
            assert namespace['RUN_DIR'] == namespace['RUN_ROOT'] / 'nested/safe-run'
            assert not namespace['RUN_ROOT'].exists()
            assert not blocked & set(sys.modules)
        ''')

    def test_explicit_steps_are_separate_and_close_is_owned_and_repeatable(self):
        self.run_notebook_script('''
            execute('configuration')
            events = []
            class SyntheticCPUBackend:
                metadata = {'fixture': True, 'execution_kind': 'synthetic_cpu_fixture'}
                def close(self):
                    events.append('close')
            def fixture_download(model_key, settings):
                events.append('download')
                assert model_key == namespace['MODEL_KEY']
                return Path(settings['cache_dir']) / 'synthetic-fixture'
            def fixture_load(model_key, settings):
                events.append('load')
                return SyntheticCPUBackend()
            def fixture_run(run_dir, model_key, settings, data_root, *, backend):
                events.append('run')
                assert backend is namespace['backend']
                assert settings['retry_errors'] is True
                return {'fixture': True, 'execution_kind': 'synthetic_cpu_fixture'}
            import traversability_hazard_inference.download as downloads
            downloads.download_weights = fixture_download
            namespace['load_backend'] = fixture_load
            namespace['run_inference'] = fixture_run
            namespace['DOWNLOAD_WEIGHTS'] = True
            execute('download')
            assert events == ['download']
            assert namespace['backend'] is None
            namespace['LOAD_MODEL'] = True
            execute('load')
            assert events == ['download', 'load']
            owned = namespace['backend']
            try:
                execute('load')
                raise AssertionError('Double loading should have failed')
            except RuntimeError:
                pass
            assert namespace['backend'] is owned
            namespace['RUN_INFERENCE'] = True
            namespace['RETRY_ERRORS'] = True
            namespace['MODEL_KEY'] = 'qwen3_5_4b'
            try:
                execute('inference')
                raise AssertionError('Changed model should have failed')
            except RuntimeError:
                pass
            assert events == ['download', 'load']
            namespace['MODEL_KEY'] = 'qwen3_vl_4b'
            execute('inference')
            assert events == ['download', 'load', 'run']
            assert namespace['backend'] is owned
            execute('close')
            execute('close')
            assert events == ['download', 'load', 'run', 'close']
            assert namespace['backend'] is None
            assert not blocked & set(sys.modules)
        ''')

    def test_inference_interruption_closes_only_injected_owned_backend(self):
        self.run_notebook_script('''
            execute('configuration')
            events = []
            class SyntheticCPUBackend:
                metadata = {'fixture': True, 'execution_kind': 'synthetic_cpu_fixture'}
                def close(self):
                    events.append('close')
            namespace['load_backend'] = lambda model_key, settings: SyntheticCPUBackend()
            def interrupt_fixture(*args, **kwargs):
                raise KeyboardInterrupt('synthetic CPU fixture interruption')
            namespace['run_inference'] = interrupt_fixture
            namespace['LOAD_MODEL'] = True
            execute('load')
            namespace['RUN_INFERENCE'] = True
            try:
                execute('inference')
                raise AssertionError('Fixture interruption should have escaped')
            except KeyboardInterrupt:
                pass
            assert events == ['close']
            assert namespace['backend'] is None
            assert namespace['_backend_model_key'] is None
            execute('close')
            assert events == ['close']
            assert not blocked & set(sys.modules)
        ''')


if __name__ == "__main__":
    unittest.main()
