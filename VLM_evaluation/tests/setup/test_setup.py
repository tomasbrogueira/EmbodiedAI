"""CPU-only setup tests; all installation and launch subprocesses are mocked."""

from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))
from traversability_benchmark import environment


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


setup = load_script("setup_env")
launcher = load_script("start_notebook")


class EnvironmentTests(unittest.TestCase):
    def test_explicit_roots_override_environment_and_relative_roots_use_repository(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(os.environ, {
            "TRAVERSABILITY_DATA_ROOT": "env-data",
            "TRAVERSABILITY_RUN_ROOT": "env-runs",
            "TRAVERSABILITY_CACHE_ROOT": "env-cache",
        }):
            repository = Path(temporary).resolve()
            roots = environment.resolve_roots(repository, data_root="chosen-data")
            self.assertEqual(roots.data_root, repository / "chosen-data")
            self.assertEqual(roots.run_root, repository / "env-runs")
            self.assertEqual(roots.cache_root, repository / "env-cache")
            self.assertFalse(roots.data_root.exists())

    def test_defaults_and_empty_explicit_root(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(os.environ, {}, clear=True):
            roots = environment.resolve_roots(temporary)
            self.assertEqual(roots.run_root, Path(temporary).resolve() / "runs")
            self.assertEqual(roots.cache_root.name, ".cache")
            with self.assertRaisesRegex(ValueError, "nonempty"):
                environment.resolve_roots(temporary, cache_root="")

    def test_report_is_json_serializable_and_missing_paths_are_not_created(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(os.environ, {}, clear=True):
            roots = environment.resolve_roots(temporary)
            with mock.patch.object(environment, "_gpu_report") as gpu:
                report = environment.environment_report(roots, gpu_check=False)
            gpu.assert_not_called()
            json.dumps(report)
            self.assertFalse(report["gpu"]["checked"])
            self.assertIsNone(report["gpu"]["cuda_available"])
            for item in report["paths"].values():
                self.assertFalse(item["exists"])
                self.assertEqual(Path(item["disk_probe_path"]), Path(temporary).resolve())
                self.assertGreater(item["disk_total_bytes"], 0)
                self.assertFalse(Path(item["path"]).exists())

    def test_file_root_is_actionable(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "file"
            path.write_text("fixture", encoding="utf-8")
            report = environment._path_report(path)
            self.assertFalse(report["is_directory"])
            self.assertIn("choose a directory", report["error"])

    def test_absent_torch_does_not_import_it_or_invent_hardware(self):
        with mock.patch.object(environment.importlib, "import_module") as import_module:
            report = environment._gpu_report(False)
        import_module.assert_not_called()
        self.assertIsNone(report["cuda_available"])
        self.assertIsNone(report["device_count"])
        self.assertEqual(report["devices"], [])

    def test_gpu_probe_reads_properties_without_tensor_allocation(self):
        cuda = mock.Mock()
        cuda.is_available.return_value = True
        cuda.device_count.return_value = 1
        cuda.get_device_properties.return_value = SimpleNamespace(name="fixture GPU", total_memory=123456)
        torch = SimpleNamespace(__version__="fixture", version=SimpleNamespace(cuda="12.6"), cuda=cuda)
        with mock.patch.object(environment.importlib, "import_module", return_value=torch):
            report = environment._gpu_report(True)
        self.assertEqual(report["devices"], [{"index": 0, "name": "fixture GPU", "total_memory_bytes": 123456}])
        self.assertIsNone(report["error"])
        self.assertEqual([call[0] for call in cuda.method_calls], ["is_available", "device_count", "get_device_properties"])

    def test_broken_optional_torch_reports_error(self):
        with mock.patch.object(environment.importlib, "import_module", side_effect=ImportError("fixture DLL error")):
            report = environment._gpu_report(True)
        self.assertIn("fixture DLL error", report["error"])
        self.assertIsNone(report["device_count"])


class SetupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repository = Path(self.temporary.name).resolve()
        (self.repository / "pyproject.toml").write_text("[project]\nname='fixture'\nversion='0'\n", encoding="utf-8")
        (self.repository / "requirements").mkdir()
        (self.repository / "requirements/base.txt").write_text("setuptools>=68\nwheel>=0.43\n", encoding="utf-8")
        (self.repository / "requirements/notebooks.txt").write_text("ipykernel>=6\njupyterlab>=4\n", encoding="utf-8")
        self.environment_patch = mock.patch.dict(os.environ, {}, clear=True)
        self.environment_patch.start()
        self.addCleanup(self.environment_patch.stop)

    def plan(self, *arguments):
        args = setup.create_parser().parse_args(["--repo-root", str(self.repository), *arguments])
        return setup.build_plan(args)

    def test_cpu_default_has_no_models_notebooks_fragments_or_launch(self):
        plan = self.plan()
        self.assertEqual([command["label"] for command in plan["commands"]], ["base", "editable", "check"])
        self.assertEqual(plan["constraints"], [])
        self.assertEqual(plan["pending_modules"], [])
        self.assertEqual(plan["profile"], "cpu")
        self.assertEqual(Path(plan["env_dir"]), self.repository / ".venv-cpu")
        for command in plan["commands"]:
            self.assertEqual(Path(command["argv"][0]), setup.environment_python(Path(plan["env_dir"])))
            self.assertNotIn("jupyterlab", command["argv"])
            self.assertNotIn("torch", command["argv"])
        editable = plan["commands"][1]["argv"]
        self.assertIn("--no-build-isolation", editable)
        self.assertIn("--no-deps", editable)
        self.assertFalse(Path(plan["env_dir"]).exists())

    def test_only_selected_peer_fragments_are_consumed_without_modifying_them(self):
        data = self.repository / "requirements/data.txt"
        inference = self.repository / "requirements/inference.txt"
        data.write_text("Pillow>=10\n", encoding="utf-8")
        inference.write_text("torch>=2\n", encoding="utf-8")
        initial = data.read_bytes(), inference.read_bytes()
        plan = self.plan("--with-module", "data", "--with-module", "evaluation")
        self.assertIn(str(data), plan["requirements"])
        self.assertNotIn(str(inference), plan["requirements"])
        self.assertEqual(plan["pending_modules"], ["evaluation"])
        self.assertEqual(initial, (data.read_bytes(), inference.read_bytes()))

    def test_cuda_requires_explicit_supported_wheel_index(self):
        for arguments in (("--cuda",), ("--with-module", "inference"), ("--cuda", "--torch-index-url", "https://example.test/whl/cu126")):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.plan(*arguments)
        plan = self.plan("--cuda", "--torch-index-url", "https://download.pytorch.org/whl/cu126")
        self.assertEqual(plan["pending_modules"], ["inference"])
        self.assertEqual(plan["cuda_build"], "cu126")
        self.assertIn("torch==2.13.0+cu126", plan["constraints"])
        self.assertIn("transformers==5.18.0", plan["constraints"])
        self.assertEqual([command["label"] for command in plan["commands"]], ["base", "cuda", "models", "editable", "check"])

    def test_cuda_peer_fragments_are_constrained_to_same_model_stack(self):
        fragment = self.repository / "requirements/inference.txt"
        fragment.write_text("torch>=2.6\ntransformers>=5\n", encoding="utf-8")
        plan = self.plan("--cuda", "--torch-index-url", "https://download.pytorch.org/whl/cu130")
        command = next(command for command in plan["commands"] if command["label"] == "fragment:inference.txt")
        self.assertIn(str(Path(plan["env_dir"]) / setup.CONSTRAINTS_NAME), command["argv"])
        self.assertEqual(fragment.read_text(encoding="utf-8"), "torch>=2.6\ntransformers>=5\n")

    def _hazard_fragments(self):
        for name in setup.HAZARD_MODULES:
            (self.repository / f"requirements/{name}.txt").write_text("numpy>=1.26,<2\n", encoding="utf-8")

    def test_hazard_cpu_fragments_do_not_pull_models(self):
        self._hazard_fragments()
        plan = self.plan("--with-module", "hazard_data", "--with-module", "hazard_evaluation")
        self.assertEqual([Path(path).name for path in plan["requirements"]], ["base.txt", "hazard_data.txt", "hazard_evaluation.txt"])
        self.assertEqual(plan["constraints"], [])
        self.assertFalse(any(command["label"] in ("cuda", "models", "sam3") for command in plan["commands"]))

    def test_real_hazard_fragments_accept_the_same_unified_version_constraints(self):
        from packaging.requirements import Requirement
        from packaging.utils import canonicalize_name
        proposed = {canonicalize_name(name): version for name, version in setup.HAZARD_GPU_VERSIONS.items()}
        proposed.update(torch=proposed["torch"] + "+cu128", torchvision=proposed["torchvision"] + "+cu128")
        for name in setup.HAZARD_MODULES:
            fragment = setup.REPOSITORY_ROOT / f"requirements/{name}.txt"
            for line in fragment.read_text(encoding="utf-8").splitlines():
                content = line.split("#", 1)[0].strip()
                if not content:
                    continue
                requirement = Requirement(content)
                key = canonicalize_name(requirement.name)
                if key in proposed:
                    self.assertIn(proposed[key], requirement.specifier, f"{fragment.name}: {content}")

    def test_hazard_gpu_is_explicit_and_reconciles_all_fragments(self):
        self._hazard_fragments()
        with mock.patch.object(setup.sys, "version_info", (3, 12, 13)):
            plan = self.plan("--cuda", "--gpu-profile", "hazard", "--torch-index-url", "https://download.pytorch.org/whl/cu128",
                             "--driver-version", "596.58", "--with-module", "hazard_data", "--with-module", "hazard_evaluation")
        self.assertEqual(plan["profile"], "hazard_cuda")
        self.assertEqual(plan["pending_modules"], [])
        self.assertEqual(plan["minimum_python"], [3, 12])
        for pin in ("torch==2.10.0+cu128", "torchvision==0.25.0+cu128", "numpy==1.26.4", "huggingface-hub==1.31.0", "setuptools==80.9.0"):
            self.assertIn(pin, plan["constraints"])
        self.assertEqual({Path(path).name for path in plan["requirements"]},
                         {"base.txt", *(f"{name}.txt" for name in setup.HAZARD_MODULES if name != "hazard_visualization")})
        sam3 = next(command for command in plan["commands"] if command["label"] == "sam3")
        self.assertIn(setup.SAM3_URL, sam3["argv"])
        self.assertIn("--no-deps", sam3["argv"])
        self.assertFalse(Path(plan["env_dir"]).exists())

    def test_hazard_gpu_rejects_unverified_driver_unsupported_python_and_legacy_pins(self):
        self._hazard_fragments()
        arguments = ("--cuda", "--gpu-profile", "hazard", "--torch-index-url", "https://download.pytorch.org/whl/cu128")
        with mock.patch.object(setup.sys, "version_info", (3, 12, 13)):
            with self.assertRaisesRegex(ValueError, "driver-version"):
                self.plan(*arguments)
            with self.assertRaisesRegex(ValueError, "minimum"):
                self.plan(*arguments, "--driver-version", "510.00")
            with self.assertRaisesRegex(ValueError, "Legacy inference"):
                self.plan(*arguments, "--driver-version", "596.58", "--with-module", "inference")
        with mock.patch.object(setup.sys, "version_info", (3, 11, 9)):
            with self.assertRaisesRegex(ValueError, "Python 3.12"):
                self.plan(*arguments, "--driver-version", "596.58")
        with mock.patch.object(setup.sys, "version_info", (3, 13, 7)):
            with self.assertRaisesRegex(ValueError, "NumPy"):
                self.plan(*arguments, "--driver-version", "596.58")

    def test_hazard_tooling_uses_standard_package_source_before_cuda(self):
        self._hazard_fragments()
        wheelhouse = self.repository / "wheelhouse"
        wheelhouse.mkdir()
        for source_arguments in ((), ("--wheelhouse", str(wheelhouse))):
            with self.subTest(source_arguments=source_arguments), mock.patch.object(setup.sys, "version_info", (3, 12, 13)):
                plan = self.plan("--cuda", "--gpu-profile", "hazard", "--torch-index-url", "https://download.pytorch.org/whl/cu126",
                                 "--driver-version", "535.247.01", *source_arguments)
            labels = [command["label"] for command in plan["commands"]]
            self.assertLess(labels.index("base"), labels.index("tooling"))
            self.assertLess(labels.index("tooling"), labels.index("cuda"))
            tooling = plan["commands"][labels.index("tooling")]["argv"]
            self.assertIn("setuptools==80.9.0", tooling)
            self.assertIn(str(Path(plan["env_dir"]) / setup.CONSTRAINTS_NAME), tooling)
            self.assertNotIn("--index-url", tooling)
            self.assertNotIn("--no-index", tooling)
            if source_arguments:
                self.assertEqual(tooling[tooling.index("--find-links") + 1], str(wheelhouse))
            else:
                self.assertNotIn("--find-links", tooling)
            cuda = plan["commands"][labels.index("cuda")]["argv"]
            self.assertEqual(cuda[cuda.index("--index-url") + 1], "https://download.pytorch.org/whl/cu126")

    def test_hazard_model_fragments_require_hazard_gpu_and_missing_fragments_fail(self):
        for name in ("hazard_inference", "hazard_segmentation"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "gpu-profile hazard"):
                self.plan("--with-module", name)
        with self.assertRaisesRegex(ValueError, "Requested hazard fragment"):
            self.plan("--with-module", "hazard_data")

    def test_offline_hazard_gpu_requires_local_pinned_sam3_source(self):
        self._hazard_fragments()
        wheelhouse = self.repository / "wheelhouse"
        wheelhouse.mkdir()
        arguments = ("--cuda", "--gpu-profile", "hazard", "--torch-index-url", "https://download.pytorch.org/whl/cu128",
                     "--driver-version", "596.58", "--offline", "--wheelhouse", str(wheelhouse))
        with mock.patch.object(setup.sys, "version_info", (3, 12, 13)):
            with self.assertRaisesRegex(ValueError, "sam3-source"):
                self.plan(*arguments)
            source = self.repository / "cached-sam3"
            (source / "sam3").mkdir(parents=True)
            (source / "sam3/model_builder.py").write_text("# placeholder checked by installation provenance gate\n", encoding="utf-8")
            plan = self.plan(*arguments, "--sam3-source", str(source))
        sam3 = next(command for command in plan["commands"] if command["label"] == "sam3")
        self.assertIn("--no-index", sam3["argv"])
        self.assertIn(str(source), sam3["argv"])
        self.assertNotIn(setup.SAM3_URL, sam3["argv"])
        self.assertEqual(plan["sam3_revision"], setup.SAM3_REVISION)
        labels = [command["label"] for command in plan["commands"]]
        self.assertLess(labels.index("tooling"), labels.index("cuda"))
        tooling = plan["commands"][labels.index("tooling")]["argv"]
        self.assertIn("setuptools==80.9.0", tooling)
        for command in plan["commands"]:
            if command["label"] != "check":
                self.assertIn("--no-index", command["argv"])
                self.assertIn("--only-binary=:all:", command["argv"])
                self.assertEqual(command["argv"][command["argv"].index("--find-links") + 1], str(wheelhouse))
                self.assertNotIn("--index-url", command["argv"])

    def test_local_sam3_wrong_revision_blocks_before_environment_writes(self):
        plan = self.plan()
        plan["sam3_source"] = str(self.repository / "sam3")
        with mock.patch.object(setup.subprocess, "run", side_effect=[SimpleNamespace(stdout="wrong-revision\n"), SimpleNamespace(stdout="")]):
            with self.assertRaisesRegex(ValueError, "pinned official revision"):
                setup.execute_plan(plan)
        self.assertFalse(Path(plan["env_dir"]).exists())

    def test_hazard_gpu_does_not_replace_an_existing_legacy_gpu_environment(self):
        plan = self.plan()
        plan["profile"] = "hazard_cuda"
        env_dir = Path(plan["env_dir"])
        env_dir.mkdir()
        (env_dir / "pyvenv.cfg").write_text("include-system-site-packages = false\n", encoding="utf-8")
        python = setup.environment_python(env_dir)
        python.parent.mkdir(parents=True, exist_ok=True)
        python.touch()
        manifest = env_dir / setup.MANIFEST_NAME
        manifest.write_text(json.dumps({"profile": "cuda", "status": "installed"}), encoding="utf-8")
        original = manifest.read_bytes()
        with mock.patch.object(setup.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "separate isolated environments"):
                setup.execute_plan(plan)
        run.assert_not_called()
        self.assertEqual(manifest.read_bytes(), original)

    def test_offline_uses_only_wheelhouse_for_every_install_and_no_build_isolation(self):
        wheelhouse = self.repository / "wheelhouse"
        wheelhouse.mkdir()
        plan = self.plan("--offline", "--wheelhouse", str(wheelhouse), "--notebooks")
        for command in plan["commands"]:
            if command["label"] not in ("check", "kernel"):
                self.assertIn("--no-index", command["argv"])
                self.assertIn("--find-links", command["argv"])
                self.assertNotIn("--index-url", command["argv"])
        self.assertEqual(plan["environment"]["HF_HUB_OFFLINE"], "1")
        with self.assertRaisesRegex(ValueError, "wheelhouse"):
            self.plan("--offline")

    def test_offline_rejects_nested_remote_requirements(self):
        wheelhouse = self.repository / "wheelhouse"
        wheelhouse.mkdir()
        (self.repository / "requirements/data.txt").write_text("-r model.txt\n", encoding="utf-8")
        (self.repository / "requirements/model.txt").write_text("sam2 @ git+https://github.com/example/sam.git\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "remote requirements"):
            self.plan("--offline", "--wheelhouse", str(wheelhouse), "--with-module", "data")

    def test_kernel_is_environment_scoped_and_cache_roots_are_external(self):
        bulk = self.repository / "external-bulk"
        plan = self.plan("--notebooks", "--env-dir", str(bulk / "env"), "--cache-root", str(bulk / "cache"))
        command = plan["commands"][-1]
        self.assertEqual(command["label"], "kernel")
        self.assertIn("--prefix", command["argv"])
        self.assertNotIn("--user", command["argv"])
        self.assertEqual(command["argv"][command["argv"].index("--prefix") + 1], str(bulk / "env"))
        for key in ("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE", "HF_DATASETS_CACHE", "TORCH_HOME", "JUPYTER_DATA_DIR", "JUPYTER_CONFIG_DIR", "JUPYTER_RUNTIME_DIR", "PIP_CACHE_DIR", "TMPDIR", "TEMP", "TMP"):
            self.assertTrue(Path(plan["environment"][key]).is_relative_to(bulk / "cache"))

    def test_dry_run_does_not_create_environment(self):
        with mock.patch.object(setup, "execute_plan") as execute, redirect_stdout(io.StringIO()) as output:
            status = setup.main(["--repo-root", str(self.repository), "--dry-run"])
        self.assertEqual(status, 0)
        execute.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())["profile"], "cpu")
        self.assertFalse((self.repository / ".venv-cpu").exists())

    def test_nonisolated_or_nonempty_targets_are_rejected(self):
        env_dir = self.repository / "env"
        env_dir.mkdir()
        (env_dir / "marker").write_text("fixture", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "nonempty"):
            setup._existing_environment(env_dir)
        (env_dir / "pyvenv.cfg").write_text("include-system-site-packages = true\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "system site packages"):
            setup._existing_environment(env_dir)

    def test_kernel_collision_does_not_overwrite_another_interpreter(self):
        env_dir = self.repository / "env"
        spec = env_dir / "share/jupyter/kernels/fixture/kernel.json"
        spec.parent.mkdir(parents=True)
        spec.write_text(json.dumps({"argv": [str(self.repository / "other/python")]}), encoding="utf-8")
        original = spec.read_bytes()
        with self.assertRaisesRegex(ValueError, "another interpreter"):
            setup._check_kernel_collision(env_dir, "fixture")
        self.assertEqual(spec.read_bytes(), original)

    def test_mock_install_records_versions_and_injects_kernel_paths(self):
        plan = self.plan("--notebooks", "--kernel-name", "fixture")
        env_dir = Path(plan["env_dir"])

        def create_environment(target):
            for key in ("TMPDIR", "TEMP", "TMP"):
                self.assertEqual(os.environ[key], str(Path(plan["roots"]["cache_root"]) / "tmp"))
                self.assertTrue(Path(os.environ[key]).is_dir())
            target.mkdir(parents=True)
            (target / "pyvenv.cfg").write_text("include-system-site-packages = false\n", encoding="utf-8")
            python = setup.environment_python(target)
            python.parent.mkdir(parents=True, exist_ok=True)
            python.touch()

        def run(arguments, **kwargs):
            self.assertEqual(Path(arguments[0]), setup.environment_python(env_dir))
            self.assertEqual(kwargs["env"]["TRAVERSABILITY_CACHE_ROOT"], plan["roots"]["cache_root"])
            self.assertNotIn("PIP_TARGET", kwargs["env"])
            self.assertNotIn("PYTHONPATH", kwargs["env"])
            self.assertEqual(kwargs["env"]["PIP_CONFIG_FILE"], os.devnull)
            for key in ("TMPDIR", "TEMP", "TMP"):
                self.assertEqual(kwargs["env"][key], str(Path(plan["roots"]["cache_root"]) / "tmp"))
            if arguments[1] == "-c":
                return SimpleNamespace(stdout=json.dumps({"prefix": str(env_dir), "base_prefix": "fixture-base", "version": [3, 11, 0]}))
            if arguments[2:4] == ["ipykernel", "install"]:
                spec = env_dir / "share/jupyter/kernels/fixture/kernel.json"
                spec.parent.mkdir(parents=True)
                spec.write_text(json.dumps({"argv": [str(setup.environment_python(env_dir)), "-m", "ipykernel_launcher"]}), encoding="utf-8")
            return SimpleNamespace(stdout="setuptools==fixture\nwheel==fixture\n")

        with mock.patch.dict(os.environ, {"PIP_TARGET": "fixture-unsafe", "PYTHONPATH": "fixture-unsafe"}), mock.patch.object(setup.venv, "EnvBuilder") as builder, mock.patch.object(setup.subprocess, "run", side_effect=run), redirect_stdout(io.StringIO()):
            builder.return_value.create.side_effect = create_environment
            setup.execute_plan(plan)
        builder.assert_called_once_with(with_pip=True, system_site_packages=False)
        manifest = json.loads((env_dir / setup.MANIFEST_NAME).read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "installed")
        self.assertEqual(manifest["installed_packages"], ["setuptools==fixture", "wheel==fixture"])
        kernel = json.loads((env_dir / "share/jupyter/kernels/fixture/kernel.json").read_text(encoding="utf-8"))
        self.assertEqual(kernel["env"]["TRAVERSABILITY_DATA_ROOT"], plan["roots"]["data_root"])
        self.assertEqual(kernel["env"]["HF_HOME"], plan["environment"]["HF_HOME"])
        self.assertEqual(kernel["env"]["TMPDIR"], plan["environment"]["TMPDIR"])
        self.assertNotIn("TMPDIR", os.environ)

    def test_failed_install_does_not_write_success_manifest(self):
        plan = self.plan()
        env_dir = Path(plan["env_dir"])
        env_dir.mkdir()
        (env_dir / "pyvenv.cfg").write_text("include-system-site-packages = false\n", encoding="utf-8")
        python = setup.environment_python(env_dir)
        python.parent.mkdir(parents=True, exist_ok=True)
        python.touch()
        identity = SimpleNamespace(stdout=json.dumps({"prefix": str(env_dir), "base_prefix": "fixture-base", "version": [3, 11, 0]}))
        with mock.patch.object(setup.subprocess, "run", side_effect=[identity, subprocess.CalledProcessError(1, "fixture install")]), redirect_stdout(io.StringIO()):
            with self.assertRaises(subprocess.CalledProcessError):
                setup.execute_plan(plan)
        self.assertFalse((env_dir / setup.MANIFEST_NAME).exists())

    def test_failed_reconfiguration_invalidates_previous_success_manifest(self):
        plan = self.plan()
        env_dir = Path(plan["env_dir"])
        env_dir.mkdir()
        (env_dir / "pyvenv.cfg").write_text("include-system-site-packages = false\n", encoding="utf-8")
        python = setup.environment_python(env_dir)
        python.parent.mkdir(parents=True, exist_ok=True)
        python.touch()
        manifest = env_dir / setup.MANIFEST_NAME
        manifest.write_text(json.dumps({"status": "installed", "notebooks": True}), encoding="utf-8")
        identity = SimpleNamespace(stdout=json.dumps({"prefix": str(env_dir), "base_prefix": "fixture-base", "version": [3, 11, 0]}))
        with mock.patch.object(setup.subprocess, "run", side_effect=[identity, subprocess.CalledProcessError(1, "fixture install")]), redirect_stdout(io.StringIO()):
            with self.assertRaises(subprocess.CalledProcessError):
                setup.execute_plan(plan)
        self.assertEqual(json.loads(manifest.read_text(encoding="utf-8"))["status"], "setup_in_progress")


class NotebookLaunchTests(unittest.TestCase):
    def test_dry_run_always_binds_localhost_and_keeps_authentication(self):
        args = launcher.create_parser().parse_args(["--env-dir", "fixture-env", "--port", "8765", "--dry-run"])
        plan = launcher.build_launch_plan(args)
        self.assertIn("--ServerApp.ip=127.0.0.1", plan["argv"])
        self.assertIn("--ServerApp.port=8765", plan["argv"])
        self.assertIn("--ServerApp.port_retries=0", plan["argv"])
        self.assertFalse(any("token" in argument or "password" in argument for argument in plan["argv"]))
        self.assertFalse(any("0.0.0.0" in argument or "ssh" in argument for argument in plan["argv"]))

    def test_launch_requires_completed_notebook_setup_and_valid_port(self):
        with tempfile.TemporaryDirectory() as temporary:
            for arguments in (["--env-dir", temporary], ["--env-dir", temporary, "--port", "0", "--dry-run"]):
                with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                    launcher.build_launch_plan(launcher.create_parser().parse_args(arguments))

    def test_launcher_uses_recorded_cache_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            env_dir = Path(temporary).resolve()
            python = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            python.parent.mkdir(parents=True)
            python.touch()
            (env_dir / "pyvenv.cfg").write_text("fixture", encoding="utf-8")
            (env_dir / launcher.MANIFEST_NAME).write_text(json.dumps({
                "env_dir": str(env_dir), "notebooks": True, "status": "installed",
                "environment": {"HF_HOME": str(env_dir / "external-cache")},
            }), encoding="utf-8")
            with mock.patch.dict(os.environ, {"PYTHONPATH": "fixture-unsafe", "PIP_TARGET": "fixture-unsafe"}), mock.patch.object(launcher.subprocess, "call", return_value=0) as launch:
                self.assertEqual(launcher.main(["--env-dir", str(env_dir)]), 0)
            self.assertEqual(launch.call_args.kwargs["env"]["HF_HOME"], str(env_dir / "external-cache"))
            self.assertNotIn("PYTHONPATH", launch.call_args.kwargs["env"])
            self.assertNotIn("PIP_TARGET", launch.call_args.kwargs["env"])
            self.assertEqual(Path(launch.call_args.args[0][0]), python)


if __name__ == "__main__":
    unittest.main()
