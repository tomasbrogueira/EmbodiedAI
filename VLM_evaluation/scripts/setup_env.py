#!/usr/bin/env python3
"""Create an isolated installation; model and notebook dependencies are opt-in."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import urlparse
import venv

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
from traversability_benchmark.environment import RootPaths, resolve_roots


GPU_VERSIONS = {
    "torch": "2.13.0",
    "torchvision": "0.28.0",
    "transformers": "5.18.0",
    "accelerate": "1.15.0",
    "bitsandbytes": "0.50.2",
}
SUPPORTED_CUDA_WHEELS = {"cu126": "12.6", "cu130": "13.0", "cu132": "13.2"}
HAZARD_GPU_VERSIONS = {
    "torch": "2.10.0", "torchvision": "0.25.0", "transformers": "5.18.0",
    "accelerate": "1.15.0", "bitsandbytes": "0.50.2", "numpy": "1.26.4",
    "Pillow": "11.3.0", "huggingface-hub": "1.31.0", "setuptools": "80.9.0",
}
HAZARD_CUDA_WHEELS = {"cu126": "12.6", "cu128": "12.8", "cu130": "13.0"}
HAZARD_MODULES = ("hazard_data", "hazard_inference", "hazard_evaluation", "hazard_segmentation", "hazard_visualization")
SAM3_REVISION = "2345a4ad109ac29c569da749c91d84f10dc08c40"
SAM3_URL = f"sam3 @ git+https://github.com/facebookresearch/sam3.git@{SAM3_REVISION}"
MANIFEST_NAME = "traversability-environment.json"
CONSTRAINTS_NAME = "traversability-constraints.txt"


def create_parser() -> argparse.ArgumentParser:
    """Return the portable setup command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--env-dir", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--notebooks", action="store_true", help="Install JupyterLab and register an environment-scoped kernel")
    parser.add_argument("--with-module", action="append", choices=("data", "evaluation", "inference", *HAZARD_MODULES), default=[])
    parser.add_argument("--cuda", action="store_true", help="Install the proposed pinned GPU/model stack; loads no weights")
    parser.add_argument("--gpu-profile", choices=("legacy", "hazard"), default="legacy", help="Hazard selects the unified Python 3.12 Qwen/SAM3 proposal")
    parser.add_argument("--torch-index-url", help="Explicit official CUDA wheel index compatible with the selected profile and GPU driver")
    parser.add_argument("--driver-version", help="Observed NVIDIA driver version; required for the hazard GPU proposal")
    parser.add_argument("--sam3-source", type=Path, help="Existing clean official SAM3 checkout at the pinned revision; supports offline installation")
    parser.add_argument("--offline", action="store_true", help="Install only from the supplied local wheelhouse")
    parser.add_argument("--wheelhouse", type=Path)
    parser.add_argument("--kernel-name")
    parser.add_argument("--kernel-display-name")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without creating or changing anything")
    return parser


def environment_variables(roots: RootPaths, *, offline: bool = False) -> dict[str, str]:
    """Keep data, runs and package/notebook caches in configurable roots."""
    cache = roots.cache_root
    values = {
        "TRAVERSABILITY_DATA_ROOT": str(roots.data_root),
        "TRAVERSABILITY_RUN_ROOT": str(roots.run_root),
        "TRAVERSABILITY_CACHE_ROOT": str(cache),
        "HF_HOME": str(cache / "huggingface"),
        "HF_HUB_CACHE": str(cache / "huggingface/hub"),
        "HF_XET_CACHE": str(cache / "huggingface/xet"),
        "HF_DATASETS_CACHE": str(cache / "huggingface/datasets"),
        "TORCH_HOME": str(cache / "torch"),
        "XDG_CACHE_HOME": str(cache),
        "PIP_CACHE_DIR": str(cache / "pip"),
        "JUPYTER_DATA_DIR": str(cache / "jupyter/data"),
        "JUPYTER_CONFIG_DIR": str(cache / "jupyter/config"),
        "JUPYTER_RUNTIME_DIR": str(cache / "jupyter/runtime"),
        "TMPDIR": str(cache / "tmp"),
        "TEMP": str(cache / "tmp"),
        "TMP": str(cache / "tmp"),
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_CONFIG_FILE": os.devnull,
        "PYTHONNOUSERSITE": "1",
    }
    if offline:
        values.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return values


def environment_python(env_dir: Path) -> Path:
    return env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _absolute(path: Path, repository: Path) -> Path:
    expanded = path.expanduser()
    return (expanded if expanded.is_absolute() else repository / expanded).resolve()


def _validate_offline_requirements(path: Path, seen: set[Path] | None = None) -> None:
    # --no-index alone does not disable direct-URL or VCS requirements.
    seen = set() if seen is None else seen
    path = path.resolve()
    if path in seen:
        return
    seen.add(path)
    for line in path.read_text(encoding="utf-8").splitlines():
        content = line.split("#", 1)[0].strip()
        if not content:
            continue
        if re.search(r"(?:https?|git|hg|svn|bzr|ssh)(?:://|\+)", content, re.IGNORECASE):
            raise ValueError(f"Offline installation cannot consume remote requirements in {path.name}; supply local wheels instead")
        match = re.match(r"(?:-r\s*|--requirement(?:=|\s+)|-c\s*|--constraint(?:=|\s+))(.+)$", content)
        if match:
            included = Path(match.group(1).strip().strip("\"'"))
            _validate_offline_requirements(included if included.is_absolute() else path.parent / included, seen)


def _cuda_build(index_url: str | None, profile: str = "legacy") -> str:
    if not index_url:
        raise ValueError("--cuda requires an explicit --torch-index-url chosen for the GPU driver")
    parsed = urlparse(index_url)
    build = parsed.path.rstrip("/").rsplit("/", 1)[-1]
    if (
        parsed.scheme != "https"
        or parsed.netloc != "download.pytorch.org"
        or parsed.path.rstrip("/") != f"/whl/{build}"
        or parsed.query
        or parsed.fragment
        or build not in (HAZARD_CUDA_WHEELS if profile == "hazard" else SUPPORTED_CUDA_WHEELS)
    ):
        choices = ", ".join(HAZARD_CUDA_WHEELS if profile == "hazard" else SUPPORTED_CUDA_WHEELS)
        raise ValueError(f"Use an official https://download.pytorch.org/whl/ index ({choices}) for the {profile} proposal")
    return build


def _driver_compatibility(version: str | None, build: str) -> dict:
    if not version or not re.fullmatch(r"\d+(?:\.\d+){1,2}", version):
        raise ValueError("Hazard GPU setup requires --driver-version from the target host's current nvidia-smi")
    parts = tuple(int(part) for part in version.split("."))
    # CUDA family minimums from NVIDIA's compatibility table. This is a
    # prerequisite check, not proof that a model/kernel or its PTX will execute.
    minimum = (580, 0) if build.startswith("cu13") else ((528, 33) if os.name == "nt" else (525, 60))
    if parts < minimum:
        raise ValueError(f"Driver {version} is below the CUDA family minimum for {build}; no automatic wheel fallback")
    return {"observed_driver_version": version, "cuda_build": build,
            "family_minimum": ".".join(map(str, minimum)), "status": "prerequisite_only; GPU execution untested"}


def build_plan(args: argparse.Namespace) -> dict:
    """Resolve setup inputs and commands without installing or writing files."""
    repository = args.repo_root.expanduser().resolve()
    if not (repository / "pyproject.toml").is_file():
        raise ValueError(f"No pyproject.toml in repository {repository}")
    if sys.version_info < (3, 11):
        raise ValueError("Run setup with Python 3.11 or newer; unified hazard GPU preparation uses Python 3.12")
    hazard_gpu = args.cuda and args.gpu_profile == "hazard"
    if args.gpu_profile != "legacy" and not args.cuda:
        raise ValueError("--gpu-profile requires explicit --cuda")
    if hazard_gpu and not (3, 12) <= sys.version_info[:2] < (3, 13):
        raise ValueError("Use Python 3.12 for the unified hazard GPU proposal: NumPy 1.26.4 has no Python 3.13+ wheels")
    env_dir = _absolute(args.env_dir or Path(".venv-hazard-gpu" if hazard_gpu else ".venv-gpu" if args.cuda else ".venv-cpu"), repository)
    if env_dir == repository or env_dir in repository.parents:
        raise ValueError("The environment must have its own directory, not the repository or an ancestor")
    roots = resolve_roots(repository, data_root=args.data_root, run_root=args.run_root, cache_root=args.cache_root)
    cuda_build = _cuda_build(args.torch_index_url, args.gpu_profile) if args.cuda else None
    driver_check = _driver_compatibility(args.driver_version, cuda_build) if hazard_gpu else None
    if args.torch_index_url and not args.cuda:
        raise ValueError("--torch-index-url requires --cuda")
    if "inference" in args.with_module and not args.cuda:
        raise ValueError("Inference dependencies require explicit --cuda/--torch-index-url; fake-backend CPU tests need no inference dependencies")
    if any(module in args.with_module for module in ("hazard_inference", "hazard_segmentation")) and not hazard_gpu:
        raise ValueError("Hazard model fragments require --cuda --gpu-profile hazard; CPU fixtures need only hazard_data/hazard_evaluation")
    if hazard_gpu and "inference" in args.with_module:
        raise ValueError("Legacy inference pins belong in their own environment; select hazard_inference in the unified hazard profile")
    if args.driver_version and not hazard_gpu:
        raise ValueError("--driver-version is used with --cuda --gpu-profile hazard")
    sam3_source = _absolute(args.sam3_source, repository) if args.sam3_source else None
    if sam3_source and (not hazard_gpu or not (sam3_source / "sam3/model_builder.py").is_file()):
        raise ValueError("--sam3-source requires the hazard GPU profile and an existing official SAM3 checkout")
    if hazard_gpu and args.offline and not sam3_source:
        raise ValueError("Offline hazard GPU setup needs --sam3-source at the pinned official revision; no network or model fallback")
    if args.kernel_name and not args.notebooks:
        raise ValueError("--kernel-name requires --notebooks")
    wheelhouse = _absolute(args.wheelhouse, repository) if args.wheelhouse is not None else None
    if args.offline and (wheelhouse is None or not wheelhouse.is_dir()):
        raise ValueError("--offline requires an existing --wheelhouse directory containing build and requested dependency wheels")
    if wheelhouse is not None and not wheelhouse.is_dir():
        raise ValueError(f"Wheelhouse directory does not exist: {wheelhouse}")

    requirements = [repository / "requirements/base.txt"]
    if args.notebooks:
        requirements.append(repository / "requirements/notebooks.txt")
    for required in requirements:
        if not required.is_file():
            raise ValueError(f"Required setup fragment is missing: {required}")
    implicit = ["hazard_inference", "hazard_segmentation"] if hazard_gpu else ["inference"] if args.cuda else []
    modules = list(dict.fromkeys([*args.with_module, *implicit]))
    pending_modules = []
    for module in modules:
        fragment = repository / f"requirements/{module}.txt"
        if fragment.is_file():
            requirements.append(fragment)
        else:
            if module in HAZARD_MODULES:
                raise ValueError(f"Requested hazard fragment is missing: {fragment}")
            pending_modules.append(module)
    if args.offline:
        for fragment in requirements:
            _validate_offline_requirements(fragment)

    python = str(environment_python(env_dir))
    pip = [python, "-m", "pip", "install", "--no-input", "--disable-pip-version-check"]
    source_flags = []
    if args.offline:
        source_flags.extend(["--no-index", "--only-binary=:all:"])
    if wheelhouse is not None:
        source_flags.extend(["--find-links", str(wheelhouse)])
    constraints = []
    if args.cuda:
        gpu_versions = HAZARD_GPU_VERSIONS if hazard_gpu else GPU_VERSIONS
        constraints = [
            f"torch=={gpu_versions['torch']}+{cuda_build}",
            f"torchvision=={gpu_versions['torchvision']}+{cuda_build}",
            *[f"{name}=={version}" for name, version in gpu_versions.items() if name not in ("torch", "torchvision")],
        ]
    constraint_flags = ["-c", str(env_dir / CONSTRAINTS_NAME)] if constraints else []
    commands = [{"label": "base", "argv": [*pip, *source_flags, "-r", str(requirements[0])]}]
    if args.cuda:
        cuda_source = source_flags if args.offline else [*source_flags, "--index-url", args.torch_index_url]
        commands.append({"label": "cuda", "argv": [*pip, *cuda_source, *constraint_flags, *constraints[:2]]})
        commands.append({"label": "models", "argv": [*pip, *source_flags, *constraint_flags, *constraints[2:]]})
    for fragment in requirements[1:]:
        commands.append({"label": f"fragment:{fragment.name}", "argv": [*pip, *source_flags, *constraint_flags, "-r", str(fragment)]})
    if hazard_gpu:
        sam3_requirement = ["-e", str(sam3_source)] if sam3_source else [SAM3_URL]
        commands.append({"label": "sam3", "argv": [*pip, *source_flags, *constraint_flags, "--no-build-isolation", "--no-deps", *sam3_requirement]})
    commands.append({
        "label": "editable",
        "argv": [*pip, *source_flags, "--no-build-isolation", "--no-deps", "-e", str(repository)],
    })
    commands.append({"label": "check", "argv": [python, "-m", "pip", "check"]})
    name = args.kernel_name or "traversability-" + hashlib.sha256(str(env_dir).encode()).hexdigest()[:10]
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise ValueError("Kernel names may contain only letters, numbers, dots, underscores and hyphens")
    if args.notebooks:
        commands.append({
            "label": "kernel",
            "argv": [python, "-m", "ipykernel", "install", "--prefix", str(env_dir), "--name", name,
                     "--display-name", args.kernel_display_name or f"Python (Traversability {'GPU' if args.cuda else 'CPU'})"],
        })
    return {
        "repo_root": str(repository),
        "env_dir": str(env_dir),
        "roots": {name: str(getattr(roots, name)) for name in ("data_root", "run_root", "cache_root")},
        "environment": environment_variables(roots, offline=args.offline),
        "profile": "hazard_cuda" if hazard_gpu else "cuda" if args.cuda else "cpu",
        "minimum_python": [3, 12] if hazard_gpu else [3, 11],
        "gpu_profile": args.gpu_profile if args.cuda else None,
        "driver_compatibility": driver_check,
        "sam3_source": str(sam3_source) if sam3_source else None,
        "sam3_revision": SAM3_REVISION if hazard_gpu else None,
        "version_status": "proposed; installed versions are recorded after setup, not GPU-benchmark validation",
        "cuda_build": cuda_build,
        "constraints": constraints,
        "offline": args.offline,
        "notebooks": args.notebooks,
        "kernel_name": name if args.notebooks else None,
        "requirements": [str(path) for path in requirements],
        "pending_modules": pending_modules,
        "commands": commands,
    }


def _existing_environment(env_dir: Path) -> None:
    if not env_dir.exists():
        return
    if not env_dir.is_dir():
        raise ValueError(f"Environment path is not a directory: {env_dir}")
    config = env_dir / "pyvenv.cfg"
    if not config.is_file():
        if any(env_dir.iterdir()):
            raise ValueError("Refusing to install into a nonempty directory without pyvenv.cfg; choose a new isolated environment")
        return
    if re.search(r"include-system-site-packages\s*=\s*true", config.read_text(encoding="utf-8"), re.IGNORECASE):
        raise ValueError("Refusing an environment with system site packages; choose a new isolated environment")
    if not environment_python(env_dir).is_file():
        raise ValueError("Existing environment has no Python interpreter; choose a new environment directory")


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _check_kernel_collision(env_dir: Path, name: str) -> None:
    spec = env_dir / "share/jupyter/kernels" / name / "kernel.json"
    if spec.exists():
        existing = json.loads(spec.read_text(encoding="utf-8"))
        argv = existing.get("argv", [])
        if not argv or Path(argv[0]).resolve() != environment_python(env_dir).resolve():
            raise ValueError(f"Refusing to overwrite kernel {name} belonging to another interpreter")


def execute_plan(plan: dict) -> None:
    """Install only into the requested isolated environment; never launch a job."""
    env_dir = Path(plan["env_dir"])
    _existing_environment(env_dir)
    previous_manifest = env_dir / MANIFEST_NAME
    if previous_manifest.is_file():
        previous_profile = json.loads(previous_manifest.read_text(encoding="utf-8")).get("profile")
        if {previous_profile, plan["profile"]} == {"cuda", "hazard_cuda"}:
            raise ValueError("Legacy and unified hazard GPU profiles need separate isolated environments; choose a new --env-dir")
    if plan.get("sam3_source"):
        source = plan["sam3_source"]
        revision = subprocess.run(["git", "-C", source, "rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", source, "status", "--porcelain", "--untracked-files=no"], check=True, text=True, capture_output=True).stdout.strip()
        if revision != SAM3_REVISION or dirty:
            raise ValueError("SAM3 source must be clean at the pinned official revision; no checkout changes or fallback are performed")
    if plan["notebooks"]:
        _check_kernel_collision(env_dir, plan["kernel_name"])
    Path(plan["environment"]["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    if not (env_dir / "pyvenv.cfg").is_file():
        # venv's ensurepip subprocess also needs bulk-storage temporary files.
        temporary_keys = ("TMPDIR", "TEMP", "TMP")
        previous = {key: os.environ.get(key) for key in temporary_keys}
        try:
            os.environ.update({key: plan["environment"][key] for key in temporary_keys})
            venv.EnvBuilder(with_pip=True, system_site_packages=False).create(env_dir)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    runtime_environment = dict(os.environ)
    # Ambient pip destinations/Python overrides must not escape the chosen venv.
    for key in ("PIP_TARGET", "PIP_PREFIX", "PIP_USER", "PYTHONHOME", "PYTHONPATH"):
        runtime_environment.pop(key, None)
    runtime_environment.update(plan["environment"])
    python = str(environment_python(env_dir))
    probe = subprocess.run(
        [python, "-c", "import json,sys; print(json.dumps({'prefix':sys.prefix,'base_prefix':sys.base_prefix,'version':list(sys.version_info[:3])}))"],
        check=True, text=True, capture_output=True, env=runtime_environment,
    )
    identity = json.loads(probe.stdout)
    minimum_python = tuple(plan.get("minimum_python", [3, 11]))
    if Path(identity["prefix"]).resolve() != env_dir or identity["prefix"] == identity["base_prefix"] or tuple(identity["version"]) < minimum_python:
        raise ValueError(f"Target Python is not an isolated Python {'.'.join(map(str, minimum_python))}+ environment")
    if plan.get("gpu_profile") == "hazard" and tuple(identity["version"][:2]) != (3, 12):
        raise ValueError("The selected hazard GPU environment must use Python 3.12 for the pinned NumPy wheel")
    for name, root in plan["roots"].items():
        path = Path(root)
        if path.exists() and not path.is_dir():
            raise ValueError(f"{name} is a file; choose a directory")
        path.mkdir(parents=True, exist_ok=True)
    for key in ("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE", "HF_DATASETS_CACHE", "TORCH_HOME", "PIP_CACHE_DIR", "JUPYTER_DATA_DIR", "JUPYTER_CONFIG_DIR", "JUPYTER_RUNTIME_DIR", "TMPDIR"):
        Path(plan["environment"][key]).mkdir(parents=True, exist_ok=True)
    if plan["constraints"]:
        (env_dir / CONSTRAINTS_NAME).write_text("\n".join(plan["constraints"]) + "\n", encoding="utf-8")
    existing_manifest = env_dir / MANIFEST_NAME
    if existing_manifest.exists():
        pending = {key: value for key, value in plan.items() if key != "commands"}
        pending["status"] = "setup_in_progress"
        _write_json(existing_manifest, pending)
    for command in plan["commands"]:
        print(f"Setup: {command['label']}", flush=True)
        subprocess.run(command["argv"], check=True, env=runtime_environment)
        if command["label"] == "kernel":
            spec = env_dir / "share/jupyter/kernels" / plan["kernel_name"] / "kernel.json"
            content = json.loads(spec.read_text(encoding="utf-8"))
            content.setdefault("env", {}).update(plan["environment"])
            _write_json(spec, content)
    if plan["cuda_build"]:
        result = subprocess.run(
            [python, "-c", "import torch; print(torch.version.cuda or '')"],
            check=True, text=True, capture_output=True, env=runtime_environment,
        )
        wheel_versions = HAZARD_CUDA_WHEELS if plan.get("gpu_profile") == "hazard" else SUPPORTED_CUDA_WHEELS
        if result.stdout.strip() != wheel_versions[plan["cuda_build"]]:
            raise ValueError("Installed Torch CUDA runtime differs from the explicit wheel index; no automatic fallback is permitted")
    versions = subprocess.run([python, "-m", "pip", "freeze", "--all"], check=True, text=True, capture_output=True, env=runtime_environment)
    manifest = {key: value for key, value in plan.items() if key != "commands"}
    manifest.update(installed_at=datetime.now(timezone.utc).isoformat(), python=identity, installed_packages=versions.stdout.splitlines(), status="installed")
    _write_json(env_dir / MANIFEST_NAME, manifest)
    print(f"Environment ready: {env_dir}")
    if plan["pending_modules"]:
        print("Peer requirements still pending: " + ", ".join(plan["pending_modules"]))
    print("No model weights, notebook server or benchmark jobs were started.")


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
        plan = build_plan(args)
        if args.dry_run:
            print(json.dumps(plan, indent=2))
        else:
            execute_plan(plan)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"Setup failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
