"""Read-only environment and path checks; importing this module loads no models."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import shutil
import sys
from typing import Any


@dataclass(frozen=True)
class RootPaths:
    """External input, generated-run and cache locations."""

    data_root: Path
    run_root: Path
    cache_root: Path


def _repository_root(repo_root: str | os.PathLike[str] | None) -> Path:
    if repo_root is not None:
        return Path(repo_root).expanduser().resolve()
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (
            (candidate / "src").is_dir() or (candidate / "docs/interfaces.md").is_file()
        ):
            return candidate
    source_root = Path(__file__).resolve().parents[2]
    return source_root if (source_root / "pyproject.toml").is_file() else current


def resolve_roots(
    repo_root: str | os.PathLike[str] | None = None,
    *,
    data_root: str | os.PathLike[str] | None = None,
    run_root: str | os.PathLike[str] | None = None,
    cache_root: str | os.PathLike[str] | None = None,
) -> RootPaths:
    """Resolve explicit roots, then environment overrides, then repository defaults."""
    repository = _repository_root(repo_root)

    def resolve(value: str | os.PathLike[str] | None, variable: str, default: str) -> Path:
        selected = value if value is not None else os.environ.get(variable) or default
        if not os.fspath(selected).strip():
            raise ValueError(f"{variable} must be a nonempty path")
        path = Path(selected).expanduser()
        if not path.is_absolute():
            path = repository / path
        return path.resolve()

    return RootPaths(
        resolve(data_root, "TRAVERSABILITY_DATA_ROOT", "data"),
        resolve(run_root, "TRAVERSABILITY_RUN_ROOT", "runs"),
        resolve(cache_root, "TRAVERSABILITY_CACHE_ROOT", ".cache"),
    )


def _path_report(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "is_directory": path.is_dir(),
        "readable": None,
        "writable": None,
        "disk_probe_path": None,
        "disk_total_bytes": None,
        "disk_free_bytes": None,
        "error": None,
    }
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if probe.is_file():
        probe = probe.parent
    if not probe.exists():
        report["error"] = "No existing ancestor is accessible; check the configured root."
        return report
    report["disk_probe_path"] = str(probe)
    report["readable"] = os.access(path if path.exists() else probe, os.R_OK)
    report["writable"] = os.access(path if path.exists() else probe, os.W_OK)
    try:
        disk = shutil.disk_usage(probe)
        report.update(disk_total_bytes=disk.total, disk_free_bytes=disk.free)
    except OSError as error:
        report["error"] = f"{type(error).__name__}: {error}"
    if path.exists() and not path.is_dir():
        report["error"] = "Configured root exists as a file; choose a directory."
    return report


_OPTIONAL_PACKAGES = {
    "traversability_data": "embodiedai-traversability",
    "traversability_inference": "embodiedai-traversability",
    "traversability_evaluation": "embodiedai-traversability",
    "PIL": "Pillow",
    "numpy": "numpy",
    "torch": "torch",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "bitsandbytes": "bitsandbytes",
    "ipykernel": "ipykernel",
    "jupyterlab": "jupyterlab",
}


def _package_report(module: str, distribution: str) -> dict[str, Any]:
    report: dict[str, Any] = {"available": False, "version": None, "error": None}
    try:
        report["available"] = importlib.util.find_spec(module) is not None
    except (ImportError, ValueError, AttributeError) as error:
        report["error"] = f"{type(error).__name__}: {error}"
    try:
        report["version"] = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        pass
    return report


def _gpu_report(torch_available: bool) -> dict[str, Any]:
    report: dict[str, Any] = {
        "checked": True,
        "torch_available": torch_available,
        "cuda_available": None,
        "device_count": None,
        "devices": [],
        "error": None,
    }
    if not torch_available:
        report["error"] = "PyTorch is optional; install the explicit CUDA profile to check it."
        return report
    try:
        torch = importlib.import_module("torch")
        report["torch_version"] = str(torch.__version__)
        report["compiled_cuda_version"] = torch.version.cuda
        available = bool(torch.cuda.is_available())
        report["cuda_available"] = available
        report["device_count"] = int(torch.cuda.device_count()) if available else 0
        if available:
            for index in range(report["device_count"]):
                device = torch.cuda.get_device_properties(index)
                report["devices"].append(
                    {"index": index, "name": device.name, "total_memory_bytes": int(device.total_memory)}
                )
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    return report


def environment_report(roots: RootPaths, *, gpu_check: bool = True) -> dict[str, Any]:
    """Check paths and software without downloads, model loads or tensor allocations."""
    packages = {name: _package_report(name, distribution) for name, distribution in _OPTIONAL_PACKAGES.items()}
    gpu = (
        _gpu_report(packages["torch"]["available"])
        if gpu_check
        else {
            "checked": False,
            "torch_available": packages["torch"]["available"],
            "cuda_available": None,
            "device_count": None,
            "devices": [],
            "error": None,
        }
    )
    return {
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
            "supported": sys.version_info >= (3, 11),
        },
        "packages": packages,
        "paths": {name: _path_report(getattr(roots, name)) for name in ("data_root", "run_root", "cache_root")},
        "gpu": gpu,
        "notes": [
            "Package availability checks resolve imports; optional model libraries are not imported.",
            "GPU checks query PyTorch hardware availability and properties without allocating tensors.",
            "Writability uses access checks, not file creation; missing-root disk values refer to its existing ancestor.",
        ],
    }
