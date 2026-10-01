"""Explicit optional SAM2 preprocessing; importing this module loads no model."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import subprocess
import tempfile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from .masks import _is_fixture, import_sam_masks
from .storage import atomic_json, data_path, exclusive_lock, hash_file, publish_file, read_json, read_jsonl


OFFICIAL_REPOSITORY = "https://github.com/facebookresearch/sam2"
VERIFIED_OFFICIAL_REVISION = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
VERIFIED_ON = "2026-10-01"
CHECKPOINT_BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"
DEFAULT_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
_CHECKPOINTS = {
    "tiny": "sam2.1_hiera_tiny.pt",
    "small": "sam2.1_hiera_small.pt",
    "base_plus": "sam2.1_hiera_base_plus.pt",
    "large": "sam2.1_hiera_large.pt",
}
_GENERATOR_DEFAULTS = {
    "points_per_side": 32,
    "points_per_batch": 64,
    "pred_iou_thresh": 0.8,
    "stability_score_thresh": 0.95,
    "stability_score_offset": 1.0,
    "mask_threshold": 0.0,
    "box_nms_thresh": 0.7,
    "crop_n_layers": 0,
    "crop_nms_thresh": 0.7,
    "crop_overlap_ratio": 512 / 1500,
    "crop_n_points_downscale_factor": 1,
    "point_grids": None,
    "min_mask_region_area": 0,
    "output_mode": "binary_mask",
    "use_m2m": False,
    "multimask_output": True,
}


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _installed_revision(package_dir: Path):
    try:
        direct = importlib.metadata.distribution("SAM-2").read_text("direct_url.json")
        if direct:
            revision = json.loads(direct).get("vcs_info", {}).get("commit_id")
            if revision:
                return revision
    except (importlib.metadata.PackageNotFoundError, ValueError):
        pass
    try:
        root_result = subprocess.run(
            ["git", "-C", str(package_dir), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if root_result.returncode != 0:
            return None
        source_root = Path(root_result.stdout.strip()).resolve()
        # A venv inside another Git checkout must not inherit that checkout's SHA.
        if (source_root / "sam2").resolve() != package_dir.resolve():
            return None
        result = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


class _SAM2Generator:
    def __init__(self, generator, torch_module, metadata):
        self._generator = generator
        self._torch = torch_module
        self.metadata = metadata

    def generate(self, rgb):
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("SAM2 requires original-size HWC uint8 RGB")
        with self._torch.inference_mode():
            return self._generator.generate(rgb)


def load_sam2_generator(
    checkpoint_path,
    model_config=DEFAULT_MODEL_CONFIG,
    device="cpu",
    generator_settings=None,
    sam2_revision=None,
):
    """Explicitly load the official generator from a local checkpoint.

    Install the official SAM-2 package in an isolated environment first. This
    function never downloads weights or installs dependencies.
    """
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise FileNotFoundError(
            f"SAM2 checkpoint missing: {checkpoint}. Explicitly download one official "
            "SAM2.1 checkpoint or supply an existing local .pt file."
        )
    try:
        import torch
        import sam2
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except ImportError as error:
        raise ImportError(
            "Optional SAM2 is not installed. Follow https://github.com/facebookresearch/sam2 "
            "in an isolated Python>=3.10 environment with torch>=2.5.1 and torchvision>=0.20.1; "
            "Windows guidance recommends Ubuntu under WSL. Existing aligned masks can be imported instead."
        ) from error
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable; choose device='cpu' or a GPU environment")
    settings = dict(_GENERATOR_DEFAULTS)
    supplied = dict(generator_settings or {})
    unknown = set(supplied) - set(settings)
    if unknown:
        raise ValueError(f"Unsupported SAM2 generator settings: {sorted(unknown)}")
    settings.update(supplied)
    if settings["output_mode"] != "binary_mask":
        raise ValueError("SAM2 preprocessing requires output_mode='binary_mask'")
    json.dumps(_jsonable(settings), allow_nan=False)
    model = build_sam2(model_config, str(checkpoint), device=device, apply_postprocessing=False)
    generator = SAM2AutomaticMaskGenerator(model, **settings)
    package_dir = Path(sam2.__file__).resolve().parent
    config_path = package_dir / model_config
    if Path(model_config).is_file():
        config_path = Path(model_config)
    metadata = {
        "implementation": "official_sam2",
        "origin": "sam2_automatic",
        "official_repository": OFFICIAL_REPOSITORY,
        "instructions_verified_on": VERIFIED_ON,
        "instructions_verified_revision": VERIFIED_OFFICIAL_REVISION,
        "installed_code_revision": _installed_revision(package_dir),
        "declared_code_revision": sam2_revision,
        "checkpoint_name": checkpoint.name,
        "checkpoint_sha256": hash_file(checkpoint),
        "model_config": str(model_config),
        "model_config_sha256": hash_file(config_path) if config_path.is_file() else None,
        "device": str(device),
        "dtype": "float32",
        "autocast": False,
        "apply_postprocessing": False,
        "generator_settings": _jsonable(settings),
        "settings_basis": "official defaults originally chosen for HieraL; not calibrated for this dataset",
        "preprocessing": {
            "input": "original-size HWC uint8 RGB; no external resizing or EXIF transform",
            "internal_resolution": getattr(model, "image_size", None),
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "mask_coordinates": "upsampled and uncropped to original RGB H,W",
        },
        "versions": {name: _version(name) for name in ("SAM-2", "torch", "torchvision", "numpy", "Pillow")},
    }
    return _SAM2Generator(generator, torch, metadata)


def generate_sam2_regions(run_dir, data_root, generator, *, frame_ids=None) -> list[dict]:
    """Explicitly generate and import all masks without semantic-label inputs.

    Existing generated frames are reused only after verifying saved RGB/mask
    hashes. A fixture generator is accepted only in a marked fixture run.
    """
    run_dir, data_root = Path(run_dir), Path(data_root)
    frames = read_jsonl(run_dir / "frames.jsonl")
    frame_by_id = {frame["frame_id"]: frame for frame in frames}
    requested = list(frame_by_id) if frame_ids is None else list(frame_ids)
    if len(set(requested)) != len(requested) or not set(requested).issubset(frame_by_id):
        raise ValueError("frame_ids must contain unique known frame IDs")
    provenance = dict(getattr(generator, "metadata", {}))
    fixture = provenance.get("implementation") != "official_sam2"
    if fixture and not _is_fixture(run_dir, frames):
        raise ValueError("Injected generators are allowed only in a marked fixture run")
    origin = "synthetic_fixture" if fixture else "sam2_automatic"
    if fixture:
        provenance = {**provenance, "implementation": "synthetic_fixture", "is_fixture": True}
    json.dumps(_jsonable(provenance), allow_nan=False)
    result = []
    for frame_id in requested:
        frame = frame_by_id[frame_id]
        image_path = data_path(data_root, frame["image_path"])
        image_hash = hash_file(image_path)
        saved = read_json(run_dir / "metadata" / "sam2.json", default={}).get("frames", {}).get(frame_id)
        if saved is not None:
            if saved["image_sha256"] != image_hash or saved["generator"] != provenance:
                raise ValueError(f"Saved SAM2 generator/RGB differs for {frame_id}")
            regions = {item["region_id"]: item for item in read_jsonl(run_dir / "regions.jsonl")}
            for region_id, mask_hash in saved["mask_hashes"].items():
                region = regions.get(region_id)
                if region is None or hash_file(data_path(data_root, region["mask_path"])) != mask_hash:
                    raise ValueError(f"Saved SAM2 mask differs for {region_id}")
                result.append(region)
            continue
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        outputs = list(generator.generate(rgb))
        records = []
        with tempfile.TemporaryDirectory(prefix=".sam2-", dir=data_root) as directory:
            for index, output in enumerate(outputs):
                if not isinstance(output, dict) or not isinstance(output.get("segmentation"), np.ndarray):
                    raise ValueError("SAM2 generator must return binary segmentation arrays")
                mask = output["segmentation"]
                if mask.shape != rgb.shape[:2] or mask.ndim != 2:
                    raise ValueError(f"SAM2 output dimensions do not match original RGB for {frame_id}")
                if not np.all(np.isin(mask, (0, 1, False, True))):
                    raise ValueError("SAM2 segmentation array must be binary")
                if not mask.any():
                    raise ValueError("SAM2 returned an empty region")
                source_path = Path(directory) / f"{index:05d}.png"
                Image.fromarray(mask.astype(np.uint8) * 255).save(source_path, format="PNG")
                output_details = {key: _jsonable(value) for key, value in output.items() if key != "segmentation"}
                records.append({
                    "region_id": f"{frame_id}:sam2:{index:05d}",
                    "frame_id": frame_id,
                    "mask_path": str(source_path.resolve()),
                    "image_sha256": image_hash,
                    "generator": {"settings": _jsonable(provenance), "output": output_details},
                })
            imported = import_sam_masks(run_dir, data_root, records, mask_origin=origin)
        details = {
            "image_sha256": image_hash,
            "generator": _jsonable(provenance),
            "mask_hashes": {region["region_id"]: hash_file(data_path(data_root, region["mask_path"])) for region in imported},
            "mask_count": len(imported),
        }
        with exclusive_lock(run_dir / ".data.lock"):
            metadata = read_json(run_dir / "metadata" / "sam2.json", default={"frames": {}})
            previous = metadata.setdefault("frames", {}).get(frame_id)
            if previous is not None and previous != details:
                raise ValueError(f"Concurrent SAM2 generation differs for {frame_id}")
            metadata["frames"][frame_id] = details
            atomic_json(run_dir / "metadata" / "sam2.json", metadata)
        result.extend(imported)
    return result


def download_sam2_checkpoint(destination, *, model="tiny", allow_download=False, expected_sha256=None) -> Path:
    """Explicitly download one official checkpoint through a temporary file.

    Existing completed files are preserved. expected_sha256 is an optional
    caller-supplied checksum; the publisher does not provide one here.
    """
    if model not in _CHECKPOINTS:
        raise ValueError(f"Unknown SAM2.1 model {model!r}; choose {sorted(_CHECKPOINTS)}")
    destination = Path(destination)
    if destination.exists():
        if not destination.is_file() or destination.stat().st_size == 0:
            raise ValueError(f"Existing checkpoint destination is not a completed file: {destination}")
        if expected_sha256 is not None and hash_file(destination) != expected_sha256:
            raise ValueError("Existing checkpoint checksum differs; completed data was preserved")
        return destination
    if not allow_download:
        raise RuntimeError("SAM2 checkpoint download is opt-in; set allow_download=True in an explicit notebook action")
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"{CHECKPOINT_BASE_URL}/{_CHECKPOINTS[model]}"
    with tempfile.NamedTemporaryFile(prefix=".checkpoint-", suffix=".part", dir=destination.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with urlopen(Request(url, headers={"User-Agent": "traversability-data/1"}), timeout=30) as response:
            if "text/html" in response.headers.get("Content-Type", "").lower():
                raise RuntimeError("Official checkpoint URL returned an HTML access/error page")
            expected_bytes = response.headers.get("Content-Length")
            with temporary_path.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        size = temporary_path.stat().st_size
        if not size or (expected_bytes is not None and size != int(expected_bytes)):
            raise RuntimeError("Checkpoint transfer was incomplete; no completed file was replaced")
        checksum = hash_file(temporary_path)
        if expected_sha256 is not None and checksum != expected_sha256:
            raise ValueError("Downloaded checkpoint checksum does not match expected_sha256")
        publish_file(temporary_path, destination)
        sidecar = destination.with_name(destination.name + ".metadata.json")
        if not sidecar.exists():
            atomic_json(sidecar, {"url": url, "sha256": checksum, "bytes": size, "model": model})
        return destination
    except (HTTPError, URLError, OSError) as error:
        raise RuntimeError(
            f"Official SAM2 checkpoint could not be downloaded from {url}: {error}. "
            "Existing data was preserved; retry explicitly or import a manually obtained local checkpoint."
        ) from error
    finally:
        temporary_path.unlink(missing_ok=True)
