"""Prepare a local-only SAM3 replay config from a completed geometry-only run.

Run in the isolated KTH pipeline environment. This helper verifies
existing assets/metadata and writes configuration; it loads no Torch/model,
connects to no server, downloads nothing and starts no inference job.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib
import importlib.metadata
from importlib.util import find_spec
import json
from pathlib import Path
import re
import sys

CONTRACT_ID = "semantic_mapping_v1"
PRESERVED_FIELDS = (
    "contract_id", "schema_version", "protocol_id", "mode", "sequence",
    "geometry", "geometry_identity", "fusion", "robot", "planning", "goals",
    "semantic_keyframe_ids", "runtime", "evaluation_grid", "hardware_budget",
)
_ROOT = Path(__file__).resolve().parents[2]
_ESTABLISHED_HAZARD_ROOT = Path("/home/jovyan/EmbodiedAI")


def _read(path):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(record, label):
    if (record.get("contract_id") != CONTRACT_ID
        or type(record.get("schema_version")) is not int or record["schema_version"] != 1):
        raise ValueError(f"{label} contract/schema mismatch")


def _versions():
    versions = {"python": sys.version.split()[0]}
    for name in ("torch", "torchvision", "numpy", "Pillow", "sam3", "timm",
                 "einops", "ftfy", "iopath", "huggingface-hub", "setuptools"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    if sys.version_info < (3, 12):
        raise ValueError("Official SAM3 requires isolated Python >=3.12")
    for name in ("torch", "torchvision", "numpy", "Pillow", "sam3", "timm", "einops",
                 "ftfy", "iopath", "huggingface-hub", "setuptools"):
        if versions[name] is None:
            raise ValueError(f"Missing installed SAM3 dependency metadata: {name}")
    numpy_version = re.match(r"(\d+)\.(\d+)", versions["numpy"])
    if numpy_version is None or not (1, 26) <= tuple(map(int, numpy_version.groups())) < (2, 0):
        raise ValueError("Pinned SAM3 requires NumPy >=1.26,<2; preserve the hazard stack")
    torch_version = re.match(r"(\d+)\.(\d+)", versions["torch"])
    if torch_version is None or tuple(map(int, torch_version.groups())) < (2, 7):
        raise ValueError("Official SAM3 requires Torch >=2.7")
    # The installed wheel suffix is metadata only; actual driver/CUDA availability
    # is checked by the executor and verified model loader before inference.
    versions["verification_scope"] = "distribution metadata only; no Torch import or GPU availability check"
    return versions


def _verify_sam_assets(settings):
    component_sources = str(_ROOT / "VLM_evaluation" / "src")
    if component_sources not in sys.path:
        sys.path.insert(0, component_sources)
    module = importlib.import_module("traversability_hazard_segmentation.sam3_adapter")
    configured = module._settings(settings)
    code = module._verify_code(configured)
    checkpoint, checkpoint_identity = module._resolve_checkpoint(configured)
    # Resolve the already cached file once. Inference uses this exact local path
    # rather than making another HF lookup, even when the original was a symlink.
    settings.update(cache_root=str(checkpoint.parent), checkpoint_path=checkpoint.name)
    spec = find_spec("sam3")
    return {"checkpoint": str(checkpoint), "checkpoint_verification": checkpoint_identity,
        "code_verification": code, "installed_sam3_source": list(spec.submodule_search_locations or []),
        "downloads": False, "model_loaded": False}


def build_replay_config(geometry_run, sequence_path, environment, *, cache_root=None,
                        sam_checkpoint=None, sam_source=None):
    if environment not in {"indoor", "outdoor"}:
        raise ValueError("Environment must be explicitly indoor or outdoor")
    geometry_run, sequence_path = Path(geometry_run).resolve(), Path(sequence_path).resolve()
    identity = _read(geometry_run / "run.json")
    base = _read(geometry_run / "config.resolved.json")
    sequence = _read(sequence_path)
    manifest = _read(geometry_run / "geometry" / "manifest.json")
    for label, record in (("Run", identity), ("Resolved config", base), ("Sequence", sequence), ("Geometry", manifest)):
        _identity(record, label)
    if identity.get("status") != "complete" or identity.get("pipeline_id") != "geometry_only":
        raise ValueError("Replay requires a completed geometry_only run")
    if manifest.get("status") != "complete":
        raise ValueError("Run geometry manifest is not complete")
    if any(record.get("fixture") is not False for record in (identity, base, sequence)):
        raise ValueError("KTH replay helper requires real, explicitly nonfixture inputs")
    if base.get("mode") != "quality_replay":
        raise ValueError("Saved-geometry replay requires the original quality_replay mode")
    missing = [key for key in PRESERVED_FIELDS if key not in base]
    if missing:
        raise ValueError("Resolved experiment is missing frozen fields: " + ", ".join(missing))
    # Shared digest recipe has ASCII escaping; all manifest IDs/paths must retain
    # their actual saved recipe. Reuse it rather than making a private identity.
    if str(_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(_ROOT / "src"))
    from pipeline_common.io import digest_json
    actual_sequence_digest = digest_json({key: value for key, value in sequence.items() if key != "manifest_digest"})
    if sequence.get("manifest_digest") != actual_sequence_digest:
        raise ValueError("Sequence manifest digest mismatch")
    for key in ("sequence_id", "split", "manifest_digest"):
        if base["sequence"].get(key) != sequence.get(key):
            raise ValueError(f"Sequence {key} differs from the frozen geometry run")
    cache = Path(manifest.get("cache_path", geometry_run / "geometry" / "cache")).resolve()
    cached_manifest = _read(cache / "manifest.json")
    _identity(cached_manifest, "Cached geometry")
    if cached_manifest.get("status") != "complete":
        raise ValueError("Verified common geometry cache is not complete")
    if digest_json(cached_manifest.get("input_identity")) != cached_manifest.get("input_fingerprint"):
        raise ValueError("Geometry cache input identity fingerprint mismatch")
    if cached_manifest.get("input_identity", {}).get("sequence_digest") != sequence["manifest_digest"]:
        raise ValueError("Geometry cache belongs to another sequence")
    from pipeline_common.geometry import geometry_settings_identity
    if cached_manifest.get("input_identity", {}).get("settings") != geometry_settings_identity(base["geometry"], False):
        raise ValueError("Resolved geometry settings differ from the verified cache")
    for key in ("geometry_fingerprint", "input_fingerprint", "processed_grid_id", "archive_sha256",
                "map_frame", "units", "up", "scale", "pose_revision"):
        if key not in manifest or cached_manifest.get(key) != manifest[key]:
            raise ValueError(f"Geometry cache/run {key} mismatch")
    if _file_digest(cache / "geometry.npz") != cached_manifest["archive_sha256"]:
        raise ValueError("Geometry archive bytes have changed")
    expected_geometry = base["geometry_identity"]
    for key in ("geometry_fingerprint", "input_fingerprint", "processed_grid_id", "units", "up", "scale", "pose_revision"):
        if expected_geometry.get(key) != manifest.get(key):
            raise ValueError(f"Resolved geometry {key} mismatch")
    selected = _read(_ROOT / "configs" / "pipelines" / f"ground_surface_{environment}.json")["pipeline"]
    selected = deepcopy(selected)
    selected["fixture"] = False
    sam = selected["sam3"]
    sam.update(fixture=False, enable_model_loading=True, allow_downloads=False, local_files_only=True,
        cache_root=str(Path(cache_root or _ESTABLISHED_HAZARD_ROOT / ".cache/kth").expanduser().resolve()),
        checkpoint_path=None, device=base["geometry"].get("device", "cuda:0"))
    if not re.fullmatch(r"cuda:(0|[1-9][0-9]*)", sam["device"]):
        raise ValueError("SAM must preserve the existing single indexed CUDA device")
    if sam_checkpoint is not None:
        checkpoint = Path(sam_checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError("Existing local SAM3 checkpoint is absent")
        sam.update(cache_root=str(checkpoint.parent), checkpoint_path=checkpoint.name)
    if sam_source is not None:
        sam["code_source_root"] = str(Path(sam_source).expanduser().resolve())
    runtime = _versions()
    assets = _verify_sam_assets(sam)
    prepared = deepcopy(base)
    prepared.update(pipeline_id="ground_surface", fixture=False, pipeline=selected, pipeline_config=selected)
    if isinstance(prepared.get("pipelines"), dict):
        prepared["pipelines"]["ground_surface"] = selected
    prepared.pop("config_digest", None)
    frozen = {key: base[key] for key in PRESERVED_FIELDS}
    if any(prepared[key] != value for key, value in frozen.items()):
        raise AssertionError("Surface configuration altered a frozen common experiment field")
    report = {"contract_id": CONTRACT_ID, "schema_version": 1, "status": "assets_verified",
        "pipeline_id": "ground_surface", "environment": environment, "prompt": selected["prompt"],
        "geometry_run": str(geometry_run), "sequence": str(sequence_path), "geometry_cache": str(cache),
        "frozen_common_fields": list(PRESERVED_FIELDS), "frozen_common_digest": _digest(frozen),
        "geometry_run_sha256": _file_digest(geometry_run / "run.json"),
        "geometry_config_sha256": _file_digest(geometry_run / "config.resolved.json"),
        "geometry_manifest_sha256": _file_digest(geometry_run / "geometry" / "manifest.json"),
        "sequence_manifest_sha256": _file_digest(sequence_path), "runtime": runtime, "sam_assets": assets,
        "model_settings": sam, "gpu_ready": None,
        "gpu_readiness_reason": "Executor must check current allocation/resources, CUDA >=12.6 and actual pinned SAM3 loading",
        "planning": "unchanged; unavailable without original verified metric scale/up and actual robot profile"}
    return prepared, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-run", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--environment", choices=("indoor", "outdoor"), required=True)
    parser.add_argument("--cache-root", help="Existing hazard cache root; default /home/jovyan/EmbodiedAI/.cache/kth")
    parser.add_argument("--sam-checkpoint", help="Existing local file; otherwise resolve the pinned HF cache offline")
    parser.add_argument("--sam-source", help="Optional clean official SAM3 checkout; otherwise verify installed VCS provenance")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    report_path = output.with_suffix(".preflight.json")
    if output == report_path:
        parser.error("Config and preflight report must have distinct paths")
    if output.exists() or report_path.exists():
        parser.error("Choose new config/preflight paths; existing files are preserved")
    try:
        configured, report = build_replay_config(args.geometry_run, args.sequence, args.environment,
            cache_root=args.cache_root, sam_checkpoint=args.sam_checkpoint, sam_source=args.sam_source)
    except Exception as error:
        parser.exit(2, f"Ground-surface configuration blocked: {type(error).__name__}: {error}\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(configured, indent=2, allow_nan=False) + "\n")
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "prompt": report["prompt"], "config": str(output),
        "preflight": str(report_path), "geometry_cache": report["geometry_cache"], "gpu_ready": None}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
