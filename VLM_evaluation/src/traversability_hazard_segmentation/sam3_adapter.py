"""Lazy, checkpoint-pinned official SAM3 image adapter (no reference inputs)."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Any


TASK_ID = "hazard_prompt_v1"
SCHEMA_VERSION = 1
CODE_REVISION = "2345a4ad109ac29c569da749c91d84f10dc08c40"
CHECKPOINT_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
CHECKPOINT_SHA256 = "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
CHECKPOINT_SIZE = 3450062241
_SOURCE_BLOBS = {
    "model_builder.py": "602906b325e5e3a99b9b15395ba16b1b2f5b2643",
    "model/sam3_image_processor.py": "f8439f50c5ee9e862938ad75be8ae06440e0b44f",
}
_LOAD_TOKEN = object()
_DEFAULTS = {
    "task_id": TASK_ID, "schema_version": SCHEMA_VERSION,
    "model_key": "sam3_image", "code_repository": "https://github.com/facebookresearch/sam3",
    "code_revision": CODE_REVISION, "checkpoint_repository": "facebook/sam3",
    "checkpoint_revision": CHECKPOINT_REVISION, "checkpoint_filename": "sam3.pt",
    "checkpoint_sha256": CHECKPOINT_SHA256, "checkpoint_size_bytes": CHECKPOINT_SIZE,
    "resolution": 1008, "confidence_threshold": 0.5, "mask_probability_threshold": 0.5,
    "device": "cuda:0", "fixture": False, "fixture_id": "sam3_processor_mock_v1",
    "enable_model_loading": False, "allow_downloads": False, "local_files_only": True,
    "checkpoint_path": None, "cache_root": None, "code_source_root": None,
    "policy_path": "configs/hazards/policy.json", "policy_id": "visible_avoid_concepts_v1",
}


class SegmenterUnavailable(RuntimeError):
    """A real model cannot be loaded under the requested, audited constraints."""

    def __init__(self, error_code: str, message: str):
        self.error_code = error_code
        super().__init__(f"{error_code}: {message}")


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ValueError("path must be a nonempty POSIX-relative path")
    parts = PurePosixPath(relative)
    if parts.is_absolute() or any(p in ("", ".", "..") for p in relative.split("/")):
        raise ValueError("path must not be absolute or contain empty/dot/parent components")
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    for part in relative.split("/"):
        if any(ord(c) < 32 or c in '<>:"|?*' for c in part) or part.endswith((".", " ")) or part.split(".")[0].upper() in reserved:
            raise ValueError("path contains a nonportable filename component")
    target = (root.resolve() / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("path escapes root")
    return target


def _component_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _runtime_root(value) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else _component_root() / path).resolve()


def _settings(settings: dict) -> dict:
    if not isinstance(settings, dict):
        raise ValueError("SAM settings must be a dictionary")
    configured = dict(_DEFAULTS)
    configured.update({k: v for k, v in settings.items() if k != "processor"})
    for key in ("fixture", "enable_model_loading", "allow_downloads", "local_files_only"):
        if type(configured[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    if configured["task_id"] != TASK_ID or type(configured["schema_version"]) is not int or configured["schema_version"] != SCHEMA_VERSION:
        raise ValueError("SAM task/schema identity mismatch")
    for key in ("model_key", "code_repository", "code_revision", "checkpoint_repository",
                "checkpoint_revision", "checkpoint_filename", "checkpoint_sha256", "checkpoint_size_bytes"):
        if configured[key] != _DEFAULTS[key]:
            raise SegmenterUnavailable("model_identity_mismatch", f"{key} must identify the verified official SAM3 image model")
    if not re.fullmatch(r"cuda:(0|[1-9][0-9]*)", str(configured["device"])):
        raise ValueError("device must name exactly one explicit indexed CUDA device, e.g. cuda:0")
    if type(configured["resolution"]) is not int or configured["resolution"] <= 0:
        raise ValueError("resolution must be a positive integer")
    for key in ("confidence_threshold", "mask_probability_threshold"):
        value = configured[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{key} must be a finite probability")
    if configured["allow_downloads"] and configured["local_files_only"]:
        raise ValueError("allow_downloads=true conflicts with local_files_only=true")
    runtime_policy = configured["policy_path"]
    if not isinstance(runtime_policy, (str, os.PathLike)):
        raise ValueError("policy_path must be a runtime path or POSIX-relative component path")
    runtime_policy = Path(runtime_policy).expanduser()
    policy_path = runtime_policy.resolve() if runtime_policy.is_absolute() else _portable_path(_component_root(), os.fspath(configured["policy_path"]))
    common = importlib.import_module(f"{__package__}.common")
    policy = common.read_json(policy_path)
    if not isinstance(policy, dict):
        raise ValueError("policy must be a JSON object")
    common.validate_task(policy, "policy")
    if policy.get("policy_id") != configured["policy_id"]:
        raise ValueError("policy task/schema/id mismatch")
    policy_hash, alias_hash = _fingerprint(policy), _fingerprint(policy["aliases"])
    for key, expected in (("policy_hash", policy_hash), ("alias_hash", alias_hash)):
        if key in configured and configured[key] != expected:
            raise ValueError(f"{key} mismatch")
        configured[key] = expected
    configured["policy_file_sha256"] = _sha256(policy_path)
    configured["vague_unusable_phrases"] = list(policy["vague_unusable_phrases"])
    configured["sam3_existing_mask_threshold"] = 0.5
    configured["mask_comparison"] = "strictly_greater_than"
    configured["postprocessing"] = "official_bilinear_original_size_then_sigmoid; no additional morphology"
    configured["execution_kind"] = "fixture" if configured["fixture"] else "real_model"
    if configured["fixture"]:
        if not isinstance(configured["fixture_id"], str) or not configured["fixture_id"].strip():
            raise ValueError("fixture_id must be a nonempty string")
        configured["fixture_identity"] = _fingerprint({"fixture_id": configured["fixture_id"], "model_key": "sam3_image", "synthetic": True})
    return configured


def _software() -> dict:
    versions = {"python": sys.version.split()[0]}
    for name in ("torch", "torchvision", "numpy", "Pillow", "sam3", "timm", "huggingface-hub", "iopath", "ftfy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _verify_code(settings: dict) -> dict:
    """Require immutable VCS provenance or a clean pinned source checkout."""
    spec = importlib.util.find_spec("sam3")
    if spec is None or not spec.submodule_search_locations:
        raise SegmenterUnavailable("dependency_unavailable", "Install the pinned official sam3 package in the isolated environment")
    package_root = Path(next(iter(spec.submodule_search_locations))).resolve()
    provenance = None
    try:
        raw = importlib.metadata.distribution("sam3").read_text("direct_url.json")
        direct = json.loads(raw) if raw else {}
        url = direct.get("url", "").removesuffix(".git").rstrip("/")
        if url == _DEFAULTS["code_repository"] and direct.get("vcs_info", {}).get("commit_id") == CODE_REVISION:
            provenance = "installed_official_vcs_commit"
    except (importlib.metadata.PackageNotFoundError, ValueError):
        pass
    checkout = settings.get("code_source_root")
    if checkout is not None:
        source = _runtime_root(checkout)
        if package_root != (source / "sam3").resolve():
            raise SegmenterUnavailable("code_provenance_mismatch", "code_source_root does not contain the imported SAM3 package")
        try:
            revision = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], check=True,
                                      capture_output=True, text=True).stdout.strip()
            dirty = subprocess.run(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
                                   check=True, capture_output=True, text=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise SegmenterUnavailable("code_provenance_unavailable", "Cannot verify the pinned SAM3 source checkout") from error
        if revision != CODE_REVISION or dirty:
            raise SegmenterUnavailable("code_provenance_mismatch", "SAM3 source checkout must be clean at the verified code revision")
        provenance = "clean_official_revision_checkout"
    if provenance is None:
        raise SegmenterUnavailable("code_provenance_required", "Supply official pinned pip VCS provenance or code_source_root for a clean pinned checkout")
    for relative, expected_blob in _SOURCE_BLOBS.items():
        try:
            source_bytes = (package_root / relative).read_bytes()
        except OSError as error:
            raise SegmenterUnavailable("code_source_unavailable", f"Cannot read pinned SAM3 source {relative}") from error
        actual = hashlib.sha1(b"blob " + str(len(source_bytes)).encode() + b"\0" + source_bytes).hexdigest()
        if actual != expected_blob:
            raise SegmenterUnavailable("code_source_mismatch", f"Official SAM3 source integrity failed for {relative}")
    return {"status": "verified", "method": provenance, "source_blobs": dict(_SOURCE_BLOBS)}


def _resolve_checkpoint(settings: dict) -> tuple[Path, dict]:
    cache_root = _runtime_root(settings.get("cache_root") or os.environ.get("TRAVERSABILITY_CACHE_ROOT") or ".cache")
    relative = settings.get("checkpoint_path")
    if relative is not None:
        path = _portable_path(cache_root, relative)
        method = "local_file_sha256"
    else:
        try:
            hub = importlib.import_module("huggingface_hub")
            path = Path(hub.hf_hub_download(repo_id="facebook/sam3", filename="sam3.pt",
                        revision=CHECKPOINT_REVISION, cache_dir=str(cache_root / "huggingface/hub"),
                        local_files_only=not settings["allow_downloads"] or settings["local_files_only"])).resolve()
            method = "pinned_hf_cache_sha256" if not settings["allow_downloads"] else "pinned_hf_download_sha256"
        except ImportError as error:
            raise SegmenterUnavailable("dependency_unavailable", "huggingface_hub is needed to resolve a pre-cached checkpoint") from error
        except Exception as error:
            raise SegmenterUnavailable("checkpoint_unavailable", "The pinned checkpoint is not cached/access-approved; supply a local file or explicitly enable approved downloads") from error
    if not path.is_file():
        raise SegmenterUnavailable("checkpoint_unavailable", "The configured checkpoint file does not exist")
    if path.stat().st_size != CHECKPOINT_SIZE or _sha256(path) != CHECKPOINT_SHA256:
        raise SegmenterUnavailable("checkpoint_integrity_mismatch", "Checkpoint size/SHA256 does not match the verified official immutable revision")
    return path, {"status": "verified", "method": method, "revision": CHECKPOINT_REVISION,
                  "sha256": CHECKPOINT_SHA256, "size_bytes": CHECKPOINT_SIZE}


def _image_modules():
    return importlib.import_module("numpy"), importlib.import_module("PIL.Image")


def _validate_input(frame: dict, prompts: list[str], data_root):
    if frame.get("input_contract_id") is not None:
        from pipeline_common.input_bridge import validate_model_input
        path = validate_model_input(frame, data_root)
        if not isinstance(prompts, list) or len(prompts) > 32 or any(not isinstance(p, str) or not p.strip() or len(p) > 80 for p in prompts):
            raise ValueError("prompts must contain at most 32 nonempty strings, each at most 80 characters")
        return path
    try:
        common = importlib.import_module(f"{__package__}.common")
    except ModuleNotFoundError as error:
        if error.name != f"{__package__}.common":
            raise
        common = None
    if common is not None:
        path = common.validate_frame(frame, data_root)
        common.validate_prompts(prompts)
        return Path(path)
    if not isinstance(frame, dict) or not isinstance(frame.get("frame_id"), str) or not frame["frame_id"]:
        raise ValueError("frame_id is required")
    for key in ("width", "height"):
        if type(frame.get(key)) is not int or frame[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not isinstance(prompts, list) or len(prompts) > 32 or any(not isinstance(p, str) or not p.strip() or len(p) > 80 for p in prompts):
        raise ValueError("prompts must contain at most 32 nonempty strings, each at most 80 characters")
    if not re.fullmatch(r"[0-9a-f]{64}", str(frame.get("image_sha256", ""))):
        raise ValueError("image_sha256 is required")
    path = _portable_path(Path(data_root), frame["image_path"])
    if _sha256(path) != frame["image_sha256"]:
        raise ValueError("image_sha256 mismatch")
    return path


def _cpu_array(value, np):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    # NumPy cannot represent Torch BF16. Preserve its values in float32 before
    # capturing autocast probabilities/scores; boolean masks remain boolean.
    if str(getattr(value, "dtype", "")) == "torch.bfloat16":
        value = value.float()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.array(value, copy=True)


def _capture(output, np, height: int, width: int, threshold: float):
    """Copy outputs before state mutation; preserve every valid instance/score pair."""
    masks, scores = [], []
    try:
        if not isinstance(output, dict):
            raise ValueError("SAM3 output must be a state dictionary")
        use_probabilities = "masks_logits" in output
        raw_masks = output.get("masks_logits" if use_probabilities else "masks")
        if raw_masks is None:
            raise ValueError("SAM3 output lacks masks")
        raw_scores = _cpu_array(output.get("scores", []), np).reshape(-1)
        if isinstance(raw_masks, list):
            instances = [_cpu_array(item, np) for item in raw_masks]
        else:
            array = _cpu_array(raw_masks, np)
            if array.ndim == 2:
                array = array[None, ...]
            if array.ndim not in (3, 4):
                raise ValueError("SAM3 masks must be N,H,W or N,1,H,W")
            instances = list(array)
        for index, instance in enumerate(instances):
            if instance.shape == (1, height, width):
                instance = instance[0]
            if instance.shape != (height, width):
                raise ValueError("SAM3 mask dimensions differ from original RGB")
            if index >= len(raw_scores):
                raise ValueError("SAM3 masks/scores count mismatch")
            score = float(raw_scores[index])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("SAM3 confidence must be a finite probability")
            if use_probabilities or instance.dtype.kind != "b":
                if not np.isfinite(instance).all() or ((instance < 0) | (instance > 1)).any():
                    raise ValueError("SAM3 mask probabilities must be finite in [0,1]")
                mask = instance > threshold
            else:
                if threshold != 0.5:
                    raise ValueError("Configured mask probability threshold requires SAM3 masks_logits probabilities")
                mask = instance
            masks.append(np.array(mask, dtype=np.bool_, copy=True))
            scores.append(score)
        if len(raw_scores) != len(instances):
            raise ValueError("SAM3 masks/scores count mismatch")
        return masks, scores, None
    except Exception as error:
        return masks, scores, f"invalid_sam_output:{type(error).__name__}:{error}"


class Sam3Segmenter:
    """SAM3 processor ownership and per-image prompt isolation."""

    def __init__(self, settings: dict, *, processor, model=None, _load_token=None):
        self.settings = _settings(settings)
        if not self.settings["fixture"] and _load_token is not _LOAD_TOKEN:
            raise SegmenterUnavailable("fixture_mode_required", "Injected processors require explicit fixture=true")
        for method in ("set_image", "reset_all_prompts", "set_text_prompt"):
            if not callable(getattr(processor, method, None)):
                raise ValueError(f"processor must implement {method}")
        self.processor = processor
        self.model = model
        self._owned = _load_token is _LOAD_TOKEN
        self._closed = False
        # Runtime roots are caller configuration, never portable artifact identity.
        self.settings.pop("cache_root", None)
        self.settings.pop("code_source_root", None)
        self.settings.pop("policy_path", None)
        self.fixture_identity = self.settings.get("fixture_identity")
        self.metadata = dict(self.settings)
        self.metadata["software"] = _software()
        self.metadata["model_validation"] = "mocked_api_only" if self.settings["fixture"] else "real_load_only_not_gpu_profiled"
        if not self.settings["fixture"]:
            self.metadata["inference_context"] = {
                "device": self.settings["device"], "inference_mode": True,
                "autocast_device_type": "cuda", "autocast_dtype": "torch.bfloat16",
                "bfloat16_output_capture": "float32_before_numpy",
            }

    @contextmanager
    def _processor_context(self):
        if self.settings["fixture"]:
            yield
            return
        torch = importlib.import_module("torch")
        # The pinned official image example uses BF16 autocast for fused layers.
        # Scope it to this adapter call while selecting its explicit CUDA index.
        with torch.cuda.device(int(self.settings["device"].split(":")[1])), \
             torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            yield

    def close(self) -> None:
        """Release only this adapter's references; never clear global CUDA caches."""
        self.processor = None
        self.model = None
        self._closed = True

    def segment_image(self, frame: dict, prompts: list[str], data_root) -> dict:
        np, Image = _image_modules()
        height = frame.get("height", 0) if isinstance(frame, dict) else 0
        width = frame.get("width", 0) if isinstance(frame, dict) else 0
        union = np.zeros((0, 0), dtype=np.bool_)
        frame_id = frame.get("frame_id") if isinstance(frame, dict) else None
        result = {"queries": [], "frame": {"frame_id": frame_id, "union_mask": union,
                  "status": "ok", "error_code": None, "sam_query_count": 0}, "settings": dict(self.metadata)}
        try:
            if self._closed:
                raise ValueError("segmenter_closed")
            path = _validate_input(frame, prompts, data_root)
            frame_schema = frame.get("schema_version", SCHEMA_VERSION)
            expected_task = "semantic_mapping_v1" if frame.get("input_contract_id") == "semantic_mapping_v1" else TASK_ID
            if frame.get("task_id", expected_task) != expected_task or type(frame_schema) is not int or frame_schema != SCHEMA_VERSION:
                raise ValueError("frame task/schema identity mismatch")
            if "fixture" in frame and (type(frame["fixture"]) is not bool or frame["fixture"] != self.settings["fixture"]):
                raise ValueError("frame fixture identity mismatch")
            if frame.get("input_contract_id") == "semantic_mapping_v1":
                from pipeline_common.input_bridge import load_model_rgb
                image = load_model_rgb(frame, data_root)
            else:
                with Image.open(path) as original:
                    image = original.convert("RGB")
                    if image.size != (width, height):
                        raise ValueError("original RGB dimensions mismatch")
            union = np.zeros((height, width), dtype=np.bool_)
            result["frame"]["union_mask"] = union
            with self._processor_context():
                state = self.processor.set_image(image)
        except Exception as error:
            result["frame"].update(status="error", error_code=f"sam_image_error:{type(error).__name__}:{error}")
            # A valid frame whose image encoding failed still needs explicit
            # failed query records so the controller can preserve exact joins.
            if union.shape == (height, width) and height > 0 and width > 0:
                result["queries"] = [{"phrase": phrase, "masks": [], "scores": [],
                    "union_mask": np.zeros((height, width), dtype=np.bool_), "status": "error",
                    "error_code": result["frame"]["error_code"], "sam_query_count": 0}
                    for phrase in prompts]
            return result
        vague = {" ".join(p.casefold().split()) for p in self.settings["vague_unusable_phrases"]}
        for phrase in prompts:
            query = {"phrase": phrase, "masks": [], "scores": [],
                     "union_mask": np.zeros((height, width), dtype=np.bool_),
                     "status": "ok", "error_code": None, "sam_query_count": 0}
            if " ".join(phrase.casefold().split()) in vague:
                query.update(status="error", error_code="vague_unusable_phrase")
            else:
                reset_succeeded = False
                try:
                    with self._processor_context():
                        self.processor.reset_all_prompts(state)
                        reset_succeeded = True
                        query["sam_query_count"] = 1
                        output = self.processor.set_text_prompt(state=state, prompt=phrase)
                    query["masks"], query["scores"], capture_error = _capture(output, np, height, width, self.settings["mask_probability_threshold"])
                    if capture_error:
                        query.update(status="error", error_code=capture_error)
                except Exception as error:
                    query.update(status="error", error_code=f"sam_query_error:{type(error).__name__}:{error}")
                    if reset_succeeded:
                        partial = getattr(error, "partial_output", state)
                        query["masks"], query["scores"], _ = _capture(partial, np, height, width, self.settings["mask_probability_threshold"])
            for mask in query["masks"]:
                query["union_mask"] |= mask
            union |= query["union_mask"]
            result["frame"]["sam_query_count"] += query["sam_query_count"]
            result["queries"].append(query)
        if any(query["status"] == "error" for query in result["queries"]):
            result["frame"].update(status="error", error_code="sam_query_failure")
        return result


def load_segmenter(settings: dict) -> Sam3Segmenter:
    """Load only verified official SAM3; optional injected processor is fixture-only."""
    configured = _settings(settings)
    injected = settings.get("processor")
    if injected is not None:
        if not configured["fixture"]:
            raise SegmenterUnavailable("fixture_mode_required", "Injected processors require explicit fixture=true")
        return Sam3Segmenter(configured, processor=injected)
    if configured["fixture"]:
        raise SegmenterUnavailable("fixture_processor_required", "Supply an explicit mock processor for fixture mode")
    if not configured["enable_model_loading"]:
        raise SegmenterUnavailable("model_loading_disabled", "Set enable_model_loading=true only for an explicitly enabled real run")
    if sys.version_info < (3, 12):
        raise SegmenterUnavailable("python_version_unavailable", "Official SAM3 requires isolated Python 3.12 or higher")
    code_identity = _verify_code(configured)
    checkpoint, checkpoint_identity = _resolve_checkpoint(configured)
    try:
        torch = importlib.import_module("torch")
        if not torch.cuda.is_available():
            raise SegmenterUnavailable("cuda_unavailable", "Official real SAM3 runs require an explicitly selected CUDA GPU")
        device_index = int(configured["device"].split(":")[1])
        if device_index >= torch.cuda.device_count():
            raise SegmenterUnavailable("cuda_device_unavailable", "Requested indexed CUDA device is not visible")
        versions = _software()
        version = tuple(int(p) for p in re.match(r"(\d+)\.(\d+)", str(versions["torch"])).groups())
        if version < (2, 7):
            raise SegmenterUnavailable("dependency_version_unavailable", "Official SAM3 requires PyTorch >=2.7")
        numpy_version = tuple(int(p) for p in re.match(r"(\d+)\.(\d+)", str(versions["numpy"])).groups())
        if not (1, 26) <= numpy_version < (2, 0):
            raise SegmenterUnavailable("dependency_version_unavailable", "Pinned SAM3 requires numpy>=1.26,<2")
        cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
        if not cuda_version or tuple(int(p) for p in cuda_version.split(".")[:2]) < (12, 6):
            raise SegmenterUnavailable("cuda_version_unavailable", "Official SAM3 requires CUDA 12.6 or higher")
        # Upstream only moves the model for the literal 'cuda'. The device context
        # selects the explicit indexed device without an implicit CPU fallback.
        with torch.cuda.device(device_index):
            builder = importlib.import_module("sam3.model_builder")
            processor_module = importlib.import_module("sam3.model.sam3_image_processor")
            model = builder.build_sam3_image_model(device="cuda", eval_mode=True,
                        checkpoint_path=str(checkpoint), load_from_HF=False,
                        enable_segmentation=True, enable_inst_interactivity=False, compile=False)
            processor = processor_module.Sam3Processor(model, resolution=configured["resolution"],
                         device=configured["device"], confidence_threshold=configured["confidence_threshold"])
        segmenter = Sam3Segmenter(configured, processor=processor, model=model, _load_token=_LOAD_TOKEN)
        segmenter.metadata.update(code_verification=code_identity, checkpoint_verification=checkpoint_identity,
                                  dependency_compatibility="proposed_stack; real GPU validation pending",
                                  upstream_limitations=["SAM3 model_builder import probes visible CUDA device 0 and enables TF32 on supported devices",
                                                       "upstream checkpoint loader reports missing keys without raising"])
        return segmenter
    except SegmenterUnavailable:
        raise
    except ImportError as error:
        raise SegmenterUnavailable("dependency_unavailable", f"Official SAM3 dependency/import is unavailable: {error}") from error
    except Exception as error:
        raise SegmenterUnavailable("sam3_load_failed", f"Official pinned SAM3 could not load: {type(error).__name__}: {error}") from error
