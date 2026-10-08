"""Lazy, local-checkpoint adapters for the SAM 3 + LingBot-Map baseline.

Each call owns and releases its model. Run ``reconstruct`` before
``segment_paths`` so the two large models do not occupy the GPU together.
No model weights, additional models, or input data are downloaded here.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import gc
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterator

import numpy as np


def _local_checkpoint(checkpoint: Path) -> Path:
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Model checkpoint is not a local file: {checkpoint}")
    return checkpoint


@contextmanager
def _source_path(source_root: Path, package: str, *, required: bool) -> Iterator[None]:
    """Make a vendored source package importable without changing the checkout."""
    root = Path(source_root).expanduser().resolve()
    present = (root / package).is_dir()
    if required and not present:
        raise FileNotFoundError(
            f"Missing {package} sources at {root}. Initialize the pinned submodules first."
        )
    entry = str(root)
    previous_index = sys.path.index(entry) if entry in sys.path else None
    if present:
        if previous_index is not None:
            sys.path.remove(entry)
        sys.path.insert(0, entry)
    try:
        yield
    finally:
        if present:
            sys.path.remove(entry)
            if previous_index is not None:
                sys.path.insert(previous_index, entry)


def _torch_runtime(device: str) -> tuple[Any, Any, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install the GPU baseline environment before model inference.") from exc
    target = torch.device(device)
    if target.type not in {"cuda", "cpu"}:
        raise ValueError("The baseline supports a single CUDA device, or CPU for testing.")
    if target.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested {device}, but PyTorch cannot see an NVIDIA GPU.")
        if target.index is None:
            target = torch.device("cuda:0")
        if target.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {target} does not exist.")
        dtype = (
            torch.bfloat16
            if torch.cuda.get_device_capability(target)[0] >= 8
            else torch.float16
        )
    else:
        dtype = torch.float32
    return torch, target, dtype


def _autocast(torch: Any, device: Any, dtype: Any) -> Any:
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=dtype)
    return nullcontext()


def _device_context(torch: Any, device: Any) -> Any:
    return torch.cuda.device(device) if device.type == "cuda" else nullcontext()


def _empty_cache(torch: Any, device: Any) -> None:
    gc.collect()
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def _single_sequence(value: Any, *, ndim: int, name: str) -> np.ndarray:
    array = _numpy(value)
    if array.ndim == ndim + 1 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != ndim:
        raise ValueError(f"LingBot {name} has unexpected shape {array.shape}.")
    return np.ascontiguousarray(array, dtype=np.float32)


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _source_revision(source_root: Path, package: str | None = None) -> str | None:
    """Read an actual checkout revision or pip's recorded source commit."""
    root = Path(source_root).resolve()
    if (root / ".git").exists():
        try:
            result = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            pass
    if package:
        try:
            direct_url = metadata.distribution(package).read_text("direct_url.json")
            if direct_url:
                return json.loads(direct_url).get("vcs_info", {}).get("commit_id")
        except (metadata.PackageNotFoundError, OSError, ValueError):
            pass
    return None


def reconstruct(
    frame_paths: list[Path],
    *,
    checkpoint: Path,
    source_root: Path,
    device: str = "cuda:0",
    image_size: int = 518,
    keyframe_interval: int = 1,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Reconstruct one ordered sequence, returning CPU arrays on the crop grid.

    ``extrinsic`` is the upstream decoded **world-to-camera** [R|t] matrix,
    in OpenCV coordinates. ``images`` is the exact preprocessed RGB tensor
    quantized to uint8; pass it directly to ``segment_paths``.
    """
    checkpoint = _local_checkpoint(checkpoint)
    paths = [Path(path).expanduser().resolve() for path in frame_paths]
    if not paths:
        raise ValueError("At least one ordered frame is required.")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Input frame is not a local file: {path}")
    if image_size <= 0 or image_size % 14:
        raise ValueError("LingBot image_size must be positive and divisible by patch size 14.")
    if keyframe_interval < 1:
        raise ValueError("keyframe_interval must be at least 1.")

    torch, target, dtype = _torch_runtime(device)
    model = payload = state_dict = predictions = images = None
    extrinsic = intrinsic = None
    with _source_path(source_root, "lingbot_map", required=True), _device_context(torch, target):
        try:
            try:
                import lingbot_map
                from lingbot_map.models.gct_stream import GCTStream
                from lingbot_map.utils.load_fn import load_and_preprocess_images
                from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
            except ImportError as exc:
                raise RuntimeError(
                    "LingBot-Map dependencies are missing; install the pinned source package "
                    "in the baseline environment. The adapter uses SDPA, without FlashInfer."
                ) from exc
            package_file = getattr(lingbot_map, "__file__", None)
            if package_file and not Path(package_file).resolve().is_relative_to(Path(source_root).resolve()):
                raise RuntimeError(
                    "A different LingBot-Map checkout was already imported. Start a new process "
                    "to use the requested pinned sources."
                )

            # Load on CPU first so checkpoint tensors do not duplicate GPU weights.
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict):
                raise ValueError("LingBot checkpoint must contain a state dictionary.")
            state_dict = payload.get("model", payload)
            if not isinstance(state_dict, dict) or not state_dict:
                raise ValueError("LingBot checkpoint has no model weights.")
            checkpoint_has_point_head = any(key.startswith("point_head.") for key in state_dict)
            scale_frames = min(8, len(paths))
            model = GCTStream(
                img_size=image_size,
                patch_size=14,
                enable_point=False,
                enable_3d_rope=True,
                max_frame_num=max(1024, len(paths)),
                kv_cache_sliding_window=64,
                kv_cache_scale_frames=scale_frames,
                kv_cache_cross_frame_special=True,
                kv_cache_include_scale_frames=True,
                use_sdpa=True,
                camera_num_iterations=4,
            )
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                raise ValueError(
                    "LingBot checkpoint is incompatible; required model weights are missing: "
                    + ", ".join(missing[:8])
                )
            payload = state_dict = None
            model = model.to(target).eval()
            # Match the upstream demo: keep prediction heads in float32.
            if dtype != torch.float32:
                model.aggregator = model.aggregator.to(dtype=dtype)

            images = load_and_preprocess_images(
                [str(path) for path in paths], mode="crop", image_size=image_size, patch_size=14
            )
            if images.ndim != 4 or images.shape[:2] != (len(paths), 3):
                raise ValueError(f"Unexpected preprocessed image shape: {tuple(images.shape)}")
            height, width = images.shape[-2:]
            # Inputs remain on CPU; upstream streaming moves one block at a time.
            with torch.inference_mode(), _autocast(torch, target, dtype):
                predictions = model.inference_streaming(
                    images,
                    num_scale_frames=scale_frames,
                    keyframe_interval=keyframe_interval,
                    output_device=torch.device("cpu"),
                )

            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                predictions["pose_enc"], (height, width)
            )
            # Match the streaming demo's depth/pose geometry. Unlike an optional
            # independently learned point head, this unprojection preserves the
            # exact source-pixel correspondence checked during fusion.
            with torch.inference_mode():
                predictions["world_points"] = model._unproject_depth_to_world(
                    predictions["depth"], predictions["pose_enc"]
                )
            predictions["world_points_conf"] = predictions["depth_conf"]

            geometry = {
                "images": np.clip(
                    images.detach().cpu().permute(0, 2, 3, 1).numpy() * 255, 0, 255
                ).astype(np.uint8),
                "world_points": _single_sequence(predictions["world_points"], ndim=4, name="world_points"),
                "world_points_conf": _single_sequence(
                    predictions["world_points_conf"], ndim=3, name="world_points_conf"
                ),
                "depth": _single_sequence(predictions["depth"], ndim=4, name="depth"),
                "extrinsic": _single_sequence(extrinsic, ndim=3, name="extrinsic"),
                "intrinsic": _single_sequence(intrinsic, ndim=3, name="intrinsic"),
            }
            expected = {
                "world_points": (len(paths), height, width, 3),
                "world_points_conf": (len(paths), height, width),
                "depth": (len(paths), height, width, 1),
                "extrinsic": (len(paths), 3, 4),
                "intrinsic": (len(paths), 3, 3),
            }
            for key, shape in expected.items():
                if geometry[key].shape != shape:
                    raise ValueError(f"LingBot {key} shape {geometry[key].shape} does not match {shape}.")
            info = {
                "model": "LingBot-Map GCTStream",
                "checkpoint": str(checkpoint),
                "checkpoint_size_bytes": checkpoint.stat().st_size,
                "source_root": str(Path(source_root).resolve()),
                "source_revision": _source_revision(Path(source_root), "lingbot-map"),
                "package_version": _version("lingbot-map"),
                "torch_version": str(torch.__version__),
                "device": str(target),
                "dtype": str(dtype),
                "attention_backend": "sdpa",
                "preprocessing": "upstream crop (EXIF orientation, resize, crop/pad)",
                "image_size": image_size,
                "processed_shape": [len(paths), height, width, 3],
                "keyframe_interval": keyframe_interval,
                "num_scale_frames": scale_frames,
                "point_source": "depth_unprojection",
                "point_confidence_source": "depth_conf",
                "checkpoint_point_head_unused": checkpoint_has_point_head,
                "extrinsic_convention": "world_to_camera",
                "coordinate_convention": "OpenCV: x right, y down, z forward",
                "unexpected_checkpoint_keys": list(unexpected),
            }
            return geometry, info
        finally:
            # Release all GPU references before the SAM adapter is called.
            model = payload = state_dict = predictions = images = None
            extrinsic = intrinsic = None
            _empty_cache(torch, target)


def merge_instance_scores(
    masks: np.ndarray,
    scores: np.ndarray,
    image_hw: tuple[int, int],
    confidence_threshold: float = 0.5,
) -> np.ndarray:
    """Union SAM instances with the maximum retained score at each positive pixel."""
    height, width = image_hw
    masks = np.asarray(masks)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("SAM confidence_threshold must be between 0 and 1.")
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.shape != (len(scores), height, width):
        raise ValueError(
            f"SAM masks must have shape (N, {height}, {width}), got {masks.shape}; "
            f"there are {len(scores)} scores."
        )
    if masks.dtype.kind not in "biuf" or not np.isfinite(masks).all():
        raise ValueError("SAM masks must contain finite boolean or numeric values.")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("SAM instance scores must be finite probabilities between 0 and 1.")
    merged = np.zeros((height, width), dtype=np.float32)
    for mask, score in zip(masks, scores):
        if score >= confidence_threshold:
            positive = mask if mask.dtype == np.bool_ else mask > 0.5
            merged[positive] = np.maximum(merged[positive], score)
    return merged


def segment_paths(
    images: np.ndarray,
    *,
    checkpoint: Path,
    source_root: Path,
    prompt: str = "Path",
    confidence_threshold: float = 0.5,
    device: str = "cuda:0",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Segment only the supplied text concept on the LingBot processed RGB grid."""
    checkpoint = _local_checkpoint(checkpoint)
    images = np.asarray(images)
    if images.ndim != 4 or images.shape[-1] != 3 or not images.shape[0]:
        raise ValueError("SAM images must have shape [S, H, W, 3] with at least one frame.")
    if images.dtype != np.uint8:
        raise ValueError("SAM requires the uint8 processed RGB images returned by reconstruct.")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("SAM requires one nonempty text prompt.")
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("SAM confidence_threshold must be between 0 and 1.")
    torch, target, dtype = _torch_runtime(device)
    model = processor = state = result = payload = state_dict = image_state = None
    # The wrapper's setup installs Meta's package here; an installed sam3 package
    # is also supported. We use its image processor directly to supply a local
    # checkpoint and avoid the wrapper constructor's automatic HF download.
    sam_source = Path(source_root) / "third_party" / "sam3"
    with _source_path(sam_source, "sam3", required=False), _device_context(torch, target):
        try:
            try:
                from PIL import Image
                import sam3
                from sam3.model_builder import build_sam3_image_model
                from sam3.model.sam3_image_processor import Sam3Processor
            except ImportError as exc:
                raise RuntimeError(
                    "SAM 3 dependencies are missing; install the pinned Meta SAM 3 "
                    "source package and baseline dependencies before inference."
                ) from exc

            # Match the pinned builder's detector-prefix checkpoint mapping,
            # while rejecting missing weights instead of merely printing them.
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict):
                raise ValueError("SAM 3 checkpoint must contain a state dictionary.")
            state_dict = payload.get("model", payload)
            if not isinstance(state_dict, dict):
                raise ValueError("SAM 3 checkpoint has no model state dictionary.")
            image_state = {
                key.replace("detector.", ""): value
                for key, value in state_dict.items()
                if "detector" in key
            }
            if not image_state:
                raise ValueError("SAM 3 checkpoint has no detector weights; provide the official sam3.pt.")
            # The builder recognizes only the literal 'cuda'. Its positional
            # caches also allocate on the current CUDA device during construction;
            # the surrounding device context binds those to the selected GPU.
            model = build_sam3_image_model(
                checkpoint_path=None,
                load_from_HF=False,
                device="cpu",
                eval_mode=True,
                enable_segmentation=True,
                enable_inst_interactivity=False,
                compile=False,
            )
            missing, unexpected = model.load_state_dict(image_state, strict=False)
            if missing:
                raise ValueError(
                    "SAM 3 checkpoint is incompatible; required model weights are missing: "
                    + ", ".join(missing[:8])
                )
            payload = state_dict = image_state = None
            model = model.to(target).eval()
            processor = Sam3Processor(
                model, device=str(target), confidence_threshold=confidence_threshold
            )
            count, height, width, _ = images.shape
            path_scores = np.zeros((count, height, width), dtype=np.float32)
            instance_counts = []
            with torch.inference_mode(), _autocast(torch, target, dtype):
                for frame_index, rgb in enumerate(images):
                    state = processor.set_image(Image.fromarray(rgb))
                    processor.reset_all_prompts(state)
                    result = processor.set_text_prompt(state=state, prompt=prompt)
                    masks = _numpy(result["masks"])
                    scores = _numpy(result["scores"])
                    path_scores[frame_index] = merge_instance_scores(
                        masks, scores, (height, width), confidence_threshold
                    )
                    instance_counts.append(int(np.asarray(scores).size))
                    # Features and instance masks belong to this frame only.
                    state = result = None
            package_file = getattr(sam3, "__file__", None)
            package_source = Path(package_file).resolve().parent.parent if package_file else sam_source
            info = {
                "model": "SAM 3 image model",
                "checkpoint": str(checkpoint),
                "checkpoint_size_bytes": checkpoint.stat().st_size,
                "source_root": str(Path(source_root).resolve()),
                "wrapper_source_revision": _source_revision(Path(source_root)),
                "sam3_package_source": str(package_source),
                "source_revision": _source_revision(package_source, "sam3"),
                "package_version": _version("sam3"),
                "torch_version": str(torch.__version__),
                "device": str(target),
                "dtype": str(dtype),
                "prompt": prompt,
                "confidence_threshold": confidence_threshold,
                "processed_shape": list(images.shape),
                "instance_counts": instance_counts,
                "pixel_score": "maximum instance confidence on union of binary masks",
                "model_weights_downloaded": False,
                "unexpected_checkpoint_keys": list(unexpected),
            }
            return path_scores, info
        finally:
            model = processor = state = result = payload = state_dict = image_state = None
            _empty_cache(torch, target)
