"""CLI and portable, pixel-aligned handoffs for the Path mapping baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
GEOMETRY_KEYS = ("images", "world_points", "world_points_conf", "depth", "extrinsic", "intrinsic")


def image_hashes(images):
    return np.asarray([hashlib.sha256(image.tobytes()).hexdigest() for image in images])


def geometry_fingerprint(geometry):
    """Bind masks to geometry, image pixels, ordering and poses, not filenames."""
    digest = hashlib.sha256()
    for key in GEOMETRY_KEYS:
        array = np.ascontiguousarray(geometry[key])
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(array.shape).encode())
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def validate_geometry(geometry, *, allow_single_frame=False):
    missing = set(GEOMETRY_KEYS) - geometry.keys()
    if missing:
        raise ValueError(f"Geometry is missing: {', '.join(sorted(missing))}")
    images = geometry["images"]
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("Processed RGB images must have shape [S,H,W,3] and dtype uint8")
    s, h, w, _ = images.shape
    if min(s, h, w) < 1 or (s < 2 and not allow_single_frame):
        raise ValueError("A scene map requires at least two frames; use --allow-single-frame for a diagnostic")
    shapes = {"world_points": (s, h, w, 3), "world_points_conf": (s, h, w),
              "extrinsic": (s, 3, 4), "intrinsic": (s, 3, 3)}
    for key, expected in shapes.items():
        value = geometry[key]
        if value.shape != expected or not np.issubdtype(value.dtype, np.floating):
            raise ValueError(f"{key} must be a floating array with shape {expected}")
    depth = geometry["depth"]
    if depth.shape not in ((s, h, w), (s, h, w, 1)) or not np.issubdtype(depth.dtype, np.floating):
        raise ValueError("depth must be a floating array on the processed image grid")
    for key in ("extrinsic", "intrinsic"):
        if not np.isfinite(geometry[key]).all():
            raise ValueError(f"{key} contains invalid camera parameters")
    k = geometry["intrinsic"]
    if np.any(k[:, 0, 0] <= 0) or np.any(k[:, 1, 1] <= 0):
        raise ValueError("Camera focal lengths must be positive")
    if not np.allclose(k[:, 2], [0, 0, 1], atol=1e-5):
        raise ValueError("Intrinsics must use the standard homogeneous camera convention")
    rotations = geometry["extrinsic"][:, :, :3]
    if not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-3):
        raise ValueError("Extrinsics must contain world-to-camera rigid rotations")
    if not np.allclose(np.linalg.det(rotations), 1, atol=1e-3):
        raise ValueError("Extrinsic rotations must preserve orientation")
    return s, h, w


def reprojection_diagnostic(geometry, min_point_confidence=1.5):
    """Check source pixels against decoded W2C cameras before accepting fusion."""
    median_samples = []
    point_count = 0
    max_error = None
    behind = 0
    for index, points in enumerate(geometry["world_points"]):
        depth = geometry["depth"][index].reshape(points.shape[:2])
        confidence = geometry["world_points_conf"][index]
        valid = (np.isfinite(points).all(-1) & np.isfinite(depth) & (depth > 0)
                 & np.isfinite(confidence) & (confidence > 0) & (confidence >= min_point_confidence))
        pixels = np.argwhere(valid)
        if not len(pixels):
            continue
        xyz = points[pixels[:, 0], pixels[:, 1]]
        extrinsic, intrinsic = geometry["extrinsic"][index], geometry["intrinsic"][index]
        camera = xyz @ extrinsic[:, :3].T + extrinsic[:, 3]
        positive = camera[:, 2] > 0
        behind += int((~positive).sum())
        if positive.any():
            projected = camera[positive] @ intrinsic.T
            uv = projected[:, :2] / projected[:, 2, None]
            errors = np.linalg.norm(uv - pixels[positive, ::-1], axis=1)
            if not np.isfinite(errors).all():
                raise ValueError("Geometry failed pixel alignment: nonfinite projected pixels")
            point_count += len(errors)
            max_error = max(max_error or 0, float(errors.max()))
            # Validate every point, but keep only bounded diagnostic samples
            # for the reported median across long sequences.
            sample = errors[np.linspace(0, len(errors) - 1, min(len(errors), 128), dtype=int)]
            median_samples.extend(sample.tolist())
    result = {"sample_count": point_count, "validated_point_count": point_count + behind,
              "behind_camera_count": behind, "max_error_pixels": max_error,
              "median_error_pixels": float(np.median(median_samples)) if median_samples else None,
              "median_sample_count": len(median_samples),
              "pose_convention": "world_to_camera", "coverage": "all_valid_geometry_pixels"}
    if behind or (max_error is not None and max_error > 1.0):
        raise ValueError(f"Geometry failed pixel alignment / world-to-camera validation: {result}")
    return result


def _write_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_geometry(path, geometry, source_paths, metadata):
    _write_npz(path, **{key: geometry[key] for key in GEOMETRY_KEYS},
               schema_version=np.asarray(SCHEMA_VERSION),
               source_paths=np.asarray([str(p) for p in source_paths]),
               processed_image_sha256=image_hashes(geometry["images"]),
               metadata_json=np.asarray(json.dumps(metadata, allow_nan=False)),
               geometry_fingerprint=np.asarray(geometry_fingerprint(geometry)))


def load_geometry(path):
    with np.load(path, allow_pickle=False) as saved:
        if saved["schema_version"].item() != SCHEMA_VERSION:
            raise ValueError("Unsupported geometry archive schema")
        geometry = {key: saved[key] for key in GEOMETRY_KEYS}
        metadata = json.loads(saved["metadata_json"].item())
        sources = saved["source_paths"].tolist()
        if len(sources) != len(geometry["images"]):
            raise ValueError("Geometry source frame count does not match the images")
        if not np.array_equal(saved["processed_image_sha256"], image_hashes(geometry["images"])):
            raise ValueError("Geometry processed image hashes do not match")
        if saved["geometry_fingerprint"].item() != geometry_fingerprint(geometry):
            raise ValueError("Geometry archive fingerprint does not match")
    return geometry, sources, metadata


def save_scores(path, scores, geometry, prompt, metadata):
    _write_npz(path, schema_version=np.asarray(SCHEMA_VERSION), path_scores=scores,
               prompt=np.asarray(prompt), geometry_fingerprint=np.asarray(geometry_fingerprint(geometry)),
               metadata_json=np.asarray(json.dumps(metadata, allow_nan=False)))


def load_scores(path, geometry, prompt):
    with np.load(path, allow_pickle=False) as saved:
        if saved["schema_version"].item() != SCHEMA_VERSION:
            raise ValueError("Unsupported SAM archive schema")
        if saved["prompt"].item() != prompt:
            raise ValueError("Saved SAM prompt does not match the requested prompt")
        if saved["geometry_fingerprint"].item() != geometry_fingerprint(geometry):
            raise ValueError("SAM masks belong to a different geometry / frame order / pixel grid")
        scores = saved["path_scores"]
        metadata = json.loads(saved["metadata_json"].item())
    validate_scores(scores, geometry["images"].shape[:3])
    return scores, metadata


def validate_scores(scores, shape):
    if scores.shape != shape or not np.issubdtype(scores.dtype, np.floating):
        raise ValueError(f"SAM scores must be floating probabilities with shape {shape}")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("SAM scores must be finite probabilities in [0,1]")


def _natural_key(path):
    return ([(1, int(part)) if part.isdigit() else (0, part.casefold())
             for part in re.split(r"(\d+)", path.name)], path.name)


def collect_frames(folder, *, stride=1, max_frames=100):
    if not folder.is_dir():
        raise ValueError(f"Frame folder does not exist: {folder}")
    paths = sorted((path for path in folder.iterdir()
                    if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}),
                   key=_natural_key)
    return paths[::stride][:max_frames]


def extract_video(path, output, *, fps=2.0, stride=1, max_frames=100):
    import cv2
    if not path.is_file():
        raise ValueError(f"Video does not exist: {path}")
    capture = cv2.VideoCapture(str(path))
    try:
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or not math.isfinite(source_fps) or source_fps <= 0:
            raise ValueError("Video cannot be opened or has no valid frame rate")
        interval = max(1, round(source_fps / fps)) * stride
        output.mkdir(parents=True, exist_ok=True)
        paths, index = [], 0
        while len(paths) < max_frames:
            success, frame = capture.read()
            if not success:
                break
            if index % interval == 0:
                destination = output / f"{index:09d}.png"
                if not cv2.imwrite(str(destination), frame):
                    raise OSError(f"Could not write extracted frame: {destination}")
                paths.append(destination)
            index += 1
        return paths
    finally:
        capture.release()


def fixture():
    """Synthetic alignment and fusion evidence; never a model measurement."""
    s, h, w = 3, 8, 12
    v, u = np.mgrid[:h, :w]
    depth = np.full((s, h, w, 1), 2.0, dtype=np.float32)
    intrinsic = np.tile(np.array([[12, 0, 5.5], [0, 12, 3.5], [0, 0, 1]], np.float32), (s, 1, 1))
    extrinsic = np.tile(np.eye(4, dtype=np.float32)[:3], (s, 1, 1))
    xyz = np.stack(((u - 5.5) / 6, (v - 3.5) / 6, np.full_like(u, 2)), axis=-1).astype(np.float32)
    images = np.tile(np.stack((u * 15 + 40, v * 20 + 30, np.full_like(u, 90)), -1).astype(np.uint8), (s, 1, 1, 1))
    scores = np.zeros((s, h, w), dtype=np.float32)
    scores[:, :, 3:9] = .85
    # One frame disputes the left edge, exercising negative semantic evidence.
    scores[2, :, 3] = 0
    geometry = {"images": images, "world_points": np.tile(xyz, (s, 1, 1, 1)),
                "world_points_conf": np.full((s, h, w), 2.0, np.float32),
                "depth": depth, "intrinsic": intrinsic, "extrinsic": extrinsic}
    return geometry, [f"synthetic:{i}" for i in range(s)], scores


def parser():
    result = argparse.ArgumentParser(description="SAM3 Path + LingBot-Map baseline; no VLM or extra segmentation model")
    inputs = result.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--frames", type=Path, help="Ordered image folder (natural filename order)")
    inputs.add_argument("--video", type=Path)
    inputs.add_argument("--geometry", type=Path, help="Saved geometry.npz from the reconstruct stage")
    inputs.add_argument("--fixture", action="store_true", help="CPU-only synthetic contract check")
    result.add_argument("--stage", choices=("all", "reconstruct", "segment", "fuse"), default="all")
    result.add_argument("--scores", type=Path, help="Saved sam_scores.npz; required for --stage fuse")
    result.add_argument("--lingbot-checkpoint", type=Path)
    result.add_argument("--sam-checkpoint", type=Path)
    result.add_argument("--lingbot-source", type=Path, default=REPO_ROOT / "src/lingbot-map")
    result.add_argument("--sam-source", type=Path, default=REPO_ROOT / "src/sam3-robot")
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--prompt", default="Path")
    result.add_argument("--image-size", type=int, default=518)
    result.add_argument("--keyframe-interval", type=int, default=1)
    result.add_argument("--stride", type=int, default=1)
    result.add_argument("--max-frames", type=int, default=100)
    result.add_argument("--fps", type=float, default=2.0)
    result.add_argument("--voxel-size", type=float, default=0.05, help="In reconstruction units, or metres when calibrated")
    result.add_argument("--meters-per-unit", type=float, help="Explicit scale calibration; omitted means unknown metric scale")
    result.add_argument("--map-origin", type=float, nargs=3, default=(0, 0, 0))
    result.add_argument("--min-point-confidence", type=float, default=1.5)
    result.add_argument("--sam-confidence", type=float, default=0.5)
    result.add_argument("--path-probability", type=float, default=0.5)
    result.add_argument("--min-observations", type=int, default=2)
    result.add_argument("--allow-single-frame", action="store_true")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--overwrite", action="store_true", help="Replace artifacts in an existing output folder")
    result.add_argument("--serve", action="store_true")
    result.add_argument("--port", type=int, default=8080)
    result.add_argument("--host", default="127.0.0.1")
    return result


def run(args):
    if args.prompt != "Path":
        raise ValueError("This baseline uses the exact prompt Path")
    for name in ("stride", "max_frames", "image_size", "keyframe_interval", "min_observations"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    for name in ("fps", "voxel_size"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("sam_confidence", "path_probability"):
        if not math.isfinite(getattr(args, name)) or not 0 <= getattr(args, name) <= 1:
            raise ValueError(f"{name} must be a probability in [0,1]")
    if not math.isfinite(args.min_point_confidence) or args.min_point_confidence < 0:
        raise ValueError("min_point_confidence must be finite and nonnegative")
    if not all(math.isfinite(x) for x in args.map_origin):
        raise ValueError("map_origin must be finite")
    if args.meters_per_unit is not None and (not math.isfinite(args.meters_per_unit) or args.meters_per_unit <= 0):
        raise ValueError("meters_per_unit must be finite and positive")
    if args.stage == "fuse" and not args.fixture and (args.geometry is None or args.scores is None):
        raise ValueError("--stage fuse requires both --geometry and --scores")
    if args.scores and args.stage == "reconstruct":
        raise ValueError("--scores is not used by the reconstruct stage")
    if args.fixture and args.scores:
        raise ValueError("--fixture generates synthetic scores; do not combine it with --scores")
    if args.serve and args.stage not in ("all", "fuse"):
        raise ValueError("--serve requires a fused map")
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise ValueError("Output directory is nonempty; choose another directory or pass --overwrite")
    model_metadata, sam_metadata = {}, {}
    if args.fixture:
        geometry, source_paths, scores = fixture()
        model_metadata = sam_metadata = {"fixture": True, "models_executed": False}
    elif args.geometry:
        geometry, source_paths, model_metadata = load_geometry(args.geometry)
    else:
        if args.lingbot_checkpoint is None or not args.lingbot_checkpoint.is_file():
            raise ValueError("Provide an existing local --lingbot-checkpoint")
        if args.stage != "reconstruct" and args.scores is None and (args.sam_checkpoint is None or not args.sam_checkpoint.is_file()):
            raise ValueError("Provide an existing local --sam-checkpoint")
        if args.frames:
            source_paths = collect_frames(args.frames, stride=args.stride, max_frames=args.max_frames)
        else:
            source_paths = extract_video(args.video, args.output / "extracted_frames", fps=args.fps,
                                         stride=args.stride, max_frames=args.max_frames)
        if len(source_paths) < (1 if args.allow_single_frame else 2):
            raise ValueError("Not enough selected input frames")
        from .models import reconstruct
        geometry, model_metadata = reconstruct(source_paths, checkpoint=args.lingbot_checkpoint,
                                               source_root=args.lingbot_source, device=args.device,
                                               image_size=args.image_size, keyframe_interval=args.keyframe_interval)
    shape = validate_geometry(geometry, allow_single_frame=args.allow_single_frame)
    alignment = reprojection_diagnostic(geometry, args.min_point_confidence)
    args.output.mkdir(parents=True, exist_ok=True)
    # summary.json is a completion marker; a failed overwrite must not retain
    # the previous run's successful summary.
    (args.output / "summary.json").unlink(missing_ok=True)
    save_geometry(args.output / "geometry.npz", geometry, source_paths, model_metadata)
    processed = args.output / "processed_frames"
    processed.mkdir(exist_ok=True)
    for index, image in enumerate(geometry["images"]):
        Image.fromarray(image).save(processed / f"{index:06d}.png")
    for existing in processed.glob("*.png"):
        if re.fullmatch(r"\d{6}\.png", existing.name) and int(existing.stem) >= shape[0]:
            existing.unlink()
    base_summary = {"schema_version": SCHEMA_VERSION, "pipeline": "sam3_lingbot_path_v1",
                    "fixture": bool(model_metadata.get("fixture", False)), "prompt": args.prompt,
                    "frame_count": shape[0], "processed_shape": list(shape),
                    "source_paths": [str(p) for p in source_paths], "geometry_model": model_metadata,
                    "alignment": alignment}
    if args.stage == "reconstruct":
        base_summary.update(completed_stage="reconstruct", artifacts=["geometry.npz", "processed_frames/", "summary.json"])
        (args.output / "summary.json").write_text(json.dumps(base_summary, indent=2), encoding="utf-8")
        return base_summary
    if args.scores:
        scores, sam_metadata = load_scores(args.scores, geometry, args.prompt)
    elif not args.fixture:
        if args.sam_checkpoint is None or not args.sam_checkpoint.is_file():
            raise ValueError("Provide an existing local --sam-checkpoint")
        from .models import segment_paths
        scores, sam_metadata = segment_paths(geometry["images"], checkpoint=args.sam_checkpoint,
                                              source_root=args.sam_source, prompt=args.prompt,
                                              confidence_threshold=args.sam_confidence, device=args.device)
    validate_scores(scores, shape)
    save_scores(args.output / "sam_scores.npz", scores, geometry, args.prompt, sam_metadata)
    base_summary.update({"segmentation_model": sam_metadata,
                         "fixture": bool(base_summary["fixture"] or sam_metadata.get("fixture", False))})
    if args.stage == "segment":
        base_summary.update(completed_stage="segment", artifacts=["geometry.npz", "processed_frames/", "sam_scores.npz", "summary.json"])
        (args.output / "summary.json").write_text(json.dumps(base_summary, indent=2), encoding="utf-8")
        return base_summary
    from .fusion import fuse_frame_sequence
    scale = args.meters_per_unit or 1.0
    result = fuse_frame_sequence(geometry["world_points"] * scale, geometry["world_points_conf"],
                                 geometry["depth"] * scale, scores, geometry["images"],
                                 voxel_size=args.voxel_size, map_origin=args.map_origin,
                                 min_point_confidence=args.min_point_confidence,
                                 path_probability_threshold=args.path_probability,
                                 min_observations=args.min_observations)
    config = {"voxel_size": args.voxel_size, "map_origin": list(args.map_origin),
              "min_point_confidence": args.min_point_confidence,
              "sam_confidence": sam_metadata.get("confidence_threshold"),
              "path_probability_threshold": args.path_probability, "min_observations": args.min_observations,
              "meters_per_reconstruction_unit": args.meters_per_unit,
              "coordinate_unit": "metres" if args.meters_per_unit else "reconstruction_units",
              "observation_count": "distinct_frames_per_voxel", "weighting": "raw_geometry_confidence",
              "voxel_denominator": "all_valid_geometry_pixels"}
    scaled_geometry = {**geometry, "extrinsic": geometry["extrinsic"].copy()}
    scaled_geometry["extrinsic"][:, :, 3] *= scale
    from .artifacts import write_artifacts, serve_map
    export = write_artifacts(args.output, result, geometry["images"], scores,
                             summary={**base_summary, "completed_stage": "fuse", "configuration": config},
                             geometry=scaled_geometry)
    if args.serve:
        serve_map(result, scaled_geometry, port=args.port, host=args.host)
    return {**base_summary, **export, "configuration": config}


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = run(args)
    except (ValueError, OSError, ImportError, RuntimeError, KeyError) as error:
        print(f"Path mapping failed: {error}", file=sys.stderr)
        return 1
    print(f"Completed {args.stage}: {args.output.resolve()}")
    print(f"Frames: {result['frame_count']}; synthetic fixture: {result['fixture']}")
    return 0
