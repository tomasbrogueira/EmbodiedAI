"""CPU-only movies of saved sampled RGB, masks, voxel journals and saved routes.

No producer, model, planner or evaluator is imported. Inputs are read-only;
output must be a new directory outside all input artifact directories.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import subprocess
import sys
import textwrap
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

MAX_ARRAY_BYTES = 256 * 1024 * 1024
MAX_GEOMETRY_ARCHIVE_BYTES = 1024 * 1024 * 1024
MAX_JSON_BYTES = 64 * 1024 * 1024
PIPELINES = {"geometry_only", "ground_surface", "fixed_hazards", "qwen_hazards"}
RESEARCH_LABEL = "Geometry-only research path - assumed scale, ground direction and robot"
COLORS = {"candidate_surface": (25, 194, 177), "hazard": (239, 127, 49),
          "diagnostic": (220, 78, 178), "geometry": (165, 176, 187),
          "research": (195, 93, 229), "native_path": (245, 193, 61)}


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _read(path, *, rows=False):
    path = Path(path)
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(f"JSON input exceeds 64 MiB: {path}")
    def reject(value):
        raise ValueError(f"Non-finite JSON value: {value}")
    value = path.read_text(encoding="utf-8")
    if rows:
        return [json.loads(line, parse_constant=reject) for line in value.splitlines() if line.strip()]
    return json.loads(value, parse_constant=reject)


def _safe(root, relative):
    text = str(relative)
    if (not text or PurePosixPath(text).is_absolute() or PureWindowsPath(text).is_absolute()
            or ".." in PurePosixPath(text.replace("\\", "/")).parts):
        raise ValueError(f"Unsafe saved artifact path: {text}")
    path = (Path(root) / text).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError(f"Saved artifact escapes its directory: {text}")
    return path


def load_npz_checked(path, *, max_bytes=MAX_ARRAY_BYTES, names=None):
    """Validate every NPY header before allocation; never deserialize objects."""
    if not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("NPZ byte limit must be positive")
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            archive_names = [item.filename for item in infos]
            if len(archive_names) != len(set(archive_names)) or not infos:
                raise ValueError("Empty or duplicate-member NPZ")
            if sum(item.file_size for item in infos) > max_bytes:
                raise ValueError("NPZ uncompressed bytes exceed read limit")
            allocated = 0
            for item in infos:
                if "/" in item.filename or "\\" in item.filename or not item.filename.endswith(".npy"):
                    raise ValueError("NPZ must contain only flat NPY members")
                with archive.open(item) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                    elif version == (2, 0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                    else:
                        raise ValueError("Unsupported NPY header version")
                    if dtype.hasobject or dtype.fields or dtype.kind not in "biufc":
                        raise ValueError("NPZ contains a nonnumeric/object/structured array")
                    if len(shape) > 8 or any(not isinstance(n, int) or n < 0 for n in shape):
                        raise ValueError("Invalid array shape")
                    size = math.prod(shape) * dtype.itemsize
                    allocated += size
                    if allocated > max_bytes or size != item.file_size - stream.tell():
                        raise ValueError("NPY allocation/header does not match bounded payload")
        with np.load(path, allow_pickle=False) as saved:
            selected = saved.files if names is None else [name for name in names if name in saved.files]
            return {name: saved[name] for name in selected}
    except (zipfile.BadZipFile, EOFError, OSError) as error:
        raise ValueError(f"Invalid NPZ: {path}") from error


def _rgb(path):
    with Image.open(path) as image:
        if image.width * image.height * 3 > MAX_ARRAY_BYTES:
            raise ValueError("RGB image exceeds read limit")
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _rgb_hash(rgb):
    return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()


def _indices(value):
    array = np.asarray(value)
    if array.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    if array.ndim != 2 or array.shape[1] != 3 or array.dtype.kind not in "iu":
        raise ValueError("Voxel indices must be integer Nx3")
    if np.any(np.abs(array.astype(float)) > 2 ** 50):
        raise ValueError("Voxel indices exceed stable rendering range")
    return array.astype(np.int64)


def _points(value):
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return np.empty((0, 3), dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
        raise ValueError("Saved route must contain finite XYZ map points")
    return array


def load_export_inputs(run_dir, sequence_path, *, cache_dir=None, research_plan_path=None):
    run_dir, sequence_path = Path(run_dir).resolve(), Path(sequence_path).resolve()
    hashes = {}
    def read(path, **kwargs):
        hashes[str(Path(path).resolve())] = sha256(path)
        return _read(path, **kwargs)
    def arrays(path, **kwargs):
        hashes[str(Path(path).resolve())] = sha256(path)
        return load_npz_checked(path, **kwargs)
    run = read(run_dir / "run.json")
    if (run.get("contract_id") != "semantic_mapping_v1" or run.get("schema_version") != 1
            or run.get("pipeline_id") not in PIPELINES):
        raise ValueError("Expected a schema-1 common saved run")
    config = read(run_dir / "config.resolved.json")
    if run.get("config_digest") != config.get("config_digest") or _digest(
            {k: v for k, v in config.items() if k != "config_digest"}) != config["config_digest"]:
        raise ValueError("Resolved config digest differs from saved run")
    sequence = read(sequence_path)
    sequence_digest = _digest({k: v for k, v in sequence.items() if k != "manifest_digest"})
    if sequence_digest != sequence.get("manifest_digest") or sequence_digest != run["sequence"]["manifest_digest"]:
        raise ValueError("Sequence manifest digest differs from run")
    geometry = read(run_dir / "geometry/manifest.json")
    map_manifest = read(run_dir / "map/manifest.json")
    planning = read(run_dir / "planning/manifest.json")
    plans_path = run_dir / "planning/plans.jsonl"
    plans = read(plans_path, rows=True) if plans_path.is_file() else []
    cache = Path(cache_dir).resolve() if cache_dir is not None else Path(geometry.get("cache_path", run_dir / "geometry/cache"))
    if cache_dir is None and not cache.is_absolute():
        cache = run_dir / "geometry" / cache
    cache = cache.resolve()
    cache_manifest = read(cache / "manifest.json")
    identity_keys = ("geometry_fingerprint", "input_fingerprint", "processed_grid_id", "archive_sha256",
                     "map_frame", "units", "pose_revision")
    for key in identity_keys:
        if geometry.get(key) != cache_manifest.get(key) or geometry.get(key) is None:
            raise ValueError(f"Geometry/cache identity mismatch: {key}")
    for record in (run, config):
        for key in ("geometry_fingerprint", "input_fingerprint", "processed_grid_id", "units", "pose_revision"):
            if record.get("geometry_identity", {}).get(key) != geometry.get(key):
                raise ValueError(f"Saved run/config geometry identity mismatch: {key}")
    for key in ("geometry_fingerprint", "map_frame", "units", "pose_revision"):
        if map_manifest.get(key) != geometry.get(key):
            raise ValueError(f"Map/geometry identity mismatch: {key}")
    if geometry.get("sequence_digest") != sequence_digest:
        raise ValueError("Geometry cache belongs to another sequence")
    archive = cache / "geometry.npz"
    if sha256(archive) != geometry["archive_sha256"]:
        raise ValueError("Geometry archive SHA256 mismatch")
    geometry_arrays = arrays(archive, max_bytes=MAX_GEOMETRY_ARCHIVE_BYTES, names=("images", "extrinsic"))
    images = geometry_arrays.get("images")
    if images is None or images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("Saved geometry RGB must be uint8 NHW3")
    source_frames = sequence.get("frames", [])
    if len(images) != len(source_frames) or not source_frames:
        raise ValueError("Geometry/sequence frame count mismatch")
    trajectory = arrays(run_dir / "geometry/camera_trajectory.npz")
    poses = trajectory.get("world_to_camera")
    if (poses is None or poses.shape != (len(source_frames), 4, 4) or not np.isfinite(poses).all()
            or not np.allclose(poses[:, 3], [0, 0, 0, 1])):
        raise ValueError("Invalid saved camera trajectory")
    expected_times = np.array([row.get("timestamp_ns") if row.get("timestamp_ns") is not None else -1 for row in source_frames])
    if not np.array_equal(trajectory.get("timestamp_ns"), expected_times):
        raise ValueError("Camera trajectory timestamps differ from sequence")
    extrinsic = geometry_arrays.get("extrinsic")
    if extrinsic is None or not np.allclose(poses[:, :3], extrinsic):
        raise ValueError("Camera trajectory differs from exact geometry archive")
    try:
        camera_points = np.array([np.linalg.solve(pose[:3, :3], -pose[:3, 3]) for pose in poses])
    except np.linalg.LinAlgError as error:
        raise ValueError("Camera trajectory contains singular poses") from error
    semantics_path = run_dir / "semantics/frames.jsonl"
    semantics = read(semantics_path, rows=True) if semantics_path.is_file() else []
    semantic_by_id = {}
    for row in semantics:
        if row.get("frame_id") in semantic_by_id:
            raise ValueError("Duplicate semantic frame")
        semantic_by_id[row["frame_id"]] = row
    dispositions = read(run_dir / "frames.jsonl", rows=True)
    disposition_by_id = {row["frame_id"]: row for row in dispositions}
    if len(disposition_by_id) != len(dispositions):
        raise ValueError("Duplicate frame disposition")
    frames = []
    for index, source in enumerate(source_frames):
        source_path = _safe(sequence_path.parent, source["image_path"])
        source_rgb = _rgb(source_path)
        encoded, decoded = sha256(source_path), _rgb_hash(source_rgb)
        hashes[str(source_path)] = encoded
        if encoded != source.get("encoded_file_sha256") or decoded != source.get("decoded_rgb_sha256"):
            raise ValueError("Source RGB identity mismatch")
        if source_rgb.shape[:2] != (source["height"], source["width"]):
            raise ValueError("Source image dimensions differ from sequence")
        path = _safe(cache, f"processed_frames/{index:06d}.png")
        rgb = _rgb(path)
        encoded, decoded = sha256(path), _rgb_hash(rgb)
        hashes[str(path)] = encoded
        if rgb.shape != images[index].shape or not np.array_equal(rgb, images[index]):
            raise ValueError("Processed PNG differs from exact saved geometry RGB")
        semantic = semantic_by_id.get(source["frame_id"])
        if semantic is not None:
            for key, expected in (("timestamp_ns", source.get("timestamp_ns")),
                                  ("sequence_id", sequence["sequence_id"]),
                                  ("geometry_fingerprint", geometry["geometry_fingerprint"]),
                                  ("processed_grid_id", geometry["processed_grid_id"]),
                                  ("decoded_rgb_sha256", decoded)):
                if semantic.get(key) != expected:
                    raise ValueError(f"Semantic/source grid join differs: {key}")
            input_record = semantic.get("adapter_provenance", {}).get("input", {})
            if input_record:
                if input_record.get("encoded_file_sha256") != encoded or input_record.get("decoded_rgb_sha256") != decoded:
                    raise ValueError("Semantic processed PNG identity mismatch")
                if input_record.get("source_to_processed") != geometry["transforms"][index]:
                    raise ValueError("Semantic source-to-processed transform mismatch")
                source_identity = input_record.get("source_rgb_identity", {})
                for key in ("encoded_file_sha256", "decoded_rgb_sha256", "image_path"):
                    if source_identity.get(key) != source.get(key):
                        raise ValueError("Semantic source image identity mismatch")
            for query in semantic.get("queries", []):
                for instance in query.get("instances", []):
                    if instance.get("processed_grid_id") != geometry["processed_grid_id"]:
                        raise ValueError("Mask processed grid differs")
                    mask_path = _safe(run_dir, instance["mask_path"])
                    if sha256(mask_path) != instance.get("mask_sha256"):
                        raise ValueError("Mask NPZ SHA256 mismatch")
                    mask = arrays(mask_path).get(instance.get("mask_key", "mask"))
                    if mask is None or mask.dtype != np.bool_ or mask.shape != rgb.shape[:2]:
                        raise ValueError("Mask must be boolean on exact processed RGB grid")
                    instance["mask"] = mask
        transform = geometry["transforms"][index]
        matrix = np.asarray(transform.get("matrix"), float)
        if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
                or not np.allclose(matrix[2], [0, 0, 1]) or abs(np.linalg.det(matrix)) < 1e-12
                or transform.get("source_shape") != list(source_rgb.shape[:2])
                or transform.get("processed_shape") != list(rgb.shape[:2])):
            raise ValueError("Invalid source-to-processed affine transform or dimensions")
        frames.append({"frame_id": source["frame_id"], "timestamp_ns": source.get("timestamp_ns"),
                       "rgb": rgb, "source_rgb": source_rgb, "transform": transform, "semantic": semantic,
                       "disposition": disposition_by_id.get(source["frame_id"], {"status": "not_recorded"}),
                       "source": source})
    if set(semantic_by_id) - {frame["frame_id"] for frame in frames}:
        raise ValueError("Semantic frame missing from exact sequence")
    final_arrays = arrays(run_dir / "map/voxels.npz")
    final_indices = _indices(final_arrays["voxel_indices"])
    origin = np.asarray(map_manifest["origin"], dtype=float)
    voxel_size = float(map_manifest["voxel_size"])
    if origin.shape != (3,) or not np.isfinite(origin).all() or not math.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("Invalid frozen voxel grid")
    if "centers" in final_arrays and not np.allclose(final_arrays["centers"], origin + (final_indices + .5) * voxel_size):
        raise ValueError("Saved voxel centers differ from recorded grid")
    contributions = read(run_dir / "map/contributions.jsonl", rows=True)
    by_id = {frame["frame_id"]: frame for frame in frames}
    seen = set()
    final_set = {tuple(index) for index in final_indices}
    for row in contributions:
        if row.get("contribution_id") in seen:
            raise ValueError("Duplicate saved contribution")
        seen.add(row.get("contribution_id"))
        frame = by_id.get(row.get("frame_id"))
        if frame is None:
            raise ValueError("Contribution frame missing from sequence")
        for key, expected in (("timestamp_ns", frame["timestamp_ns"]),
                              ("geometry_fingerprint", geometry["geometry_fingerprint"]),
                              ("processed_grid_id", geometry["processed_grid_id"]),
                              ("decoded_rgb_sha256", _rgb_hash(frame["rgb"]))):
            if row.get(key) != expected:
                raise ValueError(f"Contribution identity mismatch: {key}")
        if row.get("kind") in ("geometry", "semantic"):
            indices = _indices(row["voxel_indices"])
            if any(tuple(index) not in final_set for index in indices):
                raise ValueError("Journal contains voxel absent from final saved map")
            for name in (("geometry_weight",) if row["kind"] == "geometry" else ("positive_weight", "observed_weight")):
                weights = np.asarray(row[name], dtype=float)
                if weights.shape != (len(indices),) or not np.isfinite(weights).all() or np.any(weights < 0):
                    raise ValueError("Invalid journal evidence weights")
            if row["kind"] == "semantic":
                semantic = frame["semantic"]
                queries = {query["query_id"]: query for query in (semantic or {}).get("queries", [])}
                if semantic is None or semantic.get("status") not in ("ok", "partial") or any(
                        key not in queries or queries[key].get("status") != "ok" for key in row.get("query_ids", [])):
                    raise ValueError("Journal would fuse failed/unavailable semantic query")
                if np.any(np.asarray(row["positive_weight"]) > np.asarray(row["observed_weight"]) + 1e-6):
                    raise ValueError("Positive evidence exceeds observed evidence")
    native_paths = []
    if planning.get("availability") == "available":
        for plan in plans:
            if plan.get("status") == "ok":
                if plan.get("map_frame") != geometry["map_frame"]:
                    raise ValueError("Saved native plan uses another map frame")
                native_paths.append(_points(plan.get("path", [])))
    research = read(research_plan_path) if research_plan_path else None
    if research is not None:
        if (research.get("schema_version") != 1 or research.get("artifact_kind") != "research_illustration_plan"
                or research.get("research_illustration") is not True or research.get("safety_validated") is not False
                or research.get("frame_scope") != "final_cumulative_map_posthoc"):
            raise ValueError("Research plan must explicitly declare final posthoc illustration and no safety validation")
        for key in ("geometry_fingerprint", "input_fingerprint", "processed_grid_id", "archive_sha256", "map_frame", "units"):
            if research.get("source", {}).get(key) != geometry[key]:
                raise ValueError(f"Research plan source differs: {key}")
        research["validated_path"] = _points(research.get("path_points", [])) if research.get("status") == "ok" else _points([])
    return {"run_dir": run_dir, "sequence_path": sequence_path, "cache": cache,
            "run": run, "config": config, "sequence": sequence, "geometry": geometry,
            "map": map_manifest, "planning": planning, "plans": plans, "native_paths": native_paths,
            "research_plan": research, "frames": frames, "contributions": contributions,
            "final_voxel_indices": final_indices, "camera_trajectory": camera_points, "input_sha256": hashes}


def sampled_timeline(frames, *, video_fps=20, unknown_frame_seconds=.5, end_hold_seconds=2):
    for value in (video_fps, unknown_frame_seconds, end_hold_seconds):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Playback rate and holds must be finite and positive")
    if not frames:
        raise ValueError("No sampled frames")
    times = [frame.get("timestamp_ns") for frame in frames]
    known = all(isinstance(t, int) and not isinstance(t, bool) and t >= 0 for t in times)
    if known and any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Known source timestamps must be strictly increasing")
    durations = [(b - a) / 1e9 for a, b in zip(times, times[1:])] if known else [unknown_frame_seconds] * (len(frames) - 1)
    durations.append(end_hold_seconds)
    repeats = [max(1, int(math.floor(seconds * video_fps + .5))) for seconds in durations]
    if sum(repeats) > video_fps * 3600:
        raise ValueError("Movie exceeds bounded one-hour export; use a shorter sequence")
    return {"durations_seconds": durations, "repeat_counts": repeats,
            "encoded_durations_seconds": [n / video_fps for n in repeats],
            "playback_mode": "sampled_source_timestamps_held_no_interpolation" if known else "unknown_source_times_frame_index_pacing",
            "timestamp_provenance": [frame.get('source', {}).get('timestamp_provenance', {}) for frame in frames],
            "video_fps": video_fps, "end_hold_seconds": end_hold_seconds,
            "unknown_frame_seconds": unknown_frame_seconds}


def cumulative_states(data):
    geometry = set()
    states, prior_semantic = [], []
    expiry = data["map"].get("semantic_expiry_ns")
    for index, frame in enumerate(data["frames"]):
        for row in data["contributions"]:
            if row["frame_id"] != frame["frame_id"]:
                continue
            if row["kind"] == "geometry" and row.get("status") == "ok":
                geometry.update(tuple(item) for item in row["voxel_indices"])
            elif row["kind"] == "semantic":
                prior_semantic.append(row)
        totals = {}
        for row in prior_semantic:
            if expiry is not None:
                now, then = frame["timestamp_ns"], row.get("timestamp_ns")
                if now is None or then is None or now - then > expiry:
                    continue
            for voxel, positive, observed in zip(row["voxel_indices"], row["positive_weight"], row["observed_weight"]):
                key = (tuple(voxel), row.get("role"), row["concept_id"])
                total = totals.setdefault(key, [0., 0.])
                total[0] += positive
                total[1] += observed
        positives, hazards, scores = set(), set(), {}
        for (voxel, role, concept), (positive, observed) in totals.items():
            if positive > 0:
                (positives if role == "candidate_surface" else hazards).add(voxel)
                scores[voxel] = max(scores.get(voxel, 0), positive / observed if observed > 0 else 0)
        final = index == len(data["frames"]) - 1
        states.append({"frame_id": frame["frame_id"], "voxel_indices": _indices(sorted(geometry)),
                       "positive_indices": _indices(sorted(positives)), "hazard_indices": _indices(sorted(hazards)),
                       "scores": scores, "native_paths": data["native_paths"] if final else [],
                       "camera_trajectory": data["camera_trajectory"][:index + 1],
                       "research_path": data["research_plan"]["validated_path"] if final and data["research_plan"] else _points([])})
    if geometry != {tuple(item) for item in data["final_voxel_indices"]}:
        raise ValueError("Complete geometry journal does not reproduce final voxel coverage")
    return states


def voxel_faces(indices, origin, voxel_size):
    """Six actual cuboid faces in recorded map units, including negative indices."""
    corners = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
    faces = np.array([[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
                      [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])
    points = np.asarray(origin) + (_indices(indices)[:, None, :] + corners[None, :, :]) * voxel_size
    return points[:, faces]


def _font(size):
    for name in ("DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def _text(draw, xy, text, *, size=19, fill=(227, 232, 238), width=65):
    lines = textwrap.wrap(str(text), width=width) or [""]
    draw.multiline_text(xy, "\n".join(lines), font=_font(size), fill=fill, spacing=4)
    return len(lines) * (size + 4)


def _time_label(frame, first):
    t = frame["timestamp_ns"]
    if t is None:
        return "sample time unknown"
    offset = (t - first) / 1e9 if first is not None else t / 1e9
    clock = frame.get('source', {}).get('timestamp_provenance', {}).get('clock')
    label = 'video time' if clock == 'video_presentation_timeline' else 'sample time'
    return f"{label} {offset:.3f}s"


def project_mask_to_source(mask, transform, source_shape):
    """Reproject binary processed evidence using its saved source pixel mapping.

    The stored matrix maps integer source pixel centers to processed centers.
    Pillow's inverse sampling maps output source coordinates to input processed
    coordinates; center/corner conversion below preserves that same mapping.
    EXIF orientation is already included in the stored matrix.
    """
    matrix = np.asarray(transform["matrix"], float)
    sampling = matrix.copy()
    sampling[:2, 2] += .5 - matrix[:2, :2] @ np.array([.5, .5])
    result = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).transform(
        (int(source_shape[1]), int(source_shape[0])), Image.Transform.AFFINE,
        tuple(sampling[:2].reshape(-1)), resample=Image.Resampling.NEAREST, fillcolor=0)
    return np.asarray(result) != 0


def render_segmentation(data, frame, index, *, width=800, height=800, rgb_view="source"):
    canvas = Image.new("RGB", (width, height), (19, 26, 36))
    draw = ImageDraw.Draw(canvas)
    _text(draw, (24, 18), f"Video with segmentation | {index + 1}/{len(data['frames'])}", size=23)
    _text(draw, (24, 52), f"{frame['frame_id']} | {_time_label(frame, data['frames'][0]['timestamp_ns'])}", size=16, width=85)
    if data["run"].get("fixture"):
        _text(draw, (24, 85), "SYNTHETIC FIXTURE: software check; no real model accuracy claim", size=14, fill=(245, 193, 61), width=95)
    rgb = (frame["source_rgb"] if rgb_view == "source" else frame["rgb"]).astype(float)
    semantic = frame["semantic"]
    labels = []
    overlay = rgb.copy()
    # Fill each role's union once; preserve per-instance boundaries separately.
    unions, masks = {}, []
    if semantic:
        for query in semantic.get("queries", []):
            diagnostic = semantic.get("status") not in ("ok", "partial") or query.get("status") != "ok"
            role = "diagnostic" if diagnostic else query.get("role", "hazard")
            color = COLORS.get(role, COLORS["hazard"])
            instances = query.get("instances", [])
            message = f'"{query.get("original_phrase", "unknown")}" | {query.get("status", "unknown")} | {len(instances)} instance(s)'
            if not instances and query.get("status") == "ok":
                message += " | empty result"
            if query.get("error"):
                message += f" | error: {query['error']}"
            labels.append(message)
            for instance in instances:
                mask = instance["mask"]
                if rgb_view == "source":
                    mask = project_mask_to_source(mask, frame["transform"], rgb.shape[:2])
                role_union = unions.setdefault(role, np.zeros(mask.shape, bool))
                role_union |= mask
                masks.append((mask, color))
    else:
        labels.append("No saved semantic result for this sampled frame")
    if rgb_view == "source":
        valid = np.ones(frame["rgb"].shape[:2], bool)
        left, top, right, bottom = frame["transform"].get("pad_ltrb", [0, 0, 0, 0])
        if top: valid[:top] = False
        if bottom: valid[-bottom:] = False
        if left: valid[:, :left] = False
        if right: valid[:, -right:] = False
        coverage = project_mask_to_source(valid, frame["transform"], rgb.shape[:2])
        overlay[~coverage] *= .9
        inner = np.zeros(coverage.shape, bool)
        inner[1:-1, 1:-1] = coverage[1:-1, 1:-1] & coverage[:-2, 1:-1] & coverage[2:, 1:-1] & coverage[1:-1, :-2] & coverage[1:-1, 2:]
        overlay[coverage & ~inner] = (188, 191, 198)
    for role, mask in unions.items():
        overlay[mask] = .62 * overlay[mask] + .38 * np.asarray(COLORS.get(role, COLORS["hazard"]))
    for mask, color in masks:
        inner = np.zeros(mask.shape, bool)
        inner[1:-1, 1:-1] = mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:]
        overlay[mask & ~inner] = color
    image = Image.fromarray(overlay.clip(0, 255).astype(np.uint8))
    factor = min((width - 48) / image.width, (height - 300) / image.height)
    image = image.resize((max(1, round(image.width * factor)), max(1, round(image.height * factor))), Image.Resampling.LANCZOS)
    canvas.paste(image, ((width - image.width) // 2, 110 + (height - 300 - image.height) // 2))
    y = height - 175
    for label in labels:
        y += _text(draw, (24, y), label, size=17, width=85) + 3
    _text(draw, (24, height - 65), "Teal: candidate surface | Orange: hazard evidence | Pink: diagnostic mask", size=15, width=95)
    _text(draw, (24, height - 38), "Masks within model crop; outside is unknown. Sampled frames held, no interpolation.", size=15, width=95)
    return canvas


def _view_matrix(elevation=23, azimuth=-58):
    elevation, azimuth = np.radians([elevation, azimuth])
    toward = np.array([math.cos(elevation) * math.cos(azimuth), math.cos(elevation) * math.sin(azimuth), math.sin(elevation)])
    right = np.array([-math.sin(azimuth), math.cos(azimuth), 0])
    return np.stack([right, np.cross(toward, right), toward], axis=1)


def display_view_matrix(research_plan):
    """Rotate only the display camera; saved map coordinates stay native."""
    normal = (research_plan or {}).get('assumed_up_vector')
    if normal is None:
        return _view_matrix()
    up = np.asarray(normal, dtype=float)
    if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) < 1e-10:
        raise ValueError('Invalid explicitly assumed display ground direction')
    up /= np.linalg.norm(up)
    reference = np.eye(3)[int(np.argmin(np.abs(up)))]
    first = np.cross(reference, up)
    first /= np.linalg.norm(first)
    second = np.cross(up, first)
    basis = np.stack([first, second, up])
    return basis.T @ _view_matrix()


def render_voxels(data, frame, state, index, selected, *, width=800, height=800):
    canvas = Image.new("RGB", (width, height), (19, 26, 36))
    draw = ImageDraw.Draw(canvas)
    _text(draw, (24, 18), f"Growing voxel map | {index + 1}/{len(data['frames'])}", size=23)
    _text(draw, (24, 52), f"{frame['frame_id']} | {_time_label(frame, data['frames'][0]['timestamp_ns'])}", size=16, width=85)
    if data["run"].get("fixture"):
        _text(draw, (24, 85), "SYNTHETIC FIXTURE: software check; no real model accuracy claim", size=14, fill=(245, 193, 61), width=95)
    origin, size = np.array(data["map"]["origin"]), data["map"]["voxel_size"]
    view = display_view_matrix(data['research_plan'])
    final_indices = data["final_voxel_indices"]
    if len(final_indices):
        lower, upper = origin + final_indices.min(0) * size, origin + (final_indices.max(0) + 1) * size
        bounds_points = np.array([[x, y, z] for x in (lower[0], upper[0]) for y in (lower[1], upper[1]) for z in (lower[2], upper[2])])
    else:
        bounds_points = np.array([origin, origin + size])
    # Routes expand the fixed view once, never change the camera during playback.
    routes = data["native_paths"] + [data["camera_trajectory"]] + ([data["research_plan"]["validated_path"]] if data["research_plan"] else [])
    if any(len(route) for route in routes):
        bounds_points = np.concatenate([bounds_points] + [r for r in routes if len(r)])
    projected = bounds_points @ view
    low, high = projected[:, :2].min(0), projected[:, :2].max(0)
    scale = min((width - 110) / max(high[0] - low[0], size), (height - 365) / max(high[1] - low[1], size)) * .9
    center = (low + high) / 2
    def project(points):
        p = np.asarray(points) @ view
        xy = (p[..., :2] - center) * scale
        xy[..., 0] += width / 2
        xy[..., 1] = height / 2 + 5 - xy[..., 1]
        return xy, p[..., 2]
    indices = np.array([item for item in state["voxel_indices"] if tuple(item) in selected], dtype=np.int64).reshape(-1, 3)
    faces = voxel_faces(indices, origin, size)
    polygons, depth = project(faces)
    positive = {tuple(v) for v in state["positive_indices"]}
    hazard = {tuple(v) for v in state["hazard_indices"]}
    jobs = []
    for voxel, polys, depths in zip(indices, polygons, depth):
        key = tuple(voxel)
        role = "hazard" if key in hazard else "candidate_surface" if key in positive else "geometry"
        color = np.asarray(COLORS[role])
        for face_index, (polygon, z) in enumerate(zip(polys, depths)):
            shade = [.57, .96, .7, .84, .68, .83][face_index]
            jobs.append((float(z.mean()), polygon, tuple((color * shade).astype(int))))
    for _, polygon, color in sorted(jobs, key=lambda job: job[0]):
        draw.polygon([tuple(p) for p in polygon], fill=color, outline=(43, 53, 65))
    camera_xy, _ = project(state["camera_trajectory"])
    if len(camera_xy) > 1:
        draw.line([tuple(p) for p in camera_xy], fill=(107, 165, 244), width=2)
    if len(camera_xy):
        point = camera_xy[-1]
        draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill=(107, 165, 244))
        draw.text((point[0] + 8, point[1] - 10), "Camera", font=_font(14), fill=(107, 165, 244))
    def route(points, color, dashed=False):
        if len(points) < 2:
            return
        xy, _ = project(points)
        for a, b in zip(xy, xy[1:]):
            if dashed:
                for start in np.arange(0, 1, .13):
                    finish = min(start + .07, 1)
                    draw.line([tuple(a + (b - a) * start), tuple(a + (b - a) * finish)], fill=color, width=5)
            else:
                draw.line([tuple(a), tuple(b)], fill=color, width=5)
        for point, label in ((xy[0], "S"), (xy[-1], "G")):
            draw.ellipse((point[0] - 8, point[1] - 8, point[0] + 8, point[1] + 8), fill=color)
            draw.text((point[0] + 11, point[1] - 12), label, font=_font(18), fill=color)
    for path in state["native_paths"]:
        route(path, COLORS["native_path"])
    route(state["research_path"], COLORS["research"], dashed=True)
    # Native XYZ triad: display axes without treating any axis as verified gravity.
    anchor = np.array([width - 120, height - 190], float)
    for axis, name, color in zip(np.eye(3), "XYZ", [(242, 105, 104), (97, 207, 134), (107, 165, 244)]):
        delta = axis @ view
        end = anchor + np.array([delta[0], -delta[1]]) * 55
        draw.line([tuple(anchor), tuple(end)], fill=color, width=3)
        draw.text(tuple(end), name, font=_font(16), fill=color)
    unit = data["map"].get("units", "unknown")
    count = len(state["voxel_indices"])
    _text(draw, (24, height - 210), f"{count} saved voxels | {len(indices)} drawn | cube edge {size:g} {unit}", size=16, width=83)
    orientation = 'view aligned to assumed ground' if (data['research_plan'] or {}).get('assumed_up_vector') is not None else 'native XYZ view'
    _text(draw, (24, height - 184), f"Blue: recorded camera | {orientation}; axes remain native XYZ", size=15, width=90)
    _text(draw, (24, height - 155), "Grey: observed geometry | Teal/orange: any positive semantic weight", size=15, width=90)
    _text(draw, (24, height - 129), "Zero positive evidence is unknown. Evidence is not probability or safe terrain.", size=15, width=95)
    planning = data["planning"]
    _text(draw, (24, height - 100), f"Original planning: {planning.get('availability', 'unknown')} | {planning.get('reason') or 'saved diagnostic route; final frame only'}", size=15, width=95)
    if data["research_plan"]:
        _text(draw, (24, height - 61), RESEARCH_LABEL, size=15, fill=COLORS["research"], width=90)
        _text(draw, (24, height - 22), f"Purple dashed: final-map geometry route | {data['research_plan'].get('status')} | no safety claim", size=13, width=95)
    else:
        _text(draw, (24, height - 38), "Yellow: saved final diagnostic route. No navigation or safety validation.", size=15, width=95)
    return canvas


def _ffmpeg_binary():
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError("MP4 export needs ffmpeg on PATH or imageio-ffmpeg; --frames-only writes verified PNG previews") from None


def _encode(binary, directory, filename, timeline, output, threads):
    # concat repeats use image filenames, no shell interpolation or input mutation.
    concat = output / f"{filename}.frames.txt"
    entries = []
    for index, repeats in enumerate(timeline["repeat_counts"]):
        relative = f"{directory}/{index:06d}.png"
        entries.extend(f"file '{relative}'\n" for _ in range(repeats))
    concat.write_text("".join(entries), encoding="utf-8")
    target = output / filename
    result = subprocess.run([binary, "-hide_banner", "-loglevel", "error", "-n", "-threads", str(threads),
                             "-r", str(timeline["video_fps"]), "-f", "concat", "-safe", "1", "-i", str(concat),
                             "-an", "-c:v", "libx264", "-threads", str(threads), "-crf", "20",
                             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)],
                            capture_output=True, text=True, timeout=3600)
    (output / f"{filename}.encoder.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode or not target.is_file() or not target.stat().st_size:
        raise RuntimeError(f"Encoder failed for {filename}; see saved encoder log")
    return target


def export_movies(data, output, *, video_fps=20, unknown_frame_seconds=.5, end_hold_seconds=2,
                  max_voxels=6000, panel_size=800, threads=4, frames_only=False, rgb_view="source"):
    output = Path(output).resolve()
    for root in (data["run_dir"], data["sequence_path"].parent, data["cache"]):
        if output == root or output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError("Output must be separate from all saved input directories")
    if output.exists():
        raise ValueError("Refusing existing export directory; choose a new output path")
    if not isinstance(max_voxels, int) or max_voxels <= 0 or not isinstance(panel_size, int) or not 400 <= panel_size <= 1600 or panel_size % 2:
        raise ValueError("Use a positive voxel cap and an even panel size between 400 and 1600")
    if not isinstance(threads, int) or not 1 <= threads <= 4:
        raise ValueError("CPU encoder threads must be 1..4")
    if rgb_view not in ("source", "processed"):
        raise ValueError("RGB view must be source or processed")
    timeline = sampled_timeline(data["frames"], video_fps=video_fps,
                                unknown_frame_seconds=unknown_frame_seconds, end_hold_seconds=end_hold_seconds)
    states = cumulative_states(data)
    final = data["final_voxel_indices"]
    chosen = np.linspace(0, len(final) - 1, min(max_voxels, len(final)), dtype=int) if len(final) else []
    selected = {tuple(final[index]) for index in chosen}
    binary = None if frames_only else _ffmpeg_binary()
    config = {"schema_version": 1, "artifact_kind": "saved_pipeline_video_export", "read_only_inputs": True,
              "timeline": timeline, "max_voxels": max_voxels, "selected_voxels": len(selected),
              "voxel_rendering": "six_faces_recorded_grid_cuboids_stable_orthographic_XYZ",
              "display_up": (data['research_plan'] or {}).get('assumed_up_vector'),
              "display_up_is_assumed": (data['research_plan'] or {}).get('assumed_up_vector') is not None,
              "saved_map_coordinates_changed": False,
              "display_selection": "fixed_evenly_spaced_final_lexicographic_indices",
              "semantic_coloring": "any_positive_weight_not_traversability_not_probability",
              "semantic_expiry_ns": data["map"].get("semantic_expiry_ns"),
              "path_scope": "final_cumulative_map_posthoc", "research_label": RESEARCH_LABEL if data["research_plan"] else None,
              "camera_trajectory": "blue_saved_world_to_camera_centers_current_and_past_only_separate_from_planned_route",
              "panel_size": panel_size, "cpu_threads": threads, "frames_only": frames_only,
              "rgb_view": rgb_view, "mask_projection": "saved_source_to_processed_center_affine_nearest_no_extrapolation",
              "encoder": binary, "fixture": data["run"].get("fixture")}
    config["npz_limits_bytes"] = {"geometry_archive_headers": MAX_GEOMETRY_ARCHIVE_BYTES,
                                  "other_arrays": MAX_ARRAY_BYTES,
                                  "geometry_selected_members": ["images", "extrinsic"]}
    manifest = {"schema_version": 1, "status": "running", "run_dir": str(data["run_dir"]),
                "sequence_path": str(data["sequence_path"]), "input_sha256": data["input_sha256"],
                "geometry_identity": {key: data["geometry"].get(key) for key in ("geometry_fingerprint", "input_fingerprint", "processed_grid_id", "archive_sha256", "map_frame", "units", "scale", "up")},
                "coverage": [], "outputs_sha256": {}, "original_planning": data["planning"],
                "research_plan": {key: value for key, value in (data["research_plan"] or {}).items() if key != "validated_path"},
                "limitations": ["Only saved sampled frames are displayed; holds have no interpolated masks.",
                                "Masks from the exact processed grid are reprojected to source RGB through its saved transform; outside the crop is unknown.",
                                "Voxel map shows observed surfaces, not observed free space.",
                                "Semantic positive weights are candidate evidence, not terrain or safety certification.",
                                "Saved routes appear only on the final cumulative map; no online planning claim."]}
    output.mkdir(parents=True)
    def write_manifest():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "export_config.json").write_text(json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_manifest()
    try:
        for directory in ("segmentation_frames", "voxel_map_frames", "combined_frames"):
            (output / directory).mkdir()
        for index, (frame, state) in enumerate(zip(data["frames"], states)):
            segmentation = render_segmentation(data, frame, index, width=panel_size, height=panel_size, rgb_view=rgb_view)
            voxels = render_voxels(data, frame, state, index, selected, width=panel_size, height=panel_size)
            combined = Image.new("RGB", (panel_size * 2, panel_size))
            combined.paste(segmentation, (0, 0)); combined.paste(voxels, (panel_size, 0))
            for directory, image in (("segmentation_frames", segmentation), ("voxel_map_frames", voxels), ("combined_frames", combined)):
                path = output / directory / f"{index:06d}.png"
                image.save(path)
                manifest["outputs_sha256"][str(path.relative_to(output))] = sha256(path)
            semantic = frame["semantic"] or {}
            manifest["coverage"].append({"frame_id": frame["frame_id"], "timestamp_ns": frame["timestamp_ns"],
                                         "timestamp_provenance": frame['source'].get('timestamp_provenance', {}),
                                         "disposition": frame["disposition"], "semantic_status": semantic.get("status", "not_recorded"),
                                         "queries": [{"query_id": q.get("query_id"), "prompt": q.get("original_phrase"),
                                                      "status": q.get("status"), "error": q.get("error"), "instances": len(q.get("instances", []))}
                                                     for q in semantic.get("queries", [])],
                                         "cumulative_voxels": len(state["voxel_indices"]),
                                         "positive_candidate_voxels": len(state["positive_indices"]),
                                         "positive_hazard_voxels": len(state["hazard_indices"]),
                                         "encoded_repeat_count": timeline["repeat_counts"][index],
                                         "research_path_drawn": bool(len(state["research_path"]))})
            write_manifest()
        if not frames_only:
            for directory, name in (("segmentation_frames", "segmentation.mp4"), ("voxel_map_frames", "voxel_map.mp4"), ("combined_frames", "combined.mp4")):
                path = _encode(binary, directory, name, timeline, output, threads)
                manifest["outputs_sha256"][name] = sha256(path)
        manifest["status"] = "complete_frames_only" if frames_only else "complete"
        manifest["sampled_frame_count"] = len(data["frames"])
        manifest["encoded_frame_count"] = sum(timeline["repeat_counts"])
        manifest["query_status_counts"] = dict(Counter(q.get("status", "unknown") for f in data["frames"] for q in (f["semantic"] or {}).get("queries", [])))
        write_manifest()
    except Exception as error:
        manifest["status"], manifest["error"] = "failed", str(error)
        write_manifest()
        raise
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--sequence", required=True, type=Path)
    parser.add_argument("--cache", type=Path, help="Relocated exact saved geometry cache")
    parser.add_argument("--research-plan", type=Path, help="Separate assumption-labelled final-map illustration")
    parser.add_argument("--output", required=True, type=Path, help="New directory outside saved inputs")
    parser.add_argument("--video-fps", type=float, default=20)
    parser.add_argument("--unknown-frame-seconds", type=float, default=.5)
    parser.add_argument("--end-hold-seconds", type=float, default=2)
    parser.add_argument("--max-voxels", type=int, default=6000)
    parser.add_argument("--panel-size", type=int, default=800)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--frames-only", action="store_true", help="PNG previews when an MP4 encoder is unavailable")
    parser.add_argument("--rgb-view", choices=("source", "processed"), default="source", help="Original source aspect ratio by default")
    args = parser.parse_args(argv)
    try:
        data = load_export_inputs(args.run, args.sequence, cache_dir=args.cache, research_plan_path=args.research_plan)
        result = export_movies(data, args.output, video_fps=args.video_fps, unknown_frame_seconds=args.unknown_frame_seconds,
                               end_hold_seconds=args.end_hold_seconds, max_voxels=args.max_voxels,
                               panel_size=args.panel_size, threads=args.threads, frames_only=args.frames_only, rgb_view=args.rgb_view)
        print(json.dumps({"status": result["status"], "output": str(args.output.resolve()),
                          "sampled_frames": result["sampled_frame_count"]}))
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Export stopped: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
