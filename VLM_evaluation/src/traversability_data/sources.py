"""Deterministic source imports retaining original RGB raster and recorded timestamps."""

from __future__ import annotations

import math
import re
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from .storage import atomic_json, data_path, exclusive_lock, hash_file, publish_file, read_json


def uniform_indices(size: int, count: int) -> list[int]:
    """Choose uniformly spaced, unique indices including the two endpoints."""
    if type(size) is not int or type(count) is not int or count < 1 or size < count:
        raise ValueError(f"Need at least {count} available frames, found {size}")
    if count == 1:
        return [(size - 1) // 2]
    return [int(math.floor(index * (size - 1) / (count - 1) + 0.5)) for index in range(count)]


def _natural(value):
    return tuple((1, int(part)) if part.isdigit() else (0, part.casefold()) for part in re.split(r"(\d+)", str(value)))


def _identifier(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ValueError(f"{name} must be a simple stable identifier (letters, digits, dot, underscore or dash)")
    return value


def _images(directory: Path, *, semantic=False) -> dict[str, Path]:
    found = {}
    stems = set()
    extensions = {".png"} if semantic else {".png", ".jpg", ".jpeg"}
    if not directory.is_dir():
        return found
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in extensions:
            _identifier(path.stem, "frame filename stem")
            if path.stem.casefold() in stems:
                raise ValueError(f"Duplicate frame stem {path.stem!r} in {directory}")
            stems.add(path.stem.casefold())
            found[path.stem] = path
    return found


def _directories(roots, sequence, names):
    found = []
    for root in roots:
        if not root.is_dir():
            continue
        candidates = [root] if root.name in names else []
        candidates.extend(path for name in names for path in root.rglob(name))
        for path in candidates:
            if path.parent.name == sequence and path not in found:
                found.append(path)
    return found


def _merge_images(directories, *, semantic=False):
    merged = {}
    for directory in directories:
        for stem, path in _images(directory, semantic=semantic).items():
            if stem in merged and merged[stem] != path:
                raise ValueError(f"Ambiguous duplicate frame {stem}: {merged[stem]} and {path}")
            merged[stem] = path
    return merged


def _recipe(recipe):
    if recipe is None or recipe == "public_rellis":
        return {f"{index:05d}": {"keyframes": 20, "split": "development" if index == 0 else "test"} for index in range(5)}
    if recipe in ("legacy_phone", "legacy_70_plus_phone"):
        return {f"{index:05d}": {"keyframes": 10 if index == 0 else 15, "split": "development" if index == 0 else "test"} for index in range(5)}
    if not isinstance(recipe, dict):
        raise ValueError("recipe must be a named recipe or a recording configuration dictionary")
    if "rellis" in recipe:
        recipe = recipe["rellis"]
    if "proposed_development" in recipe or "proposed_test" in recipe:
        parsed = {}
        for split, field in (("development", "proposed_development"), ("test", "proposed_test")):
            for row in recipe.get(field, []):
                sequence = row["recording"]
                if sequence in parsed:
                    raise ValueError(f"Recording {sequence} appears in multiple splits")
                parsed[sequence] = {"keyframes": row["keyframes"], "split": split}
        recipe = parsed
    parsed = {}
    for sequence, row in recipe.items():
        if sequence not in {f"{index:05d}" for index in range(5)}:
            raise ValueError(f"Unexpected RELLIS recording {sequence!r}")
        if not isinstance(row, dict) or type(row.get("keyframes")) is not int or row["keyframes"] < 1 or row.get("split") not in {"development", "test"}:
            raise ValueError(f"Invalid recipe for recording {sequence}")
        if row["split"] != ("development" if sequence == "00000" else "test"):
            raise ValueError(f"RELLIS recording split leakage: {sequence}")
        parsed[sequence] = dict(row)
    if not parsed:
        raise ValueError("RELLIS recipe is empty")
    return parsed


def _validate_pair(image, semantic=None):
    with Image.open(image) as rgb:
        rgb.load()
        if rgb.mode not in {"RGB", "RGBA"}:
            raise ValueError(f"Source image must preserve an RGB raster: {image}")
        dimensions = rgb.size
    if semantic is not None:
        with Image.open(semantic) as mask:
            mask.load()
            if mask.size != dimensions or np.asarray(mask).ndim != 2:
                raise ValueError(f"Semantic aid must be single-channel and aligned with original RGB: {semantic}")
    return dimensions


def _publish_pairs(pairs):
    for source, destination in pairs:
        if destination.exists() and (not destination.is_file() or hash_file(source) != hash_file(destination)):
            raise FileExistsError(f"Completed input differs: {destination}")
    for source, destination in pairs:
        destination.parent.mkdir(parents=True, exist_ok=True)
        publish_file(source, destination)


def import_rellis(data_root, source_dir=None, recipe=None, *, allow_partial=True) -> dict:
    """Prepare a frozen recipe's RGB/semantic aids and report absent or short recordings."""
    root = Path(data_root).expanduser().resolve()
    if source_dir is not None:
        sources = [Path(source_dir).expanduser().resolve()]
        if not sources[0].is_dir():
            raise FileNotFoundError(f"RELLIS source directory does not exist: {sources[0]}")
    else:
        sources = [root / "raw/rellis_rgb", root / "raw/rellis_semantic"]
    frames, aids, missing, pairs = [], {}, [], []
    for sequence, settings in sorted(_recipe(recipe).items()):
        rgb_directories = _directories(sources, sequence, {"pylon_camera_node"})
        semantic_directories = _directories(sources, sequence, {"pylon_camera_node_label_id"})
        # A repeated import can use the completed prepared files if raw files were removed.
        images = _merge_images(rgb_directories) if rgb_directories else _images(root / "rellis" / sequence / "rgb")
        semantics = _merge_images(semantic_directories, semantic=True) if semantic_directories else _images(root / "rellis" / sequence / "semantic", semantic=True)
        eligible = sorted(set(images) & set(semantics) if semantics else images, key=_natural)
        count = settings["keyframes"]
        if len(eligible) < count:
            missing.append({"sequence_id": sequence, "requested": count, "available": len(eligible), "reason": "recording_absent" if not images else "insufficient_annotated_frames"})
            continue
        if not semantics:
            missing.append({"sequence_id": sequence, "requested": count, "available": 0, "reason": "semantic_ids_absent; import the exact RELLIS semantic-ID archive for automatic references"})
        for index in uniform_indices(len(eligible), count):
            stem = eligible[index]
            image, semantic = images[stem], semantics.get(stem)
            _validate_pair(image, semantic)
            frame_id = f"rellis:{sequence}:{stem}"
            image_relative = f"rellis/{sequence}/rgb/{image.name}"
            pairs.append((image, data_path(root, image_relative, must_exist=False)))
            frames.append({"frame_id": frame_id, "source": "rellis", "scene_id": "rellis_campus", "sequence_id": sequence, "timestamp_s": None, "image_path": image_relative, "split": settings["split"]})
            if semantic is not None:
                semantic_relative = f"rellis/{sequence}/semantic/{semantic.name}"
                pairs.append((semantic, data_path(root, semantic_relative, must_exist=False)))
                aids[frame_id] = semantic_relative
    if missing and not allow_partial:
        raise ValueError(f"Incomplete RELLIS preparation: {missing}. Import RGB and semantic-ID archives or use allow_partial=True to report available coverage")
    _publish_pairs(pairs)
    return {"frames": frames, "semantic_aids": aids, "missing": missing}


def import_tum(data_root, source_dir, sequence_id, split="test") -> list[dict]:
    """Import only TUM RGB listed in rgb.txt, retaining its recorded timestamps."""
    if sequence_id not in {"freiburg1_floor", "freiburg3_walking_xyz"} or split not in {"development", "test"}:
        raise ValueError("Choose the manifest's floor/walking_xyz sequence and development/test split")
    source = Path(source_dir).expanduser().resolve()
    candidates = [source / "rgb.txt"] if (source / "rgb.txt").is_file() else list(source.rglob("rgb.txt"))
    candidates = [path for path in candidates if path.parent.name == "rgbd_dataset_" + sequence_id or path.parent == source]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one {sequence_id} rgb.txt below {source}, found {len(candidates)}")
    manifest = candidates[0]
    frames, pairs, ids, paths, timestamps = [], [], set(), set(), set()
    for number, line in enumerate(manifest.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"Malformed rgb.txt row {number}: expected timestamp filename")
        timestamp = float(fields[0])
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError(f"Invalid TUM timestamp on row {number}")
        relative = fields[1]
        image = data_path(manifest.parent, relative)
        if not relative.startswith("rgb/") or image.suffix.lower() != ".png":
            raise ValueError(f"TUM RGB manifest contains a non-RGB path: {relative}")
        stem = _identifier(image.stem, "TUM frame filename")
        frame_id = f"tum:{sequence_id}:{stem}"
        if frame_id in ids or relative in paths or timestamp in timestamps:
            raise ValueError(f"Duplicate TUM frame/path/timestamp on row {number}")
        ids.add(frame_id)
        paths.add(relative)
        timestamps.add(timestamp)
        _validate_pair(image)
        destination_relative = f"tum/{sequence_id}/{relative}"
        pairs.append((image, data_path(data_root, destination_relative, must_exist=False)))
        frames.append({"frame_id": frame_id, "source": "tum", "scene_id": "tum_office", "sequence_id": sequence_id, "timestamp_s": timestamp, "image_path": destination_relative, "split": split})
    if not frames:
        raise ValueError("TUM rgb.txt contains no RGB frames")
    if any(left["timestamp_s"] > right["timestamp_s"] for left, right in zip(frames, frames[1:])):
        raise ValueError("TUM rgb.txt timestamps are not chronological")
    _publish_pairs(pairs)
    return frames


def _import_phone_video_unlocked(data_root, video_path, location, clip_id, keyframes=5) -> list[dict]:
    """Explicitly decode a phone clip at original raster size, retaining PTS provenance."""
    if location not in {"A", "B", "C"}:
        raise ValueError("Phone location must be A (development), B or C (test)")
    _identifier(clip_id, "clip_id")
    video = Path(video_path).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Phone clip is pending: {video}")
    if type(keyframes) is not int or keyframes < 1:
        raise ValueError("keyframes must be a positive integer")
    relative_base = f"phone/{location}/{clip_id}"
    metadata_path = data_path(data_root, relative_base + "/import_metadata.json", must_exist=False)
    identity = {"video_sha256": hash_file(video), "location": location, "clip_id": clip_id, "keyframes": keyframes}
    existing = read_json(metadata_path) if metadata_path.exists() else None
    if existing is not None:
        if existing.get("identity") != identity:
            raise FileExistsError("A completed phone import has different clip bytes or keyframe settings; choose a new clip_id")
        for frame in existing["frames"]:
            image_path = data_path(data_root, frame["image_path"])
            expected_hash = existing.get("image_sha256", {}).get(frame["image_path"])
            if expected_hash is not None and hash_file(image_path) != expected_hash:
                raise ValueError(f"Completed phone raster differs from its import metadata: {image_path}")
            _validate_pair(image_path)
        return existing["frames"]
    try:
        import av
    except ImportError as exc:
        raise ImportError("Phone video import needs optional PyAV (av) in the isolated data environment; existing image imports need no decoder") from exc

    def detail(frame, index):
        if getattr(frame, "is_corrupt", False):
            raise ValueError(f"Corrupt decoded phone frame {index}")
        pts, time_base = frame.pts, frame.time_base
        timestamp = float(pts * time_base) if pts is not None and time_base is not None else None
        if timestamp is not None and not math.isfinite(timestamp):
            raise ValueError(f"Invalid decoded phone timestamp {index}")
        return {"decoded_index": index, "pts": int(pts) if pts is not None else None, "time_base": str(time_base) if time_base is not None else None, "timestamp_s": timestamp, "width": frame.width, "height": frame.height, "display_rotation_degrees": getattr(frame, "rotation", None)}

    decoded = []
    with av.open(str(video)) as container:
        if not container.streams.video:
            raise ValueError("Phone clip has no video stream")
        stream = container.streams.video[0]
        stream_metadata = dict(stream.metadata)
        for index, frame in enumerate(container.decode(stream)):
            decoded.append(detail(frame, index))
    indices = uniform_indices(len(decoded), keyframes)
    recorded_times = [row["timestamp_s"] for row in decoded if row["timestamp_s"] is not None]
    if any(left > right for left, right in zip(recorded_times, recorded_times[1:])):
        raise ValueError("Decoded phone presentation timestamps are not chronological")
    root = Path(data_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    frames, pairs = [], []
    with tempfile.TemporaryDirectory(prefix=".phone-import-", dir=root) as temporary:
        with av.open(str(video)) as container:
            stream = container.streams.video[0]
            for index, frame in enumerate(container.decode(stream)):
                if index not in indices:
                    continue
                if detail(frame, index) != decoded[index]:
                    raise ValueError("Phone decoding changed between timestamp discovery and extraction")
                image = frame.to_image()
                if image.size != (frame.width, frame.height):
                    raise ValueError("Phone decoder changed the original RGB raster dimensions")
                image_relative = f"{relative_base}/rgb/{index:08d}.png"
                staged = Path(temporary) / f"{index:08d}.png"
                image.save(staged, format="PNG")
                pairs.append((staged, data_path(root, image_relative, must_exist=False)))
                frames.append({"frame_id": f"phone:{location}_{clip_id}:{index:08d}", "source": "phone", "scene_id": f"phone:{location}", "sequence_id": f"{location}_{clip_id}", "timestamp_s": decoded[index]["timestamp_s"], "image_path": image_relative, "split": "development" if location == "A" else "test"})
        if len(frames) != keyframes:
            raise ValueError("Phone extraction did not produce every selected keyframe")
        _publish_pairs(pairs)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(metadata_path, {"identity": identity, "source_filename": video.name, "decoder": "PyAV", "decoder_version": av.__version__, "ffmpeg_library_versions": getattr(av, "library_versions", {}), "stream_metadata": stream_metadata, "orientation_transform_applied": "none", "sampling": "uniform decoded-frame indices including endpoints", "decoded_frame_count": len(decoded), "selected_frame_provenance": [decoded[index] for index in indices], "frames": frames, "image_sha256": {frame["image_path"]: hash_file(data_path(root, frame["image_path"])) for frame in frames}})
    return frames


def import_phone_video(data_root, video_path, location, clip_id, keyframes=5) -> list[dict]:
    """Explicitly decode original RGB/PTS; preserve an equal completed clip import."""
    if location not in {"A", "B", "C"}:
        raise ValueError("Phone location must be A (development), B or C (test)")
    _identifier(clip_id, "clip_id")
    if not Path(video_path).expanduser().is_file():
        raise FileNotFoundError(f"Phone clip is pending: {video_path}")
    lock_path = data_path(data_root, f"phone/{location}/{clip_id}/import.lock", must_exist=False)
    with exclusive_lock(lock_path):
        return _import_phone_video_unlocked(data_root, video_path, location, clip_id, keyframes)
