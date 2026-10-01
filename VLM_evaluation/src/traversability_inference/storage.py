"""Strict portable manifest reads and crash-safe local artifact storage."""

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import tempfile


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Invalid JSON number: {value}")


def _loads(text):
    return json.loads(text, object_pairs_hook=_unique_object,
                      parse_constant=_invalid_constant)


def read_json(path):
    """Read one strict JSON object, rejecting duplicate keys and nonfinite values."""
    path = Path(path)
    try:
        value = _loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def read_jsonl(path):
    """Read strict UTF-8 JSONL objects; blank or malformed published rows fail."""
    path = Path(path)
    records = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                try:
                    record = _loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("Expected a JSON object")
                except ValueError as error:
                    raise ValueError(f"Invalid JSONL in {path}, line {number}: {error}") from error
                records.append(record)
    except UnicodeError as error:
        raise ValueError(f"Invalid UTF-8 in {path}") from error
    return records


def _string(record, key, kind):
    if not isinstance(record.get(key), str) or not record[key].strip():
        raise ValueError(f"{kind}.{key} must be a nonempty string")


def validate_relative_path(value):
    """Return a safe POSIX manifest path, never an absolute or parent path."""
    if (not isinstance(value, str) or not value or "\\" in value
            or ":" in value or "\x00" in value):
        raise ValueError("Manifest input paths must be nonempty POSIX relative paths")
    parts = value.split("/")
    if (PurePosixPath(value).is_absolute() or PureWindowsPath(value).drive
            or any(part in ("", ".", "..") for part in parts)):
        raise ValueError(f"Unsafe manifest input path: {value}")
    return value


def _unique_ids(records, key, kind):
    seen = set()
    for record in records:
        _string(record, key, kind)
        value = record[key]
        if value in seen:
            raise ValueError(f"Duplicate {key}: {value}")
        seen.add(value)
    return seen


def validate_manifests(frames, regions):
    """Validate frozen input records and joins without consulting annotations."""
    frame_ids = _unique_ids(frames, "frame_id", "frame")
    _unique_ids(regions, "region_id", "region")
    for frame in frames:
        for key in ("source", "scene_id", "sequence_id"):
            _string(frame, key, "frame")
        validate_relative_path(frame.get("image_path"))
        if frame.get("split") not in ("development", "test"):
            raise ValueError(f"Invalid frame split: {frame.get('split')}")
        if "timestamp_s" not in frame:
            raise ValueError("frame.timestamp_s is required, and may be null")
        timestamp = frame["timestamp_s"]
        if timestamp is not None and (isinstance(timestamp, bool)
                                     or not isinstance(timestamp, (int, float))
                                     or not math.isfinite(timestamp)):
            raise ValueError("frame.timestamp_s must be a finite number or null")
    for region in regions:
        _string(region, "frame_id", "region")
        if region["frame_id"] not in frame_ids:
            raise ValueError(f"Region refers to missing frame: {region['frame_id']}")
        validate_relative_path(region.get("mask_path"))
        for key in ("planning_relevant", "selected_for_classification"):
            if type(region.get(key)) is not bool:
                raise ValueError(f"region.{key} must be a boolean")


def read_manifests(run_dir):
    """Return validated frozen frames and regions, preserving every field."""
    run_dir = Path(run_dir)
    frames = read_jsonl(run_dir / "frames.jsonl")
    regions = read_jsonl(run_dir / "regions.jsonl")
    validate_manifests(frames, regions)
    return frames, regions


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def stable_fingerprint(value):
    """Hash canonical JSON without dependence on dictionary insertion order."""
    return hashlib.sha256(_canonical(value)).hexdigest()


def content_identity(path):
    """Hash one input's bytes, representing a missing file deterministically."""
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except FileNotFoundError:
        return {"status": "missing"}
    return {"status": "present", "sha256": digest.hexdigest()}


def input_identity(frames, regions, data_root):
    """Fingerprint frozen records and image/mask bytes, including missing sentinels."""
    from .preparation import input_path

    validate_manifests(frames, regions)
    paths = {frame["image_path"] for frame in frames}
    paths.update(region["mask_path"] for region in regions)
    contents = {}
    for relative in sorted(paths):
        contents[relative] = content_identity(input_path(data_root, relative))
    # IDs make identity insensitive to shuffled manifest rows, while every field
    # (including original timestamps) remains part of the frozen input identity.
    return {
        "frames": sorted(frames, key=lambda item: item["frame_id"]),
        "regions": sorted(regions, key=lambda item: item["region_id"]),
        "contents": contents,
    }


def _atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{path.name}.",
                                         suffix=".tmp", dir=path.parent,
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        if os.name != "nt":
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_json(path, value):
    """Publish a JSON artifact with a flushed same-directory atomic replacement."""
    _atomic_bytes(path, _canonical(value) + b"\n")


def atomic_jsonl(path, records):
    """Publish complete accumulated JSONL without exposing a partial final file."""
    _atomic_bytes(path, b"".join(_canonical(record) + b"\n" for record in records))


@contextmanager
def exclusive_lock(path):
    """Hold a nonblocking process lock; the OS releases it after a crash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        # Leave the file in place: unlinking a lock permits separate lock inodes.
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"Inference output is locked: {path}") from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
