"""Independent strict reads, portable paths and atomic hazard artifact storage.

Policy-independent primitives adapted from the legacy storage implementation.
No legacy modules are imported.
"""

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import tempfile

FRAME_KEYS = (
    "frame_id", "source", "scene_id", "sequence_id", "timestamp_s",
    "image_path", "split", "width", "height", "image_sha256",
)


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def _invalid_constant(value):
    raise ValueError(f"Nonfinite JSON number: {value}")


def _finite_float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Nonfinite JSON number: {value}")
    return parsed


def strict_loads(text):
    return json.loads(text, object_pairs_hook=_unique_object,
                      parse_constant=_invalid_constant, parse_float=_finite_float)


def read_json(path):
    value = strict_loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path):
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                value = strict_loads(line)
                if not isinstance(value, dict):
                    raise ValueError("Expected JSON object")
            except (ValueError, UnicodeError) as exc:
                raise ValueError(f"Invalid JSONL {path}, line {number}: {exc}") from exc
            records.append(value)
    return records


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def stable_fingerprint(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def validate_relative_path(value):
    if (not isinstance(value, str) or not value or "\\" in value
            or ":" in value or "\x00" in value):
        raise ValueError("Stored paths must be nonempty POSIX-relative paths")
    if (PurePosixPath(value).is_absolute() or PureWindowsPath(value).drive
            or any(part in ("", ".", "..") for part in value.split("/"))):
        raise ValueError(f"Unsafe relative path: {value!r}")
    return value


def safe_path(root, relative):
    validate_relative_path(relative)
    resolved_root = Path(root).expanduser().resolve()
    target = (resolved_root / relative).resolve()
    if not target.is_relative_to(resolved_root):
        raise ValueError(f"Path escapes its root: {relative!r}")
    return target


def validate_frames(frames):
    if not isinstance(frames, list):
        raise ValueError("Frames must be a list")
    seen = set()
    for frame in frames:
        if not isinstance(frame, dict) or not set(FRAME_KEYS) <= frame.keys():
            raise ValueError("Frame is missing hazard identity fields")
        if "region_id" in frame or "mask_path" in frame:
            raise ValueError("Legacy region fields are not hazard frames")
        for key in ("frame_id", "source", "scene_id", "sequence_id"):
            if not isinstance(frame[key], str) or not frame[key].strip():
                raise ValueError(f"frame.{key} must be nonempty text")
        if frame["frame_id"] in seen:
            raise ValueError(f"Duplicate frame_id: {frame['frame_id']}")
        seen.add(frame["frame_id"])
        validate_relative_path(frame["image_path"])
        if frame["split"] not in ("development", "test"):
            raise ValueError("frame.split must be development or test")
        timestamp = frame["timestamp_s"]
        if timestamp is not None and (isinstance(timestamp, bool)
                                     or not isinstance(timestamp, (int, float))
                                     or not math.isfinite(timestamp)):
            raise ValueError("frame.timestamp_s must be finite or null")
        for key in ("width", "height"):
            if type(frame[key]) is not int or frame[key] <= 0:
                raise ValueError(f"frame.{key} must be a positive integer")
        digest = frame["image_sha256"]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise ValueError("frame.image_sha256 must be lowercase SHA-256")


def content_identity(path):
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except FileNotFoundError:
        return {"status": "missing"}
    except OSError as exc:
        # No machine-specific path/message in portable identity.
        return {"status": "unreadable", "error_type": type(exc).__name__}
    return {"status": "present", "sha256": digest.hexdigest()}


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
    _atomic_bytes(path, canonical_bytes(value) + b"\n")


def atomic_jsonl(path, records):
    _atomic_bytes(path, b"".join(canonical_bytes(row) + b"\n" for row in records))


@contextmanager
def exclusive_lock(path):
    """OS-managed process lock released on process death; retain the lock inode."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
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
        except OSError as exc:
            raise RuntimeError(f"Hazard output is locked: {path}") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
