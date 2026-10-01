"""Portable paths and atomic, preserving artifact storage."""

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import tempfile


@dataclass(frozen=True)
class DataRoots:
    data_root: Path
    run_root: Path


def repository_root(start=None):
    """Find the checkout using its fixed interface document."""
    here = Path(start or __file__).resolve()
    if here.is_file():
        here = here.parent
    for candidate in (here, *here.parents):
        if (candidate / "docs/interfaces.md").is_file():
            return candidate
    raise FileNotFoundError("Cannot locate repository; pass repo_root explicitly.")


def resolve_roots(repo_root=None, data_root=None, run_root=None):
    """Resolve explicit arguments, environment variables, then repo defaults."""
    repo = Path(repo_root).resolve() if repo_root else repository_root()
    def resolve(value, variable, default):
        path = Path(value or os.environ.get(variable) or default).expanduser()
        return (path if path.is_absolute() else repo / path).resolve()
    return DataRoots(resolve(data_root, "TRAVERSABILITY_DATA_ROOT", "data"),
                     resolve(run_root, "TRAVERSABILITY_RUN_ROOT", "runs"))


def data_path(root, relative, must_exist=True):
    """Resolve a safe POSIX-relative path inside the configured root."""
    if not isinstance(relative, str) or not relative or "\\" in relative or "\0" in relative:
        raise ValueError(f"Expected a POSIX relative path: {relative!r}")
    parts = relative.split("/")
    if (PurePosixPath(relative).is_absolute() or PureWindowsPath(relative).drive
            or any(part in ("", ".", "..") or ":" in part for part in parts)):
        raise ValueError(f"Unsafe data path: {relative}")
    base = Path(root).resolve()
    result = base.joinpath(*parts).resolve()
    if not result.is_relative_to(base):
        raise ValueError(f"Data path escapes root: {relative}")
    if must_exist and not result.is_file():
        raise FileNotFoundError(f"Missing data file: {result}")
    return result


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"Invalid JSON number: {value}")


def read_json(path, default=None):
    """Read strict JSON; return default only if the file does not exist."""
    path = Path(path)
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object,
                      parse_constant=_constant)


def read_jsonl(path, missing_ok=False):
    """Read UTF-8 object rows, rejecting malformed or blank published rows."""
    path = Path(path)
    if missing_ok and not path.exists():
        return []
    result = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line, object_pairs_hook=_object, parse_constant=_constant)
                if not isinstance(row, dict):
                    raise ValueError("Expected a JSON object")
            except ValueError as error:
                raise ValueError(f"{path}:{number}: {error}") from error
            result.append(row)
    return result


def _atomic(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_json(path, value):
    """Atomically publish a complete JSON object."""
    _atomic(path, (json.dumps(value, sort_keys=True, ensure_ascii=False,
                             allow_nan=False, indent=2) + "\n").encode("utf-8"))


def atomic_jsonl(path, records):
    """Atomically publish complete JSONL; callers merge existing rows first."""
    content = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False,
                                  allow_nan=False) + "\n" for row in records)
    _atomic(path, content.encode("utf-8"))


@contextmanager
def exclusive_lock(path):
    """Use an OS-managed, nonblocking lock released after process failure."""
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
        except OSError as error:
            raise RuntimeError(f"Artifact is locked by another writer: {path}") from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def hash_file(path):
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish_file(source, destination):
    """Copy a file atomically; preserve equal destinations and reject conflicts."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(destination.with_name(destination.name + ".lock")):
        if destination.exists():
            if hash_file(source) != hash_file(destination):
                raise ValueError(f"Completed file differs; use a new destination: {destination}")
            return destination
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".import-",
                                             suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                with source.open("rb") as incoming:
                    shutil.copyfileobj(incoming, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return destination


def merge_records(existing, incoming, id_key):
    """Merge immutable records by ID, rejecting duplicates and changed records."""
    result = {}
    for rows in (existing, incoming):
        seen = set()
        for row in rows:
            identifier = row.get(id_key)
            if not isinstance(identifier, str) or not identifier:
                raise ValueError(f"Missing {id_key}")
            if identifier in seen:
                raise ValueError(f"Duplicate {id_key}: {identifier}")
            seen.add(identifier)
            if identifier in result and result[identifier] != row:
                raise ValueError(f"Existing {id_key} changed: {identifier}")
            result[identifier] = row
    return [result[key] for key in sorted(result)]
