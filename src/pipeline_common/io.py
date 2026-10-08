"""Portable strict JSON, atomic writes and named SHA256 recipes."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import os
import tempfile
import time
from contextlib import contextmanager
import numpy as np

ENCODED_HASH_RECIPE = "sha256_file_bytes_v1"
RGB_HASH_RECIPE = "sha256_rgb_uint8_c_order_v1"
PROJECT_CODE_HASH_RECIPE = "sha256_project_python_sources_v1"

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def rgb_sha256(rgb):
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("RGB hash requires uint8[H,W,3]")
    return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()

def json_default(value):
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)

def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False, default=json_default).encode()).hexdigest()

def project_code_identity(repo_root=None):
    """Fingerprint project Python sources on disk, excluding vendored models."""
    repo=Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[2]
    roots=("src/pipeline_common","src/path_mapping","src/pipelines","VLM_evaluation/src")
    paths={path for folder in roots for path in (repo/folder).rglob("*.py") if path.is_file()}
    paths.update(path for path in (repo/"src").glob("*.py") if path.is_file())
    rows=[{"path":path.relative_to(repo).as_posix(),"sha256":file_sha256(path)} for path in sorted(paths,key=lambda p:p.relative_to(repo).as_posix())
        if path.resolve().is_relative_to(repo)]
    return {"recipe":PROJECT_CODE_HASH_RECIPE,"source_roots":[*roots,"src/*.py"],
        "file_count":len(rows),"files":rows,"sha256":digest_json(rows)}

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Invalid JSON number {x}")))

@contextmanager
def _atomic_stream(path, *, binary=False):
    """Replace a file only after a complete write, using a private sibling temp."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        options = {} if binary else {"encoding": "utf-8", "newline": "\n"}
        with os.fdopen(descriptor, "wb" if binary else "w", **options) as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(4):
            try:
                temporary.replace(path)
                break
            except PermissionError as error:
                # Windows can briefly deny simultaneous replace/open operations.
                if getattr(error,"winerror",None) not in {5,32} or attempt==3: raise
                time.sleep(.01*2**attempt)
    finally:
        temporary.unlink(missing_ok=True)

def write_json(path, value):
    with _atomic_stream(path) as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False, default=json_default)
        stream.write("\n")

def write_jsonl(path, rows):
    with _atomic_stream(path) as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False, default=json_default) + "\n")

def write_npz(path, **arrays):
    if any(np.asarray(a).dtype.hasobject for a in arrays.values()):
        raise ValueError("Object/pickle arrays are prohibited")
    with _atomic_stream(path, binary=True) as stream:
        np.savez_compressed(stream, **arrays)

def safe_path(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Asset must be a relative path inside its root")
    path = (root / relative).resolve()
    if not path.is_relative_to(root): raise ValueError("Asset escapes root")
    return path
