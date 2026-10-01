"""Local CPU-test helpers; all generated inputs live in temporary directories."""

import hashlib
import json
from pathlib import Path
import sys

COMPONENT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(COMPONENT / "src"))


def config(base, run_name="fixture"):
    base = Path(base)
    return {
        "data_root": str(base / "data"),
        "run_root": str(base / "runs"),
        "cache_root": str(base / "cache"),
        "run_name": run_name,
    }


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_snapshot(root):
    root = Path(root)
    return {
        path.relative_to(root).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*") if path.is_file()
    }


FRAME_KEYS = {
    "frame_id", "source", "scene_id", "sequence_id", "timestamp_s", "image_path",
    "split", "width", "height", "image_sha256",
}
REFERENCE_KEYS = {
    "frame_id", "scored_concepts", "present_concepts", "absence_scoring_eligible",
    "concept_masks", "concept_pixel_counts", "valid_mask_path", "annotation_source",
    "reference_scope", "status",
}
