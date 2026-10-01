"""Reuse legacy preserving storage, with canonical hazard identities."""

import hashlib
import json
import re

from traversability_data.storage import (
    atomic_json, atomic_jsonl, data_path, exclusive_lock, hash_file,
    publish_file, read_json, read_jsonl,
)


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def safe_name(value):
    """Reject unsafe Windows/Linux single-component names."""
    if (not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value)
            or value.endswith(".") or value.split(".")[0].upper() in
            {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
             *(f"LPT{i}" for i in range(1, 10))}):
        raise ValueError(f"Unsafe filesystem name: {value!r}")
    return value


def fingerprint(metadata, frames, references):
    identity = {key: value for key, value in metadata.items() if key != "dataset_fingerprint"}
    return digest({"metadata": identity, "frames": frames, "references": references})


def input_signature(frames, references, assets):
    """Pin the independently evaluable RGB/reference inputs before inference."""
    paths = {row["image_path"] for row in frames}
    for row in references:
        paths.add(row["valid_mask_path"])
        paths.update(row["concept_masks"].values())
    return digest({"frames": sorted(frames, key=lambda row: row["frame_id"]),
                   "references": sorted(references, key=lambda row: row["frame_id"]),
                   "assets": {path: assets[path] for path in sorted(paths)}})
