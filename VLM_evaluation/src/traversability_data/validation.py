"""Validate the fixed JSONL contracts and aligned image/mask joins."""

import math
import hashlib
from collections import Counter

from .storage import data_path

LABELS = {"traversable", "non_traversable", "unknown"}
QUALITIES = {"valid", "mixed", "broken", "unchecked"}


def _index(rows, key):
    result = {}
    for row in rows:
        identifier = row.get(key)
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"Missing {key}")
        if identifier in result:
            raise ValueError(f"Duplicate {key}: {identifier}")
        result[identifier] = row
    return result


def validate_records(frames, regions, annotations=None, data_root=None):
    """Validate IDs, labels, splits and optionally original RGB/mask dimensions."""
    frame_index = _index(frames, "frame_id")
    region_index = _index(regions, "region_id")
    groups, image_splits, sizes, pixel_splits = {}, {}, {}, {}
    for frame in frames:
        for field in ("source", "scene_id", "sequence_id", "image_path"):
            if not isinstance(frame.get(field), str) or not frame[field].strip():
                raise ValueError(f"Frame {frame['frame_id']} missing {field}")
        split = frame.get("split")
        if split not in {"development", "test"}:
            raise ValueError(f"Invalid split: {split}")
        if "timestamp_s" not in frame:
            raise ValueError("timestamp_s is required (null is allowed)")
        timestamp = frame["timestamp_s"]
        if timestamp is not None and (isinstance(timestamp, bool)
                or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp)):
            raise ValueError("timestamp_s must be finite or null")
        key = (frame["source"], frame["sequence_id"])
        if frame["source"] == "phone":
            key = ("phone", frame["scene_id"])
        if groups.setdefault(key, split) != split:
            raise ValueError(f"Recording/location split leakage: {key}")
        if image_splits.setdefault(frame["image_path"], split) != split:
            raise ValueError(f"Image split leakage: {frame['image_path']}")
        if frame["source"] == "rellis":
            expected = "development" if frame["sequence_id"] == "00000" else "test"
            if frame["sequence_id"] not in {f"{i:05d}" for i in range(5)} or split != expected:
                raise ValueError("RELLIS sequence split leakage or unknown recording")
        if frame["source"] == "phone":
            location = frame["scene_id"].split(":")[-1]
            if location in {"A", "B", "C"} and split != ("development" if location == "A" else "test"):
                raise ValueError("Phone location split leakage")
        path = data_path(data_root or ".", frame["image_path"], must_exist=data_root is not None)
        if data_root is not None:
            from PIL import Image
            with Image.open(path) as image:
                image.load()
                if image.mode not in {"RGB", "RGBA"}:
                    raise ValueError(f"Expected RGB image: {frame['image_path']}")
                sizes[frame["frame_id"]] = image.size
                pixels = hashlib.sha256()
                pixels.update(str(image.size).encode("ascii"))
                pixels.update(image.convert("RGB").tobytes())
                if pixel_splits.setdefault(pixels.hexdigest(), split) != split:
                    raise ValueError(f"Identical RGB pixel content split leakage: {frame['frame_id']}")
    for region in regions:
        if region.get("frame_id") not in frame_index:
            raise ValueError(f"Region joins missing frame: {region.get('frame_id')}")
        for field in ("planning_relevant", "selected_for_classification"):
            if type(region.get(field)) is not bool:
                raise ValueError(f"Region {field} must be boolean")
        if region["selected_for_classification"] and not region["planning_relevant"]:
            raise ValueError("Selected mask must be planning-relevant")
        path = data_path(data_root or ".", region.get("mask_path"), must_exist=data_root is not None)
        if data_root is not None:
            from PIL import Image
            with Image.open(path) as mask:
                mask.load()
                if mask.format != "PNG" or mask.mode not in {"1", "L", "I", "I;16"}:
                    raise ValueError("Mask must be a single-channel PNG")
                if mask.size != sizes[region["frame_id"]]:
                    raise ValueError(f"Mask dimensions do not match RGB: {region['region_id']}")
                if mask.getbbox() is None:
                    raise ValueError(f"Empty mask: {region['region_id']}")
    if annotations is not None:
        _index(annotations, "region_id")
        for row in annotations:
            if row["region_id"] not in region_index:
                raise ValueError(f"Annotation joins missing region: {row['region_id']}")
            for field in ("reference_label", "semantic_class", "hazard_type", "mask_quality",
                          "annotation_status", "robot_profile_id"):
                if field not in row:
                    raise ValueError(f"Annotation missing {field}")
            if row["reference_label"] is not None and row["reference_label"] not in LABELS:
                raise ValueError("Invalid reference label")
            if row["mask_quality"] not in QUALITIES:
                raise ValueError("Invalid mask quality")
            if row["annotation_status"] not in {"pending", "complete"}:
                raise ValueError("Invalid annotation status")
            if row["annotation_status"] == "complete" and row["reference_label"] not in LABELS:
                raise ValueError("Completed annotations require an explicit reference label")
            if row["annotation_status"] == "pending" and row["reference_label"] is not None:
                raise ValueError("Pending labels must remain null")
            if not isinstance(row["robot_profile_id"], str) or not row["robot_profile_id"]:
                raise ValueError("Missing robot profile")
            for field in ("semantic_class", "hazard_type"):
                if row[field] is not None and not isinstance(row[field], str):
                    raise ValueError(f"{field} must be a string or null")
    return {"frames": len(frames), "regions": len(regions),
            "annotations": len(annotations or []),
            "frames_by_split": dict(Counter(f["split"] for f in frames))}
