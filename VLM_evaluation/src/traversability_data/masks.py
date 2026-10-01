"""Aligned SAM-mask import and deterministic, immutable region selection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from .storage import (
    atomic_json,
    atomic_jsonl,
    data_path,
    exclusive_lock,
    hash_file,
    publish_file,
    read_json,
    read_jsonl,
)
from .validation import validate_records


SELECTION_RULE = "binary_geometry_dedup_area_thirds_sha256_v1"
_ORIGINS = {"external_sam", "sam2", "sam2_automatic", "synthetic_fixture"}


def _is_fixture(run_dir: Path, frames: list[dict]) -> bool:
    metadata = read_json(run_dir / "metadata" / "data.json", default={})
    return metadata.get("is_fixture") is True and all(
        frame.get("source") == "fixture" for frame in frames
    )


def _binary_mask(path: Path, image_size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        if image.format != "PNG" or len(image.getbands()) != 1:
            raise ValueError(f"Mask must be a single-channel PNG: {path}")
        values = np.asarray(image)
        if image.size != image_size or values.ndim != 2:
            raise ValueError(f"Mask dimensions do not match original RGB: {path}")
        unique = np.unique(values)
        if len(unique) > 2 or (len(unique) == 2 and unique[0] != 0):
            raise ValueError(f"Mask must contain zero and a single foreground value: {path}")
        mask = values != 0
        if not mask.any():
            raise ValueError(f"Mask is empty: {path}")
        return mask.copy()


def _geometry(mask: np.ndarray) -> dict:
    rows, columns = np.nonzero(mask)
    height, width = mask.shape
    digest = hashlib.sha256()
    digest.update(f"{height},{width}\0".encode("ascii"))
    digest.update(np.packbits(mask, axis=None).tobytes())
    return {
        "geometry_sha256": digest.hexdigest(),
        "area": int(mask.sum()),
        "bbox_xywh": [
            int(columns.min()),
            int(rows.min()),
            int(columns.max() - columns.min() + 1),
            int(rows.max() - rows.min() + 1),
        ],
        "centroid_xy": [float(columns.mean()), float(rows.mean())],
        "dimensions_hw": [height, width],
    }


def _origin(value: str) -> str:
    if value not in _ORIGINS:
        raise ValueError(
            "Mask origin must identify independent SAM predictions; semantic/reference "
            "dataset masks are annotation aids and cannot be imported as SAM masks."
        )
    return value


def import_sam_masks(
    run_dir, data_root, mask_records, *, mask_origin="external_sam"
) -> list[dict]:
    """Import original-RGB-aligned binary PNGs, preserving existing region data.

    Each record supplies region_id, frame_id and mask_path (or source_mask_path).
    Source paths may be external absolute paths or relative to data_root. Optional
    generator metadata stays in metadata/masks.json, outside inference records.
    """
    from .run import add_regions

    run_dir, data_root = Path(run_dir), Path(data_root)
    origin = _origin(mask_origin)
    records = list(mask_records)
    if len({record.get("region_id") for record in records}) != len(records):
        raise ValueError("Duplicate region IDs in mask import")
    frames = read_jsonl(run_dir / "frames.jsonl")
    frame_by_id = {frame["frame_id"]: frame for frame in frames}
    run_metadata = read_json(run_dir / "metadata" / "data.json", default={})
    prepared_image_hashes = run_metadata.get("image_sha256", {})
    semantic_aid_paths = {
        data_path(data_root, relative, must_exist=False)
        for relative in run_metadata.get("semantic_aids", {}).values()
    }
    validate_records(frames, [], data_root=data_root)
    if origin == "synthetic_fixture" and not _is_fixture(run_dir, frames):
        raise ValueError("Synthetic masks are allowed only in a marked fixture run")
    prepared = []
    for record in records:
        region_id, frame_id = record.get("region_id"), record.get("frame_id")
        if not isinstance(region_id, str) or not region_id:
            raise ValueError("Mask import requires a nonempty region_id")
        if frame_id not in frame_by_id:
            raise ValueError(f"Unknown frame ID: {frame_id}")
        declared = record.get("mask_origin", record.get("origin", origin))
        if _origin(declared) != origin:
            raise ValueError(f"Conflicting mask origins for {region_id}")
        source = record.get("source_mask_path", record.get("mask_path"))
        if source is None:
            raise ValueError(f"Mask import requires mask_path for {region_id}")
        source = Path(source)
        source = source if source.is_absolute() else data_path(data_root, source.as_posix())
        if source.resolve() in semantic_aid_paths:
            raise ValueError(
                f"Known dataset semantic aid cannot be imported as a SAM prediction: {source}"
            )
        image_path = data_path(data_root, frame_by_id[frame_id]["image_path"])
        with Image.open(image_path) as image:
            image_size = image.size
        image_hash = hash_file(image_path)
        if prepared_image_hashes.get(frame_id, image_hash) != image_hash:
            raise ValueError(f"Prepared RGB changed for frame {frame_id}")
        if record.get("image_sha256") not in (None, image_hash):
            raise ValueError(f"Producer RGB hash does not match frame {frame_id}")
        mask = _binary_mask(source, image_size)
        geometry = _geometry(mask)
        relevant = record.get("planning_relevant", True)
        if not isinstance(relevant, bool):
            raise ValueError("planning_relevant must be boolean")
        generator = record.get("generator")
        json.dumps(generator, allow_nan=False)
        frame_key = hashlib.sha256(frame_id.encode("utf-8")).hexdigest()
        region_key = hashlib.sha256(region_id.encode("utf-8")).hexdigest()
        relative = f"prepared_masks/sam/{frame_key}/{region_key}.png"
        prepared.append(({
            "region_id": region_id,
            "frame_id": frame_id,
            "mask_path": relative,
            "planning_relevant": relevant,
            "selected_for_classification": False,
        }, mask, {
            "origin": origin,
            "image_sha256": image_hash,
            **geometry,
            "generator": generator,
        }))

    # add_regions owns this same lock; publish first, then call it without nesting.
    with exclusive_lock(run_dir / ".data.lock"):
        existing = read_jsonl(run_dir / "regions.jsonl", missing_ok=True)
        by_id = {region["region_id"]: region for region in existing}
        provenance = read_json(run_dir / "metadata" / "masks.json", default={})
        frozen = read_json(run_dir / "metadata" / "selection.json", default={}).get("frames", {})
        for region, mask, details in prepared:
            previous = by_id.get(region["region_id"])
            if previous is None and region["frame_id"] in frozen:
                raise ValueError(f"Cannot add masks to frozen frame {region['frame_id']}")
            if previous is not None:
                if any(previous[key] != region[key] for key in ("frame_id", "planning_relevant")):
                    raise ValueError(f"Existing region differs: {region['region_id']}")
                old_path = data_path(data_root, previous["mask_path"])
                old_mask = _binary_mask(old_path, (mask.shape[1], mask.shape[0]))
                if not np.array_equal(mask, old_mask):
                    raise ValueError(f"Existing mask differs: {region['region_id']}")
                region.update(previous)
            destination = data_path(data_root, region["mask_path"], must_exist=False)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and not np.array_equal(
                mask, _binary_mask(destination, (mask.shape[1], mask.shape[0]))
            ):
                raise ValueError(f"Completed destination mask differs: {region['region_id']}")
            if not destination.exists():
                with tempfile.NamedTemporaryFile(
                    prefix=".mask-", suffix=".png", dir=destination.parent, delete=False
                ) as temporary:
                    temporary_path = Path(temporary.name)
                try:
                    Image.fromarray(mask.astype(np.uint8) * 255).save(temporary_path, format="PNG")
                    publish_file(temporary_path, destination)
                finally:
                    temporary_path.unlink(missing_ok=True)
            details["mask_sha256"] = hash_file(destination)
            previous_details = provenance.get(region["region_id"])
            if previous_details is not None and previous_details != details:
                raise ValueError(f"Existing mask provenance differs: {region['region_id']}")
        proposed = dict(by_id)
        proposed.update({region["region_id"]: region for region, _, _ in prepared})
        validate_records(frames, list(proposed.values()), data_root=data_root)

    result = [region for region, _, _ in prepared]
    add_regions(
        run_dir, result, data_root,
        provenance={region["region_id"]: details for region, _, details in prepared},
    )
    return result


def _rank(seed: int, frame_id: str, region_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{frame_id}\0{region_id}".encode("utf-8")).hexdigest()


def _select(candidates: list[dict], frame_id: str, seed: int) -> tuple[list[str], dict]:
    unique, duplicate_of = {}, {}
    for candidate in sorted(candidates, key=lambda item: item["region_id"]):
        if not candidate["planning_relevant"]:
            continue
        geometry = candidate["geometry_sha256"]
        if geometry in unique:
            duplicate_of[candidate["region_id"]] = unique[geometry]["region_id"]
        else:
            unique[geometry] = candidate
    ordered = sorted(unique.values(), key=lambda item: (item["area"], item["region_id"]))
    if len(ordered) <= 5:
        return sorted(item["region_id"] for item in ordered), duplicate_of
    quotient, remainder = divmod(len(ordered), 3)
    lengths = [quotient + (index < remainder) for index in range(3)]
    selected, offset = [], 0
    rank = lambda item: (_rank(seed, frame_id, item["region_id"]), item["region_id"])
    for length, quota in zip(lengths, (2, 1, 2)):
        selected.extend(sorted(ordered[offset:offset + length], key=rank)[:quota])
        offset += length
    if len(selected) < 5:
        chosen = {item["region_id"] for item in selected}
        remaining = [item for item in ordered if item["region_id"] not in chosen]
        selected.extend(sorted(remaining, key=rank)[:5 - len(selected)])
    return sorted(item["region_id"] for item in selected), duplicate_of


def freeze_selection(run_dir, data_root, seed=0) -> dict:
    """Freeze up to five independent area-stratified masks per frame.

    An existing freeze may be resumed unchanged or extended with new frame IDs.
    Every original mask remains in regions.jsonl, including exact duplicates.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("Selection seed must be an integer")
    run_dir, data_root = Path(run_dir), Path(data_root)
    with exclusive_lock(run_dir / ".data.lock"):
        frames = read_jsonl(run_dir / "frames.jsonl")
        regions = read_jsonl(run_dir / "regions.jsonl", missing_ok=True)
        validate_records(frames, regions, data_root=data_root)
        previous = read_json(run_dir / "metadata" / "selection.json", default={})
        if previous and (
            previous.get("rule") != SELECTION_RULE or previous.get("seed") != seed
        ):
            raise ValueError("Selection rule/seed differs from the existing freeze")
        previous_frames = previous.get("frames", {})
        frame_ids = {frame["frame_id"] for frame in frames}
        if not set(previous_frames).issubset(frame_ids):
            raise ValueError("A previously frozen frame was removed")
        metadata = {
            "version": 1,
            "rule": SELECTION_RULE,
            "seed": seed,
            "hash_rank_input": "UTF-8: seed + NUL + frame_id + NUL + region_id",
            "deduplication": "exact binary geometry; lowest lexicographic region_id",
            "strata": "area ascending, then region_id; contiguous thirds; remainder earliest",
            "quotas_small_medium_large": [2, 1, 2],
            "pending_frame_ids": [],
            "frames": {},
        }
        selected_ids = set()
        by_frame = {frame_id: [] for frame_id in frame_ids}
        for region in regions:
            by_frame[region["frame_id"]].append(region)
        for frame in sorted(frames, key=lambda item: item["frame_id"]):
            frame_id = frame["frame_id"]
            if not by_frame[frame_id]:
                if frame_id in previous_frames:
                    raise ValueError(f"Masks were removed from frozen frame: {frame_id}")
                metadata["pending_frame_ids"].append(frame_id)
                continue
            image_path = data_path(data_root, frame["image_path"])
            with Image.open(image_path) as image:
                size = image.size
            candidates = []
            for region in sorted(by_frame[frame_id], key=lambda item: item["region_id"]):
                mask_path = data_path(data_root, region["mask_path"])
                mask = _binary_mask(mask_path, size)
                candidates.append({
                    "region_id": region["region_id"],
                    "mask_path": region["mask_path"],
                    "mask_sha256": hash_file(mask_path),
                    "planning_relevant": region["planning_relevant"],
                    **_geometry(mask),
                })
            selected, duplicate_of = _select(candidates, frame_id, seed)
            details = {
                "frame_record": dict(frame),
                "image_path": frame["image_path"],
                "image_sha256": hash_file(image_path),
                "dimensions_hw": [size[1], size[0]],
                "candidates": candidates,
                "duplicate_of": duplicate_of,
                "selected_region_ids": selected,
            }
            if frame_id in previous_frames:
                if previous_frames[frame_id] != details:
                    raise ValueError(f"Frozen frame/masks changed: {frame_id}")
                if any(
                    region["selected_for_classification"] != (region["region_id"] in selected)
                    for region in by_frame[frame_id]
                ):
                    raise ValueError(f"Frozen selection flags changed: {frame_id}")
            metadata["frames"][frame_id] = details
            selected_ids.update(selected)
        updated = [
            {**region, "selected_for_classification": region["region_id"] in selected_ids}
            for region in regions
        ]
        atomic_jsonl(run_dir / "regions.jsonl", updated)
        atomic_json(run_dir / "metadata" / "selection.json", metadata)
        return metadata
