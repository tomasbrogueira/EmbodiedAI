"""Prepare resumable runs without changing existing labels or inputs."""

from collections import Counter
from pathlib import Path

from .storage import (atomic_json, atomic_jsonl, data_path, exclusive_lock, hash_file,
                      merge_records, publish_file, read_json, read_jsonl)
from .validation import validate_records

DEFAULT_POLICY = "rellis_material_v1"
DEFAULT_POLICY_TEXT = (
    "Dirt, asphalt and concrete are semantically permitted, conditional on geometric "
    "feasibility. Obstacles, liquids, mud, rubble and non-support classes are avoided. "
    "Grass and void are unresolved. This material policy does not establish physical traversability."
)


def pending_annotation(region_id, robot_profile_id=DEFAULT_POLICY):
    """Create a missing annotation, never a reference-unknown example."""
    return {"region_id": region_id, "reference_label": None, "semantic_class": None,
            "hazard_type": None, "mask_quality": "unchecked", "annotation_status": "pending",
            "robot_profile_id": robot_profile_id}


def _ensure_annotations(run_dir, frames, regions, data_root, robot_profile_id):
    with exclusive_lock(run_dir / ".annotations.lock"):
        annotations = read_jsonl(run_dir / "annotations.jsonl", missing_ok=True)
        validate_records(frames, regions, annotations, data_root=data_root)
        existing = {row["region_id"]: row for row in annotations}
        for region in regions:
            existing.setdefault(region["region_id"], pending_annotation(region["region_id"], robot_profile_id))
        atomic_jsonl(run_dir / "annotations.jsonl", [existing[key] for key in sorted(existing)])


def prepare_run(run_dir, frames, data_root, *, metadata=None, semantic_aids=None):
    """Append immutable frames and initialize pending rows while preserving completed work."""
    run_dir = Path(run_dir)
    incoming = list(frames)
    with exclusive_lock(run_dir / ".data.lock"):
        existing_frames = read_jsonl(run_dir / "frames.jsonl", missing_ok=True)
        existing_ids = {frame["frame_id"] for frame in existing_frames}
        sampled_recordings = {(frame["source"], frame["sequence_id"]) for frame in existing_frames}
        for frame in incoming:
            if frame["frame_id"] not in existing_ids and (frame["source"], frame["sequence_id"]) in sampled_recordings:
                raise ValueError("Prepared keyframe sample changed within a recording; use a new run")
        frames = merge_records(existing_frames, incoming, "frame_id")
        regions = read_jsonl(run_dir / "regions.jsonl", missing_ok=True)
        validate_records(frames, regions, data_root=data_root)
        # Metadata updates share the annotation lock with automatic reference transfer.
        with exclusive_lock(run_dir / ".annotations.lock"):
            saved = read_json(run_dir / "metadata/data.json", default={})
            supplied = dict(metadata or {})
            supplied.setdefault("robot_profile_id", saved.get("robot_profile_id", DEFAULT_POLICY))
            supplied.setdefault("robot_policy", saved.get("robot_policy", DEFAULT_POLICY_TEXT))
            supplied.setdefault("is_fixture", saved.get("is_fixture", False))
            for key in ("robot_profile_id", "robot_policy", "is_fixture", "recipe", "expected_frames"):
                if key in saved and key in supplied and saved[key] != supplied[key]:
                    raise ValueError(f"Frozen run metadata changed: {key}; use a new run")
            saved.update(supplied)
            if saved["is_fixture"] != all(frame["source"] == "fixture" for frame in frames) and frames:
                raise ValueError("Fixture and real frames must use separate runs")
            aids = dict(saved.get("semantic_aids", {}))
            for frame_id, relative in (semantic_aids or {}).items():
                if frame_id not in {frame["frame_id"] for frame in frames}:
                    raise ValueError(f"Semantic aid joins unknown frame: {frame_id}")
                data_path(data_root, relative)
                if frame_id in aids and aids[frame_id] != relative:
                    raise ValueError(f"Existing semantic aid differs: {frame_id}")
                aids[frame_id] = relative
            saved["semantic_aids"] = aids
            hashes = dict(saved.get("image_sha256", {}))
            for frame in frames:
                digest = hash_file(data_path(data_root, frame["image_path"]))
                previous = hashes.get(frame["frame_id"])
                if previous is not None and previous != digest:
                    raise ValueError(f"Prepared RGB changed: {frame['frame_id']}")
                hashes[frame["frame_id"]] = digest
            saved["image_sha256"] = hashes
            saved["schema_version"] = 1
            saved["split_limit"] = "RELLIS held-out recordings on one campus; no independent geographic-site claim"
            annotations = read_jsonl(run_dir / "annotations.jsonl", missing_ok=True)
            validate_records(frames, regions, annotations, data_root=data_root)
            indexed = {row["region_id"]: row for row in annotations}
            for region in regions:
                indexed.setdefault(region["region_id"], pending_annotation(region["region_id"], saved["robot_profile_id"]))
            atomic_jsonl(run_dir / "frames.jsonl", frames)
            atomic_jsonl(run_dir / "regions.jsonl", regions)
            atomic_json(run_dir / "metadata/data.json", saved)
            atomic_jsonl(run_dir / "annotations.jsonl", [indexed[key] for key in sorted(indexed)])
    return run_dir


def add_regions(run_dir, regions, data_root, provenance=None):
    """Append validated masks; existing frozen frame inventories are immutable."""
    run_dir = Path(run_dir)
    incoming = list(regions)
    with exclusive_lock(run_dir / ".data.lock"):
        frames = read_jsonl(run_dir / "frames.jsonl")
        old = read_jsonl(run_dir / "regions.jsonl", missing_ok=True)
        known = {row["region_id"] for row in old}
        frozen = read_json(run_dir / "metadata/selection.json", default={}).get("frames", {})
        for row in incoming:
            if row["region_id"] not in known and row["frame_id"] in frozen:
                raise ValueError(f"Cannot add masks to frozen frame: {row['frame_id']}")
        merged = merge_records(old, incoming, "region_id")
        validate_records(frames, merged, data_root=data_root)
        metadata = read_json(run_dir / "metadata/data.json", default={})
        # A crash after regions publication is resumable: preparation fills only
        # absent pending rows and leaves every completed annotation untouched.
        with exclusive_lock(run_dir / ".annotations.lock"):
            annotations = read_jsonl(run_dir / "annotations.jsonl", missing_ok=True)
            validate_records(frames, merged, annotations, data_root=data_root)
            indexed = {row["region_id"]: row for row in annotations}
            for region in merged:
                indexed.setdefault(region["region_id"], pending_annotation(region["region_id"], metadata.get("robot_profile_id", DEFAULT_POLICY)))
            atomic_jsonl(run_dir / "regions.jsonl", merged)
            atomic_jsonl(run_dir / "annotations.jsonl", [indexed[key] for key in sorted(indexed)])
        if provenance is not None:
            existing = read_json(run_dir / "metadata/masks.json", default={})
            for key, value in provenance.items():
                if key in existing and existing[key] != value:
                    raise ValueError(f"Mask provenance changed: {key}")
                existing[key] = value
            atomic_json(run_dir / "metadata/masks.json", existing)
    return merged


def coverage_summary(run_dir):
    """Report actual experiment coverage and excluded/pending inputs without padding."""
    run_dir = Path(run_dir)
    frames = read_jsonl(run_dir / "frames.jsonl")
    regions = read_jsonl(run_dir / "regions.jsonl")
    annotations = read_jsonl(run_dir / "annotations.jsonl")
    validate_records(frames, regions, annotations)
    metadata = read_json(run_dir / "metadata/data.json", default={})
    fixture = metadata.get("is_fixture", False)
    expected = metadata.get("expected_frames", {"development": 20, "test": 80})
    recipe = metadata.get("recipe")
    planned_sources = ({"rellis", "phone"} if recipe == "legacy_phone" else
                       {"phone"} if recipe == "phone_extension" else {"rellis"})
    actual = Counter(frame["split"] for frame in frames if not fixture and frame["source"] in planned_sources)
    selected = {row["region_id"] for row in regions if row["selected_for_classification"]}
    eligible = [row for row in annotations if row["region_id"] in selected
                and row["annotation_status"] == "complete" and row["mask_quality"] == "valid"]
    return {"is_fixture": fixture, "expected_frames_by_split": expected,
            "real_frames_by_split": dict(actual),
            "missing_frames_by_split": {key: max(0, count - actual[key]) for key, count in expected.items()},
            "stored_frames": len(frames), "planning_relevant_regions": sum(row["planning_relevant"] for row in regions),
            "stored_frames_by_source": dict(Counter(frame["source"] for frame in frames)),
            "selected_regions": len(selected), "completed_valid_selected_references": len(eligible),
            "pending_selected_references": sum(row["region_id"] in selected and row["annotation_status"] == "pending" for row in annotations),
            "frames_without_masks": len({row["frame_id"] for row in frames} - {row["frame_id"] for row in regions}),
            "phone_clips": "present_unverified" if any(row["source"] == "phone" for row in frames) else "pending_optional",
            "reference_scope": "semantic_material_policy; physical feasibility and phone cup/cable accuracy unestablished",
            "complete_planned_frame_coverage": not fixture and bool(expected) and all(actual[key] == count for key, count in expected.items())}


def export_run(run_dir, data_root, destination):
    """Validate and copy an immutable artifact snapshot; RGB/masks stay under data_root."""
    from .masks import freeze_selection
    run_dir, destination = Path(run_dir), Path(destination)
    if run_dir.resolve() == destination.resolve():
        raise ValueError("Export destination must be a separate directory")
    selection = read_json(run_dir / "metadata/selection.json", default={})
    if not selection:
        raise ValueError("Freeze region selection before export")
    freeze_selection(run_dir, data_root, seed=selection["seed"])
    with exclusive_lock(run_dir / ".data.lock"), exclusive_lock(run_dir / ".annotations.lock"):
        frames, regions, annotations = [read_jsonl(run_dir / name) for name in
                                        ("frames.jsonl", "regions.jsonl", "annotations.jsonl")]
        validate_records(frames, regions, annotations, data_root=data_root)
        transfer = read_jsonl(run_dir / "metadata/reference_transfer.jsonl", missing_ok=True)
        transfer = merge_records([], transfer, "region_id")
        region_index = {row["region_id"]: row for row in regions}
        annotation_index = {row["region_id"]: row for row in annotations}
        for row in transfer:
            identifier = row["region_id"]
            if identifier not in region_index:
                raise ValueError(f"Reference transfer joins unknown region: {identifier}")
            annotation = annotation_index.get(identifier)
            if annotation is not None and (
                ("reference_label" in row and row["reference_label"] != annotation["reference_label"])
                or ("reference_annotation_source" in row and row["reference_annotation_source"] != annotation.get("annotation_source", "manual"))
            ):
                raise ValueError("Reference transfer is stale; retry the identical annotation save or derive references before export")
        for name in ("frames.jsonl", "regions.jsonl", "annotations.jsonl"):
            publish_file(run_dir / name, destination / name)
        for source in sorted((run_dir / "metadata").glob("*")):
            if source.is_file() and source.suffix in {".json", ".jsonl"}:
                publish_file(source, destination / "metadata" / source.name)
        atomic_json(destination / "coverage.json", coverage_summary(run_dir))
    return destination
