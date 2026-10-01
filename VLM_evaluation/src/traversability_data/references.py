"""Transfer original RELLIS semantic IDs to SAM masks for policy references."""

from collections import Counter
from pathlib import Path

from .storage import (atomic_json, atomic_jsonl, data_path, exclusive_lock,
                      hash_file, read_json, read_jsonl)
from .validation import validate_records


POLICY_ID = "rellis_material_v1"
CLASS_NAMES = {
    0: "void", 1: "dirt", 3: "grass", 4: "tree", 5: "pole", 6: "water",
    7: "sky", 8: "vehicle", 9: "object", 10: "asphalt", 12: "building",
    15: "log", 17: "person", 18: "fence", 19: "bush", 23: "concrete",
    27: "barrier", 31: "puddle", 33: "mud", 34: "rubble",
}
PERMITTED_IDS = frozenset((1, 10, 23))
AVOIDED_IDS = frozenset((4, 5, 6, 7, 8, 9, 12, 15, 17, 18, 19, 27, 31, 33, 34))
OFFICIAL_IDS_URL = "https://raw.githubusercontent.com/unmannedlab/RELLIS-3D/main/benchmarks/SalsaNext/train/tasks/semantic/config/labels/rellis.yaml"


def _pending(region_id, reason, quality="unchecked", hazard_type=None):
    return {
        "region_id": region_id, "reference_label": None, "semantic_class": None,
        "hazard_type": hazard_type, "mask_quality": quality,
        "annotation_status": "pending", "robot_profile_id": POLICY_ID,
        "annotation_source": "dataset_policy",
        "reference_scope": "semantic_material_policy",
        "reference_exclusion_reason": reason,
    }


def derive_references(run_dir, data_root, semantic_aids=None,
                      coverage_threshold=.95, purity_threshold=.95):
    """Derive explicit policy references; missing, mixed or unresolved aids stay pending."""
    import numpy as np
    from PIL import Image, __version__ as pillow_version

    for name, value in (("coverage_threshold", coverage_threshold),
                        ("purity_threshold", purity_threshold)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not .95 <= value <= 1:
            raise ValueError(f"{name} must be at least 0.95 and at most one for the public policy")
    run_dir = Path(run_dir)
    with exclusive_lock(run_dir / ".annotations.lock"):
        frames = read_jsonl(run_dir / "frames.jsonl")
        regions = read_jsonl(run_dir / "regions.jsonl")
        annotations = read_jsonl(run_dir / "annotations.jsonl", missing_ok=True)
        validate_records(frames, regions, annotations, data_root=data_root)
        metadata_path = run_dir / "metadata" / "data.json"
        metadata = read_json(metadata_path, default={})
        if metadata.get("robot_profile_id", POLICY_ID) != POLICY_ID:
            raise ValueError(f"Automatic references require frozen policy {POLICY_ID}")
        frozen_rules = metadata.get("automatic_references", {})
        if frozen_rules and any(frozen_rules.get(key) != value for key, value in
                                (("coverage_threshold", coverage_threshold),
                                 ("purity_threshold", purity_threshold))):
            raise ValueError("Frozen reference thresholds differ; use a new run")
        aids = metadata.get("semantic_aids", {}) if semantic_aids is None else semantic_aids
        if not isinstance(aids, dict):
            raise ValueError("semantic_aids must map frame IDs to data-root-relative paths")
        indexed_frames = {row["frame_id"]: row for row in frames}
        unexpected_frames = set(aids) - set(indexed_frames)
        if unexpected_frames:
            raise ValueError(f"Semantic aids reference unknown frame IDs: {sorted(unexpected_frames)}")
        aid_cache = {}
        # Validate every supplied ground-truth image, including pixels outside SAM masks.
        for frame_id, relative in aids.items():
            path = data_path(data_root, relative)
            with Image.open(path) as image:
                if len(image.getbands()) != 1:
                    raise ValueError(f"Semantic aid must contain original single-channel IDs: {relative}")
                aid = np.asarray(image).copy()
                aid_size = image.size
            if aid.ndim != 2 or not (np.issubdtype(aid.dtype, np.integer) or aid.dtype == np.bool_):
                raise ValueError(f"Semantic aid must contain integer original IDs: {relative}")
            with Image.open(data_path(data_root, indexed_frames[frame_id]["image_path"])) as rgb:
                if rgb.size != aid_size:
                    raise ValueError(f"Semantic aid/RGB dimensions differ for {frame_id}")
            unexpected = set(int(x) for x in np.unique(aid)) - set(CLASS_NAMES)
            if unexpected:
                raise ValueError(f"Unexpected original RELLIS IDs in {relative}: {sorted(unexpected)}")
            aid_cache[frame_id] = (aid, relative, hash_file(path))

        old_annotations = {row["region_id"]: row for row in annotations}
        transfer_path = run_dir / "metadata" / "reference_transfer.jsonl"
        old_transfer_rows = read_jsonl(transfer_path, missing_ok=True)
        old_transfers = {}
        region_ids = {row["region_id"] for row in regions}
        for row in old_transfer_rows:
            region_id = row["region_id"]
            if region_id in old_transfers or region_id not in region_ids:
                raise ValueError(f"Duplicate or unknown reference-transfer region ID: {region_id}")
            old_transfers[region_id] = row
        merged, transfers = [], []
        for region in regions:
            region_id, frame_id = region["region_id"], region["frame_id"]
            path = data_path(data_root, region["mask_path"])
            with Image.open(path) as image:
                mask = np.asarray(image) != 0
            mask_pixels = int(np.count_nonzero(mask))
            if mask_pixels == 0:
                raise ValueError(f"Empty SAM mask: {region_id}")
            transfer = {
                "region_id": region_id, "frame_id": frame_id,
                "mask_pixels": mask_pixels, "mask_sha256": hash_file(path),
                "semantic_aid_path": None, "semantic_aid_sha256": None,
                "nonvoid_pixels": None, "nonvoid_coverage": None,
                "class_counts": {}, "class_fractions": {}, "avoided_pixels": None,
                "avoided_fraction": None, "contains_annotated_avoided_pixels": None,
                "dominant_class_id": None, "dominant_class_fraction": None,
                "coverage_threshold": coverage_threshold, "purity_threshold": purity_threshold,
                "policy_id": POLICY_ID, "is_fixture": bool(metadata.get("is_fixture", False)),
            }
            candidate = _pending(region_id, "missing_semantic_aid")
            if frame_id in aid_cache:
                aid, relative, aid_hash = aid_cache[frame_id]
                ids, counts = np.unique(aid[mask], return_counts=True)
                class_counts = {int(key): int(value) for key, value in zip(ids, counts)}
                nonvoid = mask_pixels - class_counts.get(0, 0)
                coverage = nonvoid / mask_pixels
                avoided = sum(class_counts.get(key, 0) for key in AVOIDED_IDS)
                fractions = {str(key): value / nonvoid for key, value in class_counts.items()
                             if key != 0 and nonvoid}
                nonvoid_counts = {key: value for key, value in class_counts.items() if key != 0}
                dominant = max(nonvoid_counts, key=lambda key: (nonvoid_counts[key], -key)) if nonvoid else None
                purity = nonvoid_counts[dominant] / nonvoid if nonvoid else 0.0
                transfer.update({
                    "semantic_aid_path": relative, "semantic_aid_sha256": aid_hash,
                    "nonvoid_pixels": nonvoid, "nonvoid_coverage": coverage,
                    "class_counts": {str(key): value for key, value in class_counts.items()},
                    "class_fractions": fractions, "avoided_pixels": avoided,
                    "avoided_fraction": avoided / mask_pixels,
                    "contains_annotated_avoided_pixels": avoided > 0,
                    "dominant_class_id": dominant, "dominant_class_fraction": purity,
                })
                if dominant in PERMITTED_IDS and avoided:
                    candidate = _pending(region_id, "annotated_hazard_overlap", "mixed",
                                         "annotated_avoided_pixel_overlap")
                elif coverage < coverage_threshold:
                    candidate = _pending(region_id, "insufficient_nonvoid_coverage")
                elif purity < purity_threshold:
                    candidate = _pending(region_id, "insufficient_class_purity", "mixed")
                elif dominant not in PERMITTED_IDS | AVOIDED_IDS:
                    candidate = _pending(region_id, "unresolved_class")
                else:
                    candidate = {
                        "region_id": region_id,
                        "reference_label": "traversable" if dominant in PERMITTED_IDS else "non_traversable",
                        "semantic_class": CLASS_NAMES[dominant],
                        "hazard_type": None if dominant in PERMITTED_IDS else CLASS_NAMES[dominant],
                        "mask_quality": "valid", "annotation_status": "complete",
                        "robot_profile_id": POLICY_ID, "annotation_source": "dataset_policy",
                        "reference_scope": "semantic_material_policy",
                        "reference_exclusion_reason": None,
                    }
            existing = old_annotations.get(region_id)
            if existing and existing["annotation_status"] == "complete":
                if existing.get("annotation_source") == "dataset_policy":
                    if any(existing.get(key) != value for key, value in candidate.items()):
                        raise ValueError(f"Completed automatic reference differs: {region_id}; use a new run")
                candidate = existing
            transfer["reference_label"] = candidate["reference_label"]
            transfer["reference_exclusion_reason"] = candidate.get("reference_exclusion_reason")
            transfer["reference_annotation_source"] = candidate.get("annotation_source", "manual")
            if (existing and existing["annotation_status"] == "complete"
                    and existing.get("annotation_source") == "dataset_policy"
                    and region_id in old_transfers and old_transfers[region_id] != transfer):
                raise ValueError(f"Frozen reference provenance differs: {region_id}; use a new run")
            merged.append(candidate)
            transfers.append(transfer)
        validate_records(frames, regions, merged, data_root=data_root)
        metadata["automatic_references"] = {
            "policy_id": POLICY_ID, "reference_scope": "semantic_material_policy",
            "coverage_threshold": coverage_threshold, "purity_threshold": purity_threshold,
            "original_ids_url": OFFICIAL_IDS_URL,
            "software_versions": {"numpy": np.__version__, "Pillow": pillow_version},
            "region_count": len(regions),
            "complete_count": sum(row["annotation_status"] == "complete" for row in merged),
            "pending_count": sum(row["annotation_status"] == "pending" for row in merged),
            "dataset_policy_complete_count": sum(row["annotation_status"] == "complete"
                and row.get("annotation_source") == "dataset_policy" for row in merged),
            "preserved_manual_complete_count": sum(row["annotation_status"] == "complete"
                and row.get("annotation_source") != "dataset_policy" for row in merged),
            "exclusion_counts": dict(Counter(row.get("reference_exclusion_reason") for row in merged
                if row["annotation_status"] == "pending")),
        }
        # Persist provenance first so completed references never lack their transfer record.
        atomic_jsonl(transfer_path, transfers)
        atomic_json(metadata_path, metadata)
        atomic_jsonl(run_dir / "annotations.jsonl", merged)
        return merged
