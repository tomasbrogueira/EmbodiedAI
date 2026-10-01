"""Validate published joins, frozen identities and annotation-derived rasters."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from .configuration import TASK_ID, SEQUENCES
from .rellis import derive_rellis_reference
from .storage import data_path, digest, fingerprint, hash_file, input_signature, read_json, read_jsonl, safe_name

FRAME_FIELDS = {"frame_id", "source", "scene_id", "sequence_id", "timestamp_s", "image_path",
                "split", "width", "height", "image_sha256"}
REFERENCE_FIELDS = {"frame_id", "scored_concepts", "present_concepts", "absence_scoring_eligible",
                    "concept_masks", "concept_pixel_counts", "valid_mask_path", "annotation_source",
                    "reference_scope", "status"}


def _index(rows):
    result = {}
    for row in rows:
        identifier = row.get("frame_id")
        if not isinstance(identifier, str) or not identifier or identifier in result:
            raise ValueError(f"Invalid/duplicate frame_id: {identifier}")
        result[identifier] = row
    return result


def _mask(root, relative, size):
    path = data_path(root, relative)
    with Image.open(path) as image:
        image.load()
        array = np.asarray(image).copy()
        if image.format != "PNG" or image.mode != "L" or image.size != size or not set(np.unique(array)).issubset({0, 255}):
            raise ValueError(f"Reference mask must be aligned single-channel binary PNG: {relative}")
    return array > 0


def validate_run(run_dir, data_root):
    """Return a validation summary or an actionable malformed-artifact error."""
    try:
        return _validate_run(run_dir, data_root)
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError) as error:
        raise ValueError(f"Malformed hazard dataset artifact: {error}") from error


def _validate_run(run_dir, data_root):
    """Return coverage on valid complete pairs, including honestly partial runs."""
    run, root = Path(run_dir).resolve(), Path(data_root).resolve()
    for name in ("frames.jsonl", "references.jsonl", "metadata/dataset.json"):
        data_path(run, name)
    if any((run / name).exists() for name in ("regions.jsonl", "annotations.jsonl", "metadata/data.json")):
        raise ValueError("Legacy artifacts cannot identify a hazard run")
    metadata = read_json(run / "metadata/dataset.json")
    required = {"task_id", "schema_version", "run_name", "fixture", "policy_sha256", "alias_sha256", "policy",
                "policy_source_text", "sampling", "selection", "annotations", "assets", "reference_details",
                "source_images", "coverage", "dataset_fingerprint"}
    if not required.issubset(metadata):
        raise ValueError(f"Dataset metadata missing: {sorted(required - metadata.keys())}")
    if metadata["task_id"] != TASK_ID or type(metadata["schema_version"]) is not int or metadata["schema_version"] != 1 or type(metadata["fixture"]) is not bool:
        raise ValueError("Dataset task/schema/fixture identity mismatch")
    safe_name(metadata["run_name"])
    policy = metadata["policy"]
    if (hashlib.sha256(metadata["policy_source_text"].encode("utf-8")).hexdigest() != metadata["policy_sha256"]
            or json.loads(metadata["policy_source_text"]) != policy
            or digest(policy["aliases"]) != metadata["alias_sha256"]
            or policy.get("task_id") != TASK_ID or policy.get("schema_version") != 1):
        raise ValueError("Policy/alias hashes or identity changed")
    fixture = metadata["fixture"]
    profile = metadata.get("fixture_profile", "small")
    if profile not in {"small", "protocol"} or (not fixture and profile != "small"):
        raise ValueError("Invalid synthetic fixture profile")
    small = fixture and profile == "small"
    expected_sampling = {"seed": 0, "rellis_frames_per_sequence": 1 if small else 20,
                         "coco_small_instance_pixels": 1024}
    if metadata["sampling"] != expected_sampling:
        raise ValueError("Frozen sampling mismatch")
    frames, references = read_jsonl(run / "frames.jsonl"), read_jsonl(run / "references.jsonl")
    frame_index, ref_index = _index(frames), _index(references)
    if frame_index.keys() != ref_index.keys():
        raise ValueError("Frame/reference joins must match exactly")
    if frames != sorted(frames, key=lambda r: r["frame_id"]) or references != sorted(references, key=lambda r: r["frame_id"]):
        raise ValueError("Published records must have stable frame-ID order")
    if set(metadata["annotations"]) != set(frame_index) or set(metadata["reference_details"]) != set(frame_index):
        raise ValueError("Annotation/detail joins must match completed frames")
    assets = metadata["assets"]
    for relative, expected in assets.items():
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected) or hash_file(data_path(root, relative)) != expected:
            raise ValueError(f"Frozen asset hash changed: {relative}")
    selection = metadata["selection"]
    if set(selection) - {"rellis", "coco"} or "rellis" not in selection:
        raise ValueError("Unknown/missing source selection")
    selected_rellis = {}
    for sequence, chosen in selection["rellis"].items():
        if sequence not in SEQUENCES or chosen["algorithm"] != "natural_uniform_endpoints_v1" or chosen["requested"] != expected_sampling["rellis_frames_per_sequence"]:
            raise ValueError("RELLIS sequence/sample identity mismatch")
        if len(chosen["selected"]) != chosen["requested"] or chosen["available_at_freeze"] < chosen["requested"]:
            raise ValueError("RELLIS sample count mismatch")
        from traversability_data.sources import _natural, uniform_indices
        candidates = chosen["candidate_stems"]
        if (len(candidates) != chosen["available_at_freeze"] or len(candidates) != len(set(candidates))
                or candidates != sorted(candidates, key=lambda stem: (_natural(stem), stem))
                or chosen["selected_indices"] != uniform_indices(len(candidates), chosen["requested"])
                or [row["stem"] for row in chosen["selected"]] != [candidates[i] for i in chosen["selected_indices"]]):
            raise ValueError("RELLIS uniform natural-sorted selection changed")
        for stem in candidates:
            safe_name(stem)
        for row in chosen["selected"]:
            if row["frame_id"] != f"rellis:{sequence}:{row['stem']}" or row["frame_id"] in selected_rellis:
                raise ValueError("RELLIS selected frame identity mismatch/duplicate")
            safe_name(row["stem"])
            selected_rellis[row["frame_id"]] = row
    coco = None
    selected_coco = {}
    coco_selection = selection.get("coco")
    used_assets = set()
    if coco_selection is not None:
        from .coco import CocoDataset
        annotation_path = data_path(root, coco_selection["annotation_path"])
        if hash_file(annotation_path) != coco_selection["annotation_sha256"]:
            raise ValueError("Frozen COCO annotation hash changed")
        used_assets.add(coco_selection["annotation_path"])
        if coco_selection["image_dir"] is not None:
            data_path(root, coco_selection["image_dir"], must_exist=False)
        coco = CocoDataset(annotation_path, policy)
        reconstructed = coco.select(fixture=small)
        original_selection = {k: v for k, v in coco_selection.items() if k not in {"annotation_path", "annotation_sha256", "image_dir"}}
        if reconstructed != original_selection:
            raise ValueError("COCO frozen selected IDs, category coverage or sampling changed")
        for row in coco_selection["selected"]:
            identifier = f"coco:val2017:{row['image_id']:012d}"
            if identifier in selected_coco:
                raise ValueError("COCO selected split overlap/duplicate")
            selected_coco[identifier] = row
        expected_images = {str(r["image_id"]): coco.images[r["image_id"]] for r in coco_selection["selected"]}
        if metadata["source_images"].get("coco") != expected_images:
            raise ValueError("COCO source image selection/provenance mismatch")
    elif metadata["source_images"]:
        raise ValueError("Source images without frozen COCO selection")
    seen_paths, seen_pixels, mask_paths = set(), {}, set()
    for identifier, frame in frame_index.items():
        if set(frame) != FRAME_FIELDS or frame["source"] not in {"rellis", "coco"} or frame["split"] not in {"development", "test"}:
            raise ValueError(f"Frame schema/source/split mismatch: {identifier}")
        if not all(isinstance(frame[k], str) and frame[k] for k in ("scene_id", "sequence_id")) or frame["timestamp_s"] is not None:
            raise ValueError("Dataset timestamps must remain null; scene/sequence IDs are required")
        if any(type(frame[k]) is not int or frame[k] < 1 for k in ("width", "height")):
            raise ValueError("Original dimensions must be positive integers")
        if frame["image_path"] in seen_paths or assets.get(frame["image_path"]) != frame["image_sha256"]:
            raise ValueError("Duplicate image path or original RGB hash mismatch")
        seen_paths.add(frame["image_path"])
        used_assets.add(frame["image_path"])
        with Image.open(data_path(root, frame["image_path"])) as rgb:
            rgb.load()
            if rgb.mode not in {"RGB", "RGBA"} or rgb.size != (frame["width"], frame["height"]):
                raise ValueError("Frame dimensions/mode do not match original RGB")
            pixel_hash = hashlib.sha256(str(rgb.size).encode("ascii") + rgb.convert("RGB").tobytes()).hexdigest()
            if seen_pixels.setdefault(pixel_hash, frame["split"]) != frame["split"]:
                raise ValueError("Identical RGB content leaks between development and test")
        annotation = metadata["annotations"][identifier]
        used_assets.add(annotation["path"])
        if assets.get(annotation["path"]) != annotation["sha256"]:
            raise ValueError("Annotation hash/provenance mismatch")
        if frame["source"] == "rellis":
            chosen = selected_rellis.get(identifier)
            sequence = frame["sequence_id"]
            if chosen is None or sequence not in SEQUENCES or frame["split"] != ("development" if sequence == "00000" else "test"):
                raise ValueError("RELLIS selection/split separation mismatch")
            if (frame["image_path"] != chosen["image_path"] or frame["image_sha256"] != chosen["image_sha256"]
                    or annotation["path"] != chosen["annotation_path"] or annotation["sha256"] != chosen["annotation_sha256"]
                    or annotation["kind"] != "rellis_original_semantic_ids" or frame["scene_id"] != "rellis_campus"):
                raise ValueError("RELLIS original import identity mismatch")
            masks, valid, diagnostic = derive_rellis_reference(data_path(root, annotation["path"]), policy)
            scored = [c for c in policy["canonical_prompts"] if c in policy["datasets"]["rellis"]["hazard_label_ids"]]
            eligible = diagnostic["ignored_fraction"] <= policy["datasets"]["rellis"]["absence_scoring_max_ignored_fraction"]
        else:
            chosen = selected_coco.get(identifier)
            if chosen is None or frame["split"] != chosen["split"] or frame["sequence_id"] != "val2017" or frame["scene_id"] != f"coco:{chosen['image_id']:012d}":
                raise ValueError("COCO selection/split separation mismatch")
            if (annotation["kind"] != "coco_instances_val2017" or annotation.get("image_id") != chosen["image_id"]
                    or annotation["path"] != coco_selection["annotation_path"]):
                raise ValueError("COCO original annotation identity mismatch")
            image = coco.images[chosen["image_id"]]
            prefix = "fixtures/hazard_prompt_v1/imports/" if fixture else ""
            if frame["image_path"] != f"{prefix}coco/val2017/{image['file_name']}":
                raise ValueError("COCO RGB path differs from frozen image identity")
            if (frame["width"], frame["height"]) != (image["width"], image["height"]):
                raise ValueError("COCO annotation dimensions differ from RGB")
            masks, valid, diagnostic = coco.reference(chosen["image_id"])
            scored = list(policy["datasets"]["coco"]["scored_category_names"])
            eligible = True
        ref = ref_index[identifier]
        present = [c for c in scored if c in masks]
        if (set(ref) != REFERENCE_FIELDS or ref["scored_concepts"] != scored or ref["present_concepts"] != present
                or type(ref["absence_scoring_eligible"]) is not bool or ref["absence_scoring_eligible"] != eligible
                or set(ref["concept_masks"]) != set(present) or set(ref["concept_pixel_counts"]) != set(present)
                or ref["status"] != "complete" or ref["annotation_source"] != "dataset_policy"
                or ref["reference_scope"] != "annotated_hazard_concepts"):
            raise ValueError(f"Reference schema/presence/coverage mismatch: {identifier}")
        if metadata["reference_details"][identifier] != diagnostic:
            raise ValueError("Tiny/area/ignored-coverage diagnostics changed")
        for c, expected in [(c, masks[c]) for c in present] + [("valid", valid)]:
            relative = ref["valid_mask_path"] if c == "valid" else ref["concept_masks"][c]
            if relative in mask_paths:
                raise ValueError("Duplicate reference asset path")
            mask_paths.add(relative)
            used_assets.add(relative)
            actual = _mask(root, relative, (frame["width"], frame["height"]))
            if not np.array_equal(actual, expected):
                raise ValueError(f"Mask disagrees with original annotations: {identifier}/{c}")
            if c != "valid" and (type(ref["concept_pixel_counts"][c]) is not int or ref["concept_pixel_counts"][c] != int(actual.sum()) or not actual.any()):
                raise ValueError("Concept mask count/presence mismatch")
    if set(selected_rellis) != {f["frame_id"] for f in frames if f["source"] == "rellis"}:
        raise ValueError("RELLIS selected/completed pair count mismatch")
    if used_assets != set(assets):
        raise ValueError("Asset inventory differs from referenced inputs/masks")
    # Pending rows must name precisely the missing selections/source strata.
    pending = metadata["coverage"]["pending_inputs"]
    expected_missing_coco = {r["image_id"] for key, r in selected_coco.items() if key not in frame_index}
    pending_coco = [r["image_id"] for r in pending if r.get("reason") == "selected_rgb_missing"]
    if len(pending_coco) != len(set(pending_coco)) or set(pending_coco) != expected_missing_coco:
        raise ValueError("COCO missing selected image coverage mismatch")
    pending_rellis = [r["sequence_id"] for r in pending if r.get("source") == "rellis"]
    if len(pending_rellis) != len(set(pending_rellis)) or set(pending_rellis) != set(SEQUENCES) - set(selection["rellis"]):
        raise ValueError("RELLIS pending sequence coverage mismatch")
    missing_annotations = [r for r in pending if r.get("reason") == "instances_val2017_annotations_missing"]
    if len(missing_annotations) != int(coco_selection is None):
        raise ValueError("COCO annotation pending coverage mismatch")
    if any(r.get("reason") not in {"selected_rgb_missing", "insufficient_aligned_rgb_ids", "instances_val2017_annotations_missing"} for r in pending):
        raise ValueError("Unknown pending source state")
    from .runner import coverage_for
    expected_coverage = coverage_for(frames, references, selection, pending, policy, fixture, profile)
    if metadata["coverage"] != expected_coverage:
        raise ValueError("Source/split/category/count coverage changed")
    if "observed_input_signature" in metadata and metadata["observed_input_signature"] != input_signature(frames, references, assets):
        raise ValueError("Frozen observed input signature changed")
    if "selected_frame_ids" in metadata and metadata["selected_frame_ids"] != sorted(frame_index):
        raise ValueError("Published selected frame IDs changed")
    actual_fingerprint = fingerprint(metadata, frames, references)
    if actual_fingerprint != metadata["dataset_fingerprint"]:
        raise ValueError("Dataset fingerprint changed")
    counts = {"frames": len(frames), "references": len(references), "assets": len(assets)}
    return {"valid": True, "task_id": TASK_ID, "schema_version": 1, "fixture": fixture,
            "frames": len(frames), "references": len(references), "counts": counts,
            "coverage": expected_coverage, "dataset_fingerprint": actual_fingerprint}
