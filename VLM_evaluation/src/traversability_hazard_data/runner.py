"""Prepare frozen image/reference pairs without implicit network actions."""

import hashlib
import importlib.metadata
from pathlib import Path
import shutil
import tempfile

from PIL import Image
import numpy as np

from .configuration import TASK_ID, SEQUENCES, resolve_config
from .rellis import derive_rellis_reference, discover_sequence, select_sequence
from .storage import (atomic_json, atomic_jsonl, data_path, digest, exclusive_lock,
                      fingerprint, hash_file, input_signature, publish_file, read_json, read_jsonl)


def _check_run(run):
    if not run.exists():
        return None
    metadata = data_path(run, "metadata/dataset.json", must_exist=False)
    legacy = ("regions.jsonl", "annotations.jsonl", "metadata/data.json")
    if any((run / name).exists() for name in legacy):
        raise ValueError("Refuse legacy region-classification run; choose a new hazard run")
    if metadata.exists():
        saved = read_json(metadata)
        if saved.get("task_id") != TASK_ID or saved.get("schema_version") != 1:
            raise ValueError("Run task/schema mismatch")
        return saved
    if any(path.name != ".hazard-data.lock" for path in run.iterdir()):
        raise ValueError("Refuse unidentified nonempty run; choose a new hazard run")
    return None


def _downstream(run):
    def populated(path):
        if path.is_symlink() or path.is_file():
            return True
        return path.is_dir() and any(populated(child) for child in path.iterdir())
    if any(populated(run / name) for name in ("predictions", "evaluation", "segmentation", "benchmark")):
        return True
    directory = run / "metadata"
    return directory.is_dir() and any(path.name != "dataset.json" and
                                     (path.is_symlink() or path.suffix == ".json")
                                     for path in directory.iterdir())


def coverage_for(frames, references, selection, pending, policy, fixture, fixture_profile="small"):
    small = fixture and fixture_profile == "small"
    expected = {"rellis": {"development": 1 if small else 20, "test": 4 if small else 80},
                "coco": {"development": 3 if small else 8, "test": 3 if small else 32}}
    actual = {source: {split: 0 for split in ("development", "test")} for source in expected}
    concepts = {source: {split: {} for split in ("development", "test")} for source in expected}
    by_id = {row["frame_id"]: row for row in references}
    eligible = {source: {split: 0 for split in ("development", "test")} for source in expected}
    tiny = {source: {split: {} for split in ("development", "test")} for source in expected}
    coco_polarity = {split: {"positive": 0, "negative": 0} for split in expected["coco"]}
    for row in frames:
        source, split = row["source"], row["split"]
        actual[source][split] += 1
        ref = by_id[row["frame_id"]]
        if source == "coco":
            coco_polarity[split]["positive" if ref["present_concepts"] else "negative"] += 1
        eligible[source][split] += int(ref["absence_scoring_eligible"])
        for c in ref["present_concepts"]:
            concepts[source][split][c] = concepts[source][split].get(c, 0) + 1
            if ref["concept_pixel_counts"][c] / (row["width"] * row["height"]) < policy["tiny_concept_area_fraction"]:
                tiny[source][split][c] = tiny[source][split].get(c, 0) + 1
    missing = {source: {split: expected[source][split] - actual[source][split] for split in expected[source]}
               for source in expected}
    strata = list((selection.get("coco") or {}).get("missing_strata", []))
    for source in expected:
        scored = [c for c in policy["canonical_prompts"] if c in policy["datasets"]["rellis"]["hazard_label_ids"]] if source == "rellis" else policy["datasets"]["coco"]["scored_category_names"]
        for split in expected[source]:
            for c in scored:
                if not concepts[source][split].get(c):
                    strata.append({"source": source, "split": split, "concept": c,
                                   "reason": "no_materialized_positive_reference"})
    return {"expected_frames_by_source_split": expected, "frames_by_source_split": actual,
            "missing_frames_by_source_split": missing, "concept_positive_frames": concepts,
            "coco_frames_by_split_polarity": coco_polarity,
            "tiny_concept_frames": tiny, "absence_eligible_frames": eligible,
            "pending_inputs": pending, "missing_strata": strata,
            "pending_optional": policy["pending_coverage"],
            "complete_planned_frame_coverage": not fixture and not pending and all(v == 0 for counts in missing.values() for v in counts.values())}


def prepare_run(config, *, fixture=False):
    """Prepare immutable pairs; fill missing inputs only before downstream work."""
    if type(fixture) is not bool:
        raise ValueError("fixture must be explicitly boolean")
    cfg = resolve_config(config)
    profile = cfg.get("fixture_profile", "small")
    if not fixture and profile != "small":
        raise ValueError("fixture_profile=protocol is only valid with fixture=True")
    small = fixture and profile == "small"
    root = cfg["data_root"]
    run = data_path(cfg["run_root"], cfg["run_name"], must_exist=False)
    previous = _check_run(run)
    if previous is not None:
        if previous.get("fixture") is not fixture:
            raise ValueError("Fixture and real runs cannot share identity")
        if previous.get("fixture_profile", "small") != profile:
            raise ValueError("Synthetic fixture profile changed; choose a new run")
        from .validation import validate_run
        validate_run(run, root)
    policy_bytes = cfg["policy_path"].read_bytes()
    policy = read_json(cfg["policy_path"])
    policy_hash = hashlib.sha256(policy_bytes).hexdigest()
    if policy.get("task_id") != TASK_ID or policy.get("schema_version") != 1:
        raise ValueError("Policy task/schema mismatch")
    if previous and (previous["policy_sha256"] != policy_hash or previous["run_name"] != cfg["run_name"]):
        raise ValueError("Frozen policy/run identity changed; use a new run")
    if fixture:
        from .fixture import create_fixture_sources
        source, annotation, image_dir = create_fixture_sources(root, policy, profile=profile)
        cfg["rellis"]["source_dir"] = source
        cfg["coco"].update(annotation_path=annotation, image_dir=image_dir)
    prefix = "fixtures/hazard_prompt_v1/imports/" if fixture else ""
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(data_path(run, ".hazard-data.lock", must_exist=False)):
        locked = _check_run(run)
        if locked != previous:
            raise RuntimeError("Run changed while acquiring preparation lock; retry")
        frames = read_jsonl(run / "frames.jsonl") if previous else []
        refs = read_jsonl(run / "references.jsonl") if previous else []
        metadata = dict(previous or {})
        if not previous:
            metadata["software"] = {name: importlib.metadata.version(name) for name in ("numpy", "Pillow", "pycocotools")}
        selection = dict(metadata.get("selection", {}))
        selection["rellis"] = dict(selection.get("rellis", {}))
        annotations = dict(metadata.get("annotations", {}))
        details = dict(metadata.get("reference_details", {}))
        assets = dict(metadata.get("assets", {}))
        source_images = dict(metadata.get("source_images", {}))
        existing = {row["frame_id"] for row in frames}
        pending, staged = [], []
        with tempfile.TemporaryDirectory(prefix=".hazard-stage-", dir=root) as temp:
            temp = Path(temp)
            def copy(source, relative):
                source = Path(source)
                data_path(root, relative, must_exist=False)
                digest_value = hash_file(source)
                if relative in assets and assets[relative] != digest_value:
                    raise ValueError(f"Completed input changed: {relative}")
                assets[relative] = digest_value
                staged.append((source, relative))
            def pair(frame, masks, valid, diagnostic, annotation_info):
                frame_id = frame["frame_id"]
                with Image.open(annotation_info.pop("_rgb")) as rgb:
                    rgb.load()
                    if rgb.mode not in {"RGB", "RGBA"}:
                        raise ValueError("Expected original RGB image")
                    width, height = rgb.size
                if valid.shape != (height, width):
                    raise ValueError("Annotation dimensions must align with original RGB")
                frame.update(width=width, height=height, image_sha256=assets[frame["image_path"]])
                scored = [c for c in policy["canonical_prompts"] if c in policy["datasets"]["rellis"]["hazard_label_ids"]] if frame["source"] == "rellis" else list(policy["datasets"]["coco"]["scored_category_names"])
                present = [c for c in scored if c in masks]
                base = f"hazard_references/{cfg['run_name']}/{hashlib.sha256(frame_id.encode('utf-8')).hexdigest()}"
                paths = {}
                for concept, mask in [(c, masks[c]) for c in present] + [("valid", valid)]:
                    relative = f"{base}/{concept}.png"
                    staged_path = temp / f"{len(staged):06d}.png"
                    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(staged_path, format="PNG")
                    copy(staged_path, relative)
                    paths[concept] = relative
                eligible = (frame["source"] == "coco" or diagnostic["ignored_fraction"] <=
                            policy["datasets"]["rellis"]["absence_scoring_max_ignored_fraction"])
                frames.append(frame)
                refs.append({"frame_id": frame_id, "scored_concepts": scored,
                             "present_concepts": present, "absence_scoring_eligible": eligible,
                             "concept_masks": {c: paths[c] for c in present},
                             "concept_pixel_counts": {c: int(masks[c].sum()) for c in present},
                             "valid_mask_path": paths["valid"], "annotation_source": "dataset_policy",
                             "reference_scope": "annotated_hazard_concepts", "status": "complete"})
                annotations[frame_id] = annotation_info
                details[frame_id] = diagnostic
            for sequence in SEQUENCES:
                chosen = selection["rellis"].get(sequence)
                if chosen is None:
                    chosen, missing = select_sequence(root, cfg["rellis"]["source_dir"], sequence, 1 if small else 20)
                    if missing:
                        pending.append(missing)
                        continue
                    if prefix:
                        chosen = dict(chosen, selected=[dict(row, image_path=prefix + row["image_path"], annotation_path=prefix + row["annotation_path"]) for row in chosen["selected"]])
                    selection["rellis"][sequence] = chosen
                images, labels, _ = discover_sequence(root, cfg["rellis"]["source_dir"], sequence)
                for row in chosen["selected"]:
                    # Check selected original bytes if originals are still available.
                    for mapping, key in ((images, "image"), (labels, "annotation")):
                        original = mapping.get(row["stem"])
                        if original and hash_file(original) != row[key + "_sha256"]:
                            raise ValueError(f"Frozen RELLIS {key} changed: {row['frame_id']}")
                    if row["frame_id"] in existing:
                        continue
                    image = images.get(row["stem"])
                    label = labels.get(row["stem"])
                    if image is None or label is None:
                        raise FileNotFoundError(f"Frozen RELLIS pair missing: {row['frame_id']}")
                    masks, valid, diagnostic = derive_rellis_reference(label, policy)
                    copy(image, row["image_path"])
                    copy(label, row["annotation_path"])
                    pair({"frame_id": row["frame_id"], "source": "rellis", "scene_id": "rellis_campus",
                          "sequence_id": sequence, "timestamp_s": None, "image_path": row["image_path"],
                          "split": "development" if sequence == "00000" else "test"}, masks, valid, diagnostic,
                         {"kind": "rellis_original_semantic_ids", "path": row["annotation_path"],
                          "sha256": row["annotation_sha256"], "provenance": "RELLIS-3D original camera semantic-ID annotations", "_rgb": image})
            from .coco import CocoDataset
            coco_selection = selection.get("coco")
            annotation_path = cfg["coco"]["annotation_path"]
            if coco_selection:
                if annotation_path.is_file() and hash_file(annotation_path) != coco_selection["annotation_sha256"]:
                    raise ValueError("Frozen COCO annotation bytes changed; use a new run")
                annotation_path = data_path(root, coco_selection["annotation_path"])
            if annotation_path.is_file():
                dataset = CocoDataset(annotation_path, policy)
                if coco_selection is None:
                    coco_selection = dataset.select(fixture=small)
                    stored_annotation = f"{prefix}coco/annotations/instances_val2017_{dataset.annotation_sha256}.json"
                    copy(annotation_path, stored_annotation)
                    try:
                        image_relative = cfg["coco"]["image_dir"].relative_to(root).as_posix()
                    except ValueError:
                        image_relative = None
                    coco_selection.update(annotation_path=stored_annotation,
                                          annotation_sha256=dataset.annotation_sha256, image_dir=image_relative)
                    selection["coco"] = coco_selection
                    source_images["coco"] = {str(row["image_id"]): dataset.images[row["image_id"]] for row in coco_selection["selected"]}
                for row in coco_selection["selected"]:
                    image_id = row["image_id"]
                    frame_id = f"coco:val2017:{image_id:012d}"
                    image_info = dataset.images[image_id]
                    image = data_path(cfg["coco"]["image_dir"], image_info["file_name"], must_exist=False)
                    if frame_id in existing:
                        if image.is_file() and hash_file(image) != next(f["image_sha256"] for f in frames if f["frame_id"] == frame_id):
                            raise ValueError(f"Frozen COCO RGB changed: {image_id}")
                        continue
                    if not image.is_file():
                        pending.append({"source": "coco", "image_id": image_id, "split": row["split"],
                                        "reason": "selected_rgb_missing", "coco_url": image_info.get("coco_url"),
                                        "action": "Supply this frozen selected image or explicitly opt into the selected-image COCO download helper."})
                        continue
                    masks, valid, diagnostic = dataset.reference(image_id)
                    relative = f"{prefix}coco/val2017/{image_info['file_name']}"
                    copy(image, relative)
                    pair({"frame_id": frame_id, "source": "coco", "scene_id": f"coco:{image_id:012d}",
                          "sequence_id": "val2017", "timestamp_s": None, "image_path": relative, "split": row["split"]},
                         masks, valid, diagnostic, {"kind": "coco_instances_val2017", "path": coco_selection["annotation_path"],
                                                   "sha256": coco_selection["annotation_sha256"], "image_id": image_id,
                                                   "provenance": "COCO val2017 official instance segmentation, including crowds", "_rgb": image})
            else:
                pending.append({"source": "coco", "reason": "instances_val2017_annotations_missing",
                                "action": "Supply coco.annotation_path or explicitly opt into the official annotation archive helper; selection remains pending."})
            frames.sort(key=lambda row: row["frame_id"])
            refs.sort(key=lambda row: row["frame_id"])
            metadata.update(task_id=TASK_ID, schema_version=1, run_name=cfg["run_name"], fixture=fixture,
                            policy_sha256=policy_hash, alias_sha256=digest(policy["aliases"]), policy=policy,
                            policy_source_text=policy_bytes.decode("utf-8"),
                            sampling={"seed": 0, "rellis_frames_per_sequence": 1 if small else 20,
                                      "coco_small_instance_pixels": 1024}, selection=selection,
                            annotations=annotations, assets=assets, reference_details=details, source_images=source_images)
            if profile != "small":
                metadata["fixture_profile"] = profile
            # Preserve previously published identities; only new runs acquire pins.
            if previous is None or "observed_input_signature" in previous or len(frames) != len(existing):
                metadata["observed_input_signature"] = input_signature(frames, refs, assets)
                metadata["selected_frame_ids"] = [row["frame_id"] for row in frames]
            metadata["coverage"] = coverage_for(frames, refs, selection, pending, policy, fixture, profile)
            metadata["dataset_fingerprint"] = fingerprint(metadata, frames, refs)
            if previous == metadata:
                return run
            if previous is not None and _downstream(run):
                raise ValueError("Downstream artifacts freeze the dataset; additions require a new run")
            # Validate the complete candidate using a staging root before publication.
            overlay = temp / "data"
            overlay.mkdir()
            for relative in assets:
                source = next((s for s, r in reversed(staged) if r == relative), None)
                source = source or data_path(root, relative)
                destination = overlay.joinpath(*relative.split("/"))
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
            candidate = temp / "run"
            atomic_jsonl(candidate / "frames.jsonl", frames)
            atomic_jsonl(candidate / "references.jsonl", refs)
            atomic_json(candidate / "metadata/dataset.json", metadata)
            from .validation import validate_run
            validate_run(candidate, overlay)
            for _, relative in staged:
                destination = data_path(root, relative, must_exist=False)
                if destination.exists() and hash_file(destination) != assets[relative]:
                    raise ValueError(f"Completed destination differs; use a new run: {relative}")
            for _, relative in staged:
                data_path(root, relative + ".lock", must_exist=False)
                publish_file(data_path(overlay, relative), data_path(root, relative, must_exist=False))
            atomic_jsonl(run / "frames.jsonl", frames)
            atomic_jsonl(run / "references.jsonl", refs)
            atomic_json(run / "metadata/dataset.json", metadata)
    return run
