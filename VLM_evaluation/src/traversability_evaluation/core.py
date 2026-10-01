"""Evaluate frozen artifacts without importing data, inference or benchmarking."""

from collections import Counter
from copy import deepcopy
import os
from pathlib import Path

from .artifacts import (ArtifactError, data_path, fingerprint, input_identity, inspect_images,
                        read_json, read_jsonl, validate_metadata,
                        validate_predictions, validate_records)
from .metrics import count_metrics, rate

SPLITS = ("development", "test")
CATEGORIES = ("recognition", "target_grounding", "policy", "segmentation", "unreviewed")
_RELLIS_POLICY = dict.fromkeys(("dirt", "asphalt", "concrete"), "traversable")
_RELLIS_POLICY.update(dict.fromkeys(("tree", "pole", "water", "sky", "vehicle", "object",
                                   "building", "log", "person", "fence", "bush", "barrier",
                                   "puddle", "mud", "rubble"), "non_traversable"))


def _name(value, kind):
    if (not isinstance(value, str) or not value.strip() or value in (".", "..")
            or any(char in value for char in "/\\:\x00")):
        raise ValueError(f"{kind} must be a single nonempty portable path component")
    return value


def _configuration(config):
    if not isinstance(config, dict):
        raise ValueError("Evaluation config must be a dictionary")
    config = deepcopy(config)
    for key in ("metadata_fields", "coverage", "split_group_fields", "expected_recording_splits",
                "class_policy_mapping", "expected_metadata"):
        if key in config and not isinstance(config[key], dict):
            raise ValueError(f"{key} must be a dictionary")
    for key in ("expected_frames", "expected_recordings", "phone_hazard_clips"):
        if key in config.get("coverage", {}) and not isinstance(config["coverage"][key], dict):
            raise ValueError(f"coverage.{key} must be a dictionary")
    if "avoided_pixel_count_field" in config and (not isinstance(config["avoided_pixel_count_field"], str)
                                                 or not config["avoided_pixel_count_field"]):
        raise ValueError("avoided_pixel_count_field must be a nonempty field name")
    if type(config.get("fixture", False)) is not bool:
        raise ValueError("fixture must be boolean")
    config.setdefault("fixture", False)
    config.setdefault("reference_policy_id", "rellis_material_v1")
    if not isinstance(config["reference_policy_id"], str) or not config["reference_policy_id"].strip():
        raise ValueError("reference_policy_id must be a nonempty string")
    config["data_root"] = str(Path(config.get("data_root", os.environ.get("TRAVERSABILITY_DATA_ROOT", "data"))).resolve())
    config["run_root"] = str(Path(config.get("run_root", os.environ.get("TRAVERSABILITY_RUN_ROOT", "runs"))).resolve())
    if "comparisons" in config:
        entries = config["comparisons"]
        if not isinstance(entries, list) or not entries or any(not isinstance(row, dict) for row in entries):
            raise ValueError("comparisons must be a nonempty list of configuration entries")
    else:
        keys = config.get("model_keys", [])
        if (not isinstance(keys, list) or not keys or any(not isinstance(key, str) for key in keys)
                or len(set(keys)) != len(keys)):
            raise ValueError("model_keys must be a nonempty list of unique model keys")
        entries = [{"run_name": config.get("run_name"), "model_key": key} for key in keys]
    seen = set()
    for entry in entries:
        _name(entry.get("run_name"), "run_name")
        _name(entry.get("model_key"), "model_key")
        if "expected_metadata" in entry and not isinstance(entry["expected_metadata"], dict):
            raise ValueError("comparison.expected_metadata must be a dictionary")
        pair = (entry["run_name"], entry["model_key"])
        if pair in seen:
            raise ValueError("Duplicate comparison run/model; use separate runs for configurations")
        seen.add(pair)
        entry.setdefault("configuration_label", f"{entry['run_name']}/{entry['model_key']}")
        if not isinstance(entry["configuration_label"], str) or not entry["configuration_label"]:
            raise ValueError("configuration_label must be nonempty text")
        run_dir = Path(config["run_root"]) / entry["run_name"]
        if not run_dir.resolve().is_relative_to(Path(config["run_root"])):
            raise ValueError("Run directory escapes configured run_root")
    config["comparisons"] = entries
    config.setdefault("run_name", entries[0]["run_name"])
    _name(config["run_name"], "run_name")
    return config


def _issue(report, level, code, message, **context):
    report["validation"][level].append(dict(code=code, message=message, **context))


def _fixture_record(row):
    return (row.get("fixture") is True or row.get("is_fixture") is True
            or str(row.get("source", "")).lower().startswith(("fixture", "synthetic", "fake")))


def _read_run(run_dir, config, report):
    root = config["data_root"]
    frames = read_jsonl(run_dir / "frames.jsonl", "frame_id")
    regions = read_jsonl(run_dir / "regions.jsonl", "region_id")
    annotation_path = run_dir / "annotations.jsonl"
    annotations = read_jsonl(annotation_path, "region_id") if annotation_path.exists() else {}
    if not annotation_path.exists():
        _issue(report, "warnings", "missing_annotation_file", "No annotations.jsonl; references are unavailable",
               run_name=run_dir.name)
    validate_records(frames, regions, annotations, root, config["reference_policy_id"], config)
    data_metadata_path = run_dir / "metadata" / "data.json"
    if data_metadata_path.exists():
        data_metadata = read_json(data_metadata_path)
        if not config["fixture"] and _fixture_record(data_metadata):
            raise ArtifactError("Data metadata identifies a fixture; set fixture=true")
        if data_metadata.get("robot_profile_id", config["reference_policy_id"]) != config["reference_policy_id"]:
            raise ArtifactError("Data metadata reference policy mismatch")
    if not config["fixture"] and any(_fixture_record(row) for rows in (frames, regions, annotations) for row in rows.values()):
        raise ArtifactError("Synthetic fixture artifacts require fixture=true and cannot count as real coverage")
    inputs = input_identity(frames, regions, root)
    images, masks = inspect_images(frames, regions, root)
    selected = {key: row for key, row in regions.items() if row["selected_for_classification"]}
    primary = {}
    for key, region in selected.items():
        annotation = annotations.get(key)
        if (annotation is not None and annotation["annotation_status"] == "complete"
                and annotation["mask_quality"] == "valid" and masks[key]["defect"] is None):
            primary[key] = region
    transfer_path = run_dir / "metadata" / "reference_transfer.jsonl"
    transfer = read_jsonl(transfer_path, "region_id") if transfer_path.exists() else {}
    field = config.get("avoided_pixel_count_field", "avoided_pixels")
    for key, row in transfer.items():
        if key not in regions:
            raise ArtifactError(f"Reference transfer refers to missing region: {key}")
        if field not in row or (row[field] is not None and (type(row[field]) is not int or row[field] < 0)):
            raise ArtifactError(f"Transfer {field} must be a nonnegative integer or null: {key}")
        for policy_field in ("robot_profile_id", "policy_id"):
            if row.get(policy_field, config["reference_policy_id"]) != config["reference_policy_id"]:
                raise ArtifactError(f"Reference transfer policy mismatch: {key}")
        if not config["fixture"] and _fixture_record(row):
            raise ArtifactError("Fixture reference-transfer records cannot count as real coverage")
        region = regions[key]
        if "frame_id" in row and row["frame_id"] != region["frame_id"]:
            raise ArtifactError(f"Reference transfer frame identity mismatch: {key}")
        if "mask_sha256" in row:
            if row["mask_sha256"] != inputs["contents"][region["mask_path"]].get("sha256"):
                raise ArtifactError(f"Reference transfer mask fingerprint mismatch: {key}")
        if "mask_pixels" in row:
            if type(row["mask_pixels"]) is not int or row["mask_pixels"] <= 0:
                raise ArtifactError(f"Reference transfer mask_pixels must be positive: {key}")
            if key in masks and masks[key]["foreground_pixels"] is not None and row["mask_pixels"] != masks[key]["foreground_pixels"]:
                raise ArtifactError(f"Reference transfer foreground pixel count mismatch: {key}")
            if row[field] is not None and row[field] > row["mask_pixels"]:
                raise ArtifactError(f"Reference transfer avoided pixel count exceeds mask: {key}")
        if key in annotations:
            if "reference_label" in row and row["reference_label"] != annotations[key]["reference_label"]:
                raise ArtifactError(f"Reference transfer label mismatch: {key}")
            if ("reference_annotation_source" in row and row["reference_annotation_source"]
                    != annotations[key].get("annotation_source", "manual")):
                raise ArtifactError(f"Reference transfer annotation provenance mismatch: {key}")
            if (annotations[key].get("annotation_source") == "dataset_policy"
                    and annotations[key]["reference_label"] == "traversable" and row[field]):
                raise ArtifactError(f"Automatic permitted reference contains annotated avoided pixels: {key}")
        if row.get("semantic_aid_path") is not None:
            import hashlib
            aid_path = data_path(root, row["semantic_aid_path"])
            try:
                with aid_path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
            except OSError as error:
                raise ArtifactError(f"Reference semantic aid unavailable: {key}: {error}") from error
            if row.get("semantic_aid_sha256") != digest:
                raise ArtifactError(f"Reference semantic aid fingerprint mismatch: {key}")
    frozen = {"inputs": inputs, "annotations": [annotations[key] for key in sorted(annotations)],
              "reference_transfer": [transfer[key] for key in sorted(transfer)],
              "reference_policy_id": config["reference_policy_id"]}
    return {"frames": frames, "regions": regions, "annotations": annotations, "selected": selected,
            "primary": primary, "transfer": transfer, "images": images, "masks": masks,
            "inputs": inputs, "dataset_fingerprint": fingerprint(frozen)}


def _coverage(run, config, report):
    frames, selected, annotations = run["frames"], run["selected"], run["annotations"]
    settings = config.get("coverage", {})
    expected = settings.get("expected_frames", {"rellis": {"development": 20, "test": 80}})
    for source, quotas in expected.items():
        if (not isinstance(quotas, dict) or any(split not in SPLITS or type(count) is not int or count < 0
                                               for split, count in quotas.items())):
            raise ValueError("coverage.expected_frames must map source to nonnegative development/test counts")
    sources = sorted({frame["source"] for frame in frames.values()} | set(expected))
    rows = []
    for source in sources:
        for split in SPLITS:
            available = [key for key, frame in frames.items() if frame["source"] == source and frame["split"] == split]
            readable = sum(run["images"][key]["defect"] is None for key in available)
            rows.append({"source": source, "split": split, "expected": expected.get(source, {}).get(split),
                         "available": len(available), "readable": readable,
                         "real_available": 0 if config["fixture"] else len(available),
                         "real_readable": 0 if config["fixture"] else readable})
    recording_quotas = settings.get("expected_recordings", {"rellis": {
        "00000": 20, "00001": 20, "00002": 20, "00003": 20, "00004": 20,
    }})
    recordings = []
    for source in sorted(set(recording_quotas) | {frame["source"] for frame in frames.values()}):
        quotas = recording_quotas.get(source, {})
        if not isinstance(quotas, dict) or any(type(value) is not int or value < 0 for value in quotas.values()):
            raise ValueError("coverage.expected_recordings must map source/recording to nonnegative frame counts")
        for recording in sorted(set(quotas) | {frame["sequence_id"] for frame in frames.values() if frame["source"] == source}):
            keys = [key for key, frame in frames.items() if frame["source"] == source and frame["sequence_id"] == recording]
            recordings.append({"source": source, "sequence_id": recording, "expected": quotas.get(recording),
                               "available": len(keys), "readable": sum(run["images"][key]["defect"] is None for key in keys),
                               "real_readable": 0 if config["fixture"] else sum(run["images"][key]["defect"] is None for key in keys),
                               "split": frames[keys[0]]["split"] if keys else ("development" if recording == "00000" else "test")})
    missing = sorted(set(selected) - set(annotations))
    counts = Counter()
    quality = Counter({key: 0 for key in ("valid", "mixed", "broken", "unchecked", "missing")})
    reasons = Counter()
    class_size = Counter()
    for key, region in selected.items():
        annotation = annotations.get(key)
        quality[annotation["mask_quality"] if annotation else "missing"] += 1
        if annotation is None:
            reasons["missing_annotation"] += 1
        else:
            counts[annotation["annotation_status"]] += 1
            counts["complete_unknown"] += annotation["annotation_status"] == "complete" and annotation["reference_label"] == "unknown"
            if annotation["annotation_status"] == "pending":
                reasons[annotation.get("reference_exclusion_reason") or "pending_annotation"] += 1
            if annotation["mask_quality"] != "valid":
                reasons[f"mask_quality_{annotation['mask_quality']}"] += 1
        if run["masks"][key]["defect"]:
            reasons["physical_mask_defect"] += 1
        frame = frames[region["frame_id"]]
        fraction = run["masks"][key]["area_fraction"]
        size = "unavailable" if fraction is None else "<1%" if fraction < .01 else "1%-10%" if fraction < .1 else ">=10%"
        semantic = annotation.get("semantic_class") if annotation else None
        class_size[(frame["source"], frame["split"], semantic, size, key in run["primary"])] += 1
    phone_config = settings.get("phone_hazard_clips", {})
    phone_frames = [frame for frame in frames.values() if frame["source"].lower().startswith("phone")]
    clips = {(frame["source"], frame["sequence_id"]) for frame in phone_frames}
    expected_clips = phone_config.get("expected", 6)
    if type(expected_clips) is not int or expected_clips < 0:
        raise ValueError("coverage.phone_hazard_clips.expected must be a nonnegative integer")
    real_clips = 0 if config["fixture"] else len(clips)
    full_frames = (all(row["real_readable"] >= row["expected"] for row in rows if row["expected"] is not None)
                   and all(row["real_readable"] >= row["expected"] for row in recordings if row["expected"] is not None))
    selected_frames = {region["frame_id"] for region in selected.values()}
    missing_selected_frames = sorted(key for key, frame in frames.items() if frame["source"] in expected and key not in selected_frames)
    assessed_exclusions = {"unresolved_class", "insufficient_nonvoid_coverage", "insufficient_class_purity", "annotated_hazard_overlap"}
    pending_unresolved = [annotation for key, annotation in annotations.items() if key in selected
                          and annotation["annotation_status"] == "pending"
                          and not (annotation.get("annotation_source") == "dataset_policy"
                                   and annotation.get("reference_exclusion_reason") in assessed_exclusions)]
    reference_complete = bool(run["primary"]) and not missing and not pending_unresolved and not missing_selected_frames
    if config["fixture"]:
        status = "fixture_only"
    elif full_frames and reference_complete:
        status = "public_material_policy_complete_hazard_coverage_provisional"
    else:
        status = "public_subset_provisional"
    coverage = {
        "status": status, "fixture": config["fixture"], "frames": rows, "recordings": recordings,
        "frames_without_selected_regions": missing_selected_frames,
        "annotations": {"expected": len(selected), "available": len(selected) - len(missing),
                        "missing": len(missing), "missing_ids": missing, "pending": counts["pending"],
                        "complete": counts["complete"], "complete_unknown": counts["complete_unknown"],
                        "primary_eligible": len(run["primary"]), "by_mask_quality": dict(quality),
                        "exclusion_reasons": dict(reasons), "exclusion_counts_may_overlap": True},
        "all_annotation_records": len(annotations), "selected_regions": len(selected),
        "all_regions": len(run["regions"]),
        "phone_hazard_clips": {"expected": expected_clips, "available_recordings": real_clips,
                               "not_available": max(expected_clips - real_clips, 0),
                               "status": "optional_pending" if real_clips < expected_clips else "present_unverified",
                               "small_hazard_coverage": "not_established"},
        "class_size": [{"source": key[0], "split": key[1], "semantic_class": key[2], "area_bin": key[3],
                        "primary_eligible": key[4], "regions": count} for key, count in sorted(class_size.items(), key=lambda item: str(item[0]))],
        "interpretation": ["RELLIS test recordings are held out within one campus, not independent geographic sites.",
                           "Region counts are observations; neighboring regions are not independent scenes.",
                           "Material-policy references do not establish physical traversability or cup/cable accuracy.",
                           "Phone recording availability does not establish checked hazard annotations."],
    }
    for key in missing:
        _issue(report, "warnings", "missing_annotation", "Selected region has no annotation record", region_id=key)
    for key, mask in run["masks"].items():
        if mask["defect"]:
            _issue(report, "warnings", "physical_mask_defect", mask["defect"], region_id=key)
    report["mask_breakdown"] = []
    for source in sorted({frame["source"] for frame in frames.values()}):
        for split in SPLITS:
            keys = [key for key, region in selected.items() if frames[region["frame_id"]]["source"] == source
                    and frames[region["frame_id"]]["split"] == split]
            for mask_quality in ("valid", "mixed", "broken", "unchecked", "missing"):
                matched = [key for key in keys if (annotations[key]["mask_quality"] if key in annotations else "missing") == mask_quality]
                report["mask_breakdown"].append({"source": source, "split": split, "mask_quality": mask_quality,
                                                 "regions": len(matched), "complete": sum(key in annotations and annotations[key]["annotation_status"] == "complete" for key in matched),
                                                 "physical_defects": sum(run["masks"][key]["defect"] is not None for key in matched)})
    return coverage


def _identity(entry, metadata_digest=None, configuration_digest=None):
    return {"run_name": entry["run_name"], "model_key": entry["model_key"],
            "configuration_label": entry["configuration_label"],
            "configuration_fingerprint": configuration_digest, "metadata_fingerprint": metadata_digest}


def _rows(run, predictions, identity, coverage_status, *, invalid=False):
    all_missing_primary = sorted(set(run["primary"]) - set(predictions)) if run else []
    sources = sorted({frame["source"] for frame in run["frames"].values()}) if run else []
    rows = []
    for source in ["all", *sources]:
        for split in SPLITS:
            def matching(key):
                frame = run["frames"][run["regions"][key]["frame_id"]]
                return frame["split"] == split and (source == "all" or frame["source"] == source)
            selected = [key for key in run["selected"] if matching(key)] if run else []
            primary = [key for key in run["primary"] if matching(key)] if run else []
            missing = sorted(set(selected) - set(predictions))
            missing_primary = sorted(set(primary) - set(predictions))
            blocked = invalid or bool(all_missing_primary)
            result = dict(identity, source=source, split=split, coverage_status=coverage_status,
                          status="invalid" if invalid else "incomplete" if blocked else "scored" if primary else "no_eligible_regions",
                          eligible_regions=len(primary), selected_regions=len(selected),
                          missing_predictions=len(missing), missing_prediction_ids=missing,
                          missing_primary_predictions=len(missing_primary),
                          prediction_failures=sum(key in predictions and predictions[key]["status"] == "error" for key in primary),
                          selected_prediction_failures=sum(key in predictions and predictions[key]["status"] == "error" for key in selected),
                          metrics=None, confusion_matrix=None)
            if not blocked:
                result.update(count_metrics([{"reference_label": run["annotations"][key]["reference_label"],
                                             "prediction": predictions[key]} for key in primary]))
            rows.append(result)
    return rows


def _hazard_rows(run, predictions, identity, config, *, invalid=False):
    field = config.get("avoided_pixel_count_field", "avoided_pixels")
    sources = sorted({frame["source"] for frame in run["frames"].values()})
    rows = []
    for source in ["all", *sources]:
        for split in SPLITS:
            selected = [key for key, region in run["selected"].items() if run["frames"][region["frame_id"]]["split"] == split
                        and (source == "all" or run["frames"][region["frame_id"]]["source"] == source)]
            missing_transfer = sorted(key for key in selected if key not in run["transfer"]
                                      or run["transfer"][key][field] is None or run["masks"][key]["defect"])
            hazards = [key for key in selected if key in run["transfer"] and (run["transfer"][key][field] or 0) > 0]
            missing = sorted(set(hazards) - set(predictions))
            status = "invalid" if invalid else "unavailable" if missing_transfer or not run["transfer"] else "incomplete" if missing else "scored"
            numerator = sum(key in predictions and predictions[key]["status"] == "ok"
                            and predictions[key]["label"] == "traversable" for key in hazards)
            rows.append(dict(identity, source=source, split=split, status=status,
                             metric=rate(numerator, len(hazards)) if status == "scored" else None,
                             hazard_regions=len(hazards), missing_prediction_ids=missing,
                             missing_transfer_ids=missing_transfer,
                             avoided_pixel_count_total=sum(run["transfer"][key][field] for key in hazards),
                             region_pixel_counts={key: run["transfer"][key][field] for key in hazards}))
    return rows


def _suggestion(annotation, prediction, mask, policy, config):
    if mask["defect"] or (annotation and annotation["mask_quality"] in ("mixed", "broken")):
        return "segmentation", mask["defect"] or f"mask_quality_{annotation['mask_quality']}"
    if not annotation or annotation["annotation_status"] != "complete" or not prediction or prediction["status"] != "ok":
        return None, None
    reference = (annotation.get("semantic_class") or "").strip().casefold()
    predicted = (prediction.get("semantic_class") or "").strip().casefold()
    if reference and predicted and reference != predicted:
        return "recognition", "semantic_class_disagreement; target grounding is also a possible explanation"
    mapping = config.get("class_policy_mapping", _RELLIS_POLICY if policy == "rellis_material_v1" else {})
    if reference and reference == predicted and reference in mapping and annotation["reference_label"] == mapping[reference] and prediction["label"] != mapping[reference]:
        return "policy", "matching_canonical_class_with_policy_label_disagreement"
    return None, None


def _cases(run, predictions, identity, config):
    result = []
    field = config.get("avoided_pixel_count_field", "avoided_pixels")
    for key, region in run["selected"].items():
        annotation, prediction, mask = run["annotations"].get(key), predictions.get(key), run["masks"][key]
        effective = None if prediction is None else "unknown" if prediction["status"] == "error" else prediction["label"]
        reference = annotation["reference_label"] if annotation else None
        semantic_ref = annotation.get("semantic_class") if annotation else None
        semantic_pred = prediction.get("semantic_class") if prediction else None
        hazard_pixels = run["transfer"].get(key, {}).get(field)
        flags = []
        if prediction is None:
            flags.append("missing_prediction")
        elif prediction["status"] == "error":
            flags.append("prediction_failure")
        if reference is not None and effective is not None and reference != effective:
            flags.append("label_disagreement")
        if semantic_ref and semantic_pred and semantic_ref.strip().casefold() != semantic_pred.strip().casefold():
            flags.append("semantic_class_disagreement")
        if mask["defect"] or (annotation and annotation["mask_quality"] in ("mixed", "broken")):
            flags.append("mask_defect")
        if hazard_pixels and effective == "traversable":
            flags.append("annotated_hazard_acceptance")
        if not flags:
            continue
        suggestion, evidence = _suggestion(annotation, prediction, mask, config["reference_policy_id"], config)
        frame = run["frames"][region["frame_id"]]
        priority = 0 if reference == "non_traversable" and effective == "traversable" else 1 if "annotated_hazard_acceptance" in flags else 2 if "mask_defect" in flags else 3 if "prediction_failure" in flags else 4
        result.append(dict(identity, case_id=fingerprint([identity["run_name"], identity["model_key"], identity["metadata_fingerprint"], key]),
                           region_id=key, frame_id=region["frame_id"], source=frame["source"], split=frame["split"],
                           reference_label=reference, predicted_label=effective,
                           prediction_status=prediction["status"] if prediction else "missing",
                           error_code=prediction["error_code"] if prediction else None,
                           semantic_class_reference=semantic_ref, semantic_class_prediction=semantic_pred,
                           reason=prediction["reason"] if prediction else "", raw_response=prediction["raw_response"] if prediction else "",
                           mask_quality=annotation["mask_quality"] if annotation else None, mask_defect=mask["defect"],
                           avoided_pixel_count=hazard_pixels, foreground_pixels=mask["foreground_pixels"], area_fraction=mask["area_fraction"],
                           image_path=frame["image_path"], mask_path=region["mask_path"], suggested_category=suggestion,
                           suggestion_evidence=evidence, confirmed_category=None, review_note="", flags=flags, priority=priority))
    return result


def _review(report, config):
    path = config.get("review_labels_path")
    if not path:
        return
    try:
        labels = read_jsonl(path, "case_id")
        cases = {row["case_id"]: row for row in report["error_cases"]}
        for key, row in labels.items():
            if key not in cases:
                raise ArtifactError(f"Review annotation refers to unknown/stale case: {key}")
            category = row.get("category")
            if category not in CATEGORIES:
                raise ArtifactError(f"Invalid review category: {category}")
            if not isinstance(row.get("note", ""), str):
                raise ArtifactError("Review note must be a string")
            if category != "unreviewed" and not row.get("note", "").strip():
                raise ArtifactError("Confirmed review categories require a nonempty evidence note")
        for key, row in labels.items():
            cases[key]["confirmed_category"] = row["category"]
            cases[key]["review_note"] = row.get("note", "")
    except ArtifactError as error:
        _issue(report, "warnings", "invalid_review_sidecar", str(error))


def evaluate(config):
    """Validate and compare configurations; incomplete artifacts never gain scores.

    Config errors raise ValueError. Artifact errors remain actionable in the
    returned validation report, with the affected configuration marked invalid.
    This function reads only; publication is an explicit separate operation.
    """
    config = _configuration(config)
    destination = (Path(config["run_root"]) / config["run_name"]).resolve()
    if not destination.is_relative_to(Path(config["run_root"])):
        raise ValueError("Destination run escapes run_root")
    report = {"schema_version": 1, "fixture": config["fixture"], "reference_policy_id": config["reference_policy_id"],
              "data_root": config["data_root"], "destination_run_dir": str(destination), "dataset_fingerprint": None,
              "validation": {"errors": [], "warnings": []}, "coverage": {}, "comparison": [], "per_source": [],
              "mask_breakdown": [], "hazard_overlap": [], "error_cases": []}
    cache, failures, baseline = {}, {}, None
    rejected_fixtures = []
    for entry in config["comparisons"]:
        name = entry["run_name"]
        if name not in cache and name not in failures:
            try:
                cache[name] = _read_run(Path(config["run_root"]) / name, config, report)
            except ArtifactError as error:
                failures[name] = str(error)
                _issue(report, "errors", "invalid_run_artifacts", str(error), run_name=name)
        run = cache.get(name)
        identity = _identity(entry)
        if run is None:
            report["comparison"].extend(_rows(None, {}, identity, "unavailable", invalid=True))
            continue
        if baseline is None:
            baseline = run
            report["dataset_fingerprint"] = run["dataset_fingerprint"]
            report["coverage"] = _coverage(run, config, report)
        compatible = baseline["dataset_fingerprint"] == run["dataset_fingerprint"]
        predictions, invalid = {}, False
        try:
            if not compatible:
                raise ArtifactError("Compared runs have incompatible frozen inputs/references/transfer evidence")
            directory = Path(config["run_root"]) / name
            metadata = read_json(directory / "metadata" / f"{entry['model_key']}.json")
            execution = metadata.get("execution", {})
            if not isinstance(execution, dict):
                raise ArtifactError("Metadata execution must be an object")
            if not config["fixture"] and (_fixture_record(metadata) or execution.get("kind") in ("injected_fixture", "fixture", "synthetic")):
                rejected_fixtures.append(dict(run_name=name, model_key=entry["model_key"]))
                raise ArtifactError("Fixture metadata/predictions require fixture=true; they are not real experiment results")
            metadata_digest, configuration_digest = validate_metadata(metadata, entry, config["reference_policy_id"], run["inputs"], config)
            identity = _identity(entry, metadata_digest, configuration_digest)
            path = directory / "predictions" / f"{entry['model_key']}.jsonl"
            if path.exists():
                predictions = read_jsonl(path, "region_id")
            else:
                _issue(report, "warnings", "missing_prediction_file", "Prediction file is absent; all selected IDs are missing",
                       run_name=name, model_key=entry["model_key"])
            if not config["fixture"] and any(_fixture_record(row) for row in predictions.values()):
                raise ArtifactError("Fixture predictions cannot count as real experiment results")
            for warning in validate_predictions(predictions, run["frames"], run["regions"], entry["model_key"]):
                report["validation"]["warnings"].append(dict(warning, run_name=name, model_key=entry["model_key"]))
        except ArtifactError as error:
            invalid = True
            predictions = {}
            _issue(report, "errors", "invalid_configuration", str(error), run_name=name, model_key=entry["model_key"])
        status = report["coverage"]["status"]
        rows = _rows(run, predictions, identity, status, invalid=invalid)
        report["comparison"].extend(row for row in rows if row["source"] == "all")
        report["per_source"].extend(row for row in rows if row["source"] != "all")
        if compatible:
            report["hazard_overlap"].extend(_hazard_rows(run, predictions, identity, config, invalid=invalid))
        if not invalid:
            missing = sorted(set(run["selected"]) - set(predictions))
            if missing:
                _issue(report, "warnings", "missing_predictions", "Selected predictions are missing; primary gaps block all primary scores for this configuration",
                       run_name=name, model_key=entry["model_key"], region_ids=missing,
                       primary_region_ids=sorted(set(run["primary"]) - set(predictions)))
            report["error_cases"].extend(_cases(run, predictions, identity, config))
    if baseline is None:
        settings = config.get("coverage", {})
        quotas = settings.get("expected_frames", {"rellis": {"development": 20, "test": 80}})
        clips = settings.get("phone_hazard_clips", {}).get("expected", 6)
        report["coverage"] = {"status": "unavailable", "fixture": config["fixture"],
                              "frames": [{"source": source, "split": split, "expected": count,
                                          "available": None, "readable": None, "real_available": 0, "real_readable": 0}
                                         for source, splits in quotas.items() for split, count in splits.items()],
                              "annotations": {"expected": None, "available": None, "missing": None,
                                              "pending": None, "complete": None, "primary_eligible": None},
                              "phone_hazard_clips": {"expected": clips, "available_recordings": 0, "not_available": clips,
                                                     "status": "optional_pending", "small_hazard_coverage": "not_established"}}
    if rejected_fixtures:
        report["fixture"] = True
        report["coverage"]["fixture"] = True
        report["coverage"]["status"] = "fixture_provenance_rejected"
        report["coverage"]["rejected_fixture_configurations"] = rejected_fixtures
        for row in report["coverage"].get("frames", []):
            row["real_available"] = row["real_readable"] = 0
        for row in report["coverage"].get("recordings", []):
            row["real_readable"] = 0
        for row in report["comparison"] + report["per_source"]:
            row["coverage_status"] = "fixture_provenance_rejected"
    report["error_cases"].sort(key=lambda row: (row["priority"], row["run_name"], row["model_key"], row["region_id"]))
    _review(report, config)
    return report
