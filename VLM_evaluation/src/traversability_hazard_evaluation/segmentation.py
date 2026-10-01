"""Score saved fixed-SAM artifacts without importing a segmentation backend."""

from collections import defaultdict
from copy import deepcopy
import re

import numpy as np

from .artifacts import (
    ArtifactError, binary_mask, binding, fingerprint, issue, normalize, pointer, rate,
    read_json, read_records, safe_path, validate_identity,
)


def _field(obj, ctx, kind, name, default):
    return pointer(obj, binding(ctx["config"], kind, name, default))


def _vocabulary(ctx, source):
    datasets = ctx["policy"]["datasets"]
    if source == "rellis":
        return list(datasets[source]["hazard_label_ids"])
    if source == "coco":
        return list(datasets[source]["scored_category_names"])
    return []


def _integers(record, names):
    for name in names:
        if type(record.get(name)) is not int or record[name] < 0:
            raise ArtifactError(f"{name} must be a nonnegative integer")


def _status(record):
    if record.get("status") not in ("ok", "error"):
        raise ArtifactError("status must be ok or error")
    code = record.get("error_code")
    if record["status"] == "ok" and code is not None:
        raise ArtifactError("Successful record has error_code")
    if record["status"] == "error" and (not isinstance(code, str) or not code.strip()):
        raise ArtifactError("Failed record requires error_code")


def _record_identity(record, condition):
    if record.get("task_id") != "hazard_prompt_v1" or type(record.get("schema_version")) is not int or record["schema_version"] != 1:
        raise ArtifactError("Segmentation record task/schema mismatch")
    if record.get("condition_key") != condition:
        raise ArtifactError("Segmentation record condition mismatch")


def _expected_prompts(ctx, condition, identifier):
    if condition == "fixed_policy":
        return list(ctx["policy"]["canonical_prompts"]), "ok"
    if condition == "reference_present":
        if identifier not in ctx["reference_valid"]:
            return None, "unavailable"
        return list(ctx["references"][identifier]["present_concepts"]), "ok"
    if condition.startswith("vlm__"):
        model = condition[len("vlm__"):]
        prediction = ctx["predictions"].get(model, {}).get(identifier)
        if prediction is None or identifier in ctx.get("prediction_invalid", {}).get(model, set()):
            return None, "unavailable"
        if prediction["status"] == "error":
            return [], "error"
        return list(prediction["prompts"]), "ok"
    raise ArtifactError(f"Unknown segmentation condition: {condition}")


def load_selection(ctx):
    """Load the frozen selection through explicit configurable field bindings."""
    config = ctx["config"].get("segmentation", {})
    selection = read_json(safe_path(ctx["run_dir"], config.get("selection_path", "segmentation/selection.json")))
    validate_identity(selection, ctx, "selection")
    test = _field(selection, ctx, "selection", "test_frame_ids", "/test_frame_ids")
    warmup = _field(selection, ctx, "selection", "warmup_frame_ids", "/warmup_frame_ids")
    digest = _field(selection, ctx, "selection", "selection_hash", "/selection_hash")
    if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ArtifactError("Frozen selection_hash must be a lowercase SHA-256")
    recipe = config.get("selection_hash_recipe", "declared")
    hash_verified = False
    if recipe in ("canonical_selection", "canonical_selection_dual"):
        hash_input = deepcopy(selection)
        selector = binding(ctx["config"], "selection", "selection_hash", "/selection_hash")
        parts = [part.replace("~1", "/").replace("~0", "~") for part in selector[1:].split("/")]
        parent = hash_input
        for part in parts[:-1]:
            parent = parent[int(part)] if isinstance(parent, list) else parent[part]
        if isinstance(parent, list):
            parent.pop(int(parts[-1]))
        else:
            parent.pop(parts[-1])
        if recipe == "canonical_selection_dual":
            if selection.get("fingerprint") != digest:
                raise ArtifactError("Frozen selection fingerprint differs from selection_hash")
            hash_input.pop("fingerprint", None)
        if fingerprint(hash_input) != digest:
            raise ArtifactError("Frozen selection canonical hash mismatch")
        hash_verified = True
    elif recipe != "declared":
        raise ArtifactError("selection_hash_recipe must be declared, canonical_selection or canonical_selection_dual")
    signature_binding = binding(ctx["config"], "selection", "input_signature", "/observed_input_signature")
    input_verified = False
    try:
        input_signature = pointer(selection, signature_binding)
    except ArtifactError:
        if "input_signature" in ctx["config"].get("metadata_bindings", {}).get("selection", {}):
            raise
    else:
        if input_signature != ctx.get("observed_input_signature"):
            raise ArtifactError("Frozen selection observed input signature changed")
        input_verified = True
    if recipe == "canonical_selection_dual":
        expected_identity = {"task_id": "hazard_prompt_v1", "schema_version": 1, "fixture": ctx["config"]["fixture"],
            "policy_sha256": ctx["dataset_metadata"].get("policy_sha256"),
            "policy_hash": fingerprint(ctx["policy"]), "alias_sha256": ctx["alias_hash"],
            "dataset": ctx["dataset_metadata"], "frames": [ctx["frames"][key] for key in sorted(ctx["frames"])],
            "references": [ctx["references"][key] for key in sorted(ctx["references"])], "assets": ctx["input_assets"]}
        if selection.get("identity") != expected_identity:
            raise ArtifactError("Frozen selection input records/assets/metadata changed")
        expected_selected = {key: {"image_sha256": ctx["frames"][key]["image_sha256"],
            "reference_sha256": fingerprint(ctx["references"][key])} for key in test + warmup if key in ctx["frames"] and key in ctx["references"]}
        if selection.get("selected_inputs") != expected_selected:
            raise ArtifactError("Frozen selected image/reference identity changed")
        input_verified = True
    for values, split in ((test, "test"), (warmup, "development")):
        if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
            raise ArtifactError("Frozen selection IDs must be lists of nonempty strings")
        if len(values) != len(set(values)):
            raise ArtifactError("Duplicate frozen selection frame ID")
        for identifier in values:
            frame = ctx["frames"].get(identifier)
            if frame is None or frame["split"] != split:
                raise ArtifactError(f"Frozen selection frame missing or wrong split: {identifier}")
            if identifier not in ctx["requested_ids"]:
                raise ArtifactError(f"Frozen selection frame outside declared evaluation coverage: {identifier}")
    if set(test) & set(warmup):
        raise ArtifactError("Frozen test and development selection overlap")
    expected_sources = config.get("expected_test_per_source", {"rellis": 10, "coco": 10})
    if not isinstance(expected_sources, dict) or any(source not in ("rellis", "coco") or type(count) is not int or count < 0 for source, count in expected_sources.items()):
        raise ArtifactError("Frozen source coverage must declare nonnegative integer counts")
    actual_sources = defaultdict(int)
    for identifier in test:
        actual_sources[ctx["frames"][identifier]["source"]] += 1
    if dict(actual_sources) != expected_sources:
        raise ArtifactError(f"Frozen test source coverage differs: expected {expected_sources}, received {dict(actual_sources)}")
    expected_warmup = config.get("expected_warmup_frames", 5)
    if type(expected_warmup) is not int or expected_warmup < 0 or len(warmup) != expected_warmup:
        raise ArtifactError("Frozen development warmup coverage differs")
    return {"artifact": selection, "test_frame_ids": test, "warmup_frame_ids": warmup, "selection_hash": digest,
            "hash_recipe": recipe, "hash_verified": hash_verified, "input_identity_verified": input_verified,
            "identity_verified": hash_verified and input_verified}


def _query_mask(ctx, query, condition, identifier, phrase, index, size):
    required = {"task_id", "schema_version", "frame_id", "condition_key", "query_id", "phrase", "canonical_concept", "mask_paths", "scores", "returned_instance_count", "union_mask_path", "status", "error_code"}
    if set(query) != required:
        raise ArtifactError("SAM query must have the exact frozen query fields")
    _record_identity(query, condition)
    if query.get("frame_id") != identifier or query.get("query_id") != f"{condition}:{identifier}:q{index:03d}":
        raise ArtifactError("Query/frame/index identity mismatch")
    if query.get("phrase") != phrase:
        raise ArtifactError("Saved SAM phrase differs from original required phrase")
    canonical = ctx["aliases"].get(normalize(phrase))
    if query.get("canonical_concept") != canonical:
        raise ArtifactError("Saved SAM canonical concept differs from frozen aliases")
    _status(query)
    if normalize(phrase) in ctx["vague"] and query["status"] != "error":
        raise ArtifactError("Vague/unusable phrase must be an explicit failed query")
    paths, scores = query.get("mask_paths"), query.get("scores")
    if not isinstance(paths, list) or not isinstance(scores, list) or len(paths) != len(scores):
        raise ArtifactError("Instance mask paths and scores must be aligned lists")
    if len(paths) != len(set(paths)) or any(not isinstance(path, str) for path in paths):
        raise ArtifactError("Invalid or duplicate instance mask path")
    _integers(query, ("returned_instance_count",))
    if query["returned_instance_count"] != len(paths):
        raise ArtifactError("Instance count differs from saved mask count")
    for score in scores:
        if type(score) not in (int, float) or not np.isfinite(score) or not 0 <= score <= 1:
            raise ArtifactError("Instance score must be a finite probability")
    instance_union = np.zeros((size[1], size[0]), dtype=bool)
    for path in paths:
        instance_union |= binary_mask(ctx["run_dir"], path, size)
    union_path = query.get("union_mask_path")
    if union_path is None and query["status"] == "error" and not paths:
        # A failed invocation may have no output asset; this never means success.
        union = instance_union
    else:
        union = binary_mask(ctx["run_dir"], union_path, size)
    if query["status"] == "ok" and not np.array_equal(union, instance_union):
        raise ArtifactError("Successful query union differs from its instance masks")
    return canonical, union if query["status"] == "ok" else np.zeros_like(union)


def _metrics(tp, fn, fp, available=True):
    if not available:
        return {name: {"numerator": None, "denominator": None, "rate": None} for name in ("hazard_recall", "precision", "iou")}
    return {"hazard_recall": rate(tp, tp + fn), "precision": rate(tp, tp + fp), "iou": rate(tp, tp + fn + fp)}


def _new_row(condition, source, split):
    return {"condition_key": condition, "source": source, "split": split,
            "result_kind": "no_vlm_baseline" if condition == "fixed_policy" else "segmentation_diagnostic" if condition == "reference_present" else "combined_vlm_sam",
            "expected_frames": 0, "saved_frames": 0, "pixel_tp": 0, "pixel_fn": 0, "pixel_fp": 0,
            "reference_frames": 0, "unavailable_reference_frames": 0, "failed_frames": 0,
            "required_queries": 0, "saved_queries": 0, "failed_queries": 0, "missing_queries": 0,
            "invalid_queries": 0, "unscored_queries": 0, "requested_unscored_queries": 0, "sam_query_count": 0,
            "complete": True, "metrics_available": True}


def evaluate_segmentation(ctx):
    """Evaluate optional original-size fixed-SAM outputs against saved truth."""
    config = ctx["config"].get("segmentation", {})
    result = {"enabled": bool(config.get("enabled", False)), "rows": [], "per_concept": [], "frame_audit": [], "unexpected_query_audit": [], "coverage": {"enabled": bool(config.get("enabled", False))}, "complete": True}
    if not config.get("enabled", False):
        return result
    issues = ctx["issues"]
    try:
        selection = load_selection(ctx)
    except (ArtifactError, OSError, ValueError, TypeError, KeyError) as error:
        issue(issues, "invalid_segmentation_selection", str(error), stage="segmentation")
        result["complete"] = False
        result["coverage"]["selection_valid"] = False
        return result
    result["coverage"].update(selection_valid=True, test_frame_ids=selection["test_frame_ids"], warmup_frame_ids=selection["warmup_frame_ids"], selection_hash=selection["selection_hash"],
        selection_hash_recipe=selection["hash_recipe"], selection_hash_verified=selection["hash_verified"], input_identity_verified=selection["input_identity_verified"], identity_verified=selection["identity_verified"])
    # Keep internal identity for independently loaded deployment profiles.
    result["selection"] = selection
    conditions = config.get("conditions", [])
    seen_conditions, common_settings = set(), None
    for entry in conditions:
        condition = entry if isinstance(entry, str) else entry.get("condition_key") if isinstance(entry, dict) else None
        if not isinstance(condition, str) or not condition or condition in seen_conditions:
            issue(issues, "invalid_segmentation_condition", "Condition names must be nonempty and unique", stage="segmentation", condition_key=condition)
            result["complete"] = False
            continue
        seen_conditions.add(condition)
        entry = {} if isinstance(entry, str) else entry
        base = f"segmentation/{condition}"
        condition_valid = True
        try:
            metadata = read_json(safe_path(ctx["run_dir"], entry.get("metadata_path", f"{base}/metadata.json")))
            validate_identity(metadata, ctx, "segmentation_metadata", condition_key=condition)
            if _field(metadata, ctx, "segmentation_metadata", "selection_hash", "/selection_hash") != selection["selection_hash"]:
                raise ArtifactError("Segmentation selection hash mismatch")
            dataset_hash = _field(metadata, ctx, "segmentation_metadata", "dataset_fingerprint", "/dataset_fingerprint")
            expected_hash = _field(ctx["dataset_metadata"], ctx, "dataset", "dataset_fingerprint", "/dataset_fingerprint")
            if dataset_hash != expected_hash:
                raise ArtifactError("Segmentation dataset fingerprint mismatch")
            settings = _field(metadata, ctx, "segmentation_metadata", "sam_settings", "/sam_settings")
            if not isinstance(settings, dict) or not settings:
                raise ArtifactError("Segmentation metadata requires actual SAM settings")
            for name in ("checkpoint", "revision"):
                pin = _field(settings, ctx, "sam_settings", name, "/" + name)
                if not isinstance(pin, str) or not pin:
                    raise ArtifactError(f"Segmentation SAM {name} pin must be nonempty")
            if common_settings is None:
                common_settings = settings
            elif settings != common_settings:
                raise ArtifactError("SAM checkpoint/settings differ across conditions")
            issues_before_read = len(issues)
            frame_records, bad_frames = read_records(safe_path(ctx["run_dir"], entry.get("frames_path", f"{base}/frames.jsonl")), "frame_id", issues, "segmentation")
            query_records, bad_queries = read_records(safe_path(ctx["run_dir"], entry.get("queries_path", f"{base}/queries.jsonl")), "query_id", issues, "segmentation")
            if len(issues) != issues_before_read:
                condition_valid = False
        except (ArtifactError, OSError, ValueError, TypeError, KeyError) as error:
            issue(issues, "invalid_segmentation_artifacts", str(error), stage="segmentation", condition_key=condition)
            condition_valid = False
            frame_records, query_records, bad_frames, bad_queries = {}, {}, set(), set()
        selected = set(selection["test_frame_ids"])
        if set(frame_records) - selected or bad_frames - selected:
            condition_valid = False
            issue(issues, "unexpected_segmentation_frames", "Condition contains frames outside frozen test selection", stage="segmentation", condition_key=condition)
        rows, concepts = {}, {}
        expected_query_ids = set()
        for identifier in selection["test_frame_ids"]:
            frame = ctx["frames"][identifier]
            source, split = frame["source"], frame["split"]
            row = rows.setdefault((source, split), _new_row(condition, source, split))
            row["complete"] &= condition_valid
            row["expected_frames"] += 1
            size = (frame["width"], frame["height"])
            empty = np.zeros((size[1], size[0]), dtype=bool)
            vocabulary = _vocabulary(ctx, source)
            masks = {name: empty.copy() for name in vocabulary}
            audit = {"condition_key": condition, "frame_id": identifier, "source": source, "split": split,
                     "full_union_path": None, "required_queries": 0, "missing_queries": [], "failed_queries": [], "invalid_queries": [], "unscored_queries": 0, "queries": []}
            try:
                phrases, upstream = _expected_prompts(ctx, condition, identifier)
            except (ArtifactError, KeyError, TypeError) as error:
                issue(issues, "invalid_segmentation_condition", str(error), stage="segmentation", condition_key=condition)
                phrases, upstream = None, "unavailable"
            if phrases is None:
                row["complete"] = False
                audit["upstream_status"] = "unavailable"
                phrases = []
            else:
                audit["upstream_status"] = upstream
            expected = [f"{condition}:{identifier}:q{index:03d}" for index in range(len(phrases))]
            expected_query_ids.update(expected)
            row["required_queries"] += len(expected)
            audit["required_queries"] = len(expected)
            frame_record = frame_records.get(identifier)
            frame_valid = frame_record is not None and identifier not in bad_frames and condition_valid
            if frame_record is not None:
                row["saved_frames"] += 1
                audit["full_union_path"] = frame_record.get("union_mask_path")
            if frame_valid:
                try:
                    if set(frame_record) != {"task_id", "schema_version", "frame_id", "condition_key", "query_ids",
                        "requested_queries", "completed_queries", "failed_queries", "sam_query_count", "upstream_status",
                        "union_mask_path", "status", "error_code"}:
                        raise ArtifactError("SAM frame must have the exact frozen frame fields")
                    _record_identity(frame_record, condition)
                    _status(frame_record)
                    _integers(frame_record, ("requested_queries", "completed_queries", "failed_queries", "sam_query_count"))
                    if frame_record.get("query_ids") != expected or frame_record["requested_queries"] != len(expected):
                        raise ArtifactError("Frame required-query IDs/count differ from condition prompts")
                    if frame_record.get("upstream_status") != upstream:
                        raise ArtifactError("Frame upstream status differs from saved VLM/reference condition")
                    if frame_record["sam_query_count"] > len(expected):
                        raise ArtifactError("Actual SAM call count exceeds required queries")
                    full_mask = binary_mask(ctx["run_dir"], frame_record.get("union_mask_path"), size)
                    if not phrases and upstream == "ok" and (frame_record["status"] != "ok" or np.any(full_mask)):
                        raise ArtifactError("Successful empty prompts require a successful empty full-union mask")
                    row["sam_query_count"] += frame_record["sam_query_count"]
                except (ArtifactError, OSError, ValueError, TypeError, KeyError) as error:
                    frame_valid = False
                    issue(issues, "invalid_segmentation_frame", str(error), stage="segmentation", condition_key=condition, frame_id=identifier)
            if not frame_valid:
                row["complete"] = False
                issue(issues, "missing_or_invalid_segmentation_frame", "One valid saved frame record is required", stage="segmentation", condition_key=condition, frame_id=identifier)
            success_count = failure_count = 0
            requested_concepts, success_concepts, failure_concepts = set(), set(), set()
            for index, (query_id, phrase) in enumerate(zip(expected, phrases)):
                canonical = ctx["aliases"].get(normalize(phrase))
                if canonical in vocabulary:
                    requested_concepts.add(canonical)
                else:
                    row["requested_unscored_queries"] += 1
                query = query_records.get(query_id)
                query_audit = {"query_id": query_id, "phrase": phrase, "canonical_concept": canonical, "source_scored": canonical in vocabulary,
                    "saved": query is not None, "status": query.get("status") if query else "missing", "error_code": query.get("error_code") if query else None,
                    "mask_paths": query.get("mask_paths") if query else None, "union_mask_path": query.get("union_mask_path") if query else None}
                audit["queries"].append(query_audit)
                if query_id in bad_queries:
                    row["invalid_queries"] += 1
                    row["complete"] = False
                    audit["invalid_queries"].append(query_id)
                    continue
                if query is None:
                    row["missing_queries"] += 1
                    row["complete"] = False
                    audit["missing_queries"].append(query_id)
                    continue
                row["saved_queries"] += 1
                if canonical not in vocabulary:
                    row["unscored_queries"] += 1
                    audit["unscored_queries"] += 1
                if query.get("status") == "error":
                    failure_count += 1
                    row["failed_queries"] += 1
                    audit["failed_queries"].append({"query_id": query_id, "error_code": query.get("error_code")})
                    if canonical in vocabulary:
                        failure_concepts.add(canonical)
                try:
                    canonical, mask = _query_mask(ctx, query, condition, identifier, phrase, index, size)
                    if query["status"] == "ok":
                        success_count += 1
                        if canonical in vocabulary:
                            success_concepts.add(canonical)
                            if frame_valid:
                                masks[canonical] |= mask
                except (ArtifactError, OSError, ValueError, TypeError, KeyError) as error:
                    row["invalid_queries"] += 1
                    row["complete"] = False
                    audit["invalid_queries"].append(query_id)
                    issue(issues, "invalid_segmentation_query", str(error), stage="segmentation", condition_key=condition, frame_id=identifier, query_id=query_id)
            if frame_valid:
                semantics = config.get("completed_queries_semantics", "successful")
                completed = success_count if semantics == "successful" else success_count + failure_count
                if semantics not in ("successful", "attempted"):
                    row["complete"] = False
                    issue(issues, "invalid_completed_query_semantics", "Use successful or attempted", stage="segmentation", condition_key=condition)
                if frame_record["completed_queries"] != completed or frame_record["failed_queries"] != failure_count:
                    row["complete"] = False
                    issue(issues, "segmentation_query_counter_mismatch", "Frame counters differ from saved query statuses", stage="segmentation", condition_key=condition, frame_id=identifier)
                if frame_record["status"] == "ok" and (failure_count or audit["missing_queries"] or audit["invalid_queries"] or upstream != "ok"):
                    row["complete"] = False
                    issue(issues, "segmentation_frame_status_mismatch", "Frame success contradicts failed/missing query or upstream status", stage="segmentation", condition_key=condition, frame_id=identifier)
            frame_failed = not frame_valid or upstream != "ok" or bool(failure_count or audit["missing_queries"] or audit["invalid_queries"]) or frame_record.get("status") == "error"
            row["failed_frames"] += int(frame_failed)
            audit["failed"] = frame_failed
            reference = ctx["references"].get(identifier)
            reference_available = identifier in ctx["reference_valid"] and identifier in ctx["reference_masks_valid"]
            for name in vocabulary:
                concept = concepts.setdefault((source, split, name), {"condition_key": condition, "source": source, "split": split, "concept": name,
                    "present_frames": 0, "prompted_present_frames": 0, "successful_query_present_frames": 0,
                    "failed_query_present_frames": 0, "pixel_tp": 0, "pixel_fn": 0, "pixel_fp": 0,
                    "metrics_available": True, "unavailable_reference_frames": 0})
                if identifier in ctx["reference_valid"]:
                    present = name in reference["present_concepts"]
                    concept["present_frames"] += int(present)
                    concept["prompted_present_frames"] += int(present and name in requested_concepts)
                    concept["successful_query_present_frames"] += int(present and name in success_concepts)
                    concept["failed_query_present_frames"] += int(present and name in failure_concepts)
            valid = None
            truths = {}
            if reference_available:
                try:
                    valid = binary_mask(ctx["data_root"], reference["valid_mask_path"], size)
                    for name in vocabulary:
                        truths[name] = binary_mask(ctx["data_root"], reference["concept_masks"][name], size) if name in reference["present_concepts"] else empty.copy()
                except (ArtifactError, OSError, ValueError, TypeError, KeyError) as error:
                    reference_available = False
                    issue(issues, "invalid_segmentation_reference", str(error), stage="segmentation", condition_key=condition, frame_id=identifier)
            if not reference_available:
                row["unavailable_reference_frames"] += 1
                row["metrics_available"] = False
                row["complete"] = False
                audit["pixel_metrics_available"] = False
                for name in vocabulary:
                    concepts[(source, split, name)]["metrics_available"] = False
                    concepts[(source, split, name)]["unavailable_reference_frames"] += 1
            else:
                row["reference_frames"] += 1
                truth_union, scored_union = empty.copy(), empty.copy()
                for name in vocabulary:
                    truth, prediction = truths[name] & valid, masks[name] & valid
                    truth_union |= truth
                    scored_union |= prediction
                    concept = concepts[(source, split, name)]
                    concept["pixel_tp"] += int(np.count_nonzero(prediction & truth))
                    concept["pixel_fn"] += int(np.count_nonzero(truth & ~prediction))
                    concept["pixel_fp"] += int(np.count_nonzero(prediction & ~truth))
                tp = int(np.count_nonzero(scored_union & truth_union))
                fn = int(np.count_nonzero(truth_union & ~scored_union))
                fp = int(np.count_nonzero(scored_union & ~truth_union))
                row["pixel_tp"] += tp
                row["pixel_fn"] += fn
                row["pixel_fp"] += fp
                audit.update(pixel_metrics_available=True, pixel_tp=tp, pixel_fn=fn, pixel_fp=fp, **_metrics(tp, fn, fp))
            result["frame_audit"].append(audit)
        unexpected = (set(query_records) | bad_queries) - expected_query_ids
        if unexpected:
            for row in rows.values():
                row["complete"] = False
            issue(issues, "unexpected_segmentation_queries", "Queries outside required frozen condition/frame prompts", stage="segmentation", condition_key=condition, query_ids=sorted(unexpected))
            for query_id in sorted(unexpected):
                query = query_records.get(query_id, {})
                unexpected_frame_id = query.get("frame_id")
                frame = ctx["frames"].get(unexpected_frame_id) if isinstance(unexpected_frame_id, str) else None
                phrase = query.get("phrase")
                canonical = ctx["aliases"].get(normalize(phrase)) if isinstance(phrase, str) else None
                source_scored = frame is not None and canonical in _vocabulary(ctx, frame["source"])
                result["unexpected_query_audit"].append({"condition_key": condition, "query_id": query_id,
                    "frame_id": query.get("frame_id"), "phrase": phrase, "canonical_concept": canonical,
                    "source_scored": source_scored, "status": query.get("status"), "union_mask_path": query.get("union_mask_path")})
                if frame is not None and (frame["source"], frame["split"]) in rows and not source_scored:
                    rows[(frame["source"], frame["split"])]["unscored_queries"] += 1
        for row in rows.values():
            row["status"] = "complete" if row["complete"] else "partial"
            row.update(_metrics(row["pixel_tp"], row["pixel_fn"], row["pixel_fp"], row["metrics_available"]))
            row["frame_failure_rate"] = rate(row["failed_frames"], row["expected_frames"])
            row["query_failure_rate"] = rate(row["failed_queries"], row["required_queries"])
            if not row["metrics_available"]:
                row["observed_pixel_counts"] = {key: row[key] for key in ("pixel_tp", "pixel_fn", "pixel_fp")}
                for key in ("pixel_tp", "pixel_fn", "pixel_fp"):
                    row[key] = None
            result["complete"] &= row["complete"]
            result["rows"].append(row)
        for concept in concepts.values():
            concept["prompt_coverage"] = rate(concept["prompted_present_frames"], concept["present_frames"])
            concept["successful_query_coverage"] = rate(concept["successful_query_present_frames"], concept["present_frames"])
            concept.update(_metrics(concept["pixel_tp"], concept["pixel_fn"], concept["pixel_fp"], concept["metrics_available"]))
            if not concept["metrics_available"]:
                concept["observed_pixel_counts"] = {key: concept[key] for key in ("pixel_tp", "pixel_fn", "pixel_fp")}
                for key in ("pixel_tp", "pixel_fn", "pixel_fp"):
                    concept[key] = None
            result["per_concept"].append(concept)
    if not conditions:
        result["complete"] = False
        issue(issues, "missing_segmentation_conditions", "Enabled SAM evaluation requires declared conditions", stage="segmentation")
    result["coverage"]["conditions"] = sorted(seen_conditions)
    if common_settings is not None:
        selection["sam_settings"] = common_settings
    return result
