"""Read-only evaluation of saved image-level hazard runs."""

from collections import Counter
from copy import deepcopy
import math
import os
from pathlib import Path
import re

from .artifacts import (ArtifactError, binary_mask, binding, file_hash, fingerprint,
                        issue, normalize, parse_json, pointer, read_json,
                        read_records, safe_path, validate_identity)


TASK = "hazard_prompt_v1"
POLICY_HASH = "89b2c1bf51940c579b6c0a6e7c645c82fe3501c424900ce23475a45b60c3e216"
ALIAS_HASH = "208ad2e568186aea3c4e1d3d5d1bb3b69547eae570963ce36cce901c4f9cd50f"


def _name(value, kind):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) or value in (".", ".."):
        raise ValueError(f"Invalid {kind}: {value!r}")
    return value


def _fixture_marked(row):
    return (row.get("fixture") is True or row.get("is_fixture") is True
            or row.get("execution_kind") in ("fixture", "synthetic", "injected_fixture")
            or any("fixture" in str(row.get(k, "")).casefold() or "synthetic" in str(row.get(k, "")).casefold() for k in ("kind", "annotation_source"))
            or any(_fixture_marked(value) for value in row.values() if isinstance(value, dict))
            or any(_fixture_marked(value) for items in row.values() if isinstance(items, list) for value in items if isinstance(value, dict)))


def _declared_identity(row, fixture):
    if "task_id" in row and row["task_id"] != TASK:
        raise ArtifactError("Declared row task identity mismatch")
    if "schema_version" in row and (type(row["schema_version"]) is not int or row["schema_version"] != 1):
        raise ArtifactError("Declared row schema identity mismatch")
    if "fixture" in row and (type(row["fixture"]) is not bool or row["fixture"] != fixture):
        raise ArtifactError("Declared row fixture marker mismatch")


def _config(config):
    if not isinstance(config, dict):
        raise ValueError("config must be a dict")
    result = deepcopy(config)
    component = Path(result.get("component_root", os.environ.get("TRAVERSABILITY_REPO_ROOT", Path(__file__).resolve().parents[2]))).resolve()
    # A repository override may name the parent checkout.
    if not (component / "docs/hazard_interfaces.md").is_file() and (component / "VLM_evaluation/docs/hazard_interfaces.md").is_file():
        component = component / "VLM_evaluation"
    result["component_root"] = str(component)
    for key, variable, default in (("data_root", "TRAVERSABILITY_DATA_ROOT", "data"), ("run_root", "TRAVERSABILITY_RUN_ROOT", "runs")):
        root = Path(result.get(key, os.environ.get(variable, default))).expanduser()
        result[key] = str((root if root.is_absolute() else component / root).resolve())
    result.setdefault("run_name", TASK)
    _name(result["run_name"], "run_name")
    result.setdefault("model_keys", ["qwen3_vl_4b", "qwen3_5_4b"])
    if not isinstance(result["model_keys"], list) or not result["model_keys"] or len(set(result["model_keys"])) != len(result["model_keys"]):
        raise ValueError("model_keys must be a nonempty unique list")
    for model in result["model_keys"]:
        _name(model, "model_key")
    result.setdefault("fixture", False)
    if type(result["fixture"]) is not bool:
        raise ValueError("fixture must be boolean")
    result.setdefault("coverage", {"expected_frames": {"rellis": {"development": 20, "test": 80}, "coco": {"development": 8, "test": 32}}})
    if not isinstance(result["coverage"], dict):
        raise ValueError("coverage must be an object")
    counts = result["coverage"].get("expected_frames")
    if not isinstance(counts, dict) or any(source not in ("rellis", "coco") or not isinstance(splits, dict) or any(split not in ("development", "test") or type(count) is not int or count < 0 for split, count in splits.items()) for source, splits in counts.items()):
        raise ValueError("coverage.expected_frames must declare nonnegative source/split counts")
    result.setdefault("metadata_bindings", {})
    if not isinstance(result["metadata_bindings"], dict):
        raise ValueError("metadata_bindings must be an object")
    for kind, fields in result["metadata_bindings"].items():
        if not isinstance(fields, dict) or any(not isinstance(v, str) or not v.startswith("/") for v in fields.values()):
            raise ValueError(f"metadata_bindings.{kind} must contain JSON pointers")
    result.setdefault("hash_recipes", {"policy": "canonical_json", "aliases": "canonical_json"})
    if not isinstance(result["hash_recipes"], dict) or set(result["hash_recipes"]) != {"policy", "aliases"} or result["hash_recipes"]["policy"] not in ("canonical_json", "file_bytes") or result["hash_recipes"]["aliases"] != "canonical_json":
        raise ValueError("hash_recipes supports policy canonical_json/file_bytes and aliases canonical_json")
    result.setdefault("dataset_fingerprint_recipe", "declared")
    if result["dataset_fingerprint_recipe"] not in ("declared", "frames_references_assets_v1", "hazard_data_v1"):
        raise ValueError("Unknown dataset_fingerprint_recipe")
    if "expected_input_signature" in result:
        try:
            _sha(result["expected_input_signature"], "expected_input_signature")
        except ArtifactError as error:
            raise ValueError(str(error)) from error
    result.setdefault("segmentation", {"enabled": False, "conditions": []})
    if not isinstance(result["segmentation"], dict) or type(result["segmentation"].get("enabled", False)) is not bool:
        raise ValueError("segmentation must be an object with boolean enabled")
    if not isinstance(result["segmentation"].get("conditions", []), list):
        raise ValueError("segmentation.conditions must be a list")
    if result["segmentation"].get("completed_queries_semantics", "successful") not in ("successful", "attempted"):
        raise ValueError("completed_queries_semantics must be successful or attempted")
    result.setdefault("deployment", {"profiles": []})
    if not isinstance(result["deployment"], dict) or not isinstance(result["deployment"].get("profiles", []), list):
        raise ValueError("deployment.profiles must be a list")
    return result


def _strings(value, label):
    if not isinstance(value, list) or any(not isinstance(v, str) or not v.strip() for v in value) or len(set(value)) != len(value):
        raise ArtifactError(f"{label} must be a unique list of nonempty strings")
    return set(value)


def _frame_ids(value, label):
    """An explicit pointer may bind IDs directly or a producer's frozen frames."""
    if isinstance(value, list) and value and all(isinstance(row, dict) for row in value):
        value = [row.get("frame_id") for row in value]
    return _strings(value, label)


def _sha(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ArtifactError(f"Invalid SHA-256 {label}")


def _record_identity(row):
    if row.get("task_id") != TASK or type(row.get("schema_version")) is not int or row["schema_version"] != 1:
        raise ArtifactError("Prediction task/schema identity mismatch")
    if any(key in row for key in ("region_id", "label", "reference_label", "selected_for_classification")):
        raise ArtifactError("Legacy region-classification artifact is not a hazard prediction")


def _phrases(value, policy):
    if not isinstance(value, list) or len(value) > policy["max_prompts"]:
        raise ArtifactError("Invalid prompts list or too many phrases")
    for phrase in value:
        if not isinstance(phrase, str) or not normalize(phrase) or len(phrase) > policy["max_phrase_characters"]:
            raise ArtifactError("Invalid, empty or oversized prompt phrase")
        try:
            phrase.encode("utf-8")
        except UnicodeError as error:
            raise ArtifactError("Prompt phrase contains invalid Unicode") from error
    return value


def _normalized_unique(phrases):
    return list(dict.fromkeys(normalize(p) for p in phrases))


def _original_unique(phrases):
    seen, result = set(), []
    for phrase in phrases:
        key = normalize(phrase)
        if key not in seen:
            seen.add(key)
            result.append(phrase)
    return result


def _parse_prediction_response(row, metadata=None):
    """Read JSON without discarding an unaudited trailing generation token."""
    try:
        return parse_json(row["raw_response"])
    except ValueError:
        histories = (metadata or {}).get("attempts", {})
        attempts = histories.get(row["frame_id"], []) if isinstance(histories, dict) else []
        if not isinstance(attempts, list) or not attempts:
            raise ArtifactError("Successful response has no verified terminal-EOS audit")
        attempt = attempts[-1]
        if not isinstance(attempt, dict):
            raise ArtifactError("Successful response attempt audit is malformed")
        diagnostics = attempt.get("diagnostics", {})
        if not isinstance(diagnostics, dict):
            raise ArtifactError("Successful response diagnostics are malformed")
        generation = diagnostics.get("generation", {})
        if not isinstance(generation, dict):
            raise ArtifactError("Successful response generation audit is malformed")
        response = diagnostics.get("parse_response_text")
        eos = generation.get("eos_text")
        if (attempt.get("record") != row or attempt.get("record_sha256") != fingerprint(row, ensure_ascii=True)
                or generation.get("terminal_eos") is not True or generation.get("stop_reason") != "eos"
                or type(generation.get("eos_token_id")) is not int
                or not isinstance(response, str) or not isinstance(eos, str) or not eos
                or row["raw_response"] != response + eos):
            raise ArtifactError("Successful response terminal-EOS audit is inconsistent")
        return parse_json(response)


def _validate_prediction(row, model, frames, policy, metadata=None):
    if set(row) != {"task_id", "schema_version", "frame_id", "model_key", "prompts", "raw_response", "status", "error_code"}:
        raise ArtifactError("Prediction must have the exact whole-image hazard fields")
    _record_identity(row)
    if row.get("model_key") != model or row["frame_id"] not in frames:
        raise ArtifactError("Prediction model/frame identity mismatch")
    _phrases(row.get("prompts"), policy)
    if not isinstance(row.get("raw_response"), str):
        raise ArtifactError("Prediction raw_response must be a string")
    if row.get("status") == "error":
        if not isinstance(row.get("error_code"), str) or not row["error_code"].strip():
            raise ArtifactError("Error prediction requires error_code")
        if row["prompts"]:
            raise ArtifactError("Failed generation cannot retain successful prompts")
    elif row.get("status") == "ok":
        if row.get("error_code") is not None or "error_code" not in row:
            raise ArtifactError("Successful prediction requires null error_code")
        raw = _parse_prediction_response(row, metadata)
        if set(raw) != {"prompts"}:
            raise ArtifactError("Successful raw response must have exactly prompts")
        _phrases(raw["prompts"], policy)
        if _original_unique(raw["prompts"]) != _original_unique(row["prompts"]):
            raise ArtifactError("Saved prompts disagree with raw response")
    else:
        raise ArtifactError("Invalid prediction status")


def _vocabulary(policy, source):
    if source == "rellis":
        return set(policy["datasets"][source]["hazard_label_ids"])
    return set(policy["datasets"][source]["scored_category_names"])


def _load(ctx):
    config, run, root, issues = ctx["config"], ctx["run_dir"], ctx["data_root"], ctx["issues"]
    if any((run / relative).exists() for relative in ("regions.jsonl", "annotations.jsonl", "metadata/data.json")):
        issue(issues, "legacy_region_artifacts", "Legacy region artifacts cannot score as hazard_prompt_v1", stage="data")
    frames, bad_frames = read_records(run / "frames.jsonl", "frame_id", issues, "frames")
    refs, bad_refs = read_records(run / "references.jsonl", "frame_id", issues, "references")
    ctx.update(frames=frames, references=refs, reference_valid=set(), reference_masks_valid=set(), predictions={}, prediction_invalid={}, model_metadata={}, prediction_raw_prompts={})
    ctx["fixture_detected"] = any(_fixture_marked(row) for row in [*frames.values(), *refs.values()])
    if ctx["fixture_detected"] and not config["fixture"]:
        issue(issues, "fixture_provenance_mismatch", "Marked synthetic rows cannot establish real coverage", stage="data")
    frame_structural_bad, readable = set(bad_frames), set()
    for fid, row in list(frames.items()):
        try:
            _declared_identity(row, config["fixture"])
            if row.get("source") not in ("rellis", "coco") or row.get("split") not in ("development", "test"):
                raise ArtifactError("Invalid frame source/split")
            for key in ("scene_id", "sequence_id"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ArtifactError(f"Frame missing {key}")
            if "timestamp_s" not in row or (row["timestamp_s"] is not None and (type(row["timestamp_s"]) not in (int, float) or not math.isfinite(row["timestamp_s"]))):
                raise ArtifactError("Invalid timestamp_s")
            if any(type(row.get(k)) is not int or row[k] <= 0 for k in ("width", "height")):
                raise ArtifactError("Invalid frame dimensions")
            _sha(row.get("image_sha256"), "image_sha256")
            path = safe_path(root, row.get("image_path"))
            if not config["fixture"] and row["source"] == "rellis":
                expected_split = "development" if row["sequence_id"] == "00000" else "test" if row["sequence_id"] in ("00001", "00002", "00003", "00004") else None
                if expected_split != row["split"]:
                    raise ArtifactError("RELLIS recording violates frozen split")
        except (ValueError, KeyError) as error:
            issue(issues, "invalid_frame", error, stage="data", frame_id=fid)
            frame_structural_bad.add(fid)
            frames.pop(fid)
            continue
        try:
            from PIL import Image
            with Image.open(path) as image:
                image.load()
                if image.size != (row["width"], row["height"]):
                    raise ArtifactError("RGB image dimension mismatch")
                if image.mode != "RGB":
                    raise ArtifactError("Expected original RGB image")
            if file_hash(path) != row["image_sha256"]:
                raise ArtifactError("RGB image SHA-256 mismatch")
            readable.add(fid)
        except (OSError, ValueError) as error:
            issue(issues, "invalid_image", error, stage="data", frame_id=fid)
    ctx["readable_frames"] = readable
    # COCO sequences need not define split isolation; RELLIS recordings do.
    groups = {}
    for fid, row in frames.items():
        if row["source"] == "rellis":
            key = (row["source"], row["sequence_id"])
            if groups.setdefault(key, row["split"]) != row["split"]:
                issue(issues, "split_leakage", "RELLIS sequence spans development/test", stage="data", frame_id=fid)
    metadata = {}
    try:
        metadata = read_json(run / "metadata/dataset.json")
        ctx["dataset_metadata"] = metadata
        ctx["fixture_detected"] |= _fixture_marked(metadata)
        validate_identity(metadata, ctx, "dataset")
        _sha(pointer(metadata, binding(config, "dataset", "dataset_fingerprint", "/dataset_fingerprint")), "dataset_fingerprint")
        selected = _strings(pointer(metadata, binding(config, "dataset", "selected_frame_ids", "/selected_frame_ids")), "selected_frame_ids")
        ctx["requested_ids"] = selected
        if selected != set(frames) | frame_structural_bad:
            issue(issues, "selected_ids_mismatch", "Dataset selected IDs differ from frame manifest", stage="data", missing_ids=sorted(selected - set(frames)), unexpected_ids=sorted(set(frames) - selected))
        declared = pointer(metadata, binding(config, "dataset", "expected_frames", "/coverage/expected_frames"))
        if declared != config["coverage"]["expected_frames"]:
            raise ArtifactError("Declared dataset coverage differs from evaluation coverage")
        for name in ("source_annotations", "sampling"):
            if not isinstance(pointer(metadata, binding(config, "dataset", name, "/" + name)), dict):
                raise ArtifactError(f"Dataset {name} must be an object")
        provenance = pointer(metadata, binding(config, "dataset", "source_annotations", "/source_annotations"))
        if config["dataset_fingerprint_recipe"] == "hazard_data_v1":
            if set(provenance) != set(frames):
                raise ArtifactError("Annotation provenance must join every materialized frame")
            for fid, annotation in provenance.items():
                if not isinstance(annotation, dict) or not isinstance(annotation.get("kind"), str):
                    raise ArtifactError(f"Missing annotation provenance for {fid}")
                _sha(annotation.get("sha256"), "annotation sha256")
                if file_hash(safe_path(root, annotation.get("path"))) != annotation["sha256"]:
                    raise ArtifactError(f"Annotation bytes changed for {fid}")
        else:
            for source in {row["source"] for row in frames.values()}:
                if source not in provenance or not isinstance(provenance[source], dict) or not provenance[source]:
                    raise ArtifactError(f"Missing annotation provenance for {source}")
    except (ValueError, TypeError, OSError) as error:
        issue(issues, "invalid_dataset_metadata", error, stage="data")
    ctx.setdefault("requested_ids", set(frames) | frame_structural_bad)
    ctx["dataset_metadata"] = metadata
    for fid, ref in refs.items():
        try:
            _declared_identity(ref, config["fixture"])
            if fid not in frames:
                raise ArtifactError("Reference has no valid frame")
            if ref.get("status") == "pending":
                continue
            if ref.get("status") != "complete":
                raise ArtifactError("Reference status must be complete or pending")
            vocab = _vocabulary(ctx["policy"], frames[fid]["source"])
            if _strings(ref.get("scored_concepts"), "scored_concepts") != vocab:
                raise ArtifactError("Reference source vocabulary mismatch")
            present = _strings(ref.get("present_concepts"), "present_concepts")
            if not present <= vocab:
                raise ArtifactError("Reference contains unscored present concepts")
            if type(ref.get("absence_scoring_eligible")) is not bool:
                raise ArtifactError("Reference absence eligibility must be boolean")
            if ref.get("reference_scope") != "annotated_hazard_concepts" or not isinstance(ref.get("annotation_source"), str) or not ref["annotation_source"]:
                raise ArtifactError("Reference annotation scope/provenance mismatch")
            for field in ("concept_masks", "concept_pixel_counts"):
                if not isinstance(ref.get(field), dict) or set(ref[field]) != present:
                    raise ArtifactError(f"Reference {field} keys must equal present concepts")
            for concept in present:
                count = ref["concept_pixel_counts"][concept]
                if type(count) is not int or not 1 <= count <= frames[fid]["width"] * frames[fid]["height"]:
                    raise ArtifactError("Invalid concept pixel count")
                safe_path(root, ref["concept_masks"][concept])
            safe_path(root, ref.get("valid_mask_path"))
            ctx["reference_valid"].add(fid)
        except (ValueError, KeyError) as error:
            issue(issues, "invalid_reference", error, stage="data", frame_id=fid)
            continue
        try:
            size = (frames[fid]["width"], frames[fid]["height"])
            valid = binary_mask(root, ref["valid_mask_path"], size)
            if frames[fid]["source"] == "rellis":
                eligible = (valid.size - int(valid.sum())) / valid.size <= ctx["policy"]["datasets"]["rellis"]["absence_scoring_max_ignored_fraction"]
                if eligible != ref["absence_scoring_eligible"]:
                    raise ArtifactError("RELLIS absence eligibility disagrees with ignored pixel fraction")
            elif not ref["absence_scoring_eligible"] or not valid.all():
                raise ArtifactError("COCO selected categories require exhaustive valid-pixel coverage")
            for concept in present:
                mask = binary_mask(root, ref["concept_masks"][concept], size)
                if int(mask.sum()) != ref["concept_pixel_counts"][concept] or (mask & ~valid).any():
                    raise ArtifactError("Reference mask pixel count/valid-pixel mismatch")
            ctx["reference_masks_valid"].add(fid)
        except (ValueError, OSError) as error:
            issue(issues, "invalid_reference_mask", error, stage="data", frame_id=fid)
    for model in config["model_keys"]:
        ctx["prediction_raw_prompts"][model] = {}
        # The EOS parsing audit lives in model metadata; load it before records.
        try:
            parsing_metadata = read_json(run / "metadata" / (model + ".json"))
        except ValueError:
            parsing_metadata = {}
        preds, invalid = read_records(run / "predictions" / (model + ".jsonl"), "frame_id", issues, "prediction:" + model)
        for fid, row in list(preds.items()):
            try:
                ctx["fixture_detected"] |= _fixture_marked(row)
                _validate_prediction(row, model, frames, ctx["policy"], parsing_metadata)
                if row["status"] == "ok":
                    ctx["prediction_raw_prompts"][model][fid] = _parse_prediction_response(row, parsing_metadata)["prompts"]
                if fid not in ctx["requested_ids"]:
                    raise ArtifactError("Prediction was not requested")
                if "fixture" in row and row["fixture"] is not config["fixture"]:
                    raise ArtifactError("Prediction fixture marker mismatch")
            except (ValueError, TypeError, KeyError, AttributeError) as error:
                invalid.add(fid)
                preds.pop(fid)
                issue(issues, "invalid_prediction", error, stage="prediction:" + model, frame_id=fid, model_key=model,
                      raw_response=row.get("raw_response"), raw_prompts=row.get("prompts"))
        ctx["predictions"][model], ctx["prediction_invalid"][model] = preds, invalid
        missing = ctx["requested_ids"] - preds.keys() - invalid
        if missing:
            issue(issues, "missing_requested_predictions", "One saved prediction is required for every requested frame", stage="prediction:" + model, model_key=model, frame_ids=sorted(missing))
        try:
            meta = read_json(run / "metadata" / (model + ".json"))
            ctx["fixture_detected"] |= _fixture_marked(meta)
            validate_identity(meta, ctx, "model")
            if pointer(meta, binding(config, "model", "model_key", "/model_key")) != model:
                raise ArtifactError("Model metadata identity mismatch")
            if _frame_ids(pointer(meta, binding(config, "model", "requested_frame_ids", "/requested_frame_ids")), "requested_frame_ids") != ctx["requested_ids"]:
                raise ArtifactError("Model requested-frame coverage mismatch")
            if "frozen_inputs" in config.get("metadata_bindings", {}).get("model", {}):
                frozen_inputs = pointer(meta, binding(config, "model", "frozen_inputs", "/inputs"))
                if (frozen_inputs.get("dataset") != metadata
                        or frozen_inputs.get("frames") != [frames[k] for k in sorted(frames)]):
                    raise ArtifactError("Model frozen data/frame records differ from prepared inputs")
                contents = frozen_inputs.get("rgb_contents")
                if not isinstance(contents, dict) or set(contents) != {row["image_path"] for row in frames.values()}:
                    raise ArtifactError("Model frozen RGB coverage differs from frames")
                for row in frames.values():
                    if row["frame_id"] in readable and contents[row["image_path"]] != {"status": "present", "sha256": row["image_sha256"]}:
                        raise ArtifactError("Model frozen RGB bytes differ from prepared inputs")
            selector = binding(config, "model", "execution_kind", "/execution_kind")
            try:
                execution = pointer(meta, selector)
            except ArtifactError:
                if "execution_kind" in config.get("metadata_bindings", {}).get("model", {}):
                    raise
                execution = None
            if not config["fixture"] and execution in ("fixture", "synthetic", "injected_fixture"):
                raise ArtifactError("Synthetic execution cannot be a real run")
            if "attempts" in meta:
                for fid, row in preds.items():
                    history = meta["attempts"].get(fid, [])
                    if (not history or history[-1].get("record") != row
                            or history[-1].get("record_sha256") != fingerprint(row, ensure_ascii=True)
                            or meta.get("prediction_digests", {}).get(fid) != fingerprint(row, ensure_ascii=True)):
                        raise ArtifactError("Prediction differs from its committed attempt audit")
            bases = {"task_id", "schema_version", "model_key", "configuration", "execution", "inputs", "actual_backend"}
            recipes = {"configuration_fingerprint": bases - {"actual_backend"}, "compatibility_fingerprint": bases,
                       "audit_fingerprint": {"attempts", "load_history", "prediction_digests", "pending_prediction"}}
            for name, keys in recipes.items():
                if name in meta and meta[name] != fingerprint({key: meta[key] for key in keys}, ensure_ascii=True):
                    raise ArtifactError(f"Model {name} committed fingerprint changed")
            ctx["model_metadata"][model] = meta
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            issue(issues, "invalid_model_metadata", error, stage="prediction:" + model, model_key=model)
    observed = Counter((row["source"], row["split"]) for row in frames.values())
    coverage_rows = []
    for source, splits in config["coverage"]["expected_frames"].items():
        for split, expected in splits.items():
            actual = observed[source, split]
            coverage_rows.append({"source": source, "split": split, "expected_frames": expected,
                                  "manifest_frames": actual, "readable_frames": sum(fid in readable for fid, f in frames.items() if (f["source"], f["split"]) == (source, split))})
            if actual != expected:
                issue(issues, "source_split_coverage_mismatch", f"Expected {expected}, found {actual}", stage="data", source=source, split=split)
    for (source, split), actual in observed.items():
        if split not in config["coverage"]["expected_frames"].get(source, {}):
            issue(issues, "undeclared_source_split", f"Found {actual} undeclared frames", stage="data", source=source, split=split)
    ctx["coverage_rows"] = coverage_rows
    asset_paths = {f["image_path"] for f in frames.values()}
    for ref in refs.values():
        if ref.get("status") == "complete":
            if isinstance(ref.get("concept_masks"), dict):
                asset_paths.update(p for p in ref["concept_masks"].values() if isinstance(p, str))
            if isinstance(ref.get("valid_mask_path"), str):
                asset_paths.add(ref["valid_mask_path"])
    assets = {}
    for relative in sorted(asset_paths):
        try:
            assets[relative] = file_hash(safe_path(root, relative))
        except (OSError, ValueError) as error:
            assets[relative] = {"error": str(error)}
    ctx["observed_input_signature"] = fingerprint({"frames": [frames[k] for k in sorted(frames)], "references": [refs[k] for k in sorted(refs)], "assets": assets})
    ctx["input_assets"] = assets
    ctx["input_identity_verified"] = False
    try:
        declared_fingerprint = pointer(metadata, binding(config, "dataset", "dataset_fingerprint", "/dataset_fingerprint"))
        expected = config.get("expected_input_signature")
        if config["dataset_fingerprint_recipe"] == "frames_references_assets_v1":
            expected = declared_fingerprint
        elif config["dataset_fingerprint_recipe"] == "hazard_data_v1":
            declared_assets = metadata.get("assets")
            if not isinstance(declared_assets, dict):
                raise ArtifactError("Hazard data metadata must declare all frozen assets")
            for relative, digest in declared_assets.items():
                _sha(digest, "dataset asset")
                if file_hash(safe_path(root, relative)) != digest:
                    raise ArtifactError(f"Frozen dataset asset changed: {relative}")
            observed_fingerprint = fingerprint({"metadata": {k: v for k, v in metadata.items() if k != "dataset_fingerprint"},
                                                "frames": [frames[k] for k in sorted(frames)],
                                                "references": [refs[k] for k in sorted(refs)]})
            if observed_fingerprint != declared_fingerprint:
                raise ArtifactError("Hazard data metadata/records fingerprint changed")
            selector = binding(config, "dataset", "input_signature", "/observed_input_signature")
            try:
                published_pin = pointer(metadata, selector)
            except ArtifactError:
                published_pin = None
            if expected is not None and published_pin is not None and expected != published_pin:
                raise ArtifactError("Configured input signature differs from the prepared data pin")
            expected = expected or published_pin
        if expected is not None:
            if expected != ctx["observed_input_signature"]:
                raise ArtifactError("Frozen observed input signature differs from frames/references/asset bytes")
            ctx["input_identity_verified"] = True
        ctx["declared_dataset_fingerprint"] = declared_fingerprint
    except (ValueError, TypeError, OSError) as error:
        issue(issues, "invalid_input_fingerprint", error, stage="data")


def evaluate(config):
    """Validate saved artifacts and return complete or explicitly provisional evidence."""
    try:
        config = _config(config)
    except (TypeError, AttributeError, KeyError) as error:
        raise ValueError(f"Invalid evaluation configuration: {error}") from error
    issues = []
    run = safe_path(config["run_root"], config["run_name"])
    policy_path = Path(config.get("policy_path", "configs/hazards/policy.json"))
    if not policy_path.is_absolute():
        policy_path = Path(config["component_root"]) / policy_path
    try:
        policy = read_json(policy_path)
        if fingerprint(policy) != POLICY_HASH or fingerprint(policy["aliases"]) != ALIAS_HASH:
            raise ArtifactError("Frozen hazard policy or alias table has changed")
    except (ValueError, KeyError) as error:
        return {"task_id": TASK, "schema_version": 1, "fixture": config["fixture"], "identity": {"run_name": config["run_name"]}, "validation": {"errors": [{"code": "invalid_policy", "message": str(error)}]}, "coverage": {}, "concepts": {}, "segmentation": {"enabled": False}, "deployment": {"rows": [], "complete": False}, "comparison": {"discovery_complete": False, "comparison_ready": False, "model_selection_ready": False, "winner": None}}
    ctx = {"config": config, "run_dir": run, "data_root": Path(config["data_root"]), "policy": policy, "aliases": {normalize(a): c for c, names in policy["aliases"].items() for a in names}, "vague": set(map(normalize, policy["vague_unusable_phrases"])), "issues": issues, "policy_hash": file_hash(policy_path) if config["hash_recipes"]["policy"] == "file_bytes" else POLICY_HASH, "alias_hash": ALIAS_HASH}
    _load(ctx)
    from .concepts import evaluate_concepts
    from .segmentation import evaluate_segmentation
    from .deployment import evaluate_deployment
    concepts = evaluate_concepts(ctx)
    segmentation = evaluate_segmentation(ctx) if config["segmentation"].get("enabled", False) else {"enabled": False, "rows": [], "per_concept": [], "frame_audit": [], "coverage": {}, "complete": None}
    deployment = evaluate_deployment(ctx, segmentation.get("selection"))
    data_errors = any(e.get("stage") in ("data", "frames", "references") for e in issues)
    discovery_errors = data_errors or any(str(e.get("stage", "")).startswith("prediction:") for e in issues)
    reference_complete = ctx["requested_ids"] <= ctx["reference_valid"] and ctx["requested_ids"] <= ctx["reference_masks_valid"]
    discovery_complete = concepts["complete"] and reference_complete and ctx["requested_ids"] <= ctx["readable_frames"] and not discovery_errors
    if discovery_errors:
        for row in concepts["rows"]:
            row["status"] = "provisional"
            row["counts_are_provisional"] = True
        for row in concepts["per_concept"]:
            row["counts_are_provisional"] = True
    sam_complete = segmentation.get("complete")
    configured_cost = bool(config["deployment"].get("profiles"))
    cost_complete = deployment["complete"] if configured_cost else None
    sam_identity_verified = segmentation.get("selection", {}).get("identity_verified", False) if config["segmentation"].get("enabled", False) else True
    has_test_frames = any(f["split"] == "test" for f in ctx["frames"].values())
    ready = discovery_complete and has_test_frames and ctx["input_identity_verified"] and sam_identity_verified and (sam_complete is not False) and (cost_complete is not False) and not config["fixture"]
    # Accuracy comparison completeness is distinct from availability for a resource-informed decision.
    fixture = config["fixture"] or ctx["fixture_detected"]
    cost_models = {r.get("condition_key", "").removeprefix("vlm__") for r in deployment["rows"] if r.get("valid") and r.get("complete") and r.get("comparison_ready") and r.get("latency_measured") and r.get("resource_measurements_available") and r.get("selection_identity_verified") and r.get("component") in ("vlm", "combined")}
    return {"task_id": TASK, "schema_version": 1, "fixture": fixture,
            "identity": {"run_name": config["run_name"], "run_dir": str(run), "data_root": config["data_root"], "policy_id": policy["policy_id"], "policy_hash": ctx["policy_hash"], "alias_hash": ctx["alias_hash"], "dataset_fingerprint": ctx.get("declared_dataset_fingerprint"), "observed_input_signature": ctx["observed_input_signature"], "input_identity_verified": ctx["input_identity_verified"]},
            "validation": {"errors": issues},
            "coverage": {"source_split": ctx["coverage_rows"], "requested_frame_ids": sorted(ctx["requested_ids"]), "missing_manifest_ids": sorted(ctx["requested_ids"] - ctx["frames"].keys()), "real_manifest_frames": 0 if fixture else len(ctx["frames"]), "pending_coverage": policy["pending_coverage"], "unjoined_complete_reference_positives": [{"frame_id": fid, "present_concepts": ref.get("present_concepts")} for fid, ref in ctx["references"].items() if fid not in ctx["frames"] and ref.get("status") == "complete"]},
            "concepts": concepts, "segmentation": segmentation, "deployment": deployment,
            "comparison": {"discovery_complete": discovery_complete, "segmentation_complete": sam_complete, "deployment_complete": cost_complete, "comparison_ready": ready and not fixture, "model_selection_ready": ready and not fixture and configured_cost and deployment["complete"] and set(config["model_keys"]) <= cost_models, "final_split": "test", "winner": None},
            "limitations": ["Annotated image-concept agreement; no physical-safety, contact or path claim.", "Incomplete recall treats unresolved outputs as unmatched diagnostics; precision from available outputs is provisional, not a lower bound.", "Unscored phrases are not confirmed hallucinations.", "Reference-present SAM is a diagnostic, not a guaranteed upper bound.", "Combined latency/memory require an actual combined replay; no independent peaks or p95 values are summed."] + ([] if ctx["input_identity_verified"] else ["Dataset fingerprint checked as declared identity only; bind a supported recomputation recipe or pin expected_input_signature before a real final comparison."])}
