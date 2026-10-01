"""Development-only CLIP rejection calibration and provenance validation."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

from .clip import CLIPBackend, MODEL_KEY, _normalized_clip_settings, scoring_identity, semantic_vocabulary
from .records import InferenceError
from .storage import atomic_json, input_identity, read_jsonl, read_manifests, stable_fingerprint, validate_manifests


OBJECTIVE = "maximize_useful_acceptance_subject_to_zero_observed_unsafe_acceptance_ties_higher_threshold"


def _artifact_payload(artifact):
    return {key: value for key, value in artifact.items() if key != "artifact_fingerprint"}


def load_calibration(path, expected_identity):
    """Validate a calibration without reading any annotations or references."""
    def unique_pairs(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate calibration key {key!r}")
            output[key] = value
        return output

    try:
        artifact = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)
    except FileNotFoundError as exc:
        raise InferenceError("clip_calibration_missing", f"Calibration file does not exist: {path}") from exc
    except (ValueError, TypeError, OSError, UnicodeError) as exc:
        raise InferenceError("clip_calibration_invalid", str(exc)) from exc
    try:
        if not isinstance(artifact, dict) or artifact.get("schema_version") != 1 or artifact.get("model_key") != MODEL_KEY:
            raise ValueError("unsupported calibration schema or model")
        if artifact.get("artifact_fingerprint") != stable_fingerprint(_artifact_payload(artifact)):
            raise ValueError("calibration contents do not match their fingerprint")
        if artifact.get("compatibility_fingerprint") != stable_fingerprint(expected_identity):
            raise InferenceError("clip_calibration_incompatible", "Calibration scoring configuration differs from this backend")
        if artifact.get("scoring_identity") != expected_identity or artifact.get("objective") != OBJECTIVE:
            raise ValueError("calibration identity or objective is invalid")
        if artifact.get("threshold_comparison") != "cosine_similarity >= threshold":
            raise ValueError("calibration threshold comparison is invalid")
        if type(artifact.get("reject_all")) is not bool:
            raise ValueError("reject_all must be a boolean")
        threshold = artifact.get("threshold")
        if artifact["reject_all"]:
            if threshold is not None:
                raise ValueError("reject-all calibration must have a null threshold")
        elif type(threshold) not in {int, float} or not math.isfinite(threshold) or not -1 <= threshold <= 1:
            raise ValueError("threshold must be a finite cosine score")
        for key in ("development_frames", "development_regions", "development_region_ids"):
            if not isinstance(artifact.get(key), list):
                raise ValueError(f"{key} must be a list")
        if any(frame.get("split") != "development" for frame in artifact["development_frames"]):
            raise ValueError("calibration contains non-development frames")
        validate_manifests(artifact["development_frames"], artifact["development_regions"])
        region_ids = [row["region_id"] for row in artifact["development_regions"]]
        if artifact["development_region_ids"] != sorted(region_ids):
            raise ValueError("development IDs disagree with frozen region records")
        counts = artifact.get("observed_counts", {})
        required_counts = {
            "eligible_development_regions", "eligible_reference_traversable", "eligible_reference_non_traversable",
            "successful_reference_traversable", "successful_reference_non_traversable",
            "scoring_failures", "useful_acceptances", "unsafe_acceptances",
            "useful_acceptance_denominator", "unsafe_acceptance_denominator",
        }
        if not isinstance(counts, dict) or set(counts) != required_counts or any(type(value) is not int or value < 0 for value in counts.values()):
            raise ValueError("observed calibration counts must be nonnegative integers")
        if counts.get("successful_reference_traversable", 0) <= 0 or counts.get("successful_reference_non_traversable", 0) <= 0:
            raise ValueError("calibration needs successful examples from both reference classes")
        if counts.get("unsafe_acceptances") != 0:
            raise ValueError("calibration has observed unsafe acceptance")
        if counts["eligible_development_regions"] != len(region_ids) or counts["eligible_reference_traversable"] + counts["eligible_reference_non_traversable"] != len(region_ids):
            raise ValueError("eligible counts disagree with frozen development records")
        if counts["useful_acceptance_denominator"] != counts["eligible_reference_traversable"] or counts["unsafe_acceptance_denominator"] != counts["eligible_reference_non_traversable"]:
            raise ValueError("calibration acceptance denominators disagree with eligible classes")
        successful = artifact.get("successful_development_region_ids")
        if not isinstance(successful, list) or len(set(successful)) != len(successful) or not set(successful).issubset(region_ids):
            raise ValueError("successful development IDs are invalid")
        if counts["successful_reference_traversable"] + counts["successful_reference_non_traversable"] != len(successful) or len(successful) + counts["scoring_failures"] != len(region_ids):
            raise ValueError("successful and failed scoring counts are inconsistent")
        if counts["useful_acceptances"] > counts["successful_reference_traversable"] or (artifact["reject_all"] and counts["useful_acceptances"] != 0) or (not artifact["reject_all"] and counts["useful_acceptances"] == 0):
            raise ValueError("useful acceptance counts contradict the selected threshold")
    except InferenceError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise InferenceError("clip_calibration_invalid", str(exc)) from exc
    return artifact


def validate_calibration_inputs(artifact, frames, regions, data_root):
    """Recheck frozen development inputs; annotation files are never consulted."""
    frame_by_id = {row["frame_id"]: row for row in frames}
    region_by_id = {row["region_id"]: row for row in regions}
    development_frames = artifact["development_frames"]
    development_regions = artifact["development_regions"]
    for row in development_frames:
        if frame_by_id.get(row["frame_id"]) != row:
            raise InferenceError("clip_calibration_incompatible", "Development frame manifests have changed")
    for row in development_regions:
        if region_by_id.get(row["region_id"]) != row:
            raise InferenceError("clip_calibration_incompatible", "Development region manifests have changed")
    actual = input_identity(development_frames, development_regions, data_root)
    if stable_fingerprint(actual) != artifact["development_fingerprint"]:
        raise InferenceError("clip_calibration_incompatible", "Development image or mask contents have changed")


def _eligible_development(run_dir, frames, regions, robot_profile_id):
    frame_by_id = {row["frame_id"]: row for row in frames}
    development = {
        row["region_id"]: row for row in regions
        if frame_by_id[row["frame_id"]]["split"] == "development" and row.get("selected_for_classification") is True
    }
    annotations = {}
    for row in read_jsonl(Path(run_dir) / "annotations.jsonl"):
        region_id = row.get("region_id")
        # Test references, mask fractions and reference-transfer artifacts never
        # affect eligibility, validation, scoring, thresholds or provenance.
        if region_id not in development:
            continue
        if region_id in annotations:
            raise InferenceError("duplicate_annotation_id", str(region_id))
        annotations[region_id] = row
    eligible = []
    labels = {}
    for region_id in sorted(development):
        annotation = annotations.get(region_id, {})
        if annotation.get("annotation_status") != "complete" or annotation.get("mask_quality") != "valid" or annotation.get("robot_profile_id") != robot_profile_id:
            continue
        label = annotation.get("reference_label")
        if label not in {"traversable", "non_traversable"}:
            continue
        eligible.append(development[region_id])
        labels[region_id] = label
    eligible_frame_ids = {row["frame_id"] for row in eligible}
    selected_frames = [frame_by_id[key] for key in sorted(eligible_frame_ids)]
    return selected_frames, eligible, labels


def _choose_threshold(examples):
    """Inclusive threshold; exactly tied unsafe/useful scores cannot be accepted."""
    traversable = [item for item in examples if item["candidate_label"] == "traversable"]
    candidates = sorted({item["similarity"] for item in traversable})
    best = None
    for threshold in candidates:
        accepted = [item for item in traversable if item["similarity"] >= threshold]
        unsafe = sum(item["reference_label"] == "non_traversable" for item in accepted)
        useful = sum(item["reference_label"] == "traversable" for item in accepted)
        if unsafe == 0 and useful > 0 and (best is None or (useful, threshold) > (best[0], best[1])):
            best = useful, threshold
    return (None, True, 0) if best is None else (best[1], False, best[0])


def calibrate_clip(run_dir, settings, data_root, *, backend=None):
    """Save a development-only threshold; return its immutable provenance data.

    An injected backend must provide ``score_frame`` diagnostic records with
    region/frame IDs, semantic_class, similarity, status and error_code. Its
    calibration is identified as a fixture and is unusable by real CLIP.
    """
    settings = _normalized_clip_settings(settings)
    frames, regions = read_manifests(run_dir)
    development_frames, development_regions, labels = _eligible_development(run_dir, frames, regions, settings["robot_profile_id"])
    if not {"traversable", "non_traversable"}.issubset(set(labels.values())):
        raise InferenceError("clip_calibration_insufficient_classes", "Need completed valid development examples from both reference classes matching the robot profile")
    frozen_settings = deepcopy(settings)
    # Capture input contents before loading, which can itself take time. Never
    # publish scores fitted against one dataset with a later dataset identity.
    frozen_inputs = deepcopy(input_identity(development_frames, development_regions, data_root))
    owned = backend is None
    if owned:
        backend = CLIPBackend(MODEL_KEY, settings, _for_calibration=True)
    injected = not owned
    vocabulary = {row["semantic_class"]: row["label"] for row in semantic_vocabulary(settings["robot_profile_id"])}
    examples, failures = [], []
    try:
        identity = deepcopy(scoring_identity(
            settings, getattr(backend, "metadata", {}), injected=injected,
            backend_type=f"{type(backend).__module__}.{type(backend).__qualname__}",
        ))
        frozen_backend_settings = deepcopy(getattr(backend, "settings", None))
        for frame in development_frames:
            requested = [row for row in development_regions if row["frame_id"] == frame["frame_id"]]
            try:
                scores = backend.score_frame(deepcopy(frame), deepcopy(requested), data_root)
            except Exception as exc:
                failures.extend({"region_id": row["region_id"], "error_code": "inference_error", "diagnostic": f"{type(exc).__name__}: {exc}"} for row in requested)
                continue
            if not isinstance(scores, list):
                raise InferenceError("clip_calibration_invalid_scores", "score_frame must return a list")
            by_id = {}
            requested_ids = {row["region_id"] for row in requested}
            for score in scores:
                if not isinstance(score, dict) or score.get("region_id") not in requested_ids or score["region_id"] in by_id:
                    raise InferenceError("clip_calibration_invalid_scores", "Duplicate, unexpected or malformed scoring record")
                by_id[score["region_id"]] = score
            for region in requested:
                region_id = region["region_id"]
                score = by_id.get(region_id)
                if score is None or score.get("status") != "ok":
                    failures.append({"region_id": region_id, "error_code": (score or {}).get("error_code") or "missing_score"})
                    continue
                similarity = score.get("similarity")
                semantic_class = score.get("semantic_class")
                if score.get("frame_id") != frame["frame_id"] or score.get("error_code") is not None or semantic_class not in vocabulary or type(similarity) not in {int, float} or not math.isfinite(similarity) or not -1 <= similarity <= 1:
                    raise InferenceError("clip_calibration_invalid_scores", f"Invalid successful score for {region_id}")
                candidate_label = vocabulary[semantic_class]
                if score.get("candidate_label", candidate_label) != candidate_label:
                    raise InferenceError("clip_calibration_invalid_scores", "Scoring label contradicts the frozen semantic vocabulary")
                examples.append({"region_id": region_id, "reference_label": labels[region_id], "candidate_label": candidate_label, "similarity": similarity})
        live_frames, live_regions = read_manifests(run_dir)
        live_frame_by_id = {row["frame_id"]: row for row in live_frames}
        live_region_by_id = {row["region_id"]: row for row in live_regions}
        records_changed = (
            any(live_frame_by_id.get(row["frame_id"]) != row for row in development_frames)
            or any(live_region_by_id.get(row["region_id"]) != row for row in development_regions)
        )
        if records_changed or input_identity(development_frames, development_regions, data_root) != frozen_inputs:
            raise InferenceError("clip_calibration_inputs_changed", "Development image or mask contents changed during calibration; no calibration was published")
        actual_identity = scoring_identity(
            settings, getattr(backend, "metadata", {}), injected=injected,
            backend_type=f"{type(backend).__module__}.{type(backend).__qualname__}",
        )
        if settings != frozen_settings or getattr(backend, "settings", None) != frozen_backend_settings or actual_identity != identity:
            raise InferenceError("clip_calibration_configuration_changed", "Scoring configuration changed during calibration; no calibration was published")
        class_counts = {label: sum(row["reference_label"] == label for row in examples) for label in ("traversable", "non_traversable")}
        if not all(class_counts.values()):
            raise InferenceError("clip_calibration_insufficient_successes", "Need successful development scoring examples from both reference classes")
        threshold, reject_all, useful = _choose_threshold(examples)
        artifact = {
            "schema_version": 1,
            "model_key": MODEL_KEY,
            "objective": OBJECTIVE,
            "threshold": threshold,
            "threshold_comparison": "cosine_similarity >= threshold",
            "reject_all": reject_all,
            "scoring_identity": identity,
            "compatibility_fingerprint": stable_fingerprint(identity),
            "development_frames": development_frames,
            "development_regions": development_regions,
            "development_region_ids": [row["region_id"] for row in development_regions],
            "successful_development_region_ids": [row["region_id"] for row in examples],
            "development_fingerprint": stable_fingerprint(frozen_inputs),
            "observed_counts": {
                "eligible_development_regions": len(development_regions),
                "eligible_reference_traversable": sum(label == "traversable" for label in labels.values()),
                "eligible_reference_non_traversable": sum(label == "non_traversable" for label in labels.values()),
                "successful_reference_traversable": class_counts["traversable"],
                "successful_reference_non_traversable": class_counts["non_traversable"],
                "scoring_failures": len(failures),
                "useful_acceptances": useful,
                "unsafe_acceptances": 0,
                "useful_acceptance_denominator": sum(label == "traversable" for label in labels.values()),
                "unsafe_acceptance_denominator": sum(label == "non_traversable" for label in labels.values()),
            },
            "scoring_failures": failures,
            "provenance": "completed_valid_development_annotations_matching_robot_profile",
            "safety_scope": "zero_observed_development_unsafe_acceptance; no held_out_safety_guarantee",
        }
        artifact["artifact_fingerprint"] = stable_fingerprint(artifact)
        destination = Path(settings.get("calibration_path") or Path(run_dir) / "metadata" / "clip_calibration.json")
        atomic_json(destination, artifact)
        return artifact
    finally:
        if owned:
            backend.close()
