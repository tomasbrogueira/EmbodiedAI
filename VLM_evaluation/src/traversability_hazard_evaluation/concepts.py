"""Image-concept counts from independently validated saved hazard artifacts.

No phrase expansion, linguistic judge, instance counting, or sibling runtime
package is involved. Incomplete outputs retain annotated-positive denominators;
the enclosing evaluator decides whether a comparison is ready.
"""

from collections import Counter

from .artifacts import normalize, parse_json, rate


_PHRASE_COUNTS = (
    "raw_phrase_count", "saved_phrase_count", "normalized_phrase_count",
    "normalization_duplicate_count", "mapped_concept_duplicate_count",
    "source_unscored_phrase_count", "unknown_phrase_count",
    "vague_unusable_phrase_count", "unannotated_phrase_count",
)


def _vocabulary(policy, source):
    dataset = policy["datasets"].get(source, {})
    if "hazard_label_ids" in dataset:
        return set(dataset["hazard_label_ids"])
    return set(dataset.get("scored_category_names", []))


def _reference_state(frame_id, references, valid):
    reference = references.get(frame_id)
    if reference is None:
        return "missing"
    if reference.get("status") == "pending":
        return "pending"
    if frame_id in valid and reference.get("status") == "complete":
        return "complete"
    return "invalid"


def _prediction_state(frame_id, predictions, invalid):
    if frame_id in invalid:
        return "invalid"
    prediction = predictions.get(frame_id)
    if prediction is None:
        return "missing"
    return "error" if prediction["status"] == "error" else "ok"


def _phrase_audit(model, frame, reference, reference_state, prediction,
                  prediction_state, vocabulary, aliases, vague, audited_raw_phrases=None):
    saved_phrases = list(prediction.get("prompts", [])) if prediction else []
    raw_phrases = saved_phrases
    if prediction_state == "ok" and audited_raw_phrases is not None:
        raw_phrases = list(audited_raw_phrases)
    elif prediction_state == "ok" and prediction:
        # The inference writer may deduplicate exact normalized repeats. Audit
        # their original occurrences from the already validated raw response.
        try:
            original = parse_json(prediction["raw_response"])["prompts"]
            if isinstance(original, list) and all(isinstance(phrase, str) for phrase in original):
                raw_phrases = list(original)
        except (ValueError, TypeError, KeyError):
            # Validation belongs to the reader. This internal fallback also
            # permits direct isolated scoring checks with saved-only records.
            pass
    present = set(reference["present_concepts"]) if reference_state == "complete" else set()
    absence_eligible = reference_state == "complete" and reference["absence_scoring_eligible"]
    seen_phrases, seen_concepts, concepts = set(), set(), set()
    entries, unscored = [], []
    counts = Counter(dict.fromkeys(_PHRASE_COUNTS, 0))
    counts["raw_phrase_count"] = len(raw_phrases)
    counts["saved_phrase_count"] = len(saved_phrases)
    for index, phrase in enumerate(raw_phrases):
        normalized = normalize(phrase)
        duplicate = normalized in seen_phrases
        canonical = aliases.get(normalized)
        is_vague = normalized in vague
        category, reason = "source_concept", None
        if is_vague:
            canonical, category, reason = None, "vague_unusable", "frozen_vague_phrase"
        elif canonical is None:
            category, reason = "unknown_unscored", "outside_frozen_alias_table"
        elif canonical not in vocabulary:
            category, reason = "source_unscored", "outside_source_scored_vocabulary"
        elif reference_state != "complete":
            category, reason = "unannotated_unscored", f"reference_{reference_state}"
        elif canonical not in present and not absence_eligible:
            category, reason = "unannotated_unscored", "absence_scoring_ineligible"
        alias_duplicate = not duplicate and canonical is not None and canonical in seen_concepts
        entries.append({
            "index": index, "raw_phrase": phrase, "normalized_phrase": normalized,
            "canonical_concept": canonical, "category": category,
            "unscored_reason": reason, "normalization_duplicate": duplicate,
            "mapped_concept_duplicate": alias_duplicate,
        })
        if duplicate:
            counts["normalization_duplicate_count"] += 1
            continue
        seen_phrases.add(normalized)
        counts["normalized_phrase_count"] += 1
        if alias_duplicate:
            counts["mapped_concept_duplicate_count"] += 1
        if canonical is not None:
            seen_concepts.add(canonical)
            if prediction_state == "ok":
                concepts.add(canonical)
        count_field = {
            "vague_unusable": "vague_unusable_phrase_count",
            "unknown_unscored": "unknown_phrase_count",
            "source_unscored": "source_unscored_phrase_count",
            "unannotated_unscored": "unannotated_phrase_count",
        }.get(category)
        if count_field:
            counts[count_field] += 1
        if category in ("unknown_unscored", "source_unscored", "unannotated_unscored"):
            unscored.append({"raw_phrase": phrase, "normalized_phrase": normalized,
                             "canonical_concept": canonical, "reason": reason})
    audit = {
        "model_key": model, "frame_id": frame["frame_id"],
        "source": frame["source"], "split": frame["split"],
        "prediction_status": prediction_state, "reference_status": reference_state,
        "raw_response": prediction.get("raw_response") if prediction else None,
        "raw_phrases": raw_phrases, "saved_prompts": saved_phrases, "phrases": entries,
        "predicted_concepts": sorted(concepts),
        "source_scored_concepts": sorted(concepts & vocabulary),
        "unscored_concrete_phrases": unscored,
        **dict(counts),
    }
    return audit, concepts, counts


def _row(model, source, split, frame_ids, records, vocabulary, tiny_threshold):
    counts = Counter(dict.fromkeys(_PHRASE_COUNTS, 0))
    concepts = {concept: Counter() for concept in vocabulary}
    for frame_id in frame_ids:
        record = records[frame_id]
        counts["requested_frames"] += 1
        prediction_state, reference_state = record["prediction_state"], record["reference_state"]
        if prediction_state in ("ok", "error"):
            counts["prediction_records"] += 1
        counts[{
            "ok": "successful_responses", "error": "prediction_failures",
            "missing": "missing_predictions", "invalid": "invalid_predictions",
        }[prediction_state]] += 1
        if prediction_state == "ok" and not record["prediction"]["prompts"]:
            counts["successful_empty_lists"] += 1
        counts[f"reference_{reference_state}"] += 1
        counts.update(record["phrase_counts"])
        if reference_state != "complete":
            continue
        reference, frame = record["reference"], record["frame"]
        present, predicted = set(reference["present_concepts"]), record["predicted"] & record["vocabulary"]
        counts["reference_positive_pairs"] += len(present)
        counts["recall_tp"] += len(present & predicted)
        counts["recall_fn"] += len(present - predicted)
        eligible = reference["absence_scoring_eligible"]
        if eligible:
            counts["absence_eligible_frames"] += 1
            counts["precision_tp"] += len(present & predicted)
            counts["scored_fp"] += len(predicted - present)
            counts["scored_prediction_pairs"] += len(predicted)
        for concept in record["vocabulary"]:
            concept_counts = concepts.setdefault(concept, Counter())
            if concept in present:
                matched = concept in predicted
                concept_counts["reference_positive_pairs"] += 1
                concept_counts["recall_tp" if matched else "recall_fn"] += 1
                area = reference["concept_pixel_counts"][concept] / (frame["width"] * frame["height"])
                if 0 < area < tiny_threshold:
                    counts["tiny_positive_pairs"] += 1
                    counts["tiny_tp" if matched else "tiny_fn"] += 1
                    concept_counts["tiny_positive_pairs"] += 1
                    concept_counts["tiny_tp" if matched else "tiny_fn"] += 1
            if eligible and concept in predicted:
                concept_counts["precision_tp" if concept in present else "scored_fp"] += 1
                concept_counts["scored_prediction_pairs"] += 1
    fields = (
        "requested_frames", "prediction_records", "successful_responses",
        "successful_empty_lists", "prediction_failures", "missing_predictions",
        "invalid_predictions", "reference_complete", "reference_pending",
        "reference_missing", "reference_invalid", "absence_eligible_frames",
        "reference_positive_pairs", "recall_tp", "recall_fn", "precision_tp",
        "scored_fp", "scored_prediction_pairs", "tiny_positive_pairs", "tiny_tp", "tiny_fn",
    )
    complete = not counts["missing_predictions"] and not counts["invalid_predictions"]
    annotated = not counts["reference_pending"] and not counts["reference_missing"] and not counts["reference_invalid"]
    identity = {"model_key": model, "source": source, "split": split}
    row = {
        **identity, **{field: counts[field] for field in (*fields, *_PHRASE_COUNTS)},
        "status": "complete" if complete and annotated else "partial",
        "predictions_complete": complete, "references_complete": annotated,
        "counts_are_provisional": not (complete and annotated),
        "metrics": {
            "concept_recall": rate(counts["recall_tp"], counts["reference_positive_pairs"]),
            "supported_precision": rate(counts["precision_tp"], counts["scored_prediction_pairs"]),
            "tiny_concept_recall": rate(counts["tiny_tp"], counts["tiny_positive_pairs"]),
            "failure_rate": rate(counts["prediction_failures"], counts["prediction_records"]),
        },
    }
    per_concept = []
    concept_fields = ("reference_positive_pairs", "recall_tp", "recall_fn", "precision_tp",
                      "scored_fp", "scored_prediction_pairs", "tiny_positive_pairs", "tiny_tp", "tiny_fn")
    for concept in sorted(concepts):
        item = concepts[concept]
        per_concept.append({
            **identity, "concept": concept, "counts_are_provisional": row["counts_are_provisional"],
            **{field: item[field] for field in concept_fields},
            "concept_recall": rate(item["recall_tp"], item["reference_positive_pairs"]),
            "supported_precision": rate(item["precision_tp"], item["scored_prediction_pairs"]),
            "tiny_concept_recall": rate(item["tiny_tp"], item["tiny_positive_pairs"]),
        })
    return row, per_concept


def evaluate_concepts(ctx):
    """Count known-positive discovery and annotation-supported precision.

    ``ctx`` contains independently validated records and the frozen policy.
    Returned ``complete`` concerns saved prediction coverage only; reference,
    fixture, identity, cost and final-comparison readiness belong to the caller.
    """
    frames, references = ctx["frames"], ctx["references"]
    requested = set(ctx["requested_ids"])
    known_requested = sorted(requested & frames.keys())
    policy, aliases, vague = ctx["policy"], ctx["aliases"], ctx["vague"]
    tiny_threshold = policy["tiny_concept_area_fraction"]
    rows, per_concept, missed, audits, failures, coverage = [], [], [], [], [], []
    for model in ctx["config"]["model_keys"]:
        predictions = ctx["predictions"].get(model, {})
        invalid = set(ctx["prediction_invalid"].get(model, set()))
        records = {}
        states = {frame_id: _prediction_state(frame_id, predictions, invalid) for frame_id in requested}
        missing_ids = sorted(frame_id for frame_id, state in states.items() if state == "missing")
        invalid_ids = sorted(frame_id for frame_id, state in states.items() if state == "invalid")
        failure_ids = sorted(frame_id for frame_id, state in states.items() if state == "error")
        complete = bool(requested) and not missing_ids and not invalid_ids and requested <= frames.keys()
        coverage.append({
            "model_key": model, "requested": len(requested),
            "saved_valid_records": sum(state in ("ok", "error") for state in states.values()),
            "successful": sum(state == "ok" for state in states.values()),
            "failed": len(failure_ids), "missing": len(missing_ids), "invalid": len(invalid_ids),
            "requested_ids": sorted(requested), "missing_ids": missing_ids,
            "invalid_ids": invalid_ids, "failed_ids": failure_ids,
            "complete": complete,
            "record_coverage": rate(sum(state in ("ok", "error") for state in states.values()), len(requested)),
        })
        for frame_id in known_requested:
            frame, reference = frames[frame_id], references.get(frame_id)
            prediction_state, prediction = states[frame_id], predictions.get(frame_id)
            reference_state = _reference_state(frame_id, references, ctx["reference_valid"])
            vocabulary = _vocabulary(policy, frame["source"])
            audit, predicted, phrase_counts = _phrase_audit(
                model, frame, reference, reference_state, prediction,
                prediction_state, vocabulary, aliases, vague,
                ctx.get("prediction_raw_prompts", {}).get(model, {}).get(frame_id))
            audits.append(audit)
            records[frame_id] = {
                "frame": frame, "reference": reference, "prediction": prediction,
                "reference_state": reference_state, "prediction_state": prediction_state,
                "vocabulary": vocabulary, "predicted": predicted, "phrase_counts": phrase_counts,
            }
            if prediction_state in ("missing", "invalid", "error"):
                failures.append({
                    "model_key": model, "frame_id": frame_id,
                    "source": frame["source"], "split": frame["split"],
                    "kind": {"missing": "missing_record", "invalid": "invalid_record", "error": "saved_response_error"}[prediction_state],
                    "error_code": prediction.get("error_code") if prediction else None,
                    "raw_response": prediction.get("raw_response") if prediction else None,
                })
            if reference_state == "complete":
                for concept in sorted(set(reference["present_concepts"]) - predicted):
                    pixels = reference["concept_pixel_counts"][concept]
                    area = pixels / (frame["width"] * frame["height"])
                    missed.append({
                        "model_key": model, "frame_id": frame_id,
                        "source": frame["source"], "split": frame["split"],
                        "image_path": frame["image_path"], "concept": concept,
                        "pixel_count": pixels, "area_fraction": area,
                        "tiny": 0 < area < tiny_threshold, "prediction_status": prediction_state,
                        "reason": "not_identified" if prediction_state == "ok" else {
                            "missing": "missing_prediction", "invalid": "invalid_prediction", "error": "response_error",
                        }[prediction_state],
                    })
        for split in ("development", "test"):
            split_ids = [frame_id for frame_id in known_requested if frames[frame_id]["split"] == split]
            if not split_ids:
                continue
            for source in sorted({frames[frame_id]["source"] for frame_id in split_ids}) + ["all"]:
                group = split_ids if source == "all" else [frame_id for frame_id in split_ids if frames[frame_id]["source"] == source]
                vocabulary = set().union(*(records[frame_id]["vocabulary"] for frame_id in group))
                row, concept_rows = _row(model, source, split, group, records, vocabulary, tiny_threshold)
                rows.append(row)
                per_concept.extend(concept_rows)
    return {
        "rows": rows, "per_concept": per_concept, "missed_concepts": missed,
        "phrase_audit": audits, "failures": failures, "coverage": coverage,
        "complete": bool(coverage) and all(item["complete"] for item in coverage),
    }
