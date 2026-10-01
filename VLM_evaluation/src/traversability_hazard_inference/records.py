"""Strict phrase parsing and the eight-field hazard prediction schema."""

from .configuration import TASK_ID, SCHEMA_VERSION, MODEL_SPECS
from .storage import strict_loads

PREDICTION_KEYS = frozenset(("task_id", "schema_version", "frame_id", "model_key",
                            "prompts", "raw_response", "status", "error_code"))


class InferenceError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def normalized_phrase(phrase):
    return " ".join(phrase.split()).casefold()


def _phrases(values):
    if not isinstance(values, list):
        raise InferenceError("invalid_prompts_type", "prompts must be an array")
    if len(values) > 32:
        raise InferenceError("too_many_prompts", "More than 32 original phrases")
    result, seen = [], set()
    for phrase in values:
        if not isinstance(phrase, str):
            raise InferenceError("invalid_phrase_type", "Each phrase must be a string")
        if not normalized_phrase(phrase):
            raise InferenceError("empty_phrase", "Empty or whitespace-only phrase")
        try:
            phrase.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise InferenceError("invalid_phrase_unicode", "Phrase contains an invalid Unicode scalar") from exc
        if len(phrase) > 80:
            raise InferenceError("phrase_too_long", "Phrase exceeds 80 characters")
        normalized = normalized_phrase(phrase)
        if normalized not in seen:
            result.append(phrase)
            seen.add(normalized)
    return result


def phrase_diagnostics(prompts, policy):
    aliases = {normalized_phrase(p) for values in policy["aliases"].values() for p in values}
    vague = {normalized_phrase(p) for p in policy["vague_unusable_phrases"]}
    return {
        "vague_unusable": [p for p in prompts if normalized_phrase(p) in vague],
        "unmapped": [p for p in prompts if normalized_phrase(p) not in aliases | vague],
    }


def parse_response(raw, policy):
    if not isinstance(raw, str):
        raise InferenceError("invalid_response_type", "Decoded response must be text")
    try:
        raw.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise InferenceError("invalid_response_unicode", "Decoded response contains invalid Unicode") from exc
    try:
        value = strict_loads(raw)
    except (ValueError, RecursionError) as exc:
        raise InferenceError("invalid_json", f"Response is not strict JSON: {exc}") from exc
    if not isinstance(value, dict) or set(value) != {"prompts"}:
        raise InferenceError("invalid_response_schema", "Expected exactly one object with only prompts")
    prompts = _phrases(value["prompts"])
    diagnostics = phrase_diagnostics(prompts, policy)
    diagnostics["duplicate_count"] = len(value["prompts"]) - len(prompts)
    return {"prompts": prompts, "diagnostics": diagnostics}


def prediction(frame, model_key, *, prompts=None, raw_response="", error_code=None):
    record = {
        "task_id": TASK_ID, "schema_version": SCHEMA_VERSION,
        "frame_id": frame["frame_id"], "model_key": model_key,
        "prompts": [] if error_code is not None else list(prompts or []),
        "raw_response": raw_response, "status": "error" if error_code is not None else "ok",
        "error_code": error_code,
    }
    validate_prediction(record, model_key=model_key, frame_id=frame["frame_id"])
    return record


def validate_prediction(record, model_key=None, frame_id=None):
    if not isinstance(record, dict) or set(record) != PREDICTION_KEYS:
        raise ValueError("Prediction must have exactly the eight hazard schema fields")
    if record["task_id"] != TASK_ID or type(record["schema_version"]) is not int or record["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Prediction task/schema mismatch; legacy outputs are not hazard predictions")
    if not isinstance(record["frame_id"], str) or not record["frame_id"].strip():
        raise ValueError("Prediction frame_id must be nonempty")
    if record["model_key"] not in MODEL_SPECS:
        raise ValueError("Prediction has an unknown model_key")
    if model_key is not None and record["model_key"] != model_key:
        raise ValueError("Prediction model identity mismatch")
    if frame_id is not None and record["frame_id"] != frame_id:
        raise ValueError("Prediction frame identity mismatch")
    if not isinstance(record["raw_response"], str):
        raise ValueError("raw_response must preserve decoded text")
    parsed = _phrases(record["prompts"])
    if parsed != record["prompts"]:
        raise ValueError("Saved prompts contain normalized duplicates")
    if record["status"] == "ok":
        if record["error_code"] is not None:
            raise ValueError("Successful prediction must have null error_code")
    elif record["status"] == "error":
        if not isinstance(record["error_code"], str) or not record["error_code"].strip():
            raise ValueError("Failed prediction requires a diagnostic error_code")
        if record["prompts"]:
            raise ValueError("Failed predictions cannot expose usable prompts")
    else:
        raise ValueError("Prediction status must be ok or error")
    return record
