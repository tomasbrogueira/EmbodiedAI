"""Strict generative parsing and the fixed prediction record."""

import json

LABELS = {"traversable", "non_traversable", "unknown"}
PREDICTION_KEYS = {
    "region_id", "frame_id", "model_key", "label", "semantic_class", "reason",
    "raw_response", "status", "error_code",
}


class InferenceError(Exception):
    def __init__(self, code, message, *, raw_response=""):
        super().__init__(message)
        self.code = code
        self.raw_response = raw_response


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InferenceError("parse_duplicate_key", f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_response(raw):
    """Parse precisely one JSON object without repairing generated text."""
    if not isinstance(raw, str):
        raise InferenceError("parse_type", "Model response must be text")
    try:
        obj = json.loads(raw, object_pairs_hook=_object,
                         parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except InferenceError:
        raise
    except (ValueError, TypeError) as error:
        raise InferenceError("parse_json", str(error)) from error
    if not isinstance(obj, dict) or set(obj) != {"label", "semantic_class", "reason"}:
        raise InferenceError("parse_keys", "Expected exactly label, semantic_class and reason")
    if not isinstance(obj["label"], str) or obj["label"] not in LABELS:
        raise InferenceError("parse_label", "Invalid label")
    if obj["semantic_class"] is not None and not isinstance(obj["semantic_class"], str):
        raise InferenceError("parse_type", "semantic_class must be text or null")
    if not isinstance(obj["reason"], str):
        raise InferenceError("parse_type", "reason must be text")
    return obj


def prediction(region, model_key, parsed=None, raw_response="", error_code=None, reason=""):
    result = {
        "region_id": region["region_id"], "frame_id": region["frame_id"],
        "model_key": model_key, "label": "unknown", "semantic_class": None,
        "reason": reason, "raw_response": raw_response,
        "status": "error" if error_code else "ok", "error_code": error_code,
    }
    if parsed is not None and not error_code:
        result.update({key: parsed[key] for key in ("label", "semantic_class", "reason")})
    return validate_prediction(result, region=region, model_key=model_key)


def validate_prediction(record, *, region=None, model_key=None):
    if not isinstance(record, dict) or set(record) != PREDICTION_KEYS:
        raise ValueError("Prediction must have exactly the fixed schema keys")
    for key in ("region_id", "frame_id", "model_key", "reason", "raw_response"):
        if not isinstance(record[key], str) or (key.endswith("_id") and not record[key]):
            raise ValueError(f"Invalid prediction {key}")
    if not isinstance(record["label"], str) or record["label"] not in LABELS:
        raise ValueError("Invalid prediction label")
    if record["semantic_class"] is not None and not isinstance(record["semantic_class"], str):
        raise ValueError("Invalid semantic_class")
    if record["status"] == "ok":
        if record["error_code"] is not None:
            raise ValueError("Successful prediction must have null error_code")
    elif record["status"] == "error":
        if (record["label"] != "unknown" or record["semantic_class"] is not None
                or not isinstance(record["error_code"], str) or not record["error_code"]):
            raise ValueError("Errors require unknown label, null semantic_class and error_code")
    else:
        raise ValueError("Invalid prediction status")
    if region is not None and any(record[k] != region[k] for k in ("region_id", "frame_id")):
        raise ValueError("Prediction ID join mismatch")
    if model_key is not None and record["model_key"] != model_key:
        raise ValueError("Prediction model mismatch")
    return dict(record)
