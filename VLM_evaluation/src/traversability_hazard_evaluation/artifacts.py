"""Strict, independent CPU readers for the frozen hazard artifacts."""

import hashlib
import json
import math
from pathlib import Path, PureWindowsPath


class ArtifactError(ValueError):
    """An artifact cannot establish valid scoring evidence."""


def normalize(phrase):
    return " ".join(phrase.casefold().split())


def rate(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "rate": numerator / denominator if denominator else None}


def issue(issues, code, message, **context):
    issues.append({"code": code, "message": str(message), **context})


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ArtifactError(f"Nonfinite JSON value: {value}")


def _finite(value):
    if isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, list):
        for item in value:
            _finite(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ArtifactError("Nonfinite JSON number")


def parse_json(text):
    result = json.loads(text, object_pairs_hook=_object, parse_constant=_constant)
    _finite(result)
    if not isinstance(result, dict):
        raise ArtifactError("Expected a JSON object")
    return result


def read_json(path):
    try:
        return parse_json(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as error:
        raise ArtifactError(f"Cannot read {path}: {error}") from error


def read_records(path, id_key, issues, stage):
    """Retain unique rows; never select an arbitrary duplicate's contents."""
    records, invalid, seen = {}, set(), set()
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        issue(issues, "missing_or_unreadable_artifact", error, stage=stage, path=str(path))
        return records, invalid
    for number, line in enumerate(lines, 1):
        try:
            row = parse_json(line)
            identifier = row.get(id_key)
            if not isinstance(identifier, str) or not identifier.strip():
                raise ArtifactError(f"Missing or invalid {id_key}")
            if identifier in seen:
                previous = records.pop(identifier, None)
                invalid.add(identifier)
                issue(issues, "duplicate_id", identifier, stage=stage, path=str(path), line=number,
                      previous_record=previous, raw_record=row)
            else:
                seen.add(identifier)
                records[identifier] = row
        except (ValueError, TypeError) as error:
            issue(issues, "invalid_jsonl", error, stage=stage, path=str(path), line=number, raw_line=line)
    return records, invalid


def fingerprint(value, *, ensure_ascii=False):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=ensure_ascii, allow_nan=False).encode("utf-8", errors="backslashreplace")
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def safe_path(root, relative):
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or "\x00" in relative or ":" in relative or relative.startswith("/")
            or PureWindowsPath(relative).drive
            or any(part in ("", ".", "..") for part in relative.split("/"))):
        raise ArtifactError(f"Expected safe POSIX-relative path: {relative!r}")
    root = Path(root).resolve()
    path = root.joinpath(*relative.split("/")).resolve()
    if not path.is_relative_to(root):
        raise ArtifactError(f"Path escapes configured root: {relative}")
    return path


def binary_mask(root, relative, size):
    import numpy as np
    from PIL import Image
    path = safe_path(root, relative)
    try:
        with Image.open(path) as image:
            image.load()
            if image.format != "PNG" or len(image.getbands()) != 1 or image.mode == "P":
                raise ArtifactError(f"Expected single-channel binary PNG: {relative}")
            if tuple(image.size) != tuple(size):
                raise ArtifactError(f"Mask dimension mismatch: {relative}")
            values = np.asarray(image)
            unique = set(np.unique(values).tolist())
            if not (unique <= {0, 1} or unique <= {0, 255}):
                raise ArtifactError(f"Nonbinary mask: {relative}")
            return values != 0
    except (OSError, ValueError) as error:
        raise ArtifactError(f"Cannot validate mask {relative}: {error}") from error


def pointer(value, selector):
    if not isinstance(selector, str) or not selector.startswith("/"):
        raise ArtifactError(f"Expected JSON pointer: {selector!r}")
    try:
        for part in selector[1:].split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            value = value[int(part)] if isinstance(value, list) else value[part]
        return value
    except (KeyError, TypeError, IndexError, ValueError) as error:
        raise ArtifactError(f"Missing metadata field: {selector}") from error


def binding(config, kind, name, default):
    return config.get("metadata_bindings", {}).get(kind, {}).get(name, default)


def validate_identity(obj, ctx, kind, condition_key=None):
    config = ctx["config"]
    def field(name):
        return pointer(obj, binding(config, kind, name, "/" + name))
    expected = {"task_id": "hazard_prompt_v1", "schema_version": 1,
                "fixture": config["fixture"], "policy_hash": ctx["policy_hash"],
                "alias_hash": ctx["alias_hash"]}
    for key, value in expected.items():
        actual = field(key)
        if actual != value or (key in ("schema_version", "fixture") and type(actual) is not type(value)):
            raise ArtifactError(f"{kind} {key} mismatch")
    for key, expected_value in (("run_name", config["run_name"]), ("policy_id", ctx["policy"]["policy_id"])):
        selector = binding(config, kind, key, "/" + key)
        try:
            actual = pointer(obj, selector)
        except ArtifactError:
            if key in config.get("metadata_bindings", {}).get(kind, {}):
                raise
        else:
            if actual != expected_value:
                raise ArtifactError(f"{kind} declared {key} mismatch")
    if kind != "dataset":
        digest = pointer(ctx["dataset_metadata"], binding(config, "dataset", "dataset_fingerprint", "/dataset_fingerprint"))
        if field("dataset_fingerprint") != digest:
            raise ArtifactError(f"{kind} dataset fingerprint mismatch")
    if condition_key is not None and field("condition_key") != condition_key:
        raise ArtifactError(f"{kind} condition identity mismatch")
    return obj
