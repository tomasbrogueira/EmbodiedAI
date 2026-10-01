"""Independent reads and validation of the fixed on-disk contracts."""

import hashlib
import json
import math
from pathlib import Path, PureWindowsPath

from .metrics import LABELS


class ArtifactError(ValueError):
    """An artifact cannot safely contribute scored results."""


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
        raise ArtifactError("JSON number overflows to a nonfinite value")


def read_json(path):
    try:
        result = json.loads(Path(path).read_text(encoding="utf-8"),
                            object_pairs_hook=_object, parse_constant=_constant)
        _finite(result)
    except (OSError, ValueError, UnicodeError) as error:
        raise ArtifactError(f"Cannot read JSON {path}: {error}") from error
    if not isinstance(result, dict):
        raise ArtifactError(f"Expected JSON object: {path}")
    return result


def read_jsonl(path, id_key):
    result = {}
    try:
        with Path(path).open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                try:
                    row = json.loads(line, object_pairs_hook=_object,
                                     parse_constant=_constant)
                    _finite(row)
                    if not isinstance(row, dict):
                        raise ArtifactError("Expected JSON object")
                    identifier = row.get(id_key)
                    if not isinstance(identifier, str) or not identifier.strip():
                        raise ArtifactError(f"Missing {id_key}")
                    if identifier in result:
                        raise ArtifactError(f"Duplicate {id_key}: {identifier}")
                    result[identifier] = row
                except (ValueError, TypeError) as error:
                    raise ArtifactError(f"{path}:{number}: {error}") from error
    except (OSError, UnicodeError) as error:
        raise ArtifactError(f"Cannot read JSONL {path}: {error}") from error
    return result


def fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def data_path(root, relative):
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or "\x00" in relative or ":" in relative or relative.startswith("/")
            or PureWindowsPath(relative).drive
            or any(part in ("", ".", "..") for part in relative.split("/"))):
        raise ArtifactError(f"Expected safe POSIX relative data path: {relative!r}")
    root = Path(root).resolve()
    path = root.joinpath(*relative.split("/")).resolve()
    if not path.is_relative_to(root):
        raise ArtifactError(f"Data path escapes configured root: {relative}")
    return path


def _fields(row, required, kind):
    missing = set(required) - row.keys()
    if missing:
        raise ArtifactError(f"{kind} missing fields: {sorted(missing)}")


def _text(row, key, kind):
    if not isinstance(row.get(key), str) or not row[key].strip():
        raise ArtifactError(f"{kind}.{key} must be a nonempty string")


def _nullable_text(row, key, kind):
    if row[key] is not None and not isinstance(row[key], str):
        raise ArtifactError(f"{kind}.{key} must be string or null")


def validate_records(frames, regions, annotations, root, policy, config):
    groups = {}
    for frame in frames.values():
        _fields(frame, ("source", "scene_id", "sequence_id", "timestamp_s",
                        "image_path", "split"), "frame")
        for key in ("source", "scene_id", "sequence_id"):
            _text(frame, key, "frame")
        data_path(root, frame["image_path"])
        if frame["split"] not in ("development", "test"):
            raise ArtifactError(f"Invalid frame split: {frame['frame_id']}")
        timestamp = frame["timestamp_s"]
        if timestamp is not None and (type(timestamp) not in (int, float)
                                      or not math.isfinite(timestamp)):
            raise ArtifactError("timestamp_s must be finite number or null")
        source = frame["source"]
        # Recording isolation, not fictitious geographic independence at RELLIS.
        group_field = config.get("split_group_fields", {}).get(source)
        if group_field is None:
            group_field = "scene_id" if source.lower().startswith("phone") else "sequence_id"
        if group_field not in ("scene_id", "sequence_id"):
            raise ArtifactError("split_group_fields values must be scene_id or sequence_id")
        group = (source, frame[group_field])
        if groups.setdefault(group, frame["split"]) != frame["split"]:
            raise ArtifactError(f"Split leakage in {group_field}: {group}")
        if not config.get("fixture", False) and source == "rellis":
            planned = config.get("expected_recording_splits", {
                "00000": "development", "00001": "test", "00002": "test",
                "00003": "test", "00004": "test",
            })
            if frame["sequence_id"] in planned and planned[frame["sequence_id"]] != frame["split"]:
                raise ArtifactError(f"RELLIS recording violates frozen split: {frame['sequence_id']}")
    for region in regions.values():
        _fields(region, ("frame_id", "mask_path", "planning_relevant",
                         "selected_for_classification"), "region")
        _text(region, "frame_id", "region")
        if region["frame_id"] not in frames:
            raise ArtifactError(f"Region refers to missing frame: {region['region_id']}")
        data_path(root, region["mask_path"])
        for key in ("planning_relevant", "selected_for_classification"):
            if type(region[key]) is not bool:
                raise ArtifactError(f"region.{key} must be boolean")
    for annotation in annotations.values():
        _fields(annotation, ("reference_label", "semantic_class", "hazard_type",
                             "mask_quality", "annotation_status", "robot_profile_id"), "annotation")
        if annotation["region_id"] not in regions:
            raise ArtifactError(f"Annotation refers to missing region: {annotation['region_id']}")
        if annotation["robot_profile_id"] != policy:
            raise ArtifactError(f"Annotation policy mismatch: {annotation['region_id']}")
        status, label = annotation["annotation_status"], annotation["reference_label"]
        if status == "pending":
            if label is not None:
                raise ArtifactError("Pending annotations must have null reference_label")
        elif status == "complete":
            if label not in LABELS:
                raise ArtifactError("Completed annotations require a valid reference_label")
        else:
            raise ArtifactError(f"Invalid annotation_status: {status}")
        if annotation["mask_quality"] not in ("valid", "mixed", "broken", "unchecked"):
            raise ArtifactError(f"Invalid mask_quality: {annotation['region_id']}")
        for key in ("semantic_class", "hazard_type"):
            _nullable_text(annotation, key, "annotation")
        for key in ("annotation_source", "reference_scope", "reference_exclusion_reason"):
            if key in annotation:
                _nullable_text(annotation, key, "annotation")


def input_identity(frames, regions, root):
    contents = {}
    paths = {row["image_path"] for row in frames.values()}
    paths.update(row["mask_path"] for row in regions.values())
    for relative in sorted(paths):
        path = data_path(root, relative)
        try:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        except FileNotFoundError:
            contents[relative] = {"status": "missing"}
        except OSError as error:
            raise ArtifactError(f"Cannot hash input {relative}: {error}") from error
        else:
            contents[relative] = {"status": "present", "sha256": digest.hexdigest()}
    return {"frames": [frames[key] for key in sorted(frames)],
            "regions": [regions[key] for key in sorted(regions)], "contents": contents}


def inspect_images(frames, regions, root):
    try:
        from PIL import Image
    except ImportError as error:
        raise ArtifactError("Install requirements/evaluation.txt to validate images/masks") from error
    images, masks = {}, {}
    for key, frame in frames.items():
        try:
            with Image.open(data_path(root, frame["image_path"])) as image:
                image.load()
                images[key] = {"size": list(image.size), "defect": None}
        except (OSError, ValueError) as error:
            images[key] = {"size": None, "defect": f"image_unreadable: {error}"}
    for key, region in regions.items():
        if not region["selected_for_classification"]:
            continue
        defect, pixels = images[region["frame_id"]]["defect"], None
        try:
            with Image.open(data_path(root, region["mask_path"])) as mask:
                mask.load()
                if mask.format != "PNG" or len(mask.getbands()) != 1 or mask.mode == "P":
                    defect = "mask_not_single_channel_png"
                elif list(mask.size) != images[region["frame_id"]]["size"]:
                    defect = defect or "mask_image_size_mismatch"
                else:
                    if mask.mode in ("1", "L"):
                        pixels = sum(mask.histogram()[1:])
                    else:
                        # Integer PNG point callbacks receive symbolic operands;
                        # direct nonzero counting preserves 16-bit foreground.
                        values = mask.get_flattened_data() if hasattr(mask, "get_flattened_data") else mask.getdata()
                        pixels = sum(value != 0 for value in values)
                    if not pixels:
                        defect = "mask_empty"
        except (OSError, ValueError) as error:
            defect = f"mask_unreadable: {error}"
        size = images[region["frame_id"]]["size"]
        masks[key] = {"defect": defect, "foreground_pixels": pixels,
                      "area_fraction": pixels / (size[0] * size[1]) if pixels is not None and size else None}
    return images, masks


def pointer(value, selector):
    """Read an explicit JSON pointer; do not guess metadata field nesting."""
    if not isinstance(selector, str) or not selector.startswith("/"):
        raise ArtifactError(f"Expected JSON pointer: {selector!r}")
    try:
        for part in selector[1:].split("/"):
            value = value[part.replace("~1", "/").replace("~0", "~")]
        return value
    except (KeyError, TypeError) as error:
        raise ArtifactError(f"Missing metadata field {selector}") from error


def validate_metadata(metadata, entry, policy, inputs, config):
    fields = config.get("metadata_fields", {})
    if pointer(metadata, fields.get("model_key", "/model_key")) != entry["model_key"]:
        raise ArtifactError("Metadata model identity mismatch")
    if pointer(metadata, fields.get("robot_profile_id", "/configuration/settings/robot_profile_id")) != policy:
        raise ArtifactError("Metadata robot policy identity mismatch")
    digest = fingerprint(metadata)
    expected_digest = entry.get("expected_metadata_sha256")
    if expected_digest is not None and expected_digest != digest:
        raise ArtifactError("Expected metadata SHA256 mismatch")
    expected = entry.get("expected_metadata", config.get("expected_metadata", {}))
    if not isinstance(expected, dict):
        raise ArtifactError("expected_metadata must map JSON pointers to expected values")
    for selector, value in expected.items():
        if pointer(metadata, selector) != value:
            raise ArtifactError(f"Expected metadata configuration mismatch: {selector}")
    verified = False
    if "compatibility_fingerprint" in metadata:
        required = ("metadata_version", "model_key", "configuration", "execution", "inputs")
        if set(metadata) != set(required) | {"actual_backend", "configuration_fingerprint", "compatibility_fingerprint"}:
            raise ArtifactError("Invalid inference metadata schema")
        if metadata["metadata_version"] != 1 or type(metadata["metadata_version"]) is not int:
            raise ArtifactError("Unsupported inference metadata version")
        for key in ("configuration", "execution", "inputs", "actual_backend"):
            if not isinstance(metadata[key], dict):
                raise ArtifactError(f"Inference metadata {key} must be an object")
        if metadata["execution"].get("kind") not in ("real_model", "injected_fixture"):
            raise ArtifactError("Unknown inference execution identity")
        configuration = metadata["configuration"]
        _fields(configuration, ("configuration_version", "model_key", "model", "settings", "preprocessing",
                                "prompt", "labels", "software_versions"), "configuration")
        for key in ("model", "settings", "preprocessing", "software_versions"):
            if not isinstance(configuration[key], dict):
                raise ArtifactError(f"configuration.{key} must be an object")
        for key in ("checkpoint", "revision"):
            _text(configuration["model"], key, "configuration.model")
        _text(configuration, "prompt", "configuration")
        _fields(configuration["settings"], ("robot_profile_id", "robot_policy", "language_quantization_bits",
                                           "scene_visual_token_budget", "crop_visual_token_budget", "context_token_limit",
                                           "max_new_tokens", "request_concurrency", "seed", "enable_thinking"), "configuration.settings")
        if configuration["labels"] != list(LABELS):
            raise ArtifactError("Inference configuration label vocabulary mismatch")
        actual = {key: value for key, value in metadata.items() if key != "compatibility_fingerprint"}
        if fingerprint(actual) != metadata["compatibility_fingerprint"]:
            raise ArtifactError("Inference compatibility fingerprint is corrupt")
        base = {key: metadata[key] for key in required}
        if fingerprint(base) != metadata.get("configuration_fingerprint"):
            raise ArtifactError("Inference configuration fingerprint is corrupt")
        if metadata["inputs"] != inputs:
            raise ArtifactError("Inference input identity differs from current frames, regions or file bytes")
        if metadata["configuration"].get("model_key") != entry["model_key"]:
            raise ArtifactError("Nested configuration model identity mismatch")
        verified = True
    elif "inputs" in metadata and metadata["inputs"] != inputs:
        raise ArtifactError("Metadata inputs mismatch")
    if not (verified or expected or expected_digest):
        raise ArtifactError("Unrecognized metadata needs explicit expected_metadata checks or expected_metadata_sha256")
    return digest, metadata.get("configuration_fingerprint", digest)


def validate_predictions(predictions, frames, regions, model_key):
    warnings = []
    for prediction in predictions.values():
        _fields(prediction, ("frame_id", "model_key", "label", "semantic_class", "reason",
                             "raw_response", "status", "error_code"), "prediction")
        region_id = prediction["region_id"]
        if region_id not in regions:
            raise ArtifactError(f"Prediction refers to missing region: {region_id}")
        if prediction["frame_id"] != regions[region_id]["frame_id"]:
            raise ArtifactError(f"Prediction frame identity mismatch: {region_id}")
        if prediction["model_key"] != model_key:
            raise ArtifactError(f"Prediction model identity mismatch: {region_id}")
        if prediction["label"] not in LABELS or prediction["status"] not in ("ok", "error"):
            raise ArtifactError(f"Invalid prediction label/status: {region_id}")
        _nullable_text(prediction, "semantic_class", "prediction")
        _nullable_text(prediction, "error_code", "prediction")
        for key in ("reason", "raw_response"):
            if not isinstance(prediction[key], str):
                raise ArtifactError(f"prediction.{key} must be string")
        if prediction["status"] == "error":
            if not prediction["error_code"]:
                raise ArtifactError(f"Error prediction requires error_code: {region_id}")
            if prediction["label"] != "unknown":
                warnings.append({"code": "error_label_normalized", "region_id": region_id,
                                 "message": "Error prediction is counted as unknown despite its stored label"})
        elif prediction["error_code"] is not None:
            raise ArtifactError(f"Successful prediction has error_code: {region_id}")
    return warnings
