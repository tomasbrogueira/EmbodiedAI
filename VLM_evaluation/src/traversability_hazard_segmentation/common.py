"""Model-free hazard validation and reuse of portable storage helpers."""

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

from traversability_data.storage import (_atomic as _legacy_atomic_bytes, data_path,
    exclusive_lock, hash_file, read_json, read_jsonl)

TASK_ID = "hazard_prompt_v1"
SCHEMA_VERSION = 1
CONDITIONS = ("vlm__qwen3_vl_4b", "vlm__qwen3_5_4b", "reference_present", "fixed_policy")


def atomic_bytes(path, content):
    """Retain atomic publication through brief Windows scanner/sharing locks."""
    for attempt in range(5):
        try:
            return _legacy_atomic_bytes(path, content)
        except PermissionError as error:
            if getattr(error, "winerror", None) not in {5, 32, 33} or attempt == 4:
                raise
            time.sleep(0.05 * (attempt + 1))


def _json_bytes(value, **options):
    # Failed raw model text can contain a lone surrogate. Preserve it as a JSON
    # escape while retaining the existing UTF-8 recipe for valid Unicode.
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, **options).encode("utf-8", errors="backslashreplace")


def fingerprint(value):
    return hashlib.sha256(_json_bytes(value, separators=(",", ":"))).hexdigest()


def atomic_json(path, value):
    atomic_bytes(path, _json_bytes(value, indent=2) + b"\n")


def atomic_jsonl(path, records):
    atomic_bytes(path, b"".join(_json_bytes(row) + b"\n" for row in records))


def safe_path(root, relative, must_exist=False):
    if isinstance(relative, str):
        reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
        for part in relative.split("/"):
            if any(ord(c) < 32 or c in '<>:"|?*' for c in part) or part.endswith((".", " ")) or part.split(".")[0].upper() in reserved:
                raise ValueError(f"Nonportable path component: {part!r}")
    return data_path(root, relative, must_exist=must_exist)


def image_modules():
    try:
        import numpy as np
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("CPU image validation requires NumPy and Pillow") from error
    return np, Image


def load_config(config):
    if isinstance(config, (str, os.PathLike)):
        config = read_json(config)
    if not isinstance(config, dict):
        raise ValueError("config must be an object or JSON configuration path")
    # Keep fixture clock/probe objects intact; only serializable fields enter identity.
    return {key: value if key.startswith("_") else deepcopy(value) for key, value in config.items()}


def component_root(value=None):
    root = Path(value or os.environ.get("TRAVERSABILITY_REPO_ROOT") or Path(__file__).resolve().parents[2]).expanduser().resolve()
    if not (root / "docs/hazard_interfaces.md").is_file() and (root / "VLM_evaluation/docs/hazard_interfaces.md").is_file():
        root = root / "VLM_evaluation"
    if not (root / "docs/hazard_interfaces.md").is_file():
        raise ValueError("component_root must contain docs/hazard_interfaces.md")
    return root


def safe_name(value, label="name"):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"Unsafe {label}: {value!r}")
    safe_path(Path.cwd(), value)
    return value


def resolve_config(config):
    config = load_config(config)
    root = component_root(config.get("component_root", config.get("repo_root")))
    def resolve(key, default):
        value = config.get(key) or os.environ.get("TRAVERSABILITY_" + key.upper()) or default
        path = Path(value).expanduser()
        return (path if path.is_absolute() else root / path).resolve()
    run_root = resolve("run_root", "runs")
    name = safe_name(config.get("run_name", TASK_ID), "run_name")
    run_dir = Path(config["run_dir"]).expanduser() if config.get("run_dir") else run_root / name
    if not run_dir.is_absolute():
        run_dir = root / run_dir
    return {"component_root": root, "run_dir": run_dir.resolve(),
            "data_root": resolve("data_root", "data"), "cache_root": resolve("cache_root", ".cache"),
            "policy_path": resolve("policy_path", "configs/hazards/policy.json")}


def normalize(phrase):
    return " ".join(phrase.casefold().split())


def validate_prompts(prompts):
    if not isinstance(prompts, list) or len(prompts) > 32:
        raise ValueError("prompts must be a list of at most 32 original phrases")
    seen = set()
    for phrase in prompts:
        if not isinstance(phrase, str) or not phrase.strip() or len(phrase) > 80:
            raise ValueError("Each phrase must be nonempty and at most 80 characters")
        try:
            phrase.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("Phrase contains invalid Unicode") from error
        key = normalize(phrase)
        if key in seen:
            raise ValueError("Duplicate normalized phrase in saved prompts")
        seen.add(key)
    return prompts


def validate_task(record, kind="record"):
    if record.get("task_id") != TASK_ID or type(record.get("schema_version")) is not int or record["schema_version"] != 1:
        raise ValueError(f"{kind} task/schema mismatch; legacy region records are invalid")


def _sha(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError(f"Invalid SHA-256: {label}")


def validate_frame(frame, data_root):
    for key in ("frame_id", "source", "scene_id", "sequence_id"):
        if not isinstance(frame.get(key), str) or not frame[key].strip():
            raise ValueError(f"frame.{key} must be a nonempty string")
    if frame.get("source") not in ("rellis", "coco") or frame.get("split") not in ("development", "test"):
        raise ValueError("Unknown frame source/split")
    if "timestamp_s" not in frame:
        raise ValueError("frame.timestamp_s is required (null is allowed)")
    time = frame["timestamp_s"]
    if time is not None and (type(time) not in (int, float) or not math.isfinite(time)):
        raise ValueError("Invalid timestamp_s")
    for key in ("width", "height"):
        if type(frame.get(key)) is not int or frame[key] <= 0:
            raise ValueError(f"frame.{key} must be a positive integer")
    _sha(frame.get("image_sha256"), "image_sha256")
    path = safe_path(data_root, frame.get("image_path"), must_exist=True)
    if hash_file(path) != frame["image_sha256"]:
        raise ValueError(f"Image hash mismatch: {frame['frame_id']}")
    _, Image = image_modules()
    with Image.open(path) as image:
        image.load()
        if image.size != (frame["width"], frame["height"]):
            raise ValueError(f"Image dimensions mismatch: {frame['frame_id']}")
        # Conversion is performed by the adapter, without EXIF rotation/resizing.
        image.convert("RGB")
    return path


def binary_mask(root, relative, frame):
    np, Image = image_modules()
    path = safe_path(root, relative, must_exist=True)
    with Image.open(path) as image:
        image.load()
        if image.format != "PNG" or len(image.getbands()) != 1 or image.mode == "P":
            raise ValueError(f"Expected single-channel binary PNG: {relative}")
        if image.size != (frame["width"], frame["height"]):
            raise ValueError(f"Mask dimensions mismatch: {relative}")
        array = np.asarray(image)
        values = set(np.unique(array).tolist())
        if not (values <= {0, 1} or values <= {0, 255}):
            raise ValueError(f"Nonbinary mask: {relative}")
        return (array != 0).copy()


def _unique(rows, key, kind):
    result = {}
    for row in rows:
        identifier = row.get(key)
        if not isinstance(identifier, str) or not identifier or identifier in result:
            raise ValueError(f"Missing/duplicate {kind} {key}: {identifier}")
        result[identifier] = row
    return result


def verify_policy_metadata(metadata, inputs, kind):
    # Field names define the recipe. Do not guess that a wrong hash is compatible.
    config = metadata.get("configuration", metadata)
    found = False
    for key, expected in (("policy_sha256", inputs["policy_sha256"]),
                          ("policy_hash", inputs["policy_hash"]),
                          ("alias_sha256", inputs["alias_sha256"]),
                          ("alias_hash", inputs["alias_sha256"])):
        value = metadata.get(key, config.get(key))
        if value is not None:
            if value != expected:
                raise ValueError(f"{kind} {key} mismatch")
            found = found or key.startswith("policy")
    if not found or not any(key in metadata or key in config for key in ("alias_hash", "alias_sha256")):
        raise ValueError(f"{kind} must declare policy and alias hashes")


def read_inputs(run_dir, data_root, policy_path=None, fixture=None):
    run_dir, data_root = Path(run_dir), Path(data_root)
    policy_path = Path(policy_path) if policy_path else component_root() / "configs/hazards/policy.json"
    policy = read_json(policy_path)
    validate_task(policy, "policy")
    if policy.get("policy_id") != "visible_avoid_concepts_v1":
        raise ValueError("Frozen policy identity mismatch")
    if len(policy.get("canonical_prompts", [])) != 16 or len(set(policy["canonical_prompts"])) != 16:
        raise ValueError("Frozen policy must have 16 unique canonical prompts")
    dataset = read_json(safe_path(run_dir, "metadata/dataset.json", True))
    validate_task(dataset, "dataset")
    if dataset.get("policy_id", policy["policy_id"]) != policy["policy_id"]:
        raise ValueError("Dataset policy identity mismatch")
    if type(dataset.get("fixture")) is not bool or (fixture is not None and dataset["fixture"] is not fixture):
        raise ValueError("Dataset fixture provenance mismatch")
    _sha(dataset.get("dataset_fingerprint"), "dataset_fingerprint")
    inputs = {"policy": policy, "dataset": dataset, "fixture": dataset["fixture"],
              "policy_sha256": hash_file(policy_path), "policy_hash": fingerprint(policy),
              "alias_sha256": fingerprint(policy["aliases"])}
    verify_policy_metadata(dataset, inputs, "dataset")
    frames = _unique(read_jsonl(safe_path(run_dir, "frames.jsonl", True)), "frame_id", "frame")
    references = _unique(read_jsonl(safe_path(run_dir, "references.jsonl", True)), "frame_id", "reference")
    if set(frames) != set(references):
        raise ValueError("Frames/references must have an exact one-to-one join")
    selected = dataset.get("selected_frame_ids", sorted(frames))
    if not isinstance(selected, list) or len(selected) != len(set(selected)) or set(selected) != set(frames):
        raise ValueError("Dataset selected IDs mismatch")
    assets, paths = {}, set()
    for identifier, frame in frames.items():
        if "fixture" in frame and frame["fixture"] is not dataset["fixture"]:
            raise ValueError("Frame fixture provenance mismatch")
        validate_frame(frame, data_root)
        if frame["image_path"] in paths:
            raise ValueError("Duplicate image_path")
        paths.add(frame["image_path"])
        assets[frame["image_path"]] = frame["image_sha256"]
        ref = references[identifier]
        if ref.get("status") != "complete":
            raise ValueError(f"Reference incomplete: {identifier}")
        if "fixture" in ref and ref["fixture"] is not dataset["fixture"]:
            raise ValueError("Reference fixture provenance mismatch")
        vocabulary = list(policy["datasets"]["rellis"]["hazard_label_ids"]) if frame["source"] == "rellis" else policy["datasets"]["coco"]["scored_category_names"]
        scored, present = ref.get("scored_concepts"), ref.get("present_concepts")
        if not isinstance(scored, list) or len(scored) != len(set(scored)) or set(scored) != set(vocabulary):
            raise ValueError("Reference scored vocabulary mismatch")
        if not isinstance(present, list) or len(present) != len(set(present)) or not set(present) <= set(scored):
            raise ValueError("Reference present concepts mismatch")
        if type(ref.get("absence_scoring_eligible")) is not bool:
            raise ValueError("Reference absence eligibility must be boolean")
        masks, counts = ref.get("concept_masks"), ref.get("concept_pixel_counts")
        if not isinstance(masks, dict) or set(masks) != set(present) or not isinstance(counts, dict) or set(counts) != set(present):
            raise ValueError("Reference concepts/masks/pixel-count joins mismatch")
        relative = ref.get("valid_mask_path")
        valid = binary_mask(data_root, relative, frame)
        area = frame["width"] * frame["height"]
        ignored_fraction = (area - int(valid.sum())) / area
        if frame["source"] == "rellis":
            eligible = ignored_fraction <= policy["datasets"]["rellis"]["absence_scoring_max_ignored_fraction"]
            if ref["absence_scoring_eligible"] is not eligible:
                raise ValueError("Reference absence eligibility disagrees with RELLIS evaluable pixels")
        elif not ref["absence_scoring_eligible"] or not valid.all():
            raise ValueError("COCO references require exhaustive evaluable pixels and absence eligibility")
        for concept, relative in masks.items():
            mask = binary_mask(data_root, relative, frame)
            if type(counts[concept]) is not int or counts[concept] <= 0 or int(mask.sum()) != counts[concept]:
                raise ValueError("Reference concept pixel count mismatch")
            if (mask & ~valid).any():
                raise ValueError("Reference hazard masks must lie within evaluable pixels")
            if relative in paths:
                raise ValueError("Duplicate reference asset path")
            paths.add(relative)
            assets[relative] = hash_file(safe_path(data_root, relative, True))
        relative = ref.get("valid_mask_path")
        if relative in paths:
            raise ValueError("Duplicate valid reference asset path")
        paths.add(relative)
        assets[relative] = hash_file(safe_path(data_root, relative, True))
        for key in ("annotation_source", "reference_scope"):
            if not isinstance(ref.get(key), str) or not ref[key]:
                raise ValueError(f"Missing reference {key}")
    inputs.update(frames=frames, references=references)
    inputs["identity"] = {"task_id": TASK_ID, "schema_version": 1, "fixture": dataset["fixture"],
        "policy_sha256": inputs["policy_sha256"], "policy_hash": inputs["policy_hash"],
        "alias_sha256": inputs["alias_sha256"], "dataset": dataset,
        "frames": [frames[key] for key in sorted(frames)],
        "references": [references[key] for key in sorted(references)], "assets": assets}
    return inputs


def adapter_frame(frame):
    """Pass only RGB/frame identity to inference adapters, never reference fields."""
    keys = ("frame_id", "source", "scene_id", "sequence_id", "timestamp_s", "image_path",
            "split", "width", "height", "image_sha256", "fixture")
    return {key: deepcopy(frame[key]) for key in keys if key in frame}


def portable_settings(value):
    if isinstance(value, dict):
        return {key: portable_settings(item) for key, item in value.items()
                if not key.startswith("_") and key not in ("processor", "checkpoint_path", "source_path", "code_source_root", "cache_root", "cache_dir", "policy_path", "run_dir", "data_root", "run_root", "component_root")}
    if isinstance(value, (tuple, list)):
        return [portable_settings(item) for item in value]
    return value
