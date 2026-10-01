"""Prediction-independent seeded hazard selection."""

from collections import Counter
import hashlib
from pathlib import Path

from .common import (TASK_ID, atomic_json, exclusive_lock, fingerprint, read_json,
    read_inputs, resolve_config, safe_path)


def _selection(inputs, test_frames_per_source, warmup_frames, seed):
    if type(test_frames_per_source) is not int or test_frames_per_source <= 0 or type(warmup_frames) is not int or warmup_frames <= 0 or type(seed) is not int:
        raise ValueError("Selection sizes must be positive integers and seed an integer")
    def ranked(rows):
        return sorted(rows, key=lambda key: (hashlib.sha256(f"{seed}:{key}".encode()).hexdigest(), key))
    frames = inputs["frames"]
    test, missing = [], {}
    for source in ("rellis", "coco"):
        candidates = ranked([key for key, row in frames.items() if row["source"] == source and row["split"] == "test"])
        test.extend(candidates[:test_frames_per_source])
        missing[source] = max(0, test_frames_per_source - len(candidates))
    warmup = ranked([key for key, row in frames.items() if row["split"] == "development"])[:warmup_frames]
    missing["development"] = max(0, warmup_frames - len(warmup))
    chosen = test + warmup
    coverage = {"selected_test_per_source": dict(Counter(frames[key]["source"] for key in test)),
                "missing": missing, "concept_presence": dict(Counter(concept for key in test for concept in inputs["references"][key]["present_concepts"])),
                "pending_coverage": inputs["policy"].get("pending_coverage", [])}
    result = {"task_id": TASK_ID, "schema_version": 1, "fixture": inputs["fixture"],
        "policy_hash": inputs["policy_hash"], "policy_sha256": inputs["policy_sha256"],
        "alias_hash": inputs["alias_sha256"], "alias_sha256": inputs["alias_sha256"],
        "dataset_fingerprint": inputs["dataset"]["dataset_fingerprint"],
        "test_frame_ids": test, "warmup_frame_ids": warmup, "seed": seed,
        "test_frames_per_source": test_frames_per_source, "warmup_frames": warmup_frames,
        "selection_method": "sha256(seed:frame_id), source-stratified test; global development",
        "complete": not any(missing.values()), "coverage": coverage,
        "identity": inputs["identity"],
        "selected_inputs": {key: {"image_sha256": frames[key]["image_sha256"],
                                 "reference_sha256": fingerprint(inputs["references"][key])} for key in chosen}}
    result["fingerprint"] = fingerprint(result)
    result["selection_hash"] = result["fingerprint"]
    return result


def freeze_with_roots(run_dir, data_root, *, policy_path=None, fixture=None,
                      test_frames_per_source=10, warmup_frames=5, seed=0):
    inputs = read_inputs(run_dir, data_root, policy_path, fixture)
    result = _selection(inputs, test_frames_per_source, warmup_frames, seed)
    path = safe_path(run_dir, "segmentation/selection.json")
    with exclusive_lock(safe_path(run_dir, "segmentation/.selection.lock")):
        if path.exists():
            if read_json(path) != result:
                raise ValueError("Frozen selection/input identity changed; use a new run")
        else:
            predictions = safe_path(run_dir, "predictions")
            metadata = safe_path(run_dir, "metadata")
            generated = any(predictions.glob("*.jsonl")) if predictions.exists() else False
            generated = generated or any(item.name != "dataset.json" for item in metadata.glob("*.json"))
            generated = generated or any(safe_path(run_dir, f"segmentation/{condition}/metadata.json").exists()
                                         for condition in ("vlm__qwen3_vl_4b", "vlm__qwen3_5_4b",
                                                           "reference_present", "fixed_policy"))
            benchmark = safe_path(run_dir, "benchmark")
            generated = generated or (benchmark.exists() and any(benchmark.rglob("*.json")))
            if generated:
                raise ValueError("Freeze selection before predictions, segmentation or profiles; use a new run")
            atomic_json(path, result)
    return result


def freeze_selection(run_dir, *, data_root=None, policy_path=None,
                     test_frames_per_source=10, warmup_frames=5, seed=0):
    """Freeze with explicit roots, or configured roots when an override is absent."""
    config = {"run_dir": str(run_dir)}
    if data_root is not None:
        config["data_root"] = str(data_root)
    if policy_path is not None:
        config["policy_path"] = str(policy_path)
    roots = resolve_config(config)
    return freeze_with_roots(roots["run_dir"], roots["data_root"], policy_path=roots["policy_path"],
        test_frames_per_source=test_frames_per_source, warmup_frames=warmup_frames, seed=seed)


def validate_selection(run_dir, data_root, selection=None, *, policy_path=None, fixture=None):
    if selection is None:
        selection = read_json(safe_path(run_dir, "segmentation/selection.json", True))
    inputs = read_inputs(run_dir, data_root, policy_path, fixture)
    expected = _selection(inputs, selection["test_frames_per_source"], selection["warmup_frames"], selection["seed"])
    if selection != expected:
        raise ValueError("Frozen selection/input identity changed; use a new run/profile")
    return inputs
