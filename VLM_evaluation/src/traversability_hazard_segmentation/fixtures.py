"""Independent synthetic CPU fixtures, never experimental model evidence."""

from pathlib import Path

from .common import (TASK_ID, atomic_json, atomic_jsonl, component_root, fingerprint,
    hash_file, image_modules, normalize, read_json, validate_frame, validate_prompts)

FIXTURE_ID = "hazard_cpu_fixture_v1"


class FakeBackend:
    fixture_identity = FIXTURE_ID

    def __init__(self, model_key="qwen3_vl_4b"):
        self.model_key = model_key
        self.metadata = {"fixture": True, "fixture_id": FIXTURE_ID, "model_key": model_key,
                         "checkpoint_revision": "synthetic", "device": "cpu"}
        self.settings = {"fixture": True, "fixture_identity": FIXTURE_ID, "device": "cpu"}
        self.calls, self.closed = [], False

    def predict_image(self, frame, data_root):
        validate_frame(frame, data_root)
        self.calls.append(frame["frame_id"])
        index = int(frame["frame_id"].rsplit(":", 1)[1])
        prompts = ["water", "puddle"] if self.model_key == "qwen3_vl_4b" else ["person", "bottle", "traffic cone"]
        status, code, raw = "ok", None, None
        if index == 0:
            prompts = []
        elif index == 1:
            prompts, status, code, raw = [], "error", "invalid_json", "{bad JSON"
        elif index == 2:
            prompts = ["obstacle", "traffic cone"]
        import json
        return {"task_id": TASK_ID, "schema_version": 1, "frame_id": frame["frame_id"],
                "model_key": self.model_key, "prompts": prompts,
                "raw_response": raw if raw is not None else json.dumps({"prompts": prompts}),
                "status": status, "error_code": code}

    def close(self):
        self.closed = True


class FakeSegmenter:
    fixture_identity = FIXTURE_ID

    def __init__(self):
        self.settings = {"fixture": True, "fixture_identity": FIXTURE_ID, "model_key": "sam3_image",
            "device": "cpu", "resolution": 1008, "confidence_threshold": 0.5,
            "mask_probability_threshold": 0.5, "code_revision": "synthetic", "checkpoint_revision": "synthetic"}
        self.metadata = {"fixture": True, "fixture_id": FIXTURE_ID, "execution_kind": "injected_fixture"}
        self.calls, self.closed = [], False

    def segment_image(self, frame, prompts, data_root):
        validate_frame(frame, data_root)
        validate_prompts(prompts)
        self.calls.append((frame["frame_id"], list(prompts)))
        np, _ = image_modules()
        shape = (frame["height"], frame["width"])
        union, queries = np.zeros(shape, dtype=bool), []
        vague = {"obstacle", "obstacles", "hazard", "hazards", "unsafe area", "non-traversable objects", "non traversable objects"}
        for index, phrase in enumerate(prompts):
            mask = np.zeros(shape, dtype=bool)
            unusable = normalize(phrase) in vague
            failed = normalize(phrase) == "bottle" or unusable
            if not unusable:
                mask[index % shape[0], :] = True
            masks, scores = ([] if unusable else [mask.copy()]), ([] if unusable else [0.75])
            union |= mask
            queries.append({"phrase": phrase, "masks": masks, "scores": scores, "union_mask": mask.copy(),
                "status": "error" if failed else "ok", "error_code": "unusable_phrase" if unusable else "fixture_query_error" if failed else None,
                "sam_query_count": 0 if unusable else 1})
        failures = any(row["status"] == "error" for row in queries)
        return {"queries": queries, "frame": {"frame_id": frame["frame_id"], "union_mask": union.copy(),
                "status": "error" if failures else "ok", "error_code": "query_error" if failures else None,
                "sam_query_count": sum(row["sam_query_count"] for row in queries)}, "settings": self.settings.copy()}

    def close(self):
        self.closed = True


def prepare_fixture(run_dir, data_root):
    """Create tiny independently marked inputs under explicitly supplied empty roots."""
    run_dir, data_root = Path(run_dir).resolve(), Path(data_root).resolve()
    if (run_dir.exists() and any(run_dir.iterdir())) or (data_root.exists() and any(data_root.iterdir())):
        raise ValueError("Fixture roots must be empty; use unique directories")
    run_dir.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    np, Image = image_modules()
    policy_path = component_root() / "configs/hazards/policy.json"
    policy = read_json(policy_path)
    frames, references, assets = [], [], {}
    for source in ("rellis", "coco"):
        vocabulary = list(policy["datasets"]["rellis"]["hazard_label_ids"]) if source == "rellis" else policy["datasets"]["coco"]["scored_category_names"]
        for split, count in (("development", 3), ("test", 10)):
            for index in range(count):
                identifier = f"fixture:{source}:{split}:{index:03d}"
                base = f"fixture/{source}/{split}/{index:03d}"
                relative = base + "/rgb.png"
                image_path = data_root / relative
                image_path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (9, 5), (index * 10, 20, 30)).save(image_path)
                frames.append({"frame_id": identifier, "source": source, "scene_id": "synthetic",
                    "sequence_id": f"fixture-{source}-{split}", "timestamp_s": None, "image_path": relative,
                    "split": split, "width": 9, "height": 5, "image_sha256": hash_file(image_path), "fixture": True})
                assets[relative] = hash_file(image_path)
                concepts = [] if index == 0 else (["water"] if source == "rellis" else ["person"])
                masks, counts = {}, {}
                for concept in concepts:
                    mask = np.zeros((5, 9), dtype=np.uint8)
                    mask[0, 0] = 255
                    path = base + f"/{concept}.png"
                    Image.fromarray(mask).save(data_root / path)
                    masks[concept], counts[concept] = path, 1
                    assets[path] = hash_file(data_root / path)
                valid = base + "/valid.png"
                Image.new("L", (9, 5), 255).save(data_root / valid)
                assets[valid] = hash_file(data_root / valid)
                references.append({"frame_id": identifier, "scored_concepts": vocabulary,
                    "present_concepts": concepts, "absence_scoring_eligible": True, "concept_masks": masks,
                    "concept_pixel_counts": counts, "valid_mask_path": valid, "annotation_source": "synthetic_fixture",
                    "reference_scope": "synthetic_annotated_hazard_concepts", "status": "complete", "fixture": True})
    dataset = {"task_id": TASK_ID, "schema_version": 1, "fixture": True, "fixture_id": FIXTURE_ID,
        "policy_id": policy["policy_id"], "policy_hash": fingerprint(policy), "policy_sha256": hash_file(policy_path),
        "alias_hash": fingerprint(policy["aliases"]), "alias_sha256": fingerprint(policy["aliases"]),
        "selected_frame_ids": [row["frame_id"] for row in frames],
        "sampling": {"method": "synthetic fixture only"}, "annotation_provenance": {"rellis": "synthetic", "coco": "synthetic"},
        "annotation_hashes": {"synthetic_references": fingerprint(references)},
        "coverage": {"expected_frames": {source: {"development": 3, "test": 10} for source in ("rellis", "coco")}}}
    dataset["dataset_fingerprint"] = fingerprint({"frames": frames, "references": references, "assets": assets})
    atomic_jsonl(run_dir / "frames.jsonl", frames)
    atomic_jsonl(run_dir / "references.jsonl", references)
    atomic_json(run_dir / "metadata/dataset.json", dataset)
    # Selection must precede even synthetic predictions.
    from .selection import freeze_with_roots
    freeze_with_roots(run_dir, data_root, policy_path=policy_path, fixture=True)
    for model in ("qwen3_vl_4b", "qwen3_5_4b"):
        backend = FakeBackend(model)
        predictions = [backend.predict_image(frame, data_root) for frame in frames]
        meta = {"task_id": TASK_ID, "schema_version": 1, "fixture": True, "model_key": model,
            "execution_kind": "injected_fixture", "settings": backend.settings, "checkpoint_revision": "synthetic",
            "dataset_fingerprint": dataset["dataset_fingerprint"], "policy_hash": dataset["policy_hash"],
            "policy_sha256": dataset["policy_sha256"], "alias_hash": dataset["alias_hash"], "alias_sha256": dataset["alias_sha256"]}
        atomic_json(run_dir / f"metadata/{model}.json", meta)
        atomic_jsonl(run_dir / f"predictions/{model}.jsonl", predictions)
        backend.close()
    return {"run_dir": str(run_dir), "data_root": str(data_root), "policy_path": str(policy_path),
        "fixture": True, "sam_settings": {"fixture": True, "fixture_identity": FIXTURE_ID},
        "enable_real_run": False, "enable_combined": False, "seed": 0}
