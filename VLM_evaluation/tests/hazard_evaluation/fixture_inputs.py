"""Independently authored synthetic hazard artifacts; never experiment evidence.

No data, inference or segmentation producer is imported here. Every metadata
artifact explicitly carries fixture=true and each mask is built by hand.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


COMPONENT = Path(__file__).resolve().parents[2]
MODEL = "qwen3_vl_4b"
TASK = "hazard_prompt_v1"


def canonical_hash(value) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"),
                            ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n"
                            for value in records), encoding="utf-8")


class HazardFixture:
    """64 x 64 original-size RGB/mask assets with explicit frozen identities."""

    def __init__(self, root: Path):
        self.data_root = root / "data"
        self.run_root = root / "runs"
        self.run = self.run_root / TASK
        self.policy = json.loads((COMPONENT / "configs/hazards/policy.json").read_text("utf-8"))
        self.frames: list[dict] = []
        self.references: list[dict] = []
        self.predictions: list[dict] = []
        self.identity = {
            "task_id": TASK,
            "schema_version": 1,
            "fixture": True,
            "policy_hash": canonical_hash(self.policy),
            "alias_hash": canonical_hash(self.policy["aliases"]),
            "dataset_fingerprint": hashlib.sha256(b"independent hazard CPU fixture").hexdigest(),
        }
        self.dataset_metadata = {
            **self.identity,
            "selected_frame_ids": [],
            "source_annotations": {"rellis": {"kind": "independent_fixture", "sha256": "1" * 64},
                                   "coco": {"kind": "independent_fixture", "sha256": "2" * 64}},
            "sampling": {"kind": "independent_fixture", "seed": 0},
            "coverage": {"expected_frames": {}},
        }
        self.model_metadata = {
            **self.identity,
            "model_key": MODEL,
            "requested_frame_ids": [],
            "execution_kind": "injected_fixture",
            "checkpoint_revision": "independent-fixture",
        }
        self.config = {
            "component_root": str(COMPONENT),
            "data_root": str(self.data_root),
            "run_root": str(self.run_root),
            "run_name": TASK,
            "model_keys": [MODEL],
            "fixture": True,
            "hash_recipes": {"policy": "canonical_json", "aliases": "canonical_json"},
            "metadata_bindings": {},
            "dataset_fingerprint_recipe": "declared",
            "coverage": {"expected_frames": {}},
        }
        self.segmentation: dict[str, dict] = {}
        self.selection: dict | None = None
        self.profiles: dict[str, dict] = {}

    def scored_concepts(self, source: str) -> list[str]:
        if source == "rellis":
            return list(self.policy["datasets"]["rellis"]["hazard_label_ids"])
        return list(self.policy["datasets"]["coco"]["scored_category_names"])

    def mask(self, relative: str, positions=(), *, valid: bool = False,
             root: Path | None = None, size: tuple[int, int] = (64, 64)) -> str:
        """Positions are explicit (row, column); valid masks start fully true."""
        array = np.full((size[1], size[0]), valid, dtype=np.uint8)
        for row, column in positions:
            array[row, column] = 0 if valid else 1
        target = (root or self.data_root) / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array * 255, mode="L").save(target)
        return relative

    def frame(self, *, source="rellis", split="test", concepts: dict[str, list] | None = None,
              ignored=(), status="complete", size=(64, 64)) -> dict:
        index = len(self.frames)
        sequence = "00000" if split == "development" else "00001"
        frame_id = f"{source}:{sequence}:fixture{index:03d}"
        relative = f"fixture_rgb/{source}_{split}_{index:03d}.png"
        image = self.data_root / relative
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, (index, 30, 40)).save(image)
        frame = {
            "frame_id": frame_id,
            "source": source,
            "scene_id": "marked-synthetic-fixture",
            "sequence_id": sequence,
            "timestamp_s": None,
            "image_path": relative,
            "split": split,
            "width": size[0],
            "height": size[1],
            "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        }
        self.frames.append(frame)
        prefix = f"hazard_references/fixture{index:03d}"
        concepts = concepts or {}
        mask_paths = {concept: self.mask(f"{prefix}/{concept}.png", positions, size=size)
                      for concept, positions in concepts.items()}
        self.references.append({
            "frame_id": frame_id,
            "scored_concepts": self.scored_concepts(source),
            "present_concepts": list(concepts),
            "absence_scoring_eligible": source == "coco" or len(ignored) / (size[0] * size[1]) <= 0.05,
            "concept_masks": mask_paths,
            "concept_pixel_counts": {concept: len(set(map(tuple, positions)))
                                      for concept, positions in concepts.items()},
            "valid_mask_path": self.mask(f"{prefix}/valid.png", ignored, valid=True, size=size),
            "annotation_source": "dataset_policy",
            "reference_scope": "annotated_hazard_concepts",
            "status": status,
        })
        self._coverage()
        return frame

    def predict(self, frame: dict, prompts=(), *, status="ok", error_code=None) -> dict:
        record = {
            "task_id": TASK,
            "schema_version": 1,
            "frame_id": frame["frame_id"],
            "model_key": MODEL,
            "prompts": list(prompts),
            "raw_response": json.dumps({"prompts": list(prompts)}),
            "status": status,
            "error_code": error_code,
        }
        self.predictions.append(record)
        return record

    def _coverage(self) -> None:
        counts = Counter((frame["source"], frame["split"]) for frame in self.frames)
        coverage = {}
        for (source, split), count in counts.items():
            coverage.setdefault(source, {})[split] = count
        self.config["coverage"]["expected_frames"] = coverage
        self.dataset_metadata["coverage"]["expected_frames"] = coverage
        ids = [frame["frame_id"] for frame in self.frames]
        self.dataset_metadata["selected_frame_ids"] = ids[:]
        self.model_metadata["requested_frame_ids"] = ids[:]

    def discovery_case(self) -> None:
        """Hand calculation: recall 2/4; supported precision 1/2, not 2/3."""
        first = self.frame(concepts={"water": [(10, 10)],
                                    "tree": [(10, 20), (10, 21), (10, 22), (10, 23)]})
        self.predict(first, [" PUDDLE ", "water", "person", "cup", "traffic cone", "hazards"])
        second = self.frame(concepts={"tree": [(10, 20), (10, 21), (10, 22), (10, 23)],
                                     "water": [(11, 10), (11, 11), (11, 12), (11, 13), (11, 14)]},
                            ignored=[(row, column) for row in range(4) for column in range(64)])
        self.predict(second, ["tree"])

    def add_segmentation_case(self) -> None:
        """RELLIS person truth and scored/full unions differ; valid pixels matter."""
        frame = self.frame(source="rellis", concepts={"person": [(8, 8), (8, 9)]}, ignored=[(8, 11)])
        self.predict(frame, ["person", "cup"])
        condition = f"vlm__{MODEL}"
        test_ids = [frame["frame_id"]]
        selection_core = {
            **self.identity,
            "test_frame_ids": test_ids,
            "warmup_frame_ids": [],
            "coverage": {"rellis": {"test": 1}},
        }
        selection_hash = canonical_hash(selection_core)
        self.selection = {**selection_core, "selection_hash": selection_hash}
        metadata = {
            **self.identity,
            "selection_hash": selection_hash,
            "condition_key": condition,
            "sam_settings": {"checkpoint": "independent-fixture", "revision": "fixture",
                             "input_resolution": 64, "confidence_threshold": 0.5,
                             "mask_threshold": 0.5},
        }
        queries = []
        masks = [[(8, 8), (8, 10), (8, 11)], [(8, 9), (8, 12)]]
        for index, (phrase, positions) in enumerate(zip(["person", "cup"], masks)):
            query_id = f"{condition}:{frame['frame_id']}:q{index:03d}"
            path = self.mask(f"segmentation/{condition}/masks/fixture_q{index:03d}.png",
                             positions, root=self.run)
            queries.append({
                "task_id": TASK, "schema_version": 1,
                "frame_id": frame["frame_id"], "condition_key": condition,
                "query_id": query_id, "phrase": phrase, "canonical_concept": phrase,
                "mask_paths": [path], "scores": [0.9], "returned_instance_count": 1,
                "union_mask_path": path, "status": "ok", "error_code": None,
            })
        union_path = self.mask(f"segmentation/{condition}/masks/fixture_all_queries.png",
                               [(8, 8), (8, 9), (8, 10), (8, 11), (8, 12)], root=self.run)
        frames = [{
            "task_id": TASK, "schema_version": 1,
            "frame_id": frame["frame_id"], "condition_key": condition,
            "query_ids": [query["query_id"] for query in queries],
            "requested_queries": 2, "completed_queries": 2, "failed_queries": 0,
            "sam_query_count": 2, "upstream_status": "ok", "union_mask_path": union_path,
            "status": "ok", "error_code": None,
        }]
        self.segmentation[condition] = {"queries": queries, "frames": frames, "metadata": metadata}
        self.config["segmentation"] = {
            "enabled": True, "conditions": [condition],
            "selection_path": "segmentation/selection.json",
            "completed_queries_semantics": "successful",
            "expected_test_per_source": {"rellis": 1}, "expected_warmup_frames": 0,
        }

    def save(self) -> None:
        write_jsonl(self.run / "frames.jsonl", self.frames)
        write_jsonl(self.run / "references.jsonl", self.references)
        write_jsonl(self.run / "predictions" / f"{MODEL}.jsonl", self.predictions)
        write_json(self.run / "metadata/dataset.json", self.dataset_metadata)
        write_json(self.run / "metadata" / f"{MODEL}.json", self.model_metadata)
        if self.selection:
            write_json(self.run / "segmentation/selection.json", self.selection)
        for condition, artifacts in self.segmentation.items():
            directory = self.run / "segmentation" / condition
            write_jsonl(directory / "queries.jsonl", artifacts["queries"])
            write_jsonl(directory / "frames.jsonl", artifacts["frames"])
            write_json(directory / "metadata.json", artifacts["metadata"])
        for relative, artifacts in self.profiles.items():
            directory = self.run / relative
            write_json(directory / "summary.json", artifacts["summary"])
            write_json(directory / "metadata.json", artifacts["metadata"])
            write_jsonl(directory / "samples.jsonl", artifacts["samples"])

    def add_profile(self, component: str, *, elapsed=(1.0, 3.0), peak_bytes=100) -> dict:
        """Saved synthetic measurements, explicitly marked injected_fixture."""
        if self.selection is None:
            raise ValueError("Create a frozen segmentation selection first")
        condition = f"vlm__{MODEL}"
        relative = f"benchmark/{condition}/{component}_fixture"
        calls = 0 if component == "vlm" else 2
        measured_samples = [
            {"sample_id": f"fixture:{component}:r{repeat}", "task_id": TASK, "schema_version": 1,
             "fixture": True, "execution_kind": "injected_fixture", "component": component,
             "condition_key": condition, "frame_id": self.selection["test_frame_ids"][0],
             "phase": "measured", "repeat": repeat, "elapsed_s": duration,
             "status": "ok", "error_code": None, "sam_query_count": calls}
            for repeat, duration in enumerate(elapsed)
        ]
        summary = {
            "task_id": TASK, "schema_version": 1, "fixture": True,
            "execution_kind": "injected_fixture", "component": component,
            "condition_key": condition, "complete": True, "comparison_ready": False,
            "measured_frames": len(elapsed), "expected_measured_frames": 2,
            "unique_test_frames": 1, "failed_calls": 0,
            "sam_query_count": calls * len(elapsed),
            "latency": {"median_s": float(np.median(elapsed)), "p95_s": float(np.quantile(elapsed, 0.95))},
            "memory": {"allocated_peak_bytes": peak_bytes,
                       "incremental_allocated_peak_bytes": peak_bytes // 2,
                       "unavailable_reasons": ["Synthetic CPU fixture; no GPU telemetry measured."]},
            "loading": {"elapsed_s": 7.0}, "cold_start": {"elapsed_s": 5.0},
            "warmup": {"frames": 0},
            "phase_memory": {"loading": {"allocated_peak_bytes": 2000},
                             "warmed_inference": {"allocated_peak_bytes": peak_bytes}},
            "limitations": ["Synthetic fixture measurements are software checks only."],
        }
        metadata = {
            **self.identity, "component": component, "condition_key": condition,
            "execution_kind": "injected_fixture", "selection_hash": self.selection["selection_hash"],
            "model_key": MODEL, "checkpoint_revision": "independent-fixture",
            "settings": {"device": "cpu", "quantization": "fixture"},
            "sam_settings": copy_settings(self.segmentation[condition]["metadata"]["sam_settings"]),
            "measurement_limitations": ["Fixture only"],
        }
        profile = {"summary": summary, "metadata": metadata, "samples": measured_samples}
        self.profiles[relative] = profile
        config = self.config.setdefault("deployment", {"profiles": [], "expected_measured_repeats": 2})
        config["profiles"].append({"summary_path": f"{relative}/summary.json",
                                  "metadata_path": f"{relative}/metadata.json",
                                  "samples_path": f"{relative}/samples.jsonl"})
        return profile


def copy_settings(settings: dict) -> dict:
    """A JSON round trip keeps fixture records independently mutable."""
    return json.loads(json.dumps(settings))
