"""One frozen SAM3 surface phrase on the shared LingBot RGB grid.

The previous Path mapper supplies the alignment/archive logic. The compatible
verified SAM3 reader supplies its per-query processor interface; no model, map,
planner or scheduler is reimplemented here. A mask is candidate surface evidence.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from pipeline_common.contracts import (
    CONTRACT_ID, SCHEMA_VERSION, FramePacket, InstanceRecord, QueryRecord,
    SemanticFrame, validate_semantic,
)

PIPELINE_ID = "ground_surface"
ADAPTER_ID = "ground_surface_sam3_v1"
CONCEPT_VERSION = "ground_surface_v1"
SCORE_MEANING = "SAM3 instance confidence; uncalibrated semantic evidence weight"
PROMPT_CONFIG_IDS = {
    "ground": "ground_surface_outdoor_v1",
    "floor": "ground_surface_indoor_v1",
    "Path": "ground_surface_path_legacy_ablation_v1",
}
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _settings(config: dict) -> dict:
    if not isinstance(config, dict):
        raise ValueError("ground_surface config must be a dictionary")
    configured = deepcopy({key: value for key, value in config.items()
                           if key != "_segmenter"})
    if configured.get("pipeline_id", PIPELINE_ID) != PIPELINE_ID:
        raise ValueError("ground_surface pipeline identity mismatch")
    version = configured.get("schema_version", SCHEMA_VERSION)
    if configured.get("contract_id", CONTRACT_ID) != CONTRACT_ID or type(version) is not int or version != SCHEMA_VERSION:
        raise ValueError("ground_surface contract/schema identity mismatch")
    phrase = configured.get("prompt", "ground")
    if not isinstance(phrase, str) or phrase not in PROMPT_CONFIG_IDS:
        raise ValueError("Choose one exact frozen surface prompt: ground, floor or Path")
    configured["prompt"] = phrase
    expected_id = PROMPT_CONFIG_IDS[phrase]
    if configured.get("config_id", expected_id) != expected_id:
        raise ValueError("Surface prompt/config identity mismatch")
    configured["config_id"] = expected_id
    for key, expected in (("adapter_id", ADAPTER_ID), ("concept_id", PIPELINE_ID),
                          ("concept_mapping_version", CONCEPT_VERSION), ("role", "candidate_surface")):
        if key in configured and configured[key] != expected:
            raise ValueError(f"Surface {key} identity mismatch")
    configured["pipeline_id"] = PIPELINE_ID
    configured["adapter_id"] = ADAPTER_ID
    configured["concept_id"] = PIPELINE_ID
    configured["concept_mapping_version"] = CONCEPT_VERSION
    configured["role"] = "candidate_surface"
    if type(configured.get("fixture", False)) is not bool:
        raise ValueError("fixture must be boolean")
    configured.setdefault("fixture", False)
    selection = configured.setdefault("prompt_selection", {
        "frozen": True, "basis": "starting_proposal_unmeasured", "selection_split": "development",
    })
    if not isinstance(selection, dict) or selection.get("frozen") is not True:
        raise ValueError("Freeze one surface prompt before replay/test evaluation")
    if selection.get("selection_split", "development") != "development":
        raise ValueError("Surface prompts may only be selected on development data")
    sam = configured.setdefault("sam3", {})
    if not isinstance(sam, dict):
        raise ValueError("sam3 settings must be a dictionary")
    if sam.get("allow_downloads", False) is not False or sam.get("local_files_only", True) is not True:
        raise ValueError("ground_surface inference requires local-only weights; downloads are not supported")
    if type(sam.get("fixture", configured["fixture"])) is not bool or sam.get("fixture", configured["fixture"]) != configured["fixture"]:
        raise ValueError("SAM3/adapter fixture identity mismatch")
    sam["fixture"] = configured["fixture"]
    sam["allow_downloads"] = False
    sam["local_files_only"] = True
    return configured


def development_prompt_variants(config: dict, sequence_manifest: dict) -> list[dict]:
    """Declare ground/floor/Path runs on the same development frames/settings.

    No reference is read and no best-prompt selection is performed. The caller
    evaluates each declared run with the common evaluator and freezes its choice.
    """
    if sequence_manifest.get("split") != "development":
        raise ValueError("Prompt comparison is development-only")
    if sequence_manifest.get("contract_id") != CONTRACT_ID or sequence_manifest.get("schema_version") != 1:
        raise ValueError("Development sequence contract/schema mismatch")
    frames = sequence_manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Prompt comparison needs frozen development frames")
    base = _settings(config)
    frame_digest = _digest(frames)
    variants = []
    for phrase, config_id in PROMPT_CONFIG_IDS.items():
        variant = deepcopy(base)
        variant.update(prompt=phrase, config_id=config_id)
        variant["prompt_selection"] = {
            "frozen": True, "basis": "declared_development_ablation",
            "selection_split": "development", "comparison_frame_digest": frame_digest,
        }
        variants.append(variant)
    return variants


def _sam_module():
    """Use the existing verified processor without importing Torch at module load."""
    component_sources = str(_REPO_ROOT / "VLM_evaluation" / "src")
    if component_sources not in sys.path:
        sys.path.insert(0, component_sources)
    return importlib.import_module("traversability_hazard_segmentation.sam3_adapter")


class GroundSurfaceAdapter:
    def __init__(self, config: dict):
        self.config = _settings(config)
        injected = config.get("_segmenter")
        if injected is not None and not self.config["fixture"]:
            raise ValueError("Injected segmenters require explicit fixture=true")
        if injected is not None:
            metadata, settings = getattr(injected, "metadata", {}), getattr(injected, "settings", {})
            markers = [owner["fixture"] for owner in (metadata, settings) if "fixture" in owner]
            identity = getattr(injected, "fixture_identity", None) or metadata.get("fixture_identity")
            if not markers or any(marker is not True for marker in markers) or not isinstance(identity, str) or not identity:
                raise ValueError("Injected SAM reader must declare fixture=true and fixture_identity")
        self._segmenter = injected
        self._closed = False
        self._has_called = False
        self.config_digest = _digest(self.config)

    def _load(self):
        if self._segmenter is None:
            self._segmenter = _sam_module().load_segmenter(self.config["sam3"])
        return self._segmenter

    def _synchronize(self):
        if not self.config["fixture"]:
            # load_segmenter has already verified this explicit CUDA device.
            importlib.import_module("torch").cuda.synchronize(self.config["sam3"].get("device", "cuda:0"))

    def observe(self, frame: FramePacket) -> SemanticFrame:
        from pipeline_common.input_bridge import to_model_input

        started = time.monotonic_ns()
        query_id = "surface:" + _digest({"sequence": frame.sequence_id, "frame": frame.frame_id,
            "grid": frame.processed_grid_id, "rgb": frame.decoded_rgb_sha256,
            "config": self.config_digest})[:32]
        phrase = self.config["prompt"]
        queries = []
        model_calls = {"sam3": 0}
        query_count = 0
        query_count_reason = None
        error = None
        status = "error"
        metadata = {}
        instances = []
        loading_ms = None
        call_ms = None
        cold_call = not self._has_called
        try:
            if self._closed:
                raise RuntimeError("ground_surface_adapter_closed")
            # The common bridge re-decodes the lossless PNG and verifies both
            # hashes/pixels before the existing file-backed model reader runs.
            record, data_root = to_model_input(frame)
            load_started = time.monotonic_ns()
            needed_load = self._segmenter is None
            segmenter = self._load()
            self._synchronize()
            loading_ms = (time.monotonic_ns() - load_started) / 1e6 if needed_load else 0.0
            metadata = dict(getattr(segmenter, "metadata", {}))
            model_calls["sam3"] = 1  # one segment_image invocation, including failures
            query_count = None  # Until the backend returns audited execution counts.
            call_started = time.monotonic_ns()
            try:
                raw = segmenter.segment_image(record, [phrase], data_root)
            finally:
                self._synchronize()
                call_ms = (time.monotonic_ns() - call_started) / 1e6
                self._has_called = True
            raw_queries = raw.get("queries", [])
            raw_frame = raw.get("frame", {})
            query_count = raw_frame.get("sam_query_count")
            if type(query_count) is not int or query_count < 0 or query_count > 1:
                query_count = None
                raise ValueError("Expected zero or one actual surface text-query execution")
            if len(raw_queries) != 1 or raw_queries[0].get("phrase") != phrase:
                raise ValueError("SAM3 did not return the single original surface query")
            raw_query = raw_queries[0]
            if raw_query.get("status") not in {"ok", "error"}:
                raise ValueError("Invalid SAM3 query status")
            if raw_frame.get("status") not in {"ok", "error"}:
                raise ValueError("Invalid SAM3 frame status for one surface query")
            if type(raw_query.get("sam_query_count")) is not int or raw_query["sam_query_count"] != query_count:
                query_count = None
                raise ValueError("SAM3 frame/query execution counts disagree")
            if raw_query["status"] == "ok" and query_count != 1:
                query_count = None
                raise ValueError("Successful surface query must report one actual text execution")
            query_error = raw_query.get("error_code")
            query_status = raw_query["status"]
            if raw_frame["status"] == "error" and query_status == "ok":
                query_status = "error"
                query_error = raw_frame.get("error_code") or "sam_image_or_frame_error"
            masks, scores = raw_query.get("masks", []), raw_query.get("scores", [])
            if len(masks) != len(scores):
                raise ValueError("SAM3 instance mask/score count mismatch")
            for index, (mask, score) in enumerate(zip(masks, scores)):
                mask = np.asarray(mask)
                if mask.dtype != np.bool_ or mask.shape != frame.rgb.shape[:2]:
                    raise ValueError("SAM3 surface mask must be boolean on the shared processed RGB grid")
                if score is not None and (isinstance(score, bool) or not np.isfinite(score) or not 0 <= score <= 1):
                    raise ValueError("SAM3 instance score must be null or finite [0,1] evidence weight")
                instances.append(InstanceRecord(
                    observation_id=f"{query_id}:instance:{index}", mask=mask.copy(),
                    score=None if score is None else float(score), score_meaning=SCORE_MEANING,
                    processed_grid_id=frame.processed_grid_id,
                    metadata={"diagnostic_only": query_status != "ok"},
                ))
            # An image-level failure can never convert retained partial output
            # into usable surface evidence. Legacy frame aggregate error alone
            # is insufficient for multi-query status, but this adapter has one.
            queries = [QueryRecord(
                query_id=query_id, original_phrase=phrase, concept_id=PIPELINE_ID,
                role="candidate_surface", status=query_status, error=query_error,
                instances=instances, mapping_version=CONCEPT_VERSION,
                score_metadata={"meaning": SCORE_MEANING, "calibrated_probability": False},
            )]
            status = "ok" if query_status == "ok" else "error"
            error = query_error if status == "error" else None
        except Exception as exc:
            status = "error"
            error = {"code": getattr(exc, "error_code", "surface_observation_failed"),
                     "type": type(exc).__name__, "message": str(exc)}
            if query_count is None:
                query_count_reason = "Backend call did not return a trustworthy execution count: " + str(exc)
            instances = [replace(instance, metadata={**instance.metadata, "diagnostic_only": True})
                         for instance in instances]
            queries = [QueryRecord(query_id=query_id, original_phrase=phrase,
                concept_id=PIPELINE_ID, role="candidate_surface", status="error", error=error,
                instances=instances, mapping_version=CONCEPT_VERSION)]
        provenance = {
            "pipeline_id": PIPELINE_ID, "adapter_id": ADAPTER_ID,
            "config_id": self.config["config_id"], "config_digest": self.config_digest,
            "policy": {"policy_id": "single_candidate_surface_v1", "prompt": phrase,
                "concept_mapping_version": CONCEPT_VERSION,
                "prompt_selection": self.config["prompt_selection"],
                "meaning": "candidate ground/floor surface; geometry and robot checks required"},
            "model": metadata, "configured_model": self.config["sam3"],
            "timing": {"loading_ms": loading_ms, "sam_call_ms": call_ms,
                "clock": "monotonic_ns", "cold_call": cold_call,
                "boundary": "segment_image after load; includes file validation/image encoding/text query/CPU capture",
                "gpu_synchronized": call_ms is not None and not self.config["fixture"], "warmup_performed": False},
            "fixture": self.config["fixture"],
            "input": {"encoded_file_sha256": frame.encoded_file_sha256,
                "decoded_rgb_sha256": frame.decoded_rgb_sha256,
                "processed_grid_id": frame.processed_grid_id,
                "source_rgb_identity": frame.source_rgb_identity,
                "source_to_processed": frame.source_to_processed,
                "timestamp_provenance": frame.timestamp_provenance},
            "legacy_compatibility": {"ablation": phrase == "Path",
                "legacy_pipeline_id": "sam3_lingbot_path_v1",
                "legacy_archive_imported": False},
        }
        result = SemanticFrame(
            sequence_id=frame.sequence_id, frame_id=frame.frame_id,
            processed_grid_id=frame.processed_grid_id, decoded_rgb_sha256=frame.decoded_rgb_sha256,
            geometry_fingerprint=frame.geometry.geometry_fingerprint if frame.geometry else None,
            adapter_provenance=provenance, status=status, queries=queries,
            model_calls=model_calls, query_count=query_count, timestamp_ns=frame.timestamp_ns,
            query_count_reason=query_count_reason,
            started_monotonic_ns=started, completed_monotonic_ns=time.monotonic_ns(), error=error,
        )
        validate_semantic(result, frame)
        return result

    def close(self) -> None:
        if not self._closed:
            try:
                if self._segmenter is not None:
                    self._segmenter.close()
            finally:
                self._segmenter = None
                self._closed = True


def create_adapter(config: dict) -> GroundSurfaceAdapter:
    return GroundSurfaceAdapter(config)


def inspect_legacy_path_archive(geometry_path, scores_path, *, expected_geometry_fingerprint=None) -> dict:
    """Verify/read an old Path archive without relabeling its aggregate evidence.

    This compatibility audit performs no semantic replay: instance identities,
    per-query statuses and checkpoint hashes cannot be recovered from old scores.
    """
    from path_mapping.runner import geometry_fingerprint, load_geometry, load_scores, validate_geometry

    geometry, sources, geometry_metadata = load_geometry(Path(geometry_path))
    validate_geometry(geometry, allow_single_frame=True)
    fingerprint = geometry_fingerprint(geometry)
    if expected_geometry_fingerprint is not None and fingerprint != expected_geometry_fingerprint:
        raise ValueError("Legacy geometry fingerprint does not match the requested geometry")
    scores, sam_metadata = load_scores(Path(scores_path), geometry, "Path")
    return {"legacy_pipeline_id": "sam3_lingbot_path_v1", "legacy_schema_version": 1,
        "prompt": "Path", "geometry_fingerprint": fingerprint, "source_paths": sources,
        "processed_shape": list(scores.shape), "geometry_metadata": geometry_metadata,
        "sam_metadata": sam_metadata, "geometry_archive_sha256": _file_sha256(Path(geometry_path)),
        "scores_archive_sha256": _file_sha256(Path(scores_path)),
        "representation": "dense_maximum_over_instance_score_aggregate",
        "usable_for_new_pipeline_fusion": False,
        "reason": "Old archives lack original per-query/instance observations; no implicit ground relabeling",
        "original_instances_available": False, "original_query_status_available": False}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
