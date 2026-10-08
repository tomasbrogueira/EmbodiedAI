"""Frozen public records. Arrays are numeric; observation IDs are not tracks."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
import numpy as np

CONTRACT_ID = "semantic_mapping_v1"
SCHEMA_VERSION = 1
PIPELINE_IDS = ("geometry_only", "ground_surface", "fixed_hazards", "qwen_hazards")

@dataclass(frozen=True)
class GeometryFrame:
    points: np.ndarray
    depth: np.ndarray
    validity: np.ndarray
    confidence: np.ndarray
    intrinsics: np.ndarray
    world_to_camera: np.ndarray
    processed_grid_id: str
    geometry_fingerprint: str
    depth_kind: str = "optical_axis"
    map_frame: str = "lingbot_world"
    units: str = "reconstruction_units"
    up: tuple[float, float, float] | None = None
    scale: dict = field(default_factory=dict)
    pose_revision: str = "initial"

@dataclass(frozen=True)
class FramePacket:
    sequence_id: str
    frame_id: str
    timestamp_ns: int | None
    timestamp_provenance: dict
    rgb: np.ndarray
    image_path: Path
    encoded_file_sha256: str
    decoded_rgb_sha256: str
    processed_grid_id: str
    source_rgb_identity: dict
    source_to_processed: dict
    geometry: GeometryFrame | None = None

@dataclass(frozen=True)
class InstanceRecord:
    observation_id: str
    mask: np.ndarray
    score: float | None = None
    score_meaning: str = "unavailable"
    processed_grid_id: str | None = None
    metadata: dict = field(default_factory=dict)

@dataclass(frozen=True)
class QueryRecord:
    query_id: str
    original_phrase: str
    concept_id: str
    role: str
    status: str = "ok"
    error: Any = None
    instances: list[InstanceRecord] = field(default_factory=list)
    score_metadata: dict = field(default_factory=dict)
    mapping_version: str = "v1"

@dataclass(frozen=True)
class SemanticFrame:
    sequence_id: str
    frame_id: str
    processed_grid_id: str
    decoded_rgb_sha256: str
    geometry_fingerprint: str | None
    adapter_provenance: dict
    status: str
    queries: list[QueryRecord] = field(default_factory=list)
    model_calls: dict[str, int] = field(default_factory=dict)
    query_count: int | None = 0
    timestamp_ns: int | None = None
    started_monotonic_ns: int | None = None
    completed_monotonic_ns: int | None = None
    error: Any = None
    query_count_reason: str | None = None
    contract_id: str = CONTRACT_ID
    schema_version: int = SCHEMA_VERSION

class SemanticAdapter(Protocol):
    def observe(self, frame: FramePacket) -> SemanticFrame: ...
    def close(self) -> None: ...

def validate_semantic(result: SemanticFrame, frame: FramePacket) -> None:
    if not isinstance(result, SemanticFrame):
        raise ValueError("Adapter must return SemanticFrame")
    expected_geometry = frame.geometry.geometry_fingerprint if frame.geometry else None
    if type(result.schema_version) is not int or result.schema_version!=1:
        raise ValueError("Semantic schema_version must be integer 1")
    for key, expected in (("contract_id", CONTRACT_ID), ("schema_version", 1),
        ("sequence_id", frame.sequence_id), ("frame_id", frame.frame_id),
        ("processed_grid_id", frame.processed_grid_id),
        ("decoded_rgb_sha256", frame.decoded_rgb_sha256),
        ("geometry_fingerprint", expected_geometry)):
        if getattr(result, key) != expected:
            raise ValueError(f"Semantic {key} identity mismatch")
    if result.status not in {"ok", "partial", "error", "skipped", "not_applicable"}:
        raise ValueError("Invalid semantic status")
    if result.timestamp_ns!=frame.timestamp_ns:
        raise ValueError("Semantic capture timestamp identity mismatch")
    if result.query_count is None:
        if not result.query_count_reason and not result.adapter_provenance.get("query_count_reason") and not result.error:
            raise ValueError("Unavailable query_count requires an explicit reason")
    elif type(result.query_count) is not int or result.query_count < 0:
        raise ValueError("Query count must be a nonnegative integer or unavailable with reason")
    if any(type(n) is not int or n < 0 for n in result.model_calls.values()):
        raise ValueError("Model/query counts must be nonnegative integers")
    query_ids, instance_ids = set(), set()
    for query in result.queries:
        if not query.query_id or query.query_id in query_ids or not query.original_phrase or not query.concept_id:
            raise ValueError("Missing/duplicate query identity")
        query_ids.add(query.query_id)
        if query.role not in {"hazard", "candidate_surface"} or query.status not in {"ok", "error"}:
            raise ValueError("Invalid query role/status")
        for instance in query.instances:
            if not instance.observation_id or instance.observation_id in instance_ids:
                raise ValueError("Missing/duplicate instance identity")
            instance_ids.add(instance.observation_id)
            if instance.mask.dtype != np.bool_ or instance.mask.shape != frame.rgb.shape[:2]:
                raise ValueError("Instance mask must be boolean on the complete processed grid")
            if instance.processed_grid_id not in (None, frame.processed_grid_id):
                raise ValueError("Instance grid identity mismatch")
            if instance.score is not None and (not np.isfinite(instance.score) or not 0 <= instance.score <= 1):
                raise ValueError("Instance score must be null or finite [0,1] evidence weight")
    if result.status == "not_applicable" and (result.queries or sum(result.model_calls.values()) or result.query_count):
        raise ValueError("not_applicable must have no calls/queries")
    statuses={q.status for q in result.queries}
    if result.status=="ok" and "error" in statuses:
        raise ValueError("Successful frame contains failed queries")
    if result.status=="partial" and statuses!={"ok","error"}:
        raise ValueError("Partial frame requires both successful and failed queries")
