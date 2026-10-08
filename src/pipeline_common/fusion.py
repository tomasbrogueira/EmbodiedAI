"""Surface-only, frame-bound fusion shared by all four mapping pipelines.

Weights are evidence statistics, never calibrated probabilities. Within a
voxel, correlated geometry pixels contribute their maximum raw confidence once
per frame. Successful queries for one concept are combined by pixelwise maximum
instance weight, then capped to one maximum weighted positive and one maximum
observed weight per voxel/frame/concept. Aliases cannot increase frame support.

Expiry uses the caller's capture-clock ``now_timestamp_ns`` only. It filters the
active semantic export without deleting retained contributions or geometry.
With expiry enabled, missing capture times are temporally unknown and omitted.
No ray visibility model is implemented: missing voxels are unknown, and no
observed-free-space state is ever emitted.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import numbers
from typing import Any

import numpy as np

from .contracts import CONTRACT_ID, SCHEMA_VERSION, FramePacket, SemanticFrame, validate_semantic


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Fusion provenance must be finite JSON data") from error


def _digest(*parts: str) -> str:
    return hashlib.sha256(_canonical(parts).encode("utf-8")).hexdigest()


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(_canonical([array.dtype.str, list(array.shape)]).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _real_array(value: Any, label: str, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "iuf":
        raise ValueError(f"{label} must contain real numbers with shape {shape}")
    return array.astype(np.float64, copy=False)


def _timestamp(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or not 0 <= value <= np.iinfo(np.int64).max:
        raise ValueError(f"{label} must be null or a nonnegative int64 capture timestamp")
    return int(value)


def _latest(old: int | None, new: int | None) -> int | None:
    if old is None:
        return new
    return old if new is None else max(old, new)


@dataclass
class _FrameGeometry:
    identity: str
    points: np.ndarray
    confidence: np.ndarray
    pixel_indices: np.ndarray
    voxel_indices: np.ndarray
    inverse: np.ndarray
    weights: np.ndarray
    shape: tuple[int, int]
    timestamp_ns: int | None


class VoxelFuser:
    """Retain exact frame contributions and export deterministic numeric schemas.

    ``add_geometry(frame)`` and ``add_semantics(result, frame)`` are idempotent
    for identical replays. Reuse of a frame ID with changed geometry, pixels,
    metadata, pose or semantic observations is rejected. Pose relocation is not
    supported. ``add_semantics`` also ensures its geometry has been added.

    ``export(now_timestamp_ns=None)`` returns ``(voxels, evidence, concepts,
    contributions, metadata)``. If configured expiry is non-null, the caller
    must supply ``now_timestamp_ns`` in the same capture-clock domain. Unknown
    timestamps serialize as -1. ``observed_points()`` returns copied, valid raw
    surface positions for the common physical planner.
    """

    def __init__(self, config: dict | None = None):
        config = dict(config or {})
        size = config.get("voxel_size", 0.25)
        if isinstance(size, (bool, np.bool_)) or not isinstance(size, numbers.Real) or not math.isfinite(size) or size <= 0:
            raise ValueError("voxel_size must be positive and finite")
        self.voxel_size = float(size)
        self.origin = _real_array(config.get("origin", [0, 0, 0]), "origin", (3,)).copy()
        if not np.isfinite(self.origin).all():
            raise ValueError("origin must contain three finite coordinates")
        minimum = config.get("min_point_confidence", 0.0)
        if isinstance(minimum, (bool, np.bool_)) or not isinstance(minimum, numbers.Real) or not math.isfinite(minimum) or minimum < 0:
            raise ValueError("min_point_confidence must be finite and nonnegative")
        self.min_point_confidence = float(minimum)
        expiry = config.get("semantic_expiry_ns")
        self.semantic_expiry_ns = _timestamp(expiry, "semantic_expiry_ns")
        negatives = config.get("negative_evidence", True)
        if type(negatives) is not bool:
            raise ValueError("negative_evidence must be boolean")
        self.negative_evidence = negatives
        self._frames: dict[tuple[str, str], _FrameGeometry] = {}
        self._semantic_identities: dict[tuple[str, str], str] = {}
        self._voxels: dict[tuple[int, int, int], dict] = {}
        self._concepts: dict[str, dict] = {}
        self._semantic_contributions: list[dict] = []
        self._journal: list[dict] = []
        self._map_identity: dict | None = None

    def _prepare_geometry(self, frame: FramePacket) -> tuple[_FrameGeometry, dict | None]:
        if not isinstance(frame, FramePacket):
            raise ValueError("Geometry input must be FramePacket")
        rgb = np.asarray(frame.rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3 or min(rgb.shape[:2]) < 1:
            raise ValueError("Frame RGB must be nonempty uint8 [H,W,3]")
        for label in ("sequence_id", "frame_id", "processed_grid_id", "decoded_rgb_sha256", "encoded_file_sha256"):
            if not isinstance(getattr(frame, label), str) or not getattr(frame, label):
                raise ValueError(f"Frame {label} must be a nonempty identity")
        # Same named recipe used by the file-backed bridge: C-order uint8 RGB bytes.
        if hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest() != frame.decoded_rgb_sha256:
            raise ValueError("Decoded RGB byte identity mismatch")
        timestamp = _timestamp(frame.timestamp_ns, "frame.timestamp_ns")
        shape = rgb.shape[:2]
        identity_parts = [frame.sequence_id, frame.frame_id, frame.processed_grid_id,
            frame.decoded_rgb_sha256, frame.encoded_file_sha256, str(frame.image_path),
            timestamp, frame.timestamp_provenance, frame.source_rgb_identity, frame.source_to_processed]
        geometry = frame.geometry
        if geometry is None:
            identity = _digest(_canonical(identity_parts), "geometry_unavailable")
            return _FrameGeometry(identity, np.empty((0, 3)), np.empty(0), np.empty(0, dtype=np.int64),
                np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0), shape, timestamp), None
        if geometry.processed_grid_id != frame.processed_grid_id:
            raise ValueError("Geometry grid identity mismatch")
        for label in ("geometry_fingerprint", "map_frame", "units", "pose_revision"):
            if not isinstance(getattr(geometry, label), str) or not getattr(geometry, label):
                raise ValueError(f"Geometry {label} must be a nonempty identity")
        if geometry.depth_kind not in {"optical_axis", "ray_distance"}:
            raise ValueError("Geometry depth_kind must be optical_axis or ray_distance")
        points = _real_array(geometry.points, "geometry.points", (*shape, 3))
        depth = _real_array(geometry.depth, "geometry.depth", shape)
        confidence = _real_array(geometry.confidence, "geometry.confidence", shape)
        validity = np.asarray(geometry.validity)
        if validity.shape != shape or validity.dtype != np.bool_:
            raise ValueError("Geometry validity must be boolean on the exact RGB grid")
        intrinsic = _real_array(geometry.intrinsics, "geometry.intrinsics", (3, 3))
        if not np.isfinite(intrinsic).all() or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
            raise ValueError("Geometry intrinsics require finite positive focal lengths")
        pose = np.asarray(geometry.world_to_camera)
        if pose.shape not in {(3, 4), (4, 4)} or pose.dtype.kind not in "iuf" or not np.isfinite(pose).all():
            raise ValueError("Geometry W2C pose must be finite [3,4] or [4,4]")
        if pose.shape == (4, 4) and not np.allclose(pose[3], [0, 0, 0, 1], rtol=0, atol=1e-7):
            raise ValueError("Geometry homogeneous W2C pose has invalid last row")
        rotation = pose[:3, :3].astype(np.float64)
        if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-5) or not math.isclose(np.linalg.det(rotation), 1, abs_tol=1e-5):
            raise ValueError("Geometry W2C rotation must be a proper orthonormal rotation")
        up = None
        if geometry.up is not None:
            vector = _real_array(geometry.up, "geometry.up", (3,))
            norm = np.linalg.norm(vector)
            if not np.isfinite(vector).all() or not np.isfinite(norm) or norm == 0:
                raise ValueError("Geometry up must be a finite nonzero direction or null")
            up = (vector / norm).tolist()
        map_identity = {"map_frame": geometry.map_frame, "units": geometry.units,
            "up": up, "scale": json.loads(_canonical(geometry.scale)), "pose_revision": geometry.pose_revision}
        if self._map_identity is not None and map_identity != self._map_identity:
            raise ValueError("Geometry map frame/units/up/scale/pose revision mismatch")
        identity_parts.extend([geometry.geometry_fingerprint, geometry.depth_kind, map_identity])
        identity_parts.extend(_array_digest(np.asarray(value)) for value in
            (geometry.points, geometry.depth, geometry.confidence, geometry.validity, intrinsic, pose))
        identity = _digest(_canonical(identity_parts))
        valid = (validity & np.isfinite(points).all(axis=-1) & np.isfinite(depth) & (depth > 0)
            & np.isfinite(confidence) & (confidence > 0) & (confidence >= self.min_point_confidence))
        pixels = np.flatnonzero(valid)
        observed = points.reshape(-1, 3)[pixels].copy()
        weights = confidence.ravel()[pixels].copy()
        with np.errstate(over="ignore", invalid="ignore"):
            quantized = np.floor((observed - self.origin) / self.voxel_size)
        if not np.isfinite(quantized).all() or (quantized < -(2**63)).any() or (quantized >= 2**63).any():
            raise ValueError("Geometry coordinates produce voxel indices outside int64 range")
        indices, inverse = np.unique(quantized.astype(np.int64), axis=0, return_inverse=True)
        capped = np.zeros(len(indices), dtype=np.float64)
        np.maximum.at(capped, inverse, weights)
        centers = self.origin + (indices.astype(np.float64) + 0.5) * self.voxel_size
        if not np.isfinite(centers).all():
            raise ValueError("Voxel centers overflowed")
        return _FrameGeometry(identity, observed, weights, pixels, indices, inverse, capped, shape, timestamp), map_identity

    def add_geometry(self, frame: FramePacket) -> None:
        prepared, map_identity = self._prepare_geometry(frame)
        frame_key = (frame.sequence_id, frame.frame_id)
        prior = self._frames.get(frame_key)
        if prior is not None:
            if prior.identity != prepared.identity:
                raise ValueError("Replayed frame geometry/pixel/cache/pose identity mismatch")
            return
        # Check all additions before changing state, including numeric overflow.
        updates = []
        for index, weight in zip(prepared.voxel_indices, prepared.weights):
            key = tuple(int(v) for v in index)
            previous = self._voxels.get(key, {"weight": 0.0, "support": 0, "timestamp": None})
            total = previous["weight"] + float(weight)
            if not math.isfinite(total):
                raise ValueError("Accumulated geometry confidence overflowed")
            updates.append((key, {"weight": total, "support": previous["support"] + 1,
                "timestamp": _latest(previous["timestamp"], prepared.timestamp_ns)}))
        self._frames[frame_key] = prepared
        if map_identity is not None and self._map_identity is None:
            self._map_identity = map_identity
        self._voxels.update(updates)
        geometry = frame.geometry
        self._journal.append({"contribution_id": _digest("geometry", *frame_key, prepared.identity),
            "kind": "geometry", "sequence_id": frame.sequence_id, "frame_id": frame.frame_id,
            "timestamp_ns": frame.timestamp_ns, "processed_grid_id": frame.processed_grid_id,
            "decoded_rgb_sha256": frame.decoded_rgb_sha256,
            "geometry_fingerprint": geometry.geometry_fingerprint if geometry else None,
            "pose_revision": geometry.pose_revision if geometry else None,
            "status": "ok" if geometry else "unavailable", "valid_surface_pixels": len(prepared.points),
            "voxel_indices": prepared.voxel_indices.tolist(), "geometry_weight": prepared.weights.tolist()})

    def add_semantics(self, result: SemanticFrame, frame: FramePacket) -> None:
        validate_semantic(result, frame)
        if result.timestamp_ns != frame.timestamp_ns:
            raise ValueError("Semantic capture timestamp identity mismatch")
        # Validate the geometry and concept registry before mutating either map.
        prepared, _ = self._prepare_geometry(frame)
        frame_key = (frame.sequence_id, frame.frame_id)
        if frame_key in self._frames and self._frames[frame_key].identity != prepared.identity:
            raise ValueError("Semantic frame joined changed geometry/pixel/cache/pose identity")
        query_signature = []
        pending_concepts: dict[str, dict] = {}
        for query in result.queries:
            old = pending_concepts.get(query.concept_id, self._concepts.get(query.concept_id))
            if old is not None and (old["role"] != query.role or old["mapping_version"] != query.mapping_version):
                raise ValueError("Concept role/mapping version identity mismatch")
            if not isinstance(query.mapping_version, str) or not query.mapping_version:
                raise ValueError("Query mapping_version must be a nonempty identity")
            phrases = set(old["original_phrases"]) if old else set()
            phrases.add(query.original_phrase)
            pending_concepts[query.concept_id] = {"concept_id": query.concept_id, "role": query.role,
                "mapping_version": query.mapping_version, "original_phrases": sorted(phrases)}
            query_signature.append([query.query_id, query.original_phrase, query.concept_id, query.role,
                query.status, query.error, query.mapping_version, query.score_metadata,
                [[instance.observation_id, _array_digest(instance.mask), instance.score,
                    instance.score_meaning, instance.processed_grid_id, instance.metadata] for instance in query.instances]])
        signature = _digest(prepared.identity, _canonical([result.status, result.adapter_provenance,
            result.model_calls, result.query_count, result.timestamp_ns, result.error, query_signature]))
        old_signature = self._semantic_identities.get(frame_key)
        if old_signature is not None:
            if old_signature != signature:
                raise ValueError("Replayed semantic observation identity mismatch")
            return
        self.add_geometry(frame)
        geometry = frame.geometry
        self._concepts.update(pending_concepts)
        self._semantic_identities[frame_key] = signature
        common = {"sequence_id": frame.sequence_id, "frame_id": frame.frame_id,
            "timestamp_ns": frame.timestamp_ns, "processed_grid_id": frame.processed_grid_id,
            "decoded_rgb_sha256": frame.decoded_rgb_sha256,
            "geometry_fingerprint": geometry.geometry_fingerprint if geometry else None,
            "pose_revision": geometry.pose_revision if geometry else None,
            "adapter_provenance": json.loads(_canonical(result.adapter_provenance))}
        successful: dict[str, list] = {}
        for query in result.queries:
            accepted = result.status in {"ok", "partial"} and query.status == "ok" and geometry is not None
            self._journal.append({**common, "contribution_id": _digest("query", *frame_key, query.query_id, signature),
                "kind": "semantic_query", "query_id": query.query_id, "concept_id": query.concept_id,
                "role": query.role, "mapping_version": query.mapping_version,
                "original_phrase": query.original_phrase, "status": query.status, "frame_status": result.status,
                "error": query.error, "fused": accepted,
                "observation_ids": [instance.observation_id for instance in query.instances],
                "mask_sha256": [_array_digest(instance.mask) for instance in query.instances],
                "instance_scores": [instance.score for instance in query.instances],
                "score_meanings": [instance.score_meaning for instance in query.instances]})
            if accepted:
                successful.setdefault(query.concept_id, []).append(query)
        for concept_id, queries in sorted(successful.items()):
            scores = np.zeros(prepared.shape, dtype=np.float64)
            detected = np.zeros(prepared.shape, dtype=np.bool_)
            for query in queries:
                for instance in query.instances:
                    detected |= instance.mask
                    np.maximum(scores, np.where(instance.mask, 1.0 if instance.score is None else instance.score, 0.0), out=scores)
            pixel_scores = scores.ravel()[prepared.pixel_indices]
            detected_pixels = detected.ravel()[prepared.pixel_indices]
            positive = np.zeros(len(prepared.voxel_indices), dtype=np.float64)
            np.maximum.at(positive, prepared.inverse, prepared.confidence * pixel_scores)
            observed = prepared.weights.copy() if self.negative_evidence else np.zeros_like(positive)
            if not self.negative_evidence:
                np.maximum.at(observed, prepared.inverse[detected_pixels], prepared.confidence[detected_pixels])
            covered = observed > 0
            contribution = {**common, "contribution_id": _digest("semantic", *frame_key, concept_id, signature),
                "kind": "semantic", "concept_id": concept_id, "role": queries[0].role,
                "mapping_version": queries[0].mapping_version, "status": "ok",
                "query_ids": [query.query_id for query in queries],
                "original_phrases": [query.original_phrase for query in queries],
                "observation_ids": [instance.observation_id for query in queries for instance in query.instances],
                "voxel_indices": prepared.voxel_indices[covered].tolist(),
                "positive_weight": positive[covered].tolist(), "observed_weight": observed[covered].tolist()}
            self._semantic_contributions.append(contribution)
            self._journal.append(contribution)

    def observed_points(self) -> np.ndarray:
        """Return valid raw surface coordinates; replay never duplicates points."""
        points = [self._frames[key].points for key in sorted(self._frames)]
        return np.concatenate(points, axis=0) if points else np.empty((0, 3), dtype=np.float64)

    @property
    def points(self) -> np.ndarray:
        return self.observed_points()

    def _evidence_state(self, contribution: dict, now: int | None) -> str:
        if self.semantic_expiry_ns is None:
            return "active"
        captured = contribution["timestamp_ns"]
        if captured is None:
            return "unknown_capture_time"
        if captured > now:
            raise ValueError("Expiry reference precedes a semantic capture timestamp")
        return "stale" if now - captured > self.semantic_expiry_ns else "active"

    def export(self, now_timestamp_ns: int | None = None) -> tuple[dict, dict, list[dict], list[dict], dict]:
        now = _timestamp(now_timestamp_ns, "now_timestamp_ns")
        if self.semantic_expiry_ns is not None and now is None:
            raise ValueError("Semantic expiry requires explicit capture-clock now_timestamp_ns")
        keys = sorted(self._voxels)
        indices = np.asarray(keys, dtype=np.int64).reshape(-1, 3)
        voxels = {"voxel_indices": indices,
            "centers": self.origin + (indices.astype(np.float64) + 0.5) * self.voxel_size,
            "geometry_weight": np.asarray([self._voxels[k]["weight"] for k in keys], dtype=np.float64),
            "frame_support": np.asarray([self._voxels[k]["support"] for k in keys], dtype=np.int64),
            "last_observed_timestamp_ns": np.asarray([self._voxels[k]["timestamp"] if self._voxels[k]["timestamp"] is not None else -1 for k in keys], dtype=np.int64),
            "observation_state": np.full(len(keys), 2, dtype=np.uint8)}
        voxel_rows = {key: row for row, key in enumerate(keys)}
        concepts = [self._concepts[key] for key in sorted(self._concepts)]
        concept_rows = {concept["concept_id"]: row for row, concept in enumerate(concepts)}
        aggregates: dict[tuple[int, int], dict] = {}
        states = {contribution["contribution_id"]: self._evidence_state(contribution, now)
            for contribution in self._semantic_contributions}
        for contribution in self._semantic_contributions:
            if states[contribution["contribution_id"]] != "active":
                continue
            concept_row = concept_rows[contribution["concept_id"]]
            for index, positive, observed in zip(contribution["voxel_indices"], contribution["positive_weight"], contribution["observed_weight"]):
                key = (voxel_rows[tuple(index)], concept_row)
                total = aggregates.setdefault(key, {"positive": 0.0, "observed": 0.0, "support": 0, "timestamp": None})
                total["positive"] += positive
                total["observed"] += observed
                if not math.isfinite(total["positive"]) or not math.isfinite(total["observed"]):
                    raise ValueError("Accumulated semantic confidence overflowed")
                total["support"] += 1
                total["timestamp"] = _latest(total["timestamp"], contribution["timestamp_ns"])
        ordered = sorted(aggregates)
        evidence = {"voxel_row": np.asarray([key[0] for key in ordered], dtype=np.int64),
            "concept_row": np.asarray([key[1] for key in ordered], dtype=np.int64),
            "positive_weight": np.asarray([aggregates[k]["positive"] for k in ordered], dtype=np.float64),
            "observed_weight": np.asarray([aggregates[k]["observed"] for k in ordered], dtype=np.float64),
            "frame_support": np.asarray([aggregates[k]["support"] for k in ordered], dtype=np.int64),
            "last_seen_timestamp_ns": np.asarray([aggregates[k]["timestamp"] if aggregates[k]["timestamp"] is not None else -1 for k in ordered], dtype=np.int64),
            "evidence_score": np.asarray([aggregates[k]["positive"] / aggregates[k]["observed"] for k in ordered], dtype=np.float64)}
        journal = json.loads(_canonical(self._journal))
        for contribution in journal:
            if contribution["contribution_id"] in states:
                contribution["evidence_state"] = states[contribution["contribution_id"]]
        metadata = {"contract_id": CONTRACT_ID, "schema_version": SCHEMA_VERSION,
            **(self._map_identity or {"map_frame": None, "units": None, "up": None, "scale": {}, "pose_revision": None}),
            "voxel_size": self.voxel_size, "origin": self.origin.tolist(),
            "geometry_policy": "maximum_confidence_per_voxel_frame_surface_only_v1",
            "semantic_policy": "maximum_weighted_positive_and_observed_per_voxel_frame_concept_v1",
            "evidence_score_definition": "sum_positive_weight / sum_observed_weight; not probability",
            "unscored_instance_weight": 1.0, "negative_evidence": self.negative_evidence,
            "min_point_confidence": self.min_point_confidence,
            "free_space_capability": False, "sparse_missing_voxel_state": "unknown",
            "semantic_expiry_ns": self.semantic_expiry_ns, "expiry_now_timestamp_ns": now,
            "expiry_clock": "caller_supplied_source_capture_clock",
            "stale_semantic_contributions": sum(state == "stale" for state in states.values()),
            "unknown_time_semantic_contributions": sum(state == "unknown_capture_time" for state in states.values()),
            "expired_evidence_behavior": "omit_active_rows; absence_is_unknown; preserve_geometry_and_journal",
            "unknown_timestamp_numeric_code": -1, "concept_order": "lexicographic_concept_id",
            "voxel_order": "lexicographic_voxel_indices", "frame_count": len(self._frames),
            "semantic_frame_count": len(self._semantic_identities),
            "valid_surface_pixels": sum(len(frame.points) for frame in self._frames.values())}
        return voxels, evidence, json.loads(_canonical(concepts)), journal, metadata
