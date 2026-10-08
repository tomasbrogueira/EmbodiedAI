"""Frozen, explicitly role-mapped SAM3 vocabulary on the shared RGB grid.

No controller, references, scheduling, union-mask fusion or model fallback lives
here. Loading remains lazy and the common runner owns geometry and scheduling.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from pipeline_common.contracts import (
    FramePacket, InstanceRecord, QueryRecord, SemanticFrame, validate_semantic,
)

PIPELINE_ID = "fixed_hazards"
POLICY_ID = "visible_avoid_concepts_v1"
LEGACY_VOCABULARY_MODE = "legacy_hazards_v1"
EXPLICIT_VOCABULARY_MODE = "explicit_concepts_v1"
CANONICAL_PROMPTS = (
    "tree", "pole", "water", "vehicle", "building", "log", "person", "fence",
    "bush", "barrier", "mud", "rubble", "cup", "bottle", "chair", "dog",
)
SCORE_MEANING = "raw_sam3_detection_score; uncalibrated_evidence_weight"
_ROOT = Path(__file__).resolve().parents[2]
_POLICY_PATH = _ROOT / "VLM_evaluation/configs/hazards/policy.json"


class _LoadUnavailable(RuntimeError):
    """Replay a load diagnostic without retaining model-bearing tracebacks."""
    def __init__(self, diagnostic):
        self.error_code = diagnostic["code"]
        super().__init__(diagnostic["message"])


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _portable(value):
    if isinstance(value, dict):
        return {key: _portable(item) for key, item in value.items()
            if not key.startswith("_") and key not in {
                "processor", "cache_root", "code_source_root", "checkpoint_path", "policy_path"}}
    if isinstance(value, (tuple, list)):
        return [_portable(item) for item in value]
    return value


def _query_specs(policy, vocabulary_mode):
    """Keep the historical policy strict; custom words require explicit roles."""
    if not isinstance(policy, dict) or (policy.get("task_id") != "hazard_prompt_v1"
        or type(policy.get("schema_version")) is not int or policy["schema_version"] != 1):
        raise ValueError("Fixed vocabulary policy task/schema mismatch")
    if vocabulary_mode == LEGACY_VOCABULARY_MODE:
        if (policy.get("policy_id") != POLICY_ID
            or policy.get("canonical_prompts") != list(CANONICAL_PROMPTS)):
            raise ValueError("fixed_hazards requires the frozen ordered sixteen-concept policy")
        specs = [{"phrase": phrase, "concept_id": phrase, "role": "hazard"}
                 for phrase in CANONICAL_PROMPTS]
    elif vocabulary_mode == EXPLICIT_VOCABULARY_MODE:
        policy_id = policy.get("policy_id")
        if (not isinstance(policy_id, str) or not policy_id.strip() or policy_id == POLICY_ID
            or policy.get("vocabulary_mode") != EXPLICIT_VOCABULARY_MODE):
            raise ValueError("Explicit vocabulary requires its own versioned policy identity")
        specs = policy.get("concepts")
        if not isinstance(specs, list) or not 1 <= len(specs) <= 32:
            raise ValueError("Explicit vocabulary requires 1..32 ordered concept records")
        roles = {}
        phrases = []
        for spec in specs:
            if not isinstance(spec, dict) or set(spec) != {"phrase", "concept_id", "role"}:
                raise ValueError("Each concept requires exactly phrase, concept_id and role")
            phrase, concept, role = spec["phrase"], spec["concept_id"], spec["role"]
            if (not isinstance(phrase, str) or not phrase.strip() or phrase != phrase.strip()
                or len(phrase) > 80 or not isinstance(concept, str) or not concept.strip()
                or concept != concept.strip() or not isinstance(role, str)
                or role not in {"hazard", "candidate_surface"}):
                raise ValueError("Invalid explicit vocabulary phrase/concept/role")
            if role != "hazard" and (phrase in CANONICAL_PROMPTS or concept in CANONICAL_PROMPTS):
                raise ValueError("Historical hazard concepts cannot become candidate surfaces")
            if concept in roles and roles[concept] != role:
                raise ValueError("One concept cannot have conflicting semantic roles")
            roles[concept] = role
            phrases.append(phrase)
        if len(set(phrases)) != len(phrases) or policy.get("canonical_prompts") != phrases:
            raise ValueError("Explicit vocabulary phrases must be unique and match the ordered inventory")
    else:
        raise ValueError("Unknown fixed vocabulary mode")
    phrases = [spec["phrase"] for spec in specs]
    aliases = policy.get("aliases")
    if not isinstance(aliases, dict) or set(aliases) != set(phrases) or any(
        not isinstance(aliases[phrase], list) or phrase not in aliases[phrase]
        or any(not isinstance(alias, str) or not alias.strip() for alias in aliases[phrase])
        for phrase in phrases):
        raise ValueError("Frozen canonical concept mapping is incomplete")
    vague = policy.get("vague_unusable_phrases")
    if not isinstance(vague, list) or any(not isinstance(phrase, str) or not phrase.strip() for phrase in vague):
        raise ValueError("Policy requires explicit vague_unusable_phrases")
    return tuple(deepcopy(spec) for spec in specs)


class FixedHazardsAdapter:
    """One image encoding and the frozen original phrases per selected frame."""

    def __init__(self, config: dict):
        if not isinstance(config, dict):
            raise ValueError("fixed_hazards config must be an object")
        options = config.get("pipeline", config.get("pipeline_settings", config))
        if not isinstance(options, dict):
            raise ValueError("pipeline_settings must be an object")
        for owner in (config, options):
            version = owner.get("schema_version", 1)
            if (owner.get("contract_id", "semantic_mapping_v1") != "semantic_mapping_v1"
                or type(version) is not int or version != 1
                or owner.get("pipeline_id", PIPELINE_ID) != PIPELINE_ID):
                raise ValueError("fixed_hazards contract/schema/pipeline identity mismatch")
        self.vocabulary_mode = options.get("vocabulary_mode", LEGACY_VOCABULARY_MODE)
        if self.vocabulary_mode == EXPLICIT_VOCABULARY_MODE and not options.get("policy_path"):
            raise ValueError("Explicit vocabulary requires a policy_path")
        self.fixture = config.get("fixture", options.get("fixture", False))
        if type(self.fixture) is not bool:
            raise ValueError("fixture must be boolean")
        policy_path = Path(options.get("policy_path") or _POLICY_PATH).expanduser()
        if not policy_path.is_absolute():
            policy_path = _ROOT / policy_path
        self.policy_path = policy_path.resolve()
        policy_bytes = self.policy_path.read_bytes()
        self.policy = json.loads(policy_bytes)
        self.query_specs = _query_specs(self.policy, self.vocabulary_mode)
        self.prompts = tuple(spec["phrase"] for spec in self.query_specs)
        self.policy_id = self.policy["policy_id"]
        if "prompts" in options and options["prompts"] != list(self.prompts):
            raise ValueError("fixed_hazards cannot prefilter or reorder the frozen prompts")
        if options.get("policy_id", self.policy_id) != self.policy_id:
            raise ValueError("Fixed vocabulary policy identity mismatch")
        aliases = self.policy["aliases"]
        self.policy_provenance = {
            "policy_id": self.policy_id, "policy_version": 1,
            "component_protocol_id": "hazard_prompt_v1",
            "canonical_prompts": list(self.prompts),
            "policy_hash": _digest(self.policy),
            "policy_file_sha256": hashlib.sha256(policy_bytes).hexdigest(),
            "alias_hash": _digest(aliases),
            "hash_recipes": {"policy_hash": "sha256_canonical_json_utf8",
                "policy_file_sha256": "sha256_encoded_file_bytes",
                "alias_hash": "sha256_canonical_json_utf8"},
        }
        if self.vocabulary_mode == EXPLICIT_VOCABULARY_MODE:
            self.policy_provenance.update(vocabulary_mode=self.vocabulary_mode,
                                          concepts=deepcopy(list(self.query_specs)))
        for key in ("policy_hash", "policy_file_sha256", "alias_hash"):
            if key in options and options[key] != self.policy_provenance[key]:
                raise ValueError(f"Frozen {key} mismatch")
        self.sam_settings = deepcopy(options.get("sam_settings", {}))
        if not isinstance(self.sam_settings, dict):
            raise ValueError("sam_settings must be an object")
        if (self.sam_settings.get("allow_downloads", False) is not False
            or self.sam_settings.get("local_files_only", True) is not True):
            raise ValueError("Mapping inference requires pre-cached SAM3 weights; downloads are disabled")
        if "enable_model_loading" in options:
            if type(options["enable_model_loading"]) is not bool:
                raise ValueError("enable_model_loading must be boolean")
            if "enable_model_loading" in self.sam_settings and self.sam_settings["enable_model_loading"] != options["enable_model_loading"]:
                raise ValueError("SAM model-loading flags disagree")
            self.sam_settings["enable_model_loading"] = options["enable_model_loading"]
        if "fixture" in self.sam_settings and self.sam_settings["fixture"] is not self.fixture:
            raise ValueError("SAM fixture provenance mismatch")
        self.sam_settings.update(fixture=self.fixture, allow_downloads=False, local_files_only=True,
            policy_path=str(self.policy_path),
            policy_id=self.policy_id, policy_hash=self.policy_provenance["policy_hash"],
            alias_hash=self.policy_provenance["alias_hash"])
        # Private injection is deliberately restricted to marked CPU fixtures.
        self._segmenter = options.get("_segmenter", config.get("_segmenter"))
        self._owned = False
        self._closed = False
        self._load_duration_ns = None
        self._loading_this_call_ns = 0
        self._load_error = None
        if self._segmenter is not None:
            if not self.fixture:
                raise ValueError("Injected segmenters require fixture=true")
            self._check_fixture(self._segmenter)

    @staticmethod
    def _check_fixture(segmenter):
        metadata = getattr(segmenter, "metadata", {})
        settings = getattr(segmenter, "settings", {})
        markers = [record["fixture"] for record in (metadata, settings) if "fixture" in record]
        identity = (getattr(segmenter, "fixture_identity", None)
            or metadata.get("fixture_identity") or settings.get("fixture_identity"))
        if not markers or any(marker is not True for marker in markers) or not isinstance(identity, str) or not identity:
            raise ValueError("Injected segmenter requires stable explicit fixture provenance")

    def _get_segmenter(self):
        if self._closed:
            raise RuntimeError("fixed_hazards adapter is closed")
        if self._load_error is not None:
            raise _LoadUnavailable(self._load_error)
        if self._segmenter is None:
            source = str(_ROOT / "VLM_evaluation/src")
            if source not in sys.path:
                sys.path.insert(0, source)
            from traversability_hazard_segmentation import load_segmenter
            load_started = time.monotonic_ns()
            try:
                self._segmenter = load_segmenter(self.sam_settings)
            except Exception as error:
                self._load_error = {"code": getattr(error, "error_code", "sam_load_unavailable"),
                    "type": type(error).__name__, "message": str(error)}
                raise
            finally:
                self._load_duration_ns = time.monotonic_ns() - load_started
                self._loading_this_call_ns = self._load_duration_ns
            self._owned = True
        if self.fixture:
            self._check_fixture(self._segmenter)
        return self._segmenter

    def _segmenter_identity(self, segmenter):
        return {"metadata": _portable(deepcopy(getattr(segmenter, "metadata", {}))),
            "settings": _portable(deepcopy(getattr(segmenter, "settings", {}))),
            "fixture_identity": getattr(segmenter, "fixture_identity", None)}

    def _validate_result_settings(self, settings):
        if not isinstance(settings, dict):
            raise ValueError("SAM result settings must be an object")
        if "fixture" in settings and settings["fixture"] is not self.fixture:
            raise ValueError("SAM result fixture provenance mismatch")
        expected = {**self.sam_settings,
            "policy_file_sha256": self.policy_provenance["policy_file_sha256"]}
        actual_metadata = getattr(self._segmenter, "metadata", {})
        for key in ("model_key", "code_revision", "checkpoint_revision", "checkpoint_sha256",
            "resolution", "confidence_threshold", "mask_probability_threshold", "device",
            "policy_id", "policy_hash", "policy_file_sha256", "alias_hash", "fixture_identity"):
            if key in settings:
                for value in (expected, actual_metadata):
                    if key in value and settings[key] != value[key]:
                        raise ValueError(f"SAM actual {key} provenance mismatch")
        if not self.fixture and any(key not in settings for key in (
            "code_revision", "checkpoint_revision", "checkpoint_sha256", "inference_context",
            "device", "policy_hash", "policy_file_sha256", "alias_hash")):
            raise ValueError("Real SAM result requires complete actual model provenance")

    def _provenance(self, frame, result_settings):
        if result_settings is not None:
            self._validate_result_settings(result_settings)
        return {
            "adapter_id": "fixed_hazards_sam3_v1", "pipeline_id": PIPELINE_ID,
            "fixture": self.fixture,
            "execution_kind": "fixture" if self.fixture else "real_model",
            "policy": deepcopy(self.policy_provenance),
            "sam_settings_requested": _portable(self.sam_settings),
            "sam_settings_actual": _portable(result_settings),
            "sam_adapter_identity": self._segmenter_identity(self._segmenter),
            "model_loading": {"duration_ns": self._load_duration_ns,
                "clock": "monotonic", "borrowed": not self._owned},
            "input": {"encoded_file_sha256": frame.encoded_file_sha256,
                "decoded_rgb_sha256": frame.decoded_rgb_sha256,
                "processed_grid_id": frame.processed_grid_id,
                "source_rgb_identity": deepcopy(frame.source_rgb_identity),
                "source_to_processed": deepcopy(frame.source_to_processed),
                "timestamp_provenance": deepcopy(frame.timestamp_provenance)},
            "counts": {"requested_queries": len(self.prompts)},
        }

    def _synchronize(self):
        if not self.fixture:
            torch = importlib.import_module("torch")
            torch.cuda.synchronize(device=self.sam_settings.get("device", "cuda:0"))

    def _failure(self, frame, error, *, stage, calls, count, started, completed, timing_available=True):
        code = getattr(error, "error_code", f"{stage}_failure")
        diagnostic = {"code": code, "stage": stage, "type": type(error).__name__, "message": str(error)}
        unavailable = stage == "model_loading"
        reason = "SAM attempt did not complete with audited actual text execution counts" if count is None else None
        provenance = self._provenance(frame, None)
        provenance.update(availability="unavailable" if unavailable else "available",
            availability_reason=str(error) if unavailable else None,
            query_count_reason=reason,
            timing={"sam_call_ms": (completed - started) / 1e6 if calls and timing_available else None,
                "loading_ms": self._loading_this_call_ns / 1e6, "clock": "monotonic_ns",
                "scope": "fixture_cpu" if self.fixture else "staged_replay",
                "synchronized": timing_available,
                "unavailable_reason": None if timing_available else "selected-device CUDA synchronization failed"})
        provenance["counts"].update(executed_queries=count, successful_queries=0,
            failed_queries=len(self.prompts), query_count_status="unavailable" if count is None else "available")
        identity = self._observation_identity(frame)
        queries = [QueryRecord(query_id=f"fixed_hazards:{identity}:q{index:02d}",
            original_phrase=spec["phrase"], concept_id=spec["concept_id"], role=spec["role"], status="error",
            error=diagnostic, mapping_version=self.policy_id,
            score_metadata={"sam_query_count": count, "query_count_status": "unavailable" if count is None else "available"})
            for index, spec in enumerate(self.query_specs)]
        semantic = SemanticFrame(sequence_id=frame.sequence_id, frame_id=frame.frame_id,
            processed_grid_id=frame.processed_grid_id, decoded_rgb_sha256=frame.decoded_rgb_sha256,
            geometry_fingerprint=frame.geometry.geometry_fingerprint if frame.geometry else None,
            adapter_provenance=provenance, status="error", queries=queries,
            model_calls={"sam3_segment_image": calls}, query_count=count,
            query_count_reason=reason, timestamp_ns=frame.timestamp_ns,
            started_monotonic_ns=started, completed_monotonic_ns=completed, error=diagnostic)
        validate_semantic(semantic, frame)
        return semantic

    def _observation_identity(self, frame):
        return _digest({"sequence_id": frame.sequence_id, "frame_id": frame.frame_id,
            "grid": frame.processed_grid_id, "rgb": frame.decoded_rgb_sha256,
            "policy": self.policy_provenance["policy_hash"]})

    def _translate(self, frame, result, started, completed):
        if not isinstance(result, dict) or not isinstance(result.get("frame"), dict):
            raise ValueError("SAM result must contain frame and query records")
        raw_frame = result["frame"]
        if raw_frame.get("frame_id") != frame.frame_id or raw_frame.get("status") not in {"ok", "error"}:
            raise ValueError("SAM returned mismatched frame identity/status")
        if ((raw_frame["status"] == "ok" and raw_frame.get("error_code") is not None)
            or (raw_frame["status"] == "error" and
                (not isinstance(raw_frame.get("error_code"), str) or not raw_frame["error_code"]))):
            raise ValueError("SAM frame status/error semantics are invalid")
        raw_queries = result.get("queries")
        if not isinstance(raw_queries, list):
            raise ValueError("SAM result queries must be a list")
        # Early image validation errors may have no legacy query records. The
        # explicitly reported zero executions allow diagnostic joins, not masks.
        if (not raw_queries and raw_frame["status"] == "error"
            and raw_frame.get("sam_query_count") == 0):
            raw_queries = [{"phrase": phrase, "masks": [], "scores": [],
                "status": "error", "error_code": raw_frame.get("error_code"),
                "sam_query_count": 0} for phrase in self.prompts]
        if len(raw_queries) != len(self.prompts):
            raise ValueError("SAM returned incomplete frozen query inventory")
        queries, executed = [], 0
        identity = self._observation_identity(frame)
        for index, (spec, raw) in enumerate(zip(self.query_specs, raw_queries)):
            phrase = spec["phrase"]
            if not isinstance(raw, dict) or raw.get("phrase") != phrase or raw.get("status") not in {"ok", "error"}:
                raise ValueError("SAM returned changed phrase order/status")
            status, error, count = raw["status"], raw.get("error_code"), raw.get("sam_query_count")
            if ((status == "ok" and error is not None)
                or (status == "error" and (not isinstance(error, str) or not error))):
                raise ValueError("SAM query error semantics are invalid")
            if type(count) is not int or count not in (0, 1):
                raise ValueError("SAM query must report its actual text-call count")
            if status == "ok" and count != 1:
                raise ValueError("Successful fixed-hazard query must report one actual text execution")
            executed += count
            masks, scores = raw.get("masks"), raw.get("scores")
            if not isinstance(masks, (list, tuple)) or not isinstance(scores, (list, tuple)) or len(masks) != len(scores):
                raise ValueError("SAM instance masks/scores must align")
            query_id = f"fixed_hazards:{identity}:q{index:02d}"
            instances = []
            for instance_index, (mask, score) in enumerate(zip(masks, scores)):
                array = np.asarray(mask)
                if array.dtype != np.bool_ or array.shape != frame.rgb.shape[:2]:
                    raise ValueError("SAM instance must be boolean on the complete processed grid")
                if score is not None and (isinstance(score, (bool, np.bool_))
                    or not isinstance(score, (int, float, np.number)) or not np.isfinite(score) or not 0 <= score <= 1):
                    raise ValueError("SAM score must be null or finite in [0,1]")
                instances.append(InstanceRecord(
                    observation_id=f"{query_id}:i{instance_index:03d}",
                    mask=np.array(array, dtype=np.bool_, copy=True),
                    score=None if score is None else float(score),
                    score_meaning="unavailable" if score is None else SCORE_MEANING,
                    processed_grid_id=frame.processed_grid_id,
                    metadata={"fusion_eligible": status == "ok", "diagnostic_partial": status == "error"},
                ))
            queries.append(QueryRecord(query_id=query_id, original_phrase=phrase,
                concept_id=spec["concept_id"], role=spec["role"], status=status, error=error,
                instances=instances, mapping_version=self.policy_id,
                score_metadata={"sam_query_count": count, "score_meaning": SCORE_MEANING,
                    "calibrated_traversability_probability": False}))
        if type(raw_frame.get("sam_query_count")) is not int or raw_frame["sam_query_count"] != executed:
            raise ValueError("SAM frame and individual text-call counts disagree")
        failures = sum(query.status == "error" for query in queries)
        if (raw_frame["status"] == "error" and raw_frame["error_code"].startswith("sam_image_error:")
            and failures != len(queries)):
            raise ValueError("SAM image failure cannot expose successful query evidence")
        status = "ok" if failures == 0 else "error" if failures == len(queries) else "partial"
        if raw_frame["status"] == "error" and not failures:
            raise ValueError("SAM aggregate error cannot expose successful query evidence")
        provenance = self._provenance(frame, result.get("settings", {}))
        provenance.update(availability="available", timing={
            "sam_call_ms": (completed - started) / 1e6,
            "loading_ms": self._loading_this_call_ns / 1e6, "clock": "monotonic_ns",
            "scope": "fixture_cpu" if self.fixture else "staged_replay",
            "synchronized": True,
            "boundary": "segment_image; selected-device synchronization before/after genuine calls"})
        provenance["counts"].update(executed_queries=executed,
            successful_queries=len(queries) - failures, failed_queries=failures,
            query_count_status="available")
        semantic = SemanticFrame(sequence_id=frame.sequence_id, frame_id=frame.frame_id,
            processed_grid_id=frame.processed_grid_id, decoded_rgb_sha256=frame.decoded_rgb_sha256,
            geometry_fingerprint=frame.geometry.geometry_fingerprint if frame.geometry else None,
            adapter_provenance=provenance, status=status, queries=queries,
            model_calls={"sam3_segment_image": 1}, query_count=executed,
            timestamp_ns=frame.timestamp_ns, started_monotonic_ns=started,
            completed_monotonic_ns=completed,
            error=raw_frame.get("error_code") if status != "ok" else None)
        validate_semantic(semantic, frame)
        return semantic

    def observe(self, frame: FramePacket) -> SemanticFrame:
        from pipeline_common.input_bridge import to_model_input
        if self._closed:
            raise RuntimeError("fixed_hazards adapter is closed")
        self._loading_this_call_ns = 0
        model_frame, data_root = to_model_input(frame)
        # The shared bridge includes nested packet provenance. A backend owns
        # neither that provenance nor this packet, even if it mutates its input.
        model_frame = deepcopy(model_frame)
        model_frame["fixture"] = self.fixture
        input_identity = _digest(model_frame)
        loading_started = time.monotonic_ns()
        try:
            segmenter = self._get_segmenter()
        except Exception as error:
            return self._failure(frame, error, stage="model_loading", calls=0, count=0,
                started=loading_started, completed=time.monotonic_ns())
        adapter_identity = self._segmenter_identity(segmenter)
        synchronization_started = time.monotonic_ns()
        try:
            self._synchronize()
        except Exception as error:
            return self._failure(frame, error, stage="sam3_synchronization", calls=0, count=0,
                started=synchronization_started, completed=time.monotonic_ns(), timing_available=False)
        started = time.monotonic_ns()
        try:
            result = segmenter.segment_image(model_frame, list(self.prompts), data_root)
        except Exception as error:
            synchronized = True
            try:
                self._synchronize()
            except Exception:
                synchronized = False
            return self._failure(frame, error, stage="sam3_segment_image", calls=1, count=None,
                started=started, completed=time.monotonic_ns(), timing_available=synchronized)
        try:
            self._synchronize()
        except Exception as error:
            return self._failure(frame, error, stage="sam3_synchronization", calls=1, count=None,
                started=started, completed=time.monotonic_ns(), timing_available=False)
        completed = time.monotonic_ns()
        # Both persisted bytes and decoded pixels must still describe the same
        # immutable packet after the file-backed model read.
        current, _ = to_model_input(frame)
        current = deepcopy(current)
        current["fixture"] = self.fixture
        if _digest(current) != input_identity or _digest(model_frame) != input_identity:
            raise ValueError("SAM input/frame provenance changed during inference")
        if self._segmenter_identity(segmenter) != adapter_identity:
            raise ValueError("SAM model/settings identity changed during inference")
        return self._translate(frame, result, started, completed)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self._owned and self._segmenter is not None:
                self._segmenter.close()
        finally:
            self._segmenter = None
            self._load_error = None


def create_adapter(config: dict) -> FixedHazardsAdapter:
    return FixedHazardsAdapter(config)
