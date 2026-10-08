"""Synchronous whole processed-RGB Qwen discovery followed by per-phrase SAM3.

The common runner owns geometry, scheduling, fusion and evaluation. Importing
this module loads neither Torch nor a model. Injected providers are fixture-only.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import sys
import time

import numpy as np

from pipeline_common.contracts import FramePacket, InstanceRecord, QueryRecord, SemanticFrame


MODEL_KEYS = frozenset({"qwen3_5_4b", "qwen3_vl_4b"})
EMPTY_DISCOVERY_POLICIES = frozenset({"skip_sam", "encode_image"})
_COMPONENT_SOURCE = Path(__file__).resolve().parents[2] / "VLM_evaluation" / "src"


def _legacy_imports():
    # The repository supports direct scripts without installing its local packages.
    source = str(_COMPONENT_SOURCE)
    if source not in sys.path:
        sys.path.insert(0, source)
    from traversability_hazard_inference.configuration import (
        MODEL_SPECS, configuration_snapshot, load_policy, normalize_settings,
    )
    from traversability_hazard_inference.records import parse_response, validate_prediction
    return MODEL_SPECS, configuration_snapshot, load_policy, normalize_settings, parse_response, validate_prediction


def _fixture_identity(provider, label):
    metadata = getattr(provider, "metadata", {})
    settings = getattr(provider, "settings", {})
    markers = [owner["fixture"] for owner in (metadata, settings) if "fixture" in owner]
    identity = (getattr(provider, "fixture_identity", None)
                or metadata.get("fixture_identity") or settings.get("fixture_identity"))
    if not markers or any(marker is not True for marker in markers) or not isinstance(identity, str) or not identity:
        raise ValueError(f"Injected {label} requires fixture=true and fixture_identity")
    return identity


def _error(code, error):
    return {"code": code, "type": type(error).__name__, "message": str(error)}


class QwenHazardsAdapter:
    """One selected Qwen and one SAM3, with no internal queue or model fallback."""

    def __init__(self, config: dict):
        if not isinstance(config, dict):
            raise TypeError("Adapter configuration must be a dictionary")
        if "model_keys" in config or "fallback_model_key" in config:
            raise ValueError("Select exactly one of the two 4B Qwen model_key variants")
        self.model_key = config.get("model_key")
        if self.model_key not in MODEL_KEYS:
            raise ValueError("model_key must be qwen3_5_4b or qwen3_vl_4b")
        self.fixture = config.get("fixture", False)
        self.enable_model_loading = config.get("enable_model_loading", False)
        if type(self.fixture) is not bool or type(self.enable_model_loading) is not bool:
            raise ValueError("fixture and enable_model_loading must be boolean")
        self.empty_policy = config.get("empty_discovery_policy", "skip_sam")
        if self.empty_policy not in EMPTY_DISCOVERY_POLICIES:
            raise ValueError("empty_discovery_policy must be skip_sam or encode_image")
        self._backend = config.get("_backend")
        self._segmenter = config.get("_segmenter")
        self._clock = config.get("_clock", time.monotonic_ns)
        if any(key in config for key in ("_backend", "_segmenter", "_clock")) and not self.fixture:
            raise ValueError("Injected providers and clocks require explicit fixture=true")
        if self.fixture and self._backend is None:
            raise ValueError("Fixture execution requires an explicit Qwen backend")
        self._owned_backend = self._owned_segmenter = False
        self._closed = False
        specs, snapshot, load_policy, normalize, _, _ = _legacy_imports()
        self._model_spec = deepcopy(specs[self.model_key])
        for key in ("checkpoint", "revision"):
            if config.get("model", {}).get(key, self._model_spec[key]) != self._model_spec[key]:
                raise ValueError(f"Configured Qwen {key} differs from the frozen variant")
        self.qwen_settings = normalize(self.model_key, config.get("qwen_settings", {}))
        if self.qwen_settings["local_files_only"] is not True:
            raise ValueError("Mapping inference requires pre-cached Qwen weights; downloads are disabled")
        if self.qwen_settings["retry_errors"]:
            raise ValueError("Automatic retries are disabled")
        self.qwen_settings["fixture"] = self.fixture
        self._policy = load_policy(self.qwen_settings)
        self._configuration_snapshot = snapshot(self.model_key, self.qwen_settings)
        if config.get("policy_sha256", self._configuration_snapshot["policy_sha256"]) != self._configuration_snapshot["policy_sha256"]:
            raise ValueError("Frozen policy file hash mismatch")
        self.sam_settings = deepcopy(config.get("sam_settings", {}))
        if self.sam_settings.get("allow_downloads", False) is not False or self.sam_settings.get("local_files_only", True) is not True:
            raise ValueError("Mapping inference requires pre-cached SAM3 weights; downloads are disabled")
        if self.sam_settings.get("device", self.qwen_settings["device"]) != self.qwen_settings["device"]:
            raise ValueError("Qwen and SAM3 must share one explicit CUDA device")
        self.sam_settings.setdefault("device", self.qwen_settings["device"])
        self.sam_settings.setdefault("policy_path", self.qwen_settings["policy_path"])
        for key, expected in (("policy_id", self._policy["policy_id"]),
                              ("policy_hash", self._configuration_snapshot["policy_fingerprint"]),
                              ("alias_hash", self._configuration_snapshot["alias_sha256"])):
            if key in self.sam_settings and self.sam_settings[key] != expected:
                raise ValueError(f"SAM and Qwen {key} differ")
            self.sam_settings[key] = expected
        self.sam_settings.update(fixture=self.fixture, enable_model_loading=self.enable_model_loading,
                                 allow_downloads=False, local_files_only=True)
        self._aliases = {" ".join(alias.split()).casefold(): concept
                         for concept, aliases in self._policy["aliases"].items() for alias in aliases}
        self._mapping_version = self._policy["policy_id"]
        self._fixture_ids = {}
        for label, provider in (("qwen", self._backend), ("sam3", self._segmenter)):
            if provider is not None:
                self._fixture_ids[label] = _fixture_identity(provider, label)
        self.metadata = {
            "adapter": "qwen_hazards", "adapter_version": 1, "model_key": self.model_key,
            "model": deepcopy(self._model_spec), "fixture": self.fixture,
            "execution_kind": "fixture" if self.fixture else "pretrained_inference",
            "fixture_identities": deepcopy(self._fixture_ids),
            "configuration_snapshot": deepcopy(self._configuration_snapshot),
            "empty_discovery_policy": self.empty_policy, "input_space": "lingbot_processed_rgb",
            "qwen_confidence_probability": None,
        }

    def _ensure_backend(self):
        if self._backend is None:
            if not self.enable_model_loading:
                raise RuntimeError("model_loading_disabled: enable_model_loading=true is required")
            from traversability_hazard_inference import load_backend
            self._backend = load_backend(self.model_key, deepcopy(self.qwen_settings))
            self._owned_backend = True
        if getattr(self._backend, "model_key", None) != self.model_key:
            raise ValueError("Loaded Qwen model_key differs from the selected variant")
        metadata = getattr(self._backend, "metadata", {})
        if self.fixture:
            if _fixture_identity(self._backend, "qwen") != self._fixture_ids["qwen"]:
                raise ValueError("Qwen fixture identity changed")
        if not self.fixture:
            for key, expected in (("checkpoint", self._model_spec["checkpoint"]), ("revision", self._model_spec["revision"])):
                if metadata.get(key) != expected:
                    raise ValueError(f"Loaded Qwen {key} differs from its frozen identity")
            if metadata.get("quantization", {}).get("compute_dtype") != "torch.bfloat16":
                raise ValueError("The declared mapping variant requires BF16 floating vision/compute")
        for key in ("policy_sha256", "policy_fingerprint", "alias_sha256", "prompt_sha256"):
            if key in metadata and metadata[key] != self._configuration_snapshot[key]:
                raise ValueError(f"Loaded Qwen {key} differs from the frozen discovery policy")
        return self._backend

    def _ensure_segmenter(self):
        if self._segmenter is None:
            if self.fixture:
                raise RuntimeError("fixture_segmenter_required: supply an explicit fixture SAM provider")
            if not self.enable_model_loading:
                raise RuntimeError("model_loading_disabled: enable_model_loading=true is required")
            from traversability_hazard_segmentation import load_segmenter
            self._segmenter = load_segmenter(deepcopy(self.sam_settings))
            self._owned_segmenter = True
        metadata = getattr(self._segmenter, "metadata", {})
        if self.fixture and _fixture_identity(self._segmenter, "sam3") != self._fixture_ids.get("sam3"):
            raise ValueError("SAM fixture identity changed")
        for key, expected in (("alias_hash", self._configuration_snapshot["alias_sha256"]),
                              ("policy_hash", self._configuration_snapshot["policy_fingerprint"]),
                              ("policy_file_sha256", self._configuration_snapshot["policy_sha256"])):
            if key in metadata and metadata[key] != expected:
                raise ValueError(f"SAM and Qwen {key} differ")
        return self._segmenter

    def _concept(self, phrase):
        normalized = " ".join(phrase.split()).casefold()
        return self._aliases.get(normalized, "unmapped:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest())

    def _synchronize(self):
        if not self.fixture:
            import torch
            torch.cuda.synchronize(self.qwen_settings["device"])

    @contextmanager
    def _stage(self, provenance, name, *, model_call=False, loaded_this_call=None):
        timestamps = {"clock": "monotonic_ns", "start_monotonic_ns": self._clock(),
                      "completion_monotonic_ns": None, "duration_ms": None,
                      "synchronized": False, "timing_valid": False,
                      "boundary": "whole adapter call" if model_call else "model availability/load"}
        if loaded_this_call is not None:
            timestamps["loaded_this_call"] = loaded_this_call
        provenance["stage_timestamps"][name] = timestamps
        entered = False
        try:
            if model_call:
                self._synchronize()
                timestamps["start_monotonic_ns"] = self._clock()
            entered = True
            yield
        except BaseException as error:
            timestamps["error"] = _error("stage_failure", error)
            raise
        finally:
            # Never reset process-wide peaks; the common resource collector owns
            # joint measurements. Synchronize empty image encoding too.
            if model_call:
                try:
                    self._synchronize()
                    timestamps["synchronized"] = not self.fixture
                    timestamps["timing_valid"] = entered
                except Exception as error:
                    timestamps["timing_error"] = _error("synchronization_failure", error)
                    raise
                finally:
                    timestamps["completion_monotonic_ns"] = self._clock()
            else:
                timestamps["completion_monotonic_ns"] = self._clock()
                timestamps["timing_valid"] = True
            if timestamps["timing_valid"]:
                timestamps["duration_ms"] = (timestamps["completion_monotonic_ns"] - timestamps["start_monotonic_ns"]) / 1e6

    def _query_id(self, frame, index, phrase):
        identity = "\0".join((frame.sequence_id, frame.frame_id, frame.processed_grid_id,
                               frame.encoded_file_sha256, frame.decoded_rgb_sha256,
                               self.model_key, str(index), phrase))
        return "qwen_hazards:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def _audit_sam_count(self, returned, frame, prompts):
        """Retain auditable executions even when later mask validation fails."""
        if not isinstance(returned, dict) or not isinstance(returned.get("frame"), dict) or not isinstance(returned.get("queries"), list):
            raise ValueError("SAM query execution records are unavailable")
        row, queries = returned["frame"], returned["queries"]
        actual = row.get("sam_query_count")
        if row.get("frame_id") != frame.frame_id or row.get("status") not in {"ok", "error"}:
            raise ValueError("SAM execution frame identity/status mismatch")
        if type(actual) is not int or not 0 <= actual <= len(prompts):
            raise ValueError("SAM actual text-query count is unavailable or invalid")
        if not queries and row["status"] == "error" and actual == 0:
            return 0
        if len(queries) != len(prompts) or any(not isinstance(query, dict) or query.get("phrase") != phrase for query, phrase in zip(queries, prompts)):
            raise ValueError("SAM execution phrase joins differ from discovery")
        counts = [query.get("sam_query_count") for query in queries]
        if any(type(count) is not int or not 0 <= count <= 1 for count in counts) or sum(counts) != actual:
            raise ValueError("SAM execution counters are invalid or inconsistent")
        return actual

    def _translate_sam(self, returned, frame, prompts):
        if not isinstance(returned, dict) or not isinstance(returned.get("frame"), dict) or not isinstance(returned.get("queries"), list):
            raise ValueError("SAM must return frame and individual query records")
        row, queries = returned["frame"], returned["queries"]
        if row.get("frame_id") != frame.frame_id or row.get("status") not in {"ok", "error"}:
            raise ValueError("SAM frame identity/status mismatch")
        if not queries and row["status"] == "error" and row.get("sam_query_count") == 0:
            queries = [{"phrase": phrase, "masks": [], "scores": [], "status": "error",
                        "error_code": row.get("error_code"), "sam_query_count": 0} for phrase in prompts]
        if len(queries) != len(prompts) or any(not isinstance(query, dict) or query.get("phrase") != phrase for query, phrase in zip(queries, prompts)):
            raise ValueError("SAM queries were missing, reordered or rewritten")
        actual = row.get("sam_query_count")
        if type(actual) is not int or not 0 <= actual <= len(prompts):
            raise ValueError("SAM actual text-query count is unavailable or invalid")
        if (row["status"] == "ok" and row.get("error_code") is not None) or (row["status"] == "error" and (not isinstance(row.get("error_code"), str) or not row["error_code"])):
            raise ValueError("SAM frame status/error mismatch")
        records, individual_count = [], 0
        for index, (query, phrase) in enumerate(zip(queries, prompts)):
            status, error = query.get("status"), query.get("error_code")
            if status not in {"ok", "error"} or (status == "ok" and error is not None) or (status == "error" and (not isinstance(error, str) or not error)):
                raise ValueError("SAM query status/error mismatch")
            count = query.get("sam_query_count")
            if type(count) is not int or not 0 <= count <= 1:
                raise ValueError("SAM individual text-query count is invalid")
            individual_count += count
            masks, scores = query.get("masks"), query.get("scores")
            if not isinstance(masks, list) or not isinstance(scores, list) or len(masks) != len(scores):
                raise ValueError("SAM masks and scores must have matching counts")
            query_id = self._query_id(frame, index, phrase)
            instances = []
            for instance_index, (mask, score) in enumerate(zip(masks, scores)):
                if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.shape != frame.rgb.shape[:2]:
                    raise ValueError("SAM instance must be boolean on the exact processed RGB grid")
                if score is not None and (isinstance(score, (bool, np.bool_)) or not isinstance(score, (int, float, np.number)) or not math.isfinite(float(score)) or not 0 <= float(score) <= 1):
                    raise ValueError("SAM score must be a finite evidence weight in [0,1]")
                copied = np.array(mask, dtype=np.bool_, copy=True)
                copied.setflags(write=False)
                instances.append(InstanceRecord(
                    observation_id=f"{query_id}:instance:{instance_index}", mask=copied,
                    score=None if score is None else float(score),
                    score_meaning="unavailable" if score is None else "raw_sam3_detection_score; uncalibrated_evidence_weight",
                    processed_grid_id=frame.processed_grid_id,
                    metadata={"diagnostic_only": status == "error"},
                ))
            records.append(QueryRecord(
                query_id=query_id, original_phrase=phrase, concept_id=self._concept(phrase),
                role="hazard", status=status, error=error, instances=instances,
                score_metadata={"sam_query_count": count, "qwen_confidence_probability": None},
                mapping_version=self._mapping_version,
            ))
        if individual_count != actual:
            raise ValueError("SAM frame and individual text-query counts differ")
        successes = sum(query.status == "ok" for query in records)
        failures = len(records) - successes
        # The legacy aggregate marks mixed outcomes error. Only per-query status
        # determines fusion eligibility; never consume the all-query union.
        image_error = row["status"] == "error" and row.get("error_code") != "sam_query_failure"
        if image_error or (not records and row["status"] == "error") or (failures and not successes):
            status = "error"
        elif failures:
            status = "partial"
        else:
            status = "ok"
        return records, actual, status, row.get("error_code") if status != "ok" else None

    def observe(self, frame: FramePacket) -> SemanticFrame:
        from pipeline_common.input_bridge import to_model_input
        started = self._clock()
        provenance = deepcopy(self.metadata)
        provenance["input"] = {
            "input_space": "lingbot_processed_rgb", "sequence_id": frame.sequence_id,
            "frame_id": frame.frame_id, "processed_grid_id": frame.processed_grid_id,
            "encoded_file_sha256": frame.encoded_file_sha256,
            "decoded_rgb_sha256": frame.decoded_rgb_sha256,
            "width": frame.rgb.shape[1], "height": frame.rgb.shape[0],
            "source_rgb_identity": deepcopy(frame.source_rgb_identity),
            "source_to_processed": deepcopy(frame.source_to_processed),
            "timestamp_provenance": deepcopy(frame.timestamp_provenance),
            "pose_revision": frame.geometry.pose_revision if frame.geometry else None,
        }
        provenance["stage_timestamps"] = {}
        provenance["observe_timing_boundary"] = "input checks, optional loading, Qwen, optional SAM and output validation; loading recorded separately"
        queries, actual_queries, calls = [], 0, {"qwen_predict_image": 0, "sam3_segment_image": 0}
        status, problem = "error", None
        stage = "input_validation"
        try:
            if self._closed:
                raise RuntimeError("adapter_closed")
            record, data_root = to_model_input(frame)
            record = deepcopy(record)
            # This is a generic validated RGB record, containing no geometry,
            # masks, reference labels or image-specific annotation names.
            stage = "qwen_loading"
            was_loaded = self._backend is not None
            with self._stage(provenance, stage, loaded_this_call=not was_loaded):
                backend = self._ensure_backend()
            provenance["qwen_backend_metadata"] = deepcopy(getattr(backend, "metadata", {}))
            stage = "qwen"
            try:
                with self._stage(provenance, stage, model_call=True):
                    calls["qwen_predict_image"] = 1
                    predicted = backend.predict_image(deepcopy(record), data_root)
                    provenance["qwen_prediction"] = deepcopy(predicted)
            finally:
                diagnostics = deepcopy(getattr(backend, "last_diagnostics", {}))
                provenance["qwen_diagnostics"] = diagnostics
            provenance["qwen_prediction"] = deepcopy(predicted)
            _, _, _, _, parse_response, validate_prediction = _legacy_imports()
            validate_prediction(predicted, model_key=self.model_key, frame_id=frame.frame_id)
            provenance["strict_parser_outcome"] = {"status": predicted["status"], "error_code": predicted["error_code"]}
            provenance["actual_output_tokens"] = diagnostics.get("tokens", {}).get("output_tokens")
            provenance["preprocessing"] = {
                "input_space": "lingbot_processed_rgb",
                "legacy_description_scope": "original whole RGB image means the supplied processed-grid image",
                "backend_metadata_snapshot": deepcopy(provenance["qwen_backend_metadata"]),
                "actual_qwen_resize": deepcopy(diagnostics.get("preprocessing")),
                **deepcopy(provenance["input"]),
            }
            current_record, current_root = to_model_input(frame)
            if current_record != record or current_root != data_root:
                raise ValueError("Model input identity changed during Qwen discovery")
            if predicted["status"] == "error":
                problem = {"code": predicted["error_code"], "stage": "qwen", "diagnostics": deepcopy(diagnostics)}
            else:
                generation = diagnostics.get("generation", {})
                if generation.get("terminal_eos") is not True or generation.get("stop_reason") != "eos":
                    raise ValueError("Qwen success has no verified terminal EOS")
                parsed = parse_response(diagnostics.get("parse_response_text"), self._policy)
                if parsed["prompts"] != predicted["prompts"]:
                    raise ValueError("Qwen parser result differs from the returned original phrases")
                provenance["strict_parser_outcome"] = {"status": "ok", "error_code": None, "diagnostics": deepcopy(parsed["diagnostics"])}
                prompts = list(predicted["prompts"])
                provenance["requested_prompt_count"] = len(prompts)
                # Re-decode before the second file-backed reader, including when
                # Qwen has altered its local input dict or the persisted file.
                if not prompts and self.empty_policy == "skip_sam":
                    status = "ok"
                    provenance["sam_disposition"] = "skipped_successful_empty_discovery"
                else:
                    stage = "sam3_loading"
                    was_loaded = self._segmenter is not None
                    with self._stage(provenance, stage, loaded_this_call=not was_loaded):
                        segmenter = self._ensure_segmenter()
                    provenance["sam_backend_metadata"] = deepcopy(getattr(segmenter, "metadata", {}))
                    stage = "sam3"
                    try:
                        with self._stage(provenance, stage, model_call=True):
                            calls["sam3_segment_image"] = 1
                            actual_queries = None  # Unexpected crashes cannot establish zero text executions.
                            returned = segmenter.segment_image(deepcopy(record), prompts, data_root)
                    finally:
                        provenance["sam_diagnostics"] = deepcopy(getattr(segmenter, "last_diagnostics", {}))
                    actual_queries = self._audit_sam_count(returned, frame, prompts)
                    queries, actual_queries, status, error = self._translate_sam(returned, frame, prompts)
                    provenance["sam_legacy_frame_status"] = returned["frame"]["status"]
                    provenance["sam_returned_settings"] = deepcopy(returned.get("settings", {}))
                    if error:
                        problem = {"code": error, "stage": "sam3"}
                    to_model_input(frame)  # Detect any mutation by the SAM reader.
        except Exception as error:
            status = "error"
            problem = {**_error(f"{stage}_failure", error), "stage": stage}
            if stage == "qwen":
                provenance["strict_parser_outcome"] = {"status": "error", "error_code": getattr(error, "code", problem["code"])}
        provenance["query_counts"] = {
            "requested": provenance.get("requested_prompt_count", 0),
            "actual_sam_text_executions": actual_queries,
            "availability": "available" if actual_queries is not None else "unavailable",
            "unavailable_reason": None if actual_queries is not None else "SAM crashed or returned unauditable query counts",
            "successful_queries": sum(query.status == "ok" for query in queries),
            "failed_queries": sum(query.status == "error" for query in queries),
        }
        stages = provenance["stage_timestamps"]
        calls_timing = [stages[key] for key in ("qwen", "sam3") if key in stages]
        total_call_ms = sum(row["duration_ms"] for row in calls_timing) if calls_timing and all(row["timing_valid"] for row in calls_timing) else None
        provenance["timing"] = {
            "clock": "monotonic_ns",
            "semantic_call_ms": total_call_ms,
            "model_calls_ms": total_call_ms,  # Common runner's multiple-model timing hook.
            "qwen_call_ms": stages.get("qwen", {}).get("duration_ms"),
            "sam3_call_ms": stages.get("sam3", {}).get("duration_ms"),
            "loading_ms": sum(row["duration_ms"] for row in stages.values() if row.get("loaded_this_call") and row["duration_ms"] is not None),
            "gpu_synchronized": bool(calls_timing) and all(row["synchronized"] and row["timing_valid"] for row in calls_timing),
            "scope": "CPU fixture adapter calls" if self.fixture else "synchronized Qwen and SAM adapter calls",
            "semantic_boundary": "Qwen adapter plus optional SAM adapter; excludes loading and bridge/output checks",
        }
        reason = provenance["query_counts"]["unavailable_reason"]
        return SemanticFrame(
            sequence_id=frame.sequence_id, frame_id=frame.frame_id,
            processed_grid_id=frame.processed_grid_id, decoded_rgb_sha256=frame.decoded_rgb_sha256,
            geometry_fingerprint=frame.geometry.geometry_fingerprint if frame.geometry else None,
            adapter_provenance=provenance, status=status, queries=queries,
            model_calls=calls, query_count=actual_queries, timestamp_ns=frame.timestamp_ns,
            started_monotonic_ns=started, completed_monotonic_ns=self._clock(), error=problem,
            query_count_reason=reason,
        )

    def close(self):
        if self._closed:
            return
        self._closed = True
        failures = []
        for provider, owned in ((self._segmenter, self._owned_segmenter), (self._backend, self._owned_backend)):
            if owned and provider is not None:
                try:
                    provider.close()
                except Exception as error:
                    failures.append(error)
        self._segmenter = self._backend = None
        if failures:
            raise RuntimeError("Owned model cleanup failed") from failures[0]


def create_adapter(config: dict):
    return QwenHazardsAdapter(config)
