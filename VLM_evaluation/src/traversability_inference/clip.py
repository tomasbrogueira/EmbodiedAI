"""Crop-only CLIP adapter with an explicitly calibrated rejection threshold.

Importing this module never imports Torch/Transformers or downloads weights.
Cosine similarities are diagnostics, not calibrated confidence probabilities.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import gc
from threading import RLock

from .configuration import ROBOT_POLICIES, configuration_metadata, normalize_settings
from .preparation import prepare_target
from .records import InferenceError, prediction


MODEL_KEY = "clip_vit_b32"
MODEL_ID = "openai/clip-vit-base-patch32"
REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"

# Ordering also freezes the deterministic tie break for equal cosine scores.
# (semantic class, prompt, rellis_material_v1 mapping, small_wheeled_v1 mapping)
VOCABULARY = (
    ("dry_floor", "a photo of a dry floor surface", "unknown", "traversable"),
    ("concrete", "a photo of a concrete ground surface", "traversable", "traversable"),
    ("asphalt", "a photo of an asphalt ground surface", "traversable", "traversable"),
    ("compact_soil", "a photo of compact dirt ground", "traversable", "traversable"),
    ("short_grass", "a photo of short grass on the ground", "unknown", "traversable"),
    ("person", "a photo of a person", "non_traversable", "non_traversable"),
    ("animal", "a photo of an animal", "non_traversable", "non_traversable"),
    ("liquid", "a photo of water or a puddle on the ground", "non_traversable", "non_traversable"),
    ("mud", "a photo of wet muddy ground", "non_traversable", "non_traversable"),
    ("dense_vegetation", "a photo of dense bushes or tall vegetation", "non_traversable", "non_traversable"),
    ("fragile_belongings", "a photo of fragile personal belongings", "non_traversable", "non_traversable"),
    ("cable", "a photo of cables lying on the ground", "non_traversable", "non_traversable"),
    ("loose_obstacle", "a photo of a loose object obstructing the ground", "non_traversable", "non_traversable"),
    ("tree", "a photo of a tree trunk", "non_traversable", "non_traversable"),
    ("pole", "a photo of a pole", "non_traversable", "non_traversable"),
    ("sky", "a photo of the sky", "non_traversable", "non_traversable"),
    ("vehicle", "a photo of a vehicle", "non_traversable", "non_traversable"),
    ("building", "a photo of a building wall", "non_traversable", "non_traversable"),
    ("log", "a photo of a fallen log", "non_traversable", "non_traversable"),
    ("fence", "a photo of a fence", "non_traversable", "non_traversable"),
    ("barrier", "a photo of a barrier", "non_traversable", "non_traversable"),
    ("rubble", "a photo of rubble or loose rocks", "non_traversable", "non_traversable"),
)


def semantic_vocabulary(robot_profile_id):
    """Return the frozen prompts and mappings for the selected policy."""
    if robot_profile_id not in {"rellis_material_v1", "small_wheeled_v1"}:
        raise InferenceError("unsupported_robot_profile", str(robot_profile_id))
    column = 2 if robot_profile_id == "rellis_material_v1" else 3
    return [
        {"semantic_class": item[0], "prompt": item[1], "label": item[column]}
        for item in VOCABULARY
    ]


def _normalized_clip_settings(settings):
    resolved = normalize_settings(MODEL_KEY, settings)
    profile = resolved["robot_profile_id"]
    semantic_vocabulary(profile)
    if resolved["robot_policy"] != ROBOT_POLICIES[profile]:
        raise InferenceError("unsupported_robot_policy", "CLIP requires the exact frozen policy for its semantic mapping")
    return resolved


def scoring_identity(settings, backend_metadata, *, injected=False, backend_type=None):
    """Identity shared by calibration and classification, without the threshold."""
    metadata = dict(backend_metadata)
    metadata.pop("calibration", None)
    identity = {
        "configuration": configuration_metadata(MODEL_KEY, settings),
        "vocabulary": semantic_vocabulary(settings["robot_profile_id"]),
        "actual_backend": metadata,
        "execution_kind": "injected_fixture" if injected else "real_model",
    }
    if injected:
        identity["fixture_type"] = backend_type
    return identity


def _json_diagnostics(vocabulary, similarities):
    if len(similarities) != len(vocabulary):
        raise InferenceError("clip_invalid_scores", "CLIP returned an incorrect score count")
    scores = [float(value) for value in similarities]
    if any(not math.isfinite(value) or not -1.000001 <= value <= 1.000001 for value in scores):
        raise InferenceError("clip_invalid_scores", "CLIP returned non-finite or invalid cosine scores")
    # Numerical roundoff can exceed the cosine interval by a few ulps.
    scores = [max(-1.0, min(1.0, value)) for value in scores]
    best = max(range(len(scores)), key=scores.__getitem__)
    selected = vocabulary[best]
    raw = json.dumps(
        {
            "semantic_class": selected["semantic_class"],
            "cosine_similarity": scores[best],
            "similarities": {item["semantic_class"]: value for item, value in zip(vocabulary, scores)},
            "calibrated_probability": False,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return selected, scores[best], raw


class CLIPBackend:
    """Pinned floating-point CLIP on one explicit CUDA device."""

    def __init__(self, model_key, settings, *, _for_calibration=False):
        if model_key != MODEL_KEY:
            raise ValueError(f"CLIP does not implement {model_key!r}")
        self.model_key = model_key
        self.settings = _normalized_clip_settings(settings)
        self._vocabulary = semantic_vocabulary(self.settings["robot_profile_id"])
        self._lock = RLock()
        self._validated_data_roots = set()
        self._model = self._processor = self._text_features = self._torch = None
        self._closed = False
        self._calibration = None
        if not _for_calibration and not self.settings.get("calibration_path"):
            raise InferenceError("clip_calibration_missing", "Set calibration_path to a compatible development calibration file")
        if not _for_calibration and not Path(self.settings["calibration_path"]).is_file():
            raise InferenceError("clip_calibration_missing", "The explicitly selected development calibration file does not exist")
        try:
            self._load_model()
            if not _for_calibration:
                from .calibration import load_calibration

                self._calibration = load_calibration(
                    self.settings["calibration_path"],
                    scoring_identity(self.settings, self.metadata),
                )
                self.metadata["calibration"] = self._calibration
        except Exception:
            self.close()
            raise

    def _load_model(self):
        import torch
        from transformers import AutoProcessor, CLIPModel

        self._torch = torch
        device = self.settings.get("device", "cuda:0")
        if not isinstance(device, str) or not device.startswith("cuda:") or not device[5:].isdigit():
            raise InferenceError("invalid_device", "CLIP requires one explicit CUDA device such as cuda:0")
        if not torch.cuda.is_available():
            raise InferenceError("cuda_unavailable", "CLIP model loading requires CUDA; CPU tests use injected backends")
        common = {"revision": REVISION}
        if self.settings.get("cache_dir") is not None:
            common["cache_dir"] = self.settings["cache_dir"]
        if self.settings.get("local_files_only") is not None:
            common["local_files_only"] = self.settings["local_files_only"]
        self._processor = AutoProcessor.from_pretrained(MODEL_ID, **common)
        self._model = CLIPModel.from_pretrained(MODEL_ID, dtype=torch.float32, **common)
        self._model = self._model.to(device).eval()
        text_inputs = self._processor(
            text=[item["prompt"] for item in self._vocabulary],
            padding=True,
            return_tensors="pt",
        )
        text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
        with torch.inference_mode():
            features = self._model.get_text_features(**text_inputs)
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            self._text_features = features / features.norm(dim=-1, keepdim=True)
        image_processor = self._processor.image_processor
        self.metadata = {
            "model_id": MODEL_ID,
            "revision": REVISION,
            "device": device,
            "parameter_devices": sorted({str(parameter.device) for parameter in self._model.parameters()}),
            "parameter_dtypes": sorted({str(parameter.dtype) for parameter in self._model.parameters()}),
            "quantization": None,
            "preprocessing": {
                "target": "original_rgb_mask_bbox_crop",
                "crop_padding_per_side": 0.2,
                "size": image_processor.size,
                "crop_size": image_processor.crop_size,
                "image_mean": image_processor.image_mean,
                "image_std": image_processor.image_std,
                "do_resize": image_processor.do_resize,
                "do_center_crop": image_processor.do_center_crop,
            },
            "vocabulary": self._vocabulary,
            "score": "normalized_image_text_cosine_uncalibrated",
        }

    def _score_crop(self, crop):
        inputs = self._processor(images=crop, return_tensors="pt")
        inputs = {key: value.to(self.metadata["device"]) for key, value in inputs.items()}
        with self._torch.inference_mode():
            features = self._model.get_image_features(**inputs)
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            features = features / features.norm(dim=-1, keepdim=True)
            return (features @ self._text_features.T)[0].float().cpu().tolist()

    def score_frame(self, frame, regions, data_root):
        """Prepare and score crops without classification or reference access.

        This helper deliberately has a diagnostic score schema, not the fixed
        prediction schema, so calibration cannot masquerade as inference.
        """
        with self._lock:
            return self._score_frame_unlocked(frame, regions, data_root)

    def _score_frame_unlocked(self, frame, regions, data_root):
        output = []
        for region in regions:
            result = {
                "region_id": region["region_id"],
                "frame_id": frame["frame_id"],
                "semantic_class": None,
                "candidate_label": "unknown",
                "similarity": None,
                "raw_response": "",
                "status": "error",
                "error_code": None,
            }
            try:
                if self._closed:
                    raise InferenceError("backend_closed", "CLIP backend is closed")
                target = prepare_target(frame, region, data_root)
                selected, similarity, raw = _json_diagnostics(self._vocabulary, self._score_crop(target.crop))
                result.update(
                    semantic_class=selected["semantic_class"], candidate_label=selected["label"],
                    similarity=similarity, raw_response=raw, status="ok",
                )
            except InferenceError as exc:
                result["error_code"] = exc.code
                result["reason"] = str(exc)
            except Exception as exc:
                result["error_code"] = "inference_error"
                result["reason"] = f"{type(exc).__name__}: {exc}"
            output.append(result)
        return output

    def predict_frame(self, frame, regions, data_root):
        with self._lock:
            return self._predict_frame_unlocked(frame, regions, data_root)

    def _predict_frame_unlocked(self, frame, regions, data_root):
        if self._calibration is None:
            return [prediction(region, self.model_key, error_code="clip_calibration_missing", reason="CLIP classification requires development calibration") for region in regions]
        # A benchmark may call the adapter directly, without the run reader.
        # Verify the calibration's frozen inputs once per data root before any
        # classifications; the runner additionally checks the manifest joins.
        root = str(Path(data_root).resolve())
        if root not in self._validated_data_roots:
            from .calibration import validate_calibration_inputs

            try:
                validate_calibration_inputs(
                    self._calibration, self._calibration["development_frames"],
                    self._calibration["development_regions"], data_root,
                )
            except InferenceError as exc:
                return [prediction(region, self.model_key, error_code=exc.code, reason=str(exc)) for region in regions]
            except (ValueError, KeyError, TypeError) as exc:
                return [prediction(region, self.model_key, error_code="clip_calibration_invalid", reason=str(exc)) for region in regions]
            self._validated_data_roots.add(root)
        output = []
        for region, score in zip(regions, self.score_frame(frame, regions, data_root)):
            if score["status"] != "ok":
                output.append(prediction(region, self.model_key, raw_response=score["raw_response"], error_code=score["error_code"], reason=score.get("reason", "CLIP scoring failed")))
                continue
            rejected = self._calibration["reject_all"] or score["similarity"] < self._calibration["threshold"]
            label = "unknown" if rejected else score["candidate_label"]
            reason = "CLIP semantic cosine match rejected by development threshold" if rejected else "CLIP highest cosine semantic match mapped through frozen robot policy"
            output.append(prediction(region, self.model_key, parsed={"label": label, "semantic_class": score["semantic_class"], "reason": reason}, raw_response=score["raw_response"]))
        return output

    def close(self):
        with self._lock:
            self._close_unlocked()

    def _close_unlocked(self):
        if self._closed:
            return
        self._closed = True
        self._text_features = self._model = self._processor = None
        torch = self._torch
        self._torch = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            with torch.cuda.device(self.settings["device"]):
                torch.cuda.empty_cache()
