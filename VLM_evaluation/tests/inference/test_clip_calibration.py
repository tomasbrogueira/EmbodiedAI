"""CPU fixtures only: these do not establish GPU performance or safety."""

from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace
import sys
from threading import RLock

import pytest
from PIL import Image

from traversability_inference.calibration import (
    calibrate_clip, load_calibration, validate_calibration_inputs,
)
from traversability_inference.clip import (
    CLIPBackend, MODEL_ID, MODEL_KEY, REVISION, _json_diagnostics,
    scoring_identity, semantic_vocabulary,
)
from traversability_inference.configuration import normalize_settings
from traversability_inference.records import InferenceError, PREDICTION_KEYS
from traversability_inference.storage import input_identity, stable_fingerprint


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")


def prepared_run(tmp_path, labels=("traversable", "non_traversable"), *, extra_test=True):
    run = tmp_path / "run"
    data = tmp_path / "data"
    data.mkdir()
    frames, regions, annotations = [], [], []
    all_labels = [*labels, *( ["traversable"] if extra_test else [] )]
    for index, label in enumerate(all_labels):
        frame_id, region_id = f"frame:{index}", f"region:{index}"
        split = "test" if extra_test and index == len(labels) else "development"
        frames.append({"frame_id": frame_id, "source": "synthetic", "scene_id": "fixture", "sequence_id": "0", "timestamp_s": index * 0.25, "image_path": f"image{index}.png", "split": split})
        regions.append({"region_id": region_id, "frame_id": frame_id, "mask_path": f"mask{index}.png", "planning_relevant": True, "selected_for_classification": True})
        annotations.append({"region_id": region_id, "reference_label": label, "semantic_class": "unused_reference_class", "hazard_type": None, "mask_quality": "valid", "annotation_status": "complete", "robot_profile_id": "rellis_material_v1"})
        Image.new("RGB", (8, 8), (index, 40, 80)).save(data / f"image{index}.png")
        Image.new("L", (8, 8), 255).save(data / f"mask{index}.png")
    write_jsonl(run / "frames.jsonl", list(reversed(frames)))
    write_jsonl(run / "regions.jsonl", list(reversed(regions)))
    write_jsonl(run / "annotations.jsonl", list(reversed(annotations)))
    return run, data, frames, regions, annotations


class ScoringFixture:
    metadata = {"fixture_identity": "synthetic_clip_scores_v1"}

    def __init__(self, scores):
        self.scores = scores
        self.requested = []

    def score_frame(self, frame, regions, data_root):
        self.requested.extend(row["region_id"] for row in regions)
        output = []
        for row in reversed(regions):
            score = self.scores[row["region_id"]]
            if score is None:
                output.append({"region_id": row["region_id"], "frame_id": frame["frame_id"], "status": "error", "error_code": "fixture_error"})
            else:
                semantic, similarity = score
                output.append({"region_id": row["region_id"], "frame_id": frame["frame_id"], "semantic_class": semantic, "similarity": similarity, "status": "ok", "error_code": None})
        return output


def fixture_identity(backend):
    return scoring_identity(normalize_settings(MODEL_KEY, {}), backend.metadata, injected=True, backend_type=f"{type(backend).__module__}.{type(backend).__qualname__}")


def test_frozen_vocabulary_policy_and_cosine_diagnostics():
    rellis = semantic_vocabulary("rellis_material_v1")
    wheeled = semantic_vocabulary("small_wheeled_v1")
    assert {row["semantic_class"]: row["label"] for row in rellis}["short_grass"] == "unknown"
    assert {row["semantic_class"]: row["label"] for row in wheeled}["short_grass"] == "traversable"
    assert {row["semantic_class"]: row["label"] for row in rellis}["sky"] == "non_traversable"
    assert [row["prompt"] for row in rellis] == [row["prompt"] for row in wheeled]
    chosen, score, raw = _json_diagnostics(rellis, [0.5] * len(rellis))
    assert chosen["semantic_class"] == "dry_floor"  # Stable vocabulary tie break.
    assert score == 0.5
    assert json.loads(raw)["calibrated_probability"] is False
    assert len(json.loads(raw)["similarities"]) == len(rellis)
    with pytest.raises(InferenceError, match="non-finite"):
        _json_diagnostics(rellis, [float("nan")] * len(rellis))


@pytest.mark.parametrize("scores,threshold,useful", [
    ([0.7, 0.8, 0.69], 0.7, 2),
    ([0.7, 0.8, 0.7], 0.8, 1),
    ([-0.2, 0.1, -0.3], -0.2, 2),
])
def test_inclusive_calibration_boundary_and_higher_threshold(tmp_path, scores, threshold, useful):
    run, data, *_ = prepared_run(tmp_path, ("traversable", "traversable", "non_traversable"))
    backend = ScoringFixture({f"region:{index}": ("concrete", score) for index, score in enumerate(scores)})
    artifact = calibrate_clip(run, {}, data, backend=backend)
    assert artifact["threshold"] == threshold
    assert artifact["reject_all"] is False
    assert artifact["observed_counts"]["useful_acceptances"] == useful
    assert artifact["observed_counts"]["unsafe_acceptances"] == 0
    assert artifact["development_region_ids"] == ["region:0", "region:1", "region:2"]
    assert set(backend.requested) == set(artifact["development_region_ids"])
    assert load_calibration(run / "metadata" / "clip_calibration.json", fixture_identity(backend)) == artifact


def test_reject_all_is_explicit_when_no_safe_useful_acceptance(tmp_path):
    run, data, *_ = prepared_run(tmp_path)
    backend = ScoringFixture({"region:0": ("concrete", 0.8), "region:1": ("concrete", 0.9)})
    artifact = calibrate_clip(run, {}, data, backend=backend)
    assert artifact["reject_all"] is True
    assert artifact["threshold"] is None
    assert artifact["observed_counts"]["useful_acceptances"] == 0


def test_non_traversable_winner_is_not_unsafe_acceptance(tmp_path):
    run, data, *_ = prepared_run(tmp_path)
    backend = ScoringFixture({"region:0": ("concrete", 0.4), "region:1": ("person", 0.99)})
    artifact = calibrate_clip(run, {}, data, backend=backend)
    assert artifact["threshold"] == 0.4
    assert artifact["observed_counts"]["useful_acceptances"] == 1


def test_test_references_and_reference_transfer_do_not_participate(tmp_path):
    run, data, frames, regions, annotations = prepared_run(tmp_path)
    backend = ScoringFixture({"region:0": ("concrete", 0.8), "region:1": ("mud", 0.9)})
    first = calibrate_clip(run, {}, data, backend=backend)
    annotations[-1].update(reference_label={"invalid": "test reference"}, mask_quality="broken", robot_profile_id="other")
    annotations.append(dict(annotations[-1]))  # Test-only duplicate is irrelevant.
    write_jsonl(run / "annotations.jsonl", annotations)
    write_jsonl(run / "metadata" / "reference_transfer.jsonl", [{"region_id": "region:0", "fractions": "never read"}])
    Image.new("RGB", (8, 8), "red").save(data / frames[-1]["image_path"])
    second = calibrate_clip(run, {}, data, backend=backend)
    assert first == second


@pytest.mark.parametrize("change", [
    {"mask_quality": "mixed"}, {"annotation_status": "pending"},
    {"robot_profile_id": "small_wheeled_v1"}, {"reference_label": "unknown"},
])
def test_only_completed_valid_matching_profile_known_development_classes(tmp_path, change):
    run, data, frames, regions, annotations = prepared_run(tmp_path)
    annotations[1].update(change)
    write_jsonl(run / "annotations.jsonl", annotations)
    with pytest.raises(InferenceError) as caught:
        calibrate_clip(run, {}, data, backend=ScoringFixture({}))
    assert caught.value.code == "clip_calibration_insufficient_classes"
    assert not (run / "metadata" / "clip_calibration.json").exists()


def test_both_reference_classes_need_successful_scores(tmp_path):
    run, data, *_ = prepared_run(tmp_path)
    with pytest.raises(InferenceError) as caught:
        calibrate_clip(run, {}, data, backend=ScoringFixture({"region:0": ("concrete", 0.8), "region:1": None}))
    assert caught.value.code == "clip_calibration_insufficient_successes"


def test_duplicate_development_annotations_fail_and_unselected_regions_are_excluded(tmp_path):
    run, data, frames, regions, annotations = prepared_run(tmp_path)
    write_jsonl(run / "annotations.jsonl", [*annotations, dict(annotations[0])])
    with pytest.raises(InferenceError) as caught:
        calibrate_clip(run, {}, data, backend=ScoringFixture({}))
    assert caught.value.code == "duplicate_annotation_id"
    write_jsonl(run / "annotations.jsonl", annotations)
    regions[1]["selected_for_classification"] = False
    write_jsonl(run / "regions.jsonl", regions)
    with pytest.raises(InferenceError) as caught:
        calibrate_clip(run, {}, data, backend=ScoringFixture({}))
    assert caught.value.code == "clip_calibration_insufficient_classes"


def test_cache_and_calibration_paths_do_not_change_scoring_identity(tmp_path):
    settings_a = normalize_settings(MODEL_KEY, {"cache_dir": str(tmp_path / "cache_a"), "calibration_path": str(tmp_path / "calibration_a.json")})
    settings_b = normalize_settings(MODEL_KEY, {"cache_dir": str(tmp_path / "cache_b"), "calibration_path": str(tmp_path / "calibration_b.json"), "local_files_only": True})
    assert scoring_identity(settings_a, {}) == scoring_identity(settings_b, {})


@pytest.mark.parametrize("mutation", ["image", "mask", "manifest", "metadata", "backend_settings"])
def test_calibration_rejects_mid_score_drift_and_preserves_previous_artifact(tmp_path, mutation):
    run, data, frames, regions, _ = prepared_run(tmp_path)
    scores = {"region:0": ("concrete", 0.8), "region:1": ("mud", 0.9)}
    calibrate_clip(run, {}, data, backend=ScoringFixture(scores))
    destination = run / "metadata" / "clip_calibration.json"
    previous = destination.read_bytes()

    class DriftingFixture(ScoringFixture):
        def __init__(self):
            super().__init__(scores)
            self.metadata = {"nested": {"identity": 1}}
            self.settings = normalize_settings(MODEL_KEY, {})
            self.changed = False

        def score_frame(self, frame, requested, data_root):
            output = super().score_frame(frame, requested, data_root)
            if not self.changed:
                self.changed = True
                if mutation == "image":
                    Image.new("RGB", (8, 8), "red").save(data / frames[0]["image_path"])
                elif mutation == "mask":
                    Image.new("L", (8, 8), 0).save(data / regions[0]["mask_path"])
                elif mutation == "manifest":
                    frames[0]["timestamp_s"] = 999
                    write_jsonl(run / "frames.jsonl", frames)
                elif mutation == "metadata":
                    self.metadata["nested"]["identity"] = 2
                else:
                    self.settings["context_token_limit"] = 1024
            return output

    with pytest.raises(InferenceError) as caught:
        calibrate_clip(run, {}, data, backend=DriftingFixture())
    expected_code = "clip_calibration_inputs_changed" if mutation in {"image", "mask", "manifest"} else "clip_calibration_configuration_changed"
    assert caught.value.code == expected_code
    assert destination.read_bytes() == previous


def test_scoring_backend_cannot_mutate_frozen_calibration_records(tmp_path):
    run, data, frames, regions, _ = prepared_run(tmp_path)

    class ArgumentMutatingFixture(ScoringFixture):
        def score_frame(self, frame, requested, data_root):
            output = super().score_frame(frame, requested, data_root)
            frame["timestamp_s"] = 999
            requested[0]["mask_path"] = "mutated_backend_argument.png"
            return output

    artifact = calibrate_clip(run, {}, data, backend=ArgumentMutatingFixture({"region:0": ("concrete", 0.8), "region:1": ("mud", 0.9)}))
    assert artifact["development_frames"] == frames[:2]
    assert artifact["development_regions"] == regions[:2]
    assert artifact["development_fingerprint"] == stable_fingerprint(input_identity(frames[:2], regions[:2], data))


def test_fixture_calibration_cannot_classify_with_real_identity(tmp_path):
    run, data, *_ = prepared_run(tmp_path)
    backend = ScoringFixture({"region:0": ("concrete", 0.8), "region:1": ("mud", 0.9)})
    calibrate_clip(run, {}, data, backend=backend)
    with pytest.raises(InferenceError) as caught:
        load_calibration(run / "metadata" / "clip_calibration.json", scoring_identity(normalize_settings(MODEL_KEY, {}), backend.metadata))
    assert caught.value.code == "clip_calibration_incompatible"


def test_calibration_integrity_and_configuration_compatibility(tmp_path):
    run, data, *_ = prepared_run(tmp_path)
    backend = ScoringFixture({"region:0": ("concrete", 0.8), "region:1": ("mud", 0.9)})
    artifact = calibrate_clip(run, {}, data, backend=backend)
    changed = fixture_identity(backend)
    changed["configuration"]["settings"]["context_token_limit"] = 1024
    with pytest.raises(InferenceError) as caught:
        load_calibration(run / "metadata" / "clip_calibration.json", changed)
    assert caught.value.code == "clip_calibration_incompatible"
    artifact["threshold"] = 0.1
    (run / "metadata" / "clip_calibration.json").write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(InferenceError) as caught:
        load_calibration(run / "metadata" / "clip_calibration.json", fixture_identity(backend))
    assert caught.value.code == "clip_calibration_invalid"


@pytest.mark.parametrize("changed_input", ["image", "mask", "manifest"])
def test_calibration_rejects_changed_development_inputs(tmp_path, changed_input):
    run, data, frames, regions, _ = prepared_run(tmp_path)
    artifact = calibrate_clip(run, {}, data, backend=ScoringFixture({"region:0": ("concrete", 0.8), "region:1": ("mud", 0.9)}))
    validate_calibration_inputs(artifact, frames, regions, data)
    if changed_input == "image":
        Image.new("RGB", (8, 8), "red").save(data / frames[0]["image_path"])
    elif changed_input == "mask":
        Image.new("L", (8, 8), 0).save(data / regions[0]["mask_path"])
    else:
        frames[0]["timestamp_s"] = 99
    with pytest.raises(InferenceError) as caught:
        validate_calibration_inputs(artifact, frames, regions, data)
    assert caught.value.code == "clip_calibration_incompatible"


def cpu_adapter(threshold=0.5, *, reject_all=False):
    adapter = CLIPBackend.__new__(CLIPBackend)
    adapter.model_key = MODEL_KEY
    adapter._vocabulary = semantic_vocabulary("rellis_material_v1")
    adapter._closed = False
    adapter._lock = RLock()
    adapter._validated_data_roots = set()
    adapter._torch = adapter._model = adapter._processor = adapter._text_features = None
    adapter._calibration = {"threshold": threshold, "reject_all": reject_all, "development_frames": [], "development_regions": [], "development_fingerprint": stable_fingerprint(input_identity([], [], Path.cwd()))}
    return adapter


def test_classification_preserves_raw_cosines_and_threshold_boundary(tmp_path):
    run, data, frames, regions, _ = prepared_run(tmp_path)
    adapter = cpu_adapter()
    adapter._score_crop = lambda crop: [0.1, 0.5] + [0.0] * (len(adapter._vocabulary) - 2)
    result = adapter.predict_frame(frames[0], [regions[0]], data)[0]
    assert set(result) == PREDICTION_KEYS
    assert result["label"] == "traversable" and result["status"] == "ok"
    assert json.loads(result["raw_response"])["cosine_similarity"] == 0.5
    adapter._calibration["threshold"] = 0.500001
    assert adapter.predict_frame(frames[0], [regions[0]], data)[0]["label"] == "unknown"
    adapter._calibration.update(reject_all=True, threshold=None)
    assert adapter.predict_frame(frames[0], [regions[0]], data)[0]["label"] == "unknown"
    adapter.close()
    adapter.close()
    closed = adapter.predict_frame(frames[0], [regions[0]], data)[0]
    assert closed["label"] == "unknown" and closed["error_code"] == "backend_closed"


def test_classification_mask_errors_and_missing_calibration(tmp_path):
    run, data, frames, regions, _ = prepared_run(tmp_path)
    adapter = cpu_adapter()
    Image.new("L", (8, 8), 0).save(data / regions[0]["mask_path"])
    result = adapter.predict_frame(frames[0], [regions[0]], data)[0]
    assert result["status"] == "error" and result["error_code"] == "mask_empty"
    adapter._calibration = None
    result = adapter.predict_frame(frames[0], [regions[0]], data)[0]
    assert result["error_code"] == "clip_calibration_missing"


def test_direct_classification_checks_calibration_input_contents(tmp_path):
    run, data, frames, regions, _ = prepared_run(tmp_path)
    artifact = calibrate_clip(run, {}, data, backend=ScoringFixture({"region:0": ("concrete", 0.8), "region:1": ("mud", 0.9)}))
    adapter = cpu_adapter()
    adapter._calibration = artifact
    adapter._score_crop = lambda crop: pytest.fail("incompatible calibration must be rejected before scoring")
    Image.new("RGB", (8, 8), "red").save(data / frames[0]["image_path"])
    result = adapter.predict_frame(frames[2], [regions[2]], data)[0]
    assert result["status"] == "error" and result["error_code"] == "clip_calibration_incompatible"


def test_custom_clip_policy_is_rejected_before_loading():
    with pytest.raises(InferenceError) as caught:
        CLIPBackend(MODEL_KEY, {"robot_policy": "permit everything"})
    assert caught.value.code == "unsupported_robot_policy"


def test_mocked_loader_is_pinned_floating_point_single_device_and_close_idempotent(monkeypatch):
    calls = []

    class Tensor:
        def to(self, device):
            calls.append(("tensor_to", device))
            return self

        def norm(self, **kwargs):
            return self

        def __truediv__(self, other):
            return self

    class Processor:
        image_processor = SimpleNamespace(size={"shortest_edge": 224}, crop_size={"height": 224, "width": 224}, image_mean=[0.1] * 3, image_std=[0.2] * 3, do_resize=True, do_center_crop=True)

        @classmethod
        def from_pretrained(cls, checkpoint, **kwargs):
            calls.append(("processor_load", checkpoint, kwargs))
            return cls()

        def __call__(self, **kwargs):
            calls.append(("processor_inputs", kwargs))
            return {"input_ids": Tensor()}

    class Model:
        @classmethod
        def from_pretrained(cls, checkpoint, **kwargs):
            calls.append(("model_load", checkpoint, kwargs))
            return cls()

        def to(self, device):
            calls.append(("model_to", device))
            return self

        def eval(self):
            return self

        def get_text_features(self, **kwargs):
            return Tensor()

        def parameters(self):
            return [SimpleNamespace(device="cuda:0", dtype="torch.float32")]

    torch = SimpleNamespace(float32="torch.float32", cuda=SimpleNamespace(is_available=lambda: True, device=lambda device: nullcontext(), empty_cache=lambda: calls.append(("empty_cache",))), inference_mode=nullcontext)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoProcessor=Processor, CLIPModel=Model))
    adapter = CLIPBackend(MODEL_KEY, {"cache_dir": "portable-test-cache", "local_files_only": True}, _for_calibration=True)
    processor_load = next(item for item in calls if item[0] == "processor_load")
    model_load = next(item for item in calls if item[0] == "model_load")
    assert processor_load[1] == model_load[1] == MODEL_ID
    assert processor_load[2]["revision"] == model_load[2]["revision"] == REVISION
    assert model_load[2]["dtype"] == "torch.float32"
    assert "device_map" not in model_load[2]
    assert ("model_to", "cuda:0") in calls
    assert adapter.metadata["parameter_dtypes"] == ["torch.float32"]
    adapter.close()
    adapter.close()
    assert calls.count(("empty_cache",)) == 1


def test_loading_requires_explicit_calibration_before_model_import():
    with pytest.raises(InferenceError) as caught:
        CLIPBackend(MODEL_KEY, {})
    assert caught.value.code == "clip_calibration_missing"
    with pytest.raises(InferenceError) as caught:
        CLIPBackend(MODEL_KEY, {"calibration_path": "missing-fixture-calibration.json"})
    assert caught.value.code == "clip_calibration_missing"
