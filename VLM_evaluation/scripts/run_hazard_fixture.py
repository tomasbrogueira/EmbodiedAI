"""Explicit, offline integration replay. All artifacts are synthetic CPU checks."""

from copy import deepcopy
import json
from pathlib import Path
import sys

COMPONENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMPONENT / "src"))
MODELS = ("qwen3_vl_4b", "qwen3_5_4b")
FIXTURE_ID = "hazard_cpu_fixture_v1"


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class SyntheticProbe:
    available = False
    unavailable_reason = "Synthetic CPU fixture; GPU memory unmeasured"

    def synchronize(self):
        pass

    def reset_peak(self):
        pass

    def snapshot(self):
        return {"allocated_bytes": None, "reserved_bytes": None,
                "device_used_bytes": None, "device_total_bytes": None}

    def peaks(self):
        return {"allocated_peak_bytes": None, "reserved_peak_bytes": None}


class RGBFixtureBackend:
    """Read RGB plus common policy only; never open annotations or references."""
    fixture_identity = FIXTURE_ID

    def __init__(self, model_key, settings, clock=None):
        from traversability_hazard_inference.configuration import configuration_snapshot, load_policy
        self.model_key, self.clock = model_key, clock
        snapshot = configuration_snapshot(model_key, settings)
        self.settings = deepcopy(snapshot["settings"])
        self.policy = load_policy(settings)
        self.metadata = {"fixture": True, "fixture_id": FIXTURE_ID, "model_key": model_key,
                         "checkpoint_revision": snapshot["model"]["revision"],
                         "implementation": "synthetic_rgb_rules", "device": "cpu"}
        self.last_diagnostics, self.calls = {}, []

    def predict_image(self, frame, data_root):
        from PIL import Image
        from traversability_hazard_inference.records import InferenceError, parse_response, prediction
        assert not {"present_concepts", "concept_masks", "scored_concepts", "annotation_path"} & frame.keys()
        with Image.open(Path(data_root) / frame["image_path"]) as image:
            image.load()
            red, green, _ = image.convert("RGB").getpixel((0, 0))
        # These colors belong to the marked synthetic RGB generator, not ground truth.
        index = (green - 20) // 5 if red == 140 else green - 80
        phrases = ([], ["person"], ["person", "traffic cone"], ["water", "puddle"],
                   ["bottle", "chair"], ["tree", "person"])[index % 6]
        if self.model_key == MODELS[1] and index % 6 == 3:
            phrases = ["person", "traffic cone"]
        raw = json.dumps({"prompts": phrases})
        error = None
        if index % 6 == 1:
            raw = "{bad JSON"
        elif index % 11 == 7:
            raw, error = '{"prompts":["person"', "generation_truncated"
        self.calls.append((frame["frame_id"], raw))
        self.last_diagnostics = {"fixture": True, "generation": {
            "terminal_eos": error is None, "stop_reason": "length_limit" if error else "eos",
            "eos_text": "" if error is None else None}}
        if self.clock:
            self.clock.advance(0.2)
        try:
            if error:
                raise InferenceError(error, "Marked synthetic truncation")
            parsed = parse_response(raw, self.policy)
            self.last_diagnostics.update(parsed["diagnostics"], parse_response_text=raw)
            return prediction(frame, self.model_key, prompts=parsed["prompts"], raw_response=raw)
        except InferenceError as exc:
            return prediction(frame, self.model_key, raw_response=raw, error_code=exc.code)

    def close(self):
        pass


def run_fixture(output_root, *, resume=False):
    """Run public producers and evaluator in new, explicit, isolated roots."""
    from traversability_hazard_data import prepare_run, validate_run
    from traversability_hazard_inference import run_inference
    from traversability_hazard_inference.storage import atomic_json, atomic_jsonl, read_json, read_jsonl
    from traversability_hazard_segmentation import freeze_selection, run_conditions
    from traversability_hazard_segmentation.fixtures import FakeSegmenter
    from traversability_hazard_benchmark import profile_backend
    from traversability_hazard_evaluation import evaluate, export_report

    output = Path(output_root).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        metadata = output / "runs/hazard_prompt_v1_fixture/metadata/dataset.json"
        if not resume or not metadata.is_file():
            raise ValueError("Fixture output root must be new/empty or an explicitly resumed fixture")
        saved_identity = read_json(metadata)
        if (saved_identity.get("task_id") != "hazard_prompt_v1" or saved_identity.get("fixture") is not True
                or saved_identity.get("fixture_profile") != "protocol"):
            raise ValueError("Resume requires this workflow's marked protocol fixture")
    output.mkdir(parents=True, exist_ok=True)
    data, runs, cache = output / "data", output / "runs", output / "cache"
    cfg = {"run_name": "hazard_prompt_v1_fixture", "fixture_profile": "protocol",
           "data_root": str(data), "run_root": str(runs), "cache_root": str(cache)}
    print("10: generating synthetic RGB/source annotations and references", flush=True)
    run = prepare_run(cfg, fixture=True)
    validation = validate_run(run, data)
    assert validation["valid"] and validation["fixture"] and validation["frames"] == 140
    references = read_jsonl(run / "references.jsonl")
    assert any(not row["absence_scoring_eligible"] for row in references)
    assert any(row["concept_pixel_counts"].get("water") == 2 for row in references)
    selection = freeze_selection(run, data_root=data)
    assert selection["complete"] and len(selection["test_frame_ids"]) == 20
    settings = read_json(COMPONENT / "configs/hazards/inference.json")["settings"]
    settings.update(fixture=True, cache_dir=str(cache))
    print("11: marked fake whole-RGB VLM responses for both conditions", flush=True)
    inference = {}
    for model in MODELS:
        inference[model] = run_inference(run, model, settings, data,
                                       backend=RGBFixtureBackend(model, settings))
        assert inference[model]["complete"] and inference[model]["fixture"]
    records = read_jsonl(run / f"predictions/{MODELS[0]}.jsonl")
    assert any(row["status"] == "ok" and row["prompts"] == [] for row in records)
    assert {"invalid_json", "generation_truncated"} <= {row["error_code"] for row in records}
    assert any("traffic cone" in row["prompts"] for row in records)

    clock = Clock()
    sam_settings = read_json(COMPONENT / "configs/hazards/segmentation.json")["sam_settings"]
    sam_settings.update(fixture=True, fixture_identity=FIXTURE_ID, device="cpu")

    class Segmenter(FakeSegmenter):
        def __init__(self):
            super().__init__()
            self.settings = deepcopy(sam_settings)

        def segment_image(self, frame, prompts, data_root):
            result = super().segment_image(frame, prompts, data_root)
            clock.advance(0.3)
            return result

    segmenter = Segmenter()
    condition_cfg = {**cfg, "run_dir": str(run), "fixture": True,
                     "sam_settings": sam_settings, "enable_real_run": False}
    print("13: fake SAM outputs for both VLMs, reference-present and fixed policy", flush=True)
    segmentation = run_conditions(condition_cfg, segmenter=segmenter)
    assert segmentation["complete"] and not segmentation["comparison_ready"]
    atomic_json(output / "segmentation_summary.json", segmentation)
    assert any(row["status"] == "error" for row in read_jsonl(run / "segmentation/fixed_policy/queries.jsonl"))
    evaluation_cfg = read_json(COMPONENT / "configs/hazards/evaluation.json")
    evaluation_cfg.update(run_root=str(runs), data_root=str(data), run_name=run.name, fixture=True)
    evaluation_cfg["segmentation"]["enabled"] = True
    report = evaluate(evaluation_cfg)
    if report["validation"]["errors"]:
        raise AssertionError(json.dumps(report["validation"], indent=2))
    assert report["concepts"]["missed_concepts"] and not report["comparison"]["comparison_ready"]

    profiles, declared_profiles = {}, []
    schedule = [("vlm", f"vlm__{model}") for model in MODELS]
    schedule += [("sam", condition) for condition in evaluation_cfg["segmentation"]["conditions"]]
    schedule += [("combined", f"vlm__{MODELS[0]}")]
    for component, condition in schedule:
        print(f"13: synthetic {component} profile {condition}", flush=True)
        profile_id = f"{component}_fixture_v1"
        profile_cfg = {**condition_cfg, "component": component, "condition_key": condition,
                       "profile_id": profile_id, "enable_combined": component == "combined",
                       "vlm_settings": deepcopy(RGBFixtureBackend(MODELS[0], settings).settings),
                       "_clock": clock, "_memory_probe": SyntheticProbe()}
        model = condition.removeprefix("vlm__")
        backend = RGBFixtureBackend(model, settings, clock) if component in {"vlm", "combined"} else None
        profile = profile_backend(profile_cfg, backend=backend,
                                  segmenter=segmenter if component in {"sam", "combined"} else None)
        assert profile["summary"]["fixture"] and profile["summary"]["complete"] and not profile["summary"]["comparison_ready"]
        profiles[f"{condition}/{profile_id}"] = profile
        base = f"benchmark/{condition}/{profile_id}"
        declared_profiles.append({"summary_path": base + "/summary.json",
                                  "metadata_path": base + "/metadata.json",
                                  "samples_path": base + "/samples.jsonl"})
    evaluation_cfg["deployment"]["profiles"] = declared_profiles
    print("12: discovery, source pixel metrics and synthetic resource summaries", flush=True)
    report = evaluate(evaluation_cfg)
    if report["validation"]["errors"]:
        raise AssertionError(json.dumps(report["validation"], indent=2))
    paths = export_report(report, run / "evaluation")

    # Changed input and legacy identities cannot silently resume; restore fixture bytes.
    rejection_checks = {}
    frame = read_jsonl(run / "frames.jsonl")[0]
    rgb = data / frame["image_path"]
    original = rgb.read_bytes()
    rgb.write_bytes(original + b"changed-input-fixture")
    try:
        for name, action in {
            "data": lambda: validate_run(run, data),
            "inference": lambda: run_inference(run, MODELS[0], settings, data, backend=RGBFixtureBackend(MODELS[0], settings)),
            "segmentation": lambda: run_conditions(condition_cfg, segmenter=segmenter),
            "profile": lambda: profile_backend(profile_cfg, backend=backend, segmenter=segmenter),
        }.items():
            try:
                action()
            except (ValueError, RuntimeError):
                rejection_checks[name] = True
            else:
                raise AssertionError(f"Changed input was accepted by {name}")
    finally:
        rgb.write_bytes(original)
    predictions = run / f"predictions/{MODELS[0]}.jsonl"
    saved = predictions.read_bytes()
    try:
        atomic_jsonl(predictions, records[:-1])
        missing = evaluate(evaluation_cfg)
        assert not missing["comparison"]["discovery_complete"]
        rejection_checks["missing_record_blocks_comparison"] = True
        legacy_rows = deepcopy(records)
        legacy_rows[0]["task_id"] = "region_classification_v1"
        atomic_jsonl(predictions, legacy_rows)
        legacy = evaluate(evaluation_cfg)
        assert legacy["validation"]["errors"]
        rejection_checks["legacy_scoring"] = True
    finally:
        predictions.write_bytes(saved)
    legacy_run = runs / "legacy_fixture"
    legacy_run.mkdir(exist_ok=True)
    legacy_file = legacy_run / "regions.jsonl"
    legacy_bytes = b'{"region_id":"old"}\n'
    if legacy_file.exists():
        if legacy_file.read_bytes() != legacy_bytes:
            raise ValueError("Existing legacy fixture bytes changed; refusing replacement")
    else:
        legacy_file.write_bytes(legacy_bytes)
    before = (legacy_run / "regions.jsonl").read_bytes()
    try:
        run_inference(legacy_run, MODELS[0], settings, data, backend=RGBFixtureBackend(MODELS[0], settings))
    except ValueError:
        rejection_checks["legacy_resume"] = True
    else:
        raise AssertionError("Legacy region run resumed as hazard task")
    assert (legacy_run / "regions.jsonl").read_bytes() == before
    summary = {"task_id": "hazard_prompt_v1", "schema_version": 1, "fixture": True,
               "execution_kind": "injected_fixture", "real_accuracy_evidence": False,
               "real_memory_evidence": False, "run_dir": str(run), "data_validation": validation,
               "selection_counts": {"test": 20, "warmup": 5}, "inference": inference,
               "segmentation": segmentation, "profiles": list(profiles),
               "rejection_checks": rejection_checks, "exports": paths,
               "comparison": report["comparison"]}
    atomic_json(output / "fixture_summary.json", summary)
    atomic_json(output / "evaluation_config.json", evaluation_cfg)
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, help="New or empty isolated fixture directory")
    parser.add_argument("--resume", action="store_true", help="Resume an unchanged marked fixture from this workflow")
    arguments = parser.parse_args()
    print(json.dumps(run_fixture(arguments.output_root, resume=arguments.resume), indent=2))
