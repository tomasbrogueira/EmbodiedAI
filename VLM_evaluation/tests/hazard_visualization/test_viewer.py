"""Focused CPU saved-artifact contracts; no producer, model or judge runs."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pytest

from traversability_hazard_evaluation.artifacts import ArtifactError, binary_mask, safe_path
from traversability_hazard_visualization import (
    load_run, reconstruct_overlay, render_discovery, render_segmentation,
    render_deployment, render_image, export_figure, create_viewer,
)
from traversability_hazard_visualization.rendering import image_diagnostics, rate_text

COMPONENT = Path(__file__).resolve().parents[2]
FRAME = "coco:val2017:exact:01"
MODEL = "qwen3_vl_4b"
CONDITION = "vlm__" + MODEL


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def mask(path, pixels, size=(4, 3)):
    values = np.zeros((size[1], size[0]), dtype=np.uint8)
    for index in pixels:
        values.flat[index] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values).save(path)


@pytest.fixture
def saved_fixture(tmp_path):
    data, run = tmp_path / "data", tmp_path / "runs/saved"
    data.mkdir()
    Image.new("RGB", (4, 3), color=(80, 120, 150)).save(data / "rgb.png")
    frame = {"frame_id": FRAME, "source": "coco", "split": "test", "image_path": "rgb.png", "width": 4, "height": 3}
    ref = {"frame_id": FRAME, "status": "complete", "scored_concepts": ["person", "cup", "bottle", "chair", "dog"],
           "present_concepts": ["person"], "concept_masks": {"person": "person.png"},
           "valid_mask_path": "valid.png", "absence_scoring_eligible": False}
    write_rows(run / "frames.jsonl", [frame])
    write_rows(run / "references.jsonl", [ref])
    mask(data / "person.png", [0, 4])
    mask(data / "valid.png", [i for i in range(12) if i != 2])
    mask(run / "sam/scored.png", [0, 2])
    mask(run / "sam/unscored.png", [1])
    mask(run / "sam/partial.png", [3, 4])
    mask(run / "sam/all.png", [0, 1, 2, 3, 4])
    queries = []
    for i, (phrase, canonical, path, status, scored) in enumerate([
            ("person", "person", "scored", "ok", True),
            ("traffic cone", None, "unscored", "ok", False),
            ("person", "person", "partial", "error", True)]):
        queries.append({"query_id": f"{CONDITION}:{FRAME}:q{i:03d}", "phrase": phrase,
                        "canonical_concept": canonical, "source_scored": scored, "saved": True,
                        "status": status, "error_code": "query_failure" if status == "error" else None,
                        "union_mask_path": f"sam/{path}.png", "mask_paths": [f"sam/{path}.png"]})
    audit = {"condition_key": CONDITION, "frame_id": FRAME, "source": "coco", "split": "test",
             "queries": queries, "full_union_path": "sam/all.png", "failed": True,
             "upstream_status": "ok", "required_queries": 3,
             "invalid_queries": [], "missing_queries": [], "failed_queries": [{"query_id": queries[2]["query_id"]}],
             "pixel_metrics_available": True, "pixel_tp": 1, "pixel_fn": 1, "pixel_fp": 0}
    rate = {"numerator": 1, "denominator": 2, "rate": .5}
    null = {"numerator": 0, "denominator": 0, "rate": None}
    row = {"model_key": MODEL, "source": "coco", "split": "test", "requested_frames": 1, "prediction_records": 1,
           "reference_complete": 1, "status": "partial", "counts_are_provisional": True,
           "metrics": {"concept_recall": rate, "supported_precision": rate, "tiny_concept_recall": null, "failure_rate": rate}}
    phrase = {"frame_id": FRAME, "model_key": MODEL, "source": "coco", "split": "test", "prediction_status": "ok",
              "raw_response": '{"prompts":["person","traffic cone"]}', "saved_prompts": ["person", "traffic cone"],
              "phrases": [{"raw_phrase": "person", "canonical_concept": "person", "category": "source_concept"},
                          {"raw_phrase": "traffic cone", "canonical_concept": None, "category": "unknown_unscored"}]}
    profile = {"component": "combined", "condition_key": CONDITION, "summary_path": f"benchmark/{CONDITION}/explicit_profile/summary.json",
               "valid": True, "complete": False, "comparison_ready": False, "fixture": True,
               "latency": {"median_s": None, "p95_s": .5}, "memory": {"allocated_peak_bytes": None,
               "reserved_peak_bytes": 30, "device_used_peak_bytes": 99, "incremental_allocated_peak_bytes": None,
               "baseline_allocated_bytes": 10, "unavailable_reasons": ["synthetic fixture"]},
               "budget": {"status": "unmeasured", "provenance": "proposed"}, "measured_samples": 1, "valid_measured_calls": 0,
               "expected_measured_frames": 2, "unique_test_frames": 1, "failed_calls": 1, "sam_query_count": None}
    report = {"task_id": "hazard_prompt_v1", "schema_version": 1, "fixture": True,
              "identity": {"run_name": "saved", "data_root": "historical/unusable/path"}, "validation": {"errors": []},
              "coverage": {"requested_frame_ids": [FRAME], "pending_coverage": ["cables"]},
              "comparison": {"comparison_ready": False, "model_selection_ready": False},
              "concepts": {"rows": [row, {**row, "source": "all"}], "complete": False,
                           "phrase_audit": [phrase], "failures": [],
                           "missed_concepts": [{"frame_id": FRAME, "model_key": MODEL, "source": "coco", "split": "test", "concept": "person"}],
                           "per_concept": [{**{k: row[k] for k in ("model_key", "source", "split", "counts_are_provisional")},
                                            "concept": "person", "concept_recall": rate}]},
              "segmentation": {"frame_audit": [audit], "complete": True, "rows": [], "per_concept": []},
              "deployment": {"rows": [profile], "complete": False}}
    report_path = run / "evaluation/hazard_report.json"
    write_json(report_path, report)
    config = {"component_root": str(COMPONENT), "data_root": str(data), "run_root": str(tmp_path / "runs"),
              "run_name": "saved", "output_dir": str(tmp_path / "exports")}
    return config, report, report_path


def test_source_scored_success_survives_other_query_failure(saved_fixture):
    config, _, _ = saved_fixture
    saved = load_run(config)
    result = reconstruct_overlay(saved, FRAME, CONDITION)
    assert result["counts"] == {"pixel_tp": 1, "pixel_fn": 1, "pixel_fp": 0}
    assert np.flatnonzero(result["scored"]).tolist() == [0]
    assert np.flatnonzero(binary_mask(saved.run_dir, "sam/all.png", (4, 3))).tolist() == [0, 1, 2, 3, 4]
    assert not result["scored"].flat[2]  # ignored pixel
    assert not result["scored"].flat[4]  # failed partial annotated positive remains missed


@pytest.mark.parametrize("field", ["invalid_queries", "missing_queries"])
def test_successful_but_evaluator_excluded_query_is_not_scored(saved_fixture, field):
    config, report, path = saved_fixture
    audit = report["segmentation"]["frame_audit"][0]
    audit[field] = [audit["queries"][0]["query_id"]]
    audit.update(pixel_tp=0, pixel_fn=2, pixel_fp=0)
    write_json(path, report)
    assert not reconstruct_overlay(load_run(config), FRAME, CONDITION)["scored"].any()


@pytest.mark.parametrize("mutation", ["counts", "instance_union", "dimensions", "missing_asset", "frame_rejected", "reference_unavailable"])
def test_unavailable_overlay_never_changes_metrics(saved_fixture, mutation):
    config, report, path = saved_fixture
    if mutation == "counts":
        report["segmentation"]["frame_audit"][0]["pixel_fp"] = 8
    elif mutation == "frame_rejected":
        report["validation"]["errors"] = [{"code": "invalid_segmentation_frame", "condition_key": CONDITION, "frame_id": FRAME}]
    elif mutation == "reference_unavailable":
        report["segmentation"]["frame_audit"][0]["pixel_metrics_available"] = False
    elif mutation == "instance_union":
        mask(Path(config["run_root"])/"saved/sam/instance.png", [])
        report["segmentation"]["frame_audit"][0]["queries"][0]["mask_paths"] = ["sam/instance.png"]
    elif mutation == "dimensions":
        mask(Path(config["run_root"])/"saved/sam/scored.png", [0], (2, 2))
    elif mutation == "missing_asset":
        (Path(config["run_root"])/"saved/sam/scored.png").unlink()
    write_json(path, report)
    saved = load_run(config)
    before = copy.deepcopy(saved.report)
    with pytest.raises(ArtifactError):
        reconstruct_overlay(saved, FRAME, CONDITION)
    figure = render_image(saved, FRAME, model=MODEL, condition=CONDITION)
    assert any("unavailable" in ax.get_title() for ax in figure.axes)
    assert saved.report == before
    plt.close(figure)


@pytest.mark.parametrize("relative", ["../escape.png", "/absolute.png", "C:/asset.png", "a\\b.png", "a/../b", "a//b"])
def test_path_escapes_rejected_before_any_mask_read(saved_fixture, relative):
    config, report, path = saved_fixture
    report["segmentation"]["frame_audit"][0]["queries"][0]["union_mask_path"] = relative
    write_json(path, report)
    with pytest.raises(ArtifactError, match="path"):
        load_run(config)


def test_symlink_containment(tmp_path):
    root, outside = tmp_path/"inside", tmp_path/"outside"
    root.mkdir()
    outside.mkdir()
    try:
        (root/"escape").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Native symlink creation unavailable on this host")
    with pytest.raises(ArtifactError, match="escapes"):
        safe_path(root, "escape/mask.png")


@pytest.mark.parametrize("kind", ["frame_duplicate", "reference_join", "phrase_join", "query_join", "report_duplicate", "profile_duplicate", "legacy", "schema_boolean"])
def test_identity_and_exact_join_rejection(saved_fixture, kind):
    config, report, path = saved_fixture
    run = Path(config["run_root"])/"saved"
    if kind == "frame_duplicate":
        row = json.loads((run/"frames.jsonl").read_text())
        write_rows(run/"frames.jsonl", [row, row])
    elif kind == "reference_join":
        row = json.loads((run/"references.jsonl").read_text())
        row["frame_id"] = FRAME+"0"
        write_rows(run/"references.jsonl", [row])
    elif kind == "phrase_join":
        report["concepts"]["phrase_audit"][0]["split"] = "development"
    elif kind == "query_join":
        report["segmentation"]["frame_audit"][0]["queries"][0]["query_id"] = "wrong-frame:q000"
    elif kind == "report_duplicate":
        report["concepts"]["rows"].append(report["concepts"]["rows"][0])
    elif kind == "profile_duplicate":
        report["deployment"]["rows"].append(report["deployment"]["rows"][0])
    elif kind == "legacy":
        report["task_id"] = "region_classification_v1"
    elif kind == "schema_boolean":
        report["schema_version"] = True
    write_json(path, report)
    with pytest.raises(ArtifactError):
        load_run(config)


def test_null_provisional_fixture_and_no_double_count(saved_fixture):
    config, report, _ = saved_fixture
    saved = load_run(config)
    assert len(saved.rows("concepts", source="coco", split="test")) == 1
    assert len(saved.rows("concepts", source="all", split="test")) == 1
    assert rate_text({"rate": None, "numerator": 0, "denominator": 0}) == "unavailable (0/0)"
    figure = render_discovery(saved, source="coco")
    assert "SYNTHETIC DATA" in figure._suptitle.get_text()
    assert figure._hazard_export["provisional"]
    assert any("PROVISIONAL" in t.get_text() for t in figure.texts)
    assert figure._hazard_export["selected_rows"][0]["metrics"] == report["concepts"]["rows"][0]["metrics"]
    plt.close(figure)


def test_explicit_profiles_keep_nulls_memory_and_budget(saved_fixture):
    saved = load_run(saved_fixture[0])
    empty = render_deployment(saved)
    assert not empty._hazard_export["selected_rows"]
    key = next(iter(saved.profiles))
    figure = render_deployment(saved, [key])
    row = figure._hazard_export["selected_rows"][0]
    assert row["latency"]["median_s"] is None
    assert row["memory"]["allocated_peak_bytes"] is None
    assert row["memory"]["device_used_peak_bytes"] == 99
    assert row["budget"]["provenance"] == "proposed"
    assert key == ("combined", CONDITION, "explicit_profile")
    with pytest.raises(ArtifactError):
        render_deployment(saved, [key, key])
    plt.close("all")


def test_sam_direct_rates_and_prompt_coverage_remain_separate(saved_fixture):
    config, report, path = saved_fixture
    half = {"rate": .5, "numerator": 1, "denominator": 2}
    unavailable = {"rate": None, "numerator": None, "denominator": None}
    row = {"condition_key": CONDITION, "source": "coco", "split": "test", "saved_frames": 1,
           "expected_frames": 1, "reference_frames": 1, "status": "partial", "complete": False,
           "metrics_available": False, "hazard_recall": unavailable, "precision": unavailable,
           "iou": unavailable, "frame_failure_rate": half, "query_failure_rate": half}
    report["segmentation"]["rows"] = [row]
    report["segmentation"]["per_concept"] = [{**row, "concept": "person", "prompt_coverage": half,
                                               "successful_query_coverage": half}]
    write_json(path, report)
    fig = render_segmentation(load_run(config), source="coco")
    assert len(fig.axes[1].patches) == 2  # Prompt and success coverage, unavailable pixel recall has no bar.
    assert fig._hazard_export["provisional"]
    assert fig._hazard_export["selected_rows"][0]["hazard_recall"] == unavailable
    plt.close(fig)


def test_real_report_has_no_synthetic_label_and_keeps_readiness_false(saved_fixture):
    config, report, path = saved_fixture
    report["fixture"] = False
    report["deployment"]["rows"][0]["fixture"] = False
    write_json(path, report)
    saved = load_run(config)
    fig = render_discovery(saved)
    assert "SYNTHETIC" not in fig._suptitle.get_text()
    assert saved.status()["comparison"]["comparison_ready"] is False
    plt.close(fig)


def test_rejected_profile_without_component_stays_visible(saved_fixture):
    config, report, path = saved_fixture
    report["deployment"]["rows"] = [{"summary_path": "benchmark/bad/invalid/summary.json", "valid": False, "complete": False, "error": "missing samples"}]
    write_json(path, report)
    saved = load_run(config)
    fig = render_deployment(saved, list(saved.profiles))
    assert fig._hazard_export["selected_rows"][0]["error"] == "missing samples"
    plt.close(fig)


def test_eight_missing_native_profiles_keep_distinct_expected_identities(saved_fixture):
    config, report, path = saved_fixture
    conditions = ["vlm__qwen3_vl_4b", "vlm__qwen3_5_4b"]
    planned = [(condition, "kth_v1_vlm") for condition in conditions]
    planned += [(condition, "kth_v1_sam") for condition in conditions + ["reference_present", "fixed_policy"]]
    planned += [(condition, "kth_v1_combined") for condition in conditions]
    rows = [{"summary_path": f"benchmark/{condition}/{profile}/summary.json",
             "valid": False, "complete": False, "comparison_ready": False,
             "error": "missing profile metadata"} for condition, profile in planned]
    report["deployment"]["rows"] = rows
    write_json(path, report)
    saved = load_run(config)
    assert set(saved.profiles) == {("unavailable", "expected:" + condition, profile)
                                   for condition, profile in planned}
    assert list(saved.profiles.values()) == rows
    assert all("component" not in row and "condition_key" not in row for row in saved.profiles.values())
    figure = render_deployment(saved, list(saved.profiles))
    assert len(figure._hazard_export["selected_rows"]) == 8
    assert all(row["valid"] is False and row["comparison_ready"] is False
               for row in figure._hazard_export["selected_rows"])
    assert saved.status()["comparison"]["model_selection_ready"] is False
    plt.close(figure)
    report["deployment"]["rows"].append(copy.deepcopy(rows[0]))
    write_json(path, report)
    with pytest.raises(ArtifactError, match="duplicate deployment profile identity"):
        load_run(config)


@pytest.mark.parametrize("summary_path", ["../benchmark/c/p/summary.json",
                                         "benchmark/c/../p/summary.json",
                                         "benchmark\\c\\p\\summary.json"])
def test_expected_profile_identity_requires_safe_summary_path(saved_fixture, summary_path):
    config, report, path = saved_fixture
    report["deployment"]["rows"] = [{"summary_path": summary_path, "valid": False}]
    write_json(path, report)
    with pytest.raises(ArtifactError):
        load_run(config)


def test_expected_profile_path_does_not_replace_observed_identity(saved_fixture):
    config, report, path = saved_fixture
    row = report["deployment"]["rows"][0]
    row["summary_path"] = "benchmark/expected_condition/explicit_profile/summary.json"
    write_json(path, report)
    saved = load_run(config)
    assert next(iter(saved.profiles)) == ("combined", CONDITION, "explicit_profile")
    assert next(iter(saved.profiles.values())) == row


@pytest.mark.parametrize("outcome", ["ok_empty", "error", "missing"])
def test_response_failure_and_empty_distinction(saved_fixture, outcome):
    config, report, path = saved_fixture
    audit = report["concepts"]["phrase_audit"][0]
    if outcome == "missing":
        report["concepts"]["phrase_audit"] = []
    elif outcome == "error":
        audit["prediction_status"] = "error"
    else:
        audit.update(saved_prompts=[], phrases=[], raw_response='{"prompts":[]}')
    write_json(path, report)
    saved = load_run(config)
    actual = image_diagnostics(saved, FRAME, MODEL)["outcome"]
    assert {"ok_empty": "Successful empty", "error": "Response failure", "missing": "Missing/invalid"}[outcome] in actual


def test_export_is_opt_in_contained_labeled_and_non_overwriting(saved_fixture):
    config, _, path = saved_fixture
    before = path.read_bytes()
    saved = load_run(config)
    fig = render_image(saved, FRAME, model=MODEL, condition=CONDITION)
    with pytest.raises(ArtifactError, match="disabled"):
        export_figure(saved, fig, "image.png", enabled=True)
    config["export_enabled"] = True
    saved = load_run(config)
    for bad in ["../source.png", "bad.pdf"]:
        with pytest.raises(ArtifactError):
            export_figure(saved, fig, bad, enabled=True)
    target = export_figure(saved, fig, "image.svg", enabled=True)
    metadata = json.loads(target.with_name(target.name+".json").read_text())
    assert metadata["fixture"] and metadata["diagnostics"]["phrase_audit"]["phrases"][1]["raw_phrase"] == "traffic cone"
    assert "SYNTHETIC DATA" in target.read_text()
    with pytest.raises(ArtifactError, match="overwrite"):
        export_figure(saved, fig, "image.svg", enabled=True)
    assert path.read_bytes() == before
    for protected in [config["data_root"], str(path.parent), str(COMPONENT / "notebooks")]:
        with pytest.raises(ArtifactError, match="overlaps"):
            load_run({**config, "output_dir": protected})
    plt.close(fig)


def test_missing_report_and_optional_artifacts_are_informative(tmp_path, saved_fixture):
    saved = load_run({"component_root": str(COMPONENT), "data_root": str(tmp_path/"absent-data"), "run_root": str(tmp_path/"absent-runs")})
    assert saved.report is None and "No saved hazard report" in saved.messages[0]
    assert not (tmp_path/"absent-data").exists()
    for figure in [render_discovery(saved), render_image(saved), render_segmentation(saved), render_deployment(saved)]:
        assert figure._hazard_export["fixture"] is False
    config, report, path = saved_fixture
    report["segmentation"] = {}
    report["deployment"] = {}
    write_json(path, report)
    discovery_only = load_run(config)
    assert discovery_only.phrases and not discovery_only.sam_audits
    render_discovery(discovery_only)
    plt.close("all")


def test_relocated_roots_and_environment_defaults(saved_fixture, tmp_path, monkeypatch):
    config, _, _ = saved_fixture
    import shutil
    relocated = tmp_path/"relocated"
    shutil.copytree(Path(config["data_root"]), relocated/"data")
    shutil.copytree(Path(config["run_root"]), relocated/"runs")
    monkeypatch.setenv("TRAVERSABILITY_DATA_ROOT", str(relocated/"data"))
    monkeypatch.setenv("TRAVERSABILITY_RUN_ROOT", str(relocated/"runs"))
    monkeypatch.setenv("TRAVERSABILITY_REPO_ROOT", str(COMPONENT.parent))
    saved = load_run({"run_name": "saved", "output_dir": str(tmp_path/"render")})
    assert saved.component_root == COMPONENT
    assert saved.rgb(FRAME).shape == (3, 4, 3)
    assert reconstruct_overlay(saved, FRAME, CONDITION)["counts"]["pixel_tp"] == 1
    explicit = load_run(config)
    assert explicit.data_root == Path(config["data_root"])


def test_widget_selectors_only_render_indexed_evidence(saved_fixture, monkeypatch):
    pytest.importorskip("ipywidgets")
    from unittest.mock import Mock
    monkeypatch.setattr("IPython.display.display", Mock())
    saved = load_run(saved_fixture[0])
    viewer = create_viewer(saved)
    tabs = viewer.children[-1]
    assert len(tabs.children) == 4
    controls = tabs.children[1].children[0].children
    assert controls[1].value == "test"
    assert controls[2].value == FRAME
    controls[-1].value = .7
    controls[-2].value = "failed_partial"
    controls[5].value = saved.sam_audits[(CONDITION, FRAME)]["queries"][1]["query_id"]
    assert tabs.children[3].children[0].children[0].value == ()
    plt.close("all")


@pytest.mark.parametrize("use_fixture", [False, True])
def test_notebook_headless_no_models_or_artifact_mutation(saved_fixture, tmp_path, use_fixture):
    config, _, _ = saved_fixture
    if not use_fixture:
        config = {**config, "run_root": str(tmp_path/"missing")}
    # A fresh process blocks imports, including producer packages; report reading
    # may import evaluator readers but must never invoke evaluate().
    script = '''
import importlib.abc, sys, json, hashlib
from pathlib import Path
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'torch','transformers','sam3','traversability_hazard_inference','traversability_hazard_segmentation','traversability_hazard_benchmark','traversability_hazard_data'}:
   raise AssertionError('Forbidden import: '+fullname)
sys.meta_path.insert(0, Guard())
import traversability_hazard_evaluation
def forbidden_evaluate(*args, **kwargs):
 raise AssertionError('Visualization invoked evaluator')
traversability_hazard_evaluation.evaluate=forbidden_evaluate
config=json.loads(sys.argv[1])
def snapshot():
 return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for root in (config['data_root'],config['run_root']) for p in Path(root).rglob('*') if p.is_file()}
before=snapshot()
ns={'__name__':'__main__'}
nb=json.loads(Path('notebooks/14_hazard_visualization.ipynb').read_text())
for cell in nb['cells']:
 if cell['cell_type']=='code':
  exec(compile(''.join(cell['source']),cell['id'],'exec'),ns)
  if cell['id']=='hazard-visualization-config':
   ns.update(config=config,SHOW_WIDGETS=False,EXPORT=False,FRAME_ID="coco:val2017:exact:01",MODEL="qwen3_vl_4b",CONDITION="vlm__qwen3_vl_4b")
assert before==snapshot()
assert not {'torch','transformers','sam3'} & set(sys.modules)
'''
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONPATH": str(COMPONENT/"src"), "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run([sys.executable, "-B", "-c", script, json.dumps(config)], cwd=COMPONENT, env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout+result.stderr


def test_cpu_setup_and_notebook_registration():
    def module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        obj = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(obj)
        return obj
    setup = module("hazard_visualization_setup", COMPONENT/"scripts/setup_env.py")
    plan = setup.build_plan(setup.create_parser().parse_args(["--repo-root", str(COMPONENT), "--with-module", "hazard_visualization", "--dry-run"]))
    assert plan["profile"] == "cpu" and not plan["constraints"]
    assert any(p.endswith("hazard_visualization.txt") for p in plan["requirements"])
    checker = module("hazard_visualization_checker", COMPONENT/"scripts/check_notebooks.py")
    assert "14_hazard_visualization.ipynb" in checker.HAZARD_NOTEBOOKS
    assert checker.safe_overrides("14_hazard_visualization.ipynb", fixture=True) == {"SHOW_WIDGETS": False, "EXPORT": False}
