"""Index JSON evidence once; read only selected RGB and original-size masks."""

from dataclasses import dataclass, field
from pathlib import Path
import os

import numpy as np
from PIL import Image

from traversability_hazard_evaluation.artifacts import (
    ArtifactError, binary_mask, read_json, read_records, safe_path,
)


def _index(rows, keys):
    result = {}
    for row in rows:
        key = tuple(row.get(k) for k in keys)
        if any(not isinstance(v, str) or not v for v in key):
            raise ArtifactError(f"Invalid identity {keys}: {key}")
        if key in result:
            raise ArtifactError(f"Duplicate identity {keys}: {key}")
        result[key] = row
    return result


def _records(path):
    issues = []
    rows, invalid = read_records(path, "frame_id", issues, "visualization")
    if issues or invalid:
        raise ArtifactError(f"Invalid JSONL {path}: {issues}")
    return rows


def _identity(row):
    if (row.get("task_id") != "hazard_prompt_v1"
            or type(row.get("schema_version")) is not int or row["schema_version"] != 1):
        raise ArtifactError("Expected hazard_prompt_v1 schema 1; legacy region artifacts are unsupported")


def check_output_directory(output, data, runs, component):
    """Contain explicit exports without touching or re-indexing input artifacts."""
    output = Path(output).resolve()
    protected = [data, runs, *[component / name for name in
                              ("src", "notebooks", "configs", "docs", "scripts", "tests", "requirements", ".git")]]
    for path in protected:
        path = Path(path).resolve()
        if output.is_relative_to(path) or path.is_relative_to(output):
            raise ArtifactError("Visualization output directory overlaps saved inputs or source files")
    return output


def profile_identity(row):
    # Native report rows retain summary_path rather than a separate profile_id.
    profile = row.get("profile_id")
    if not profile:
        path = row.get("summary_path")
        if not path:
            raise ArtifactError("Deployment row has no profile identity")
        profile = Path(path).parent.name
    # Rejected profiles may have only summary_path plus an error. Preserve them
    # visibly without inferring their component from a directory name.
    return (row.get("component", "unavailable"), row.get("condition_key", "unavailable"), profile)


@dataclass
class SavedRun:
    config: dict
    component_root: Path
    data_root: Path
    run_dir: Path
    output_dir: Path
    report: dict | None = None
    frames: dict = field(default_factory=dict)
    references: dict = field(default_factory=dict)
    phrases: dict = field(default_factory=dict)
    sam_audits: dict = field(default_factory=dict)
    predictions: dict = field(default_factory=dict)
    profiles: dict = field(default_factory=dict)
    messages: list = field(default_factory=list)

    @property
    def fixture(self):
        return bool(self.report and self.report["fixture"])

    def rows(self, section, table="rows", **selectors):
        return [row for row in (self.report or {}).get(section, {}).get(table, [])
                if all(value is None or row.get(key) == value for key, value in selectors.items())]

    def status(self):
        return {"fixture": self.fixture, "messages": self.messages,
                **{k: (self.report or {}).get(k, {}) for k in ("validation", "coverage", "comparison")},
                "completeness": {s: (self.report or {}).get(s, {}).get("complete")
                                 for s in ("concepts", "segmentation", "deployment")}}

    def rgb(self, frame_id):
        frame = self.frames[frame_id]
        try:
            with Image.open(safe_path(self.data_root, frame["image_path"])) as image:
                if image.size != (frame["width"], frame["height"]):
                    raise ArtifactError("RGB dimensions differ from saved frame")
                return np.asarray(image.convert("RGB")).copy()
        except OSError as error:
            raise ArtifactError(f"RGB unavailable: {error}") from error

    def reference_masks(self, frame_id):
        frame, ref = self.frames[frame_id], self.references.get(frame_id)
        if not ref or ref.get("status") != "complete":
            raise ArtifactError("Annotation reference pending, missing or incomplete")
        size = (frame["width"], frame["height"])
        valid = binary_mask(self.data_root, ref["valid_mask_path"], size)
        masks = {name: binary_mask(self.data_root, path, size) & valid
                 for name, path in ref["concept_masks"].items() if name in ref["scored_concepts"]}
        union = np.zeros_like(valid)
        for mask in masks.values():
            union |= mask
        return valid, masks, union


def load_run(config=None):
    """Load a saved report and exact joins; missing inputs yield explicit empty states.

    Explicit roots override environment roots. Embedded historical absolute roots
    in the report are informational only, so saved runs may be relocated.
    """
    config = dict(config or {})
    component = Path(config.get("component_root") or os.environ.get("TRAVERSABILITY_REPO_ROOT")
                     or Path(__file__).resolve().parents[2]).expanduser().resolve()
    if (component / "VLM_evaluation/pyproject.toml").is_file():
        component /= "VLM_evaluation"
    def root(key, default):
        value = Path(config.get(key) or os.environ.get("TRAVERSABILITY_" + key.upper()) or default).expanduser()
        return (value if value.is_absolute() else component / value).resolve()
    data, runs = root("data_root", "data"), root("run_root", "runs")
    name = config.get("run_name", "hazard_prompt_v1")
    if not isinstance(name, str) or "/" in name:
        raise ArtifactError("run_name must be one safe path component")
    run = safe_path(runs, name)
    output = root("output_dir", "visualization_outputs")
    output = check_output_directory(output, data, runs, component)
    saved = SavedRun(config, component, data, run, output)
    path = safe_path(run, "evaluation/hazard_report.json")
    if not path.is_file():
        saved.messages.append(f"No saved hazard report at {path}. Supply an evaluated saved run; no fixture is loaded implicitly.")
        return saved
    report = read_json(path)
    _identity(report)
    if type(report.get("fixture")) is not bool:
        raise ArtifactError("Report must declare a boolean fixture marker")
    if report.get("identity", {}).get("run_name") != name:
        raise ArtifactError("Report run_name does not match selected run")
    saved.report = report
    for section, keys in (("concepts", ("model_key", "source", "split")),
                          ("segmentation", ("condition_key", "source", "split"))):
        _index(report.get(section, {}).get("rows", []), keys)
        _index(report.get(section, {}).get("per_concept", []), (*keys, "concept"))
    saved.phrases = _index(report.get("concepts", {}).get("phrase_audit", []), ("model_key", "frame_id"))
    saved.sam_audits = _index(report.get("segmentation", {}).get("frame_audit", []), ("condition_key", "frame_id"))
    _index(report.get("concepts", {}).get("missed_concepts", []), ("model_key", "frame_id", "concept"))
    for audit in saved.sam_audits.values():
        _index(audit.get("queries", []), ("query_id",))
        for query in audit.get("queries", []):
            if not query["query_id"].startswith(f"{audit['condition_key']}:{audit['frame_id']}:q"):
                raise ArtifactError("SAM query has a mismatched exact frame/condition join")
    for row in report.get("deployment", {}).get("rows", []):
        if "task_id" in row:
            _identity(row)
        if "fixture" in row and row["fixture"] != report["fixture"]:
            raise ArtifactError("Deployment/report fixture identity mismatch")
        key = profile_identity(row)
        if key in saved.profiles or any(not isinstance(v, str) or not v for v in key):
            raise ArtifactError(f"Invalid or duplicate deployment profile identity: {key}")
        if row.get("summary_path"):
            safe_path(run, row["summary_path"])
        saved.profiles[key] = row
    for filename, attribute in (("frames.jsonl", "frames"), ("references.jsonl", "references")):
        path = safe_path(run, filename)
        if path.is_file():
            setattr(saved, attribute, _records(path))
        else:
            saved.messages.append(f"Missing {filename}; image/reference view unavailable. Report tables remain usable.")
    if saved.frames:
        if set(saved.references) - set(saved.frames):
            raise ArtifactError("Reference frame IDs do not join exactly to frame manifest")
        requested = report.get("coverage", {}).get("requested_frame_ids", [])
        if not isinstance(requested, list) or len(set(requested)) != len(requested):
            raise ArtifactError("Duplicate or invalid report requested frame identities")
        if set(saved.frames) - set(requested):
            raise ArtifactError("Frame manifest IDs outside report's exact requested frame IDs")
        for frame in saved.frames.values():
            if any(type(frame.get(k)) is not int or frame[k] <= 0 for k in ("width", "height")):
                raise ArtifactError("Invalid frame dimensions")
            safe_path(data, frame.get("image_path"))
        policy = read_json(component / "configs/hazards/policy.json")
        vocabularies = {"rellis": set(policy["datasets"]["rellis"]["hazard_label_ids"]),
                        "coco": set(policy["datasets"]["coco"]["scored_category_names"])}
        for ref in saved.references.values():
            if ref.get("status") == "complete":
                vocabulary = vocabularies.get(saved.frames[ref["frame_id"]]["source"])
                if vocabulary is None or set(ref.get("scored_concepts", [])) != vocabulary:
                    raise ArtifactError("Reference vocabulary differs from frozen source contract")
                if not set(ref.get("present_concepts", [])) <= vocabulary:
                    raise ArtifactError("Reference positive concept outside source vocabulary")
                safe_path(data, ref.get("valid_mask_path"))
                for concept, path in ref.get("concept_masks", {}).items():
                    if concept not in vocabulary:
                        raise ArtifactError("Reference mask outside source vocabulary")
                    safe_path(data, path)
                if not set(ref.get("present_concepts", [])) <= set(ref.get("concept_masks", {})):
                    raise ArtifactError("Missing source-reference concept mask paths")
        for audit in [*saved.phrases.values(), *saved.sam_audits.values(),
                      *report.get("concepts", {}).get("missed_concepts", []),
                      *report.get("concepts", {}).get("failures", [])]:
            frame = saved.frames.get(audit.get("frame_id"))
            if frame is None or any(audit.get(k) != frame[k] for k in ("source", "split")):
                raise ArtifactError("Report evidence has a mismatched exact frame/source/split join")
        # Validate all query paths before reading any selected mask; exact joins
        # remain opaque (no substring matching on frame IDs).
        for model in sorted({key[0] for key in saved.phrases}):
            path = safe_path(run, f"predictions/{model}.jsonl")
            if not path.is_file():
                continue  # Embedded phrase audit is sufficient; no duplicate export required.
            for identifier, prediction in _records(path).items():
                _identity(prediction)
                if identifier not in saved.frames or prediction.get("model_key") != model:
                    raise ArtifactError("Prediction model/frame identity mismatch")
                audit = saved.phrases.get((model, identifier))
                if audit and (audit.get("raw_response") != prediction.get("raw_response")
                              or audit.get("saved_prompts") != prediction.get("prompts")
                              or audit.get("prediction_status") != prediction.get("status")):
                    raise ArtifactError("Prediction differs from saved evaluated phrase audit")
                saved.predictions[(model, identifier)] = prediction
    for audit in saved.sam_audits.values():
        if audit.get("full_union_path"):
            safe_path(run, audit["full_union_path"])
        for query in audit.get("queries", []):
            for path in [*(query.get("mask_paths") or []), *([query["union_mask_path"]] if query.get("union_mask_path") else [])]:
                safe_path(run, path)
    return saved


def reconstruct_overlay(saved, frame_id, condition):
    """Reconstruct source-scored successful masks, following saved evaluator audits.

    A failed frame can retain valid successful queries. Failed partial, unscored,
    invalid and missing queries never enter scoring. No report metric is changed.
    """
    audit = saved.sam_audits.get((condition, frame_id))
    if audit is None:
        raise ArtifactError("No saved SAM frame audit; supply evaluated fixed-SAM outputs for this frame/condition")
    if audit.get("pixel_metrics_available") is not True:
        raise ArtifactError("Scored overlay unavailable: saved pixel metrics unavailable")
    errors = (saved.report or {}).get("validation", {}).get("errors", [])
    if any(e.get("condition_key") == condition and e.get("frame_id") == frame_id
           and e.get("code") in ("invalid_segmentation_frame", "missing_or_invalid_segmentation_frame") for e in errors):
        raise ArtifactError("Scored overlay unavailable: evaluator rejected the SAM frame")
    valid, truths, truth = saved.reference_masks(frame_id)
    size = (valid.shape[1], valid.shape[0])
    scored = np.zeros_like(valid)
    by_concept = {}
    excluded = set(audit.get("invalid_queries", [])) | set(audit.get("missing_queries", []))
    excluded |= {q["query_id"] if isinstance(q, dict) else q for q in audit.get("failed_queries", [])}
    for query in audit.get("queries", []):
        if (query["query_id"] in excluded or query.get("status") != "ok"
                or not query.get("saved") or not query.get("source_scored")):
            continue
        canonical = query.get("canonical_concept")
        if canonical not in saved.references[frame_id]["scored_concepts"]:
            raise ArtifactError("SAM source-scored query contradicts source vocabulary")
        paths = query.get("mask_paths")
        if not isinstance(paths, list) or len(paths) != len(set(paths)):
            raise ArtifactError("Invalid successful query instance paths")
        mask = binary_mask(saved.run_dir, query.get("union_mask_path"), size)
        instances = np.zeros_like(valid)
        for path in paths:
            instances |= binary_mask(saved.run_dir, path, size)
        if not np.array_equal(mask, instances):
            raise ArtifactError("Successful query union differs from saved instance masks")
        mask &= valid
        scored |= mask
        by_concept.setdefault(canonical, np.zeros_like(valid))[:] |= mask
    counts = {"pixel_tp": int(np.count_nonzero(scored & truth)),
              "pixel_fn": int(np.count_nonzero(truth & ~scored)),
              "pixel_fp": int(np.count_nonzero(scored & ~truth))}
    for key, value in counts.items():
        if audit.get(key) is not None and audit[key] != value:
            raise ArtifactError(f"Scored overlay unavailable: reconstructed {key}={value} differs from saved {audit[key]}")
    return {"scored": scored, "valid": valid, "reference": truth, "reference_concepts": truths,
            "prediction_concepts": by_concept, "counts": counts, "audit": audit}
