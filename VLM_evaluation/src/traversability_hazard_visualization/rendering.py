"""Public Matplotlib render path, independent of notebook/widget frontends."""

import json
from pathlib import Path
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from traversability_hazard_evaluation.artifacts import ArtifactError, binary_mask, safe_path
from .loading import reconstruct_overlay

COLORS = {"reference": "#2ca25f", "scored": "#3182bd", "ignored": "#969696",
          "all_query": "#756bb1", "failed_partial": "#e6550d", "query": "#d4a017"}
METRICS = ("concept_recall", "supported_precision", "tiny_concept_recall", "failure_rate")


def rate_text(value, available=True):
    value = value or {}
    counts = f"{value.get('numerator', '?')}/{value.get('denominator', '?')}"
    return f"{value['rate']:.1%} ({counts})" if available and value.get("rate") is not None else f"unavailable ({counts})"


def _plain(value):
    return "unavailable" if value is None else str(value)


def _stamp(saved, figure, title, context, evidence=None):
    status = saved.status()
    provisional = any(r.get("counts_are_provisional") or r.get("complete") is False
                      or r.get("status") in ("partial", "incomplete") for r in (evidence or []))
    fixture = "SYNTHETIC DATA / SOFTWARE CHECK ONLY — scripted latency is not hardware evidence\n" if saved.fixture else ""
    caption = "\n".join(textwrap.wrap(f"Run: {saved.run_dir.name} | {context}", 155))
    figure.suptitle(f"{fixture}{title}\n{caption}", fontsize=11,
                   color="#a33a13" if saved.fixture else "#222222", y=.99)
    comparison = (saved.report or {}).get("comparison", {})
    errors = len((saved.report or {}).get("validation", {}).get("errors", []))
    complete = status["completeness"]
    footer = (f"{'PROVISIONAL / INCOMPLETE' if provisional else 'Saved evidence'} | validation errors: {errors} | "
              f"complete: discovery={complete['concepts']}, SAM={complete['segmentation']}, deployment={complete['deployment']}\n"
              f"comparison_ready={comparison.get('comparison_ready', 'unavailable')}; "
              f"model_selection_ready={comparison.get('model_selection_ready', 'unavailable')}. "
              "Annotated concepts only; no physical safety or path-quality claim.")
    figure.text(.02, .012, footer, fontsize=8, va="bottom")
    figure._hazard_export = {"run_name": saved.run_dir.name, "fixture": saved.fixture,
                            "evidence_kind": "synthetic_software_check" if saved.fixture else "saved_report",
                            "context": context, "provisional": provisional, "status": status,
                            "selected_rows": evidence or []}
    return figure


def _empty(saved, title, message, context="no selected samples"):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis("off")
    ax.text(.03, .65, textwrap.fill(message, 110), transform=ax.transAxes, va="top", fontsize=11)
    fig.subplots_adjust(top=.72, bottom=.2)
    return _stamp(saved, fig, title, context)


def _table(ax, columns, rows, widths=None):
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="left", colWidths=widths)
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    # Matplotlib's default cell heights ignore embedded newlines. Size each
    # row by its actual line count, in physical units, so evidence never overlaps.
    axis_height = ax.get_position().height * ax.figure.get_figheight()
    for r, texts in enumerate([columns, *rows]):
        lines = max(str(t).count("\n") + 1 for t in texts)
        height = (lines * .14 + .12) / axis_height
        for c in range(len(columns)):
            table[r, c].set_height(height)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if r == 0:
            cell.set_facecolor("#e9eef4")
            cell.set_text_props(weight="bold")
    return table


def _rates(ax, rows, keys, label_key):
    y = np.arange(len(rows))
    width = .75 / len(keys)
    for j, key in enumerate(keys):
        positions = y + (j - (len(keys)-1)/2) * width
        for i, row in enumerate(rows):
            metric = row.get(key, {})
            available = row.get("metrics_available", True) if key in ("hazard_recall", "precision", "iou", "concept_recall") else True
            value = metric.get("rate") if available else None
            if value is None:
                ax.text(.01, positions[i], f"{key}: unavailable", va="center", fontsize=7)
            else:
                ax.barh(positions[i], value, height=width*.85, color=f"C{j}", label=key if i == 0 else None)
                ax.text(min(value+.015, .97), positions[i],
                        f"{metric.get('numerator')}/{metric.get('denominator')}", fontsize=7, va="center")
    ax.set_yticks(y, [row[label_key] for row in rows], fontsize=8)
    ax.set_xlim(0, 1.2)
    ax.set_xticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    handles = [Patch(color=f"C{j}", label=key.replace("_", " ")) for j, key in enumerate(keys)]
    ax.legend(handles=handles, fontsize=8, loc="upper center", bbox_to_anchor=(.5, 1.22), ncol=len(keys))


def render_discovery(saved, *, source="all", split="test", model=None):
    """Exact saved rows, never rate averages or sums of source/all rows."""
    rows = saved.rows("concepts", source=source, split=split, model_key=model)
    context = f"source={source}, split={split} ({'development only' if split == 'development' else 'test'}), model={model or 'all models'}"
    if not rows:
        return _empty(saved, "VLM discovery", "No saved discovery rows for this selection. " + " ".join(saved.messages), context)
    concepts = saved.rows("concepts", "per_concept", source=source, split=split, model_key=model)
    fig, axes = plt.subplots(2, 1, figsize=(13, max(6, 3 + len(concepts)*.23)),
                             gridspec_kw={"height_ratios": [max(1, len(rows)*.5), max(2, len(concepts)*.23)]})
    fig.subplots_adjust(top=.79, bottom=.14, left=.24, right=.98, hspace=.65)
    table_rows = [[r["model_key"], f"{r.get('prediction_records', '?')}/{r.get('requested_frames', '?')} records\n"
                   f"{r.get('reference_complete', '?')} references", *[rate_text(r.get("metrics", {}).get(k), r.get("metrics_available", True)) for k in METRICS],
                   f"{r.get('status', 'unknown')}\nprovisional={r.get('counts_are_provisional', 'unknown')}" ] for r in rows]
    _table(axes[0], ["Model", "Coverage", "Concept recall", "Supported precision", "Tiny recall", "Response failures", "Status"], table_rows)
    if concepts:
        plot_rows = [{**r, "label": r["concept"] + " / " + r["model_key"]} for r in concepts]
        _rates(axes[1], plot_rows, ["concept_recall"], "label")
    else:
        axes[1].axis("off")
        axes[1].text(.05, .5, "Per-concept rows unavailable for this source. Select an individual source.")
    return _stamp(saved, fig, "VLM discovery — annotation-supported rates", context + f" | n={sum(r.get('requested_frames', 0) for r in rows)} model-frame requests", rows + concepts)


def render_segmentation(saved, *, source="rellis", split="test", condition=None):
    rows = saved.rows("segmentation", source=source, split=split, condition_key=condition)
    context = f"source={source}, split={split}, condition={condition or 'all conditions'}"
    if not rows:
        return _empty(saved, "SAM handoff", "No saved SAM metrics for this selection. Supply an evaluated fixed-SAM stage; discovery works independently.", context)
    plot_condition = condition or rows[0]["condition_key"]
    concepts = saved.rows("segmentation", "per_concept", source=source, split=split, condition_key=plot_condition)
    fig, axes = plt.subplots(2, 1, figsize=(14, max(7, 4+len(concepts)*.5)),
                             gridspec_kw={"height_ratios": [max(2, len(rows)*.5), max(2, len(concepts)*.5)]})
    fig.subplots_adjust(top=.8, bottom=.13, left=.28, right=.98, hspace=.55)
    keys = ["hazard_recall", "precision", "iou", "frame_failure_rate", "query_failure_rate"]
    table_rows = [[r["condition_key"] + "\n" + {"fixed_policy": "no-VLM baseline", "reference_present": "diagnostic control"}.get(r["condition_key"], "combined VLM+SAM"),
                   f"{r.get('saved_frames')}/{r.get('expected_frames')} saved\n{r.get('reference_frames')} references",
                   *[rate_text(r.get(k), r.get("metrics_available", True) if k in keys[:3] else True) for k in keys],
                   f"{r.get('status')}\nmetrics={r.get('metrics_available')}" ] for r in rows]
    _table(axes[0], ["Condition / interpretation", "Selected frames", "Hazard recall", "Precision", "IoU", "Frame failures", "Query failures", "Status"], table_rows,
           [.2, .12, .12, .12, .1, .12, .12, .1])
    if concepts:
        _rates(axes[1], [{**r, "label": r["concept"] + " / " + r["condition_key"]} for r in concepts],
               ["prompt_coverage", "successful_query_coverage", "hazard_recall"], "label")
    else:
        axes[1].axis("off")
    return _stamp(saved, fig, "SAM handoff — prompts, query success and pixel coverage",
                  context + f" | coverage plot={plot_condition} | n={sum(r.get('expected_frames', 0) for r in rows)} condition-frame slots", rows+concepts)


def image_diagnostics(saved, frame_id, model, condition=None):
    audit = saved.phrases.get((model, frame_id))
    prediction = saved.predictions.get((model, frame_id))
    if audit is None:
        outcome = "Missing/invalid saved phrase audit; no successful-empty claim."
    elif audit.get("prediction_status") != "ok":
        outcome = "Response failure: " + str(audit.get("prediction_status"))
    elif not audit.get("saved_prompts"):
        outcome = "Successful empty hazard output."
    else:
        outcome = "Successful saved response."
    return {"outcome": outcome, "phrase_audit": audit, "prediction": prediction,
            "missed_annotated_concepts": saved.rows("concepts", "missed_concepts", model_key=model, frame_id=frame_id),
            "response_failures": saved.rows("concepts", "failures", model_key=model, frame_id=frame_id),
            "segmentation_audit": saved.sam_audits.get((condition, frame_id)),
            "validation_errors": [e for e in (saved.report or {}).get("validation", {}).get("errors", [])
                                  if e.get("frame_id") == frame_id and e.get("model_key", model) == model
                                  and e.get("condition_key", condition) == condition]}


def _overlay(ax, mask, color, opacity):
    from matplotlib.colors import to_rgba
    rgba = np.zeros((*mask.shape, 4))
    rgba[mask] = to_rgba(color, opacity)
    ax.imshow(rgba, interpolation="nearest", origin="upper")


def render_image(saved, frame_id=None, *, model=None, condition=None, query_id=None,
                 concept=None, opacity=.45, diagnostic=None):
    """RGB, source-reference and reconstructed scored union in the same coordinates.

    diagnostic may explicitly be all_query or failed_partial. A selected query is
    always diagnostic, including successful queries outside the source vocabulary.
    """
    if not 0 <= opacity <= 1:
        raise ValueError("opacity must be between zero and one")
    if frame_id is None or frame_id not in saved.frames:
        return _empty(saved, "Image explorer", "No selected saved frame. Supply frames.jsonl and RGB assets under the configured data root.")
    frame = saved.frames[frame_id]
    context = f"{frame['source']}/{frame['split']} | {frame_id} | model={model} | condition={condition} | query={query_id} | n=1 frame"
    try:
        rgb = saved.rgb(frame_id)
    except (ArtifactError, KeyError) as error:
        return _empty(saved, "Image explorer", str(error), context)
    fig = plt.figure(figsize=(14, 9))
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    text_ax = fig.add_subplot(grid[1, :])
    for ax, title in zip(axes, ["RGB", "Annotation reference", "Scored prediction"]):
        ax.imshow(rgb, interpolation="nearest", origin="upper")
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-.5, frame["width"]-.5)
        ax.set_ylim(frame["height"]-.5, -.5)
        ax.set_xlabel("image x / pixels", fontsize=8)
        ax.set_ylabel("image y / pixels", fontsize=8)
    notices = []
    handles = []
    valid = None
    try:
        valid, truths, truth = saved.reference_masks(frame_id)
        if concept:
            if concept not in saved.references[frame_id]["scored_concepts"]:
                raise ArtifactError("Selected concept is outside source vocabulary")
            truth = truths.get(concept, np.zeros_like(valid))
        _overlay(axes[1], truth, COLORS["reference"], opacity)
        handles.append(Patch(color=COLORS["reference"], label="annotation reference (valid pixels)"))
        for ax in axes[1:]:
            _overlay(ax, ~valid, COLORS["ignored"], opacity)
        handles.append(Patch(color=COLORS["ignored"], label="ignored / not evaluated"))
    except (ArtifactError, KeyError) as error:
        notices.append(str(error))
        axes[1].set_title("Annotation reference unavailable")
    try:
        overlay = reconstruct_overlay(saved, frame_id, condition)
        mask = overlay["scored"] if not concept else overlay["prediction_concepts"].get(concept, np.zeros_like(overlay["valid"]))
        _overlay(axes[2], mask, COLORS["scored"], opacity)
        handles.append(Patch(color=COLORS["scored"], label="successful source-scored SAM (valid pixels)"))
        notices.append(f"Reconstructed counts match saved audit: {overlay['counts']}")
    except (ArtifactError, KeyError) as error:
        axes[2].set_title("Scored prediction unavailable")
        notices.append(str(error))
    audit = saved.sam_audits.get((condition, frame_id))
    try:
        diagnostic_mask = None
        size = (frame["width"], frame["height"])
        color = None
        label = None
        if diagnostic is not None and diagnostic not in ("all_query", "failed_partial"):
            raise ValueError("Unknown diagnostic overlay")
        if diagnostic and audit:
            diagnostic_mask = np.zeros((size[1], size[0]), dtype=bool)
            if diagnostic == "all_query":
                diagnostic_mask |= binary_mask(saved.run_dir, audit["full_union_path"], size)
            else:
                for query in audit["queries"]:
                    if query.get("status") == "error" and query.get("union_mask_path"):
                        diagnostic_mask |= binary_mask(saved.run_dir, query["union_mask_path"], size)
            color, label = COLORS[diagnostic], diagnostic.replace("_", " ") + " — diagnostic, unscored"
        if query_id:
            if not audit:
                raise ArtifactError("No saved queries for this condition/frame")
            query = next((q for q in audit["queries"] if q["query_id"] == query_id), None)
            if query is None:
                raise ArtifactError("Query does not join selected condition/frame")
            diagnostic_mask = binary_mask(saved.run_dir, query.get("union_mask_path"), size)
            color, label = COLORS["query"], f"selected query: {query['phrase']} ({query.get('status')}) — diagnostic"
        if diagnostic_mask is not None:
            # A separate diagnostic replaces the scored image, retaining its title.
            axes[2].clear()
            axes[2].imshow(rgb, interpolation="nearest", origin="upper")
            _overlay(axes[2], diagnostic_mask, color, opacity)
            if valid is not None:
                _overlay(axes[2], ~valid, COLORS["ignored"], opacity)
            axes[2].set_title(label, fontsize=9)
            axes[2].set_xlim(-.5, frame["width"]-.5)
            axes[2].set_ylim(frame["height"]-.5, -.5)
            axes[2].set_xlabel("image x / pixels", fontsize=8)
            axes[2].set_ylabel("image y / pixels", fontsize=8)
            handles = [h for h in handles if h.get_label() != "successful source-scored SAM (valid pixels)"]
            handles.append(Patch(color=color, label=label))
    except (ArtifactError, KeyError) as error:
        notices.append("Diagnostic overlay unavailable: " + str(error))
    diagnostics = image_diagnostics(saved, frame_id, model, condition)
    phrase_audit = diagnostics["phrase_audit"] or {}
    lines = [diagnostics["outcome"], "Exact saved original phrases (audit):"]
    for phrase in phrase_audit.get("phrases", []):
        category = phrase.get("category")
        scored = "source-scored" if category == "source_concept" else "unscored"
        lines.append(f"  {phrase.get('raw_phrase')!r} → {phrase.get('canonical_concept')!r} | {scored} / {category}"
                     f" | {phrase.get('unscored_reason') or ''}")
    if not phrase_audit.get("phrases"):
        lines.append("  " + repr(phrase_audit.get("saved_prompts", "unavailable")))
    lines.append("Raw response: " + repr(phrase_audit.get("raw_response", "unavailable")))
    lines.append("Missed annotated concepts: " + ", ".join(r["concept"] for r in diagnostics["missed_annotated_concepts"]))
    lines.append("Response errors: " + repr([r.get("error_code", r.get("kind")) for r in diagnostics["response_failures"]]))
    if audit:
        lines.append(f"SAM upstream={audit.get('upstream_status')}; frame_failed={audit.get('failed')}; "
                     f"required={audit.get('required_queries')}; failed={audit.get('failed_queries')}; "
                     f"invalid={audit.get('invalid_queries')}; missing={audit.get('missing_queries')}")
    if diagnostics["validation_errors"]:
        lines.append("Validation: " + repr(diagnostics["validation_errors"]))
    lines.extend(notices)
    text_ax.axis("off")
    # Full diagnostics are available in widgets and export sidecar, without clipping
    # long raw responses/phrase lists into an unreadable figure.
    wrapped = [line for original in lines for line in textwrap.wrap(original, 155, replace_whitespace=False)]
    preview = wrapped[:17]
    if len(wrapped) > 17:
        preview.append("… Full original phrases/response and diagnostics: notebook details / export metadata.")
    text_ax.text(0, .95, "\n".join(preview), fontsize=9, va="top", transform=text_ax.transAxes)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .12), ncol=2, fontsize=8)
    fig.subplots_adjust(top=.79, bottom=.22, left=.06, right=.98, hspace=.2)
    _stamp(saved, fig, "Saved image evidence" + (f" — concept={concept}" if concept else ""), context, [audit] if audit else [])
    fig._hazard_export["diagnostics"] = diagnostics
    return fig


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        return [(key, text) for k, v in value.items() for key, text in _flatten(v, f"{prefix}.{k}" if prefix else k)]
    return [(prefix, _plain(value))]


def render_deployment(saved, profile_ids=None):
    """Explicit (component, condition, profile) identities; never selects newest."""
    if not profile_ids:
        return _empty(saved, "Deployment profiles", "Select explicit component / condition / profile IDs. Isolated VLM, isolated SAM and actual combined profiles remain separate; no stage p95 or memory peaks are added.")
    keys = [tuple(k) for k in profile_ids]
    if len(keys) != len(set(keys)):
        raise ArtifactError("Duplicate selected profile identity")
    try:
        rows = [saved.profiles[k] for k in keys]
    except KeyError as error:
        raise ArtifactError(f"Unknown explicit profile identity: {error}") from error
    memory_rows = []
    for key, row in zip(keys, rows):
        fields = {"memory": row.get("memory"), "memory_attribution": row.get("memory_attribution"),
                  "baseline_scope": row.get("baseline_scope"),
                  "budget": row.get("budget"), "target_budget_bytes": row.get("target_budget_bytes"),
                  "latency_measured": row.get("latency_measured"), "limitations": row.get("limitations"),
                  "error": row.get("error")}
        memory_rows += [[key[2], field.replace("device_used", "shared_device_used"),
                         "\n".join(textwrap.wrap(text, 75))] for field, text in _flatten(fields)]
    height = max(7, 3.5 + len(memory_rows)*.24)
    fig, axes = plt.subplots(2, 1, figsize=(15, height), gridspec_kw={"height_ratios": [2, max(2, len(memory_rows)*.24)]})
    fig.subplots_adjust(top=.88 if height < 12 else .95, bottom=.14 if height < 12 else .05, left=.02, right=.98, hspace=.1)
    table_rows = []
    for key, row in zip(keys, rows):
        latency = row.get("latency") or {}
        values = [_plain(latency.get(k)) + " s" for k in ("median_s", "p95_s")]
        table_rows.append([" / ".join(key), f"valid={row.get('valid')}\ncomplete={row.get('complete')}\nready={row.get('comparison_ready')}",
                           *values, f"{row.get('valid_measured_calls', 'unavailable')} timed\n{row.get('measured_samples', row.get('measured_frames'))}/{row.get('expected_measured_frames')} slots\n{row.get('unique_test_frames')} test frames",
                           f"{row.get('failed_calls')} failures\n{_plain(row.get('sam_query_count'))} SAM calls\n{row.get('unavailable_sam_query_counts', 'unavailable')} unknown query counts"])
    _table(axes[0], ["Explicit component / condition / profile", "Validity", "Median", "p95", "Measured samples", "Failures / queries"], table_rows,
           [.28, .14, .12, .12, .18, .16])
    _table(axes[1], ["Profile", "Resource / provenance field (bytes unless stated)", "Saved value"], memory_rows, [.23, .37, .4])
    return _stamp(saved, fig, "Deployment — allocator allocated / reserved / sampled shared-device usage (separate counters)",
                  "source=selected protocol, split=test measured / development warmup | " + "; ".join(" / ".join(k) for k in keys), rows)


def export_figure(saved, figure, filename, *, enabled=False):
    """Explicit opt-in PNG/SVG plus evidence sidecar; never overwrite any file."""
    if not enabled or not saved.config.get("export_enabled", False):
        raise ArtifactError("Export disabled; set config export_enabled and enabled=True explicitly")
    if not isinstance(filename, str) or "/" in filename or Path(filename).suffix.lower() not in (".png", ".svg"):
        raise ArtifactError("Export filename must be a safe PNG/SVG basename")
    # Re-resolve configuration to protect against a replaced directory/symlink.
    from .loading import check_output_directory
    destination = check_output_directory(saved.output_dir, saved.data_root, saved.run_dir.parent, saved.component_root)
    if destination != saved.output_dir:
        raise ArtifactError("Visualization output directory changed since loading")
    target = safe_path(destination, filename)
    sidecar = safe_path(destination, filename + ".json")
    if target.exists() or sidecar.exists():
        raise ArtifactError("Refusing to overwrite existing visualization exports")
    if not hasattr(figure, "_hazard_export"):
        raise ArtifactError("Export only accepts figures rendered by this viewer")
    destination.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        figure.savefig(stream, format=target.suffix[1:].lower(), dpi=150)
    with sidecar.open("x", encoding="utf-8") as stream:
        json.dump(figure._hazard_export, stream, indent=2, ensure_ascii=True, allow_nan=False)
    return target
