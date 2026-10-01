"""Compact exports and a small evidence-based classification review gallery."""

import base64
import csv
from html import escape
import io
import json
import os
from pathlib import Path
import tempfile


LABELS = ("traversable", "non_traversable", "unknown")
METRICS = ("unsafe_acceptance", "useful_acceptance", "predicted_unknown_rate",
           "reference_unknown_acceptance")
CATEGORIES = ("recognition", "target_grounding", "policy", "segmentation", "unreviewed")


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return _json(value)
    return value


def _csv(rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _cell(row.get(key)) for key in fieldnames})
    return stream.getvalue()


def _flat_comparison(row):
    result = {key: value for key, value in row.items() if key not in {"metrics", "confusion_matrix"}}
    for name in METRICS:
        metric = (row.get("metrics") or {}).get(name)
        for field in ("numerator", "denominator", "rate"):
            result[f"{name}_{field}"] = metric.get(field) if metric else None
    return result


def _atomic_write(path, content):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _rate(metric):
    if metric is None:
        return "unavailable"
    n, d, rate = metric["numerator"], metric["denominator"], metric["rate"]
    value = "null" if rate is None else f"{100 * rate:.1f}%"
    return f"{value} ({n}/{d})"


def comparison_html(report):
    """Return one comparison table; every displayed rate includes its counts."""
    show_source = any(row.get("source", "all") != "all" for row in report.get("comparison", []))
    headings = ["Configuration / model / run", "Split", "Status", "Eligible", "Unsafe acceptance",
                "Useful acceptance", "Predicted unknown", "Reference-unknown acceptance",
                "Failures", "Missing", "Coverage"]
    if show_source:
        headings.insert(2, "Source")
    rows = []
    for row in report.get("comparison", []):
        identity = " / ".join(str(row.get(key, "")) for key in
                              ("configuration_label", "model_key", "run_name"))
        split = "development (tuning)" if row.get("split") == "development" else "test (held-out recordings)"
        metrics = row.get("metrics") or {}
        failures = f"{row.get('prediction_failures', 0)} primary; {row.get('selected_prediction_failures', 0)} selected"
        values = [identity, split, row.get("status"), row.get("eligible_regions", 0)]
        values.extend(_rate(metrics.get(name)) for name in METRICS)
        values.extend([failures, row.get("missing_predictions", 0), row.get("coverage_status", "")])
        if show_source:
            values.insert(2, row.get("source", "all"))
        rows.append("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in values) + "</tr>")
    banner = "<p><strong>SYNTHETIC FIXTURE — excluded from experiment results.</strong></p>" if report.get("fixture") else ""
    return (banner + "<p>Development results are for tuning. RELLIS test results use held-out recordings on one campus; "
            "regions within a frame are not independent scenes. Public coverage is provisional for phone hazards.</p>"
            "<div style='overflow-x:auto'><table style='border-collapse:collapse' border='1' cellpadding='5'>"
            "<thead><tr>" + "".join(f"<th>{heading}</th>" for heading in headings) + "</tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></div>")


def _resolve_asset(data_root, value):
    if not value:
        raise ValueError("asset path absent")
    root = Path(data_root).resolve()
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("asset outside data root")
    return path


def _thumbnail(case, data_root):
    from PIL import Image, ImageChops, ImageFilter

    try:
        with Image.open(_resolve_asset(data_root, case.get("image_path"))) as opened:
            image = opened.convert("RGB")
        original_size = image.size
        image.thumbnail((360, 220), Image.Resampling.LANCZOS)
        message = ""
        try:
            with Image.open(_resolve_asset(data_root, case.get("mask_path"))) as opened:
                if opened.size != original_size or len(opened.getbands()) != 1:
                    raise ValueError("mask not aligned or not single-channel")
                # All nonzero values are foreground, including 16-bit masks.
                pixels = opened.get_flattened_data() if hasattr(opened, "get_flattened_data") else opened.getdata()
                mask = Image.new("L", opened.size)
                mask.putdata([255 if pixel else 0 for pixel in pixels])
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
            overlay = Image.blend(image, Image.new("RGB", image.size, (255, 200, 20)), 0.25)
            image = Image.composite(overlay, image, mask)
            edge = ImageChops.subtract(mask, mask.filter(ImageFilter.MinFilter(3)))
            image.paste((255, 220, 20), mask=edge)
        except (OSError, ValueError, TypeError) as exc:
            message = f"Mask preview unavailable: {exc}"
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"<img alt='RGB with selected mask in yellow' src='data:image/png;base64,{encoded}'>" + (
            f"<p>{escape(message)}</p>" if message else "")
    except (OSError, ValueError, TypeError) as exc:
        return f"<p>Image preview unavailable: {escape(str(exc))}</p>"


def _case_priority(case):
    unsafe = case.get("predicted_label") == "traversable" and (
        case.get("reference_label") == "non_traversable" or (case.get("avoided_pixel_count") or 0) > 0)
    if unsafe:
        priority = 0
    elif case.get("mask_defect") or case.get("mask_quality") in {"mixed", "broken"}:
        priority = 1
    elif case.get("prediction_status") in {"error", "missing"}:
        priority = 2
    else:
        priority = 3
    return priority, str(case.get("run_name", "")), str(case.get("model_key", "")), str(case.get("region_id", ""))


def render_review_gallery(report, *, limit=20):
    """Return HTML for at most 20 cases, with tentative and confirmed attribution distinct."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("limit must be a nonnegative integer")
    limit = min(limit, 20)
    cases = sorted(report.get("error_cases", []), key=_case_priority)[:limit]
    groups = {category: [] for category in CATEGORIES}
    for case in cases:
        category = case.get("confirmed_category") or case.get("suggested_category") or "unreviewed"
        groups[category if category in groups else "unreviewed"].append(case)
    blocks = ["<p>Optional review of at most 20 examples. Suggestions are tentative; confirmed categories require "
              "explicit review evidence. No automatic causal judgment or additional VLM reference is used.</p>",
              f"<p>Showing {len(cases)} of {len(report.get('error_cases', []))} cases; priority: unsafe acceptances, "
              "mask defects, prediction failures.</p>"]
    if report.get("fixture"):
        blocks.append("<p><strong>SYNTHETIC FIXTURE — qualitative demonstration only.</strong></p>")
    for category, records in groups.items():
        blocks.append(f"<h3>{escape(category.replace('_', ' ').title())} ({len(records)})</h3>")
        if not records:
            blocks.append("<p>No cases in this displayed group.</p>")
            continue
        blocks.append("<div style='display:flex;flex-wrap:wrap;gap:12px'>")
        for case in records:
            confirmed = case.get("confirmed_category")
            attribution = f"Confirmed: {confirmed}" if confirmed else "Unreviewed"
            if case.get("suggested_category"):
                attribution += f"; tentative suggestion: {case['suggested_category']}"
            details = {
                "Run / model / split": f"{case.get('run_name', '')} / {case.get('model_key', '')} / {case.get('split', '')}",
                "Reference → prediction": f"{case.get('reference_label')} → {case.get('predicted_label')}",
                "Semantic classes": f"{case.get('semantic_class_reference')} → {case.get('semantic_class_prediction')}",
                "Prediction status": f"{case.get('prediction_status')} ({case.get('error_code')})",
                "Mask": f"{case.get('mask_quality')} / {case.get('mask_defect')}",
                "Annotated avoided pixels": case.get("avoided_pixel_count"),
                "Attribution": attribution,
                "Suggestion evidence": case.get("suggestion_evidence"),
                "Review evidence": case.get("review_note"),
                "Reason": case.get("reason"),
            }
            blocks.append("<article style='max-width:380px;border:1px solid #aaa;padding:8px'>"
                          f"<strong>{escape(str(case.get('region_id', '')))}</strong>"
                          + _thumbnail(case, report.get("data_root", ".")) + "<dl>"
                          + "".join(f"<dt>{escape(key)}</dt><dd>{escape(str(value))}</dd>" for key, value in details.items())
                          + "</dl><details><summary>Raw response</summary><pre style='white-space:pre-wrap'>"
                          + escape(str(case.get("raw_response") or "")) + "</pre></details></article>")
        blocks.append("</div>")
    return "".join(blocks)


def _confusion_rows(report):
    result = []
    seen = set()
    for row in report.get("comparison", []) + report.get("per_source", []):
        matrix = row.get("confusion_matrix")
        identity = {key: row.get(key) for key in
                    ("run_name", "model_key", "configuration_label", "split", "source", "status")}
        marker = tuple(identity.values())
        if marker in seen:
            continue
        seen.add(marker)
        for reference_index, reference in enumerate(LABELS):
            for prediction_index, prediction in enumerate(LABELS):
                result.append(dict(identity, reference_label=reference, predicted_label=prediction,
                                   count=matrix[reference_index][prediction_index] if matrix is not None else None))
    return result


def _confusion_plots(report, destination):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    paths = []
    for index, row in enumerate(report.get("comparison", [])):
        matrix = row.get("confusion_matrix")
        if matrix is None:
            continue
        figure, axis = plt.subplots(figsize=(5, 4))
        axis.imshow(matrix, cmap="Blues")
        axis.set(xticks=range(3), yticks=range(3), xticklabels=LABELS, yticklabels=LABELS,
                 xlabel="Prediction", ylabel="Reference",
                 title=f"{row.get('configuration_label')} · {row.get('split')}")
        axis.tick_params(axis="x", labelrotation=20)
        for y in range(3):
            for x in range(3):
                axis.text(x, y, str(matrix[y][x]), ha="center", va="center")
        figure.tight_layout()
        path = destination / f"confusion_{index:03d}.png"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination, suffix=".png", delete=False) as handle:
                temporary = Path(handle.name)
            figure.savefig(temporary, dpi=120)
            os.replace(temporary, path)
        finally:
            plt.close(figure)
            if temporary is not None and temporary.exists():
                temporary.unlink()
        paths.append(str(path))
    return paths


def export_report(report, output_dir=None, *, plots=False):
    """Atomically export compact artifacts only below the destination run/evaluation."""
    if not report.get("destination_run_dir"):
        raise ValueError("report has no destination_run_dir")
    run_dir = Path(report["destination_run_dir"]).resolve()
    allowed = (run_dir / "evaluation").resolve()
    if not allowed.is_relative_to(run_dir):
        raise ValueError("evaluation directory escapes the destination run")
    destination = Path(output_dir).resolve() if output_dir is not None else allowed
    if not destination.is_relative_to(allowed):
        raise ValueError("output_dir must stay inside the destination run's evaluation directory")
    destination.mkdir(parents=True, exist_ok=True)
    comparisons = [_flat_comparison(row) for row in report.get("comparison", [])]
    sources = [_flat_comparison(row) for row in report.get("per_source", [])]
    # Empty tables retain useful headers, so absence of results remains explicit.
    identity = ["run_name", "model_key", "configuration_label", "split", "source", "status"]
    gallery = ("<!doctype html><html lang='en'><meta charset='utf-8'><title>Classification review</title>"
               "<body><h1>Classification review</h1>" + comparison_html(report)
               + render_review_gallery(report) + "</body></html>")
    payloads = {
        "report.json": _json(report) + "\n",
        "coverage.json": _json(report.get("coverage", {})) + "\n",
        "comparison.csv": _csv(comparisons, None if comparisons else identity),
        "per_source.csv": _csv(sources, None if sources else identity),
        "confusion.csv": _csv(_confusion_rows(report), identity + ["reference_label", "predicted_label", "count"]),
        "mask_breakdown.csv": _csv(report.get("mask_breakdown", []), None if report.get("mask_breakdown") else identity),
        "hazard_overlap.csv": _csv(report.get("hazard_overlap", []), None if report.get("hazard_overlap") else identity),
        "error_cases.csv": _csv(report.get("error_cases", []), None if report.get("error_cases") else ["case_id", "region_id", "suggested_category", "confirmed_category", "review_note"]),
        "review_gallery.html": gallery,
    }
    paths = {}
    for name, content in payloads.items():
        path = destination / name
        _atomic_write(path, content)
        paths[name] = str(path)
    if plots:
        paths["plots"] = _confusion_plots(report, destination)
    return paths
