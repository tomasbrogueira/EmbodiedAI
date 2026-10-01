"""Portable exports, without visualization or interpretation of unscored nouns."""

import csv
import io
import json
import os
from pathlib import Path
import tempfile


def _atomic(path, text):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent,
                                         prefix="." + path.name, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _flatten(row, prefix=""):
    result = {}
    for key, value in row.items():
        name = prefix + key
        if isinstance(value, dict):
            result.update(_flatten(value, name + "."))
        elif isinstance(value, (list, tuple)):
            result[name] = json.dumps(value, ensure_ascii=False, allow_nan=False)
        else:
            result[name] = value
    return result


def _csv(rows):
    flattened = [_flatten(row) for row in rows]
    names = list(dict.fromkeys(key for row in flattened for key in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=names)
    writer.writeheader()
    writer.writerows(flattened)
    return stream.getvalue()


def export_report(report, output_dir):
    """Write evaluator-owned names only and return their absolute paths."""
    if not isinstance(report, dict) or report.get("task_id") != "hazard_prompt_v1" or report.get("schema_version") != 1:
        raise ValueError("Expected a hazard_prompt_v1 report")
    destination = Path(output_dir).expanduser().resolve()
    # Construct all serialization before publishing any file.
    concepts, segmentation, deployment = (report.get(key, {}) for key in ("concepts", "segmentation", "deployment"))
    contents = {
        "hazard_report.json": json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        "coverage.json": json.dumps({"fixture": report["fixture"], "coverage": report.get("coverage", {}), "prediction_coverage": concepts.get("coverage", []), "segmentation_coverage": segmentation.get("coverage", {})}, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        "concept_metrics.csv": _csv(concepts.get("rows", [])),
        "per_concept.csv": _csv(concepts.get("per_concept", [])),
        "missed_concepts.csv": _csv(concepts.get("missed_concepts", [])),
        "segmentation_metrics.csv": _csv(segmentation.get("rows", [])),
        "segmentation_concept_coverage.csv": _csv(segmentation.get("per_concept", [])),
        "deployment.csv": _csv(deployment.get("rows", [])),
    }
    for name, rows in (("phrase_audit.jsonl", concepts.get("phrase_audit", [])),
                       ("errors.jsonl", report.get("validation", {}).get("errors", []) + concepts.get("failures", [])),
                       ("segmentation_frames.jsonl", segmentation.get("frame_audit", []))):
        contents[name] = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)
    # Failed model text may contain lone Unicode surrogates preserved by the
    # inference audit. Escape those code points before any UTF-8 publication.
    contents = {name: content.encode("utf-8", errors="backslashreplace").decode("utf-8")
                for name, content in contents.items()}
    # Refuse an unsafe named target before replacing any previously saved report.
    for name in contents:
        path = destination / name
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(f"Unsafe export target: {path}")
    destination.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, content in contents.items():
        path = destination / name
        _atomic(path, content)
        paths[name] = str(path)
    return paths
