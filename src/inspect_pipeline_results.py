"""Read saved mapping results on CPU. No producer, evaluator or model imports."""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
import html
from io import BytesIO
import json
from pathlib import Path
import zipfile

import numpy as np

PIPELINES = {"geometry_only", "ground_surface", "fixed_hazards", "qwen_hazards"}
MAX_ARRAY_BYTES = 256 * 1024 * 1024


def default_outputs_root(project_root=None):
    server = Path("/home/jovyan/EmbodiedAI-pipelines/outputs")
    project = Path(project_root or Path(__file__).resolve().parents[1])
    return server if server.is_dir() else project / "outputs" / "pipeline_fixture_v1"


def _json(path, default=None):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).is_file() else default


def _rows(path):
    if not Path(path).is_file():
        return []
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # An in-progress final line is not a completed record.
    return rows


def discover_runs(root):
    root = Path(root).expanduser().resolve()
    runs = []
    for path in root.rglob("run.json") if root.is_dir() else []:
        try:
            record = _json(path)
            pipeline = record.get("pipeline_id", record.get("pipeline"))
            if record.get("contract_id") != "semantic_mapping_v1" or pipeline not in PIPELINES:
                continue
            runs.append({"path": path.parent, "pipeline": pipeline,
                         "fixture": record.get("fixture"), "status": record.get("status", "unknown"),
                         "modified": path.stat().st_mtime})
        except (OSError, ValueError, TypeError):
            continue
    return sorted(runs, key=lambda row: row["modified"], reverse=True)


def choose_run(root):
    runs = discover_runs(root)
    for predicate in (lambda r: r["fixture"] is False and r["pipeline"] == "geometry_only",
                      lambda r: r["fixture"] is False,
                      lambda r: r["fixture"] is True and r["pipeline"] == "geometry_only",
                      lambda r: r["fixture"] is True):
        for row in runs:
            if predicate(row):
                return row["path"]
    raise FileNotFoundError(f"No common pipeline run found under {Path(root).resolve()}")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _arrays(path, names):
    if not Path(path).is_file():
        return {}
    try:
        with zipfile.ZipFile(path) as archive:
            size = sum(info.file_size for info in archive.infolist()
                       if info.filename in {name + ".npy" for name in names})
        if size > MAX_ARRAY_BYTES:
            raise ValueError("Selected arrays exceed the viewer's 256 MiB read limit")
        with np.load(path, allow_pickle=False) as saved:
            return {name: saved[name] for name in names if name in saved.files}
    except (zipfile.BadZipFile, EOFError) as error:
        raise ValueError(f"Incomplete or invalid saved array archive: {path}") from error


def _cache(run_dir, manifest, override, warnings):
    candidates = []
    if override:
        candidates.append(Path(override).expanduser().resolve())
    elif manifest.get("cache_path"):
        declared = Path(manifest["cache_path"])
        candidates.append(declared if declared.is_absolute() else run_dir / "geometry" / declared)
    if not override:
        candidates.append(run_dir / "geometry" / "cache")
        # Portable fixture layout; accept only its exact declared archive hash.
        candidates.append(run_dir.parent.parent / "geometry_cache")
    expected = manifest.get("archive_sha256")
    for directory in candidates:
        archive = directory / "geometry.npz"
        if not archive.is_file():
            continue
        if not expected or _sha256(archive) != expected:
            warnings.append(f"Ignored unverified geometry cache: {archive}")
            continue
        return directory.resolve()
    warnings.append("Verified source geometry cache unavailable; using saved voxel centers where possible.")
    return None


def load_run(run_dir, *, cache_dir=None, report_path=None):
    run_dir = Path(run_dir).expanduser().resolve()
    metadata = _json(run_dir / "run.json", {})
    if metadata.get("contract_id") != "semantic_mapping_v1" or metadata.get("pipeline_id") not in PIPELINES:
        raise ValueError("Select a common pipeline run directory containing run.json")
    warnings = []
    geometry = _json(run_dir / "geometry/manifest.json", {})
    result = {"path": run_dir, "run": metadata, "summary": _json(run_dir / "summary.json", {}),
              "geometry": geometry, "map": _json(run_dir / "map/manifest.json", {}),
              "planning": _json(run_dir / "planning/manifest.json", {}),
              "frames": _rows(run_dir / "frames.jsonl"), "semantics": _rows(run_dir / "semantics/frames.jsonl"),
              "concepts": _json(run_dir / "map/concepts.json", []), "warnings": warnings,
              "report": None}
    result["cache"] = _cache(run_dir, geometry, cache_dir, warnings)
    candidates = [Path(report_path)] if report_path else [
        run_dir / "evaluation/report.json",
        run_dir.parent.parent / "evaluations" / metadata["pipeline_id"] / "report.json"]
    identity = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":"),
                                        allow_nan=False).encode()).hexdigest()
    for candidate in candidates:
        report = _json(candidate)
        if report and report.get("run_identity") == identity:
            result["report"] = report
            result["report_path"] = candidate.resolve()
            break
        if report:
            warnings.append(f"Ignored report for a different run identity: {candidate}")
    return result


def summary_text(result):
    run, summary, planning = result["run"], result["summary"], result["planning"]
    lines = [f"Run: {result['path']}", f"Pipeline: {run['pipeline_id']} | Status: {run.get('status', 'unknown')}",
             f"Mode: {run.get('mode', 'unknown')}"]
    if run.get("fixture") is True:
        lines.append("FIXTURE: synthetic software checks. These are not real model accuracy, speed or memory results.")
    elif run.get("fixture") is not False:
        lines.append("Fixture provenance unavailable; do not treat these results as real measurements.")
    if run.get("status") != "complete":
        lines.append("This run is not complete. Saved counts and plots may be partial.")
    required = summary.get("required_frame_attempts", run.get("sequence", {}).get("frame_count", "unknown"))
    lines.append(f"Frames: {len(result['frames'])} saved dispositions / {required} required; "
                 f"statuses: {dict(Counter(row.get('status', 'unknown') for row in result['frames']))}")
    lines.append(f"Semantic frame statuses: {dict(Counter(row.get('status', 'unknown') for row in result['semantics']))}")
    lines.append(f"Saved counts: {summary.get('counts', {})}")
    lines.append(f"Planning: {planning.get('availability', run.get('planning_status', 'unavailable'))}; "
                 f"reason: {planning.get('reason') or run.get('reason') or 'none recorded'}")
    lines.append("Planning output is an internal diagnostic; it does not certify robot safety or successful navigation.")
    geometry = result["geometry"]
    lines.append(f"Map frame: {geometry.get('map_frame', 'unknown')}; units: {geometry.get('units', 'unknown')}; "
                 f"declared up: {geometry.get('up')}")
    lines.append("Geometry uses saved map coordinates. No XY ground plane or gravity direction is inferred; scale is not applied again.")
    report = result["report"]
    if report is None or report.get("reference") is None:
        lines.append("No matching independent-reference evaluation found: accuracy is unavailable; plots are inspection only.")
    else:
        lines.append(f"Saved evaluation: {result['report_path']}")
        lines.append(f"Reference: {report['reference'].get('reference_id', 'declared reference')}; "
                     "only metrics explicitly marked available have supporting coverage.")
        if report.get("fixture"):
            lines.append("Reference metrics are synthetic fixture results, not real accuracy.")
    lines.extend("Note: " + note for note in result["warnings"])
    return "\n".join(lines)


def _indices(count, limit):
    return np.linspace(0, count - 1, min(count, limit), dtype=np.int64) if count else np.empty(0, np.int64)


def _cloud(result, limit):
    cache = result["cache"]
    if cache:
        try:
            arrays = _arrays(cache / "geometry.npz", ["world_points", "images", "depth", "world_points_conf"])
            points, rgb = arrays["world_points"], arrays["images"]
            if points.shape != rgb.shape or points.ndim != 4 or points.shape[-1] != 3:
                raise ValueError("Source points/RGB do not share a pixel grid")
            ids = _indices(points.size // 3, limit * 3)
            selected = points.reshape(-1, 3)[ids]
            colors = rgb.reshape(-1, 3)[ids].astype(float) / 255
            valid = np.isfinite(selected).all(1)
            if "depth" in arrays:
                depth = arrays["depth"].reshape(-1)[ids]
                valid &= np.isfinite(depth) & (depth > 0)
            if "world_points_conf" in arrays:
                confidence = arrays["world_points_conf"].reshape(-1)[ids]
                minimum = result["geometry"].get("settings", {}).get("min_confidence", 0)
                valid &= np.isfinite(confidence) & (confidence > 0) & (confidence >= minimum)
            _, height, width, _ = points.shape
            frame_ids, pixels = np.divmod(ids, height * width)
            ys, xs = np.divmod(pixels, width)
            for index, transform in enumerate(result["geometry"].get("transforms", [])):
                left, top, right, bottom = transform.get("pad_ltrb", [0, 0, 0, 0])
                affected = frame_ids == index
                valid &= ~affected | ((xs >= left) & (xs < width - right) & (ys >= top) & (ys < height - bottom))
            selected, colors = selected[valid], colors[valid]
            kept = _indices(len(selected), limit)
            return selected[kept], colors[kept], "source-colored world points (sampled)"
        except (OSError, ValueError, KeyError) as error:
            result["warnings"].append(f"Source-colored geometry unavailable: {error}")
    arrays = _arrays(result["path"] / "map/voxels.npz", ["centers"])
    points = arrays.get("centers", np.empty((0, 3)))
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Voxel centers must have shape [N,3]")
    points = points[_indices(len(points), limit)]
    points = points[np.isfinite(points).all(1)]
    return points, "0.6", "voxel centers (sampled; neutral color)"


def _axes(ax, points, units):
    unit = "m" if units == "metres" else "unscaled/declared units"
    for name in ("x", "y", "z"):
        getattr(ax, f"set_{name}label")(f"Map {name.upper()} ({unit})")
    if len(points):
        extent = np.ptp(points, axis=0)
        ax.set_box_aspect(np.maximum(extent, max(float(extent.max()) * .1, 1e-5)))


def create_figures(result, *, max_points=20000, max_concepts=16):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from PIL import Image

    if not 1 <= max_points <= 20000:
        raise ValueError("max_points must be between 1 and 20000")
    figures = []
    cache = result["cache"]
    images = sorted((cache / "processed_frames").glob("*.png")) if cache else []
    if images:
        picked = [images[i] for i in _indices(len(images), 6)]
        fig, axes = plt.subplots(1, len(picked), figsize=(max(5, 2.4 * len(picked)), 3), squeeze=False)
        for ax, path in zip(axes.ravel(), picked):
            try:
                with Image.open(path) as image:
                    preview = image.copy(); preview.thumbnail((600, 600))
                    ax.imshow(preview)
            except (OSError, ValueError) as error:
                ax.text(.1, .5, "Saved preview unavailable", transform=ax.transAxes)
                result["warnings"].append(f"Processed preview unavailable: {path.name}: {error}")
            ax.set_title(path.stem); ax.axis("off")
        fig.suptitle("Saved processed RGB frames"); fig.tight_layout()
        figures.append(("Processed frames", fig))
    try:
        points, colors, label = _cloud(result, max_points)
        fig = plt.figure(figsize=(9, 7)); ax = fig.add_subplot(111, projection="3d")
        if len(points):
            ax.scatter(*points.T, c=colors, s=2, depthshade=False)
        axis_points = points
        trajectory = _arrays(result["path"] / "geometry/camera_trajectory.npz", ["world_to_camera"])
        poses = trajectory.get("world_to_camera")
        if poses is not None and poses.ndim == 3 and poses.shape[1:] in ((3, 4), (4, 4)):
            centers = -np.einsum("nji,nj->ni", poses[:, :3, :3], poses[:, :3, 3])
            centers = centers[np.isfinite(centers).all(1)]
            if len(centers):
                ax.plot(*centers.T, color="crimson", marker="o", markersize=3, label="camera centers")
                ax.legend()
                axis_points = np.concatenate([points, centers])
        _axes(ax, axis_points, result["geometry"].get("units")); ax.set_title(label + "\nSaved map axes; no ground orientation inferred")
        figures.append(("Geometry and cameras", fig))
    except (OSError, ValueError, KeyError) as error:
        result["warnings"].append(f"Geometry plot unavailable: {error}")
    try:
        voxels = _arrays(result["path"] / "map/voxels.npz", ["centers"])
        evidence = _arrays(result["path"] / "map/semantic_evidence.npz", ["voxel_row", "concept_row", "evidence_score"])
        concepts = result["concepts"]
        concepts = concepts if isinstance(concepts, list) else concepts.get("concepts", concepts.get("registry", []))
        centers = voxels.get("centers", np.empty((0, 3)))
        vr, cr, scores = (evidence.get(key, np.empty(0)) for key in ("voxel_row", "concept_row", "evidence_score"))
        if not (len(vr) == len(cr) == len(scores)) or (len(vr) and (vr.dtype.kind not in "iu" or cr.dtype.kind not in "iu")):
            raise ValueError("Invalid sparse semantic evidence indices")
        if len(vr) and ((vr < 0).any() or (vr >= len(centers)).any() or (cr < 0).any() or (cr >= len(concepts)).any()):
            raise ValueError("Semantic rows reference unavailable voxel/concept identities")
        for index, concept in enumerate(concepts[:max_concepts]):
            concept = {"concept_id": concept} if isinstance(concept, str) else concept
            selected = (cr == index) & np.isfinite(scores) & (scores > 0)
            ids = np.flatnonzero(selected); ids = ids[_indices(len(ids), max_points)]
            positive = centers[vr[ids]] if len(ids) else np.empty((0, 3))
            fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection="3d")
            if len(positive):
                finite = np.isfinite(positive).all(1)
                points = positive[finite]
                plot = ax.scatter(*points.T, c=scores[ids][finite], cmap="viridis", s=5)
                fig.colorbar(plot, ax=ax, shrink=.6, label="Weighted evidence statistic; not safety probability")
            else:
                points = positive
                ax.text2D(.1, .5, "No positive stored evidence\nAbsence does not certify safety", transform=ax.transAxes)
            _axes(ax, points, result["map"].get("units")); ax.set_title(f"{concept['concept_id']} ({concept.get('role', 'declared concept')})")
            figures.append(("Semantic evidence: " + concept["concept_id"], fig))
        if len(concepts) > max_concepts:
            result["warnings"].append(f"Showing {max_concepts}/{len(concepts)} concepts; raise max_concepts to inspect the others.")
    except (OSError, ValueError, KeyError, IndexError) as error:
        result["warnings"].append(f"Semantic plots unavailable: {error}")
    try:
        arrays = _arrays(result["path"] / "planning/costmap.npz", ["decision_state"])
        state = arrays.get("decision_state")
        if state is not None and state.ndim == 2 and state.size:
            if not np.isin(state, [0, 1, 2]).all():
                raise ValueError("Unknown decision-state codes")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.imshow(state, origin="lower", cmap=ListedColormap(["lightgray", "tomato", "seagreen"]), norm=BoundaryNorm([-.5, .5, 1.5, 2.5], 3))
            ax.set_title("Robot-policy grid: gray unknown, red blocked, green traversable\nInternal diagnostic; safety/navigation unverified")
            ax.set_xlabel("Grid column"); ax.set_ylabel("Grid row")
            figures.append(("Planning diagnostic", fig))
    except (OSError, ValueError, KeyError) as error:
        result["warnings"].append(f"Planning plot unavailable: {error}")
    return figures


def export_html(result, output, *, max_points=20000):
    import matplotlib.pyplot as plt
    output = Path(output).expanduser().resolve()
    if output == result["path"] or output.is_relative_to(result["path"]):
        raise ValueError("Viewer export must be outside the saved run directory")
    figures = create_figures(result, max_points=max_points)
    parts = ["<!doctype html><meta charset='utf-8'><title>Saved pipeline results</title>",
             "<style>body{font:16px system-ui;max-width:1000px;margin:32px auto;padding:0 16px}pre{white-space:pre-wrap}img{max-width:100%}</style>",
             "<h1>Saved pipeline results</h1><pre>" + html.escape(summary_text(result)) + "</pre>"]
    for title, fig in figures:
        buffer = BytesIO(); fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
        parts.append("<h2>" + html.escape(title) + "</h2><img alt='" + html.escape(title, quote=True) +
                     "' src='data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode() + "'>")
        plt.close(fig)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "report.html"; path.write_text("\n".join(parts), encoding="utf-8")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_outputs_root())
    parser.add_argument("--run", type=Path)
    parser.add_argument("--cache", type=Path, help="Explicit relocated cache directory; archive identity is verified")
    parser.add_argument("--report", type=Path, help="Existing evaluator report.json; run identity must match")
    parser.add_argument("--output", type=Path, help="Optional separate directory for a standalone static report.html")
    args = parser.parse_args(argv)
    result = load_run(args.run or choose_run(args.root), cache_dir=args.cache, report_path=args.report)
    print(summary_text(result))
    if args.output:
        print(f"Static viewer: {export_html(result, args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
