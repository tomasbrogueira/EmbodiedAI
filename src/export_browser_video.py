"""CPU-only self-contained HTML players for a completed saved video export.

Video and thumbnail bytes are inline so authenticated Jupyter subresource
requests are unnecessary. No JavaScript, models or security settings are used.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
import html
from io import BytesIO
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import statistics
import sys
import time
from urllib.parse import quote

from PIL import Image

MAX_MP4_BYTES = 32 * 1024 * 1024
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_PNG_BYTES = 16 * 1024 * 1024
PLAYERS = (("combined", "Video and voxel map", "combined_frames"),
           ("segmentation", "Video with segmentation", "segmentation_frames"),
           ("voxel_map", "Growing voxel map", "voxel_map_frames"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    if Path(path).stat().st_size > MAX_JSON_BYTES:
        raise ValueError(f"Metadata exceeds 16 MiB: {path}")
    def reject(value):
        raise ValueError(f"Non-finite JSON value: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def safe_asset(root, relative):
    text = str(relative)
    parts = PurePosixPath(text.replace("\\", "/")).parts
    if not text or PurePosixPath(text).is_absolute() or PureWindowsPath(text).is_absolute() or ".." in parts:
        raise ValueError("Unsafe export asset path")
    path = (root / text).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Export asset escapes its directory")
    return path


def asset_href(path, page_dir):
    """Navigation links stay relative to the authenticated Jupyter HTML URL."""
    relative = os.path.relpath(Path(path).resolve(), Path(page_dir).resolve()).replace("\\", "/")
    return quote(relative, safe="/")


def data_url(payload, mime):
    return f"data:{mime};base64," + base64.b64encode(payload).decode("ascii")


def thumbnail(path):
    with Image.open(path) as image:
        if image.format != "PNG" or image.width * image.height > 16_000_000:
            raise ValueError("Expected a bounded saved PNG preview")
        image = image.convert("RGB")
        image.thumbnail((960, 540), Image.Resampling.LANCZOS)
        output = BytesIO()
        image.save(output, format="JPEG", quality=84, optimize=True)
    return output.getvalue()


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def describe_export(manifest, config, sequence=None):
    rows = manifest["coverage"]
    times = [row.get("timestamp_ns") for row in rows]
    clocks = {row.get("timestamp_provenance", {}).get("clock") for row in rows}
    if sequence:
        clocks.update(row.get("timestamp_provenance", {}).get("clock") for row in sequence.get("frames", []))
    clock = "video" if "video_presentation_timeline" in clocks else "sample"
    known = all(isinstance(t, int) and not isinstance(t, bool) and t >= 0 for t in times)
    facts = [f"{len(rows)} saved frames. Sampled results are held between updates; no masks are interpolated for unsampled frames."]
    timing = {"clock": clock, "timestamp_provenance": [row.get("timestamp_provenance", {}) for row in rows]}
    if known:
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("Saved timestamps must increase")
        start, finish = times[0] / 1e9, times[-1] / 1e9
        timing.update(first_time_seconds=start, last_time_seconds=finish)
        if len(times) > 1:
            gaps = [(b - a) / 1e9 for a, b in zip(times, times[1:])]
            median = statistics.median(gaps)
            timing.update(median_spacing_seconds=median, effective_samples_per_second=1 / median,
                          minimum_spacing_seconds=min(gaps), maximum_spacing_seconds=max(gaps))
            cadence = f"about {1 / median:.3g} saved frames/s; actual median spacing {median:.6g} s"
            if max(gaps) - min(gaps) > .000001:
                cadence += f" (range {min(gaps):.6g}–{max(gaps):.6g} s)"
            facts.append(f"{clock.capitalize()} time {start:.3f}–{finish:.3f} s · {cadence}.")
        else:
            facts.append(f"One saved frame at {clock} time {start:.3f} s.")
    else:
        facts.append("Source timestamps are incomplete; this movie uses labeled frame pacing.")
    facts.append("Times are decoded video presentation times, not wall-clock capture times." if clock == "video"
                 else "Times use the saved sample clock; wall-clock capture timing is not established.")
    hold = config.get("timeline", {}).get("end_hold_seconds")
    if _number(hold):
        facts.append(f"The final result holds for {hold:g} s. Playback holds contain no new observations.")
    queries = [q for row in rows for q in row.get("queries", [])]
    prompts = sorted({q.get("prompt") for q in queries if isinstance(q.get("prompt"), str)})
    statuses = Counter(str(q.get("status", "unknown")) for q in queries)
    if queries:
        status_text = ", ".join(f"{count} {status}" for status, count in sorted(statuses.items()))
        instances = sum(q.get("instances", 0) for q in queries if isinstance(q.get("instances"), int))
        facts.append(f"Prompt: {', '.join(prompts) or 'not recorded'} · queries: {status_text} · {instances} saved mask instances. Empty and failed results remain visible.")
    final = rows[-1]
    facts.append(f"The map grows from past observations: {final.get('cumulative_voxels', 'unknown')} saved voxel cubes. Blue is reconstructed camera motion, separate from a planned route.")
    facts.append("Teal/orange show positive semantic evidence. Grey shows observed geometry; missing evidence and unseen space are unknown, not safe terrain.")
    original = manifest.get("original_planning", {})
    facts.append(f"Original planning: {original.get('availability', 'unknown')}. {original.get('reason') or 'Saved diagnostic output; no navigation safety certification.'}")
    research = manifest.get("research_plan") or {}
    assumptions = research.get("assumptions") or {}
    if research:
        geometry_only = research.get("planning_basis") == "geometry_only" and research.get("semantic_guidance") is False
        basis = "geometry-only" if geometry_only else "saved research"
        facts.append(f"Purple route: {basis} illustration · {research.get('status', 'unknown')}. It uses the final map only; segmentation does not guide this route." if geometry_only
                     else f"Purple route: {basis} illustration · {research.get('status', 'unknown')}. The route uses the final map only.")
        if research.get("reason"):
            facts.append(f"Research result: {research['reason']}")
        selection = (research.get("endpoints") or {}).get("selection")
        if selection == "automatic_demonstration_endpoints_on_largest_observed_traversable_component":
            facts.append("Endpoints: automatically selected demonstration cells on the largest observed component admitted by the assumed research checks.")
        elif selection == "camera_first_last_xy_projected_to_observed_support":
            facts.append("Endpoints: the first and last camera positions projected onto observed supporting cells; endpoint locations remain assumed.")
        else:
            facts.append("Research endpoint selection is unavailable; no endpoint locations are established.")
        facts.append("Research illustration — scale, ground direction and robot are assumptions; no safety or accuracy claim.")
        plane = research.get("ground_plane")
        ground_mode = (assumptions.get("ground_estimation") or {}).get("mode")
        if not plane:
            facts.append("No ground plane is available in this research result; the requested ground policy does not establish one.")
        elif ground_mode == "explicit_plane" and plane.get("method") == "explicit_user_assumed_plane":
            facts.append("Ground plane: explicitly supplied as a research assumption; ground direction is not measured gravity.")
        elif ground_mode == "camera_up_constrained_ransac" and plane.get("method") == "camera_up_prior_constrained_observed_plane_ransac":
            facts.append("Ground plane: fitted to saved geometry using an assumed camera-up prior. Floor identity and gravity remain unverified.")
        else:
            facts.append("A saved research ground plane is present, but its estimation method is not established by this metadata; ground direction remains assumed.")
        robot = assumptions.get("robot") or {}
        values = []
        factor = assumptions.get("metres_per_native_unit")
        if _number(factor):
            values.append(f"{factor:g} m per native map unit")
        for key, title in (("footprint_radius", "robot radius"), ("height", "height"), ("clearance", "clearance")):
            if _number(robot.get(key)):
                values.append(f"{title} {robot[key]:g} m")
        if values:
            facts.append("Assumed: " + "; ".join(values) + ". These are not measured calibration values.")
    if config.get("fixture") is True:
        facts.insert(0, "Synthetic fixture: software demonstration, not real model performance or accuracy.")
    return facts, timing


def render_player(label, title, video, first, last, facts, links, siblings):
    esc = html.escape
    fact_html = "".join(f"<li>{esc(str(fact))}</li>" for fact in facts)
    link_html = " · ".join(f'<a href="{esc(href, quote=True)}">{esc(name)}</a>' for name, href in links)
    nav = " · ".join(f'<a href="{esc(href, quote=True)}">{esc(name)}</a>' for name, href in siblings)
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(label)} — {esc(title)}</title><style>
body{{margin:0;background:#121a25;color:#edf1f5;font:17px/1.5 system-ui,sans-serif}}main{{max-width:1200px;margin:auto;padding:24px}}
h1{{font-size:29px;margin:0}}h2{{font-size:20px}}a{{color:#84d5ed}}video{{display:block;width:100%;max-height:78vh;background:#080d15;margin:20px 0}}
.previews{{display:flex;gap:18px;flex-wrap:wrap}}figure{{margin:0;flex:1;min-width:240px}}figure img{{width:100%;height:auto}}figcaption{{color:#c8d3df}}
li{{margin:9px 0}}nav,.sources{{padding:15px 0}}.note{{color:#bdcbd8}}</style></head>
<body><main><h1>{esc(label)}</h1><h2>{esc(title)}</h2><nav aria-label="Other movie views">{nav}</nav>
<video controls playsinline preload="metadata" poster="{data_url(first, 'image/jpeg')}"><source src="{data_url(video, 'video/mp4')}" type="video/mp4">Use the original MP4 link below if your browser cannot play this video.</video>
<p class="note">Use the video controls to play, pause and seek. Video and posters are embedded in this page.</p>
<div class="previews"><figure><img src="{data_url(first, 'image/jpeg')}" alt="First saved frame"><figcaption>First saved frame</figcaption></figure>
<figure><img src="{data_url(last, 'image/jpeg')}" alt="Final saved result"><figcaption>Final saved result</figcaption></figure></div>
<h2>What the movie shows</h2><ul>{fact_html}</ul><h2>Original files and provenance</h2><p class="sources">{link_html}</p>
</main></body></html>'''


def export_browser_players(export_dir, output_dir, label):
    export_dir, output_dir = Path(export_dir).resolve(), Path(output_dir).resolve()
    if not isinstance(label, str) or not label.strip() or len(label) > 200:
        raise ValueError("Use a nonempty label of at most 200 characters")
    if output_dir == export_dir or output_dir.is_relative_to(export_dir) or export_dir.is_relative_to(output_dir):
        raise ValueError("Browser output must be separate from the original export")
    if output_dir.exists():
        raise ValueError("Refusing an existing browser output directory")
    inputs = {}
    def record(path):
        digest = sha256(path)
        inputs[str(Path(path).resolve())] = digest
        return digest
    manifest_path, config_path = export_dir / "manifest.json", export_dir / "export_config.json"
    record(manifest_path); record(config_path)
    source = read_json(manifest_path)
    config = read_json(config_path)
    for field in ("run_dir", "sequence_path"):
        if source.get(field):
            original_root = Path(source[field]).resolve()
            if field == "sequence_path":
                original_root = original_root.parent
            if output_dir == original_root or output_dir.is_relative_to(original_root):
                raise ValueError("Browser output must be separate from the original run and sequence")
    if source.get("schema_version") != 1 or source.get("status") != "complete" or config.get("frames_only") is True:
        raise ValueError("Expected a completed MP4 export, not partial or frames-only output")
    coverage = source.get("coverage")
    if not isinstance(coverage, list) or not coverage or source.get("sampled_frame_count") != len(coverage):
        raise ValueError("Saved frame coverage/count differs")
    last_index = len(coverage) - 1
    selected_assets = {}
    for kind, title, frames_dir in PLAYERS:
        files = [f"{kind}.mp4", f"{frames_dir}/000000.png", f"{frames_dir}/{last_index:06d}.png"]
        for relative in files:
            path = safe_asset(export_dir, relative)
            limit = MAX_MP4_BYTES if relative.endswith(".mp4") else MAX_PNG_BYTES
            if not 0 < path.stat().st_size <= limit:
                raise ValueError(f"Empty or oversized browser asset: {relative}; MP4 cap is 32 MiB")
            digest = record(path)
            if source.get("outputs_sha256", {}).get(relative) != digest:
                raise ValueError(f"Saved asset hash mismatch: {relative}")
        selected_assets[kind] = (safe_asset(export_dir, files[0]), safe_asset(export_dir, files[1]), safe_asset(export_dir, files[2]))
    # Optional original metadata is read only when present and covered by the
    # export's recorded input hashes. Missing relocated inputs remain unlinked.
    sequence = None
    sequence_path = Path(source["sequence_path"]) if source.get("sequence_path") else None
    if sequence_path is not None and sequence_path.is_file():
        expected = source.get("input_sha256", {}).get(str(sequence_path.resolve()))
        if not expected or record(sequence_path) != expected:
            raise ValueError("Original sequence metadata hash differs from the saved export")
        sequence = read_json(sequence_path)
    facts, timing = describe_export(source, config, sequence)
    links = []
    def link(name, path):
        if path is not None and Path(path).exists():
            links.append((name, asset_href(path, output_dir)))
    for kind, title, _ in PLAYERS:
        link(f"Original {title.lower()} MP4", export_dir / f"{kind}.mp4")
    if sequence:
        original = sequence.get("input_provenance", {}).get("path")
        if original and Path(original).is_file():
            link("Source recording", Path(original))
    run = Path(source["run_dir"]) if source.get("run_dir") else None
    if run:
        for name, relative in (("Voxel data", "map/voxels.npz"), ("Semantic evidence", "map/semantic_evidence.npz"),
                               ("Masks and queries", "semantics/frames.jsonl"), ("Mask files", "semantics/masks"),
                               ("Original run", "run.json"), ("Map metadata", "map/manifest.json")):
            link(name, run / relative)
    link("Source sequence", sequence_path)
    link("Movie provenance", manifest_path); link("Movie settings", config_path)
    for path_text in source.get("input_sha256", {}):
        path = Path(path_text)
        if path.name == "research_plan.json":
            link("Research plan", path); link("Assumptions", path.parent / "assumptions.json")
    helper = Path(__file__).resolve()
    helper_hash = record(helper)
    browser_config = {"schema_version": 1, "artifact_kind": "self_contained_browser_video",
                      "label": label.strip(), "export_dir": str(export_dir), "helper_sha256": helper_hash,
                      "input_sha256": inputs, "max_mp4_bytes": MAX_MP4_BYTES,
                      "video_delivery": "inline_data_video_mp4", "thumbnail_delivery": "inline_data_image_jpeg",
                      "javascript": False, "original_inputs_read_only": True,
                      "facts": facts, "timing": timing, "original_links": links}
    result = {"schema_version": 1, "status": "running", "label": label.strip(),
              "created_time_ns": time.time_ns(), "helper_sha256": helper_hash,
              "input_sha256": inputs, "outputs_sha256": {}, "players": {},
              "self_hash_policy": "manifest hash is in manifest.sha256; no self-referential hash",
              "source_export_status": source["status"]}
    output_dir.mkdir(parents=True)
    def write_manifest():
        (output_dir / "manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    try:
        config_output = output_dir / "browser_config.json"
        config_output.write_text(json.dumps(browser_config, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        result["outputs_sha256"][config_output.name] = sha256(config_output)
        write_manifest()
        siblings = [(title, f"{kind}.html") for kind, title, _ in PLAYERS]
        for kind, title, _ in PLAYERS:
            video_path, first_path, last_path = selected_assets[kind]
            first, last = thumbnail(first_path), thumbnail(last_path)
            for suffix, payload in (("first", first), ("last", last)):
                thumb = output_dir / f"{kind}_{suffix}.jpg"
                thumb.write_bytes(payload)
                result["outputs_sha256"][thumb.name] = sha256(thumb)
            player = output_dir / f"{kind}.html"
            player.write_text(render_player(label.strip(), title, video_path.read_bytes(), first, last,
                                            facts, links, siblings), encoding="utf-8")
            result["outputs_sha256"][player.name] = sha256(player)
            result["players"][kind] = {"html": player.name, "first_thumbnail": f"{kind}_first.jpg",
                                        "last_thumbnail": f"{kind}_last.jpg", "source_mp4_sha256": inputs[str(video_path)]}
        result["status"] = "complete"
        write_manifest()
        (output_dir / "manifest.sha256").write_text(sha256(output_dir / "manifest.json") + "  manifest.json\n", encoding="ascii")
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_manifest()
        raise
    return result


def gallery_card(browser_dir, *, page_dir=None, label=None):
    """Return an inline-thumbnail HTML card for a separate batch gallery."""
    browser_dir = Path(browser_dir).resolve()
    manifest_path = safe_asset(browser_dir, "manifest.json")
    checksum_path = safe_asset(browser_dir, "manifest.sha256")
    if checksum_path.stat().st_size > 256:
        raise ValueError("Invalid gallery manifest checksum sidecar")
    fields = checksum_path.read_text(encoding="ascii").split()
    if (len(fields) != 2 or fields[1] != "manifest.json" or len(fields[0]) != 64
            or any(char not in "0123456789abcdef" for char in fields[0])
            or sha256(manifest_path) != fields[0]):
        raise ValueError("Gallery manifest checksum differs")
    manifest = read_json(manifest_path)
    if manifest.get("status") != "complete":
        raise ValueError("Gallery requires completed browser players")
    player = manifest["players"]["combined"]
    thumb = safe_asset(browser_dir, player["first_thumbnail"])
    if sha256(thumb) != manifest["outputs_sha256"].get(thumb.name):
        raise ValueError("Gallery thumbnail hash differs")
    page_dir = Path(page_dir or browser_dir.parent).resolve()
    links = []
    for kind, title, _ in PLAYERS:
        relative = manifest["players"][kind]["html"]
        player_path = safe_asset(browser_dir, relative)
        if player_path.suffix != ".html" or sha256(player_path) != manifest["outputs_sha256"].get(relative):
            raise ValueError("Gallery player path/hash differs")
        links.append(f'<a href="{html.escape(asset_href(player_path, page_dir), quote=True)}">{html.escape(title)}</a>')
    text_links = " · ".join(links)
    return (f'<article class="video-card"><h2>{html.escape(label or manifest["label"])}</h2>'
            f'<img src="{data_url(thumb.read_bytes(), "image/jpeg")}" alt="First saved video and voxel map" style="width:100%;max-width:600px">'
            f'<p>{text_links}</p><p>Sampled frames held between updates. Research illustration; no safety claim.</p></article>')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", dest="export_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args(argv)
    try:
        result = export_browser_players(args.export_dir, args.output, args.label)
        print(json.dumps({"status": result["status"], "output": str(args.output.resolve()), "players": list(result["players"])}))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Browser export stopped: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
