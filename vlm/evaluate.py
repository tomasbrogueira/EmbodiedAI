# Support direct compatibility-script execution without environment mutation.
if __package__ in (None, ''):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
"""Evaluate VLM proposals on video keyframes without running SAM or LingBot."""

import argparse
import html
import io
import json
import math
import subprocess
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from orchestrator.shared import parse_proposal, write_json
from vlm.propose import DEFAULT_MODEL, INSTRUCTION, QwenProposer


def video_info(path):
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate:format=duration",
        "-of", "json", str(path),
    ], capture_output=True, check=True, text=True)
    data = json.loads(result.stdout)
    return {"duration_seconds": float(data["format"]["duration"]), **data["streams"][0]}


def decode_frame(path, seconds):
    result = subprocess.run([
        "ffmpeg", "-v", "error", "-ss", str(seconds), "-i", str(path),
        "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-threads", "1", "-",
    ], capture_output=True, check=True)
    return Image.open(io.BytesIO(result.stdout)).convert("RGB")


def preprocess(image):
    # Match the mapper's width=518, patch-rounded height, center-crop procedure.
    height = max(14, round(image.height * 518 / image.width / 14) * 14)
    image = image.resize((518, height), Image.Resampling.BICUBIC)
    if height > 518:
        top = (height - 518) // 2
        image = image.crop((0, top, 518, top + 518))
    return image


def contact_sheet(directory, records):
    width, height, columns = 280, 350, 3
    sheet = Image.new("RGB", (width * columns, height * math.ceil(len(records) / columns)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    for idx, record in enumerate(records):
        x, y = (idx % columns) * width, (idx // columns) * height
        image = Image.open(directory / record["image"]).convert("RGB")
        image.thumbnail((width - 12, 260))
        sheet.paste(image, (x + 6, y + 4))
        label = f'{record["timestamp_seconds"]:.2f}s: '
        label += ", ".join(record["prompts"]) if record["prompts"] is not None else "INVALID JSON"
        if record["prompts"] == []:
            label += "none"
        # Wrap captions at word boundaries so long responses remain readable.
        lines, line = [], ""
        for word in label.split():
            candidate = (line + " " + word).strip()
            if draw.textlength(candidate, font=font) > width - 12:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
        draw.multiline_text((x + 6, y + 267), "\n".join(lines[:6]), fill="black", font=font)
    sheet.save(directory / "contact_sheet.jpg", quality=90)


def write_report(root, report):
    rows, details = [], []
    for video in report["videos"]:
        directory = video["artifact_directory"]
        records = video["frames"]
        prompts = sorted({p for r in records for p in (r["prompts"] or [])})
        invalid = sum(r["prompts"] is None for r in records)
        rows.append(f'<tr><td><a href="{directory}/contact_sheet.jpg">{html.escape(video["video"])}</a></td>'
                    f'<td>{len(records)}</td><td>{invalid}</td><td>{html.escape(", ".join(prompts)) or "none"}</td></tr>')
        cards = []
        for record in records:
            image = f'{directory}/{record["image"]}'
            context = f'{directory}/{record["context_image"]}'
            cards.append(f'<figure><a href="{context}"><img src="{image}" loading="lazy"></a>'
                         f'<figcaption>{record["timestamp_seconds"]:.2f}s — '
                         f'{html.escape(str(record["prompts"]))}</figcaption>'
                         f'<details><summary>Raw response</summary><pre>{html.escape(record["raw_response"])}</pre></details></figure>')
        details.append(f'<h2>{html.escape(video["video"])}</h2><div class="frames">{"".join(cards)}</div>')
    page = '<!doctype html><meta charset="utf-8"><title>VLM video evaluation</title>'
    page += '<style>body{font-family:system-ui;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px}.frames{display:flex;flex-wrap:wrap}figure{width:260px;margin:10px}img{width:260px}pre{white-space:pre-wrap}figcaption{overflow-wrap:anywhere}</style>'
    page += '<h1>VLM video evaluation</h1><p>Model: ' + html.escape(report["model"]) + '</p>'
    page += f'<p>Sampling: every {report["interval_seconds"]} seconds, plus a frame near the end. '
    page += 'Inputs use LingBot-style resize/crop. Click a frame for the uncropped context. '
    page += 'No ground-truth annotations: outputs are proposals, not measured detection accuracy.</p>'
    page += '<details><summary>Instruction</summary><pre>' + html.escape(report["instruction"]) + '</pre></details>'
    page += '<table><tr><th>Video / contact sheet</th><th>Frames</th><th>Invalid responses</th><th>Proposed concepts</th></tr>'
    page += "".join(rows) + '</table>' + "".join(details)
    (root / "index.html").write_text(page)
    write_json(root / "results.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", type=Path, default=Path("data/videos"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=2.0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--robot", default="a small wheeled ground robot")
    parser.add_argument("--instruction-file", type=Path, help="Override for prompt comparison")
    parser.add_argument("--resume", action="store_true", help="Resume the same instruction and sampling run")
    args = parser.parse_args()
    if not math.isfinite(args.interval_seconds) or args.interval_seconds <= 0:
        parser.error("--interval-seconds must be finite and positive")
    videos = sorted(p for p in args.videos_dir.rglob("*") if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"})
    if not videos:
        parser.error("No videos found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    instruction = (args.instruction_file.read_text() if args.instruction_file else INSTRUCTION).replace("{robot}", args.robot)
    if args.resume and (args.output_dir / "results.json").exists():
        previous = json.loads((args.output_dir / "results.json").read_text())
        if (previous["instruction"] != instruction or previous["model"] != args.model
                or previous["interval_seconds"] != args.interval_seconds):
            parser.error("Resume requires the same model, instruction and sampling interval")
    report = {"model": args.model, "interval_seconds": args.interval_seconds,
              "instruction": instruction, "preprocessing": "LingBot-style 518-width resize and center crop; PIL bicubic",
              "ground_truth_available": False, "videos": []}
    (args.output_dir / "instruction.txt").write_text(instruction)
    proposer = QwenProposer(args.model, args.device)
    for path in videos:
        info = video_info(path)
        duration = info["duration_seconds"]
        # Stay sufficiently before EOF even on variable-frame-rate clips.
        last = max(0.0, duration - 0.2)
        timestamps = [i * args.interval_seconds for i in range(math.floor(last / args.interval_seconds) + 1)]
        if last - timestamps[-1] > 0.5:
            timestamps.append(last)
        directory = args.output_dir / path.parent.name / path.stem
        directory.mkdir(parents=True, exist_ok=True)
        records = []
        if args.resume and (directory / "results.json").exists():
            records = json.loads((directory / "results.json").read_text())["frames"]
            if [r["timestamp_seconds"] for r in records] != timestamps[:len(records)]:
                parser.error(f"Saved timestamps do not match {path}")
        video = {"video": str(path), "metadata": info,
                 "artifact_directory": directory.relative_to(args.output_dir).as_posix(), "frames": records}
        report["videos"].append(video)
        for idx, seconds in enumerate(timestamps):
            if idx < len(records):
                continue
            context = decode_frame(path, seconds)
            image = preprocess(context)
            image_name, context_name = f"frame_{idx:03d}.jpg", f"context_{idx:03d}.jpg"
            image.save(directory / image_name, quality=95)
            context.thumbnail((960, 960))
            context.save(directory / context_name, quality=90)
            start = time.perf_counter()
            raw = proposer.propose(image, instruction)
            elapsed = time.perf_counter() - start
            error = None
            try:
                prompts = parse_proposal(raw)
            except (ValueError, KeyError) as exc:
                prompts, error = None, str(exc)
            record = {"timestamp_seconds": seconds, "image": image_name,
                      "context_image": context_name, "raw_response": raw,
                      "prompts": prompts, "error": error, "inference_seconds": elapsed}
            records.append(record)
            write_json(directory / "results.json", video)
            print(f"[{path.parent.name}/{path.stem}] {seconds:.2f}s: {prompts} ({elapsed:.2f}s)", flush=True)
        contact_sheet(directory, records)
        write_report(args.output_dir, report)
    print(f"Evaluated {len(videos)} videos; report: {args.output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
