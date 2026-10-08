"""CPU packaging of verified native LingBot-map captures and saved segmentation.

Only complete saved previews of at most nine frames are admitted. The native
WebGL pixels are pasted unchanged between explanatory captions; no geometry,
mask, model, renderer or route is recomputed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import textwrap
import time

import numpy as np
from PIL import Image, ImageDraw, __version__ as PILLOW_VERSION

from export_browser_video import (
    MAX_MP4_BYTES, MAX_PNG_BYTES, asset_href, read_json, render_player, safe_asset,
    sha256, thumbnail,
)
from export_pipeline_video import _font, sampled_timeline

SOURCE_REVISION = "849e690bb086103637e44b1e91878d9d43a8bf0c"
ENCODER_ADAPTER_SHA256 = "d546e85a5c42368fe4fa0f577b1c98070293eee187ce695e80a28cb0a0cd2f78"
RENDERER = "official_lingbot_PointCloudViewer_Viser_WebGL_get_render"
PLAYERS = (("combined", "Native reconstruction + video masks"),
           ("segmentation", "Video masks — saved segmentation evidence"),
           ("native_map", "Native LingBot-map point cloud"))
IDENTITY = ("geometry_fingerprint", "input_fingerprint", "processed_grid_id",
            "archive_sha256", "map_frame", "units", "scale", "up")


def make_encoder(stop):
    """Reuse the frozen encoder that passed the independent Linux lifecycle suite."""
    adapter = Path(__file__).with_name("serve_lingbot_view.py")
    if sha256(adapter)!=ENCODER_ADAPTER_SHA256:
        raise ValueError("Owned encoder adapter differs from the reviewed frozen hash")
    from serve_lingbot_view import OwnedEncoder
    return OwnedEncoder(stop)


class EncoderLifetime:
    """Own subprocess groups through cleanup; signal handlers never skip finally."""
    def __init__(self,max_seconds=600):
        if os.name!="posix" or not Path("/proc/self/stat").is_file():
            raise RuntimeError("Bounded encoding requires the reviewed Linux process-group runtime")
        self.deadline = time.monotonic()+max_seconds
        self.stop = threading.Event()
        self.handlers = {}
        self.encoder = make_encoder(self.stop)
        self.calls = []

    def __enter__(self):
        for number in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
            self.handlers[number] = signal.signal(number,lambda *_:self.stop.set())
        return self

    def __exit__(self,*_):
        try:
            self.encoder.cancel()
        finally:
            for number,handler in self.handlers.items():
                signal.signal(number,handler)

    def call(self,command,log_path,*,timeout):
        if self.stop.is_set() or time.monotonic()>=self.deadline:
            raise RuntimeError("Encoder interrupted or preview deadline reached")
        limit = min(self.deadline,time.monotonic()+timeout)
        record = {"command":list(command),"log_path":str(log_path),"returncode":None}
        try:
            record["returncode"] = self.encoder.run(command,log_path,limit)
            return record["returncode"]
        finally:
            try:
                record["cleanup"] = self.encoder.cancel()
            finally:
                self.calls.append(record)


def lifecycle_self_test():
    """Focused CPU Linux smoke: normal/failure, TERM/INT, timeout, unrelated PID."""
    if os.name!="posix":
        return {"status":"skipped","reason":"Linux process-group checks require POSIX /proc"}
    import tempfile
    with tempfile.TemporaryDirectory(prefix="native_encoder_lifecycle_") as directory:
        root = Path(directory)
        unrelated = subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"],start_new_session=True)
        checks = []
        try:
            for code,expected in (("pass",0),("raise SystemExit(7)",7)):
                with EncoderLifetime(15) as lifetime:
                    exit_code = lifetime.call([sys.executable,"-c",code],root/"log",timeout=5)
                    assert exit_code==expected and lifetime.encoder.cleanup["reaped"] and lifetime.encoder.cleanup["group_gone"]
                    checks.append("normal_exit" if expected==0 else "child_failure_reaped")
            for number in (signal.SIGTERM,signal.SIGINT):
                with EncoderLifetime(15) as lifetime:
                    timer = threading.Timer(.3,lambda n=number:os.kill(os.getpid(),n))
                    timer.start()
                    try:
                        lifetime.call([sys.executable,"-c","import time;time.sleep(30)"],root/"log",timeout=5)
                    except RuntimeError:
                        assert lifetime.encoder.cleanup["reaped"] and lifetime.encoder.cleanup["group_gone"]
                    else:
                        raise AssertionError("Signal did not interrupt owned encoder")
                    finally:
                        timer.cancel();timer.join()
                    checks.append(signal.Signals(number).name+"_cleanup")
            with EncoderLifetime(15) as lifetime:
                try:
                    lifetime.call([sys.executable,"-c","import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)"],
                                  root/"log",timeout=.3)
                except RuntimeError:
                    assert lifetime.encoder.cleanup["reaped"] and lifetime.encoder.cleanup["group_gone"]
                else:
                    raise AssertionError("Deadline did not interrupt owned encoder")
                checks.append("deadline_TERM_KILL_reaped")
            original_popen = subprocess.Popen
            def fail_wait_once(*args,**kwargs):
                child = original_popen(*args,**kwargs)
                original_wait = child.wait
                def failed_wait(*args,**kwargs):
                    child.wait = original_wait
                    raise RuntimeError("injected monitor exception")
                child.wait = failed_wait
                return child
            with EncoderLifetime(15) as lifetime:
                subprocess.Popen = fail_wait_once
                try:
                    lifetime.call([sys.executable,"-c","import time;time.sleep(30)"],root/"log",timeout=5)
                except RuntimeError:
                    assert lifetime.encoder.cleanup["reaped"] and lifetime.encoder.cleanup["group_gone"]
                else:
                    raise AssertionError("Injected exception did not run encoder cleanup")
                finally:
                    subprocess.Popen = original_popen
                checks.append("exception_cleanup_reaped")
            assert unrelated.poll() is None
            checks.append("unrelated_process_preserved")
        finally:
            unrelated.terminate();unrelated.wait(timeout=5)
    return {"status":"passed","checks":checks}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    os.replace(temporary, path)


def _png(path):
    if not 0 < path.stat().st_size <= MAX_PNG_BYTES:
        raise ValueError("Empty or oversized saved PNG")
    with Image.open(path) as image:
        if image.format != "PNG" or image.size != (800,800):
            raise ValueError("Native preview requires the saved 800×800 PNG grid")
        return np.asarray(image.convert("RGB")).copy()


def load_preview(native_dir, segmentation_dir, native_manifest_sha256):
    """Admit hash-pinned complete outputs; join every source sample exactly."""
    native_dir, segmentation_dir = Path(native_dir).resolve(), Path(segmentation_dir).resolve()
    if not isinstance(native_manifest_sha256,str) or not re.fullmatch(r"[0-9a-f]{64}",native_manifest_sha256):
        raise ValueError("Supply the reviewed final native manifest SHA256")
    inputs = {}
    def record(path):
        digest = sha256(path)
        inputs[str(path.resolve())] = digest
        return digest
    def metadata(root, name):
        path = safe_asset(root,name)
        record(path)
        return read_json(path)
    native_path = safe_asset(native_dir,"manifest.json")
    if record(native_path) != native_manifest_sha256:
        raise ValueError("Native render manifest SHA256 differs from the reviewed snapshot")
    native = read_json(native_path)
    native_config = metadata(native_dir,"viewer_config.json")
    segmentation = metadata(segmentation_dir,"manifest.json")
    segmentation_config = metadata(segmentation_dir,"export_config.json")
    if (native.get("schema_version")!=1 or native.get("status")!="complete"
            or native.get("capture",{}).get("status")!="complete"
            or native.get("capture",{}).get("renderer")!=RENDERER):
        raise ValueError("Require final complete genuine native WebGL capture, after viewer shutdown")
    if native.get("outputs_sha256",{}).get("viewer_config.json") != inputs[str((native_dir/"viewer_config.json").resolve())]:
        raise ValueError("Native viewer settings hash differs from capture manifest")
    if (native_config.get("source_revision")!=SOURCE_REVISION or native_config.get("device")!="cpu"
            or native_config.get("model") is not None or native_config.get("sky_segmentation") is not False
            or native_config.get("saved_xyz_changed") is not False
            or native_config.get("saved_extrinsic")!="world_to_camera_unchanged"
            or native_config.get("native_use_point_map") is not True):
        raise ValueError("Native capture lacks the reviewed exact saved-cache CPU provenance")
    if (segmentation.get("schema_version")!=1 or segmentation.get("status")!="complete"
            or segmentation_config.get("frames_only") is not False
            or segmentation_config.get("rgb_view")!="source"
            or segmentation_config.get("mask_projection")!="saved_source_to_processed_center_affine_nearest_no_extrapolation"):
        raise ValueError("Require a complete original-frame segmentation export using the exact saved transform")
    native_rows = native.get("coverage")
    segmentation_rows = segmentation.get("coverage")
    captures = native["capture"].get("frames")
    if (not isinstance(native_rows,list) or not isinstance(segmentation_rows,list) or not isinstance(captures,list)
            or not 1<=len(native_rows)<=9 or len(native_rows)!=len(segmentation_rows) or len(native_rows)!=len(captures)
            or segmentation.get("sampled_frame_count")!=len(native_rows)):
        raise ValueError("Exact saved native/segmentation frame counts differ or exceed nine")
    for field in ("run_dir","sequence_path","original_planning","research_plan"):
        if native.get(field)!=segmentation.get(field):
            raise ValueError("Native/segmentation source differs: "+field)
    for key in IDENTITY:
        if key not in native.get("geometry_identity",{}) or key not in segmentation.get("geometry_identity",{}):
            raise ValueError("Saved geometry identity is missing: "+key)
        if native["geometry_identity"][key]!=segmentation["geometry_identity"][key]:
            raise ValueError("Native/segmentation geometry differs: "+key)
    common_hashes = set(native.get("input_sha256",{})) & set(segmentation.get("input_sha256",{}))
    if not common_hashes or any(native["input_sha256"][key]!=segmentation["input_sha256"][key] for key in common_hashes):
        raise ValueError("Native/segmentation saved input hashes differ")
    seen = set()
    native_pixels, segmentation_pixels = [], []
    research = native.get("research_plan") or {}
    if research and (research.get("planning_basis")!="geometry_only" or research.get("semantic_guidance") is not False):
        raise ValueError("Saved research illustration must explicitly be geometry-only without semantic guidance")
    for index,(nrow,srow,capture) in enumerate(zip(native_rows,segmentation_rows,captures)):
        frame_id = nrow.get("frame_id")
        if not isinstance(frame_id,str) or frame_id in seen:
            raise ValueError("Missing or duplicated native frame identity")
        seen.add(frame_id)
        for field in ("frame_id","timestamp_ns","timestamp_provenance","disposition","semantic_status"):
            if nrow.get(field)!=srow.get(field):
                raise ValueError("Exact native/segmentation sample join differs: "+field)
        if capture.get("frame_id")!=frame_id or capture.get("timestamp_ns")!=nrow.get("timestamp_ns"):
            raise ValueError("Native capture order/timestamp differs from source sample")
        if capture.get("visible_frame_indices")!=list(range(index+1)):
            raise ValueError("Native capture would include a wrong/future cumulative frame")
        if type(capture.get("research_route_visible")) is not bool:
            raise ValueError("Native route visibility is missing")
        if capture["research_route_visible"] and (research.get("status")!="ok" or index!=len(native_rows)-1):
            raise ValueError("Native capture claims an unavailable or nonfinal research route")
        def normalized_queries(row):
            return [{key:q.get(key) for key in ("prompt","status","instances","error")} for q in row.get("queries",[])]
        if normalized_queries(nrow)!=normalized_queries(srow):
            raise ValueError("Native/segmentation saved query status or empty result differs")
        for root,manifest,relative,collection in (
                (native_dir,native,f"native_frames/{index:06d}.png",native_pixels),
                (segmentation_dir,segmentation,f"segmentation_frames/{index:06d}.png",segmentation_pixels)):
            path = safe_asset(root,relative)
            if record(path)!=manifest.get("outputs_sha256",{}).get(relative):
                raise ValueError("Saved preview PNG hash differs: "+relative)
            collection.append(_png(path))
    timeline = sampled_timeline([{"timestamp_ns":row["timestamp_ns"],"source":{"timestamp_provenance":row.get("timestamp_provenance",{})}}
                                 for row in native_rows], video_fps=20,end_hold_seconds=2)
    if native_config.get("timeline")!=timeline or segmentation_config.get("timeline")!=timeline:
        raise ValueError("Native/segmentation held source timestamp timeline differs")
    if segmentation.get("encoded_frame_count")!=sum(timeline["repeat_counts"]) or any(
            row.get("encoded_repeat_count")!=count for row,count in zip(segmentation_rows,timeline["repeat_counts"])):
        raise ValueError("Saved segmentation encoding differs from the sampled held timeline")
    for root,manifest,relative in ((segmentation_dir,segmentation,"segmentation.mp4"),
                                   (native_dir,native,"native.mp4")):
        path = safe_asset(root,relative)
        if relative=="native.mp4" and not path.exists():
            if relative in manifest.get("outputs_sha256",{}):
                raise ValueError("Recorded native MP4 is missing")
            continue
        if not 0<path.stat().st_size<=MAX_MP4_BYTES or record(path)!=manifest.get("outputs_sha256",{}).get(relative):
            raise ValueError("Original saved MP4 hash/size differs: "+relative)
    for path_text,digest in list(inputs.items()):
        if sha256(path_text)!=digest:
            raise ValueError("Saved preview input changed during admission")
    return {"native_dir":native_dir,"segmentation_dir":segmentation_dir,"native":native,
            "native_config":native_config,"segmentation":segmentation,"segmentation_config":segmentation_config,
            "native_pixels":native_pixels,"segmentation_pixels":segmentation_pixels,"timeline":timeline,"input_sha256":inputs}


def ensure_output(output,data):
    output = Path(output).resolve()
    roots = [data["native_dir"],data["segmentation_dir"]]
    for source in (data["native"],data["segmentation"]):
        for field in ("run_dir","sequence_path"):
            if source.get(field):
                root = Path(source[field]).resolve()
                roots.append(root.parent if field=="sequence_path" else root)
    if output.exists() or any(output.is_relative_to(root) for root in roots):
        raise ValueError("Native preview output must be fresh and separate from every original input")
    return output


def _draw(draw,xy,text,size=18,width=77):
    draw.multiline_text(xy,"\n".join(textwrap.wrap(str(text),width=width)),font=_font(size),fill=(231,238,245),spacing=3)


def panels(data,index):
    row = data["native"]["coverage"][index]
    research = data["native"].get("research_plan") or {}
    provenance = row.get("timestamp_provenance",{})
    clock = "video" if provenance.get("clock")=="video_presentation_timeline" else "sample"
    timestamp = row.get("timestamp_ns")
    time_label = f"{clock} time {timestamp/1e9:.3f}s" if timestamp is not None else "source time unknown"
    footer = "Sampled results held between updates; no new masks or geometry during holds."
    if research:
        status = research.get("status","unknown")
        route = data["native"]["capture"]["frames"][index]["research_route_visible"]
        footer += (" Research illustration: saved assumed route at final map." if route else
                   " Research illustration: no path found; no route drawn." if status=="no_path" else
                   " Research illustration: blocked inputs; no route drawn." if status=="blocked_inputs" else
                   " Research route is reserved for the final saved map.")
        footer += " Assumed scale, ground and robot; no safety claim."
    native = Image.new("RGB",(800,1000),(17,25,37))
    segmentation = Image.new("RGB",(800,1000),(17,25,37))
    for image,title,pixels in ((native,"Native LingBot-map reconstruction",data["native_pixels"][index]),
                               (segmentation,"Video masks — saved evidence",data["segmentation_pixels"][index])):
        image.paste(Image.fromarray(pixels),(0,104))
        draw = ImageDraw.Draw(image)
        _draw(draw,(18,12),title,size=27,width=48)
        _draw(draw,(18,50),f"Saved sample {index+1}/{len(data['native_pixels'])} · {time_label}",size=20,width=63)
        _draw(draw,(18,80),"RGB point cloud; blue = recorded camera motion" if image is native else
              "Turquoise = ground/floor mask evidence within the model crop",size=16,width=83)
        _draw(draw,(18,914),footer,size=16,width=88)
    combined = Image.new("RGB",(1600,1000))
    combined.paste(segmentation,(0,0));combined.paste(native,(800,0))
    return {"native_map":native,"segmentation":segmentation,"combined":combined}


def facts(data):
    native,segmentation = data["native"],data["segmentation"]
    rows = native["coverage"]
    times = [row["timestamp_ns"] for row in rows]
    clock = "video" if any(row.get("timestamp_provenance",{}).get("clock")=="video_presentation_timeline" for row in rows) else "sample"
    result = ["The point cloud is rendered by LingBot-map’s official Viser viewer from the exact saved cache. RGB points and camera frustums are native; blue camera motion is our added overlay.",
              "Segmentation is a separate saved overlay on the original video frame. The model crop bounds its masks; unseen areas and empty results are unknown.",
              f"{len(rows)} saved samples. Movies run at 20 encoded frames/s by holding each sample to the next saved timestamp, with a final 2-second hold. No new geometry or masks are inferred during holds."]
    if all(isinstance(t,int) and not isinstance(t,bool) and t>=0 for t in times):
        result.append(f"Saved {clock} times: {times[0]/1e9:.3f}–{times[-1]/1e9:.3f}s. Encoded timing is rounded to 20 fps; exact timestamps remain in provenance.")
    result.append("Video times are decoded presentation timestamps, not wall-clock capture times." if clock=="video" else
                  "The saved sample clock does not establish wall-clock capture timing.")
    queries = [q for row in rows for q in row.get("queries",[])]
    prompts = sorted({str(q.get("prompt")) for q in queries})
    statuses = Counter(str(q.get("status","unknown")) for q in queries)
    result.append("Prompts: "+(", ".join(prompts) or "none recorded")+". Query results: "+
                  (", ".join(f"{count} {status}" for status,count in sorted(statuses.items())) or "none recorded")+
                  ". Saved empty and failed results remain visible; mask colors mean instance/evidence identity, never traversability.")
    overlays = data["native_config"].get("optional_added_overlays",{})
    if overlays.get("semantic_positive_evidence"):
        result.append("Our added turquoise/orange voxel-center overlay indicates positive candidate/hazard evidence. It is separate from native RGB geometry and does not certify terrain.")
    if overlays.get("saved_fused_voxel_cuboids"):
        result.append("Our optional wireframe cuboids show the saved fused voxel grid; they are an added overlay on the native point cloud.")
    units = native["geometry_identity"].get("units","unknown")
    result.append(f"Saved XYZ, W2C cameras and units ({units}) are unchanged. Observed surfaces do not establish free space, terrain safety or reconstruction accuracy.")
    original = native.get("original_planning") or {}
    result.append(f"Original planning remains {original.get('availability','unknown')}: {original.get('reason') or 'no navigation safety claim'}.")
    research = native.get("research_plan") or {}
    if research:
        status = research.get("status","unknown")
        drawn = any(row["research_route_visible"] for row in native["capture"]["frames"])
        result.append(f"Separate research illustration: {status}. "+("The saved route appears only on the final cumulative map." if drawn else "No route is drawn.")+
                      " This plan is geometry-only; segmentation does not guide it.")
        if research.get("reason"):
            result.append("Research result: "+str(research["reason"])+".")
        assumptions = research.get("assumptions") or {}
        plane = research.get("ground_plane") or {}
        mode = (assumptions.get("ground_estimation") or {}).get("mode")
        if mode=="explicit_plane" and plane.get("method")=="explicit_user_assumed_plane":
            result.append("Ground plane is explicitly assumed, not measured gravity.")
        elif mode=="camera_up_constrained_ransac" and plane.get("method")=="camera_up_prior_constrained_observed_plane_ransac":
            result.append("Ground plane is fitted to saved geometry using an assumed camera-up prior. Gravity and floor identity remain unverified.")
        else:
            result.append("Ground calibration is unavailable or unverified in this research result.")
        selection = (research.get("endpoints") or {}).get("selection")
        if selection=="automatic_demonstration_endpoints_on_largest_observed_traversable_component":
            result.append("Research endpoints are automatic demonstration cells on the largest observed component admitted by the assumed checks.")
        elif selection=="camera_first_last_xy_projected_to_observed_support":
            result.append("Research endpoints are first/last camera positions projected to observed support; their locations remain assumed.")
        else:
            result.append("No research endpoints are established by this saved result.")
        robot = assumptions.get("robot") or {}
        values = []
        for value,title in ((assumptions.get("metres_per_native_unit"),"metres per native unit"),
                            (robot.get("footprint_radius"),"robot radius m"),(robot.get("height"),"robot height m"),
                            (robot.get("clearance"),"robot clearance m")):
            if isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value):
                values.append(f"{value:g} {title}")
        result.append("Research illustration — assumed scale, ground direction and robot; no safety claim. "+"; ".join(values)+".")
    if data["segmentation_config"].get("fixture"):
        result.insert(0,"Synthetic fixture: this demonstrates software behavior, not actual model performance.")
    return result


def main_facts(data):
    rows = data["native"]["coverage"]
    times = [row["timestamp_ns"] for row in rows]
    research = data["native"].get("research_plan") or {}
    status = research.get("status")
    drawn = any(row["research_route_visible"] for row in data["native"]["capture"]["frames"])
    route = ("Research illustration: no route found." if status=="no_path" else
             "Research illustration: blocked inputs; no route shown." if status=="blocked_inputs" else
             "Research illustration: the saved assumed route appears only on the final map." if drawn else
             "No planned route is displayed.")
    time_note = (f"Saved video/sample times {times[0]/1e9:.3f}–{times[-1]/1e9:.3f}s." if
                 all(isinstance(t,int) and not isinstance(t,bool) for t in times) else "Source times are incomplete.")
    return [f"{len(rows)}-frame preview: LingBot-map’s native point cloud beside the saved video masks.",
            "Turquoise = ground/floor mask evidence; blue = recorded camera motion. Masks are candidate evidence, not traversability.",
            route+" Scale, ground direction and robot are assumptions; no safety claim.",
            time_note+" Samples are held between updates; the final view holds2seconds. No new masks or geometry appear during holds."]


def player_html(label,title,video,first,last,data,links):
    page = render_player(label,title,video,first,last,main_facts(data),links,
                         [(name,f"{kind}.html") for kind,name in PLAYERS])
    details = ("<details><summary>Assumptions and provenance notes</summary><ul>"+
               "".join("<li>"+html.escape(fact)+"</li>" for fact in facts(data))+"</ul></details>")
    return page.replace("<h2>Original files and provenance</h2>",details+"<h2>Original files and provenance</h2>")


def encode_and_probe(binary,probe,output,kind,timeline,lifetime):
    listing = output/f"{kind}.frames.txt"
    listing.write_text("".join(f"file '{kind}_frames/{index:06d}.png'\n"*count
                              for index,count in enumerate(timeline["repeat_counts"])),encoding="utf-8")
    target = output/f"{kind}.mp4"
    exit_code = lifetime.call([binary,"-hide_banner","-loglevel","error","-n","-threads","4","-r","20",
                             "-f","concat","-safe","1","-i",str(listing),"-an","-c:v","libx264","-threads","4",
                             "-pix_fmt","yuv420p","-crf","20","-movflags","+faststart",str(target)],
                             output/f"{kind}.encoder.log",timeout=120)
    if exit_code or not target.is_file() or not 0<target.stat().st_size<=MAX_MP4_BYTES:
        raise RuntimeError("MP4 encoding failed/oversized: "+kind)
    probe_output = output/f"{kind}.probe.json"
    exit_code = lifetime.call([probe,"-v","error","-count_frames","-select_streams","v:0","-show_streams","-show_format",
                              "-threads","4","-of","json",str(target)],
                             probe_output,timeout=30)
    if exit_code:
        raise RuntimeError("MP4 validation failed: "+kind)
    report = read_json(probe_output)
    streams = report.get("streams",[])
    expected_count = sum(timeline["repeat_counts"])
    expected_width = 1600 if kind=="combined" else 800
    if len(streams)!=1:
        raise RuntimeError("Expected one video stream")
    stream = streams[0]
    if (stream.get("codec_name")!="h264" or stream.get("pix_fmt")!="yuv420p" or stream.get("width")!=expected_width
            or stream.get("height")!=1000 or int(stream.get("nb_read_frames",-1))!=expected_count
            or stream.get("avg_frame_rate")!="20/1"
            or abs(float(report.get("format",{}).get("duration",-1))-expected_count/20)>.051):
        raise RuntimeError("Encoded frame count/dimensions/rate/duration differs: "+kind)
    write_json(output/f"{kind}.ffprobe.json",report)
    return target


def export_preview(data,output,label,*,max_seconds=600):
    if not isinstance(label,str) or not label.strip() or len(label)>200:
        raise ValueError("Use a readable nonempty label of at most200 characters")
    output = ensure_output(output,data)
    binary,probe = shutil.which("ffmpeg"),shutil.which("ffprobe")
    if not binary or not probe:
        raise RuntimeError("Native preview packaging requires existing ffmpeg and ffprobe; no install is performed")
    if not isinstance(max_seconds,int) or not 1<=max_seconds<=900:
        raise ValueError("CPU preview lifetime must be within1–900seconds")
    lifetime = EncoderLifetime(max_seconds)
    output.mkdir(parents=True,exist_ok=False)
    font = _font(18)
    font_path = getattr(font,"path",None)
    config = {"schema_version":1,"artifact_kind":"native_lingbot_saved_preview","helper_sha256":sha256(__file__),
              "read_only_inputs":True,"input_sha256":data["input_sha256"],"timeline":data["timeline"],
              "native_pixel_rectangle_xywh":[0,104,800,800],"native_pixels_unchanged":True,
              "panel_size":[800,1000],"annotation":"deterministic_Pillow_captions_only","pillow_version":PILLOW_VERSION,
              "annotation_font_name":list(font.getname()),
              "annotation_font_path":font_path if isinstance(font_path,str) else "Pillow bundled font",
              "annotation_font_sha256":sha256(font_path) if isinstance(font_path,str) and Path(font_path).is_file() else None,
              "support_helpers_sha256":{name:sha256(Path(__file__).with_name(name))
                                        for name in ("export_browser_video.py","export_pipeline_video.py","serve_lingbot_view.py")},
              "owned_encoder_adapter_sha256":ENCODER_ADAPTER_SHA256,
              "encoder":binary,"probe":probe,"cpu_threads":4,"browser_mp4_cap_bytes":MAX_MP4_BYTES,
              "max_runtime_seconds":max_seconds,"encoder_ownership":"new_session_TERM_then_KILL_and_reap"}
    result = {"schema_version":1,"status":"running","label":label.strip(),"helper_sha256":sha256(__file__),
              "input_sha256":data["input_sha256"],"outputs_sha256":{},"players":{},
              "coverage":data["segmentation"]["coverage"],"sampled_frame_count":len(data["native_pixels"]),
              "encoded_frame_count":sum(data["timeline"]["repeat_counts"]),"geometry_identity":data["native"]["geometry_identity"],
              "original_planning":data["native"]["original_planning"],"research_plan":data["native"].get("research_plan"),
              "native_renderer":RENDERER,"native_viewer_config":data["native_config"],"facts":facts(data)}
    def record_output(path):
        result["outputs_sha256"][str(path.relative_to(output))] = sha256(path)
    def save_manifest():
        write_json(output/"manifest.json",result)
    lifetime.__enter__()
    try:
        write_json(output/"export_config.json",config);record_output(output/"export_config.json");save_manifest()
        for kind,_ in PLAYERS:
            (output/f"{kind}_frames").mkdir()
        for index in range(len(data["native_pixels"])):
            for kind,image in panels(data,index).items():
                path = output/f"{kind}_frames/{index:06d}.png"
                image.save(path);record_output(path)
        links = []
        def link(name,path):
            if path.exists():
                links.append((name,asset_href(path,output)))
        link("Original native captures",data["native_dir"]/"native_frames")
        link("Original native MP4",data["native_dir"]/"native.mp4")
        link("Original segmentation MP4",data["segmentation_dir"]/"segmentation.mp4")
        link("Native capture provenance",data["native_dir"]/"manifest.json")
        link("Native viewer settings",data["native_dir"]/"viewer_config.json")
        link("Segmentation provenance",data["segmentation_dir"]/"manifest.json")
        link("Segmentation settings",data["segmentation_dir"]/"export_config.json")
        run = Path(data["native"].get("run_dir",""))
        for title,relative in (("Masks and queries","semantics/frames.jsonl"),("Mask files","semantics/masks"),
                               ("Saved voxel data","map/voxels.npz"),("Original run","run.json")):
            link(title,run/relative)
        link("Source sequence",Path(data["native"].get("sequence_path","")))
        for path_text in data["native"].get("input_sha256",{}):
            path = Path(path_text)
            if path.name=="research_plan.json":
                link("Separate research plan",path);link("Research assumptions",path.parent/"assumptions.json")
        for kind,title in PLAYERS:
            mp4 = encode_and_probe(binary,probe,output,kind,data["timeline"],lifetime)
            for path in (mp4,output/f"{kind}.ffprobe.json",output/f"{kind}.encoder.log",
                         output/f"{kind}.probe.json",output/f"{kind}.frames.txt"):
                record_output(path)
            first,last = thumbnail(output/f"{kind}_frames/000000.png"),thumbnail(output/f"{kind}_frames/{len(data['native_pixels'])-1:06d}.png")
            for suffix,payload in (("first",first),("last",last)):
                path = output/f"{kind}_{suffix}.jpg";path.write_bytes(payload);record_output(path)
            player_links = [("This MP4",f"{kind}.mp4"),("Export provenance","manifest.json"),("Export settings","export_config.json")]+links
            page = output/f"{kind}.html"
            page.write_text(player_html(label.strip(),title,mp4.read_bytes(),first,last,data,player_links),encoding="utf-8")
            record_output(page)
            result["players"][kind] = {"html":page.name,"mp4":mp4.name,"first_thumbnail":f"{kind}_first.jpg","last_thumbnail":f"{kind}_last.jpg"}
            save_manifest()
        for path_text,digest in data["input_sha256"].items():
            if sha256(path_text)!=digest:
                raise RuntimeError("Saved source changed during packaging")
        result["status"] = "complete";save_manifest()
        (output/"manifest.sha256").write_text(sha256(output/"manifest.json")+"  manifest.json\n",encoding="ascii")
    except BaseException as error:
        result.update(status="failed",error=str(error));save_manifest()
        raise
    finally:
        try:
            lifetime.__exit__()
        except BaseException as error:
            result.update(status="failed",error="Encoder cleanup failed: "+str(error))
            raise
        finally:
            result["encoder_lifecycle"] = lifetime.calls
            save_manifest()
            if result["status"]=="complete":
                (output/"manifest.sha256").write_text(sha256(output/"manifest.json")+"  manifest.json\n",encoding="ascii")
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv==["--self-test-lifecycle"]:
        print(json.dumps(lifecycle_self_test()));return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native",required=True,type=Path,help="Completed native viewer output after clean shutdown")
    parser.add_argument("--native-manifest-sha256",required=True,help="Reviewed final native manifest SHA256")
    parser.add_argument("--segmentation-export",required=True,type=Path,help="Completed source-frame segmentation export")
    parser.add_argument("--output",required=True,type=Path,help="Fresh separate CPU preview directory")
    parser.add_argument("--label",required=True)
    parser.add_argument("--max-runtime",type=int,default=600,help="Bounded CPU export lifetime, at most900seconds")
    args = parser.parse_args(argv)
    try:
        data = load_preview(args.native,args.segmentation_export,args.native_manifest_sha256)
        result = export_preview(data,args.output,args.label,max_seconds=args.max_runtime)
        print(json.dumps({"status":result["status"],"output":str(args.output.resolve()),"players":list(result["players"])}))
    except Exception as error:
        print("Native preview export stopped: "+str(error),file=sys.stderr)
        return 1
    return 0


if __name__=="__main__":
    raise SystemExit(main())
