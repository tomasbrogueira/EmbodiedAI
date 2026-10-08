"""CPU-only segmentation movies from a complete saved sampled recording.

Exact saved source RGB, mask grids and source transforms are validated by the
common read-only loader. No model, mask inference, geometry or map renderer runs.
The explicit recording admission is1–58 samples, never every source video frame.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time

if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for variable in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "4"

from PIL import ImageDraw

from export_pipeline_video import (
    MAX_GEOMETRY_ARCHIVE_BYTES, _text, load_export_inputs, sampled_timeline, sha256,
    render_segmentation,
)

ENCODER_SHA256 = "d546e85a5c42368fe4fa0f577b1c98070293eee187ce695e80a28cb0a0cd2f78"
MAX_FRAMES = 58
MAX_MP4_BYTES = 32*1024*1024
IDENTITY_KEYS = ("geometry_fingerprint","input_fingerprint","processed_grid_id","archive_sha256",
                 "map_frame","units","scale","up")
MASK_PROJECTION = "saved_source_to_processed_center_affine_nearest_no_extrapolation"


def write_json(path,value):
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    os.replace(temporary,path)


def load_recording(run,sequence,cache,research_plan,expected_frames):
    if type(expected_frames) is not int or not 1<=expected_frames<=MAX_FRAMES:
        raise ValueError("Expected sampled frame count must be an explicit integer within1–58")
    data = load_export_inputs(run,sequence,cache_dir=cache,research_plan_path=research_plan)
    if len(data["frames"])!=expected_frames:
        raise ValueError("Saved complete sequence differs from the exact expected sample count; no truncation")
    if data["run"].get("sequence",{}).get("frame_count")!=expected_frames:
        raise ValueError("Saved run sequence count differs from the exact expected sample count")
    data["timeline"] = sampled_timeline(data["frames"],video_fps=20,end_hold_seconds=2)
    data["expected_sampled_frames"] = expected_frames
    return data


def fresh_output(path,data,research_plan=None):
    output = Path(path).resolve()
    roots = [data["run_dir"],data["sequence_path"].parent,data["cache"]]
    if research_plan:
        roots.append(Path(research_plan).resolve().parent)
    if output.exists() or any(output.is_relative_to(root) for root in roots):
        raise ValueError("Segmentation output must be fresh and separate from every saved input")
    return output


def saved_research(data):
    return {key:value for key,value in (data["research_plan"] or {}).items() if key!="validated_path"}


def frame_coverage(frame,repeats):
    semantic = frame["semantic"] or {}
    return {"frame_id":frame["frame_id"],"timestamp_ns":frame["timestamp_ns"],
            "timestamp_provenance":frame["source"].get("timestamp_provenance",{}),
            "disposition":frame["disposition"],"semantic_status":semantic.get("status","not_recorded"),
            "queries":[{"query_id":query.get("query_id"),"prompt":query.get("original_phrase"),
                        "status":query.get("status"),"error":query.get("error"),"instances":len(query.get("instances",[]))}
                       for query in semantic.get("queries",[])],"encoded_repeat_count":repeats,
            "research_path_drawn":False}


def render_frame(data,frame,index):
    image = render_segmentation(data,frame,index,width=800,height=800,rgb_view="source")
    # The source RGB/mask image begins below110px; the reserved header contains
    # captions only. Saved status/empty-result labels and mask pixels stay intact.
    draw = ImageDraw.Draw(image)
    draw.rectangle((0,0,800,47),fill=(19,26,36))
    _text(draw,(24,18),f"Video masks — saved evidence | {index+1}/{len(data['frames'])}",size=23,width=60)
    if not data["run"].get("fixture"):
        research = data["research_plan"] or {}
        message = ("Research illustration: no path found; no route drawn." if research.get("status")=="no_path" else
                   "Research illustration: blocked inputs; no route drawn." if research.get("status")=="blocked_inputs" else
                   "Research route is separate; segmentation does not guide it." if research else
                   "Candidate mask evidence; original planning remains as recorded.")
        _text(draw,(24,85),message,size=14,width=95)
    return image


def make_encoder(stop):
    path = Path(__file__).with_name("serve_lingbot_view.py").resolve()
    if sha256(path)!=ENCODER_SHA256:
        raise ValueError("OwnedEncoder helper differs from the reviewed frozen SHA256")
    module = importlib.import_module("serve_lingbot_view")
    if Path(module.__file__).resolve()!=path:
        raise ValueError("OwnedEncoder was imported outside the frozen source path")
    return module.OwnedEncoder(stop)


def encode_and_probe(data,output,encoder,stop,deadline,binary,probe,lifecycle):
    listing = output/"segmentation.mp4.frames.txt"
    listing.write_text("".join(f"file 'segmentation_frames/{index:06d}.png'\n"*count
                              for index,count in enumerate(data["timeline"]["repeat_counts"])),encoding="utf-8")
    def run(command,log,timeout):
        record = {"command":command,"log_path":str(log),"returncode":None}
        try:
            if stop.is_set() or time.monotonic()>=deadline:
                raise RuntimeError("Segmentation export interrupted or exceeded its lifetime")
            record["returncode"] = encoder.run(command,log,min(deadline,time.monotonic()+timeout))
            return record["returncode"]
        finally:
            try:
                record["cleanup"] = encoder.cancel()
            finally:
                lifecycle.append(record)
    target = output/"segmentation.mp4"
    code = run([binary,"-hide_banner","-loglevel","error","-n","-threads","4","-r","20",
                "-f","concat","-safe","1","-i",str(listing),"-an","-c:v","libx264","-threads","4",
                "-crf","20","-pix_fmt","yuv420p","-movflags","+faststart",str(target)],output/"segmentation.encoder.log",180)
    if code or not target.is_file() or not 0<target.stat().st_size<=MAX_MP4_BYTES:
        raise RuntimeError("Segmentation MP4 encoding failed or exceeded32MiB; see saved encoder log")
    report_path = output/"segmentation.ffprobe.json"
    code = run([probe,"-v","error","-count_frames","-select_streams","v:0","-show_streams","-show_format",
                "-threads","4","-of","json",str(target)],report_path,60)
    if code or report_path.stat().st_size>16*1024*1024:
        raise RuntimeError("Segmentation MP4 probe failed")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    streams = report.get("streams",[])
    count = sum(data["timeline"]["repeat_counts"])
    if len(streams)!=1:
        raise RuntimeError("Expected exactly one segmentation video stream")
    stream = streams[0]
    if (stream.get("codec_name")!="h264" or stream.get("pix_fmt")!="yuv420p"
            or stream.get("width")!=800 or stream.get("height")!=800 or stream.get("avg_frame_rate")!="20/1"
            or int(stream.get("nb_read_frames",-1))!=count
            or abs(float(report.get("format",{}).get("duration",-1))-count/20)>.051):
        raise RuntimeError("Encoded segmentation frame count, dimensions, duration or rate differs")
    return (target,listing,report_path,output/"segmentation.encoder.log")


def export_recording(data,output,*,research_plan=None,max_runtime=600,stage_deadline=None):
    if type(max_runtime) is not int or not 1<=max_runtime<=900:
        raise ValueError("CPU export lifetime must be within1–900seconds")
    output = fresh_output(output,data,research_plan)
    if os.name!="posix":
        raise RuntimeError("Encoding requires the reviewed Linux OwnedEncoder process-group runtime")
    binary,probe = shutil.which("ffmpeg"),shutil.which("ffprobe")
    if not binary or not probe:
        raise RuntimeError("Existing ffmpeg and ffprobe are required; no installation is performed")
    stop = threading.Event()
    encoder = make_encoder(stop)
    deadline = time.monotonic()+max_runtime if stage_deadline is None else stage_deadline
    if time.monotonic()>=deadline:
        raise RuntimeError("Saved-input admission exceeded the whole CPU-stage lifetime")
    config = {"schema_version":1,"artifact_kind":"saved_segmentation_recording_export","read_only_inputs":True,
              "helper_sha256":sha256(__file__),"owned_encoder_adapter_sha256":ENCODER_SHA256,
              "saved_input_validator_sha256":sha256(Path(__file__).with_name("export_pipeline_video.py")),
              "timeline":data["timeline"],"expected_sampled_frames":data["expected_sampled_frames"],
              "maximum_sampled_frames":MAX_FRAMES,"source_sampling_scope":"complete_saved_samples_not_every_source_frame",
              "panel_size":800,"cpu_threads":4,"max_runtime_seconds":max_runtime,
              "lifetime_includes_input_admission":stage_deadline is not None,"frames_only":False,"rgb_view":"source",
              "mask_projection":MASK_PROJECTION,"encoder":binary,"probe":probe,"fixture":data["run"].get("fixture"),
              "max_mp4_bytes":MAX_MP4_BYTES,"geometry_archive_read_cap_bytes":MAX_GEOMETRY_ARCHIVE_BYTES,
              "geometry_renderer":None,"model_inference":False,"mask_inference":False,
              "path_scope":"no_path_rendered_in_segmentation_panel"}
    manifest = {"schema_version":1,"status":"running","run_dir":str(data["run_dir"]),"sequence_path":str(data["sequence_path"]),
                "helper_sha256":sha256(__file__),"input_sha256":data["input_sha256"],
                "geometry_identity":{key:data["geometry"].get(key) for key in IDENTITY_KEYS},
                "expected_sampled_frames":data["expected_sampled_frames"],"sampled_frame_count":len(data["frames"]),
                "encoded_frame_count":sum(data["timeline"]["repeat_counts"]),"original_planning":data["planning"],
                "research_plan":saved_research(data),"coverage":[],"outputs_sha256":{},"encoder_lifecycle":[],
                "limitations":["Only saved sampled frames are displayed; holds contain no new mask or geometry inference.",
                               "Masks are projected onto source RGB through the exact saved grid transform; outside the crop is unknown.",
                               "Ground/floor masks are candidate evidence, not terrain safety or traversability.",
                               "Research planning is separate and assumption-labelled; this segmentation movie draws no route."]}
    output.mkdir(parents=True,exist_ok=False)
    def record(path):
        manifest["outputs_sha256"][str(path.relative_to(output))] = sha256(path)
    def save():
        write_json(output/"manifest.json",manifest)
    handlers = {}
    try:
        for number in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
            handlers[number] = signal.signal(number,lambda *_:stop.set())
        write_json(output/"export_config.json",config);record(output/"export_config.json");save()
        (output/"segmentation_frames").mkdir()
        for index,(frame,repeats) in enumerate(zip(data["frames"],data["timeline"]["repeat_counts"])):
            if stop.is_set() or time.monotonic()>=deadline:
                raise RuntimeError("Saved-frame export interrupted or exceeded its lifetime")
            path = output/f"segmentation_frames/{index:06d}.png"
            render_frame(data,frame,index).save(path);record(path)
            manifest["coverage"].append(frame_coverage(frame,repeats));save()
        for path in encode_and_probe(data,output,encoder,stop,deadline,binary,probe,manifest["encoder_lifecycle"]):
            record(path)
        for path_text,digest in data["input_sha256"].items():
            if stop.is_set() or time.monotonic()>=deadline:
                raise RuntimeError("Source verification interrupted or exceeded the whole CPU-stage lifetime")
            if sha256(path_text)!=digest:
                raise RuntimeError("Saved source changed during segmentation export")
        if stop.is_set() or time.monotonic()>=deadline:
            raise RuntimeError("Segmentation export interrupted or exceeded the whole CPU-stage lifetime")
        manifest["query_status_counts"] = dict(Counter(query.get("status","unknown") for frame in manifest["coverage"] for query in frame["queries"]))
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="failed",error=str(error))
        raise
    finally:
        stop.set()
        try:
            manifest["encoder_cleanup"] = encoder.cancel()
        except BaseException as error:
            manifest.update(status="failed",error="Owned encoder cleanup failed: "+str(error))
            raise
        finally:
            for number,handler in handlers.items():
                signal.signal(number,handler)
            save()
            if manifest["status"]=="complete":
                (output/"manifest.sha256").write_text(sha256(output/"manifest.json")+"  manifest.json\n",encoding="ascii")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True,type=Path)
    parser.add_argument("--sequence",required=True,type=Path)
    parser.add_argument("--cache",type=Path,help="Exact relocated saved geometry cache")
    parser.add_argument("--research-plan",type=Path,help="Separate saved assumption-labelled research result")
    parser.add_argument("--output",required=True,type=Path,help="Fresh separate segmentation output")
    parser.add_argument("--expected-frames",required=True,type=int,help="Exact complete saved sample count1–58")
    parser.add_argument("--max-runtime",type=int,default=600,help="Bounded CPU lifetime, at most900seconds")
    args = parser.parse_args(argv)
    if not 1<=args.max_runtime<=900:
        parser.error("Whole CPU-stage lifetime must be within1–900seconds")
    stage_deadline = time.monotonic()+args.max_runtime
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    try:
        data = load_recording(args.run,args.sequence,args.cache,args.research_plan,args.expected_frames)
        result = export_recording(data,args.output,research_plan=args.research_plan,max_runtime=args.max_runtime,
                                  stage_deadline=stage_deadline)
        print(json.dumps({"status":result["status"],"sampled_frames":result["sampled_frame_count"],"output":str(args.output.resolve())}))
    except Exception as error:
        print("Saved segmentation recording stopped: "+str(error),file=sys.stderr)
        return 1
    return 0


if __name__=="__main__":
    raise SystemExit(main())
