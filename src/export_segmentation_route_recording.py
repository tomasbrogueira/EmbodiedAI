"""CPU-only mask movies with the saved research route projected into RGB pixels.

Exact saved source RGB, mask grids and source transforms are validated by the
common read-only loader. No model, mask inference, geometry or map renderer runs.
The static route is a final-map illustration. An optional separately admitted
navigation replay supplies a short, replanned native route for each saved view.
The explicit recording admission is1–58 samples, never every source video frame.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import json
import math
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

import numpy as np
from PIL import Image, ImageDraw

from export_pipeline_video import (
    MAX_GEOMETRY_ARCHIVE_BYTES, _text, load_export_inputs, sampled_timeline, sha256,
    render_segmentation, load_npz_checked,
)
from pipeline_common.route_projection import project_route_to_source

ENCODER_SHA256 = "d546e85a5c42368fe4fa0f577b1c98070293eee187ce695e80a28cb0a0cd2f78"
MAX_FRAMES = 58
MAX_MP4_BYTES = 32*1024*1024
IDENTITY_KEYS = ("geometry_fingerprint","input_fingerprint","processed_grid_id","archive_sha256",
                 "map_frame","units","scale","up")
MASK_PROJECTION = "saved_source_to_processed_center_affine_nearest_no_extrapolation"
BASE_HELPER_SHA256 = "d084e18f9546316788a78393580b6f3916929a3f442ede75f4ab5b2a0c86d8c7"
PATH_SCOPE = "saved_research_route_projected_to_source_rgb"
PROJECTION_METHOD = "saved_opencv_world_to_camera_intrinsics_inverse_pixel_center_affine_depth_checked_v1"
ROUTE_COLOR = (214, 118, 255)
REPLAY_SCOPE = "observation_prefix_map_replay"
REPLAY_PATH_SCOPE = "per_frame_replanned_next_steps_projected_to_source_rgb"
MAX_REPLAY_JSON_BYTES = 16*1024*1024
REPLAY_IDENTITY_KEYS = ("geometry_fingerprint", "processed_grid_id", "input_fingerprint", "map_frame", "units")


def _native_path(value, name):
    array = np.asarray(value)
    if array.size == 0:
        if array.shape not in ((0,),(0,3)):
            raise ValueError(name+" has an invalid empty XYZ shape")
        return np.empty((0,3),float)
    if (array.ndim!=2 or array.shape[1:]!=(3,) or len(array)>4096
            or array.dtype.kind not in "iuf" or not np.isfinite(array).all()):
        raise ValueError(name+" must be at most4096 finite native XYZ points")
    return array.astype(float,copy=False)


def _path_length(path):
    return float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum()) if len(path)>1 else 0.


def _display_prefix(path,display):
    """A horizon may interpolate its last edge, but may not invent a shortcut."""
    if len(display)<2 or len(display)>len(path) or not np.allclose(display[:-1],path[:len(display)-1],rtol=0,atol=1e-8):
        return False
    start,end=path[len(display)-2:len(display)]
    delta=end-start; denominator=float(delta@delta)
    if denominator<=1e-18:
        return bool(np.allclose(display[-1],end,rtol=0,atol=1e-8))
    fraction=float((display[-1]-start)@delta/denominator)
    return (-1e-8<=fraction<=1+1e-8 and
            np.allclose(display[-1],start+np.clip(fraction,0,1)*delta,rtol=0,atol=1e-8))


def load_replanning_replay(data,path,*,research_plan_path=None):
    """Independently admit prefix-map replay; leave the static-plan gate intact."""
    path=Path(path).resolve()
    if path.stat().st_size>MAX_REPLAY_JSON_BYTES:
        raise ValueError("Navigation replay exceeds its bounded16MiB JSON admission")
    with path.open("rb") as stream:
        contents=stream.read(MAX_REPLAY_JSON_BYTES+1)
    if len(contents)>MAX_REPLAY_JSON_BYTES:
        raise ValueError("Navigation replay exceeds its bounded16MiB JSON admission")
    initial=hashlib.sha256(contents).hexdigest()
    def nonfinite(value):
        raise ValueError("Navigation replay contains nonfinite JSON "+value)
    replay=json.loads(contents,parse_constant=nonfinite)
    try:
        json.dumps(replay,allow_nan=False)
    except (ValueError,TypeError) as error:
        raise ValueError("Navigation replay must contain finite JSON values") from error
    if (not isinstance(replay,dict) or type(replay.get("schema_version")) is not int or replay["schema_version"]!=1
            or replay.get("artifact_kind")!="research_navigation_replay" or replay.get("research_illustration") is not True
            or replay.get("perception_scope")!="cached_geometry_replay" or replay.get("frame_scope")!=REPLAY_SCOPE):
        raise ValueError("Explicit schema1 research navigation cached-geometry prefix replay required")
    source=replay.get("source")
    plan=data.get("research_plan") or {}
    if not isinstance(source,dict) or not plan:
        raise ValueError("Navigation replay requires separately validated research assumptions and source identities")
    for key in REPLAY_IDENTITY_KEYS:
        if source.get(key)!=data["geometry"].get(key) or source.get(key)!=plan.get("source",{}).get(key):
            raise ValueError("Navigation replay source differs: "+key)
    if "archive_sha256" in source and (source["archive_sha256"]!=data["geometry"].get("archive_sha256")
            or source["archive_sha256"]!=plan.get("source",{}).get("archive_sha256")):
        raise ValueError("Navigation replay source differs: archive_sha256")
    if research_plan_path is not None and source.get("research_plan_sha256")!=sha256(research_plan_path):
        raise ValueError("Navigation replay research plan hash differs")
    original_hashes=source.get("original_input_sha256")
    if original_hashes is not None:
        if not isinstance(original_hashes,dict) or not original_hashes:
            raise ValueError("Navigation replay original input hashes must be a nonempty mapping")
        for saved_path,digest in original_hashes.items():
            if data["input_sha256"].get(saved_path)!=digest:
                raise ValueError("Navigation replay original input hash differs: "+saved_path)
    mission=replay.get("mission")
    if (not isinstance(mission,dict) or not isinstance(mission.get("id"),str) or not mission["id"]
            or mission.get("policy")!="persistent_forward_corridor" or mission.get("state")!="active"
            or mission.get("completed") is not False):
        raise ValueError("Navigation replay must retain an active uncompleted forward mission")
    horizon=replay.get("display_horizon_assumed_m")
    factor=plan.get("assumptions",{}).get("metres_per_native_unit")
    if any(isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0
           for value in (horizon,factor)) or horizon>2:
        raise ValueError("Navigation replay horizon must be at most2 assumed metres with a finite positive research scale")
    frames=replay.get("frames")
    if not isinstance(frames,list) or len(frames)!=len(data["frames"]):
        raise ValueError("Navigation replay must cover the exact complete saved sampled sequence")
    for index,(entry,frame) in enumerate(zip(frames,data["frames"])):
        if (not isinstance(entry,dict) or entry.get("frame_id")!=frame["frame_id"]
                or "timestamp_ns" not in entry or entry["timestamp_ns"]!=frame["timestamp_ns"]
                or (entry["timestamp_ns"] is not None and type(entry["timestamp_ns"]) is not int)
                or type(entry.get("observed_through_index")) is not int or entry["observed_through_index"]!=index):
            raise ValueError("Navigation replay frame order, timestamp or observation cutoff differs")
        full=_native_path(entry.get("path_points"),"Replay full path")
        display=_native_path(entry.get("display_path_points"),"Replay display path")
        if entry.get("status")=="ok":
            if not _display_prefix(full,display):
                raise ValueError("Navigation display path must be an ordered prefix of its full route")
        elif entry.get("status") in {"awaiting_support","awaiting_observation"}:
            if len(full) or len(display):
                raise ValueError("Awaiting support cannot display an invented route")
        else:
            raise ValueError("Navigation replay frame status must be ok, awaiting_support or awaiting_observation")
        if _path_length(display)*factor>horizon+1e-7:
            raise ValueError("Navigation replay display path exceeds its short assumed-metre horizon")
        if not isinstance(entry.get("navigation_state"),(str,dict)) or not isinstance(entry.get("diagnostics"),dict):
            raise ValueError("Navigation replay requires explicit navigation state and diagnostics")
        navigation=entry["navigation_state"]
        if isinstance(navigation,dict) and (navigation.get("mission_state")!="active"
                or navigation.get("mission_completed") is not False):
            raise ValueError("Each navigation observation must retain the active uncompleted mission")
    terminal=replay.get("terminal_state")
    if (not isinstance(terminal,dict) or terminal.get("mission_state")!="active"
            or terminal.get("execution_state")!="awaiting_observations" or terminal.get("mission_completed") is not False
            or terminal.get("reason")!="recording_ended"):
        raise ValueError("Recording EOF must retain an active mission awaiting observations")
    retained=_native_path(terminal.get("retained_display_path_points"),"Retained terminal path")
    if _path_length(retained)*factor>horizon+1e-7:
        raise ValueError("Retained terminal path exceeds the admitted display horizon")
    last_display=_native_path(frames[-1]["display_path_points"],"Last admitted display path")
    if retained.shape!=last_display.shape or not np.allclose(retained,last_display,rtol=0,atol=1e-8):
        raise ValueError("Recording EOF must retain exactly the last admitted next steps, without a fabricated replan")
    if sha256(path)!=initial:
        raise ValueError("Navigation replay changed during admission")
    data["input_sha256"][str(path)]=initial
    data["replanning_replay"]=replay
    data["replanning_replay_path"]=path
    return replay


def source_image_rectangle(source_shape, width=800, height=800):
    """Exact fitted image dimensions used by the unchanged mask renderer."""
    source_height, source_width = source_shape
    factor = min((width-48)/source_width, (height-300)/source_height)
    image_width = max(1, round(source_width*factor))
    image_height = max(1, round(source_height*factor))
    return ((width-image_width)//2, 110+(height-300-image_height)//2,
            image_width, image_height)


def map_source_pixels(points, source_shape, rectangle):
    """Pillow resize maps pixel centres, including the half-pixel offset."""
    values = np.asarray(points, dtype=float)
    if values.shape[-1:] != (2,) or not np.isfinite(values).all():
        raise ValueError("Projected source pixels must be finite pairs")
    source_height, source_width = source_shape
    left, top, width, height = rectangle
    return (values+.5)*np.array([width/source_width, height/source_height])-.5+np.array([left, top])


def prepare_route_projection(data):
    """Load exact same-frame camera/depth evidence; never change the saved plan."""
    base = Path(__file__).with_name("export_segmentation_recording.py")
    if sha256(base) != BASE_HELPER_SHA256:
        raise ValueError("Reviewed mask-only helper differs from its frozen source hash")
    plan = data["research_plan"] or {}
    path = np.asarray(plan.get("validated_path", np.empty((0, 3))), dtype=float)
    replay=data.get("replanning_replay")
    if not replay and len(path) < 2:
        data["route_projections"] = [
            {"source_segments":np.empty((0,2,2)), "source_arrowheads":np.empty((0,3,2)),
             "report":{"status":"no_route", "visible_segments":0, "arrow_count":0,
                       "research_status":plan.get("status", "not_recorded")}}
            for _ in data["frames"]]
        return
    archive = data["cache"] / "geometry.npz"
    if sha256(archive) != data["geometry"]["archive_sha256"]:
        raise ValueError("Projection geometry differs from the mask renderer source")
    arrays = load_npz_checked(archive, max_bytes=MAX_GEOMETRY_ARCHIVE_BYTES,
                             names=("intrinsic", "extrinsic", "depth", "world_points_conf"))
    count = len(data["frames"])
    keys = ("intrinsic", "extrinsic", "depth", "world_points_conf")
    if any(key not in arrays for key in keys):
        raise ValueError("Exact saved camera calibration and registered depth are required")
    k, poses, depth, confidence = (arrays[key] for key in keys)
    shape = (count, *data["frames"][0]["rgb"].shape[:2])
    if depth.shape == (*shape,1):
        depth = depth[...,0]
    if (k.shape != (count,3,3) or poses.shape != (count,3,4)
            or depth.shape != shape or confidence.shape != shape
            or not np.isfinite(k).all() or not np.isfinite(poses).all()):
        raise ValueError("Projection calibration/depth differs from the exact saved processed grid")
    minimum = data["geometry"].get("settings",{}).get("min_confidence")
    voxel_size = float(data["map"]["voxel_size"])
    if (not isinstance(minimum,(float,int)) or isinstance(minimum,bool)
            or not math.isfinite(minimum) or minimum < 0
            or not math.isfinite(voxel_size) or voxel_size <= 0):
        raise ValueError("Saved projection confidence or voxel resolution is invalid")
    result = []
    for index, frame in enumerate(data["frames"]):
        navigation=replay["frames"][index] if replay else None
        frame_path=_native_path(navigation["display_path_points"],"Replay display path") if navigation else path
        rectangle = source_image_rectangle(frame["source_rgb"].shape[:2])
        display_scale = min(rectangle[2]/frame["source_rgb"].shape[1],
                            rectangle[3]/frame["source_rgb"].shape[0])
        pose = np.eye(4)
        pose[:3] = poses[index]
        valid = np.isfinite(confidence[index]) & (confidence[index] >= minimum)
        valid &= np.isfinite(depth[index]) & (depth[index] > 0)
        projected = project_route_to_source(frame_path,k[index],pose,frame["transform"],
            depth=depth[index],depth_valid=valid,
            depth_absolute_tolerance=math.sqrt(3)*voxel_size/2,
            depth_relative_tolerance=.03, sample_spacing_pixels=2.0, max_samples=20000,
            arrowhead_length_pixels=16/display_scale, arrow_spacing_pixels=90/display_scale)
        projected["report"].update({"method":PROJECTION_METHOD,
            "source_image_rectangle":list(rectangle), "depth_min_confidence":minimum,
            "depth_absolute_tolerance_native_units":math.sqrt(3)*voxel_size/2,
            "depth_relative_tolerance":.03, "route_direction":"saved_start_to_goal",
            "route_scope":REPLAY_SCOPE if replay else "final_cumulative_map_posthoc", "source_coordinate_units":"pixel_centres"})
        if navigation:
            projected["report"].update({"planner_recomputed_route":navigation["status"]=="ok",
                "mission_id":replay["mission"]["id"],"mission_state":"active","mission_completed":False,
                "navigation_state":navigation["navigation_state"],"navigation_status":navigation["status"],
                "observed_through_index":index,"display_horizon_assumed_m":replay["display_horizon_assumed_m"],
                "display_path_length_assumed_m":_path_length(frame_path)*plan["assumptions"]["metres_per_native_unit"],
                "perception_scope":"cached_geometry_replay"})
        result.append(projected)
    if sha256(archive) != data["geometry"]["archive_sha256"]:
        raise ValueError("Projection geometry changed during admission")
    data["route_projections"] = result


def draw_projected_route(image, projection, source_shape):
    """Keep graphics inside the actual RGB rectangle, away from captions."""
    rectangle = source_image_rectangle(source_shape)
    left, top, width, height = rectangle
    layer = Image.new("RGBA", (width,height))
    draw = ImageDraw.Draw(layer)
    def coordinates(values):
        return map_source_pixels(values,source_shape,rectangle)-np.array([left,top])
    segments = coordinates(projection["source_segments"])
    for segment in segments:
        draw.line([tuple(point) for point in segment],fill=(25,15,34,230),width=7)
    for segment in segments:
        draw.line([tuple(point) for point in segment],fill=(*ROUTE_COLOR,255),width=4)
    for triangle in coordinates(projection["source_arrowheads"]):
        points = [tuple(point) for point in triangle]
        draw.polygon(points,fill=(*ROUTE_COLOR,255))
        draw.line(points+[points[0]],fill=(255,244,255,255),width=2)
    image.paste(layer,(left,top),layer)
    return image


def write_json(path,value):
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    os.replace(temporary,path)


def load_recording(run,sequence,cache,research_plan,expected_frames,*,replanning_replay=None):
    if type(expected_frames) is not int or not 1<=expected_frames<=MAX_FRAMES:
        raise ValueError("Expected sampled frame count must be an explicit integer within1–58")
    data = load_export_inputs(run,sequence,cache_dir=cache,research_plan_path=research_plan)
    if len(data["frames"])!=expected_frames:
        raise ValueError("Saved complete sequence differs from the exact expected sample count; no truncation")
    if data["run"].get("sequence",{}).get("frame_count")!=expected_frames:
        raise ValueError("Saved run sequence count differs from the exact expected sample count")
    data["timeline"] = sampled_timeline(data["frames"],video_fps=20,end_hold_seconds=2)
    data["expected_sampled_frames"] = expected_frames
    if replanning_replay is not None:
        load_replanning_replay(data,replanning_replay,research_plan_path=research_plan)
    prepare_route_projection(data)
    return data


def fresh_output(path,data,research_plan=None):
    output = Path(path).resolve()
    roots = [data["run_dir"],data["sequence_path"].parent,data["cache"]]
    if research_plan:
        roots.append(Path(research_plan).resolve().parent)
    if data.get("replanning_replay_path"):
        roots.append(data["replanning_replay_path"].parent)
    if output.exists() or any(output.is_relative_to(root) for root in roots):
        raise ValueError("Segmentation output must be fresh and separate from every saved input")
    return output


def saved_research(data):
    return {key:value for key,value in (data["research_plan"] or {}).items() if key!="validated_path"}


def frame_coverage(frame,repeats,projection):
    semantic = frame["semantic"] or {}
    return {"frame_id":frame["frame_id"],"timestamp_ns":frame["timestamp_ns"],
            "timestamp_provenance":frame["source"].get("timestamp_provenance",{}),
            "disposition":frame["disposition"],"semantic_status":semantic.get("status","not_recorded"),
            "queries":[{"query_id":query.get("query_id"),"prompt":query.get("original_phrase"),
                        "status":query.get("status"),"error":query.get("error"),"instances":len(query.get("instances",[]))}
                       for query in semantic.get("queries",[])],"encoded_repeat_count":repeats,
            "research_path_drawn":bool(len(projection["source_segments"])),
            "route_projection":projection["report"]}


def render_frame(data,frame,index):
    image = render_segmentation(data,frame,index,width=800,height=800,rgb_view="source")
    projection = data["route_projections"][index]
    draw_projected_route(image,projection,frame["source_rgb"].shape[:2])
    # The source RGB/mask image begins below110px; the reserved header contains
    # captions only. Saved status/empty-result labels and mask pixels stay intact.
    draw = ImageDraw.Draw(image)
    draw.rectangle((0,0,800,47),fill=(19,26,36))
    replay=data.get("replanning_replay")
    title="Masks + next steps" if replay else "Masks + planned direction"
    _text(draw,(24,18),f"{title} | {index+1}/{len(data['frames'])}",size=23,width=60)
    if not data["run"].get("fixture"):
        research = data["research_plan"] or {}
        message = ("Awaiting new observations; mission remains active." if replay and projection["report"].get("navigation_status")=="awaiting_observation" else
                   "Awaiting observed support; mission remains active." if replay and projection["report"].get("navigation_status")=="awaiting_support" else
                   "Purple: replanned next steps from this view; mission active." if replay and projection["report"]["status"]=="visible" else
                   "Next steps hidden here by the view/depth; mission remains active." if replay else
                   "Research illustration: no path found; no route drawn." if research.get("status")=="no_path" else
                   "Research illustration: blocked inputs; no route drawn." if research.get("status")=="blocked_inputs" else
                   "Purple arrows: saved route start to goal (posthoc illustration)." if projection["report"]["arrow_count"]>0 else
                   "Purple line: visible route section; no direction arrow here." if projection["report"]["status"]=="visible" else
                   "Route not shown here: outside view or uncertain/hidden depth." if research else
                   "Candidate mask evidence; original planning remains as recorded.")
        _text(draw,(24,85),message,size=14,width=95)
    draw.rectangle((0,734,800,799),fill=(19,26,36))
    _text(draw,(24,740),"Purple: planned direction | Teal: floor mask | Orange: hazard mask",size=15,width=95)
    footer=("Recording paused / mission active - awaiting observations." if replay and index==len(data["frames"])-1 else
            "Prefix-map research replay; only the next 2 assumed metres shown." if replay else
            "Saved camera projection; final-map research route, not live guidance.")
    _text(draw,(24,772),footer,size=13,width=95)
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
    replay=data.get("replanning_replay")
    route_scope=REPLAY_SCOPE if replay else "final_cumulative_map_posthoc"
    path_scope=REPLAY_PATH_SCOPE if replay else PATH_SCOPE
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
              "path_scope":path_scope,"route_projection_method":PROJECTION_METHOD,
              "route_projection_helper_sha256":sha256(Path(__file__).parent/"pipeline_common/route_projection.py"),
              "frozen_base_helper_sha256":BASE_HELPER_SHA256,
              "route_direction":"saved_start_to_goal", "route_scope":route_scope}
    if replay:
        config.update(perception_scope="cached_geometry_replay",replanning_replay_sha256=data["input_sha256"][str(data["replanning_replay_path"])],
                      display_horizon_assumed_m=replay["display_horizon_assumed_m"],mission=replay["mission"],
                      terminal_state=replay["terminal_state"])
    manifest = {"schema_version":1,"status":"running","run_dir":str(data["run_dir"]),"sequence_path":str(data["sequence_path"]),
                "helper_sha256":sha256(__file__),"input_sha256":data["input_sha256"],
                "geometry_identity":{key:data["geometry"].get(key) for key in IDENTITY_KEYS},
                "expected_sampled_frames":data["expected_sampled_frames"],"sampled_frame_count":len(data["frames"]),
                "encoded_frame_count":sum(data["timeline"]["repeat_counts"]),"original_planning":data["planning"],
                "research_plan":saved_research(data),"coverage":[],"outputs_sha256":{},"encoder_lifecycle":[],
                "limitations":["Only saved sampled frames are displayed; holds contain no new mask or geometry inference.",
                               "Masks are projected onto source RGB through the exact saved grid transform; outside the crop is unknown.",
                               "Ground/floor masks are candidate evidence, not terrain safety or traversability.",
                               "Purple arrows project the unchanged saved final-map research route, ordered from start to goal.",
                               "Only depth-compatible visible parts inside the saved model crop are drawn; missing depth is excluded.",
                               "Visibility uses saved reconstruction depth with half-voxel-diagonal plus3% tolerance, not verified physical visibility.",
                               "This is a posthoc research illustration, not live robot guidance or independently validated route safety."]}
    manifest["path_scope"] = path_scope
    manifest["route_projection_method"] = PROJECTION_METHOD
    manifest["route_scope"] = route_scope
    if replay:
        manifest.update(mission=replay["mission"],terminal_state=replay["terminal_state"],
                        perception_scope="cached_geometry_replay",display_horizon_assumed_m=replay["display_horizon_assumed_m"])
        manifest["limitations"][3]="Purple arrows display each saved prefix-map replan's short horizon, not the full mission route."
        manifest["limitations"][-1]="This uses cached offline geometry; replanning replay does not establish live perception or route safety."
        manifest["limitations"].append("Recording EOF pauses observations; the mission remains active and no post-EOF replan is fabricated.")
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
        (output/"route_projection_pixels").mkdir()
        for index,(frame,repeats) in enumerate(zip(data["frames"],data["timeline"]["repeat_counts"])):
            if stop.is_set() or time.monotonic()>=deadline:
                raise RuntimeError("Saved-frame export interrupted or exceeded its lifetime")
            path = output/f"segmentation_frames/{index:06d}.png"
            projection = data["route_projections"][index]
            pixel_path = output/f"route_projection_pixels/{index:06d}.json"
            projection["report"]["source_pixel_annotation_file"] = pixel_path.relative_to(output).as_posix()
            write_json(pixel_path,{"schema_version":1,"frame_id":frame["frame_id"],
                "timestamp_ns":frame["timestamp_ns"],"report":projection["report"],
                "source_segments":projection["source_segments"].tolist(),
                "source_arrowheads":projection["source_arrowheads"].tolist()})
            if pixel_path.stat().st_size > 4*1024*1024:
                raise RuntimeError("Source-pixel annotation exceeds its bounded4MiB scope")
            record(pixel_path)
            render_frame(data,frame,index).save(path);record(path)
            manifest["coverage"].append(frame_coverage(frame,repeats,data["route_projections"][index]));save()
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
        manifest["route_projection_status_counts"] = dict(Counter(frame["route_projection"]["status"] for frame in manifest["coverage"]))
        manifest["frames_with_direction_arrows"] = sum(frame["route_projection"]["arrow_count"]>0 for frame in manifest["coverage"])
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
    parser.add_argument("--replanning-replay",type=Path,help="Separate admitted per-frame short-horizon navigation replay")
    parser.add_argument("--output",required=True,type=Path,help="Fresh separate segmentation output")
    parser.add_argument("--expected-frames",required=True,type=int,help="Exact complete saved sample count1–58")
    parser.add_argument("--max-runtime",type=int,default=600,help="Bounded CPU lifetime, at most900seconds")
    args = parser.parse_args(argv)
    if not 1<=args.max_runtime<=900:
        parser.error("Whole CPU-stage lifetime must be within1–900seconds")
    if args.replanning_replay is not None and args.research_plan is None:
        parser.error("--replanning-replay requires --research-plan for separately admitted research assumptions")
    stage_deadline = time.monotonic()+args.max_runtime
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    try:
        data = load_recording(args.run,args.sequence,args.cache,args.research_plan,args.expected_frames,
                              replanning_replay=args.replanning_replay)
        result = export_recording(data,args.output,research_plan=args.research_plan,max_runtime=args.max_runtime,
                                  stage_deadline=stage_deadline)
        print(json.dumps({"status":result["status"],"sampled_frames":result["sampled_frame_count"],"output":str(args.output.resolve())}))
    except Exception as error:
        print("Saved segmentation recording stopped: "+str(error),file=sys.stderr)
        return 1
    return 0


if __name__=="__main__":
    raise SystemExit(main())
