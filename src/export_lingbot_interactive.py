"""Offline official Viser playback of a complete saved LingBot point-cloud scene.

CPU-only, exact saved samples, native XYZ/W2C and separately labelled saved
evidence. The official frontend is packaged separately; this exporter writes
scene.viser and its provenance, without models, captures or video encoders.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import select
import signal
import sys
import threading
import time
from types import SimpleNamespace

ADAPTER_SHA = "1ab730c4e0d495c447264f53f3455d5bf2b6a76512b4c0c03661c3d8ed2239e0"
SEGMENTATION_SHA = "d084e18f9546316788a78393580b6f3916929a3f442ede75f4ab5b2a0c86d8c7"
SOURCE_REVISION = "849e690bb086103637e44b1e91878d9d43a8bf0c"
MAX_SCENE_BYTES = 64 * 1024 * 1024
IDENTITY = ("geometry_fingerprint", "input_fingerprint", "processed_grid_id", "archive_sha256",
            "map_frame", "units", "scale", "up")


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def logical_timeline(frames):
    times = [frame.get("timestamp_ns") for frame in frames]
    if (not times or len(times) > 58 or any(type(value) is not int or value < 0 for value in times)
            or any(right <= left for left, right in zip(times, times[1:]))):
        raise ValueError("Require1–58 exact saved samples with strictly increasing nonnegative timestamps")
    ids = [frame.get("frame_id") for frame in frames]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Saved frame identities must be unique and ordered")
    return {"source_time_origin_ns": times[0], "transition_seconds": [(value - times[0])/1e9 for value in times],
            "final_hold_seconds": 2, "duration_seconds": (times[-1] - times[0])/1e9 + 2,
            "chronology": "saved_source_pts_held_samples_no_interpolation", "timing_quantized": False}


def validate_controls(expected_frames, max_runtime, port):
    if (type(expected_frames) is not int or not 1 <= expected_frames <= 58
            or type(max_runtime) is not int or not 1 <= max_runtime <= 900
            or type(port) is not int or not 1024 <= port <= 65535):
        raise ValueError("Require exact1–58 samples, lifetime1–900seconds and a nonprivileged port")


def frame_coverage(frame):
    return {"frame_id": frame["frame_id"], "timestamp_ns": frame["timestamp_ns"],
            "timestamp_provenance": frame["source"].get("timestamp_provenance", {}),
            "disposition": frame["disposition"], "semantic_status": (frame["semantic"] or {}).get("status", "not_recorded"),
            "queries": [{"query_id": query.get("query_id"), "prompt": query.get("original_phrase"),
                         "status": query.get("status"), "error": query.get("error"), "instances": len(query.get("instances", []))}
                        for query in (frame["semantic"] or {}).get("queries", [])]}


def record_transitions(viewer, overlays, serializer, frames, timeline, states, check):
    if (timeline != logical_timeline(frames) or len(states) != len(frames)
            or len(viewer.frame_nodes) != len(frames)):
        raise ValueError("Native timeline, frame nodes and cumulative saved states must join exactly")
    transitions=[]
    for index,(frame,at) in enumerate(zip(frames,timeline["transition_seconds"])):
        check()
        if index:serializer.insert_sleep(at-timeline["transition_seconds"][index-1])
        viewer.gui_timestep.value=index  # plain logical handle, never a native GUI setter
        with viewer.server.atomic():overlays.tick()
        expected=[number<=index for number in range(len(frames))]
        if [bool(node.visible) for node in viewer.frame_nodes]!=expected:
            raise RuntimeError("Native serialized cumulative visibility differs")
        transitions.append({"frame_id":frame["frame_id"],"timestamp_ns":frame["timestamp_ns"],
            "timestamp_provenance":frame["source"].get("timestamp_provenance",{}),"queries":frame_coverage(frame)["queries"],
            "logical_seconds":at,"visible_frame_indices":list(range(index+1)),
            "saved_camera_trajectory_points":len(states[index]["camera_trajectory"]),
            "research_route_visible":len(states[index]["research_path"])>=2})
    serializer.insert_sleep(2)
    return transitions


def validate_scene_payload(payload):
    if type(payload) is not bytes or not 0<len(payload)<=MAX_SCENE_BYTES:
        raise ValueError("Official scene recording is empty or exceeds64MiB")


def verify_segmentation(folder, data):
    folder = Path(folder).resolve()
    manifest_path, config_path = folder/"manifest.json", folder/"export_config.json"
    manifest, config = read(manifest_path), read(config_path)
    sidecar = (folder/"manifest.sha256").read_text(encoding="ascii").split()
    if sidecar != [sha256(manifest_path), "manifest.json"]:
        raise ValueError("Saved segmentation manifest checksum differs")
    if (manifest.get("status") != "complete" or config.get("helper_sha256") != SEGMENTATION_SHA
            or config.get("frames_only") is not False or config.get("rgb_view") != "source"
            or config.get("mask_projection") != "saved_source_to_processed_center_affine_nearest_no_extrapolation"
            or config.get("timeline") != data["timeline"]
            or config.get("expected_sampled_frames") != len(data["frames"])
            or manifest.get("sampled_frame_count") != len(data["frames"])
            or manifest.get("run_dir") != str(data["run_dir"])
            or manifest.get("sequence_path") != str(data["sequence_path"])
            or manifest.get("original_planning") != data["planning"]
            or manifest.get("research_plan") != {key:value for key,value in (data["research_plan"] or {}).items() if key!="validated_path"}
            or manifest.get("geometry_identity") != {key:data["geometry"].get(key) for key in IDENTITY}):
        raise ValueError("Segmentation and native scene must join the exact saved source, geometry and research result")
    if any(manifest.get("input_sha256", {}).get(path) != digest for path,digest in data["input_sha256"].items()):
        raise ValueError("Segmentation original input hashes differ from the saved native scene inputs")
    rows = manifest.get("coverage", [])
    if len(rows) != len(data["frames"]):
        raise ValueError("Segmentation coverage count differs")
    for frame, row, repeats in zip(data["frames"], rows, data["timeline"]["repeat_counts"]):
        if (any(row.get(key) != value for key,value in frame_coverage(frame).items())
                or row.get("encoded_repeat_count") != repeats):
            raise ValueError("Saved segmentation frame order, timestamp or query evidence differs")
    outputs = manifest.get("outputs_sha256", {})
    required = {"export_config.json", "segmentation.mp4"} | {f"segmentation_frames/{index:06d}.png" for index in range(len(rows))}
    if not required <= outputs.keys():
        raise ValueError("Segmentation must include its genuine saved movie and every sample PNG")
    if not 0 < (folder/"segmentation.mp4").stat().st_size <= 32*1024*1024:
        raise ValueError("Saved segmentation movie is empty or exceeds32MiB")
    for relative, expected in outputs.items():
        path = (folder/relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(folder) or not path.is_file() or sha256(path)!=expected:
            raise ValueError("Saved segmentation output hash/path differs")
        data["input_sha256"][str(path)] = expected
    for path in (manifest_path, folder/"manifest.sha256"):
        data["input_sha256"][str(path.resolve())] = sha256(path)
    return folder, manifest


def process_memory():
    result={"threads":0,"vmhwm_bytes":0}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("Threads:"): result["threads"]=int(line.split()[1])
        elif line.startswith("VmHWM:"): result["vmhwm_bytes"]=int(line.split()[1])*1024
    return result


def runtime_identity(viser):
    package = Path(viser.__file__).resolve().parent
    names=("_viser.py","infra/_infra.py","_messages.py","_scene_api.py","_gui_handles.py")
    sources={name:sha256(package/name) for name in names}
    build=package/"client/build"
    if not (build/"index.html").is_file():
        raise ValueError("Installed official Viser frontend build is missing")
    files=[{"relative_path":str(path.relative_to(build)).replace("\\","/"),"sha256":sha256(path),"size_bytes":path.stat().st_size}
           for path in sorted(build.rglob("*")) if path.is_file()]
    return {"viser_version":importlib.metadata.version("viser"),"resolved_package_root":str(package),
            "python_sources_sha256":sources,"frontend_inventory":files,
            "frontend_inventory_sha256":hashlib.sha256(json.dumps(files,sort_keys=True,separators=(",",":")).encode()).hexdigest()}


def export(args):
    deadline=time.monotonic()+args.max_runtime
    validate_controls(args.expected_frames,args.max_runtime,args.port)
    if os.name!="posix" or not hasattr(os,"sched_getaffinity"):
        raise RuntimeError("Interactive scene export requires the reviewed Linux CPU runtime")
    if "torch" in sys.modules:
        raise RuntimeError("Run this fresh CPU exporter before importing Torch")
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[name]="4"
    allowed=sorted(os.sched_getaffinity(0))
    if len(allowed)<4:raise RuntimeError("Four allowed CPUs are required")
    os.sched_setaffinity(0,allowed[:4])
    adapter_path=Path(__file__).with_name("serve_lingbot_recording.py").resolve()
    if sha256(adapter_path)!=ADAPTER_SHA:raise ValueError("Frozen native saved-input adapter bytes differ")
    adapter=importlib.import_module("serve_lingbot_recording")
    if Path(adapter.__file__).resolve()!=adapter_path:raise ValueError("Native adapter imported outside frozen source")
    stop,finished=threading.Event(),threading.Event()
    parent=os.getppid()
    pipe=adapter.inherited_controller(args)
    source=args.source_root.resolve()
    expected_lock=source.parent.parent/"outputs/.pipeline2_video_execution.lock"
    if args.execution_lock_path.resolve()!=expected_lock.resolve():raise ValueError("Use the isolated Pipeline2 execution lock")
    global_lease=adapter.NativeLease(source,args.port);global_lease.path=expected_lock
    port_lease=adapter.NativeLease(source,args.port)
    observed={"max_process_threads":0,"sampled_helper_vmhwm_bytes":0}
    output=None;viewer=None;manifest=None
    handlers={number:signal.signal(number,lambda *_:stop.set()) for number in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP)}
    def check():
        if stop.is_set() or time.monotonic()>=deadline:raise RuntimeError("Interactive export interrupted or exceeded whole-stage lifetime")
    def watchdog():
        while not finished.wait(.05):
            try:
                memory=process_memory()
                observed["max_process_threads"]=max(observed["max_process_threads"],memory["threads"])
                observed["sampled_helper_vmhwm_bytes"]=max(observed["sampled_helper_vmhwm_bytes"],memory["vmhwm_bytes"])
                readable=select.select([pipe],[],[],0)[0] if pipe is not None else []
            except BaseException as error:
                observed["watchdog_error"]=str(error);stop.set();readable=[]
            if os.getppid()!=parent or readable or stop.is_set() or time.monotonic()>=deadline:
                stop.set()
                if not finished.wait(5):
                    if output is not None:
                        try:write(output/"shutdown_failure.json",{"status":"failed","reason":"whole_stage_or_parent_loss_cleanup_timeout"})
                        except BaseException:pass
                    os._exit(124)
                return
    watcher=threading.Thread(target=watchdog,name="interactive-scene-lifetime",daemon=True)
    watcher.start()
    try:
        global_lease.acquire();port_lease.acquire();check()
        source,source_hashes=adapter.verify_source(source);check()
        data=adapter.prepare_inputs(args.run,args.sequence,args.cache,args.research_plan,expected_frames=args.expected_frames);check()
        timeline=logical_timeline(data["frames"])
        segmentation,segmanifest=verify_segmentation(args.segmentation_export,data);check()
        if args.output.resolve().is_relative_to(segmentation):raise ValueError("Fresh scene output must be outside saved segmentation")
        output=adapter.output_directory(args.output,data,source,args.research_plan)
        config={"schema_version":1,"helper_sha256":sha256(__file__),"native_adapter_sha256":ADAPTER_SHA,
                "source_revision":SOURCE_REVISION,"expected_sampled_frames":args.expected_frames,"device":"cpu",
                "model":None,"sky_segmentation":False,"native_use_point_map":True,"saved_xyz_changed":False,
                "saved_extrinsic":"world_to_camera_unchanged","actual_cpu_affinity":sorted(os.sched_getaffinity(0)),
                "cpu_threads":4,"max_runtime_seconds":args.max_runtime,"native_cameras_hidden":True,
                "max_display_points":adapter.MAX_POINTS,"max_semantic_centers":adapter.MAX_SEMANTIC_CENTERS,
                "scene_file":"scene.viser","scene_byte_cap":MAX_SCENE_BYTES,
                "serializer":"official_Viser_get_scene_serializer","chronology":timeline["chronology"],"timeline":timeline,
                "saved_segmentation_timeline":data["timeline"],"paired_controls_are_synchronized":False,
                "timestep_driver":"explicit_logical_index_without_GUI_callbacks","frontend_packaged":False,
                "semantic_overlay":"saved_positive_evidence_centers_separate_from_native_RGB","fused_voxels":False}
        manifest={"schema_version":1,"status":"running","helper_sha256":sha256(__file__),
                  "run_dir":str(data["run_dir"]),"sequence_path":str(data["sequence_path"]),"input_sha256":data["input_sha256"],
                  "geometry_identity":{key:data["geometry"].get(key) for key in IDENTITY},"original_planning":data["planning"],
                  "research_plan":{key:value for key,value in (data["research_plan"] or {}).items() if key!="validated_path"},
                  "expected_sampled_frames":args.expected_frames,"sampled_frame_count":len(data["frames"]),
                  "coverage":[frame_coverage(frame) for frame in data["frames"]],"transitions":[],"outputs_sha256":{},
                  "limitations":["Native point cloud uses saved samples; unsampled moments contain no new geometry or masks.",
                                 "Turquoise/orange centers are added saved semantic evidence, not traversability.",
                                 "Blue recorded camera motion is separate from a final-only assumed research route.",
                                 "Offline controls belong to the official Viser frontend; Python GUI callbacks are excluded."]}
        write(output/"manifest.json",manifest)
        sys.path.insert(0,str(source))
        module=importlib.import_module("lingbot_map.vis.point_cloud_viewer")
        if not Path(module.__file__).resolve().is_relative_to(source):raise ValueError("Official native viewer imported outside pinned checkout")
        module.torch.set_num_threads(4)
        if module.torch.cuda.is_available():raise RuntimeError("Interactive export must create no CUDA context")
        config["actual_cuda_available"]=False
        runtime=runtime_identity(module.viser)
        if runtime["viser_version"]!="1.1.1":raise ValueError("Use the reviewed official Viser1.1.1 runtime")
        config["native_runtime"]=runtime
        viewer=adapter.construct_loopback_viewer(module,data["native_pred"],args.port,data["native_downsample"],data["native_threshold"],.001)
        for handle in viewer.recording_controls.handles:handle.disabled=True
        # Exactly the native animate() initialization, without its infinite loop,
        # GUI timestep callbacks or any connected-client camera mutations.
        viewer.server.scene.add_frame("/frames",show_axes=False)
        viewer.frame_nodes=[]
        for step in viewer.all_steps:
            check()
            viewer.frame_nodes.append(viewer.server.scene.add_frame(f"/frames/{step}",show_axes=False))
            viewer.add_pc(step)
        viewer.gui_timestep=SimpleNamespace(value=0)
        viewer.fourd=False
        count=sum(len(points) for points in viewer.vis_pts_list)
        if not 0<count<=adapter.MAX_POINTS or len(viewer.vis_pts_list)!=args.expected_frames:raise ValueError("Actual native display exceeds its bounded point admission")
        config["actual_display_point_count"]=count
        pose=adapter.native_display_pose(viewer)
        config["display_only_camera"]=pose
        config["initial_camera_property_mapping"]={"position":"position","look_at":"look_at","up":"up_direction","fov":"fov"}
        for property_name,pose_key in config["initial_camera_property_mapping"].items():
            if not hasattr(viewer.server.initial_camera,property_name):raise RuntimeError("Official initial-camera API is missing: "+property_name)
            setattr(viewer.server.initial_camera,property_name,pose[pose_key])
        overlays=adapter.SavedOverlays(viewer,data,semantics=True,voxels=False,research=bool(args.research_plan))
        overlays.tick()
        config["semantic_overlay_sampling"]=overlays.semantic_sampling
        viewer.server.flush();check()
        serializer=viewer.server.get_scene_serializer()
        config["serializer_signatures"]={"insert_sleep":str(inspect.signature(serializer.insert_sleep)),"serialize":str(inspect.signature(serializer.serialize))}
        manifest["transitions"]=record_transitions(viewer,overlays,serializer,data["frames"],timeline,data["states"],check)
        check()
        payload=serializer.serialize();check()
        validate_scene_payload(payload)
        (output/"scene.viser").write_bytes(payload)
        manifest.update(scene_file="scene.viser",scene_sha256=sha256(output/"scene.viser"),scene_size_bytes=len(payload),timeline=timeline)
        for path,digest in data["input_sha256"].items():
            check()
            if sha256(path)!=digest:raise RuntimeError("Saved source changed during interactive export")
        if adapter.verify_source(source)[1]!=source_hashes:raise RuntimeError("Pinned native source changed during export")
        if sha256(adapter_path)!=ADAPTER_SHA:raise RuntimeError("Frozen native adapter changed during export")
        if sha256(__file__)!=config["helper_sha256"]:raise RuntimeError("Interactive exporter changed during export")
        if runtime_identity(module.viser)!=runtime:raise RuntimeError("Official Viser sources or frontend changed during export")
        manifest["source_inputs_unchanged"]=True
        config["source_files_sha256"]=source_hashes
        check();manifest["status"]="complete"
    except BaseException as error:
        if manifest is not None:manifest.update(status="failed",error=str(error))
        raise
    finally:
        stop.set()
        errors=[]
        if viewer is not None:
            try:viewer.server.stop()
            except BaseException as error:errors.append(str(error))
            try:viewer.recording_controls.restore_methods()
            except BaseException as error:errors.append(str(error))
        try:
            finished.set();watcher.join(timeout=.2)
            if manifest is not None:
                manifest["shutdown"]={"server_stopped":not errors,"animation_thread_started":False,"encoder_started":False,
                                      "watchdog_stopped":not watcher.is_alive(),"observed":observed,"errors":errors}
                if errors or watcher.is_alive() or observed.get("watchdog_error") or time.monotonic()>=deadline:
                    manifest["status"]="failed"
                config["observed_process_threads_and_memory"]=observed
                config["memory_measurement"]="Sampled helper-process VmHWM only, not aggregate descendant memory"
                write(output/"viewer_config.json",config)
                manifest["outputs_sha256"]={path.name:sha256(path) for path in (output/"viewer_config.json",output/"scene.viser") if path.is_file()}
                write(output/"manifest.json",manifest)
                if manifest["status"]=="complete":(output/"manifest.sha256").write_text(sha256(output/"manifest.json")+"  manifest.json\n",encoding="ascii")
        finally:
            try:port_lease.close()
            finally:
                try:global_lease.close()
                finally:
                    finished.set();watcher.join(timeout=.2)
                    for number,handler in handlers.items():signal.signal(number,handler)
                    for descriptor in (args.parent_pipe_fd,args.controller_lock_fd):
                        if descriptor is not None:os.close(descriptor)
    if manifest["status"]!="complete":raise RuntimeError("Interactive scene shutdown did not finish cleanly")
    return manifest


def main(argv=None):
    if argv is None:argv=sys.argv[1:]
    if argv==["--self-test"]:return self_test()
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("source-root","run","sequence","segmentation-export","output","execution-lock-path"):
        parser.add_argument("--"+name,type=Path,required=True)
    for name in ("cache","research-plan"):parser.add_argument("--"+name,type=Path)
    parser.add_argument("--expected-frames",type=int,required=True)
    parser.add_argument("--max-runtime",type=int,default=900)
    parser.add_argument("--port",type=int,default=8891)
    parser.add_argument("--parent-pipe-fd",type=int)
    parser.add_argument("--controller-lock-fd",type=int)
    args=parser.parse_args(argv)
    try:validate_controls(args.expected_frames,args.max_runtime,args.port)
    except ValueError as error:parser.error(str(error))
    try:
        result=export(args)
        print(json.dumps({"status":result["status"],"scene_sha256":result["scene_sha256"],"sampled_frames":result["sampled_frame_count"]}),flush=True)
    except Exception as error:
        print("Interactive native export stopped: "+str(error),file=sys.stderr);return 1
    return 0


def self_test():
    """Only new chronology, source-admission and control-bound CPU fixtures."""
    from contextlib import nullcontext
    from copy import deepcopy
    import math
    import tempfile
    import unittest

    def frames():
        return [{"frame_id":f"frame-{index}","timestamp_ns":1_000_000_000+index*500_000_001,
                 "source":{"timestamp_provenance":{"clock":"fixture_exact_source_pts"}},
                 "disposition":"processed","semantic":{"status":"ok","queries":[
                     {"query_id":f"query-{index}","original_phrase":"ground","status":"ok","error":None,
                      "instances":[] if index%2 else [{"fixture":True}]}]}} for index in range(58)]

    class Tests(unittest.TestCase):
        def test_exact_58_sample_timing(self):
            value=logical_timeline(frames())
            self.assertEqual(value["transition_seconds"][-1],28.500000057)
            self.assertEqual(value["duration_seconds"],30.500000057)
            self.assertFalse(value["timing_quantized"])

        def test_invalid_timing_and_frame_identity(self):
            for mutate in (lambda f:f.clear(),lambda f:f.append(f[-1]),
                           lambda f:f[1].update(timestamp_ns=f[0]["timestamp_ns"]),
                           lambda f:f[0].update(timestamp_ns=True),lambda f:f[0].update(timestamp_ns=None),
                           lambda f:f[0].update(timestamp_ns=-1),lambda f:f[1].update(frame_id=f[0]["frame_id"])):
                with self.subTest(mutation=mutate):
                    value=frames();mutate(value)
                    with self.assertRaises(ValueError):logical_timeline(value)

        def test_control_bounds(self):
            validate_controls(58,900,8891)
            for values in ((0,900,8891),(59,900,8891),(True,900,8891),(9,901,8891),
                           (9,0,8891),(9,True,8891),(9,900,1023),(9,900,65536)):
                with self.subTest(values=values):
                    with self.assertRaises(ValueError):validate_controls(*values)

        def test_explicit_cumulative_prefix_queries_and_final_route(self):
            source=frames();viewer=SimpleNamespace(gui_timestep=SimpleNamespace(value=0),
                server=SimpleNamespace(atomic=nullcontext),frame_nodes=[SimpleNamespace(visible=False) for _ in source])
            states=[{"camera_trajectory":[[0,0,0]]*(index+1),"research_path":[] if index<57 else [[0,0,0],[1,0,0]]}
                    for index in range(58)]
            elapsed=[0.0];sleeps=[]
            def tick():
                for index,node in enumerate(viewer.frame_nodes):node.visible=index<=viewer.gui_timestep.value
            def sleep(delta):sleeps.append(delta);elapsed[0]+=delta
            recorded=record_transitions(viewer,SimpleNamespace(tick=tick),SimpleNamespace(insert_sleep=sleep),
                source,logical_timeline(source),states,lambda:None)
            self.assertEqual(len(recorded),58);self.assertEqual(recorded[43]["visible_frame_indices"],list(range(44)))
            self.assertEqual(recorded[11]["queries"],frame_coverage(source[11])["queries"])
            self.assertFalse(any(row["research_route_visible"] for row in recorded[:-1]))
            self.assertTrue(recorded[-1]["research_route_visible"])
            self.assertEqual(sleeps[-1],2);self.assertTrue(math.isclose(elapsed[0],30.500000057,abs_tol=1e-12))

        def test_visibility_or_state_mismatch_stops(self):
            source=frames()[:1];viewer=SimpleNamespace(gui_timestep=SimpleNamespace(value=0),
                server=SimpleNamespace(atomic=nullcontext),frame_nodes=[SimpleNamespace(visible=False)])
            states=[{"camera_trajectory":[],"research_path":[]}]
            with self.assertRaises(RuntimeError):record_transitions(viewer,SimpleNamespace(tick=lambda:None),
                SimpleNamespace(insert_sleep=lambda _:None),source,logical_timeline(source),states,lambda:None)
            with self.assertRaises(ValueError):record_transitions(viewer,None,None,source,logical_timeline(source),[],lambda:None)

        def test_scene_size_and_type_guard(self):
            validate_scene_payload(b"fixture")
            for payload in (b"",bytearray(b"x"),b"x"*(MAX_SCENE_BYTES+1)):
                with self.assertRaises(ValueError):validate_scene_payload(payload)

        def fixture(self,base):
            folder=base/"segmentation";folder.mkdir();(folder/"segmentation_frames").mkdir()
            source=base/"source";source.write_bytes(b"saved source fixture")
            data={"frames":frames(),"run_dir":base/"run","sequence_path":base/"sequence.json",
                  "planning":{"status":"blocked_unverified_inputs"},"research_plan":{"status":"no_path","path_points":[]},
                  "geometry":{key:"fixture-"+key for key in IDENTITY},"input_sha256":{str(source):sha256(source)},
                  "timeline":{"video_fps":20,"repeat_counts":[10]*57+[40],"fixture":True}}
            config={"helper_sha256":SEGMENTATION_SHA,"frames_only":False,"rgb_view":"source",
                    "mask_projection":"saved_source_to_processed_center_affine_nearest_no_extrapolation",
                    "expected_sampled_frames":58,"timeline":data["timeline"]}
            write(folder/"export_config.json",config);(folder/"segmentation.mp4").write_bytes(b"metadata fixture, not encoded video")
            for index in range(58):(folder/f"segmentation_frames/{index:06d}.png").write_bytes(b"metadata fixture PNG")
            manifest={"status":"complete","run_dir":str(data["run_dir"]),"sequence_path":str(data["sequence_path"]),
                      "sampled_frame_count":58,"original_planning":data["planning"],"research_plan":data["research_plan"],
                      "geometry_identity":data["geometry"],"input_sha256":dict(data["input_sha256"]),
                      "coverage":[dict(frame_coverage(f),encoded_repeat_count=r) for f,r in zip(data["frames"],data["timeline"]["repeat_counts"])],
                      "outputs_sha256":{str(p.relative_to(folder)).replace("\\","/"):sha256(p) for p in folder.rglob("*") if p.is_file()}}
            return folder,data,manifest

        def save_fixture_manifest(self,folder,manifest):
            write(folder/"manifest.json",manifest)
            (folder/"manifest.sha256").write_text(sha256(folder/"manifest.json")+"  manifest.json\n",encoding="ascii")

        def test_58_source_hash_admission_read_only(self):
            with tempfile.TemporaryDirectory() as temp:
                folder,data,manifest=self.fixture(Path(temp));self.save_fixture_manifest(folder,manifest)
                before={str(path):sha256(path) for path in folder.rglob("*") if path.is_file()}
                verify_segmentation(folder,data)
                self.assertEqual(before,{str(path):sha256(path) for path in folder.rglob("*") if path.is_file()})
                self.assertIn(str((folder/"manifest.json").resolve()),data["input_sha256"])

        def test_source_hash_geometry_query_order_and_repeat_mismatches(self):
            with tempfile.TemporaryDirectory() as temp:
                folder,data,manifest=self.fixture(Path(temp))
                mutations=(lambda m:m["input_sha256"].update({next(iter(data["input_sha256"])):"changed"}),
                           lambda m:m["geometry_identity"].update(units="different"),
                           lambda m:m["coverage"].reverse(),lambda m:m["coverage"][0].update(timestamp_ns=17),
                           lambda m:m["coverage"][0]["queries"][0].update(status="failed"),
                           lambda m:m["coverage"][0].update(encoded_repeat_count=1),
                           lambda m:m.update(status="failed"),lambda m:m["research_plan"].update(path_points=[[1,2,3]]))
                for mutate in mutations:
                    with self.subTest(mutation=mutate):
                        altered=deepcopy(manifest);mutate(altered);self.save_fixture_manifest(folder,altered)
                        with self.assertRaises(ValueError):verify_segmentation(folder,deepcopy(data))

        def test_saved_asset_hash_and_sidecar_rejected(self):
            with tempfile.TemporaryDirectory() as temp:
                folder,data,manifest=self.fixture(Path(temp));self.save_fixture_manifest(folder,manifest)
                (folder/"segmentation_frames/000057.png").write_bytes(b"changed")
                with self.assertRaises(ValueError):verify_segmentation(folder,deepcopy(data))
                (folder/"manifest.sha256").write_text("wrong",encoding="ascii")
                with self.assertRaises(ValueError):verify_segmentation(folder,deepcopy(data))

        def test_saved_asset_traversal_rejected(self):
            with tempfile.TemporaryDirectory() as temp:
                folder,data,manifest=self.fixture(Path(temp));outside=Path(temp)/"outside";outside.write_bytes(b"outside")
                manifest["outputs_sha256"]["../outside"]=sha256(outside);self.save_fixture_manifest(folder,manifest)
                with self.assertRaises(ValueError):verify_segmentation(folder,data)

    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    print(json.dumps({"status":"passed" if result.wasSuccessful() else "failed","tests":result.testsRun,
                     "helper_sha256":sha256(__file__),"scope":"New pure CPU chronology/source/control fixtures; no Linux lifecycle, Viser, model or encoded-video claim"}),flush=True)
    return 0 if result.wasSuccessful() else 1


if __name__=="__main__":raise SystemExit(main())
