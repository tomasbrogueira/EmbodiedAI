"""Bounded CPU capture of one complete saved evaluation recording in LingBot-map.

No model, checkpoint, producer, sky segmentation or inference is used. The
official viewer renders the saved point map and cameras; our optional overlays
are separately labelled. A connected browser supplies genuine WebGL captures.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import signal
import select
import stat
import subprocess
import sys
import threading
import time

import numpy as np
from PIL import Image

from export_pipeline_video import (
    MAX_GEOMETRY_ARCHIVE_BYTES, cumulative_states, load_export_inputs,
    load_npz_checked, sampled_timeline, sha256,
)

SOURCE_REVISION = "849e690bb086103637e44b1e91878d9d43a8bf0c"
MAX_FRAMES = 58
MAX_SECONDS = 900
MAX_POINTS = 300000
MAX_OVERLAY_VOXELS = 6000
MAX_SEMANTIC_CENTERS = 6000
RESEARCH_LABEL = "Research illustration — assumed scale, ground direction and robot; no safety claim"


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def cpu_rss_bytes():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])*1024
    raise RuntimeError("Linux CPU resident-memory diagnostic is unavailable")


def verify_source(source_root):
    root = Path(source_root).resolve()
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                                text=True, timeout=15)
        if result.returncode:
            raise ValueError("Cannot verify the pinned LingBot-map checkout: " + result.stderr.strip())
        return result.stdout.strip()
    if Path(git("rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError("--source-root must be the LingBot-map checkout itself")
    if git("rev-parse", "HEAD") != SOURCE_REVISION:
        raise ValueError("LingBot-map source revision differs from the reviewed pin")
    if git("diff", "HEAD", "--name-only", "--", "lingbot_map"):
        raise ValueError("LingBot-map package has modified tracked source")
    hashes = {str(path.relative_to(root)): sha256(path)
              for path in sorted((root / "lingbot_map").rglob("*.py"))}
    if "lingbot_map/vis/point_cloud_viewer.py" not in hashes:
        # Windows relative paths use backslashes; provenance uses one spelling.
        hashes = {name.replace("\\", "/"): value for name, value in hashes.items()}
    if "lingbot_map/vis/point_cloud_viewer.py" not in hashes:
        raise ValueError("Pinned native viewer source is missing")
    return root, hashes


def prepare_inputs(run, sequence, cache=None, research_plan=None, *, expected_frames):
    """Reuse the saved run/sequence/mask/journal joins, then validate numeric geometry."""
    if type(expected_frames) is not int or not 1 <= expected_frames <= MAX_FRAMES:
        raise ValueError("A recording requires an exact expected count within 1–58 saved frames")
    data = load_export_inputs(run, sequence, cache_dir=cache, research_plan_path=research_plan)
    frames = data["frames"]
    if len(frames) != expected_frames:
        raise ValueError("Recording requires its exact complete saved sequence; no truncation or count substitution")
    arrays = load_npz_checked(data["cache"] / "geometry.npz", max_bytes=MAX_GEOMETRY_ARCHIVE_BYTES,
                              names=("images", "world_points", "world_points_conf", "depth", "extrinsic", "intrinsic"))
    if sha256(data["cache"] / "geometry.npz") != data["geometry"]["archive_sha256"]:
        raise ValueError("Geometry archive changed during saved-cache admission")
    required = ("images", "world_points", "world_points_conf", "depth", "extrinsic", "intrinsic")
    if any(name not in arrays for name in required):
        raise ValueError("Exact saved geometry arrays are missing")
    images = arrays["images"]
    n, height, width, channels = images.shape
    if images.dtype != np.uint8 or channels != 3 or n != len(frames):
        raise ValueError("Saved RGB must be exact NHWC uint8")
    for name, shape in (("world_points", (n, height, width, 3)), ("world_points_conf", (n, height, width)),
                        ("extrinsic", (n, 3, 4)), ("intrinsic", (n, 3, 3))):
        array = arrays[name]
        if array.shape != shape or array.dtype.kind != "f":
            raise ValueError("Invalid saved geometry array: " + name)
    depth = arrays["depth"]
    if depth.shape == (n, height, width, 1):
        depth = depth[..., 0]
    if depth.shape != (n, height, width) or depth.dtype.kind != "f":
        raise ValueError("Invalid saved optical-axis depth")
    extrinsic, intrinsic = arrays["extrinsic"], arrays["intrinsic"]
    if (not np.isfinite(extrinsic).all() or not np.isfinite(intrinsic).all()
            or np.any(intrinsic[:, (0, 1), (0, 1)] <= 0)
            or not np.allclose(intrinsic[:, 2], [0, 0, 1])):
        raise ValueError("Invalid saved camera matrices")
    rotation = extrinsic[:, :3, :3]
    if (not np.allclose(rotation @ rotation.transpose(0, 2, 1), np.eye(3), atol=1e-3)
            or not np.allclose(np.linalg.det(rotation), 1, atol=1e-3)):
        raise ValueError("Native viewer requires saved rigid W2C poses")
    minimum = float(data["geometry"].get("settings", {}).get("min_confidence",
                    data["config"].get("geometry", {}).get("min_confidence", 1.5)))
    if not math.isfinite(minimum) or not 1 < minimum <= 5:
        raise ValueError("Reviewed preview requires a saved confidence threshold in (1,5]")
    points, confidence = arrays["world_points"], arrays["world_points_conf"]
    valid = (np.isfinite(points).all(-1) & np.isfinite(depth) & (depth > 0)
             & np.isfinite(confidence) & (confidence > 0) & (confidence >= minimum))
    for index, frame in enumerate(frames):
        pad = frame["transform"].get("pad_ltrb")
        if (not isinstance(pad, list) or len(pad) != 4
                or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in pad)):
            raise ValueError("Saved padding is missing or invalid")
        left, top, right, bottom = pad
        if left + right >= width or top + bottom >= height:
            raise ValueError("Saved padding covers the entire model grid")
        if top: valid[index, :top] = False
        if bottom: valid[index, -bottom:] = False
        if left: valid[index, :, :left] = False
        if right: valid[index, :, -right:] = False
    # Display-only NaNs preserve the recorded common accepted-pixel mask. The
    # archive, poses, RGB, numeric XYZ and confidence are never modified in place.
    displayed_points = points.copy()
    displayed_points[~valid] = np.nan
    counts = valid.reshape(n, -1).sum(1).astype(int)
    downsample = 10
    while sum((int(count) + downsample - 1) // downsample for count in counts) > MAX_POINTS:
        downsample += 1
    if downsample > 1000:
        raise ValueError("Native display cannot meet its bounded point admission")
    native_threshold = float(np.nextafter(np.float32(minimum), np.float32(-np.inf)))
    pred = {"images": images.transpose(0, 3, 1, 2).astype(np.float32) / 255,
            "world_points": displayed_points, "world_points_conf": confidence,
            "depth": depth[..., None], "extrinsic": extrinsic, "intrinsic": intrinsic}
    data.update(native_pred=pred, valid_point_counts=counts.tolist(), native_downsample=downsample,
                retained_native_geometry_bytes=sum(array.nbytes for array in pred.values()),
                min_confidence=minimum, native_threshold=native_threshold,
                states=cumulative_states(data), timeline=sampled_timeline(frames, video_fps=20,
                                                                         end_hold_seconds=2))
    return data


def output_directory(output, data, source_root, research_plan=None):
    output = Path(output).resolve()
    roots = [data["run_dir"], data["sequence_path"].parent, data["cache"], Path(source_root).resolve()]
    if research_plan:
        roots.append(Path(research_plan).resolve().parent)
    if output.exists() or any(output.is_relative_to(root) for root in roots):
        raise ValueError("Output must be a fresh separate directory outside all saved inputs and source")
    output.mkdir(parents=True, exist_ok=False)
    return output


class _PlaybackStopped(Exception):
    pass


class NativePlayback:
    """Interrupt only native animate's sleep, without editing vendor files."""
    def __init__(self, viewer, module, stop, tick):
        self.viewer, self.module, self.stop, self.tick = viewer, module, stop, tick
        self.original_time = module.time
        self.thread_id = None
        self.error = None
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run, name="lingbot-native-playback", daemon=True)

    def __getattr__(self, name):
        return getattr(self.original_time, name)

    def sleep(self, seconds):
        if threading.get_ident() != self.thread_id:
            return self.original_time.sleep(seconds)
        self.tick()
        self.ready.set()
        if self.stop.wait(seconds):
            raise _PlaybackStopped()

    def _run(self):
        self.thread_id = threading.get_ident()
        try:
            self.viewer.animate()
        except _PlaybackStopped:
            pass
        except BaseException as error:
            self.error = error
            self.stop.set()
        finally:
            self.ready.set()

    def start(self):
        self.module.time = self
        self.viewer.on_replay = True
        original_checkbox = self.viewer.server.gui.add_checkbox
        def paused_checkbox(label, *args, **kwargs):
            if label == "Playing":
                if args:
                    args = (False, *args[1:])
                else:
                    kwargs["initial_value"] = False
                handle = original_checkbox(label, *args, **kwargs)
                handle.disabled = True
                return handle
            return original_checkbox(label, *args, **kwargs)
        self.viewer.server.gui.add_checkbox = paused_checkbox
        try:
            self.thread.start()
            if not self.ready.wait(30) or self.error:
                raise RuntimeError("Native animation did not initialize") from self.error
        finally:
            self.viewer.server.gui.add_checkbox = original_checkbox

    def close(self):
        self.stop.set()
        self.thread.join(timeout=3)
        self.module.time = self.original_time
        if self.thread.is_alive():
            raise RuntimeError("Native animation failed to stop within three seconds")


class GuiControls:
    """Collect native local-button handles too, so capture freezes every control."""
    def __init__(self, gui):
        self.handles = []
        self.gui = gui
        self.originals = {}
        for name in ("add_button", "add_button_group", "add_checkbox", "add_slider", "add_number", "add_text", "add_dropdown"):
            original = getattr(gui, name, None)
            if not callable(original):
                continue
            self.originals[name] = original
            def tracked(*args, _original=original, **kwargs):
                handle = _original(*args, **kwargs)
                if hasattr(handle, "disabled"):
                    self.handles.append(handle)
                return handle
            setattr(gui, name, tracked)

    def restore_methods(self):
        for name, original in self.originals.items():
            setattr(self.gui, name, original)


def construct_loopback_viewer(module, pred, port, downsample, threshold, point_size):
    """Confine the upstream hardcoded host during this isolated constructor only."""
    original = module.viser.ViserServer
    created = []
    registries = []
    def loopback_factory(*args, **kwargs):
        if args:
            raise RuntimeError("Unexpected positional native server constructor")
        kwargs["host"] = "127.0.0.1"
        server = original(**kwargs)
        created.append(server)
        registries.append(GuiControls(server.gui))
        return server
    module.viser.ViserServer = loopback_factory
    try:
        viewer = module.PointCloudViewer(model=None, device="cpu", port=port, pred_dict=pred,
                                         use_point_map=True, mask_sky=False, show_camera=False,
                                         depth_stride=1, vis_threshold=threshold, point_size=point_size)
        if (not callable(getattr(viewer.server, "get_host", None))
                or not callable(getattr(viewer.server, "get_port", None))
                or viewer.server.get_host() != "127.0.0.1"
                or viewer.server.get_port() != port):
            raise RuntimeError("Native server did not bind the exact requested loopback host and port")
    except BaseException:
        for server in created:
            server.stop()
        for registry in registries:
            registry.restore_methods()
        raise
    finally:
        module.viser.ViserServer = original
    viewer.downsample_slider.value = downsample
    viewer.downsample_slider.disabled = True  # preserve the aggregate browser point cap
    viewer.recording_controls = registries[0]
    viewer.show_camera_checkbox.disabled = True
    viewer.vis_threshold_slider.disabled = True
    viewer.psize_slider.disabled = True
    for name in ("screenshot_button", "glb_export_button", "save_video_button"):
        getattr(viewer, name).disabled = True  # their unchecked file paths/fallbacks are not our export
    return viewer


def native_display_pose(viewer, *, sample_cap=20000):
    """Fit a display camera to native accepted points; do not transform geometry."""
    samples = []
    per_frame = max(1, sample_cap // max(1, len(viewer.all_steps)))
    if len(viewer.vis_pts_list) != len(viewer.all_steps):
        raise RuntimeError("Native point-cloud handles are not fully initialized")
    for points in viewer.vis_pts_list:
        points = np.asarray(points).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points):
            stride = max(1, math.ceil(len(points) / per_frame))
            samples.append(points[::stride])
    if not samples:
        raise RuntimeError("Native accepted point cloud has no finite display samples")
    points = np.concatenate(samples)
    lower, upper = np.percentile(points, [5, 95], axis=0)
    center = (lower + upper) / 2
    radius = max(float(np.linalg.norm(upper - lower) / 2), 1e-4)
    first = viewer.all_steps[0]
    rotation = np.asarray(viewer.cam_dict["R"][first], dtype=float)
    first_position = np.asarray(viewer.cam_dict["t"][first], dtype=float)
    # Look toward the observed scene from the first camera's side, rather than
    # the opposite side selected by the native world's +Z Overview preset.
    direction = first_position - center
    if np.linalg.norm(direction) < radius * .02:
        direction = -rotation[:, 2]
    direction /= np.linalg.norm(direction)
    up = -rotation[:, 1]
    up /= np.linalg.norm(up)
    if abs(float(direction @ up)) > .95:
        direction = -rotation[:, 2]
        direction /= np.linalg.norm(direction)
    yaw = math.radians(12)
    base_direction = direction.copy()
    direction = (direction * math.cos(yaw) + np.cross(up, direction) * math.sin(yaw)
                 + up * float(up @ direction) * (1-math.cos(yaw)))
    fov = math.radians(50)
    distance = radius / math.sin(fov / 2) * 1.25
    return {"method": "first_saved_camera_side_robust_accepted_point_fit_display_only",
            "geometry_changed": False, "percentile_bounds": [5, 95],
            "sample_count": len(points), "sample_cap": sample_cap,
            "native_bounds": [lower.tolist(), upper.tolist()], "pivot": center.tolist(),
            "position": (center + direction * distance).tolist(),
            "look_at": center.tolist(), "up_direction": up.tolist(), "fov": fov,
            "up_provenance": "negative_y_axis_of_first_saved_camera_to_world_rotation",
            "first_saved_camera_position": first_position.tolist(),
            "base_center_to_camera_direction": base_direction.tolist(), "display_yaw_degrees": 12,
            "distance_native_units": distance}


def _cube_mesh(indices, origin, size):
    corners = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                        [0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=np.float32)
    triangles = np.array([[0,3,2],[0,2,1],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
                          [1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]], dtype=np.uint32)
    vertices = (np.asarray(origin) + (indices[:,None,:] + corners) * size).reshape(-1,3).astype(np.float32)
    faces = (triangles[None,:,:] + 8*np.arange(len(indices))[:,None,None]).reshape(-1,3).astype(np.uint32)
    return vertices, faces


def get_native_render(camera, *, timeout):
    """Bound the public blocking native API even on Viser versions without timeout."""
    done = threading.Event()
    result, errors = [], []
    def request():
        try:
            kwargs = {"height":800,"width":800}
            parameters = inspect.signature(camera.get_render).parameters
            if "timeout" in parameters:
                kwargs["timeout"] = timeout
            if "transport_format" in parameters:
                kwargs["transport_format"] = "png"
            result.append(camera.get_render(**kwargs))
        except BaseException as error:
            errors.append(error)
        finally:
            done.set()
    worker = threading.Thread(target=request,name="viser-native-render-request",daemon=True)
    worker.start()
    if not done.wait(timeout):
        raise RuntimeError("Native WebGL capture timed out; no image fallback is allowed")
    if errors:
        raise RuntimeError("Native WebGL capture failed") from errors[0]
    return result[0]


class SavedOverlays:
    def __init__(self, viewer, data, *, semantics=False, voxels=False, research=False):
        self.viewer, self.data = viewer, data
        self.semantics, self.voxels, self.research = semantics, voxels, research
        self.handles = []
        self.previous = None
        self.frame_facts = None
        self.lock = threading.RLock()
        union = sorted({(kind,*map(int,index)) for state in data["states"]
                        for kind in ("positive_indices","hazard_indices") for index in state[kind]}) if semantics else []
        selection = [union[index] for index in np.linspace(0,len(union)-1,min(len(union),MAX_SEMANTIC_CENTERS),dtype=int)] if union else []
        self.semantic_selected = set(selection)
        self.semantic_sampling = {"policy":"fixed_evenly_spaced_lexicographic_union_kind_native_voxel_index",
                                  "cap":MAX_SEMANTIC_CENTERS,"union_cells":len(union),"selected_cells":len(selection),
                                  "selection_sha256":hashlib.sha256(json.dumps(selection).encode()).hexdigest(),
                                  "saved_evidence_changed":False}
        if (research or len(data["frames"]) > 1) and not callable(getattr(viewer.server.scene, "add_line_segments", None)):
            raise RuntimeError("Installed Viser lacks the native line-segment overlay API")
        if voxels and not callable(getattr(viewer.server.scene, "add_mesh_simple", None)):
            raise RuntimeError("Installed Viser lacks the native mesh overlay API")
        if voxels and len(data["final_voxel_indices"]) > MAX_OVERLAY_VOXELS:
            raise ValueError("Optional fused voxel overlay exceeds its 6000-cube cap")

    def line(self, name, points, color, width):
        if len(points) < 2:
            return
        segments = np.stack([points[:-1], points[1:]], axis=1).astype(np.float32)
        colors = np.broadcast_to(np.asarray(color, dtype=np.uint8), segments.shape).copy()
        self.handles.append(self.viewer.server.scene.add_line_segments(name, points=segments,
                                                                      colors=colors, line_width=width))

    def tick(self):
        if not hasattr(self.viewer, "gui_timestep") or not hasattr(self.viewer, "frame_nodes"):
            return
        with self.lock:
            self.viewer.update_frame_visibility()
            index = int(self.viewer.gui_timestep.value)
            if index == self.previous:
                return
            for handle in self.handles:
                handle.remove()
            self.handles.clear()
            state = self.data["states"][index]
            if self.frame_facts is not None:
                frame = self.data["frames"][index]
                queries = (frame["semantic"] or {}).get("queries", [])
                query_text = "; ".join(f"{html.escape(str(q.get('original_phrase','')))}: "
                                       f"{html.escape(str(q.get('status','unknown')))}, {len(q.get('instances',[]))} instances"
                                       for q in queries) or "No semantic query recorded"
                timestamp = frame["timestamp_ns"]
                source_time = f"{timestamp/1e9:.3f}s" if timestamp is not None else "unknown"
                self.frame_facts.content = (f"Saved sample {index+1}/{len(self.data['frames'])}; "
                                            f"{html.escape(frame['frame_id'])}; source timestamp {source_time}.\n\n"+query_text)
            self.line("/adapter/saved_camera_trajectory", state["camera_trajectory"], (75,156,255), 3)
            if self.semantics:
                for name, color in (("positive_indices", (25,194,177)), ("hazard_indices", (239,127,49))):
                    indices = state[name]
                    indices = np.asarray([index for index in indices if (name,*map(int,index)) in self.semantic_selected],dtype=np.int64).reshape(-1,3)
                    if len(indices):
                        centers = np.asarray(self.data["map"]["origin"]) + (indices+.5)*self.data["map"]["voxel_size"]
                        self.handles.append(self.viewer.server.scene.add_point_cloud(
                            "/adapter/"+name, points=centers.astype(np.float32),
                            colors=np.broadcast_to(np.array(color,np.uint8), centers.shape).copy(), point_size=.006))
            if self.voxels and len(state["voxel_indices"]):
                vertices, faces = _cube_mesh(state["voxel_indices"], self.data["map"]["origin"], self.data["map"]["voxel_size"])
                self.handles.append(self.viewer.server.scene.add_mesh_simple(
                    "/adapter/saved_fused_voxel_cuboids", vertices=vertices, faces=faces,
                    color=(170,180,195), wireframe=True, opacity=.25))
            if self.research:
                self.line("/adapter/final_only_assumed_research_route", state["research_path"], (212,100,240), 6)
            self.previous = index
            self.viewer.server.flush()


def capture_native(viewer, client, data, output, manifest, write_manifest, stop, deadline, overlays):
    """Capture native WebGL only; upstream RGB fallback is never called."""
    destination = output / "native_frames"
    destination.mkdir(exist_ok=False)
    before = int(viewer.gui_timestep.value)
    old_fourd = viewer.fourd
    viewer.fourd = False
    viewer.gui_timestep.disabled = True
    threshold_disabled = viewer.vis_threshold_slider.disabled
    viewer.vis_threshold_slider.disabled = True
    disabled_controls = []
    for handle in viewer.recording_controls.handles:
        if hasattr(handle,"disabled") and not isinstance(handle,(str,type)):
            disabled_controls.append((handle,handle.disabled))
            handle.disabled = True
    camera = client.camera
    camera_state = {name: np.asarray(getattr(camera, name)).tolist()
                    for name in ("position", "look_at", "up_direction", "wxyz", "fov")}
    manifest["capture"] = {"status": "running", "renderer": "official_lingbot_PointCloudViewer_Viser_WebGL_get_render",
                           "client_id": getattr(client,"client_id",None), "camera": camera_state, "frames": []}
    write_manifest()
    try:
        for index, frame in enumerate(data["frames"]):
            if stop.is_set() or time.monotonic() >= deadline:
                raise RuntimeError("Native capture interrupted or exceeded its bounded lifetime")
            viewer.gui_timestep.value = index
            overlays.tick()
            for name in ("position", "look_at", "up_direction", "fov"):
                setattr(camera, name, camera_state[name])
            viewer.server.flush()
            time.sleep(.15)
            # Native timestep callbacks toggle only current/previous nodes and
            # may arrive asynchronously. Settle, restore cumulative visibility,
            # and verify both sides of the actual native capture.
            viewer.update_frame_visibility()
            viewer.server.flush()
            expected = [j<=index for j in range(len(data["frames"]))]
            def stable():
                return (int(viewer.gui_timestep.value)==index and not viewer.fourd
                        and [bool(node.visible) for node in viewer.frame_nodes]==expected
                        and all(np.allclose(np.asarray(getattr(camera,name)),camera_state[name],atol=1e-6,rtol=1e-6)
                                for name in ("position","look_at","up_direction","fov")))
            if not stable():
                raise RuntimeError("Native cumulative frame visibility differs before capture")
            rendered = get_native_render(camera,timeout=max(.1,min(15,deadline-time.monotonic())))
            if not stable():
                raise RuntimeError("Native frame visibility or view changed during capture")
            if rendered is None:
                raise RuntimeError("Native WebGL returned no image; RGB fallback is forbidden")
            pixels = np.asarray(rendered)
            if pixels.dtype != np.uint8 or pixels.shape not in ((800,800,3),(800,800,4)):
                raise RuntimeError("Native WebGL returned an invalid image")
            path = destination / f"{index:06d}.png"
            Image.fromarray(pixels[:,:,:3]).save(path)
            manifest["outputs_sha256"][str(path.relative_to(output))] = sha256(path)
            manifest["capture"]["frames"].append({"frame_id": frame["frame_id"], "timestamp_ns": frame["timestamp_ns"],
                                                   "visible_frame_indices": list(range(index+1)),
                                                   "research_route_visible": bool(len(data["states"][index]["research_path"])) and overlays.research})
            write_manifest()
            print(json.dumps({"status":"capturing","saved_sample":index+1,
                              "expected_sampled_frames":len(data["frames"]),
                              "frame_id":frame["frame_id"],"timestamp_ns":frame["timestamp_ns"]}),flush=True)
        manifest["capture"]["status"] = "complete"
    except BaseException as error:
        manifest["capture"].update(status="failed", error=str(error))
        raise
    finally:
        viewer.gui_timestep.value = before
        viewer.fourd = old_fourd
        viewer.gui_timestep.disabled = False
        viewer.vis_threshold_slider.disabled = threshold_disabled
        for handle,disabled in disabled_controls:
            handle.disabled = disabled
        overlays.tick()
        write_manifest()


class NativeLease:
    """One native adapter per pinned checkout/port, held until child cleanup."""
    def __init__(self, source, port):
        self.path = Path(source).parent / f".{Path(source).name}.native-viewer-port-{port}.lock"
        self.file = None

    def acquire(self):
        if os.name != "posix":
            raise RuntimeError("The bounded native preview requires the Linux flock runtime")
        import fcntl
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps({"pid": os.getpid(), "port_lock": str(self.path)}) + "\n")
            handle.flush()
        except BaseException:
            handle.close()
            raise RuntimeError("Another native preview owns this pinned checkout/port")
        self.file = handle

    def close(self):
        if self.file:
            self.file.close()
            self.file = None


def _serve(args, stop, deadline, output_ready):
    if "torch" in sys.modules:
        raise RuntimeError("Run this isolated CPU CLI before importing Torch")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    if len(affinity) != 4:
        raise RuntimeError("The native CPU preview requires an explicit four-CPU affinity")
    source, source_hashes = verify_source(args.source_root)
    if stop.is_set() or time.monotonic() >= deadline:
        raise RuntimeError("Recording deadline expired during source admission")
    data = prepare_inputs(args.run, args.sequence, args.cache, args.research_plan,
                          expected_frames=args.expected_frames)
    if stop.is_set() or time.monotonic() >= deadline:
        raise RuntimeError("Recording deadline expired during saved-array preparation")
    output = output_directory(args.output, data, source, args.research_plan)
    output_ready.set()
    config = {"schema_version":1,"adapter_sha256":sha256(__file__),"source_revision":SOURCE_REVISION,
              "source_root":str(source),"native_source_sha256":source_hashes,"bind_host":"127.0.0.1",
              "saved_join_validator_sha256":sha256(Path(__file__).with_name("export_pipeline_video.py")),
              "port":args.port,"device":"cpu","model":None,"sky_segmentation":False,
              "saved_xyz_changed":False,"saved_extrinsic":"world_to_camera_unchanged",
              "native_use_point_map":True,"processed_rgb_conversion":"NHWC_uint8_to_NCHW_float32_div255",
              "accepted_pixel_min_confidence_inclusive":data["min_confidence"],
              "native_strict_visibility_threshold":data["native_threshold"],
              "invalid_and_padding_display_points":"NaN_on_copy_only",
              "downsample_factor":data["native_downsample"],"max_display_points":MAX_POINTS,
              "point_size_native_units":args.point_size,"max_runtime_seconds":args.max_runtime,
              "retained_native_geometry_bytes":data["retained_native_geometry_bytes"],
              "actual_cpu_affinity":affinity,"native_cameras_hidden":True,
              "expected_sampled_frames":args.expected_frames,"admission":"exact_complete_recording_1_to_58_saved_samples",
              "capture_output":"native_WebGL_PNG_only","exit_after_capture":args.exit_after_capture,
              "optional_added_overlays":{"semantic_positive_evidence":args.semantic_evidence,
                                        "saved_fused_voxel_cuboids":args.fused_voxels,
                                        "final_only_assumed_research_route":bool(args.research_plan)},
              "timeline":data["timeline"],"renderer":"LingBot-map official PointCloudViewer via Viser"}
    manifest = {"schema_version":1,"status":"initializing","run_dir":str(data["run_dir"]),
                "sequence_path":str(data["sequence_path"]),"input_sha256":data["input_sha256"],
                "geometry_identity":data["geometry"],"original_planning":data["planning"],
                "research_plan":{key:value for key,value in (data["research_plan"] or {}).items() if key!="validated_path"},
                "valid_point_counts":data["valid_point_counts"],"outputs_sha256":{},"capture":{"status":"not_requested"},
                "coverage":[{"frame_id":f["frame_id"],"timestamp_ns":f["timestamp_ns"],
                             "timestamp_provenance":f["source"].get("timestamp_provenance",{}),
                             "disposition":f["disposition"],"semantic_status":(f["semantic"] or {}).get("status","not_recorded"),
                             "queries":[{"query_id":q.get("query_id"),"prompt":q.get("original_phrase"),"status":q.get("status"),
                                         "instances":len(q.get("instances",[])),"error":q.get("error")}
                                        for q in (f["semantic"] or {}).get("queries",[])]} for f in data["frames"]]}
    write_json(output/"viewer_config.json",config)
    manifest["outputs_sha256"]["viewer_config.json"] = sha256(output/"viewer_config.json")
    manifest_lock = threading.RLock()
    def write_manifest():
        with manifest_lock:
            write_json(output/"manifest.json",manifest)
    write_manifest()
    lease = NativeLease(source, args.port)
    viewer = playback = None
    capture_thread = None
    capture_lock = threading.Lock()
    capture_errors = []
    try:
        lease.acquire()
        sys.path.insert(0,str(source))
        module = importlib.import_module("lingbot_map.vis.point_cloud_viewer")
        if not Path(module.__file__).resolve().is_relative_to(source):
            raise RuntimeError("Imported viewer is outside the pinned source checkout")
        module.torch.set_num_threads(4)
        cuda_available = bool(module.torch.cuda.is_available())
        if cuda_available:
            raise RuntimeError("CUDA is visible in the isolated CPU native preview")
        viewer = construct_loopback_viewer(module,data["native_pred"],args.port,data["native_downsample"],
                                          data["native_threshold"],args.point_size)
        for name,filename in (("screenshot_path","unused_screenshot.png"),("glb_output_path","unused_scene.glb"),
                              ("video_output_path","unused_upstream_video.mp4")):
            handle = getattr(viewer,name)
            handle.value = str(output/filename)
            handle.disabled = True
        overlays = SavedOverlays(viewer,data,semantics=args.semantic_evidence,voxels=args.fused_voxels,
                                 research=bool(args.research_plan))
        playback = NativePlayback(viewer,module,stop,overlays.tick)
        playback.start()
        actual_display_points = sum(len(points) for points in viewer.vis_pts_list)
        if not 0 < actual_display_points <= MAX_POINTS:
            raise RuntimeError("Actual native accepted point count violates the 300000-point bound")
        display_pose = native_display_pose(viewer)
        config.update(actual_cuda_available=cuda_available, actual_bind_host=viewer.server.get_host(),
                      actual_bind_port=viewer.server.get_port(), exclusive_lease=str(lease.path),
                      display_only_camera=display_pose,
                      semantic_overlay_sampling=overlays.semantic_sampling,
                      overlay_counts={"final_positive_candidate_voxel_centers":len(data["states"][-1]["positive_indices"]),
                                      "final_hazard_voxel_centers":len(data["states"][-1]["hazard_indices"]),
                                      "fused_voxel_cuboids":len(data["final_voxel_indices"]) if args.fused_voxels else 0},
                      actual_display_point_count=actual_display_points,
                      native_runtime={"python":sys.version.split()[0],"numpy":np.__version__,
                                      "torch":str(module.torch.__version__),
                                      "viser":str(getattr(module.viser,"__version__","unavailable"))})
        config["observed_cpu_rss_bytes_after_native_initialization"] = cpu_rss_bytes()
        write_json(output/"viewer_config.json",config)
        manifest["outputs_sha256"]["viewer_config.json"] = sha256(output/"viewer_config.json")
        research = data["research_plan"]
        prompt = "; ".join(sorted({str(query.get("original_phrase","")) for frame in data["frames"]
                                   for query in (frame["semantic"] or {}).get("queries",[])})) or "No semantic query recorded"
        facts = ("### LingBot-map native reconstruction\n"
                 "Dense RGB points use the exact saved reconstruction. Blue is the saved camera trajectory.\n\n"
                 f"{len(data['frames'])} sampled frames; model crop RGB; {html.escape(str(data['geometry']['units']))} native units. "
                 "Unsampled moments have no new geometry or masks. Original scale, ground direction and robot remain as recorded.\n\n"
                 f"Prompts: {html.escape(prompt)}. Original planning: {html.escape(str(data['planning'].get('availability','unknown')))}.\n\n")
        if args.semantic_evidence:
            facts += "Our added overlay: turquoise/orange voxel-center points mean positive candidate/hazard evidence; never traversability or free space.\n\n"
            if overlays.semantic_sampling["union_cells"]>MAX_SEMANTIC_CENTERS:
                facts += "The evidence overlay shows a fixed bounded subset; all original semantic evidence remains saved.\n\n"
        if args.fused_voxels:
            facts += "Our added overlay: wireframe cuboids are the saved fused voxel grid, separate from native dense points.\n\n"
        if research:
            facts += RESEARCH_LABEL+". Purple route is geometry-only and appears only on the final saved cumulative frame. "
            facts += "Research status: "+html.escape(str(research.get("status")))+". SAM masks do not guide this path.\n\n"
            assumptions = research.get("assumptions", {})
            plane = research.get("ground_plane") or {}
            method = plane.get("method")
            ground = ("declared assumed plane" if method=="explicit_user_assumed_plane" else
                      "observed plane fitted under an assumed camera-up direction" if method=="camera_up_prior_constrained_observed_plane_ransac" else
                      "no ground plane available")
            endpoints = (research.get("endpoints") or {}).get("selection")
            endpoint_text = ("automatic demonstration cells on the largest observed component" if
                             endpoints=="automatic_demonstration_endpoints_on_largest_observed_traversable_component" else
                             "first/last camera positions projected to observed support, as assumed endpoints" if
                             endpoints=="camera_first_last_xy_projected_to_observed_support" else "no saved selected endpoints")
            facts += "Assumptions: "+html.escape(str(assumptions.get("metres_per_native_unit","unknown")))+" metres per native unit; "
            facts += html.escape(str(assumptions.get("robot",{})))+" robot; "+ground+". Endpoints: "+endpoint_text+". Gravity and floor identity are unverified.\n\n"
        viewer.server.gui.add_markdown(facts)
        overlays.frame_facts = viewer.server.gui.add_markdown("Saved frame status is initializing")
        overlays.previous = None
        overlays.tick()
        status = viewer.server.gui.add_text("Native capture",initial_value="Ready; an open browser is required",disabled=True)
        capture_button = viewer.server.gui.add_button("Capture native sampled frames")
        def begin_capture(client):
            nonlocal capture_thread
            if not capture_lock.acquire(blocking=False):
                return
            if (output/"native_frames").exists():
                capture_lock.release()
                status.value = "This fresh output already has captures; use a new output for another recording"
                return
            def worker():
                try:
                    if stop.wait(.8):
                        raise RuntimeError("Native capture interrupted before browser initialization")
                    status.value = "Capturing official WebGL; preserve this view until complete"
                    capture_native(viewer,client,data,output,manifest,write_manifest,stop,deadline,overlays)
                    write_manifest()
                    status.value = "Native capture complete"
                except BaseException as error:
                    capture_errors.append(error)
                    status.value = "Native capture failed: "+str(error)
                    stop.set()
                finally:
                    capture_lock.release()
            capture_thread = threading.Thread(target=worker,name="lingbot-native-capture",daemon=True)
            capture_thread.start()
        @capture_button.on_click
        def capture(event):
            begin_capture(event.client)
        @viewer.server.on_client_connect
        def connected(client):
            for name in ("up_direction", "position", "look_at", "fov"):
                setattr(client.camera, name, display_pose[name])
            if args.capture_on_connect:
                begin_capture(client)
        manifest["status"] = "serving"
        manifest["local_url"] = f"http://127.0.0.1:{args.port}/"
        write_manifest()
        print(json.dumps({"status":"serving","local_url":manifest["local_url"],"output":str(output)},ensure_ascii=False),flush=True)
        while not stop.wait(.2) and time.monotonic()<deadline:
            if playback.error:
                raise RuntimeError("Native animation failed") from playback.error
            if (args.exit_after_capture and capture_thread is not None and not capture_thread.is_alive()
                    and manifest["capture"]["status"] == "complete"):
                manifest["shutdown_reason"] = "capture_complete_worker_finished"
                stop.set()
                break
        if capture_errors:
            raise RuntimeError("Native capture failed") from capture_errors[0]
        if capture_thread:
            capture_thread.join(timeout=2)
        if capture_thread and capture_thread.is_alive():
            raise RuntimeError("Native capture did not finish before the preview was closed")
        if manifest["capture"]["status"] != "complete":
            raise RuntimeError("Recording closed without complete browser WebGL captures; no fallback is allowed")
        manifest["status"] = "complete"
        manifest["observed_cpu_rss_bytes_after_capture"] = cpu_rss_bytes()
        manifest.setdefault("shutdown_reason", "bounded_runtime" if time.monotonic()>=deadline else "signal_or_stop")
    except BaseException as error:
        manifest.update(status="failed",error=str(error))
        raise
    finally:
        stop.set()
        cleanup_errors = []
        if playback:
            try:
                playback.close()
            except Exception as error:
                cleanup_errors.append(str(error))
        if viewer:
            try:
                viewer.server.stop()
            except Exception as error:
                cleanup_errors.append(str(error))
            finally:
                try:
                    viewer.recording_controls.restore_methods()
                except Exception as error:
                    cleanup_errors.append(str(error))
        if capture_thread:
            capture_thread.join(timeout=2)
        lease.close()
        if cleanup_errors:
            manifest.update(status="failed",cleanup_errors=cleanup_errors)
        write_manifest()
    if manifest["status"]=="failed":
        raise RuntimeError("Native viewer cleanup failed")
    print(json.dumps({"status":"closed","capture_status":manifest["capture"]["status"],
                      "expected_sampled_frames":args.expected_frames,"output":str(output),
                      "shutdown_reason":manifest["shutdown_reason"]}),flush=True)
    return manifest


def inherited_controller(args):
    """Admit an inherited pipe read end and an already-held controller lock."""
    pipe_fd, lock_fd = args.parent_pipe_fd, args.controller_lock_fd
    if (pipe_fd is None) != (lock_fd is None):
        raise ValueError("Parent pipe and controller lock descriptors must be supplied together")
    if pipe_fd is None:
        return None
    if os.name!="posix" or pipe_fd<3 or lock_fd<3 or pipe_fd==lock_fd:
        raise ValueError("Controller mode requires two distinct inherited Linux descriptors≥3")
    import fcntl
    if not stat.S_ISFIFO(os.fstat(pipe_fd).st_mode) or fcntl.fcntl(pipe_fd,fcntl.F_GETFL)&os.O_ACCMODE != os.O_RDONLY:
        raise ValueError("Parent liveness descriptor must be an inherited pipe read end")
    if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
        raise ValueError("Controller exclusion descriptor must be an inherited regular lock file")
    lock_path = Path(f"/proc/self/fd/{lock_fd}").resolve(strict=True)
    with lock_path.open("a") as independent:
        try:
            fcntl.flock(independent,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise ValueError("Inherited controller exclusion lock is not already held")
    fcntl.flock(lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    return pipe_fd


def serve(args):
    """The deadline covers source checks, array preparation, browser wait and capture."""
    stop, finished, output_ready = threading.Event(), threading.Event(), threading.Event()
    deadline = time.monotonic()+args.max_runtime
    parent_pid = os.getppid()
    pipe_fd = inherited_controller(args)
    source = args.source_root.resolve()
    expected_lock = source.parent.parent / "outputs/.pipeline2_video_execution.lock"
    if args.execution_lock_path.resolve()!=expected_lock.resolve():
        raise ValueError("Native model-exclusion lock must be the isolated Pipeline2 execution lock")
    execution_lease = NativeLease(source,args.port)
    execution_lease.path = expected_lock
    handlers = {number:signal.signal(number,lambda *_:stop.set())
                for number in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
    def watchdog():
        while not finished.wait(.05):
            parent_gone = os.getppid()!=parent_pid
            pipe_closed = False
            if pipe_fd is not None:
                readable,_,_ = select.select([pipe_fd],[],[],0)
                pipe_closed = bool(readable) and os.read(pipe_fd,4096)==b""
            if stop.is_set() or parent_gone or pipe_closed or time.monotonic()>=deadline:
                stop.set()
                if not finished.wait(5):
                    if output_ready.is_set():
                        write_json(Path(args.output)/"shutdown_failure.json",
                                   {"status":"failed","reason":"whole_recording_deadline_or_parent_loss_cleanup_timeout"})
                    os._exit(124)
                return
    watcher = threading.Thread(target=watchdog,name="native-recording-whole-stage-bound",daemon=True)
    watcher.start()
    try:
        execution_lease.acquire()
        return _serve(args,stop,deadline,output_ready)
    finally:
        execution_lease.close()
        finished.set()
        watcher.join(timeout=.2)
        for number,handler in handlers.items():
            signal.signal(number,handler)
        for descriptor in (args.parent_pipe_fd,args.controller_lock_fd):
            if descriptor is not None:
                os.close(descriptor)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root",required=True,type=Path,help="Exact pinned src/lingbot-map checkout")
    parser.add_argument("--run",required=True,type=Path)
    parser.add_argument("--sequence",required=True,type=Path)
    parser.add_argument("--cache",type=Path,help="Relocated exact saved cache directory")
    parser.add_argument("--research-plan",type=Path,help="Separate assumption-labelled final-map illustration")
    parser.add_argument("--output",required=True,type=Path,help="Fresh separate provenance/capture directory")
    parser.add_argument("--port",type=int,default=8080)
    parser.add_argument("--max-runtime",type=int,default=900,help="Hard preview lifetime, at most900seconds")
    parser.add_argument("--expected-frames",type=int,required=True,help="Exact complete saved recording count, within1–58")
    parser.add_argument("--execution-lock-path",type=Path,required=True,help="Isolated outputs/.pipeline2_video_execution.lock")
    parser.add_argument("--parent-pipe-fd",type=int,help="Inherited controller liveness pipe read end; paired with controller-lock-fd")
    parser.add_argument("--controller-lock-fd",type=int,help="Inherited already-held controller exclusion flock descriptor")
    parser.add_argument("--point-size",type=float,default=.001,help="Native display size only; no scale calibration")
    parser.add_argument("--semantic-evidence",action="store_true",help="Our positive-evidence voxel-center overlay")
    parser.add_argument("--fused-voxels",action="store_true",help="Our optional actual saved-grid wireframe cuboids")
    parser.add_argument("--capture-on-connect",action="store_true",help="Capture all saved frames on first browser connection")
    parser.add_argument("--exit-after-capture",action="store_true",help="Close cleanly once the PNG capture worker has finished")
    args = parser.parse_args(argv)
    if not 1024<=args.port<=65535 or not 1<=args.max_runtime<=MAX_SECONDS or not 1<=args.expected_frames<=MAX_FRAMES:
        parser.error("Port1024–65535; lifetime1–900seconds; exact complete recording1–58frames")
    if not math.isfinite(args.point_size) or not .00001<=args.point_size<=.1:
        parser.error("Native display point size must be within .00001–.1")
    try:
        serve(args)
    except Exception as error:
        print("Native saved-cache viewer failed: "+str(error),file=sys.stderr)
        return 1
    return 0


if __name__=="__main__":
    raise SystemExit(main())
