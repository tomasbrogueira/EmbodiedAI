"""Inspect each recording's unchanged map, camera estimates and verified route."""

import argparse
import threading
import time
from pathlib import Path

import numpy as np
from visualization.dataset_status import read as read_json, FEASIBILITY_PENDING, adapter_hash, cost_map_hash, outcome, reviewed_output


def signature(paths):
    result = []
    for path in paths:
        try:
            stat = path.stat()
            result.append((stat.st_size, stat.st_mtime_ns))
        except OSError:
            result.append(None)
    return tuple(result)


def palette(cost, limits):
    colours = np.full((len(cost), 3), (150, 155, 165), dtype=np.uint8)
    valid = np.isfinite(cost)
    low, high = limits
    t = np.clip((cost[valid] - low) / max(high - low, 1e-12), 0, 1)
    stops = np.array(((30, 70, 240), (0, 220, 245), (255, 225, 40), (235, 50, 40)))
    colours[valid] = np.column_stack([
        np.interp(t, np.linspace(0, 1, len(stops)), stops[:, i]) for i in range(3)
    ]).astype(np.uint8)
    return colours


def colour_view(view, mode):
    if mode == "Source":
        return view["source"], "**Source:** recorded appearance, not traversal costs or steepness."
    if mode == "Steepness":
        return view["steepness_colours"], (
            "**Steepness:** blue → cyan → yellow → red: **0° flat to 90° vertical**, relative to the estimated ground. "
            "Grey: unknown slope. This shows surface tilt only, independent of obstacles and traversal cost."
            if view["slope_sample"] else "**Steepness:** no valid slope estimates are available; grey means unknown slope.")
    if mode != "Cost map":
        raise ValueError("Unknown colour mode")
    low, high = view["limits"]
    legend = (f"**Cost colours:** blue → cyan → yellow → red: low to high finite traversal cost.\n\n"
              f"Range **{low:.5g}–{high:.5g}** (clipped). Red is costly but traversable. Grey: **unavailable** (infinite cost, insufficient evidence or pending).\n\n"
              f"Showing {view['sampled_rows']:,} evenly sampled original map rows. "
              "Display sampling does not change the planning graph.") if view["finite_sample"] else (
                  FEASIBILITY_PENDING if view["stale_costs"] else
                  "Grey: unavailable (infinite cost, insufficient evidence or pending). No finite traversal cost is available.")
    return view["cost_colours"], legend


def load_view(video, report, stale_costs=False):
    output = Path(video["output"])
    data = output / "data"
    points = np.load(data / "points.npy", mmap_mode="r")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Map must contain original XYZ point rows")
    rows = np.linspace(0, len(points) - 1, min(250_000, len(points)), dtype=np.int64)
    xyz = np.asarray(points[rows])
    valid = np.isfinite(xyz).all(axis=1)
    rows, xyz = rows[valid], xyz[valid]
    if len(points) and not len(xyz):
        raise ValueError("No finite map points are available yet")
    source = np.full((len(rows), 3), 155, dtype=np.uint8)
    if (data / "colours.npy").exists():
        colours = np.load(data / "colours.npy", mmap_mode="r")
        if colours.ndim != 2 or colours.shape[0] != len(points) or colours.shape[1] < 3:
            raise ValueError("Source colours do not match original point rows")
        source = np.asarray(colours[rows, :3], dtype=np.uint8)
    slopes = np.full(len(rows), np.nan)
    if (data / "slope.npy").exists():
        slope = np.load(data / "slope.npy", mmap_mode="r")
        if slope.shape != (len(points),) or slope.dtype.kind not in "iuf":
            raise ValueError("Slopes do not match original point rows")
        slopes = np.asarray(slope[rows], dtype=float)
        slopes[(slopes < 0) | (slopes > np.pi / 2)] = np.nan
    costs = np.full(len(rows), np.nan)
    cost = None
    if not stale_costs and (data / "cost.npy").exists():
        cost = np.load(data / "cost.npy", mmap_mode="r")
        if cost.shape != (len(points),):
            raise ValueError("Costs do not match original point rows")
        costs = np.asarray(cost[rows])
    finite = costs[np.isfinite(costs)]
    limits = None if stale_costs else report.get("cost_range")
    if isinstance(limits, dict):
        limits = [limits.get("min"), limits.get("max")]
    try:
        limits = np.asarray(limits, dtype=float).reshape(2)
        if not np.isfinite(limits).all() or limits[1] < limits[0]:
            raise ValueError("Invalid colour range")
    except (ValueError, TypeError):
        limits = np.percentile(finite, [5, 95]) if len(finite) else np.array([0., 1.])
    cameras = np.empty((0, 3))
    if (data / "cameras.npy").exists():
        cameras = np.array(np.load(data / "cameras.npy", mmap_mode="r"))
        if cameras.ndim != 2 or cameras.shape[1] != 3:
            raise ValueError("Camera positions must have XYZ columns")
    selected = {} if stale_costs else (report.get("selected") or {})
    start, goal = selected.get("start"), selected.get("goal")
    segment = None
    if start and goal:
        first, last = int(start["frame_id"]), int(goal["frame_id"])
        if not 0 <= first <= last < len(cameras):
            raise ValueError("Selected video frames do not match camera rows")
        segment = (first, last)
        shown_cameras = cameras[first:last + 1]
    else:
        shown_cameras = cameras
    route = np.empty((0, 3))
    partial = not stale_costs and report.get("status") == "partial"
    route_note = FEASIBILITY_PENDING if stale_costs else ""
    if not stale_costs and (output / "route.npz").exists():
        if report.get("status") in ("ok", "degenerate_route", "partial") and report.get("optimal") is True and report.get("correspondence_verified") is True:
            with np.load(output / "route.npz") as saved:
                ids, route = saved["ids"], saved["positions"]
            if ids.ndim != 1 or ids.dtype.kind not in "iu" or len(ids) == 0:
                raise ValueError("Route must identify original point rows")
            if (ids < 0).any() or (ids >= len(points)).any() or not np.array_equal(route, points[ids]):
                raise ValueError("Route positions do not match their original point rows")
            reached = report.get("reached") if partial else goal
            if partial and (report.get("goal_reached") is not False or
                            report.get("optimality_scope") != "nearest_reachable_endpoint" or not reached):
                raise ValueError("Partial route must certify an unreachable goal and nearest reachable endpoint")
            if not segment or ids[0] != int(start["id"]) or ids[-1] != int(reached["id"]):
                raise ValueError("Route endpoints do not match the fixed start and reported destination")
            if partial and (ids[-1] == int(goal["id"]) or not np.array_equal(route[-1], reached.get("ground_point"))):
                raise ValueError("Partial endpoint does not match its original map row")
            if cost is None or not np.isfinite(cost[ids]).all():
                raise ValueError("Route contains an unavailable point")
        else:
            route_note = "Route withheld until optimality and point correspondence are verified."
    up = np.array([0., -1., 0.])
    reference_available = False
    if (data / "reference.npy").exists():
        reference = np.asarray(np.load(data / "reference.npy"), dtype=float).reshape(3)
        norm = np.linalg.norm(reference)
        if np.isfinite(reference).all() and norm > 0:
            up, reference_available = reference / norm, True
    endpoints = []
    if segment:
        for label, endpoint in (("start", start), ("goal", goal)):
            row, frame = int(endpoint["id"]), int(endpoint["frame_id"])
            if not 0 <= row < len(points):
                raise ValueError("Ground endpoint must identify an original point row")
            ground, camera = np.asarray(points[row], dtype=float), cameras[frame]
            mode = endpoint.get("association_mode", "beneath_camera")
            if mode == "visible_ground":
                frame_ids = np.load(data / "frame_ids.npy", mmap_mode="r")
                if frame_ids.shape != (len(points),) or int(frame_ids[row]) != frame:
                    raise ValueError("Visible ground endpoint was not observed in its associated video frame")
            if endpoint.get("ground_point") is not None and not np.array_equal(ground, endpoint["ground_point"]):
                raise ValueError("Ground endpoint position differs from its original map row")
            delta = ground - camera
            drop = float(-delta @ up)
            endpoints.append(dict(label=label, frame=frame, mode=mode, ground=ground,
                                  lateral=float(np.linalg.norm(delta + drop * up)),
                                  drop=drop, distance=float(np.linalg.norm(delta))))
    # Frame the main observed scene; isolated depth outliers remain in the map.
    context = np.concatenate((shown_cameras, route, np.asarray([e["ground"] for e in endpoints]).reshape(-1, 3)))
    context = context[np.isfinite(context).all(axis=1)]
    box = np.percentile(xyz, [1, 99], axis=0) if len(xyz) else (
        np.stack((context.min(axis=0), context.max(axis=0))) if len(context) else np.zeros((2, 3)))
    if len(context):
        box[0] = np.minimum(box[0], context.min(axis=0))
        box[1] = np.maximum(box[1], context.max(axis=0))
    provenance = read_json(data / "provenance.json")
    radius = provenance.get("settings", {}).get("radius")
    if not isinstance(radius, (int, float)) or not np.isfinite(radius) or radius <= 0:
        radius = max(float(np.linalg.norm(box[1] - box[0])) / 200, 1e-6)
    return dict(xyz=xyz, source=source, cost_colours=palette(costs, limits), limits=limits,
                steepness_colours=palette(slopes, (0., np.pi / 2)), slope_sample=int(np.isfinite(slopes).sum()),
                finite_sample=len(finite), cameras=shown_cameras, route=route, segment=segment,
                endpoints=endpoints, partial=partial, empty_map=len(points) == 0, stale_costs=stale_costs,
                up=up, reference_available=reference_available, box=box, radius=radius,
                reference_report=provenance.get("reference_report", {}), route_note=route_note, sampled_rows=len(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video", help="Initial video relative path from the manifest")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8094)
    args = parser.parse_args()
    videos = read_json(args.manifest).get("videos", [])
    if not videos:
        raise ValueError("The manifest contains no videos")
    if args.video is not None and args.video not in {v["relative_path"] for v in videos}:
        parser.error("--video must match a relative_path in the manifest")
    from visualization.viser_route_overlay import create_server
    server = create_server(host=args.host, port=args.port)
    lock = threading.RLock()
    state = dict(video=None, signature=None, view=None, handles=[], cloud=None, camera=[], route=[], labels=[])
    server.gui.add_markdown("# Dataset paths\nCyan: **estimated camera trajectory**.\n\nMagenta with white outline: **verified minimum-cost route**. Drawn over the map for visibility, at its exact ground coordinates. A partial route stops at an orange marker; its unreachable goal is red.")
    batch_summary = server.gui.add_markdown("")
    initial = args.video or next((v["relative_path"] for v in videos
                    if (ready := read_json(Path(v["output"]) / "status.json")).get("stage") == "complete"
                    and ready.get("status") == "ok"), videos[0]["relative_path"])
    choices = server.gui.add_dropdown("Video", options=tuple(v["relative_path"] for v in videos), initial_value=initial)
    summary = server.gui.add_markdown("")
    colour_mode = server.gui.add_dropdown("Colour", options=("Source", "Cost map", "Steepness"), initial_value="Cost map")
    camera_toggle = server.gui.add_checkbox("Camera trajectory", initial_value=True)
    route_toggle = server.gui.add_checkbox("Dijkstra route", initial_value=True)
    labels_toggle = server.gui.add_checkbox("Endpoint labels", initial_value=False)
    size = server.gui.add_slider("Point size", min=.01, max=.20, step=.01, initial_value=.04)
    fit_button = server.gui.add_button("Frame scene")
    save_button = server.gui.add_button("Save current view")
    save_note = server.gui.add_markdown("")
    legend = server.gui.add_markdown("")

    def fit(client):
        view = state["view"]
        if client is None or view is None:
            return
        up, box = view["up"], view["box"]
        side = np.eye(3)[np.argmin(np.abs(up))]
        side = side - side.dot(up) * up
        side /= np.linalg.norm(side)
        target = box.mean(axis=0)
        distance = max(float(np.linalg.norm(box[1] - box[0])) * 1.2, view["radius"] * 4)
        with client.atomic():
            client.camera.up_direction = up
            client.camera.position = target + distance * (.8 * up + .6 * side)
            client.camera.look_at = target

    def draw(view, video):
        for handle in state["handles"]:
            handle.remove()
        state.update(handles=[], cloud=None, camera=[], route=[], labels=[], view=view)
        if view is None:
            return
        server.scene.set_up_direction(view["up"])

        def remember(handle, group=None, is_label=False):
            state["handles"].append(handle)
            if group:
                state[group].append(handle)
                handle.visible = camera_toggle.value if group == "camera" else route_toggle.value
            if is_label:
                state["labels"].append((handle, group))
                handle.visible = handle.visible and labels_toggle.value
            return handle

        if len(view["xyz"]):
            state["cloud"] = remember(server.scene.add_point_cloud(
                "/dataset/map", points=view["xyz"], colors=colour_view(view, colour_mode.value)[0],
                point_size=view["radius"] * size.value, precision="float32", point_shape="circle"))
        for group, positions, colour in (("camera", view["cameras"], (35, 225, 255)),
                                           ("route", view["route"], (235, 35, 180))):
            if group == "route" and len(positions) and np.all(positions == positions[0]):
                text = "No progress from start; goal unreachable" if view["partial"] else "Dijkstra: start = goal"
                remember(server.scene.add_label("/dataset/route/zero_length", text=text,
                                               position=positions[0], depth_test=False), group)
            if len(positions) > 1:
                pairs = np.stack((positions[:-1], positions[1:]), axis=1)
                pairs = pairs[np.isfinite(pairs).all(axis=(1, 2))]
                if len(pairs):
                    if group == "route":
                        remember(server.scene.add_line_segments("/dataset/route/outline", points=pairs,
                                 colors=(255, 255, 255), thickness=10, thickness_units="screen"), group)
                    remember(server.scene.add_line_segments(f"/dataset/{group}/line", points=pairs,
                             colors=colour, thickness=4 if group == "camera" else 6, thickness_units="screen"), group)
            if group == "camera" and len(positions):
                for index, label, colour in ((0, "start", (90, 235, 125)), (-1, "end", (255, 165, 55))):
                    position = positions[index]
                    if not np.isfinite(position).all():
                        continue
                    remember(server.scene.add_icosphere(f"/dataset/{group}/{label}", radius=view["radius"] * .2,
                                                       color=colour, position=position), group)
                    if group == "camera":
                        remember(server.scene.add_label(f"/dataset/{group}/{label}_label", text=f"Camera {label}",
                                                       position=position, depth_test=False), group, is_label=True)
        for endpoint in view["endpoints"]:
            label, position = endpoint["label"], endpoint["ground"]
            colour = (90, 235, 125) if label == "start" else (255, 165, 55)
            if view["partial"] and label == "goal":
                colour = (235, 50, 40)
            text = "Start" if label == "start" else "Goal unreachable" if view["partial"] else "Goal"
            remember(server.scene.add_icosphere(f"/dataset/ground/{label}", radius=view["radius"] * .2,
                                               color=colour, position=position), "route")
            remember(server.scene.add_label(f"/dataset/ground/{label}_label", text=text,
                                           position=position, depth_test=False), "route", is_label=True)
        if view["partial"] and len(view["route"]):
            position = view["route"][-1]
            remember(server.scene.add_icosphere("/dataset/ground/reached", radius=view["radius"] * .2,
                                               color=(255, 165, 55), position=position), "route")
            remember(server.scene.add_label("/dataset/ground/reached_label", text="Nearest reachable point",
                                           position=position, depth_test=False), "route", is_label=True)

    def refresh(force=False):
        nonlocal videos
        with lock:
            manifest = read_json(args.manifest)
            if manifest.get("videos"):
                videos = manifest["videos"]
            current_hash = cost_map_hash(manifest)
            current_prepare_hash = adapter_hash("dataset_prepare.py")
            current_association_hash = adapter_hash("dataset_path.py")
            counts = {"verified routes": 0, "verified partial routes": 0, "completed without route": 0, "running": 0, "pending": 0, "errors": 0}
            for item in videos:
                item_output = Path(item["output"])
                progress, finished, _ = reviewed_output(item_output, read_json(item_output / "status.json"),
                    read_json(item_output / "report.json"), current_hash, current_prepare_hash, current_association_hash)
                counts[outcome(progress, finished)] += 1
            batch_summary.content = " · ".join(f"**{number}** {name}" for name, number in counts.items())
            options = tuple(v["relative_path"] for v in videos)
            if tuple(choices.options) != options:
                previous = choices.value
                choices.options = options
                choices.value = previous if previous in options else options[0]
            video = next(v for v in videos if v["relative_path"] == choices.value)
            output, data = Path(video["output"]), Path(video["output"]) / "data"
            status, report = read_json(output / "status.json"), read_json(output / "report.json")
            status, report, stale = reviewed_output(output, status, report, current_hash, current_prepare_hash, current_association_hash)
            changed_video = video["id"] != state["video"]
            files = [data / name for name in ("points.npy", "colours.npy", "cost.npy", "slope.npy", "reference.npy", "cameras.npy", "provenance.json")]
            sig = signature(files + [output / "route.npz", output / "report.json"]) + (
                current_hash, current_prepare_hash, current_association_hash, stale, status.get("stage"), status.get("status"))
            load_error = ""
            if force or changed_video or sig != state["signature"]:
                try:
                    first_map = state["view"] is None
                    view = load_view(video, report, stale) if (data / "points.npy").exists() else None
                    draw(view, video)
                    state.update(video=video["id"], signature=sig)
                    if changed_video or first_map or force:
                        for client in server.get_clients().values():
                            fit(client)
                except (OSError, ValueError, KeyError, IndexError) as error:
                    load_error = f"Viewer check: {error}"
                    draw(None, video)
                    state["video"] = video["id"]
            view = state["view"]
            stage, result = status.get("stage", "queued"), status.get("status", "pending")
            lines = [f"**{video['relative_path']}** · SAM: **{video.get('prompt', 'unspecified')}**",
                     f"**{stage} · {result}**", str(status.get("message", ""))]
            start, goal = (report.get("selected") or {}).get("start"), (report.get("selected") or {}).get("goal")
            if view and view["segment"]:
                first, last = view["segment"]
                frames = int(report.get("frame_count", video.get("frame_count", last + 1)))
                interval = f"Frames **{first}–{last}**"
                if start.get("time_seconds") is not None and goal.get("time_seconds") is not None:
                    interval += f" · **{start['time_seconds']:.2f}–{goal['time_seconds']:.2f} s**"
                visible_ground = any(endpoint["mode"] == "visible_ground" for endpoint in view["endpoints"])
                heading = "### Video segment · visible-ground association" if visible_ground else "### Supported video segment"
                if view["partial"]:
                    heading = "### Intended video segment · goal unreachable"
                lines += [heading, interval,
                          f"Omitted: **{first}** initial and **{max(0, frames - last - 1)}** final frames."]
                if visible_ground:
                    lines.append("Visible-ground endpoints are observed in their associated video frames. Routes use ground coordinates; camera-to-ground offsets are not route edges.")
                for endpoint in view["endpoints"]:
                    mode = "visible ground" if endpoint["mode"] == "visible_ground" else "beneath camera"
                    lines.append(f"**{endpoint['label'].capitalize()}: {mode}** · lateral **{endpoint['lateral']:.5g}**, "
                                 f"drop **{endpoint['drop']:.5g}**, 3D offset **{endpoint['distance']:.5g}** map units.")
            elif view and np.isfinite(view["cameras"]).all(axis=1).any():
                lines.append("**Full camera trajectory shown for context; no supported segment selected.**")
            else:
                lines.append("**No camera trajectory is displayed.**")
            if report and report.get("status") not in ("missing_reference", "updated_feasibility_pending"):
                lines.append(f"Endpoint associations: **{report.get('supported_count', 0)} / {report.get('frame_count', video.get('frame_count', '?'))}** poses associated.")
                reasons = report.get("reason_counts") or {}
                if reasons:
                    lines.append("Pose results: " + "; ".join(f"{key.replace('_', ' ')}: {value}" for key, value in reasons.items()))
            if view and len(view["route"]):
                if view["partial"]:
                    lines.append("**Partial route — goal unreachable.** The original start is retained. The route ends at the reachable point closest to the intended goal (orange marker); the intended goal remains red. The cyan trajectory shows the intended video interval, not successful route coverage.")
                    lines.append(f"**Verified minimum-cost route to the nearest reachable point** · {len(view['route']):,} original points · cost **{float(report['total_cost']):.5g}**.")
                elif report.get("status") == "degenerate_route":
                    lines.append("**Start and goal are the same ground point: an optimal zero-length, zero-cost route.** The camera may still move or return to its starting area.")
                else:
                    lines.append(f"**Verified minimum-cost route** · {len(view['route']):,} original points · cost **{float(report['total_cost']):.5g}**.")
                if report.get("route_length") is not None:
                    lines.append(f"Length: **{float(report['route_length']):.5g} map units**.")
            else:
                lines.append("**No verified Dijkstra route displayed.**")
            if view:
                if view["empty_map"]:
                    lines.append("**The map contains no point rows.**")
                lines.append(view["route_note"])
                reference = view["reference_report"]
                window = reference.get("selected_window")
                if view["reference_available"] and window:
                    first_reference, last_reference = (int(Path(name).stem.split("_")[-1]) for name in (window[0], window[-1]))
                    lines.append(f"Ground direction established from video frames **{first_reference}–{last_reference}**.")
                if not view["reference_available"]:
                    surface = video.get("prompt", "ground")
                    explanations = {
                        "missing_floor_masks": f"No {surface} masks were available to establish the ground direction.",
                        "insufficient_reference_frames": f"Too few opening frames contained a reliable {surface} patch to establish the ground direction.",
                        "inconsistent_reference_frames": f"The opening {surface} patches disagreed about the ground direction.",
                    }
                    scanned = reference.get("reference_scan_version")
                    if scanned:
                        explanations["insufficient_reference_frames"] = f"No checked video window established a reliable ground direction. The final checked window had too few reliable {surface} patches."
                        explanations["inconsistent_reference_frames"] = f"No checked video window established a reliable ground direction. The {surface} patches in the final checked window disagreed about the ground direction."
                        lines.append(f"Checked **{reference.get('inspected_frame_count', 0)} frames** across **{reference.get('windows_checked', 0)} windows**.")
                    lines.append(explanations.get(reference.get("reason"), "The ground direction could not be established."))
                    frames = reference.get("frames") or []
                    if frames and all(frame.get("point_count") == 0 for frame in frames):
                        location = "final checked window" if scanned else f"first {len(frames)} frames"
                        lines.append(f"The {location} had **no usable {surface} points** after the existing checks.")
                    elif frames:
                        lines.append(f"{surface.capitalize()} points in the reported window: " + ", ".join(str(frame.get("point_count", "?")) for frame in frames) + ".")
                    lines.append("**Display orientation is arbitrary; no route is inferred.**")
                legend.content = colour_view(view, colour_mode.value)[1]
            else:
                legend.content = ("No map is available for this completed result." if stage == "complete"
                                  else "Map unavailable; see the reported error." if result == "error" or stage == "error" or load_error
                                  else "Map will appear when reconstruction is ready.")
            if load_error:
                lines.append(f"**{load_error}**")
            summary.content = "\n\n".join(line for line in lines if line)

    @choices.on_update
    def selected(_):
        refresh()

    @colour_mode.on_update
    def colours(_):
        with lock:
            if state["cloud"] is not None:
                state["cloud"].colors, legend.content = colour_view(state["view"], colour_mode.value)

    @size.on_update
    def point_size(_):
        with lock:
            if state["cloud"] is not None:
                state["cloud"].point_size = state["view"]["radius"] * size.value

    @camera_toggle.on_update
    def camera_visibility(_):
        with lock:
            for handle in state["camera"]:
                handle.visible = camera_toggle.value
            for handle, group in state["labels"]:
                if group == "camera":
                    handle.visible = camera_toggle.value and labels_toggle.value

    @route_toggle.on_update
    def route_visibility(_):
        with lock:
            for handle in state["route"]:
                handle.visible = route_toggle.value
            for handle, group in state["labels"]:
                if group == "route":
                    handle.visible = route_toggle.value and labels_toggle.value

    @labels_toggle.on_update
    def label_visibility(_):
        with lock:
            for handle, group in state["labels"]:
                handle.visible = labels_toggle.value and (camera_toggle.value if group == "camera" else route_toggle.value)

    @fit_button.on_click
    def whole_map(event):
        with lock:
            fit(event.client)

    @server.on_client_connect
    def connected(client):
        with lock:
            fit(client)

    @save_button.on_click
    def save_view(event):
        from PIL import Image
        with lock:
            if event.client is not None and state["view"] is not None:
                video = next(v for v in videos if v["id"] == state["video"])
                rendered = event.client.camera.get_render(600, 900, timeout=30)
                Image.fromarray(rendered).save(Path(video["output"]) / "preview.png")
                save_note.content = f"Saved **{video['name']}** view as preview.png beside its results."

    refresh(force=True)
    print(f"Dataset viewer ready at http://{args.host}:{args.port}", flush=True)
    while True:
        time.sleep(5)
        try:
            refresh()
        except Exception as error:
            print(f"Viewer refresh: {error}", flush=True)


if __name__ == "__main__":
    main()
