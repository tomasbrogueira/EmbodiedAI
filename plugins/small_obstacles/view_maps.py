#!/usr/bin/env python3
"""Compare existing map exports in Viser without modifying the mapping pipeline."""

import argparse
from functools import lru_cache
from pathlib import Path
import sys
import threading
import time

import numpy as np
import viser

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "orchestrator"))
from view_ply import load_ply


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("maps", type=Path, nargs="+", help="PLY files or directories to search recursively")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    # Two float32 XYZ + uint8 RGB clouds must fit the Jupyter proxy's
    # default 10 MiB WebSocket message limit, including Viser metadata.
    parser.add_argument("--max-points", type=int, default=200_000)
    args = parser.parse_args()
    if args.max_points < 0 or not 1 <= args.port <= 65535:
        parser.error("Use a non-negative point limit and a port between 1 and 65535.")
    paths = sorted({p.resolve() for item in args.maps for p in
                    (item.rglob("*.ply") if item.is_dir() else [item])})
    if not paths or any(not p.is_file() for p in paths):
        parser.error("No valid PLY files found.")
    labels = {f"{i + 1}: {p.parent.parent.name}/{p.parent.name}/{p.name}": p
              for i, p in enumerate(paths)}
    names = list(labels)

    @lru_cache(maxsize=2)
    def load(name):
        points, colors = load_ply(labels[name], args.max_points)
        lo, hi = np.percentile(points, [2, 98], axis=0)
        center = (lo + hi) / 2
        extent = max(float(np.max(hi - lo)), 1e-3)
        return points, colors, center, extent

    server = viser.ViserServer(host=args.host, port=args.port, label="LingBot map comparison")
    server.scene.set_up_direction("+z")
    server.gui.add_markdown("Compare accumulated maps from whole videos. Each map uses the existing frame selection. Controls are shared between connected viewers.")
    left = server.gui.add_dropdown("Map A", names, initial_value=names[0])
    right = server.gui.add_dropdown("Map B", names, initial_value=names[min(1, len(names) - 1)])
    layout = server.gui.add_dropdown("Layout", ["Side by side", "A only", "B only", "Overlay (shared coordinates)"])
    color_mode = server.gui.add_dropdown("Appearance", ["RGB + obstacle markings", "Neutral geometry"])
    size = server.gui.add_slider("Point size / scene extent", min=0.0001, max=0.005, step=0.0001, initial_value=0.001)
    reset = server.gui.add_button("Reset camera")
    info = server.gui.add_markdown("")
    server.gui.add_markdown("Neutral geometry hides all color markings; original RGB under red points is absent from the exported PLY. Side-by-side views are centered for display, without changing scale or files. Overlay is meaningful only for maps with shared coordinates. Red RGB values alone are not an authoritative obstacle label.")
    lock = threading.RLock()
    handles = []
    camera_target = np.zeros(3)
    camera_extent = 1.0

    def reset_camera(_=None):
        direction = np.array([1., -1., .8])
        direction /= np.linalg.norm(direction)
        for camera in [server.initial_camera] + [c.camera for c in server.get_clients().values()]:
            camera.look_at = camera_target
            camera.position = camera_target + direction * camera_extent * 2.5
            camera.near = max(1e-5, camera_extent * 1e-5)
            camera.far = max(1000., camera_extent * 20)

    def refresh(_=None):
        nonlocal camera_target, camera_extent
        with lock:
            try:
                chosen = [("A", left.value)] if layout.value == "A only" else [("B", right.value)] if layout.value == "B only" else [("A", left.value), ("B", right.value)]
                loaded = [(slot, name, load(name)) for slot, name in chosen]
                extent = max(item[2][3] for item in loaded)
                for handle in handles:
                    handle.remove()
                handles.clear()
                lower, upper, text = [], [], []
                for i, (slot, name, (points, colors, center, _)) in enumerate(loaded):
                    position = np.zeros(3)
                    if layout.value == "Side by side":
                        position = -center + np.array([(i - .5) * extent * 1.5, 0., 0.])
                    display_colors = colors if color_mode.value == "RGB + obstacle markings" else np.full_like(colors, 180)
                    handles.append(server.scene.add_point_cloud(f"/map_{slot}", points=points, colors=display_colors, position=position, point_size=extent * size.value, point_shape="circle", precision="float32"))
                    lower.append(points.min(axis=0) + position)
                    upper.append(points.max(axis=0) + position)
                    text.append(f"**{slot}:** {name} — {len(points):,} displayed points")
                lo, hi = np.min(lower, axis=0), np.max(upper, axis=0)
                camera_target = (lo + hi) / 2
                camera_extent = max(float(np.max(hi - lo)), 1e-3)
                info.content = "\n\n".join(text)
            except Exception as exc:
                info.content = f"Could not load map: {exc}"
                print(f"Map load failed: {exc}", flush=True)

    for control in (left, right, layout, color_mode, size):
        control.on_update(refresh)
    reset.on_click(reset_camera)
    refresh()
    reset_camera()
    print(f"Comparison viewer ready on {args.host}:{args.port}; {len(paths)} maps available.", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
