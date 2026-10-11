#!/usr/bin/env python3
"""Viser view of a running incremental map (bounded whole-history sampling)."""
import argparse
import json
from pathlib import Path
import time
import sys

import numpy as np
import viser

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from plugins.small_obstacles.scene_pose import PoseOverlay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--port', type=int, default=8082)
    args = parser.parse_args()
    server = viser.ViserServer(host='0.0.0.0', port=args.port, label='Incremental LingBot obstacle map')
    server.scene.set_up_direction('+z')
    status = server.gui.add_markdown('Waiting for the first mapped frame / model loading…')
    server.gui.add_markdown('Live display samples the whole accumulated map, up to the configured point limit. The full map is saved as obstacle_map.ply. Red points are mask observations; naturally red surfaces can have the same RGB.')
    mtime, initialized = None, False
    overlay = PoseOverlay(server)
    extent = 1.
    try:
        while True:
            summary = args.directory / 'status.json'
            if summary.exists():
                data = json.loads(summary.read_text())
                status.content = '\n\n'.join(f'**{k}:** {data.get(k, "—")}' for k in
                    ('status', 'loading_model', 'source_time', 'geometry_frames', 'mapped_frames',
                     'geometry_hz', 'mapping_hz', 'kv_cache_window', 'gpu_sampled_peak_gib',
                     'gpu_min_sampled_free_gib', 'prompts', 'points', 'marked_points', 'error'))
            path = args.directory / 'display.npz'
            if path.exists() and path.stat().st_mtime_ns != mtime:
                current = path.stat().st_mtime_ns
                with np.load(path, allow_pickle=False) as data:
                    points, colors = data['points'].copy(), data['colors'].copy()
                if len(points):
                    lo, hi = np.percentile(points, [2, 98], axis=0)
                    extent = max(float(np.max(hi - lo)), 1e-3)
                    server.scene.add_point_cloud('/live_map', points=points, colors=colors,
                        point_size=extent * .001, precision='float32', point_shape='circle')
                    if not initialized:
                        center = (lo + hi) / 2
                        server.initial_camera.position = center + np.array([1., -1., .8]) * extent * 2.5
                        server.initial_camera.look_at = center
                        for client in server.get_clients().values():
                            client.camera.position = server.initial_camera.position
                            client.camera.look_at = server.initial_camera.look_at
                        initialized = True
                mtime = current
            if summary.exists():
                overlay.update(data.get('camera_pose'),extent)
            time.sleep(.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == '__main__':
    main()
