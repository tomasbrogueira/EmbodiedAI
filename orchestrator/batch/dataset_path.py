# Support direct compatibility-script execution without environment mutation.
if __package__ in (None, ''):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
"""Recording artifact adapter around independent planner route algorithms."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree
from path_planner.routes import *
def json_value(value):
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(json_value(value), indent=2), encoding='utf-8')
    os.replace(temporary, path)


def save_route(output, ids, points, total_cost):
    ids = np.asarray(ids, dtype=np.int64)
    temporary = output / 'route.tmp.npz'
    np.savez(temporary, ids=ids, positions=points[ids], total_cost=float(total_cost))
    os.replace(temporary, output / 'route.npz')


def verify_cost_source(provenance, source_path):
    expected = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
    if provenance.get('cost_source_sha256') != expected:
        raise ValueError('Saved costs use a different cost-map source; recompute costs before planning')


def run(args):
    output = args.output.resolve()
    data = output / 'data' if (output / 'data').is_dir() else output
    report = json.loads((output / 'report.json').read_text(encoding='utf-8'))
    if report.get('status') == 'missing_clearance_calibration':
        save_route(output, [], np.empty((0, 3)), np.inf)
        return 0
    started = time.monotonic()
    def update(stage, **details):
        candidate = details.pop('candidate', None)
        if candidate is not None:
            np.savez(output / 'candidate.tmp.npz', **candidate)
            os.replace(output / 'candidate.tmp.npz', output / 'candidate.npz')
        atomic_json(output / 'status.json', {'name': report.get('name'), 'stage': stage, 'status': 'running',
                                            'elapsed_seconds': time.monotonic() - started, **details})
        print(stage, json.dumps(json_value(details)), flush=True)
    try:
        report.update(status='running', optimal=False, route_points=0, total_cost=None,
                      route_length=0, all_edges_verified=False,
                      selected={'start': None, 'goal': None}, start=None, goal=None,
                      reached=None, goal_reached=False, optimality_scope=None)
        atomic_json(output / 'report.json', report)
        save_route(output, [], np.empty((0, 3)), np.inf)
        sys.path.insert(0, str(args.repo / 'path-planning'))
        from path_planner.cost_map import CostMap, _settings
        import path_planner.cost_map as cost_map
        from path_planner.planner import _aligned_costs, find_path
        from path_planner.recording_endpoints import select_endpoint
        update('associating_video_poses')
        provenance = json.loads((data / 'provenance.json').read_text(encoding='utf-8'))
        verify_cost_source(provenance, cost_map.__file__)
        report['cost_source_sha256'] = provenance['cost_source_sha256']
        if not provenance.get('correspondence_verified') or not report.get('correspondence_verified'):
            raise ValueError('Point, cost, mask and video-pose correspondence must be verified first')
        points = np.asarray(np.load(data / 'points.npy', mmap_mode='r'), dtype=np.float64)
        costs = np.load(data / 'cost.npy', mmap_mode='r')
        mask = np.load(data / 'support_mask.npy', mmap_mode='r')
        reference = np.load(data / 'reference.npy')
        cameras, timestamps = np.load(data / 'cameras.npy'), np.load(data / 'timestamps.npy')
        frame_ids = np.load(data / 'frame_ids.npy', mmap_mode='r')
        if (mask.shape != (len(points),) or mask.dtype.kind != 'b'
                or frame_ids.shape != (len(points),) or frame_ids.dtype.kind not in 'iu'
                or np.any(frame_ids < 0) or np.any(frame_ids >= len(cameras))
                or cameras.ndim != 2 or cameras.shape[1] != 3
                or timestamps.shape != (len(cameras),) or not np.isfinite(timestamps).all()
                or np.any(np.diff(timestamps) < 0)
                or report.get('frame_count') != len(cameras)
                or report.get('total_rows') != len(points)):
            raise ValueError('Invalid saved mask, camera or timestamp alignment')
        if reference.shape != (3,):
            raise ValueError('Invalid saved reference shape')
        if not np.isfinite(reference).all() or not np.any(reference):
            reference = None
        elif not np.isclose(np.linalg.norm(reference), 1.0, rtol=0, atol=1e-12):
            raise ValueError('Saved reference must be the exact normalized CostMap reference')
        settings = _settings(provenance['settings'] | {'support_mask': mask})
        _aligned_costs(points, SimpleNamespace(points=points, cost=costs))
        if np.any(np.isfinite(costs) & (~mask | ~np.isfinite(points).all(axis=1))):
            raise ValueError('Finite costs must refer to finite, labelled support rows')
        poses = associate(points, costs, mask, reference, settings, cameras, timestamps, select_endpoint, frame_ids)
        atomic_json(output / 'allposes.json', poses)
        state, start, goal = select_segment(poses)
        attempts = 0
        report.update(supported_count=sum(pose['reason'] == 'ok' for pose in poses),
                      reason_counts=dict(Counter(pose['reason'] for pose in poses)),
                      association_mode_counts=dict(Counter(pose['association_mode'] for pose in poses if pose['reason'] == 'ok')),
                      initial_supported_interval={'start': start, 'goal': goal},
                      selected={'start': None, 'goal': None}, start=None, goal=None,
                      earliest_latest_supported_attempted=state in ('ready', 'degenerate_route'),
                      segment_label='supported video segment', endpoint_full_array_verified=True,
                      endpoint_policy='Keep the first supported video start and target the last supported video goal. If disconnected, end at the reachable original row closest in 3D to the intended goal, breaking exact ties by original row ID. Prefer beneath-camera association; only missing support may use finite-cost ground below the camera observed in that same frame. No camera-to-ground graph edge is added.')
        atomic_json(output / 'report.json', report)
        ids, total, optimal, verified = np.empty(0, dtype=np.int64), np.inf, False, False
        if state == 'degenerate_route':
            ids, total, optimal, verified = np.asarray([start['id']]), 0.0, True, True
        elif state == 'ready':
            update('restoring_connection_queries')
            cm = restore_cost_map(points, costs, mask, reference, settings, CostMap)
            start, goal, result, attempts = fixed_start_segment(
                cm, poses, NativeSearch(args.library), find_path, update)
            state = result.status
            ids, total, optimal, verified = result.ids, result.total_cost, True, True
        save_route(output, ids, points, total)
        visible_ends = sum(endpoint is not None and endpoint['association_mode'] == 'visible_ground'
                           for endpoint in (start, goal))
        interval_modes = Counter(pose['association_mode'] for pose in poses
                                 if start and goal and start['frame_id'] <= pose['frame_id'] <= goal['frame_id']
                                 and pose['reason'] == 'ok')
        report.update(status=state, optimal=optimal, all_edges_verified=verified, start=start, goal=goal,
                      segment_label=('partial route from supported video start; intended goal unreachable' if state == 'partial' else
                                     'supported video segment (visible-ground association)' if visible_ends else 'supported video segment'),
                      reached={'id': int(ids[-1]), 'ground_point': points[ids[-1]]} if len(ids) else None,
                      goal_reached=bool(len(ids) and goal and ids[-1] == goal['id']),
                      optimality_scope='nearest_reachable_endpoint' if state == 'partial' else 'intended_goal' if len(ids) else None,
                      selected_visible_ground_count=visible_ends,
                      selected_interval_association_mode_counts=dict(interval_modes),
                      selected_interval_visible_ground_count=interval_modes['visible_ground'],
                      selected={'start': start, 'goal': goal}, connection_attempts=attempts,
                      selected_coverage_seconds=goal['time_seconds'] - start['time_seconds'] if start and goal else None,
                      excluded_prefix_frames=start['frame_id'] if start else None,
                      excluded_suffix_frames=len(cameras) - 1 - goal['frame_id'] if goal else None,
                      used_connected_subinterval=False,
                      original_row_ids_verified=True, original_costs_reused=True,
                      route_points=len(ids), total_cost=total,
                      route_length=float(np.linalg.norm(np.diff(points[ids], axis=0), axis=1).sum()),
                      elapsed_path_seconds=time.monotonic() - started)
        atomic_json(output / 'report.json', report)
        update('complete', status=state, route_points=len(ids), total_cost=total)
        return 0
    except Exception as error:
        # An interrupted/failed calculation must never leave an older successful route visible.
        save_route(output, [], np.empty((0, 3)), np.inf)
        report.update(status='error', error=f'{type(error).__name__}: {error}',
                      optimal=False, route_points=0, total_cost=None)
        atomic_json(output / 'report.json', report)
        update('error', error=report['error'])
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--library', type=Path,
                        default=Path(__file__).resolve().parents[2] / 'path-planning/corridor_astar_feasible.so')
    sys.exit(run(parser.parse_args()))
