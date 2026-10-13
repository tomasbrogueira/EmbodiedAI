"""Exact supported-video paths using saved costs; all artifacts stay outside the repo."""

import argparse
import ctypes
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree


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


def with_visible_ground(result, points, costs, mask, reference, camera, frame, frame_ids, rows=None):
    """Missing beneath-camera support may use observed ground in this exact frame."""
    mode = 'beneath_camera'
    if result['reason'] == 'missing_support' and frame_ids is not None and reference is not None:
        if rows is None:
            rows = np.flatnonzero(frame_ids == frame)
        rows = np.sort(rows)
        rows = rows[(frame_ids[rows] == frame) & mask[rows] & np.isfinite(costs[rows])
                    & np.isfinite(points[rows]).all(axis=1)]
        delta = points[rows] - camera
        below = -(delta @ reference) > 0
        rows, delta = rows[below], delta[below]
        if len(rows):
            row = int(rows[np.argmin(np.linalg.norm(delta, axis=1))])
            offset = points[row] - camera
            drop = float(-offset @ reference)
            lateral = float(np.linalg.norm(offset + drop * reference))
            result = dict(reason='ok', id=row, camera=np.array(camera), projected=None,
                          offset=offset.copy(), distance=lateral, drop=drop)
            mode = 'visible_ground'
    if result['reason'] != 'ok':
        return result | {'association_mode': None}
    row = result['id']
    offset = points[row] - camera
    drop = float(-offset @ reference)
    return result | dict(association_mode=mode, camera=np.array(camera), ground_point=points[row].copy(),
                         camera_offset=offset, camera_distance=float(np.linalg.norm(offset)),
                         lateral_distance=float(np.linalg.norm(offset + drop * reference)),
                         observed_frame_id=int(frame_ids[row]) if frame_ids is not None else None)


def floor_projection(points, rows, basis, chunk_size=1000000):
    projected = np.empty((len(rows), 2), dtype=np.float64)
    for first in range(0, len(rows), chunk_size):
        last = first + chunk_size
        projected[first:last] = points[rows[first:last]] @ basis
    return projected


def global_endpoint(points, costs, mask, reference, settings, camera, select_endpoint, chunk_size=1000000):
    """Full scan independent of the spatial index; rank geometry before checking cost."""
    winner, winning_first, nearest_distance = None, None, np.inf
    camera = None if camera is None else np.asarray(camera)
    if camera is not None and camera.shape == (3,) and np.isfinite(camera).all() and reference is not None:
        for first in range(0, len(points), chunk_size):
            last = first + chunk_size
            rows = np.flatnonzero(mask[first:last] & np.isfinite(points[first:last]).all(axis=1)) + first
            delta = (points[rows] - camera) / settings['radius']
            drop = -delta @ reference
            lateral = np.linalg.norm(delta + drop[:, None] * reference, axis=1)
            keep = (drop > 0) & (lateral <= .5)
            rows, delta = rows[keep], delta[keep]
            if len(rows):
                distances = np.linalg.norm(delta, axis=1)
                nearest = int(np.argmin(distances))
                # Chunks and rows ascend, so strict improvement preserves original-ID ties.
                if winner is None or distances[nearest] < nearest_distance:
                    winner, nearest_distance = int(rows[nearest]), float(distances[nearest])
                    winning_first = first
    first, last = (0, 0) if winning_first is None else (winning_first, winning_first + chunk_size)
    subset = SimpleNamespace(points=points[first:last], cost=costs[first:last], reference=reference,
                             config=settings | {'support_mask': mask[first:last]})
    # Keep the winning chunk's operation shape: a singleton dot product can round
    # differently at the exact lateral boundary. Production supplies all diagnostics.
    result = select_endpoint(subset.points, subset, camera)
    if result['id'] is not None:
        result['id'] += first
    return result


def associate(points, costs, mask, reference, settings, cameras, timestamps, select_endpoint, frame_ids=None):
    """Spatial indexing only narrows candidates; production selects original row IDs."""
    floor_rows = np.flatnonzero(mask & np.isfinite(points).all(axis=1))
    ordered_frames = frame_ids is not None and not np.any(frame_ids[1:] < frame_ids[:-1])
    if reference is not None:
        axis = np.eye(3)[np.argmin(np.abs(reference))]
        first = np.cross(reference, axis)
        first /= np.linalg.norm(first)
        basis = np.column_stack((first, np.cross(reference, first)))
        tree = cKDTree(floor_projection(points, floor_rows, basis))
        radius = settings['radius'] / 2
        # Expand candidate lookup slightly; select_endpoint enforces the exact tolerance.
        query_radius = radius + 1e-8 * settings['radius']
    reports = []
    for frame, camera in enumerate(cameras):
        rows = np.empty(0, dtype=np.int64)
        if reference is not None and np.isfinite(camera).all():
            nearby = tree.query_ball_point(camera @ basis, query_radius)
            rows = np.sort(floor_rows[nearby])
        subset = SimpleNamespace(points=points[rows], cost=costs[rows], reference=reference,
                                 config=settings | {'support_mask': mask[rows]})
        result = select_endpoint(subset.points, subset, camera)
        if result['id'] is not None:
            result['id'] = int(rows[result['id']])
        frame_rows = None
        if ordered_frames and result['reason'] == 'missing_support':
            first_row, last_row = np.searchsorted(frame_ids, [frame, frame + 1])
            frame_rows = np.arange(first_row, last_row, dtype=np.int64)
        result = with_visible_ground(result, points, costs, mask, reference, camera, frame,
                                     frame_ids, frame_rows)
        reports.append(result | {'frame_id': frame, 'time_seconds': float(timestamps[frame])})
    if reference is not None:
        del tree
    del floor_rows
    selected = [result for result in reports if result['reason'] == 'ok']
    # Recheck the selected endpoints against every original point, independently of the index.
    verify_associations(points, costs, mask, reference, settings, cameras,
                        [selected[0], selected[-1]] if selected else [], select_endpoint, frame_ids)
    return reports


def verify_associations(points, costs, mask, reference, settings, cameras, candidates, select_endpoint, frame_ids=None):
    for candidate in candidates:
        check = global_endpoint(points, costs, mask, reference, settings,
                                cameras[candidate['frame_id']], select_endpoint)
        check = with_visible_ground(check, points, costs, mask, reference, cameras[candidate['frame_id']],
                                    candidate['frame_id'], frame_ids)
        scale = max(settings['radius'], np.max(np.abs(cameras[candidate['frame_id']])),
                    np.max(np.abs(points[candidate['id']])) if candidate['id'] is not None else 0.)
        diagnostic_tolerance = 16 * np.finfo(float).eps * scale
        for key in ('reason', 'id', 'projected', 'offset', 'distance', 'drop', 'association_mode',
                    'ground_point', 'camera_offset', 'camera_distance', 'lateral_distance', 'observed_frame_id'):
            actual, expected = check.get(key), candidate.get(key)
            # Only derived reporting values allow machine roundoff; the selected
            # row, acceptance reason and original coordinates remain exact.
            if key in ('projected', 'offset', 'distance', 'drop', 'lateral_distance') and actual is not None and expected is not None:
                equal = np.allclose(actual, expected, rtol=0, atol=diagnostic_tolerance)
            elif isinstance(actual, np.ndarray):
                equal = np.array_equal(actual, expected)
            else:
                equal = actual == expected
            if not equal:
                raise RuntimeError(f'Indexed endpoint differs from full-array selection: {key}')


def select_segment(poses):
    supported = [pose for pose in poses if pose['reason'] == 'ok']
    if not supported:
        return 'no_supported_pose', None, None
    if len(supported) == 1:
        return 'single_supported_pose', supported[0], supported[0]
    start, goal = supported[0], supported[-1]
    return ('degenerate_route' if start['id'] == goal['id'] else 'ready'), start, goal


def restore_cost_map(points, costs, mask, reference, settings, CostMap):
    """Restore the existing connection queries without recomputing any cost."""
    cm = CostMap.__new__(CostMap)
    cm.points, cm.cost, cm.reference, cm.config = points, costs, reference, settings
    finite = np.isfinite(points).all(axis=1)
    cm.rows = np.flatnonzero(finite)
    cm.obstacle_rows = np.flatnonzero(finite & (~mask | settings.get('obstacle_mask', np.zeros(len(points), dtype=bool))))
    cm.tree = cKDTree(points if len(cm.rows) == len(points) else points[cm.rows])
    cm.obstacles = cKDTree(points[cm.obstacle_rows])
    if hasattr(CostMap, '_index_overhead') and reference is not None:
        cm._index_overhead()
    # _connected only queries nearest support distance: coincident copies do not change it.
    cm.support_rows = np.flatnonzero(np.isfinite(costs))
    cm.support = cKDTree(points[cm.support_rows])
    return cm


def edge_allowed(cm, first, second):
    delta = cm.points[[second]] - cm.points[first]
    radius2 = cm.config['connection_radius'] ** 2
    return (np.einsum('ij,ij->i', delta, delta)[0] <= radius2
            and cm._connected(int(first), int(second)))


def edge_diagnostic(cm, first, second):
    a, b = cm.points[min(first, second)], cm.points[max(first, second)]
    delta, c = b - a, cm.config
    length, height = np.linalg.norm(delta), delta @ cm.reference
    horizontal = np.linalg.norm(delta - height * cm.reference)
    steps = max(1, int(np.ceil(length / (c['max_gap'] / 4))))
    padding = length / (2 * steps)
    probes = a + (np.arange(steps + 1) / steps)[:, None] * delta
    distances = cm.support.query(probes)[0]
    worst = int(np.argmax(distances))
    distance = float(distances[worst])
    nearest, obstacle_row = np.inf, None
    for rows in cm._near((a + b) / 2, length / 2 + c['obstacle_margin'],
                         cm.obstacles, cm.obstacle_rows):
        offset = cm.points[rows] - a
        t = np.clip(offset @ delta / (length * length), 0, 1) if length else np.zeros(len(rows))
        obstacle_distances = np.linalg.norm(offset - t[:, None] * delta, axis=1)
        closest = int(np.argmin(obstacle_distances))
        if obstacle_distances[closest] < nearest:
            nearest, obstacle_row = float(obstacle_distances[closest]), int(rows[closest])
    reason = ('radius' if np.einsum('i,i->', delta, delta) > c['connection_radius'] ** 2 else
              'slope' if c.get('slope_policy', 'enforce') == 'enforce' and length and abs(height) >= horizontal * np.tan(c['max_slope']) else
              'support_gap' if distance + padding > c['max_gap'] / 2 else
              'obstacle' if nearest <= c['obstacle_margin'] else
              'overhead_obstacle' if hasattr(cm, 'obstacle_free') and not cm.obstacle_free(a, b) else 'unexplained')
    return dict(ids=[int(first), int(second)], positions=cm.points[[first, second]],
                reason=reason, length=length, height=height, horizontal=horizontal,
                steps=steps, padding=padding, max_probe_distance=distance,
                gap_limit=c['max_gap'] / 2, nearest_obstacle_in_query=nearest,
                obstacle_margin=c['obstacle_margin'],
                bad_probe=probes[worst] if distance + padding > c['max_gap'] / 2 else None,
                bad_probe_distance=distance if distance + padding > c['max_gap'] / 2 else None,
                obstacle_row=obstacle_row,
                obstacle_point=cm.points[obstacle_row] if obstacle_row is not None else None)


def route_cost(cm, ids):
    a, b = ids[:-1], ids[1:]
    length = np.linalg.norm(cm.points[b] - cm.points[a], axis=1)
    return float(np.sum(length * (cm.cost[a] / 2 + cm.cost[b] / 2)))


def feasible_bound(cm, finite_rows, associated_ids, start, goal, update):
    """A sampled subgraph supplies only an upper bound, never a no-path conclusion."""
    for size in (36000, 72000, 144000):
        sampled = finite_rows[np.linspace(0, len(finite_rows) - 1,
                                         min(size, len(finite_rows)), dtype=np.int64)]
        rows = np.union1d(sampled, associated_ids)
        x, costs = cm.points[rows], cm.cost[rows]
        tree = cKDTree(x)
        radius = cm.config['connection_radius']
        # Avoid materializing a dense sampled graph; the exact search still runs below.
        pair_count = (tree.count_neighbors(tree, radius) - len(rows)) // 2
        update('finding_feasible_bound', sampled_rows=len(rows), sampled_edges=int(pair_count))
        if pair_count > 10000000:
            continue
        pairs = tree.query_pairs(radius, output_type='ndarray')
        if not len(pairs):
            continue
        a, b = pairs.T
        delta = x[b] - x[a]
        keep = np.einsum('ij,ij->i', delta, delta) <= radius * radius
        a, b, delta = a[keep], b[keep], delta[keep]
        weight = np.linalg.norm(delta, axis=1) * (costs[a] / 2 + costs[b] / 2)
        active = np.ones(len(a), dtype=bool)
        keys = a * len(rows) + b
        source, target = np.searchsorted(rows, [start, goal])
        for attempt in range(128):
            graph = coo_matrix((weight[active], (a[active], b[active])),
                               shape=(len(rows), len(rows))).tocsr()
            distance, previous = dijkstra(graph, directed=False, indices=source,
                                          return_predecessors=True)
            if not np.isfinite(distance[target]):
                break
            local, row = [int(target)], int(target)
            while row != source:
                row = int(previous[row])
                if row < 0 or len(local) > len(rows):
                    raise RuntimeError('Invalid sampled predecessor chain')
                local.append(row)
            local = np.asarray(local[::-1], dtype=np.int64)
            ids = rows[local]
            invalid = [(int(i), int(j)) for i, j, u, v in
                       zip(local[:-1], local[1:], ids[:-1], ids[1:])
                       if not edge_allowed(cm, u, v)]
            if not invalid:
                return ids, route_cost(cm, ids)
            bad_keys = [min(i, j) * len(rows) + max(i, j) for i, j in invalid]
            active &= ~np.isin(keys, bad_keys)
        if len(sampled) == len(finite_rows):
            break
    return None, None


class NativeSearch:
    """Bind the shipped clearance-aware search; certify its results in exact_route."""
    supports_feasibility = supports_clearance = True

    def __init__(self, path):
        self.library = ctypes.CDLL(str(path))
        self.supports_policy = hasattr(self.library, 'corridor_astar_policy')
        self.function = self.library.corridor_astar_policy if self.supports_policy else self.library.corridor_astar_clearance
        self.function.argtypes = [ctypes.POINTER(ctypes.c_double)] * 3 + [
            ctypes.c_int64, ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_int64, ctypes.c_int64,
            ctypes.POINTER(ctypes.c_int64), ctypes.c_int64,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_double), ctypes.c_int64,
            ctypes.POINTER(ctypes.c_int64), ctypes.c_int64,
            ctypes.POINTER(ctypes.c_int64), ctypes.c_int64,
            ctypes.c_double, ctypes.c_double, ctypes.c_double]
        if self.supports_policy:
            self.function.argtypes += [ctypes.c_int]
        self.function.restype = ctypes.c_int

    def prepare_map(self, cm, support_rows):
        if getattr(self, '_map', None) is not cm:
            self._map = cm
            self._original = np.ascontiguousarray(cm.points, dtype=np.float64)
            self._support_rows = np.ascontiguousarray(support_rows, dtype=np.int64)
            self._obstacle_rows = np.ascontiguousarray(cm.obstacle_rows, dtype=np.int64)

    def __call__(self, x, costs, reference, radius, slope, cmin, upper, start, goal, banned,
                 max_gap=None, obstacle_margin=None, clearance_height=None):
        if not hasattr(self, '_map') or max_gap is None or obstacle_margin is None:
            raise ValueError('Full feasibility search requires the original prepared map and settings')
        if clearance_height is None:
            raise ValueError('Clearance search requires the original clearance height')
        x, costs, reference = [np.ascontiguousarray(value, dtype=np.float64)
                               for value in (x, costs, reference)]
        pairs = np.asarray(sorted(banned), dtype=np.int64).reshape(-1, 2)
        distance = np.empty(len(x), dtype=np.float64)
        previous = np.empty(len(x), dtype=np.int64)
        doubles = lambda value: value.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        integers = lambda value: value.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        arguments = (doubles(x), doubles(costs), doubles(reference), len(x),
            radius, np.tan(slope), cmin, upper, start, goal,
            integers(pairs), len(pairs), doubles(distance), integers(previous),
            doubles(self._original), len(self._original),
            integers(self._support_rows), len(self._support_rows),
            integers(self._obstacle_rows), len(self._obstacle_rows),
            float(max_gap), float(obstacle_margin), float(clearance_height))
        enforce = self._map.config.get('slope_policy', 'enforce') == 'enforce'
        if not enforce and not self.supports_policy:
            raise RuntimeError('Recompile native search to use the slope-disabled policy')
        status = self.function(*arguments, int(enforce)) if self.supports_policy else self.function(*arguments)
        if status < 0:
            raise RuntimeError(f'Native exact search failed with code {status}')
        return status, distance, previous


def nearest_reachable(points, rows, distance, goal, chunk_size=1000000):
    """Closest original row in an exhausted reachable set; original-ID ties."""
    winner, closest = None, np.inf
    for first in range(0, len(rows), chunk_size):
        local = np.flatnonzero(np.isfinite(distance[first:first + chunk_size])) + first
        if len(local):
            separation = np.linalg.norm(points[rows[local]] - points[goal], axis=1)
            best = int(np.argmin(separation))
            if winner is None or separation[best] < closest:
                winner, closest = int(local[best]), float(separation[best])
    if winner is None:
        raise RuntimeError('Exhausted search did not retain its starting point')
    return winner


def exact_route(cm, start, goal, associated_ids, native, find_path, update,
                on_disconnected=None, allow_partial=False):
    rows = cm.support_rows if hasattr(cm, 'support_rows') else np.flatnonzero(np.isfinite(cm.cost))
    feasible_native = getattr(native, 'supports_feasibility', False)
    if feasible_native:
        native.prepare_map(cm, rows)
    bound_ids, upper = feasible_bound(cm, rows, associated_ids, start, goal, update)
    has_bound = upper is not None
    cmin = float(np.min(cm.cost[rows]))
    if has_bound:
        # Any improving path must lie in this ellipse. Tightening the minimum
        # row cost after pruning keeps the same bound valid by induction.
        for _ in range(4):
            pieces = []
            for offset in range(0, len(rows), 1000000):
                part = rows[offset:offset + 1000000]
                lower = (np.linalg.norm(cm.points[part] - cm.points[start], axis=1)
                         + np.linalg.norm(cm.points[part] - cm.points[goal], axis=1)) * cmin
                pieces.append(part[lower <= upper + 1e-9 * max(1.0, upper)])
            rows = np.concatenate(pieces)
            del pieces, part, lower
            next_minimum = float(np.min(cm.cost[rows]))
            if next_minimum == cmin:
                break
            cmin = next_minimum
    else:
        # Cover g+h for every reachable relaxed vertex, not just a route to the
        # target. Exhaustion then also gives a valid disconnection partition.
        extent = np.maximum(abs(cm.tree.mins - cm.points[goal]),
                            abs(cm.tree.maxes - cm.points[goal]))
        upper = float((np.longdouble(max(1, len(rows) - 1))
                       * cm.config['connection_radius'] * np.max(cm.cost[rows])
                       + np.longdouble(cmin) * np.linalg.norm(extent))
                      * (1 + 4 * len(rows) * np.finfo(float).eps))
        if not np.isfinite(upper):
            raise RuntimeError('The unrestricted search bound is not representable')
    local_start, local_goal = np.searchsorted(rows, [start, goal])
    if (local_start >= len(rows) or local_goal >= len(rows)
            or rows[local_start] != start or rows[local_goal] != goal):
        raise RuntimeError('Safe search bounds unexpectedly excluded an endpoint')
    x, costs = np.ascontiguousarray(cm.points[rows]), np.ascontiguousarray(cm.cost[rows])
    witnessed = getattr(native, 'supports_witnesses', False)
    if witnessed and not hasattr(cm, '_path_support_witnesses'):
        cm._path_support_witnesses, cm._path_obstacle_witnesses = {}, {}
    banned, attempt = set(), 0
    while True:
        attempt += 1
        update('exact_path_search', retained_rows=len(rows), attempt=attempt,
               banned_edges=len(banned), verified_upper_bound=upper if has_bound else None)
        # Search from the fixed start so exhaustion also supplies the partial route.
        witness_arguments = {}
        if feasible_native:
            witness_arguments = dict(max_gap=cm.config['max_gap'], obstacle_margin=cm.config['obstacle_margin'])
            if getattr(native, 'supports_clearance', False):
                witness_arguments['clearance_height'] = cm.config['clearance_height']
        elif witnessed:
            witness_arguments = dict(
                support_witnesses=[(*point, distance) for point, distance in cm._path_support_witnesses.items()],
                obstacle_witnesses=list(cm._path_obstacle_witnesses.values()),
                max_gap=cm.config['max_gap'], obstacle_margin=cm.config['obstacle_margin'])
        status, distance, previous = native(x, costs, cm.reference,
                                            cm.config['connection_radius'], cm.config['max_slope'],
                                            cmin, upper, int(local_start), int(local_goal), banned,
                                            **witness_arguments)
        target = int(local_goal)
        if status == 0:
            if has_bound:
                raise RuntimeError('Exact search lost a previously verified feasible route')
            if on_disconnected is not None:
                local_associated = np.searchsorted(rows, associated_ids)
                on_disconnected(associated_ids[np.isfinite(distance[local_associated])])
            if not allow_partial:
                return None
            target = nearest_reachable(cm.points, rows, distance, goal)
        local, row = [target], target
        while row != local_start:
            row = int(previous[row])
            if row < 0 or len(local) > len(rows):
                raise RuntimeError('Invalid exact-search predecessor chain')
            local.append(row)
        local = np.asarray(local[::-1], dtype=np.int64)
        certificate = rows[local]
        update('validating_candidate_route', attempt=attempt, candidate_optimal=False,
               route_points=len(certificate), candidate_cost=float(distance[target]),
               candidate_goal_reached=status == 1,
               candidate={'ids': certificate, 'positions': cm.points[certificate],
                          'total_cost': float(distance[target]), 'optimal': False})
        invalid = [(min(int(i), int(j)), max(int(i), int(j))) for i, j, u, v in
                   zip(local[:-1], local[1:], certificate[:-1], certificate[1:])
                   if not edge_allowed(cm, u, v)]
        if not invalid:
            break
        before = len(banned)
        banned.update(invalid)
        if len(banned) == before:
            raise RuntimeError('Exact search reused an excluded edge')
        diagnostics = [edge_diagnostic(cm, int(rows[a]), int(rows[b]))
                       for a, b in (invalid if witnessed else invalid[:2])]
        if witnessed:
            for edge in diagnostics:
                if edge['bad_probe'] is not None:
                    cm._path_support_witnesses[tuple(edge['bad_probe'])] = edge['bad_probe_distance']
                if edge['nearest_obstacle_in_query'] <= cm.config['obstacle_margin']:
                    cm._path_obstacle_witnesses[edge['obstacle_row']] = edge['obstacle_point']
        update('candidate_edges_rejected', attempt=attempt, invalid_edges=len(invalid),
               banned_edges=len(banned), candidate_cost=float(distance[target]),
               support_witnesses=len(cm._path_support_witnesses) if witnessed else 0,
               obstacle_witnesses=len(cm._path_obstacle_witnesses) if witnessed else 0,
               examples=diagnostics[:2])
        del distance, previous
    certificate_cost, reached = float(distance[target]), int(rows[target])
    del x, costs, rows, distance, previous
    # The relaxed graph contains every permitted original edge. Its minimum is
    # now fully feasible, so it is also a global minimum of the original graph.
    # On exhaustion, its reachable set contains every originally reachable row.
    # A feasible route to its closest row also certifies the closest reachable
    # endpoint; no second search or movement of the original start is needed.
    # Verify using unchanged production find_path/neighbours on the certificate's
    # induced rows; support and obstacle queries still use the complete map.
    certificate_rows = np.unique(certificate)
    certificate_tree = cKDTree(cm.points[certificate_rows])
    original_near = cm._near
    def certificate_near(position, radius, tree=None, rows=None, **kwargs):
        if tree is None:
            tree, rows = certificate_tree, certificate_rows
        yield from original_near(position, radius, tree, rows, **kwargs)
    cm._near = certificate_near
    try:
        result = find_path(cm.points, cm, start, reached)
    finally:
        cm._near = original_near
    if (result.status != 'ok' or not np.isclose(result.total_cost, certificate_cost,
                                               rtol=1e-9, atol=1e-9)
            or not np.array_equal(result.positions, cm.points[result.ids])
            or any(not edge_allowed(cm, a, b) for a, b in zip(result.ids[:-1], result.ids[1:]))):
        raise RuntimeError('Production planner verification disagreed with the exact certificate')
    if reached != goal:
        result.status = 'partial'
    return result


def fixed_start_segment(cm, poses, native, find_path, update):
    """Keep the first supported pose and its intended final video endpoint."""
    state, start, goal = select_segment(poses)
    if state == 'degenerate_route':
        return start, goal, find_path(cm.points, cm, start['id'], goal['id']), 0
    if state != 'ready':
        return start, goal, None, 0
    associated = np.asarray(sorted({pose['id'] for pose in poses if pose['reason'] == 'ok'}), dtype=np.int64)
    result = exact_route(cm, start['id'], goal['id'], associated, native, find_path, update,
                         allow_partial=True)
    return start, goal, result, 1
