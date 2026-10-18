"""Small graph and endpoint checks for the external batch path stage."""

from heapq import heappop, heappush
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(os.environ.get('DATASET_REPO', Path.cwd())) / 'path-planning'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cost_map import CostMap, _settings
from planner import find_path
import dataset_path as stage


def simple_endpoint(points, cm, camera):
    result = {'reason': 'missing_support', 'id': None}
    if cm.reference is None:
        return result | {'reason': 'missing_reference'}
    delta = points - camera
    drop = -delta @ cm.reference
    lateral = np.linalg.norm(delta + drop[:, None] * cm.reference, axis=1)
    rows = np.flatnonzero(cm.config['support_mask'] & (drop > 0)
                          & (lateral <= cm.config['radius'] / 2))
    if not len(rows):
        return result
    row = rows[np.argmin(np.linalg.norm(delta[rows], axis=1))]
    return result | ({'reason': 'ok', 'id': int(row)} if np.isfinite(cm.cost[row])
                     else {'reason': 'blocked_support'})


def tube_endpoint(points, cm, camera):
    """Current server selector arithmetic, independent of the chunked scan."""
    result = dict(reason='missing_pose', id=None, camera=None, projected=None, offset=None, distance=None)
    if camera is None or np.shape(camera) != (3,) or not np.isfinite(camera).all():
        return result
    result['camera'] = np.array(camera)
    if cm.reference is None:
        return result | {'reason': 'missing_reference'}
    rows = np.flatnonzero(cm.config['support_mask'] & np.isfinite(points).all(axis=1))
    radius = cm.config['radius']
    delta = (points[rows] - camera) / radius
    drop = -delta @ cm.reference
    lateral = np.linalg.norm(delta + drop[:, None] * cm.reference, axis=1)
    keep = (drop > 0) & (lateral <= .5)
    rows, delta, drop = rows[keep], delta[keep], drop[keep]
    if not len(rows):
        return result | {'reason': 'missing_support'}
    nearest = np.argmin(np.linalg.norm(delta, axis=1))
    row, depth = int(rows[nearest]), float(drop[nearest] * radius)
    projected = camera - depth * cm.reference
    offset = points[row] - projected
    result.update(projected=projected, offset=offset, distance=float(np.linalg.norm(offset)), drop=depth)
    return result | ({'reason': 'ok', 'id': row} if np.isfinite(cm.cost[row])
                     else {'reason': 'blocked_support'})


def python_relaxation(x, cost, ref, radius, slope, cmin, upper, start, goal, banned):
    distance, previous = np.full(len(x), np.inf), np.full(len(x), -1, dtype=np.int64)
    distance[start], queue = 0, [(0.0, start)]
    while queue:
        total, row = heappop(queue)
        if total != distance[row]:
            continue
        if row == goal:
            return 1, distance, previous
        for other in range(len(x)):
            if row == other or (min(row, other), max(row, other)) in banned:
                continue
            delta = x[other] - x[row]
            length, height = np.linalg.norm(delta), delta @ ref
            if length > radius or abs(height) > np.linalg.norm(delta - height * ref) * np.tan(slope) + 1e-12:
                continue
            candidate = total + length * (cost[row] / 2 + cost[other] / 2)
            if candidate < distance[other] and candidate <= upper + 1e-9:
                distance[other], previous[other] = candidate, row
                heappush(queue, (candidate, other))
    return 0, distance, previous


class WitnessRecorder:
    supports_witnesses = True

    def __init__(self, native):
        # These two protocol tests exercise witness collection even when the
        # selected native library filters all failures before returning a route.
        self.native = python_relaxation if getattr(native, 'supports_feasibility', False) else native
        self.batches = []

    def __call__(self, *args, **kwargs):
        self.batches.append(kwargs)
        return self.native(*args, **(kwargs if getattr(self.native, 'supports_witnesses', False) else {}))


class BatchPathTests(unittest.TestCase):
    def test_chunked_floor_projection_matches_original_rows(self):
        points = np.arange(51, dtype=float).reshape(-1, 3) / 7
        rows = np.array([1, 3, 4, 7, 9, 12, 16])
        basis = np.array([[.6, 0], [0, 1], [-.8, 0]])
        # BLAS may round a final one-row product differently; candidate lookup
        # already includes a much larger 1e-8 * radius expansion.
        np.testing.assert_allclose(stage.floor_projection(points, rows, basis, chunk_size=2),
                                    points[rows] @ basis, rtol=1e-15, atol=0)
        self.assertEqual(stage.floor_projection(points, rows[:0], basis, chunk_size=2).shape, (0, 2))

    def test_chunked_global_endpoint_matches_full_scan_and_keeps_blocked_nearest(self):
        points = np.array([[0., 0, -2], [.5, 0, 0], [np.nan, 0, 0],
                           [-.5, 0, 0], [0, 0, 1], [0, 0, .25], [0, 0, .5]])
        camera, reference = np.array([0., 0, 1]), np.array([0., 0, 1])
        mask = np.array([True] * 6 + [False])
        settings = {'radius': 1.}
        for costs, expected in [(np.ones(7), ('ok', 5)),
                                (np.array([1, 1, 1, 1, 1, np.inf, 1]), ('blocked_support', None))]:
            cm = SimpleNamespace(points=points, cost=costs, config=settings | {'support_mask': mask}, reference=reference)
            for chunk_size in (1, 2, 3, 8):
                actual = stage.global_endpoint(points, costs, mask, reference, settings, camera,
                                               tube_endpoint, chunk_size=chunk_size)
                self.assertEqual(stage.json_value(actual), stage.json_value(tube_endpoint(points, cm, camera)))
                self.assertEqual((actual['reason'], actual['id']), expected)
        # Equal-distance boundary candidates straddle chunks: lowest original ID wins.
        mask[5] = False
        actual = stage.global_endpoint(points, np.ones(7), mask, reference, settings, camera,
                                       tube_endpoint, chunk_size=2)
        self.assertEqual(actual['id'], 1)

    def test_chunked_missing_support_falls_back_but_blocked_nearest_does_not(self):
        points = np.array([[3., 0, 0], [2, 0, 0], [1, 0, 0], [0, 0, 0]])
        costs, mask = np.array([1., 1, 1, np.inf]), np.array([True, True, True, False])
        frames, reference, camera = np.array([0, 0, 1, 0]), np.array([0., 0, 1]), np.array([0., 0, 1])
        settings = {'radius': .4}
        scan = stage.global_endpoint
        with patch.object(stage, 'global_endpoint', side_effect=lambda *a: scan(*a, chunk_size=2)):
            poses = stage.associate(points, costs, mask, reference, settings, camera[None, :],
                                    [0], tube_endpoint, frames)
            self.assertEqual((poses[0]['id'], poses[0]['association_mode']), (1, 'visible_ground'))
            stage.verify_associations(points, costs, mask, reference, settings, camera[None, :],
                                      poses, tube_endpoint, frames)
            mask[-1] = True
            blocked = stage.global_endpoint(points, costs, mask, reference, settings, camera, tube_endpoint)
            blocked = stage.with_visible_ground(blocked, points, costs, mask, reference, camera, 0, frames)
            self.assertEqual((blocked['reason'], blocked['id'], blocked['association_mode']),
                              ('blocked_support', None, None))

    def test_chunked_global_endpoint_preserves_missing_inputs(self):
        points, costs, mask = np.zeros((5, 3)), np.ones(5), np.ones(5, dtype=bool)
        for camera, reference in [(None, np.array([0., 0, 1])),
                                  (np.array([0., 0, np.nan]), np.array([0., 0, 1])),
                                  (np.array([0., 0, 1]), None),
                                  (np.array([5., 0, 1]), np.array([0., 0, 1]))]:
            cm = SimpleNamespace(points=points, cost=costs, reference=reference, config={'radius': 1., 'support_mask': mask})
            actual = stage.global_endpoint(points, costs, mask, reference, {'radius': 1.}, camera,
                                           tube_endpoint, chunk_size=2)
            self.assertEqual(stage.json_value(actual), stage.json_value(tube_endpoint(points, cm, camera)))

    def test_chunked_boundary_winner_keeps_production_operation_shape(self):
        reference = np.array([-.7804534760983168, .5866065379689757, .2162987315221854])
        points = np.array([[1.8679119571837217, -1.229649962221206, -.9863133944008194],
                           [2.3302149016255687, -1.5771273430481096, -1.1144383160318252],
                           [1.3179392783212878, -.8162780201320221, -.8338912495158923],
                           [.504607861722595, -.2049596762154414, -.6084805631583654],
                           [.8841463574186007, -.49022941977430023, -.7136677366376757],
                           [.6861453260493205, -.34140735275439726, -.6587927527509173]])
        costs, mask = np.ones(6), np.ones(6, dtype=bool)
        actual = stage.global_endpoint(points, costs, mask, reference, {'radius': 1.}, np.zeros(3),
                                       tube_endpoint, chunk_size=3)
        cm = SimpleNamespace(points=points, cost=costs, reference=reference, config={'radius': 1., 'support_mask': mask})
        expected = tube_endpoint(points, cm, np.zeros(3))
        self.assertEqual((actual['reason'], actual['id']), (expected['reason'], expected['id']))

    def test_tilted_reference_audit_allows_only_derived_roundoff(self):
        rng = np.random.default_rng(57)
        scan = stage.global_endpoint
        for _ in range(20):
            reference = rng.normal(size=3)
            reference /= np.linalg.norm(reference)
            points, camera = rng.normal(size=(103, 3)), rng.normal(size=3)
            costs, mask, settings = np.ones(103), np.ones(103, dtype=bool), {'radius': 1.}
            with patch.object(stage, 'global_endpoint', side_effect=lambda *a: scan(*a, chunk_size=2)):
                poses = stage.associate(points, costs, mask, reference, settings, camera[None, :], [0], tube_endpoint)
                if poses[0]['reason'] == 'ok':
                    corrupt = poses[0] | {'distance': poses[0]['distance'] + 1e-8}
                    with self.assertRaisesRegex(RuntimeError, 'distance'):
                        stage.verify_associations(points, costs, mask, reference, settings,
                                                  camera[None, :], [corrupt], tube_endpoint)

    def test_supported_interval_uses_video_order_and_original_row_ids(self):
        points = np.array([[30., 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0]])
        cameras = np.array([[10., 0, 1], [0, 0, 1], [1, 0, 1], [10, 0, 1]])
        mask = np.array([False, True, True, True])
        poses = stage.associate(points, np.array([np.inf, 1, 1, 1]), mask,
                                np.array([0., 0, 1]), {'radius': .4}, cameras,
                                np.array([0., .3, .7, 1]), simple_endpoint)
        status, start, goal = stage.select_segment(poses)
        self.assertEqual((status, start['frame_id'], start['id'], start['time_seconds'],
                          goal['frame_id'], goal['id']), ('ready', 1, 1, .3, 2, 2))

    def test_rejected_blocked_and_missing_reference_do_not_move_endpoints(self):
        points = np.array([[0., 0, 0], [.1, 0, 0]])
        cameras = np.array([[0., 0, 1], [2, 0, 1]])
        args = (points, np.array([np.inf, 1]), np.ones(2, dtype=bool))
        poses = stage.associate(*args, np.array([0., 0, 1]), {'radius': 1},
                                cameras, np.arange(2), simple_endpoint)
        self.assertEqual([p['reason'] for p in poses], ['blocked_support', 'missing_support'])
        self.assertEqual(stage.select_segment(poses)[0], 'no_supported_pose')
        poses = stage.associate(*args, None, {'radius': 1}, cameras, np.arange(2), simple_endpoint)
        self.assertTrue(all(p['reason'] == 'missing_reference' for p in poses))

    def test_single_and_same_row_are_explicit(self):
        pose = {'reason': 'ok', 'id': 7, 'frame_id': 2}
        self.assertEqual(stage.select_segment([pose])[0], 'single_supported_pose')
        self.assertEqual(stage.select_segment([pose, pose | {'frame_id': 5}])[0], 'degenerate_route')

    def test_visible_ground_uses_same_frame_finite_rows_below_camera_and_original_ties(self):
        points = np.array([[2., 0, 0], [1, 0, 0], [2, 0, 0], [.1, 0, 2], [.5, 0, 0], [.25, 0, 0]])
        costs, mask = np.array([1, 1, 1, 1, np.inf, np.inf]), np.array([True] * 5 + [False])
        frame_ids = np.array([0, 2, 0, 0, 0, 0])
        cameras = np.array([[0., 0, 1]] * 3)
        poses = stage.associate(points, costs, mask, np.array([0., 0, 1]), {'radius': .2},
                                cameras, np.arange(3), simple_endpoint, frame_ids)
        first, missing, last = poses
        self.assertEqual((first['id'], first['association_mode'], first['observed_frame_id']),
                          (0, 'visible_ground', 0))
        self.assertEqual((missing['reason'], missing['id']), ('missing_support', None))
        self.assertEqual((last['id'], last['observed_frame_id']), (1, 2))
        self.assertIsNone(first['projected'])
        np.testing.assert_array_equal(first['ground_point'], points[0])
        np.testing.assert_array_equal(first['camera_offset'], [2, 0, -1])
        self.assertEqual((first['lateral_distance'], first['drop']), (2, 1))
        self.assertAlmostEqual(first['camera_distance'], np.sqrt(5))
        self.assertEqual(len(points), 6)
        with self.assertRaisesRegex(RuntimeError, 'observed_frame_id'):
            stage.verify_associations(points, costs, mask, np.array([0., 0, 1]), {'radius': .2},
                                       cameras, [first | {'observed_frame_id': 2}], simple_endpoint, frame_ids)

    def test_visible_ground_sorted_frame_index_does_not_fill_unobserved_frames(self):
        points = np.array([[2., 0, 0], [2, 0, 0], [1, 0, 0]])
        poses = stage.associate(points, np.ones(3), np.ones(3, dtype=bool), np.array([0., 0, 1]),
                                {'radius': .2}, np.array([[0., 0, 1]] * 3), np.arange(3),
                                simple_endpoint, np.array([0, 0, 2]))
        self.assertEqual([pose['id'] for pose in poses], [0, None, 2])

    def test_beneath_camera_precedence_and_blocked_reference_failures_never_fall_back(self):
        points = np.array([[0., 0, 0], [.3, 0, .9]])
        mask, frames = np.ones(2, dtype=bool), np.array([1, 0])
        cameras, reference = np.array([[0., 0, 1]] * 2), np.array([0., 0, 1])
        def associate(costs, ref=reference):
            return stage.associate(points, np.asarray(costs), mask, ref, {'radius': .2},
                                    cameras, np.arange(2), simple_endpoint, frames)[0]
        direct = associate([1, 1])
        self.assertEqual((direct['association_mode'], direct['id']), ('beneath_camera', 0))
        self.assertEqual(associate([np.inf, 1])['reason'], 'blocked_support')
        self.assertEqual(associate([1, 1], None)['reason'], 'missing_reference')

    def cost_map(self, points, cost, mask=None, connection=1.01):
        points, cost = np.asarray(points, dtype=float), np.asarray(cost, dtype=float)
        mask = np.ones(len(points), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        settings = _settings({'radius': 3., 'connection_radius': connection,
                              'max_gap': 3., 'obstacle_margin': .01, 'support_mask': mask})
        return stage.restore_cost_map(points, cost, mask, np.array([0., 0, 1]), settings, CostMap)

    def solve(self, cm, start, goal, native=None):
        if native is None:
            native = self.native()
        return stage.exact_route(cm, start, goal, np.array([start, goal]), native,
                                 find_path, lambda *args, **kwargs: None)

    def native(self):
        library = os.environ.get('DATASET_NATIVE_LIBRARY')
        return stage.NativeSearch(library) if library else python_relaxation

    def poses(self, ids, times=None):
        return [{'reason': 'ok', 'id': row, 'frame_id': frame,
                 'time_seconds': float(frame if times is None else times[frame])}
                for frame, row in enumerate(ids)]

    def segment(self, cm, poses):
        return stage.fixed_start_segment(cm, poses, self.native(), find_path,
                                               lambda *args, **kwargs: None)

    def test_weighted_optimal_path_matches_production(self):
        cm = self.cost_map([[0, 0, 0], [1, 0, 0], [2, 0, 0],
                            [0, 1, 0], [1, 1, 0], [2, 1, 0]], [1, 20, 1, 1, 1, 1])
        expected = find_path(cm.points, cm, 0, 2)
        result = self.solve(cm, 0, 2)
        self.assertEqual(result.total_cost, 4)
        self.assertEqual(result.total_cost, expected.total_cost)
        np.testing.assert_array_equal(result.ids, [0, 3, 4, 5, 2])

    def test_disconnected_graph_and_observed_barrier(self):
        cm = self.cost_map([[0, 0, 0], [.3, 0, 0], [3, 0, 0]], [1, 1, 1], connection=.35)
        self.assertIsNone(self.solve(cm, 0, 2))
        # The relaxed graph crosses an observed barrier; lazy validation must remove that edge.
        cm = self.cost_map([[0, 0, 0], [.3, 0, 0], [.6, 0, 0], [.45, 0, 0]],
                           [1, 1, 1, np.inf], [True, True, True, False], connection=.35)
        self.assertIsNone(self.solve(cm, 0, 2))

    def test_coincident_rows_retain_zero_cost_edge(self):
        cm = self.cost_map([[0, 0, 0], [0, 0, 0]], [1, 2])
        result = self.solve(cm, 0, 1)
        self.assertEqual(result.total_cost, 0)
        np.testing.assert_array_equal(result.ids, [0, 1])

    def test_missing_sampled_bound_still_searches_full_graph(self):
        cm = self.cost_map([[0, 0, 0], [.3, 0, 0], [.6, 0, 0]], [1, 1, 1], connection=.35)
        with patch.object(stage, 'feasible_bound', return_value=(None, None)):
            result = self.solve(cm, 0, 2)
        self.assertAlmostEqual(result.total_cost, .6)

    def test_partial_keeps_original_start_and_reuses_exhausted_tree_once(self):
        cm = self.cost_map([[0, 0, 0], [.3, 0, 0], [3, 0, 0], [3.3, 0, 0]],
                           [1] * 4, connection=.35)
        calls = []
        def counted(*args, **kwargs):
            calls.append((args[7], args[8]))
            return python_relaxation(*args)
        poses = self.poses([0, 2, 3])
        with patch.object(stage, 'feasible_bound', return_value=(None, None)):
            start, goal, result, attempts = stage.fixed_start_segment(
                cm, poses, counted, find_path, lambda *a, **k: None)
        self.assertEqual((start['id'], goal['id'], result.status, attempts), (0, 3, 'partial', 1))
        np.testing.assert_array_equal(result.ids, [0, 1])
        self.assertEqual(calls, [(0, 3)])
        self.assertAlmostEqual(result.total_cost, .3)

    def test_partial_original_id_tie_and_zero_length_reachability(self):
        cm = self.cost_map([[0, 0, 0], [.3, .1, 0], [.3, -.1, 0], [2, 0, 0]],
                           [1] * 4, connection=.35)
        _, goal, result, _ = self.segment(cm, self.poses([0, 3]))
        self.assertEqual((goal['id'], result.ids[-1], result.status), (3, 1, 'partial'))
        rows = np.array([0, 1, 2])
        self.assertEqual(stage.nearest_reachable(cm.points, rows, np.zeros(3), 3, chunk_size=1), 1)
        cm = self.cost_map([[0, 0, 0], [0, 0, 0], [10, 0, 0]], [1] * 3)
        cm.config['connection_radius'] = 0
        _, goal, result, _ = self.segment(cm, self.poses([1, 2]))
        self.assertEqual((goal['id'], result.status, result.total_cost), (2, 'partial', 0))
        np.testing.assert_array_equal(result.ids, [1, 0])

    def test_partial_can_be_start_only_and_never_advances_to_later_video_pose(self):
        cm = self.cost_map([[0, 0, 0], [3, 0, 0], [3.3, 0, 0]], [1] * 3, connection=.35)
        start, goal, result, _ = self.segment(cm, self.poses([0, 1, 2]))
        self.assertEqual((start['id'], goal['id'], result.status, result.total_cost), (0, 2, 'partial', 0))
        np.testing.assert_array_equal(result.ids, [0])
        start, goal, result, attempts = self.segment(cm, self.poses([1, 0, 1]))
        self.assertEqual((start['id'], goal['id'], result.status, attempts), (1, 1, 'ok', 0))

    def test_partial_certificate_removes_false_reachability_before_selecting_end(self):
        cm = self.cost_map([[0, 0, 0], [.3, 0, 0], [.6, 0, 0], [.45, 0, 0], [3, 0, 0]],
                           [1, 1, 1, np.inf, 1], [True, True, True, False, True], connection=.35)
        updates = []
        with patch.object(stage, 'feasible_bound', return_value=(None, None)):
            result = stage.exact_route(cm, 0, 4, np.array([0, 4]), python_relaxation, find_path,
                                      lambda name, **kw: updates.append((name, kw)), allow_partial=True)
        np.testing.assert_array_equal(result.ids, [0, 1])
        self.assertEqual(result.status, 'partial')
        self.assertEqual(sum(name == 'exact_path_search' for name, _ in updates), 2)
        self.assertTrue(any(name == 'candidate_edges_rejected' for name, _ in updates))

    def test_partial_exhaustion_covers_long_detour_without_sampled_bound(self):
        cm = self.cost_map([[0, 0, 0], [0, 1, 0], [0, 2, 0], [1, 2, 0],
                            [2, 2, 0], [2, 1, 0], [2, 0, 0], [4, 0, 0]], [1] * 8)
        with patch.object(stage, 'feasible_bound', return_value=(None, None)):
            _, _, result, _ = self.segment(cm, self.poses([0, 7]))
        self.assertEqual((result.status, result.ids[-1], result.total_cost), ('partial', 6, 6))

    def test_fixed_start_matches_exhaustive_original_graph(self):
        rng = np.random.default_rng(421)
        for _ in range(10):
            points = np.column_stack((rng.integers(0, 5, 12) * .3,
                                       rng.integers(0, 3, 12) * .3, np.zeros(12)))
            cm = self.cost_map(points, rng.uniform(1, 3, 12), connection=.31)
            start, goal = map(int, rng.choice(12, 2, replace=False))
            reference = [find_path(cm.points, cm, start, row) for row in range(12)]
            reached = min((row for row, path in enumerate(reference) if path.status == 'ok'),
                          key=lambda row: (np.linalg.norm(points[row] - points[goal]), row))
            if reference[goal].status == 'ok':
                reached = goal
            with patch.object(stage, 'feasible_bound', return_value=(None, None)):
                _, intended, result, _ = self.segment(cm, self.poses([start, goal]))
            self.assertEqual((intended['id'], result.ids[0], result.ids[-1]), (goal, start, reached))
            self.assertAlmostEqual(result.total_cost, reference[reached].total_cost)
            self.assertEqual(result.status, 'ok' if reached == goal else 'partial')

    def test_saved_partial_report_keeps_intended_goal_and_labels_actual_end(self):
        import gc
        import hashlib
        import json
        import cost_map
        cm = self.cost_map([[0, 0, 0], [.3, 0, 0], [3, 0, 0]], [1] * 3, connection=.35)
        poses = [p | {'association_mode': 'beneath_camera'} for p in self.poses([0, 2])]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            output = Path(folder)
            arrays = dict(points=cm.points, cost=cm.cost, support_mask=np.ones(3, bool), reference=cm.reference,
                          cameras=np.array([[0, 0, 1], [3, 0, 1]]), timestamps=np.arange(2), frame_ids=np.array([0, 0, 1]))
            for name, value in arrays.items():
                np.save(output / (name + '.npy'), value)
            stage.atomic_json(output / 'report.json', dict(frame_count=2, total_rows=3, correspondence_verified=True))
            settings = {k: v for k, v in cm.config.items() if k != 'support_mask'}
            stage.atomic_json(output / 'provenance.json', dict(settings=settings, correspondence_verified=True,
                              cost_source_sha256=hashlib.sha256(Path(cost_map.__file__).read_bytes()).hexdigest()))
            args = SimpleNamespace(output=output, repo=Path(os.environ.get('DATASET_REPO', Path.cwd())), library=None)
            with patch.object(stage, 'associate', return_value=poses), patch.object(stage, 'NativeSearch', return_value=python_relaxation):
                self.assertEqual(stage.run(args), 0)
            gc.collect()  # Release the restored map's bound-method cycle before Windows removes memmaps.
            report = json.loads((output / 'report.json').read_text())
            self.assertEqual((report['status'], report['goal_reached'], report['optimality_scope']),
                              ('partial', False, 'nearest_reachable_endpoint'))
            self.assertEqual((report['selected']['start']['id'], report['selected']['goal']['id'], report['reached']['id']), (0, 2, 1))
            self.assertTrue(report['optimal'] and report['all_edges_verified'])
            self.assertFalse(report['used_connected_subinterval'])

    def test_solver_failure_with_known_feasible_bound_is_not_no_path(self):
        cm = self.cost_map([[0, 0, 0], [.3, 0, 0]], [1, 1], connection=.35)
        with self.assertRaisesRegex(RuntimeError, 'lost a previously verified'):
            self.solve(cm, 0, 1, lambda *args: (0, None, None))


    def test_exhausted_cut_covers_distant_component_and_zero_edges(self):
        for connection, x in ((.35, [0, .3, .6, 100, 100.3, 100.6]),
                              (0, [0, 0, 0, 100, 100, 100])):
            with self.subTest(connection=connection):
                cm = self.cost_map([[v, 0, 0] for v in x], [1] * 6, connection=.35)
                cm.config['connection_radius'] = connection
                cuts = []
                result = stage.exact_route(cm, 0, 5, np.arange(6), self.native(), find_path,
                                           lambda *args, **kwargs: None,
                                           lambda reached: cuts.append(set(map(int, reached))))
                self.assertIsNone(result)
                self.assertEqual(cuts, [{0, 1, 2}])
        # The far component requires a six-edge U-shaped detour between its tips.
        cm = self.cost_map([[0, 0, 0], [100, 0, 0], [101, 0, 0], [102, 0, 0],
                            [102, 1, 0], [102, 2, 0], [101, 2, 0], [100, 2, 0]], [1] * 8)
        cuts = []
        stage.exact_route(cm, 7, 0, np.arange(8), self.native(), find_path,
                          lambda *args, **kwargs: None,
                          lambda reached: cuts.append(set(map(int, reached))))
        self.assertEqual(cuts, [set(range(1, 8))])





    def test_support_witness_uses_exact_production_probe_and_keeps_valid_route(self):
        cm = self.cost_map([[0, 0, 0], [.06, 0, 0], [.23, 0, 0], [.249, 0, 0]],
                           [1] * 4, connection=.35)
        cm.config['max_gap'] = .2
        diagnostic = stage.edge_diagnostic(cm, 0, 3)
        self.assertEqual((diagnostic['steps'], diagnostic['reason']), (5, 'support_gap'))
        expected_probe = cm.points[0] + (np.arange(6) / 5)[3] * (cm.points[3] - cm.points[0])
        np.testing.assert_array_equal(diagnostic['bad_probe'], expected_probe)
        self.assertEqual(diagnostic['bad_probe_distance'], cm.support.query(expected_probe)[0])
        expected = find_path(cm.points, cm, 0, 3)
        recorder = WitnessRecorder(self.native())
        result = self.solve(cm, 0, 3, recorder)
        self.assertAlmostEqual(result.total_cost, expected.total_cost)
        self.assertTrue(any(batch['support_witnesses'] for batch in recorder.batches))
        for point, distance in cm._path_support_witnesses.items():
            self.assertEqual(distance, cm.support.query(point)[0])
        # Witnesses can be reused across another endpoint pair in this same map.
        self.solve(cm, 1, 3, recorder)
        self.assertTrue(recorder.batches[-1]['support_witnesses'])

    def test_obstacle_witness_is_observed_row_and_keeps_valid_detour(self):
        cm = self.cost_map([[0, 0, 0], [.5, 0, 0], [1, 0, 0],
                            [0, .5, 0], [.5, .5, 0], [1, .5, 0], [.25, 0, 0]],
                           [1] * 6 + [np.inf], [True] * 6 + [False], connection=.51)
        expected = find_path(cm.points, cm, 0, 2)
        recorder = WitnessRecorder(self.native())
        result = self.solve(cm, 0, 2, recorder)
        self.assertEqual(result.total_cost, expected.total_cost)
        self.assertEqual(result.total_cost, 2)
        self.assertTrue(any(batch['obstacle_witnesses'] for batch in recorder.batches))
        self.assertEqual(set(cm._path_obstacle_witnesses), {6})
        np.testing.assert_array_equal(cm._path_obstacle_witnesses[6], cm.points[6])
        self.assertFalse(cm.config['support_mask'][6])
        self.assertEqual(len(cm.points), 7)

    def test_full_feasibility_adapter_keeps_original_support_outside_search_ellipse(self):
        cm = self.cost_map([[0, 0, 0], [.3, 0, 0], [.6, 0, 0], [100, 0, 0], [100, 1, 0]],
                           [1, 1, 1, 1, np.inf], [True, True, True, True, False], connection=.35)
        prepared = stage.NativeSearch.__new__(stage.NativeSearch)
        prepared.prepare_map(cm, cm.support_rows)
        self.assertTrue(np.shares_memory(prepared._original, cm.points))
        self.assertTrue(np.shares_memory(prepared._support_rows, cm.support_rows))
        self.assertTrue(np.shares_memory(prepared._obstacle_rows, cm.obstacle_rows))
        class Probe:
            supports_feasibility = True
            def prepare_map(self, mapped, rows):
                self.support_rows = rows.copy()
                self.obstacle_rows = mapped.obstacle_rows.copy()
            def __call__(self, *args, **kwargs):
                self.search_rows = len(args[0])
                return python_relaxation(*args)
        probe = Probe()
        self.assertAlmostEqual(self.solve(cm, 0, 2, probe).total_cost, .6)
        self.assertEqual(probe.search_rows, 3)
        np.testing.assert_array_equal(probe.support_rows, [0, 1, 2, 3])
        np.testing.assert_array_equal(probe.obstacle_rows, [4])

    def test_empty_failure_artifact_and_strict_json(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            output = Path(directory)
            stage.save_route(output, [], np.empty((0, 3)), np.inf)
            with np.load(output / 'route.npz') as result:
                self.assertEqual(result['positions'].shape, (0, 3))
                self.assertTrue(np.isinf(result['total_cost']))
            stage.atomic_json(output / 'status.json', {'total_cost': np.inf})
            self.assertEqual((output / 'status.json').read_text(), '{\n  "total_cost": null\n}')


if __name__ == '__main__':
    unittest.main()
