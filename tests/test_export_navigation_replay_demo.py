"""Metric horizon and replay-presentation regressions; no models or planner."""
from __future__ import annotations

from pathlib import Path
import copy
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from export_navigation_replay_demo import (
    admit_numeric_derivation, admit_reference_and_mission, admit_row_state,
    admitted_display_route, circle_clipped_routes, map_box_for_size,
    metric_scale_available, presentation_size, render_frame, replay_status_label, semantic_mask_path, uses_display_up_fallback,
)
from export_pipeline_video import sampled_timeline


class LocalPreviewTests(unittest.TestCase):
    def reference_fixture(self):
        identity = {key: key + '_identity' for key in ['geometry_fingerprint', 'input_fingerprint', 'processed_grid_id', 'map_frame', 'units']}
        paths = [Path('/geometry.npz').resolve(), Path('/sequence.json').resolve(), Path('/manifest.json').resolve(), Path('/reference.json').resolve()]
        hashes = {str(path): str(i) * 64 for i, path in enumerate(paths)}
        reference = {'source': identity.copy(), 'assumptions': {'metres_per_native_unit': 4.}, 'assumed_up_vector': [0, -1, 0]}
        replay = copy.deepcopy(reference)
        replay['source']['local_input_sha256'] = hashes.copy()
        replay['mission'] = {'policy': 'persistent_forward_corridor', 'id': 'test_forward', 'state': 'active', 'completed': False,
                             'origin_native': [0, 0, 0], 'forward_native': [0, 0, 1]}
        return replay, identity, reference, hashes, paths

    def test_modified_reference_assumptions_identity_and_missing_local_hash_are_refused(self):
        original = self.reference_fixture()
        admit_reference_and_mission(*original)
        for mutation in ['scale', 'up', 'identity', 'hash', 'mission']:
            with self.subTest(mutation=mutation):
                replay, manifest, reference, hashes, paths = copy.deepcopy(original)
                if mutation == 'scale':
                    replay['assumptions']['metres_per_native_unit'] = 1.
                elif mutation == 'up':
                    replay['assumed_up_vector'] = [0, 0, 1]
                elif mutation == 'identity':
                    reference['source']['geometry_fingerprint'] = 'other'
                elif mutation == 'hash':
                    del replay['source']['local_input_sha256'][str(paths[-1])]
                else:
                    replay['mission']['completed'] = True
                with self.assertRaises(ValueError):
                    admit_reference_and_mission(replay, manifest, reference, hashes, paths)

    def test_row_execution_and_lifetime_must_match_status(self):
        row = {'status': 'awaiting_observation', 'path_points': [], 'display_path_points': [],
               'navigation_state': {'execution_state': 'awaiting_observations', 'mission_state': 'active', 'mission_completed': False}}
        admit_row_state(row)
        row['navigation_state']['execution_state'] = 'ready'
        with self.assertRaises(ValueError):
            admit_row_state(row)
        row['navigation_state']['execution_state'] = 'awaiting_observations'
        row['display_path_points'] = [[0, 0, 0], [0, 1, 0]]
        with self.assertRaises(ValueError):
            admit_row_state(row)

    def test_outside_segment_crosses_circle_with_exact_boundary_intersections(self):
        pieces = circle_clipped_routes([[-3, 0, 7], [3, 0, 9]], [0, 0, 0], np.eye(3), 1.)
        self.assertEqual(len(pieces), 1)
        np.testing.assert_allclose(pieces[0], [[-2, 0, 7 + 1/3], [2, 0, 8 + 2/3]])

    def test_outside_excursion_does_not_connect_two_inside_pieces(self):
        pieces = circle_clipped_routes([[0, 0, 0], [3, 0, 0], [3, 3, 0], [0, 0, 0]],
                                      [0, 0, 0], np.eye(3), 1.)
        self.assertEqual(len(pieces), 2)
        np.testing.assert_allclose(pieces[0][-1], [2, 0, 0])
        np.testing.assert_allclose(pieces[1][0], [np.sqrt(2), np.sqrt(2), 0])
        self.assertFalse(np.array_equal(pieces[0][-1], pieces[1][0]))

    def test_declared_scale_and_ground_basis_control_horizontal_radius(self):
        basis = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)
        pieces = circle_clipped_routes([[0, 0, 0], [0, 20, 2]], [0, 0, 0], basis, 4.)
        np.testing.assert_allclose(pieces[0][-1], [0, 5, .5])
        self.assertAlmostEqual(np.linalg.norm(pieces[0][-1] @ basis[:2].T * 4), 2.)

    def test_tangent_and_degenerate_outside_segments_are_not_drawn(self):
        self.assertEqual(circle_clipped_routes([[-3, 2, 0], [3, 2, 0]], [0, 0, 0], np.eye(3), 1.), [])
        self.assertEqual(circle_clipped_routes([[3, 0, 0], [3, 0, 1]], [0, 0, 0], np.eye(3), 1.), [])

    def test_invalid_basis_or_nonmetric_scale_is_refused(self):
        with self.assertRaises(ValueError):
            circle_clipped_routes([], [0, 0, 0], np.ones((3, 3)), 1.)
        with self.assertRaises(ValueError):
            circle_clipped_routes([], [0, 0, 0], np.eye(3), 0.)

    def test_final_hold_keeps_guidance_without_mutating_active_mission(self):
        part = np.array([[0, 0, 0], [0, 1, 0]], float)
        frame = {
            'row': {'timestamp_ns': 0, 'navigation_state': {'execution_state': 'ready', 'mission_state': 'active'}},
            'rgb': np.full((8, 8, 3), 130, np.uint8), 'floor': np.ones((8, 8), bool),
            'transform': {'matrix': np.eye(3).tolist()}, 'parts': [part.copy()],
            'projections': [], 'agent': np.zeros(3), 'agent_xy': np.zeros(2),
            'costmap': {'origin': np.array([-3., -3.]), 'resolution': np.array([.1]),
                        'decision_state': np.full((60, 60), 2, np.uint8)},
        }
        data = {'frames': [frame], 'basis': np.eye(3), 'scale': 1.}
        active = np.asarray(render_frame(data, 0))
        paused = np.asarray(render_frame(data, 0, paused=True))
        # Pausing changes only the bottom status, retaining the video and local map.
        np.testing.assert_array_equal(active[:-100], paused[:-100])
        self.assertFalse(np.array_equal(active[-100:], paused[-100:]))
        np.testing.assert_array_equal(frame['parts'][0], part)
        self.assertEqual(frame['row']['navigation_state']['mission_state'], 'active')

    def test_bent_route_inside_circle_cannot_expose_more_than_two_metres_of_arc(self):
        row = {'path_points': [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
               'display_path_points': [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
               'agent_ground_point_native': [0, 0, 0], 'agent_requested_xy_assumed_m': [0, 0]}
        with self.assertRaises(ValueError):
            admitted_display_route(row, 1.)
        row['display_path_points'] = [[0, 0, 0], [1, 0, 0], [1, 1, 0]]
        display, agent, requested = admitted_display_route(row, 1.)
        self.assertEqual(len(display), 3)

    def test_waiting_without_ground_has_requested_xy_only_and_no_route(self):
        row = {'path_points': [], 'display_path_points': [],
               'agent_ground_point_native': None, 'agent_requested_xy_assumed_m': [.2, 1.]}
        display, agent, requested = admitted_display_route(row, 4.)
        self.assertIsNone(agent)
        self.assertEqual(len(display), 0)
        np.testing.assert_array_equal(requested, [.2, 1.])
        row['path_points'] = [[0, 0, 0], [.1, 0, 0]]
        row['display_path_points'] = row['path_points']
        with self.assertRaises(ValueError):
            admitted_display_route(row, 4.)

    def test_source_sample_holds_and_final_two_seconds_at_twenty_fps(self):
        timeline = sampled_timeline([{'timestamp_ns': i * 500500000} for i in range(22)], video_fps=20, end_hold_seconds=2)
        self.assertEqual(timeline['repeat_counts'][:-1], [10] * 21)
        self.assertEqual(timeline['repeat_counts'][-1], 40)
        self.assertEqual(sum(timeline['repeat_counts']), 250)

    def test_generic_portrait_and_landscape_layout_preserve_source_aspect(self):
        self.assertEqual(presentation_size((1280, 720)), (1080, 1920))
        self.assertEqual(presentation_size((720, 1280)), (1920, 1080))
        for shape in ((1280, 720), (720, 1280)):
            size = presentation_size(shape)
            left, top, right, bottom = map_box_for_size(size)
            self.assertTrue(0 < left < right < size[0])
            self.assertTrue(0 < top < bottom < size[1])
            self.assertEqual(right-left, bottom-top)

    def test_unverified_original_pose_has_no_ground_or_route_and_visible_status(self):
        row = {'status': 'pose_unverified', 'path_points': [], 'display_path_points': [],
               'agent_ground_point_native': None,
               'navigation_state': {'execution_state': 'pose_unverified', 'mission_state': 'active', 'mission_completed': False}}
        admit_row_state(row)
        self.assertEqual(replay_status_label({'row': row}), 'Pose unverified')
        self.assertEqual(replay_status_label({'row': row}, paused=True), 'Pose unverified')
        row['agent_ground_point_native'] = [0, 0, 0]
        with self.assertRaises(ValueError):
            admit_row_state(row)

    def test_display_only_upright_fallback_requires_blocked_reference_and_no_path(self):
        replay, manifest, reference, hashes, paths = self.reference_fixture()
        replay['source']['basis_source'] = 'assumed_upright_first_camera_for_unavailable_map'
        replay['frames'] = [{'path_points': [], 'display_path_points': []}]
        reference.update(status='blocked_inputs', assumed_up_vector=None)
        admit_reference_and_mission(replay, manifest, reference, hashes, paths)
        self.assertTrue(uses_display_up_fallback(replay, reference))
        replay['frames'][0]['path_points'] = [[0, 0, 0], [0, 0, 1]]
        with self.assertRaises(ValueError):
            uses_display_up_fallback(replay, reference)
        replay['frames'][0]['path_points'] = []
        reference['assumed_up_vector'] = [0, -1, 0]
        with self.assertRaises(ValueError):
            uses_display_up_fallback(replay, reference)

    def test_reference_numeric_dispatch_policy_cannot_disagree_with_manifest(self):
        derivation = {'method': 'saved_pose_passthrough_unverified_display_only',
                      'source_archive_sha256': 'a'*64, 'source_geometry_fingerprint': 'b'*64,
                      'pose_correction_verified': False, 'planning_admitted': False}
        reference = {'source': {'derived_geometry': derivation.copy()}}
        admit_numeric_derivation(reference, derivation)
        reference['source']['derived_geometry']['pose_correction_verified'] = True
        with self.assertRaises(ValueError):
            admit_numeric_derivation(reference, derivation)

    def test_metric_marker_requires_applied_height_reference_and_verified_pose(self):
        reference = {'research_calibration': {'camera_height_reference': {'availability': 'applied'}}}
        self.assertTrue(metric_scale_available(reference, True))
        self.assertFalse(metric_scale_available(reference, False))
        self.assertFalse(metric_scale_available(reference, True, display_up_fallback=True))
        reference['research_calibration']['camera_height_reference']['availability'] = 'unavailable'
        self.assertFalse(metric_scale_available(reference, True))

    def test_standard_derived_and_flattened_semantic_mask_roots_remain_exact(self):
        base = Path('/test_derived_run').resolve()
        self.assertEqual(semantic_mask_path(base/'semantics/frames.jsonl', 'semantics/masks/mask.npz'),
                         base/'semantics/masks/mask.npz')
        self.assertEqual(semantic_mask_path(base/'data/semantics_frames.jsonl', 'masks/mask.npz'),
                         base/'data/masks/mask.npz')
        self.assertEqual(semantic_mask_path(base/'data/frames.jsonl', 'masks/mask.npz', base),
                         base/'masks/mask.npz')
        with self.assertRaises(ValueError):
            semantic_mask_path(base/'semantics/frames.jsonl', '../outside.npz')

    def test_unit_up_renormalization_roundoff_is_admitted_but_changed_direction_is_not(self):
        replay, manifest, reference, hashes, paths = self.reference_fixture()
        reference['assumed_up_vector'] = [0, -.9999999999999999, 0]
        admit_reference_and_mission(replay, manifest, reference, hashes, paths)
        replay['assumed_up_vector'] = [1e-5, -1, 0]
        with self.assertRaises(ValueError):
            admit_reference_and_mission(replay, manifest, reference, hashes, paths)


if __name__ == '__main__':
    unittest.main()
