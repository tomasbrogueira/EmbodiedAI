import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import sys
from copy import deepcopy
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from build_navigation_replay import build_replay, build_batch, replay_reference, sha256
from derive_pose_corrected_run import numeric_for_derivation, CORRECTED_METHOD
from export_pipeline_video import _digest
from path_mapping.runner import geometry_fingerprint
from pipeline_research_plan import DEFAULT_ASSUMPTIONS


class ReplaySourceAdmissionTests(unittest.TestCase):
    def fixture(self, root):
        inputs = root/'inputs'
        inputs.mkdir()
        geometry = inputs/'geometry.npz'
        geometry.write_bytes(b'not read before source admission')
        source = dict(geometry_fingerprint='a'*64, input_fingerprint='b'*64,
                      processed_grid_id='c'*64, map_frame='world', units='reconstruction_units',
                      recording_id='clip', derived_geometry={'source_archive_sha256': hashlib.sha256(geometry.read_bytes()).hexdigest()})
        sequence = {'sequence_id': 'clip', 'frames': [{'frame_id': 'first', 'timestamp_ns': 0}]}
        sequence['manifest_digest'] = _digest(sequence)
        manifest = {**{k: source[k] for k in ('geometry_fingerprint', 'input_fingerprint', 'processed_grid_id', 'map_frame', 'units')},
                    'sequence_digest': sequence['manifest_digest']}
        files = [geometry, inputs/'plan.json', inputs/'sequence.json', inputs/'manifest.json']
        for path, value in zip(files[1:], [{'source': source}, sequence, manifest]):
            path.write_text(json.dumps(value))
        return files, root/'result'

    def test_changed_sequence_cannot_reuse_stored_digest(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
            files, output = self.fixture(Path(d))
            sequence = json.loads(files[2].read_text())
            sequence['frames'][0]['timestamp_ns'] = 99
            files[2].write_text(json.dumps(sequence))
            with self.assertRaisesRegex(ValueError, 'digest does not match'):
                build_replay(*files, output)
            self.assertFalse(output.exists())

    def test_reference_plan_cannot_relabel_the_geometry_grid(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
            files, output = self.fixture(Path(d))
            plan = json.loads(files[1].read_text())
            plan['source']['processed_grid_id'] = 'z'*64
            files[1].write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, 'processed_grid_id'):
                build_replay(*files, output)
            self.assertFalse(output.exists())

    def test_horizon_longer_than_requested_contract_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
            files, output = self.fixture(Path(d))
            with self.assertRaisesRegex(ValueError, r'\(0,2\]'):
                build_replay(*files, output, horizon_m=3)
            self.assertFalse(output.exists())

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
            files, output = self.fixture(Path(d))
            output.mkdir()
            (output/'keep.txt').write_text('existing export')
            with self.assertRaisesRegex(ValueError, 'fresh output'):
                build_replay(*files, output)
            self.assertEqual((output/'keep.txt').read_text(), 'existing export')


class GenericRecordingReplayTests(unittest.TestCase):
    def sources(self, root, *, recording='outdoor_stairs', empty_first=False, unverified=False, no_plane=False):
        inputs = root / recording
        inputs.mkdir()
        n, h, w = 3, 8, 8
        rotation = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]], np.float32)
        poses = np.empty((n, 3, 4), np.float32)
        poses[:, :, :3] = rotation
        poses[:, :, 3] = [[0., 0., 1.], [.2, 0., 1.], [5., 0., 1.]]
        intrinsic = np.tile(np.array([[4., 0., 3.5], [0., 4., 0.], [0., 0., 1.]], np.float32), (n, 1, 1))
        depth = np.broadcast_to((4 / np.maximum(np.arange(h), 1))[None, :, None, None], (n, h, w, 1)).astype(np.float32).copy()
        confidence = np.full((n, h, w), 2., np.float32)
        if empty_first:
            confidence[0] = 0.
        original = {'images': np.zeros((n, h, w, 3), np.uint8), 'depth': depth,
                    'world_points': np.zeros((n, h, w, 3), np.float32), 'world_points_conf': confidence,
                    'intrinsic': intrinsic, 'extrinsic': poses}
        geometry = inputs/'geometry.npz'
        np.savez_compressed(geometry, **original)
        derivation = {'method': 'saved_pose_passthrough_unverified_display_only' if unverified else CORRECTED_METHOD,
                      'pose_correction_verified': not unverified, 'source_archive_sha256': sha256(geometry)}
        numeric = numeric_for_derivation(original, derivation)
        source = dict(geometry_fingerprint=geometry_fingerprint(numeric), input_fingerprint='b'*64,
                      processed_grid_id='c'*64, map_frame='world', units='reconstruction_units',
                      archive_sha256='d'*64, recording_id=recording, derived_geometry=derivation,
                      voxel_origin_native=[0., 0., 0.], voxel_size_native=.05)
        frames = [{'frame_id': f'frame_{i:06d}', 'timestamp_ns': i*500_000_000,
                   'timestamp_provenance': {'source_frame_index': i*15}} for i in range(n)]
        sequence = {'sequence_id': recording, 'frames': frames}
        sequence['manifest_digest'] = _digest(sequence)
        manifest = {**{k: source[k] for k in ('geometry_fingerprint', 'input_fingerprint', 'processed_grid_id', 'map_frame', 'units', 'archive_sha256')},
                    'sequence_digest': sequence['manifest_digest'], 'settings': {'min_confidence': 1.5},
                    'transforms': [{'pad_ltrb': [0, 0, 0, 0]} for _ in frames], 'derivation': derivation}
        assumptions = deepcopy(DEFAULT_ASSUMPTIONS)
        assumptions['assumption_id'] = recording+'_upright_camera_scale_assumptions'
        plan = {'source': source, 'assumptions': assumptions,
                'status': 'blocked_inputs' if no_plane else 'ok',
                'reason': 'No observed plane' if no_plane else None,
                'assumed_up_vector': None if no_plane else [0., 0., 1.],
                'ground_plane': None if no_plane else {'normal_native': [0., 0., 1.], 'point_native': [0., 0., 0.]},
                'slope_reference': {'source': 'assumed_upright_first_camera'}}
        files = [geometry, inputs/'reference.json', inputs/'sequence.json', inputs/'manifest.json']
        for path, value in zip(files[1:], (plan, sequence, manifest)):
            path.write_text(json.dumps(value), encoding='utf-8')
        return files, plan, manifest, numeric

    def supported_map(self, *args, **kwargs):
        shape = (50, 30)
        arrays = {'origin': np.array([-1.5, 0.]), 'resolution': np.array([.1]),
                  'projection_basis': np.eye(3), 'geometry_state': np.full(shape, 2, np.uint8),
                  'decision_state': np.full(shape, 2, np.uint8), 'support_height': np.zeros(shape),
                  'policy_blocked_mask': np.zeros(shape, bool), 'costs': np.ones(shape)}
        metadata = {'availability': 'available', 'footprint_radius_with_clearance': .12,
                    'profile': deepcopy(DEFAULT_ASSUMPTIONS['robot'])}
        return arrays, metadata

    def test_late_support_initializes_at_current_observation_then_never_relocates(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            files, _, _, _ = self.sources(root, empty_first=True)
            before = {str(p): sha256(p) for p in files}
            with patch('build_navigation_replay.research_costmap', side_effect=self.supported_map):
                replay = build_replay(*files, root/'replay')
            self.assertEqual(replay['mission']['id'], 'outdoor_stairs_persistent_forward_corridor_v1')
            self.assertEqual(replay['mission']['initialization_observed_index'], 1)
            self.assertFalse(replay['source']['level_floor_assumed'])
            self.assertEqual([f['status'] for f in replay['frames']], ['awaiting_support', 'ok', 'awaiting_support'])
            first, initialized, lost = replay['frames']
            self.assertIsNone(first['agent_ground_point_native'])
            self.assertEqual(first['diagnostics']['accepted_prefix_points'], 0)
            self.assertEqual(first['diagnostics']['occupied_prefix_voxels'], 0)
            np.testing.assert_allclose(initialized['agent_requested_xy_assumed_m'], replay['mission']['initial_agent_ground_native'][:2])
            self.assertAlmostEqual(initialized['diagnostics']['display_length_assumed_m'], 2.)
            self.assertEqual(lost['path_points'], [])
            self.assertIsNone(lost['agent_ground_point_native'])
            self.assertEqual(replay['terminal_state']['retained_display_path_points'], [])
            self.assertFalse(replay['terminal_state']['mission_completed'])
            self.assertEqual(replay['terminal_state']['execution_state'], 'awaiting_observations')
            self.assertEqual(before, {str(p): sha256(p) for p in files})
            with np.load(root/'replay'/first['costmap_file'], allow_pickle=False) as saved:
                self.assertFalse(saved['decision_state'].any())
                self.assertFalse(saved['observed_mask'].any())

    def test_unavailable_plane_produces_all_unknown_no_ground_and_no_route(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            files, _, _, numeric = self.sources(root, no_plane=True)
            with patch('build_navigation_replay.research_costmap') as calculate:
                replay = build_replay(*files, root/'replay')
            calculate.assert_not_called()
            expected = -numeric['extrinsic'][0, 1, :3]
            np.testing.assert_allclose(replay['assumed_up_vector'], expected/np.linalg.norm(expected))
            self.assertEqual(replay['source']['basis_source'], 'assumed_upright_first_camera_for_unavailable_map')
            self.assertFalse(replay['mission']['initialized_on_observed_support'])
            for row in replay['frames']:
                self.assertEqual(row['status'], 'awaiting_support')
                self.assertIsNone(row['agent_ground_point_native'])
                self.assertEqual(row['path_points'], [])
                self.assertFalse(row['diagnostics']['research_inputs_available'])

    def test_pose_unverified_suppresses_plan_even_with_available_plane_and_support(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            files, _, _, _ = self.sources(root, unverified=True)
            with patch('build_navigation_replay.research_costmap', side_effect=self.supported_map) as calculate:
                replay = build_replay(*files, root/'replay')
            calculate.assert_not_called()
            self.assertFalse(replay['source']['pose_correction_verified'])
            for row in replay['frames']:
                self.assertEqual(row['status'], 'pose_unverified')
                self.assertEqual(row['navigation_state']['execution_state'], 'pose_unverified')
                self.assertEqual(row['display_path_points'], [])
                self.assertIsNone(row['agent_ground_point_native'])

    def test_foreign_level_declaration_is_refused_and_slope_up_is_retained(self):
        assumptions = deepcopy(DEFAULT_ASSUMPTIONS)
        up = np.array([0., -.1, 1.]); up /= np.linalg.norm(up)
        plan = {'assumptions': assumptions, 'status': 'ok', 'assumed_up_vector': up.tolist(),
                'ground_plane': {'normal_native': [0., -.3, 1.], 'point_native': [0., 0., 0.]}}
        manifest = {'geometry_fingerprint': 'a'*64}
        _, retained, _, _, policy = replay_reference(plan, manifest, 'stairs', np.eye(3)[None])
        np.testing.assert_allclose(retained, up)
        self.assertFalse(policy['level_floor_assumed'])
        plan['assumptions']['level_reference'] = {'recording_id': '5506', 'geometry_fingerprint': 'a'*64,
            'declaration': 'user_declared_level_floor', 'evidence': 'Only 5506 is declared level'}
        with self.assertRaisesRegex(ValueError, 'another recording'):
            replay_reference(plan, manifest, 'stairs', np.eye(3)[None])

    def test_batch_retains_independent_missions_and_receipts(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            entries = []
            for name in ('indoor_a', 'outdoor_stairs'):
                files, _, _, _ = self.sources(root, recording=name, no_plane=True)
                entries.append({'recording_id': name, **{k: str(p) for k, p in zip(
                    ('geometry', 'reference_plan', 'sequence', 'geometry_manifest'), files)}})
            config = root/'batch.json'
            config.write_text(json.dumps({'schema_version': 1,
                'artifact_kind': 'research_navigation_replay_batch_config', 'research_illustration': True,
                'recordings': entries}), encoding='utf-8')
            result = build_batch(config, root/'batch_output')
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(len(result['recordings']), 2)
            for row in result['recordings']:
                self.assertEqual(row['frame_status_counts']['awaiting_support'], 3)
                saved = json.loads((Path(row['output'])/'replanning_replay.json').read_text())
                self.assertTrue(saved['mission']['id'].startswith(row['recording_id']))
                self.assertFalse(saved['terminal_state']['mission_completed'])

    def test_already_corrected_archive_is_admitted_without_a_second_inversion(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            files, plan, manifest, numeric = self.sources(root, no_plane=True)
            corrected = files[0].parent/'corrected.npz'
            np.savez_compressed(corrected, **numeric)
            manifest['archive_sha256'] = plan['source']['archive_sha256'] = sha256(corrected)
            files[1].write_text(json.dumps(plan), encoding='utf-8')
            files[3].write_text(json.dumps(manifest), encoding='utf-8')
            files[0] = corrected
            replay = build_replay(*files, root/'replay', geometry_kind='corrected_w2c')
            self.assertEqual(replay['source']['geometry_fingerprint'], geometry_fingerprint(numeric))
            self.assertEqual(replay['source']['geometry_input_kind'], 'corrected_w2c')
            rotation, translation = numeric['extrinsic'][1, :, :3], numeric['extrinsic'][1, :, 3]
            np.testing.assert_allclose(replay['frames'][1]['camera_center_native'], -rotation.T@translation)
            self.assertEqual(replay['frames'][1]['path_points'], [])


if __name__ == '__main__':
    unittest.main()
