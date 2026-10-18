"""Cost-only migration keeps geometry/slopes immutable and rejects stale provenance."""
import json
import gc
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

import dataset_prepare as prepare
import dataset_path as path_stage
from dataset_path import verify_cost_source


class CostRevisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.close_arrays)
        self.root = Path(self.temp.name)
        self.out, self.repo = self.root / 'video', self.root / 'repo'
        self.data = self.out / 'data'
        self.data.mkdir(parents=True)
        (self.repo / 'path-planning').mkdir(parents=True)
        (self.repo / 'orchestrator').mkdir()
        self.source = self.repo / 'path-planning/cost_map.py'
        self.source.write_text('new cost source')
        (self.repo / 'path-planning/slope.py').write_text('unchanged slopes')
        video = self.root / 'input.mp4'
        video.write_bytes(b'unchanged recording')
        self.entry = dict(name='test', input_sha256=prepare.digest(video), video=str(video),
                          prompt='ground', frame_count=2, timestamps=[0., 1.], model_settings={'same': True})
        self.gpu = dict(input_sha256=self.entry['input_sha256'], prompt='ground', frames=2,
                        model_settings=self.entry['model_settings'], lingbot=str(self.root / 'lingbot'),
                        sam=str(self.root / 'sam'))
        for name, value in [('entry', self.entry), ('gpu', self.gpu)]:
            prepare.write(self.out / f'{name}.json', value)
        self.identity = dict(input_sha256=self.entry['input_sha256'], prompt='ground',
            model_settings=self.entry['model_settings'], frame_count=2, timestamps=[0., 1.],
            lingbot=self.gpu['lingbot'], sam=self.gpu['sam'],
            filters=dict(frame_step=2, pixel_step=1, min_depth_conf=2.5, max_depth=25., patch_size=14),
            source_hashes={str(path.relative_to(self.repo)): prepare.digest(path)
                          for folder in ('path-planning', 'orchestrator')
                          for path in sorted((self.repo / folder).glob('*.py'))})
        self.old = self.identity | {'source_hashes': self.identity['source_hashes'] |
                                   {str(Path('path-planning/cost_map.py')): 'old cost source'}}
        self.arrays = dict(points=np.array([[0., 0, 0], [.1, 0, 0]], dtype=np.float32),
                          slope=np.zeros(2), quality=np.ones(2), done=np.ones(2, dtype=bool),
                          support_mask=np.ones(2, dtype=bool), frame_ids=np.array([0, 0]),
                          reference=np.array([0., 0, 1.]), cameras=np.array([[0., 0, 1.]] * 2),
                          timestamps=np.array([0., 1.]))
        for name, values in self.arrays.items():
            np.save(self.data / f'{name}.npy', values)

    def close_arrays(self):
        for name in ('POINTS', 'SLOPES', 'QUALITY', 'DONE'):
            if hasattr(prepare, name):
                delattr(prepare, name)
        gc.collect()

    def legacy_receipts(self):
        key = prepare.identity_hash(self.old)
        prepare.write(self.out / 'geometry.json', dict(identity=key, total_rows=2,
            reference_available=True, reference=[0., 0, 1.], reference_scan_version=1,
            reference_report={'slope_config': {'radius': .2, 'neighbors': 128, 'min_samples': 6}},
            source_ply='original.ply', source_ply_sha256='original cloud'))
        prepare.write(self.out / 'slopes-started.json', {'identity': key})
        prepare.write(self.out / 'prepared.json', {'identity': key, 'status': 'ok'})
        prepare.write(self.data / 'provenance.json', self.old | {'correspondence_verified': True})

    def test_only_cost_source_change_migrates_receipts_not_arrays(self):
        self.legacy_receipts()
        before = {path: path.read_bytes() for path in self.data.glob('*.npy')}
        key = prepare.migrate_geometry(self.out, self.identity)
        self.assertEqual(key, prepare.identity_hash(prepare.geometry_identity(self.identity)))
        self.assertEqual(prepare.read(self.out / 'geometry.json')['identity'], key)
        self.assertEqual(prepare.read(self.out / 'slopes-started.json')['identity'], key)
        self.assertEqual(key, prepare.migrate_geometry(self.out, self.identity))
        self.assertTrue(all(path.read_bytes() == value for path, value in before.items()))

    def test_changed_geometry_source_is_rejected_without_writes(self):
        self.legacy_receipts()
        before = (self.out / 'geometry.json').read_bytes()
        changed = self.identity | {'source_hashes': self.identity['source_hashes'] |
                                  {str(Path('path-planning/slope.py')): 'changed slope model'}}
        with self.assertRaisesRegex(ValueError, 'non-cost source'):
            prepare.migrate_geometry(self.out, changed)
        self.assertEqual((self.out / 'geometry.json').read_bytes(), before)

    def test_cost_only_pipeline_skips_geometry_and_slope_work(self):
        self.legacy_receipts()
        before = {path: path.read_bytes() for path in self.data.glob('*.npy')}
        builds = []
        def build(points, slopes, reference, config):
            builds.append(True)
            return SimpleNamespace(points=np.asarray(points, dtype=float), cost=np.array([1., 2.]), reference=reference,
                                   config=config | {'clearance_height': config['radius'], 'max_neighbors': 1},
                                   _index_overhead=lambda: None, _blocked_nodes=lambda p: p[:, 0] == 0)
        unavailable = lambda *a, **k: self.fail('Geometry/model work must not run for a cost-only revision')
        modules = dict(cost_map=SimpleNamespace(cKDTree=cKDTree, build_cost_map=build),
                       cost_evidence=SimpleNamespace(load_cost_evidence=unavailable),
                       io_utils=SimpleNamespace(load_extrinsic=unavailable),
                       view_ply=SimpleNamespace(load_ply=unavailable))
        def baseline_poses(points, costs, *args, **kwargs):
            np.testing.assert_array_equal(costs, [1., 2.])
            return [dict(reason='ok', association_mode='beneath_camera', drop=1., frame_id=0, id=0)]
        with patch.dict(sys.modules, modules), patch.object(sys, 'argv', ['prepare', '--repo', str(self.repo), '--output', str(self.out)]), \
             patch.object(path_stage, 'associate', side_effect=baseline_poses) as associate:
            prepare.main()
            prepare.main()  # A matching new cost receipt is reused.
        self.assertEqual(associate.call_count, 1)
        self.assertIsNone(associate.call_args.kwargs['frame_ids'])
        self.assertEqual(associate.call_args.args[0].dtype, np.float64)
        self.assertEqual(len(builds), 1)
        self.assertTrue(all(path.read_bytes() == value for path, value in before.items()))
        np.testing.assert_array_equal(np.load(self.data / 'cost.npy'), [np.inf, 2.])
        provenance = prepare.read(self.data / 'provenance.json')
        self.assertEqual(provenance['clearance_calibration']['point_row_ids'], [0])
        self.assertEqual(provenance['settings']['clearance_height'], 1.)
        verify_cost_source(provenance, self.source)
        with self.assertRaisesRegex(ValueError, 'recompute costs'):
            verify_cost_source(provenance | {'cost_source_sha256': 'old'}, self.source)

    def test_missing_calibration_saves_only_infinite_costs_and_empty_route(self):
        self.legacy_receipts()
        calls = []
        def build(points, slopes, reference, config):
            calls.append(True)
            return SimpleNamespace(points=np.asarray(points, dtype=float), cost=np.array([1., 2.]),
                                   reference=reference, config=config | {'clearance_height': config['radius']})
        unavailable = lambda *a, **k: self.fail('Geometry/model work must not run for a cost-only revision')
        modules = dict(cost_map=SimpleNamespace(cKDTree=cKDTree, build_cost_map=build),
                       cost_evidence=SimpleNamespace(load_cost_evidence=unavailable),
                       io_utils=SimpleNamespace(load_extrinsic=unavailable), view_ply=SimpleNamespace(load_ply=unavailable))
        np.savez(self.out / 'route.npz', ids=[1], positions=[[.1, 0, 0]], total_cost=0)
        with patch.dict(sys.modules, modules), patch.object(sys, 'argv', ['prepare', '--repo', str(self.repo), '--output', str(self.out)]), \
             patch.object(path_stage, 'associate', return_value=[]) as associate:
            prepare.main()
            prepare.main()
        self.assertEqual((len(calls), associate.call_count), (1, 1))
        self.assertTrue(np.isinf(np.load(self.data / 'cost.npy')).all())
        report, provenance = prepare.read(self.out / 'report.json'), prepare.read(self.data / 'provenance.json')
        self.assertEqual((report['status'], report['finite_cost_rows'], report['route_points']),
                         ('missing_clearance_calibration', 0, 0))
        self.assertIsNone(provenance['clearance_calibration']['height'])
        self.assertEqual(prepare.read(self.out / 'status.json')['stage'], 'complete')
        with np.load(self.out / 'route.npz') as route:
            self.assertEqual(len(route['ids']), 0)
            self.assertTrue(np.isinf(route['total_cost']))

    def test_atomic_cost_replacement_preserves_linked_original(self):
        original = self.root / 'original-cost.npy'
        np.save(original, np.array([1., 1.]))
        try:
            (self.data / 'cost.npy').symlink_to(original)
        except OSError as error:
            if sys.platform == 'win32':
                self.skipTest(f'Windows sandbox cannot create symbolic links: {error}')
            raise
        prepare.save(self.data / 'cost.npy', np.array([np.inf, 2.]))
        np.testing.assert_array_equal(np.load(original), [1., 1.])
        np.testing.assert_array_equal(np.load(self.data / 'cost.npy'), [np.inf, 2.])

    def test_reused_demo_import_reverifies_rows_without_touching_original_arrays(self):
        ply = self.root / 'original.ply'
        ply.write_bytes(b'original verified cloud')
        original = dict(recording='test', source_ply=str(ply), source_sha256=prepare.digest(ply),
                        source_slope=str(self.data / 'slope.npy'), settings={'radius': .2}, n_ref=[0., 0, 1.],
                        all_original_xyz_rgb_and_row_ids_verified=True, correspondence_verified=True,
                        input_sha256=self.entry['input_sha256'], prompt='ground', frame_count=2)
        prepare.write(self.data / 'provenance.json', original)
        frames = Path(self.gpu['lingbot'])
        frames.mkdir()
        for frame in range(2):
            np.savez(frames / f'frame_{frame:06d}.npz', extrinsic=np.column_stack([np.eye(3), [0., 0, -1.]]))
        before = {path: path.read_bytes() for path in self.data.glob('*.npy')}
        evidence_calls = []
        def evidence(points, *args, **kwargs):
            evidence_calls.append(kwargs)
            np.testing.assert_array_equal(points, self.arrays['points'])
            return {name: self.arrays[name] for name in ('support_mask', 'frame_ids')}
        prepare.import_reused_geometry(self.out, self.entry, self.gpu, self.identity,
                                       self.identity['filters'], evidence, lambda value: value)
        geometry = prepare.read(self.out / 'geometry.json')
        self.assertEqual(len(evidence_calls), 1)
        self.assertTrue(evidence_calls[0]['include_frame_ids'])
        self.assertTrue(geometry['reused_slopes'])
        self.assertEqual(geometry['reference_report']['slope_config']['radius'], .2)
        self.assertEqual(geometry['legacy_provenance'], original)
        self.assertTrue(all(path.read_bytes() == value for path, value in before.items()))
        self.assertFalse((self.out / 'slopes-started.json').exists())


if __name__ == '__main__':
    unittest.main()
