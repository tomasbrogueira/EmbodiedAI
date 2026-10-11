"""Frozen camera-height calibration and cache/failure contracts."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

import dataset_prepare as prepare
import dataset_path as path_stage


class CameraClearanceTests(unittest.TestCase):
    def poses(self, drops):
        return [dict(reason='ok', association_mode='beneath_camera', drop=drop,
                     frame_id=frame, id=100 + frame) for frame, drop in enumerate(drops)]

    def test_median_rejects_outlier_influence_and_records_original_evidence(self):
        poses = self.poses([.7, .8, .9, 1., 90.])
        result = prepare.camera_clearance(poses, .1)
        self.assertEqual((result['reason'], result['height'], result['sample_count']), ('ok', .9, 5))
        self.assertEqual(result['frame_ids'], list(range(5)))
        self.assertEqual(result['point_row_ids'], list(range(100, 105)))
        poses[0]['drop'] = 800.
        self.assertEqual(result['drops'][0], .7)  # Frozen evidence, no later pose mutation.

    def test_only_valid_positive_beneath_camera_support_contributes(self):
        poses = self.poses([.4, 0, -1, np.nan, np.inf, None, 100, 100, 100])
        poses[-3]['association_mode'] = 'visible_ground'
        poses[-2]['reason'] = 'blocked_support'
        poses[-1]['association_mode'] = None
        result = prepare.camera_clearance(poses, .5)
        self.assertEqual((result['height'], result['sample_count']), (.5, 1))
        self.assertEqual(result['point_row_ids'], [100])

    def test_missing_calibration_never_substitutes_radius(self):
        for poses in ([], self.poses([None, np.nan, -1, 0]),
                      [self.poses([2])[0] | {'association_mode': 'visible_ground'}]):
            result = prepare.camera_clearance(poses, .2)
            self.assertEqual(result['reason'], 'missing_clearance_calibration')
            self.assertEqual(result['sample_count'], 0)
            self.assertIsNone(result['height'])

    def test_relative_scale_changes_height_but_not_calibration_sources(self):
        baseline = prepare.camera_clearance(self.poses([.7, .8, .9, 100]), .2)
        for scale in (.001, 7., 1000.):
            actual = prepare.camera_clearance(self.poses(np.array([.7, .8, .9, 100]) * scale), .2 * scale)
            self.assertAlmostEqual(actual['height'] / scale, baseline['height'])
            self.assertEqual(actual['point_row_ids'], baseline['point_row_ids'])

    def test_cache_requires_adapter_policy_and_frozen_settings_height(self):
        identity = dict(geometry_identity='geometry', cost_source_sha256='cost',
                        prepare_adapter_sha256='new adapter', calibration_association_sha256='association',
                        clearance_calibration_policy=prepare.CALIBRATION_POLICY)
        calibration = prepare.camera_clearance(self.poses([.7, .8, .9]), .1)
        provenance = identity | dict(clearance_calibration=calibration, settings={'clearance_height': .8})
        receipt = dict(identity=prepare.identity_hash(identity), cost_source_sha256='cost', status='ok', clearance_height=.8)
        self.assertTrue(prepare.costs_current(receipt, provenance, identity))
        for field, value in (('prepare_adapter_sha256', 'old adapter'), ('clearance_calibration_policy', 0),
                             ('calibration_association_sha256', 'old association')):
            self.assertFalse(prepare.costs_current(receipt, provenance, identity | {field: value}))
            self.assertFalse(prepare.costs_current(receipt, provenance | {field: value}, identity))
        self.assertFalse(prepare.costs_current(receipt, provenance | {'settings': {'clearance_height': .1}}, identity))
        self.assertFalse(prepare.costs_current(receipt | {'clearance_height': .1}, provenance, identity))

    def test_missing_calibration_survives_direct_path_invocation_without_old_route(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            out = Path(directory)
            report = dict(status='missing_clearance_calibration', optimal=False, route_points=0)
            state = dict(stage='complete', status='missing_clearance_calibration')
            path_stage.atomic_json(out / 'report.json', report)
            path_stage.atomic_json(out / 'status.json', state)
            np.savez(out / 'route.npz', ids=[123], positions=[[0, 0, 0]], total_cost=0)
            self.assertEqual(path_stage.run(SimpleNamespace(output=out)), 0)
            self.assertEqual(json.loads((out / 'report.json').read_text()), report)
            self.assertEqual(json.loads((out / 'status.json').read_text()), state)
            with np.load(out / 'route.npz') as route:
                self.assertEqual(len(route['ids']), 0)
                self.assertTrue(np.isinf(route['total_cost']))


if __name__ == '__main__':
    unittest.main()
