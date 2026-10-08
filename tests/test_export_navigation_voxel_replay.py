"""Recorded-grid prefix fidelity and honest empty-map cube presentation."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from export_navigation_voxel_replay import (
    BACKGROUND, build_voxel_prefixes, fixed_projection, project, render_voxel_frame,
)


class VoxelReplayTests(unittest.TestCase):
    def fixture(self, *, empty=False):
        numeric = {
            'world_points': np.array([[[[-.01, 0, 0], [.01, 0, 0]]],
                                      [[[.01, 0, 0], [.05, 0, 0]]]], float),
            'depth': np.ones((2, 1, 2, 1)),
            'world_points_conf': np.zeros((2, 1, 2)) if empty else np.ones((2, 1, 2)),
            'extrinsic': np.broadcast_to(np.eye(4)[:3], (2, 3, 4)).copy(),
        }
        frames = []
        for i in range(2):
            frames.append({'floor': np.array([[i == 0, i == 1]]), 'parts': [], 'agent': None,
                           'row': {'frame_id': f'sample_{i}', 'status': 'awaiting_support',
                                   'navigation_state': {'execution_state': 'awaiting_support'},
                                   'diagnostics': {'occupied_prefix_voxels': 0 if empty else i+2,
                                                   'accepted_prefix_points': 0 if empty else (i+1)*2}}})
        return {'numeric': numeric, 'frames': frames,
                'manifest': {'settings': {'min_confidence': .5},
                             'transforms': [{'pad_ltrb': [0, 0, 0, 0]}]*2},
                'reference_plan': {'source': {'voxel_origin_native': [0, 0, 0], 'voxel_size_native': .05}},
                'replay': {'assumed_up_vector': [0, 0, 1]}}

    def test_prefix_cubes_use_exact_negative_grid_indices_and_prior_floor_evidence(self):
        data = self.fixture()
        cubes = build_voxel_prefixes(data)
        self.assertEqual([row['voxel_count'] for row in cubes['counts']], [2, 3])
        np.testing.assert_array_equal(cubes['prefixes'][1][0], [[-1, 0, 0], [0, 0, 0], [1, 0, 0]])
        self.assertEqual(cubes['prefixes'][0][1], {(-1, 0, 0)})
        self.assertEqual(cubes['prefixes'][1][1], {(-1, 0, 0), (1, 0, 0)})
        data['frames'][1]['row']['diagnostics']['occupied_prefix_voxels'] = 4
        with self.assertRaises(ValueError):
            build_voxel_prefixes(data)

    def test_empty_prefix_has_no_inferred_agent_cube_or_route(self):
        data = self.fixture(empty=True)
        cubes = build_voxel_prefixes(data)
        projection = fixed_projection(data, cubes, 512)
        self.assertTrue(np.isfinite(projection['image_scale']))
        image = np.asarray(render_voxel_frame(data, cubes, projection, 0))
        np.testing.assert_array_equal(image[:-70], np.broadcast_to(BACKGROUND, image[:-70].shape))
        self.assertIsNone(data['frames'][0]['agent'])

    def test_fixed_oblique_framing_keeps_complete_cube_corners_inside_canvas(self):
        data = self.fixture()
        cubes = build_voxel_prefixes(data)
        projection = fixed_projection(data, cubes, 512)
        indices = cubes['prefixes'][-1][0]
        corners = cubes['origin'] + (indices[:, None] + np.array([[0, 0, 0], [1, 1, 1]])[None]) * cubes['voxel_size']
        xy, _ = project(corners, projection)
        self.assertTrue(np.all(xy > 0))
        self.assertTrue(np.all(xy < 512))


if __name__ == '__main__':
    unittest.main()
