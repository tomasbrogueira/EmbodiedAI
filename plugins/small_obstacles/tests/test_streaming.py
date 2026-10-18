from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from plugins.small_obstacles.run_streaming import project_frame
from plugins.small_obstacles.stream_state import DisplayBuffer, PromptState, StreamingPly


class StreamingTests(unittest.TestCase):
    def test_prompt_age_uses_observation_time_and_empty_results_do_not_erase_recent_hazard(self):
        state = PromptState(ttl=2., max_prompts=2)
        state.observe(['log', 'bag'], 1.)
        state.observe([], 2.)
        self.assertEqual(state.active(2.), ['log', 'bag'])
        state.observe(['rock'], 2.5)
        self.assertEqual(state.active(2.5), ['rock', 'log'])
        self.assertEqual(state.active(3.1), ['rock'])
        self.assertEqual(state.active(4.6), [])

    def test_display_bounded_and_keeps_point_color_pairs(self):
        view = DisplayBuffer(5)
        for n in (3, 9, 0, 4):
            points = np.arange(n * 3, dtype=np.float32).reshape(n, 3)
            view.append(points, points.astype(np.uint8))
            p, c = view.arrays()
            self.assertLessEqual(len(p), 5)
            np.testing.assert_array_equal(p.astype(np.uint8), c)

    def test_display_retains_earlier_scene_observations(self):
        view = DisplayBuffer(100)
        view.append(np.zeros((1000, 3), np.float32), np.zeros((1000, 3), np.uint8))
        view.append(np.ones((1000, 3), np.float32), np.ones((1000, 3), np.uint8))
        points, _ = view.arrays()
        self.assertTrue((points[:, 0] == 0).any())
        self.assertTrue((points[:, 0] == 1).any())
        self.assertEqual(view.count, 2000)

    def test_equivalent_log_names_use_one_sam_query(self):
        state = PromptState()
        state.observe(['fallen log', 'detached log', 'log'], 0.)
        self.assertEqual(state.active(0.), ['log'])

    def test_incremental_projection_and_ply_match_unchanged_mapper(self):
        root = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(root / 'orchestrator'))
        from semantic_map import build_semantic_point_cloud
        from view_ply import read_ply_header
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            predictions, masks = directory / 'predictions', directory / 'masks'
            predictions.mkdir()
            masks.mkdir()
            arrays = []
            writer = StreamingPly(directory / 'stream.ply')
            for index in (0, 2):
                depth = np.ones((28, 28), dtype=np.float32)
                depth[0, 0] = 26  # Same original depth filter.
                conf = np.full_like(depth, 3)
                conf[0, 1] = 2.5  # Threshold is strictly greater, not >=.
                mask = np.zeros_like(depth, dtype=bool)
                mask[3, 4] = True
                prediction = dict(depth=depth, depth_conf=conf,
                    images=np.full((28, 28, 3), 30, np.uint8), intrinsic=np.eye(3),
                    extrinsic=np.c_[np.eye(3), [index, 0, 0]])
                np.savez(predictions / f'frame_{index:06d}.npz', **prediction)
                np.savez(masks / f'video_obstacles_frame{index:06d}.npz', out_binary_masks=mask)
                points, colors, marked = project_frame(prediction, mask)
                self.assertEqual(marked, 1)
                writer.append(points, colors)
                count, dtype, offset = read_ply_header(writer.path)
                self.assertEqual(count, writer.count)
                self.assertEqual(writer.path.stat().st_size - offset, count * dtype.itemsize)
            writer.close()
            config = SimpleNamespace(frame_step=2, pixel_step=1, lingbot_dir=predictions,
                sam_dir=masks, video_name='video', prompt='obstacles', video_path=directory / 'video.mp4',
                min_depth_conf=2.5, max_depth=25, segment_color=np.array([255, 0, 0], dtype=np.uint8),
                patch_size=14, out_file=directory / 'original.ply')
            build_semantic_point_cloud(config)
            for path in (directory / 'stream.ply', directory / 'original.ply'):
                count, dtype, offset = read_ply_header(path)
                arrays.append(np.frombuffer(path.read_bytes()[offset:], dtype=dtype))
            np.testing.assert_array_equal(*arrays)

    def test_current_frame_alignment_is_required(self):
        prediction = dict(depth=np.ones((28, 28)), depth_conf=np.ones((28, 28)),
            images=np.zeros((28, 28, 3)), intrinsic=np.eye(3), extrinsic=np.c_[np.eye(3), np.zeros(3)])
        with self.assertRaisesRegex(ValueError, 'matching dimensions'):
            project_frame(prediction, np.zeros((14, 14), dtype=bool))


if __name__ == '__main__':
    unittest.main()
