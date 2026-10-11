from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from plugins.small_obstacles.run_streaming import video_frames


class Video:
    def __init__(self, frames=12, metadata_count=None):
        self.frames = frames
        self.metadata_count = frames if metadata_count is None else metadata_count
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, key):
        import cv2
        return 30. if key == cv2.CAP_PROP_FPS else self.metadata_count

    def grab(self):
        if self.index >= self.frames:
            return False
        self.index += 1
        return True

    def read(self):
        if self.index >= self.frames:
            return False, None
        frame = np.full((28, 28, 3), self.index, np.uint8)
        self.index += 1
        return True, frame

    def release(self):
        self.released = True


class FrameCountTests(unittest.TestCase):
    def collect(self, video, count=4, **overrides):
        args = SimpleNamespace(video_path=Path('fixture.mp4'), input_fps=4., image_size=28,
            realtime=False, max_frames=0, frame_count=count, max_frame_num=1024)
        for key, value in overrides.items():
            setattr(args, key, value)
        info = {}
        with patch('cv2.VideoCapture', return_value=video), \
             patch('plugins.small_obstacles.run_streaming.time.perf_counter', side_effect=lambda: 1000.*video.index):
            frames = list(video_frames(args, info))
        return frames, info

    def test_exact_count_covers_entire_video_and_waiting_does_not_drop_frames(self):
        video = Video()
        frames, info = self.collect(video)
        self.assertEqual([frame['source_index'] for frame in frames], [0, 3, 7, 11])
        self.assertEqual([int(frame['rgb'][0, 0, 0]) for frame in frames], [0, 3, 7, 11])
        self.assertEqual(frames[-1]['source_time'], 11/30.)
        self.assertEqual(info['target_geometry_frames'], 4)
        self.assertEqual(info['dropped_sampled_frames'], 0)
        self.assertTrue(video.released)

    def test_requested_count_above_available_retains_every_unique_frame(self):
        frames, info = self.collect(Video(), count=100)
        self.assertEqual([f['source_index'] for f in frames], list(range(12)))
        self.assertEqual(info['requested_frame_count'], 100)
        self.assertEqual(info['target_geometry_frames'], 12)

    def test_original_sampling_and_prefix_limit_stay_the_same(self):
        frames, info = self.collect(Video(50), count=0, max_frames=3)
        self.assertEqual([f['source_index'] for f in frames], [0, 8, 16])
        self.assertEqual(info['input_stride'], 8)
        self.assertEqual(info['dropped_sampled_frames'], 0)

    def test_truncated_video_cannot_be_reported_as_exact_count_success(self):
        video = Video(frames=8, metadata_count=12)
        with self.assertRaisesRegex(ValueError, 'ended before selected frame'):
            self.collect(video)
        self.assertTrue(video.released)

    def test_conflicting_modes_and_model_limit_rejected_with_capture_closed(self):
        for settings in ({'realtime': True}, {'max_frames': 2}, {'max_frame_num': 3}):
            video = Video()
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.collect(video, **settings)
            self.assertTrue(video.released)


if __name__ == '__main__':
    unittest.main()
