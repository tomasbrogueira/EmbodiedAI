"""Synthetic checks for automatic relative-floor selection."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from floor_reference import estimate_floor_reference


def frame(identifier=0, angle=0):
    x, y = np.meshgrid(np.arange(40) * .02 - .4, np.arange(40) * .02 - .4)
    return {"id": identifier, "points": np.stack((x, y, x * np.tan(angle)), axis=-1),
            "floor_mask": np.ones(x.shape, dtype=bool), "camera": np.array([0., 0., 1.])}


class FloorReferenceTests(unittest.TestCase):
    def test_rotated_frame_and_camera_side(self):
        rotation, _ = np.linalg.qr([[1., 2., 3.], [3., -2., 1.], [2., 1., -3.]])
        for side in (1, -1):
            frames = [frame(i) for i in range(3)]
            for item in frames:
                item["points"] = item["points"] @ rotation.T + [4, -2, 7]
                item["camera"] = side * rotation[:, 2] + [4, -2, 7]
            normal, report = estimate_floor_reference(frames)
            assert_allclose(normal, side * rotation[:, 2], atol=1e-7)
            self.assertEqual(report["reason"], "ok")

    def test_small_noise_and_five_percent_outliers(self):
        rng = np.random.default_rng(1)
        frames = [frame(i) for i in range(3)]
        for item in frames:
            points = item["points"]
            points[..., 2] += rng.normal(0, .0005, points.shape[:2])
            points[..., 2][rng.random(points.shape[:2]) < .05] += .3
        normal, report = estimate_floor_reference(frames)
        assert_allclose(normal, [0, 0, 1], atol=.001)
        self.assertEqual(report["reason"], "ok")

    def test_insufficient_frames_or_missing_floor(self):
        missing = [frame(i) for i in range(5)]
        for item in missing:
            item["floor_mask"][:] = False
        for frames in ([], [frame()], missing):
            normal, report = estimate_floor_reference(frames)
            self.assertIsNone(normal)
            self.assertEqual(report["reason"], "insufficient_reference_frames")

    def test_only_first_five_frames_are_considered(self):
        def frames():
            for i in range(5):
                yield frame(i)
            raise AssertionError("Read beyond the first five frames")
        normal, report = estimate_floor_reference(frames())
        assert_allclose(normal, [0, 0, 1], atol=1e-7)
        self.assertEqual(len(report["frames"]), 5)

    def test_inconsistent_floor_directions(self):
        frames = [frame(i, np.deg2rad(angle)) for i, angle in enumerate((0, 0, 15))]
        normal, report = estimate_floor_reference(frames)
        self.assertIsNone(normal)
        self.assertEqual(report["reason"], "inconsistent_reference_frames")
        self.assertGreater(report["max_disagreement_degrees"], 5)

    def test_two_height_levels_do_not_define_one_floor(self):
        frames = [frame(i) for i in range(3)]
        for item in frames:
            item["points"][:, 20:, 2] += .3
        normal, report = estimate_floor_reference(frames)
        self.assertIsNone(normal)
        self.assertEqual(report["reason"], "insufficient_reference_frames")


if __name__ == "__main__":
    unittest.main()
