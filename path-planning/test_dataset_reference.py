"""Run with DATASET_REPO set to the production repository."""

import os
import sys
import unittest
from pathlib import Path

import numpy as np

if os.environ.get("DATASET_REPO"):
    sys.path.insert(0, str(Path(os.environ["DATASET_REPO"]) / "path-planning"))

from dataset_reference import scan_reference
from floor_reference import estimate_floor_reference


def frame(index, supported=True):
    x, z = np.meshgrid(np.linspace(-.2, .2, 25), np.linspace(-.2, .2, 25))
    return dict(id=f"frame_{index:06}.npz", points=np.stack((x, np.zeros_like(x), z), axis=-1),
                floor_mask=np.full(x.shape, supported), camera=np.array([0., 1., 0.]))


class ReferenceScanTests(unittest.TestCase):
    def test_first_five_are_unchanged_and_stop_reading(self):
        frames = [frame(i) for i in range(5)]
        expected, evidence = estimate_floor_reference(frames)

        def stream():
            yield from frames
            raise AssertionError("Read beyond a successful opening window")

        normal, report = scan_reference(stream())
        np.testing.assert_array_equal(normal, expected)
        for key, value in evidence.items():
            self.assertEqual(report[key], value)
        self.assertEqual(report["selected_window"], [f["id"] for f in frames])
        self.assertEqual(report["inspected_frame_count"], 5)
        self.assertEqual(report["windows_checked"], 1)
        self.assertEqual(report["slope_config"]["radius"], 20 * np.median([f["spacing"] for f in evidence["frames"]]))

    def test_late_first_stable_window(self):
        normal, report = scan_reference(frame(i, supported=i >= 5) for i in range(12))
        self.assertIsNotNone(normal)
        self.assertEqual(report["selected_window"], [f"frame_{i:06}.npz" for i in range(3, 8)])
        self.assertEqual(report["inspected_frame_count"], 8)
        self.assertEqual(report["windows_checked"], 4)
        self.assertEqual(sum(f["reason"] == "accepted" for f in report["frames"]), 3)

    def test_no_stable_window_is_explicit(self):
        normal, report = scan_reference(frame(i, False) for i in range(9))
        self.assertIsNone(normal)
        self.assertIsNone(report["selected_window"])
        self.assertEqual(report["reason"], "insufficient_reference_frames")
        self.assertEqual(report["inspected_frame_count"], 9)
        self.assertEqual(report["windows_checked"], 5)
        self.assertNotIn("slope_config", report)

    def test_short_recording_preserves_existing_behavior(self):
        frames = [frame(i) for i in range(3)]
        expected, _ = estimate_floor_reference(frames)
        normal, report = scan_reference(iter(frames))
        np.testing.assert_array_equal(normal, expected)
        self.assertEqual(report["inspected_frame_count"], 3)


if __name__ == "__main__":
    unittest.main()
