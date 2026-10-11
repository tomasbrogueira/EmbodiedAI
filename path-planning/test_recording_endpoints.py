"""Recording endpoints retain observed support and original cloud row IDs."""

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from cost_map import build_cost_map
from recording_endpoints import find_recording_path, select_endpoint


def floor_points(angle=0.0):
    x, y = np.meshgrid(np.arange(-6, 7) * .01, np.arange(-6, 7) * .01)
    return np.column_stack((x.ravel(), y.ravel(), x.ravel() * np.tan(angle)))


def costs_for(points, slope=None, reference=(0, 0, 1), radius=.081):
    return build_cost_map(points, np.zeros(len(points)) if slope is None else slope,
                          reference, {"radius": radius,
                                      "support_mask": np.ones(len(points), bool)})


class EndpointTests(unittest.TestCase):
    def test_flat_projection_offset_and_original_row(self):
        points = floor_points()
        camera = np.array([.002, .003, .2])
        result = select_endpoint(points, costs_for(points), camera)
        expected = int(np.argmin(np.linalg.norm(points, axis=1)))
        self.assertEqual((result["reason"], result["id"]), ("ok", expected))
        assert_allclose(result["camera"], camera)
        assert_allclose(result["projected"], [.002, .003, 0], atol=1e-14)
        assert_allclose(result["offset"], [-.002, -.003, 0], atol=1e-14)
        self.assertAlmostEqual(result["distance"], np.hypot(.002, .003))

    def test_ramp_projection_and_transformed_scaled_geometry(self):
        angle = np.deg2rad(20)
        points = floor_points(angle)
        camera = np.array([.012, .003, .2])
        costs = costs_for(points, np.full(len(points), angle))
        original = select_endpoint(points, costs, camera)
        self.assertEqual(original["reason"], "ok")
        assert_allclose(original["projected"], [.012, .003, points[original["id"], 2]])
        rotation = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
        translation = np.array([.4, -.2, .3])
        for scale in (.001, 37.):
            with self.subTest(scale=scale):
                transformed = scale * (points @ rotation.T + translation)
                mapped = costs_for(transformed, np.full(len(points), angle),
                                   rotation[:, 2], radius=.081 * scale)
                result = select_endpoint(transformed, mapped,
                                         scale * (rotation @ camera + translation))
                self.assertEqual((result["reason"], result["id"]),
                                 ("ok", original["id"]))
                assert_allclose(result["projected"],
                                scale * (rotation @ original["projected"] + translation))
                assert_allclose(result["offset"], scale * rotation @ original["offset"],
                                atol=1e-12)

    def test_missing_reference_pose_and_support(self):
        points = floor_points()
        costs = costs_for(points)
        for camera in (None, [np.nan, 0, .2]):
            self.assertEqual(select_endpoint(points, costs, camera)["reason"], "missing_pose")
        result = select_endpoint(points, costs_for(points, reference=None), [0, 0, .2])
        self.assertEqual(result["reason"], "missing_reference")
        for camera in ([1., 0., .2], [0., 0., -.1]):
            result = select_endpoint(points, costs, camera)
            self.assertEqual((result["reason"], result["id"]), ("missing_support", None))
        self.assertEqual(select_endpoint(points, costs, [0, 0, .2],
                                         max_drop=.1)["reason"], "missing_support")

    def test_stacked_support_uses_closest_3d_row_and_existing_cost(self):
        floor = floor_points()
        points = np.vstack((floor, floor + [.002, .003, .03]))
        costs = costs_for(points)
        camera = np.array([0, 0, .2])
        expected = int(np.argmin(np.linalg.norm(points - camera, axis=1)))
        result = select_endpoint(points, costs, camera)
        self.assertEqual((result["reason"], result["id"]), ("ok", expected))
        self.assertGreaterEqual(expected, len(floor))
        costs.cost[expected] = np.inf
        result = select_endpoint(points, costs, camera)
        self.assertEqual((result["reason"], result["id"]), ("blocked_support", None))

    def test_single_tube_candidate_reuses_surrounding_support(self):
        angles = np.arange(8) * np.pi / 4
        points = np.vstack(([0., 0., 0.],
                            np.column_stack((.06*np.cos(angles), .06*np.sin(angles),
                                             np.zeros(8)))))
        costs = costs_for(points)
        self.assertTrue(np.isfinite(costs.cost[0]))
        self.assertEqual(select_endpoint(points, costs, [0, 0, .2])["id"], 0)

    def test_closed_lateral_limit_and_strict_below_camera(self):
        from types import SimpleNamespace
        radius = .08
        points = np.array([[radius/2, 0, 0], [radius/2+1e-6, 0, .1], [0, 0, .2]])
        costs = SimpleNamespace(points=points, cost=np.ones(3), reference=np.array([0,0,1]),
                                config={"radius":radius,"support_mask":np.ones(3,bool)})
        result = select_endpoint(points, costs, [0,0,.2])
        self.assertEqual(result["id"], 0)
        self.assertAlmostEqual(result["distance"], radius/2)
        self.assertAlmostEqual(result["drop"], .2)

    def test_nearest_unknown_support_is_not_skipped(self):
        points = floor_points()
        centre = int(np.argmin(np.linalg.norm(points, axis=1)))
        slope = np.zeros(len(points))
        slope[centre] = np.nan
        costs = costs_for(points, slope)
        self.assertTrue(np.isfinite(costs.cost).any())
        result = select_endpoint(points, costs, [0, 0, .2])
        self.assertEqual((result["reason"], result["id"]), ("blocked_support", None))

    def test_duplicates_keep_lowest_row_and_do_not_create_support(self):
        floor = floor_points()
        centre = int(np.argmin(np.linalg.norm(floor, axis=1)))
        points = np.vstack((floor, np.tile(floor[centre], (12, 1))))
        result = select_endpoint(points, costs_for(points), [0, 0, .2])
        self.assertEqual((result["reason"], result["id"]), ("ok", centre))
        points = np.tile([[0., 0., 0.]], (12, 1))
        result = select_endpoint(points, costs_for(points), [0, 0, .2])
        self.assertNotEqual(result["reason"], "ok")
        self.assertIsNone(result["id"])

    def test_reordered_cloud_is_rejected(self):
        points = floor_points()
        with self.assertRaises(ValueError):
            select_endpoint(points[::-1], costs_for(points), [0, 0, .2])


class RecordingTests(unittest.TestCase):
    def test_numeric_recording_endpoints_and_correspondence(self):
        points = floor_points()
        costs = costs_for(points)
        io = types.ModuleType("io_utils")
        io.frame_number = lambda path: int(path.stem.split("_")[-1])
        io.load_extrinsic = lambda data: data
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory, \
                patch.dict(sys.modules, {"io_utils": io}), \
                patch("recording_endpoints.load_cost_evidence",
                      return_value={"support_mask": costs.config["support_mask"]}) as evidence:
            for frame_id, camera in ((10, [.02, 0, .2]), (2, [-.02, 0, .2]),
                                     (3, [1., 0, .2])):
                np.savez(Path(directory) / f"frame_{frame_id}.npz",
                         extrinsic=np.column_stack((np.eye(3), -np.array(camera))))
            result, report = find_recording_path(points, costs, directory, directory,
                                                "video", frame_step=2, pixel_step=1)
            self.assertEqual(result.status, "ok")
            self.assertTrue(report["correspondence_verified"])
            self.assertEqual((report["start"]["frame_id"], report["goal"]["frame_id"]),
                             (2, 10))
            assert_allclose(result.positions[[0, -1]], [[-.02, 0, 0], [.02, 0, 0]])
            self.assertEqual(evidence.call_args.kwargs["frame_step"], 2)
            self.assertEqual(evidence.call_args.kwargs["pixel_step"], 1)
            # Refuse incompatible labels even when the same points were supplied.
            evidence.return_value = {"support_mask": np.zeros(len(points), bool)}
            with self.assertRaises(ValueError):
                find_recording_path(points, costs, directory, directory, "video",
                                    frame_step=2, pixel_step=1)
            evidence.return_value = {"support_mask": costs.config["support_mask"]}
            evidence.side_effect = ValueError("Cloud rows do not match saved frame")
            with self.assertRaises(ValueError):
                find_recording_path(points, costs, directory, directory, "video",
                                    frame_step=2, pixel_step=1)
            evidence.side_effect = None
            # Do not silently substitute the next available recording pose.
            for missing_frame, status, key in ((2, "invalid_start", "start"),
                                               (10, "invalid_goal", "goal")):
                with self.subTest(missing_frame=missing_frame):
                    filename = Path(directory) / f"frame_{missing_frame}.npz"
                    with np.load(filename) as data:
                        extrinsic = data["extrinsic"].copy()
                    np.savez(filename, depth=np.ones((2, 2)))
                    result, report = find_recording_path(points, costs, directory,
                                                        directory, "video",
                                                        frame_step=2, pixel_step=1)
                    self.assertEqual(result.status, status)
                    self.assertEqual(report[key]["reason"], "missing_pose")
                    assert_array_equal(result.ids, [])
                    self.assertTrue(np.isinf(result.total_cost))
                    np.savez(filename, extrinsic=extrinsic)
            for invalid in (np.column_stack((np.eye(3), [np.nan, 0, 0])),
                            np.zeros((3, 4)), np.column_stack((2 * np.eye(3), [0, 0, 0]))):
                np.savez(Path(directory) / "frame_10.npz", extrinsic=invalid)
                result, report = find_recording_path(points, costs, directory, directory,
                                                    "video", frame_step=2, pixel_step=1)
                self.assertEqual(result.status, "invalid_goal")
                self.assertEqual(report["goal"]["reason"], "invalid_pose")


if __name__ == "__main__":
    unittest.main()
