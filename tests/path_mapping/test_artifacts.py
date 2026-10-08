"""Artifact contract checks using tiny, deterministic CPU-only maps."""

import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from path_mapping.artifacts import PATH_COLOR, UNCERTAIN_COLOR, serve_map, write_artifacts
from path_mapping.fusion import fuse_frame_sequence


def fixture():
    points = np.tile(np.array([[[[0.1, 0.1, 1.1], [1.1, 0.1, 1.1], [2.1, 0.1, 1.1]]]]), (2, 1, 1, 1))
    rgb = np.full(points.shape, (10, 20, 30), dtype=np.uint8)
    scores = np.tile(np.array([[[1.0, 0.25, 0.0]]]), (2, 1, 1))
    result = fuse_frame_sequence(
        points, np.full(scores.shape, 2.0), np.ones(scores.shape), scores, rgb,
        voxel_size=1, min_observations=2,
    )
    extrinsic = np.tile(np.eye(4)[:3], (2, 1, 1))
    extrinsic[:, :3, 3] = (-2, -3, -4)
    intrinsic = np.tile(np.array([[100, 0, 1.5], [0, 100, 0.5], [0, 0, 1]]), (2, 1, 1))
    geometry = {"extrinsic": extrinsic, "intrinsic": intrinsic, "image_shape": (1, 3)}
    return result, rgb, scores, geometry


def read_ply(path):
    content = path.read_bytes()
    header, payload = content.split(b"end_header\n", 1)
    dtype = [("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
             ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    return header.decode("ascii"), np.frombuffer(payload, dtype=dtype)


class ArtifactTests(unittest.TestCase):
    def test_export_retains_evidence_and_labels_visualization(self):
        result, rgb, scores, geometry = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            report = write_artifacts(
                output, result, rgb, scores, summary={"prompt": "Path", "voxel_size": np.float64(1)},
                geometry=geometry, max_context_points=4,
            )
            saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["counts"], report["counts"])
            self.assertEqual(report["counts"]["accepted_path_voxels"], 1)
            self.assertEqual(report["counts"]["uncertain_path_voxels"], 1)
            self.assertEqual(report["counts"]["valid_context_points"], 6)
            self.assertEqual(report["counts"]["exported_context_points"], 4)
            self.assertIn("metric scale unverified", saved["coordinate_units"])
            with np.load(output / "path_points.npz", allow_pickle=False) as points:
                np.testing.assert_array_equal(points["points"], result.path_points)
                np.testing.assert_array_equal(points["scores"], result.path_scores)
                np.testing.assert_array_equal(points["frame_indices"], (0, 0, 1, 1))
            with np.load(output / "path_voxels.npz", allow_pickle=False) as voxels:
                np.testing.assert_array_equal(voxels["probabilities"], (1, 0.25, 0))
                np.testing.assert_array_equal(voxels["observations"], (2, 2, 2))
                np.testing.assert_array_equal(voxels["path_flags"], (True, False, False))
            header, vertices = read_ply(output / "path_map.ply")
            self.assertIn("element vertex 2", header)
            self.assertEqual(len(vertices), 2)
            np.testing.assert_array_equal(vertices["x"], (0.5, 1.5))
            self.assertEqual(tuple(vertices[0][name] for name in ("red", "green", "blue")), PATH_COLOR)
            self.assertEqual(tuple(vertices[1][name] for name in ("red", "green", "blue")), UNCERTAIN_COLOR)
            self.assertEqual(len(read_ply(output / "context_cloud.ply")[1]), 4)
            with Image.open(output / "overlays" / "000000.png") as image:
                overlay = np.array(image)
            np.testing.assert_array_equal(overlay[0, 2], rgb[0, 0, 2])
            self.assertGreater(overlay[0, 0, 1], rgb[0, 0, 0, 1])
            self.assertGreater(overlay[0, 0, 1], overlay[0, 1, 1])
            with np.load(output / "camera_trajectory.npz", allow_pickle=False) as trajectory:
                np.testing.assert_array_equal(trajectory["camera_to_world"][:, :3, 3], ((2, 3, 4), (2, 3, 4)))
                np.testing.assert_allclose(trajectory["camera_to_world"] @ trajectory["world_to_camera"], np.tile(np.eye(4), (2, 1, 1)))

    def test_empty_map_is_still_reviewable(self):
        _, rgb, scores, _ = fixture()
        points = np.zeros(rgb.shape)
        result = fuse_frame_sequence(points, np.zeros(scores.shape), np.zeros(scores.shape), scores, rgb)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            report = write_artifacts(output, result, rgb, scores, summary={})
            self.assertEqual(report["counts"]["accepted_path_voxels"], 0)
            self.assertEqual(len(read_ply(output / "path_map.ply")[1]), 0)
            self.assertEqual(len(read_ply(output / "context_cloud.ply")[1]), 0)
            with np.load(output / "path_points.npz", allow_pickle=False) as points:
                self.assertEqual(points["points"].shape, (0, 3))

    def test_rerun_removes_stale_generated_overlays_and_trajectory(self):
        result, rgb, scores, geometry = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            write_artifacts(output, result, rgb, scores, summary={}, geometry=geometry)
            stale = output / "overlays" / "000008.png"
            stale.write_bytes(b"obsolete")
            unrelated = output / "overlays" / "reference.png"
            unrelated.write_bytes(b"reference")
            write_artifacts(output, result, rgb, scores, summary={})
            self.assertFalse(stale.exists())
            self.assertFalse((output / "camera_trajectory.npz").exists())
            self.assertEqual(unrelated.read_bytes(), b"reference")

    def test_calibrated_units_and_replay_artifacts_are_recorded(self):
        result, rgb, scores, _ = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "geometry.npz").touch()
            (output / "sam_scores.npz").touch()
            (output / "processed_frames").mkdir()
            report = write_artifacts(
                output, result, rgb, scores,
                summary={"configuration": {"coordinate_unit": "metres", "meters_per_reconstruction_unit": 2}},
            )
            self.assertEqual(report["coordinate_units"], "metres (user-supplied reconstruction scale)")
            for name in ("geometry.npz", "sam_scores.npz", "processed_frames/"):
                self.assertIn(name, report["artifacts"])
            header, _ = read_ply(output / "path_map.ply")
            self.assertIn("coordinate units are recorded in summary.json", header)

    def test_invalid_alignment_and_pose_fail_before_outputs(self):
        result, rgb, scores, geometry = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "map"
            with self.assertRaises(ValueError):
                write_artifacts(output, result, rgb, scores[:, :, :2], summary={})
            self.assertFalse(output.exists())
            geometry["extrinsic"][:, :3, :3] = 0
            with self.assertRaisesRegex(ValueError, "singular"):
                write_artifacts(output, result, rgb, scores, summary={}, geometry=geometry)
            self.assertFalse(output.exists())

    def test_viewer_uses_inverted_poses_and_stops_server(self):
        result, _, _, geometry = fixture()
        server = Mock()
        server.scene.add_point_cloud.side_effect = [SimpleNamespace(visible=True), SimpleNamespace(visible=False), SimpleNamespace(visible=True)]
        server.gui.add_checkbox.side_effect = lambda *args, **kwargs: SimpleNamespace(on_update=lambda callback: callback)
        viser_module = ModuleType("viser")
        viser_module.ViserServer = Mock(return_value=server)
        transforms_module = ModuleType("viser.transforms")
        transforms_module.SO3 = SimpleNamespace(from_matrix=lambda matrix: SimpleNamespace(wxyz=np.array((1, 0, 0, 0))))
        with patch.dict(sys.modules, {"viser": viser_module, "viser.transforms": transforms_module}):
            with patch("path_mapping.artifacts.time.sleep", side_effect=KeyboardInterrupt):
                serve_map(result, geometry)
        viser_module.ViserServer.assert_called_once_with(host="127.0.0.1", port=8080)
        np.testing.assert_array_equal(server.scene.add_camera_frustum.call_args_list[0].kwargs["position"], (2, 3, 4))
        server.stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
