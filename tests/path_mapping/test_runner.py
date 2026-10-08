"""CPU contract checks for saved pixel-aligned baseline stage handoffs."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from path_mapping import runner


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.geometry, self.sources, self.scores = runner.fixture()

    def rewrite_archive(self, path, **updates):
        with np.load(path, allow_pickle=False) as saved:
            content = {key: saved[key] for key in saved.files}
        content.update(updates)
        with path.open("wb") as stream:
            np.savez_compressed(stream, **content)

    def test_geometry_and_scores_round_trip_without_models(self):
        geometry_path = self.root / "geometry.npz"
        score_path = self.root / "scores.npz"
        metadata = {"fixture": True, "extrinsic_convention": "world_to_camera"}
        runner.save_geometry(geometry_path, self.geometry, self.sources, metadata)
        restored, sources, restored_metadata = runner.load_geometry(geometry_path)
        self.assertEqual(sources, self.sources)
        self.assertEqual(restored_metadata, metadata)
        runner.validate_geometry(restored)
        alignment = runner.reprojection_diagnostic(restored)
        self.assertLess(alignment["max_error_pixels"], 1e-5)
        for key in runner.GEOMETRY_KEYS:
            np.testing.assert_array_equal(restored[key], self.geometry[key])
        runner.save_scores(score_path, self.scores, restored, "Path", metadata)
        restored_scores, restored_metadata = runner.load_scores(score_path, restored, "Path")
        np.testing.assert_array_equal(restored_scores, self.scores)
        self.assertEqual(restored_metadata, metadata)

    def test_geometry_rejects_changed_pixels_hash_and_frame_count(self):
        path = self.root / "geometry.npz"
        for change in ("pixels", "hash", "source_count"):
            with self.subTest(change=change):
                runner.save_geometry(path, self.geometry, self.sources, {})
                if change == "pixels":
                    changed = self.geometry["images"].copy()
                    changed[0, 0, 0, 0] ^= 1
                    self.rewrite_archive(path, images=changed)
                elif change == "hash":
                    self.rewrite_archive(path, processed_image_sha256=np.asarray(["0" * 64] * 3))
                else:
                    self.rewrite_archive(path, source_paths=np.asarray(self.sources[:2]))
                with self.assertRaises(ValueError):
                    runner.load_geometry(path)

    def test_geometry_fingerprint_rejects_pose_tampering(self):
        path = self.root / "geometry.npz"
        runner.save_geometry(path, self.geometry, self.sources, {})
        changed = self.geometry["extrinsic"].copy()
        changed[1, 0, 3] = 0.25
        self.rewrite_archive(path, extrinsic=changed)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            runner.load_geometry(path)

    def test_scores_reject_different_geometry_order_pose_or_prompt(self):
        path = self.root / "scores.npz"
        # Distinct image content ensures a permutation is a different sequence.
        self.geometry["images"][0, 0, 0] = [1, 2, 3]
        runner.save_scores(path, self.scores, self.geometry, "Path", {})
        permutation = np.asarray([2, 0, 1])
        permuted = {key: value[permutation] for key, value in self.geometry.items()}
        with self.assertRaisesRegex(ValueError, "different geometry"):
            runner.load_scores(path, permuted, "Path")
        changed = copy.deepcopy(self.geometry)
        changed["extrinsic"][0, 0, 3] = 0.5
        with self.assertRaisesRegex(ValueError, "different geometry"):
            runner.load_scores(path, changed, "Path")
        with self.assertRaisesRegex(ValueError, "prompt"):
            runner.load_scores(path, self.geometry, "path")

    def test_saved_scores_reject_wrong_grid_and_nonfinite_values(self):
        path = self.root / "scores.npz"
        cases = [np.zeros((3, 8, 11), np.float32), np.zeros((3, 8, 12), np.uint8), self.scores.copy(), self.scores.copy()]
        cases[2][0, 0, 0] = np.nan
        cases[3][0, 0, 0] = 1.1
        for scores in cases:
            with self.subTest(shape=scores.shape, dtype=scores.dtype):
                runner.save_scores(path, scores, self.geometry, "Path", {})
                with self.assertRaises(ValueError):
                    runner.load_scores(path, self.geometry, "Path")

    def test_geometry_requires_sequence_and_rigid_positive_focal_cameras(self):
        single = {key: value[:1] for key, value in self.geometry.items()}
        with self.assertRaisesRegex(ValueError, "at least two"):
            runner.validate_geometry(single)
        self.assertEqual(runner.validate_geometry(single, allow_single_frame=True), (1, 8, 12))
        for problem in ("negative_focal", "reflection", "not_rigid", "nan_intrinsic"):
            with self.subTest(problem=problem):
                geometry = copy.deepcopy(self.geometry)
                if problem == "negative_focal":
                    geometry["intrinsic"][0, 0, 0] = -12
                elif problem == "reflection":
                    geometry["extrinsic"][0, 0, 0] = -1
                elif problem == "not_rigid":
                    geometry["extrinsic"][0, 0, 0] = 2
                else:
                    geometry["intrinsic"][0, 0, 0] = np.nan
                with self.assertRaises(ValueError):
                    runner.validate_geometry(geometry)

    def test_reprojection_uses_world_to_camera_with_nontrivial_pose(self):
        angle = np.pi / 6
        rotation = np.asarray([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]], np.float32)
        translation = np.asarray([0.3, -0.1, 0.4], np.float32)
        camera_points = self.geometry["world_points"][1].copy()
        self.geometry["world_points"][1] = (camera_points - translation) @ rotation
        self.geometry["extrinsic"][1, :, :3] = rotation
        self.geometry["extrinsic"][1, :, 3] = translation
        runner.validate_geometry(self.geometry)
        result = runner.reprojection_diagnostic(self.geometry)
        self.assertLess(result["max_error_pixels"], 1e-4)
        # Substituting the inverse pose must fail, despite still being SE(3).
        self.geometry["extrinsic"][1, :, :3] = rotation.T
        self.geometry["extrinsic"][1, :, 3] = -rotation.T @ translation
        with self.assertRaisesRegex(ValueError, "pixel alignment"):
            runner.reprojection_diagnostic(self.geometry)

    def test_reprojection_rejects_behind_camera_and_pixel_mismatch(self):
        behind = copy.deepcopy(self.geometry)
        behind["world_points"][0, 0, 0, 2] = -1
        with self.assertRaisesRegex(ValueError, "pixel alignment"):
            runner.reprojection_diagnostic(behind)
        off_grid = copy.deepcopy(self.geometry)
        off_grid["world_points"][0, 0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, "pixel alignment"):
            runner.reprojection_diagnostic(off_grid)

    def test_reprojection_rejects_invalid_pixels_outside_diagnostic_sampling(self):
        height, width, count = 20, 20, 2
        v, u = np.mgrid[:height, :width]
        camera_points = np.stack(((u - 10) * 0.02, (v - 10) * 0.02, np.full_like(u, 2)), axis=-1).astype(np.float32)
        geometry = {
            "images": np.zeros((count, height, width, 3), np.uint8),
            "world_points": np.tile(camera_points, (count, 1, 1, 1)),
            "world_points_conf": np.full((count, height, width), 2, np.float32),
            "depth": np.full((count, height, width, 1), 2, np.float32),
            "intrinsic": np.tile(np.asarray([[100, 0, 10], [0, 100, 10], [0, 0, 1]], np.float32), (count, 1, 1)),
            "extrinsic": np.tile(np.eye(4, dtype=np.float32)[:3], (count, 1, 1)),
        }
        # Flattened pixel one is absent from the old 128-point linspace sample.
        self.assertNotIn(1, np.linspace(0, height * width - 1, 128, dtype=int))
        for corruption in ("behind_camera", "off_grid"):
            with self.subTest(corruption=corruption):
                invalid = copy.deepcopy(geometry)
                if corruption == "behind_camera":
                    invalid["world_points"][0, 0, 1, 2] = -1
                else:
                    invalid["world_points"][0, 0, 1, 0] += 0.1
                with self.assertRaisesRegex(ValueError, "pixel alignment"):
                    runner.reprojection_diagnostic(invalid)

    def test_collect_frames_uses_natural_order_and_selection(self):
        for name in ("frame10.png", "frame2.png", "frame1.png", "other.txt"):
            (self.root / name).write_text("placeholder", encoding="utf-8")
        selected = runner.collect_frames(self.root, stride=2, max_frames=2)
        self.assertEqual([path.name for path in selected], ["frame1.png", "frame10.png"])

    def test_reconstruct_stage_uses_only_geometry_model(self):
        folder = self.root / "frames"
        folder.mkdir()
        for index, image in enumerate(self.geometry["images"]):
            Image.fromarray(image).save(folder / f"frame{index}.png")
        checkpoint = self.root / "lingbot.pt"
        checkpoint.write_bytes(b"mocked local checkpoint")
        output = self.root / "reconstruct"
        args = runner.parser().parse_args(["--frames", str(folder), "--stage", "reconstruct", "--lingbot-checkpoint", str(checkpoint), "--output", str(output)])
        with patch("path_mapping.models.reconstruct", return_value=(self.geometry, {"fixture": True})) as reconstruct, patch("path_mapping.models.segment_paths") as segment:
            result = runner.run(args)
        reconstruct.assert_called_once()
        segment.assert_not_called()
        self.assertTrue(result["fixture"])
        self.assertEqual(len(list((output / "processed_frames").glob("*.png"))), 3)
        self.assertFalse((output / "sam_scores.npz").exists())
        self.assertEqual(json.loads((output / "summary.json").read_text(encoding="utf-8"))["completed_stage"], "reconstruct")

    def test_fuse_handoff_runs_no_models_and_calibrates_points_and_poses(self):
        geometry_path = self.root / "geometry.npz"
        score_path = self.root / "scores.npz"
        self.geometry["extrinsic"][1, 0, 3] = 0.3
        self.geometry["world_points"][1, :, :, 0] -= 0.3
        runner.save_geometry(geometry_path, self.geometry, self.sources, {"fixture": True})
        runner.save_scores(score_path, self.scores, self.geometry, "Path", {"fixture": True, "confidence_threshold": 0.7})
        args = runner.parser().parse_args(["--geometry", str(geometry_path), "--scores", str(score_path), "--stage", "fuse", "--meters-per-unit", "2", "--output", str(self.root / "fused")])
        with patch("path_mapping.models.reconstruct") as reconstruct, patch("path_mapping.models.segment_paths") as segment, patch("path_mapping.artifacts.write_artifacts", return_value={}) as export:
            result = runner.run(args)
        reconstruct.assert_not_called()
        segment.assert_not_called()
        fusion_result = export.call_args.args[1]
        np.testing.assert_allclose(fusion_result.context_points, self.geometry["world_points"].reshape(-1, 3) * 2)
        scaled_geometry = export.call_args.kwargs["geometry"]
        np.testing.assert_allclose(scaled_geometry["extrinsic"][:, :, 3], self.geometry["extrinsic"][:, :, 3] * 2)
        self.assertEqual(result["configuration"]["coordinate_unit"], "metres")
        # A CPU replay preserves the producing SAM threshold, even though the
        # current CLI's default for a new SAM invocation is 0.5.
        self.assertEqual(result["configuration"]["sam_confidence"], 0.7)

    def test_segment_handoff_uses_exact_saved_rgb_and_skips_reconstruction(self):
        geometry_path = self.root / "geometry.npz"
        runner.save_geometry(geometry_path, self.geometry, self.sources, {"fixture": True})
        checkpoint = self.root / "sam.pt"
        checkpoint.write_bytes(b"mocked local checkpoint")
        output = self.root / "segmented"
        args = runner.parser().parse_args(["--geometry", str(geometry_path), "--stage", "segment", "--sam-checkpoint", str(checkpoint), "--output", str(output)])
        with patch("path_mapping.models.reconstruct") as reconstruct, patch("path_mapping.models.segment_paths", return_value=(self.scores, {"fixture": True})) as segment:
            runner.run(args)
        reconstruct.assert_not_called()
        segment.assert_called_once()
        np.testing.assert_array_equal(segment.call_args.args[0], self.geometry["images"])
        self.assertEqual(segment.call_args.kwargs["prompt"], "Path")
        saved_scores, metadata = runner.load_scores(output / "sam_scores.npz", self.geometry, "Path")
        np.testing.assert_array_equal(saved_scores, self.scores)
        self.assertTrue(metadata["fixture"])
        self.assertEqual(json.loads((output / "summary.json").read_text(encoding="utf-8"))["completed_stage"], "segment")

    def test_overwrite_removes_stale_generated_processed_frames(self):
        output = self.root / "overwritten"
        processed = output / "processed_frames"
        processed.mkdir(parents=True)
        Image.fromarray(self.geometry["images"][0]).save(processed / "000999.png")
        Image.fromarray(self.geometry["images"][0]).save(processed / "manual.png")
        args = runner.parser().parse_args(["--fixture", "--stage", "reconstruct", "--output", str(output), "--overwrite"])
        runner.run(args)
        self.assertFalse((processed / "000999.png").exists())
        self.assertTrue((processed / "manual.png").exists())
        generated = [path.name for path in processed.glob("*.png") if path.stem.isdigit()]
        self.assertEqual(sorted(generated), ["000000.png", "000001.png", "000002.png"])

    def test_invalid_cli_combinations_fail_before_model_calls(self):
        output = self.root / "invalid"
        cases = [
            ["--fixture", "--prompt", "path"],
            ["--fixture", "--stage", "reconstruct", "--serve"],
            ["--fixture", "--scores", "missing.npz"],
            ["--geometry", "missing.npz", "--stage", "fuse"],
            ["--fixture", "--min-observations", "0"],
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments):
                args = runner.parser().parse_args([*arguments, "--output", str(output)])
                with patch("path_mapping.models.reconstruct") as reconstruct, patch("path_mapping.models.segment_paths") as segment, self.assertRaises(ValueError):
                    runner.run(args)
                reconstruct.assert_not_called()
                segment.assert_not_called()


if __name__ == "__main__":
    unittest.main()
