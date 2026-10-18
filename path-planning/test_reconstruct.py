"""CPU-only regressions for distributed reconstruction initialization."""

import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    repo = Path(os.environ.get("DATASET_REPO", Path(__file__).resolve().parents[1]))
    spec = importlib.util.spec_from_file_location("reconstruct_under_test", repo / "path-planning/reconstruct.py")
    reconstruct = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reconstruct)


@unittest.skipIf(torch is None, "Reconstruction requires torch")
class ReconstructionAnchorTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(mode="streaming", num_scale_frames=8)
        self.model = object()
        self.images = torch.arange(17, dtype=torch.float32).reshape(17, 1, 1, 1)

    def run_wrapper(self, run, images=None):
        return reconstruct.run_with_anchors(
            run, self.model, self.images if images is None else images, self.args
        )

    def test_anchors_cover_recording_before_remaining_frames(self):
        received = []

        def run(model, images, args):
            self.assertIs(model, self.model)
            self.assertIs(args, self.args)
            self.assertEqual(images.device.type, "cpu")
            received.extend(images[:, 0, 0, 0].tolist())
            return {}

        result = self.run_wrapper(run)
        self.assertEqual(received, [0, 2, 5, 7, 9, 11, 14, 16, 1, 3, 4, 6, 8, 10, 12, 13, 15])
        np.testing.assert_array_equal(result["initialization_source_frames"], [0, 2, 5, 7, 9, 11, 14, 16])
        self.assertEqual(result["reconstruction_policy"].item(), "uniform_scale_anchors_v1")

    def test_restores_geometry_poses_confidence_and_images_together(self):
        fields = (
            "pose_enc", "depth", "depth_conf", "world_points", "world_points_conf",
            "frame_type", "is_keyframe", "extrinsic", "intrinsic", "images",
        )
        expected = {
            name: np.arange(17 * 6).reshape(17, 2, 3) + index * 1000
            for index, name in enumerate(fields)
        }
        metadata = {"source": "unchanged"}

        def run(model, images, args):
            ids = images[:, 0, 0, 0].numpy().astype(int)
            return {**{name: value[ids] for name, value in expected.items()}, "metadata": metadata}

        result = self.run_wrapper(run)
        for name, value in expected.items():
            with self.subTest(field=name):
                np.testing.assert_array_equal(result[name], value)
        self.assertIs(result["metadata"], metadata)

    def test_short_recordings_use_each_frame_once(self):
        for count in (1, 3, 8):
            with self.subTest(frames=count):
                images = self.images[:count]

                def run(model, ordered, args):
                    torch.testing.assert_close(ordered, images)
                    return {"images": ordered.numpy()}

                result = self.run_wrapper(run, images)
                np.testing.assert_array_equal(result["images"], images.numpy())
                np.testing.assert_array_equal(result["initialization_source_frames"], np.arange(count))

    def test_official_batched_frame_flags_restore_original_order(self):
        def run(model, ordered, args):
            ids = ordered[:, 0, 0, 0].numpy().astype(int)
            return {'frame_type': ids.astype(np.uint8)[None],
                    'is_keyframe': (ids % 3 == 0)[None], 'images': ordered.numpy()}
        result = self.run_wrapper(run)
        np.testing.assert_array_equal(result['frame_type'], np.arange(17, dtype=np.uint8))
        np.testing.assert_array_equal(result['is_keyframe'], np.arange(17) % 3 == 0)
        np.testing.assert_array_equal(result['images'], self.images.numpy())

    def test_batched_input_reorders_frames_without_mixing_batches(self):
        images = torch.stack((self.images, self.images + 100))
        order = [0, 2, 5, 7, 9, 11, 14, 16, 1, 3, 4, 6, 8, 10, 12, 13, 15]

        def run(model, ordered, args):
            torch.testing.assert_close(ordered, images[:, order])
            # Inference returns frame-first arrays even for batched input.
            return {"depth": ordered[0, :, 0, 0, 0].numpy()}

        result = self.run_wrapper(run, images)
        np.testing.assert_array_equal(result["depth"], np.arange(17))

    def test_non_streaming_mode_is_rejected_before_inference(self):
        def run(*args):
            self.fail("Inference must not run with an unsupported mode")

        for mode in ("offline", "window", ""):
            with self.subTest(mode=mode):
                self.args.mode = mode
                with self.assertRaisesRegex(ValueError, "streaming"):
                    self.run_wrapper(run)

    def test_invalid_input_shape_is_rejected_before_inference(self):
        def run(*args):
            self.fail("Inference must not run with an invalid frame shape")

        for shape in ((17, 1, 1), (1, 17, 1, 1, 1, 1)):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "NCHW or BNCHW"):
                    self.run_wrapper(run, torch.zeros(shape))

    def test_empty_video_or_nonpositive_anchor_count_is_rejected(self):
        def run(*args):
            self.fail("Inference must not run without frames or initialization anchors")

        for shape in ((0, 1, 1, 1), (1, 0, 1, 1, 1)):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "nonempty"):
                    self.run_wrapper(run, torch.zeros(shape))
        for count in (0, -1):
            with self.subTest(anchors=count):
                self.args.num_scale_frames = count
                with self.assertRaisesRegex(ValueError, "positive"):
                    self.run_wrapper(run)

    def test_misaligned_output_is_rejected(self):
        for value in ([1] * 17, np.array(1), np.zeros((16, 2)), np.zeros((1, 17, 2))):
            with self.subTest(shape=getattr(value, "shape", "list")):
                with self.assertRaisesRegex(ValueError, "frame axis in depth"):
                    self.run_wrapper(lambda *args: {"depth": value})


if __name__ == "__main__":
    unittest.main()
