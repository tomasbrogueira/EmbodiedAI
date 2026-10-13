import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from plugins.small_obstacles.common import parse_proposal, rgb_digest
from plugins.small_obstacles.project import project
from plugins.small_obstacles.propose import propose
from plugins.small_obstacles.segment import segment, union_masks


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.lingbot = self.root / "lingbot"
        self.lingbot.mkdir()
        self.masks = self.root / "masks"
        self.prompts = self.root / "prompts.json"
        self.rgb = np.full((4, 4, 3), 30, dtype=np.uint8)
        for idx in (0, 1):
            np.savez(
                self.lingbot / f"frame_{idx:06d}.npz",
                images=self.rgb, depth=np.ones((4, 4, 1)),
                depth_conf=np.full((4, 4), 3.0),
                intrinsic=np.eye(3), extrinsic=np.c_[np.eye(3), np.zeros(3)],
            )

    def segment_args(self):
        return SimpleNamespace(
            lingbot_dir=self.lingbot, prompts=self.prompts,
            output_dir=self.masks, frame_step=1, max_frames=None,
            device="cpu", sam_threshold=0.5, checkpoint=None,
        )

    def projection_args(self):
        return SimpleNamespace(
            lingbot_dir=self.lingbot, masks_dir=self.masks,
            output=self.root / "map.ply", pixel_step=1,
            min_depth_conf=2.5, max_depth=25.0, color=[255, 0, 0],
        )

    def test_json_validation_and_deduplication(self):
        self.assertEqual(parse_proposal('```json\n{"prompts": [" Shoe ", "shoe", "cable"]}\n```'), ["shoe", "cable"])
        for bad in ('{"prompts": "shoe"}', '{"prompts": [null]}', '{"prompts": [""]}', 'shoe'):
            with self.assertRaises(ValueError):
                parse_proposal(bad)

    def test_discover_concepts_on_later_frames(self):
        class Proposer:
            def __init__(self):
                self.responses = iter(['{"prompts": []}', '{"prompts": ["shoe", "cable"]}'])

            def propose(self, image, instruction):
                return next(self.responses)

        args = SimpleNamespace(
            lingbot_dir=self.lingbot, keyframe_step=1, max_frames=None,
            model="test", device="cpu", max_new_tokens=128,
            robot="wheeled robot", ground_crop=False, output=self.prompts,
        )
        propose(args, Proposer())
        data = json.loads(self.prompts.read_text())
        self.assertEqual(data["prompts"], ["shoe", "cable"])
        self.assertEqual(len(data["observations"]), 2)

    def test_multiple_prompts_union_and_real_world_projection(self):
        class Segmenter:
            def __init__(self):
                self.image_calls = 0
                self.prompt_calls = []

            def set_image(self, image):
                self.image_calls += 1

            def masks(self, prompt):
                self.prompt_calls.append(prompt)
                masks = np.zeros((2, 1, 4, 4), dtype=bool)
                masks[0, 0, 0, 0] = True
                masks[1, 0, 1 if prompt == "shoe" else 2, 1] = True
                return masks

        self.prompts.write_text('["shoe", "cable"]')
        predictor = Segmenter()
        segment(self.segment_args(), predictor)
        self.assertEqual(predictor.image_calls, 2)
        self.assertEqual(predictor.prompt_calls, ["shoe", "cable"] * 2)
        with np.load(self.masks / "frame_000000.npz") as data:
            self.assertEqual(int(data["out_binary_masks"].sum()), 3)
            self.assertEqual(data["rgb_sha256"].item(), rgb_digest(self.rgb))
        # Use the actual official LingBot unprojection and PLY writer.
        args = self.projection_args()
        project(args)
        raw = args.output.read_bytes()
        header, payload = raw.split(b"end_header\n", 1)
        self.assertIn(b"element vertex 32", header)
        dtype = np.dtype([(axis, "<f4") for axis in "xyz"] + [(c, "u1") for c in ("r", "g", "b")])
        vertices = np.frombuffer(payload, dtype=dtype)
        self.assertEqual(int((vertices["r"] == 255).sum()), 6)
        np.testing.assert_allclose([vertices["x"][5], vertices["y"][5], vertices["z"][5]], [1, 1, 1])
        self.assertEqual(vertices["r"][3], 30)

    def test_empty_proposals_do_not_load_sam(self):
        self.prompts.write_text('{"prompts": []}')
        with patch("plugins.small_obstacles.segment.SamSegmenter") as builder:
            segment(self.segment_args())
            builder.assert_not_called()
        with np.load(self.masks / "frame_000000.npz") as data:
            self.assertFalse(data["out_binary_masks"].any())

    def test_empty_instances_and_alignment(self):
        self.assertFalse(union_masks(np.zeros((0, 1, 4, 4)), (4, 4)).any())
        with self.assertRaises(ValueError):
            union_masks(np.ones((1, 1, 3, 4)), (4, 4))

    def test_projection_rejects_changed_source(self):
        self.prompts.write_text('[]')
        segment(self.segment_args())
        path = self.lingbot / "frame_000000.npz"
        with np.load(path) as data:
            arrays = dict(data)
        arrays["images"] = self.rgb + 1
        np.savez(path, **arrays)
        with self.assertRaisesRegex(ValueError, "different RGB"):
            project(self.projection_args())

    def test_depth_filters_remove_masked_points(self):
        self.prompts.write_text('[]')
        segment(self.segment_args())
        for path in self.lingbot.glob("frame_*.npz"):
            with np.load(path) as data:
                arrays = dict(data)
            arrays["depth_conf"][0, 0] = 1.0
            arrays["depth"][1, 0] = 30.0
            arrays["depth"][2, 0] = 0.0
            np.savez(path, **arrays)
        project(self.projection_args())
        self.assertIn(b"element vertex 26", self.projection_args().output.read_bytes())


if __name__ == "__main__":
    unittest.main()
