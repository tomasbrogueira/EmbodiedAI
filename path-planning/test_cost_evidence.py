import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from cost_evidence import load_cost_evidence


def frame(depth=5.0):
    return {"depth": np.full((21, 21), depth),
            "intrinsic": np.array([[6., 0., 10.], [0., 6., 10.], [0., 0., 1.]]),
            "extrinsic": np.column_stack((np.eye(3), np.zeros(3)))}


class RowEvidenceTests(unittest.TestCase):
    def test_original_sampling_filtering_order_and_missing_masks(self):
        def world(depth, intrinsic, extrinsic):
            y, x = np.indices(depth.shape)
            return np.stack((x, y, depth), axis=-1)

        io = types.ModuleType("io_utils")
        io.frame_number = lambda path: int(path.stem.split("_")[-1])
        io.load_depth_map = lambda data, name: data
        io.load_intrinsic = io.load_extrinsic = lambda data: data
        geometry = types.ModuleType("lingbot_geometry")
        geometry.depth_to_world_points = world
        masks = types.ModuleType("sam_masks")
        masks.build_sam_index = lambda *args: {10: "mask10"}
        mask = np.ones((3, 3), dtype=bool)
        mask[2, 2] = False
        masks.load_sam_mask = lambda path: mask
        masks.preprocess_mask = lambda data, **kwargs: data
        modules = {"io_utils": io, "lingbot_geometry": geometry, "sam_masks": masks}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory, \
                patch.dict(sys.modules, modules):
            chunks = []
            for frame_id in [10, 2, 1]:
                depth = np.full((3, 3), 1.123456789 + frame_id)
                depth[2, 0] = 25.0
                confidence = np.full((3, 3), 3.)
                confidence[0, 2] = 2.5
                evidence = frame()
                np.savez(Path(directory) / f"frame_{frame_id}.npz", depth=depth,
                         depth_conf=confidence, intrinsic=evidence["intrinsic"],
                         extrinsic=evidence["extrinsic"])
                if frame_id % 2 == 0:
                    chunks.append(world(depth, None, None)[[0, 2], [0, 2]].astype(np.float32))
            points = np.concatenate(chunks).astype(float)
            config = dict(frame_step=2, pixel_step=2)
            result = load_cost_evidence(points, directory, directory, "video", **config)
            np.testing.assert_array_equal(result["support_mask"], [True, False, False, False])
            self.assertEqual(set(result), {"support_mask"})
            for changed in [points[::-1], points[:-1], np.concatenate([points, points[:1]])]:
                with self.assertRaises(ValueError):
                    load_cost_evidence(changed, directory, directory, "video", **config)
            masks.build_sam_index = lambda *args: {}
            result = load_cost_evidence(points, directory, directory, "video", **config)
            self.assertFalse(result["support_mask"].any())


if __name__ == "__main__":
    unittest.main()
