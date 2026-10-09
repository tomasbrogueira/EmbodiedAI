"""LingBot-MAP depth-to-world geometry utilities."""

import sys
from pathlib import Path

import numpy as np


# Resolve the project root independently of config.py.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LINGBOT_ROOT = PROJECT_ROOT / "lingbot-map"

if not LINGBOT_ROOT.is_dir():
    raise FileNotFoundError(
        f"LingBot-MAP source directory not found: {LINGBOT_ROOT}"
    )

if str(LINGBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(LINGBOT_ROOT))

from lingbot_map.utils.geometry import (
    unproject_depth_map_to_point_map,
)


def depth_to_world_points(depth, intrinsic, extrinsic):
    """Project a depth map into 3D using official LingBot-MAP geometry.

    The saved extrinsic is passed directly to the official helper.
    No additional pose transformations are applied.
    """

    depth_batch = depth[
        np.newaxis, ..., np.newaxis
    ].astype(np.float32)

    extrinsic_batch = extrinsic[
        np.newaxis, ...
    ].astype(np.float32)

    intrinsic_batch = intrinsic[
        np.newaxis, ...
    ].astype(np.float32)

    world_points_batch = unproject_depth_map_to_point_map(
        depth_batch,
        extrinsic_batch,
        intrinsic_batch,
    )

    result = world_points_batch[0]

    # Support NumPy arrays and PyTorch tensors.
    if hasattr(result, "detach"):
        result = result.detach().cpu().numpy()

    world_points = np.asarray(result, dtype=np.float32)
    expected_shape = (*depth.shape, 3)

    if world_points.shape != expected_shape:
        raise ValueError(
            f"Official unprojection returned {world_points.shape}; "
            f"expected {expected_shape}."
        )

    return world_points