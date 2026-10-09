"""Generic helpers for reading and validating LingBot-MAP frame data."""

import re
from pathlib import Path

import numpy as np


def frame_number(path: Path) -> int:
    """Extract the trailing frame number from a filename stem."""
    match = re.search(r"(\d+)$", path.stem)
    if match is None:
        raise ValueError(f"Cannot extract frame number from {path.name}")
    return int(match.group(1))


def load_depth_map(array, name: str = "depth") -> np.ndarray:
    """Load a 2D depth or confidence array as float32."""
    result = np.asarray(array).squeeze()
    if result.ndim != 2:
        raise ValueError(f"Expected a 2D {name} map, got {result.shape}")
    return result.astype(np.float32)


def load_intrinsic(array) -> np.ndarray:
    """Load camera intrinsics as a validated 3 x 3 matrix."""
    K = np.asarray(array).squeeze()
    if K.shape == (4, 4):
        K = K[:3, :3]
    if K.shape != (3, 3):
        raise ValueError(f"Expected 3 x 3 intrinsic matrix, got {K.shape}")

    K = K.astype(np.float32)
    if not np.isfinite(K).all():
        raise ValueError("Intrinsic matrix contains non-finite values.")
    if abs(float(K[0, 0])) < 1e-8:
        raise ValueError("Invalid focal length fx.")
    if abs(float(K[1, 1])) < 1e-8:
        raise ValueError("Invalid focal length fy.")
    return K


def load_extrinsic(array) -> np.ndarray:
    """Load saved extrinsics without inverting or modifying the pose."""
    extrinsic = np.asarray(array).squeeze()
    if extrinsic.shape == (4, 4):
        extrinsic = extrinsic[:3, :4]
    if extrinsic.shape != (3, 4):
        raise ValueError(
            f"Expected 3 x 4 or 4 x 4 extrinsics, got {extrinsic.shape}"
        )

    extrinsic = extrinsic.astype(np.float32)
    if not np.isfinite(extrinsic).all():
        raise ValueError("Extrinsic matrix contains non-finite values.")
    return extrinsic


def load_rgb(image) -> np.ndarray:
    """Convert an image to H x W x 3 uint8 RGB."""
    image = np.asarray(image).squeeze()
    if image.ndim != 3:
        raise ValueError(f"Unexpected image dimensions: {image.shape}")

    # Convert channel-first RGB to channel-last RGB if necessary.
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected H x W x 3 RGB, got {image.shape}")

    if np.issubdtype(image.dtype, np.floating):
        finite = image[np.isfinite(image)]
        if finite.size > 0 and finite.max() <= 1.5:
            image = image * 255.0

    image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(image, 0, 255).astype(np.uint8)
