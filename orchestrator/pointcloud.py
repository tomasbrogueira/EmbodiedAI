"""Point-cloud output helpers."""

from pathlib import Path

import numpy as np


def save_binary_ply(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
) -> None:
    """Save XYZ coordinates and RGB colors in binary PLY format."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {points.shape}")
    if colors.shape != points.shape:
        raise ValueError(
            f"Expected colors shape {points.shape}, got {colors.shape}"
        )

    vertex_dtype = np.dtype([
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ])

    vertices = np.empty(len(points), dtype=vertex_dtype)
    vertices["x"] = points[:, 0]
    vertices["y"] = points[:, 1]
    vertices["z"] = points[:, 2]
    vertices["red"] = colors[:, 0]
    vertices["green"] = colors[:, 1]
    vertices["blue"] = colors[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        file.write(header.encode("ascii"))
        file.write(vertices.tobytes())
