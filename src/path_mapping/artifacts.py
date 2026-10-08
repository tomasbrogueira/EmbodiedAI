"""CPU-only exports and an optional local viewer for the path baseline.

Coordinates retain the caller's units; an explicit reconstruction scale may
convert them to metres. Green path labels do not imply robot traversability.
"""

from __future__ import annotations

import json
import math
import numbers
from pathlib import Path
import re
import time
from typing import TYPE_CHECKING, Mapping

import numpy as np

if TYPE_CHECKING:
    from .fusion import FusionResult


PATH_COLOR = (50, 220, 90)
UNCERTAIN_COLOR = (255, 210, 35)
CONTEXT_COLOR = (145, 145, 145)


def _sample_indices(count: int, maximum: int | None) -> np.ndarray:
    """Deterministically sample the full sequence, including its endpoints."""
    if maximum is not None:
        if isinstance(maximum, (bool, np.bool_)) or not isinstance(maximum, numbers.Integral) or maximum < 0:
            raise ValueError("max_context_points must be a nonnegative integer or None")
    if maximum is None or count <= maximum:
        return np.arange(count, dtype=np.int64)
    if maximum == 0:
        return np.empty(0, dtype=np.int64)
    return np.linspace(0, count - 1, maximum, dtype=np.int64)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON")


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Write a compact binary PLY without losing the input coordinate precision."""
    points = np.asarray(points)
    colors = np.asarray(colors)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("PLY points must have finite shape [N,3]")
    if colors.shape != points.shape or colors.dtype != np.uint8:
        raise ValueError("PLY colors must be uint8 with shape [N,3]")
    records = np.empty(
        len(points),
        dtype=[("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
               ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    for index, name in enumerate(("x", "y", "z")):
        records[name] = points[:, index]
    for index, name in enumerate(("red", "green", "blue")):
        records[name] = colors[:, index]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "comment coordinate units are recorded in summary.json\n"
        f"element vertex {len(points)}\n"
        "property double x\nproperty double y\nproperty double z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with path.open("wb") as file:
        file.write(header.encode("ascii"))
        records.tofile(file)


def _camera_trajectory(geometry: Mapping, frame_count: int | None = None) -> dict:
    """Normalize decoded LingBot world-to-camera poses and invert them once."""
    extrinsic = np.asarray(geometry["extrinsic"], dtype=np.float64)
    if extrinsic.ndim != 3 or extrinsic.shape[1:] not in ((3, 4), (4, 4)):
        raise ValueError("extrinsic must have shape [S,3,4] or [S,4,4]")
    if not np.isfinite(extrinsic).all():
        raise ValueError("extrinsic must contain finite world-to-camera poses")
    if frame_count is not None and len(extrinsic) != frame_count:
        raise ValueError("Camera trajectory must have one pose per processed frame")
    world_to_camera = np.broadcast_to(np.eye(4), (len(extrinsic), 4, 4)).copy()
    world_to_camera[:, :3, :] = extrinsic[:, :3, :]
    if extrinsic.shape[1] == 4 and not np.allclose(extrinsic[:, 3, :], (0, 0, 0, 1)):
        raise ValueError("Homogeneous extrinsic poses must end with [0,0,0,1]")
    try:
        camera_to_world = np.linalg.inv(world_to_camera)
    except np.linalg.LinAlgError as error:
        raise ValueError("Camera trajectory contains a singular world-to-camera pose") from error
    intrinsic = np.asarray(geometry["intrinsic"], dtype=np.float64)
    if intrinsic.shape != (len(extrinsic), 3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError("intrinsic must have finite shape [S,3,3]")
    if (intrinsic[:, (0, 1), (0, 1)] <= 0).any():
        raise ValueError("intrinsic focal lengths must be positive")
    return {"world_to_camera": world_to_camera, "camera_to_world": camera_to_world, "intrinsic": intrinsic}


def write_artifacts(
    output: Path,
    result: FusionResult,
    images: np.ndarray,
    path_scores: np.ndarray,
    *,
    summary: dict,
    geometry: Mapping | None = None,
    max_context_points: int | None = 100000,
) -> dict:
    """Write aligned overlays, complete path evidence, voxels, clouds and summary.

    ``images`` is uint8 ``[S,H,W,3]``; ``path_scores`` is the matching probability
    grid. NPZ files retain all evidence. Only the gray context PLY is sampled to
    ``max_context_points``; set that to ``None`` for the full context cloud.
    Existing generated files are replaced when rerunning into the same directory.

    ``path_points.npz`` contains points/colors/scores/confidence/frame_indices.
    ``path_voxels.npz`` contains indices/centers/probabilities/observations/
    path_flags/path_weights/total_weights/point_counts/colors. Zero-probability
    voxels remain in the NPZ; the PLY displays accepted voxels in green and other
    voxels with positive path evidence in yellow.
    """
    from PIL import Image

    images = np.asarray(images)
    scores = np.asarray(path_scores)
    if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8 or min(images.shape[:3]) < 1:
        raise ValueError("images must be nonempty uint8 [S,H,W,3]")
    if scores.shape != images.shape[:3] or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("path_scores must match the image grid with finite values in [0,1]")
    context_indices = _sample_indices(len(result.context_points), max_context_points)
    trajectory = _camera_trajectory(geometry, len(images)) if geometry is not None else None
    # Validate metadata before creating files. summary.json is the completion record.
    report = dict(summary)
    json.dumps(report, default=_json_default, allow_nan=False)
    output = Path(output)
    overlays = output / "overlays"
    overlays.mkdir(parents=True, exist_ok=True)
    color = np.asarray(PATH_COLOR, dtype=np.float32)
    for index, (rgb, score) in enumerate(zip(images, scores)):
        alpha = np.asarray(score, dtype=np.float32)[..., None] * 0.55
        overlay = np.rint(rgb.astype(np.float32) * (1.0 - alpha) + color * alpha).astype(np.uint8)
        Image.fromarray(overlay).save(overlays / f"{index:06d}.png")
    for existing in overlays.glob("*.png"):
        if re.fullmatch(r"\d{6}\.png", existing.name) and int(existing.stem) >= len(images):
            existing.unlink()

    np.savez_compressed(
        output / "path_points.npz",
        points=result.path_points,
        colors=result.path_colors,
        scores=result.path_scores,
        confidence=result.path_confidence,
        frame_indices=result.path_frame_indices,
    )
    np.savez_compressed(
        output / "path_voxels.npz",
        indices=result.voxel_indices,
        centers=result.voxel_centers,
        probabilities=result.path_probabilities,
        observations=result.observations,
        path_flags=result.path_flags,
        path_weights=result.path_weights,
        total_weights=result.total_weights,
        point_counts=result.point_counts,
        colors=result.voxel_colors,
    )
    displayed = result.path_flags | (result.path_probabilities > 0)
    voxel_colors = np.full((int(displayed.sum()), 3), UNCERTAIN_COLOR, dtype=np.uint8)
    voxel_colors[result.path_flags[displayed]] = PATH_COLOR
    _write_ply(output / "path_map.ply", result.voxel_centers[displayed], voxel_colors)
    gray = np.full((len(context_indices), 3), CONTEXT_COLOR, dtype=np.uint8)
    _write_ply(output / "context_cloud.ply", result.context_points[context_indices], gray)
    artifact_names = ["overlays/", "path_points.npz", "path_voxels.npz", "path_map.ply", "context_cloud.ply", "summary.json"]
    for name in ("geometry.npz", "sam_scores.npz", "processed_frames/"):
        if (output / name).exists():
            artifact_names.append(name)
    if trajectory is not None:
        np.savez_compressed(output / "camera_trajectory.npz", **trajectory)
        artifact_names.append("camera_trajectory.npz")
    elif (output / "camera_trajectory.npz").exists():
        # A prior export's poses must not appear to belong to the current map.
        (output / "camera_trajectory.npz").unlink()
    counts = dict(report.get("counts", {}))
    counts.update(
        frames=len(images),
        processed_height=images.shape[1],
        processed_width=images.shape[2],
        positive_pixels=int(np.count_nonzero(scores)),
        valid_context_points=len(result.context_points),
        exported_context_points=len(context_indices),
        positive_3d_points=len(result.path_points),
        voxels=len(result.voxel_centers),
        accepted_path_voxels=int(result.path_flags.sum()),
        uncertain_path_voxels=int((displayed & ~result.path_flags).sum()),
    )
    report.update(
        counts=counts,
        artifacts=artifact_names,
        context_sampling={"max_points": max_context_points, "method": "uniform sequence indices"},
        visualization_colors={"path": PATH_COLOR, "uncertain": UNCERTAIN_COLOR, "context": CONTEXT_COLOR},
    )
    configuration = report.get("configuration", {})
    units = configuration.get("coordinate_unit", "reconstruction_units")
    unit_description = "metres (user-supplied reconstruction scale)" if units == "metres" else "native reconstruction units; metric scale unverified"
    report.setdefault("coordinate_units", unit_description)
    report.setdefault("label_meaning", "Path prompt evidence; robot traversability unverified")
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, default=_json_default, allow_nan=False) + "\n", encoding="utf-8",
    )
    return report


def serve_map(
    result: FusionResult,
    geometry: Mapping,
    *,
    port: int = 8080,
    host: str = "127.0.0.1",
    point_size: float = 0.01,
    max_context_points: int | None = 100000,
) -> None:
    """Serve separate context, positive-point, voxel and camera layers until Ctrl-C.

    Viser is imported only when requested. ``geometry`` contains decoded W2C
    ``extrinsic``, ``intrinsic`` and processed ``image_shape=(H,W)``. RGB images
    may optionally be provided as ``images`` with shape ``[S,H,W,3]``.
    """
    if isinstance(port, bool) or not isinstance(port, numbers.Integral) or not 1 <= port <= 65535:
        raise ValueError("port must be an integer in [1,65535]")
    if not isinstance(host, str) or not host.strip():
        raise ValueError("host must be a nonempty address")
    if not isinstance(point_size, numbers.Real) or isinstance(point_size, bool) or not math.isfinite(point_size) or point_size <= 0:
        raise ValueError("point_size must be positive and finite")
    context_indices = _sample_indices(len(result.context_points), max_context_points)
    trajectory = _camera_trajectory(geometry)
    images = geometry.get("images")
    if images is not None:
        images = np.asarray(images)
        if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8 or len(images) != len(trajectory["intrinsic"]):
            raise ValueError("Viewer images must be uint8 [S,H,W,3] aligned with the camera trajectory")
        height, width = images.shape[1:3]
    else:
        shape = geometry.get("image_shape")
        if shape is None or len(shape) != 2 or any(isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1 for value in shape):
            raise ValueError("Viewer geometry requires processed image_shape=(H,W)")
        height, width = shape
    try:
        import viser
        from viser.transforms import SO3
    except ImportError as error:
        raise RuntimeError("The interactive viewer requires viser; install the baseline requirements or omit --serve") from error

    server = viser.ViserServer(host=host, port=int(port))
    try:
        # The first decoded camera uses OpenCV's +Y-down convention.
        if len(trajectory["camera_to_world"]):
            server.scene.set_up_direction(-trajectory["camera_to_world"][0, :3, 1])
        context = server.scene.add_point_cloud(
            "/context", points=result.context_points[context_indices], colors=CONTEXT_COLOR,
            point_size=float(point_size),
        )
        positive = server.scene.add_point_cloud(
            "/path_points", points=result.path_points, colors=PATH_COLOR,
            point_size=float(point_size), visible=False,
        )
        selected = result.path_flags | (result.path_probabilities > 0)
        colors = np.full((int(selected.sum()), 3), UNCERTAIN_COLOR, dtype=np.uint8)
        colors[result.path_flags[selected]] = PATH_COLOR
        voxels = server.scene.add_point_cloud(
            "/path_voxels", points=result.voxel_centers[selected], colors=colors,
            point_size=float(point_size) * 2.0,
        )
        handles = {"Context": context, "Positive path points": positive, "Path voxels": voxels}
        for label, handle in handles.items():
            toggle = server.gui.add_checkbox(label, initial_value=handle.visible)

            @toggle.on_update
            def update_visibility(event, scene_handle=handle):
                scene_handle.visible = event.target.value

        camera_handles = []
        for index, (pose, intrinsic) in enumerate(zip(trajectory["camera_to_world"], trajectory["intrinsic"])):
            camera_handles.append(server.scene.add_camera_frustum(
                f"/cameras/{index:06d}",
                fov=float(2 * np.arctan(height / (2 * intrinsic[1, 1]))),
                aspect=float(width / height),
                scale=float(point_size) * 10,
                wxyz=SO3.from_matrix(pose[:3, :3]).wxyz,
                position=pose[:3, 3],
                color=(100, 150, 255),
                image=None if images is None else images[index],
            ))
        camera_toggle = server.gui.add_checkbox("Camera trajectory", initial_value=True)

        @camera_toggle.on_update
        def update_cameras(event):
            for handle in camera_handles:
                handle.visible = event.target.value

        print(f"Path map viewer: http://{host}:{port} (Ctrl-C to stop)", flush=True)
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
