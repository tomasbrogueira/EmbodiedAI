"""Associate recording poses with observed support, then use the point planner."""

from numbers import Real
from pathlib import Path

import numpy as np

from path_planner.planner import _aligned_costs, _failure, find_path


def load_cost_evidence(*args, **kwargs):
    """Legacy recording-file adapter, imported only when explicitly requested."""
    from orchestrator.batch.cost_evidence import load_cost_evidence as load
    return load(*args, **kwargs)


def select_endpoint(points, cost_map, camera, *, max_drop=None):
    """Choose the closest observed floor row below the camera within radius/2.

    Reuse its existing cost; blocked nearest support is never skipped. The
    reported projection is on the downward ray at the selected row's height.
    """
    points, costs = _aligned_costs(points, cost_map)
    if max_drop is not None and (isinstance(max_drop, (bool, np.bool_))
            or not isinstance(max_drop, Real) or not np.isfinite(max_drop)
            or max_drop <= 0):
        raise ValueError("max_drop must be finite and positive, or None")
    result = dict(reason="missing_pose", id=None, camera=None, projected=None,
                  offset=None, distance=None)
    if camera is None:
        return result
    camera = np.asarray(camera, dtype=float)
    if camera.shape != (3,) or not np.isfinite(camera).all():
        return result
    result["camera"] = camera.copy()
    reference = cost_map.reference
    if reference is None:
        return result | {"reason": "missing_reference"}
    config = cost_map.config
    radius = config["radius"]
    rows = np.flatnonzero(config["support_mask"] & np.isfinite(points).all(axis=1))
    delta = (points[rows] - camera) / radius
    drop = -delta @ reference
    lateral = np.linalg.norm(delta + drop[:, None] * reference, axis=1)
    keep = (drop > 0) & (lateral <= .5)
    if max_drop is not None:
        keep &= drop <= max_drop / radius
    rows, delta, drop = rows[keep], delta[keep], drop[keep]
    if not len(rows):
        return result | {"reason": "missing_support"}
    # Input order gives the lowest original row ID for exact distance ties.
    nearest = int(np.argmin(np.linalg.norm(delta, axis=1)))
    row = int(rows[nearest])
    depth = float(drop[nearest] * radius)
    projected = camera - depth * reference
    offset = points[row] - projected
    result.update(projected=projected, offset=offset,
                  distance=float(np.linalg.norm(offset)), drop=depth)
    if not np.isfinite(costs[row]):
        return result | {"reason": "blocked_support"}
    return result | {"reason": "ok", "id": row}


def find_recording_path(points, cost_map, lingbot_dir, sam_dir, video_name, *,
                        max_drop=None, **evidence_settings):
    """Return (PathResult, endpoint report) using first/last saved recording poses.

    Pass the cloud's frame_step, pixel_step and any nondefault export filters.
    The existing evidence loader checks every XYZ row against these predictions;
    floor labels must also match this cost map. Pose errors never skip a frame.
    """
    from io_utils import frame_number, load_extrinsic

    points, _ = _aligned_costs(points, cost_map)
    paths = sorted(Path(lingbot_dir).glob("frame_*.npz"), key=frame_number)
    report = {"correspondence_verified": False}
    cameras = []
    for name, path in zip(("start", "goal"),
                          (paths[0], paths[-1]) if paths else (None, None)):
        camera, reason = None, "missing_pose"
        if path is not None:
            with np.load(path, allow_pickle=False) as data:
                if "extrinsic" in data:
                    try:
                        extrinsic = load_extrinsic(data["extrinsic"])
                        rotation = extrinsic[:, :3]
                        if (not np.allclose(rotation.T @ rotation, np.eye(3),
                                            atol=1e-4, rtol=0)
                                or np.linalg.det(rotation) <= 0):
                            raise ValueError("Pose rotation must be proper orthonormal")
                        camera = -rotation.T @ extrinsic[:, 3]
                        reason = "ok" if np.isfinite(camera).all() else "invalid_pose"
                    except ValueError:
                        reason = "invalid_pose"
        cameras.append(camera)
        report[name] = dict(reason=reason, id=None, camera=camera,
                            frame_id=frame_number(path) if path else None)
    # Return both pose diagnostics before asking the geometry adapter to load
    # frames that may have missing extrinsics.
    for name in ("start", "goal"):
        if report[name]["reason"] != "ok":
            return _failure("invalid_" + name), report
    evidence = load_cost_evidence(points, lingbot_dir, sam_dir, video_name,
                                  **evidence_settings)
    if not np.array_equal(evidence["support_mask"], cost_map.config["support_mask"]):
        raise ValueError("Recording floor labels do not match this cost map")
    report["correspondence_verified"] = True
    for name, camera in zip(("start", "goal"), cameras):
        report[name].update(select_endpoint(points, cost_map, camera,
                                            max_drop=max_drop))
    return find_path(points, cost_map, report["start"]["id"],
                     report["goal"]["id"]), report
