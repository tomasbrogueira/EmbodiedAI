"""Estimate a relative floor normal from existing masked, world-frame depth points."""

from itertools import islice

import numpy as np
from scipy.ndimage import binary_erosion

MAX_FRAMES = 5
MIN_POINTS = 100
MAX_SAMPLES = 2000
RANSAC_TRIALS = 128
MIN_SUPPORT = 0.85
MAX_DISAGREEMENT_DEGREES = 5.0


def _fit_frame(frame, rng):
    points = np.asarray(frame["points"], dtype=float)
    mask = np.asarray(frame["floor_mask"], dtype=bool)
    camera = np.asarray(frame["camera"], dtype=float)
    if points.shape != (*mask.shape, 3) or mask.ndim != 2 or camera.shape != (3,):
        raise ValueError("Expected points[H,W,3], floor_mask[H,W] and camera[3].")
    mask = binary_erosion(mask & np.isfinite(points).all(axis=-1),
                          structure=np.ones((3, 3)), iterations=3)
    candidates = np.unique(points[mask], axis=0)
    info = {"id": frame["id"], "point_count": len(candidates), "support": 0.0}
    if len(candidates) < MIN_POINTS or not np.isfinite(camera).all():
        return None, info | {"reason": "insufficient_points_or_invalid_camera"}
    vertical, horizontal = mask[1:] & mask[:-1], mask[:, 1:] & mask[:, :-1]
    spacing = np.concatenate([
        np.linalg.norm(points[1:][vertical] - points[:-1][vertical], axis=-1),
        np.linalg.norm(points[:, 1:][horizontal] - points[:, :-1][horizontal], axis=-1),
    ])
    spacing = spacing[np.isfinite(spacing) & (spacing > 0)]
    if not len(spacing):
        return None, info | {"reason": "missing_pixel_spacing"}
    if len(candidates) > MAX_SAMPLES:
        candidates = candidates[rng.choice(len(candidates), MAX_SAMPLES, replace=False)]
    tolerance = min(2 * np.median(spacing), 0.01 * np.linalg.norm(np.ptp(candidates, axis=0)))
    info.update(spacing=float(np.median(spacing)), tolerance=float(tolerance),
                sample_count=len(candidates))
    if not np.isfinite(tolerance) or tolerance <= 0:
        return None, info | {"reason": "invalid_tolerance"}
    best = np.zeros(len(candidates), dtype=bool)
    for _ in range(RANSAC_TRIALS):
        a, b, c = candidates[rng.choice(len(candidates), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length == 0:
            continue
        inliers = np.abs((candidates - a) @ (normal / length)) <= tolerance
        if inliers.sum() > best.sum():
            best = inliers
    info["support"] = float(best.mean())
    if best.sum() < MIN_POINTS or info["support"] < MIN_SUPPORT:
        return None, info | {"reason": "insufficient_plane_support"}
    centre = candidates[best].mean(axis=0)
    centred = candidates[best] - centre
    values, vectors = np.linalg.eigh(centred.T @ centred)
    if values[1] <= 0 or values[1] < 0.02 * values[2] or values[0] > 0.01 * values[1]:
        return None, info | {"reason": "poor_plane_fit"}
    normal = vectors[:, 0]
    info["support"] = float((np.abs((candidates - centre) @ normal) <= tolerance).mean())
    if info["support"] < MIN_SUPPORT:
        return None, info | {"reason": "insufficient_refitted_support"}
    toward_camera = float(normal @ (camera - centre))
    if abs(toward_camera) <= tolerance:
        return None, info | {"reason": "ambiguous_camera_side"}
    normal *= np.sign(toward_camera)
    return normal, info | {"reason": "accepted"}


def estimate_floor_reference(frames):
    """Return (normal or None, evidence) from the first five frame dictionaries.

    Each frame supplies id, points[H,W,3], geometry-filtered floor_mask[H,W],
    and its camera[3] centre in the same world frame. No gravity is inferred.
    """
    rng = np.random.default_rng(0)
    normals, reports = [], []
    for frame in islice(frames, MAX_FRAMES):
        normal, info = _fit_frame(frame, rng)
        reports.append(info)
        if normal is not None:
            normals.append(normal)
    report = {"frames": reports, "max_disagreement_degrees": None}
    if len(normals) < 3:
        return None, report | {"reason": "insufficient_reference_frames"}
    normals = np.asarray(normals)
    dots = np.clip(normals @ normals.T, -1, 1)
    disagreement = float(np.degrees(np.arccos(np.abs(dots))).max())
    report["max_disagreement_degrees"] = disagreement
    if disagreement > MAX_DISAGREEMENT_DEGREES or np.any(dots < 0):
        return None, report | {"reason": "inconsistent_reference_frames"}
    normal = normals.mean(axis=0)
    normal /= np.linalg.norm(normal)
    return normal, report | {"reason": "ok"}


def estimate_local_floor_reference(frames, scene_report=None):
    """Fallback from locally planar near-field ground in three distinct frames.

    Roughness/curvature over a whole image need not prevent identifying a local
    reference. Each crop retains the original support, plane and consistency
    requirements; multiple crops of one frame never count as multiple frames.
    """
    rng=np.random.default_rng(0);normals=[];reports=[];seen=set()
    scene={int(f['id']):f for f in (scene_report or {}).get('frames',[])}
    for frame in islice(frames,MAX_FRAMES):
        identity=int(frame['id'])
        if identity in seen:continue
        seen.add(identity);mask=np.asarray(frame['floor_mask'],bool);h,w=mask.shape
        y,x=np.where(mask&np.isfinite(frame['points']).all(-1));chosen=None;attempts=[]
        if len(y):
            size=min(64,h,w)
            for fraction in (.85,.6,.35):
                cy=int(np.quantile(y,fraction));near=abs(y-cy)<size//4
                cx=int(np.median(x[near]));y0=int(np.clip(cy-size//2,0,h-size));x0=int(np.clip(cx-size//2,0,w-size))
                local=dict(frame,points=frame['points'][y0:y0+size,x0:x0+size],floor_mask=mask[y0:y0+size,x0:x0+size])
                normal,info=_fit_frame(local,rng);attempts.append(info['reason'])
                if normal is not None:
                    chosen=info|dict(crop=[x0,y0,x0+size,y0+size],scene_spacing=scene.get(identity,{}).get('spacing',info['spacing']))
                    normals.append(normal);break
        reports.append(chosen or dict(id=identity,reason='no_supported_local_plane',attempts=attempts))
    report=dict(frames=reports,method='local ground patches',max_disagreement_degrees=None)
    if len(normals)<3:return None,report|dict(reason='insufficient_reference_frames')
    normals=np.asarray(normals);dots=np.clip(normals@normals.T,-1,1)
    disagreement=float(np.degrees(np.arccos(abs(dots))).max());report['max_disagreement_degrees']=disagreement
    if disagreement>MAX_DISAGREEMENT_DEGREES or np.any(dots<0):return None,report|dict(reason='inconsistent_reference_frames')
    normal=normals.mean(0);normal/=np.linalg.norm(normal)
    return normal,report|dict(reason='ok')
