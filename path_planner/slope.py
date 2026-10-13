"""Point-wise inclination and optional local geometric quality (NumPy/SciPy)."""
from numbers import Integral, Real
import numpy as np
from scipy.spatial import cKDTree


def _settings(config):
    if not isinstance(config, dict) or 'radius' not in config:
        raise ValueError('config must be a dictionary containing radius')
    if config.keys() - {'radius', 'neighbors', 'min_samples'}:
        raise ValueError('Supported slope settings: radius, neighbors, min_samples')
    s = {'neighbors': 128, 'min_samples': 6, **config}
    if (isinstance(s['radius'], (bool, np.bool_)) or not isinstance(s['radius'], Real)
            or not np.isfinite(s['radius']) or s['radius'] <= 0):
        raise ValueError('radius must be positive and finite')
    for key in ('neighbors', 'min_samples'):
        if isinstance(s[key], (bool, np.bool_)) or not isinstance(s[key], Integral):
            raise ValueError(f'{key} must be an integer')
    if not 6 <= s['min_samples'] <= s['neighbors']:
        raise ValueError('Require 6 <= min_samples <= neighbors')
    return s


def _plane(patches, weights):
    total = weights.sum(axis=1)
    centre = np.einsum('bk,bkj->bj', weights, patches) / total[:, None]
    z = patches - centre[:, None, :]
    cov = np.einsum('bki,bkj,bk->bij', z, z, weights) / total[:, None, None]
    values, vectors = np.linalg.eigh(cov)
    return centre, np.maximum(values, 0), vectors[:, :, 0]


def _fit_patches(patches, valid, min_samples=6):
    """Fit bounded distinct neighbours; query is origin, units are radius-scaled."""
    normals = np.full((len(patches), 3), np.nan)
    quality = np.zeros(len(patches))
    rows = np.flatnonzero(valid.sum(axis=1) >= min_samples)
    if not len(rows):
        return normals, quality
    p, mask = patches[rows], valid[rows]
    centre, initial, normal = _plane(p, mask.astype(float))
    residual = np.einsum('bkj,bj->bk', p - centre[:, None, :], normal)
    median = np.nanmedian(np.where(mask, residual, np.nan), axis=1)
    deviation = residual - median[:, None]
    scale = np.maximum(1.4826 * np.nanmedian(
        np.where(mask, np.abs(deviation), np.nan), axis=1), 1e-9)
    cutoff = 1.345 * scale
    weights = mask * np.minimum(1., cutoff[:, None] /
                               np.maximum(np.abs(deviation), cutoff[:, None]))
    centre, values, normal = _plane(p, weights)
    effective = weights.sum(axis=1) ** 2 / (weights * weights).sum(axis=1)
    # Numerical identifiability only; poor but defined planes retain an estimate.
    good = ((effective >= min_samples - 1e-9)
            & (initial[:, 2] > 0) & (values[:, 2] > 0)
            & (initial[:, 1] - initial[:, 0] > 1e-6 * initial[:, 2])
            & (values[:, 1] - values[:, 0] > 1e-6 * values[:, 2]))
    residual = np.einsum('bkj,bj->bk', p - centre[:, None, :], normal)
    median = np.nanmedian(np.where(mask, residual, np.nan), axis=1)
    scale = np.maximum(1.4826 * np.nanmedian(
        np.where(mask, np.abs(residual - median[:, None]), np.nan), axis=1), 1e-9)
    query_residual = np.abs(-np.einsum('bj,bj->b', centre, normal) - median)
    agreement = np.minimum(1., 1.345 * scale / np.maximum(query_residual, 1.345 * scale))
    tiny = np.finfo(float).tiny
    shape = np.minimum((initial[:, 1] - initial[:, 0]) / np.maximum(initial[:, 2], tiny),
                       (values[:, 1] - values[:, 0]) / np.maximum(values[:, 2], tiny))
    normals[rows[good]] = normal[good]
    quality[rows[good]] = np.clip(shape[good], 0, 1) * agreement[good]
    return normals, quality


def estimate_slope(points, n_ref, config, *, frame_ids=None, return_quality=False):
    """Return radians[N], optionally (radians[N], quality[N]), in original row order.

    Fit up to neighbors distinct same-frame positions within radius, with one
    Huber-weighted PCA refit. Without frame_ids all rows are one observation set.
    Quality is a local shape/query-agreement score in [0,1], not a probability or
    traversability certificate. Undefined normals have slope NaN and quality 0.
    """
    s = _settings(config)
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or points.dtype.kind not in 'iuf':
        raise ValueError('points must be a real array with shape (N,3)')
    if frame_ids is not None:
        frame_ids = np.asarray(frame_ids)
        if frame_ids.shape != (len(points),) or frame_ids.dtype.kind not in 'iu':
            raise ValueError('frame_ids must be an integer array with shape (N,)')
    ref = np.asarray(n_ref)
    if ref.shape != (3,) or ref.dtype.kind not in 'iuf' or not np.isfinite(ref).all():
        raise ValueError('n_ref must have three finite real values')
    ref = ref.astype(float)
    if not np.any(ref):
        raise ValueError('n_ref must be nonzero')
    ref /= np.max(np.abs(ref))
    ref /= np.linalg.norm(ref)
    slope, quality = np.full(len(points), np.nan), np.zeros(len(points))
    finite = np.flatnonzero(np.isfinite(points).all(axis=1))
    labels = np.zeros(len(finite), dtype=int) if frame_ids is None else frame_ids[finite]
    for label in np.unique(labels):
        rows = finite[labels == label]
        unique, inverse = np.unique(points[rows].astype(float), axis=0, return_inverse=True)
        if len(unique) < s['min_samples']:
            continue
        tree = cKDTree(unique)
        angles, scores = np.full(len(unique), np.nan), np.zeros(len(unique))
        k = min(s['neighbors'], len(unique))
        for start in range(0, len(unique), 2048):
            query = unique[start:start + 2048]
            distance, index = tree.query(query, k=k,
                distance_upper_bound=np.nextafter(s['radius'], np.inf))
            valid = np.isfinite(distance)
            patches = (unique[np.minimum(index, len(unique)-1)] - query[:, None, :]) / s['radius']
            patches[~valid] = 0
            normal, score = _fit_patches(patches, valid, s['min_samples'])
            angles[start:start+len(query)] = np.arccos(np.clip(np.abs(normal @ ref), 0, 1))
            scores[start:start+len(query)] = score
        slope[rows], quality[rows] = angles[inverse], scores[inverse]
    return (slope, quality) if return_quality else slope
