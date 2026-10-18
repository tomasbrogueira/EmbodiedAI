# Support direct compatibility-script execution without environment mutation.
if __package__ in (None, ''):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
"""Compare corresponding RGB/depth observations, without ground-truth claims."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from orchestrator.paths import ROOT
sys.path.insert(0, str(ROOT))


def fit_similarity(source, target):
    """Least-squares global Sim3: target ≈ scale * source @ rotation.T + translation."""
    source, target = np.asarray(source, np.float64), np.asarray(target, np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError('Expected matching N x 3 arrays with at least three points')
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    x, y = source - source_mean, target - target_mean
    variance = np.mean(np.sum(x*x, axis=1))
    if variance <= 1e-12:
        raise ValueError('Cannot align a degenerate point cloud')
    u, singular, vt = np.linalg.svd(y.T @ x / len(x))
    sign = np.ones(3)
    sign[-1] = np.linalg.det(u @ vt)
    rotation = (u * sign) @ vt
    scale = float((singular * sign).sum() / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def percentiles(values):
    values = np.asarray(values)
    return {'n': int(values.size), 'median': float(np.median(values)),
            'p95': float(np.percentile(values, 95)), 'rmse': float(np.sqrt(np.mean(values**2)))}


def compare(candidate, reference):
    from orchestrator.shared import read_rgb, rgb_digest
    candidate_paths = sorted(candidate.glob('frame_*.npz'))
    reference_paths = sorted(reference.glob('frame_*.npz'))
    if [p.name for p in candidate_paths] != [p.name for p in reference_paths] or not candidate_paths:
        raise ValueError('Candidate and reference must contain identical source frame files')
    sys.path.insert(0, str(ROOT / 'orchestrator'))
    from lingbot_geometry import depth_to_world_points
    from io_utils import load_depth_map, load_intrinsic, load_extrinsic
    frames, source_points, target_points, sample_depths = [], [], [], []
    for cp, rp in zip(candidate_paths, reference_paths):
        with np.load(cp) as data:
            c = {k: data[k].copy() for k in data.files}
        with np.load(rp) as data:
            r = {k: data[k].copy() for k in data.files}
        if rgb_digest(read_rgb(cp)) != rgb_digest(read_rgb(rp)):
            raise ValueError(f'RGB mismatch: {cp.name}')
        cd, rd = load_depth_map(c['depth']), load_depth_map(r['depth'])
        cc, rc = load_depth_map(c['depth_conf']), load_depth_map(r['depth_conf'])
        cw = depth_to_world_points(cd, load_intrinsic(c['intrinsic']), load_extrinsic(c['extrinsic']))
        rw = depth_to_world_points(rd, load_intrinsic(r['intrinsic']), load_extrinsic(r['extrinsic']))
        valid = (np.isfinite(cw).all(-1) & np.isfinite(rw).all(-1) & np.isfinite(cd) & np.isfinite(rd)
                 & np.isfinite(cc) & np.isfinite(rc) & (cc > 2.5) & (rc > 2.5)
                 & (cd > 0) & (rd > 0) & (cd < 25) & (rd < 25))
        pixels = np.flatnonzero(valid)
        if not len(pixels):
            raise ValueError(f'No common valid geometry in {cp.name}')
        selected = pixels[np.linspace(0, len(pixels)-1, min(2048, len(pixels)), dtype=int)]
        source_points.append(cw.reshape(-1, 3)[selected])
        target_points.append(rw.reshape(-1, 3)[selected])
        sample_depths.append(rd.ravel()[selected])
        frames.append({'frame': cp.name, 'common_valid_pixels': len(pixels),
            'bit_exact': {k: bool(np.array_equal(c[k], r[k]))
                          for k in ('depth', 'depth_conf', 'intrinsic', 'extrinsic')},
            'relative_depth_difference': percentiles(np.abs(cd[valid]-rd[valid])/rd[valid])})
    x, y, depths = np.concatenate(source_points), np.concatenate(target_points), np.concatenate(sample_depths)
    scale, rotation, translation = fit_similarity(x, y)
    raw_error = np.linalg.norm(x-y, axis=1)
    aligned_error = np.linalg.norm(scale * x @ rotation.T + translation - y, axis=1)
    offset = 0
    for frame, points, frame_depths in zip(frames, source_points, sample_depths):
        count = len(points)
        frame['aligned_point_difference'] = percentiles(aligned_error[offset:offset+count])
        frame['aligned_point_difference_over_reference_depth'] = percentiles(aligned_error[offset:offset+count]/frame_depths)
        offset += count
    return {'candidate': str(candidate.resolve()), 'reference': str(reference.resolve()),
        'rgb_hashes_match': True, 'frames': frames,
        'all_prediction_arrays_bit_exact_frames': sum(all(f['bit_exact'].values()) for f in frames),
        'global_alignment': {'scale': scale, 'rotation': rotation.tolist(), 'translation': translation.tolist()},
        'raw_point_difference_model_units': percentiles(raw_error),
        'aligned_point_difference_model_units': percentiles(aligned_error),
        'aligned_point_difference_over_reference_depth': percentiles(aligned_error/depths),
        'interpretation': 'Global Sim3 removes coordinate/scale differences; residuals compare predictions, not accuracy against ground truth.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.candidate, args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('all_prediction_arrays_bit_exact_frames',
        'aligned_point_difference_over_reference_depth')}, indent=2))


if __name__ == '__main__':
    main()
