# Support direct compatibility-script execution without environment mutation.
if __package__ in (None, ''):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from orchestrator.paths import source_path
"""Prepare unchanged per-recording point rows and existing geometric costs."""

import argparse
import hashlib
import json
import multiprocessing
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from orchestrator.batch.dataset_inventory import sha256 as digest

CALIBRATION_POLICY = 1


def camera_clearance(poses, radius):
    """Freeze height from baseline beneath-camera support, never filtered poses."""
    samples = [pose for pose in poses if pose.get('reason') == 'ok'
               and pose.get('association_mode') == 'beneath_camera'
               and pose.get('drop') is not None and np.isfinite(pose['drop']) and pose['drop'] > 0]
    drops = [float(pose['drop']) for pose in samples]
    return dict(policy=CALIBRATION_POLICY, reason='ok' if samples else 'missing_clearance_calibration',
                height=float(max(radius, np.median(drops))) if samples else None,
                baseline_height=float(radius), sample_count=len(samples), drops=drops,
                frame_ids=[int(pose['frame_id']) for pose in samples],
                point_row_ids=[int(pose['id']) for pose in samples])


def costs_current(prepared, provenance, cost_identity):
    if prepared.get('identity') != identity_hash(cost_identity):
        return False
    if any(provenance.get(key) != cost_identity[key] for key in
           ('cost_source_sha256', 'prepare_adapter_sha256', 'calibration_association_sha256', 'clearance_calibration_policy')):
        return False
    if prepared.get('cost_source_sha256') != cost_identity['cost_source_sha256']:
        return False
    calibration = provenance.get('clearance_calibration', {})
    if prepared.get('status') == 'ok':
        height = calibration.get('height')
        return (calibration.get('reason') == 'ok' and calibration.get('sample_count', 0) > 0
                and height is not None and height == prepared.get('clearance_height')
                and height == provenance.get('settings', {}).get('clearance_height'))
    return (prepared.get('status') == 'missing_reference' or
            prepared.get('status') == calibration.get('reason') == 'missing_clearance_calibration')


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    def numpy_json(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Cannot encode {type(item).__name__}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=numpy_json))
    temporary.replace(path)


def save(path, values):
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, values)
    temporary.replace(path)


def identity_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def geometry_identity(identity):
    return identity | {'source_hashes': {name: value for name, value in identity['source_hashes'].items()
                                        if Path(name).name != 'cost_map.py'}}


def migrate_geometry(out, identity):
    """Allow only a cost-source change; geometry and slope arrays stay untouched."""
    key = identity_hash(geometry_identity(identity))
    geometry_path, slope_path = out / 'geometry.json', out / 'slopes-started.json'
    if not geometry_path.exists():
        return key
    geometry = read(geometry_path)
    allowed = {key}
    provenance_path = out / 'data/provenance.json'
    if provenance_path.exists():
        old = read(provenance_path)
        if old.get('correspondence_verified') and all(name in old for name in identity):
            previous = {name: old[name] for name in identity}
            if geometry_identity(previous) == geometry_identity(identity):
                allowed.add(identity_hash(previous))
    if geometry['identity'] not in allowed:
        raise ValueError('Cached geometry inputs or non-cost source changed; use a fresh output folder')
    slope = read(slope_path) if slope_path.exists() else None
    if slope is not None and slope['identity'] not in allowed:
        raise ValueError('Slope checkpoint belongs to different geometry or source')
    if geometry['identity'] != key:
        write(geometry_path, geometry | {'identity': key})
    if slope is not None and slope['identity'] != key:
        write(slope_path, slope | {'identity': key})
    return key


def import_reused_geometry(out, entry, gpu, identity, filters, load_cost_evidence, load_extrinsic):
    """Reverify the original demo's linked rows, retaining its reference and slopes."""
    data, old = out / 'data', read(out / 'data/provenance.json')
    if (not old.get('correspondence_verified') or not old.get('all_original_xyz_rgb_and_row_ids_verified')
            or old.get('input_sha256') != entry['input_sha256'] or old.get('prompt') != entry['prompt']
            or old.get('frame_count') != entry['frame_count']):
        raise ValueError('Reused geometry lacks matching verified recording provenance')
    ply = Path(old['source_ply'])
    if digest(ply) != old['source_sha256']:
        raise ValueError('Reused original point cloud changed')
    points, slope = np.load(data / 'points.npy', mmap_mode='r'), np.load(data / 'slope.npy', mmap_mode='r')
    if slope.shape != (len(points),) or (data / 'slope.npy').resolve() != Path(old['source_slope']).resolve():
        raise ValueError('Reused slope rows do not identify the original aligned array')
    evidence = load_cost_evidence(points, gpu['lingbot'], gpu['sam'], entry['name'],
                                 prompt=entry['prompt'], include_frame_ids=True, **filters)
    for name in ('support_mask', 'frame_ids'):
        if not np.array_equal(evidence[name], np.load(data / f'{name}.npy', mmap_mode='r')):
            raise ValueError(f'Reused original {name} correspondence differs')
    del evidence
    cameras, timestamps = np.load(data / 'cameras.npy'), np.load(data / 'timestamps.npy')
    if cameras.shape != (entry['frame_count'], 3) or not np.array_equal(timestamps, entry['timestamps']):
        raise ValueError('Reused camera and timestamp rows differ from the recording')
    paths = sorted(Path(gpu['lingbot']).glob('frame_*.npz'), key=lambda path: int(path.stem.split('_')[-1]))
    if [int(path.stem.split('_')[-1]) for path in paths] != list(range(len(cameras))):
        raise ValueError('Reused reconstruction does not cover every camera frame')
    for frame, (camera, path) in enumerate(zip(cameras, paths)):
        with np.load(path, allow_pickle=False) as saved:
            extrinsic = load_extrinsic(saved['extrinsic'])
        if not np.array_equal(camera, -extrinsic[:, :3].T @ extrinsic[:, 3]):
            raise ValueError(f'Reused camera pose differs at frame {frame}')
    reference = np.load(data / 'reference.npy')
    if not np.allclose(reference, old['n_ref'], rtol=0, atol=1e-12):
        raise ValueError('Reused reference differs from its original provenance')
    report = dict(reason='reused_verified_reference', frames=[],
                  slope_config=dict(radius=old['settings']['radius']))
    write(out / 'geometry.json', dict(identity=identity_hash(geometry_identity(identity)), total_rows=len(points),
          reference_available=True, reference=reference, reference_scan_version=1, reference_report=report,
          source_ply=str(ply), source_ply_sha256=old['source_sha256'], reused_slopes=True, legacy_provenance=old))


def slope_frame(frame):
    from slope import estimate_slope

    start, end = BOUNDS[frame:frame + 2]
    if end > start:
        slopes, quality = estimate_slope(POINTS[start:end], REFERENCE, SLOPE_CONFIG, return_quality=True)
        SLOPES[start:end], QUALITY[start:end] = slopes, quality
        SLOPES.flush()
        QUALITY.flush()
    DONE[frame] = True
    DONE.flush()
    return frame


def main():
    global POINTS, REFERENCE, SLOPE_CONFIG, BOUNDS, SLOPES, QUALITY, DONE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    if args.output.is_symlink() or (args.output / 'data').is_symlink():
        raise ValueError('Cost revisions require a batch-local output/data directory, not a directory symlink')
    out, repo = args.output.resolve(), args.repo.resolve()
    data = out / "data"
    data.mkdir(exist_ok=True)
    sys.path[:0] = [str(repo / "path-planning"), str(repo / "orchestrator")]
    from cost_evidence import load_cost_evidence
    from io_utils import load_extrinsic
    from dataset_reference import SCAN_VERSION, reference_from_video
    import orchestrator.batch.dataset_reference as dataset_reference
    import dataset_path
    from orchestrator.ply_io import load_ply
    import cost_map

    entry, gpu = read(out / "entry.json"), read(out / "gpu.json")
    previous_report = read(out / 'report.json') if (out / 'report.json').exists() else {}
    reused = previous_report.get('reused_verified_output')
    count = int(entry["frame_count"])
    if (gpu["input_sha256"] != entry["input_sha256"] or gpu["prompt"] != entry["prompt"]
            or gpu["frames"] != count
            or (gpu.get('model_settings') != entry['model_settings'] and not (reused and 'model_settings' not in gpu))):
        raise ValueError("GPU results do not identify this input, prompt and model settings")
    if digest(Path(entry["video"])) != entry["input_sha256"]:
        raise ValueError("Input video changed after the dataset manifest was created")
    filters = dict(frame_step=2, pixel_step=1, min_depth_conf=2.5, max_depth=25., patch_size=14)
    sources = {str(p.relative_to(repo)): digest(source_path(repo, str(p.relative_to(repo)))) for folder in ("path-planning", "orchestrator")
               for p in sorted((repo / folder).glob("*.py")) if not p.name.startswith("test_")}
    identity = dict(input_sha256=entry["input_sha256"], prompt=entry["prompt"],
                    model_settings=gpu.get('model_settings'), source_hashes=sources, filters=filters,
                    lingbot=gpu["lingbot"], sam=gpu["sam"], frame_count=count, timestamps=entry["timestamps"])
    cost_source = digest(source_path(repo, 'path-planning/cost_map.py'))
    prepare_adapter = digest(Path(__file__))
    calibration_association = digest(Path(dataset_path.__file__))
    receipt = out / "prepared.json"
    geometry = out / "geometry.json"
    if reused and not geometry.exists():
        import_reused_geometry(out, entry, gpu, identity, filters, load_cost_evidence, load_extrinsic)
    key = migrate_geometry(out, identity)
    cost_identity = dict(geometry_identity=key, cost_source_sha256=cost_source,
                         prepare_adapter_sha256=prepare_adapter, calibration_association_sha256=calibration_association,
                         clearance_calibration_policy=CALIBRATION_POLICY)
    cost_key = identity_hash(cost_identity)
    if receipt.exists():
        prepared = read(receipt)
        if prepared["status"] == "missing_reference" and not read(geometry).get("reference_scan_version"):
            receipt.unlink()  # Retry only the formerly opening-window reference failure.
        elif costs_current(prepared, read(data / 'provenance.json'), cost_identity):
            print("Prepared map and costs already verified", flush=True)
            return

    def status(stage, message):
        write(out / "status.json", dict(stage=stage, status="running", message=message))
        print(message, flush=True)

    if not geometry.exists():
        status("map", "Building the original semantic point cloud")
        ply = out / f"{entry['name']}_{entry['prompt']}_semantic_map.ply"
        subprocess.run([sys.executable, "-B", "-u", str(repo / "orchestrator" / "main.py"),
                        "--video-path", entry["video"], "--lingbot-dir", gpu["lingbot"],
                        "--sam-dir", gpu["sam"], "--output", str(ply), "--prompt", entry["prompt"],
                        "--frame-step", "2", "--pixel-step", "1", "--min-depth-conf", "2.5",
                        "--max-depth", "25", "--patch-size", "14", "--segment-color", "255,0,0"], check=True)
        points, colours = load_ply(ply, max_points=0)
        colours = np.asarray(colours)
        if colours.dtype != np.uint8:
            if colours.dtype.kind == "f" and colours.size and np.nanmax(colours) <= 1:
                colours = colours * 255
            colours = np.clip(colours, 0, 255).astype(np.uint8)
        evidence = load_cost_evidence(points, gpu["lingbot"], gpu["sam"], entry["name"],
                                      prompt=entry["prompt"], include_frame_ids=True, **filters)
        frame_ids = evidence["frame_ids"]
        if frame_ids.shape != (len(points),) or (np.diff(frame_ids) < 0).any() or (frame_ids < 0).any() or (frame_ids >= count).any():
            raise ValueError("Point provenance does not match ordered original video frames")
        paths = sorted(Path(gpu["lingbot"]).glob("frame_*.npz"), key=lambda p: int(p.stem.split("_")[-1]))
        if [int(p.stem.split("_")[-1]) for p in paths] != list(range(count)):
            raise ValueError("Recording poses do not cover every original video frame")
        cameras = np.empty((count, 3), dtype=np.float64)
        for frame, path in enumerate(paths):
            with np.load(path, allow_pickle=False) as saved:
                extrinsic = load_extrinsic(saved["extrinsic"])
            rotation, translation = extrinsic[:, :3], extrinsic[:, 3]
            if (not np.isfinite(extrinsic).all() or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4, rtol=0)
                    or np.linalg.det(rotation) <= 0):
                raise ValueError(f"Invalid rigid camera pose at frame {frame}")
            cameras[frame] = -rotation.T @ translation
        timestamps = np.asarray(entry["timestamps"], dtype=np.float64)
        if timestamps.shape != (count,) or not np.isfinite(timestamps).all() or (np.diff(timestamps) < 0).any():
            raise ValueError("Actual video timestamps do not correspond to camera rows")
        reference, reference_report = reference_from_video(gpu["lingbot"], gpu["sam"], entry["name"], prompt=entry["prompt"])
        for name, array in dict(points=np.asarray(points, dtype=np.float32), colours=colours,
                                support_mask=evidence["support_mask"], frame_ids=frame_ids,
                                cameras=cameras, timestamps=timestamps,
                                reference=np.full(3, np.nan) if reference is None else reference).items():
            save(data / f"{name}.npy", array)
        write(geometry, dict(identity=key, total_rows=len(points), reference_available=reference is not None,
                             reference=None if reference is None else np.asarray(reference).tolist(),
                             reference_scan_version=SCAN_VERSION,
                             reference_report=reference_report, source_ply=str(ply), source_ply_sha256=digest(ply)))
        del points, colours, evidence, frame_ids, cameras

    saved_geometry = read(geometry)
    if not saved_geometry["reference_available"] and not saved_geometry.get("reference_scan_version"):
        status("map", "Searching later video frames for a reliable ground direction")
        reference, reference_report = reference_from_video(gpu["lingbot"], gpu["sam"], entry["name"], prompt=entry["prompt"])
        saved_geometry.update(reference_available=reference is not None,
                              reference=None if reference is None else np.asarray(reference).tolist(),
                              reference_report=reference_report, reference_scan_version=SCAN_VERSION)
        save(data / "reference.npy", np.full(3, np.nan) if reference is None else reference)
        write(geometry, saved_geometry)
    total = saved_geometry["total_rows"]
    provenance = dict(identity, geometry_identity=key, cost_source_sha256=cost_source,
                      prepare_adapter_sha256=prepare_adapter, clearance_calibration_policy=CALIBRATION_POLICY,
                      calibration_association_sha256=calibration_association,
                      correspondence_verified=True, source_ply=saved_geometry["source_ply"],
                      source_ply_sha256=saved_geometry["source_ply_sha256"],
                      reference_adapter_sha256=digest(Path(dataset_reference.__file__)),
                      reference_report=saved_geometry["reference_report"], settings={})
    if 'legacy_provenance' in saved_geometry:
        provenance['legacy_provenance'] = saved_geometry['legacy_provenance']
        provenance['geometry_revalidated_from_legacy'] = True
        provenance['source_hashes_scope'] = 'Current validation and cost code; historical geometry and slope code were not recorded'
    report = dict(name=entry["name"], frame_count=count, total_rows=total, finite_cost_rows=0, cost_range=None,
                  correspondence_verified=True, supported_count=0, selected=dict(start=None, goal=None),
                  route_points=0, optimal=False, reason_counts={}, cost_source_sha256=cost_source,
                  prepare_adapter_sha256=prepare_adapter, clearance_calibration_policy=CALIBRATION_POLICY)
    report['calibration_association_sha256'] = calibration_association
    if reused:
        report['reused_verified_output'] = reused
    write(out / 'report.json', report | {'status': 'pending'})
    if not saved_geometry["reference_available"]:
        cost = np.lib.format.open_memmap(data / "cost.tmp.npy", mode="w+", dtype=np.float64, shape=(total,))
        cost[:] = np.inf
        cost.flush()
        del cost
        (data / 'cost.tmp.npy').replace(data / 'cost.npy')
        report.update(status="missing_reference", reason_counts=dict(missing_reference=count))
        write(data / "provenance.json", provenance)
        write(out / "report.json", report)
        write(receipt, dict(identity=cost_key, cost_source_sha256=cost_source, status="missing_reference"))
        write(out / "status.json", dict(stage="complete", status="missing_reference",
                                        message="Automatic reference unavailable; context map only, no route"))
        return

    POINTS = np.load(data / "points.npy", mmap_mode="r")
    REFERENCE = np.asarray(saved_geometry["reference"], dtype=np.float64)
    SLOPE_CONFIG = saved_geometry["reference_report"]["slope_config"]
    frame_ids = np.load(data / "frame_ids.npy", mmap_mode="r")
    BOUNDS = np.searchsorted(frame_ids, np.arange(count + 1))
    slope_receipt = out / "slopes-started.json"
    resume = slope_receipt.exists()
    if resume and read(slope_receipt)["identity"] != key:
        raise ValueError("Slope checkpoint belongs to different inputs/settings")
    complete = saved_geometry.get('reused_slopes') or (resume and np.load(data / 'done.npy').all())
    arrays = []
    for name, dtype, shape, initial in (() if complete else (("slope", np.float64, (total,), np.nan),
                                        ("quality", np.float64, (total,), np.nan),
                                        ("done", bool, (count,), False))):
        array = np.lib.format.open_memmap(data / f"{name}.npy", mode="r+" if resume else "w+", dtype=dtype, shape=shape)
        if not resume:
            array[:] = initial
            array.flush()
        arrays.append(array)
    if complete:
        SLOPES = np.load(data / 'slope.npy', mmap_mode='r')
        pending = []
    else:
        SLOPES, QUALITY, DONE = arrays
        write(slope_receipt, dict(identity=key))
        pending = np.flatnonzero(~DONE).tolist()
    status("costs", f"Estimating per-frame surface slopes: {count - len(pending)}/{count} frames ready")
    if pending:
        with ProcessPoolExecutor(max_workers=min(16, len(pending)), mp_context=multiprocessing.get_context("fork")) as pool:
            for _ in pool.map(slope_frame, pending, chunksize=1):
                status("costs", f"Estimating per-frame surface slopes: {int(DONE.sum())}/{count} frames ready")
    radius = SLOPE_CONFIG["radius"]
    base_tree = cost_map.cKDTree

    class ParallelTree(base_tree):
        def query(self, positions, *positional, **keywords):
            keywords.setdefault("workers", 16)
            # Explicit support bounds stay unchanged. Clearance above r is capped by the existing formula.
            if len(positional) < 4:
                keywords.setdefault("distance_upper_bound", radius)
            return super().query(positions, *positional, **keywords)

    cost_map.cKDTree = ParallelTree
    status("costs", "Computing existing support and obstacle-clearance costs")
    cm = cost_map.build_cost_map(POINTS, SLOPES, REFERENCE,
                                 dict(radius=radius, support_mask=np.load(data / "support_mask.npy", mmap_mode="r")))
    if not np.isfinite(cm.reference).all() or not np.isclose(np.linalg.norm(cm.reference), 1., atol=1e-12, rtol=0):
        raise ValueError("The production cost map did not return a unit reference")
    from dataset_path import associate, save_route
    from recording_endpoints import select_endpoint
    status('costs', 'Calibrating upward clearance from baseline beneath-camera ground')
    poses = associate(cm.points, cm.cost, cm.config['support_mask'], cm.reference, cm.config,
                      np.load(data / 'cameras.npy'), np.load(data / 'timestamps.npy'),
                      select_endpoint, frame_ids=None)
    calibration = camera_clearance(poses, radius)
    provenance['clearance_calibration'] = calibration
    report['clearance_calibration'] = calibration
    if calibration['reason'] != 'ok':
        cm.cost[:] = np.inf
        save_route(out, [], cm.points, np.inf)
    elif calibration['height'] > radius:
        cm.config['clearance_height'] = calibration['height']
        cm._index_overhead()
        for first in range(0, total, cm.config['max_neighbors']):
            rows = np.flatnonzero(np.isfinite(cm.cost[first:first + cm.config['max_neighbors']])) + first
            cm.cost[rows[cm._blocked_nodes(cm.points[rows])]] = np.inf
    save(data / "cost.npy", cm.cost)
    if not np.array_equal(np.load(data / 'reference.npy'), cm.reference):
        save(data / "reference.npy", cm.reference)
    provenance["settings"] = {key: value.item() if isinstance(value, np.generic) else value
                              for key, value in cm.config.items() if not key.endswith("_mask")}
    report["finite_cost_rows"] = sum(int(np.isfinite(cm.cost[i:i + 1_000_000]).sum()) for i in range(0, total, 1_000_000))
    sample = cm.cost[np.linspace(0, total - 1, min(total, 250_000), dtype=np.int64)] if total else np.empty(0)
    sample = sample[np.isfinite(sample)]
    state = 'ok' if calibration['reason'] == 'ok' else 'missing_clearance_calibration'
    report.update(status='pending' if state == 'ok' else state,
                  cost_range=np.percentile(sample, [5, 95]).tolist() if len(sample) else None)
    if state != 'ok':
        report['reason_counts'] = {state: count}
    write(data / "provenance.json", provenance)
    write(out / "report.json", report)
    write(receipt, dict(identity=cost_key, cost_source_sha256=cost_source, status=state,
                        clearance_height=calibration['height']))
    if state != 'ok':
        write(out / 'status.json', dict(stage='complete', status=state,
              message='No valid beneath-camera height calibration; context map only, no route'))
        return
    status("path", "Map, costs, camera poses and original point correspondence verified; selecting endpoints")


if __name__ == "__main__":
    main()
