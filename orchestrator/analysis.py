"""Independent steepness, cost and route artifact stages around numerical APIs."""
from pathlib import Path
import numpy as np
from orchestrator.artifacts import verify, publish, save_array, write_json, code_identity, digest
from orchestrator.semantics import POSITIVE, NEGATIVE
from path_planner.cost_map import build_cost_map, _settings
from path_planner.slope import estimate_slope


def load_snapshot(path):
    path = Path(path).resolve()
    receipt = verify(path, 'map')
    required = ('points', 'source_rgb', 'support_mask', 'obstacle_mask', 'frame_ids',
                'pixel_ids', 'cameras', 'source_frame_ids', 'timestamps', 'frame_offsets')
    arrays = {key: np.load(path / f'{key}.npy', mmap_mode='r') for key in required}
    n = len(arrays['points'])
    if arrays['points'].shape != (n, 3) or arrays['source_rgb'].shape != (n, 3):
        raise ValueError('Invalid map point/RGB shape')
    for key in ('support_mask', 'obstacle_mask'):
        if arrays[key].shape != (n,) or arrays[key].dtype.kind != 'b':
            raise ValueError('Map masks must be aligned Boolean arrays')
    for key in ('frame_ids', 'pixel_ids'):
        if arrays[key].shape != (n,) or arrays[key].dtype.kind not in 'iu':
            raise ValueError('Map source correspondence is invalid')
    k = len(arrays['cameras'])
    if arrays['cameras'].shape != (k, 3) or arrays['source_frame_ids'].shape != (k,) or arrays['timestamps'].shape != (k,):
        raise ValueError('Camera/source/timestamp correspondence is invalid')
    if np.any(np.diff(arrays['source_frame_ids']) <= 0) or not np.isfinite(arrays['timestamps']).all() or np.any(np.diff(arrays['timestamps']) < 0):
        raise ValueError('Invalid camera ordering or timestamps')
    if not np.isin(arrays['frame_ids'], arrays['source_frame_ids']).all():
        raise ValueError('Point frame IDs do not identify camera source frames')
    for key in POSITIVE + NEGATIVE:
        file = path / f'{key}.npy'
        if file.exists():
            arrays[key] = np.load(file, mmap_mode='r')
    return receipt, arrays


def reference_and_radius(receipt, data, reference=None, radius=None, source='base_support'):
    if source not in ('base_support', 'traversability'):
        raise ValueError('reference_source must be base_support or traversability')
    report = {'reason': 'supplied', 'source': source}
    if reference is None or radius is None:
        from path_planner.floor_reference import estimate_floor_reference
        # Preserve full-resolution erosion and the original five-frame scan
        # even when the exported planning cloud uses pixel sampling.
        metadata = receipt['metadata']
        if source == 'base_support' and metadata.get('support_dir'):
            from orchestrator.io_utils import frame_number, load_depth_map, load_extrinsic, load_intrinsic
            from orchestrator.sam_masks import build_sam_index, load_sam_mask, preprocess_mask
            from lingbot.geometry import depth_to_world_points
            from orchestrator.batch.dataset_reference import scan_reference
            predictions, support_dir = Path(metadata['predictions_dir']), Path(metadata['support_dir'])
            masks = build_sam_index(support_dir, metadata['video_name'], metadata['support_prompt'])
            def observations():
                filters = metadata['filters']
                for file in sorted(predictions.glob('frame_*.npz'), key=frame_number):
                    if file.name not in metadata['sources'] or digest(file) != metadata['sources'][file.name]:
                        raise ValueError('Reference geometry source changed after projection')
                    mask_path = masks.get(frame_number(file))
                    if mask_path is None:
                        raise ValueError('Missing reference support source')
                    expected = metadata.get('reference_support_sources', {}).get(mask_path.name)
                    if expected is None or digest(mask_path) != expected:
                        raise ValueError('Reference support source changed after projection')
                    with np.load(file, allow_pickle=False) as saved:
                        depth, conf = load_depth_map(saved['depth']), load_depth_map(saved['depth_conf'])
                        extrinsic, intrinsic = load_extrinsic(saved['extrinsic']), load_intrinsic(saved['intrinsic'])
                    points = depth_to_world_points(depth, intrinsic, extrinsic)
                    valid = np.isfinite(points).all(-1) & np.isfinite(depth) & (depth > 0) & np.isfinite(conf)
                    if filters['min_depth_conf'] is not None: valid &= conf > filters['min_depth_conf']
                    if filters['max_depth'] is not None: valid &= depth < filters['max_depth']
                    mask = preprocess_mask(load_sam_mask(mask_path), depth.shape, 14)
                    yield dict(id=file.name, points=points, floor_mask=mask & valid,
                               camera=-extrinsic[:,:3].T @ extrinsic[:,3])
            normal, report = scan_reference(observations())
            if normal is None:
                return None, None, report | {'source': source}
            if reference is None: reference = normal
            if radius is None: radius = report['slope_config']['radius']
            normal = np.asarray(reference, dtype=float)
            if normal.shape != (3,) or not np.isfinite(normal).all() or not np.any(normal) or not np.isfinite(radius) or radius <= 0:
                raise ValueError('Invalid reference/radius')
            return normal / np.linalg.norm(normal), float(radius), report | {'source': source}
        mask = data.get('base_support_mask') if source == 'base_support' else data['support_mask']
        if mask is None:
            return None, None, {'reason': 'missing_reference_support', 'source': source}
        frames = []
        camera_by_source = dict(zip(data['source_frame_ids'].tolist(), data['cameras']))
        for number, entry in enumerate(receipt['metadata']['frames']):
            h, w = entry['shape']
            first, last = data['frame_offsets'][number:number + 2]
            points, selected = np.full((h * w, 3), np.nan), np.zeros(h * w, dtype=bool)
            pixels = data['pixel_ids'][first:last]
            points[pixels] = data['points'][first:last]
            selected[pixels] = mask[first:last]
            frames.append(dict(id=entry['source_index'], points=points.reshape(h, w, 3),
                               floor_mask=selected.reshape(h, w), camera=camera_by_source[entry['source_index']]))
            if len(frames) < 5:
                continue
            normal, report = estimate_floor_reference(frames[-5:])
            if normal is not None:
                break
        else:
            normal, report = estimate_floor_reference(frames[-5:])
        if normal is None:
            return None, None, report | {'source': source}
        if reference is None:
            reference = normal
        if radius is None:
            spacing = [f['spacing'] for f in report['frames'] if f['reason'] == 'accepted']
            radius = float(20 * np.median(spacing))
    reference = np.asarray(reference, dtype=float)
    if reference.shape != (3,) or not np.isfinite(reference).all() or np.linalg.norm(reference) == 0:
        raise ValueError('Reference must be a finite nonzero vector')
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('Radius must be finite and positive')
    return reference / np.linalg.norm(reference), float(radius), report | {'source': source}


def steepness(snapshot, output, *, reference=None, radius=None, reference_source='base_support'):
    receipt, data = load_snapshot(snapshot)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    normal, radius, report = reference_and_radius(receipt, data, reference, radius, reference_source)
    metadata = dict(snapshot=str(Path(snapshot).resolve()), snapshot_identity=receipt['identity'],
                    reference_report=report, code_identity=code_identity(), radius=radius)
    if normal is None:
        return publish(output, 'steepness', metadata | {'status': 'missing_reference'})
    values, quality = estimate_slope(data['points'], normal, {'radius': radius},
                                     frame_ids=data['frame_ids'], return_quality=True)
    for key, array in dict(reference=normal, slope=values, quality=quality).items():
        save_array(output / f'{key}.npy', array)
    return publish(output, 'steepness', metadata | {'status': 'ok', 'reference': normal})


def costs(snapshot, output, *, slope_policy='enforce', slopes=None, reference=None, radius=None,
          reference_source='base_support', clearance_height=None, calibration_source='base_support', settings=None):
    from path_planner.recording_endpoints import select_endpoint
    from path_planner.routes import associate
    from orchestrator.batch.dataset_prepare import camera_clearance
    if slope_policy not in ('enforce', 'disabled'):
        raise ValueError('Unknown slope policy')
    receipt, data = load_snapshot(snapshot)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    slope_values, slope_identity = None, None
    if slope_policy == 'enforce':
        if slopes is None:
            raise ValueError('Slope-aware costs require a compatible steepness artifact')
        slope_receipt = verify(slopes, 'steepness')
        if slope_receipt['metadata']['snapshot_identity'] != receipt['identity'] or slope_receipt['metadata']['code_identity'] != code_identity():
            raise ValueError('Steepness belongs to a different snapshot or implementation')
        if slope_receipt['metadata']['status'] == 'ok':
            slope_values = np.load(Path(slopes) / 'slope.npy', mmap_mode='r')
            saved_reference = np.load(Path(slopes) / 'reference.npy')
            if reference is not None and not np.allclose(saved_reference, np.asarray(reference) / np.linalg.norm(reference), rtol=0, atol=1e-12):
                raise ValueError('Requested reference differs from saved steepness reference')
            if radius is not None and radius != slope_receipt['metadata']['radius']:
                raise ValueError('Requested radius differs from saved steepness radius')
            reference, radius = saved_reference, slope_receipt['metadata']['radius']
        slope_identity = slope_receipt['identity']
    normal, radius, ref_report = reference_and_radius(receipt, data, reference, radius, reference_source)
    metadata = dict(snapshot=str(Path(snapshot).resolve()), snapshot_identity=receipt['identity'],
                    slope_policy=slope_policy, slope_identity=slope_identity,
                    reference_report=ref_report, code_identity=code_identity())
    if normal is None or (slope_policy == 'enforce' and slope_values is None):
        save_array(output / 'cost.npy', np.full(len(data['points']), np.inf))
        return publish(output, 'costs', metadata | {'status': 'missing_reference'})
    config = dict(settings or {}) | {'radius': radius, 'slope_policy': slope_policy}
    if any(k.endswith('_mask') for k in config):
        raise ValueError('Semantic masks must come from the verified map, not settings')
    if calibration_source not in ('base_support', 'traversability'):
        raise ValueError('calibration_source must be base_support or traversability')
    if clearance_height is None:
        baseline_mask = data.get('base_support_mask') if calibration_source == 'base_support' else data['support_mask']
        calibration = dict(reason='missing_clearance_calibration', source=calibration_source)
        if baseline_mask is not None:
            baseline = build_cost_map(data['points'], slope_values, normal, config | {'support_mask': baseline_mask})
            poses = associate(baseline.points, baseline.cost, baseline_mask, baseline.reference,
                              baseline.config, data['cameras'], data['timestamps'], select_endpoint)
            calibration = camera_clearance(poses, radius) | {'source': calibration_source}
        if calibration['reason'] != 'ok':
            save_array(output / 'cost.npy', np.full(len(data['points']), np.inf))
            return publish(output, 'costs', metadata | {'status': 'missing_clearance_calibration', 'calibration': calibration})
        clearance_height = calibration['height']
    else:
        if not np.isfinite(clearance_height) or clearance_height <= 0:
            raise ValueError('Supplied clearance_height must be finite and positive')
        clearance_height = max(radius, clearance_height)
        calibration = dict(reason='ok', source='supplied', height=clearance_height)
    cm = build_cost_map(data['points'], slope_values, normal,
                        config | {'support_mask': data['support_mask'], 'obstacle_mask': data['obstacle_mask'],
                                  'clearance_height': clearance_height})
    save_array(output / 'cost.npy', cm.cost)
    save_array(output / 'reference.npy', cm.reference)
    settings = {k: v for k, v in cm.config.items() if not k.endswith('_mask')}
    metadata.update(status='ok', settings=settings, calibration=calibration,
                    finite_rows=int(np.isfinite(cm.cost).sum()))
    return publish(output, 'costs', metadata)


def route(cost_directory, output, *, start=None, goal=None, library=None):
    from path_planner.routes import restore_cost_map, associate, select_segment, exact_route, NativeSearch
    from path_planner.recording_endpoints import select_endpoint
    from path_planner.planner import find_path
    cost_receipt = verify(cost_directory, 'costs')
    metadata = cost_receipt['metadata']
    if metadata['code_identity'] != code_identity():
        raise ValueError('Saved costs use a different implementation')
    map_receipt, data = load_snapshot(metadata['snapshot'])
    if map_receipt['identity'] != metadata['snapshot_identity']:
        raise ValueError('Saved costs identify a different map snapshot')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status=metadata['status'], cost_identity=cost_receipt['identity'],
                  snapshot_identity=map_receipt['identity'], snapshot=metadata['snapshot'],
                  cost_directory=str(Path(cost_directory).resolve()), slope_policy=metadata['slope_policy'],
                  optimal=False, correspondence_verified=True, route_points=0)
    ids, total = np.empty(0, dtype=np.int64), np.inf
    if metadata['status'] == 'ok':
        values = np.load(Path(cost_directory) / 'cost.npy', mmap_mode='r')
        normal = np.load(Path(cost_directory) / 'reference.npy')
        config = _settings(metadata['settings'] | {'support_mask': data['support_mask'], 'obstacle_mask': data['obstacle_mask']})
        cm = restore_cost_map(data['points'].astype(float), values, data['support_mask'], normal, config,
                              __import__('path_planner.cost_map', fromlist=['CostMap']).CostMap)
        poses = []
        if start is None and goal is None:
            # Translate sparse source IDs to camera-sequence rows only for the
            # existing visible-ground association; published source IDs stay intact.
            frame_rows = np.searchsorted(data['source_frame_ids'], data['frame_ids'])
            poses = associate(cm.points, cm.cost, data['support_mask'], normal, config,
                              data['cameras'], data['timestamps'], select_endpoint, frame_rows)
            for pose in poses:
                pose['source_frame_index'] = int(data['source_frame_ids'][pose['frame_id']])
            state, first, last = select_segment(poses)
            report.update(status=state, start=first, goal=last)
            if state in ('ready', 'degenerate_route'):
                start, goal = first['id'], last['id']
        elif start is None or goal is None:
            raise ValueError('Supply both start and goal row IDs')
        if start is not None and goal is not None:
            # Validate endpoints before native indexing and bounds.
            n = len(cm.points)
            invalid = next((name for row, name in ((start, 'invalid_start'), (goal, 'invalid_goal'))
                            if type(row) is not int or not 0 <= row < n or not np.isfinite(cm.cost[row])), None)
            if invalid:
                report['status'] = invalid
            else:
                if library is not None:
                    result = exact_route(cm, start, goal, np.asarray(sorted({start, goal})), NativeSearch(library),
                                         find_path, lambda *a, **k: None, allow_partial=True)
                else:
                    from path_planner.planner import find_partial_path
                    result = find_partial_path(cm.points, cm, start, goal)
                ids, total = result.ids, result.total_cost
                report.update(status=result.status, optimal=True, goal_reached=bool(len(ids) and ids[-1] == goal),
                              start_id=start, goal_id=goal, reached_id=int(ids[-1]) if len(ids) else None,
                              all_edges_verified=True, optimality_scope='nearest_reachable_endpoint' if result.status == 'partial' else 'intended_goal')
        write_json(output / 'allposes.json', poses)
    np.savez(output / 'route.npz', ids=ids, positions=data['points'][ids], total_cost=total)
    report.update(route_points=len(ids), total_cost=total,
                  native_sha256=digest(library) if library else None, code_identity=code_identity())
    write_json(output / 'report.json', report)
    return publish(output, 'route', report)
