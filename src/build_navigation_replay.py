"""Replan a persistent research mission from cached observation prefixes.

No inference or physical execution is performed. Cached LingBot initialization
and the fixed research reference may use later observations; only admission to
the replay map is restricted to the current prefix. Original inputs are read
only, and every output lives in a new directory.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '4'

import numpy as np
from path_mapping.runner import geometry_fingerprint, validate_geometry
from pipeline_research_plan import research_costmap, validate_assumptions
from pipeline_common.planning import _basis, _empty, TRAVERSABLE
from pipeline_common.research_route import _local_point, segment_supported
from export_pipeline_video import _digest, load_npz_checked, MAX_GEOMETRY_ARCHIVE_BYTES

MAX_FRAMES = 256
IDENTITY_KEYS = ('geometry_fingerprint', 'input_fingerprint', 'processed_grid_id', 'map_frame', 'units')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def accepted_geometry(numeric, manifest):
    depth = numeric['depth'].reshape(numeric['world_points'].shape[:-1])
    conf = numeric['world_points_conf']
    threshold = manifest['settings']['min_confidence']
    valid = (np.isfinite(numeric['world_points']).all(-1) & np.isfinite(depth) & (depth > 0)
             & np.isfinite(conf) & (conf > 0) & (conf >= threshold))
    if len(manifest['transforms']) != len(valid):
        raise ValueError('Geometry transforms do not match the saved frame count')
    for i, transform in enumerate(manifest['transforms']):
        left, top, right, bottom = transform['pad_ltrb']
        if left: valid[i, :, :left] = False
        if top: valid[i, :top] = False
        if right: valid[i, :, -right:] = False
        if bottom: valid[i, -bottom:] = False
    return valid


def initial_support(arrays, metadata, target_native, factor, max_adjustment=.5):
    """Initialize on actual observed support, never connect unseen camera floor."""
    target = np.asarray(target_native) * factor @ arrays['projection_basis'].T
    cells = [tuple(map(int, c)) for c in np.argwhere(arrays['decision_state'] == TRAVERSABLE)]
    if not cells:
        raise ValueError('No full-footprint support for research initialization')
    points = np.asarray([_local_point(c, arrays) for c in cells])
    distances = np.linalg.norm(points[:, :2] - target[:2], axis=1)
    candidates = sorted(range(len(cells)), key=lambda j: (distances[j], cells[j]))
    for j in candidates:
        point = points[j]
        if (distances[j] <= max_adjustment
                and abs(point[2]-target[2]) <= metadata['profile']['max_step']+3*metadata['profile']['max_roughness']
                and segment_supported(point, point, arrays, metadata)):
            return point, {'initialization': 'first_observed_forward_floor_patch',
                           'source_target_native': np.asarray(target_native).tolist(),
                           'initial_cell': list(cells[j]),
                           'support_adjustment_assumed_m': float(distances[j]),
                           'camera_to_agent_horizontal_offset_assumed': True,
                           'unsupported_camera_to_agent_bridge_drawn': False}
    raise ValueError('Observed initialization patch has no nearby supported robot footprint')


def replay_reference(plan, manifest, recording_id, rotations):
    """Retain each recording's declared research reference without levelling it.

    A missing ground/scale reference disables planning. A camera-upright basis
    can still orient an unknown-only display; it is never supporting ground.
    """
    assumptions = validate_assumptions(plan['assumptions'])
    for key in ('level_reference', 'camera_height_reference'):
        if key not in assumptions:
            continue
        ref = assumptions[key]
        if ref['recording_id'] != recording_id or ref['geometry_fingerprint'] != manifest['geometry_fingerprint']:
            raise ValueError(key+' belongs to another recording or corrected geometry')
    value = plan.get('assumed_up_vector')
    fallback = value is None
    if fallback:
        if plan.get('status') != 'blocked_inputs':
            raise ValueError('A reference without assumed up must explicitly have blocked_inputs status')
        up = -np.asarray(rotations[0, 1], float)
        basis_source = 'assumed_upright_first_camera_for_unavailable_map'
    else:
        up = np.asarray(value, float)
        basis_source = plan.get('slope_reference', {}).get('source', 'explicit_reference_plan_assumed_up')
    basis = _basis(up)
    up = basis[2]
    plane = deepcopy(plan.get('ground_plane'))
    available = plan.get('status') != 'blocked_inputs' and not fallback and plane is not None
    if plane is not None:
        for key in ('normal_native', 'point_native'):
            vector = np.asarray(plane.get(key), float)
            if vector.shape != (3,) or not np.isfinite(vector).all():
                raise ValueError('Reference ground plane requires finite '+key)
        if np.linalg.norm(plane['normal_native']) < 1e-12:
            raise ValueError('Reference ground plane normal must be nonzero')
    calibration = plan.get('research_calibration', {}).get('camera_height_reference', {})
    policy = {
        'recording_id': recording_id, 'reference_plan_status': plan.get('status'),
        'research_inputs_available': available,
        'reference_unavailable_reason': None if available else (plan.get('reason') or 'No observed ground reference is available'),
        'basis_source': basis_source,
        'scale_source': calibration.get('method', 'explicit_reference_plan_scale_assumption'),
        'level_floor_assumed': 'level_reference' in assumptions,
        'gravity_measured': False, 'physical_calibration_verified': False,
    }
    return assumptions, up, basis, plane, policy


def unknown_costmap(basis, assumptions, requested_xy, reason):
    """Nonempty bounded display grid with zero observed or supporting cells."""
    arrays, metadata = _empty(reason)
    shape = (3, 3)
    for key, value in list(arrays.items()):
        if value.ndim != 2 or key == 'projection_basis':
            continue
        fill = False if value.dtype.kind == 'b' else (0 if value.dtype.kind in 'iu' else np.nan)
        arrays[key] = np.full(shape, fill, dtype=value.dtype)
    arrays['costs'].fill(np.inf)
    arrays['unknown_mask'].fill(True)
    resolution = assumptions['grid_resolution_m']
    arrays.update(origin=np.asarray(requested_xy, float)-1.5*resolution,
                  resolution=np.array([resolution], float), up=basis[2].copy(),
                  projection_basis=basis.copy())
    robot = assumptions['robot']
    metadata.update(shape=list(shape), origin=arrays['origin'].tolist(), resolution=resolution,
                    projection_basis=basis.tolist(), up=basis[2].tolist(),
                    units='assumed_metres', scale={'assumed': True, 'metres_per_native_unit': assumptions['metres_per_native_unit']},
                    profile=deepcopy(robot), footprint_radius_with_clearance=robot['footprint_radius']+robot['clearance'],
                    unknown_rule='blocked', research_illustration=True, safety_validated=False,
                    calibration_verified=False, observed_surface_cells=0, support_cells=0)
    return arrays, metadata


def waiting_route(reason):
    return {'status': 'awaiting_support', 'reason': reason, 'path_points_assumed_m': [],
            'display_path_points_assumed_m': [], 'agent_ground_point_assumed_m': None,
            'path_cells': [], 'swept_footprint_checked': False}


def build_replay(geometry_archive, reference_plan_path, sequence_path, manifest_path,
                 output, *, horizon_m=2., geometry_kind='original_saved_c2w', expected_recording_id=None):
    from pipeline_common.navigation_replay import plan_next_steps
    output = Path(output).resolve()
    paths = [Path(p).resolve() for p in (geometry_archive, reference_plan_path, sequence_path, manifest_path)]
    if output.exists() or any(output == p or output.is_relative_to(p.parent) for p in paths):
        raise ValueError('Choose a fresh output directory separate from every saved input')
    if not np.isfinite(horizon_m) or not 0 < horizon_m <= 2:
        raise ValueError('The displayed research horizon must be finite and in (0,2] metres')
    hashes = {str(p): sha256(p) for p in paths}
    plan, sequence, manifest = (_read(p) for p in paths[1:])
    if _digest({k: v for k, v in sequence.items() if k != 'manifest_digest'}) != sequence.get('manifest_digest'):
        raise ValueError('Saved sequence manifest digest does not match its contents')
    if (sequence.get('sequence_id') != plan['source']['recording_id']
            or sequence.get('manifest_digest') != manifest['sequence_digest']):
        raise ValueError('Saved sequence is not the one bound to the corrected geometry')
    for key in IDENTITY_KEYS:
        if manifest.get(key) != plan['source'].get(key):
            raise ValueError('Corrected plan and geometry manifest identity differs: '+key)
    recording_id = sequence['sequence_id']
    if expected_recording_id is not None and recording_id != expected_recording_id:
        raise ValueError('Batch recording_id differs from the exact saved sequence identity')
    if geometry_kind not in {'original_saved_c2w', 'corrected_w2c'}:
        raise ValueError('geometry_kind must explicitly identify original_saved_c2w or corrected_w2c')
    expected_archive = (plan['source']['derived_geometry']['source_archive_sha256']
                        if geometry_kind == 'original_saved_c2w' else manifest['archive_sha256'])
    if hashes[str(paths[0])] != expected_archive:
        raise ValueError('Geometry archive differs from declared corrected review provenance')
    provenance = manifest.get('derivation', {})
    pose_verified = provenance.get('pose_correction_verified', True) is not False
    if not pose_verified and provenance.get('method') != 'saved_pose_passthrough_unverified_display_only':
        raise ValueError('Unverified pose display requires the explicit unchanged-pose passthrough provenance')
    numeric = load_npz_checked(paths[0], max_bytes=MAX_GEOMETRY_ARCHIVE_BYTES)
    if geometry_kind == 'original_saved_c2w':
        # The derivation helper owns the verified correction versus untouched
        # display-only passthrough distinction; no second pose inversion.
        from derive_pose_corrected_run import numeric_for_derivation
        numeric = numeric_for_derivation(numeric, provenance)
    validate_geometry(numeric, allow_single_frame=True)
    fingerprint = geometry_fingerprint(numeric)
    if fingerprint != plan['source']['geometry_fingerprint'] or fingerprint != manifest['geometry_fingerprint']:
        raise ValueError('Recomputed corrected numeric geometry differs from the saved review')
    frames = sequence['frames']
    if not 1 <= len(frames) <= MAX_FRAMES or len(numeric['images']) != len(frames):
        raise ValueError('Require the exact complete saved sample count within 1-'+str(MAX_FRAMES))
    ids = [f['frame_id'] for f in frames]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Saved frame identities must be unique nonempty strings')
    timestamps = [f['timestamp_ns'] for f in frames]
    if any(type(t) is not int or t < 0 for t in timestamps) or any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError('Saved sample timestamps must strictly increase')
    valid = accepted_geometry(numeric, manifest)
    rotations = numeric['extrinsic'][:, :, :3]
    assumptions, up, basis, plane, reference = replay_reference(plan, manifest, recording_id, rotations)
    if not pose_verified:
        reference.update(research_inputs_available=False,
                         reference_unavailable_reason='Pose convention is unverified; awaiting independent pose evidence')
    factor = assumptions['metres_per_native_unit']
    cameras = -np.einsum('sji,sj->si', rotations, numeric['extrinsic'][:, :, 3])
    camera_local = cameras * factor @ basis.T
    forward = rotations[0, 2] @ basis.T
    forward = forward[:2] / np.linalg.norm(forward[:2])
    voxel_size = plan['source']['voxel_size_native']
    voxel_origin = np.asarray(plan['source']['voxel_origin_native'])
    if (isinstance(voxel_size, bool) or not isinstance(voxel_size, (int, float))
            or not np.isfinite(voxel_size) or voxel_size <= 0
            or voxel_origin.shape != (3,) or not np.isfinite(voxel_origin).all()):
        raise ValueError('Reference voxel origin/size must be finite and explicit')
    output.mkdir(parents=True, exist_ok=False)
    (output/'costmaps').mkdir()
    replay = {'schema_version': 1, 'artifact_kind': 'research_navigation_replay',
        'research_illustration': True, 'planning_basis': 'geometry_only', 'semantic_guidance': False,
        'frame_scope': 'observation_prefix_map_replay', 'perception_scope': 'cached_geometry_replay',
        'display_horizon_assumed_m': float(horizon_m), 'assumptions': assumptions,
        'assumed_up_vector': up.tolist(), 'projection_basis': basis.tolist(),
        'path_coordinate_units': 'original_native_map_units',
        'source': {**{k: plan['source'][k] for k in (*IDENTITY_KEYS, 'archive_sha256')},
                   'research_plan_sha256': hashes[str(paths[1])],
                   'local_input_sha256': hashes,
                   'geometry_input_kind': geometry_kind, 'recording_id': recording_id,
                   'pose_correction_verified': pose_verified,
                   'voxel_size_native': float(voxel_size), 'voxel_origin_native': voxel_origin.tolist(),
                   'corrected_numeric_method': (provenance.get('method', 'decode_saved_c2w_invert_to_w2c_unproject_saved_depth')
                         if geometry_kind == 'original_saved_c2w' else 'admit_saved_corrected_w2c_geometry'),
                   **reference,
                   'models_executed': False},
        'mission': {'id': recording_id+'_persistent_forward_corridor_v1',
                    'policy': 'persistent_forward_corridor', 'state': 'active', 'completed': False,
                    'goal_policy': 'reachable_observed_frontier_along_fixed_corridor_direction',
                    'goal_is_video_end': False,
                    'rolling_goal_is_mission_completion': False,
                    'agent_motion': 'simulated_initial_support_plus_recorded_camera_horizontal_displacement',
                    'origin_native': cameras[0].tolist(),
                    'origin_kind': 'camera_reference_until_supported_initialization',
                    'forward_native': (np.r_[forward, 0.] @ basis).tolist(),
                    'initialized_on_observed_support': False},
        'limitations': ['Cached reconstruction is replayed, not recomputed online; LingBot bootstrap may use later initialization frames.',
                        'Up/scale are fixed offline research assumptions from the corrected review.',
                        'The simulated robot initializes only when an observed forward patch supports its footprint; until then no ground position or route is asserted.',
                        'Robot movement follows recorded displacement by assumption; no controller or physical execution is simulated.',
                        'Unknown and blocked cells cannot provide continuation; EOF supplies no new observations.'],
        'frames': []}
    raw_prefix = []
    voxel_indices = np.empty((0, 3), np.int64)
    all_metadata = []
    initial = None
    initialization_index = None
    start_time = time.monotonic()
    for i, frame in enumerate(frames):
        raw = numeric['world_points'][i][valid[i]]
        raw_prefix.append(raw)
        indices = np.unique(np.floor((raw-voxel_origin)/voxel_size).astype(np.int64), axis=0)
        voxel_indices = np.unique(np.concatenate((voxel_indices, indices)), axis=0)
        centers = voxel_origin + (voxel_indices + .5) * voxel_size
        prefix_count = sum(len(p) for p in raw_prefix)
        requested = (camera_local[i, :2].copy() if initial is None else
                     initial[:2]+camera_local[i, :2]-camera_local[initialization_index, :2])
        map_reason = reference['reference_unavailable_reason']
        if reference['research_inputs_available'] and prefix_count:
            try:
                arrays, metadata = research_costmap(np.concatenate(raw_prefix), plane, assumptions,
                    up_native=up, voxels_native={'centers': centers}, voxel_size_native=voxel_size)
            except ValueError as error:
                if 'grid size bound' not in str(error):
                    raise
                map_reason = str(error)
                arrays, metadata = unknown_costmap(basis, assumptions, requested, map_reason)
            if metadata.get('availability') != 'available':
                map_reason = metadata.get('reason') or 'Research prefix costmap is unavailable'
                arrays, metadata = unknown_costmap(basis, assumptions, requested, map_reason)
        else:
            map_reason = map_reason or 'No accepted surface observations in this prefix'
            arrays, metadata = unknown_costmap(basis, assumptions, requested, map_reason)
        initialization_reason = None
        if initial is None and metadata.get('availability') == 'available':
            h, w = valid[i].shape
            x, y = w//2, min(h-2, int(.9*h))
            patch = numeric['world_points'][i, y-1:y+2, x-1:x+2]
            admitted = valid[i, y-1:y+2, x-1:x+2]
            if np.count_nonzero(admitted) < 3:
                initialization_reason = 'Current observed forward patch lacks depth support'
            else:
                try:
                    initial, initialization = initial_support(arrays, metadata, np.median(patch[admitted], axis=0), factor)
                except ValueError as error:
                    initialization_reason = str(error)
                else:
                    initialization_index = i
                    requested = initial[:2].copy()
                    replay['mission'].update(initialization=initialization,
                        initialization_observed_index=i, initialized_on_observed_support=True,
                        origin_kind='observed_supported_initial_ground',
                        origin_native=(initial @ basis / factor).tolist(),
                        initial_agent_ground_native=(initial @ basis / factor).tolist())
        if initial is None or metadata.get('availability') != 'available':
            route = waiting_route(map_reason or initialization_reason or 'No supported initialization is available')
            if not pose_verified:
                route['status'] = 'pose_unverified'
        else:
            route = plan_next_steps(arrays, metadata, requested, initial[:2], forward,
                                    horizon_m=horizon_m)
        path = np.asarray(route.get('path_points_assumed_m', []), float).reshape(-1, 3)/factor
        display = np.asarray(route.get('display_path_points_assumed_m', []), float).reshape(-1, 3)/factor
        ground = route.get('agent_ground_point_assumed_m')
        relative = f'costmaps/{i:04d}.npz'
        np.savez_compressed(output/relative, **arrays)
        row = {'frame_index': i, 'frame_id': frame['frame_id'], 'timestamp_ns': frame['timestamp_ns'],
            'source_frame_index': frame['timestamp_provenance']['source_frame_index'],
            'observed_through_index': i, 'status': route['status'], 'reason': route['reason'],
            'path_points': path.tolist(), 'display_path_points': display.tolist(),
            'agent_requested_xy_assumed_m': requested.tolist(),
            'agent_ground_point_native': None if ground is None else (np.asarray(ground)/factor).tolist(),
            'camera_center_native': cameras[i].tolist(),
            'costmap_file': relative, 'costmap_sha256': sha256(output/relative),
            'path_cells': [list(map(int, cell)) for cell in route.get('path_cells', [])],
            'navigation_state': {'mission_state': 'active', 'mission_completed': False,
                                  'execution_state': ('ready' if route['status'] == 'ok' else
                                       'pose_unverified' if route['status'] == 'pose_unverified' else
                                       'awaiting_observations' if route['status'] == 'awaiting_observation' else 'awaiting_support')},
            'diagnostics': {'accepted_prefix_points': prefix_count,
                            'occupied_prefix_voxels': len(centers), 'grid_shape': list(arrays['costs'].shape),
                            'research_inputs_available': reference['research_inputs_available'],
                            'costmap_availability': metadata.get('availability'),
                            'initialized_on_observed_support': initial is not None,
                            'initialization_observed_index': initialization_index,
                            'display_length_assumed_m': float(np.linalg.norm(np.diff(display, axis=0), axis=1).sum()*factor),
                            'planned_length_assumed_m': float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()*factor),
                            'goal_is_temporary_frontier': True, 'future_frames_admitted_to_map': 0,
                            'swept_footprint_checked': route.get('swept_footprint_checked', False)}}
        replay['frames'].append(row)
        all_metadata.append({'frame_index': i, 'frame_id': frame['frame_id'], 'metadata': metadata})
        print(json.dumps({'frame': i+1, 'frames': len(frames), 'status': route['status'],
                          'display_m': row['diagnostics']['display_length_assumed_m'],
                          'internal_plan_m': row['diagnostics']['planned_length_assumed_m'],
                          'elapsed_seconds': round(time.monotonic()-start_time, 1)}), flush=True)
    last = replay['frames'][-1]
    replay['terminal_state'] = {'mission_state': 'active', 'execution_state': 'awaiting_observations',
        'mission_completed': False, 'reason': 'recording_ended',
        'last_observed_timestamp_ns': last['timestamp_ns'],
        'retained_display_path_points': last['display_path_points'],
        'retained_internal_path_points': last['path_points'],
        'new_observations_after_end': 0,
        'end_of_recording_is_goal_arrival': False}
    _write(output/'prefix_costmap_metadata.json', all_metadata)
    _write(output/'replanning_replay.json', replay)
    after = {str(p): sha256(p) for p in paths}
    if after != hashes:
        raise RuntimeError('A saved input changed during replay construction')
    _write(output/'replay_build_receipt.json', {'status': 'complete', 'frames': len(frames),
           'original_inputs_unchanged': True, 'input_hashes': hashes,
           'geometry_fingerprint': fingerprint, 'models_executed': False,
           'mission_completed': False, 'elapsed_seconds': time.monotonic()-start_time,
           'replay_sha256': sha256(output/'replanning_replay.json')})
    return replay


def build_batch(config_path, output):
    """Build independently bound recordings; failures retain explicit receipts.

    Schema 1: {artifact_kind: research_navigation_replay_batch_config,
    research_illustration: true, recordings: [{recording_id, geometry,
    reference_plan, sequence, geometry_manifest, geometry_kind?}]}. Paths are
    relative to the config file. No research assumption is shared or invented.
    """
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config_hash = sha256(config_path)
    config = _read(config_path)
    if (config.get('schema_version') != 1
            or config.get('artifact_kind') != 'research_navigation_replay_batch_config'
            or config.get('research_illustration') is not True):
        raise ValueError('Explicit schema-1 research_navigation_replay_batch_config required')
    records = config.get('recordings')
    if not isinstance(records, list) or not 1 <= len(records) <= 256:
        raise ValueError('Batch requires 1-256 explicit recording entries')
    ids = [r.get('recording_id') for r in records]
    if any(not isinstance(i, str) or not i or Path(i).name != i or i in {'.', '..'} for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Batch recording IDs must be unique safe directory names')
    admitted = []
    for record in records:
        paths = []
        for key in ('geometry', 'reference_plan', 'sequence', 'geometry_manifest'):
            value = record.get(key)
            if not isinstance(value, str) or not value:
                raise ValueError('Every recording requires explicit '+key+' path')
            path = (config_path.parent / value).resolve()
            if not path.is_file() or output.is_relative_to(path.parent):
                raise ValueError('Batch inputs must exist outside the fresh output directory')
            paths.append(path)
        admitted.append((record, paths))
    if output.exists() or config_path.is_relative_to(output):
        raise ValueError('Batch output must be fresh and separate from the config')
    output.mkdir(parents=True)
    rows = []
    for record, paths in admitted:
        destination = output / record['recording_id']
        try:
            replay = build_replay(*paths, destination, expected_recording_id=record['recording_id'],
                                  geometry_kind=record.get('geometry_kind', 'original_saved_c2w'))
        except (ValueError, OSError, KeyError) as error:
            row = {'recording_id': record['recording_id'], 'status': 'failed',
                   'error': type(error).__name__+': '+str(error), 'output': str(destination)}
        else:
            statuses = {state: sum(f['status'] == state for f in replay['frames'])
                        for state in ('ok', 'awaiting_support', 'awaiting_observation', 'pose_unverified')}
            row = {'recording_id': record['recording_id'], 'status': 'complete',
                   'output': str(destination), 'frames': len(replay['frames']),
                   'frame_status_counts': statuses, 'mission_state': 'active', 'mission_completed': False}
        rows.append(row)
        _write(output/'batch_receipt.json', {'schema_version': 1, 'artifact_kind': 'research_navigation_replay_batch_receipt',
               'status': 'running', 'config_sha256': config_hash, 'models_executed': False, 'recordings': rows})
    if sha256(config_path) != config_hash:
        raise RuntimeError('Batch config changed during replay construction')
    receipt = {'schema_version': 1, 'artifact_kind': 'research_navigation_replay_batch_receipt',
               'status': 'complete' if all(r['status'] == 'complete' for r in rows) else 'completed_with_failures',
               'config_sha256': config_hash, 'models_executed': False, 'recordings': rows}
    _write(output/'batch_receipt.json', receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-config', type=Path)
    parser.add_argument('--geometry', type=Path)
    parser.add_argument('--reference-plan', type=Path)
    parser.add_argument('--sequence', type=Path)
    parser.add_argument('--geometry-manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--horizon-m', type=float, default=2.)
    parser.add_argument('--geometry-kind', choices=('original_saved_c2w', 'corrected_w2c'), default='original_saved_c2w')
    args = parser.parse_args(argv)
    inputs = (args.geometry, args.reference_plan, args.sequence, args.geometry_manifest)
    if args.batch_config is not None:
        if any(p is not None for p in inputs) or args.horizon_m != 2. or args.geometry_kind != 'original_saved_c2w':
            parser.error('--batch-config cannot be combined with per-recording inputs/options')
        return build_batch(args.batch_config, args.output)
    if any(p is None for p in inputs):
        parser.error('A single recording requires --geometry, --reference-plan, --sequence and --geometry-manifest')
    build_replay(args.geometry, args.reference_plan, args.sequence, args.geometry_manifest,
                 args.output, horizon_m=args.horizon_m, geometry_kind=args.geometry_kind)


if __name__ == '__main__':
    main()
