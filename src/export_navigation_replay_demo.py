"""CPU demonstration movie of an observation-prefix navigation replay.

The renderer loads cached RGB/depth/masks and already planned per-frame routes.
It does not execute perception, planning, a robot controller, or server work.
The source video is decoded only at the exact saved sample frame indices.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw

from export_pipeline_video import (
    MAX_GEOMETRY_ARCHIVE_BYTES, _digest, _font, _read, _rgb_hash, _safe,
    load_npz_checked, project_mask_to_source, sampled_timeline, sha256,
)
from path_mapping.runner import GEOMETRY_KEYS, geometry_fingerprint
from pipeline_common.navigation_replay import clip_route_horizon
from pipeline_common.route_projection import project_route_to_source

FPS = 20
END_HOLD_SECONDS = 2
HORIZON_M = 2.
SIZE = (1080, 1920)
MAP_BOX = (752, 28, 1052, 328)
NAVY = (25, 47, 68)
TEAL = (18, 170, 158)
PURPLE = (161, 74, 221)
GRAY = (96, 112, 128)
LIGHT = (235, 241, 246)
WHITE = (255, 255, 255)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def route_points(value):
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return np.empty((0, 3), float)
    if array.ndim != 2 or array.shape[1] != 3 or len(array) > 4096 or not np.isfinite(array).all():
        raise ValueError('Display route must be finite native XYZ with at most 4096 points')
    return array


def circle_clipped_routes(path, agent, basis, metres_per_native_unit, horizon=HORIZON_M):
    """Clip every ordered segment to the agent's horizontal metric circle.

    Disconnected inside pieces remain separate, so an outside excursion cannot
    become an invented line between two re-entry points.
    """
    path = route_points(path)
    agent = np.asarray(agent, float)
    basis = np.asarray(basis, float)
    if agent.shape != (3,) or basis.shape != (3, 3) or not np.isfinite(agent).all():
        raise ValueError('Agent and projection basis must be finite XYZ/3x3')
    if not np.isfinite(basis).all() or not np.allclose(basis @ basis.T, np.eye(3), atol=1e-6):
        raise ValueError('Projection basis must be orthonormal')
    if not math.isfinite(metres_per_native_unit) or metres_per_native_unit <= 0 or not math.isfinite(horizon) or horizon <= 0:
        raise ValueError('Scale and horizon must be positive')
    xy = (path - agent) @ basis[:2].T * metres_per_native_unit
    pieces = []
    for p0, p1, q0, q1 in zip(path[:-1], path[1:], xy[:-1], xy[1:]):
        delta = q1 - q0
        a = float(delta @ delta)
        b = float(2 * q0 @ delta)
        c = float(q0 @ q0 - horizon ** 2)
        if a < 1e-20:
            if c > 1e-10:
                continue
            lo, hi = 0., 1.
        else:
            discriminant = b * b - 4 * a * c
            if discriminant < 0:
                continue
            root = math.sqrt(max(0., discriminant))
            lo = max(0., (-b - root) / (2 * a))
            hi = min(1., (-b + root) / (2 * a))
            if hi - lo <= 1e-12:
                continue
        start = p0 + lo * (p1 - p0)
        end = p0 + hi * (p1 - p0)
        if pieces and np.linalg.norm(pieces[-1][-1] - start) < 1e-9:
            pieces[-1].append(end)
        else:
            pieces.append([start, end])
    return [np.asarray(piece) for piece in pieces]


def _agent(row):
    if row.get('agent_ground_point_native') is None:
        return None
    value = np.asarray(row.get('agent_ground_point_native'), float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError('Every replay row requires a finite simulated ground agent point')
    return value


def admitted_display_route(row, scale):
    """Require the exact ordered first two metres of the already saved plan."""
    full = route_points(row.get('path_points', []))
    display = route_points(row.get('display_path_points', []))
    expected = clip_route_horizon(full * scale, HORIZON_M) / scale
    if display.shape != expected.shape or not np.allclose(display, expected, rtol=0, atol=1e-7):
        raise ValueError('Display route must be the ordered two-metre arc-length prefix of the saved plan')
    agent = _agent(row)
    if len(display) and (agent is None or not np.allclose(display[0], agent, rtol=0, atol=1e-7)):
        raise ValueError('A nonempty route must start at the finite supported simulated agent ground point')
    requested = np.asarray(row.get('agent_requested_xy_assumed_m'), float)
    if requested.shape != (2,) or not np.isfinite(requested).all():
        raise ValueError('Every replay row requires finite requested simulated-agent XY')
    return display, agent, requested


def decode_saved_samples(video_path, source_frames):
    """Sequential decode preserves each original decoder-reported frame index."""
    indices = [f['timestamp_provenance']['source_frame_index'] for f in source_frames]
    if any(type(i) is not int or i < 0 for i in indices) or any(b <= a for a, b in zip(indices, indices[1:])):
        raise ValueError('Source video frame indices must be increasing nonnegative integers')
    wanted = dict(zip(indices, source_frames))
    decoded = {}
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError('Cannot decode the original source video')
        for i in range(indices[-1] + 1):
            ok, bgr = capture.read()
            if not ok:
                raise ValueError('Source video ends before its last saved sample')
            if i not in wanted:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            record = wanted[i]
            if rgb.shape[:2] != (record['height'], record['width']) or _rgb_hash(rgb) != record['decoded_rgb_sha256']:
                raise ValueError(f'Decoded source video sample differs from saved pixels: {record["frame_id"]}')
            decoded[record['frame_id']] = rgb
    finally:
        capture.release()
    return decoded


def admit_reference_and_mission(replay, manifest, reference, actual_hashes, required_local_paths):
    """Bind rendering assumptions and lifetime to the immutable reference plan."""
    for key in ('geometry_fingerprint', 'input_fingerprint', 'processed_grid_id', 'map_frame', 'units'):
        if replay.get('source', {}).get(key) != manifest.get(key) or reference.get('source', {}).get(key) != manifest.get(key):
            raise ValueError('Replay/reference/corrected geometry identity mismatch: ' + key)
    if replay.get('assumptions') != reference.get('assumptions'):
        raise ValueError('Replay assumptions/up differ from the exact reference plan')
    if not uses_display_up_fallback(replay, reference):
        replay_up = np.asarray(replay.get('assumed_up_vector'), float)
        reference_up = np.asarray(reference.get('assumed_up_vector'), float)
        if (replay_up.shape != (3,) or reference_up.shape != (3,)
                or not np.isfinite(replay_up).all() or not np.isfinite(reference_up).all()
                or not np.allclose(replay_up, reference_up, rtol=0, atol=1e-12)):
            raise ValueError('Replay assumed up differs from the exact reference plan direction')
    declared = replay.get('source', {}).get('local_input_sha256')
    if not isinstance(declared, dict):
        raise ValueError('Replay requires its original local input SHA256 mapping')
    declared = {str(Path(path).resolve()): digest for path, digest in declared.items()}
    for path in required_local_paths:
        path = str(Path(path).resolve())
        if declared.get(path) != actual_hashes.get(path):
            raise ValueError('Replay local input SHA256 differs or is missing: ' + path)
    mission = replay.get('mission', {})
    if mission.get('policy') != 'persistent_forward_corridor' or not isinstance(mission.get('id'), str) or not mission['id'].strip() or len(mission['id']) > 128 or mission.get('state') != 'active' or mission.get('completed') is not False:
        raise ValueError('Replay requires a named active persistent forward mission')
    for key in ('origin_native', 'forward_native'):
        value = np.asarray(mission.get(key), float)
        if value.shape != (3,) or not np.isfinite(value).all() or (key == 'forward_native' and np.linalg.norm(value) < 1e-12):
            raise ValueError('Mission requires a finite native origin and nonzero forward direction')


def uses_display_up_fallback(replay, reference):
    """An unavailable map may use first-camera upright solely to frame display."""
    marker = 'assumed_upright_first_camera_for_unavailable_map'
    if replay.get('source', {}).get('basis_source') != marker:
        return False
    up = np.asarray(reference.get('assumed_up_vector'), float)
    has_reference_up = up.shape == (3,) and np.isfinite(up).all() and np.linalg.norm(up) > 1e-12
    if reference.get('status') != 'blocked_inputs' or has_reference_up:
        raise ValueError('Display-only upright fallback requires unavailable reference map/up')
    if any(len(route_points(row.get(key, []))) for row in replay.get('frames', [])
           for key in ('path_points', 'display_path_points')):
        raise ValueError('Display-only upright fallback cannot admit any route')
    return True


def admit_row_state(row):
    expected = {'ok': 'ready', 'awaiting_support': 'awaiting_support', 'awaiting_observation': 'awaiting_observations',
                'pose_unverified': 'pose_unverified'}
    state = row.get('navigation_state', {})
    if row.get('status') not in expected or state.get('execution_state') != expected[row['status']]:
        raise ValueError('Replay row status and navigation execution state disagree')
    if state.get('mission_state') != 'active' or state.get('mission_completed') is not False:
        raise ValueError('Every displayed replay frame must retain the active forward mission')
    full = route_points(row.get('path_points', []))
    display = route_points(row.get('display_path_points', []))
    if row['status'] == 'ok' and len(display) < 2:
        raise ValueError('A ready replay row requires an actual local display route')
    if row['status'] != 'ok' and (len(full) or len(display)):
        raise ValueError('Waiting replay rows cannot invent guidance')
    if row['status'] == 'pose_unverified' and row.get('agent_ground_point_native') is not None:
        raise ValueError('Pose-unverified display cannot assert a supported ground agent')


def admit_numeric_derivation(reference, derivation):
    binding = reference.get('source', {}).get('derived_geometry')
    if not isinstance(binding, dict):
        raise ValueError('Reference plan requires its declared numeric derivation binding')
    for key in ('method', 'source_archive_sha256', 'source_geometry_fingerprint'):
        if not derivation.get(key) or binding.get(key) != derivation[key]:
            raise ValueError('Reference-plan numeric derivation differs from geometry manifest: ' + key)
    for key in ('pose_correction_verified', 'planning_admitted'):
        if binding.get(key, True) is not derivation.get(key, True):
            raise ValueError('Reference-plan derivation policy differs from geometry manifest: ' + key)


def metric_scale_available(reference, pose_verified, *, display_up_fallback=False):
    availability = reference.get('research_calibration', {}).get('camera_height_reference', {}).get('availability')
    return pose_verified is True and not display_up_fallback and availability == 'applied'


def semantic_mask_path(semantics_path, relative, semantic_root=None):
    """Honor run-relative paths in standard semantics/frames.jsonl exports.

    Portable flattened metadata uses paths relative to the JSONL directory.
    Other layouts can state their root explicitly; no existence-based fallback
    selects a different mask when a declared path is missing.
    """
    root = Path(semantic_root).resolve() if semantic_root is not None else Path(semantics_path).resolve().parent
    parts = Path(relative).parts
    if semantic_root is None and root.name == 'semantics' and parts[:1] == ('semantics',):
        root = root.parent
    return _safe(root, relative)


def load_demo(replay_path, geometry_path, video_path, sequence_path, manifest_path, semantics_path, reference_plan_path, *, include_numeric=False, semantic_root=None):
    paths = list(map(lambda p: Path(p).resolve(), [replay_path, geometry_path, video_path, sequence_path, manifest_path, semantics_path, reference_plan_path]))
    replay_path, geometry_path, video_path, sequence_path, manifest_path, semantics_path, reference_plan_path = paths
    hashes = {str(path): sha256(path) for path in paths}
    replay, sequence, manifest = map(_read, [replay_path, sequence_path, manifest_path])
    reference = _read(reference_plan_path)
    if replay.get('artifact_kind') != 'research_navigation_replay' or replay.get('schema_version') != 1 or replay.get('research_illustration') is not True:
        raise ValueError('Expected schema 1 research_navigation_replay')
    if replay.get('frame_scope') != 'observation_prefix_map_replay' or replay.get('perception_scope') != 'cached_geometry_replay':
        raise ValueError('Demo requires the explicit cached-geometry observation-prefix scope')
    if replay.get('display_horizon_assumed_m') != HORIZON_M:
        raise ValueError('This demonstration requires the declared 2 m display horizon')
    if replay.get('source', {}).get('research_plan_sha256') != hashes[str(reference_plan_path)]:
        raise ValueError('Replay reference-plan SHA256 differs from the real local plan file')
    admit_reference_and_mission(replay, manifest, reference, hashes,
                                [geometry_path, sequence_path, manifest_path, reference_plan_path])
    if _digest({k: v for k, v in sequence.items() if k != 'manifest_digest'}) != sequence.get('manifest_digest'):
        raise ValueError('Sequence manifest digest differs')
    if manifest.get('sequence_digest') != sequence['manifest_digest']:
        raise ValueError('Geometry is registered to another sequence')
    if hashes[str(video_path)] != sequence.get('input_provenance', {}).get('encoded_video_sha256'):
        raise ValueError('Original source video SHA256 differs from the saved sequence')
    derivation = manifest.get('derivation', {})
    admit_numeric_derivation(reference, derivation)
    if hashes[str(geometry_path)] != derivation.get('source_archive_sha256'):
        raise ValueError('Original geometry archive SHA256 differs from the corrected manifest')
    geometry = load_npz_checked(geometry_path, max_bytes=MAX_GEOMETRY_ARCHIVE_BYTES, names=GEOMETRY_KEYS)
    if geometry_fingerprint(geometry) != derivation.get('source_geometry_fingerprint'):
        raise ValueError('Original numeric geometry fingerprint differs')
    from derive_pose_corrected_run import numeric_for_derivation
    corrected = numeric_for_derivation(geometry, derivation)
    if geometry_fingerprint(corrected) != manifest['geometry_fingerprint']:
        raise ValueError('Declared numeric derivation differs from its saved geometry identity')
    for key in ('images', 'depth', 'intrinsic', 'world_points_conf'):
        if not np.array_equal(corrected[key], geometry[key], equal_nan=True):
            raise ValueError('Pose correction changed immutable saved data: ' + key)
    sources = sequence['frames']
    rows = replay.get('frames', [])
    if len(rows) != len(sources) or len(rows) != len(corrected['images']) or not 1 <= len(rows) <= 256:
        raise ValueError('Replay, video sequence and geometry frame counts differ')
    transforms = manifest['input_identity']['transforms']
    if len(transforms) != len(rows):
        raise ValueError('One exact source affine is required for every frame')
    scale = replay['assumptions']['metres_per_native_unit']
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('Replay assumed metric scale must be positive')
    basis = np.asarray(replay['projection_basis'], float)
    circle_clipped_routes([], [0, 0, 0], basis, scale)
    if not np.allclose(basis[2], replay['assumed_up_vector'], atol=1e-8):
        raise ValueError('Replay ground basis disagrees with its assumed up vector')
    if uses_display_up_fallback(replay, reference):
        expected_up = -corrected['extrinsic'][0, 1, :3].astype(float)
        expected_up /= np.linalg.norm(expected_up)
        if not np.allclose(replay['assumed_up_vector'], expected_up, rtol=0, atol=1e-8):
            raise ValueError('Display-only upright fallback differs from the saved first camera pose')
    pose_verified = derivation.get('pose_correction_verified', True)
    if pose_verified is not True and (pose_verified is not False or derivation.get('method') != 'saved_pose_passthrough_unverified_display_only'):
        raise ValueError('Unverified pose requires the explicit original-pose display-only derivation')
    scale_available = metric_scale_available(reference, pose_verified,
                                            display_up_fallback=uses_display_up_fallback(replay, reference))
    if not scale_available and any(len(route_points(row.get(key, []))) for row in rows
                                   for key in ('path_points', 'display_path_points')):
        raise ValueError('Unavailable height-reference scale cannot admit a metric display route')
    voxel_size = float(reference['source']['voxel_size_native'])
    if not math.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError('Reference plan requires a finite positive native voxel size')
    semantic_rows = _read(semantics_path, rows=True)
    semantic_by_id = {row['frame_id']: row for row in semantic_rows}
    if len(semantic_rows) != len(rows) or len(semantic_by_id) != len(rows):
        raise ValueError('Semantic mask frame identities/count differ')
    metadata_path = replay_path.parent / 'prefix_costmap_metadata.json'
    metadata = _read(metadata_path)
    hashes[str(metadata_path)] = sha256(metadata_path)
    if not isinstance(metadata, list) or len(metadata) != len(rows):
        raise ValueError('One prefix costmap metadata record is required per frame')
    video = decode_saved_samples(video_path, sources)
    frames = []
    minimum = float(manifest['settings']['min_confidence'])
    for i, (row, source, transform, meta) in enumerate(zip(rows, sources, transforms, metadata)):
        if row.get('frame_index') != i or row.get('observed_through_index') != i or row.get('frame_id') != source['frame_id'] or row.get('timestamp_ns') != source['timestamp_ns']:
            raise ValueError('Replay frame identity/timestamp/prefix does not match its saved sample')
        admit_row_state(row)
        if (row['status'] == 'pose_unverified') != (pose_verified is False):
            raise ValueError('Replay row pose state disagrees with immutable numeric derivation')
        if (transform.get('source_shape') != list(video[source['frame_id']].shape[:2])
                or transform.get('processed_shape') != list(corrected['images'].shape[1:3])):
            raise ValueError('Saved source affine shapes differ from actual source video/processed RGB')
        semantic = semantic_by_id[source['frame_id']]
        for key, expected in [('geometry_fingerprint', manifest['geometry_fingerprint']), ('processed_grid_id', manifest['processed_grid_id']), ('timestamp_ns', source['timestamp_ns']), ('sequence_id', sequence['sequence_id']), ('decoded_rgb_sha256', _rgb_hash(corrected['images'][i]))]:
            if semantic.get(key) != expected:
                raise ValueError('Semantic/saved processed-grid join differs: ' + key)
        if semantic['adapter_provenance']['input']['source_to_processed'] != transform:
            raise ValueError('Semantic source affine differs from saved geometry grid')
        floor = np.zeros(corrected['images'].shape[1:3], bool)
        for query in semantic['queries']:
            if query['role'] != 'candidate_surface':
                continue
            for instance in query['instances']:
                mask_path = semantic_mask_path(semantics_path, instance['mask_path'], semantic_root)
                hashes[str(mask_path)] = sha256(mask_path)
                if hashes[str(mask_path)] != instance['mask_sha256']:
                    raise ValueError('Saved floor mask archive SHA256 differs')
                mask = load_npz_checked(mask_path).get(instance.get('mask_key', 'mask'))
                if mask is None or mask.dtype != bool or mask.shape != floor.shape:
                    raise ValueError('Floor mask is not boolean on the exact saved RGB grid')
                floor |= mask
        depth = corrected['depth'][i, ..., 0]
        conf = corrected['world_points_conf'][i]
        valid = np.isfinite(conf) & (conf >= minimum) & np.isfinite(depth) & (depth > 0)
        display, agent, requested_xy = admitted_display_route(row, scale)
        parts = circle_clipped_routes(display, agent, basis, scale) if agent is not None else []
        visible = []
        for part in parts:
            visible.append(project_route_to_source(
                part, corrected['intrinsic'][i], corrected['extrinsic'][i], transform,
                depth=depth, depth_valid=valid,
                depth_absolute_tolerance=math.sqrt(3) * voxel_size / 2,
                depth_relative_tolerance=.03, sample_spacing_pixels=2,
                arrowhead_length_pixels=34, arrow_spacing_pixels=180,
            ))
        costmap_path = _safe(replay_path.parent, row['costmap_file'])
        hashes[str(costmap_path)] = sha256(costmap_path)
        if hashes[str(costmap_path)] != row['costmap_sha256']:
            raise ValueError('Prefix costmap archive SHA256 differs')
        costmap = load_npz_checked(costmap_path)
        if costmap.get('decision_state', np.empty(0)).ndim != 2:
            raise ValueError('Prefix costmap decision-state grid is required')
        if not np.allclose(costmap['projection_basis'], basis, atol=1e-8):
            raise ValueError('Prefix costmap and replay projection bases differ')
        frames.append({'row': row, 'source': source, 'rgb': video[source['frame_id']],
                       'transform': transform, 'floor': floor, 'parts': parts,
                       'projections': visible, 'agent': agent,
                       'agent_xy': agent @ basis[:2].T * scale if agent is not None else requested_xy,
                       'costmap': costmap,
                       'costmap_metadata': meta})
    terminal = replay.get('terminal_state', {})
    if terminal.get('mission_state') != 'active' or terminal.get('mission_completed') is not False or terminal.get('execution_state') != 'awaiting_observations' or terminal.get('reason') != 'recording_ended':
        raise ValueError('Recording end must await observations with its mission active')
    if not np.array_equal(route_points(terminal.get('retained_display_path_points', [])), route_points(rows[-1].get('display_path_points', []))):
        raise ValueError('Terminal hold must retain the last displayed guidance')
    data = {'frames': frames, 'replay': replay, 'manifest': manifest, 'reference_plan': reference,
            'pose_correction_verified': pose_verified, 'numeric_derivation_method': derivation['method'],
            'metric_scale_available': scale_available,
            'semantic_root_override': str(Path(semantic_root).resolve()) if semantic_root is not None else None,
            'scale': float(scale), 'basis': basis, 'input_sha256': hashes,
            'timeline': sampled_timeline([{'timestamp_ns': source['timestamp_ns'], 'source': source} for source in sources], video_fps=FPS, end_hold_seconds=END_HOLD_SECONDS)}
    if include_numeric:
        data['numeric'] = corrected
    return data


def _text(draw, xy, text, size=22, fill=NAVY):
    draw.text(xy, str(text), font=_font(size), fill=fill)


def _badge(draw, box, text, fill, text_fill=WHITE, size=18):
    draw.rounded_rectangle(box, radius=14, fill=fill)
    width = draw.textlength(text, font=_font(size))
    _text(draw, ((box[0] + box[2] - width) / 2, box[1] + 9), text, size, text_fill)


def presentation_size(source_shape, max_side=1920):
    """Preserve the source aspect ratio with even encoding dimensions."""
    height, width = source_shape
    if min(height, width) <= 0 or not isinstance(max_side, int) or not 512 <= max_side <= 3840:
        raise ValueError('Source dimensions and bounded presentation size are required')
    factor = max_side / max(height, width)
    return tuple(max(2, 2 * round(value * factor / 2)) for value in (width, height))


def map_box_for_size(size):
    side = round(min(size) * 300 / 1080)
    margin = round(min(size) * 28 / 1080)
    return (size[0] - margin - side, margin, size[0] - margin, margin + side)


def fit_source_rectangle(source_shape, size=SIZE):
    height, width = source_shape
    factor = min(size[0] / width, size[1] / height)
    w, h = round(width * factor), round(height * factor)
    return ((size[0] - w) // 2, (size[1] - h) // 2, w, h)


def source_to_canvas(values, source_shape, rectangle):
    left, top, width, height = rectangle
    return (np.asarray(values) + .5) * [width / source_shape[1], height / source_shape[0]] - .5 + [left, top]


def draw_photo(canvas, frame):
    rgb = frame['rgb'].copy()
    mask = project_mask_to_source(frame['floor'], frame['transform'], rgb.shape[:2])
    rgb[mask] = np.round(rgb[mask] * .77 + np.array(TEAL) * .23).astype(np.uint8)
    photo = Image.fromarray(rgb)
    rectangle = fit_source_rectangle(rgb.shape[:2], canvas.size)
    left, top, width, height = rectangle
    canvas.paste(photo.resize((width, height), Image.Resampling.LANCZOS), (left, top))
    layer = Image.new('RGBA', (width, height))
    draw = ImageDraw.Draw(layer)
    for projection in frame['projections']:
        segments = source_to_canvas(projection['source_segments'], rgb.shape[:2], rectangle) - [left, top]
        for segment in segments:
            draw.line([tuple(p) for p in segment], fill=(35, 20, 55, 230), width=10)
            draw.line([tuple(p) for p in segment], fill=(*PURPLE, 255), width=6)
        heads = source_to_canvas(projection['source_arrowheads'], rgb.shape[:2], rectangle) - [left, top]
        for head in heads:
            points = [tuple(p) for p in head]
            draw.polygon(points, fill=(*PURPLE, 255))
            draw.line(points + [points[0]], fill=(255, 248, 255, 255), width=1)
    canvas.paste(layer, (left, top), layer)


def draw_map(canvas, frame, basis, scale, box=MAP_BOX, *, show_metric_scale=True):
    """Track the simulated agent; show only this prefix grid and 2 m guidance."""
    draw = ImageDraw.Draw(canvas)
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    agent_xy = frame['agent_xy']
    span = 5.4
    pixels_per_metre = min(width, height) / span
    x0 = agent_xy[0] - width / pixels_per_metre / 2
    y1 = agent_xy[1] + height / pixels_per_metre / 2
    def pixel(points):
        xy = np.asarray(points) @ basis[:2].T * scale
        return np.stack([left + (xy[..., 0] - x0) * pixels_per_metre,
                         top + (y1 - xy[..., 1]) * pixels_per_metre], axis=-1)
    costmap = frame['costmap']
    origin = np.asarray(costmap['origin'], float)
    resolution = float(np.asarray(costmap['resolution']).ravel()[0])
    state = costmap['decision_state']
    if origin.shape != (2,) or resolution <= 0 or not np.isfinite(origin).all() or not np.isin(state, [0, 1, 2]).all():
        raise ValueError('Invalid prefix costmap grid origin/resolution/codes')
    xx = x0 + (np.arange(width) + .5) / pixels_per_metre
    yy = y1 - (np.arange(height) + .5) / pixels_per_metre
    xi = np.floor((xx - origin[0]) / resolution).astype(int)
    yi = np.floor((yy - origin[1]) / resolution).astype(int)
    inside = (yi[:, None] >= 0) & (yi[:, None] < state.shape[0]) & (xi[None, :] >= 0) & (xi[None, :] < state.shape[1])
    indices = state[np.clip(yi, 0, state.shape[0] - 1)[:, None], np.clip(xi, 0, state.shape[1] - 1)[None, :]]
    indices = np.where(inside, indices, 0)
    palette = np.array([[239, 242, 244], [113, 123, 128], [212, 229, 222]], dtype=np.uint8)
    inset = Image.fromarray(palette[indices])
    canvas.paste(inset, (left, top))
    cx, cy = left + width / 2, top + height / 2
    if show_metric_scale:
        radius = HORIZON_M * pixels_per_metre
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=(172, 133, 196), width=1)
    for part in frame['parts']:
        points = [tuple(p) for p in pixel(part)]
        draw.line(points, fill=WHITE, width=6)
        draw.line(points, fill=PURPLE, width=3)
        end = points[-1]
        draw.ellipse((end[0] - 3, end[1] - 3, end[0] + 3, end[1] + 3), fill=PURPLE)
    draw.ellipse((cx - 6, cy - 6, cx + 6, cy + 6),
                 fill=NAVY if frame['agent'] is not None else WHITE, outline=WHITE, width=2)
    if frame['agent'] is not None:
        draw.polygon([(cx, cy - 12), (cx - 5, cy - 3), (cx + 5, cy - 3)], fill=NAVY)
    draw.rectangle(box, outline=(220, 227, 231), width=1)
    if show_metric_scale:
        draw.line((left + 14, bottom - 14, left + 14 + pixels_per_metre, bottom - 14), fill=NAVY, width=2)
        _text(draw, (left + 14, bottom - 36), '1 m', 16)


def replay_status_label(frame, *, paused=False):
    if frame['row'].get('navigation_state', {}).get('execution_state') == 'pose_unverified':
        return 'Pose unverified'
    if paused:
        return 'Awaiting observations'
    state = frame['row'].get('navigation_state', {}).get('execution_state')
    return {'awaiting_support': 'Awaiting support',
            'awaiting_observations': 'Awaiting observations',
            'blocked': 'Blocked'}.get(state)


def render_frame(data, index, *, paused=False):
    frame = data['frames'][index]
    size = data.get('size', SIZE)
    canvas = Image.new('RGB', size, (0, 0, 0))
    draw_photo(canvas, frame)
    draw_map(canvas, frame, data['basis'], data['scale'], data.get('map_box', MAP_BOX),
             show_metric_scale=data.get('metric_scale_available', True))
    label = replay_status_label(frame, paused=paused)
    if label:
        font = _font(30)
        layer = Image.new('RGBA', size)
        draw = ImageDraw.Draw(layer)
        width = draw.textlength(label, font=font)
        x = (size[0] - width) / 2
        y = size[1] - 78
        draw.rounded_rectangle((x - 16, y - 8, x + width + 16, y + 44), radius=6, fill=(0, 0, 0, 130))
        draw.text((x, y), label, font=font, fill=WHITE)
        canvas.paste(layer, (0, 0), layer)
    return canvas


def _binary(name, explicit=None):
    path = Path(explicit).resolve() if explicit else None
    if path is not None and path.is_file():
        return str(path)
    if path is not None:
        raise ValueError(f'Explicit {name} executable does not exist')
    found = shutil.which(name)
    if found:
        return found
    if name == 'ffmpeg':
        roots = [Path.home() / 'AppData/Local/Microsoft/WinGet/Packages']
        for root in roots:
            candidates = sorted(root.glob('Gyan.FFmpeg_*/ffmpeg-*/bin/ffmpeg.exe'))
            if candidates:
                return str(candidates[-1])
    raise ValueError(f'{name} executable is unavailable; pass its explicit path')


def export_demo(data, output, *, ffmpeg=None, frames_only=False):
    size = data.get('size', SIZE)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Demo output must be a fresh directory')
    for path in data['input_sha256']:
        if Path(path).resolve().is_relative_to(output):
            raise ValueError('Output directory cannot contain a saved input')
    output.mkdir(parents=True)
    frames_dir = output / 'frames'
    proofs_dir = output / 'proofs'
    frames_dir.mkdir(); proofs_dir.mkdir()
    reports = []
    proof_indices = sorted(set([0, len(data['frames']) // 2, len(data['frames']) - 1]))
    for i, frame in enumerate(data['frames']):
        paused = i == len(data['frames']) - 1
        image = render_frame(data, i, paused=paused)
        image.save(frames_dir / f'{i:06d}.png')
        if i in proof_indices:
            image.save(proofs_dir / f'{i:02d}_{"recording_paused" if paused else "local_plan"}.png')
        distances = [np.linalg.norm((part - frame['agent']) @ data['basis'][:2].T * data['scale'], axis=1) for part in frame['parts']]
        reports.append({'frame_index': i, 'frame_id': frame['row']['frame_id'],
                        'timestamp_ns': frame['row']['timestamp_ns'], 'observed_through_index': i,
                        'paused_recording_hold': paused, 'display_path_parts': len(frame['parts']),
                        'execution_state': frame['row']['navigation_state']['execution_state'],
                        'status_label': replay_status_label(frame, paused=paused),
                        'agent_ground_available': frame['agent'] is not None,
                        'map_marker_scope': 'supported_ground_agent' if frame['agent'] is not None else 'requested_xy_only',
                        'maximum_display_radius_assumed_m': max((float(d.max()) for d in distances), default=0),
                        'full_floor_mask_displayed': True,
                        'projection_reports': [p['report'] for p in frame['projections']]})
    timeline = data['timeline']
    listing = output / 'navigation_replay.frames.txt'
    listing.write_text(''.join(f"file 'frames/{i:06d}.png'\n" * count for i, count in enumerate(timeline['repeat_counts'])), encoding='utf-8')
    target = output / 'navigation_replay.mp4'
    probe_report = None
    if not frames_only:
        binary = _binary('ffmpeg', ffmpeg)
        command = [binary, '-hide_banner', '-loglevel', 'error', '-n', '-r', str(FPS),
                   '-f', 'concat', '-safe', '1', '-i', str(listing), '-an', '-c:v', 'libx264',
                   '-threads', '4', '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(target)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        (output / 'encoder.log').write_text(result.stdout + result.stderr, encoding='utf-8')
        if result.returncode or not target.is_file():
            raise RuntimeError('Navigation replay MP4 encoding failed; see encoder.log')
        extension = '.exe' if Path(binary).suffix.lower() == '.exe' else ''
        probe = Path(binary).with_name('ffprobe' + extension)
        if not probe.is_file():
            probe = Path(_binary('ffprobe'))
        result = subprocess.run([str(probe), '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                                 '-show_streams', '-show_format', '-of', 'json', str(target)],
                                capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise RuntimeError('Encoded replay probe failed')
        probe_report = json.loads(result.stdout)
        write_json(output / 'ffprobe.json', probe_report)
        streams = probe_report.get('streams', [])
        expected_count = sum(timeline['repeat_counts'])
        if len(streams) != 1 or streams[0].get('codec_name') != 'h264' or streams[0].get('pix_fmt') != 'yuv420p' or streams[0].get('avg_frame_rate') != '20/1' or streams[0].get('width') != size[0] or streams[0].get('height') != size[1] or int(streams[0].get('nb_read_frames', -1)) != expected_count:
            raise RuntimeError('MP4 count/rate/dimensions/codec differ from admitted replay')
        if abs(float(probe_report['format']['duration']) - expected_count / FPS) > .051:
            raise RuntimeError('Encoded replay duration differs from timestamp-held timeline')
    for path, expected in data['input_sha256'].items():
        if sha256(path) != expected:
            raise RuntimeError('Saved input changed during rendering: ' + path)
    receipt = {
        'schema_version': 1, 'artifact_kind': 'research_navigation_replay_demo',
        'models_executed': False, 'planner_executed': False, 'robot_commands_executed': False,
        'frame_scope': 'observation_prefix_map_replay', 'perception_scope': 'cached_geometry_replay',
        'pose_correction_verified': data['pose_correction_verified'],
        'numeric_derivation_method': data['numeric_derivation_method'],
        'metric_scale_available': data['metric_scale_available'],
        'camera_height_reference_availability': data['reference_plan'].get('research_calibration', {}).get('camera_height_reference', {}).get('availability'),
        'assumed_metres_per_native_unit': data['scale'], 'display_horizon_assumed_m': HORIZON_M,
        'horizon_rule': 'horizontal_circle_about_simulated_ground_agent',
        'display_plan_admission': 'ordered_first_two_metres_of_full_route_euclidean_arc_length',
        'mask_scope': 'full_saved_SAM_candidate_surface_mask', 'mission_completed': False,
        'semantic_mask_path_contract': 'standard_semantics_directory_paths_are_run_relative; flattened_metadata_paths_are_jsonl_relative; explicit_override_optional',
        'semantic_root_override': data['semantic_root_override'],
        'terminal_state': data['replay']['terminal_state'], 'timeline': timeline,
        'source_samples': len(data['frames']), 'encoded_frame_count': sum(timeline['repeat_counts']),
        'dimensions': list(size), 'input_sha256': data['input_sha256'],
        'presentation': {'layout': 'full_source_aspect_video_with_small_map_inset',
                         'map_inset_box': list(data.get('map_box', MAP_BOX)),
                         'on_screen_text': (['1 m'] if data['metric_scale_available'] else []) + ['Awaiting support (unavailable route only)', 'Awaiting observations (waiting or end hold only)', 'Pose unverified (explicit original-pose display only)'],
                         'explanations_and_assumptions': 'saved_receipts'},
        'input_hashes_unchanged': True, 'renderer_sha256': sha256(__file__),
        'geometry_correction_helper_sha256': sha256(Path(__file__).with_name('derive_pose_corrected_run.py')),
        'route_projection_helper_sha256': sha256(Path(__file__).parent / 'pipeline_common/route_projection.py'),
        'projection_frames': reports, 'frames_only': frames_only,
        'video_path': str(target) if not frames_only else None,
        'video_sha256': sha256(target) if not frames_only else None,
    }
    write_json(output / 'render_receipt.json', receipt)
    print(json.dumps({'output': str(output), 'source_samples': len(data['frames']),
                      'encoded_frame_count': receipt['encoded_frame_count'],
                      'duration_seconds': receipt['encoded_frame_count'] / FPS,
                      'mission_state': 'active', 'end_hold_seconds': END_HOLD_SECONDS}))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('replay', 'geometry', 'video', 'sequence', 'geometry-manifest', 'semantics', 'reference-plan', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--ffmpeg', type=Path)
    parser.add_argument('--frames-only', action='store_true')
    parser.add_argument('--semantic-root', type=Path, help='Explicit root for mask_path fields; standard derived-run layout is detected')
    parser.add_argument('--max-side', type=int, default=1920, help='Longest canvas side; preserves source aspect ratio')
    args = parser.parse_args(argv)
    data = load_demo(args.replay, args.geometry, args.video, args.sequence, args.geometry_manifest, args.semantics,
                     args.reference_plan, semantic_root=args.semantic_root)
    data['size'] = presentation_size(data['frames'][0]['rgb'].shape[:2], args.max_side)
    data['map_box'] = map_box_for_size(data['size'])
    export_demo(data, args.output, ffmpeg=args.ffmpeg, frames_only=args.frames_only)


if __name__ == '__main__':
    main()
