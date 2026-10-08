"""Render minimal recorded-grid cubes and saved next-two-metre replay routes.

CPU presentation only: no models, map planning, physical calibration, or robot
commands. Every cumulative cube prefix is reconstructed from immutable cached
geometry solely to verify and display the already saved replay evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from build_navigation_replay import accepted_geometry
from export_navigation_replay_demo import (
    FPS, _binary, load_demo, replay_status_label, write_json,
)
from export_pipeline_video import _font, display_view_matrix, sha256, voxel_faces

PURPLE, NAVY = (158, 66, 210), (26, 46, 67)
BACKGROUND = (245, 248, 250)


def build_voxel_prefixes(data):
    """Exact recorded-grid indices; require the producer's counts each prefix."""
    numeric = data['numeric']
    valid = accepted_geometry(numeric, data['manifest'])
    source = data['reference_plan']['source']
    origin = np.asarray(source['voxel_origin_native'], float)
    voxel_size = float(source['voxel_size_native'])
    if origin.shape != (3,) or not np.isfinite(origin).all() or not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError('Reference plan requires a finite voxel origin and positive native cell size')
    accumulated = np.empty((0, 3), np.int64)
    floor_set = set()
    prefixes, counts = [], []
    accepted_points = 0
    for i, frame in enumerate(data['frames']):
        xyz = numeric['world_points'][i]
        raw = xyz[valid[i]]
        indices = np.unique(np.floor((raw - origin) / voxel_size).astype(np.int64), axis=0)
        accumulated = np.unique(np.concatenate((accumulated, indices)), axis=0)
        positive = xyz[valid[i] & frame['floor']]
        if len(positive):
            semantic_indices = np.unique(np.floor((positive - origin) / voxel_size).astype(np.int64), axis=0)
            floor_set.update(map(tuple, semantic_indices))
        accepted_points += int(valid[i].sum())
        diagnostics = frame['row']['diagnostics']
        if len(accumulated) != diagnostics['occupied_prefix_voxels']:
            raise ValueError(f'Prefix {i} voxel count differs from saved replay')
        if accepted_points != diagnostics['accepted_prefix_points']:
            raise ValueError(f'Prefix {i} accepted geometry count differs from saved replay')
        prefixes.append((accumulated.copy(), floor_set.copy()))
        counts.append({'frame_index': i, 'frame_id': frame['row']['frame_id'],
                       'voxel_count': len(accumulated), 'accepted_prefix_points': accepted_points,
                       'SAM_floor_evidence_voxel_count': len(floor_set), 'source_prefix_only': True})
    return {'prefixes': prefixes, 'counts': counts, 'origin': origin, 'voxel_size': voxel_size}


def fixed_projection(data, cubes, size):
    """Fixed oblique display view, with complete recorded cubes inside bounds."""
    view = display_view_matrix({'assumed_up_vector': data['replay']['assumed_up_vector']})
    final = cubes['prefixes'][-1][0]
    origin, voxel_size = cubes['origin'], cubes['voxel_size']
    bounds = []
    if len(final):
        low = origin + final.min(0) * voxel_size
        high = origin + (final.max(0) + 1) * voxel_size
        bounds.append(np.array([[x, y, z] for x in (low[0], high[0])
                               for y in (low[1], high[1]) for z in (low[2], high[2])]))
    bounds.extend(part for frame in data['frames'] for part in frame['parts'])
    bounds.extend(frame['agent'][None] for frame in data['frames'] if frame['agent'] is not None)
    # An empty map has no inferred ground point. Camera centers only establish
    # a bounded blank display frame; they are never drawn as support or routes.
    if not bounds:
        extrinsic = data['numeric']['extrinsic']
        bounds.append(-np.einsum('sji,sj->si', extrinsic[:, :, :3], extrinsic[:, :, 3]))
    projected = np.concatenate(bounds) @ view
    lower, upper = projected[:, :2].min(0), projected[:, :2].max(0)
    span = np.maximum(upper - lower, voxel_size * 2)
    scale = float(min((size - 90) / span) * .94)
    return {'view': view, 'center': (lower + upper) / 2, 'image_scale': scale, 'size': size}


def project(points, projection):
    coordinates = np.asarray(points) @ projection['view']
    xy = (coordinates[..., :2] - projection['center']) * projection['image_scale']
    xy[..., 0] += projection['size'] / 2
    xy[..., 1] = projection['size'] / 2 - xy[..., 1]
    return xy, coordinates[..., 2]


def render_voxel_frame(data, cubes, projection, index, *, paused=False):
    size = projection['size']
    image = Image.new('RGB', (size, size), BACKGROUND)
    draw = ImageDraw.Draw(image)
    indices, floor_indices = cubes['prefixes'][index]
    frame = data['frames'][index]
    if len(indices):
        faces = voxel_faces(indices, cubes['origin'], cubes['voxel_size'])
        polygons, depths = project(faces, projection)
        bases = np.array([(143, 176, 157) if tuple(v) in floor_indices else (181, 187, 191) for v in indices])
        shades = np.array([.57, .96, .70, .84, .68, .83])
        colors = np.round(bases[:, None, :] * shades[None, :, None]).astype(np.uint8).reshape(-1, 3)
        polygons = polygons.reshape(-1, 4, 2)
        order = np.argsort(depths.mean(axis=-1).ravel(), kind='stable')
        for j in order:
            color = colors[j]
            edge = tuple(map(int, np.round(color * .74)))
            draw.polygon([tuple(p) for p in polygons[j]], fill=tuple(map(int, color)), outline=edge)
    factor = size / 1080
    for part in frame['parts']:
        xy, _ = project(part, projection)
        points = [tuple(p) for p in xy]
        draw.line(points, fill=(247, 240, 253), width=max(2, round(11 * factor)))
        draw.line(points, fill=PURPLE, width=max(1, round(6 * factor)))
        if len(xy) > 1:
            direction = xy[-1] - xy[-2]
            norm = np.linalg.norm(direction)
            if norm > 1e-9:
                direction /= norm
                side = np.array([-direction[1], direction[0]]) * 6 * factor
                tail = xy[-1] - direction * 18 * factor
                draw.polygon([tuple(xy[-1]), tuple(tail + side), tuple(tail - side)], fill=PURPLE)
    if frame['agent'] is not None:
        p, _ = project(frame['agent'], projection)
        radius = 8 * factor
        draw.ellipse((p[0]-radius, p[1]-radius, p[0]+radius, p[1]+radius),
                     fill=NAVY, outline=(249, 251, 253), width=max(1, round(2 * factor)))
    label = replay_status_label(frame, paused=paused)
    if label:
        font = _font(max(14, round(26 * factor)))
        text_width = draw.textlength(label, font=font)
        draw.text(((size-text_width)/2, size-51*factor), label, font=font, fill=NAVY)
    return image


def export_voxel_demo(data, output, *, side=1080, ffmpeg=None, frames_only=False):
    if not isinstance(side, int) or side % 2 or not 512 <= side <= 2160:
        raise ValueError('Cube presentation side must be an even integer from 512 to 2160')
    output = Path(output).resolve()
    if output.exists() or any(Path(path).resolve().is_relative_to(output) for path in data['input_sha256']):
        raise ValueError('Voxel export requires a fresh output directory separate from saved inputs')
    cubes = build_voxel_prefixes(data)
    projection = fixed_projection(data, cubes, side)
    output.mkdir(parents=True)
    (output / 'frames').mkdir()
    (output / 'proofs').mkdir()
    proof_indices = {0, len(data['frames']) // 2, len(data['frames']) - 1}
    for i in range(len(data['frames'])):
        image = render_voxel_frame(data, cubes, projection, i, paused=i == len(data['frames'])-1)
        image.save(output / 'frames' / f'{i:06d}.png')
        if i in proof_indices:
            image.save(output / 'proofs' / f'{i:02d}_voxel_replay.png')
    timeline = data['timeline']
    count = sum(timeline['repeat_counts'])
    listing = output / 'voxel_replay.frames.txt'
    listing.write_text(''.join(f"file 'frames/{i:06d}.png'\n" * n
                              for i, n in enumerate(timeline['repeat_counts'])), encoding='utf-8')
    target = output / 'voxel_replay.mp4'
    if not frames_only:
        binary = _binary('ffmpeg', ffmpeg)
        result = subprocess.run([binary, '-hide_banner', '-loglevel', 'error', '-n', '-r', str(FPS),
                                 '-f', 'concat', '-safe', '1', '-i', str(listing), '-an', '-c:v', 'libx264',
                                 '-threads', '4', '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(target)],
                                capture_output=True, text=True, timeout=600)
        (output / 'encoder.log').write_text(result.stdout + result.stderr, encoding='utf-8')
        if result.returncode or not target.is_file():
            raise RuntimeError('Voxel MP4 encoder failed; see encoder.log')
        probe = Path(binary).with_name('ffprobe.exe' if Path(binary).suffix.lower() == '.exe' else 'ffprobe')
        if not probe.is_file():
            probe = Path(_binary('ffprobe'))
        result = subprocess.run([str(probe), '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                                 '-show_streams', '-show_format', '-of', 'json', str(target)],
                                capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise RuntimeError('Voxel MP4 probe failed')
        report = json.loads(result.stdout)
        write_json(output / 'ffprobe.json', report)
        streams = report.get('streams', [])
        expected = ('h264', 'yuv420p', side, side, '20/1', count)
        if len(streams) != 1 or tuple(streams[0].get(k) for k in ('codec_name', 'pix_fmt', 'width', 'height', 'avg_frame_rate')) + (int(streams[0].get('nb_read_frames', -1)),) != expected:
            raise RuntimeError('Voxel movie count/rate/codec/dimensions differ')
        if abs(float(report['format']['duration']) - count/FPS) > .051:
            raise RuntimeError('Voxel movie duration differs from timestamp-held timeline')
    for path, expected in data['input_sha256'].items():
        if sha256(path) != expected:
            raise RuntimeError('Saved source changed during voxel export: ' + path)
    write_json(output / 'render_receipt.json', {
        'artifact_kind': 'research_voxel_navigation_replay_video', 'schema_version': 1,
        'models_executed': False, 'new_maps_produced': False, 'new_plans_produced': False,
        'source_geometry_fingerprint': data['manifest']['derivation']['source_geometry_fingerprint'],
        'corrected_geometry_fingerprint': data['manifest']['geometry_fingerprint'],
        'geometry_numeric_method': 'derive_pose_corrected_run.numeric_for_derivation',
        'numeric_derivation_method': data['numeric_derivation_method'],
        'pose_correction_verified': data['pose_correction_verified'],
        'geometry_filter_method': 'build_navigation_replay.accepted_geometry',
        'voxel_origin_native': cubes['origin'].tolist(), 'voxel_size_native': cubes['voxel_size'],
        'voxel_size_assumed_m': cubes['voxel_size'] * data['scale'] if data['metric_scale_available'] else None,
        'metric_scale_available': data['metric_scale_available'],
        'camera_height_reference_availability': data['reference_plan'].get('research_calibration', {}).get('camera_height_reference', {}).get('availability'),
        'metres_per_native_unit_reference_value': data['scale'],
        'semantic_cube_color_rule': 'subdued_green_if_any_accepted_saved_SAM_floor_pixel_maps_to_this_prefix_voxel_else_gray',
        'semantic_color_is_floor_validation': False,
        'cube_geometry': 'six_recorded_grid_cell_faces_export_pipeline_video.voxel_faces',
        'fixed_oblique_view_matrix_native': projection['view'].tolist(),
        'fixed_image_scale_pixels_per_native_unit': projection['image_scale'],
        'fixed_projected_center_native': projection['center'].tolist(),
        'framing_scope': 'fixed_final_recorded_map_bounds; empty_map_uses_camera_centers_for_blank_display_only',
        'coordinate_projection': 'native_xyz @ fixed_view_matrix; display_up_from_explicit_replay_assumption',
        'assumptions': data['replay']['assumptions'], 'assumed_up_vector': data['replay']['assumed_up_vector'],
        'route_scope': 'existing_per_prefix_two_metre_display_routes', 'route_replanned': False,
        'route_is_screen_overlay_not_depth_occlusion_test': True,
        'agent_scope': 'existing_supported_simulated_agent_ground_point; unavailable_ground_not_drawn',
        'prefix_counts_match_replay': True, 'prefix_counts': cubes['counts'],
        'frame_execution_states': [f['row']['navigation_state']['execution_state'] for f in data['frames']],
        'timeline': timeline, 'mission_completed': False, 'terminal_state': data['replay']['terminal_state'],
        'input_sha256': data['input_sha256'], 'inputs_unchanged': True,
        'semantic_root_override': data['semantic_root_override'],
        'renderer_sha256': sha256(__file__),
        'geometry_correction_helper_sha256': sha256(Path(__file__).with_name('derive_pose_corrected_run.py')),
        'cube_helper_sha256': sha256(Path(__file__).with_name('export_pipeline_video.py')),
        'presentation': 'minimal_full_frame_cube_view; only waiting status label',
        'dimensions': [side, side], 'encoded_frames': count, 'duration_seconds': count/FPS,
        'source_samples': len(data['frames']), 'encoded_frame_count': count,
        'frames_only': frames_only, 'video_path': str(target) if not frames_only else None,
        'video_sha256': sha256(target) if not frames_only else None,
    })
    print(json.dumps({'output': str(output), 'prefix_counts': [c['voxel_count'] for c in cubes['counts']],
                      'encoded_frames': count, 'fps': FPS, 'duration_seconds': count/FPS}), flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('replay', 'geometry', 'video', 'sequence', 'geometry-manifest', 'semantics', 'reference-plan', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--ffmpeg', type=Path)
    parser.add_argument('--frames-only', action='store_true')
    parser.add_argument('--semantic-root', type=Path, help='Explicit root for mask_path fields; standard derived-run layout is detected')
    parser.add_argument('--side', type=int, default=1080)
    args = parser.parse_args(argv)
    data = load_demo(args.replay, args.geometry, args.video, args.sequence, args.geometry_manifest,
                     args.semantics, args.reference_plan, include_numeric=True, semantic_root=args.semantic_root)
    export_voxel_demo(data, args.output, side=args.side, ffmpeg=args.ffmpeg, frames_only=args.frames_only)


if __name__ == '__main__':
    main()
