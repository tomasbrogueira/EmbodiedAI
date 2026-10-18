"""Canonical mask-to-world projection preserving every selected point row."""
import json
from pathlib import Path
import numpy as np
from orchestrator.artifacts import digest, publish, save_array
from orchestrator.io_utils import frame_number, load_depth_map, load_extrinsic, load_intrinsic, load_rgb
from orchestrator.shared import rgb_digest
from orchestrator.semantics import CHANNELS, FUSION_POLICY, fuse, colors


def frame_arrays(prediction, masks, *, pixel_step=1, min_depth_conf=2.5, max_depth=25., unproject=None):
    if not isinstance(pixel_step, int) or isinstance(pixel_step, bool) or pixel_step < 1:
        raise ValueError('pixel_step must be a positive integer')
    if min_depth_conf is not None and (not np.isfinite(min_depth_conf) or min_depth_conf < 0):
        raise ValueError('min_depth_conf must be nonnegative or None')
    if max_depth is not None and (not np.isfinite(max_depth) or max_depth <= 0):
        raise ValueError('max_depth must be positive or None')
    if unproject is None:
        from lingbot.geometry import depth_to_world_points
        unproject = depth_to_world_points
    depth, confidence = load_depth_map(prediction['depth']), load_depth_map(prediction['depth_conf'])
    rgb = load_rgb(prediction['images'])
    if depth.shape != confidence.shape or rgb.shape[:2] != depth.shape:
        raise ValueError('RGB/depth/confidence dimensions do not match')
    fuse(masks, depth.shape)
    intrinsic, extrinsic = load_intrinsic(prediction['intrinsic']), load_extrinsic(prediction['extrinsic'])
    rotation = extrinsic[:, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-4) or np.linalg.det(rotation) <= 0:
        raise ValueError('Camera rotation must be proper orthonormal')
    xyz = np.asarray(unproject(depth, intrinsic, extrinsic), dtype=np.float32)
    if xyz.shape != (*depth.shape, 3):
        raise ValueError('Unexpected world point shape')
    valid = np.isfinite(xyz).all(-1) & np.isfinite(depth) & (depth > 0) & np.isfinite(confidence)
    if max_depth is not None:
        valid &= depth < max_depth
    if min_depth_conf is not None:
        valid &= confidence > min_depth_conf
    take = valid[::pixel_step, ::pixel_step]
    rows = {name: value[::pixel_step, ::pixel_step][take] for name, value in masks.items()}
    support, blocked = fuse(rows, (int(take.sum()),))
    return dict(points=xyz[::pixel_step, ::pixel_step][take],
                source_rgb=rgb[::pixel_step, ::pixel_step][take],
                pixel_ids=np.arange(depth.size).reshape(depth.shape)[::pixel_step, ::pixel_step][take],
                support_mask=support, obstacle_mask=blocked, **rows), -rotation.T @ extrinsic[:, 3], depth.shape


def build_snapshot(predictions, output, *, support_dir=None, semantic_dir=None, video_name=None,
                   support_prompt='floor', frame_step=2, pixel_step=1, source_fps=None,
                   timestamps=None, min_depth_conf=2.5, max_depth=25., unproject=None,
                   reconstruction_policy='offline_uniform_anchors', selected_frames=None):
    """Read immutable saved geometry; source hashes and pixel IDs bind row identity."""
    from orchestrator.sam_masks import build_sam_index, load_sam_mask, preprocess_mask
    if frame_step < 1:
        raise ValueError('frame_step must be positive')
    paths = sorted(Path(predictions).glob('frame_*.npz'), key=frame_number)
    if not paths:
        raise ValueError('No saved geometry predictions')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    support_index = build_sam_index(Path(support_dir), video_name, support_prompt) if support_dir else {}
    manifest = json.loads((Path(semantic_dir) / 'manifest.json').read_text()) if semantic_dir else None
    selected = [p for p in paths if frame_number(p) % frame_step == 0] if selected_frames is None else [p for p in paths if p.name in selected_frames]
    if selected_frames is not None and [p.name for p in selected] != list(selected_frames):
        raise ValueError('Selected source frames differ from saved geometry')
    selected_names = {p.name for p in selected}
    if not selected:
        raise ValueError('Sampling selects no geometry frames')
    if manifest and (manifest.get('schema_version') != 1 or manifest.get('frames') != [p.name for p in selected]):
        raise ValueError('Semantic manifest must identify exactly the selected geometry frames')
    frames, arrays, cameras, source_ids, source_times = [], [], [], [], []
    enabled = {'base_support_mask'} if support_dir else set()
    if manifest:
        enabled.update(manifest.get('enabled_channels', [k for k, v in manifest.get('query_groups', {}).items() if v]))
        if not manifest.get('query_groups'):
            enabled.add('vlm_non_traversability_mask')
    sources = {}
    for path in paths:
        index = frame_number(path)
        with np.load(path, allow_pickle=False) as file:
            prediction = {k: file[k] for k in file.files}
        rgb = load_rgb(prediction['images'])
        extrinsic = load_extrinsic(prediction['extrinsic'])
        rotation = extrinsic[:, :3]
        if not np.isfinite(extrinsic).all() or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-4) or np.linalg.det(rotation) <= 0:
            raise ValueError(f'Invalid camera pose in {path.name}')
        cameras.append(-rotation.T @ extrinsic[:, 3])
        source_ids.append(index)
        if timestamps is not None:
            source_times.append(float(timestamps[index]))
        elif source_fps is not None and np.isfinite(source_fps) and source_fps > 0:
            source_times.append(index / source_fps)
        else:
            raise ValueError('Projection needs source timestamps or a positive source FPS')
        sources[path.name] = digest(path)
        if path.name not in selected_names:
            continue
        masks = {}
        if support_dir:
            mask_path = support_index.get(index)
            if mask_path is None:
                raise ValueError(f'Missing required support mask for source frame {index}')
            masks['base_support_mask'] = preprocess_mask(load_sam_mask(mask_path), rgb.shape[:2], 14)
            sources['support/' + mask_path.name] = digest(mask_path)
        if semantic_dir:
            mask_path = Path(semantic_dir) / path.name
            with np.load(mask_path, allow_pickle=False) as file:
                if str(file['rgb_sha256'].item()) != rgb_digest(rgb):
                    raise ValueError(f'{path.name}: mask/RGB provenance mismatch')
                for channel in enabled - ({'base_support_mask'} if support_dir else set()):
                    key = channel if channel in file else 'out_binary_masks' if channel == 'vlm_non_traversability_mask' and not manifest.get('query_groups') else None
                    if key is None:
                        raise ValueError(f'Missing required {channel} for {path.name}')
                    masks[channel] = np.asarray(file[key])
            sources['semantic/' + path.name] = digest(mask_path)
        rows, _, shape = frame_arrays(prediction, masks, pixel_step=pixel_step,
                                     min_depth_conf=min_depth_conf, max_depth=max_depth, unproject=unproject)
        rows['frame_ids'] = np.full(len(rows['points']), index, dtype=np.int64)
        arrays.append(rows)
        frames.append(dict(source_index=index, shape=list(shape), rows=len(rows['points']), rgb_sha256=rgb_digest(rgb)))
    if np.any(np.diff(source_times) < 0) or not np.isfinite(source_times).all():
        raise ValueError('Source timestamps must be finite and ordered')
    for key in arrays[0]:
        save_array(output / f'{key}.npy', np.concatenate([a[key] for a in arrays]))
    for key, value in dict(cameras=cameras, source_frame_ids=source_ids, timestamps=source_times).items():
        save_array(output / f'{key}.npy', np.asarray(value))
    offsets = np.r_[0, np.cumsum([f['rows'] for f in frames])]
    save_array(output / 'frame_offsets.npy', offsets)
    metadata = dict(reconstruction_policy=reconstruction_policy, sources=sources, frames=frames,
                    predictions_dir=str(Path(predictions).resolve()),
                    support_dir=str(Path(support_dir).resolve()) if support_dir else None,
                    video_name=video_name, support_prompt=support_prompt,
                    reference_support_sources={p.name: digest(p) for p in support_index.values()},
                    enabled_channels=sorted(enabled), fusion_policy=FUSION_POLICY,
                    filters=dict(frame_step=frame_step, pixel_step=pixel_step,
                                 min_depth_conf=min_depth_conf, max_depth=max_depth,
                                 selected_frames=[p.name for p in selected]),
                    correspondence_verified=True,
                    semantic_manifest_sha256=digest(Path(semantic_dir) / 'manifest.json') if semantic_dir else None)
    from orchestrator.pointcloud import save_binary_ply
    xyz, rgb = np.load(output / 'points.npy'), np.load(output / 'source_rgb.npy')
    positive, negative = np.load(output / 'support_mask.npy'), np.load(output / 'obstacle_mask.npy')
    save_binary_ply(output / 'map.ply', xyz, colors(rgb, positive, negative))
    return publish(output, 'map', metadata)
