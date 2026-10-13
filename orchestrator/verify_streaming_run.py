# Support direct compatibility-script execution without environment mutation.
if __package__ in (None, ''):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
"""Verify recorded frame/mask contracts and the optional SAM runtime on real RGB."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

from orchestrator.paths import ROOT
sys.path.insert(0, str(ROOT))


def verify_map(run):
    from orchestrator.shared import rgb_digest, read_rgb
    sys.path.insert(0, str(ROOT / 'orchestrator'))
    from semantic_map import build_semantic_point_cloud
    from view_ply import read_ply_header
    config = json.loads((run / 'config.json').read_text())
    status = json.loads((run / 'status.json').read_text())
    if status['status'] != 'complete' or status['dropped_sampled_frames']:
        raise ValueError('Expected a complete run with no dropped sampled frames')
    events = [json.loads(line) for line in (run / 'events.jsonl').read_text().splitlines()]
    geometry = [e for e in events if e['type'] == 'geometry_frame']
    mapped = [e for e in events if e['type'] == 'frame']
    expected = list(range(0, status['source_frame_count'], status['input_stride']))
    if config['max_frames']:
        expected = expected[:config['max_frames']]
    if [e['source_index'] for e in geometry] != expected:
        raise ValueError('Geometry source indices differ from the sampling schedule')
    if [e['source_index'] for e in mapped] != expected[::config['map_step']]:
        raise ValueError('Mapped source indices differ from the requested frame step')
    predictions = sorted((run / 'predictions').glob('frame_*.npz'))
    if [p.name for p in predictions] != [f'frame_{e["source_index"]:06d}.npz' for e in mapped]:
        raise ValueError('Saved predictions differ from mapped frames')
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        masks = directory / 'masks'
        masks.mkdir()
        for path, event in zip(predictions, mapped):
            rgb = read_rgb(path)
            with np.load(run / 'masks' / path.name) as data:
                mask, digest = data['out_binary_masks'].copy(), str(data['rgb_sha256'])
            if mask.dtype != np.bool_ or mask.shape != rgb.shape[:2]:
                raise ValueError(f'Invalid native mask: {path.name}')
            if rgb_digest(rgb) != digest or digest != event['rgb_sha256']:
                raise ValueError(f'RGB/mask provenance mismatch: {path.name}')
            if int(mask.sum()) != event['mask_pixels']:
                raise ValueError(f'Mask count mismatch: {path.name}')
            source_index = event['source_index']
            (masks / f'video_obstacles_frame{source_index:06d}.npz').symlink_to((run/'masks'/path.name).resolve())
        output = directory / 'original.ply'
        parameters = SimpleNamespace(frame_step=1, pixel_step=1, lingbot_dir=run/'predictions',
            sam_dir=masks, video_name='video', prompt='obstacles', video_path=Path(config['video_path']),
            min_depth_conf=2.5, max_depth=25, segment_color=np.array([255, 0, 0], dtype=np.uint8),
            patch_size=14, out_file=output)
        with redirect_stdout(sys.stderr):
            build_semantic_point_cloud(parameters)
        original_count, _, original_offset = read_ply_header(output)
        stream_count, _, stream_offset = read_ply_header(run/'obstacle_map.ply')
        if original_count != stream_count or stream_count != status['points']:
            raise ValueError('Original/streaming PLY counts differ')
        streams = [np.memmap(path, dtype=np.uint8, mode='r', offset=offset)
                   for path, offset in [(output, original_offset), (run/'obstacle_map.ply', stream_offset)]]
        if streams[0].shape != streams[1].shape:
            raise ValueError('PLY payload lengths differ')
        for start in range(0, len(streams[0]), 2**20):
            if not np.array_equal(streams[0][start:start+2**20], streams[1][start:start+2**20]):
                raise ValueError('Original/streaming PLY payloads differ')
    return {'geometry_frames': len(geometry), 'mapped_frames': len(mapped),
        'no_drops': True, 'mask_rgb_hashes_match': True,
        'original_mapper_ply_payload_bit_exact': True, 'points': stream_count}


def verify_sam(run, all_videos=False):
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    from PIL import Image
    from orchestrator.shared import read_rgb
    from sam.segment import union_masks
    from sam.sam_runtime import CachedSamSegmenter
    config = json.loads((run/'config.json').read_text())
    with redirect_stdout(sys.stderr):
        model = CachedSamSegmenter('cuda', .5, config.get('sam_checkpoint'))
    events = [json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()]
    cases = [(run/'predictions'/f'frame_{e["source_index"]:06d}.npz', e['prompts'],
              run/'masks'/f'frame_{e["source_index"]:06d}.npz') for e in events if e['type']=='frame']
    if all_videos:
        for path in sorted((ROOT/'data/vlm_maps/timed_run').glob('*/*/prompts.json')):
            frames = sorted((path.parent/'lingbot').glob('*/frame_*.npz'))
            if frames:
                cases.append((frames[0], json.loads(path.read_text())['prompts'][:8] or ['log'], None))
    results = []
    for frame, prompts, saved in cases:
        rgb = read_rgb(frame)
        reference = np.zeros(rgb.shape[:2], dtype=bool)
        model.set_image(Image.fromarray(rgb))
        for prompt in prompts:
            reference |= union_masks(model.masks(prompt), reference.shape)
        candidate = model.union(prompts, reference.shape)
        exact = bool(np.array_equal(reference, candidate))
        saved_exact = None
        if saved is not None:
            with np.load(saved) as data:
                saved_exact = bool(np.array_equal(reference, data['out_binary_masks']))
        results.append({'frame': str(frame), 'prompts': prompts, 'pixels': int(reference.sum()),
                        'adapter_bit_exact': exact, 'saved_pipeline_mask_bit_exact': saved_exact})
    if not all(r['adapter_bit_exact'] and r['saved_pipeline_mask_bit_exact'] is not False for r in results):
        raise ValueError('SAM reference/candidate mask mismatch')
    return {'cases': results, 'all_masks_bit_exact': True, 'text_cache': model.cache_stats()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--role', choices=['map', 'sam'], required=True)
    parser.add_argument('--all-videos', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify_map(args.run) if args.role == 'map' else verify_sam(args.run, args.all_videos)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(f'{args.role} verification passed: {args.output}')


if __name__ == '__main__':
    main()
