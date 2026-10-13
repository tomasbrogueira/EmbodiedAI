"""Geometry references on exactly the frames of a no-drop plugin run.

Reuses the original causal loop and original windowed batch inference helper.
No VLM/SAM is loaded; standalone timings are not complete-pipeline rates.
"""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--modes', nargs='+', choices=['causal64', 'windowed32'], default=['causal64', 'windowed32'])
    args = parser.parse_args()
    config = json.loads((args.run / 'config.json').read_text())
    if config['realtime']:
        parser.error('Reference requires a run without frame dropping')
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    from plugins.small_obstacles.common import rgb_digest
    from plugins.small_obstacles.run_streaming import video_frames
    from plugins.small_obstacles.stream_worker import LingBotStream
    config['kv_cache_window'] = 64
    config['video_path'] = Path(config['video_path'])
    state = {}
    frames = list(video_frames(SimpleNamespace(**config), state))
    source_indices = [frame['source_index'] for frame in frames]
    candidate_events = [json.loads(line) for line in (args.run / 'events.jsonl').read_text().splitlines()]
    candidate_frames = [e for e in candidate_events if e['type'] == 'geometry_frame']
    if source_indices != [e['source_index'] for e in candidate_frames]:
        raise ValueError('Candidate and reference source indices do not match')
    if any(rgb_digest(frame['rgb']) != event['rgb_sha256'] for frame, event in zip(frames, candidate_frames)):
        raise ValueError('Candidate and reference decoded RGB hashes do not match')
    t = time.perf_counter()
    with redirect_stdout(sys.stderr):
        runner = LingBotStream(config)
    results = {'run': str(args.run.resolve()), 'source_indices': source_indices,
               'rgb_hashes_match': True, 'load_seconds': time.perf_counter() - t,
               'camera_iterations': config['camera_iterations'], 'results': {}}

    def save_frame(directory, frame, arrays):
        np.savez(directory / f'frame_{frame["source_index"]:06d}.npz',
                 images=frame['rgb'], **{k: arrays[k] for k in ('depth', 'depth_conf', 'intrinsic', 'extrinsic')})

    for mode in args.modes:
        directory = args.output / mode
        directory.mkdir()
        runner.model.clean_kv_cache()
        runner.count = 0
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        if mode == 'causal64':
            timings = []
            with tempfile.TemporaryDirectory() as temporary:
                inp, out = str(Path(temporary) / 'input.npz'), str(Path(temporary) / 'output.npz')
                batches = [frames[:2]] + [[f] for f in frames[2:]]
                sequence = 0
                for batch in batches:
                    np.savez(inp, rgb=np.stack([f['rgb'] for f in batch]))
                    t = time.perf_counter()
                    with redirect_stdout(sys.stderr):
                        runner.frame({'input': inp, 'output': out})
                    timings.append(time.perf_counter() - t)
                    with np.load(out) as data:
                        predictions = {k: data[k].copy() for k in data.files}
                    for i, frame in enumerate(batch):
                        if sequence % config['map_step'] == 0:
                            save_frame(directory, frame, {k: v[i] for k, v in predictions.items()})
                        sequence += 1
            extra = {'geometry_batch_seconds': timings, 'kv_cache_window': 64}
        else:
            sys.path.insert(0, str(ROOT / 'lingbot-map'))
            from demo_render.batch_demo import run_inference
            images = torch.from_numpy(np.stack([f['rgb'] for f in frames])).permute(0, 3, 1, 2).float() / 255
            parameters = SimpleNamespace(mode='windowed', num_scale_frames=2, keyframe_interval=2,
                window_size=32, overlap_size=None, overlap_keyframes=8, scale_mode='median')
            with redirect_stdout(sys.stderr):
                predictions = run_inference(runner.model, images, parameters)
            for sequence, frame in enumerate(frames):
                if sequence % config['map_step'] == 0:
                    save_frame(directory, frame, {k: predictions[k][sequence]
                               for k in ('depth', 'depth_conf', 'intrinsic', 'extrinsic')})
            extra = {'window_size_keyframes': 32, 'overlap_keyframes': 8}
        torch.cuda.synchronize()
        results['results'][mode] = {'seconds_including_reference_saves': time.perf_counter() - started,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(), **extra}
        (args.output / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
        print(mode, results['results'][mode]['seconds_including_reference_saves'], flush=True)


if __name__ == '__main__':
    main()
