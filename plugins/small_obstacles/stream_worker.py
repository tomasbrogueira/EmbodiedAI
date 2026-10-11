"""Persistent model workers for the optional incremental runner (JSON-lines RPC)."""
import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class LingBotStream:
    """The existing model's causal scale/forward loop, without collecting a video."""
    def __init__(self, config):
        import torch
        from types import SimpleNamespace
        sys.path.insert(0, str(ROOT / 'lingbot-map'))
        from demo import load_model, postprocess
        self.torch, self.postprocess = torch, postprocess
        self.config = config
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        args = SimpleNamespace(mode='windowed', image_size=config['image_size'], patch_size=14,
            enable_3d_rope=True, max_frame_num=config['max_frame_num'],
            kv_cache_sliding_window=config.get('kv_cache_window', 64), num_scale_frames=2, use_sdpa=True,
            camera_num_iterations=config['camera_iterations'], model_path=config['lingbot_checkpoint'])
        # Load checkpoint on CPU to avoid a second full GPU weight copy.
        self.model = load_model(args, 'cpu')
        self.model.aggregator.to(dtype=self.dtype)
        self.model.to('cuda').eval()
        self.model.clean_kv_cache()
        self.count = 0
        self.scale_frames = 0

    def reset(self):
        self.model.clean_kv_cache()
        self.model._set_skip_append(False)
        self.count = self.scale_frames = 0

    def frame(self, request):
        torch = self.torch
        with np.load(request['input'], allow_pickle=False) as data:
            rgb = data['rgb'].copy()  # T,H,W,3; already original OpenCV preprocessing.
        images = torch.from_numpy(rgb).permute(0, 3, 1, 2).float() / 255.
        batch = images.unsqueeze(0).to('cuda')
        if self.count == 0:
            self.scale_frames = len(images)
            block = self.scale_frames
            skip = False
        else:
            if len(images) != 1:
                raise ValueError('After initialization, send one frame at a time')
            block = 1
            skip = (self.count - self.scale_frames) % 2 != 0
        if self.count + len(images) > self.config['max_frame_num']:
            raise ValueError('Stream exceeds max-frame-num; increase the configured limit')
        if skip:
            self.model._set_skip_append(True)
        try:
            with torch.inference_mode(), torch.autocast('cuda', dtype=self.dtype):
                pred = self.model.forward(batch, num_frame_for_scale=self.scale_frames,
                    num_frame_per_block=block, causal_inference=True)
            # Exactly the same postprocessing as the existing batch demo,
            # including its extrinsic convention. Do not invert poses again.
            pred, _ = self.postprocess(pred, images)
        finally:
            if skip:
                self.model._set_skip_append(False)
        arrays = {k: pred[k].float().numpy() for k in ('depth', 'depth_conf', 'intrinsic', 'extrinsic')}
        np.savez(request['output'], images=rgb, **arrays)
        self.count += len(images)
        return {'frames': len(images)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', choices=['lingbot', 'vlm', 'sam'], required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    import torch
    threads = config.get('worker_threads', 0)
    if threads:
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
    from plugins.small_obstacles.runtime_metrics import CudaMemoryMonitor, cache_memory, host_memory, release_unused_cuda
    protocol = sys.stdout
    monitor = None

    def send(value):
        protocol.write(json.dumps(value) + '\n')
        protocol.flush()

    try:
        started = time.perf_counter()
        with redirect_stdout(sys.stderr):
            if args.role == 'lingbot':
                model = LingBotStream(config)
            elif args.role == 'vlm':
                from plugins.small_obstacles.propose import QwenProposer
                model = QwenProposer(config['vlm_model'], max_new_tokens=config['max_new_tokens'])
            else:
                if config.get('sam_text_cache', False):
                    from plugins.small_obstacles.sam_runtime import CachedSamSegmenter
                    model = CachedSamSegmenter('cuda', .5, config.get('sam_checkpoint'),
                                               config.get('sam_text_cache_size', 16))
                else:
                    from plugins.small_obstacles.segment import SamSegmenter
                    model = SamSegmenter('cuda', .5, config.get('sam_checkpoint'))
        monitor = CudaMemoryMonitor(torch, args.role, args.config.parent / f'memory_{args.role}.jsonl',
                                    config.get('memory_sample_interval', .1))
        send({'status': 'ready', 'load_seconds': time.perf_counter() - started,
              'memory': monitor.snapshot(), 'host_memory': host_memory(),
              'torch_threads': torch.get_num_threads(),
              'torch_interop_threads': torch.get_num_interop_threads()})
        for line in sys.stdin:
            request = json.loads(line)
            if request.get('op') == 'close':
                break
            started = time.perf_counter()
            try:
                before = monitor.snapshot()
                with redirect_stdout(sys.stderr):
                    operation = request.get('op', 'frame')
                    if operation in ('reset', 'release_sequence'):
                        if args.role == 'lingbot':
                            model.reset()
                        elif args.role == 'sam' and hasattr(model, 'state'):
                            del model.state
                        # Keep the bounded text cache belonging to this SAM
                        # model, while discarding previous image features.
                        torch.cuda.empty_cache()
                        monitor.close()
                        if operation == 'reset':
                            config = json.loads(Path(request['config']).read_text())
                            sys.stderr.flush()
                            with (Path(request['config']).parent / f'{args.role}.log').open('a') as log:
                                os.dup2(log.fileno(), sys.stderr.fileno())
                            torch.cuda.reset_peak_memory_stats()
                            monitor = CudaMemoryMonitor(torch, args.role,
                                Path(request['config']).parent / f'memory_{args.role}.jsonl',
                                config.get('memory_sample_interval', .1))
                            if args.role == 'lingbot':
                                model.config = config
                        result = {'sequence_reset': True}
                    elif operation != 'frame':
                        raise ValueError(f'Unknown worker operation: {operation}')
                    elif args.role == 'lingbot':
                        result = model.frame(request)
                    else:
                        from PIL import Image
                        with np.load(request['input'], allow_pickle=False) as data:
                            rgb = data['rgb'].copy()
                        image = Image.fromarray(rgb)
                        if args.role == 'vlm':
                            from plugins.small_obstacles.common import parse_proposal
                            raw = model.propose(image, config['instruction'])
                            result = {'raw_response': raw}
                            try:
                                result['prompts'] = parse_proposal(raw)
                            except ValueError as exc:
                                result['proposal_error'] = str(exc)
                        else:
                            from plugins.small_obstacles.segment import union_masks
                            mask = np.zeros(rgb.shape[:2], dtype=bool)
                            if request['prompts']:
                                model.set_image(image)
                                if config.get('sam_text_cache', False):
                                    mask = model.union(request['prompts'], mask.shape)
                                else:
                                    for prompt in request['prompts']:
                                        mask |= union_masks(model.masks(prompt), mask.shape)
                            np.savez(request['output'], out_binary_masks=mask)
                            result = {'mask_pixels': int(mask.sum()),
                                      'text_cache': model.cache_stats() if config.get('sam_text_cache', False) else None}
                result['cuda_cache_released'] = release_unused_cuda(torch,
                    config.get('cuda_cache_policy', 'retain'), int(config.get('cuda_cache_min_free_gib', 2.) * 2**30))
                if args.role == 'lingbot':
                    result['cache'] = cache_memory(model.model)
                send({'status': 'ok', 'elapsed_seconds': time.perf_counter() - started,
                      'memory_before': before, 'memory': monitor.snapshot(),
                      'host_memory': host_memory(), **result})
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                failure = {'status': 'error', 'error': str(exc)}
                try:
                    failure.update(memory=monitor.snapshot(), host_memory=host_memory())
                except Exception as sampling_error:
                    failure['memory_error'] = str(sampling_error)
                send(failure)
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        send({'status': 'error', 'error': str(exc)})
        raise SystemExit(1)
    finally:
        if monitor is not None:
            monitor.close()


if __name__ == '__main__':
    main()
