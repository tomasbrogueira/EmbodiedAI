"""Read-only diagnostic benchmarks; does not alter the streaming pipeline.

Run each role in its existing Python environment. Results are isolated model
timings, not end-to-end throughput or evidence of perception accuracy.
"""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def stats(values):
    return dict(n=len(values), median_ms=1000 * statistics.median(values),
                p90_ms=1000 * float(np.percentile(values, 90)),
                min_ms=1000 * min(values), max_ms=1000 * max(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', choices=['lingbot', 'cache', 'sam', 'vlm', 'cpu'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'data/vlm_maps/streaming_final/config.json')
    parser.add_argument('--frame', type=Path, default=ROOT / 'data/vlm_maps/streaming_final/predictions/frame_000042.npz')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    with np.load(args.frame) as data:
        prediction = {k: data[k].copy() for k in data.files}
    rgb = prediction['images'].squeeze()
    if rgb.shape[0] == 3:
        rgb = rgb.transpose(1, 2, 0)
    if rgb.dtype != np.uint8:
        rgb = np.round(rgb * 255).astype(np.uint8)
    result = {'role': args.role, 'frame': str(args.frame), 'rgb_shape': list(rgb.shape),
              'clock': 'perf_counter; CUDA synchronized at diagnostic stage boundaries',
              'measurements': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + '\n')

    def record(**entry):
        result['measurements'].append(entry)
        save()
        print(json.dumps(entry), flush=True)

    if args.role != 'cpu':
        import torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(4)

        def timed(fn):
            torch.cuda.synchronize()
            t = time.perf_counter()
            value = fn()
            torch.cuda.synchronize()
            return value, time.perf_counter() - t

        def memory():
            return {'allocated_gib': torch.cuda.memory_allocated() / 2**30,
                    'reserved_gib': torch.cuda.memory_reserved() / 2**30,
                    'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30}

        result['runtime'] = {'torch': torch.__version__, 'cuda_build': torch.version.cuda,
            'device': str(torch.cuda.get_device_properties(0)), 'interop_threads': 4}
        from PIL import Image
        image = Image.fromarray(rgb)

    if args.role == 'sam':
        from plugins.small_obstacles.segment import SamSegmenter, union_masks
        t = time.perf_counter()
        with redirect_stdout(sys.stderr):
            sam = SamSegmenter('cuda', .5, config['sam_checkpoint'])
        result['load_seconds_threads4'] = time.perf_counter() - t
        prompts = ['log', 'bag', 'stone block', 'puddle', 'cable', 'bottle', 'open hole', 'branch']
        for threads in [112, 4, 1, 8]:
            torch.set_num_threads(threads)
            for count in [1, 3, 8]:
                rows = []
                for repeat in range(6):
                    _, image_seconds = timed(lambda: sam.set_image(image))
                    mask = np.zeros(rgb.shape[:2], dtype=bool)
                    query_seconds = []
                    for prompt in prompts[:count]:
                        output, seconds = timed(lambda: sam.masks(prompt))
                        mask |= union_masks(output, mask.shape)
                        query_seconds.append(seconds)
                    rows.append({'image': image_seconds, 'queries': sum(query_seconds),
                                 'total': image_seconds + sum(query_seconds), 'mask_pixels': int(mask.sum())})
                record(threads=threads, prompts=count, image=stats([r['image'] for r in rows[2:]]),
                       queries=stats([r['queries'] for r in rows[2:]]),
                       total=stats([r['total'] for r in rows[2:]]), raw=rows, memory=memory())
        torch.set_num_threads(4)
        # Test text-feature caching and GPU union without changing production APIs.
        cache = {}
        with torch.inference_mode(), sam._autocast():
            for prompt in prompts:
                cache[prompt] = sam.processor.model.backbone.forward_text([prompt], device='cuda')
        for count in [1, 3, 8]:
            rows = []
            for repeat in range(6):
                sam.set_image(image)
                torch.cuda.synchronize()
                reference = np.zeros(rgb.shape[:2], dtype=bool)
                t = time.perf_counter()
                for prompt in prompts[:count]:
                    reference |= union_masks(sam.masks(prompt), reference.shape)
                torch.cuda.synchronize()
                baseline = time.perf_counter() - t

                def cached_union():
                    mask = torch.zeros(rgb.shape[:2], dtype=torch.bool, device='cuda')
                    with torch.inference_mode(), sam._autocast():
                        for prompt in prompts[:count]:
                            sam.processor.reset_all_prompts(sam.state)
                            sam.state['backbone_out'].update(cache[prompt])
                            sam.state['geometric_prompt'] = sam.processor.model._get_dummy_prompt()
                            output = sam.processor._forward_grounding(sam.state)['masks']
                            mask |= output[:, 0].any(dim=0)
                    return mask.cpu().numpy()

                candidate, candidate_seconds = timed(cached_union)
                rows.append({'baseline': baseline, 'cached_gpu_union': candidate_seconds,
                             'bit_exact': bool(np.array_equal(reference, candidate)),
                             'mask_pixels': int(candidate.sum())})
            record(experiment='text_cache_plus_gpu_union', prompts=count,
                   baseline=stats([r['baseline'] for r in rows[2:]]),
                   candidate=stats([r['cached_gpu_union'] for r in rows[2:]]), raw=rows)
        from plugins.small_obstacles.common import read_rgb
        validation = []
        for proposal_path in sorted((ROOT / 'data/vlm_maps/timed_run').glob('*/*/prompts.json')):
            proposal = json.loads(proposal_path.read_text())
            frame_paths = sorted((proposal_path.parent / 'lingbot').glob('*/frame_*.npz'))
            if not frame_paths:
                continue
            validation_rgb = read_rgb(frame_paths[0])
            concepts = proposal['prompts'][:8] or ['log']
            sam.set_image(Image.fromarray(validation_rgb))
            reference = np.zeros(validation_rgb.shape[:2], dtype=bool)
            for concept in concepts:
                reference |= union_masks(sam.masks(concept), reference.shape)
            candidate = torch.zeros(validation_rgb.shape[:2], dtype=torch.bool, device='cuda')
            with torch.inference_mode(), sam._autocast():
                for concept in concepts:
                    if concept not in cache:
                        cache[concept] = sam.processor.model.backbone.forward_text([concept], device='cuda')
                    sam.processor.reset_all_prompts(sam.state)
                    sam.state['backbone_out'].update(cache[concept])
                    sam.state['geometric_prompt'] = sam.processor.model._get_dummy_prompt()
                    output = sam.processor._forward_grounding(sam.state)['masks']
                    candidate |= output[:, 0].any(dim=0)
            actual = candidate.cpu().numpy()
            validation.append({'video': str(proposal_path.parent.relative_to(ROOT)),
                'frame': frame_paths[0].name, 'prompts': concepts,
                'mask_pixels': int(actual.sum()), 'bit_exact': bool(np.array_equal(reference, actual))})
        result['cache_union_validation'] = validation

    elif args.role == 'vlm':
        from plugins.small_obstacles.propose import QwenProposer
        t = time.perf_counter()
        with redirect_stdout(sys.stderr):
            qwen = QwenProposer(config['vlm_model'], max_new_tokens=128)
        result['load_seconds_threads4'] = time.perf_counter() - t
        result['attention'] = {name: getattr(value, '_attn_implementation', None)
            for name, value in [('root', qwen.model.config),
                ('text', qwen.model.config.text_config), ('vision', qwen.model.config.vision_config)]}
        result['dtype'] = str(qwen.model.dtype)
        result['parameter_count'] = sum(p.numel() for p in qwen.model.parameters())
        for threads in [112, 4, 1, 8]:
            torch.set_num_threads(threads)
            rows = []
            for repeat in range(4):
                messages = [{'role': 'user', 'content': [
                    {'type': 'image', 'image': image}, {'type': 'text', 'text': config['instruction']}]}]
                inputs, pre = timed(lambda: qwen.processor.apply_chat_template(messages, tokenize=True,
                    add_generation_prompt=True, return_dict=True, return_tensors='pt').to(qwen.model.device))
                with torch.inference_mode():
                    output, generation = timed(lambda: qwen.model.generate(**inputs,
                        max_new_tokens=128, do_sample=False))
                raw = qwen.processor.batch_decode(output[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
                rows.append({'preprocess': pre, 'generation': generation,
                    'input_tokens': inputs['input_ids'].shape[1],
                    'image_grid_thw': inputs['image_grid_thw'].tolist(),
                    'output_tokens': output.shape[1] - inputs['input_ids'].shape[1], 'response': raw})
            record(threads=threads, preprocessing=stats([r['preprocess'] for r in rows[1:]]),
                generation=stats([r['generation'] for r in rows[1:]]), raw=rows, memory=memory())

    elif args.role == 'lingbot':
        from plugins.small_obstacles.stream_worker import LingBotStream
        config['camera_iterations'] = 4
        t = time.perf_counter()
        with redirect_stdout(sys.stderr):
            runner = LingBotStream(config)
        result['load_seconds_threads4'] = time.perf_counter() - t
        # Repeated same-image input deliberately isolates cache/compute cost.
        # These artificial trajectories provide no geometry quality evidence.
        stages = {}
        original_forward, original_post = runner.model.forward, runner.postprocess

        def forward(*a, **kw):
            value, seconds = timed(lambda: original_forward(*a, **kw))
            stages['forward'] = seconds
            return value

        def post(*a, **kw):
            value, seconds = timed(lambda: original_post(*a, **kw))
            stages['postprocess'] = seconds
            return value

        runner.model.forward, runner.postprocess = forward, post
        with tempfile.TemporaryDirectory() as temporary:
            inp, out = str(Path(temporary) / 'input.npz'), str(Path(temporary) / 'output.npz')
            for iterations in [4, 1]:
                runner.model.camera_head.num_iterations = iterations
                for threads in [112, 4, 1, 8]:
                    torch.set_num_threads(threads)
                    runner.model.clean_kv_cache()
                    runner.count = 0
                    np.savez(inp, rgb=np.stack([rgb, rgb]))
                    with redirect_stdout(sys.stderr):
                        _, bootstrap = timed(lambda: runner.frame({'input': inp, 'output': out}))
                    np.savez(inp, rgb=rgb[None])
                    rows = []
                    for i in range(16):
                        with redirect_stdout(sys.stderr):
                            _, total = timed(lambda: runner.frame({'input': inp, 'output': out}))
                        rows.append({'total': total, **stages, 'sequence': i + 2})
                    record(threads=threads, camera_iterations=iterations, bootstrap_seconds=bootstrap,
                        total=stats([r['total'] for r in rows[4:]]),
                        forward=stats([r['forward'] for r in rows[4:]]),
                        postprocess=stats([r['postprocess'] for r in rows[4:]]), raw=rows, memory=memory())
            torch.set_num_threads(4)
            runner.model.camera_head.num_iterations = 4
            runner.model.clean_kv_cache()
            runner.count = 0
            np.savez(inp, rgb=np.stack([rgb, rgb]))
            with redirect_stdout(sys.stderr):
                runner.frame({'input': inp, 'output': out})
            np.savez(inp, rgb=rgb[None])
            rows = []
            for i in range(160):
                with redirect_stdout(sys.stderr):
                    _, total = timed(lambda: runner.frame({'input': inp, 'output': out}))
                rows.append({'total': total, **stages, 'sequence': i + 2, 'memory': memory()})
            for start, stop in [(4, 20), (60, 80), (140, 160)]:
                record(experiment='cache_growth', start_sequence=start + 2, stop_sequence=stop + 2,
                    total=stats([r['total'] for r in rows[start:stop]]), raw=rows[start:stop],
                    memory=rows[stop - 1]['memory'])
            try:
                from torch.profiler import profile, ProfilerActivity
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    with redirect_stdout(sys.stderr):
                        runner.frame({'input': inp, 'output': out})
                result['attention_profiler_events'] = [
                    {'name': e.key, 'calls': e.count, 'device_time_us': e.device_time_total}
                    for e in prof.key_averages() if any(s in e.key.lower()
                        for s in ['attention', 'flash', 'efficient', 'softmax'])]
            except Exception as exc:
                result['profiler_error'] = repr(exc)
        runner.model.clean_kv_cache()
        # Small kernel probe tests installed compiler/driver, not whole models.
        try:
            f = torch.compile(lambda x: x.sin() * x + 1, fullgraph=True)
            x = torch.rand((512, 512), device='cuda')
            value, elapsed = timed(lambda: f(x))
            result['compiler_smoke'] = {'ok': bool(torch.allclose(value, x.sin() * x + 1)),
                                        'first_call_seconds': elapsed}
        except Exception as exc:
            result['compiler_smoke'] = {'ok': False, 'error': repr(exc)}

    elif args.role == 'cache':
        from plugins.small_obstacles.stream_worker import LingBotStream
        config['camera_iterations'] = 4
        with redirect_stdout(sys.stderr):
            runner = LingBotStream(config)

        def describe_cache():
            cache = runner.model.aggregator.kv_cache
            by_dtype = {}
            for key, tensor in cache.items():
                if isinstance(tensor, torch.Tensor):
                    name = str(tensor.dtype)
                    by_dtype[name] = by_dtype.get(name, 0) + tensor.numel() * tensor.element_size()
            return {'bytes_by_dtype': by_dtype, 'examples': {key: {
                'shape': list(cache[key].shape), 'dtype': str(cache[key].dtype)}
                for key in ['k_0', 'v_0'] if isinstance(cache.get(key), torch.Tensor)}}

        def cast_cache():
            # Exploratory storage experiment only. Attention still runs through
            # the original autocast SDPA implementation on the following frame.
            cache = runner.model.aggregator.kv_cache
            # Snapshot keys rather than (key, tensor) pairs: keeping all old
            # tensors alive while making replacements can double peak memory.
            for key in list(cache):
                tensor = cache[key]
                if key.startswith(('k_', 'v_')) and isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32:
                    cache[key] = tensor.to(torch.bfloat16)

        with tempfile.TemporaryDirectory() as temporary:
            inp, out = str(Path(temporary) / 'input.npz'), str(Path(temporary) / 'output.npz')
            reference_outputs = []
            for candidate in [False, True]:
                runner.model.clean_kv_cache()
                runner.count = 0
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                np.savez(inp, rgb=np.stack([rgb, rgb]))
                with redirect_stdout(sys.stderr):
                    runner.frame({'input': inp, 'output': out})
                if candidate:
                    cast_cache()
                result['candidate_initial_cache' if candidate else 'baseline_initial_cache'] = describe_cache()
                np.savez(inp, rgb=rgb[None])
                rows = []
                for i in range(160 if candidate else 16):
                    def step():
                        with redirect_stdout(sys.stderr):
                            runner.frame({'input': inp, 'output': out})
                        if candidate:
                            cast_cache()
                    try:
                        _, total = timed(step)
                    except RuntimeError as exc:
                        result['failure'] = {'experiment': 'bf16_cache_storage' if candidate else 'original_cache_storage',
                            'sequence': i + 2, 'error': str(exc), 'memory': memory()}
                        save()
                        raise
                    row = {'total': total, 'sequence': i + 2, 'memory': memory()}
                    if i < 16:
                        with np.load(out) as data:
                            arrays = {k: data[k].copy() for k in ('depth', 'depth_conf', 'intrinsic', 'extrinsic')}
                        if not candidate:
                            reference_outputs.append(arrays)
                        else:
                            row['bit_exact'] = {k: bool(np.array_equal(value, reference_outputs[i][k])) for k, value in arrays.items()}
                            row['max_absolute_difference'] = {k: float(np.max(np.abs(value-reference_outputs[i][k]))) for k, value in arrays.items()}
                    rows.append(row)
                    if i + 1 in [16, 80, 160]:
                        record(experiment='bf16_cache_storage' if candidate else 'original_cache_storage',
                            end_sequence=i + 3, timing=stats([r['total'] for r in rows[max(4, i-19):i+1]]),
                            memory=memory(), cache=describe_cache(), raw=rows[max(0,i-19):i+1])

    else:
        from plugins.small_obstacles.run_streaming import project_frame
        from plugins.small_obstacles.stream_state import DisplayBuffer, StreamingPly
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        t = time.perf_counter()
        points, colors, _ = project_frame(prediction, mask)
        result['first_projection_seconds'] = time.perf_counter() - t
        rows = []
        for _ in range(40):
            t = time.perf_counter()
            project_frame(prediction, mask)
            rows.append(time.perf_counter() - t)
        record(stage='projection', timing=stats(rows), points=len(points))
        buffer = DisplayBuffer()
        buffer.append(points, colors)
        rows = []
        for _ in range(40):
            t = time.perf_counter()
            buffer.append(points, colors)
            rows.append(time.perf_counter() - t)
        record(stage='display_reservoir_append', timing=stats(rows))
        for directory in [args.output.parent, Path('/dev/shm')]:
            with tempfile.TemporaryDirectory(dir=directory) as temporary:
                path = Path(temporary) / 'frame.npz'
                rows = []
                for _ in range(40):
                    t = time.perf_counter()
                    np.savez(path, **prediction)
                    with np.load(path) as data:
                        values = {k: data[k].copy() for k in data.files}
                    rows.append(time.perf_counter() - t)
                record(stage='npz_write_read', filesystem=str(directory), timing=stats(rows), bytes=path.stat().st_size)
                writer = StreamingPly(Path(temporary) / 'map.ply')
                rows = []
                for _ in range(20):
                    t = time.perf_counter()
                    writer.append(points, colors)
                    rows.append(time.perf_counter() - t)
                writer.close()
                record(stage='ply_append', filesystem=str(directory), timing=stats(rows), bytes_per_frame=len(points)*15)
        import cv2
        cap = cv2.VideoCapture(config['video_path'])
        rows = []
        while True:
            t = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            rows.append(time.perf_counter() - t)
        cap.release()
        record(stage='decode_each_source_frame', timing=stats(rows))

    save()


if __name__ == '__main__':
    main()
