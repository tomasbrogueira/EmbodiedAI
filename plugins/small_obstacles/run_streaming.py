"""Incremental video -> persistent LingBot/VLM/SAM -> map, in a separate plugin.

Run as python -m plugins.small_obstacles.run_streaming --help in an environment
with NumPy, OpenCV, and the original mapper's geometry dependencies.
"""
import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

import numpy as np

from .common import rgb_digest
from .propose import DEFAULT_MODEL
from .prompt_presets import instruction_settings, validate_instruction, validate_sam_prompts
from .model_session import required_roles
from .run_existing_pipeline import write_json
from .stream_state import DisplayBuffer, PromptState, StreamingPly
from .runtime_metrics import host_memory

ROOT = Path(__file__).resolve().parents[2]


class WorkerError(RuntimeError):
    def __init__(self, role, message):
        super().__init__(f'{role}: {message["error"]}; inspect {role}.log')
        self.role, self.message = role, message


class Worker:
    def __init__(self, role, python, config, directory):
        self.role, self.pending = role, False
        self.messages = queue.Queue()
        self.log = (directory / f'{role}.log').open('w')
        settings = json.loads(Path(config).read_text())
        environment = {**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONNOUSERSITE': '1'}
        if settings.get('worker_threads', 0):
            for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
                environment[name] = str(settings['worker_threads'])
        self.process = subprocess.Popen([python, str(ROOT / 'plugins/small_obstacles/stream_worker.py'),
            '--role', role, '--config', str(config)], cwd=ROOT, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1,
            env=environment)
        def read():
            try:
                for line in self.process.stdout:
                    self.messages.put(json.loads(line))
            except Exception as exc:
                self.messages.put({'status': 'error', 'error': str(exc)})
            finally:
                self.messages.put({'status': 'error', 'error': 'Worker exited; inspect its log'})
        threading.Thread(target=read, daemon=True).start()
        try:
            self.ready = self.receive(timeout=600)
        except BaseException:
            self.close()
            raise

    def submit(self, **request):
        if self.pending:
            raise RuntimeError(f'{self.role} already has an outstanding request')
        self.process.stdin.write(json.dumps(request) + '\n')
        self.process.stdin.flush()
        self.pending = True

    def receive(self, timeout=300):
        try:
            message = self.messages.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f'{self.role} timed out; inspect {self.role}.log')
        self.pending = False
        if message['status'] == 'error':
            raise WorkerError(self.role, message)
        return message

    def poll(self):
        return self.receive(timeout=0) if not self.messages.empty() else None

    def call(self, **request):
        self.submit(**request)
        return self.receive()

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"op":"close"}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        self.log.close()


def preprocess(rgb, image_size=518):
    """Match the existing batch demo's OpenCV resize and center crop exactly."""
    import cv2
    h, w = rgb.shape[:2]
    new_h = max(14, round(h * (image_size / w) / 14) * 14)
    rgb = cv2.resize(rgb, (image_size, new_h), interpolation=cv2.INTER_CUBIC)
    if new_h > image_size:
        start = (new_h - image_size) // 2
        rgb = rgb[start:start + image_size]
    return np.ascontiguousarray(rgb)


def project_frame(prediction, mask):
    """Same original mask alignment, geometry helper, and validity thresholds."""
    sys.path.insert(0, str(ROOT / 'orchestrator'))
    from io_utils import load_depth_map, load_intrinsic, load_extrinsic, load_rgb
    from lingbot_geometry import depth_to_world_points
    from sam_masks import preprocess_mask
    depth = load_depth_map(prediction['depth'])
    conf = load_depth_map(prediction['depth_conf'])
    rgb = load_rgb(prediction['images'])
    if depth.shape != conf.shape or depth.shape != rgb.shape[:2] or mask.shape != depth.shape:
        raise ValueError('Geometry, RGB and current-frame mask must have matching dimensions')
    aligned = preprocess_mask(mask, depth.shape, 14)
    if not np.array_equal(aligned, mask):
        raise ValueError('Original patch-size 14 preprocessing would change this mask')
    points = depth_to_world_points(depth, load_intrinsic(prediction['intrinsic']), load_extrinsic(prediction['extrinsic']))
    valid = (np.isfinite(points).all(axis=-1) & np.isfinite(depth) & np.isfinite(conf)
             & (depth > 0) & (depth < 25) & (conf > 2.5))
    colors = rgb[valid].copy()
    labels = aligned[valid]
    colors[labels] = [255, 0, 0]
    return points[valid].astype(np.float32), colors, int(labels.sum())


def video_frames(args, info):
    """Bounded decoding: lossless sampled input or paced latest-frame playback."""
    import cv2
    cap = cv2.VideoCapture(str(args.video_path))
    if not cap.isOpened():
        raise ValueError(f'Cannot open {args.video_path}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError('Video has no valid FPS metadata')
    requested = getattr(args, 'frame_count', 0)
    source_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if requested and (args.realtime or args.max_frames or requested < 2):
        cap.release()
        raise ValueError('Frame-count mode requires at least 2 frames, no realtime pacing and no max-frames limit')
    if requested and source_count < 2:
        cap.release()
        raise ValueError('Frame-count mode requires valid metadata for at least 2 source frames')
    target_count = min(requested, source_count) if requested else None
    if target_count and target_count > args.max_frame_num:
        cap.release()
        raise ValueError(f'Selected {target_count} frames exceed max-frame-num ({args.max_frame_num})')
    stride = max(1, round(fps / args.input_fps)) if args.input_fps else 1
    info.update(source_fps=fps, source_frame_count=source_count,
                input_stride=stride, nominal_input_fps=fps / stride, dropped_sampled_frames=0)
    if requested:
        info.update(sampling_mode='offline frame count', requested_frame_count=requested,
                    target_geometry_frames=target_count, input_stride=None,
                    nominal_input_fps=target_count*fps/source_count)
    index, next_index, emitted = 0, 0, 0
    started = None
    try:
        while not args.max_frames or emitted < args.max_frames:
            if target_count is not None and emitted >= target_count:
                break
            target = next_index
            if target_count is not None:
                # Uniform unique indices, including both ends, decoded in order.
                target = emitted*(source_count-1)//(target_count-1)
            if args.realtime and emitted >= 2:
                if started is None:
                    started = time.perf_counter() - next_index / fps
                target = max(target, int((time.perf_counter() - started) * fps / stride) * stride)
                delay = started + target / fps - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            if target_count is None:
                info['dropped_sampled_frames'] += max(0, (target - next_index) // stride)
            while index < target:
                if not cap.grab():
                    if target_count is not None:
                        raise ValueError(f'Video ended before selected frame {target}; decoded {emitted}/{target_count}')
                    return
                index += 1
            ok, frame = cap.read()
            if not ok:
                if target_count is not None:
                    raise ValueError(f'Cannot decode selected frame {target}; decoded {emitted}/{target_count}')
                return
            captured = time.perf_counter()
            rgb = preprocess(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), args.image_size)
            yield {'source_index': index, 'source_time': index / fps, 'captured': captured, 'rgb': rgb}
            index += 1
            next_index = target + stride
            emitted += 1
    finally:
        cap.release()


class RunCancelled(Exception):
    pass


def run(args, model_session=None, stop_event=None, output_stream=None):
    manual_prompts = getattr(args, 'manual_sam_prompts', None)
    if manual_prompts is not None:
        manual_prompts = validate_sam_prompts(manual_prompts, args.max_prompts)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    work = args.output_dir / 'work'
    work.mkdir()
    config = {**vars(args), **instruction_settings(args)}
    config.update(proposal_mode='manual' if manual_prompts is not None else 'vlm',
                  manual_sam_prompts=manual_prompts)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    config_path = args.output_dir / 'config.json'
    write_json(config_path, config)
    workers = []
    writer = None
    events = (args.output_dir / 'events.jsonl').open('w')
    state = {'status': 'loading', 'geometry_mode': 'causal streaming with existing model and KV cache',
             'geometry_frames': 0, 'mapped_frames': 0, 'vlm_calls': 0, 'invalid_proposals': 0,
             'points': 0, 'marked_points': 0, 'mapping_frame_step': args.map_step,
             'realtime': args.realtime, 'requested_frame_count': getattr(args, 'frame_count', 0),
             'display_mode': 'uniform sample of the whole accumulated map',
             'kv_cache_window': args.kv_cache_window, 'gpu_memory': {},
             'vlm_prompt_name': config['vlm_prompt_name'],
             'vlm_instruction_sha256': config['vlm_instruction_sha256'],
             'proposal_mode': config['proposal_mode'], 'manual_sam_prompts': manual_prompts,
             'vlm_skipped': manual_prompts is not None}
    started = time.perf_counter()
    running_started = None
    frames = None
    session_started = False
    prompts = PromptState(args.prompt_ttl, args.max_prompts)
    display = DisplayBuffer(args.display_points)
    from .scene_pose import PoseHistory
    pose_history = PoseHistory()

    def event(value):
        events.write(json.dumps(value) + '\n')
        events.flush()

    def status():
        state['controller_host_memory'] = host_memory()
        snapshots = list(state['gpu_memory'].values())
        if snapshots:
            peak = max(m['device_sampled_peak_used_bytes'] for m in snapshots)
            state['gpu_sampled_peak_gib'] = round(peak / 2**30, 3)
            state['gpu_min_sampled_free_gib'] = round((snapshots[-1]['device_total_bytes']-peak) / 2**30, 3)
        state['wall_seconds'] = time.perf_counter() - started
        if running_started is not None:
            elapsed = time.perf_counter() - running_started
            state.update(processing_seconds=elapsed, geometry_hz=state['geometry_frames'] / max(elapsed, 1e-9),
                         mapping_hz=state['mapped_frames'] / max(elapsed, 1e-9))
        write_json(args.output_dir / 'status.json', state)

    def memory(role, result, source_indices=None):
        if 'memory' in result:
            state['gpu_memory'][role] = result['memory']
        if result.get('text_cache') is not None:
            state['sam_text_cache'] = result['text_cache']
        event({'type': 'worker_memory', 'role': role,
               'source_indices': source_indices,
               **{key: result[key] for key in ('memory', 'memory_before', 'host_memory', 'cache',
                                              'text_cache', 'cuda_cache_released') if key in result}})

    viewer = None
    vlm_source = None
    last_vlm_time = -float('inf')
    last_display = -float('inf')
    try:
        if args.serve:
            viewer_log = (args.output_dir / 'viewer.log').open('w')
            viewer = subprocess.Popen([args.viewer_python, str(ROOT / 'plugins/small_obstacles/view_stream.py'),
                str(args.output_dir), '--port', str(args.port)], cwd=ROOT, stdout=viewer_log, stderr=subprocess.STDOUT)
            viewer_log.close()
            state['viewer_pid'] = viewer.pid
            state['viewer_port'] = args.port
        status()
        state['worker_pids'] = {}
        state['models_reused'] = False
        models = {}
        if manual_prompts is not None:
            state['vlm_load_seconds'] = 0.
            event({'type': 'manual_sam_prompts', 'prompts': manual_prompts})

        def loading(role):
            state['loading_model'] = role
            status()

        def ready(role, worker, result):
            workers.append(worker)
            models[role] = worker
            state[role + '_load_seconds'] = result['load_seconds']
            state['worker_pids'][role] = worker.process.pid
            state['models_reused'] = state['models_reused'] or result.get('reused', False)
            memory(role, result)
            event({'type': 'model_ready', 'role': role, 'process_id': worker.process.pid,
                   'reused': result.get('reused', False), 'load_seconds': result['load_seconds']})
            print(f'{role} {"reused" if result.get("reused") else "loaded"} in {result["load_seconds"]:.2f}s',
                  flush=True, file=output_stream)

        if model_session is None:
            for role in required_roles(config):
                loading(role)
                worker = Worker(role, getattr(args, role + '_python'), config_path, args.output_dir)
                ready(role, worker, worker.ready)
        else:
            model_session.begin(config_path, args.output_dir, loading, ready)
            session_started = True
            state['model_session_id'] = model_session.id
        lingbot, vlm, sam = models['lingbot'], models.get('vlm'), models['sam']
        frames = iter(video_frames(args, state))
        first = []
        for _ in range(2):
            try:
                first.append(next(frames))
            except StopIteration:
                break
        if not first:
            raise ValueError('Video contains no decodable frames')
        state.update(status='running', loading_model=None)
        writer = StreamingPly(args.output_dir / 'obstacle_map.ply')
        running_started = time.perf_counter()
        sequence = 0

        def accept_proposal(result):
            nonlocal vlm_source
            state['vlm_calls'] += 1
            if 'proposal_error' in result:
                state['invalid_proposals'] += 1
            else:
                prompts.observe(result['prompts'], vlm_source['source_time'])
            event({'type': 'vlm', **vlm_source, **result})
            state['last_vlm_source_time'] = vlm_source['source_time']
            memory('vlm', result, [vlm_source['source_index']])
            vlm_source = None

        batch = first
        while batch:
            if stop_event is not None and stop_event.is_set():
                raise RunCancelled()
            loop_start = time.perf_counter()
            np.savez(work / 'lingbot_input.npz', rgb=np.stack([f['rgb'] for f in batch]))
            state['pending_geometry_source_indices'] = [frame['source_index'] for frame in batch]
            geometry = lingbot.call(op='frame', input=str(work / 'lingbot_input.npz'), output=str(work / 'geometry.npz'))
            state['pending_geometry_source_indices'] = None
            memory('lingbot', geometry, [frame['source_index'] for frame in batch])
            with np.load(work / 'geometry.npz', allow_pickle=False) as data:
                predictions = {key: data[key].copy() for key in data.files}
            for i, frame in enumerate(batch):
                if stop_event is not None and stop_event.is_set():
                    raise RunCancelled()
                state['geometry_frames'] += 1
                state['last_geometry_source_index'] = frame['source_index']
                state['last_geometry_source_time'] = frame['source_time']
                event({'type': 'geometry_frame', 'sequence_index': sequence,
                       'source_index': frame['source_index'], 'source_time': frame['source_time'],
                       'rgb_sha256': rgb_digest(frame['rgb'])})
                state['camera_pose'] = pose_history.observe(
                    {key: value[i] for key,value in predictions.items()},
                    frame['source_index'], frame['source_time'])
                # Pose also advances on geometry frames that are not mapped.
                if sequence % args.map_step:
                    status()
                if vlm is not None and vlm.pending:
                    result = vlm.poll()
                    if result is not None:
                        accept_proposal(result)
                if sequence % args.map_step == 0:
                    rgb = frame['rgb']
                    if vlm is not None and not vlm.pending and frame['source_time'] - last_vlm_time >= args.vlm_interval:
                        np.savez(work / 'vlm_input.npz', rgb=rgb)
                        vlm_source = {'source_index': frame['source_index'], 'source_time': frame['source_time'],
                                      'rgb_sha256': rgb_digest(rgb)}
                        vlm.submit(op='frame', input=str(work / 'vlm_input.npz'))
                        last_vlm_time = frame['source_time']
                        # Bootstrap concepts once; later refreshes do not block geometry.
                        if state['mapped_frames'] == 0:
                            accept_proposal(vlm.receive())
                    active = list(manual_prompts) if manual_prompts is not None else prompts.active(frame['source_time'])
                    mask = np.zeros(rgb.shape[:2], dtype=bool)
                    sam_seconds = 0.
                    if active:
                        np.savez(work / 'sam_input.npz', rgb=rgb)
                        result = sam.call(op='frame', input=str(work / 'sam_input.npz'),
                                          output=str(work / 'mask.npz'), prompts=active)
                        sam_seconds = result['elapsed_seconds']
                        memory('sam', result, [frame['source_index']])
                        with np.load(work / 'mask.npz', allow_pickle=False) as data:
                            mask = data['out_binary_masks'].copy()
                    prediction = {key: value[i] for key, value in predictions.items()}
                    mapped_at = time.perf_counter()
                    points, colors, marked = project_frame(prediction, mask)
                    writer.append(points, colors)
                    display.append(points, colors)
                    state.update(mapped_frames=state['mapped_frames'] + 1, points=writer.count,
                        marked_points=state['marked_points'] + marked, source_index=frame['source_index'],
                        source_time=frame['source_time'], prompts=active)
                    if state['mapped_frames'] == 1:
                        state['first_map_wall_seconds'] = time.perf_counter() - started
                        state['first_map_processing_seconds'] = time.perf_counter() - running_started
                    entry = {'type': 'frame', 'sequence_index': sequence, 'source_index': frame['source_index'],
                        'source_time': frame['source_time'], 'rgb_sha256': rgb_digest(rgb), 'prompts': active,
                        'vlm_source_time': state.get('last_vlm_source_time'), 'mask_pixels': int(mask.sum()),
                        'points': len(points), 'marked_points': marked, 'sam_seconds': sam_seconds,
                        'lingbot_batch_seconds': geometry['elapsed_seconds'],
                        'projection_seconds': time.perf_counter() - mapped_at,
                        'capture_to_map_seconds': time.perf_counter() - frame['captured']}
                    event(entry)
                    if args.save_frames:
                        name = f'frame_{frame["source_index"]:06d}.npz'
                        for directory in ('predictions', 'masks'):
                            (args.output_dir / directory).mkdir(exist_ok=True)
                        np.savez(args.output_dir / 'predictions' / name, **prediction)
                        np.savez_compressed(args.output_dir / 'masks' / name, out_binary_masks=mask,
                                            rgb_sha256=np.asarray(rgb_digest(rgb)))
                    if time.perf_counter() - last_display >= args.display_interval:
                        points_view, colors_view = display.arrays()
                        temporary = work / 'display.npz'
                        np.savez(temporary, points=points_view, colors=colors_view)
                        temporary.replace(args.output_dir / 'display.npz')
                        last_display = time.perf_counter()
                    status()
                    print(f'frame {frame["source_index"]} t={frame["source_time"]:.2f}s '
                          f'map={state["mapping_hz"]:.2f}Hz prompts={active} marked={marked}', flush=True, file=output_stream)
                sequence += 1
            event({'type': 'geometry_batch', 'frames': len(batch), 'seconds': geometry['elapsed_seconds'],
                   'loop_seconds': time.perf_counter() - loop_start,
                   'source_indices': [frame['source_index'] for frame in batch], 'cache': geometry.get('cache')})
            try:
                batch = [next(frames)]
            except StopIteration:
                batch = []
        if vlm is not None and vlm.pending:
            accept_proposal(vlm.receive())
        points_view, colors_view = display.arrays()
        np.savez(work / 'display.npz', points=points_view, colors=colors_view)
        (work / 'display.npz').replace(args.output_dir / 'display.npz')
        state['status'] = 'complete'
    except RunCancelled:
        state['status'] = 'stopped'
        points_view, colors_view = display.arrays()
        np.savez(work / 'display.npz', points=points_view, colors=colors_view)
        (work / 'display.npz').replace(args.output_dir / 'display.npz')
    except BaseException as exc:
        state.update(status='failed', error=str(exc))
        if isinstance(exc, WorkerError):
            memory(exc.role, exc.message)
            event({'type': 'worker_error', 'role': exc.role, **exc.message})
        raise
    finally:
        status()
        if frames is not None:
            frames.close()
        if writer is not None:
            writer.close()
        try:
            if model_session is not None and session_started:
                for role, result in model_session.finish(state['status'] in ('complete', 'stopped')).items():
                    event({'type': 'sequence_released', 'role': role, **result})
                state['models_resident'] = model_session.loaded
                # Keep processing Hz separate from between-video cleanup.
                write_json(args.output_dir / 'status.json', state)
            elif model_session is None:
                for worker in reversed(workers):
                    worker.close()
        finally:
            events.close()
        # The viewer remains available after EOF, just like the original viewer.
        print(json.dumps(state, indent=2), flush=True, file=output_stream)
    return viewer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video-path', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory; existing outputs are never overwritten')
    for role in ('lingbot', 'vlm', 'sam', 'viewer'):
        parser.add_argument('--' + role + '-python', default=os.environ.get(role.upper() + '_PYTHON', sys.executable))
    parser.add_argument('--lingbot-checkpoint', type=Path, default=ROOT / 'lingbot-map/lingbot-map.pt')
    parser.add_argument('--sam-checkpoint', default=None)
    parser.add_argument('--vlm-model', default=DEFAULT_MODEL)
    parser.add_argument('--vlm-instruction-file', type=Path,
                        help='UTF-8 file containing a custom VLM instruction; keep the JSON prompts response contract')
    parser.add_argument('--vlm-prompt-name', help='Label recorded alongside the exact instruction in each run')
    parser.add_argument('--sam-prompts', nargs='+', dest='manual_sam_prompts',
                        help='Skip Qwen and use these SAM object names on every mapped frame')
    parser.add_argument('--input-fps', type=float, default=4., help='Sample input near this FPS; 0 retains every frame')
    parser.add_argument('--frame-count', type=int, default=0,
                        help='Offline: select this many unique frames across the whole video (capped at available frames); overrides input-fps, no dropping')
    parser.add_argument('--realtime', action='store_true', help='Pace video and skip late sampled frames instead of accumulating lag')
    parser.add_argument('--map-step', type=int, default=2, help='Map every Nth retained geometry frame, as in original settings')
    parser.add_argument('--vlm-interval', type=float, default=1., help='Minimum seconds of source-video time between proposal requests')
    parser.add_argument('--prompt-ttl', type=float, default=5., help='Expire concepts by their proposal source-video timestamp')
    parser.add_argument('--max-prompts', type=int, default=8)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--image-size', type=int, default=518)
    parser.add_argument('--camera-iterations', type=int, default=4, help='Original default 4; 1 is an opt-in speed/quality tradeoff')
    parser.add_argument('--max-frame-num', type=int, default=1024)
    parser.add_argument('--kv-cache-window', type=int, default=64,
                        help='Recent patch keyframes kept by existing LingBot eviction; smaller values change temporal context')
    parser.add_argument('--memory-sample-interval', type=float, default=.1,
                        help='Seconds between device-wide memory samples per worker; 0 disables background traces')
    parser.add_argument('--worker-threads', type=int, default=0,
                        help='CPU/BLAS threads per model worker; 0 retains environment defaults')
    parser.add_argument('--cuda-cache-policy', choices=['retain', 'release', 'budget'], default='retain',
                        help='Retain unused blocks, release after requests, or release when device free memory is low')
    parser.add_argument('--cuda-cache-min-free-gib', type=float, default=2.,
                        help='Free-memory threshold for budget policy; not a guarantee of transient peak headroom')
    parser.add_argument('--sam-text-cache', action='store_true',
                        help='Optional cached SAM text features plus a single GPU union of native masks')
    parser.add_argument('--sam-text-cache-size', type=int, default=16)
    parser.add_argument('--max-frames', type=int, default=0, help='Limit retained input frames for a smoke run')
    parser.add_argument('--save-frames', action='store_true', help='Save predictions and authoritative masks for mapped frames')
    parser.add_argument('--serve', action='store_true', help='Start the live Viser viewer; remains available after EOF')
    parser.add_argument('--port', type=int, default=8082)
    parser.add_argument('--display-points', type=int, default=200_000)
    parser.add_argument('--display-interval', type=float, default=.5)
    args = parser.parse_args(argv)
    if args.frame_count < 0 or args.frame_count == 1:
        parser.error('Use frame-count 0 for FPS sampling, or at least 2 for offline frame-count mode')
    if args.frame_count and (args.realtime or args.max_frames):
        parser.error('Frame-count mode cannot be combined with realtime or max-frames')
    args.vlm_instruction = None
    if args.vlm_instruction_file is not None:
        args.vlm_instruction_file = args.vlm_instruction_file.resolve()
        try:
            args.vlm_instruction = validate_instruction(args.vlm_instruction_file.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    if args.manual_sam_prompts is not None:
        try:
            args.manual_sam_prompts = validate_sam_prompts(args.manual_sam_prompts, args.max_prompts)
        except ValueError as exc:
            parser.error(str(exc))
    if args.input_fps < 0 or args.map_step < 1 or args.vlm_interval <= 0 or args.prompt_ttl <= 0:
        parser.error('Use non-negative input FPS, positive frame step, VLM interval and prompt TTL')
    if any(value <= 0 for value in (args.max_prompts, args.max_new_tokens, args.camera_iterations,
                                   args.max_frame_num, args.kv_cache_window, args.sam_text_cache_size,
                                   args.display_points, args.display_interval)):
        parser.error('Limits and display interval must be positive')
    if args.max_frames < 0 or args.image_size < 14 or args.image_size % 14 or not 1 <= args.port <= 65535:
        parser.error('Use non-negative max-frames, image size divisible by 14 and a valid port')
    if args.worker_threads < 0 or not np.isfinite(args.memory_sample_interval) or args.memory_sample_interval < 0:
        parser.error('Use non-negative worker threads and a finite non-negative memory sample interval')
    if not np.isfinite(args.cuda_cache_min_free_gib) or args.cuda_cache_min_free_gib <= 0:
        parser.error('Use a finite positive CUDA free-memory threshold')
    args.video_path = args.video_path.resolve()
    args.output_dir = args.output_dir.resolve()
    args.lingbot_checkpoint = args.lingbot_checkpoint.resolve()
    if not args.video_path.is_file() or not args.lingbot_checkpoint.is_file():
        parser.error('Video and LingBot checkpoint must exist')
    if args.output_dir.exists():
        parser.error('Choose a new output directory')
    return args


def main():
    args = parse_args()
    viewer = run(args)
    if viewer is not None:
        print('Mapping complete; keeping Viser available. Press Ctrl+C to stop the viewer.', flush=True)
        try:
            while viewer.poll() is None:
                time.sleep(1)
        except KeyboardInterrupt:
            viewer.terminate()
            viewer.wait(timeout=10)


if __name__ == '__main__':
    main()
