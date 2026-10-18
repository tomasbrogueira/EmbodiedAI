# Support direct compatibility-script execution without environment mutation.
if __package__ in (None, ''):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
"""Optional timed VLM/SAM runner that feeds the unchanged original mapper.

Run --help for batch use. No original reconstruction or orchestration files are
modified. The adapt subcommand verifies masks before exporting mapper filenames.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

from orchestrator.paths import ROOT
FRAME_STEP = 2
MAPPING_SETTINGS = dict(frame_step=2, pixel_step=1, min_depth_conf=2.5,
                        max_depth=25, patch_size=14, segment_color='255,0,0')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def selected_frames(directory):
    numbered = []
    for path in directory.glob('frame_*.npz'):
        match = re.fullmatch(r'frame_(\d+)\.npz', path.name)
        if not match:
            raise ValueError(f'Unexpected frame filename: {path.name}')
        numbered.append((int(match[1]), path))
    return [p for i, p in sorted(numbered) if i % FRAME_STEP == 0]


def adapt(args):
    import numpy as np
    sys.path.insert(0, str(ROOT / 'orchestrator'))
    from io_utils import load_rgb, load_depth_map
    from sam_masks import preprocess_mask
    selection = json.loads(args.selection_file.read_text())['frames']
    manifest = json.loads((args.masks_dir / 'manifest.json').read_text())
    if manifest['schema_version'] != 1 or manifest['frames'] != selection:
        raise ValueError('SAM manifest does not match the selected mapping frames')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name in selection:
        if Path(name).name != name or not re.fullmatch(r'frame_\d+\.npz', name):
            raise ValueError(f'Invalid frame name: {name}')
        with np.load(args.lingbot_dir / name, allow_pickle=False) as prediction:
            rgb = load_rgb(prediction['images'])
            depth = load_depth_map(prediction['depth'])
            confidence = load_depth_map(prediction['depth_conf'])
        with np.load(args.masks_dir / name, allow_pickle=False) as data:
            mask = np.asarray(data['out_binary_masks'], dtype=bool)
            digest = hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()
            if data['rgb_sha256'].item() != digest:
                raise ValueError(f'{name}: mask/source RGB mismatch')
        if not (mask.shape == depth.shape == confidence.shape == rgb.shape[:2]):
            raise ValueError(f'{name}: mask/source dimensions mismatch')
        # Keep the original patch setting and prove its preprocessing is identity.
        if not np.array_equal(mask, preprocess_mask(mask, depth.shape, 14)):
            raise ValueError(f'{name}: original mapper preprocessing would alter the mask')
        index = int(re.search(r'(\d+)\.npz$', name)[1])
        shutil.copyfile(args.masks_dir / name,
                        args.output_dir / f'{args.video_name}_obstacles_frame{index:06d}.npz')
    print(f'Adapted {len(selection)} aligned union masks for the unchanged mapper', flush=True)


def timed_stage(record, name, command, directory):
    started = time.perf_counter()
    stage = {'name': name, 'command': command, 'started_utc': datetime.now(timezone.utc).isoformat(),
             'status': 'running', 'log': f'{name}.log'}
    record['stages'].append(stage)
    write_json(directory / 'timings.json', record)
    print(f'[{record["video"]}] START {name}', flush=True)
    lines = []
    with (directory / stage['log']).open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1,
                                   env={**os.environ, 'PYTHONUNBUFFERED': '1'})
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                lines.append(line)
                print(f'[{name}] {line.rstrip()}', flush=True)
            returncode = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    stage.update(elapsed_seconds=time.perf_counter()-started, returncode=returncode,
                 status='complete' if returncode == 0 else 'failed')
    if name == 'lingbot':
        match = re.search(r'Inference done in ([\d.]+)s \(([\d.]+) FPS\)', ''.join(lines))
        if match:
            stage['reported_inference_seconds'] = float(match[1])
            stage['reported_inference_fps'] = float(match[2])
    count = record.get('selected_frame_count') if name != 'lingbot' else record.get('prediction_frame_count')
    if count:
        stage['frames_per_wall_second'] = count / stage['elapsed_seconds']
    write_json(directory / 'timings.json', record)
    print(f'[{record["video"]}] END {name}: {stage["elapsed_seconds"]:.2f}s', flush=True)
    if returncode:
        raise RuntimeError(f'{name} failed (exit {returncode}); see {directory / stage["log"]}')


def run_video(args, video, directory):
    if directory.exists():
        raise ValueError(f'Output already exists; choose a new output root: {directory}')
    directory.mkdir(parents=True)
    start = time.perf_counter()
    record = {'video': str(video), 'started_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'running', 'mapping_settings': MAPPING_SETTINGS,
              'vlm_sampling': 'same frame indices as original mapper: index modulo 2 == 0',
              'timing_scope': 'process wall times including imports, model loading and IO; rendering disabled',
              'stages': []}
    predictions = directory / 'lingbot' / video.stem if args.reconstruct else args.lingbot_root / video.stem
    record['lingbot_dir'] = str(predictions)
    try:
        if args.reconstruct:
            timed_stage(record, 'lingbot', [args.lingbot_python,
                str(ROOT / 'lingbot-map/demo_render/batch_demo.py'), '--video_path', str(video),
                '--output_folder', str(directory / 'lingbot'), '--model_path', str(args.lingbot_checkpoint),
                '--mode', 'windowed', '--window_size', '32', '--keyframe_interval', '2',
                '--num_scale_frames', '2', '--overlap_keyframes', '8', '--use_sdpa',
                '--no_render', '--save_predictions'], directory)
            # batch_demo can catch scene failures and still exit 0: verify its result.
            result = json.loads((directory / 'lingbot/batch_results.json').read_text())
            if not result['results'] or not all(r['success'] for r in result['results']):
                raise RuntimeError('LingBot reported a failed scene; inspect lingbot.log')
        elif not predictions.is_dir():
            raise FileNotFoundError(f'Predictions missing: {predictions}')
        frames = selected_frames(predictions)
        if not frames:
            raise ValueError('No selected prediction frames')
        record['prediction_frame_count'] = len(list(predictions.glob('frame_*.npz')))
        record['selected_frame_count'] = len(frames)
        if args.reconstruct:
            s = record['stages'][0]
            s['frames_per_wall_second'] = record['prediction_frame_count'] / s['elapsed_seconds']
        selection = directory / 'selection.json'
        write_json(selection, {'frame_step': FRAME_STEP, 'frames': [p.name for p in frames]})
        # A separate read-only view lets both plugin stages select the exact modulo
        # indices used by the original mapper, even if source frame numbers have gaps.
        view = directory / 'selected_predictions'
        view.mkdir()
        for path in frames:
            (view / path.name).symlink_to(path.resolve())
        timed_stage(record, 'vlm', [args.vlm_python, '-m', 'plugins.small_obstacles', 'propose',
            '--lingbot-dir', str(view), '--output', str(directory / 'prompts.json'),
            '--model', args.vlm_model, '--keyframe-step', '1'], directory)
        prompts = json.loads((directory / 'prompts.json').read_text())['prompts']
        record['prompts'] = prompts
        record['sam_prompt_applications'] = len(prompts) * len(frames)
        timed_stage(record, 'sam', [args.sam_python, '-m', 'plugins.small_obstacles', 'segment',
            '--lingbot-dir', str(view), '--prompts', str(directory / 'prompts.json'),
            '--output-dir', str(directory / 'masks'), '--frame-step', '1'], directory)
        timed_stage(record, 'adapt_masks', [args.mapper_python, str(ROOT / 'plugins/small_obstacles/run_existing_pipeline.py'), 'adapt',
            '--lingbot-dir', str(predictions), '--masks-dir', str(directory / 'masks'),
            '--output-dir', str(directory / 'mapper_masks'), '--video-name', video.stem,
            '--selection-file', str(selection)], directory)
        timed_stage(record, 'mapping', [args.mapper_python, str(ROOT / 'orchestrator/main.py'),
            '--video-path', str(video), '--lingbot-dir', str(predictions),
            '--sam-dir', str(directory / 'mapper_masks'), '--prompt', 'obstacles',
            '--output', str(directory / 'obstacle_map.ply'), '--frame-step', '2', '--pixel-step', '1',
            '--min-depth-conf', '2.5', '--max-depth', '25', '--patch-size', '14',
            '--segment-color', '255,0,0'], directory)
        record['status'] = 'complete'
    except Exception as exc:
        record.update(status='failed', error=str(exc))
        print(f'FAILED {video}: {exc}', flush=True)
    finally:
        record['total_seconds'] = time.perf_counter()-start
        write_json(directory / 'timings.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest='stage', required=True)
    run = stages.add_parser('run', help='Timed single-video or batch pipeline')
    inputs = run.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--video-path', type=Path)
    inputs.add_argument('--videos-dir', type=Path)
    run.add_argument('--output-dir', type=Path, required=True, help='New output root; never overwrites a scene')
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument('--reconstruct', action='store_true', help='Run existing LingBot into isolated output directories')
    source.add_argument('--lingbot-root', type=Path, help='Reuse existing ROOT/VIDEO_STEM predictions; LingBot time is excluded')
    for role in ('lingbot', 'vlm', 'sam', 'mapper'):
        run.add_argument(f'--{role}-python', default=os.environ.get(f'{role.upper()}_PYTHON', sys.executable))
    run.add_argument('--lingbot-checkpoint', type=Path, default=ROOT / 'lingbot-map/lingbot-map.pt')
    run.add_argument('--vlm-model', default='Qwen/Qwen3-VL-4B-Instruct')
    run.add_argument('--dry-run', action='store_true')
    export = stages.add_parser('adapt', help='Validate and export existing mapper mask filenames')
    for key in ('lingbot-dir', 'masks-dir', 'output-dir', 'selection-file'):
        export.add_argument('--'+key, type=Path, required=True)
    export.add_argument('--video-name', required=True)
    args = parser.parse_args()
    if args.stage == 'adapt':
        adapt(args)
        return
    args.output_dir = args.output_dir.resolve()
    if args.lingbot_root: args.lingbot_root = args.lingbot_root.resolve()
    args.lingbot_checkpoint = args.lingbot_checkpoint.resolve()
    videos = [args.video_path.resolve()] if args.video_path else sorted(
        p.resolve() for p in args.videos_dir.rglob('*') if p.suffix.lower() in {'.mp4','.mov','.avi','.mkv'})
    if not videos or any(not p.is_file() for p in videos): parser.error('No input videos found')
    # Reject duplicate scene paths and existing outputs before any expensive work.
    dirs = [args.output_dir / p.parent.name / p.stem for p in videos]
    if len(set(dirs)) != len(dirs): parser.error('Duplicate scene output paths')
    if any(d.exists() for d in dirs): parser.error('Scene output already exists; use a new --output-dir')
    if args.dry_run:
        print(json.dumps({'videos':[str(p) for p in videos], 'outputs':[str(d) for d in dirs],
                          'mapping_settings':MAPPING_SETTINGS, 'vlm_frame_step':2,
                          'sam_frame_step':2, 'reconstruct':args.reconstruct},indent=2))
        return
    started = time.perf_counter()
    batch = {'settings': MAPPING_SETTINGS, 'started_utc':datetime.now(timezone.utc).isoformat(), 'videos':[]}
    for video,directory in zip(videos,dirs):
        batch['videos'].append(run_video(args,video,directory))
        batch['elapsed_seconds'] = time.perf_counter()-started
        write_json(args.output_dir / 'batch_timings.json', batch)
        if __package__:
            from orchestrator.timing_report import report
        else:
            from timing_report import report
        report(args.output_dir)
    failed = sum(r['status'] != 'complete' for r in batch['videos'])
    print(f'Completed {len(videos)-failed}/{len(videos)} videos in {batch["elapsed_seconds"]:.2f}s',flush=True)
    if failed: raise SystemExit(1)


if __name__ == '__main__':
    main()
