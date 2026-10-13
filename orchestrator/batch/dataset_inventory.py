# Support direct compatibility-script execution without environment mutation.
if __package__ in (None, ''):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
"""Inventory full recordings without replacing mismatched or completed runs."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time


GEOMETRY = dict(frame_step=2, pixel_step=1, min_depth_conf=2.5, max_depth=25, patch_size=14)
MODEL = dict(mode='streaming', kv_cache_sliding_window=32, keyframe_interval=2,
             num_scale_frames=8, use_sdpa=True)


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def read(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def probe(video):
    result = json.loads(command('ffprobe', '-v', 'error', '-select_streams', 'v:0',
                               '-show_frames', '-show_entries',
                               'frame=best_effort_timestamp_time:stream=nb_frames:format=duration',
                               '-of', 'json', str(video)))
    timestamps = [float(frame['best_effort_timestamp_time']) for frame in result['frames']]
    if not timestamps or not all(math.isfinite(t) for t in timestamps):
        raise ValueError(f'{video}: missing or invalid frame timestamps')
    if any(b < a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError(f'{video}: frame timestamps are not ordered')
    declared = result['streams'][0].get('nb_frames', 'N/A')
    if declared != 'N/A' and int(declared) != len(timestamps):
        raise ValueError(f'{video}: declared and decoded frame counts differ')
    duration = float(result['format']['duration'])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f'{video}: invalid duration')
    return dict(duration=duration, frame_count=len(timestamps), timestamps=timestamps)


def inventory(repo, output, pull=False):
    repo, output = repo.resolve(), output.resolve()
    if output == repo or repo in output.parents:
        raise ValueError('The output directory must be outside the repository')
    if pull:
        # Never stash, reset, or resolve conflicts in the user's checkout.
        subprocess.run(['git', '-C', str(repo), 'pull', '--ff-only'], check=True)
    old = read(output / 'manifest.json') or {}
    model = MODEL | {'reconstruction_source_sha256': sha256(repo / 'path-planning/reconstruct.py')} | {name + '_commit': command('git', '-C', str(repo / folder), 'rev-parse', 'HEAD')
                     for name, folder in [('lingbot', 'lingbot-map'), ('sam', 'sam3')]}
    videos, mismatches = [], []
    root = repo / 'data' / 'videos'
    for video in sorted(root.rglob('*')):
        relative = video.relative_to(root)
        if not video.is_file() or video.suffix.lower() != '.mp4' or any(p.startswith('.') for p in relative.parts):
            continue
        if relative.parts[0] not in ('indoor', 'outdoor'):
            raise ValueError(f'Unknown video category: {relative}; floor/ground prompt cannot be inferred')
        identity = relative.with_suffix('').as_posix()
        run = output / 'runs' / identity
        entry = dict(id=identity, relative_path=relative.as_posix(), name=video.stem,
                     video=str(video), output=str(run), prompt='floor' if relative.parts[0] == 'indoor' else 'ground',
                     input_sha256=sha256(video), model_settings=model, **probe(video))
        existing = read(run / 'entry.json')
        if existing:
            changed = [key for key, value in entry.items() if existing.get(key) != value]
            if old.get('geometry_settings') != GEOMETRY:
                changed.append('geometry_settings')
            if changed:
                mismatches.append(dict(id=identity, changed=changed))
        elif run.exists() and any(run.iterdir()):
            mismatches.append(dict(id=identity, changed=['existing run has no entry.json']))
        videos.append(entry)
    if not videos:
        raise ValueError(f'No recordings found under {root}')
    ids = [entry['id'].casefold() for entry in videos]
    if len(ids) != len(set(ids)):
        raise ValueError('Video paths collide after removing extensions or ignoring case')
    output.mkdir(parents=True, exist_ok=True)
    if mismatches:
        write(output / 'inventory-mismatches.json', mismatches)
        raise ValueError('Cached inputs/settings differ; existing manifest and runs were preserved. '
                         'See inventory-mismatches.json and use a separate output directory.')
    for entry in videos:
        run = Path(entry['output'])
        run.mkdir(parents=True, exist_ok=True)
        if not (run / 'entry.json').exists():
            write(run / 'entry.json', entry)
        if not (run / 'status.json').exists():
            write(run / 'status.json', dict(stage='queued', status='pending',
                                           message='Waiting for full reconstruction and segmentation'))
    manifest = dict(repo=str(repo), dataset_commit=command('git', '-C', str(repo), 'rev-parse', 'HEAD'),
                    geometry_settings=GEOMETRY, created=time.time(), videos=videos)
    write(output / 'manifest.json', manifest)
    for name in ('gpu-finished.json', 'cpu-finished.json'):
        (output / name).unlink(missing_ok=True)
    print(f'Inventoried {len(videos)} videos: {output / "manifest.json"}', flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pull', action='store_true', help='Fast-forward a clean tracked checkout first')
    args = parser.parse_args()
    locks = []
    if os.name == 'posix':
        import fcntl
        for name in ('gpu.lock', 'cpu.lock'):
            path = args.output / name
            if path.exists():
                lock = path.open('r')
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    parser.error('Wait for the active batch workers before updating the inventory')
                locks.append(lock)
    inventory(args.repo, args.output, args.pull)
