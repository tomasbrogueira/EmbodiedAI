"""Sequential full-video inference with validated resume and bounded concurrency."""
import argparse
from pathlib import Path
import fcntl
import json
import os
import subprocess
import time
import traceback
from dataset_resources import allocation, write
from dataset_inventory import MODEL, sha256

SOURCE_ROOT = Path(__file__).resolve().parent
ROOT = SOURCE_ROOT


def status(out, stage, state='running', message=''):
    write(out / 'status.json', dict(stage=stage, status=state, message=message, updated=time.time()))


def frames(path, count, masks=False):
    found = sorted(int(p.stem[-6:] if masks else p.stem.split('_')[-1])
                   for p in path.glob('*.npz' if masks else 'frame_*.npz'))
    return found == list(range(count))


def command(repo, out, stage, args, cwd, valid):
    record = out / (stage + '-command.json')
    if record.exists() and json.loads(record.read_text()) == args and valid():
        print('Reusing verified', out.name, stage, flush=True)
        return
    while True:
        probe = subprocess.check_output([str(repo / '.setup/envs/lingbot-map/bin/python'), '-B', '-c',
            'import torch,json; print(json.dumps(dict(devices=torch.cuda.device_count(),memory=torch.cuda.mem_get_info())))'], text=True)
        cuda = json.loads(probe.strip().splitlines()[-1])
        if cuda['devices'] != 1:
            raise RuntimeError('Expected the existing single allocated GPU')
        if cuda['memory'][0] >= .85 * cuda['memory'][1]:
            break
        status(out, 'waiting_for_gpu', 'pending', 'Waiting for the allocated GPU to become free')
        time.sleep(20)
    status(out, stage)
    env = os.environ | {'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
                        'PYTHONDONTWRITEBYTECODE': '1', 'HF_HOME': str(repo / '.setup/cache/huggingface'),
                        'HF_HUB_OFFLINE': '1'}
    write(record, args)
    with (out / (stage + '.log')).open('w') as log:
        subprocess.run(args, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    if not valid():
        raise RuntimeError(f'{stage}: outputs do not contain every video frame')


def run(repo, entry):
    expected = MODEL | {'reconstruction_source_sha256': sha256(repo / 'reconstruct.py')}
    if any(entry.get('model_settings', {}).get(key) != value for key, value in expected.items()):
        raise RuntimeError('Reconstruction source/settings changed; inventory a separate output directory')
    out, name, count = Path(entry['output']), entry['name'], entry['frame_count']
    receipt = out / 'gpu.json'
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if any(saved[key] != entry[key] for key in ('input_sha256', 'prompt', 'model_settings')):
            raise RuntimeError('Cached model outputs do not match the input/settings')
        if saved['frames'] != count or not frames(Path(saved['lingbot']), count) or not frames(Path(saved['sam']), count, True):
            raise RuntimeError('Cached model outputs are incomplete')
        return
    with allocation(ROOT, entry, 'models'):
        ling, sam = out / 'lingbot', out / 'sam'
        def ling_valid():
            result = ling / 'batch_results.json'
            if not result.exists():
                return False
            results = json.loads(result.read_text()).get('results', [])
            return len(results) == 1 and results[0].get('success') and frames(ling / name, count)
        command(repo, out, 'reconstruction', [str(repo / '.setup/envs/lingbot-map/bin/python'), '-B', '-u',
            str(repo / 'reconstruct.py'), '--video_path', entry['video'], '--output_folder', str(ling),
            '--model_path', str(repo / 'lingbot-map/lingbot-map.pt'), '--mode', 'streaming', '--kv_cache_sliding_window', '32',
            '--keyframe_interval', '2', '--num_scale_frames', '8', '--max_frame_num', str(max(1024, count)),
            '--use_sdpa', '--save_predictions', '--no_render', '--no_original_video', '--num_workers', '4'], repo / 'lingbot-map', ling_valid)
        command(repo, out, 'masks', [str(repo / '.setup/envs/sam3/bin/python'), '-B', '-u', str(SOURCE_ROOT / 'generate_masks.py'),
            '--video_path', entry['video'], '--output_folder', str(sam), '--prompt', entry['prompt']], repo / 'sam3',
            lambda: frames(sam / (name + '_masks'), count, True))
        write(receipt, dict(input_sha256=entry['input_sha256'], prompt=entry['prompt'], frames=count,
                           lingbot=str(ling / name), sam=str(sam / (name + '_masks')), model_settings=entry['model_settings']))
        status(out, 'ready', 'pending', 'Full-video model outputs verified; awaiting map and path processing')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    ROOT = args.manifest.resolve().parent
    lock = (ROOT / 'gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads(args.manifest.read_text())
    for entry in sorted(manifest['videos'], key=lambda v: (v['frame_count'], v['relative_path'])):
        try:
            run(Path(manifest['repo']), entry)
        except Exception as error:
            traceback.print_exc()
            status(Path(entry['output']), 'models', 'error', str(error))
            write(Path(entry['output']) / 'gpu-error.json', {'error': str(error), 'traceback': traceback.format_exc()})
    write(ROOT / 'gpu-finished.json', {'finished': time.time(), 'videos': len(manifest['videos'])})
