# Support direct compatibility-script execution without environment mutation.
if __package__ in (None, ''):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from orchestrator.paths import source_path
"""Verify routes while one preparation thread works through short recordings."""
import argparse
from concurrent.futures import CancelledError, ThreadPoolExecutor
from pathlib import Path
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import threading
import time
import traceback
from orchestrator.batch.dataset_resources import allocation, write

SOURCE_ROOT = Path(__file__).resolve().parent
ROOT = SOURCE_ROOT
POLICY = 5  # Calibrated upward clearance, with a fixed-start partial route if needed.


def cost_source_hash(repo):
    return hashlib.sha256((source_path(repo, 'path-planning/cost_map.py')).read_bytes()).hexdigest()


def prepare_adapter_hash():
    return hashlib.sha256((SOURCE_ROOT / 'dataset_prepare.py').read_bytes()).hexdigest()


def calibration_association_hash():
    return hashlib.sha256((SOURCE_ROOT / 'dataset_path.py').read_bytes()).hexdigest()


def reused(entry):
    report = Path(entry['output']) / 'report.json'
    return report.exists() and json.loads(report.read_text()).get('reused_verified_output')


def command(repo, entry, script):
    out = Path(entry['output'])
    python = str(repo / '.setup/envs/orchestrator-visualizer/bin/python')
    env = os.environ | {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                        'PYTHONDONTWRITEBYTECODE': '1'}
    with (out / (Path(script).stem + '.log')).open('a') as log:
        subprocess.run([python, '-B', '-u', str(SOURCE_ROOT / script), '--repo', str(repo), '--output', str(out)],
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def prepare_ahead(repo, entry, stopping=lambda: False):
    if stopping():
        raise CancelledError()
    try:
        with allocation(ROOT, entry, 'geometry-and-path'):
            if stopping():
                raise CancelledError()
            command(repo, entry, 'dataset_prepare.py')
        out = Path(entry['output'])
        if json.loads((out / 'report.json').read_text()).get('status') == 'pending':
            write(out / 'status.json', {'stage': 'path', 'status': 'pending',
                                       'message': 'Map and costs prepared; waiting for route search'})
    except CancelledError:
        raise
    except Exception as error:
        failure(entry, error)
        raise


def run(repo, entry, prepared=False, before_path=None):
    out = Path(entry['output'])
    with allocation(ROOT, entry, 'geometry-and-path'):
        if not prepared:
            command(repo, entry, 'dataset_prepare.py')
        if json.loads((out / 'report.json').read_text()).get('status') not in ('missing_reference', 'missing_clearance_calibration'):
            if before_path:
                before_path()
            command(repo, entry, 'dataset_path.py')
    report = json.loads((out / 'report.json').read_text())
    write(out / 'cpu.json', {'input_sha256': entry['input_sha256'], 'policy': POLICY,
                           'cost_source_sha256': report['cost_source_sha256'],
                           'prepare_adapter_sha256': report['prepare_adapter_sha256'],
                           'calibration_association_sha256': report['calibration_association_sha256'],
                           'status': report['status'], 'finished': time.time()})


def failure(entry, error):
    traceback.print_exc()
    write(Path(entry['output']) / 'status.json', {'stage': 'geometry-or-path', 'status': 'error', 'message': str(error)})
    write(Path(entry['output']) / 'cpu-error.json', {'error': str(error), 'traceback': traceback.format_exc()})


def work(stopping=lambda: False, manifest_path=None):
    manifest_path = Path(manifest_path) if manifest_path is not None else ROOT / 'manifest.json'
    attempted = set()
    ahead = {}
    priority = lambda v: (v['frame_count'], v['id'])
    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            while not stopping():
                manifest = json.loads(manifest_path.read_text())
                repo = Path(manifest['repo'])
                cost_source = cost_source_hash(repo)
                prepare_adapter = prepare_adapter_hash()
                calibration_association = calibration_association_hash()
                pending = []
                for candidate in manifest['videos']:
                    out = Path(candidate['output'])
                    receipt = json.loads((out / 'cpu.json').read_text()) if (out / 'cpu.json').exists() else {}
                    current = (receipt.get('policy') == POLICY and receipt.get('input_sha256') == candidate['input_sha256']
                               and receipt.get('cost_source_sha256') == cost_source
                               and receipt.get('prepare_adapter_sha256') == prepare_adapter
                               and receipt.get('calibration_association_sha256') == calibration_association)
                    if candidate['id'] not in attempted and not current and (out / 'gpu.json').exists():
                        pending.append(candidate)
                prepared = bool(ahead)
                if ahead:
                    entry, future = ahead.pop(min(ahead, key=lambda key: priority(ahead[key][0])))
                    try:
                        future.result()
                    except Exception:
                        continue  # Preparation already reported failure, or was cancelled by a stop.
                elif pending:
                    entry = min(pending, key=priority)
                    attempted.add(entry['id'])
                elif (ROOT / 'gpu-finished.json').exists():
                    write(ROOT / 'cpu-finished.json', {'finished': time.time(), 'attempted': sorted(attempted)})
                    break
                else:
                    time.sleep(10)
                    continue
                if stopping():
                    break

                def prefetch():
                    if stopping() or entry['frame_count'] > 900 or reused(entry):
                        return
                    candidates = [v for v in pending if v['id'] not in attempted
                                  and v['frame_count'] <= 900 and not reused(v)]
                    for following in sorted(candidates, key=priority):
                        if stopping():
                            break
                        attempted.add(following['id'])  # Reserve it before its thread can write anything.
                        ahead[following['id']] = (following, pool.submit(prepare_ahead, repo, following, stopping))
                try:
                    run(repo, entry, prepared=prepared, before_path=prefetch)
                except Exception as error:
                    failure(entry, error)
        finally:
            for _, future in ahead.values():
                future.cancel()  # Cancel queued work; finish only the already-active preparation.
            for _, future in ahead.values():
                try:
                    future.result()
                except Exception:
                    pass  # Active failures were reported; cancellation is not a processing error.


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    ROOT = args.manifest.resolve().parent
    lock = (ROOT / 'cpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stopping = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stopping.set())
    work(stopping.is_set, args.manifest.resolve())
