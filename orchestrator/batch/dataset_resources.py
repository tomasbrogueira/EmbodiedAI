"""Keep long recordings exclusive within this batch's 64 GiB allocation."""
from contextlib import contextmanager, nullcontext
from pathlib import Path
import fcntl
import json
import os
import shutil
import time


@contextmanager
def allocation(root, entry, stage):
    from orchestrator.resources import gpu_allocation
    with gpu_allocation() if stage == 'models' else nullcontext():
        yield from _allocation(root, entry, stage)


def _allocation(root, entry, stage):
    with (root / 'compute.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX if entry['frame_count'] > 900 else fcntl.LOCK_SH)
        resource = {'stage': stage, 'time': time.time(), 'load': os.getloadavg(),
                    'free_disk': shutil.disk_usage(root).free,
                    'memory_current': int(Path('/sys/fs/cgroup/memory.current').read_text())}
        if resource['free_disk'] < 100 * 1024**3:
            raise RuntimeError('Less than 100 GiB free storage')
        (Path(entry['output']) / (stage + '-resources.json')).write_text(json.dumps(resource))
        yield


def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)
