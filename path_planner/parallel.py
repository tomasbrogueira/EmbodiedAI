"""Bound CPU work without changing numerical neighbourhoods or point rows."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from numbers import Integral
import os
from pathlib import Path


def worker_count(workers):
    if isinstance(workers, bool) or not isinstance(workers, Integral) or workers < 0:
        raise ValueError('workers must be a nonnegative integer; 0 selects automatically')
    if workers:
        return int(workers)
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count() or 1
    try:
        quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if quota != 'max':
            available = min(available, max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    # Limit simultaneous neighbourhood tensors and leave room for the viewer.
    return max(1, min(12, available))


def completed(function, jobs, workers):
    """Return finished independent batches; cancel queued work on failure."""
    workers = worker_count(workers)
    if workers == 1:
        for job in jobs:
            yield function(job)
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        try:
            futures = [executor.submit(function, job) for job in jobs]
            for future in as_completed(futures):
                yield future.result()
        finally:
            for future in futures:
                future.cancel()
