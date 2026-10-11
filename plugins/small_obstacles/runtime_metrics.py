"""Memory telemetry without NVML or an extra CUDA context in the controller."""
import json
from pathlib import Path
import threading
import time


def host_memory():
    result = {}
    status = Path('/proc/self/status')
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith(('VmRSS:', 'VmHWM:')):
                key, value, _ = line.split()
                result[key.rstrip(':') + '_bytes'] = int(value) * 1024
    for name in ('memory.current', 'memory.max'):
        path = Path('/sys/fs/cgroup') / name
        if path.exists():
            value = path.read_text().strip()
            result['cgroup_' + name.replace('.', '_')] = int(value) if value != 'max' else None
    return result


class CudaMemoryMonitor:
    """Per-worker allocator peaks plus sampled device-wide use across workers.

    Device use is sampled, so its maximum is a lower bound on transient peaks.
    Allocator counters refer only to this process; do not sum their peak values.
    """
    def __init__(self, torch, role, path, interval=.1):
        self.torch, self.role, self.path, self.interval = torch, role, Path(path), interval
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.device_peak_used = self.process_peak_allocated = self.process_peak_reserved = 0
        self.thread = None
        if interval > 0:
            self.thread = threading.Thread(target=self._sample_loop, daemon=True)
            self.thread.start()

    def snapshot(self):
        free, total = self.torch.cuda.mem_get_info()
        allocated = self.torch.cuda.memory_allocated()
        reserved = self.torch.cuda.memory_reserved()
        peak_allocated = self.torch.cuda.max_memory_allocated()
        peak_reserved = self.torch.cuda.max_memory_reserved()
        with self.lock:
            self.device_peak_used = max(self.device_peak_used, total - free)
            self.process_peak_allocated = max(self.process_peak_allocated, peak_allocated)
            self.process_peak_reserved = max(self.process_peak_reserved, peak_reserved)
            result = {'role': self.role, 'unix_seconds': time.time(),
                'device_total_bytes': total, 'device_free_bytes': free, 'device_used_bytes': total - free,
                'device_sampled_peak_used_bytes': self.device_peak_used,
                'allocated_bytes': allocated, 'reserved_bytes': reserved,
                'peak_allocated_bytes': self.process_peak_allocated,
                'peak_reserved_bytes': self.process_peak_reserved}
        return result

    def _sample_loop(self):
        with self.path.open('w') as output:
            while not self.stop.is_set():
                try:
                    value = self.snapshot()
                except Exception as exc:
                    value = {'role': self.role, 'unix_seconds': time.time(), 'sampling_error': str(exc)}
                output.write(json.dumps(value) + '\n')
                output.flush()
                self.stop.wait(self.interval)

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2)


def cache_memory(model):
    """Describe the active aggregator cache, including retained special tokens."""
    by_dtype, keyframes, special_tokens = {}, 0, 0
    for key, tensor in model.aggregator.kv_cache.items():
        if not hasattr(tensor, 'numel'):
            continue
        dtype = str(tensor.dtype)
        by_dtype[dtype] = by_dtype.get(dtype, 0) + tensor.numel() * tensor.element_size()
        if key == 'k_0':
            keyframes = tensor.shape[2]
        elif key == 'k_0_special':
            special_tokens = tensor.shape[2] * tensor.shape[3]
    return {'bytes': sum(by_dtype.values()), 'bytes_by_dtype': by_dtype,
            'patch_keyframes': keyframes, 'older_special_tokens': special_tokens}


def release_unused_cuda(torch, policy, min_free_bytes):
    """Release only unused allocator blocks; live tensors/caches remain intact."""
    release = policy == 'release' or (policy == 'budget' and torch.cuda.mem_get_info()[0] < min_free_bytes)
    if release:
        torch.cuda.empty_cache()
    return release
