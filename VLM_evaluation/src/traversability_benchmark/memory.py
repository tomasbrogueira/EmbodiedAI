"""Optional GPU instrumentation; importing this module does not import Torch."""

import threading

SNAPSHOT_KEYS = ("allocated_bytes", "reserved_bytes", "device_used_bytes", "device_total_bytes")
PEAK_KEYS = ("allocated_peak_bytes", "reserved_peak_bytes")


class NullMemoryProbe:
    """Represent unavailable GPU statistics explicitly, including CPU fixtures."""

    available = False
    unavailable_reason = "GPU measurement unavailable (CPU or no optional Torch/CUDA dependency)"

    def synchronize(self):
        """CPU execution needs no CUDA synchronization."""

    def reset_peak(self):
        """No allocator exists to reset."""

    def snapshot(self):
        """Return null statistics rather than fabricated zero measurements."""
        return dict.fromkeys(SNAPSHOT_KEYS)

    def peaks(self):
        """Return null allocator peaks."""
        return dict.fromkeys(PEAK_KEYS)


class TorchMemoryProbe:
    """Measure a selected CUDA device; exclusively own its peak-counter resets."""

    available = True

    def __init__(self, device=0):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use NullMemoryProbe for CPU profiling")
        self.torch = torch
        self.device = device
        self.unavailable_reason = None

    def synchronize(self):
        """Wait for kernels on the selected device."""
        self.torch.cuda.synchronize(self.device)

    def reset_peak(self):
        """Reset current-process allocator peaks without releasing allocations."""
        self.torch.cuda.reset_peak_memory_stats(self.device)

    def snapshot(self):
        """Return allocator counters and CUDA device-wide free/total usage."""
        snapshot = {
            "allocated_bytes": self.torch.cuda.memory_allocated(self.device),
            "reserved_bytes": self.torch.cuda.memory_reserved(self.device),
            "device_used_bytes": None, "device_total_bytes": None,
        }
        try:
            free, total = self.torch.cuda.mem_get_info(self.device)
            snapshot.update(device_used_bytes=total - free, device_total_bytes=total)
        except Exception as error:
            snapshot["unavailable_reason"] = f"device-wide usage: {type(error).__name__}: {error}"
        return snapshot

    def peaks(self):
        """Return exact peaks of the current-process Torch allocator."""
        return {
            "allocated_peak_bytes": self.torch.cuda.max_memory_allocated(self.device),
            "reserved_peak_bytes": self.torch.cuda.max_memory_reserved(self.device),
        }

    def describe(self):
        """Return selected GPU hardware/runtime identity without allocating tensors."""
        properties = self.torch.cuda.get_device_properties(self.device)
        return {"device": str(self.device), "name": properties.name,
                "compute_capability": [properties.major, properties.minor],
                "device_total_bytes": properties.total_memory,
                "cuda_runtime_version": self.torch.version.cuda}


class _DeviceSampler:
    def __init__(self, probe, interval_s=0.02):
        self.probe, self.interval_s = probe, interval_s
        self.stop_event = threading.Event()
        self.snapshots, self.errors = [], []
        self.thread = None

    def capture(self):
        try:
            self.snapshots.append(self.probe.snapshot())
        except Exception as error:
            self.errors.append(f"snapshot: {type(error).__name__}: {error}")

    def start(self):
        self.capture()
        if getattr(self.probe, "available", True):
            self.thread = threading.Thread(target=self._poll, name="traversability-memory", daemon=True)
            self.thread.start()

    def _poll(self):
        while not self.stop_event.wait(self.interval_s):
            self.capture()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        self.capture()


def _maximum(values):
    valid = [v for v in values if v is not None]
    return max(valid) if valid else None


def _delta(peak, baseline):
    return None if peak is None or baseline is None else max(0, peak - baseline)


def memory_report(probe, sampler, baseline, errors=(), allocator_peaks_valid=True):
    reasons = list(errors) + list(sampler.errors)
    reasons.extend(s["unavailable_reason"] for s in sampler.snapshots if s.get("unavailable_reason"))
    try:
        peaks = probe.peaks() if allocator_peaks_valid else dict.fromkeys(PEAK_KEYS)
    except Exception as error:
        peaks = dict.fromkeys(PEAK_KEYS)
        reasons.append(f"peaks: {type(error).__name__}: {error}")
    if not allocator_peaks_valid:
        reasons.append("Allocator peaks unavailable: interval peak counters were not reset successfully")
    values = {
        "allocated_peak_bytes": peaks.get("allocated_peak_bytes"),
        "reserved_peak_bytes": peaks.get("reserved_peak_bytes"),
        "device_used_peak_bytes": _maximum(s.get("device_used_bytes") for s in sampler.snapshots),
        "device_total_bytes": _maximum(s.get("device_total_bytes") for s in sampler.snapshots),
    }
    for name, base_name in (("allocated", "allocated_bytes"), ("reserved", "reserved_bytes"), ("device_used", "device_used_bytes")):
        values[f"incremental_{name}_peak_bytes"] = _delta(values[f"{name}_peak_bytes"], baseline.get(base_name))
    if all(v is None for v in values.values()):
        reasons.append(getattr(probe, "unavailable_reason", "GPU telemetry unavailable") or "GPU telemetry unavailable")
    else:
        reasons.extend(f"GPU statistic unavailable: {key}" for key, value in values.items() if value is None)
    values.update(device_peak_is_sampled=True, sampling_interval_s=sampler.interval_s, unavailable_reasons=list(dict.fromkeys(reasons)))
    return values
