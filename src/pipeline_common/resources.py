"""Actual process memory measurements without importing Torch or initializing CUDA.

The shared runner owns one collector across reconstruction, semantics, planning
and artifact writes. Torch allocator peaks are read directly, never reconstructed
from model-level measurements. Staged LingBot and semantic residency does not
demonstrate simultaneous model fit. RSS is a sampled current-process peak and
can miss bursts shorter than the sampling interval; it excludes child processes.
"""
from __future__ import annotations

from copy import deepcopy
import importlib
import math
import numbers
import os
import sys
from threading import Event, Lock, Thread
import time


MEMORY_METRIC_IDS = (
    "joint_gpu_allocated_peak_bytes",
    "joint_gpu_reserved_peak_bytes",
    "host_rss_peak_bytes",
)
GPU_SCOPE = ("actual joint process peak for staged runner geometry and semantic residency; "
             "does not establish simultaneous LingBot and semantic model fit")
HOST_SCOPE = "actual current-process sampled RSS peak across the staged runner"


def _bytes(value):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
        raise ValueError("Memory measurement must be a nonnegative integer byte count")
    return int(value)


class ResourceCollector:
    """Collect one run window; ``stop`` is idempotent and returns three metrics.

    ``device`` identifies the user's single allocated visible CUDA device. Only
    an already imported Torch module is inspected. If CUDA is initialized at
    ``start``, this common collector resets its allocator peaks exactly once.
    Reset failure invalidates GPU measurements because their baseline cannot be
    scoped to this run. When CUDA initializes during the window, its allocator
    peaks naturally cover the process since that initialization.

    ``host_enabled=False`` explicitly disables optional psutil sampling. Missing
    psutil, access errors, and sampler failures produce null/unavailable metrics.
    This class never imports Torch, launches models, or resets peaks at ``stop``.
    """

    def __init__(self, sample_interval_s=0.05, device="cuda:0", *, host_enabled=True):
        if (isinstance(sample_interval_s, bool) or not isinstance(sample_interval_s, numbers.Real)
                or not math.isfinite(sample_interval_s) or sample_interval_s <= 0):
            raise ValueError("sample_interval_s must be positive and finite")
        if not isinstance(device, str) or not device.startswith("cuda:") or not device[5:].isdigit():
            raise ValueError("Resource collector requires one explicit cuda:N device")
        if type(host_enabled) is not bool:
            raise ValueError("host_enabled must be boolean")
        self.sample_interval_s = float(sample_interval_s)
        self.device = device
        self.host_enabled = host_enabled
        self._started_ns = None
        self._stopped_ns = None
        self._report = None
        self._stop_event = Event()
        self._lock = Lock()
        self._thread = None
        self._process = None
        self._host_peak = 0
        self._host_samples = 0
        self._host_error = None
        self._host_first_ns = None
        self._host_last_ns = None
        self._torch_at_start = None
        self._gpu_error = None
        self._gpu_baseline = "CUDA initialized after collector start"
        self._gpu_start_ns = None
        self._gpu_reset_count = 0

    def _sample_host(self):
        try:
            rss = _bytes(self._process.memory_info().rss)
            sampled_ns = time.monotonic_ns()
            with self._lock:
                self._host_peak = max(self._host_peak, rss)
                self._host_samples += 1
                if self._host_first_ns is None:
                    self._host_first_ns = sampled_ns
                self._host_last_ns = sampled_ns
        except Exception as error:
            with self._lock:
                self._host_error = f"RSS sampling unavailable: {type(error).__name__}: {error}"
            self._stop_event.set()

    def _sample_loop(self):
        while not self._stop_event.wait(self.sample_interval_s):
            self._sample_host()

    def start(self):
        if self._started_ns is not None:
            raise RuntimeError("ResourceCollector already started; use one collector per run")
        self._started_ns = time.monotonic_ns()
        self._torch_at_start = sys.modules.get("torch")
        if self._torch_at_start is not None:
            try:
                cuda = self._torch_at_start.cuda
                if cuda.is_initialized():
                    # Do not initialize CUDA, select devices, or collect model peaks.
                    cuda.reset_peak_memory_stats(self.device)
                    self._gpu_reset_count = 1
                    self._gpu_start_ns = time.monotonic_ns()
                    self._gpu_baseline = "common collector reset existing initialized CUDA allocator at start"
            except Exception as error:
                self._gpu_error = f"Cannot establish CUDA run baseline: {type(error).__name__}: {error}"
        if not self.host_enabled:
            self._host_error = "RSS sampling explicitly disabled"
            return
        try:
            psutil = importlib.import_module("psutil")
            self._process = psutil.Process(os.getpid())
        except Exception as error:
            self._host_error = f"Optional psutil RSS unavailable: {type(error).__name__}: {error}"
            return
        self._sample_host()
        if self._host_error is None:
            try:
                self._thread = Thread(target=self._sample_loop, name="semantic-mapping-rss", daemon=True)
                self._thread.start()
            except Exception as error:
                self._thread = None
                self._host_error = f"RSS sampler unavailable: {type(error).__name__}: {error}"

    def _base_metric(self, scope):
        return {"value": None, "unit": "bytes", "status": "unavailable", "reason": None,
            "collection_scope": scope, "process_id": os.getpid(), "clock": "monotonic_ns",
            "collection_start_monotonic_ns": self._started_ns,
            "collection_stop_monotonic_ns": self._stopped_ns,
            "collection_duration_s": (self._stopped_ns - self._started_ns) / 1e9}

    def _gpu_metrics(self):
        allocated, reserved = (self._base_metric(GPU_SCOPE) for _ in range(2))
        common = {"device": self.device, "collector_peak_reset_count": self._gpu_reset_count,
            "peak_baseline": self._gpu_baseline, "allocator_peak_start_monotonic_ns": self._gpu_start_ns,
            "measurement_source": "PyTorch current-process CUDA caching allocator",
            "excludes": "GPU driver, other processes and non-Torch GPU allocations"}
        allocated.update(common)
        reserved.update(common)
        reason = self._gpu_error
        torch = sys.modules.get("torch")
        if reason is None:
            if torch is None:
                reason = "Torch not loaded; collector does not import Torch or initialize CUDA"
            elif self._torch_at_start is not None and torch is not self._torch_at_start:
                reason = "Torch module changed during collection; CUDA baseline is unverifiable"
            else:
                try:
                    if not torch.cuda.is_initialized():
                        reason = "CUDA was not initialized during the collection window"
                except Exception as error:
                    reason = f"CUDA initialization state unavailable: {type(error).__name__}: {error}"
        if reason is not None:
            allocated["reason"] = reserved["reason"] = reason
            return allocated, reserved
        # Independent failures stay explicit; neither peak is computed from the other.
        for metric, reader in ((allocated, "max_memory_allocated"), (reserved, "max_memory_reserved")):
            try:
                metric["value"] = _bytes(getattr(torch.cuda, reader)(self.device))
                metric.update(status="available", reason=None, peak_read_monotonic_ns=time.monotonic_ns())
            except Exception as error:
                metric["reason"] = f"CUDA allocator peak unavailable: {type(error).__name__}: {error}"
        return allocated, reserved

    def stop(self):
        if self._started_ns is None:
            raise RuntimeError("ResourceCollector must start before stop")
        if self._report is not None:
            return deepcopy(self._report)
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 2 * self.sample_interval_s))
            if self._thread.is_alive():
                self._host_error = "RSS sampler did not stop within the bounded collection window"
        if self._process is not None and self._host_error is None:
            self._sample_host()
        self._stopped_ns = time.monotonic_ns()
        host = self._base_metric(HOST_SCOPE)
        host.update(sample_interval_s=self.sample_interval_s, sample_count=self._host_samples,
            first_sample_monotonic_ns=self._host_first_ns, last_sample_monotonic_ns=self._host_last_ns,
            measurement_source="optional psutil.Process.memory_info().rss",
            sampled_peak_limitation="may miss bursts shorter than sample interval; excludes child processes")
        if self._host_error is None and self._host_samples:
            host.update(value=self._host_peak, status="available", reason=None)
        else:
            host["reason"] = self._host_error or "No RSS samples collected"
        allocated, reserved = self._gpu_metrics()
        self._report = dict(zip(MEMORY_METRIC_IDS, (allocated, reserved, host)))
        return deepcopy(self._report)
