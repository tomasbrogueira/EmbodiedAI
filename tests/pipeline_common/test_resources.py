"""CPU-only resource collection with fake allocator counters and optional RSS."""
import json
import subprocess
import sys
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pipeline_common.resources import MEMORY_METRIC_IDS, ResourceCollector


class FakeCuda:
    def __init__(self, initialized=True, *, reset_error=False, allocated=101, reserved=211):
        self.initialized = initialized
        self.reset_error = reset_error
        self.allocated = allocated
        self.reserved = reserved
        self.calls = []

    def is_initialized(self):
        self.calls.append(("is_initialized",))
        return self.initialized

    def reset_peak_memory_stats(self, device):
        self.calls.append(("reset", device))
        if self.reset_error:
            raise RuntimeError("fake reset failure")

    def max_memory_allocated(self, device):
        self.calls.append(("allocated", device))
        return self.allocated

    def max_memory_reserved(self, device):
        self.calls.append(("reserved", device))
        return self.reserved


class FakeProcess:
    def __init__(self, rss_values, *, fail_after=None):
        self.values = list(rss_values)
        self.calls = 0
        self.fail_after = fail_after
        self.failed = Event()

    def memory_info(self):
        if self.fail_after is not None and self.calls >= self.fail_after:
            self.failed.set()
            raise PermissionError("fake access failure")
        value = self.values[min(self.calls, len(self.values) - 1)]
        self.calls += 1
        return SimpleNamespace(rss=value)


class ResourceTests(unittest.TestCase):
    def test_import_never_loads_torch(self):
        completed = subprocess.run([sys.executable, "-c",
            "import sys; import pipeline_common.resources; assert 'torch' not in sys.modules"],
            capture_output=True, text=True, timeout=5)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_no_torch_and_disabled_host_produce_three_null_metrics(self):
        with patch.dict(sys.modules, {"torch": None}):
            collector = ResourceCollector(host_enabled=False)
            collector.start()
            result = collector.stop()
        self.assertEqual(tuple(result), MEMORY_METRIC_IDS)
        for measurement in result.values():
            self.assertIsNone(measurement["value"])
            self.assertEqual(measurement["status"], "unavailable")
            self.assertEqual(measurement["unit"], "bytes")
            self.assertTrue(measurement["reason"])
            self.assertGreaterEqual(measurement["collection_duration_s"], 0)
        self.assertIn("does not import", result[MEMORY_METRIC_IDS[0]]["reason"])
        json.dumps(result, allow_nan=False)

    def test_initialized_cuda_one_common_reset_and_actual_independent_peaks(self):
        cuda = FakeCuda()
        with patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=cuda)}):
            collector = ResourceCollector(device="cuda:2", host_enabled=False)
            collector.start()
            first = collector.stop()
            # Mutation of a returned report cannot change future idempotent stop.
            first[MEMORY_METRIC_IDS[0]]["value"] = 999
            result = collector.stop()
        self.assertEqual(cuda.calls.count(("reset", "cuda:2")), 1)
        self.assertEqual(cuda.calls.count(("allocated", "cuda:2")), 1)
        self.assertEqual(cuda.calls.count(("reserved", "cuda:2")), 1)
        self.assertEqual(result[MEMORY_METRIC_IDS[0]]["value"], 101)
        self.assertEqual(result[MEMORY_METRIC_IDS[1]]["value"], 211)
        for identifier in MEMORY_METRIC_IDS[:2]:
            measurement = result[identifier]
            self.assertEqual(measurement["status"], "available")
            self.assertEqual(measurement["collector_peak_reset_count"], 1)
            self.assertIn("joint process peak", measurement["collection_scope"])
            self.assertIn("does not establish simultaneous", measurement["collection_scope"])
            self.assertIsNotNone(measurement["allocator_peak_start_monotonic_ns"])

    def test_existing_uninitialized_torch_uses_new_cuda_lifetime_without_reset(self):
        cuda = FakeCuda(initialized=False)
        with patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=cuda)}):
            collector = ResourceCollector(host_enabled=False)
            collector.start()
            cuda.initialized = True  # Fake the runner's model initializing CUDA.
            result = collector.stop()
        self.assertFalse(any(call[0] == "reset" for call in cuda.calls))
        self.assertEqual(result[MEMORY_METRIC_IDS[0]]["value"], 101)
        self.assertEqual(result[MEMORY_METRIC_IDS[0]]["collector_peak_reset_count"], 0)
        self.assertIn("initialized after", result[MEMORY_METRIC_IDS[0]]["peak_baseline"])

    def test_torch_loaded_during_window_uses_actual_process_peaks(self):
        cuda = FakeCuda()
        with patch.dict(sys.modules, {"torch": None}):
            collector = ResourceCollector(host_enabled=False)
            collector.start()
            sys.modules["torch"] = SimpleNamespace(cuda=cuda)
            result = collector.stop()
        self.assertFalse(any(call[0] == "reset" for call in cuda.calls))
        self.assertEqual(result[MEMORY_METRIC_IDS[1]]["value"], 211)

    def test_reset_failure_invalidates_baseline_and_does_not_read_old_peaks(self):
        cuda = FakeCuda(reset_error=True)
        with patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=cuda)}):
            collector = ResourceCollector(host_enabled=False)
            collector.start()
            result = collector.stop()
        self.assertEqual(cuda.calls.count(("reset", "cuda:0")), 1)
        self.assertFalse(any(call[0] in {"allocated", "reserved"} for call in cuda.calls))
        for identifier in MEMORY_METRIC_IDS[:2]:
            self.assertEqual(result[identifier]["status"], "unavailable")
            self.assertIsNone(result[identifier]["value"])
            self.assertIn("baseline", result[identifier]["reason"])

    def test_no_cuda_initialization_and_module_replacement_remain_unavailable(self):
        cuda = FakeCuda(initialized=False)
        with patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=cuda)}):
            collector = ResourceCollector(host_enabled=False)
            collector.start()
            result = collector.stop()
        self.assertIn("not initialized", result[MEMORY_METRIC_IDS[0]]["reason"])
        self.assertFalse(any(call[0] in {"reset", "allocated", "reserved"} for call in cuda.calls))
        with patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=FakeCuda())}):
            collector = ResourceCollector(host_enabled=False)
            collector.start()
            sys.modules["torch"] = SimpleNamespace(cuda=FakeCuda())
            result = collector.stop()
        self.assertIn("module changed", result[MEMORY_METRIC_IDS[0]]["reason"])

    def test_invalid_allocated_peak_does_not_hide_independent_reserved_peak(self):
        cuda = FakeCuda(allocated=-1)
        with patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=cuda)}):
            collector = ResourceCollector(host_enabled=False)
            collector.start()
            result = collector.stop()
        self.assertIsNone(result[MEMORY_METRIC_IDS[0]]["value"])
        self.assertEqual(result[MEMORY_METRIC_IDS[0]]["status"], "unavailable")
        self.assertEqual(result[MEMORY_METRIC_IDS[1]]["value"], 211)

    def test_host_rss_reports_observed_peak_sampling_count_and_boundaries(self):
        process = FakeProcess([100, 900, 200])
        psutil = SimpleNamespace(Process=lambda pid: process)
        with patch.dict(sys.modules, {"torch": None}), patch(
                "pipeline_common.resources.importlib.import_module", return_value=psutil) as importer:
            collector = ResourceCollector(sample_interval_s=10)
            collector.start()
            collector._sample_host()  # Controlled intermediate sample, no timing race.
            result = collector.stop()
        importer.assert_called_once_with("psutil")
        host = result[MEMORY_METRIC_IDS[2]]
        self.assertEqual(host["value"], 900)
        self.assertEqual(host["sample_count"], 3)
        self.assertEqual(host["sample_interval_s"], 10)
        self.assertEqual(host["status"], "available")
        self.assertLessEqual(host["first_sample_monotonic_ns"], host["last_sample_monotonic_ns"])
        self.assertLessEqual(host["collection_start_monotonic_ns"], host["first_sample_monotonic_ns"])
        self.assertLessEqual(host["last_sample_monotonic_ns"], host["collection_stop_monotonic_ns"])
        self.assertIn("may miss bursts", host["sampled_peak_limitation"])

    def test_optional_psutil_absence_does_not_block_runner(self):
        with patch.dict(sys.modules, {"torch": None}), patch(
                "pipeline_common.resources.importlib.import_module", side_effect=ModuleNotFoundError("psutil")):
            collector = ResourceCollector()
            collector.start()
            host = collector.stop()[MEMORY_METRIC_IDS[2]]
        self.assertEqual(host["status"], "unavailable")
        self.assertIsNone(host["value"])
        self.assertIn("Optional psutil", host["reason"])

    def test_thread_sampler_failure_is_caught_and_peak_marked_unavailable(self):
        process = FakeProcess([100], fail_after=1)
        with patch.dict(sys.modules, {"torch": None}), patch(
                "pipeline_common.resources.importlib.import_module",
                return_value=SimpleNamespace(Process=lambda pid: process)):
            collector = ResourceCollector(sample_interval_s=.001)
            collector.start()
            self.assertTrue(process.failed.wait(timeout=1))
            host = collector.stop()[MEMORY_METRIC_IDS[2]]
        self.assertEqual(host["status"], "unavailable")
        self.assertIsNone(host["value"])
        self.assertEqual(host["sample_count"], 1)
        self.assertIn("fake access failure", host["reason"])

    def test_sampler_thread_launch_failure_does_not_abort_runner(self):
        process = FakeProcess([100])
        with patch.dict(sys.modules, {"torch": None}), patch(
                "pipeline_common.resources.Thread") as thread, patch(
                "pipeline_common.resources.importlib.import_module",
                return_value=SimpleNamespace(Process=lambda pid: process)):
            thread.return_value.start.side_effect = RuntimeError("fake thread unavailable")
            collector = ResourceCollector()
            collector.start()
            host = collector.stop()[MEMORY_METRIC_IDS[2]]
        self.assertEqual(host["status"], "unavailable")
        self.assertIsNone(host["value"])
        self.assertIn("fake thread unavailable", host["reason"])

    def test_lifecycle_and_configuration_validation(self):
        collector = ResourceCollector(host_enabled=False)
        with self.assertRaisesRegex(RuntimeError, "must start"):
            collector.stop()
        with patch.dict(sys.modules, {"torch": None}):
            collector.start()
            with self.assertRaisesRegex(RuntimeError, "already started"):
                collector.start()
            collector.stop()
        for kwargs in ({"sample_interval_s": 0}, {"sample_interval_s": True},
                {"sample_interval_s": float("nan")}, {"device": "cuda"},
                {"device": "cpu"}, {"device": "cuda:-1"}, {"host_enabled": 1}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    ResourceCollector(**kwargs)


if __name__ == "__main__":
    unittest.main()
