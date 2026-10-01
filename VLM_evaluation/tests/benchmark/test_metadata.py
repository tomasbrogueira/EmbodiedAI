"""Configuration provenance must be complete before publishing a comparison."""

from types import SimpleNamespace
import unittest

from traversability_benchmark.memory import TorchMemoryProbe
from traversability_benchmark.profiling import _metadata_details


class MetadataTests(unittest.TestCase):
    def backend(self):
        return SimpleNamespace(settings={"robot_policy": "frozen material policy"}, metadata={
            "revision": "fixture-revision", "processor": {"size": 32}, "quantization": {"language_bits": 4},
        })

    def test_qwen_requires_prompt_provenance(self):
        backend = self.backend()
        result = _metadata_details(backend, None, "qwen3_vl_4b", backend.settings, backend.settings)
        self.assertFalse(result["comparison_metadata_complete"])
        self.assertIn("actual_prompt", result["missing_comparison_metadata"])
        sidecar = {"model_key": "qwen3_vl_4b", "configuration": {"prompt": "fixture full prompt", "settings": backend.settings}, "actual_backend": backend.metadata}
        self.assertTrue(_metadata_details(backend, sidecar, "qwen3_vl_4b", backend.settings, backend.settings)["comparison_metadata_complete"])
        sidecar["configuration"]["settings"] = {"robot_policy": "different fixture policy"}
        self.assertFalse(_metadata_details(backend, sidecar, "qwen3_vl_4b", backend.settings, backend.settings)["comparison_metadata_complete"])

    def test_clip_requires_frozen_vocabulary_and_development_calibration(self):
        backend = self.backend()
        result = _metadata_details(backend, None, "clip_vit_b32", backend.settings, backend.settings)
        self.assertIn("actual_clip_vocabulary_and_calibration", result["missing_comparison_metadata"])
        backend.metadata.update(vocabulary=[{"class": "fixture"}], calibration={"threshold": 0.5, "fixture": True})
        self.assertTrue(_metadata_details(backend, None, "clip_vit_b32", backend.settings, backend.settings)["comparison_metadata_complete"])

    def test_gpu_identity_queries_only_the_selected_device(self):
        devices = []
        def properties(device):
            devices.append(device)
            return SimpleNamespace(name="fixture GPU", major=8, minor=6, total_memory=20_000_000_000)
        probe = TorchMemoryProbe.__new__(TorchMemoryProbe)
        probe.device = "cuda:1"
        probe.torch = SimpleNamespace(cuda=SimpleNamespace(get_device_properties=properties), version=SimpleNamespace(cuda="12.6"))
        identity = probe.describe()
        self.assertEqual(devices, ["cuda:1"])
        self.assertEqual(identity["compute_capability"], [8, 6])
        self.assertEqual(identity["cuda_runtime_version"], "12.6")


if __name__ == "__main__":
    unittest.main()
