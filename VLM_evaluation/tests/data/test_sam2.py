"""Mocked CPU SAM2 integration; no real checkpoint, model or download is used."""

from contextlib import nullcontext
import hashlib
import importlib
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from traversability_data.masks import freeze_selection
from traversability_data import sam2
from traversability_data.storage import atomic_json, atomic_jsonl, read_json, read_jsonl


class FixtureGenerator:
    metadata = {"fixture_identity": "sam_cpu_fixture_v1"}

    def __init__(self, bad_shape=False):
        self.calls = []
        self.bad_shape = bad_shape

    def generate(self, rgb):
        self.calls.append(rgb.copy())
        shape = (3, 4) if self.bad_shape else rgb.shape[:2]
        mask = np.zeros(shape, dtype=bool)
        mask[:, :2] = True
        return [{"segmentation": mask, "area": int(mask.sum()), "predicted_iou": 0.9}]


class SAM2Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data_root, self.run_dir = self.root / "data", self.root / "runs" / "fixture"
        self.data_root.mkdir()
        rgb = np.zeros((12, 20, 3), dtype=np.uint8)
        rgb[..., 0], rgb[..., 1], rgb[..., 2] = 100, 40, 70
        Image.fromarray(rgb).save(self.data_root / "rgb.png")
        self.frame_id = "fixture:dev:00000"
        atomic_jsonl(self.run_dir / "frames.jsonl", [{
            "frame_id": self.frame_id, "source": "fixture", "scene_id": "fixture-scene",
            "sequence_id": "fixture-dev", "timestamp_s": None, "image_path": "rgb.png",
            "split": "development",
        }])
        atomic_json(self.run_dir / "metadata" / "data.json", {"is_fixture": True})

    def test_module_import_and_missing_checkpoint_do_not_import_optional_models(self):
        original_import = __import__
        def guarded(name, *args, **kwargs):
            if name == "torch" or name == "sam2" or name.startswith("sam2."):
                raise AssertionError("Optional model import was attempted")
            return original_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=guarded):
            importlib.reload(sam2)
            with self.assertRaises(FileNotFoundError):
                sam2.load_sam2_generator(self.root / "missing.pt")

    def fake_dependencies(self):
        checkpoint = self.root / "mock.pt"
        checkpoint.write_bytes(b"mock weights, CPU fixture only")
        package_path = self.root / "mock_sam2"
        config = package_path / sam2.DEFAULT_MODEL_CONFIG
        config.parent.mkdir(parents=True)
        config.write_text("fixture: true\n")
        package = types.ModuleType("sam2")
        package.__file__, package.__path__ = str(package_path / "__init__.py"), [str(package_path)]
        torch = types.ModuleType("torch")
        torch.inference_mode = nullcontext
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        build = types.ModuleType("sam2.build_sam")
        automatic = types.ModuleType("sam2.automatic_mask_generator")
        calls = []
        def build_model(*args, **kwargs):
            calls.append(("build", args, kwargs))
            return types.SimpleNamespace(image_size=64)
        class Automatic:
            def __init__(self, model, **settings):
                calls.append(("generator", model, settings))
            def generate(self, rgb):
                return [{"segmentation": np.ones(rgb.shape[:2], dtype=bool)}]
        build.build_sam2, automatic.SAM2AutomaticMaskGenerator = build_model, Automatic
        return checkpoint, calls, {"torch": torch, "sam2": package, "sam2.build_sam": build, "sam2.automatic_mask_generator": automatic}

    def test_lazy_local_loader_defaults_cpu_and_captures_provenance(self):
        checkpoint, calls, modules = self.fake_dependencies()
        with patch.dict(sys.modules, modules):
            generator = sam2.load_sam2_generator(checkpoint, sam2_revision="declared-fixture-revision")
        self.assertEqual(calls[0][1], (sam2.DEFAULT_MODEL_CONFIG, str(checkpoint)))
        self.assertEqual(calls[0][2], {"device": "cpu", "apply_postprocessing": False})
        self.assertEqual(calls[1][2]["output_mode"], "binary_mask")
        self.assertEqual(generator.metadata["checkpoint_sha256"], hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        self.assertEqual(generator.metadata["declared_code_revision"], "declared-fixture-revision")
        self.assertTrue(generator.metadata["model_config_sha256"])
        self.assertEqual(generator.metadata["preprocessing"]["internal_resolution"], 64)
        self.assertEqual(generator.generate(np.zeros((12, 20, 3), dtype=np.uint8))[0]["segmentation"].shape, (12, 20))
        with self.assertRaises(ValueError):
            generator.generate(np.zeros((12, 20, 3), dtype=float))

    def test_loader_rejects_implicit_cuda_and_rle_output(self):
        checkpoint, _, modules = self.fake_dependencies()
        with patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                sam2.load_sam2_generator(checkpoint, device="cuda")
            with self.assertRaisesRegex(ValueError, "binary_mask"):
                sam2.load_sam2_generator(checkpoint, generator_settings={"output_mode": "coco_rle"})

    def test_fixture_generation_original_alignment_and_resume_reuses_masks(self):
        generator = FixtureGenerator()
        result = sam2.generate_sam2_regions(self.run_dir, self.data_root, generator)
        self.assertEqual(generator.calls[0].shape, (12, 20, 3))
        self.assertEqual(generator.calls[0].dtype, np.uint8)
        self.assertEqual(generator.calls[0][0, 0].tolist(), [100, 40, 70])
        self.assertEqual(len(result), 1)
        pending = read_jsonl(self.run_dir / "annotations.jsonl")[0]
        self.assertIsNone(pending["reference_label"])
        freeze_selection(self.run_dir, self.data_root)
        repeated = sam2.generate_sam2_regions(self.run_dir, self.data_root, generator)
        self.assertEqual(len(generator.calls), 1)
        self.assertTrue(repeated[0]["selected_for_classification"])
        origin = read_json(self.run_dir / "metadata" / "masks.json")[result[0]["region_id"]]["origin"]
        self.assertEqual(origin, "synthetic_fixture")
        self.assertEqual(read_json(self.run_dir / "metadata" / "data.json"), {"is_fixture": True})

    def test_output_shape_and_unmarked_fixture_fail_without_publishing(self):
        with self.assertRaisesRegex(ValueError, "dimensions"):
            sam2.generate_sam2_regions(self.run_dir, self.data_root, FixtureGenerator(bad_shape=True))
        self.assertFalse((self.run_dir / "regions.jsonl").exists())
        atomic_json(self.run_dir / "metadata" / "data.json", {"is_fixture": False})
        with self.assertRaisesRegex(ValueError, "fixture"):
            sam2.generate_sam2_regions(self.run_dir, self.data_root, FixtureGenerator())

    def response(self, payload, content_type="application/octet-stream"):
        response = io.BytesIO(payload)
        response.headers = {"Content-Type": content_type, "Content-Length": str(len(payload))}
        return response

    def test_checkpoint_download_is_explicit_one_file_and_preserves_completed(self):
        checkpoint = self.root / "checkpoint.pt"
        with patch.object(sam2, "urlopen") as open_url:
            with self.assertRaisesRegex(RuntimeError, "opt-in"):
                sam2.download_sam2_checkpoint(checkpoint)
            open_url.assert_not_called()
        payload = b"synthetic checkpoint bytes, not real SAM2 weights"
        checksum = hashlib.sha256(payload).hexdigest()
        with patch.object(sam2, "urlopen", return_value=self.response(payload)) as open_url:
            self.assertEqual(sam2.download_sam2_checkpoint(checkpoint, allow_download=True, expected_sha256=checksum), checkpoint)
            url = open_url.call_args.args[0].full_url
            self.assertTrue(url.endswith("/sam2.1_hiera_tiny.pt"))
            sam2.download_sam2_checkpoint(checkpoint)
            self.assertEqual(open_url.call_count, 1)
        self.assertEqual(checkpoint.read_bytes(), payload)
        self.assertEqual(list(self.root.glob(".checkpoint-*.part")), [])
        with self.assertRaisesRegex(ValueError, "preserved"):
            sam2.download_sam2_checkpoint(checkpoint, expected_sha256="wrong")

    def test_download_errors_leave_no_completed_or_temporary_checkpoint(self):
        checkpoint = self.root / "checkpoint.pt"
        with patch.object(sam2, "urlopen", return_value=self.response(b"bad")):
            with self.assertRaisesRegex(ValueError, "checksum"):
                sam2.download_sam2_checkpoint(checkpoint, allow_download=True, expected_sha256="wrong")
        with patch.object(sam2, "urlopen", return_value=self.response(b"<html>quota</html>", "text/html")):
            with self.assertRaisesRegex(RuntimeError, "HTML"):
                sam2.download_sam2_checkpoint(checkpoint, allow_download=True)
        self.assertFalse(checkpoint.exists())
        self.assertEqual(list(self.root.glob(".checkpoint-*.part")), [])


if __name__ == "__main__":
    unittest.main()
