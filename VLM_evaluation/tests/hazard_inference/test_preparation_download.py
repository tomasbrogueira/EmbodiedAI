"""Injected CPU image/Hub fixtures: no image packages or network required."""

import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from traversability_hazard_inference.configuration import MODEL_SPECS
from traversability_hazard_inference.download import download_weights
from traversability_hazard_inference.preparation import load_rgb, resize_aligned
from traversability_hazard_inference.records import InferenceError


class InjectedImage:
    """Synthetic RGB stand-in; reads only our JSON fixture bytes."""
    def __init__(self, width, height, pixels="original", mode="RGB"):
        self.width, self.height = width, height
        self.size, self.mode = (width, height), mode
        self.pixels, self.closed, self.loaded = pixels, False, False
        self.conversions, self.resize_calls = [], []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.closed = True

    def load(self):
        self.loaded = True

    def convert(self, mode):
        self.conversions.append(mode)
        return InjectedImage(self.width, self.height, self.pixels, mode)

    def resize(self, size, resample):
        self.resize_calls.append((size, resample))
        return InjectedImage(*size, pixels=self.pixels, mode=self.mode)


class PreparationFixtureTests(unittest.TestCase):
    def setUp(self):
        self.opened = []
        image_module = ModuleType("PIL.Image")
        image_module.Resampling = SimpleNamespace(BICUBIC="fixture bicubic")

        def open_image(stream):
            self.assertIsInstance(stream, BytesIO)
            data = json.loads(stream.getvalue())
            image = InjectedImage(data["width"], data["height"], data["pixels"], data["mode"])
            self.opened.append(image)
            return image

        image_module.open = open_image
        pil = ModuleType("PIL")
        pil.Image = image_module
        context = patch.dict(sys.modules, {"PIL": pil, "PIL.Image": image_module})
        context.start()
        self.addCleanup(context.stop)

    def fixture(self, root):
        contents = json.dumps({"width": 64, "height": 96, "mode": "RGBA", "pixels": "all original fixture pixels"}).encode()
        (root / "image.fixture").write_bytes(contents)
        return {"frame_id": "fixture:rgb", "image_path": "image.fixture", "width": 64,
                "height": 96, "image_sha256": hashlib.sha256(contents).hexdigest()}

    def test_original_size_pixels_and_hash_before_rgb_conversion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame = self.fixture(root)
            (root / "references.jsonl").write_text("DO NOT READ", encoding="utf-8")
            result = load_rgb({**frame, "present_concepts": ["poison label"]}, root)
            self.assertEqual(result.size, (64, 96))
            self.assertEqual(result.mode, "RGB")
            self.assertEqual(result.pixels, "all original fixture pixels")
            self.assertEqual(self.opened[0].conversions, ["RGB"])
            self.assertTrue(self.opened[0].loaded)
            self.assertTrue(self.opened[0].closed)
            self.assertFalse(result.closed)

    def test_bad_hash_and_dimensions_are_explicit_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame = self.fixture(root)
            with self.assertRaises(InferenceError) as caught:
                load_rgb({**frame, "image_sha256": "0" * 64}, root)
            self.assertEqual(caught.exception.code, "image_identity_mismatch")
            self.assertEqual(self.opened, [])
            with self.assertRaises(InferenceError) as caught:
                load_rgb({**frame, "width": 1}, root)
            self.assertEqual(caught.exception.code, "image_dimensions_mismatch")
            self.assertTrue(self.opened[-1].closed)

    def test_unavailable_and_escaping_input_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame = self.fixture(root)
            for path in ("missing.fixture", "../outside.fixture"):
                with self.subTest(path=path), self.assertRaises(InferenceError) as caught:
                    load_rgb({**frame, "image_path": path}, root)
                self.assertEqual(caught.exception.code, "image_unavailable")

    def test_invalid_image_decode_is_an_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contents = b"synthetic invalid image"
            (root / "image.fixture").write_bytes(contents)
            frame = {"image_path": "image.fixture", "image_sha256": hashlib.sha256(contents).hexdigest(), "width": 1, "height": 1}
            with self.assertRaises(InferenceError) as caught:
                load_rgb(frame, root)
            self.assertEqual(caught.exception.code, "image_decode_failure")

    def test_alignment_cap_and_extreme_aspects(self):
        for width, height in ((1800, 1200), (6000, 20), (20, 6000), (1, 1), (64, 96)):
            with self.subTest(size=(width, height)):
                original = InjectedImage(width, height)
                result = resize_aligned(original, 1024 * 32 * 32, 32)
                self.assertEqual(result.width % 32, 0)
                self.assertEqual(result.height % 32, 0)
                self.assertLessEqual(result.width * result.height, 1024 * 32 * 32)
                self.assertEqual(result.pixels, original.pixels)
                self.assertEqual(original.resize_calls, [(result.size, "fixture bicubic")])
                self.assertFalse(original.closed)
        with self.assertRaises(InferenceError):
            resize_aligned(InjectedImage(64, 96), 1, 32)


class DownloadFixtureTests(unittest.TestCase):
    def test_only_explicit_download_calls_pinned_hub_without_model_imports(self):
        calls = []
        hub = ModuleType("huggingface_hub")
        hub.snapshot_download = lambda **kwargs: calls.append(kwargs) or "fixture cached path"
        with patch.dict(sys.modules, {"huggingface_hub": hub}):
            for key in MODEL_SPECS:
                result = download_weights(key, {"cache_dir": "fixture-cache"})
                self.assertEqual(result, "fixture cached path")
                self.assertEqual(calls[-1]["repo_id"], MODEL_SPECS[key]["checkpoint"])
                self.assertEqual(calls[-1]["revision"], MODEL_SPECS[key]["revision"])
                self.assertFalse(calls[-1]["local_files_only"])
                self.assertIn("*.safetensors", calls[-1]["allow_patterns"])
            with self.assertRaises(ValueError):
                download_weights("qwen3_vl_4b", {"fixture": True})
            self.assertEqual(len(calls), 3)


if __name__ == "__main__":
    unittest.main()
