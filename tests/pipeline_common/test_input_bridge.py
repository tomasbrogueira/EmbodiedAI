"""CPU conformance for generic, exact-pixel model inputs and count audits.

The real legacy common/SAM reader modules are present throughout these tests;
only the SAM processor is injected in explicit fixture mode. No weights load.
"""
from dataclasses import replace
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
LEGACY_SOURCE = ROOT / "VLM_evaluation" / "src"
if str(LEGACY_SOURCE) not in sys.path:
    sys.path.insert(0, str(LEGACY_SOURCE))

from pipeline_common.contracts import CONTRACT_ID, FramePacket, SemanticFrame, validate_semantic
from pipeline_common import input_bridge
from pipeline_common.io import ENCODED_HASH_RECIPE, RGB_HASH_RECIPE, file_sha256, rgb_sha256
from traversability_hazard_inference.preparation import load_rgb as qwen_load_rgb
from traversability_hazard_segmentation import common as legacy_common
from traversability_hazard_segmentation import sam3_adapter


class FixtureProcessor:
    """Explicit CPU stand-in at the SAM processor boundary, retaining real readers."""

    def __init__(self):
        self.images = []
        self.prompts = []
        self.resets = 0

    def set_image(self, image):
        pixels = np.array(image, copy=True)
        self.images.append(pixels)
        return {"height": pixels.shape[0], "width": pixels.shape[1]}

    def reset_all_prompts(self, state):
        self.resets += 1
        state.pop("masks", None)
        state.pop("scores", None)

    def set_text_prompt(self, *, state, prompt):
        self.prompts.append(prompt)
        masks = np.zeros((1, state["height"], state["width"]), dtype=bool)
        masks[0, 1, 2] = True
        state.update(masks=masks, scores=np.array([.9]))
        return state


class InputBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        # Unequal channels and varied pixels reveal RGB swaps/crops/re-encoding.
        self.rgb = (np.arange(3 * 5 * 3).reshape(3, 5, 3) * 5 + 17).astype(np.uint8)
        self.path = self.root / "processed.png"
        Image.fromarray(self.rgb).save(self.path, compress_level=0)
        self.frame = FramePacket(
            sequence_id="phone_video_fixture", frame_id="frame_000007",
            timestamp_ns=None, timestamp_provenance={"clock": "unknown", "source": "fixture"},
            rgb=self.rgb.copy(), image_path=self.path,
            encoded_file_sha256=file_sha256(self.path), decoded_rgb_sha256=rgb_sha256(self.rgb),
            processed_grid_id="aligned_lingbot_grid_v1",
            source_rgb_identity={"source_grid_id": "original", "source_dimensions": [8, 12]},
            source_to_processed={"crop": [2, 1, 7, 5], "resize": [5, 3]}, geometry=None)

    def record(self):
        return input_bridge.to_model_input(self.frame)[0]

    def segmenter(self):
        processor = FixtureProcessor()
        segmenter = sam3_adapter.load_segmenter({"fixture": True, "processor": processor})
        self.addCleanup(segmenter.close)
        return segmenter, processor

    def semantic(self, **changes):
        frame = SemanticFrame(
            sequence_id=self.frame.sequence_id, frame_id=self.frame.frame_id,
            processed_grid_id=self.frame.processed_grid_id,
            decoded_rgb_sha256=self.frame.decoded_rgb_sha256, geometry_fingerprint=None,
            adapter_provenance={"fixture": True}, status="ok", query_count=0,
            timestamp_ns=self.frame.timestamp_ns)
        return replace(frame, **changes)

    def test_named_byte_and_pixel_hash_recipes_remain_distinct(self):
        record, root = input_bridge.to_model_input(self.frame)
        self.assertEqual(root, self.root.resolve())
        self.assertEqual(record["input_contract_id"], CONTRACT_ID)
        self.assertEqual(record["task_id"], CONTRACT_ID)
        self.assertEqual(record["encoded_hash_recipe"], ENCODED_HASH_RECIPE)
        self.assertEqual(record["decoded_hash_recipe"], RGB_HASH_RECIPE)
        self.assertEqual(record["image_sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(record["encoded_file_sha256"], record["image_sha256"])
        self.assertEqual(record["decoded_rgb_sha256"], hashlib.sha256(self.rgb.tobytes(order="C")).hexdigest())
        self.assertNotEqual(record["encoded_file_sha256"], record["decoded_rgb_sha256"])
        self.assertEqual(record["source_to_processed"], self.frame.source_to_processed)
        self.assertEqual(record["processed_grid_id"], self.frame.processed_grid_id)
        self.assertNotIn("source", record)
        self.assertNotIn("present_concepts", record)
        json.dumps(record, allow_nan=False)
        for changes in (
            {"image_sha256": record["decoded_rgb_sha256"]},
            {"encoded_file_sha256": record["decoded_rgb_sha256"]},
            {"decoded_rgb_sha256": record["encoded_file_sha256"]},
            {"encoded_hash_recipe": RGB_HASH_RECIPE},
            {"decoded_hash_recipe": ENCODED_HASH_RECIPE},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                input_bridge.validate_model_input({**record, **changes}, root)

    def test_same_pixels_different_png_bytes_keep_separate_encoded_identities(self):
        record = self.record()
        Image.fromarray(self.rgb).save(self.path, compress_level=9)
        changed_hash = file_sha256(self.path)
        self.assertNotEqual(changed_hash, record["encoded_file_sha256"])
        with self.assertRaisesRegex(ValueError, "Encoded-file identity"):
            input_bridge.to_model_input(self.frame)
        new_frame = replace(self.frame, encoded_file_sha256=changed_hash)
        newer, root = input_bridge.to_model_input(new_frame)
        self.assertNotEqual(newer["encoded_file_sha256"], record["encoded_file_sha256"])
        self.assertEqual(newer["decoded_rgb_sha256"], record["decoded_rgb_sha256"])
        np.testing.assert_array_equal(input_bridge.load_model_rgb(newer, root), self.rgb)

    def test_packet_requires_nonempty_uint8_rgb_and_exact_persisted_pixels(self):
        arrays = [self.rgb.astype(np.float32), self.rgb.astype(np.uint16), self.rgb[..., 0],
                  np.zeros((3, 5, 4), dtype=np.uint8), np.zeros((0, 5, 3), dtype=np.uint8),
                  self.rgb.tolist()]
        for rgb in arrays:
            with self.subTest(kind=type(rgb), shape=getattr(rgb, "shape", None)), self.assertRaises(ValueError):
                input_bridge.to_model_input(replace(self.frame, rgb=rgb))
        with self.assertRaisesRegex(ValueError, "FramePacket RGB hash"):
            input_bridge.to_model_input(replace(self.frame, decoded_rgb_sha256="0" * 64))
        changed = self.rgb.copy()
        changed[1, 2, 0] ^= 1
        with self.assertRaisesRegex(ValueError, "Decoded RGB identity"):
            input_bridge.to_model_input(replace(self.frame, rgb=changed, decoded_rgb_sha256=rgb_sha256(changed)))

    def test_noncontiguous_rgb_uses_declared_c_order_recipe(self):
        view = self.rgb[:, ::-1]
        self.assertFalse(view.flags.c_contiguous)
        Image.fromarray(view).save(self.path)
        frame = replace(self.frame, rgb=view, encoded_file_sha256=file_sha256(self.path),
                        decoded_rgb_sha256=hashlib.sha256(view.tobytes(order="C")).hexdigest())
        record, root = input_bridge.to_model_input(frame)
        np.testing.assert_array_equal(input_bridge.load_model_rgb(record, root), view)

    def test_generic_record_types_grid_dimensions_and_task_are_strict(self):
        record = self.record()
        changes = [
            {"input_contract_id": "unknown_contract"}, {"task_id": "hazard_prompt_v1"},
            {"schema_version": True}, {"schema_version": 1.0}, {"schema_version": "1"},
            {"width": True}, {"width": 5.0}, {"height": 4}, {"height": 0},
            {"sequence_id": None}, {"frame_id": 7}, {"frame_id": ""},
            {"processed_grid_id": None}, {"processed_grid_id": ""},
        ]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                input_bridge.validate_model_input({**record, **change}, self.root)

    def test_paths_must_stay_inside_declared_model_root(self):
        record = self.record()
        for path in ("../processed.png", str(self.path.resolve()), "", None, 7):
            with self.subTest(path=path), self.assertRaises((ValueError, TypeError, OSError)):
                input_bridge.load_model_rgb({**record, "image_path": path}, self.root)
        with self.assertRaises(OSError):
            input_bridge.load_model_rgb({**record, "image_path": "missing.png"}, self.root)

    def test_jpeg_gray_and_rgba_files_cannot_qualify_as_aligned_rgb_png(self):
        cases = [("JPEG", Image.fromarray(self.rgb)),
                 ("PNG", Image.fromarray(self.rgb[..., 0])),
                 ("PNG", Image.fromarray(np.concatenate((self.rgb, np.full((3, 5, 1), 255, np.uint8)), axis=2)))]
        for kind, image in cases:
            with self.subTest(kind=kind, mode=image.mode):
                image.save(self.path, format=kind)
                frame = replace(self.frame, encoded_file_sha256=file_sha256(self.path))
                with self.assertRaisesRegex(ValueError, "lossless RGB PNG"):
                    input_bridge.to_model_input(frame)

    def test_loader_decodes_verified_byte_snapshot_and_returns_owned_pixels(self):
        record = self.record()
        original_open = Image.open
        replacement = np.full_like(self.rgb, 233)
        streams = []

        def mutate_after_snapshot(stream, *args, **kwargs):
            self.assertIsInstance(stream, BytesIO)
            streams.append(stream.getvalue())
            # A file change after byte capture must not alter the decoded image.
            Image.fromarray(replacement).save(self.path)
            return original_open(stream, *args, **kwargs)

        with patch.object(Image, "open", side_effect=mutate_after_snapshot):
            image = input_bridge.load_model_rgb(record, self.root)
        self.assertEqual(hashlib.sha256(streams[0]).hexdigest(), record["encoded_file_sha256"])
        np.testing.assert_array_equal(image, self.rgb)
        self.rgb[:] = 0
        self.path.unlink()
        np.testing.assert_array_equal(image, self.frame.rgb)

    def test_sam_generic_input_uses_bridge_with_real_legacy_common_installed(self):
        record = self.record()
        self.assertEqual(Path(legacy_common.__file__).resolve(), (LEGACY_SOURCE / "traversability_hazard_segmentation/common.py").resolve())
        segmenter, processor = self.segmenter()
        with (patch.object(legacy_common, "validate_frame", wraps=legacy_common.validate_frame) as legacy,
              patch.object(input_bridge, "validate_model_input", wraps=input_bridge.validate_model_input) as validate,
              patch.object(input_bridge, "load_model_rgb", wraps=input_bridge.load_model_rgb) as load):
            result = segmenter.segment_image(record, ["water"], self.root)
        self.assertEqual(result["frame"]["status"], "ok", result["frame"]["error_code"])
        self.assertEqual(result["frame"]["sam_query_count"], 1)
        self.assertEqual(validate.call_count, 1)
        self.assertEqual(load.call_count, 1)
        self.assertEqual(legacy.call_count, 0)
        np.testing.assert_array_equal(processor.images[0], self.frame.rgb)
        self.assertEqual(processor.prompts, ["water"])
        self.assertTrue(result["queries"][0]["masks"][0][1, 2])
        self.assertTrue(result["settings"]["fixture"])

    def test_sam_task_discriminator_and_both_hashes_reject_before_encoding(self):
        record = self.record()
        for change in ({"task_id": "hazard_prompt_v1"}, {"input_contract_id": "wrong"},
                       {"schema_version": True}, {"decoded_rgb_sha256": "0" * 64},
                       {"encoded_file_sha256": "0" * 64}, {"image_sha256": "0" * 64}):
            with self.subTest(change=change):
                segmenter, processor = self.segmenter()
                result = segmenter.segment_image({**record, **change}, ["water"], self.root)
                self.assertEqual(result["frame"]["status"], "error")
                self.assertEqual(result["frame"]["sam_query_count"], 0)
                self.assertEqual(processor.images, [])
                self.assertEqual(processor.prompts, [])

    def test_sam_revalidates_snapshot_if_file_changes_after_initial_validation(self):
        record = self.record()
        segmenter, processor = self.segmenter()
        initial_validate = input_bridge.validate_model_input

        def replace_after_validation(*args, **kwargs):
            path = initial_validate(*args, **kwargs)
            Image.fromarray(np.full_like(self.rgb, 201)).save(path)
            return path

        with patch.object(input_bridge, "validate_model_input", side_effect=replace_after_validation):
            result = segmenter.segment_image(record, ["water"], self.root)
        self.assertEqual(result["frame"]["status"], "error")
        self.assertIn("Encoded-file identity mismatch", result["frame"]["error_code"])
        self.assertEqual(processor.images, [])

    def test_qwen_preparation_reaches_generic_bridge_and_ignores_reference_fields(self):
        record = {**self.record(), "present_concepts": ["poison annotation"], "references_path": "do_not_read.json"}
        with patch.object(input_bridge, "load_model_rgb", wraps=input_bridge.load_model_rgb) as load:
            image = qwen_load_rgb(record, self.root)
        load.assert_called_once_with(record, self.root)
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (5, 3))
        np.testing.assert_array_equal(image, self.frame.rgb)
        with self.assertRaisesRegex(ValueError, "Decoded RGB identity"):
            qwen_load_rgb({**record, "decoded_rgb_sha256": "0" * 64}, self.root)

    def test_legacy_validator_rejects_generic_sources_and_runtime_task(self):
        record = self.record()
        generic = {**record, "source": "phone_video", "scene_id": "fixture",
                   "split": "development", "timestamp_s": None, "fixture": True}
        with self.assertRaisesRegex(ValueError, "Unknown frame source"):
            legacy_common.validate_frame(generic, self.root)
        with self.assertRaisesRegex(ValueError, "task/schema mismatch"):
            legacy_common.validate_task(generic)
        # The original benchmark's complete fixture frame remains accepted.
        legacy = {**generic, "source": "coco", "task_id": "hazard_prompt_v1"}
        legacy.pop("input_contract_id")
        self.assertEqual(legacy_common.validate_frame(legacy, self.root), self.path)
        legacy_common.validate_task(legacy)
        segmenter, processor = self.segmenter()
        with patch.object(legacy_common, "validate_frame", wraps=legacy_common.validate_frame) as validate:
            result = segmenter.segment_image(legacy, ["water"], self.root)
        self.assertEqual(result["frame"]["status"], "ok")
        validate.assert_called_once_with(legacy, self.root)
        np.testing.assert_array_equal(processor.images[0], self.frame.rgb)

    def test_query_count_requires_genuine_nonnegative_int_or_explained_null(self):
        for value in (True, False, 1.0, "1", np.int64(1), -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_semantic(self.semantic(query_count=value), self.frame)
        for count in (0, 1, 16):
            validate_semantic(self.semantic(query_count=count), self.frame)
        with self.assertRaisesRegex(ValueError, "explicit reason"):
            validate_semantic(self.semantic(query_count=None), self.frame)
        validate_semantic(self.semantic(query_count=None, query_count_reason="SAM interrupted before an execution counter was returned"), self.frame)

    def test_null_query_count_audit_explanation_survives_serialization(self):
        from pipeline_common.runtime import semantic_record
        examples = [
            self.semantic(query_count=None, query_count_reason="explicit execution count unavailable"),
            self.semantic(query_count=None, adapter_provenance={"query_count_reason": "adapter audit unavailable"}),
            self.semantic(query_count=None, status="error", error={"code": "sam_interrupted", "message": "count unauditable"}),
        ]
        for example in examples:
            with self.subTest(reason=example.query_count_reason, error=example.error):
                validate_semantic(example, self.frame)
                row = semantic_record(example, self.root)
                self.assertIsNone(row["query_count"])
                self.assertTrue(row["query_count_reason"], "Accepted unavailable counts need a serialized top-level explanation for the evaluator")
                json.dumps(row, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
