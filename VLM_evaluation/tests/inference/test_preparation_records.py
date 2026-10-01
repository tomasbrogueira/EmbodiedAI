from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from traversability_inference.configuration import configuration_metadata, normalize_settings
from traversability_inference.preparation import input_path, prepare_target, resize_aligned
from traversability_inference.records import InferenceError, parse_response, prediction, validate_prediction


def target(tmp_path, mask):
    rgb = np.arange(12 * 14 * 3, dtype=np.uint8).reshape(12, 14, 3)
    Image.fromarray(rgb).save(tmp_path / "rgb.png")
    Image.fromarray(mask).save(tmp_path / "mask.png")
    frame = {"frame_id": "f", "image_path": "rgb.png"}
    region = {"region_id": "r", "frame_id": "f", "mask_path": "mask.png"}
    return frame, region, rgb


def test_outline_is_exterior_and_crop_is_original_padded_rgb(tmp_path):
    mask = np.zeros((12, 14), dtype=np.uint8)
    mask[3:9, 4:11] = 255
    frame, region, rgb = target(tmp_path, mask)
    prepared = prepare_target(frame, region, tmp_path)
    assert prepared.bbox == (2, 1, 13, 11)  # ceil(7*.2)=2; ceil(6*.2)=2 each side
    scene = np.asarray(prepared.scene)
    np.testing.assert_array_equal(scene[3:9, 4:11], rgb[3:9, 4:11])
    np.testing.assert_array_equal(scene[2, 3], [255, 0, 255])
    np.testing.assert_array_equal(scene[0, 0], rgb[0, 0])
    np.testing.assert_array_equal(np.asarray(prepared.crop), rgb[1:11, 2:13])
    assert prepare_target(frame, region, tmp_path).scene.tobytes() == prepared.scene.tobytes()


def test_border_padding_is_clipped(tmp_path):
    mask = np.zeros((12, 14), dtype=np.uint8)
    mask[:2, :3] = 1
    frame, region, _ = target(tmp_path, mask)
    assert prepare_target(frame, region, tmp_path).bbox == (0, 0, 4, 3)


@pytest.mark.parametrize("mask,code", [
    (np.zeros((12, 14), dtype=np.uint8), "mask_empty"),
    (np.ones((11, 14), dtype=np.uint8), "mask_shape_mismatch"),
    (np.ones((12, 14, 3), dtype=np.uint8), "mask_format_invalid"),
])
def test_mask_errors(tmp_path, mask, code):
    frame, region, _ = target(tmp_path, mask)
    with pytest.raises(InferenceError) as caught:
        prepare_target(frame, region, tmp_path)
    assert caught.value.code == code


def test_missing_and_mismatched_inputs(tmp_path):
    frame, region, _ = target(tmp_path, np.ones((12, 14), dtype=np.uint8))
    region["frame_id"] = "different"
    with pytest.raises(InferenceError, match="belong"):
        prepare_target(frame, region, tmp_path)
    region["frame_id"] = "f"
    region["mask_path"] = "missing.png"
    with pytest.raises(InferenceError) as caught:
        prepare_target(frame, region, tmp_path)
    assert caught.value.code == "mask_read_failed"


@pytest.mark.parametrize("path", ["../escape.png", "C:/escape.png", "/escape.png", "dir\\mask.png"])
def test_input_path_cannot_escape_root(tmp_path, path):
    with pytest.raises(InferenceError):
        input_path(tmp_path, path)


@pytest.mark.parametrize("size", [(1920, 1080), (1080, 1920), (3, 9000), (9000, 3), (1, 1), (640, 480)])
def test_aligned_images_respect_both_budgets(size):
    image = Image.new("RGB", size)
    for budget in (262144, 393216):
        result = resize_aligned(image, budget, 32)
        assert result.width % 32 == result.height % 32 == 0
        assert result.width * result.height <= budget


@pytest.mark.parametrize("raw,code", [
    ('```json\n{"label":"unknown","semantic_class":null,"reason":"x"}\n```', "parse_json"),
    ('prefix {"label":"unknown","semantic_class":null,"reason":"x"}', "parse_json"),
    ('{"label":"unknown","semantic_class":null,"reason":"x"} suffix', "parse_json"),
    ('{"label":"unknown","label":"traversable","semantic_class":null,"reason":"x"}', "parse_duplicate_key"),
    ('{"label":"unknown","semantic_class":null,"reason":"x","score":0.9}', "parse_keys"),
    ('{"label":"safe","semantic_class":null,"reason":"x"}', "parse_label"),
    ('{"label":true,"semantic_class":null,"reason":"x"}', "parse_label"),
    ('{"label":"unknown","semantic_class":3,"reason":"x"}', "parse_type"),
    ('{"label":"unknown","semantic_class":null,"reason":null}', "parse_type"),
    ('[{"label":"unknown","semantic_class":null,"reason":"x"}]', "parse_keys"),
])
def test_strict_parser(raw, code):
    with pytest.raises(InferenceError) as caught:
        parse_response(raw)
    assert caught.value.code == code


def test_raw_response_schema_and_unknown_errors():
    region = {"region_id": "r", "frame_id": "f"}
    raw = '  {"label":"traversable", "semantic_class":"concrete", "reason":"dry"}\n'
    record = prediction(region, "qwen3_vl_4b", parsed=parse_response(raw), raw_response=raw)
    assert record["raw_response"] == raw
    assert record["status"] == "ok"
    error = prediction(region, "qwen3_vl_4b", raw_response="bad", error_code="parse_json", reason="invalid")
    assert error["label"] == "unknown" and error["semantic_class"] is None
    assert len(error) == 9
    with pytest.raises(ValueError):
        validate_prediction(dict(error, extra=True))
    with pytest.raises(ValueError):
        validate_prediction(dict(error, label="traversable"))


def test_package_and_adapter_imports_do_not_import_model_libraries():
    source = str(Path(__file__).resolve().parents[2] / "src")
    script = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "import traversability_inference; import traversability_inference.qwen; "
        "import traversability_inference.clip; "
        "assert not any(name in sys.modules for name in "
        "['torch','transformers','bitsandbytes','accelerate','PIL','numpy'])"
    )
    subprocess.run([sys.executable, "-c", script, source], check=True)


def test_explicit_fallback_and_portable_identity():
    assert normalize_settings("qwen3_5_2b", {})["robot_profile_id"] == "rellis_material_v1"
    with pytest.raises(ValueError):
        normalize_settings("automatic", {})
    with pytest.raises(ValueError):
        normalize_settings("qwen3_5_4b", {"enable_thinking": True})
    first = configuration_metadata("qwen3_5_4b", {"cache_dir": "C:/local/cache", "calibration_path": "a"})
    second = configuration_metadata("qwen3_5_4b", {"cache_dir": "/other/cache", "calibration_path": "b"})
    assert first == second
    assert first != configuration_metadata("qwen3_5_4b", {"robot_profile_id": "small_wheeled_v1"})
