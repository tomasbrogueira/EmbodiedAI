"""Exact, lossless processed-grid input for SAM3 and Qwen model readers."""
from pathlib import Path
from io import BytesIO
import hashlib
import numpy as np
from PIL import Image
from .contracts import CONTRACT_ID, FramePacket
from .io import file_sha256, rgb_sha256, safe_path, ENCODED_HASH_RECIPE, RGB_HASH_RECIPE

def _owned_input(record: dict, data_root):
    if record.get("input_contract_id") != CONTRACT_ID or type(record.get("schema_version")) is not int or record["schema_version"] != 1:
        raise ValueError("Generic model-input contract/schema mismatch")
    for key in ("sequence_id", "frame_id", "processed_grid_id"):
        if not isinstance(record.get(key), str) or not record[key]: raise ValueError(f"Missing {key}")
    for key in ("width", "height"):
        if type(record.get(key)) is not int or record[key] < 1: raise ValueError(f"Invalid {key}")
    if record.get("encoded_hash_recipe") != ENCODED_HASH_RECIPE or record.get("decoded_hash_recipe") != RGB_HASH_RECIPE:
        raise ValueError("Model input hash recipes mismatch")
    path = safe_path(data_root, record.get("image_path", ""))
    contents = path.read_bytes()
    encoded = hashlib.sha256(contents).hexdigest()
    if encoded != record.get("encoded_file_sha256") or encoded != record.get("image_sha256"):
        raise ValueError("Encoded-file identity mismatch")
    with Image.open(BytesIO(contents)) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != (record["width"], record["height"]):
            raise ValueError("Aligned model input must be lossless RGB PNG on the processed grid")
        image.load()
        rgb = np.array(image, dtype=np.uint8)
    if rgb_sha256(rgb) != record.get("decoded_rgb_sha256"):
        raise ValueError("Decoded RGB identity mismatch")
    if record.get("task_id") not in (None, CONTRACT_ID): raise ValueError("Generic input task mismatch")
    return path, Image.fromarray(rgb)

def validate_model_input(record: dict, data_root) -> Path:
    return _owned_input(record, data_root)[0]

def to_model_input(frame: FramePacket) -> tuple[dict, Path]:
    if not isinstance(frame.rgb,np.ndarray) or frame.rgb.dtype!=np.uint8 or frame.rgb.ndim!=3 or frame.rgb.shape[-1]!=3 or min(frame.rgb.shape[:2])<1:
        raise ValueError("FramePacket.rgb must be nonempty uint8[H,W,3]")
    if rgb_sha256(frame.rgb)!=frame.decoded_rgb_sha256:
        raise ValueError("FramePacket RGB hash mismatch")
    path = Path(frame.image_path).resolve()
    record = {"input_contract_id": CONTRACT_ID, "task_id": CONTRACT_ID, "schema_version": 1,
        "sequence_id": frame.sequence_id, "frame_id": frame.frame_id,
        "processed_grid_id": frame.processed_grid_id, "image_path": path.name,
        "width": frame.rgb.shape[1], "height": frame.rgb.shape[0],
        "timestamp_ns": frame.timestamp_ns, "timestamp_provenance": frame.timestamp_provenance,
        "image_sha256": frame.encoded_file_sha256, "encoded_file_sha256": frame.encoded_file_sha256,
        "decoded_rgb_sha256": frame.decoded_rgb_sha256,
        "encoded_hash_recipe": ENCODED_HASH_RECIPE, "decoded_hash_recipe": RGB_HASH_RECIPE,
        "source_rgb_identity": frame.source_rgb_identity, "source_to_processed": frame.source_to_processed}
    _, image = _owned_input(record, path.parent)
    if not np.array_equal(np.asarray(image), frame.rgb): raise ValueError("Persisted pixels differ from FramePacket.rgb")
    return record, path.parent

def load_model_rgb(record: dict, data_root):
    return _owned_input(record, data_root)[1]
