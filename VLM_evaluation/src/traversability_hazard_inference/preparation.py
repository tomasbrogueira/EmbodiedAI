"""Whole RGB loading and policy-independent aligned resizing."""

import hashlib
from io import BytesIO
import math

from .records import InferenceError
from .storage import safe_path


def load_rgb(frame, data_root):
    """Return an owned original-size RGB image; never inspect other assets."""
    from PIL import Image

    try:
        path = safe_path(data_root, frame["image_path"])
        contents = path.read_bytes()
    except (OSError, ValueError, KeyError) as exc:
        raise InferenceError("image_unavailable", f"Cannot read RGB input: {exc}") from exc
    if hashlib.sha256(contents).hexdigest() != frame["image_sha256"]:
        raise InferenceError("image_identity_mismatch", "RGB bytes differ from the frozen frame hash")
    try:
        with Image.open(BytesIO(contents)) as original:
            if original.size != (frame["width"], frame["height"]):
                raise InferenceError("image_dimensions_mismatch", "RGB dimensions differ from the manifest")
            original.load()
            return original.convert("RGB")
    except InferenceError:
        raise
    except Exception as exc:
        raise InferenceError("image_decode_failure", f"Cannot decode original RGB: {type(exc).__name__}: {exc}") from exc


def resize_aligned(image, max_pixels, factor):
    """Adapted independent legacy resize; no crops, outlines or masks."""
    from PIL import Image

    if type(factor) is not int or factor <= 0 or type(max_pixels) is not int or max_pixels < factor * factor:
        raise InferenceError("image_budget_invalid", "Pixel budget must fit one alignment cell")
    if image.width <= 0 or image.height <= 0:
        raise InferenceError("image_dimensions_invalid", "RGB dimensions must be positive")
    scale = min(1.0, math.sqrt(max_pixels / (image.width * image.height)))
    width = max(factor, math.floor(image.width * scale / factor) * factor)
    height = max(factor, math.floor(image.height * scale / factor) * factor)
    if width * height > max_pixels:
        if width >= height:
            width = max(factor, (max_pixels // height // factor) * factor)
        else:
            height = max(factor, (max_pixels // width // factor) * factor)
    if width * height > max_pixels:
        raise InferenceError("image_budget_invalid", "Aligned image exceeds pixel cap")
    return image.resize((width, height), Image.Resampling.BICUBIC)
