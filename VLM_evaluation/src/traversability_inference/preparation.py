"""RGB-only scene outlining and original-image padded crops."""

from dataclasses import dataclass
import math
from pathlib import Path, PurePosixPath

from .records import InferenceError


@dataclass(frozen=True)
class PreparedTarget:
    scene: object
    crop: object
    bbox: tuple


def input_path(data_root, relative_path):
    """Resolve a portable input path without escaping its data root."""
    if not isinstance(relative_path, str) or not relative_path or "\\" in relative_path:
        raise InferenceError("input_path_invalid", "Input paths must be POSIX relative paths")
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts or ":" in relative_path:
        raise InferenceError("input_path_invalid", "Input path escapes its data root")
    root = Path(data_root).resolve()
    result = root.joinpath(*relative.parts).resolve()
    if not result.is_relative_to(root):
        raise InferenceError("input_path_invalid", "Resolved input path escapes data root")
    return result


def prepare_target(frame, region, data_root):
    """Render an exterior outline; crop original RGB with rounded-up 20% padding."""
    import numpy as np
    from PIL import Image

    if region.get("frame_id") != frame.get("frame_id"):
        raise InferenceError("region_frame_mismatch", "Region does not belong to this frame")
    try:
        with Image.open(input_path(data_root, frame["image_path"])) as source:
            rgb = source.convert("RGB")
    except InferenceError:
        raise
    except (OSError, KeyError, ValueError) as error:
        raise InferenceError("image_read_failed", str(error)) from error
    try:
        with Image.open(input_path(data_root, region["mask_path"])) as source:
            if source.format != "PNG" or source.mode not in {"1", "L", "I", "I;16", "P"}:
                raise InferenceError("mask_format_invalid", "Mask must be single-channel PNG")
            if source.size != rgb.size:
                raise InferenceError("mask_shape_mismatch", "Mask dimensions differ from RGB")
            mask = np.asarray(source) != 0
    except InferenceError:
        raise
    except (OSError, KeyError, ValueError) as error:
        raise InferenceError("mask_read_failed", str(error)) from error
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise InferenceError("mask_empty", "Mask contains no foreground")
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    pad_x, pad_y = math.ceil((x1 - x0) * 0.2), math.ceil((y1 - y0) * 0.2)
    bbox = (max(0, x0 - pad_x), max(0, y0 - pad_y),
            min(rgb.width, x1 + pad_x), min(rgb.height, y1 + pad_y))
    padded = np.pad(mask, 1)
    dilation = np.zeros_like(mask)
    for dy in range(3):
        for dx in range(3):
            dilation |= padded[dy:dy + rgb.height, dx:dx + rgb.width]
    exterior = dilation & ~mask
    scene = np.asarray(rgb).copy()
    scene[exterior] = (255, 0, 255)
    return PreparedTarget(Image.fromarray(scene), rgb.crop(bbox), bbox)


def resize_aligned(image, max_pixels, factor):
    """Resize independently to processor-aligned dimensions within a hard pixel cap."""
    from PIL import Image

    if type(factor) is not int or factor <= 0 or type(max_pixels) is not int or max_pixels < factor * factor:
        raise InferenceError("image_budget_invalid", "Pixel budget must fit one alignment cell")
    scale = min(1.0, math.sqrt(max_pixels / (image.width * image.height)))
    width = max(factor, math.floor(image.width * scale / factor) * factor)
    height = max(factor, math.floor(image.height * scale / factor) * factor)
    # Minimum alignment can inflate a very thin image; shrink its long dimension.
    if width * height > max_pixels:
        if width >= height:
            width = max(factor, (max_pixels // height // factor) * factor)
        else:
            height = max(factor, (max_pixels // width // factor) * factor)
    if width * height > max_pixels:
        raise InferenceError("image_budget_invalid", "Aligned image exceeds pixel cap")
    return image.resize((width, height), Image.Resampling.BICUBIC)
