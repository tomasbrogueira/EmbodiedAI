"""SAM3 mask loading, alignment, and per-frame mask indexing."""

from pathlib import Path

import numpy as np

from io_utils import frame_number


def load_sam_mask(path: Path) -> np.ndarray:
    """Load out_binary_masks and union masks into one 2D binary mask."""
    with np.load(path) as data:
        if "out_binary_masks" not in data.files:
            raise KeyError(
                f"'out_binary_masks' missing from {path}. "
                f"Available keys: {data.files}"
            )
        masks = np.asarray(data["out_binary_masks"])

    # Remove extra batch dimensions while preserving N x H x W.
    while masks.ndim > 3 and masks.shape[0] == 1:
        masks = masks[0]

    if masks.ndim == 2:
        mask = masks.astype(bool)
    elif masks.ndim == 3:
        mask = np.any(masks.astype(bool), axis=0)
    else:
        raise ValueError(f"Unexpected SAM3 mask shape in {path}: {masks.shape}")
    return mask


def preprocess_mask(
    mask: np.ndarray,
    target_shape: tuple[int, int],
    patch_size: int,
) -> np.ndarray:
    """Resize by nearest-neighbour sampling and center-crop to depth shape.

    This intentionally preserves the working preprocessing procedure from
    the original script. No OpenCV dependency is required.
    """
    original_h, original_w = mask.shape
    target_h, target_w = target_shape

    if original_h <= 0 or original_w <= 0:
        raise ValueError(f"Invalid SAM3 mask dimensions: {mask.shape}")
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")

    new_width = target_w
    new_height = (
        round(original_h * (new_width / original_w) / patch_size)
        * patch_size
    )
    if new_height <= 0:
        raise ValueError(f"Invalid resized mask height: {new_height}")

    # Nearest-neighbour resizing using NumPy index maps.
    y_indices = np.arange(new_height, dtype=np.int64) * original_h // new_height
    x_indices = np.arange(new_width, dtype=np.int64) * original_w // new_width
    resized = mask[y_indices[:, None], x_indices[None, :]]

    # Center-crop to the depth map's exact height.
    if new_height >= target_h:
        start_y = (new_height - target_h) // 2
        resized = resized[start_y : start_y + target_h, :]
    else:
        raise ValueError(
            f"Resized mask height {new_height} is smaller than the target "
            f"height {target_h}. Check the image preprocessing."
        )

    if resized.shape != target_shape:
        raise ValueError(
            f"Processed mask shape {resized.shape} does not match depth "
            f"shape {target_shape}."
        )
    return resized.astype(bool)


def build_sam_index(
    sam_dir: Path,
    video_name: str,
    prompt: str,
) -> dict[int, Path]:
    """Build a mapping from frame number to its SAM3 mask file."""
    if not sam_dir.exists():
        raise FileNotFoundError(f"SAM3 mask directory not found:\n{sam_dir}")

    sam_files = sorted(
        sam_dir.rglob(f"{video_name}_{prompt}_frame*.npz")
    )
    if not sam_files:
        raise RuntimeError(
            f"No SAM3 mask files matching '{video_name}_{prompt}_frame*.npz' "
            f"found in:\n{sam_dir}"
        )

    sam_by_frame: dict[int, Path] = {}
    for path in sam_files:
        idx = frame_number(path)
        if idx in sam_by_frame:
            raise RuntimeError(
                f"Multiple SAM3 masks found for frame {idx}:\n"
                f"{sam_by_frame[idx]}\n{path}"
            )
        sam_by_frame[idx] = path
    return sam_by_frame
