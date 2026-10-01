"""Whole-image annotation references. No model loading or implicit downloads."""

from .runner import prepare_run
from .validation import validate_run
from .downloads import download_coco_annotations, download_selected_coco_images

__all__ = ["prepare_run", "validate_run", "download_coco_annotations", "download_selected_coco_images"]
