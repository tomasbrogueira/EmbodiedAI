"""Data preparation APIs; importing this package loads no models or downloads."""

from .storage import DataRoots, resolve_roots, repository_root, read_json, read_jsonl
from .validation import validate_records
from .run import prepare_run, add_regions, coverage_summary, export_run
from .downloads import resource_catalog, download_resource, import_archive
from .sources import import_rellis, import_tum, import_phone_video, uniform_indices
from .masks import import_sam_masks, freeze_selection
from .sam2 import load_sam2_generator, generate_sam2_regions, download_sam2_checkpoint
from .annotations import save_annotation, show_region, annotation_viewer
from .references import derive_references
from .fixture import create_cpu_fixture

__all__ = ["DataRoots", "resolve_roots", "repository_root", "read_json", "read_jsonl",
           "validate_records", "prepare_run", "add_regions", "coverage_summary", "export_run",
           "resource_catalog", "download_resource", "import_archive", "import_rellis", "import_tum",
           "import_phone_video", "uniform_indices", "import_sam_masks", "freeze_selection",
           "load_sam2_generator", "generate_sam2_regions", "download_sam2_checkpoint",
           "save_annotation", "show_region", "annotation_viewer", "derive_references", "create_cpu_fixture"]
