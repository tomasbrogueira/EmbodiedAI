"""Resumable optional human annotation without replacing completed records."""

from pathlib import Path

from .storage import atomic_jsonl, data_path, exclusive_lock, read_json, read_jsonl
from .validation import validate_records


LABELS = ("traversable", "non_traversable", "unknown")
MASK_QUALITIES = ("valid", "mixed", "broken", "unchecked")


def _transfer_updates(run_dir, annotation):
    path = run_dir / "metadata/reference_transfer.jsonl"
    rows = read_jsonl(path, missing_ok=True)
    if len({row.get("region_id") for row in rows}) != len(rows):
        raise ValueError("Duplicate region_id in reference transfer")
    changed = False
    for row in rows:
        if row.get("region_id") == annotation["region_id"]:
            updates = {
                "reference_label": annotation["reference_label"],
                "reference_exclusion_reason": annotation.get("reference_exclusion_reason"),
                "reference_annotation_source": annotation.get("annotation_source", "manual"),
            }
            changed = any(row.get(key) != value for key, value in updates.items())
            row.update(updates)
    return path, rows, changed


def _publish_transfer(update):
    path, rows, changed = update
    if changed:
        try:
            atomic_jsonl(path, rows)
        except OSError as error:
            raise RuntimeError("Annotation is saved; reference transfer update failed. Retry the identical save to repair provenance.") from error


def _run_records(run_dir):
    run_dir = Path(run_dir)
    frames = read_jsonl(run_dir / "frames.jsonl")
    regions = read_jsonl(run_dir / "regions.jsonl")
    annotations = read_jsonl(run_dir / "annotations.jsonl", missing_ok=True)
    validate_records(frames, regions, annotations)
    return frames, regions, annotations


def save_annotation(run_dir, region_id, reference_label, mask_quality="unchecked",
                    semantic_class=None, hazard_type=None):
    """Save an explicit label, or null/pending, preserving completed annotations."""
    if reference_label is not None and reference_label not in LABELS:
        raise ValueError(f"Invalid reference label: {reference_label!r}")
    if mask_quality not in MASK_QUALITIES:
        raise ValueError(f"Invalid mask quality: {mask_quality!r}")
    for name, value in (("semantic_class", semantic_class), ("hazard_type", hazard_type)):
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{name} must be a string or null")
    run_dir = Path(run_dir)
    with exclusive_lock(run_dir / ".annotations.lock"):
        frames, regions, annotations = _run_records(run_dir)
        if region_id not in {region["region_id"] for region in regions}:
            raise ValueError(f"Unknown region_id: {region_id}")
        metadata = read_json(run_dir / "metadata" / "data.json", default={})
        candidate = {
            "region_id": region_id,
            "reference_label": reference_label,
            "semantic_class": semantic_class,
            "hazard_type": hazard_type,
            "mask_quality": mask_quality,
            "annotation_status": "pending" if reference_label is None else "complete",
            "robot_profile_id": metadata.get("robot_profile_id", "rellis_material_v1"),
            "annotation_source": "manual",
        }
        indexed = {row["region_id"]: row for row in annotations}
        existing = indexed.get(region_id)
        if existing and existing["annotation_status"] == "complete":
            if all(existing.get(key) == value for key, value in candidate.items()
                   if key != "annotation_source"):
                _publish_transfer(_transfer_updates(run_dir, existing))
                return dict(existing)
            raise ValueError(f"Completed annotation is immutable: {region_id}")
        indexed[region_id] = candidate
        merged = [indexed[region["region_id"]] for region in regions
                  if region["region_id"] in indexed]
        validate_records(frames, regions, merged)
        transfer_update = _transfer_updates(run_dir, candidate)
        atomic_jsonl(run_dir / "annotations.jsonl", merged)
        _publish_transfer(transfer_update)
        return dict(candidate)


def show_region(run_dir, data_root, region_id):
    """Return a matplotlib figure of the original RGB and aligned region mask."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from PIL import Image
    except ImportError as error:
        raise ImportError("Image display needs Pillow, numpy and matplotlib; install requirements/data.txt.") from error
    frames, regions, _ = _run_records(run_dir)
    region = next((row for row in regions if row["region_id"] == region_id), None)
    if region is None:
        raise ValueError(f"Unknown region_id: {region_id}")
    frame = next(row for row in frames if row["frame_id"] == region["frame_id"])
    with Image.open(data_path(data_root, frame["image_path"])) as image:
        rgb = np.asarray(image.convert("RGB"))
    with Image.open(data_path(data_root, region["mask_path"])) as image:
        if image.format != "PNG" or len(image.getbands()) != 1:
            raise ValueError("Region masks must be single-channel PNG images")
        mask = np.asarray(image) != 0
    if mask.shape != rgb.shape[:2]:
        raise ValueError(f"Mask/RGB dimensions differ for {region_id}")
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("Original RGB")
    axes[1].imshow(rgb)
    overlay = np.zeros((*mask.shape, 4), dtype=float)
    overlay[mask] = (1, 0.3, 0, 0.45)
    axes[1].imshow(overlay)
    axes[1].set_title("Aligned mask overlay")
    axes[2].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Region mask")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(region_id)
    figure.tight_layout()
    return figure


class AnnotationSession:
    """Function-based fallback: inspect pending_region_ids, show, then save."""

    def __init__(self, run_dir, data_root, selected_only=True):
        self.run_dir = Path(run_dir)
        self.data_root = Path(data_root)
        self.selected_only = selected_only

    @property
    def pending_region_ids(self):
        """Return pending planning regions in manifest order, joined by ID."""
        _, regions, annotations = _run_records(self.run_dir)
        completed = {row["region_id"] for row in annotations
                     if row["annotation_status"] == "complete"}
        return [row["region_id"] for row in regions
                if row["planning_relevant"] and row["region_id"] not in completed
                and (not self.selected_only or row["selected_for_classification"])]

    def show(self, region_id=None):
        """Display a supplied region or the first pending region."""
        region_id = region_id or next(iter(self.pending_region_ids), None)
        if region_id is None:
            raise ValueError("No pending regions remain")
        return show_region(self.run_dir, self.data_root, region_id)

    def save(self, reference_label, mask_quality="unchecked", semantic_class=None,
             hazard_type=None, region_id=None):
        """Save the supplied region, or the first pending region."""
        region_id = region_id or next(iter(self.pending_region_ids), None)
        if region_id is None:
            raise ValueError("No pending regions remain")
        return save_annotation(self.run_dir, region_id, reference_label,
                               mask_quality, semantic_class, hazard_type)


def annotation_viewer(run_dir, data_root, selected_only=True, use_widgets=True):
    """Return a resumable widget viewer, or an AnnotationSession fallback."""
    session = AnnotationSession(run_dir, data_root, selected_only)
    if not use_widgets:
        return session
    try:
        import ipywidgets as widgets
        from IPython.display import display
        import matplotlib.pyplot as plt
    except ImportError:
        return session
    region = widgets.Dropdown(description="Region:")
    label = widgets.Dropdown(options=[("Pending (no label)", None)] + [(x, x) for x in LABELS],
                             description="Label:")
    quality = widgets.Dropdown(options=MASK_QUALITIES, value="unchecked", description="Mask:")
    semantic = widgets.Text(description="Class:")
    hazard = widgets.Text(description="Hazard:")
    save = widgets.Button(description="Save annotation")
    output = widgets.Output()
    refreshing = False

    def render(change=None):
        if refreshing:
            return
        with output:
            output.clear_output(wait=True)
            if region.value is None:
                print("No pending regions remain. Completed annotations are preserved.")
                return
            _, _, rows = _run_records(session.run_dir)
            row = next((x for x in rows if x["region_id"] == region.value), {})
            label.value = row.get("reference_label")
            quality.value = row.get("mask_quality", "unchecked")
            semantic.value = row.get("semantic_class") or ""
            hazard.value = row.get("hazard_type") or ""
            figure = session.show(region.value)
            display(figure)
            plt.close(figure)

    def refresh():
        nonlocal refreshing
        refreshing = True
        previous = region.value
        region.options = session.pending_region_ids
        if previous in region.options:
            region.value = previous
        refreshing = False
        save.disabled = region.value is None
        render()

    def on_save(button):
        try:
            session.save(label.value, quality.value, semantic.value or None,
                         hazard.value or None, region.value)
            refresh()
        except (ValueError, OSError, RuntimeError) as error:
            with output:
                print(f"Not saved: {error}")

    region.observe(render, names="value")
    save.on_click(on_save)
    refresh()
    return widgets.VBox([region, widgets.HBox([label, quality]),
                         widgets.HBox([semantic, hazard]), save, output])
