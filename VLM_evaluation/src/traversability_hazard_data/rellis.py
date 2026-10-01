"""Original semantic-ID references; no regions or learning-ID remapping."""

from pathlib import Path
import numpy as np
from PIL import Image

from .configuration import SEQUENCES
from .storage import hash_file


def derive_rellis_reference(semantic_path, policy):
    with Image.open(semantic_path) as image:
        labels = np.asarray(image).copy()
        if image.format != "PNG" or labels.ndim != 2 or labels.dtype.kind not in "uib":
            raise ValueError("RELLIS IDs must be an original single-channel integer PNG")
    spec = policy["datasets"]["rellis"]
    unexpected = set(map(int, np.unique(labels))) - set(map(int, spec["original_label_ids"]))
    if unexpected:
        raise ValueError(f"Unexpected RELLIS original IDs: {sorted(unexpected)}")
    valid = ~np.isin(labels, spec["ignore_label_ids"])
    masks = {concept: np.isin(labels, ids) for concept, ids in spec["hazard_label_ids"].items()}
    masks = {concept: mask for concept, mask in masks.items() if np.any(mask)}
    fractions = {concept: int(mask.sum()) / labels.size for concept, mask in masks.items()}
    return masks, valid, {
        "concept_area_fractions": fractions,
        "tiny_concepts": [c for c in policy["canonical_prompts"] if c in fractions and
                          fractions[c] < policy["tiny_concept_area_fraction"]],
        "ignored_pixel_count": int((~valid).sum()),
        "ignored_fraction": int((~valid).sum()) / labels.size,
    }


def discover_sequence(data_root, source_dir, sequence):
    """Reuse legacy publisher-tree and existing byte-preserving import layouts."""
    from traversability_data.sources import _directories, _merge_images, _images, _natural
    root = Path(data_root)
    sources = [Path(source_dir)] if source_dir else [root / "raw/rellis_rgb", root / "raw/rellis_semantic"]
    rgb_dirs = _directories(sources, sequence, {"pylon_camera_node"})
    id_dirs = _directories(sources, sequence, {"pylon_camera_node_label_id"})
    images = _merge_images(rgb_dirs) if rgb_dirs else _images(root / "rellis" / sequence / "rgb")
    ids = _merge_images(id_dirs, semantic=True) if id_dirs else _images(root / "rellis" / sequence / "semantic", semantic=True)
    # The legacy RGB-only fallback is deliberately never eligible here.
    aligned = sorted(set(images) & set(ids), key=lambda stem: (_natural(stem), stem))
    return images, ids, aligned


def select_sequence(data_root, source_dir, sequence, count):
    from traversability_data.sources import uniform_indices
    images, labels, aligned = discover_sequence(data_root, source_dir, sequence)
    if len(aligned) < count:
        return None, {"source": "rellis", "sequence_id": sequence, "requested": count,
                      "available": len(aligned), "reason": "insufficient_aligned_rgb_ids",
                      "action": "Supply matching pylon_camera_node and pylon_camera_node_label_id RGB/ID files (or existing rellis/<sequence>/rgb and semantic imports)."}
    selected = []
    indices = uniform_indices(len(aligned), count)
    for index in indices:
        stem = aligned[index]
        selected.append({"stem": stem, "frame_id": f"rellis:{sequence}:{stem}",
                         "image_path": f"rellis/{sequence}/rgb/{images[stem].name}",
                         "annotation_path": f"rellis/{sequence}/semantic/{labels[stem].name}",
                         "image_sha256": hash_file(images[stem]),
                         "annotation_sha256": hash_file(labels[stem])})
    return {"algorithm": "natural_uniform_endpoints_v1", "requested": count,
            "available_at_freeze": len(aligned), "candidate_stems": aligned,
            "selected_indices": indices, "selected": selected}, None
