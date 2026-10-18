"""Independent semantic query groups sharing a fresh image embedding."""
import numpy as np
from PIL import Image
from vlm.prompts import normalize_prompts, validate_target

CHANNELS = ('base_support_mask', 'sam_traversability_mask', 'vlm_traversability_mask',
            'vlm_non_traversability_mask', 'manual_non_traversability_mask')


def query_groups(vlm_prompts=(), target='non_traversable', extra=(), manual=(), support=()):
    validate_target(target)
    return {
        'base_support_mask': normalize_prompts(list(support)),
        'sam_traversability_mask': normalize_prompts(list(extra)),
        'vlm_traversability_mask': normalize_prompts(list(vlm_prompts)) if target == 'traversable' else [],
        'vlm_non_traversability_mask': normalize_prompts(list(vlm_prompts)) if target == 'non_traversable' else [],
        'manual_non_traversability_mask': normalize_prompts(list(manual)),
    }


def segment_groups(segmenter, rgb, groups):
    """Return per-source Boolean masks. Fixed queries never depend on VLM state."""
    from sam.segment import union_masks
    if set(groups) - set(CHANNELS):
        raise ValueError('Unknown semantic query source')
    groups = {key: normalize_prompts(value) for key, value in groups.items()}
    masks = {key: np.zeros(rgb.shape[:2], dtype=bool) for key in groups}
    if not any(groups.values()):
        return masks
    if segmenter is None:
        raise ValueError('Active SAM queries require a segmenter')
    segmenter.set_image(Image.fromarray(rgb))
    if hasattr(segmenter, 'union_groups'):
        return segmenter.union_groups(groups, rgb.shape[:2])
    cache = {}
    for key, prompts in groups.items():
        for prompt in prompts:
            if prompt not in cache:
                cache[prompt] = union_masks(segmenter.masks(prompt), rgb.shape[:2])
            masks[key] |= cache[prompt]
    return masks
