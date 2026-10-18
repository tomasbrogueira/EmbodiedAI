"""Validate source masks and retain explicit positive/negative evidence."""
import numpy as np

POSITIVE = ('base_support_mask', 'sam_traversability_mask', 'vlm_traversability_mask')
NEGATIVE = ('vlm_non_traversability_mask', 'manual_non_traversability_mask')
CHANNELS = POSITIVE + NEGATIVE
FUSION_POLICY = 'positive_union_negative_veto_v1'


def fuse(masks, shape):
    if set(masks) - set(CHANNELS):
        raise ValueError('Unknown semantic channel')
    positive, negative = np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)
    for name, mask in masks.items():
        mask = np.asarray(mask)
        if mask.shape != shape or mask.dtype.kind != 'b':
            raise ValueError(f'{name} must be an aligned Boolean mask')
        if name in POSITIVE:
            positive |= mask
        else:
            negative |= mask
    return positive, negative


def colors(rgb, support, blocked):
    result = rgb.copy()
    result[support & ~blocked] = (30, 210, 90)
    result[blocked] = (255, 0, 0)
    return result
