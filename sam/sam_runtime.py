"""Optional SAM adapter: cached text features and one GPU mask union.

The original SamSegmenter and SAM source remain the reference implementation.
"""
from collections import OrderedDict

from sam.segment import SamSegmenter


class CachedSamSegmenter(SamSegmenter):
    def __init__(self, device, threshold, checkpoint=None, cache_size=16):
        if cache_size < 1:
            raise ValueError('Text cache size must be positive')
        super().__init__(device, threshold, checkpoint)
        self.cache_size = cache_size
        # Cache ownership is this loaded model/processor instance. Creating a
        # different model creates a different cache; no global tensor sharing.
        self.text_cache = OrderedDict()
        self.cache_hits = self.cache_misses = 0

    def _text_features(self, prompt):
        if prompt in self.text_cache:
            self.cache_hits += 1
            self.text_cache.move_to_end(prompt)
            return self.text_cache[prompt]
        self.cache_misses += 1
        features = self.processor.model.backbone.forward_text([prompt], device=self.processor.device)
        self.text_cache[prompt] = features
        if len(self.text_cache) > self.cache_size:
            self.text_cache.popitem(last=False)
        return features

    def union(self, prompts, shape):
        torch = self.torch
        with torch.inference_mode(), self._autocast():
            mask = torch.zeros(shape, dtype=torch.bool, device=self.processor.device)
            for prompt in prompts:
                self.processor.reset_all_prompts(self.state)
                self.state['backbone_out'].update(self._text_features(prompt))
                self.state['geometric_prompt'] = self.processor.model._get_dummy_prompt()
                # Retain the original instance confidence filter, interpolation
                # and mask threshold. Union only the resulting Boolean masks.
                output = self.processor._forward_grounding(self.state)['masks']
                if output.ndim != 4 or output.shape[1] != 1 or tuple(output.shape[2:]) != tuple(shape):
                    raise ValueError(f'SAM masks {tuple(output.shape)} do not match RGB/depth shape {shape}')
                mask |= output[:, 0].any(dim=0)
        return mask.cpu().numpy()

    def cache_stats(self):
        return {'hits': self.cache_hits, 'misses': self.cache_misses,
                'entries': len(self.text_cache), 'limit': self.cache_size}

    def union_groups(self, groups, shape):
        """Share query results, retaining source/class labels; one final transfer."""
        torch = self.torch
        keys, prompts = list(groups), {}
        with torch.inference_mode(), self._autocast():
            union = torch.zeros((len(keys), *shape), dtype=torch.bool, device=self.processor.device)
            for index, key in enumerate(keys):
                for prompt in groups[key]:
                    if prompt not in prompts:
                        self.processor.reset_all_prompts(self.state)
                        self.state['backbone_out'].update(self._text_features(prompt))
                        self.state['geometric_prompt'] = self.processor.model._get_dummy_prompt()
                        output = self.processor._forward_grounding(self.state)['masks']
                        if output.ndim != 4 or output.shape[1] != 1 or tuple(output.shape[2:]) != tuple(shape):
                            raise ValueError('SAM mask shape differs from the current RGB')
                        prompts[prompt] = output[:, 0].any(dim=0)
                    union[index] |= prompts[prompt]
        values = union.cpu().numpy()
        return dict(zip(keys, values))
