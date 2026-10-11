from collections import OrderedDict
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from plugins.small_obstacles.compare_geometry import fit_similarity
from plugins.small_obstacles.runtime_metrics import CudaMemoryMonitor, cache_memory, release_unused_cuda
from plugins.small_obstacles.run_streaming import WorkerError
from plugins.small_obstacles.sam_runtime import CachedSamSegmenter


class RuntimeTests(unittest.TestCase):
    def test_shared_device_memory_and_worker_peaks_remain_distinct(self):
        cuda = SimpleNamespace(mem_get_info=lambda: (40, 100), memory_allocated=lambda: 10,
            memory_reserved=lambda: 20, max_memory_allocated=lambda: 15, max_memory_reserved=lambda: 25)
        with tempfile.TemporaryDirectory() as directory:
            monitor = CudaMemoryMonitor(SimpleNamespace(cuda=cuda), 'lingbot', Path(directory)/'trace', 0)
            first = monitor.snapshot()
            cuda.mem_get_info = lambda: (70, 100)
            cuda.max_memory_allocated = lambda: 12
            second = monitor.snapshot()
            monitor.close()
        self.assertEqual(first['device_used_bytes'], 60)
        self.assertEqual(second['device_used_bytes'], 30)
        self.assertEqual(second['device_sampled_peak_used_bytes'], 60)
        self.assertEqual(second['peak_allocated_bytes'], 15)
        self.assertEqual(second['allocated_bytes'], 10)

    def test_cache_budget_includes_evicted_special_tokens(self):
        class Tensor:
            dtype = 'bf16'
            def __init__(self, shape):
                self.shape = shape
            def numel(self):
                return int(np.prod(self.shape))
            def element_size(self):
                return 2
        tensors = {'k_0': Tensor((1, 2, 10, 20, 4)), 'v_0': Tensor((1, 2, 10, 20, 4)),
                   'k_0_special': Tensor((1, 2, 5, 6, 4)), '_skip_append': False}
        result = cache_memory(SimpleNamespace(aggregator=SimpleNamespace(kv_cache=tensors)))
        self.assertEqual(result['patch_keyframes'], 10)
        self.assertEqual(result['older_special_tokens'], 30)
        self.assertEqual(result['bytes'], sum(t.numel()*2 for t in tensors.values() if hasattr(t, 'numel')))

    def test_worker_error_retains_memory_evidence(self):
        payload = {'status': 'error', 'error': 'allocation failed', 'memory': {'device_free_bytes': 5}}
        exc = WorkerError('sam', payload)
        self.assertEqual(exc.message['memory']['device_free_bytes'], 5)
        self.assertEqual(exc.role, 'sam')

    def test_unused_allocator_blocks_released_only_under_selected_policy(self):
        releases = []
        cuda = SimpleNamespace(mem_get_info=lambda: (40, 100), empty_cache=lambda: releases.append(True))
        torch = SimpleNamespace(cuda=cuda)
        self.assertFalse(release_unused_cuda(torch, 'retain', 80))
        self.assertFalse(release_unused_cuda(torch, 'budget', 30))
        self.assertTrue(release_unused_cuda(torch, 'budget', 50))
        self.assertTrue(release_unused_cuda(torch, 'release', 30))
        self.assertEqual(len(releases), 2)

    def test_similarity_alignment_recovers_global_scale_rotation_translation(self):
        rng = np.random.default_rng(42)
        source = rng.normal(size=(100, 3))
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        target = 1.7 * source @ rotation.T + [2., 3., -1.]
        scale, estimated_rotation, translation = fit_similarity(source, target)
        np.testing.assert_allclose(scale * source @ estimated_rotation.T + translation, target, atol=1e-12)
        with self.assertRaisesRegex(ValueError, 'degenerate'):
            fit_similarity(np.zeros((10, 3)), np.ones((10, 3)))

    def test_sam_cached_text_union_uses_current_image_and_bounded_cache(self):
        import torch
        calls = []
        def forward_text(prompts, device):
            calls.extend(prompts)
            return {'language_features': prompts[0]}
        class Processor:
            device = 'cpu'
            model = SimpleNamespace(backbone=SimpleNamespace(forward_text=forward_text),
                                    _get_dummy_prompt=lambda: object())
            def reset_all_prompts(self, state):
                state['backbone_out'].pop('language_features', None)
            def _forward_grounding(self, state):
                if state['image_id'] == 1:
                    mask = torch.zeros((2, 1, 14, 14), dtype=torch.bool)
                    mask[0, 0, 1, 2] = True
                    mask[1, 0, 3, 4] = True
                else:
                    mask = torch.zeros((0, 1, 14, 14), dtype=torch.bool)
                return {'masks': mask}
        adapter = object.__new__(CachedSamSegmenter)
        adapter.torch, adapter.device_type = torch, 'cpu'
        adapter.processor, adapter.cache_size = Processor(), 1
        adapter.text_cache = OrderedDict()
        adapter.cache_hits = adapter.cache_misses = 0
        adapter.state = {'image_id': 1, 'backbone_out': {}}
        first = adapter.union(['log'], (14, 14))
        self.assertEqual(int(first.sum()), 2)
        adapter.state = {'image_id': 2, 'backbone_out': {}}
        self.assertFalse(adapter.union(['log'], (14, 14)).any())
        self.assertEqual(calls, ['log'])
        self.assertFalse(adapter.union(['bag', 'log'], (14, 14)).any())
        self.assertEqual(calls, ['log', 'bag', 'log'])
        self.assertEqual(len(adapter.text_cache), 1)
        self.assertFalse(adapter.union([], (14, 14)).any())


if __name__ == '__main__':
    unittest.main()
