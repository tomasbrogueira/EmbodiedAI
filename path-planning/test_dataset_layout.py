"""Batch source paths and reconstruction identity stay independent of run data."""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

try:
    import fcntl
except ModuleNotFoundError:
    sys.modules['fcntl'] = types.SimpleNamespace(LOCK_EX=1, LOCK_SH=2, LOCK_NB=4, flock=lambda *args: None)

import dataset_cpu as cpu
import dataset_gpu as gpu
import dataset_inventory as inventory


class BatchLayoutTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo, self.batch, self.source = (self.root / name for name in ('repo', 'batch', 'source'))
        for path in (self.repo, self.batch, self.source):
            path.mkdir()
        (self.repo / 'path-planning').mkdir()
        (self.repo / 'path-planning/reconstruct.py').write_text('current uniform anchor reconstruction')
        (self.source / 'dataset_prepare.py').write_text('prepare source')
        (self.source / 'dataset_path.py').write_text('association source')

    def entry(self):
        out = self.batch / 'video'
        out.mkdir()
        return dict(id='indoor/video', relative_path='indoor/video.mp4', name='video',
                    output=str(out), video=str(self.root / 'video.mp4'), frame_count=2,
                    input_sha256='video-hash', prompt='floor',
                    model_settings=inventory.MODEL | {'reconstruction_source_sha256':
                        inventory.sha256(self.repo / 'path-planning/reconstruct.py')})

    def test_cpu_source_hashes_and_subprocess_do_not_follow_batch_root(self):
        entry = self.entry()
        with patch.object(cpu, 'ROOT', self.batch), patch.object(cpu, 'SOURCE_ROOT', self.source), \
             patch.object(cpu.subprocess, 'run') as run:
            self.assertEqual(cpu.prepare_adapter_hash(), hashlib.sha256(b'prepare source').hexdigest())
            self.assertEqual(cpu.calibration_association_hash(), hashlib.sha256(b'association source').hexdigest())
            cpu.command(self.repo, entry, 'dataset_prepare.py')
        self.assertEqual(run.call_args.args[0][3], str(self.source / 'dataset_prepare.py'))
        self.assertEqual(run.call_args.args[0][-1], entry['output'])
        self.assertTrue((Path(entry['output']) / 'dataset_prepare.log').exists())

    def test_cpu_reads_the_selected_manifest_and_publishes_in_batch(self):
        manifest = self.batch / 'selected.json'
        manifest.write_text(json.dumps({'repo': str(self.repo), 'videos': []}))
        (self.batch / 'gpu-finished.json').write_text('{}')
        with patch.object(cpu, 'ROOT', self.batch), patch.object(cpu, 'SOURCE_ROOT', self.source), \
             patch.object(cpu, 'cost_source_hash', return_value='cost-source'):
            cpu.work(manifest_path=manifest)
        self.assertTrue((self.batch / 'cpu-finished.json').exists())
        self.assertFalse((self.source / 'cpu-finished.json').exists())

    def test_gpu_uses_current_reconstruction_and_source_mask_adapter(self):
        entry = self.entry()
        with patch.object(gpu, 'ROOT', self.batch), patch.object(gpu, 'SOURCE_ROOT', self.source), \
             patch.object(gpu, 'allocation', return_value=nullcontext()), patch.object(gpu, 'command') as command:
            gpu.run(self.repo, entry)
        reconstruction, masks = [call.args[3] for call in command.call_args_list]
        self.assertEqual(reconstruction[3], str(self.repo / 'path-planning/reconstruct.py'))
        for flag, value in (('--mode', 'streaming'), ('--num_scale_frames', '8'),
                            ('--kv_cache_sliding_window', '32'), ('--keyframe_interval', '2')):
            self.assertEqual(reconstruction[reconstruction.index(flag) + 1], value)
        self.assertNotIn('--window_size', reconstruction)
        self.assertEqual(masks[3], str(self.source / 'generate_masks.py'))

    def test_gpu_rejects_changed_reconstruction_before_reusing_or_launching(self):
        entry = self.entry()
        receipt = Path(entry['output']) / 'gpu.json'
        receipt.write_text('{}')
        (self.repo / 'path-planning/reconstruct.py').write_text('changed reconstruction')
        with patch.object(gpu, 'command') as command, patch.object(gpu, 'allocation') as allocation:
            with self.assertRaisesRegex(RuntimeError, 'Reconstruction source/settings changed'):
                gpu.run(self.repo, entry)
        command.assert_not_called()
        allocation.assert_not_called()
        self.assertEqual(receipt.read_text(), '{}')

    def test_inventory_preserves_old_manifest_after_reconstruction_changes(self):
        videos = self.repo / 'data/videos/indoor'
        videos.mkdir(parents=True)
        (videos / 'sample.mp4').write_bytes(b'video')
        with patch.object(inventory, 'command', return_value='model-commit'), \
             patch.object(inventory, 'probe', return_value=dict(duration=1., frame_count=2, timestamps=[0., .5])):
            manifest = inventory.inventory(self.repo, self.batch)
            original = (self.batch / 'manifest.json').read_bytes()
            saved = manifest['videos'][0]['model_settings']
            self.assertEqual(saved['reconstruction_source_sha256'], inventory.sha256(self.repo / 'path-planning/reconstruct.py'))
            self.assertEqual(saved['num_scale_frames'], 8)
            (self.repo / 'path-planning/reconstruct.py').write_text('different anchors')
            with self.assertRaisesRegex(ValueError, 'Cached inputs/settings differ'):
                inventory.inventory(self.repo, self.batch)
        self.assertEqual((self.batch / 'manifest.json').read_bytes(), original)
        differences = json.loads((self.batch / 'inventory-mismatches.json').read_text())
        self.assertIn('model_settings', differences[0]['changed'])


if __name__ == '__main__':
    unittest.main()
