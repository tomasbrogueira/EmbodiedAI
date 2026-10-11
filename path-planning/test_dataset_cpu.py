"""Coordinator scheduling with stub subprocesses and real bounded futures."""
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

try:
    import fcntl
except ModuleNotFoundError:
    sys.modules['fcntl'] = types.SimpleNamespace(LOCK_EX=1, LOCK_SH=2, LOCK_NB=4, flock=lambda *args: None)

import dataset_cpu as cpu


class SchedulingTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parent
        self.temporary = tempfile.TemporaryDirectory(prefix='cpu-schedule-test-', dir=root)
        self.root = Path(self.temporary.name).resolve()
        assert self.root.is_relative_to(root)
        self.addCleanup(self.temporary.cleanup)
        self.entries, self.events, self.hooks = [], [], {}
        self.failures, self.missing = set(), set()
        self.missing_calibration = set()
        self.stop = threading.Event()
        source = self.root / 'repo/path-planning/cost_map.py'
        source.parent.mkdir(parents=True)
        source.write_text('cost source v4')
        self.cost_source = cpu.cost_source_hash(self.root / 'repo')
        (self.root / 'dataset_prepare.py').write_text('prepare with frozen camera calibration')
        (self.root / 'dataset_path.py').write_text('current camera association')
        with patch.object(cpu, 'ROOT', self.root), patch.object(cpu, 'SOURCE_ROOT', self.root):
            self.prepare_adapter = cpu.prepare_adapter_hash()
            self.calibration_association = cpu.calibration_association_hash()

    def add(self, name, frames=100, reused=False):
        out = self.root / 'runs' / name
        out.mkdir(parents=True)
        (out / 'gpu.json').write_text('{}')
        if reused:
            (out / 'report.json').write_text(json.dumps({'reused_verified_output': True, 'status': 'pending'}))
        entry = dict(id=name, output=str(out), frame_count=frames, input_sha256=name)
        self.entries.append(entry)
        return out

    def subprocess(self, args, **kwargs):
        stage, out = Path(args[3]).stem, Path(args[-1])
        name = next(e['id'] for e in self.entries if Path(e['output']) == out)
        self.events.append((stage, name, 'start'))
        hook = self.hooks.get((stage, name))
        if hook:
            hook()
        if (stage, name) in self.failures:
            raise subprocess.CalledProcessError(1, args)
        if stage == 'dataset_prepare':
            state = ('missing_reference' if name in self.missing else
                     'missing_clearance_calibration' if name in self.missing_calibration else 'pending')
        else:
            self.assertTrue((out / 'report.json').exists())
            state = 'ok'
        previous = json.loads((out / 'report.json').read_text()) if (out / 'report.json').exists() else {}
        (out / 'report.json').write_text(json.dumps(previous | {'status': state, 'cost_source_sha256': self.cost_source,
            'prepare_adapter_sha256': self.prepare_adapter, 'calibration_association_sha256': self.calibration_association}))
        self.events.append((stage, name, 'end'))
        return types.SimpleNamespace(returncode=0)

    def work(self):
        (self.root / 'manifest.json').write_text(json.dumps({'repo': str(self.root / 'repo'), 'videos': self.entries}))
        (self.root / 'gpu-finished.json').write_text('{}')
        with patch.object(cpu, 'ROOT', self.root), patch.object(cpu, 'SOURCE_ROOT', self.root), patch.object(cpu, 'allocation', return_value=nullcontext()), \
             patch.object(cpu.subprocess, 'run', side_effect=self.subprocess), patch.object(cpu.traceback, 'print_exc'):
            cpu.work(self.stop.is_set)

    def overlap(self):
        ready, release = threading.Event(), threading.Event()
        def prepare_b():
            ready.set()
            self.assertTrue(release.wait(3), 'Current path did not overlap preparation')
        def path_a():
            self.assertTrue(ready.wait(3), 'Look-ahead preparation never started')
            release.set()
        self.hooks[('dataset_prepare', 'B')] = prepare_b
        self.hooks[('dataset_path', 'A')] = path_a
        return path_a

    def test_overlap_order_and_no_duplicate_preparation(self):
        a, b = self.add('A'), self.add('B', 200)
        self.overlap()
        self.work()
        events = self.events
        self.assertLess(events.index(('dataset_prepare', 'B', 'start')), events.index(('dataset_path', 'A', 'end')))
        self.assertLess(events.index(('dataset_prepare', 'B', 'end')), events.index(('dataset_path', 'B', 'start')))
        for out in (a, b):
            self.assertTrue((out / 'cpu.json').exists())
            self.assertEqual(events.count(('dataset_prepare', out.name, 'start')), 1)
        self.assertTrue((self.root / 'cpu-finished.json').exists())

    def test_preparation_queue_stays_busy_and_later_results_are_consumed_once(self):
        outputs = [self.add(name, 100 * (i + 1)) for i, name in enumerate(('A', 'B', 'C', 'D'))]
        last_started = threading.Event()
        self.hooks[('dataset_prepare', 'D')] = last_started.set
        self.hooks[('dataset_path', 'A')] = lambda: self.assertTrue(last_started.wait(3), 'Preparation stopped after the first video')
        self.work()
        active = {'dataset_prepare': 0, 'dataset_path': 0}
        maxima = active.copy()
        overlapped = False
        for stage, _, event in self.events:
            active[stage] += 1 if event == 'start' else -1
            maxima[stage] = max(maxima[stage], active[stage])
            overlapped |= all(active.values())
        self.assertEqual(maxima, {'dataset_prepare': 1, 'dataset_path': 1})
        self.assertTrue(overlapped)
        for out in outputs:
            for stage in active:
                self.assertEqual(self.events.count((stage, out.name, 'start')), 1)
            self.assertTrue((out / 'cpu.json').exists())
            self.assertLess(self.events.index(('dataset_prepare', out.name, 'end')),
                            self.events.index(('dataset_path', out.name, 'start')))

    def test_failed_prefetch_is_not_retried_or_given_a_receipt(self):
        self.add('A')
        b, c = self.add('B', 200), self.add('C', 300)
        self.failures.add(('dataset_prepare', 'B'))
        self.work()
        self.assertEqual(self.events.count(('dataset_prepare', 'B', 'start')), 1)
        self.assertNotIn(('dataset_path', 'B', 'start'), self.events)
        self.assertFalse((b / 'cpu.json').exists())
        self.assertEqual(json.loads((b / 'status.json').read_text())['status'], 'error')
        self.assertTrue((c / 'cpu.json').exists())

    def test_path_failure_does_not_discard_successful_prefetch(self):
        a, b = self.add('A'), self.add('B', 200)
        self.failures.add(('dataset_path', 'A'))
        self.work()
        self.assertFalse((a / 'cpu.json').exists())
        self.assertTrue((b / 'cpu.json').exists())
        self.assertEqual(self.events.count(('dataset_prepare', 'B', 'start')), 1)

    def test_long_current_and_long_next_are_serial(self):
        self.add('A', 100)
        self.add('B', 950)
        self.add('C', 1000)
        self.work()
        self.assertLess(self.events.index(('dataset_path', 'A', 'end')),
                        self.events.index(('dataset_prepare', 'B', 'start')))
        self.assertLess(self.events.index(('dataset_path', 'B', 'end')),
                        self.events.index(('dataset_prepare', 'C', 'start')))

    def test_reused_current_and_next_are_serial_but_costs_are_reprepared(self):
        self.add('A', 100, reused=True)
        self.add('B', 200)
        self.add('C', 300, reused=True)
        self.work()
        self.assertEqual(self.events.count(('dataset_prepare', 'A', 'start')), 1)
        self.assertEqual(self.events.count(('dataset_prepare', 'C', 'start')), 1)
        self.assertLess(self.events.index(('dataset_path', 'A', 'end')), self.events.index(('dataset_prepare', 'B', 'start')))
        self.assertLess(self.events.index(('dataset_path', 'B', 'end')), self.events.index(('dataset_path', 'C', 'start')))

    def test_prefetched_missing_reference_completes_without_path(self):
        self.add('A')
        b = self.add('B', 200)
        self.missing.add('B')
        self.work()
        self.assertNotIn(('dataset_path', 'B', 'start'), self.events)
        self.assertEqual(json.loads((b / 'cpu.json').read_text())['status'], 'missing_reference')

    def test_graceful_stop_finishes_current_and_prefetch_only(self):
        a, b = self.add('A'), self.add('B', 200)
        c, d = self.add('C', 300), self.add('D', 400)
        ready, release = threading.Event(), threading.Event()
        def prepare_b():
            ready.set()
            self.assertTrue(release.wait(3))
        def stop_after_path():
            self.assertTrue(ready.wait(3))
            self.stop.set()
            release.set()
        self.hooks[('dataset_prepare', 'B')] = prepare_b
        self.hooks[('dataset_path', 'A')] = stop_after_path
        self.work()
        self.assertTrue((a / 'cpu.json').exists())
        self.assertIn(('dataset_prepare', 'B', 'end'), self.events)
        self.assertNotIn(('dataset_path', 'B', 'start'), self.events)
        self.assertFalse((b / 'cpu.json').exists())
        self.assertFalse((self.root / 'cpu-finished.json').exists())
        self.assertEqual(json.loads((b / 'status.json').read_text())['status'], 'pending')
        for out in (c, d):
            self.assertFalse(any(event[1] == out.name for event in self.events))
            self.assertFalse((out / 'cpu.json').exists())
            self.assertFalse((out / 'cpu-error.json').exists())

    def test_missing_camera_calibration_completes_without_path(self):
        out = self.add('A')
        self.missing_calibration.add('A')
        self.work()
        self.assertNotIn(('dataset_path', 'A', 'start'), self.events)
        receipt = json.loads((out / 'cpu.json').read_text())
        self.assertEqual(receipt['status'], 'missing_clearance_calibration')
        self.assertEqual(receipt['prepare_adapter_sha256'], self.prepare_adapter)

    def test_matching_receipt_is_preserved(self):
        a = self.add('A')
        self.add('B', 200)
        receipt = {'policy': cpu.POLICY, 'input_sha256': 'A', 'status': 'ok', 'finished': 1,
                   'cost_source_sha256': self.cost_source, 'prepare_adapter_sha256': self.prepare_adapter,
                   'calibration_association_sha256': self.calibration_association}
        (a / 'cpu.json').write_text(json.dumps(receipt))
        self.work()
        self.assertFalse(any(event[1] == 'A' for event in self.events))
        self.assertEqual(json.loads((a / 'cpu.json').read_text()), receipt)

    def test_old_cost_source_receipt_is_recomputed(self):
        out = self.add('A')
        (out / 'cpu.json').write_text(json.dumps({'policy': cpu.POLICY, 'input_sha256': 'A',
                                                'cost_source_sha256': 'old source', 'status': 'ok'}))
        self.work()
        self.assertIn(('dataset_prepare', 'A', 'start'), self.events)
        self.assertEqual(json.loads((out / 'cpu.json').read_text())['cost_source_sha256'], self.cost_source)

    def test_old_prepare_adapter_receipt_is_recomputed(self):
        out = self.add('A')
        (out / 'cpu.json').write_text(json.dumps({'policy': cpu.POLICY, 'input_sha256': 'A',
            'cost_source_sha256': self.cost_source, 'prepare_adapter_sha256': 'old radius-only prepare', 'status': 'ok'}))
        self.work()
        self.assertIn(('dataset_prepare', 'A', 'start'), self.events)
        self.assertEqual(json.loads((out / 'cpu.json').read_text())['prepare_adapter_sha256'], self.prepare_adapter)

    def test_old_association_adapter_receipt_is_recomputed(self):
        out = self.add('A')
        (out / 'cpu.json').write_text(json.dumps({'policy': cpu.POLICY, 'input_sha256': 'A',
            'cost_source_sha256': self.cost_source, 'prepare_adapter_sha256': self.prepare_adapter,
            'calibration_association_sha256': 'old association', 'status': 'ok'}))
        self.work()
        self.assertIn(('dataset_prepare', 'A', 'start'), self.events)
        self.assertEqual(json.loads((out / 'cpu.json').read_text())['calibration_association_sha256'], self.calibration_association)


if __name__ == '__main__':
    unittest.main()
