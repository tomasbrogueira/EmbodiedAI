"""Local-only recovery admission checks; no server, models, or stage processes."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path('C:/University/KTH/EmbodiedAI')
CANDIDATE = ROOT / '.codex-local/kth-server/navigation_all18_20261008_deadline.py'
SUPERVISOR = ROOT / '.codex-local/kth-server/pipeline2_cpu_stage_20261008_deadline.py'
FROZEN = ROOT / '.codex-local/handoffs/navigation_all18_source_20261008.zip'
FROZEN_SHA = '9b065ac58b607d4e985aec02acb6f0cbf47df9289de3765be99627b271fec560'
spec = importlib.util.spec_from_file_location('deadline_recovery_candidate', CANDIDATE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='deadline_recovery_', dir=ROOT / '.codex-local/handoffs')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.batch = self.root / 'outputs/batch'
        self.support = self.root / '.viewer_support/previous'
        self.batch.mkdir(parents=True)
        self.old_manifest, self.old_payload = m.bundle_contents(FROZEN, FROZEN_SHA)
        m.install_bundle(FROZEN, FROZEN_SHA, self.support)
        self.new_bundle = self.root / 'recovery.zip'
        self.bundle_info = m.create_recovery_bundle(FROZEN, FROZEN_SHA, CANDIDATE, SUPERVISOR, self.new_bundle)
        self.new_manifest, self.new_payload = m.bundle_contents(self.new_bundle, self.bundle_info['sha256'])

    def fixture(self):
        names = ['indoor_IMG_' + str(5506 + i) for i in range(18)]
        published = self.root / 'outputs/evaluation_video_pairs'
        published.mkdir(parents=True)
        (published / 'prior.mp4').write_bytes(b'preserved prior published video')
        clips = []
        scripts = {'derive': 'derive_pose_corrected_run.py', 'reference': 'pipeline_research_plan.py',
                   'replay': 'build_navigation_replay.py', 'navigation': 'export_navigation_replay_demo.py',
                   'voxels': 'export_navigation_voxel_replay.py'}
        for i, name in enumerate(names):
            original = self.root / m.OLD_RELATIVE / name
            sequence = original / 'sequence/sequence.json'
            video = self.root / 'videos' / (name + '.mp4')
            video.parent.mkdir(exist_ok=True)
            video.write_bytes((name + ' original encoded video').encode())
            m.write(sequence, {'input_provenance': {'path': str(video), 'encoded_video_sha256': m.sha(video)}})
            inputs = [original / 'ground_surface_run', original / 'geometry_only_run/geometry/cache',
                      sequence.parent, video, self.root / m.CAL_RELATIVE / name / 'input_assumptions.json']
            for path in inputs:
                if path.exists():
                    continue
                if path.suffix == '.json':
                    m.write(path, {'height': 1.5})
                else:
                    path.mkdir(parents=True)
                    (path / 'saved.npz').write_bytes(b'original saved cache')
            attempt = self.batch / 'clips' / name / 'attempt_01'
            attempt.mkdir(parents=True)
            failed = i == 17
            outputs, media = {}, {}
            for stage_name, script in scripts.items():
                folder = 'derived' if stage_name == 'derive' else stage_name
                artifact = attempt / folder
                artifact.mkdir()
                filename = 'result.mp4' if stage_name in ('navigation', 'voxels') else 'result.json'
                (artifact / filename).write_bytes(b'x' * (2000 if filename.endswith('.mp4') else 20))
                command = [str(self.root / '.venv-pipelines/bin/python'), str(self.support / 'src' / script),
                           '--output', str(artifact)]
                if stage_name == 'replay':
                    command = [str(self.root / '.venv-pipelines/bin/python'), str(self.support / 'src/build_navigation_replay.py'),
                               '--geometry', str(original / 'geometry_only_run/geometry/cache/geometry.npz'),
                               '--reference-plan', str(attempt / 'reference/research_plan.json'),
                               '--sequence', str(sequence),
                               '--geometry-manifest', str(attempt / 'derived/derived_run/geometry/manifest.json'),
                               '--output', str(artifact), '--horizon-m', '2']
                spec_path = attempt / 'stages' / (stage_name + '_command.json')
                m.write(spec_path, {'command': command, 'script_sha256': self.old_manifest['files']['src/' + script],
                                   'execution_lease_owner': 'controller_and_supervisor_inherited_descriptor',
                                   'root': str(self.root), 'support': str(self.support), 'batch': str(self.batch)})
                helper = [str(self.root / '.venv-pipelines/bin/python'), str(self.support / 'tools/navigation_all18_20261008.py'),
                          '--worker-stage', str(spec_path)]
                report_path = attempt / 'stages' / (stage_name + '_supervision.json')
                report = {'status': 'failed' if failed and stage_name == 'replay' else 'complete',
                          'max_runtime_seconds': 900, 'stop_reason': 'whole_cpu_stage_deadline' if failed and stage_name == 'replay' else None,
                          'helper_reaped': True, 'helper_group_gone': True, 'owned_descendants_gone': True,
                          'wrapper_sha256': self.old_manifest['files']['tools/pipeline2_cpu_stage.py'],
                          'helper_command': helper, 'actual_helper_command': helper + ['--parent-pipe-fd', '5', '--controller-lock-fd', '3'],
                          'native_child': True, 'controller_lock_held_until_child_exit': True,
                          'execution_lock_owner': 'native_child',
                          'controller_lock_path': str(self.root / 'outputs/.pipeline2_native_recordings.lock'),
                          'execution_lock_path': str(self.root / 'outputs/.pipeline2_video_execution.lock')}
                m.write(report_path, report)
                record = {'stage': stage_name, 'status': report['status'], 'command': command,
                          'output_inventory': m.file_inventory(artifact), 'supervision_sha256': m.sha(report_path)}
                m.write(attempt / 'stages' / (stage_name + '.json'), record)
                outputs[str(artifact.relative_to(self.batch))] = m.file_inventory(artifact)
                if stage_name in ('navigation', 'voxels'):
                    media[stage_name] = str((artifact / filename).relative_to(self.batch))
                if failed and stage_name == 'replay':
                    break
            clip = {'clip': name, 'status': 'failed' if failed else 'complete', 'attempt': str(attempt),
                    'source_input_inventory': {str(p): m.file_inventory(p) for p in inputs}}
            if not failed:
                clip.update(output_inventory=outputs, media=media)
            clips.append(clip)
        archive = self.batch / 'videos_all18.zip'
        archive.write_bytes(b'previous available-media delivery archive')
        receipt = {'schema_version': 1, 'status': 'complete_with_failures', 'models_executed': False,
                   'gpu_work': False, 'robot_commands_executed': False, 'published_pairs_unchanged': True,
                   'published_pairs_before': m.file_inventory(published), 'clips': clips,
                   'identity': {'source_state_sha256': m.STATE_SHA, 'bundle_sha256': FROZEN_SHA,
                                'support': str(self.support), 'batch': str(self.batch), 'selected_clips': names,
                                'orchestrator_sha256': self.old_manifest['files']['tools/navigation_all18_20261008.py']},
                   'delivery_archive': {'path': archive.name, 'sha256': m.sha(archive), 'bytes': archive.stat().st_size}}
        m.write(self.batch / 'receipt.json', receipt)
        return receipt, names

    def test_visible_supervisor_diff_and_frozen_pipeline_closure(self):
        self.assertEqual(m.execution_tool_delta(self.old_manifest, self.old_payload, self.new_manifest, self.new_payload), sorted(m.EXECUTION_TOOLS))
        self.assertEqual(self.new_manifest['files']['tools/pipeline2_cpu_stage.py'], m.RECOVERY_SUPERVISOR_SHA)
        for name in self.old_manifest['files']:
            if name not in m.EXECUTION_TOOLS:
                self.assertEqual(self.old_payload[name], self.new_payload[name])
        self.assertEqual(m.verify_support(self.support, self.old_manifest, self.old_payload), m.file_inventory(self.support))

    def test_pipeline_change_and_additional_supervisor_change_rejected(self):
        for name in ('src/build_navigation_replay.py', 'tools/pipeline2_cpu_stage.py'):
            payload, manifest = dict(self.new_payload), deepcopy(self.new_manifest)
            payload[name] += b'\n# unrelated change\n'
            manifest['files'][name] = hashlib.sha256(payload[name]).hexdigest()
            with self.assertRaises(ValueError):
                m.execution_tool_delta(self.old_manifest, self.old_payload, manifest, payload)

    def test_bundle_metadata_inventory_and_fresh_destination_required(self):
        manifest = deepcopy(self.new_manifest)
        manifest['models_executed'] = True
        with self.assertRaises(ValueError):
            m.execution_tool_delta(self.old_manifest, self.old_payload, manifest, self.new_payload)
        with self.assertRaises(ValueError):
            m.create_recovery_bundle(FROZEN, FROZEN_SHA, CANDIDATE, SUPERVISOR, self.new_bundle)
        (self.support / 'unbound.py').write_bytes(b'x')
        with self.assertRaises(ValueError):
            m.verify_support(self.support, self.old_manifest, self.old_payload)

    def test_timeout_policy_bounded_and_other_stages_stay900(self):
        for value in ('900', '2400', '3600'):
            self.assertEqual(m.replay_timeout(value), int(value))
        for value in ('899', '3601', '2400.0', '-1', 'NaN'):
            with self.assertRaises(Exception):
                m.replay_timeout(value)
        with self.assertRaises(ValueError):
            m.stage('reference', [], None, None, None, None, None, None, None, timeout=2400)
        with self.assertRaises(ValueError):
            m.stage('replay', [], None, None, None, None, None, None, None, timeout=3601)

    def test_new_stage_receipt_records_exact_deadline_and_supervisor(self):
        support = self.root / '.viewer_support/recovery'
        m.install_bundle(self.new_bundle, self.bundle_info['sha256'], support)
        attempt = self.batch / 'clips/indoor_IMG_5506/attempt_02'
        attempt.mkdir(parents=True)
        with patch.object(m, 'resource_snapshot', return_value={'admitted': False, 'reasons': ['fixture resource unavailable']}):
            with self.assertRaises(m.ResourceUnavailable):
                m.stage('replay', ['python', 'unused'], attempt / 'replay', attempt, {}, 3,
                        support, self.root, self.batch, timeout=2400)
        saved = m.read(attempt / 'stages/replay.json')
        self.assertEqual(saved['max_runtime_seconds'], 2400)
        self.assertEqual(saved['supervisor_sha256'], m.RECOVERY_SUPERVISOR_SHA)
        self.assertEqual(saved['status'], 'waiting_resources')
        environment = m.inherited_environment(self.root, support, self.batch)
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            self.assertEqual(environment[key], '4')
        self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], '')

    def test_real_finished_fixture_and_changed_completed_output(self):
        receipt, names = self.fixture()
        audit = m.migration_audit(receipt, self.root, self.batch, names, self.old_manifest)
        self.assertEqual(len(audit['retained_attempts']), 18)
        self.assertEqual(len(audit['complete_outputs']), 85)
        self.assertEqual(len(audit['deadline_failures']), 1)
        movie = self.batch / receipt['clips'][0]['media']['navigation']
        movie.write_bytes(b'changed complete output')
        with self.assertRaises(ValueError):
            m.migration_audit(receipt, self.root, self.batch, names, self.old_manifest)

    def test_running_batch_missing_source_or_no_concrete_timeout_rejected(self):
        receipt, names = self.fixture()
        running = deepcopy(receipt)
        running['status'] = 'running'
        with self.assertRaises(ValueError):
            m.migration_audit(running, self.root, self.batch, names, self.old_manifest)
        missing = deepcopy(receipt)
        missing['clips'][0]['source_input_inventory'].pop(next(iter(missing['clips'][0]['source_input_inventory'])))
        with self.assertRaises(ValueError):
            m.migration_audit(missing, self.root, self.batch, names, self.old_manifest)
        path = Path(receipt['clips'][-1]['attempt']) / 'stages/replay_supervision.json'
        report = m.read(path)
        report['stop_reason'] = 'controller_pipe_eof'
        m.write(path, report)
        with self.assertRaisesRegex(ValueError, 'No concrete'):
            m.migration_audit(receipt, self.root, self.batch, names, self.old_manifest)

    def test_unbound_timeout_report_cannot_qualify_recovery(self):
        receipt, names = self.fixture()
        path = Path(receipt['clips'][-1]['attempt']) / 'stages/replay_supervision.json'
        report = m.read(path)
        for key, value in [('wrapper_sha256', 'different source'),
                           ('helper_command', ['another stage']),
                           ('execution_lock_path', 'another lease')]:
            bad = dict(report)
            bad[key] = value
            m.write(path, bad)
            with self.assertRaises(ValueError):
                m.migration_audit(receipt, self.root, self.batch, names, self.old_manifest)
        m.write(path, report)
        command_path = path.with_name('replay.json')
        bad_record = m.read(command_path)
        bad_record['command'][1] = str(self.support / 'src/pipeline_research_plan.py')
        m.write(command_path, bad_record)
        with self.assertRaises(ValueError):
            m.migration_audit(receipt, self.root, self.batch, names, self.old_manifest)

    def test_migration_preserves_old_complete_attempts_and_history(self):
        receipt, names = self.fixture()
        previous_bytes = (self.batch / 'receipt.json').read_bytes()
        old_complete = deepcopy(receipt['clips'][0])
        new_support = self.root / '.viewer_support/recovery'
        identity = m.recovery_identity(self.root, self.batch, new_support, names, self.bundle_info['sha256'], self.new_manifest, 2400)
        args = type('Args', (), {'resume': True, 'retry_failed': True, 'previous_bundle': FROZEN,
                               'migrate_execution_tools': FROZEN_SHA, 'bundle': self.new_bundle,
                               'bundle_sha': self.bundle_info['sha256']})()
        migrated, sources = m.migrate_execution_tools(args, self.root, self.batch, new_support, names, identity, self.new_manifest, self.new_payload)
        self.assertEqual(migrated['clips'][0], old_complete)
        self.assertEqual(sources, self.new_manifest)
        self.assertEqual(m.file_inventory(self.support), m.verify_support(self.support, self.old_manifest, self.old_payload))
        event_path = self.batch / migrated['migration_history'][0]['path']
        self.assertEqual((event_path.parent / 'receipt.before.json').read_bytes(), previous_bytes)
        self.assertEqual(migrated['identity']['execution_policy']['replay_max_runtime_seconds'], 2400)
        m.verify_migration_history(migrated, self.root, self.batch)
        with self.assertRaises(FileExistsError):
            m.immutable_json(event_path, {'changed': True})
        (Path(old_complete['attempt']) / 'derived/result.json').write_bytes(b'tampered historical partial')
        with self.assertRaises(ValueError):
            m.verify_migration_history(migrated, self.root, self.batch)

    def test_rejected_migration_preserves_previous_receipt(self):
        receipt, names = self.fixture()
        previous_bytes = (self.batch / 'receipt.json').read_bytes()
        new_support = self.root / '.viewer_support/recovery'
        identity = m.recovery_identity(self.root, self.batch, new_support, names, self.bundle_info['sha256'], self.new_manifest, 2400)
        args = type('Args', (), {'resume': True, 'retry_failed': True, 'previous_bundle': FROZEN,
                               'migrate_execution_tools': 'bad previous hash', 'bundle': self.new_bundle,
                               'bundle_sha': self.bundle_info['sha256']})()
        with self.assertRaises(ValueError):
            m.migrate_execution_tools(args, self.root, self.batch, new_support, names, identity, self.new_manifest, self.new_payload)
        self.assertEqual((self.batch / 'receipt.json').read_bytes(), previous_bytes)
        self.assertFalse(new_support.exists())

    def test_original_inputs_published_pairs_and_missing_stage_rejected(self):
        receipt, names = self.fixture()
        targets = [self.root / m.OLD_RELATIVE / names[0] / 'ground_surface_run/saved.npz',
                   self.root / 'outputs/evaluation_video_pairs/prior.mp4',
                   Path(receipt['clips'][0]['attempt']) / 'stages/derive.json']
        for path in targets:
            data = path.read_bytes()
            path.unlink()
            try:
                with self.assertRaises((ValueError, FileNotFoundError)):
                    m.migration_audit(receipt, self.root, self.batch, names, self.old_manifest)
            finally:
                path.write_bytes(data)

    def test_active_execution_lease_rejected_before_install_or_receipt_change(self):
        receipt, names = self.fixture()
        m.write(self.root / m.OLD_RELATIVE / 'pilot_status.json', {
            'completed_recordings': [{'relative': name + '/saved', 'sampled_frames': 35 if i == 0 else 29}
                                     for i, name in enumerate(names)]})
        previous = (self.batch / 'receipt.json').read_bytes()
        new_support = self.root / '.viewer_support/recovery'
        lock_calls = []
        def flock(fd, flags):
            lock_calls.append((fd, flags))
            if len(lock_calls) == 3:
                raise BlockingIOError('another CPU stage still owns global execution')
        facade = SimpleNamespace(LOCK_EX=2, LOCK_NB=4, flock=flock)
        class PosixAdmission:
            name = 'posix'
            def __getattr__(self, name):
                return getattr(os, name)
        original_is_file = Path.is_file
        def is_file(path):
            return True if path.as_posix() == '/proc/self/stat' else original_is_file(path)
        original_sha = m.sha
        def frozen_state(path):
            return m.STATE_SHA if Path(path).name == 'pilot_status.json' else original_sha(path)
        signals = SimpleNamespace(SIGINT=2, SIGTERM=15, SIGHUP=1, signal=lambda *_: None)
        with patch.dict(sys.modules, {'fcntl': facade}), patch.object(m, 'os', PosixAdmission()), \
                patch.object(Path, 'is_file', is_file), patch.object(m, 'sha', frozen_state), \
                patch.object(m, 'signal', signals), patch.object(m, 'resource_snapshot', return_value={
                    'admitted': True, 'cpu_affinity': [0, 1, 2, 3]}):
            with self.assertRaises(BlockingIOError):
                m.main(['--root', str(self.root), '--bundle', str(self.new_bundle), '--bundle-sha', self.bundle_info['sha256'],
                        '--output', str(self.batch), '--support', str(new_support), '--resume', '--retry-failed',
                        '--replay-timeout', '2400', '--migrate-execution-tools', FROZEN_SHA,
                        '--previous-bundle', str(FROZEN)])
        self.assertEqual(len(lock_calls), 3)
        self.assertEqual((self.batch / 'receipt.json').read_bytes(), previous)
        self.assertFalse(new_support.exists())


if __name__ == '__main__':
    unittest.main()
