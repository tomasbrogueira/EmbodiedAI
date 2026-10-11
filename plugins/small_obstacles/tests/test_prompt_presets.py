import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from plugins.small_obstacles.prompt_presets import (
    DEFAULT_INSTRUCTION, DEFAULT_NAME, PromptPresets, instruction_settings)
from plugins.small_obstacles.run_streaming import parse_args
from plugins.small_obstacles.model_session import ModelSession
from plugins.small_obstacles.tests.test_session_pose import FakeWorker


class PromptPresetTests(unittest.TestCase):
    def test_presets_persist_and_overwrite_without_changing_run_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            store = PromptPresets(Path(directory)/'presets')
            text = 'Describe water.\nReturn {"prompts": ["pond"]}.\n'
            store.save('Water', text)
            settings = instruction_settings(SimpleNamespace(vlm_instruction=text, vlm_prompt_name='Water'))
            run_config = Path(directory)/'config.json'
            run_config.write_text(json.dumps(settings))
            fresh_store = PromptPresets(store.directory)
            self.assertEqual(fresh_store.list()['Water']['instruction'], text)
            fresh_store.save('Water', 'Changed instruction')
            fresh_store.save('Loose logs', 'Find logs')
            self.assertEqual(set(store.list()), {'Water', 'Loose logs'})
            self.assertEqual(store.list()['Water']['instruction'], 'Changed instruction')
            self.assertEqual(json.loads(run_config.read_text())['instruction'], text)
            self.assertEqual(settings['vlm_instruction_sha256'], hashlib.sha256(text.encode()).hexdigest())

    def test_invalid_edits_and_names_cannot_destroy_default_or_escape_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            store = PromptPresets(Path(directory)/'presets')
            for name, instruction in [('', 'prompt'), (DEFAULT_NAME, 'prompt'), ('test', ' \n')]:
                with self.assertRaises(ValueError):
                    store.save(name, instruction)
            store.save('../outside/ponds 🌊', 'Find visible ponds')
            self.assertEqual(len(list(store.directory.glob('*.json'))), 1)
            self.assertFalse((Path(directory)/'outside').exists())
            self.assertIn('../outside/ponds 🌊', store.list())
            self.assertEqual(instruction_settings(SimpleNamespace())['instruction'], DEFAULT_INSTRUCTION)
            with self.assertRaises(ValueError):
                instruction_settings(SimpleNamespace(vlm_instruction=''))

    def test_custom_instruction_file_is_read_verbatim_and_empty_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video, checkpoint, prompt = [root/name for name in ['video.mp4', 'checkpoint.pt', 'prompt.txt']]
            video.touch(); checkpoint.touch()
            text = 'Find logs.\nReturn JSON with "prompts".\n'
            prompt.write_text(text)
            argv = ['--video-path', str(video), '--output-dir', str(root/'run'),
                    '--lingbot-checkpoint', str(checkpoint), '--vlm-instruction-file', str(prompt),
                    '--vlm-prompt-name', 'Logs']
            settings = instruction_settings(parse_args(argv))
            self.assertEqual(settings['instruction'], text)
            self.assertEqual(settings['vlm_prompt_name'], 'Logs')
            prompt.write_text('')
            with self.assertRaises(SystemExit):
                parse_args(argv)

    def test_new_prompt_resets_worker_config_without_reloading_models(self):
        with tempfile.TemporaryDirectory() as directory:
            FakeWorker.made = []
            path = Path(directory)/'config.json'
            config = {role+'_python': 'python' for role in ['lingbot', 'vlm', 'sam']}
            config['instruction'] = 'Find logs'
            path.write_text(json.dumps(config))
            session = ModelSession(FakeWorker)
            cold = session.begin(path, path.parent, lambda role: None, lambda *args: None)
            session.finish()
            config['instruction'] = 'Find ponds'
            path.write_text(json.dumps(config))
            ready = []
            warm = session.begin(path, path.parent, lambda role: None,
                                 lambda role, worker, result: ready.append(result))
            self.assertEqual(cold, warm)
            self.assertEqual(len(FakeWorker.made), 3)
            self.assertTrue(all(result['reused'] for result in ready))
            vlm_reset = warm[1].operations[-1]
            self.assertEqual(vlm_reset['op'], 'reset')
            self.assertEqual(json.loads(Path(vlm_reset['config']).read_text())['instruction'], 'Find ponds')
            session.finish(); session.close()

    def test_manual_sessions_load_only_geometry_and_sam_and_reuse_across_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            FakeWorker.made = []
            path = Path(directory)/'config.json'
            config = {role+'_python': 'python' for role in ['lingbot', 'vlm', 'sam']}
            config['proposal_mode'] = 'manual'
            path.write_text(json.dumps(config))
            session = ModelSession(FakeWorker)
            manual = session.begin(path, path.parent, lambda role: None, lambda *args: None)
            self.assertEqual([w.role for w in manual], ['lingbot', 'sam'])
            self.assertTrue(session.loaded)
            session.finish()
            config['proposal_mode'] = 'vlm'
            path.write_text(json.dumps(config))
            automatic = session.begin(path, path.parent, lambda role: None, lambda *args: None)
            self.assertIs(automatic[0], manual[0])
            self.assertIs(automatic[2], manual[1])
            self.assertEqual(len(FakeWorker.made), 3)
            session.finish()
            config['proposal_mode'] = 'manual'
            path.write_text(json.dumps(config))
            again = session.begin(path, path.parent, lambda role: None, lambda *args: None)
            self.assertEqual(again, manual)
            self.assertEqual(len(FakeWorker.made), 3)
            session.finish(); session.close()

    def test_manual_queries_are_concrete_deduplicated_and_not_silently_truncated(self):
        from plugins.small_obstacles.prompt_presets import parse_sam_prompts
        self.assertEqual(parse_sam_prompts('Log, plastic bottle\nLOG\npond'), ['log', 'plastic bottle', 'pond'])
        for text in ['', ',\n', 'a phrase with more than eight separate object words included',
                     '\n'.join(f'object {i}' for i in range(9))]:
            with self.assertRaises(ValueError):
                parse_sam_prompts(text)


if __name__ == '__main__':
    unittest.main()
