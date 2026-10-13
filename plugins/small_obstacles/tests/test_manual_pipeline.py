import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from plugins.small_obstacles.run_streaming import parse_args, run


class ManualPipelineTests(unittest.TestCase):
    def test_direct_queries_skip_vlm_and_stay_active_after_prompt_ttl(self):
        workers = []
        queries = []

        class Worker:
            def __init__(self, role, python, config, directory):
                if role == 'vlm':
                    raise AssertionError('Manual mode must not load or query Qwen')
                self.role = role
                self.process = type('Process', (), {'pid': len(workers)+1})()
                self.ready = {'load_seconds': 0.}
                self.closed = False
                workers.append(self)

            def call(self, **request):
                if self.role == 'lingbot':
                    with np.load(request['input']) as data:
                        rgb = data['rgb'].copy()
                    n = len(rgb)
                    np.savez(request['output'], images=rgb,
                        depth=np.full((n, 28, 28), 2., np.float32),
                        depth_conf=np.full((n, 28, 28), 3., np.float32),
                        intrinsic=np.tile(np.array([[10,0,14],[0,10,14],[0,0,1]], np.float32),(n,1,1)),
                        extrinsic=np.tile(np.c_[np.eye(3), [0,0,0]], (n,1,1)))
                else:
                    queries.append(list(request['prompts']))
                    mask = np.zeros((28,28), bool)
                    mask[0,:len(request['prompts'])] = True
                    np.savez(request['output'], out_binary_masks=mask)
                return {'elapsed_seconds': .01}

            def close(self):
                self.closed = True

        def frames(args, state):
            state.update(source_fps=1., source_frame_count=3, input_stride=1, dropped_sampled_frames=0)
            for source_index, time in enumerate([0., 10., 20.]):
                yield {'source_index': source_index, 'source_time': time, 'captured': 0.,
                       'rgb': np.full((28,28,3), 100, np.uint8)}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video, checkpoint = root/'video.mp4', root/'checkpoint.pt'
            video.touch(); checkpoint.touch()
            args = parse_args(['--video-path', str(video), '--output-dir', str(root/'run'),
                '--lingbot-checkpoint', str(checkpoint), '--map-step', '1',
                '--sam-prompts', 'log', 'plastic bottle', '--save-frames'])
            with patch('plugins.small_obstacles.run_streaming.Worker', Worker), \
                 patch('plugins.small_obstacles.run_streaming.video_frames', frames):
                run(args, output_stream=io.StringIO())
            state = json.loads((args.output_dir/'status.json').read_text())
            events = [json.loads(line) for line in (args.output_dir/'events.jsonl').read_text().splitlines()]
            self.assertEqual(state['status'], 'complete')
            self.assertEqual(state['vlm_calls'], 0)
            self.assertTrue(state['vlm_skipped'])
            self.assertEqual([w.role for w in workers], ['lingbot', 'sam'])
            self.assertTrue(all(w.closed for w in workers))
            self.assertEqual(queries, [['log', 'plastic bottle']]*3)
            self.assertEqual(state['mapped_frames'], 3)
            self.assertFalse(any(e['type'] == 'vlm' for e in events))
            mapped = [e for e in events if e['type'] == 'frame']
            self.assertEqual([e['mask_pixels'] for e in mapped], [2,2,2])
            self.assertEqual([e['prompts'] for e in mapped], queries)


if __name__ == '__main__':
    unittest.main()
