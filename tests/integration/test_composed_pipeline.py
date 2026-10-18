"""Cross-component semantic, feature, artifact and native certification tests."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np

from orchestrator.artifacts import verify, write_json, publish, code_identity
from orchestrator.projection import build_snapshot, frame_arrays
from orchestrator.config import resolve
from orchestrator.pipeline import run_pipeline
from orchestrator.analysis import steepness, costs, route, load_snapshot
from orchestrator.mask_sources import assemble
from orchestrator.shared import rgb_digest
from path_planner.cost_map import build_cost_map, _settings, CostMap
from path_planner.planner import find_path, find_partial_path
from path_planner.recording_endpoints import select_endpoint
from path_planner.routes import NativeSearch, exact_route
from sam.queries import query_groups, segment_groups
from sam.segment import segment
from vlm.propose import propose, instruction_for
from vlm.prompts import parse_proposal

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / 'path-planning/corridor_astar_feasible.so'


class Segmenter:
    def __init__(self):
        self.images, self.calls = 0, []
    def set_image(self, image):
        self.images += 1
    def masks(self, prompt):
        self.calls.append(prompt)
        mask = np.zeros((28, 28), dtype=bool)
        if prompt in ('path', 'ground'):
            mask[:, 14:] = True
        else:
            mask[14, 20] = True
        return mask[None, None]


class IntegrationTests(unittest.TestCase):
    @unittest.skipUnless(NATIVE.exists(), 'Compile native search first')
    def test_failed_native_sample_still_searches_all_original_rows(self):
        points = self.points.reshape(-1, 3).astype(float)
        cm = build_cost_map(points, None, [0,0,1], dict(radius=.16,
            support_mask=np.ones(len(points),bool), slope_policy='disabled'))
        start, goal = 14*28+3, 14*28+25
        expected = find_path(points,cm,start,goal)
        with patch('path_planner.routes._native_bound', return_value=None) as sampled:
            actual = exact_route(cm,start,goal,np.array([start,goal]),NativeSearch(NATIVE),
                find_path,lambda *a,**k:None,allow_partial=True)
        sampled.assert_called_once()
        self.assertEqual(actual.status, expected.status)
        self.assertAlmostEqual(actual.total_cost, expected.total_cost)
        for first,second in zip(actual.ids[:-1],actual.ids[1:]):
            self.assertTrue(cm._connected(first,second))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.predictions = self.root / 'predictions'
        self.predictions.mkdir()
        grid = np.linspace(-.28, .28, 28)
        x, y = np.meshgrid(grid, grid)
        self.points = np.stack((x, y, np.zeros_like(x)), axis=-1).astype(np.float32)
        self.rgb = np.full((28, 28, 3), 100, dtype=np.uint8)
        self.support = self.root / 'support'
        self.support.mkdir()
        for index in (0, 2, 4):
            prediction = dict(images=self.rgb, depth=np.ones((28,28),np.float32),
                depth_conf=np.full((28,28),3,np.float32), intrinsic=np.eye(3),
                extrinsic=np.c_[np.eye(3), -np.array([index*.01,0,1.])])
            np.savez(self.predictions / f'frame_{index:06d}.npz', **prediction)
            mask = np.zeros((28, 28), bool)
            mask[:, :14] = True
            np.savez(self.support / f'video_floor_frame{index:06d}.npz', out_binary_masks=mask)
        self.unproject = lambda *args: self.points

    def semantic(self, directory, target='non_traversable', extra=('path',), prompts=('log',)):
        proposals = self.root / (directory + '-proposals.json')
        write_json(proposals, dict(prompts=list(prompts), target=target))
        args = SimpleNamespace(lingbot_dir=self.predictions, output_dir=self.root/directory,
            prompts=proposals, target=target, extra_traversability_prompts=list(extra),
            device='cpu', sam_threshold=.5, checkpoint=None, frame_step=1, max_frames=None)
        segment(args, Segmenter())
        return args.output_dir

    def snapshot(self, name='snapshot', semantic=True):
        directory = self.semantic(name + '-semantic') if semantic else None
        build_snapshot(self.predictions, self.root/name, support_dir=self.support,
            semantic_dir=directory, video_name='video', source_fps=30., frame_step=2,
            unproject=self.unproject)
        return self.root/name

    def config(self, snapshot, name, slopes, planner):
        return dict(output=str(self.root/name), inputs={'snapshot': str(snapshot)},
            analysis=dict(steepness='compute' if slopes else 'off', costs='compute' if planner else 'off',
                          reference=[0.,0.,1.], radius=.16, clearance_height=1.),
            planning=dict(enabled=planner, start=14*28+3, goal=14*28+25))

    def test_four_slope_planner_combinations_publish_only_requested_artifacts(self):
        snapshot = self.snapshot()
        for slopes, planner in ((False,False),(True,False),(False,True),(True,True)):
            name = f'run-{slopes}-{planner}'
            state = run_pipeline(self.config(snapshot, name, slopes, planner))
            self.assertEqual(state['status'], 'complete')
            self.assertEqual('steepness' in state['outputs'], slopes)
            self.assertEqual('route' in state['outputs'], planner)
            self.assertEqual((self.root/name/'steepness').exists(), slopes)
            self.assertEqual((self.root/name/'route').exists(), planner)
            if planner:
                report = verify(state['outputs']['route'], 'route')['metadata']
                self.assertEqual(report['status'], 'ok')
                self.assertEqual(report['slope_policy'], 'enforce' if slopes else 'disabled')
                self.assertTrue(report['all_edges_verified'])

    def test_extra_path_expands_cost_support_and_negative_overlap_vetoes(self):
        snapshot = self.snapshot()
        costs(snapshot, self.root/'costs', slope_policy='disabled', reference=[0,0,1], radius=.16, clearance_height=1.)
        values = np.load(self.root/'costs/cost.npy')
        path_row, blocked_row = 14*28+25, 14*28+20
        self.assertTrue(np.isfinite(values[path_row]))
        self.assertTrue(np.isinf(values[blocked_row]))
        _, data = load_snapshot(snapshot)
        cm = build_cost_map(data['points'], None, [0,0,1], dict(radius=.16, slope_policy='disabled',
            support_mask=data['support_mask'], obstacle_mask=data['obstacle_mask'], clearance_height=1.))
        self.assertIn(blocked_row, cm.obstacle_rows)
        self.assertFalse(cm.eligible[blocked_row])
        camera = data['points'][blocked_row] + [0,0,1]
        self.assertEqual(select_endpoint(cm.points, cm, camera)['reason'], 'blocked_support')

    def test_all_empty_additional_masks_preserve_baseline(self):
        points = self.points.reshape(-1,3)
        support = np.ones(len(points),bool)
        baseline = build_cost_map(points, np.zeros(len(points)), [0,0,1], dict(radius=.16, support_mask=support))
        other = build_cost_map(points, np.zeros(len(points)), [0,0,1], dict(radius=.16, support_mask=support,
            sam_traversability_mask=np.zeros(len(points),bool), obstacle_mask=np.zeros(len(points),bool)))
        np.testing.assert_array_equal(baseline.cost, other.cost)
        np.testing.assert_array_equal(baseline.neighbours(400), other.neighbours(400))

    def test_positive_masks_do_not_bypass_enabled_slope_limits(self):
        points = self.points.reshape(-1,3)
        masks = np.ones(len(points),bool)
        enforced = build_cost_map(points, np.full(len(points),np.pi/3), [0,0,1],
            dict(radius=.16, sam_traversability_mask=masks))
        disabled = build_cost_map(points, None, [0,0,1], dict(radius=.16, slope_policy='disabled', sam_traversability_mask=masks))
        self.assertTrue(np.isinf(enforced.cost).all())
        self.assertTrue(np.isfinite(disabled.cost).all())
        self.assertTrue(np.isnan(disabled.slope).all())

    @unittest.skipUnless(NATIVE.exists(), 'Compile native search first')
    def test_native_and_python_match_with_slope_disabled_and_negative_evidence(self):
        snapshot = self.snapshot()
        costs(snapshot, self.root/'costs', slope_policy='disabled', reference=[0,0,1], radius=.16, clearance_height=1.)
        first = route(self.root/'costs', self.root/'python', start=14*28+3, goal=14*28+25)
        second = route(self.root/'costs', self.root/'native', start=14*28+3, goal=14*28+25, library=NATIVE)
        self.assertAlmostEqual(first['metadata']['total_cost'], second['metadata']['total_cost'], places=9)

    @unittest.skipUnless(NATIVE.exists(), 'Compile native search first')
    def test_native_policy_disables_grade_for_vertical_edges(self):
        from path_planner.routes import restore_cost_map
        points = np.array([[0.,0,0],[0.,0,.5]])
        settings = _settings(dict(radius=1., connection_radius=.6, max_gap=3., obstacle_margin=0.,
                                  support_mask=np.ones(2,bool), slope_policy='disabled'))
        cm = restore_cost_map(points, np.ones(2), np.ones(2,bool), np.array([0.,0,1]), settings, CostMap)
        self.assertTrue(cm._connected(0,1))
        native = NativeSearch(NATIVE)
        result = exact_route(cm,0,1,np.array([0,1]),native,find_path,lambda *a,**k:None,allow_partial=True)
        self.assertEqual(result.status,'ok')
        self.assertAlmostEqual(result.total_cost,.5)

    def test_target_changes_semantic_interpretation_without_changing_query(self):
        for target in ('traversable','non_traversable'):
            masks = segment_groups(Segmenter(), self.rgb, query_groups(['log'], target, ['path']))
            self.assertTrue(masks['sam_traversability_mask'][:,14:].all())
            selected = 'vlm_traversability_mask' if target=='traversable' else 'vlm_non_traversability_mask'
            self.assertTrue(masks[selected][14,20])

    def test_query_embedding_is_shared_but_polarity_sources_remain_distinct(self):
        model = Segmenter()
        groups = query_groups(['path'], 'non_traversable', ['path'])
        masks = segment_groups(model, self.rgb, groups)
        self.assertEqual(model.images,1)
        self.assertEqual(model.calls,['path'])
        np.testing.assert_array_equal(masks['sam_traversability_mask'], masks['vlm_non_traversability_mask'])

    def test_extra_queries_work_without_vlm_proposals(self):
        model = Segmenter()
        masks = segment_groups(model,self.rgb,query_groups([], 'non_traversable',['path']))
        self.assertEqual(model.calls,['path'])
        self.assertTrue(masks['sam_traversability_mask'].any())
        self.assertFalse(masks['vlm_non_traversability_mask'].any())

    def test_streaming_extra_queries_survive_empty_vlm_and_ttl_with_vlm_off_or_on(self):
        from orchestrator.streaming import parse_args, run
        workers, requests = [], []
        class Worker:
            def __init__(self,role,python,config,directory):
                self.role,self.pending=role,False
                self.process=SimpleNamespace(pid=len(workers)+1)
                self.ready={'load_seconds':0.}
                workers.append(self)
            def submit(self,**request): self.pending=True
            def receive(self,timeout=300):
                self.pending=False
                return {'prompts':[],'raw_response':'{"prompts":[]}','elapsed_seconds':.01}
            def poll(self): return self.receive() if self.pending else None
            def call(self,**request):
                with np.load(request['input']) as value:rgb=value['rgb'].copy()
                if self.role=='lingbot':
                    n=len(rgb)
                    np.savez(request['output'],images=rgb,depth=np.full((n,28,28),2.,np.float32),
                        depth_conf=np.full((n,28,28),3.,np.float32),intrinsic=np.tile(np.eye(3),(n,1,1)),
                        extrinsic=np.tile(np.c_[np.eye(3),[0,0,0]],(n,1,1)))
                else:
                    requests.append(request['groups'])
                    masks=segment_groups(Segmenter(),rgb,request['groups'])
                    np.savez(request['output'],**masks,out_binary_masks=masks['vlm_non_traversability_mask'])
                return {'elapsed_seconds':.01}
            def close(self): pass
        def frames(args,state):
            state.update(source_fps=1.,source_frame_count=3,input_stride=1,dropped_sampled_frames=0)
            for index,timestamp in enumerate((0.,10.,20.)):
                yield dict(source_index=index,source_time=timestamp,captured=0.,rgb=self.rgb)
        video,checkpoint=self.root/'video.mp4',self.root/'model.pt'
        video.touch();checkpoint.touch()
        for disabled in (False,True):
            arguments=['--video-path',str(video),'--output-dir',str(self.root/f'stream-{disabled}'),
                '--lingbot-checkpoint',str(checkpoint),'--map-step','1','--save-frames',
                '--extra-traversability-prompts','path']
            if disabled:arguments+=['--no-vlm']
            args=parse_args(arguments)
            with patch('orchestrator.streaming.Worker',Worker),patch('orchestrator.streaming.video_frames',frames):
                run(args,output_stream=io.StringIO())
            for index in range(3):
                with np.load(args.output_dir/'masks'/f'frame_{index:06d}.npz') as value:
                    self.assertTrue(value['sam_traversability_mask'].any())
                    self.assertFalse(value['vlm_non_traversability_mask'].any())
        self.assertEqual(len(requests),6)
        self.assertTrue(all(request['sam_traversability_mask']==['path'] for request in requests))

    def test_standalone_vlm_instruction_and_target_are_bound(self):
        class Proposer:
            def propose(self,image,instruction):
                self.instruction = instruction
                return '{"prompts":["path"]}'
        model=Proposer()
        args=SimpleNamespace(lingbot_dir=self.predictions,keyframe_step=1,max_frames=1,model='fake',
            robot='robot',ground_crop=False,target='traversable',output=self.root/'proposals.json')
        propose(args,model)
        value=json.loads(args.output.read_text())
        self.assertEqual(value['target'],'traversable')
        self.assertEqual(value['observations'][0]['rgb_sha256'],rgb_digest(self.rgb))
        self.assertIn('continuous ground',model.instruction)
        with self.assertRaises(ValueError):parse_proposal('{"target":"non_traversable","prompts":[]}', 'traversable')

    def test_reused_source_target_and_query_mismatches_are_rejected(self):
        semantic=self.semantic('reused')
        modes=dict(vlm_queries='reuse',extra_traversability='reuse',manual_non_traversability='off')
        with self.assertRaisesRegex(ValueError,'missing required|target mismatch'):
            assemble(self.root/'opposite',reused=semantic,modes=modes,target='traversable',extra_prompts=['path'])
        with self.assertRaisesRegex(ValueError,'query list differs'):
            assemble(self.root/'changed',reused=semantic,modes=modes,target='non_traversable',extra_prompts=['pavement'])

    def test_disabled_reused_sources_do_not_leak_into_combined_masks(self):
        semantic=self.semantic('reused')
        result=assemble(self.root/'selected',reused=semantic,
            modes=dict(vlm_queries='off',extra_traversability='reuse',manual_non_traversability='off'),
            target='non_traversable',extra_prompts=['path'])
        with np.load(result/'frame_000000.npz') as value:
            self.assertNotIn('vlm_non_traversability_mask',value.files)
            self.assertFalse(value['out_binary_masks'].any())
            self.assertTrue(value['sam_traversability_mask'].any())

    def test_snapshot_tampering_and_rgb_mismatch_are_rejected(self):
        semantic=self.semantic('wrong')
        file=semantic/'frame_000000.npz'
        with np.load(file) as saved: arrays={k:saved[k] for k in saved.files}
        arrays['rgb_sha256']=np.asarray('different')
        np.savez(file,**arrays)
        with self.assertRaisesRegex(ValueError,'provenance mismatch'):
            build_snapshot(self.predictions,self.root/'bad',semantic_dir=semantic,source_fps=30,unproject=self.unproject)
        snapshot=self.snapshot('good')
        np.save(snapshot/'obstacle_mask.npy',np.zeros(3,bool))
        with self.assertRaisesRegex(ValueError,'Artifact changed'):
            load_snapshot(snapshot)

    def test_reuse_adds_planning_without_reconstruction_or_slope_fitting(self):
        snapshot=self.snapshot()
        first=run_pipeline(self.config(snapshot,'first',False,True))
        cfg=self.config(snapshot,'reused',False,True)
        cfg['analysis']['costs']='reuse'
        cfg['inputs']['costs']=first['outputs']['costs']
        with patch('orchestrator.pipeline.costs',create=True) as unused, patch('path_planner.slope.estimate_slope',side_effect=AssertionError):
            state=run_pipeline(cfg)
        self.assertEqual(state['status'],'complete')
        self.assertFalse((self.root/'reused/costs').exists())
        self.assertEqual(state['outputs']['costs'],first['outputs']['costs'])

    def test_missing_reference_produces_explicit_no_route(self):
        semantic=self.semantic('positive-only')
        snapshot=self.root/'no-base-reference'
        build_snapshot(self.predictions,snapshot,semantic_dir=semantic,source_fps=30,unproject=self.unproject)
        receipt=costs(snapshot,self.root/'no-reference',slope_policy='disabled')
        self.assertEqual(receipt['metadata']['status'],'missing_reference')
        final=route(self.root/'no-reference',self.root/'no-route')
        self.assertEqual(final['metadata']['status'],'missing_reference')
        self.assertEqual(final['metadata']['route_points'],0)

    def test_bad_dependencies_are_rejected_before_output_or_models(self):
        cases=[dict(planning={'enabled':True}),
               dict(sam={'vlm_queries':'compute'}),
               dict(analysis={'costs':'compute'},planning={'slope_policy':'enforce'}),
               dict(sam={'extra_traversability':{'mode':'compute','prompts':[]}})]
        for number,extra in enumerate(cases):
            cfg=dict(output=str(self.root/f'bad-{number}'))|extra
            with self.assertRaises(ValueError):resolve(cfg)
            self.assertFalse(Path(cfg['output']).exists())

    def test_cost_and_viewer_only_jobs_do_not_import_model_libraries(self):
        code="import sys; import path_planner; import orchestrator.analysis; import visualizer.integrated; assert not any(x in sys.modules for x in ['torch','transformers','sam3','viser','lingbot_map'])"
        result=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_viewer_accepts_only_matching_current_artifacts(self):
        from visualizer.integrated import load_view
        snapshot=self.snapshot()
        state=run_pipeline(self.config(snapshot,'view',True,True))
        view=load_view(self.root/'view')
        self.assertIn('Steepness',view['modes'])
        self.assertIn('Cost map',view['modes'])
        self.assertGreater(len(view['route']),0)
        raw=np.load(snapshot/'source_rgb.npy')
        self.assertTrue((raw==100).all())
        cost=Path(state['outputs']['costs'])/'cost.npy'
        np.save(cost,np.zeros(3))
        view=load_view(self.root/'view')
        self.assertNotIn('Cost map',view['modes'])
        self.assertEqual(len(view['route']),0)
        self.assertTrue(any('withheld' in note for note in view['notes']))


if __name__=='__main__':unittest.main()
