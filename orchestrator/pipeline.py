"""Composable offline pipelines with explicit compute/reuse/off stages."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
from orchestrator.artifacts import write_json, verify, identity, digest
from orchestrator.config import resolve
from orchestrator.paths import ROOT


def run_pipeline(configuration, command_runner=None):
    cfg = resolve(configuration)
    out = Path(cfg['output'])
    if out.exists():
        raise ValueError('Choose a new output directory; completed runs are never overwritten')
    out.mkdir(parents=True)
    write_json(out / 'config.json', cfg)
    state = {'schema_version': 2, 'status': 'running', 'stages': {}, 'features': cfg,
             'outputs': {}, 'config_identity': identity(cfg)}

    def status(stage, mode, **details):
        state['stages'][stage] = {'mode': mode, **details}
        write_json(out / 'status.json', state)

    def command(role, module, arguments):
        interpreter = cfg['runtime'].get(role + '_python', sys.executable)
        cmd = [interpreter, '-B', '-m', module, *map(str, arguments)]
        if command_runner is not None:
            command_runner(role, cmd)
            return
        from orchestrator.resources import gpu_allocation
        with gpu_allocation(), (out / (role + '.log')).open('a') as log:
            subprocess.run(cmd, check=True, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                           env=os.environ | {'PYTHONPATH': str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', ''),
                                             'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
                                             'PYTHONDONTWRITEBYTECODE': '1'})

    def remember(name, path):
        state['outputs'][name] = str(Path(path).resolve())

    try:
        for name, mode in (
            ('geometry', cfg['geometry']['mode']), ('vlm', cfg['vlm']['mode']),
            ('support', cfg['sam']['support']), ('vlm_queries', cfg['sam']['vlm_queries']),
            ('extra_traversability', cfg['sam']['extra_traversability']['mode']),
            ('manual_non_traversability', cfg['sam']['manual_non_traversability']['mode']),
            ('steepness', cfg['analysis']['steepness']), ('costs', cfg['analysis']['costs']),
            ('route', 'compute' if cfg['planning']['enabled'] else 'off')):
            status(name, mode, status='disabled' if mode == 'off' else 'pending')
        inputs, sam, analysis = cfg['inputs'], cfg['sam'], cfg['analysis']
        prediction_dir = Path(inputs['predictions']) if inputs.get('predictions') else None
        video = Path(inputs['video']) if inputs.get('video') else None
        if cfg['geometry']['mode'] == 'compute':
            if cfg['geometry']['policy'] != 'offline_uniform_anchors':
                raise ValueError('Use orchestrator.streaming for causal reconstruction; do not substitute offline geometry')
            root = out / 'geometry'
            command('lingbot', 'lingbot.offline', ['--video_path', video, '--output_folder', root,
                '--model_path', cfg['runtime'].get('lingbot_checkpoint', ROOT / 'lingbot-map/lingbot-map.pt'),
                '--mode', 'streaming', '--kv_cache_sliding_window', '32', '--keyframe_interval', '2',
                '--num_scale_frames', '8', '--use_sdpa', '--save_predictions', '--no_render', '--no_original_video'])
            batch = json.loads((root / 'batch_results.json').read_text())
            if not batch.get('results') or not all(r.get('success') for r in batch['results']):
                raise RuntimeError('LingBot reported incomplete/failed reconstruction')
            prediction_dir = root / video.stem
        if prediction_dir:
            if not list(prediction_dir.glob('frame_*.npz')):
                raise ValueError('No predictions available')
            remember('predictions', prediction_dir)
            status('geometry', cfg['geometry']['mode'], status='computed' if cfg['geometry']['mode'] == 'compute' else 'reused')
        # 2D VLM/SAM-only jobs can sample RGB without loading LingBot.
        needs_rgb = cfg['vlm']['mode'] == 'compute' or any(sam[k] == 'compute' for k in ('vlm_queries',)) or any(sam[k]['mode'] == 'compute' for k in ('extra_traversability', 'manual_non_traversability'))
        if needs_rgb and prediction_dir is None:
            if video is None:
                raise ValueError('VLM/SAM image queries need saved RGB predictions or input video')
            from orchestrator.streaming import video_frames
            args = SimpleNamespace(video_path=video, input_fps=cfg['geometry'].get('input_fps', 4.),
                                   frame_count=0, max_frame_num=1024, max_frames=0, realtime=False, image_size=518)
            prediction_dir = out / 'rgb_samples'
            prediction_dir.mkdir()
            for frame in video_frames(args, {}):
                np.savez(prediction_dir / f'frame_{frame["source_index"]:06d}.npz', images=frame['rgb'])
        proposal_file = Path(inputs['proposals']) if inputs.get('proposals') else None
        if cfg['vlm']['mode'] == 'compute':
            proposal_file = out / 'proposals.json'
            args = ['propose', '--lingbot-dir', prediction_dir, '--output', proposal_file,
                    '--target', cfg['vlm']['target'], '--keyframe-step', cfg['vlm'].get('keyframe_step', 30)]
            if cfg['vlm'].get('model'):
                args += ['--model', cfg['vlm']['model']]
            if cfg['vlm'].get('instruction_file'):
                args += ['--instruction-file', cfg['vlm']['instruction_file']]
            command('vlm', 'vlm', args)
        if cfg['vlm']['mode'] != 'off':
            proposals = json.loads(proposal_file.read_text())
            if proposals.get('target', 'non_traversable') != cfg['vlm']['target']:
                raise ValueError('Reused proposals have a different traversability target')
            from orchestrator.shared import read_rgb, rgb_digest
            for observation in proposals.get('observations', []):
                name = observation['frame']
                if Path(name).name != name:
                    raise ValueError('Invalid proposal source filename')
                expected = observation.get('rgb_sha256')
                if expected and (prediction_dir is None or rgb_digest(read_rgb(prediction_dir / name)) != expected):
                    raise ValueError('Reused proposals belong to different RGB')
            remember('proposals', proposal_file)
            status('vlm', cfg['vlm']['mode'], status='computed' if cfg['vlm']['mode'] == 'compute' else 'reused')
        support_dir = Path(inputs['support_dir']) if inputs.get('support_dir') else None
        if sam['support'] == 'compute':
            support_args = ['support', '--video_path', video, '--output_folder', out / 'support', '--prompt', sam['support_prompt']]
            if cfg['runtime'].get('sam_checkpoint'):
                support_args += ['--checkpoint', cfg['runtime']['sam_checkpoint']]
            command('sam', 'sam', support_args)
            support_dir = out / 'support' / (video.stem + '_masks')
        if sam['support'] != 'off':
            remember('support', support_dir)
            status('support', sam['support'], status='computed' if sam['support'] == 'compute' else 'reused')
        reused_semantic = Path(inputs['semantic_dir']) if inputs.get('semantic_dir') else None
        semantic_dir = None
        query_modes = [sam['vlm_queries'], sam['extra_traversability']['mode'], sam['manual_non_traversability']['mode']]
        if 'compute' in query_modes:
            from orchestrator.io_utils import frame_number
            selected = out / 'selected_predictions'
            selected.mkdir()
            for path in sorted(prediction_dir.glob('frame_*.npz')):
                if frame_number(path) % cfg['projection']['frame_step'] == 0:
                    (selected / path.name).symlink_to(path.resolve())
            semantic_dir = out / 'computed_semantic_masks'
            args = ['segment', '--lingbot-dir', selected, '--output-dir', semantic_dir, '--frame-step', '1']
            if sam['vlm_queries'] == 'compute':
                args += ['--prompts', proposal_file, '--target', cfg['vlm']['target']]
            if sam['extra_traversability']['mode'] == 'compute':
                args += ['--extra-traversability-prompts', *sam['extra_traversability']['prompts']]
            if sam['manual_non_traversability']['mode'] == 'compute':
                args += ['--manual-non-traversability-prompts', *sam['manual_non_traversability']['prompts']]
            if cfg['runtime'].get('sam_checkpoint'):
                args += ['--checkpoint', cfg['runtime']['sam_checkpoint']]
            command('sam', 'sam', args)
        if any(m != 'off' for m in query_modes):
            from orchestrator.mask_sources import assemble
            semantic_dir = assemble(out / 'semantic_masks', computed=semantic_dir, reused=reused_semantic,
                modes=dict(zip(('vlm_queries', 'extra_traversability', 'manual_non_traversability'), query_modes)),
                target=cfg['vlm']['target'], extra_prompts=sam['extra_traversability']['prompts'],
                manual_prompts=sam['manual_non_traversability']['prompts'])
            remember('semantic_masks', semantic_dir)
            for name, mode in zip(('vlm_queries', 'extra_traversability', 'manual_non_traversability'), query_modes):
                if mode != 'off':
                    status(name, mode, status='computed' if mode == 'compute' else 'reused')
        snapshot = Path(inputs['snapshot']) if inputs.get('snapshot') else None
        if snapshot is None and cfg['projection']['enabled']:
            from orchestrator.projection import build_snapshot
            snapshot = out / 'map'
            timestamps, source_fps = inputs.get('timestamps'), inputs.get('source_fps')
            if source_fps is None and timestamps is None and video:
                from vlm.evaluate import video_info
                numerator, denominator = map(float, video_info(video)['avg_frame_rate'].split('/'))
                source_fps = numerator / denominator
            build_snapshot(prediction_dir, snapshot, support_dir=support_dir if sam['support'] != 'off' else None,
                           semantic_dir=semantic_dir if any(m != 'off' for m in query_modes) else None,
                           video_name=video.stem if video else inputs.get('video_name'), support_prompt=sam['support_prompt'],
                           frame_step=cfg['projection']['frame_step'], pixel_step=cfg['projection']['pixel_step'],
                           timestamps=timestamps, source_fps=source_fps,
                           min_depth_conf=cfg['projection']['min_depth_conf'], max_depth=cfg['projection']['max_depth'],
                           reconstruction_policy=cfg['geometry']['policy'])
        if snapshot:
            verify(snapshot, 'map')
            remember('map', snapshot)
        from orchestrator.analysis import steepness, costs, route
        common = {key: analysis[key] for key in ('reference', 'radius', 'reference_source') if key in analysis}
        slope_dir = Path(inputs['steepness']) if inputs.get('steepness') else None
        if analysis['steepness'] == 'compute':
            slope_dir = out / 'steepness'
            steepness(snapshot, slope_dir, **common)
        if analysis['steepness'] != 'off':
            slope_receipt = verify(slope_dir, 'steepness')
            if slope_receipt['metadata']['snapshot_identity'] != verify(snapshot, 'map')['identity']:
                raise ValueError('Reused steepness snapshot mismatch')
            remember('steepness', slope_dir)
            status('steepness', analysis['steepness'], status=slope_receipt['metadata']['status'])
        cost_dir = Path(inputs['costs']) if inputs.get('costs') else None
        if analysis['costs'] == 'compute':
            cost_dir = out / 'costs'
            extra = {key: analysis[key] for key in ('clearance_height', 'calibration_source', 'settings') if key in analysis}
            costs(snapshot, cost_dir, slope_policy=cfg['planning']['slope_policy'], slopes=slope_dir, **common, **extra)
        if analysis['costs'] != 'off':
            receipt = verify(cost_dir, 'costs')
            if receipt['metadata']['snapshot_identity'] != verify(snapshot, 'map')['identity'] or receipt['metadata']['slope_policy'] != cfg['planning']['slope_policy']:
                raise ValueError('Reused costs snapshot/policy mismatch')
            remember('costs', cost_dir)
            status('costs', analysis['costs'], status=receipt['metadata']['status'])
        if cfg['planning']['enabled']:
            route_dir = out / 'route'
            options = {k: cfg['planning'][k] for k in ('start', 'goal', 'library') if k in cfg['planning']}
            if 'library' not in options and (ROOT / 'path-planning/corridor_astar_feasible.so').exists():
                options['library'] = ROOT / 'path-planning/corridor_astar_feasible.so'
            receipt = route(cost_dir, route_dir, **options)
            remember('route', route_dir)
            status('route', 'compute', status=receipt['metadata']['status'])
        state['status'] = 'complete'
        write_json(out / 'status.json', state)
        if cfg['visualizer']['enabled']:
            args = cfg['visualizer']
            interpreter = cfg['runtime'].get('viewer_python', sys.executable)
            process = subprocess.Popen([interpreter, '-m', 'visualizer', str(out), '--port', str(args['port']), '--host', args['host']], cwd=ROOT)
            state['viewer_pid'] = process.pid
            write_json(out / 'status.json', state)
        return state
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(out / 'status.json', state)
        raise
