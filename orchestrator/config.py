"""Validate requested stages before any model is loaded or output is created."""
from copy import deepcopy
from pathlib import Path
from vlm.prompts import normalize_prompts, validate_target

MODES = ('compute', 'reuse', 'off')


def resolve(value):
    if not isinstance(value, dict):
        raise ValueError('Pipeline configuration must be an object')
    cfg = deepcopy(value)
    allowed = {'inputs', 'output', 'geometry', 'vlm', 'sam', 'analysis', 'planning',
               'visualizer', 'projection', 'runtime'}
    if set(cfg) - allowed:
        raise ValueError(f'Unknown configuration sections: {sorted(set(cfg) - allowed)}')
    cfg.setdefault('inputs', {})
    cfg.setdefault('runtime', {})
    if not cfg.get('output'):
        raise ValueError('An output directory is required')
    defaults = {
        'geometry': dict(mode='off', policy='offline_uniform_anchors'),
        'vlm': dict(mode='off', target='non_traversable'),
        'analysis': dict(steepness='off', costs='off', reference_source='base_support', calibration_source='base_support', workers=0),
        'planning': dict(enabled=False, slope_policy=None),
        'visualizer': dict(enabled=False, port=8094, host='127.0.0.1'),
        'projection': dict(enabled=True, frame_step=2, pixel_step=1, min_depth_conf=2.5, max_depth=25.),
    }
    for key, default in defaults.items():
        if not isinstance(cfg.get(key, {}), dict):
            raise ValueError(f'{key} must be an object')
        cfg[key] = default | cfg.get(key, {})
    from path_planner.parallel import worker_count
    worker_count(cfg['analysis']['workers'])
    if cfg['geometry']['policy'] not in ('offline_uniform_anchors', 'causal_stream'):
        raise ValueError('Unknown reconstruction policy')
    if cfg['geometry']['mode'] == 'compute' and cfg['geometry']['policy'] == 'causal_stream':
        raise ValueError('Use orchestrator.streaming for causal reconstruction; the offline runner does not substitute geometry policies')
    sam = cfg.setdefault('sam', {})
    sam.setdefault('support', 'off')
    sam.setdefault('support_prompt', 'floor')
    sam.setdefault('vlm_queries', 'off')
    for name in ('extra_traversability', 'manual_non_traversability'):
        sam[name] = {'mode': 'off', 'prompts': []} | sam.get(name, {})
        sam[name]['prompts'] = normalize_prompts(sam[name]['prompts'])
        if sam[name]['mode'] == 'compute' and not sam[name]['prompts']:
            raise ValueError(f'{name} compute requires at least one SAM prompt')
    modes = [cfg['geometry']['mode'], cfg['vlm']['mode'], sam['support'], sam['vlm_queries'],
             sam['extra_traversability']['mode'], sam['manual_non_traversability']['mode'],
             cfg['analysis']['steepness'], cfg['analysis']['costs']]
    if any(mode not in MODES for mode in modes):
        raise ValueError('Stage modes must be compute, reuse or off')
    validate_target(cfg['vlm']['target'])
    if sam['vlm_queries'] == 'compute' and cfg['vlm']['mode'] == 'off':
        raise ValueError('VLM-query SAM needs computed or reused typed VLM proposals')
    if cfg['planning']['slope_policy'] is None:
        cfg['planning']['slope_policy'] = 'disabled' if cfg['analysis']['steepness'] == 'off' else 'enforce'
    if cfg['planning']['slope_policy'] not in ('enforce', 'disabled'):
        raise ValueError('Unknown slope policy')
    if cfg['analysis']['costs'] == 'compute' and cfg['planning']['slope_policy'] == 'enforce' and cfg['analysis']['steepness'] == 'off':
        raise ValueError('Slope-aware costs require steepness compute/reuse; select disabled policy to omit slopes')
    if cfg['planning']['enabled'] and cfg['analysis']['costs'] == 'off':
        raise ValueError('Planning requires costs compute/reuse; disabled stages are not auto-enabled')
    analysis = cfg['analysis']['steepness'] != 'off' or cfg['analysis']['costs'] != 'off' or cfg['planning']['enabled']
    if analysis and not cfg['projection']['enabled'] and not cfg['inputs'].get('snapshot'):
        raise ValueError('Analysis requires projection or a supplied snapshot')
    supplied = cfg['inputs'].get('snapshot')
    if supplied and (cfg['geometry']['mode'] != 'off' or any(mode != 'off' for mode in modes[1:6])):
        raise ValueError('A reused immutable snapshot cannot be combined with new geometry or semantic stages; project a new snapshot')
    if cfg['geometry']['mode'] == 'compute' and not cfg['inputs'].get('video'):
        raise ValueError('Reconstruction requires an input video')
    if cfg['geometry']['mode'] == 'reuse' and not cfg['inputs'].get('predictions'):
        raise ValueError('Geometry reuse requires predictions')
    if sam['support'] == 'compute' and not cfg['inputs'].get('video'):
        raise ValueError('Support-video SAM requires an input video')
    for section, mode, field in (
        ('vlm', cfg['vlm']['mode'], 'proposals'),
        ('support', sam['support'], 'support_dir'),
        ('steepness', cfg['analysis']['steepness'], 'steepness'),
        ('costs', cfg['analysis']['costs'], 'costs'),
    ):
        if mode == 'reuse' and not cfg['inputs'].get(field):
            raise ValueError(f'{section} reuse requires inputs.{field}')
    needs_reused_masks = any(mode == 'reuse' for mode in modes[3:6])
    computes_masks = any(mode == 'compute' for mode in modes[3:6])
    if needs_reused_masks and not cfg['inputs'].get('semantic_dir'):
        raise ValueError('Semantic reuse requires inputs.semantic_dir')
    if cfg['projection']['enabled'] and not supplied and cfg['geometry']['mode'] == 'off':
        if not cfg['inputs'].get('predictions'):
            # 2D-only pipelines have no geometry and no map stage.
            if analysis:
                raise ValueError('Analysis needs saved geometry or a snapshot')
            cfg['projection']['enabled'] = False
    for field in ('video', 'predictions', 'snapshot', 'proposals', 'support_dir', 'semantic_dir', 'steepness', 'costs'):
        if cfg['inputs'].get(field) and not Path(cfg['inputs'][field]).exists():
            raise ValueError(f'Input does not exist: {field}')
    cfg['output'] = str(Path(cfg['output']).resolve())
    return cfg
