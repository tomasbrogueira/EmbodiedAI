"""Select computed/reused mask sources independently without relabeling evidence."""
import json
from pathlib import Path
import numpy as np
from orchestrator.artifacts import write_json, digest


def assemble(output, *, computed=None, reused=None, modes, target, extra_prompts=(), manual_prompts=()):
    inputs = {}
    for mode, directory in (('compute', computed), ('reuse', reused)):
        if directory is not None:
            directory = Path(directory)
            manifest = json.loads((directory / 'manifest.json').read_text())
            if manifest.get('schema_version') != 1:
                raise ValueError('Unsupported semantic manifest')
            inputs[mode] = (directory, manifest)
    channel_by_stage = {
        'vlm_queries': 'vlm_traversability_mask' if target == 'traversable' else 'vlm_non_traversability_mask',
        'extra_traversability': 'sam_traversability_mask',
        'manual_non_traversability': 'manual_non_traversability_mask',
    }
    selected, frames, groups, identities = {}, None, {}, {}
    for stage, mode in modes.items():
        if mode == 'off':
            continue
        directory, manifest = inputs[mode]
        channel = channel_by_stage[stage]
        available = manifest.get('enabled_channels', [k for k, v in manifest.get('query_groups', {}).items() if v])
        if channel not in available:
            raise ValueError(f'Reused/computed semantic source is missing required {channel}')
        if stage == 'vlm_queries' and manifest.get('target', 'non_traversable') != target:
            raise ValueError('Semantic VLM target mismatch')
        requested = extra_prompts if stage == 'extra_traversability' else manual_prompts if stage == 'manual_non_traversability' else []
        saved = manifest['query_groups'][channel]
        if requested and list(requested) != saved:
            raise ValueError(f'Reused query list differs for {stage}')
        if frames is not None and frames != manifest['frames']:
            raise ValueError('Semantic source frame sets differ')
        frames = manifest['frames']
        selected[channel] = directory
        groups[channel] = saved
        identities[channel] = dict(mode=mode, manifest_sha256=digest(directory / 'manifest.json'))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name in frames or []:
        if Path(name).name != name or not name.startswith('frame_'):
            raise ValueError('Invalid semantic source filename')
        masks, rgb_hash, shape = {}, None, None
        for channel, directory in selected.items():
            with np.load(directory / name, allow_pickle=False) as saved:
                current_hash = str(saved['rgb_sha256'].item())
                mask = np.asarray(saved[channel])
                if mask.dtype.kind != 'b' or mask.ndim != 2:
                    raise ValueError('Semantic source must be a Boolean image mask')
                if rgb_hash is not None and (current_hash != rgb_hash or mask.shape != shape):
                    raise ValueError('Semantic sources identify different RGB/dimensions')
                rgb_hash, shape = current_hash, mask.shape
                masks[channel] = mask.copy()
        negative = np.zeros(shape, dtype=bool)
        for channel, mask in masks.items():
            if 'non_traversability' in channel:
                negative |= mask
        np.savez_compressed(output / name, **masks, out_binary_masks=negative, rgb_sha256=np.asarray(rgb_hash))
    write_json(output / 'manifest.json', dict(schema_version=1, frames=frames, target=target,
        query_groups=groups, enabled_channels=list(selected), sources=identities,
        alignment='SAM on saved LingBot RGB; source-specific masks'))
    return output
