# Support direct compatibility-script execution without environment mutation.
if __package__ in (None, ''):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
"""Offline LingBot reconstruction with uniformly distributed initialization views."""
from pathlib import Path
import sys

import numpy as np
import torch


def run_with_anchors(run, model, images, args):
    if args.mode != 'streaming':
        raise ValueError('Distributed initialization requires streaming inference')
    if images.ndim not in (4, 5):
        raise ValueError('Expected frames with shape NCHW or BNCHW')
    axis = images.ndim - 4
    count = images.shape[axis]
    if count < 1 or args.num_scale_frames < 1:
        raise ValueError('Expected nonempty video frames and positive scale-frame count')
    anchors = np.linspace(0, count - 1, min(args.num_scale_frames, count)).round().astype(int)
    remaining = np.ones(count, dtype=bool)
    remaining[anchors] = False
    order = np.r_[anchors, np.flatnonzero(remaining)]
    indices = torch.as_tensor(order, dtype=torch.long, device=images.device)
    result = run(model, images.index_select(axis, indices), args)
    inverse = np.argsort(order)
    # Restore every frame field together, including the images used by SAM/export.
    for key in ('pose_enc', 'depth', 'depth_conf', 'world_points', 'world_points_conf',
                'frame_type', 'is_keyframe', 'extrinsic', 'intrinsic', 'images'):
        if key in result:
            value = result[key]
            if not isinstance(value, np.ndarray) or not value.ndim or value.shape[0] != count:
                raise ValueError(f'Unexpected frame axis in {key}')
            result[key] = value[inverse]
    result["initialization_source_frames"] = anchors
    result["reconstruction_policy"] = np.array("uniform_scale_anchors_v1")
    return result


def main():
    lingbot = Path(__file__).resolve().parents[1] / 'lingbot-map'
    sys.path[:0] = [str(lingbot / 'demo_render'), str(lingbot)]
    import batch_demo
    run = batch_demo.run_inference
    batch_demo.run_inference = lambda model, images, args: run_with_anchors(run, model, images, args)
    batch_demo.main()


if __name__ == '__main__':
    main()
