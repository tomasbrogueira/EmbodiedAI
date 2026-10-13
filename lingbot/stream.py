"""Causal LingBot adapter; model imports occur on construction."""
import sys
from pathlib import Path
import numpy as np
from orchestrator.paths import ROOT

class LingBotStream:
    """The existing model's causal scale/forward loop, without collecting a video."""
    def __init__(self, config):
        import torch
        from types import SimpleNamespace
        sys.path.insert(0, str(ROOT / 'lingbot-map'))
        from demo import load_model, postprocess
        self.torch, self.postprocess = torch, postprocess
        self.config = config
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        args = SimpleNamespace(mode='windowed', image_size=config['image_size'], patch_size=14,
            enable_3d_rope=True, max_frame_num=config['max_frame_num'],
            kv_cache_sliding_window=config.get('kv_cache_window', 64), num_scale_frames=2, use_sdpa=True,
            camera_num_iterations=config['camera_iterations'], model_path=config['lingbot_checkpoint'])
        # Load checkpoint on CPU to avoid a second full GPU weight copy.
        self.model = load_model(args, 'cpu')
        self.model.aggregator.to(dtype=self.dtype)
        self.model.to('cuda').eval()
        self.model.clean_kv_cache()
        self.count = 0
        self.scale_frames = 0

    def reset(self):
        self.model.clean_kv_cache()
        self.model._set_skip_append(False)
        self.count = self.scale_frames = 0

    def frame(self, request):
        torch = self.torch
        with np.load(request['input'], allow_pickle=False) as data:
            rgb = data['rgb'].copy()  # T,H,W,3; already original OpenCV preprocessing.
        images = torch.from_numpy(rgb).permute(0, 3, 1, 2).float() / 255.
        batch = images.unsqueeze(0).to('cuda')
        if self.count == 0:
            self.scale_frames = len(images)
            block = self.scale_frames
            skip = False
        else:
            if len(images) != 1:
                raise ValueError('After initialization, send one frame at a time')
            block = 1
            skip = (self.count - self.scale_frames) % 2 != 0
        if self.count + len(images) > self.config['max_frame_num']:
            raise ValueError('Stream exceeds max-frame-num; increase the configured limit')
        if skip:
            self.model._set_skip_append(True)
        try:
            with torch.inference_mode(), torch.autocast('cuda', dtype=self.dtype):
                pred = self.model.forward(batch, num_frame_for_scale=self.scale_frames,
                    num_frame_per_block=block, causal_inference=True)
            # Exactly the same postprocessing as the existing batch demo,
            # including its extrinsic convention. Do not invert poses again.
            pred, _ = self.postprocess(pred, images)
        finally:
            if skip:
                self.model._set_skip_append(False)
        arrays = {k: pred[k].float().numpy() for k in ('depth', 'depth_conf', 'intrinsic', 'extrinsic')}
        np.savez(request['output'], images=rgb, **arrays)
        self.count += len(images)
        return {'frames': len(images)}


