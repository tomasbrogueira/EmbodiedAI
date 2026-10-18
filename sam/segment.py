"""Segment every concept and save one aligned union mask per LingBot frame."""

import sys
import json
from pathlib import Path

import numpy as np
from PIL import Image

from orchestrator.shared import frames, read_rgb, rgb_digest, write_json
from vlm.prompts import load_prompts


class SamSegmenter:
    def __init__(self, device, threshold, checkpoint=None):
        import torch

        self.torch = torch
        self.device_type = torch.device(device).type
        # Prefer the source checkout used by this project's original pipeline.
        source = Path(__file__).resolve().parents[1] / "sam3"
        if source.is_dir():
            sys.path.insert(0, str(source))
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self.processor = Sam3Processor(
            build_sam3_image_model(device=device, checkpoint_path=checkpoint),
            device=device, confidence_threshold=threshold,
        )

    def set_image(self, image):
        with self._autocast():
            self.state = self.processor.set_image(image)

    def _autocast(self):
        # SAM3's CUDA backbone requires the mixed-precision inference context.
        return self.torch.autocast(
            device_type=self.device_type, dtype=self.torch.bfloat16,
            enabled=self.device_type == "cuda",
        )

    def masks(self, prompt):
        # Retain the image embedding, remove only the previous prompt/results.
        self.processor.reset_all_prompts(self.state)
        with self._autocast():
            result = self.processor.set_text_prompt(prompt=prompt, state=self.state)
        return result["masks"].detach().cpu().numpy()


def union_masks(masks, shape):
    masks = np.asarray(masks, dtype=bool)
    # SAM image inference returns N x 1 x H x W, including N=0.
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None]
    if masks.ndim != 3 or masks.shape[1:] != shape:
        raise ValueError(f"SAM masks {masks.shape} do not match RGB/depth shape {shape}")
    return masks.any(axis=0)


def segment(args, segmenter=None):
    if args.output_dir.resolve() == args.lingbot_dir.resolve():
        raise ValueError("Mask output must be separate from LingBot predictions")
    from sam.queries import query_groups, segment_groups
    from vlm.prompts import validate_target
    proposal = json.loads(args.prompts.read_text()) if getattr(args, 'prompts', None) else []
    saved_target = proposal.get('target', 'non_traversable') if isinstance(proposal, dict) else 'non_traversable'
    target = validate_target(getattr(args, 'target', None) or saved_target)
    if isinstance(proposal, dict) and 'target' in proposal and target != saved_target:
        raise ValueError('Saved VLM proposal target differs from the requested target')
    prompts = load_prompts(args.prompts) if getattr(args, 'prompts', None) else []
    extra = getattr(args, 'extra_traversability_prompts', []) or []
    manual = getattr(args, 'manual_non_traversability_prompts', []) or []
    groups = query_groups(prompts, target, extra, manual)
    enabled = []
    if getattr(args, 'prompts', None):
        enabled.append('vlm_traversability_mask' if target == 'traversable' else 'vlm_non_traversability_mask')
    if extra:
        enabled.append('sam_traversability_mask')
    if manual:
        enabled.append('manual_non_traversability_mask')
    paths = frames(args.lingbot_dir, args.frame_step, args.max_frames)
    # Empty proposals produce valid all-false masks without loading SAM weights.
    if any(groups.values()) and segmenter is None:
        segmenter = SamSegmenter(args.device, args.sam_threshold, args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # A failed rerun must not leave the previous manifest pointing at mixed masks.
    (args.output_dir / "manifest.json").unlink(missing_ok=True)
    for path in paths:
        rgb = read_rgb(path)
        masks = segment_groups(segmenter, rgb, groups)
        mask = masks['vlm_non_traversability_mask'] | masks['manual_non_traversability_mask']
        np.savez_compressed(
            args.output_dir / path.name, out_binary_masks=mask,
            rgb_sha256=np.asarray(rgb_digest(rgb)),
            **masks,
        )
        print(f"[SAM3] {path.name}: {int(mask.sum())} obstacle pixels", flush=True)
    # Project only the explicit list from this completed run, never stale files.
    write_json(args.output_dir / "manifest.json", {
        "schema_version": 1, "lingbot_dir": str(args.lingbot_dir.resolve()),
        "prompts": prompts, "frames": [path.name for path in paths],
        "sam_threshold": args.sam_threshold, "target": target, "query_groups": groups,
        "enabled_channels": enabled,
        "alignment": "SAM3 on saved LingBot RGB; native output resolution",
    })
