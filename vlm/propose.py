"""Use a small local VLM to discover segmentation concepts on keyframes."""

from PIL import Image

from orchestrator.shared import frames, read_rgb, write_json, rgb_digest
from vlm.prompts import parse_proposal, validate_target


DEFAULT_MODEL = "Qwen/Qwen3-VL-4B-Instruct"

INSTRUCTION = """Describe the visible ground, then identify visible hazards for
Robot: {robot}. Return only JSON with "ground" (one factual sentence) and
"prompts" (a list of short object names for SAM3).

Include an object only when it is visibly loose, detached or displaced on the
ground where the robot could travel. Also include visibly identifiable bodies
of water and actual open holes or ditches. Do not assume an object is hazardous
merely because it is visible. A detached log across a path qualifies.

Exclude normal terrain and installed infrastructure: maintained gravel paths,
grass or vegetation alone, scattered leaves, intact pavement, paving stones,
installed path slabs or steps, and closed flush drain or manhole covers.
Joints, cracks or shadows alone do not make a slab broken or a hole open.
Exclude furniture, buildings, fixed posts and objects on tables. A loose rock
or displaced fragment qualifies; the intact surface underneath it does not.

Use distinct lowercase conventional object nouns of one to three words.
Name the object itself, without color, position, size or pose: use "stone block"
rather than "tilted stone block". Do not use vague labels such as "debris",
"fragment", "gravel debris", "obstacle" or "overgrown grass". If the specific
object type or its loose/displaced status is unclear, omit it. Never invent
water or holes from stains or shadows. Do not copy examples unless visible.
If no qualifying hazard is visible, "prompts" must be []. Do not speculate.
"""


class QwenProposer:
    def __init__(self, model_name, device="auto", max_new_tokens=256):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, dtype="auto", device_map=device,
        ).eval()
        self.max_new_tokens = max_new_tokens

    def propose(self, image, instruction):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": instruction},
        ]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
            )
        return self.processor.batch_decode(
            generated[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )[0]


def propose(args, proposer=None):
    paths = frames(args.lingbot_dir, args.keyframe_step, args.max_frames)
    proposer = proposer or QwenProposer(args.model, args.device, args.max_new_tokens)
    # replace, rather than format, because the instruction contains literal JSON.
    target = validate_target(getattr(args, 'target', 'non_traversable'))
    instruction = getattr(args, 'instruction', None) or instruction_for(target, args.robot)
    prompts, observations = [], []
    for path in paths:
        rgb = read_rgb(path)
        image = Image.fromarray(rgb)
        responses = []
        views = [("full", image)]
        if args.ground_crop:
            views.append(("lower_half", image.crop((0, image.height // 2, image.width, image.height))))
        for view, crop in views:
            raw = proposer.propose(crop, instruction)
            try:
                found = parse_proposal(raw, target)
            except (ValueError, KeyError) as exc:
                raise ValueError(f"Invalid VLM JSON for {path.name} ({view}): {raw!r}") from exc
            responses.append({"view": view, "prompts": found, "raw_response": raw})
            for prompt in found:
                if prompt not in prompts:
                    prompts.append(prompt)
        observations.append({"frame": path.name, "rgb_sha256": rgb_digest(rgb), "responses": responses})
        print(f"[VLM] {path.name}: {len(prompts)} distinct concepts", flush=True)
    write_json(args.output, {
        "schema_version": 1, "model": args.model, "robot": args.robot,
        "lingbot_dir": str(args.lingbot_dir.resolve()),
        "instruction": instruction, "target": target, "prompts": prompts, "observations": observations,
    })
    print(f"Saved {len(prompts)} prompts to {args.output}", flush=True)


TRAVERSABILITY_INSTRUCTION = """Describe the visible ground, then identify observed
surfaces that Robot: {robot} could potentially travel on. Return only JSON with
\"ground\" (one factual sentence) and \"prompts\" (short conventional surface names
for SAM3). Identify visibly continuous ground surfaces such as a path or pavement
only when present. Exclude water, open holes, walls, loose objects and unsupported
surfaces. Names must describe the actual visible region, not a claim of safety.
Geometry, steepness and clearance are checked separately. Use distinct lowercase
names, no color or location qualifiers. If no suitable surface is visible, return
an empty prompts list. Do not speculate or copy examples unless visible.
"""


def instruction_for(target, robot='a small wheeled ground robot'):
    validate_target(target)
    template = TRAVERSABILITY_INSTRUCTION if target == 'traversable' else INSTRUCTION
    return template.replace('{robot}', robot)
