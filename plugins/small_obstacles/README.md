# Small-obstacle mapping plugin

Optional **VLM → multi-prompt SAM3 → binary union mask → LingBot 3D map**.
The existing LingBot, SAM3 video script, and single-prompt orchestrator are
unchanged. This plugin adds semantic markings only; it does not implement path
planning or certify that a surface is traversable.

The [detailed integration report](../../docs/lingbot_obstacle_plugin_integration.md)
explains the saved prediction contract, camera-to-world geometry, scene export,
deployment commands and evaluation limits.

## Inputs and output

Use a LingBot prediction directory containing `frame_*.npz`, saved using the
existing `--save_predictions` workflow. Each prediction needs `images`, `depth`,
`depth_conf`, `intrinsic`, and `extrinsic`.

1. A small Qwen3-VL model examines sampled saved RGB frames, first describes the
   visible ground, and proposes short concrete hazard names for SAM3. It covers
   indoor and outdoor scenes, including loose objects, water bodies, and open
   holes or ditches. Ordinary pavement, grass, scattered leaves, furniture, and
   maintained gravel paths are excluded. Prompts discovered on later frames are
   included. Raw responses and source frames are saved in `prompts.json`.
2. SAM3 image inference applies **every prompt to every selected prediction**,
   computes image features once per frame, and unions all returned instance masks.
   Empty proposals or detections produce all-false masks.
3. Projection colors matching valid world points red and leaves other points in
   their original RGB colors. It calls the same official LingBot unprojection
   helper as the existing orchestrator; there is no additional pose transform,
   voxel fusion, or modification of the original predictions.

The result is `obstacle_map.ply`, a complete point cloud for the selected frames
with obstacle pixels marked in a single color. The binary masks are also retained
as `masks/frame_*.npz` under `out_binary_masks`. `masks/manifest.json` records the
prompts and frames belonging to that run. RGB hashes prevent projection of masks
onto different source images. Use a separate output directory for concurrent runs.

## Environments

Run commands from the repository root. Separate stages allow you to retain the
project's existing model environments.

- **VLM:** install `requirements-vlm.txt` in an environment with matching
  PyTorch and torchvision builds. The default model is
  [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct).
  Model weights are downloaded on first use; `--model` also accepts a local path.
- **SAM:** use the existing `sam3` environment, source checkout, and weights.
  `--checkpoint /path/to/sam3.pt` avoids the default Hugging Face weight download.
  Access to the gated SAM3 model is needed if downloading it.
- **Projection:** use the existing orchestrator environment, which already has
  NumPy, PyTorch, and the LingBot source checkout. No VLM or SAM weights are loaded.

```bash
pip install -r plugins/small_obstacles/requirements-vlm.txt
```

If user-site packages shadow your environment's CUDA PyTorch with a CPU build,
run with `PYTHONNOUSERSITE=1`. This was necessary in the workspace's base Python
environment during validation.

## Run the stages

In the VLM environment:

```bash
python -m plugins.small_obstacles propose \
  --lingbot-dir data/lingbot_output/IMG_5522 \
  --output data/small_obstacles/IMG_5522/prompts.json \
  --keyframe-step 30
```

Use `--ground-crop` to additionally inspect a lower-half crop, or `--robot` to
provide the robot description. The crop is only a discovery aid: SAM runs on the
full saved RGB frame. Inspect or edit `prompts.json` to review discovered concepts.
You can also supply a manual JSON list, e.g. `["electrical cable", "shoe"]`.

The VLM returns `{"ground": "a factual description", "prompts": ["fallen log"]}`;
only `prompts` are passed to SAM3. The ground description remains in the saved raw
response for review. Model choice and prompt tuning were evaluated on all 18
supplied videos; see [the evaluation report](../../data/small_obstacles/evaluation/summary.md).
The latest prompt excludes installed steps, flush closed covers and ordinary
grass/gravel, and requests short conventional object names. Its paired rerun
and SAM mask review are in [the improved-prompt report](../../data/small_obstacles/evaluation/improved_4b/analysis.md),
with all proposal-mask previews in [one folder](../../data/small_obstacles/evaluation/improved_proposal_masks/index.html).
Use `--model Qwen/Qwen3-VL-2B-Instruct` if desired, but its tested prompts frequently
produced generic labels or marked normal surfaces.

In the SAM3 environment:

```bash
python -m plugins.small_obstacles segment \
  --lingbot-dir data/lingbot_output/IMG_5522 \
  --prompts data/small_obstacles/IMG_5522/prompts.json \
  --output-dir data/small_obstacles/IMG_5522/masks \
  --frame-step 2
```

Use `--sam-threshold` to adjust the instance confidence cutoff (default 0.5).
Set `--frame-step 1` to segment every saved frame. Both discovery and segmentation
support `--max-frames` for small smoke runs; the two limits are independent.

In the orchestrator environment:

```bash
python -m plugins.small_obstacles project \
  --lingbot-dir data/lingbot_output/IMG_5522 \
  --masks-dir data/small_obstacles/IMG_5522/masks \
  --output data/small_obstacles/IMG_5522/obstacle_map.ply

python orchestrator/view_ply.py data/small_obstacles/IMG_5522/obstacle_map.ply
```

Projection defaults match the existing workflow: pixel step 1, depth confidence
above 2.5, depth below 25. Disable either filter with `none`. `--color 255,0,0`
controls the shared obstacle color.

## Run all three automatically

Set interpreter paths to your model environments, then use the wrapper. Each
stage runs in its own process, releasing model/GPU memory before the next stage.

```bash
VLM_PYTHON=/path/to/vlm-env/bin/python \
SAM_PYTHON=/path/to/sam3-env/bin/python \
PROJECT_PYTHON=/path/to/orchestrator-env/bin/python \
bash plugins/small_obstacles/run_pipeline.sh \
  data/lingbot_output/IMG_5522 data/small_obstacles/IMG_5522
```

If all dependencies are installed in the active environment, omit the interpreter
variables. Optional wrapper variables: `VLM_MODEL`, `KEYFRAME_STEP`, `FRAME_STEP`.

## Accuracy limits

Masks use the **RGB already saved with LingBot's depth**. SAM returns masks at
that same resolution, so neither original-video frame numbering nor an additional
resize/crop transform is needed. This avoids alignment errors, but objects lost
in LingBot's input downsampling cannot be recovered by the plugin. Ground crops
do not restore lost source detail.

The VLM proposes object classes and SAM segments matching instances anywhere in
the frame. For example, a `shoe` prompt can also mark a shoe on a shelf. Ground
contact filtering and metric hazard verification are outside this implementation.
The merged mask deliberately discards individual object identities, and there
is no temporal tracking or de-duplication of world points.

## Checks

```bash
python -m unittest discover -s plugins/small_obstacles/tests -v
```

Tests exercise proposal validation, later-frame discovery, mask unions, empty
detections, source alignment, geometry filters, and projection through the actual
LingBot helper. Model-independent tests use fake VLM/SAM responses and do not
download weights.

## Evaluate VLM proposals on videos

The evaluator needs `ffmpeg` and `ffprobe` on PATH, plus the VLM environment. It
does not run SAM3 or reconstruct geometry. It processes every video under the
input directory, sampling across each full clip at the requested interval and
near its end, using LingBot-style resize/crop. Outputs include timestamped
images, uncropped context thumbnails, raw model responses, latency measurements,
per-video contact sheets, an HTML review page, and JSON results.

```bash
python -m plugins.small_obstacles.evaluate \
  --videos-dir data/videos \
  --output-dir data/small_obstacles/evaluation/my_run \
  --interval-seconds 2
```

Use `--instruction-file` to compare prompts on the same timestamps, or `--resume`
to continue an interrupted run with the same model, prompt and interval. Saved
`instruction.txt` freezes the exact prompt. Videos have no supplied ground-truth
labels; JSON validity and proposal statistics are measured, while detection
quality needs visual review. These statistics are not precision or recall.
