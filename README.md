# Integrated perception and planning

This branch combines `path-planning-cleanup` and `vlm_improvements` into six
components: path planner, VLM, orchestrator, SAM, LingBot-MAP and visualizer.

Read [Integrated pipelines](docs/INTEGRATED_PIPELINES.md) for standalone commands,
combined runs, feature switches, traversability targets and independent SAM queries.
See [Validation](docs/INTEGRATION_VALIDATION.md) for the checks performed.

- VLM can propose traversable or non-traversable regions.
- Extra SAM traversability queries such as `path` run independently alongside it.
- Positive evidence expands support candidates; negative evidence vetoes overlaps
  in the cost map and certified routes.
- Steepness, cost generation and route search are separately selectable.
- Model environments remain separate; analysis can reuse saved snapshots.

The existing installation and command examples below remain available through
compatibility entry points.

# Installation

Clone the repo with ```git clone --single-branch --branch full-integration-live-planning https://github.com/tomasbrogueira/EmbodiedAI.git``` 

**Note**: Use the EECS cluster with VSCode, using the terminal in VSCode.

## Foundation Models

Clone both SAM3 and Lingbot-map to the EmbodiedAI folder (main repo folder). 

```bash
cd EmbodiedAI
git clone https://github.com/facebookresearch/sam3.git
git clone https://github.com/Robbyant/lingbot-map.git
```

Follow the readme.me from each lingbot-map ([https://github.com/robbyant/lingbot-map](https://github.com/robbyant/lingbot-map)) until Step.3 (inclusive) and sam3 ([https://github.com/facebookresearch/sam3?tab=readme-ov-file#installation](https://github.com/facebookresearch/sam3?tab=readme-ov-file#installation)) until Step.3 (but do not clone the repo again).

Do not forget to copy the weights for lingbot-map from https://huggingface.co/robbyant/lingbot-map to the lingbot-map folder.

## Install required dependencies for orchestration and visualization

```bash
conda create -n orchestrator-visualizer python=3.10 pip -y
conda activate orchestrator-visualizer
pip install -r path-planning/requirements.txt
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
```

## Getting lingbot-map pointcloud

For offline recordings, path-planning/reconstruct.py initializes LingBot jointly from eight evenly spaced video views, then processes the remaining frames and restores every output to its original frame number. This reduces the observed scale drift without changing masks or planning rules. It uses future video frames and is not an online method. Rebuild the map, slopes and costs when reconstruction outputs change.

```bash
conda deactivate
conda activate lingbot-map
python3 path-planning/reconstruct.py \
    --video_path data/videos/indoor/IMG_5522.mp4 \
    --output_folder data/lingbot_output/ \
    --model_path lingbot-map/lingbot-map.pt \
    --mode streaming \
    --kv_cache_sliding_window 32 \
    --keyframe_interval 2 \
    --num_scale_frames 8 \
    --use_sdpa \
    --save_predictions --no_render --no_original_video
```

## Getting SAM3 segmentations

```bash
conda deactivate
conda activate sam3
python3 path-planning/generate_masks.py \
    --video_path data/videos/indoor/IMG_5522.mp4 \
    --output_folder data/sam_output \
    --prompt floor
```

**Note:** I had to dowgrade and install new libs:

```bash
pip install setuptools==69.5.1
pip install pycocotools
pip install psutil
pip install opencv-python
pip install matplotlib
```

## Projecting SAM3 segmentations to 3D pointcloud (orchestration)

```bash
conda deactivate
conda activate orchestrator-visualizer
python orchestrator/main.py \
    --video-path data/videos/indoor/IMG_5522.mp4 \
    --lingbot-dir data/lingbot_output/IMG_5522 \
    --sam-dir data/sam_output/IMG_5522_masks \
    --output data/combined/IMG_5522_floor_semantic_map.ply \
    --prompt floor \
    --frame-step 2 \
    --pixel-step 1 \
    --min-depth-conf 2.5 \
    --max-depth 25 \
    --patch-size 14 \
    --segment-color 255,0,0
```

## Visualizing results

```bash
python3 orchestrator/view_ply.py data/combined/IMG_5522_floor_semantic_map.ply
```

## Path planning

See [path-planning](path-planning/README.md) for original-point slopes, costs, recording routes, batch tools and the three-mode Viser viewer. Keep generated outputs outside the repository.

The [workflow scripts](orchestrator/scripts/README.md) run reconstruction, masks and map export using the existing KTH environments.

## VLM and semantic mapping

The [semantic mapping plugin](plugins/small_obstacles/README.md) supports VLM proposals and multi-query SAM masks. See the [integration plan](docs/integration_merge_plan.md) for the six-component architecture.
