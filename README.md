# Installation

Clone the repo with ```git clone --single-branch --branch non_vibed https://github.com/tomasbrogueira/EmbodiedAI.git``` 

**Note**: Use the EECS cluster with VSCode, using the terminal in VSCode.

## Foundation Models

Clone both SAM3 and Lingbot-map to the EmbodiedAI folder (main repo folder). 

```bash
cd EmbodiedAI
git clone https://github.com/facebookresearch/sam3.git
git clone https://github.com/Robbyant/lingbot-map.git
mv generate_sam3_masks.py ./sam3
```

Follow the readme.me from each lingbot-map ([https://github.com/robbyant/lingbot-map](https://github.com/robbyant/lingbot-map)) until Step.3 (inclusive) and sam3 ([https://github.com/facebookresearch/sam3?tab=readme-ov-file#installation](https://github.com/facebookresearch/sam3?tab=readme-ov-file#installation)) until Step.3 (but do not clone the repo again).

Do not forget to copy the weights for lingbot-map from https://huggingface.co/robbyant/lingbot-map to the lingbot-map folder.

## Install required dependencies for orchestration and visualization

```bash
conda create -n orchestrator-visualizer python=3.10 pip -y
conda activate orchestrator-visualizer
pip install numpy scipy
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pip install viser
```

## Getting lingbot-map pointcloud

```bash
conda deactivate
conda activate lingbot-map
python3 lingbot-map/demo_render/batch_demo.py \
    --video_path data/videos/indoor/IMG_5522.mp4 \
    --output_folder data/lingbot_output/ \
    --model_path lingbot-map/lingbot-map.pt \
    --mode windowed \
    --window_size 32 \
    --keyframe_interval 2 \
    --num_scale_frames 2 \
    --overlap_keyframes 8 \
    --use_sdpa \
    --config demo_render/config/indoor.yaml \
    --save_predictions
```

## Getting SAM3 segmentations

```bash
conda deactivate
conda activate sam3
python3 sam3/generate_sam3_masks.py \
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
conda activate orchestrator
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

## TODOs

- Change the SAM3 pipeline to support multiple segmentations and update the orchestrator accordingly
- Create a single bash script to run sequentially the pipeline, instead of running it manually

