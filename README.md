# Lingbot-Map + SAM 3D Path Mapping

This project grounds a 2D SAM 3 segmentation in a 3D reconstruction. Given an
ordered image sequence or video, Lingbot-Map reconstructs the scene and SAM 3
segments pixels matching the text prompt `Path`. The positive pixels are then
assigned to the corresponding 3D points and accumulated into labelled voxels.

```text
frames/video
    -> Lingbot-Map: depth, poses, intrinsics, world points
    -> SAM 3: prompt "Path" and per-pixel mask
    -> mask pixels + 3D points
    -> voxel aggregation
    -> visual 3D path map
```

## Components

- [`src/lingbot-map/`](src/lingbot-map/) provides streaming 3D reconstruction.
- [`src/sam3-robot/`](src/sam3-robot/) provides the SAM 3 text-prompt wrapper.
- `data/videos/` contains experiment inputs.

This pipeline should use an image sequence or video. A single image can produce
a single-view depth estimate, but cannot provide a stable scene-level map.

## Setup

Both components currently require an NVIDIA GPU. Follow the component setup
instructions in [`src/lingbot-map/README.md`](src/lingbot-map/README.md) and
[`src/sam3-robot/README.md`](src/sam3-robot/README.md).

Run Lingbot-Map on ordered frames with:

```bash
cd src/lingbot-map
python demo.py \
  --model_path /path/to/lingbot-map.pt \
  --image_folder /path/to/frames \
  --export_preprocessed /path/to/processed_frames
```

Configure SAM 3 with the exact prompt `Path`. The current SAM vocabulary does
not include this concept by default, so add it to
`src/sam3-robot/sam3_terrain/concepts.py` or use a project-specific config.

## Fusion Pipeline

### 1. Keep both models pixel-aligned

Run SAM 3 on the same cropped/resized frames used by Lingbot-Map. The simplest
approach is to use Lingbot-Map's `--export_preprocessed` output. If SAM runs on
the original frames, resize the masks back with nearest-neighbour interpolation
and apply the same transform to the camera intrinsics.

For a crop `(left, top)` followed by scale `(sx, sy)`:

```text
K_processed = [[sx * fx,       0, sx * (cx - left)],
               [      0, sy * fy, sy * (cy - top)],
               [      0,        0,              1]]
```

### 2. Segment each frame

For each frame, run SAM 3 with `Path` and merge all returned instances into a
single boolean mask. Keep the SAM confidence for each positive pixel. Reject
low-confidence or invalid Lingbot points.

Lingbot-Map provides the data needed for fusion:

```text
world_points       [S, H, W, 3]  world point for each processed pixel
world_points_conf  [S, H, W]     point confidence
depth              [S, H, W, 1] depth in camera coordinates
extrinsic          [S, 3, 4]    decoded camera pose
intrinsic           [S, 3, 3]    camera matrix K
```

### 3. Project the mask into 3D

The preferred method is to use Lingbot-Map's point map directly. For every
positive pixel `(u, v)` in frame `s`:

```text
p_world = world_points[s, v, u]
```

If explicit back-projection is needed, use the depth and intrinsics:

```text
z = depth[s, v, u]
p_camera = [(u - cx) * z / fx, (v - cy) * z / fy, z, 1]
p_world = T_camera_to_world[s] @ p_camera
```

Make sure the decoded pose is camera-to-world. If it is world-to-camera, invert
it before projection. The frame index, image dimensions, and coordinate
convention must be identical across the two models.

### 4. Accumulate voxels

Choose a metric voxel size, for example `0.05` metres. Quantize each valid
positive point:

```text
voxel = floor((p_world - map_origin) / voxel_size).astype(int)
```

Accumulate observations instead of overwriting labels. A useful weighted score
is:

```text
path_weight[voxel] += sam_score * lingbot_point_confidence
total_weight[voxel] += lingbot_point_confidence
path_probability = path_weight[voxel] / max(total_weight[voxel], epsilon)
```

Mark a voxel as path when its probability and observation count pass configured
thresholds. Save voxel coordinates, path probabilities, observation counts, and
the configuration used to create the map.

## Visualization

The implementation should write an output directory such as `outputs/demo/`:

```text
overlays/        RGB frames with the SAM Path mask
path_points.npz  positive 3D points before voxelization
path_voxels.npz  voxel coordinates, scores, and counts
path_map.ply     coloured path voxels
path_map.glb     optional scene with context cloud and camera poses
summary.json     thresholds and output statistics
```

Use three views to debug the result:

1. **2D overlays:** confirm that SAM selects the intended path and that masks
   are aligned with the processed RGB frames.
2. **3D projected points:** display positive points in green over the Lingbot
   point cloud. This reveals incorrect pose inversion or frame alignment.
3. **Voxel map:** display high-confidence path voxels in green, uncertain ones
   in yellow, and the reconstructed context in gray.

Lingbot-Map already provides an interactive Viser viewer and GLB export. The
fusion viewer should retain the camera trajectory and add the path points and
voxels as separate layers. For a quick baseline visualization:

```bash
cd src/sam3-robot
python scripts/run_image.py \
  --images "data/images/*.jpg" \
  --out ../../outputs/demo/overlays
```

The final fusion command should generate the overlays and 3D artifacts, then
serve the interactive map, for example:

```bash
python src/fuse_path_into_map.py \
  --frames data/videos/indoor/demo_frames \
  --lingbot-checkpoint /path/to/lingbot-map.pt \
  --prompt Path \
  --voxel-size 0.05 \
  --output outputs/demo \
  --serve --port 8080
```

## Validation

Before accepting a map, verify that:

- the same frame and processed dimensions are used by SAM and Lingbot-Map;
- positive pixels have valid depth and point confidence;
- projected points have positive camera depth;
- reprojecting a selected 3D point returns its source pixel within about one
  pixel;
- path labels persist across neighbouring frames;
- the overlays, projected point cloud, and voxel map agree visually.