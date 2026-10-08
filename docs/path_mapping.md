# SAM3 + LingBot-Map baseline: server run

This is the first mapping baseline: ordered RGB frames → LingBot-Map geometry
→ SAM3 with the exact prompt `Path` → semantic voxel evidence. It uses these
two models only. The Qwen hazard candidate remains separate in
[pipeline candidates](pipeline_candidates.md).

Preparation and CPU contract checks are complete locally. Real checkpoint
loading, GPU memory, runtime and reconstruction quality still need a server
smoke run. No server installation or baseline job was launched during this
preparation. The other chat reported the KTH hazard evaluation complete at
20:15 Europe/Berlin on 2026-10-01; that report is not a new server resource check.

## Source checkout

The parent repository pins LingBot-Map at
`849e690bb086103637e44b1e91878d9d43a8bf0c` and sam3-robot at
`e467ecef8efdc86eaa806d27a99f749b45e4853f`. The added `.gitmodules` restores their
source URLs. The prepared `path_baseline_server_bundle.zip` contains
`.gitmodules`, `src/path_mapping`, `src/fuse_path_into_map.py`, requirements,
tests and documentation. These additions are currently local work; pulling
`main` alone does not include them until published.

Upload that bundle to the server. Use a new sibling checkout to preserve the
evaluation checkout and its local edits. For the KTH server:

```bash
cd /home/jovyan
git clone https://github.com/tomasbrogueira/EmbodiedAI.git EmbodiedAI-path-baseline
cd EmbodiedAI-path-baseline
unzip -o /absolute/path/to/path_baseline_server_bundle.zip -d .
```

Cloning the current parent commit supplies the gitlinks; then extracting the
bundle supplies their missing configuration and the new baseline code. Use a
different new directory if the sibling checkout already exists.

From the parent checkout:

```bash
git submodule update --init src/lingbot-map src/sam3-robot
```

Use an isolated environment for this baseline. Preserve the evaluation
environment and its saved results. The KTH checkout used by the other chat is
`/home/jovyan/EmbodiedAI`; the proposed baseline checkout is
`/home/jovyan/EmbodiedAI-path-baseline`. These instructions do not connect to
either server directory.

## Proposed GPU environment

Use Python 3.12 and the CUDA 12.8 PyTorch wheels. The pinned LingBot source
supports native PyTorch SDPA; this baseline uses that backend. The optional
sky model, batch renderer, Kaolin, and FlashInfer are not part of this baseline.
LingBot's README allows newer Torch for its streaming model, while SAM3 and the
wrapper's setup require the newer stack. This combined stack has not yet been
tested with real checkpoints.

```bash
python3.12 -m venv .venv-path-map
source .venv-path-map/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/path_mapping_gpu.txt
python -m pip install -e src/lingbot-map --no-deps
python -m pip install --no-deps \
  'sam3 @ git+https://github.com/facebookresearch/sam3.git@2345a4ad109ac29c569da749c91d84f10dc08c40'
python -m pip check
```

The SAM3 package source is pinned independently of the small sam3-robot wrapper.
The adapter calls Meta's image processor directly to pass the exact `Path`
prompt and an existing local checkpoint. The wrapper's default vocabulary and
automatic checkpoint download are bypassed. Model and data downloads are
explicit preparation steps; inference itself downloads no checkpoints.

Provide the official local `lingbot-map.pt` from
[Robbyant's checkpoint repository](https://huggingface.co/robbyant/lingbot-map)
and the local official `sam3.pt` from
[Meta's gated repository](https://huggingface.co/facebook/sam3). SAM3 access
must be approved separately. Reuse the server's existing SAM3 file if available.
Do not store credentials in the repository or command examples.

## Verify the server is ready

Check that the earlier evaluation is complete, then check CPU load, RAM, disk
space and GPU processes immediately before the smoke run. Use a persistent
tmux session on the compute host. Start with two to eight adjacent frames and
one free GPU. The default selection limit is 100 frames; full-resolution maps
are retained on CPU, so longer sequences need explicit RAM/disk planning.

These are read-only checks to perform on the chosen compute server:

```bash
uptime
free -h
df -h .
nvidia-smi
```

For INESC servers, read `C:\Users\tomas\.ssh\INESC_SERVER_RULES.md` in full
before connecting. Aquila is SSH/status only; use a compute alias, a free GPU,
tmux/screen and `~/.local/bin/inesc-run`. Keep bulk inputs, environments, caches,
outputs and logs under `/data/tmpC/tomasbrogueira`. Preserve remote project
references to `/home/tomasbrogueira/SERVER_RULES.md`. The KTH paths above must
not be used as an INESC storage template.

## Small real-model smoke run

From the parent checkout in the activated baseline environment, replace each
placeholder with an existing server path:

```bash
python src/fuse_path_into_map.py \
  --frames /absolute/path/to/ordered_frames \
  --lingbot-checkpoint /absolute/path/to/lingbot-map.pt \
  --sam-checkpoint /absolute/path/to/sam3.pt \
  --max-frames 8 \
  --prompt Path \
  --voxel-size 0.05 \
  --output outputs/path_smoke
```

Use `--video /absolute/path/to/video.mp4` instead of `--frames` to extract PNG
frames inside the output directory; `--fps 2` and `--stride 1` are defaults.
Image folders are sorted by natural filename order (2 before 10), not by
filesystem timestamps. Inputs should be adjacent views of one scene. A single
frame is rejected unless `--allow-single-frame` is supplied for diagnostics.

LingBot finishes and releases its model before SAM loads. Both adapters use
the selected `--device cuda:0`. To reserve a different physical GPU, set
`CUDA_VISIBLE_DEVICES` for the process and keep `--device cuda:0` inside that
single visible device. On shared servers, select only a currently free GPU.

## Replay each stage independently

Geometry and SAM scores are separate archives. This also supports separate
inference environments if the combined stack needs adjustment:

```bash
# LingBot only.
python src/fuse_path_into_map.py --stage reconstruct \
  --frames /absolute/path/to/ordered_frames \
  --lingbot-checkpoint /absolute/path/to/lingbot-map.pt \
  --max-frames 8 --output outputs/path_geometry

# SAM only, on the saved processed RGB grid.
python src/fuse_path_into_map.py --stage segment \
  --geometry outputs/path_geometry/geometry.npz \
  --sam-checkpoint /absolute/path/to/sam3.pt \
  --output outputs/path_semantics

# CPU fusion only; change voxel thresholds without rerunning either model.
python src/fuse_path_into_map.py --stage fuse \
  --geometry outputs/path_geometry/geometry.npz \
  --scores outputs/path_semantics/sam_scores.npz \
  --voxel-size 0.05 --min-observations 2 --path-probability 0.5 \
  --output outputs/path_map
```

Output directories must be empty unless `--overwrite` is explicit. Scores bind
to the geometry arrays, poses, processed pixels and frame order through a
SHA-256 fingerprint. A changed pixel grid, pose or sequence cannot silently
reuse the previous masks. NPZ archives contain numerical arrays and JSON text;
loading disables pickle.

## Geometry and evidence rules

The processed RGB tensor comes from LingBot's own crop/resize loader. SAM sees
exactly those saved uint8 pixels, and returns masks at that same resolution.
Overlapping SAM instances use their maximum confidence per positive pixel.

The baseline uses LingBot's official depth/pose unprojection to produce the
pixel-aligned world map, with `depth_conf` as its geometry confidence. Decoded
poses are OpenCV world-to-camera. The checkpoint's optional point head is
unused. Every valid point is reprojected before fusion; a point behind the
camera or more than one pixel from its source stops the run.

Finite world points, positive depth and positive geometry confidence above
`--min-point-confidence 1.5` enter the map. Raw LingBot confidence is a weight,
not a calibrated probability. Every valid pixel contributes geometry weight;
Path-positive pixels contribute their SAM score times that weight. Non-Path
observations can therefore reduce a voxel's Path evidence. Observation counts
measure distinct frames rather than the number of dense pixels in a voxel.

`--path-probability 0.5` and `--min-observations 2` select accepted Path voxels.
The saved ratio is a weighted evidence score, not a calibrated physical safety
probability. Missing Path evidence does not establish traversability. There is
no temporal tracker, planner or safety classifier in this baseline.

Monocular reconstruction does not establish metric scale. `--voxel-size 0.05`
means reconstruction units by default. Supply a measured
`--meters-per-unit SCALE` to scale world points, depths and camera translations;
then the voxel size and map origin are in metres. The summary records the
calibration value and coordinate units.

## Saved outputs and viewer

| Output | Contents |
| --- | --- |
| `processed_frames/` | Exact aligned RGB input to SAM |
| `geometry.npz` | Processed RGB, world points/confidence, depth, W2C poses, K, sources and fingerprint |
| `sam_scores.npz` | Dense Path scores, exact prompt and matching geometry fingerprint |
| `overlays/` | RGB with green Path score opacity |
| `path_points.npz` | Positive valid world points, RGB, SAM score, geometry confidence, frame indices |
| `path_voxels.npz` | Voxel indices/centers, weighted scores, distinct-frame support, flags and weights |
| `path_map.ply` | Accepted Path voxels green; other positive-evidence voxels yellow |
| `context_cloud.ply` | Deterministically sampled gray reconstruction context |
| `camera_trajectory.npz` | W2C/C2W cameras and intrinsics in the exported coordinate units |
| `summary.json` | Completion stage, provenance, alignment, thresholds, units and counts |

Add `--serve --port 8080` to a fused run for Viser's separate context, positive
points, voxel and camera layers. It binds to `127.0.0.1` by default; use an SSH
tunnel or the server's Jupyter port proxy to view it. Serving stays active until
Ctrl-C. A GLB export is not required for this baseline.

## Model-free checks

In Python 3.12 with `requirements/path_mapping_cpu.txt` installed:

```bash
PYTHONPATH=src python -m unittest discover -s tests/path_mapping -v
python src/fuse_path_into_map.py --fixture --output outputs/path_fixture
```

The fixture checks pixel alignment, voxel statistics, saved handoffs and
exports without importing Torch or loading models. It marks all outputs as
synthetic. Its results are software evidence only, not measured reconstruction
quality or model performance.
