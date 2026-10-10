# KTH setup fixes and verification

This records only the installation differences and verification for the isolated
KTH setup on 2026-10-10. The branch README remains unchanged and is the baseline.

## Checkout and isolation

- Server folder: `/home/jovyan/EmbodiedAI-non-vibed`.
- Remote: `https://github.com/tomasbrogueira/EmbodiedAI.git`.
- Branch: `non_vibed`.
- Commit: `ad248afb369ef5b398c86f005bb791f61429b5ca`.
- Commands ran in the KTH JupyterHub VS Code terminal.
- New environments: `.setup/envs/lingbot-map`, `.setup/envs/sam3` and
  `.setup/envs/orchestrator-visualizer`. Caches are under `.setup/cache/`.
  Existing project folders, environments, jobs and results were preserved.
- `/.setup/` is excluded locally through `.git/info/exclude`, including HF login
  files. Never commit the cache or authentication files.
- The README's later `conda activate orchestrator` is a naming mismatch. With
  user approval, this setup uses the created `orchestrator-visualizer` environment.

## Specific installation fixes

Both linked model installation guides were followed through Step 3. The user
required stopping there; the broader LingBot visualization/rendering installation
was not performed. The following individual exceptions were subsequently approved.

### SAM3

After the additions already listed in the branch README (including
`setuptools==69.5.1`), the real script failed to import `einops`. Dependency
checking also found a NumPy conflict: SAM3 declares `numpy>=1.26,<2`, while the
installed OpenCV and contourpy versions required NumPy 2. The approved repair was:

~~~bash
conda activate "$EAI_SETUP_ROOT/.setup/envs/sam3"
python -m pip install einops==0.8.2 numpy==1.26.4 opencv-python==4.11.0.86 contourpy==1.3.3
python -m pip check
~~~

The dependency check and subsequent real-model run passed. Evidence:
`.setup/logs/23-sam3-approved-repair.log` and
`.setup/logs/25-sam3-approved-retry.log`.
SAM3's gated `config.json` and `sam3.pt` were downloaded after user-completed HF
login into the isolated `HF_HOME`, snapshot
`3c879f39826c281e95690f02c7821c4de09afae7`.

### LingBot: Matplotlib and Viser

The batch command first failed before inference at `import matplotlib`. After
approved Matplotlib installation, the same clip failed at `import viser`. The
user then approved Viser and its normal required support packages:

~~~bash
conda activate "$EAI_SETUP_ROOT/.setup/envs/lingbot-map"
python -m pip install matplotlib==3.10.9
python -m pip install viser==1.1.1
python -m pip check
~~~

Both installations and dependency checks passed. These commands describe changes
already applied; there is no need to reinstall them before running the workflow.

- Matplotlib added contourpy 1.3.2, cycler 0.12.1, fonttools 4.65.0,
  kiwisolver 1.5.1, pyparsing 3.3.3, python-dateutil 2.9.0.post0 and six 1.17.0.
- Viser added certifi 2026.7.22, charset-normalizer 3.5.2, imageio 2.38.0,
  markdown-it-py 4.2.0, mdurl 0.1.2, msgspec 0.22.0, pygments 2.21.0,
  requests 2.34.2, rich 14.3.4, trimesh 5.1.1, urllib3 2.8.0,
  websockets 16.1.1 and zstandard 0.25.0.
- Installation reports: `.setup/manifests/lingbot-matplotlib-install.json` and
  `.setup/manifests/lingbot-viser-install.json`.
- Logs: `.setup/logs/29-lingbot-matplotlib-install.log` and
  `.setup/logs/33-lingbot-viser-install.log`.

No other dependencies or CUDA extensions were added after Viser.

## Script relocation, not code deletion

The branch README prescribes `mv generate_sam3_masks.py ./sam3`. That move accounts
for the 131 deleted lines shown by the parent repository's Git diff; the file now
lives in the nested SAM3 repository. It was verified byte-for-byte identical to
`HEAD:generate_sam3_masks.py`. No script logic was changed.
SHA-256: `a0f69cdc03fafeaae0c5b9bf4855b98342403fb4ee29e6ac2538172f018de7e2`.

## Versions and evidence

- LingBot source: `8fdf984a7f9caf391622ea5843a8410900d27ef2`;
  Python 3.10.22, torch 2.8.0+cu128, torchvision 0.23.0+cu128.
- SAM3 source: `0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc`;
  Python 3.12.15, torch 2.10.0+cu128, torchvision 0.25.0+cu128.
- Orchestrator: Python 3.10, torch 2.8.0+cpu, numpy 2.2.6, scipy 1.15.3,
  viser 1.1.1.
- LingBot checkpoint: `lingbot-map/lingbot-map.pt`; publisher SHA-256 matched.
- Full package manifests and checkpoint fingerprints are in `.setup/manifests/`.
  Installation and execution logs are in `.setup/logs/`.

## Exact commands for the small workflow

Start in the server's VS Code terminal. Activate only the environment needed for
each stage. The three environments were created at these paths:

~~~bash
cd /home/jovyan/EmbodiedAI-non-vibed
source /opt/conda/etc/profile.d/conda.sh
export EAI_SETUP_ROOT="$PWD"
export CONDA_ENVS_PATH="$PWD/.setup/envs"
export CONDA_PKGS_DIRS="$PWD/.setup/cache/conda"
export PIP_CACHE_DIR="$PWD/.setup/cache/pip"
export HF_HOME="$PWD/.setup/cache/huggingface"
export MPLCONFIGDIR="$PWD/.setup/cache/matplotlib"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
~~~

The test clip already exists. It was created once from the first 32 frames of the
README's video, preserving the original video and placing test outputs separately:

~~~bash
mkdir -p .setup/verification
ffmpeg -nostdin -n -v error -threads 4 -i data/videos/indoor/IMG_5522.mp4 -frames:v 32 -an -c:v libx264 -threads 4 -crf 18 .setup/verification/IMG_5522_32f.mp4
~~~

LingBot: the exact command below **produced 32 real prediction files**, but then
**exited 1 during video rendering** because `frustum_cull_ext` is missing. It is
not a fully successful batch run. Do not hide this exit status or treat
`batch_results.json` as successful. The saved predictions were usable by the
orchestrator in the next stage.

~~~bash
conda activate "$EAI_SETUP_ROOT/.setup/envs/lingbot-map"
python3 -u lingbot-map/demo_render/batch_demo.py --video_path .setup/verification/IMG_5522_32f.mp4 --output_folder .setup/verification/lingbot_output/ --model_path lingbot-map/lingbot-map.pt --mode windowed --window_size 32 --keyframe_interval 2 --num_scale_frames 2 --overlap_keyframes 8 --use_sdpa --config demo_render/config/indoor.yaml --save_predictions
~~~

SAM3: this command completed with exit 0 and produced 32 masks:

~~~bash
conda activate "$EAI_SETUP_ROOT/.setup/envs/sam3"
python3 -u sam3/generate_sam3_masks.py --video_path .setup/verification/IMG_5522_32f.mp4 --output_folder .setup/verification/sam_output --prompt floor
~~~

Orchestration: using those real model outputs, this command completed with exit 0:

~~~bash
conda activate "$EAI_SETUP_ROOT/.setup/envs/orchestrator-visualizer"
python orchestrator/main.py --video-path .setup/verification/IMG_5522_32f.mp4 --lingbot-dir .setup/verification/lingbot_output/IMG_5522_32f --sam-dir .setup/verification/sam_output/IMG_5522_32f_masks --output .setup/verification/combined/IMG_5522_32f_floor_semantic_map.ply --prompt floor --frame-step 2 --pixel-step 1 --min-depth-conf 2.5 --max-depth 25 --patch-size 14 --segment-color 255,0,0
~~~

Viewer: this started successfully and displayed the generated semantic point cloud
in the browser through KTH's port proxy. It was stopped after verification.

~~~bash
conda activate "$EAI_SETUP_ROOT/.setup/envs/orchestrator-visualizer"
python3 -u orchestrator/view_ply.py .setup/verification/combined/IMG_5522_32f_floor_semantic_map.ply
~~~

While it is running, use the VS Code forwarded port 8080 or this authenticated
KTH proxy URL:

`https://gpu1.eecs.kth.se/user/tombro/vscode/proxy/8080/?websocket=wss://gpu1.eecs.kth.se/user/tombro/vscode/proxy/8080`

## Observed results and remaining blocker

| Stage | Observed result | Evidence under .setup/logs/ |
| --- | --- | --- |
| LingBot inference | Real checkpoint loaded on the allocated H100 MIG; 32 frames, about 5.0 seconds; 32 prediction NPZs saved | `35-lingbot-viser-retry.log` |
| LingBot video renderer | Failed with `No module named 'frustum_cull_ext'`; batch exit 1 | Same log, corresponding exit file, and batch_results.json |
| SAM3 | Real GPU model run, exit 0, 32 masks; all boolean 850x478, numeric fields finite and aligned, every mask nonempty | `25-sam3-approved-retry.log`, `26-sam3-output-validation.log` |
| Orchestrator | Exit 0; 16 sampled frames; 3,732,781 points, including 1,129,698 segmented points (30.26%); 53.40 MiB PLY | `36-orchestrator-real-run.log` and corresponding exit file |
| Browser viewer | Loaded the PLY, displayed a 500,000-point sample, client connected; actual semantic point cloud visually observed | `37-viewer-real-run.log` plus browser inspection |

Output: `.setup/verification/combined/IMG_5522_32f_floor_semantic_map.ply`.
This demonstrates the real model-to-PLY-to-viewer path on a short clip. It does
not establish segmentation accuracy, geometric accuracy, long-video stability,
or success of the complete LingBot batch command. No mocks or old-project outputs
were used.

The remaining renderer blocker is a compiled CUDA extension defined in
`lingbot-map/demo_render/render_cuda_ext/setup.py`. The CUDA compiler `nvcc` was
not on PATH or at the checked standard CUDA locations. No compiler, Kaolin,
ONNX Runtime, FlashInfer, or rendering extension was installed. The run also
reported unavailable ONNX Runtime and FlashInfer, but inference completed with
`--use_sdpa`. Any further dependency installation or extension build requires
user approval; the initial Step 3 stopping point otherwise remains in force.
