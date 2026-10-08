# Local setup on a PC or Linux host

Keep the project checkout for source, configuration, reviewed reports and the
[18 versioned videos](../data/videos/README.md). Keep environments, model source
checkouts, checkpoints, package caches, temporary files and bulk runs in a
separate asset directory. CPU fixtures need no model source or weights.

## Clone the project

```bash
git clone --branch main --no-recurse-submodules https://github.com/tomasbrogueira/EmbodiedAI.git
cd EmbodiedAI
```

This ordinary clone includes the project videos; their original bytes are bound
by [the dataset manifest](../data/videos/manifest.json). It leaves the historical
`src/lingbot-map` and `src/sam3-robot` submodules uninitialized. The current common
pipelines can use external model sources, so neither submodule is needed for the
CPU path or the external setup below.

## Choose external storage and check the CPU path

Use Python 3.12 and Git. Run these commands from the project root, choosing an
asset directory outside the checkout. Package installation is an explicit
preparation step and can need network access; fixture execution is offline.

PowerShell:

```powershell
$assetRoot = 'C:\EmbodiedAI-assets'
New-Item -ItemType Directory -Force -Path "$assetRoot/cache/tmp" | Out-Null
$env:HF_HOME = "$assetRoot/cache/huggingface"
$env:HF_HUB_CACHE = "$assetRoot/cache/huggingface/hub"
$env:HF_XET_CACHE = "$assetRoot/cache/huggingface/xet"
$env:HF_DATASETS_CACHE = "$assetRoot/cache/huggingface/datasets"
$env:TORCH_HOME = "$assetRoot/cache/torch"
$env:XDG_CACHE_HOME = "$assetRoot/cache"
$env:PIP_CACHE_DIR = "$assetRoot/cache/pip"
$env:TMP = "$assetRoot/cache/tmp"
$env:TEMP = "$assetRoot/cache/tmp"
$env:TRAVERSABILITY_CACHE_ROOT = "$assetRoot/cache"
$env:TRAVERSABILITY_DATA_ROOT = "$assetRoot/data"
$env:TRAVERSABILITY_RUN_ROOT = "$assetRoot/runs"
py -3.12 -m venv "$assetRoot/envs/cpu"
$cpuPython = "$assetRoot/envs/cpu/Scripts/python.exe"
& $cpuPython -m pip install -r requirements/path_mapping_cpu.txt -r requirements/test.txt
$env:PYTHONPATH = 'src'
& $cpuPython -m pipeline_common.fixture --output "$assetRoot/runs/cpu-fixture-001"
& $cpuPython -m pytest -q tests
```

Linux/macOS:

On a compute server, select your account's approved bulk storage before running
this block. Tomás's INESC location is
`/data/tmpC/tomasbrogueira/embodiedai-assets`; use that absolute value for
`asset_root` only under his account. Other teammates must choose their own owned
or approved location. Replace the first assignment below before creating caches
or environments, and follow the shared-server rules described after the example.

```bash
asset_root="$HOME/embodiedai-assets"
mkdir -p "$asset_root/cache/tmp"
export HF_HOME="$asset_root/cache/huggingface"
export HF_HUB_CACHE="$asset_root/cache/huggingface/hub"
export HF_XET_CACHE="$asset_root/cache/huggingface/xet"
export HF_DATASETS_CACHE="$asset_root/cache/huggingface/datasets"
export TORCH_HOME="$asset_root/cache/torch"
export XDG_CACHE_HOME="$asset_root/cache"
export PIP_CACHE_DIR="$asset_root/cache/pip"
export TMPDIR="$asset_root/cache/tmp" TMP="$asset_root/cache/tmp" TEMP="$asset_root/cache/tmp"
export TRAVERSABILITY_CACHE_ROOT="$asset_root/cache"
export TRAVERSABILITY_DATA_ROOT="$asset_root/data"
export TRAVERSABILITY_RUN_ROOT="$asset_root/runs"
python3.12 -m venv "$asset_root/envs/cpu"
cpu_python="$asset_root/envs/cpu/bin/python"
"$cpu_python" -m pip install -r requirements/path_mapping_cpu.txt -r requirements/test.txt
PYTHONPATH=src "$cpu_python" -m pipeline_common.fixture --output "$asset_root/runs/cpu-fixture-001"
"$cpu_python" -m pytest -q tests
```

Use a fresh fixture output directory on each invocation. Its synthetic maps and
timings check software contracts, not model quality or GPU performance.
The [dependency guide](../requirements/README.md) explains the separate scopes.

Read `C:\Users\tomas\.ssh\INESC_SERVER_RULES.md` in full before connecting to
INESC, and follow the [shared-server rules](team_workflow.md#shared-server-work).
Aquila is SSH/status only. Preserve existing environments, jobs and results;
check resources before every job, use at most one free GPU, and run workloads
through tmux/screen and `~/.local/bin/inesc-run` on a compute host.

## Optional pinned external model sources

Clone model code only when preparing real inference. These commands preserve
LF source bytes on Windows too: SAM3 verifies exact upstream source blobs.
Use new destination directories and leave the pinned checkouts clean.

PowerShell:

```powershell
git clone -c core.autocrlf=false -c core.eol=lf --no-recurse-submodules https://github.com/Robbyant/lingbot-map.git "$assetRoot/src/lingbot-map"
git -C "$assetRoot/src/lingbot-map" checkout --detach 849e690bb086103637e44b1e91878d9d43a8bf0c
git clone -c core.autocrlf=false -c core.eol=lf --no-recurse-submodules https://github.com/facebookresearch/sam3.git "$assetRoot/src/sam3"
git -C "$assetRoot/src/sam3" checkout --detach 2345a4ad109ac29c569da749c91d84f10dc08c40
```

Linux:

```bash
git clone -c core.autocrlf=false -c core.eol=lf --no-recurse-submodules https://github.com/Robbyant/lingbot-map.git "$asset_root/src/lingbot-map"
git -C "$asset_root/src/lingbot-map" checkout --detach 849e690bb086103637e44b1e91878d9d43a8bf0c
git clone -c core.autocrlf=false -c core.eol=lf --no-recurse-submodules https://github.com/facebookresearch/sam3.git "$asset_root/src/sam3"
git -C "$asset_root/src/sam3" checkout --detach 2345a4ad109ac29c569da749c91d84f10dc08c40
```

Do not run `src/sam3-robot/setup.sh`: its upstream clone is not pinned to the
official SAM3 revision required by the current adapters. They call official
SAM3 directly and require it installed in the selected model environment.

## Explicit model environment preparation

GPU dependencies remain proposals; these instructions do not establish a
working CUDA stack, memory fit or real-video accuracy. João's reported LingBot
integration problem remains pending verification on his device. Coordinate any
LingBot environment changes with him and preserve known working environments.

The [VLM setup helper](../VLM_evaluation/scripts/setup_env.py) can prepare a
separate SAM3/Qwen environment with every storage root explicit. After the
external SAM3 checkout exists, set the driver value from the target host's
current `nvidia-smi` output and review this plan. CUDA 12.8 is the documented
proposal; choose a supported wheel index for the actual target driver.

PowerShell:

```powershell
$driverVersion = 'REPLACE_WITH_TARGET_DRIVER_VERSION'
& $cpuPython VLM_evaluation/scripts/setup_env.py `
  --env-dir "$assetRoot/envs/hazard-gpu" --cache-root "$assetRoot/cache" `
  --data-root "$assetRoot/data" --run-root "$assetRoot/runs" `
  --sam3-source "$assetRoot/src/sam3" `
  --with-module hazard_data --with-module hazard_evaluation `
  --cuda --gpu-profile hazard --torch-index-url https://download.pytorch.org/whl/cu128 `
  --driver-version $driverVersion --dry-run
```

Linux:

```bash
driver_version=REPLACE_WITH_TARGET_DRIVER_VERSION
"$cpu_python" VLM_evaluation/scripts/setup_env.py \
  --env-dir "$asset_root/envs/hazard-gpu" --cache-root "$asset_root/cache" \
  --data-root "$asset_root/data" --run-root "$asset_root/runs" \
  --sam3-source "$asset_root/src/sam3" \
  --with-module hazard_data --with-module hazard_evaluation \
  --cuda --gpu-profile hazard --torch-index-url https://download.pytorch.org/whl/cu128 \
  --driver-version "$driver_version" --dry-run
```

`--dry-run` prints the plan without changing files. Removing that flag is the
explicit installation step: it installs pinned packages and the clean external
SAM3 checkout into the selected environment, without downloading checkpoints or
starting a workload. The full [VLM setup guide](../VLM_evaluation/docs/setup.md)
covers offline wheelhouses and setup validation. Its environment manifest
records the selected roots and installed packages; ordinary later shell commands
still need the external cache environment variables above.

For a combined LingBot/SAM/Qwen experiment, explicitly add LingBot's supporting
packages and external source to that model environment, then check dependencies:

```powershell
$modelPython = "$assetRoot/envs/hazard-gpu/Scripts/python.exe"
& $modelPython -m pip install -r requirements/geometry_only_gpu.txt
& $modelPython -m pip install --no-build-isolation --no-deps -e "$assetRoot/src/lingbot-map"
& $modelPython -m pip check
```

On Linux use `model_python="$asset_root/envs/hazard-gpu/bin/python"` and the
equivalent `"$model_python" -m pip` commands. For geometry alone, use a separate
environment and [geometry_only_gpu.txt](../requirements/geometry_only_gpu.txt),
installing a matching CUDA Torch/Torchvision pair explicitly first. The combined
[path_mapping_gpu.txt](../requirements/path_mapping_gpu.txt) is another proposed
supporting stack; it does not install source checkouts or all Qwen dependencies.

## Local checkpoints and experiment configurations

Obtain weights as an explicit preparation step and keep credentials outside Git.
LingBot uses the local checkpoint from
[Robbyant/lingbot-map](https://huggingface.co/robbyant/lingbot-map).
SAM3 requires separately approved access to
[facebook/sam3](https://huggingface.co/facebook/sam3): retain `sam3.pt` from the
pinned revision and verify the size/hash in the
[configuration guide](pipeline_configuration.md#prepare-real-runs). Place it at
`<asset-directory>/cache/models/sam3.pt`. Qwen needs the selected pinned snapshot
and processor in `<asset-directory>/cache/huggingface/hub`; the explicit download
function and model revisions are in
[hazard inference](../VLM_evaluation/docs/hazard_inference.md).
Importing project modules and running the current mapping presets do not
download model code or weights. Missing dependencies/assets produce explicit
errors rather than substituting synthetic predictions.

Copy the chosen [preset](../configs/pipelines/) into
`<asset-directory>/configs/`, then edit that copy. For example:

```powershell
New-Item -ItemType Directory -Force -Path "$assetRoot/configs" | Out-Null
Copy-Item configs/pipelines/ground_surface_floor.json "$assetRoot/configs/ground_surface_floor.local.json"
```

```bash
mkdir -p "$asset_root/configs"
cp configs/pipelines/ground_surface_floor.json "$asset_root/configs/ground_surface_floor.local.json"
```

Retain the full preset and override these fields:

| Field | External value |
| --- | --- |
| `geometry.source_root` | Absolute external LingBot checkout path |
| `geometry.checkpoint` | Absolute external `lingbot-map.pt` file path |
| `pipeline.sam3.code_source_root` | Absolute external official SAM3 checkout path, installed in this interpreter |
| `pipeline.sam3.cache_root` | Absolute external cache directory |
| `pipeline.sam3.checkpoint_path` | `models/sam3.pt`, POSIX-relative under `cache_root` |

For fixed/Qwen hazards, SAM settings are under `pipeline.sam_settings` instead
of `pipeline.sam3`. Qwen additionally needs
`pipeline.qwen_settings.cache_dir` set to the absolute external HF hub cache.
Keep the current model revisions and hashes, `local_files_only: true` and
`allow_downloads: false`; enable model loading explicitly where the preset leaves
it disabled. SAM absolute checkpoint paths, backslashes and parent components
are rejected. Absolute roots avoid the component's relative-root resolution.

The Qwen and historical fixed-hazard files supply semantic settings: merge their
`pipeline` into a complete common experiment config. If a copied resolved config
also contains `pipeline_config` or `pipelines.<id>`, update the selected block:
`pipelines.<id>` takes precedence over `pipeline_config`, which takes precedence
over `pipeline`. Editing a lower-precedence block alone has no effect.
See [pipeline configuration](pipeline_configuration.md) for all fields and
[run storage](run_storage.md) for external output roots and geometry replay.
Use the same explicitly selected visible CUDA device across all model settings.
Calibration, an up direction, robot limits and independent references still
need to be supplied before making physical safety or accuracy claims.
