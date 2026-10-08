# Portable hazard setup

The active experiment is `hazard_prompt_v1`: references from public annotations,
whole RGB plus the common policy to each VLM, discovery scoring, then identical
SAM3 text segmentation and deployment measurement. The legacy notebooks 01–04
and their results remain separate. Use `VLM_evaluation/` as the component root.

Start with an existing Python 3.12 interpreter with `venv` and `ensurepip`.
Setup creates or updates only the explicitly selected isolated environment and
installs this component editable. It never installs into the launching interpreter,
loads a model, downloads weights/data, or starts a notebook server/job. `--dry-run`
prints the exact proposed plan without creating files. The CPU modules also work
on Python 3.11; the unified GPU proposal uses Python 3.12 because its required
NumPy 1.26.4 wheels do not support Python 3.13 or newer.

```bash
cd VLM_evaluation
python3.12 scripts/setup_env.py --env-dir .venv-hazard-cpu --notebooks \
  --with-module hazard_data --with-module hazard_evaluation
python3.12 scripts/start_notebook.py --env-dir .venv-hazard-cpu
```

The requirement selectors are `hazard_data`, `hazard_inference`,
`hazard_evaluation`, `hazard_segmentation` and CPU-only `hazard_visualization`.
The latter adds Matplotlib/ipywidgets for notebook 14's saved-artifact viewer;
it can be selected without CUDA or model fragments. CPU fixtures need only the data
and evaluation fragments, plus notebook support for notebook checks. Requesting
model fragments requires explicit `--cuda --gpu-profile hazard`. That profile
includes both model fragments and uses one coherent proposed Qwen/SAM3 stack;
legacy `--cuda` retains the separate legacy proposal. Missing requested hazard
fragments fail clearly. Setup refuses to replace an existing legacy GPU profile
with the hazard profile in the same environment. Nothing chooses another SAM
model or task automatically.

Use this notebook order:

1. **00_setup_and_checks**: inspect imports, roots and disk. GPU probing is optional.
2. **10_hazard_data**: prepare synthetic fixtures or explicitly prepare real
   RELLIS/COCO references. Actual downloads are opt-in. Freeze all intended real
   inputs before creating predictions; inspect source/split/category coverage.
   Complete real preparation freezes the SAM test/warmup selection here.
3. **11_hazard_inference**: use the frozen selection, then explicitly load and
   run each 4B VLM sequentially on the same
   frozen RGB inputs. Cached/offline loading is the default. The 2B model is an
   explicitly selected fallback; model failures never select it automatically.
4. **12_hazard_evaluation**: score saved discovery records on CPU; incomplete
   predictions block comparisons. Development rows support tuning, test rows
   support final comparison.
5. **13_hazard_segmentation_and_cost**: use the already frozen selection for both
   VLM conditions, reference-present diagnostics and the fixed-policy baseline.
   Real segmentation/profiles/combined replay require separate explicit switches.
6. Rerun **12** with saved segmentation/profile inputs enabled to export the full
   report. Joint memory and combined p95 require an actual combined replay.
7. **14_hazard_visualization**: view that saved report on CPU, including original
   phrases, reference/scored/diagnostic overlays and explicitly selected profiles.
   Export stays off by default; see [visualization](hazard_visualization.md).

Real/download actions remain disabled by default. Notebook 10's small CPU fixture
is marked synthetic. The diagnostic reference-present control supplies canonical
phrases only to its controller; references and source labels never enter VLM
inference. See [protocol](hazard_protocol.md) and [interfaces](hazard_interfaces.md)
for vocabulary, absence-eligible precision and scored-pixel rules. RELLIS supplies
100 outdoor images and COCO supplies 40 selected val2017 images when available.
Phone/cable/spill/contact cases, physical safety and temporal pipeline evaluation
remain pending.

Configure `TRAVERSABILITY_REPO_ROOT` to this component (or its parent checkout)
when launching elsewhere. Data/run/cache roots use `TRAVERSABILITY_DATA_ROOT`,
`TRAVERSABILITY_RUN_ROOT`, `TRAVERSABILITY_CACHE_ROOT`, or setup's corresponding
`--data-root`, `--run-root`, `--cache-root` options. Relative roots resolve against
this component; defaults are `data/`, `runs/`, `.cache/`. Reference assets resolve
against the data root; generated SAM masks resolve against the run directory.
Hugging Face, Torch, pip, Jupyter and temporary caches stay under the cache root.

For an environment installed before component relocation, refresh only that
selected environment's editable install:

```bash
.venv-hazard-cpu/bin/python -m pip install --no-build-isolation --no-deps -e .
.venv-hazard-cpu/bin/python -m pip check
```

On Windows use the selected environment's `Scripts/python.exe`; on Linux use
`bin/python`. [Python venv](https://docs.python.org/3/library/venv.html) and
[IPython kernels](https://ipython.readthedocs.io/en/stable/install/kernel_install.html)
describe isolation and registration. Setup records installed package versions
and its environment-scoped kernel in `traversability-environment.json`.

For portable offline CPU preparation, supply an existing wheelhouse containing
packaging tools and requested fragments, adding notebook wheels if needed:

```bash
python3.12 scripts/setup_env.py --env-dir .venv-hazard-cpu \
  --with-module hazard_data --with-module hazard_evaluation --notebooks \
  --offline --wheelhouse /path/to/wheelhouse
.venv-hazard-cpu/bin/python -m pip install --no-index --find-links /path/to/wheelhouse pytest
.venv-hazard-cpu/bin/python scripts/check_notebooks.py \
  --output-root /path/to/new/notebook_checks --with-fixture
```

Pytest is an explicit test dependency of the selected isolated environment;
include its wheels for offline checks. Follow the helper-path/importlib command
in [the integration guide](hazard_integration.md) for combined suites with
duplicate legacy filenames.

The checker defaults to notebooks 00/10/11/12/13/14 and writes only executed copies
and marked fixture artifacts to a new output directory. It forces downloads,
model loading, real inference, real profiling and combined replay off; the
fixture switch enables CPU fixtures only. `--include-legacy` adds safe notebook
04. Source notebooks must remain output-free. `--execution python` runs code
cells headlessly in an existing isolated CPU environment without requiring a
registered kernel; the default `kernel` mode uses setup's registered kernel.
For notebook 14, `--visualization-run-root`, `--visualization-data-root` and
`--visualization-run-name` explicitly select existing saved inputs, independently
of the checker sandbox roots. These options never generate a fixture; widgets
and visualization exports are forced off during checks.
[Offline pip](https://pip.pypa.io/en/stable/cli/pip_install/) uses `--no-index` and
local wheels; missing requirements fail without network fallback.

The unified GPU stack below was checked against official sources on 2026-10-01,
then reconciled across hazard fragments. These are **proposed GPU versions**,
not an installed or model-tested CUDA environment.

| Dependency | Unified proposal | Official compatibility evidence |
|---|---|---|
| Python | 3.12 | [SAM3 prerequisites](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/README.md) require Python 3.12+ |
| Torch / torchvision | 2.10.0 / 0.25.0, preferred cu128 | [Matching PyTorch wheels](https://pytorch.org/get-started/previous-versions/) provide cu126/cu128/cu130 for this pair; SAM3 requires Torch 2.7+ and CUDA 12.6+ |
| NumPy / Pillow | 1.26.4 / 11.3.0 | [Pinned SAM3 requirements](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/pyproject.toml) require NumPy >=1.26,<2 |
| Transformers / Accelerate | 5.18.0 / 1.15.0 | [Transformers 5.18 requirements](https://github.com/huggingface/transformers/blob/v5.18.0/setup.py) allow Torch >=2.5 and NumPy >=1.17 |
| Hugging Face Hub | 1.31.0 | Transformers requires >=1.31,<3; [Hub 1.31 release](https://github.com/huggingface/huggingface_hub/releases/tag/v1.31.0) replaces the conflicting SAM-only 0.34.4 proposal |
| bitsandbytes | 0.50.2 | [Official hardware/build table](https://huggingface.co/docs/bitsandbytes/v0.50.2/installation) supports the proposed CUDA 12.8 / Ada path |
| setuptools | 80.9.0 | The [pinned SAM3 builder](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/sam3/model_builder.py) imports `pkg_resources`; retain a release providing it |
| SAM3 code | 2345a4ad109ac29c569da749c91d84f10dc08c40 | One pinned official image-model path for every condition |

The shared `regex==2025.10.22` pin satisfies Transformers 5.18.0's
`regex>=2025.10.22` requirement, as confirmed by installed distribution metadata
on KTH. Dependency consistency does not establish GPU or model compatibility.

The SAM3 fragment also pins `einops==0.8.1`, `psutil==7.0.0` and
`pycocotools==2.0.11` for imports reached by the official image builder:
[rotary embeddings](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/sam3/sam/rope.py),
[video predictor](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/sam3/model/sam3_video_predictor.py), and
[COCO loaders](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/sam3/train/data/coco_json_loaders.py).
Upstream base metadata omits these, so `pip check` alone cannot validate SAM3
runtime imports. OpenCV and decord imports are delayed until video operations.

Read-only local `nvidia-smi` reported RTX 3000 Ada Laptop, compute capability 8.9,
driver 596.58 and driver-supported CUDA 13.2. NVIDIA's
[compatibility table](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
supports newer-driver backward compatibility with CUDA 12.x; cu128 is therefore
an appropriate **driver prerequisite proposal** for this machine. No Torch CUDA,
NF4 kernel, SAM3 checkpoint, VLM generation or joint-residency test was run.
Driver compatibility does not establish VRAM fit. A future target host must
supply its own current driver/version and GPU-capability evidence; local evidence
must not be reused as proof about the shared server.

Preview the explicit GPU environment (remove `--dry-run` only for a deliberately
requested isolated install on the selected host):

```bash
python3.12 scripts/setup_env.py --env-dir .venv-hazard-gpu --notebooks \
  --with-module hazard_data --with-module hazard_evaluation \
  --cuda --gpu-profile hazard --torch-index-url https://download.pytorch.org/whl/cu128 \
  --driver-version 596.58 --dry-run
```

For offline GPU preparation, add `--offline --wheelhouse /path/to/wheels` and
`--sam3-source /path/to/existing/sam3`. The latter must be a clean checkout at
exactly the pinned revision; setup verifies it before installation, never changes
its Git state, and installs it editable only into the selected environment. Set
SAM settings `code_source_root` to that checkout so the adapter can verify actual
imported code provenance. An online explicitly requested GPU install without
`--sam3-source` installs the pinned official VCS revision. Optional acceleration
kernels are not installed; their actual use must be recorded and fixed across
comparisons.

SAM3 checkpoints are [access-gated by Meta/Hugging Face](https://huggingface.co/facebook/sam3).
Request access and authenticate separately before an explicitly authorized
checkpoint download. The image checkpoint is `sam3.pt` at immutable revision
`3c879f39826c281e95690f02c7821c4de09afae7`, size 3,450,062,241 bytes and SHA256
`9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`.
Use the existing HF cache or set `checkpoint_path` to a POSIX-relative path under
the chosen cache root. Cached loading verifies size/hash; missing access, files,
provenance or dependencies yields a clear unavailable state. No SAM2, SAM3.1,
other checkpoint or task fallback is permitted. Qwen snapshots also remain pinned;
notebook 11 has a separate explicit download cell and defaults to offline loading.

Tested during integration: isolated Windows Python 3.12.13, NumPy 1.26.4,
Pillow 12.3.0, pycocotools 2.0.11, pytest 9.1.1, nbformat 5.11.1,
nbclient 0.11.0, ipykernel 7.4.0 and matplotlib 3.11.2 for preserved legacy tests.
The selected environment's 49 installed packages passed dependency consistency
checks. Setup CPU tests validate fragment selection,
GPU plan rejection gates and immutable offline source identity with mocks. These
software checks do not validate the proposed Pillow 11.3.0/GPU stack, real
accuracy, latency, memory, joint fit or the 6 decimal GB VLM target.
Safe headless copies of 00/10/11/12/13 plus legacy 04 passed; a separate marked
CPU fixture check of 00/10/11/12/13 passed. Source outputs stayed cleared.

`start_notebook.py` binds to `127.0.0.1`, keeps token authentication and opens no
browser. No server connection or job was performed. Future portable server
preparation uses an approved Python 3.12 interpreter and explicit isolated env,
data, run and cache roots under `/data/tmpC/tomasbrogueira`; keep home below
30 GB. Before any INESC connection, read
`C:\Users\tomas\.ssh\INESC_SERVER_RULES.md` in full. Aquila is SSH/status only.
Compute jobs require current CPU/RAM/disk/GPU checks, tmux/screen,
`~/.local/bin/inesc-run` and at most one free GPU. Preserve remote project
references to `/home/tomasbrogueira/SERVER_RULES.md`.
