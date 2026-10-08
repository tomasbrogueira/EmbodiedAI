# Dependency scopes

Use separate CPU, geometry and semantic-model environments. Install from the
repository root and keep environments/checkpoints/caches outside Git.

The [local setup guide](../docs/local_setup.md) recommends keeping environments,
model source clones, checkpoints, HF/Torch/pip caches and temporary files outside
the project directory. A normal clone includes the 18 dataset videos and leaves
model submodules uninitialized; no model code is required for CPU fixtures.

| File | Scope |
| --- | --- |
| path_mapping_cpu.txt | NumPy and Pillow for model-free frame preparation, saved-geometry fusion/replay and contract fixtures |
| test.txt | Full top-level CPU test dependencies, including OpenCV; includes the CPU list |
| geometry_only_gpu.txt | LingBot-only supporting packages; install a matching CUDA Torch/Torchvision stack separately |
| path_mapping_gpu.txt | Proposed combined LingBot/SAM supporting stack, including Torch/Torchvision pins |
| ../VLM_evaluation/requirements/ | Separate component benchmark, SAM and Qwen dependency groups |

Python 3.12 is the existing preparation target. The fixture needs only NumPy and
Pillow; video preparation additionally imports OpenCV. Tests do not require
loading model weights. The [top-level README](../README.md) gives a CPU setup and
fixture command; [CONTRIBUTING](../CONTRIBUTING.md) gives the checks.

The current GPU lists preserve the previously proposed model pins. A clean GPU
installation and current LingBot/SAM/Qwen checkpoint execution have not been
verified by this documentation pass. João owns the working LingBot environment
and reported LingBot issue. Record any verified environment change alongside
its source/checkpoint identities instead of guessing replacement pins.

The 8 October CPU dependency check confirmed a fresh Python 3.12 environment
with NumPy 1.26.4, Pillow 11.3.0, OpenCV 4.11.0 and pytest 9.1.1. The four-pipeline
fixture also completed in an existing Python 3.12 environment with NumPy 2.5.3
and Pillow 12.3.0. These checks establish CPU dependency/fixture behavior,
separately from GPU/model execution.

The [historical GPU setup](../docs/path_mapping.md) includes a proposed CUDA
installation order and pinned SAM source installation. It is background for
model setup, not evidence that the current blocker is solved. The combined list
does not install all Qwen packages; use
[VLM_evaluation setup](../VLM_evaluation/docs/setup.md) and
[hazard inference instructions](../VLM_evaluation/docs/hazard_inference.md) for
those dependencies and explicit local model preparation.

Requirements files do not initialize model repositories or obtain checkpoints.
Use external pinned LingBot/SAM3 checkouts and install them explicitly into the
chosen model environment. The VLM setup helper accepts `--sam3-source` plus
absolute `--env-dir`, `--cache-root`, `--data-root` and `--run-root`; preview its
GPU proposal with `--dry-run` before an explicit installation. It installs no
weights and starts no workloads. Preserve official SAM3 source bytes and point
`code_source_root` at the clean checkout actually installed in that interpreter.
Do not use the wrapper's unpinned `setup.sh` for the current common pipelines.

Follow the [server rules](../docs/team_workflow.md#shared-server-work) before
remote installs or work. Never change an existing evaluation environment merely
to test this repository.
