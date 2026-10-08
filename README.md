# EmbodiedAI: four comparable mapping pipelines

This repository compares pretrained LingBot-Map reconstruction with three
semantic alternatives on the same 18 project videos. All four use the shared
RGB/geometry handoff, semantic voxel fusion, robot policy, planner and evaluator.

| Pipeline ID | Meeting assignment, 8 October 2026 | Semantic evidence |
| --- | --- | --- |
| geometry_only | João | LingBot geometry only |
| ground_surface | Tomás | SAM3 with the exact lowercase prompt **floor** |
| fixed_hazards | Florian | SAM3 with a larger fixed vocabulary; concepts keep explicit surface/hazard roles |
| qwen_hazards | Joris, tentative | A small VLM proposes words, then SAM3 segments them |

The common runner and all four adapters are implemented. CPU fixtures exercise
their shared artifact contract. **Real video results, accuracy and runtime still
need pipeline-specific validation.** The meeting reported a LingBot-Map problem;
João will address it using his working setup. Its precise cause is not established
here. Keep that blocker distinct from historical pilots and component results.

~~~text
recording -> prepared sequence -> shared LingBot geometry and processed RGB
                                      |
                              one semantic adapter
                                      |
                          geometry + per-concept voxels
                                      |
                         robot policy and local planning
                                      |
                         evaluation -> comparison report
~~~

A surface mask is candidate evidence. Geometry, scale, up direction and robot
limits are separate requirements for physical traversability claims.

## Start with a CPU fixture

Use Python 3.12 from the repository root. This needs no GPU, checkpoints,
network access during execution, or initialized model submodules.

Follow the [external CPU setup](docs/local_setup.md#choose-external-storage-and-check-the-cpu-path)
for the exact PowerShell or Linux/macOS commands. It creates a separate
environment, redirects package and temporary caches outside the checkout,
generates a fixture there and runs the repository tests.

Choose a fresh fixture directory for each invocation. The fixture creates three
analytic images, a geometry cache and independent synthetic robot references,
then calls the real run/evaluate/compare CLIs for all four IDs. Inspect
`<asset-directory>/runs/cpu-fixture-001/comparison/comparison.json` and its metrics table.
The common fixture adapters replace model inference; production adapter checks
also live in [tests/pipelines](tests/pipelines/). Fixture timings are CPU software
checks, never model performance results.

The repository test suite covers the shared mapping system.
[VLM_evaluation](VLM_evaluation/README.md) has its own setup and tests.
See [dependency scopes](requirements/README.md) for installation limitations.

## Run a recording

Prepare each recording once and share its frozen sequence manifest. The current
preparation CLI accepts a video or an ordered frame folder and a JSON config.
Video decoding additionally needs OpenCV. Use a complete copied run config with
local checkpoint paths and the settings described in
[pipeline configuration](docs/pipeline_configuration.md). A minimal preparation
config is shown in the [repository guide](docs/repository_guide.md#prepare-a-shared-sequence).

~~~bash
python src/prepare_sequence.py --input data/videos/indoor/IMG_5506.mp4 --config /path/to/prepare.json --output data/prepared/IMG_5506
python src/run_pipeline.py --pipeline ground_surface --sequence data/prepared/IMG_5506/sequence.json --config /path/to/run.json --output runs/ground_surface/IMG_5506/attempt-001
python src/evaluate_pipeline.py --run runs/ground_surface/IMG_5506/attempt-001 --output results/ground_surface/IMG_5506/attempt-001
~~~

The configuration guide defines the preparation fields and presets. Run commands
from the repository root; replace the two config paths with real JSON files.
For matched quality replays, later runs use the first run's geometry cache via
**--geometry-cache**. [Run storage](docs/run_storage.md) explains the actual
layout, optional **--output-root** for unique attempts, and cache portability.

Without independent references, evaluation still reports supported operational
diagnostics and explicit unavailable quality metrics. When references exist,
add **--reference /path/to/reference.json**. Compare at least two evaluation
directories, repeating **--evaluation**:

~~~bash
python src/compare_pipelines.py --evaluation results/geometry_only/IMG_5506/attempt-001 --evaluation results/ground_surface/IMG_5506/attempt-001 --output results/comparisons/IMG_5506-001
~~~

Comparison checks the frozen experiment identities and reports incompatibilities.
It cannot repair different sequences, calibration, robot policies or hardware
budgets. Raw maps remain useful when calibrated planning is unavailable.

## Model setup and access

Follow [local setup on a PC or Linux host](docs/local_setup.md) for a `main`
clone, an external asset directory and explicit pinned model preparation.
A normal clone includes all 18 project videos and leaves historical model
submodules uninitialized. CPU fixtures need neither model code nor checkpoints.
Keep model source checkouts, environments, weights, HF/Torch/pip caches,
temporary files and bulk runs outside the project checkout.

The guide installs the pinned official SAM3 package from an external clean
checkout; do not use `sam3-robot/setup.sh`, which clones unpinned upstream code.
[Dependency scopes](requirements/README.md) describe the proposed GPU stacks;
their pins are not a newly verified working LingBot setup. Coordinate LingBot
environment changes with João. Keep model revisions and file hashes in the
effective config; changing them creates a new experiment.

- Obtain LingBot's local checkpoint from [Robbyant/lingbot-map](https://huggingface.co/robbyant/lingbot-map).
- Obtain SAM3 access and its local checkpoint from [facebook/sam3](https://huggingface.co/facebook/sam3); access approval is separate from cloning the wrapper.
- Qwen variants use the pinned model identities in
  [configs/pipelines](configs/pipelines/) and the separate
  [hazard inference setup](VLM_evaluation/docs/hazard_inference.md).

Store weights and caches outside Git. Current semantic adapters require
explicit local-only model preparation; missing weights or dependencies are
reported, and model failures never silently switch to fixture predictions.
The legacy fixed-hazard and Qwen templates need local paths and explicit model
loading enabled; the newer larger-vocabulary preset already enables loading.
SAM checkpoint paths are relative inside a cache root, unlike LingBot's path.
Copy presets to external local configs and override absolute source/cache roots
as shown in the setup guide. See the configuration guide before running any preset.

## Where to work and report

- [Repository guide](docs/repository_guide.md): source layout, interfaces and extension points.
- [Team workflow](docs/team_workflow.md): owners, common videos, runtime scopes, report template and Sunday handoff.
- [Contributing](CONTRIBUTING.md): branches, review and local checks.
- [Publication readiness](docs/release_readiness.md): prepared changes, verification and remaining research work.
- [Pipeline status and evidence](docs/pipeline_candidates.md): implemented features and validation limits.
- [Curated pilot evidence](reports/README.md): compact historical results retained for review.
- [Historical implementation specifications](docs/implementation/README.md): retained design context.

Use the same [18 recordings](data/videos/README.md) for every pipeline.
Report runtime, advantages, limitations, and concrete good/bad examples.
The soft deadline is **Sunday 11 October 2026**; the hard deadline will be
decided at that meeting. Compact reviewed reports can be published later;
bulk runs, renderings, model files and caches remain outside Git.

For INESC work, read the local server rules before connecting and follow the
[server section of the team guide](docs/team_workflow.md#shared-server-work).
Existing server jobs and results must be preserved.
