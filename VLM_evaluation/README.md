# VLM evaluation

The active experiment is `hazard_prompt_v1`: whole RGB image → visible avoidance
phrases from a Qwen VLM → SAM3 text segmentation. The VLM sees the frozen common
policy and RGB only. References come from public annotations and enter scoring
and the reference-present diagnostic controller.

Start with [setup](docs/setup.md), [protocol](docs/hazard_protocol.md) and
[artifact/API contracts](docs/hazard_interfaces.md). Prepare an isolated Python
3.12+ environment with all four hazard requirement fragments. CPU preparation
and checks need no models; the explicit unified GPU proposal is documented
separately from tested CPU versions. Downloads and real model actions are off
by default.

[First server run](docs/first_server_run.md): prepare the compute host, verify
checkpoint access and run a one-image GPU smoke test before the full evaluation.

Run notebooks in this order:

1. [00 setup/checks](notebooks/00_setup_and_checks.ipynb).
2. [10 data/references](notebooks/10_hazard_data.ipynb): prepare all intended
   inputs and freeze the SAM test/warmup selection before predictions.
3. [11 whole-image inference](notebooks/11_hazard_inference.ipynb): run both
   4B VLMs sequentially against the same frozen inputs and close each backend.
4. [12 discovery evaluation](notebooks/12_hazard_evaluation.ipynb).
5. [13 SAM handoff/cost](notebooks/13_hazard_segmentation_and_cost.ipynb): both
   VLM conditions, reference-present control and full 16-phrase fixed baseline.
6. Rerun 12 with segmentation/profiles enabled to export the combined report.
7. [14 saved visualization](notebooks/14_hazard_visualization.ipynb): inspect the
   report, original phrases, references, scored/diagnostic SAM masks and explicit
   deployment profiles on CPU; no producer or evaluator runs in this viewer.

The real selection is 100 RELLIS RGB/original-ID pairs and 40 COCO val2017 images
selected from instance annotations. RELLIS water/puddle share one concept; tiny
annotated hazards remain positives. COCO scores person, cup, bottle, chair and
dog. Precision uses only absence-eligible source concepts. Unscored phrases
stay visible without being called hallucinations. Errors miss positives; missing
records block comparison. Pixel scores use declared concepts and evaluable
pixels, with the all-query downstream mask saved separately.

[The integration guide](docs/hazard_integration.md) gives the complete offline
synthetic workflow, reproducible checks and artifact entry points. These fixtures
verify software contracts and carry no real accuracy, latency or memory evidence.
The KTH real-data run completed on 1 October 2026, including gated checkpoint
loading, both Qwen models, four SAM conditions and eight deployment profiles.
See [the recorded evidence and agreed pipeline recommendation](../docs/pipeline_candidates.md).
The 6 decimal GB target applies to the VLM component; LingBot integration,
semantic voxel fusion and whole-pipeline joint fit/performance still require
their own measured replay.

Module notes: [data](docs/hazard_data.md), [inference](docs/hazard_inference.md),
[evaluation](docs/hazard_evaluation.md), [SAM](docs/hazard_segmentation.md),
[profiling](docs/hazard_benchmark.md), [visualization](docs/hazard_visualization.md).
The CPU `hazard_visualization` setup selector adds Matplotlib and ipywidgets;
PNG/SVG export is explicit opt-in. Legacy region packages, notebooks 00–04
and results remain available for traceability; their artifacts cannot identify
or score as the new task. Physical safety,
floor contact, path relevance and temporal/depth/fusion/planning evaluation
remain work for the wider pipeline.

Data, results, checkpoints, caches and parent `.codex-local/` helpers are ignored
by Git. Runtime roots are portable through `TRAVERSABILITY_*_ROOT` or explicit
configuration. No server job, commit or push is part of this integration.
