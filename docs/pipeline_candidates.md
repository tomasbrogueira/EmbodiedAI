# Pipeline status and evidence

Updated for the 8 October 2026 team meeting. The four stable architecture IDs
below are implemented in this working tree. Model execution, software checks,
and independent video/navigation validation are separate milestones.
This document replaces the 1 October pre-implementation audit while retaining
its scoped component evidence.

## Registry and current status

| ID | Definition and owner | Implemented software | Remaining validation |
| --- | --- | --- | --- |
| geometry_only | João: LingBot only | Shared geometry/cache, geometry voxels, robot-conditioned local planning and evaluator; no semantic queries | Current user-reported LingBot issue; working environment; all-video reconstruction/runtime checks |
| ground_surface | Tomás: LingBot + SAM3 exact **floor** | One frozen candidate-surface phrase, aligned instance masks, provenance, common fusion/planning/evaluation | Current setup and all-video results; independent surface/robot accuracy and runtime |
| fixed_hazards | Florian: LingBot + larger fixed SAM3 vocabulary | Original 16-hazard policy plus explicit-role vocabulary mode; per-query observations and multi-concept fusion | Larger vocabulary is an unmeasured starting proposal; all-video quality/runtime and policy refinement on development data |
| qwen_hazards | Joris, tentative: small VLM → words → SAM3 + LingBot | Local-only Qwen/SAM integration, original phrases, per-concept mapping, error/count provenance and common scheduling/fusion | Local assets/loading configuration; complete current model/video validation and deployment fit |

The exact-floor preset is
[ground_surface_floor.json](../configs/pipelines/ground_surface_floor.json).
Historical indoor **floor**, outdoor **ground**, and exact **Path** ablation
presets remain available. The meeting baseline keeps lowercase **floor** across
environments; environment-specific prompt experiments are separately named
variants.

The larger vocabulary preset is
[fixed_hazards_rich_vocabulary.json](../configs/pipelines/fixed_hazards_rich_vocabulary.json),
with its [versioned role policy](../configs/pipelines/fixed_vocabulary_surface_hazards_v1.policy.json).
The ID fixed_hazards stays compatible even when its explicit vocabulary includes
candidate surfaces. Surface phrases map to candidate-surface evidence; hazard
phrases retain hazard roles. Do not infer robot safety from either noun list.

Qwen3.5-4B and Qwen3-VL-4B-Instruct are checkpoint variants of qwen_hazards,
not extra architecture families. The current comparison includes exactly four
IDs and pretrained inference only. Existing SAM2/CLIP code stays as separate
legacy/component material.

See [pipeline configuration](pipeline_configuration.md) for precise settings,
loading defaults and local path resolution; [repository guide](repository_guide.md)
for the implemented interfaces; [team workflow](team_workflow.md) for shared
evaluation and reporting. Historical [implementation prompts](implementation/README.md)
are design context.

## What the current software checks establish

The common runner accepts prepared sequences, runs one semantic adapter, projects
its successful per-query observations into shared geometry/semantic voxels, and
writes common planning/evaluation artifacts. Geometry-only emits schema-valid
empty semantics. CPU fixtures check aligned grids/hashes, contracts, failures
and the run/evaluate/compare paths. Production adapters also have model-free
checks using the real shared records and supported fixture readers.

The current LingBot geometry adapter reconstructs a sequence batch.
The bounded latest-pending semantic scheduler exists, but paced replay is a
**cached_geometry_scheduler_replay**, not a connected live geometry stream.
Full capture-to-update speed, true evidence age/update rate, simultaneous model
residency and actual robot/controller performance remain unestablished.

The map represents observed surfaces and semantic evidence, without certified
free-space clearing or persistent object tracking. Unobserved/unqueried/failed
states remain explicit. Expired evidence does not establish free space.
Missing verified metric scale/up and real robot parameters block physical
planning claims. Missing independent references make accuracy unavailable;
internal returned paths are diagnostics.

The meeting reported a current LingBot problem, assigned to João because his
device has a working setup. The documentation pass did not reproduce it or
identify a precise cause. Historical success does not prove the current setup
works, and CPU fixtures do not fix or test that model issue.

## Preserved real-model evidence

### Two-clip ground-surface pilot, 2 October 2026

[The recorded KTH pilot](implementation/ground_surface.md#completed-bounded-kth-pilot)
reports a completed geometry smoke plus indoor **floor** and outdoor **ground**
saved-geometry replays on IMG_5508 and IMG_5513. These are two short development
clips, not the full 18-video evaluation. It documents real pinned model loading
and aligned saved-cache semantics.

That pilot provides no controlled Path-prompt comparison, independent video
accuracy, verified physical scale/up/robot profile, live end-to-end throughput,
joint model residency, or robot navigation validation. Consult the original
note and retained audit artifacts for counts and timing boundaries.
Its environment differs from the earlier proposed GPU setup.
The [curated local pilot evidence](../reports/2026-10-02-pipeline2-pilot/README.md)
is available for review without reconnecting to KTH; omitted bulk assets remain
explicitly outside that publication.

### Hazard image-component evaluation, 1 October 2026

The prior audit recorded completed **hazard_prompt_v1** evidence at KTH on
100 RELLIS and 40 COCO images, four SAM conditions on 20 fixed test images, and
eight deployment profiles. Keep those component reports under their original
protocol. The values below reproduce the recorded audit; this documentation
pass did not reconnect to KTH or rerun the benchmark.

| Qwen + SAM3 | Successful measured calls | Median successful time/image | p95 successful time/image |
| --- | --- | --- | --- |
| Qwen3.5-4B + SAM3 | 40/40 | 1.343 s | 2.463 s |
| Qwen3-VL-4B-Instruct + SAM3 | 38/40 | 1.898 s | 2.674 s |

Those were warm saved-image adapter timings on the historical H100 MIG
allocation with roughly 20 GB GPU memory. Loading, LingBot, voxel fusion, video
scheduling and planning were excluded. They do not establish whole-pipeline
real-time operation. The recorded 140-image inference run included 13 Qwen3-VL
response failures and zero Qwen3.5 failures; Qwen3-VL-guided SAM recorded one
failed frame. Preserve failures rather than reporting successful latency alone.

Canonical server paths from that audit, relative to its original
/home/jovyan/EmbodiedAI checkout:

- VLM_evaluation/runs/hazard_prompt_v1/evaluation/hazard_report.json
- VLM_evaluation/runs/hazard_prompt_v1/benchmark/&lt;condition&gt;/&lt;profile&gt;/summary.json

See the [component protocol](../VLM_evaluation/docs/hazard_protocol.md) and
[profiling definitions](../VLM_evaluation/docs/hazard_benchmark.md).
Original authenticated [results notebook](https://gpu1.eecs.kth.se/user/tombro/lab/tree/EmbodiedAI/kth_results.ipynb)
access is separate from a local clone.

## Current comparison work

All owners evaluate the same 18 recordings using the common prepared sequences.
Report runtime with its actual scope, advantages and limitations, plus traceable
good/bad examples. Agree the development/test split and any extra common metrics
before using test outcomes. Compare only matched frozen experiment settings;
there is no automatic winner.

Compact reviewed results can be published incrementally with source run/config
identity and bulk artifact references. Existing server jobs and original results
remain untouched. [Team workflow](team_workflow.md) provides the report template
and the Sunday 11 October handoff.
