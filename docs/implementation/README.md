# Historical implementation specifications and notes

**The four prompt files and shared_contract.md are historical handoff
specifications.** They describe the original implementation assignments.
Their lists of missing features and instructions to launch/merge sessions are
not a current repository status report. Do not rerun those assignments without
first checking the implemented source.

Use the [top-level README](../../README.md),
[current pipeline status](../pipeline_candidates.md),
[pipeline configuration](../pipeline_configuration.md),
[run storage](../run_storage.md) and
[team workflow](../team_workflow.md) for current collaboration and commands.

## Original handoffs

| Historical session | Specification | Current implementation |
| --- | --- | --- |
| 1 | [Geometry only and shared foundation](prompt_1_geometry_only.md) | [pipeline_common](../../src/pipeline_common/), [geometry_only](../../src/pipelines/geometry_only.py) |
| 2 | [Ground-surface mapping](prompt_2_ground_surface.md) | [ground_surface](../../src/pipelines/ground_surface.py), retaining old Path tooling |
| 3 | [Fixed hazard vocabulary](prompt_3_fixed_hazards.md) | [fixed_hazards](../../src/pipelines/fixed_hazards.py) |
| 4 | [Qwen-guided hazards](prompt_4_qwen_hazards.md) | [qwen_hazards](../../src/pipelines/qwen_hazards.py), two checkpoint variants |

The [shared specification](shared_contract.md) records intended contract and
evaluation design. Current public fields and validation live in
[contracts.py](../../src/pipeline_common/contracts.py); actual run/config/metric
behavior lives in the common source and maintained guides above. Some planned
capabilities remain unavailable, notably live streamed geometry/runtime claims.

## Scoped implementation and experiment notes

- [Ground-surface implementation and two-clip KTH pilot](ground_surface.md):
  adapter details, historical 2 October evidence and recorded commands.
- [Fixed-hazard implementation](fixed_hazards.md):
  original fixed-policy integration; use the configuration guide for the newer
  explicit-role larger vocabulary.
- [Geometry runtime notes](geometry_only_runtime.md):
  earlier implementation/validation scope.
- [Navigation replay demonstration](navigation_replay_demo.md):
  research illustration and replay limits, not independently validated robot navigation.

Recorded output paths, counts, environments and pilot commands are historical
evidence. Private server helper paths may not exist in a public clone. Preserve
those notes and reports without interpreting them as commands to connect to a
server or as validation of every current pipeline on all project videos.

The earlier [Path mapper setup](../path_mapping.md) remains useful for old
stage archives and proposed GPU environment background. Its pre-implementation
status statements do not describe the newer common four-pipeline runner.

All current comparison work uses pretrained inference only: no training,
fine-tuning, distillation or learned calibration. SAM2/CLIP work remains in
VLM_evaluation as separate component/research material.
