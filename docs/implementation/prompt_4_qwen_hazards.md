# Session 4 prompt: Qwen-guided SAM3 hazard adapter

Copy the text below into a separate implementation session after session 1's
common foundation is merged.

---

Implement pipeline 4, `qwen_hazards`, in `C:\University\KTH\EmbodiedAI`.
Reuse the existing whole-image Qwen and SAM3 adapters and integrate them with
the common semantic voxel/evaluation contract. Only pretrained inference and
evaluation are in scope: no training, fine-tuning, distillation, learned calibration
or checkpoint updates. Do the implementation rather than just propose it.

## Required context and dependency

Read `docs/implementation/shared_contract.md` in full; obey the frozen
`semantic_mapping_v1`, version 1 interfaces, artifact layout and evaluation rules.
Read `docs/pipeline_candidates.md`,
`VLM_evaluation/docs/{hazard_protocol,hazard_inference,hazard_segmentation,hazard_benchmark}.md`,
`VLM_evaluation/configs/hazards/policy.json`,
`VLM_evaluation/src/traversability_hazard_inference/{qwen,records,configuration}.py`,
`VLM_evaluation/src/traversability_hazard_segmentation/sam3_adapter.py` and
`VLM_evaluation/src/traversability_hazard_benchmark/profiling.py`.
Inspect session 1's actual contracts, file-backed bridge, scheduler and tests.
Preserve other sessions' changes and applicable AGENTS instructions.

The four architecture IDs are geometry_only, ground_surface, fixed_hazards and
qwen_hazards. Do not add SAM2/CLIP. Qwen3.5-4B and Qwen3-VL-4B-Instruct are two
variants of this single pipeline, evaluated separately, with exactly one Qwen
loaded in a run. Do not add automatic model switching or a third Qwen fallback.

A prior session already implemented the SAM3 `Path` + LingBot mapper. Reuse
the shared geometry/alignment adapted from it; its one-label voxels are not yet
Qwen hazard semantics. The completed KTH `hazard_prompt_v1` component evaluation
used 100 RELLIS + 40 COCO images and separately profiled Qwen/SAM. Successful warm
combined medians were 1.343 s for Qwen3.5 and 1.898 s for Qwen3-VL, with 40/40
and 38/40 successful profile attempts respectively. These exclude LingBot,
voxel fusion, queueing and planning, and do not prove real-time joint operation.
Preserve saved results and failures; neither checkpoint is an automatic winner.
SAM3 access/checkpoints were prepared on KTH; never read or copy credentials
into prompts, logs or repo files.

Session 1 must be merged before integration. If its core is absent, report the
dependency and prepare only your adapter against the documented contract;
do not implement private geometry/fusion/scheduling/planning/evaluation systems.

## Ownership and implementation

Own `src/pipelines/qwen_hazards.py`,
`configs/pipelines/qwen_hazards_qwen3_5_4b.json`,
`configs/pipelines/qwen_hazards_qwen3_vl_4b.json`,
`tests/pipelines/test_qwen_hazards.py` and adapter documentation. Shared CLI,
schemas, input bridge, scheduler, map, planner and evaluator belong to session 1.
Use shared registry hooks; coordinate a common change with its owner instead
of forking it. Keep any necessary legacy extension backward compatible/tested.

Required entry point, identical across model variants:

```text
python src/run_pipeline.py --pipeline qwen_hazards --sequence <sequence.json> --config <one-qwen-variant.json> --output <run-dir>
```

Export `create_adapter(config)` with synchronous
`observe(FramePacket) -> SemanticFrame` / `close()`. Lazily load one backend via
`traversability_hazard_inference.load_backend(model_key, settings)` and SAM via
`traversability_hazard_segmentation.load_segmenter(settings)`. Compose their
`predict_image(frame, data_root)` and `segment_image(frame, prompts, data_root)`
methods directly. Do not invoke evaluation runners/controllers or depend on
annotation-backed selections/reference masks.

Use the whole aligned RGB supplied by session 1, not region crops. It is the
exact LingBot-processed grid used by the other adapters; preserve original
source/transform metadata and do not claim identical input to the earlier
whole-original-image benchmark. Qwen prompt content must receive no depth,
geometry, SAM masks, reference labels or image-specific annotation names.
The common policy inventory is permitted; it is not an image-specific label leak.
Preserve the existing backend metadata snapshot, but wrap it with
`input_space=lingbot_processed_rgb`, source/grid/transform identity and actual
Qwen resize dimensions. Its legacy phrase "original whole RGB image" describes
the adapter input, not the unprocessed camera source. Verify the persisted
lossless PNG decodes to the shared pixels before inference.

On successful discovery, pass original phrases unchanged and in order to SAM
on that same immutable frame. Normalize concepts with the frozen alias table
only for map bookkeeping, while preserving original words. Additional concrete
nouns stay explicit/unmapped, not automatically invalid or forced into a class.
Return `role=hazard` queries, individual SAM instances and scores/statuses,
model/policy provenance, actual model/query counts and per-stage timestamps.
Copy `last_diagnostics` per call; retain raw Qwen response, strict parser outcome
and actual generation tokens. Qwen supplies no confidence probability.

Qwen error/truncation/invalid JSON/unverified stop remains an upstream failure:
do not call SAM or replace it with a successful empty list/fixed-list fallback.
A successful empty list is a successful observation with zero queries, not proof
of traversability. In the mapping config, explicitly record whether empty
discovery skips SAM image encoding (recommended default) or reproduces the
legacy benchmark's empty-prompt call; record actual calls and don't reuse old
timing numbers as measurements of changed behavior. SAM partial-query errors
retain successes and diagnostic failed masks, with only successful query masks
eligible for fusion; its all-query union is not a safe semantic input.
If legacy empty-prompt image encoding is selected and fails, preserve that
failure even though Qwen discovery succeeded. Translate mixed SAM query outcomes
to shared partial status rather than copying legacy aggregate frame status.

Use existing pinned variants and reproducible settings: language-only NF4 4-bit
quantization where supported, BF16 vision, bounded visual/context/output tokens,
deterministic generation, thinking disabled, explicit device and concurrency 1.
Record resolved/actual settings rather than inventing successful loading or a
performance bound. Keep common SAM settings identical to fixed_hazards. Models
close owned resources without disturbing other components. No silent CPU,
random-weight, alternate-checkpoint or synthetic inference fallback.

For quality_replay use the same frozen geometry and semantic keyframes as
pipelines 2/3. For paced_runtime use only session 1's bounded latest-pending
worker: immutable in-flight frame/pose identity, recorded drops, capture/start/
completion/fusion timing and real evidence age. Keep observe synchronous; no
second queue in the adapter. Unqueried concepts are not confirmed absent.
The shared core handles geometry availability, pose changes, fusion, dynamic
evidence expiry and robot costs. An old mask must never use the newest pose.

## Acceptance and validation

Use injected backends for meaningful CPU tests: exactly one selected Qwen;
phrase order and unknown noun retention; successful empty vs upstream failure;
no SAM invocation on failure; raw response/diagnostic snapshot; exact immutable
frame/grid/hash joins; overlapping masks/classes; partial SAM errors; actual
call counts; cleanup and fixture identity. Test both empty-discovery encoding
policies, including a failed legacy image encode; test lossless pixel identity
and correctly wrapped preprocessing metadata. Integrate delayed/out-of-order
semantic completion with session 1's scheduler tests, proving bounded pending
work, dropped-frame records and correct captured-frame geometry/evidence age.

Demonstrate both configurations through the shared fixture CLI, common map and
evaluator, preserving water/person/log meanings and provenance. Compare against
fixed_hazards without changing shared geometry, SAM settings, robot policy,
planner, evaluation coverage or test thresholds. Keep legacy hazard tests passing.

Prepare small KTH smoke/replay commands. Model inference belongs on KTH, not the
user's PC; check current CPU/RAM/disk/GPU processes and allocation and use an
isolated environment before an authorized job. Preserve datasets/caches/credentials
and completed runs. Do not start bulk downloads or long evaluations as side
effects. Measure actual joint residency and growing-map memory for any whole-
pipeline speed/fit claim; the historical ~20 GB H100 MIG allocation is not enough
to infer it. If runtime validation is unavailable, state it explicitly.

Finish with implemented adapter/configs, actual test/integration results, exact
public/server commands, inspectable common outputs and remaining hardware/
robot/reference limitations. Don't select a winner without measured quality,
coverage, failure rates and full-pipeline cost.
