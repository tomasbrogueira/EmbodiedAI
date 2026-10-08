# Session 1 prompt: geometry baseline and common foundation

Copy the text below into the first implementation session in this repository.

---

Implement pipeline 1, `geometry_only`, and the shared foundation used by all four
pipelines in `C:\University\KTH\EmbodiedAI` (server checkout:
`/home/jovyan/EmbodiedAI`). This is implementation work, not just a proposal.
Only pretrained inference and evaluation are in scope. Do not train, fine-tune,
distill, learn calibration models or update pretrained weights.

## Context to read before editing

Read `docs/implementation/shared_contract.md` in full. It is the authoritative
`semantic_mapping_v1`, version 1 specification for entry points, typed interfaces,
artifact layout, fusion, planning and fair evaluation. Also read
`docs/pipeline_candidates.md`, `docs/path_mapping.md`, root `README.md`,
`src/path_mapping/{runner,models,fusion,artifacts}.py`, and relevant existing tests.
Read applicable AGENTS instructions. Preserve other sessions' local changes.

There are exactly four architectures: geometry only; SAM3 ground-surface mapping;
fixed-hazard SAM3; and Qwen-guided SAM3 hazards. The two Qwen checkpoints are
variants of the fourth architecture. Do not implement the earlier optional
SAM2/CLIP pipeline. Existing legacy code should remain intact.

A previous GPT session **already implemented** `src/fuse_path_into_map.py` and
`src/path_mapping/`: ordered RGB/video input, LingBot reconstruction, exact aligned
SAM3 `Path` segmentation, stage fingerprints, confidence-weighted single-concept
voxel fusion, exports and an optional viewer. Its preparation reports 42 model-free
tests; its saved fixture executed neither real model. Reuse that geometry/alignment
work; do not rebuild the upstream models or claim real GPU mapping validation.
The new mapping files may still be untracked: inspect the actual working tree
and preserve them; do not assume a fresh clone contains them.

KTH's separate `hazard_prompt_v1` run completed on 1 October 2026, using 100
RELLIS + 40 COCO images, four SAM conditions and eight component profiles. It
validated image components, not continuous mapping, planning or joint real-time
operation. Keep that protocol and its results unchanged.

## Your ownership and deliverables

Own `src/pipeline_common/`, `src/pipelines/__init__.py`,
`src/pipelines/geometry_only.py`, `src/prepare_sequence.py`,
`src/run_pipeline.py`, `src/evaluate_pipeline.py`, `src/compare_pipelines.py`,
common fixture/experiment configs, `configs/pipelines/geometry_only.json`,
`tests/pipeline_common/` and your adapter tests/documentation. Sessions 2–4 own
their adapters; publish the foundation and contract before they integrate.
Do not implement their production semantic models as part of this task.

1. Implement the four exact CLI interfaces in the shared contract. Validate
   sequence/config/schema identities; resolve effective settings once and save
   them. Preserve one disposition per input frame, atomic completion markers,
   explicit unavailable/blocked/failed states and fixture provenance. The runner
   must lazily register all four IDs and clearly report missing production
   adapters without importing/loading their models.
2. Implement typed `FramePacket`, `GeometryFrame`, `SemanticFrame`, query/instance
   records, `SemanticAdapter.observe(frame)`/`close()` and `create_adapter(config)`.
   Provide common lifecycle, serialization and validation helpers. Make the
   generic file-backed input bridge compatible with existing hazard adapters;
   keep encoded-image-file hashes distinct from decoded-RGB hashes. Do not
   fabricate a dataset identity for generic videos. Preserve legacy validation.
   Installed SAM3's common.validate_frame only accepts RELLIS/COCO: implement
   the explicit generic model-input dispatch in the shared contract, not just
   a file wrapper or an unreachable fallback. Keep benchmark/reference validation
   unchanged and test both paths. Persist aligned RGB losslessly and verify
   re-decoded pixels before models read the file.
3. Adapt the existing LingBot reconstruction and official depth/pose unprojection
   into a shared verified geometry cache. Record original and processed grids,
   crop/resize, intrinsics, W2C direction, axes/up, scale and pose revision.
   Check reprojection and bind cache reuse to actual frames, model/code/config,
   timestamps, transforms and calibration. Invalid cache identity must fail.
4. Implement the common geometry voxel map and multi-concept semantic fuser.
   Keep surface observations distinct from observed free-space and unknown.
   Geometry-only emits valid empty semantic artifacts and `not_applicable`
   semantic records, zero semantic calls, and uses no SAM/Qwen. The shared fuser
   still needs independent per-concept evidence so sessions 2–4 can use it.
   Preserve successful queries, original phrases, overlap/conflicts and
   contribution provenance. Failed partial masks and omitted concepts must not
   become negative/accepted evidence. Make duplicate replay idempotent; handle
   evidence aging and changed poses as specified rather than silently mixing them.
5. Implement a shared, reproducible 2.5D robot costmap and local grid planner.
   Check support, slope/roughness, steps/drop-offs and footprint/height clearance;
   do not treat walls/overhangs as supporting ground or missing depth as free.
   The versioned robot profile and independent metric scale/up are required for
   physical checks. Fixture dimensions are explicitly synthetic. If actual
   robot parameters are unavailable, finish the software and raw-map path but
   report planning as blocked_inputs, not a fabricated safe path.
   All pipelines use this same policy; semantics never override geometry violations.
6. Implement the single evaluator and four-pipeline comparison harness, including
   independent-reference validation, coverage-aware metrics, comparability gates,
   JSON/CSV outputs and interpretable summaries. Distinguish surface/hazard
   diagnostics from common decision/planning outcomes. Missing references mean
   null/unavailable; no semantic capability means not_applicable. Do not derive
   ground truth from SAM/Qwen or score physical navigation using our own map.
7. Implement `quality_replay` with frozen geometry/keyframes and `paced_runtime`
   scheduling hooks. The latter needs a bounded latest-pending semantic worker,
   immutable in-flight frame joins, queue-drop journals and capture-to-fusion
   age. Keep model adapters synchronous. Separate actual end-to-end timings and
   joint memory from cached/staged timings and model-loading costs. Do not add
   isolated p95s/peaks or call a requested playback rate measured FPS.

## Acceptance and validation

Provide one tiny model-free sequence, independent synthetic reference, explicit
robot fixture and fake semantic providers exercising all four IDs through the
same CLI, fusion, planner and evaluator. Fake providers must be fixture-only;
they are never an inference fallback. Include meaningful tests for transformed
pose/crop/scale/up and cache tampering; missing-depth holes vs observed drop-offs;
steep/rough surfaces, overhang/footprint clearance; semantic/geometry conflict;
multiple/overlapping concepts; duplicate replay, failed partial masks and omitted
queries; queue drops/out-of-order completion/aging; missing references; and refusal
to compare incompatible runs. Keep the original Path and hazard tests passing.

Finish executable CPU checks and an example four-pipeline fixture comparison.
Document precise commands, schemas, metrics and limitations. No bulk manual
labeling is requested. Do not start long GPU evaluations or bulk downloads as
implementation side effects. Real-model inference belongs on KTH, not this PC.
Before any separately authorized server job, check current CPU/RAM/disk/GPU
processes and allocation, and preserve the isolated environments, credentials
and completed runs. The historical allocation was ~20 GB H100 MIG / 64 GB RAM;
it is not a promise of current resources or joint fit. Do not connect to INESC
as a substitute; its separate host rules apply if explicitly used.

End with changed files, actual checks/results, exact public commands, artifact
examples and explicit remaining real-model validation/robot/reference inputs.
Provide a foundation handoff that sessions 2–4 can merge/import. Do not mark
full navigation or real-time capability validated without measured evidence.
