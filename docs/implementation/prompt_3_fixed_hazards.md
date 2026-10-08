# Session 3 prompt: fixed-vocabulary SAM3 hazard adapter

Copy the text below into a separate implementation session after session 1's
common foundation is merged.

---

Implement pipeline 3, `fixed_hazards`, in `C:\University\KTH\EmbodiedAI`.
Use the existing, evaluated SAM3 image adapter and connect its independent hazard
observations to the shared mapping/evaluation infrastructure. This is pretrained
inference and evaluation only: no training, fine-tuning, distillation or learned
calibration. Do the implementation, not just a design proposal.

## Required context and dependency

Read `docs/implementation/shared_contract.md` completely. It freezes
`semantic_mapping_v1`, version 1 entry points, records, output layout, fusion,
planner and evaluator. Read `docs/pipeline_candidates.md`,
`VLM_evaluation/docs/{hazard_protocol,hazard_segmentation,hazard_benchmark}.md`,
`VLM_evaluation/configs/hazards/policy.json`, and
`VLM_evaluation/src/traversability_hazard_segmentation/{sam3_adapter,controller,common}.py`.
Inspect session 1's implemented contracts, file-backed bridge and fixture harness.
Preserve other sessions' changes and all applicable AGENTS instructions.

Exactly four architectures are in scope: geometry-only; SAM3 ground surfaces;
fixed-hazard SAM3; and Qwen-guided SAM3 hazards. SAM2/CLIP is excluded. The prior
SAM3+LingBot `Path` mapper already exists in `src/path_mapping/`, with synthetic
checks; do not rebuild it or use its single scalar label as multi-concept evidence.

KTH's real `hazard_prompt_v1` evaluation completed on 1 October 2026 with 100
RELLIS and 40 COCO images, four SAM conditions and eight profiles. Fixed-policy
SAM3 was executed. These are component results, not full mapping/navigation or
joint GPU-memory evidence. Preserve those artifacts and the existing protocol.

Session 1's common foundation must be merged before integration. If it is absent,
report that dependency and prepare your adapter against the documented protocol;
do not create a private fuser, planner, evaluator or different schemas.

## Ownership and implementation

Own `src/pipelines/fixed_hazards.py`,
`configs/pipelines/fixed_hazards.json`,
`tests/pipelines/test_fixed_hazards.py` and adapter documentation. Shared entry
points/registry hooks, geometry, file-backed bridge, fusion, scheduling, planner
and evaluator are owned by session 1. Keep source changes scoped; any unavoidable
legacy adapter extension must be backward compatible and tested. Do not edit
upstream submodules, another pipeline or common schemas independently.

Required entry point:

```text
python src/run_pipeline.py --pipeline fixed_hazards --sequence <sequence.json> --config <fixed-hazards-config.json> --output <run-dir>
```

Export `create_adapter(config)` with shared `observe(FramePacket) -> SemanticFrame`
and `close()`. Lazily load through
`traversability_hazard_segmentation.load_segmenter(settings)` and call
`segmenter.segment_image(frame, prompts, data_root)` once per selected frame,
allowing the existing adapter to encode the image once and reset text state
between queries. Runtime must not call the evaluation controller/run_conditions
or require reference annotations, selection artifacts or evaluator output.

Use all 16 original canonical prompts from the frozen policy on every selected
frame, in saved order:

```text
tree, pole, water, vehicle, building, log, person, fence,
bush, barrier, mud, rubble, cup, bottle, chair, dog
```

Record policy/version/digest. Do not prefilter prompts using references, a VLM,
predicted presence or dataset labels. Keep SAM checkpoint, thresholds, mask
postprocessing and image encoding aligned with pipeline 4 for a controlled
comparison; the number of queries is an intended architecture difference.
Use shared frozen keyframes in quality_replay and the shared eligible-frame
schedule/worker in paced_runtime, never a private scheduling loop.

Use session 1's file-backed bridge for the exact LingBot-processed RGB frame,
dimensions, immutable frame/grid identity, timestamps and encoded-file hash.
Decoded-pixel hash and geometry fingerprint are separate identities. No private
resize/crop, latest-pose substitution or invented RELLIS/COCO dataset identity
for generic videos. References remain evaluator-only.

Return each query with original phrase, normalized concept/version,
`role=hazard`, individual instance masks on the processed grid, nullable raw SAM
scores/meaning, statuses/errors, actual query counts and checkpoint/settings
provenance. Keep overlaps between classes. Instance IDs are observation IDs,
not persistent objects. Failed partial masks remain diagnostic and ineligible
for fusion. The existing adapter's all-query union can include failed partial
masks: do not use it as fusion input. Successful zero detections differ from
error or unknown; none certifies safety. Detection negatives do not clear geometry.

Fusion/projection and robot traversal costs belong to the shared core. All
sixteen queried concepts must retain their own evidence; do not reduce them to
one hazard-union scalar. Unknown/invalid geometry abstains from projection while
raw semantic observations stay saved. SAM scores are not calibrated traversal
probabilities. Close only owned model resources; preserve explicit device,
BF16 capture and pinned source/checkpoint identities. Missing dependencies or
weights produce clear unavailable state, not fake empty detections/model fallback.

## Acceptance and validation

Write meaningful model-free tests with injected segmenters: exact inventory and
one segment_image call; separate overlapping concepts/instances; aligned mask
dimensions and both hashes; successful empties; image failure; partial query
failure with successful queries retained and failed masks excluded; generic input
bridge behavior; actual call counts; fixture provenance and cleanup. Demonstrate
that water/person/log remain distinguishable in the common map, including
traceability to original query/frame, and that geometry violations override
semantic traversal priors.

Run the shared CLI/fixture/evaluator against the geometry baseline and pipeline 2
when available. Keep absent references null/unavailable; don't call a sparse
static-image dataset a validated video-navigation benchmark. Preserve previous
hazard tests/results. Prepare a small KTH real-model replay command, but do not
launch bulk downloads or long evaluations as implementation side effects.
Inference belongs on the KTH server in an isolated environment with fresh
CPU/RAM/disk/GPU/allocation checks. Historical ~20 GB MIG memory is not proof of
joint fit. Preserve authentication credentials and existing completed runs.

Finish with changed files, actual CPU checks, common-layout integration evidence,
precise server validation commands and any remaining GPU/reference/robot-input
limitations. Don't claim whole-pipeline real-time operation from the completed
image component timings.
