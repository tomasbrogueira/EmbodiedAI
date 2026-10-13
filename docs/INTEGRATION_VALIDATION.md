# Integration validation

Implemented on `integration/path-planning-vlm`, using planning commit `708053a0`
and VLM commit `7f9a3a58`. The original working checkout and its modified files
were preserved. The implementation checkout is `/home/jovyan/EmbodiedAI-integration`.

## Automated checks

- 158 planner tests passed, including the real-output frame-flag regression,
  with the rebuilt native library enabled.
- 44 original VLM/SAM plugin tests passed through compatibility entry points.
- 19 integration tests passed: 221 total.
- Python compilation and Git whitespace checks passed.
- Live Viser service started without loading model weights and returned HTTP 200.
  The temporary service was stopped after verification.

The integration suite covers all four steepness/planner combinations, direct
cost-map use of extra path masks, negative-overlap precedence, blocked-nearest
endpoints, Python/native parity with slope checks disabled, shared SAM embeddings
with separate source masks, VLM targets, fixed extra prompts with VLM off/empty/
expired, artifact reuse, stale/tampered outputs, missing reference, and model-free
numerical/viewer imports. Fixtures use synthetic geometry and fake model responses.

## Existing recording smoke check

Five existing IMG_5522 prediction files and support masks were projected with
frame step 2 and pixel step 2. All 174,305 selected XYZ rows, support labels and
frame IDs matched the original evidence loader exactly. Automatic full-resolution
reference estimation and steepness completed successfully. Costs reported
`missing_clearance_calibration`, with no finite costs and no route claim; no
camera-height fallback was introduced.

Smoke artifacts and the JSON result are under
`/home/jovyan/EmbodiedAI-integration-validation/`. The check ran on saved outputs;
no reconstruction, Qwen or SAM inference was performed.

## Limits of this validation

Real GPU model stages were subsequently tested on the 20 GB allocation; see
[GPU validation](GPU_VALIDATION.md) for results, the adapter fix and the distinction
between the default calibration outcome and diagnostic routing. The initial CPU
checks used the earlier approximately 10 GiB partition. Offline model stages
execute sequentially and live sessions coordinate allocation through a shared
lock. Three-model persistent-session capacity was not stress tested.

Continuous incremental replanning is not implemented; live analysis/search runs
on the finalized immutable snapshot. Existing source-row, slope-aware clearance
and endpoint rules remain the default policy. Slope-disabled results are explicitly
identified and certified against the alternate policy.
