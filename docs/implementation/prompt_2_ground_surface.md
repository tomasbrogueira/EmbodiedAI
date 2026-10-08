# Session 2 prompt: extend the existing SAM3 + LingBot mapper

Copy the text below into a separate implementation session after session 1's
common foundation is merged.

---

Implement pipeline 2, `ground_surface`, in `C:\University\KTH\EmbodiedAI`.
**Extend and reuse the existing SAM3 + LingBot-Map mapping implementation. Do
not rebuild it.** Only pretrained inference and evaluation are in scope; no
training, fine-tuning, distillation or learned calibration.

## Required context and dependency

Read `docs/implementation/shared_contract.md` in full and use its frozen
`semantic_mapping_v1`, version 1 interfaces. Read `docs/pipeline_candidates.md`,
`docs/path_mapping.md`, root README, `src/fuse_path_into_map.py`,
`src/path_mapping/{runner,models,fusion,artifacts}.py` and their tests. Inspect
session 1's actual `src/pipeline_common/` contracts, registry and fixture harness.
Preserve other sessions' edits and applicable AGENTS rules.

The previous session implemented ordered input, LingBot reconstruction, official
depth/pose unprojection, exact processed-RGB alignment, SAM3 `Path`, geometry/SAM
fingerprints, distinct-frame weighted voxel evidence, NPZ/PLY/overlay outputs
and optional Viser. Its reported 42 model-free checks are software validation;
real-model GPU mapping remains unverified. The CLI currently hard-requires
exactly `Path`, and the result contains a single concept. Keep legacy commands,
saved artifacts and results readable; don't relabel them as new ground evidence.

The four pipelines share geometry, voxel fusion, robot costs, planner, evaluator
and entry/output points. Only this pipeline's semantic provider differs. The
other families are geometry-only, fixed-hazard SAM3 and Qwen-guided SAM3. SAM2/CLIP
is excluded. The completed KTH hazard component run is not validation of this mapper.

Session 1 must be merged before full integration. If its foundation is missing,
report that concrete dependency and prepare only your adapter/tests against the
documented contract; do not invent a private map/evaluator or claim completion.

## Ownership and required behavior

Own `src/pipelines/ground_surface.py`, its tests under
`tests/pipelines/test_ground_surface.py`, configs
`configs/pipelines/ground_surface_outdoor.json`, `ground_surface_indoor.json`,
`ground_surface_path_legacy.json`, and adapter documentation. Shared CLI,
geometry, fusion, scheduling, planning and evaluation belong to session 1.
Necessary legacy compatibility changes must be minimal and backward compatible;
do not edit upstream pinned submodules or other pipeline adapters.

Required public entry point:

```text
python src/run_pipeline.py --pipeline ground_surface --sequence <sequence.json> --config <ground-surface-config.json> --output <run-dir>
```

Export `create_adapter(config)` returning the shared
`observe(FramePacket) -> SemanticFrame` / `close()` interface. Reuse the existing
SAM3 loader/processor logic or compatible verified SAM3 adapter; expose individual
query/instance evidence rather than just the old aggregate scalar output.
Consume the exact aligned RGB grid supplied by session 1, with both file-byte
and decoded-pixel hashes. No independent resize or newest-pose substitution.

The definition is **candidate ground/floor surface segmentation → surface
evidence in LingBot voxels → common geometric/robot traversal checks**. Use
`role=candidate_surface` and concept ID `ground_surface`, while retaining the
actual original query and checkpoint/config provenance. A SAM surface mask
does not certify support, clearance or safety. Undetected ground remains
insufficient semantic evidence, not automatically an obstacle or free space.

Make one recorded surface prompt per selected frame. Support `ground` as the
outdoor starting configuration, `floor` indoors and exact `Path` as a separately
named compatibility/ablation configuration. These starting terms are proposals,
not measured improvements. Offer a development-only comparison using frozen
frames/references and the same SAM settings; freeze the chosen prompt before
test evaluation. Do not dynamically choose the best prompt using test labels,
or turn this into pipeline 3's multi-hazard vocabulary.

Keep SAM masks at the shared grid, per-instance score meaning, successful empty
queries, errors and partial failures explicit. Failed partial masks must be
diagnostic-only. Do not call raw SAM or weighted voxel scores calibrated
traversability probabilities. Retain all valid geometry observations separately
from successful concept query coverage; one pixel cloud is not many independent
frame confirmations. Projection and fusion happen in the shared package.

Preserve the legacy `Path` interface and artifact identity. Add explicit new
pipeline/config identity rather than changing the meaning of existing
`sam3_lingbot_path_v1` files. Import/replay legacy scores only through verified
hash/grid/geometry/prompt checks with honest provenance, never an implicit rename.
Do not maintain a second voxel format or planner.

## Acceptance and validation

Use the common fixture harness to exercise the actual runner and common output
layout, evaluator and comparison gates. Tests must cover ground/floor/Path config
selection and identity; original phrases and candidate_surface role; aligned
mask dimensions/hashes; overlapping instances without inflated evidence;
successful empty vs failed query; partial masks excluded from usable fusion;
legacy archive compatibility and mismatched cache rejection; and cleanup.
Demonstrate that semantic evidence cannot override a shared geometry rejection
such as a steep slope, drop-off or insufficient clearance.

Do not claim real mapping accuracy or speed from fixtures. Prepare a small KTH
replay command with saved alignment and inspectable voxel provenance. Run model
inference only on the server in an isolated environment, after current resource
checks and when that job is authorized; don't launch bulk downloads or long
evaluations as side effects. Preserve the completed hazard results and credentials.
If weights/GPU/actual robot scale/profile are unavailable, keep that limitation
explicit; no random model, CPU model substitution or synthetic success fallback.

Finish with implemented adapter/configs, passing checks, one common-layout sample
run, precise server smoke commands and remaining measured validation. A complete
hand-off means integration with the common interfaces, not just a renamed README.
