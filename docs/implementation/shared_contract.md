# Shared contract for the four evaluation pipelines

Contract ID: `semantic_mapping_v1`. Version: `1`. Recorded 1 October 2026.
This is the implementation specification. The common implementation now lives
in `src/pipeline_common/`; CPU conformance is distinct from real-model/runtime
validation. Use pretrained inference only. No training, fine-tuning,
distillation, learned calibration or checkpoint weight updates are in scope.
Development-only selection of declared prompts/thresholds is permitted; freeze
those choices before held-out evaluation.

## Scope and ownership

Exactly four pipeline IDs are valid:

| ID | Evidence provider |
|---|---|
| `geometry_only` | No semantic model; geometry baseline |
| `ground_surface` | SAM3 candidate ground/floor surfaces |
| `fixed_hazards` | SAM3 with the frozen 16-concept hazard inventory |
| `qwen_hazards` | One Qwen proposes image-specific phrases, then SAM3 |

The two 4B Qwens are variants of `qwen_hazards`. SAM2/CLIP is excluded from
this implementation/comparison scope; preserve its existing legacy files.

Session 1 owns `src/pipeline_common/`, the four common CLI files below,
`src/pipelines/__init__.py`, `src/pipelines/geometry_only.py`, common tests,
common experiment/robot fixture configuration and the comparison harness.
Sessions 2–4 own their adapter, its configs/tests and adapter documentation.
They import the common package. They must not fork the map, planner, evaluator,
schemas or entry points. Merge session 1 before integrating sessions 2–4.

Existing `src/path_mapping/` and `VLM_evaluation/` are reusable sources. Keep
their existing entry points, protocols and outputs working. Do not edit pinned
upstream submodules to build these adapters. If a shared interface needs revision,
change this contract and the shared owner first; never invent a private variant.

## Identical public entry points

All commands run from the repository root. These commands are required new
entry points, not commands available in the current checkout.

```text
python src/prepare_sequence.py --input <video-or-frame-folder> --config <prepare.json> --output <sequence-dir>
python src/run_pipeline.py --pipeline <ID> --sequence <sequence-dir>/sequence.json --config <run.json> --output <run-dir>
python src/evaluate_pipeline.py --run <run-dir> --reference <reference.json> --output <evaluation-dir>
python src/compare_pipelines.py --evaluation <eval-1> --evaluation <eval-2> --evaluation <eval-3> --evaluation <eval-4> --output <comparison-dir>
```

`--reference` is optional: absent references allow operational diagnostics only,
with reference-based quality metrics explicitly unavailable. The common runner
supports `--fixture` for explicitly synthetic model-free execution and
`--geometry-cache <cache-dir>` for verified reuse. Add shared options centrally.
Missing adapters/weights/robot inputs return an explicit unavailable/blocked
state; they never trigger silent model substitution, synthetic fallback or downloads.
Refuse overwriting an existing completed run; resumption, if supported, must
verify identities and preserve previous attempts.

## Input and configuration

`sequence.json` contains `contract_id`, `schema_version`, `sequence_id`,
`split` (`development` or `test`), `fixture`, ordered `frames`, input provenance
and a manifest digest. Each frame has a unique `frame_id`, nullable `timestamp_ns`,
relative `image_path`, encoded-file SHA256, decoded RGB SHA256, dimensions
and timestamp provenance. Hash recipes must be named; encoded-file bytes and
decoded RGB bytes are different identities. Validate paths, actual file bytes,
decoded pixels, ordering and monotonic known timestamps. Do not fabricate video
capture times from filenames. Frame-folder timing must be supplied or marked
synthetic/unknown, in which case real evidence-age metrics are unavailable.
Serialize unknown timestamps in numeric arrays as `-1`, documented in the
manifest, never as invented capture time. Name the source capture clock.
Runtime durations use a separate monotonic clock. Paced replay maps source
capture times onto a recorded replay monotonic origin at the declared playback
rate; age uses that mapped capture time. Never subtract historical capture dates
from today's completion or label cached quality-replay processing live evidence age.

References/annotations are outside this runtime manifest. All pipelines receive
the same sequence and shared experiment configuration. Their model/prompt
settings differ only in the explicitly declared pipeline section.

The resolved run configuration includes:

- Contract/protocol IDs, pipeline ID, checkpoint variant and input fingerprint.
- Pinned code/checkpoint identities, paths, device, precision, model preprocessing
  and generation/segmentation settings; actual loaded identities go in the run.
- Geometry settings, cache identity, pose revision, depth convention, map frame,
  axes, gravity/up direction, crop/resize transforms and metric-scale provenance.
- Voxel origin/resolution, geometry validity thresholds and common fusion policy.
- Frozen semantic keyframe IDs for quality replay, and separately runtime queue,
  detection cadence, expiry and scheduling policies.
- A versioned robot profile: footprint, height, clearance, maximum slope/step,
  support/roughness limits, unknown-space rule and semantic concept cost policy.
  Do not invent the real robot's dimensions/capabilities. Fixture values must be
  visibly synthetic. Missing physical parameters block planning, while raw mapping
  may complete with a recorded reason.
- Goal/start requests in the declared map frame and planner settings, plus a
  frozen evaluation ROI/grid for comparable coverage. Independent reference paths,
  coverage and taxonomy are evaluator-only inputs; runtime reads no annotations.

Map coordinates require an explicit up/gravity direction. Monocular LingBot
units are not metres until externally verified scale is supplied. Apply the
same scale to points, depth and camera translations. Without verified scale/up,
allow explicitly uncalibrated visualization, but block metric robot-cost,
clearance and navigation claims. No test-reference scale fitting is allowed.

## Frozen Python adapter interface

Define typed records and protocols in `src/pipeline_common/contracts.py`.
Array fields below describe logical in-memory records; serialized forms use
numeric NPZ and JSON, with no pickle/object arrays.

```python
class SemanticAdapter(Protocol):
    def observe(self, frame: FramePacket) -> SemanticFrame: ...
    def close(self) -> None: ...

def create_adapter(config: dict) -> SemanticAdapter: ...
```

Each adapter module exports `create_adapter`. Session 1's registry lazily imports
`src/pipelines/<ID>.py`; importing/testing common code must not load Torch,
models or CUDA. `observe` is synchronous. The shared scheduler can run it in
a bounded worker; adapters must not create a second scheduler or change its return type.

`FramePacket` includes sequence/frame ID, capture timestamp and provenance,
the complete aligned RGB frame (`uint8[H,W,3]`), its persisted image path,
encoded-file/decoded-pixel hashes, processed grid ID, source RGB identity,
source-to-processed transform and `GeometryFrame | None`. For the first common
implementation, all semantic adapters consume the exact LingBot-processed RGB
grid, not independent crops. Qwen sees that whole aligned frame, not region
crops. Retain the original image and preprocessing transform; cropped-out source
pixels are outside mapping coverage. Record this difference from the earlier
whole-original-image component benchmark.
Persist processed frames as lossless PNG. Re-decode and verify equality with
`FramePacket.rgb` and its decoded-pixel hash before either file-backed model
reads them; JPEG re-encoding cannot satisfy an exact-pixel handoff.

`GeometryFrame` includes aligned points `[H,W,3]`, depth `[H,W]`, validity and
confidence `[H,W]`, effective intrinsics, verified OpenCV world-to-camera pose,
explicit optical-axis/ray-distance depth kind, grid ID, geometry fingerprint,
map frame/units/up/scale and pose revision. Reuse official depth/pose unprojection
and the existing reprojection checks. Invalid or unavailable geometry remains
explicit; observations are never projected using another frame's pose.

`SemanticFrame` includes contract/version, sequence/frame/grid/RGB/geometry
identities, adapter/model/policy provenance, capture/start/completion timestamps,
`status` (`ok`, `partial`, `error`, `skipped`, `not_applicable`), error details,
query records and actual model-call/query counts.

Each query stores a stable `query_id`, original phrase, versioned `concept_id`,
`role` (`candidate_surface` or `hazard`), `status` (`ok` or `error`), error,
instances and available score metadata. Each instance stores observation ID,
boolean mask `[H,W]` on `FramePacket.processed_grid_id`, score or null and score
meaning. Source-grid references use the declared transform; adapters cannot
return source-size masks on the processed grid. Query/instance IDs
are observations, not persistent tracks. Keep original Qwen phrases unchanged
for SAM; normalized aliases are a separate versioned concept mapping. Unknown
nouns stay explicit with stable unmapped IDs, not silently forced into a class.

`geometry_only` returns `status=not_applicable`, zero model calls and no queries.
A successful query with no masks is still `ok`. Qwen's successful empty list
is `ok` with no queries. These differ from errors/skips and do not certify safety.
Frame `partial` retains successful and failed query records. Fuse only `ok`
queries; failed partial masks remain diagnostic. Recompute any display union
from successful queries; never use the old all-query union as semantic input.
SAM scores and raw geometry confidence are evidence weights, not calibrated
traversability probabilities. Qwen provides no confidence probability.
Translate SAM results explicitly: mixed successful/failed queries are `partial`;
all failed queries or image-encoding failure are `error`; all successful queries,
including zero detections, are `ok`. The legacy aggregate SAM frame status alone
does not implement this distinction. Successful empty Qwen discovery with SAM
skipped is `ok`; if the configured legacy empty-prompt encoding runs and fails,
preserve that SAM image error rather than returning success.

Existing hazard adapters hash encoded files and validate their own task/frame
records. Session 1 owns a common file-backed input bridge with both hash recipes.
Adapt a generic sequence honestly; never label a phone/video frame as RELLIS
or COCO merely to bypass legacy validation. Any necessary generic-frame support
must be backward compatible and retain all existing hazard protocol checks.
Generic runtime input support is a required session 1 deliverable: installed
SAM3 currently calls `common.validate_frame`, which accepts only RELLIS/COCO.
Its fallback when that module is missing is insufficient. Implement an explicit
model-input path discriminated by `input_contract_id=semantic_mapping_v1`,
validated by the shared file/grid/hash bridge, for both model readers. Keep
legacy benchmark `validate_frame` and reference/selection validation strict;
runtime compatibility must not make annotations/evaluators accept invented sources.

### Public Python names and explicit unavailability clarification

The frozen dataclasses use `GeometryFrame.points`, `validity`, `intrinsics`,
`world_to_camera`, `processed_grid_id`, `geometry_fingerprint`, `depth_kind`,
`map_frame`, `units`, `up`, `scale` and `pose_revision`. `SemanticFrame` uses
`adapter_provenance`, `model_calls`, `query_count`, `timestamp_ns`,
`started_monotonic_ns`, `completed_monotonic_ns`, `decoded_rgb_sha256` and
`geometry_fingerprint`. Queries/instances are `QueryRecord` and `InstanceRecord`;
the phrase field is `original_phrase`. An omitted instance grid inherits the
validated full frame grid. `query_count` may be null only with an explicit
`query_count_reason` after unauditable execution; known nonnegative integer
counts remain separate and comparison/evaluation never substitutes zero.

The bridge signatures are `to_model_input(frame) -> (record, data_root)`,
`validate_model_input(record, data_root) -> Path` and
`load_model_rgb(record, data_root) -> PIL.Image`. Model readers receive pixels
decoded from one owned, hash-verified byte snapshot. Run configs may place
pipeline-specific settings in `pipeline`, `pipeline_config` or `pipelines[ID]`;
the common runner resolves this once to `pipeline_config`.

The frozen semantic evidence threshold is `fusion.semantic_threshold`
(default 0.5). An explicit `evaluation.semantic_evidence_threshold` is a
separately frozen diagnostic choice; both the effective threshold and evaluation
policy enter comparison gates. Local plans require finite XYZ endpoints in the
declared map frame. Costmap numeric codes are 0 unknown, 1 blocked, 2 traversable.

The initial LingBot adapter returns geometry as a sequence batch. Its paced
semantic worker is a staged scheduling hook, recorded as
`cached_geometry_scheduler_replay`; it cannot establish live end-to-end latency,
update rate, evidence age or simultaneous model residency. These metrics remain
unavailable until an actual streamed geometry producer is connected and measured.

## Common outputs and fusion

Every pipeline emits the same layout, including empty-but-valid semantic outputs
for `geometry_only`. A completed artifact cannot be inferred from a directory existing.

```text
run.json                      # identity, status, provenance, fixture, capabilities
config.resolved.json          # frozen effective settings and digests
frames.jsonl                  # one input/frame disposition, no dropped records
geometry/manifest.json        # validated shared cache link/fingerprint/units
semantics/frames.jsonl        # SemanticFrame metadata and query/mask references
semantics/masks/<ID>.npz      # numeric masks, no pickle
map/manifest.json             # coordinates, units, scale/up, schema, concept registry
map/voxels.npz                # geometry evidence
map/concepts.json             # stable concept IDs/roles/phrases/mapping versions
map/semantic_evidence.npz     # sparse voxel-by-concept evidence
map/contributions.jsonl       # frame/query/pose contribution provenance
planning/costmap.npz          # common raster/grid costs and observed/unknown states
planning/plans.jsonl          # one outcome for every requested start/goal
events.jsonl                  # stages, attempts, errors, queue drops, timings
summary.json                  # counts/status, measured operational metrics
```

`map/voxels.npz` has `voxel_indices:int64[N,3]`, `centers:float64[N,3]`,
`geometry_weight:float64[N]`, `frame_support:int64[N]`,
`last_observed_timestamp_ns:int64[N]`, and `observation_state:uint8[N]`.
State codes: 0 unknown, 1 observed free-space, 2 observed surface. A surface
observation is not automatically an impassable obstacle or supporting floor.
Sparse missing voxels are unknown. Free-space evidence requires a declared
visibility/ray model; missing depth must not produce free space. Record whether
that capability exists; a surface-only implementation never emits state 1.
Conflicting surface/free evidence resolves conservatively to surface until a
declared geometric revision/visibility rule supports clearing it. Semantic
negatives or expiry never clear geometric observations.

`map/semantic_evidence.npz` has M sparse rows:
`voxel_row:int64[M]`, `concept_row:int64[M]`,
`positive_weight:float64[M]`, `observed_weight:float64[M]`,
`frame_support:int64[M]`, `last_seen_timestamp_ns:int64[M]`,
`evidence_score:float64[M]`. Empty arrays are required for pipeline 1.
Registry ordering is saved and deterministic; rows refer to that exact registry
and voxel array. Evidence score is a declared weighted statistic, not probability.
Track contribution IDs, original phrases and per-keyframe pose versions in the
journal so evidence can be inspected, deduplicated, expired or relocated.

The common fuser projects masks onto valid observed surfaces using exact frame
identity. Overlapping concepts remain separate. Cap correlated pixels/aliases
from the same frame/concept; repeated replay is idempotent. Only successfully
queried concepts accrue observed/negative evidence; omitted Qwen concepts are
unqueried, not confirmed absent. A successful no-detection observation may alter
that concept's detection evidence according to the frozen policy, but never clears
occupancy or certifies safety. Temporal expiry makes evidence stale/unknown;
moving-object disappearance is not free-space proof. Reject stale pose/cache
identities or relocate from retained contributions; never silently mix revisions.

## One geometry policy and planner

Session 1 implements shared support/slope/roughness, step/drop-off and footprint/
height/clearance checks and a reproducible local 2.5D costmap/grid planner.
Declare the 2.5D, static/kinematic limitations; it is not a demonstrated robot
controller. Use a common up frame and robot profile. Missing geometry/coverage
remains unknown; observed physical violations block traversal regardless of
semantics. Candidate surface evidence can support a configured semantic prior
only after geometric checks. Hazard evidence applies the same concept-cost policy
for pipelines 3/4. Conflicts and unmapped concepts remain explicit.

`costmap.npz` stores origin/resolution/up projection, costs, observed/unknown
mask and decision state (`unknown`, `blocked`, `traversable`) as versioned numeric
codes. Plans store request ID, start/goal/frame, path coordinates or empty path,
status (`ok`, `no_path`, `blocked_inputs`, `error`) and reason. Do not quietly
snap invalid endpoints into free cells or treat unknown as traversable. If
scale/up/robot inputs block planning, write schema-valid empty costmap arrays,
availability/reason metadata and one `blocked_inputs` plan for every request;
never fabricate a profile or silently omit required artifacts.

## Evaluation protocol and comparison gates

Use two explicitly different modes:

1. **`quality_replay`:** same frozen geometry cache, frame grid, semantic keyframe
   IDs, voxel/fusion parameters, robot policy, goals and planner for all four.
   Cached/staged replay timings are labeled as such. Pipeline differences are
   semantic evidence only. Pipeline 1 still processes every common geometry frame.
2. **`paced_runtime`:** actual end-to-end processing at recorded capture timing,
   including geometry, scheduling, semantics, projection/fusion and planning.
   Use common scheduling hooks and resource limits; measure each pipeline's
   actual execution/residency. Bounded latest-pending semantic work records every
   superseded ID and preserves the in-flight frame's geometry. Offline cached
   results cannot establish live speed or joint memory fit.

Freeze scene/sequence-disjoint development/test splits. Current 100 RELLIS +
40 COCO images remain component evidence. They are not a continuous-video or
navigation benchmark. Reuse public independent references where available;
declare taxonomy, alignment, metric frame, pose/scale source, valid/ignore masks
and coverage. Do not create ground truth from the evaluated models. No bulk
manual labeling or weight training is required by this implementation.

Evaluate common robot decisions and operational behavior across all four.
Also report per-concept/surface segmentation and voxel precision/recall/IoU
where the pipeline's capability and reference coverage support them. Absent
semantic capabilities are `not_applicable`, not a fabricated zero IoU.
Report geometry validity/reprojection diagnostics, decision coverage/unknown
rate, safe precision/recall and unsafe-as-traversable errors. Report planning
success/collision/clearance only against independent references; a path returned
by our own planner is an internal diagnostic, not real navigation success.
Report temporal stability with dynamic-scene context, failures, query counts,
semantic update rate, capture-to-fusion evidence age, stage/end-to-end median/p95,
loading/warmup separately, actual joint GPU/host peaks and geometry/map growth.

Each metric includes `value`, numerator/denominator where applicable, unit,
`status` (`available`, `unavailable`, `not_applicable`), reason and reference
coverage. Missing references/empty denominators produce JSON null, not zero,
NaN or inferred accuracy. Preserve failures and planned frames in denominators;
report successful-call latency separately from completion/failure rate. An
all-unknown map cannot win merely by making no unsafe decisions: pair safety
errors with usable coverage and supported recall. No automatic winner is required.

`reference.json` minimally contains contract/version, reference protocol/ID,
sequence/split digest, fixture provenance, taxonomy/version, frame/grid IDs,
map frame/axes/up/units, pose/scale/alignment provenance, asset paths/hashes,
reference kind (2D semantic, 3D semantic, robot decision or navigation), and
valid/ignore masks plus explicit coverage. Do not infer robot-safe labels merely
from a material class without an independent, declared robot-reference policy.
Unverified alignment/scale or unavailable reference kind blocks that metric.

Freeze these common metric IDs. `C` is independently labeled, valid binary
robot-decision reference coverage; `D` is the frozen evaluation grid/ROI. Both
are independent of predicted-map extents. Missing predictions in D are unknown.

| Metric ID | Definition |
|---|---|
| `decision_coverage` | Non-unknown predicted cells in D / all cells in D |
| `unknown_rate` | Unknown predicted cells in D / all cells in D |
| `safe_precision` | Reference-safe cells predicted traversable in C / all predicted-traversable cells in C |
| `safe_recall` | Reference-safe cells predicted traversable in C / all reference-safe cells in C |
| `unsafe_as_traversable_rate` | Reference-blocked cells predicted traversable in C / all reference-blocked cells in C |
| `semantic_precision.<concept_id>` | TP / (TP + FP) within declared semantic reference coverage |
| `semantic_recall.<concept_id>` | TP / (TP + FN) within declared semantic reference coverage |
| `semantic_iou.<concept_id>` | TP / (TP + FP + FN) within declared semantic reference coverage |
| `reference_valid_path_rate` | Requests with a returned path independently valid for footprint/clearance / requests with valid navigation references |
| `pipeline_failure_rate` | Failed required frame attempts / all required frame attempts, retaining absent/failed records |
| `semantic_query_count` | Actual SAM text-query executions; model-call and attempted/successful query counts also saved |
| `semantic_update_rate_hz` | Completed semantic observations / measured eligible replay duration, runtime mode only |

Use `.median` / `.p95` summaries for `geometry_stage_ms`, `semantic_call_ms`,
`end_to_end_latency_ms` and `semantic_evidence_age_ms`; record counts, clock and
timing boundaries, with synchronization when GPU work is asynchronous.
End-to-end starts at capture availability/decode in the runtime replay and ends
at the corresponding map/planning update; semantic age ends at fusion.
Loading/warmup are separate. Geometry-only semantic timing is not_applicable.
Store actual `joint_gpu_allocated_peak_bytes`, `joint_gpu_reserved_peak_bytes`
and `host_rss_peak_bytes` with collection scope; unavailable measurements stay
null. These IDs do not authorize summing separate component peaks. Additional
metrics require versioned definitions and cannot silently replace these.

`<evaluation-dir>/report.json` and `<evaluation-dir>/metrics.csv` use fixed metric IDs and
record comparability fingerprints. Compare only matching contract/protocol,
mode, sequence/split, geometry/alignment/scale/up, fusion, robot/policy,
goals/planner/evaluation ROI and declared hardware budget. For quality_replay,
actual semantic keyframe IDs match. For paced_runtime, match eligible keyframe/
cadence/scheduler policies and capture schedule; processed/dropped frames are
measured outcomes and may differ. Refuse an incompatible combined
ranking; explain differences instead. Surface/hazard-specific diagnostics are
not silently averaged into one score. Preserve `hazard_prompt_v1` reports unchanged.

## Completion and tests

Session 1 supplies a tiny synthetic sequence, robot profile, independent
reference and fake adapters exercising all four IDs through the actual CLI and
same evaluator. Fake adapters are fixture-only and never real inference fallback.
Test schema/grid/hash mismatch, nontrivial pose/crop/scale/up, altered cache,
invalid depth, missing-depth holes, observed drop-offs, steep/rough support,
wall/overhang clearance, footprint gaps, semantic/geometry conflict, overlapping
concepts, duplicate observations, partial failures, omitted queries, queue drops,
expired evidence and missing references. Do not add tests that merely mirror code.

Real-model GPU execution occurs on KTH, not on the user's PC. Before any job,
check current CPU/RAM/disk/GPU processes and allocation; use only the user's
visible device and an isolated environment. The previous allocation was an
H100 MIG slice with about 20 GB GPU memory and 64 GB container RAM; this is
historical capacity, not a current availability or measured joint-fit claim.
Preserve server datasets, caches, credentials and completed runs. Do not perform
bulk downloads or start long evaluations as implementation side effects. Prepare
small replay commands and state clearly which real-model checks remain unrun.
KTH is not INESC; Aquila/INESC-specific rules apply only if those hosts are used.
