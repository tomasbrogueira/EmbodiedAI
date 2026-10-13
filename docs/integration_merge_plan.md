# Integrating path-planning-cleanup and vlm_improvements

Status: initial offline and finalized-stream integration implemented on
`integration/path-planning-vlm`. See INTEGRATED_PIPELINES.md and
INTEGRATION_VALIDATION.md. Continuous prefix replanning remains a later extension.

Updated requirement: every stage must support standalone execution and composition.
Slope/steepness analysis and path planning are separately switchable features,
alongside VLM, SAM, reconstruction and visualization. The default combined
profile preserves the existing slope-aware planner; an explicit slope-disabled
planning profile is also required.

Updated semantic requirement: VLM instructions can target either traversable or
non-traversable regions. Independent extra SAM traversability queries (for
example, path) run alongside VLM-derived queries or without VLM. Their masks
must change the planner's cost-map support/obstacle interpretation, not merely
map colors. Target class and source are explicit for every proposal and mask.

Reviewed sources:

- `origin/path-planning-cleanup`: `708053a08fda2f75c5767b320f2763626de5c155`
- `vlm_improvements`: `7f9a3a58baa7aa3f003b3116077fae4beab890f8`
- Common ancestor: `ad248afb369ef5b398c86f005bb791f61429b5ca`

This plan uses the six requested components: path planner, VLM, orchestrator,
SAM, LingBot-MAP and visualizer. Use path-planning-cleanup as the integration
foundation and import the requested vlm_improvements branch. The current
vlm_improvements_v2 checkout is not an integration source.

## 1. Findings that determine the design

1. The seven core orchestrator Python modules and its initializer are identical
   on the two reviewed branches. Planning adds workflow scripts; VLM adds the
   optional small-obstacles plugin. Retain those additions.
2. A read-only three-way merge preview shows one textual conflict in README.md.
   This is not proof that the merged system will run correctly. Semantic and
   lifecycle integration require the work below.
3. The planner uses an explicit support mask for floor/ground. The VLM/SAM plugin
   produces an explicit obstacle mask. A combined system needs both channels.
4. The planner treats observed non-support geometry as obstacles. Simply showing
   the plugin's red PLY does not alter planning, and replacing the support mask
   with the obstacle union would invert the meaning of the data.
5. Planning reconstruction initializes from eight spread recording frames and
   restores source order. VLM incremental reconstruction initializes from the
   first two selected frames and maintains causal caches. Their outputs are
   separate reconstruction policies, not interchangeable coordinate arrays.
6. The planning adapter expects complete original-frame predictions and camera
   timestamps. The VLM runner optionally saves only mapped frames and uses
   sampled source indices. Its artifacts need an explicit adapter.
7. Current component boundaries are mixed: model runners and viewers live under
   path-planning, and model control/projection/UI live under small_obstacles.
8. The inherited VLM-branch CI references absent requirements/test.txt and root
   tests/. Retain the actual planning CI and add the real plugin test suite.
9. Preparation hashes many source files; moving files changes identity. Existing
   artifacts must not be declared current just because the implementation was
   relocated.
10. The existing VLM instruction detects hazards only. Generalize its contract
    with an explicit traversable/non-traversable target and separate instruction
    presets. Additional SAM traversability queries are a parallel independent
    source; they must not replace, depend on or expire with the VLM query list.

## 2. Target ownership and repository layout

Keep path-planning/ as the existing planner component location. Create clear
first-party directories for the other responsibilities. The LingBot component
has first-party adapters under lingbot/ and the external source checkout under
lingbot-map/. SAM follows the same pattern with sam/ and external sam3/.

```text
repo/
  path-planning/             Planner algorithms and native search
  path_planner/             Independent importable numerical implementation
    test_*.py               Existing contracts, moved only with compatibility
    corridor_astar_feasible.cpp
    planner.py, ...          Temporary compatibility entry points
  vlm/                      Proposal generation, instruction presets, evaluation
  sam/                      Support and obstacle segmentation adapters
  lingbot/                  Offline and causal reconstruction adapters
  orchestrator/             Contracts, projection, workers, scheduling, receipts
    contracts/
    pipelines/
    workers/
    scripts/
  visualizer/               Saved/live views, route overlays, camera controls
  plugins/small_obstacles/  Compatibility wrappers during migration
  tests/integration/        Cross-component artifact and lifecycle checks
  docs/
  data/videos/              Existing source recordings
  lingbot-map/              External source and weights, ignored
  sam3/                     External source and weights, ignored
```

The six components can be in one repository without being one dependency stack.
The orchestrator calls planner APIs locally and model adapters through the
existing separate-environment worker mechanism. The visualizer submits run
requests to orchestration and reads published artifacts.

| Component | Owns | Inputs and outputs | Dependency boundary |
| --- | --- | --- | --- |
| Path planner | Reference fitting, slopes, costs, node/edge predicates, endpoints, Python/native optimal search and route certification | Aligned geometry/evidence/camera arrays to costs and routes | NumPy/SciPy/native library; no model or UI imports |
| VLM | Qwen inference, selectable traversability target, prompt parsing, instruction definitions/presets and proposal evaluation | RGB + instruction + target class to timestamped typed concepts/raw response | Transformers/Torch/Pillow in VLM environment; no geometry or planner imports |
| SAM | Floor/ground video masks, typed VLM image queries, independent extra traversability queries, per-channel unions and bounded text cache | RGB/video + queries with class/source to separate typed masks and provenance | SAM3/Torch in SAM environment; no costs, route or viewer logic |
| LingBot-MAP | Model loading, offline anchors, causal cache/sequence lifecycle and official geometry adapter | Video/RGB to depth/confidence/intrinsics/extrinsics/source metadata | LingBot/Torch in model environment; no semantic interpretation |
| Orchestrator | Input inventory, sampling, worker RPC/session ownership, projection/alignment, CPU/GPU scheduling, run transactions and provenance | Requests/artifacts to verified map snapshots and stage results | Lightweight controller dependencies; model modules imported lazily only by their workers |
| Visualizer | Source/obstacle/cost/slope modes, camera trajectory, routes, prompt/run controls and status presentation | Published snapshots/status to UI; commands to orchestrator | Viser/Pillow/UI; no model loading or mutation of route/cost data |

Planner inputs use ordinary arrays and planner-owned value objects. The planner
does not need to import the orchestrator's disk schema to run synthetic tests.
Disk schemas are validated at the orchestrator boundary.

Slope analysis and route search are separate public APIs within the path planner
component. Either can run without invoking the other. The orchestrator builds a
dependency graph from the requested features rather than always executing one
fixed end-to-end chain. Each component also exposes its own CLI/API accepting
explicit inputs and producing reusable artifacts.

## 3. File migration map

| Existing implementation | Target owner/action |
| --- | --- |
| path-planning/slope.py, floor_reference.py, cost_map.py, planner.py | Path planner; preserve formulas and numerical behavior |
| path-planning/recording_endpoints.py | Keep array-based selection in planner; move prediction/mask file loading into orchestration |
| path-planning/dataset_path.py | Keep native search/certification/endpoint rules in planner; move argument parsing, progress publication and artifact IO into orchestration |
| path-planning/cost_evidence.py | Orchestrator evidence adapter; preserve exact XYZ checks |
| path-planning/video_reference.py, dataset_reference.py | Split prediction decoding into orchestrator and window/fitting decisions into planner |
| path-planning/dataset_prepare.py | Orchestrator preparation/calibration workflow calling planner APIs |
| path-planning/dataset_inventory.py, dataset_gpu.py, dataset_cpu.py, dataset_resources.py | Orchestrator batch pipelines and resource scheduling |
| path-planning/reconstruct.py | lingbot/offline.py; keep old CLI as wrapper |
| path-planning/generate_masks.py, sam_cpu_video.py | SAM support-video adapter; keep old CLI wrappers |
| small_obstacles/propose.py | VLM proposal adapter; traversable/non-traversable target and corresponding instructions |
| small_obstacles/common.py | Split prompt utilities into VLM and RGB/frame/artifact utilities into orchestration |
| small_obstacles/prompt_presets.py | VLM instruction/preset validation; UI widgets stay in visualizer |
| small_obstacles/segment.py, sam_runtime.py | Typed SAM image adapter, parallel independent traversability prompts, separate mask unions and optional text cache |
| small_obstacles/stream_worker.py | Orchestrator worker protocol; move LingBotStream into LingBot adapter and delegate each role |
| small_obstacles/model_session.py, run_streaming.py, run_existing_pipeline.py | Orchestrator model sessions and pipeline runners |
| small_obstacles/project.py; duplicated project_frame implementation | One canonical orchestrator projection service, with original CLI-compatible wrappers |
| small_obstacles/stream_state.py | Timestamped typed VLM prompt lifetime to VLM; independent fixed SAM prompts to run configuration; PLY persistence to orchestrator; reservoir display sampling to visualizer |
| small_obstacles/runtime_metrics.py | Orchestrator telemetry, with role-specific cache inspection delegated to model adapters |
| small_obstacles/live_demo.py, view_stream.py, view_maps.py, scene_pose.py, viewer_overview.py | Visualizer; session ownership and worker launch remain in orchestrator |
| path-planning/visualization/*; orchestrator/view_ply.py | Visualizer; extract pure PLY reader into orchestrator before removing preparation's viewer import |
| small_obstacles/evaluate.py | VLM diagnostic tools |
| small_obstacles/profile_runtime.py, benchmark_geometry.py, compare_geometry.py, timing_report.py, verify_streaming_run.py | Diagnostic commands under their owning component or orchestrator integration tools |

Move one family at a time with thin re-export/CLI wrappers. Introduce qualified
imports rather than relying on ambiguous modules such as common, planner or
io_utils. Check subprocess paths and source-hash readers after every migration.

## 4. Shared contracts

### Frame prediction

Record run_id, reconstruction_id, geometry_revision, reconstruction_policy,
sequence_index, source_frame_index, source_timestamp, processed RGB dimensions,
RGB SHA-256, depth, confidence, intrinsics, extrinsics and camera convention.
Store sampling separately from source frame identity. Retain raw source timestamp
and the preprocessing transform. Frames without depth cannot enter a map snapshot.

Support at least two named policies:

- offline_uniform_anchors: existing planning offline method;
- causal_stream: existing VLM stream method.

Preserve each algorithm's initialization, cache, keyframe and pose conventions.
Use the offline policy for the initial combined planning implementation.

### Segmentation artifact

Each mask has a semantic class (traversable or non_traversable) and source
(base_support_sam, vlm_sam or extra_sam). The base floor/ground support channel
also retains its reference-estimation role. Record frame/RGB identity,
dimensions, producer/model/settings identity, prompt(s), threshold, proposal
source timestamp when applicable, status and mask digest. Preserve source-specific
masks even after constructing aggregated support and obstacle channels.

Valid empty detection, intentionally disabled detection, unavailable data and
inference failure are distinct states. A failed or missing required obstacle
artifact cannot silently become a valid empty mask. Explicitly disabled obstacle
processing is a supported planner-only run mode.

Base support segmentation remains floor indoors and ground outdoors. VLM-derived
segmentation is routed into the class selected for its instruction. Additional
manual SAM queries such as path create an independent traversable channel; the
existing manual obstacle mode creates a non-traversable channel. Preserve the
original support video workflow and image-query workflow initially; they are
different algorithms. Support video masks may need the existing mapper alignment
transform, whereas image-query masks already use processed LingBot RGB. Record
and validate each path.

### VLM target and independent extra SAM queries

Expose VLM target: traversable | non_traversable. Keep the legacy hazard
instruction as the default non-traversable preset for compatibility; add a
traversability instruction preset. Custom instructions can describe either
target and must carry the selected class. Do not guess polarity from names such
as path, water, log or from response prose. Freeze target plus exact instruction
hash and preset name in each run/proposal artifact. Switching target invalidates
dependent masks/costs/routes even if the resulting prompt strings are identical.

The existing JSON prompts list can remain the model response format. The adapter
attaches configured target, source frame/time and proposal identity after parsing.
Validate a response target if supplied; disagreement is an invalid proposal.
Changing target loads the corresponding built-in instruction unless the user
explicitly selects a custom instruction tagged for that target.

Use independently configured SAM extra traversability prompts, e.g. [path].
They are run-scoped fixed queries, processed on every selected semantic frame
when enabled. They do not require a VLM response and remain active when VLM
returns [], fails, expires, is busy, or is disabled. A required VLM failure still
retains its own error state; successful extra masks must not disguise it.

Run SAM with both query groups on the same current RGB image. Compute its fresh
image embedding once, then process each distinct query with a per-source/per-class
result. The existing bounded text cache remains valid for exact strings. A prompt
seen in both classes may share its inference result but must retain both labels;
deduplication must not merge the semantic classes. Never OR positive and negative
results into the single legacy obstacle mask before cost-map interpretation.

VLM concepts retain their timestamp/TTL behavior. Extra SAM prompts do not expire
with VLM TTL. Shared SAM execution does not make their configuration, provenance
or failure status dependent on the VLM. Independent groups have separate prompt
limits; the current VLM max-prompts limit must not evict extra SAM queries.

### Map snapshot

Publish points[N,3], source_rgb[N,3], frame_ids[N], processed_pixel_ids[N] or
equivalent frame-offset/valid-pixel mapping, cameras[K,3], timestamps[K] and
source-frame IDs for camera rows. Attach enabled source masks:

- base_support_mask[N]: original floor/ground segmentation;
- sam_traversability_mask[N]: independent extra SAM traversability prompts;
- vlm_traversability_mask[N]: SAM results for traversable VLM proposals;
- vlm_non_traversability_mask[N]: SAM results for non-traversable VLM proposals;
- manual_non_traversability_mask[N]: existing direct/manual obstacle queries.

Publish derived support_mask[N] and obstacle_mask[N] alongside source masks and
the fusion-policy identity. Retain explicit disabled/unavailable states when
absent. Geometry-only snapshots are valid without either derived channel. A
planner request requires valid observed support evidence; positive extra/VLM
masks can supply candidate support when reference/radius/calibration prerequisites
are independently satisfied. A semantic-only map does not require base floor/
ground segmentation or planner calibration.

XYZ rows retain the existing ordered concatenation, float32 export precision and
filter rules. Row ID is the array index. Source RGB is preserved separately from
rendered semantic colors. The canonical correspondence manifest must bind the
reconstruction, frame selection, filters and both semantic channels.

No voxel conversion, filling, coordinate normalization, sorting or display
sampling is allowed between the canonical snapshot and planner input.

### Planner result

Every cost/route artifact references the exact map snapshot, evidence digests,
reference/calibration identity, planner/native version, feature configuration and
planning policy, including whether slopes are used.
Route IDs and positions must agree with that snapshot. Preserve existing statuses
including missing reference/calibration, endpoint failures and certified partial
routes. Distinguish baseline-only and obstacle-aware results in provenance.

## 5. Integrating traversability evidence into the cost map

Preserve original floor/ground evidence and each semantic source. Extend
build_cost_map/CostMap with aligned traversability and non-traversability evidence
plus explicit class/source provenance. Aggregate with this initial fixed policy:

```text
positive_evidence = base_support_mask
                    OR sam_traversability_mask
                    OR vlm_traversability_mask

negative_evidence = vlm_non_traversability_mask
                    OR manual_non_traversability_mask

raw_support = positive_evidence
eligible_support = positive_evidence AND NOT negative_evidence
observed_obstacles = NOT positive_evidence OR negative_evidence

support_mask = raw_support
obstacle_mask = negative_evidence
```

Disabled optional sources contribute no labels and retain explicit disabled
status. Missing or failed enabled sources are not converted to valid zero masks.
Non-traversable evidence takes precedence on every overlapping row. Source masks
remain available for inspection and dependency hashing.

This policy expands support candidates with traversable labels instead of merely
painting them green. Positive SAM/VLM labels do not automatically set zero cost
or declare an executable route. Candidates still need finite geometry, sufficient
support and the selected slope/clearance/connection predicates. Negative labels
make flagged rows infeasible and include their observed geometry in the obstacle
indexes, including whole-edge checks. Untagged non-support geometry retains the
planner's existing obstacle treatment.

Use eligible_support for node eligibility and support-query construction. Use
observed_obstacles for obstacle trees, nearest-obstacle penalties, node sweeps and
whole-edge clearance. Preserve raw_support for nearest beneath-camera association:
if the geometrically nearest support row is now blocked, report blocked_support
instead of skipping it to find a farther node. Removing flagged rows from the
endpoint's raw support mask would violate this existing rule. The raw mask is now
the positive union, so valid extra path evidence can participate in endpoints;
negative overlap still causes blocked_support rather than a farther substitution.

Implement this fusion in the planner's cost-map input/eligibility policy, not
only in orchestrator rendering. Update costs, support/obstacle indexes, saved
cost reconstruction and native row lists from the same verified source masks.
Extend every constructor/restoration path consistently, especially
dataset_path.restore_cost_map and NativeSearch.prepare_map. The native function
already accepts support and obstacle row lists; investigate retaining its ABI
while passing the updated lists. Certify all candidates with the Python CostMap.

Keep reference-estimation evidence distinct from semantic traversability. By
default estimate the ground normal and pixel-spacing radius from the original
floor/ground masks, preserving the existing robust plane checks. A path query
does not by itself supply measured ground orientation or camera clearance.
If base support is disabled, a planning run needs a supplied validated reference/
radius or an explicitly selected traversability-reference adapter that passes
the same robust checks; no invented plane fallback is allowed.

Freeze H using baseline-valid beneath-camera associations before applying
negative semantic filtering. For the default profile with no extra positive
evidence, this retains existing calibration. Prefer original base support for
calibration; positive-union calibration without base support requires an explicit
recorded policy and the same baseline geometric/beneath-camera checks. For the
slope-disabled profile baseline validity omits slope predicates. Changing positive
support or its calibration source creates a new calibration/cost identity;
changing only negative flags does not recalibrate H. Retain contributing source,
frame/row IDs and digests. Enabled slopes use original per-frame geometry.

For fixed reference/calibration/configuration, adding obstacle flags must not
create new feasible nodes or connections. Existing finite costs can increase or
become infinite. With positive evidence fixed, an empty valid negative channel
must reproduce that support profile's costs/neighbors/routes. With additional
positive and negative channels disabled or valid-empty, reproduce the original
planner baseline. Adding positive labels may open candidates only through the
unchanged geometric predicates; test costs/routes under the new evidence identity.

Keep holes/water as semantic evidence without manufacturing 3D surfaces. A pixel
with no surviving depth provides no new 3D obstacle row. The existing continuous
support checks remain necessary, and this integration does not create a metric
body-clearance or observed-free-space guarantee.

## 6. Supported execution modes and rollout order

### Standalone stages and independently selectable features

Expose stage commands as well as an orchestrator run command. CLI names below
describe planned interfaces; they are not commands implemented by this plan.

| Standalone stage | Required input | Output; stages it does not launch |
| --- | --- | --- |
| LingBot reconstruct | Video or selected RGB sequence | Geometry/poses; no VLM, SAM, slopes or route |
| VLM propose | Images, video samples or saved prediction RGB + instruction + traversability target | Typed concepts/raw responses; no reconstruction or SAM |
| SAM support | Video/RGB + floor/ground query | Support masks; no VLM, geometry analysis or route |
| SAM image queries | RGB + independent extra traversability prompts and/or typed manual/saved VLM queries | Separate source/class masks; no VLM model unless separately requested |
| Orchestrator project | Saved geometry + any enabled mask channels | Canonical geometry/semantic snapshot; no model inference required |
| Planner steepness | Canonical points + reference, or support evidence for reference estimation + neighborhood configuration | Slope/quality arrays; no cost construction, route or native search |
| Planner costs | Canonical snapshot + support/calibration + selected slope policy | Costs/graph identity; no route search |
| Planner route | Valid saved costs/map or explicit planner arrays + endpoints/policy | Certified route; no model inference or viewer |
| Visualizer view | Saved snapshot and optional matching analysis/results | Available layers; no inference or planning required |

Every stage can use previously saved compatible inputs. Missing inputs are
reported before launching workers; a standalone stage never silently runs other
components. The combined orchestrator may resolve dependencies only through
the configured compute/reuse modes and must show the resolved stages up front.

Use a frozen run configuration with explicit stage modes and feature switches:

```yaml
# Proposed configuration contract, not an existing runnable config.
geometry:
  mode: compute                    # compute | reuse | off
  policy: offline_uniform_anchors
vlm:
  mode: compute                    # compute | reuse | off
  target: non_traversable           # traversable | non_traversable
  instruction_preset: hazards       # Matching class; or a tagged custom preset
sam:
  support: compute                 # compute | reuse | off
  vlm_queries: compute             # compute | reuse | off; inherit VLM class
  extra_traversability:
    mode: compute                  # compute | reuse | off; independent of VLM
    prompts: [path]
  manual_non_traversability:
    mode: off                      # Existing manual obstacle mode
    prompts: []
analysis:
  steepness: compute               # compute | reuse | off
  costs: compute                   # compute | reuse | off
planning:
  enabled: true
  slope_policy: enforce            # enforce | disabled
visualizer:
  enabled: true
```

The VLM and extra_traversability entries are additive: both run in the example.
Switching VLM target to traversable changes only the class of VLM-derived masks;
extra path masks remain traversable and independently enabled. Saved VLM queries
retain their original target and cannot be silently reinterpreted through reuse.

The public Slope/steepness feature switch controls whether slope evidence is
computed/reused and used. The public Path planning switch controls route search.
These are independent: slopes can be enabled for inspection with planning off,
and planning can run with slopes off using the explicit policy below. Costs can
also be requested independently, for a cost-map-only result.

For new runs, the UI slope switch selects the policy: slope on means enforce
with compute/reuse; slope off means disabled. A user selecting a compatible
saved slope artifact is enabling slope evidence through reuse, not running new
slope fitting. No hidden auto-enabling or silent substitution is permitted.

### Slope-aware and slope-disabled planning policies

**Default slope-aware policy:** preserve the existing slope estimates, NaN
handling, node inclination cutoff, slope cost term, edge grade limit and all
Python/native certification rules. Valid reused slopes must match the geometry,
reference, frame grouping and slope configuration.

**Explicit slope-disabled policy:** skip slope fitting and omit its penalties
and feasibility predicates. Keep support requirements, gap checks, geometric
obstacle clearance, fixed-start endpoint rules and exact route certification.

```text
slope-aware cost:
  1 + slope_penalty + obstacle_clearance_penalty

slope-disabled cost:
  1 + obstacle_clearance_penalty
```

Slope-disabled planning must also bypass slope-based node eligibility and edge
grade rejection. Setting w_slope=0 alone is insufficient because current NaN
and slope thresholds would still reject rows/edges. Do not synthesize zero slopes
or widen max_slope to mimic disabling. Add an explicit policy argument throughout
Python CostMap, restoration, candidate bounds and native feasibility search;
version the native interface if needed. Certify against the selected policy.

The floor/ground reference, automatically derived neighborhood radius and H are
still required for support/clearance geometry. Computing these is separate from
fitting per-point slopes. Extract the spacing/radius derivation from the existing
reference report so it is available without invoking estimate_slope.

Slope-disabled routes are labeled "Slope checks disabled" in configuration,
reports and the viewer; they do not certify a slope limit. Their costs/calibration
and routes cannot reuse the slope-aware policy identity. Unknown slope values
remain unknown if older slope artifacts are displayed as historical context.

### Supported combinations

| Requested pipeline | Reconstruction | VLM | SAM channels | Steepness | Route planning |
| --- | --- | --- | --- | --- | --- |
| Geometry only | Compute/reuse | Off | Off | Off | Off |
| VLM discovery only | Not required; RGB/video supplied | Compute/reuse | Off | Off | Off |
| SAM segmentation only | Not required; RGB/video supplied | Off or saved concepts | Support and/or obstacles | Off | Off |
| Existing VLM + SAM obstacle map | Compute/reuse | Compute/reuse, or manual queries | Obstacles | Off | Off |
| VLM traversability map | Compute/reuse | Traversable target | VLM traversable | Off | Off |
| Extra SAM path map without VLM | Compute/reuse | Off | Independent extra traversability | Optional | Optional with valid reference/calibration |
| VLM non-traversable + independent SAM path | Compute/reuse | Non-traversable target | VLM negative + extra positive; base support optional subject to calibration | On/off by policy | Optional |
| Steepness inspection only | Compute/reuse | Off | Support or saved reference | Compute/reuse | Off |
| Cost map only | Compute/reuse | Optional | Support, optional obstacles | On/off by policy | Off |
| Existing slope-aware planner | Compute/reuse | Off | Support | Compute/reuse | On |
| Planner with slope disabled | Compute/reuse | Optional | Support, optional obstacles | Off | On, disabled policy |
| Full combined pipeline | Compute/reuse | Compute/reuse, or manual queries | Support + obstacles | Compute/reuse | On |
| View saved results | Reuse artifacts only | Off | Reuse available channels | Reuse if present | Reuse route if present; no search |

The viewer is optional for every compute pipeline. Individual stage commands
and headless composed runs remain first-class supported interfaces.

When geometry is off, stages requiring 3D input need a supplied validated snapshot;
2D VLM/SAM stages still work from RGB/video. When SAM inference is off, planning
can use supplied compatible support masks; no support evidence means a planner
request is rejected. VLM off plus its query stage on requires saved typed queries;
extra traversability SAM and manual non-traversability SAM use their own fixed
prompts and remain independently executable. Positive-only planning still needs
valid reference/radius/calibration rather than automatically enabling base SAM.

### Controls, disabled stages and run lifecycle

Provide separate run controls for VLM proposal generation with target selector,
base support SAM, VLM-query SAM, extra SAM traversability (enabled + prompt editor),
manual non-traversability SAM, slope/steepness, costs and path planning.
Reconstruction and visualization also have standalone/headless operation.
Keep layer visibility controls separate from feature computation: hiding
Steepness does not disable fitting, and hiding a Route does not stop search.

Disabled stages have status disabled and produce no new slope/cost/route files,
load no model/native library needed only by that stage and spawn no associated
workers. Reused artifacts have status reused with their identity, not computed.
Display only layers supported by current matching artifacts. New runs must never
show an old cost or route as their result when the corresponding feature is off.

Freeze feature configuration at run start. Changing compute/policy switches
applies to the next run or an explicitly requested derived snapshot job; never
mutate a graph during native/Python search. Layer visibility can change freely
while a run is active. Completed geometry remains reusable when analysis is
enabled later, and completed costs remain reusable when route search is enabled.

Instruction presets store their target class. Extra SAM traversability prompts
are edited/saved separately from VLM instructions and their generated concepts.
Show separate source masks and a combined support/blocked view, including label
conflicts where negative evidence wins. A display color is never the cost-map
input. The source view continues to show unmodified RGB.

### Initial delivery: offline combined planning

Run the planning branch's reconstruction once. Generate selected floor/ground,
typed VLM-query and independent extra SAM traversability masks against this same
reconstruction. Project one canonical snapshot, prepare/calibrate, calculate
costs from the positive/negative evidence policy, run certified routing and show
all outputs in the visualizer.

```text
inventory -> offline LingBot predictions
          -> SAM support masks
          -> Qwen proposals + target -> typed SAM masks
          -> independent SAM traversability prompts (e.g. path)
          -> shared projection/alignment + positive/negative cost policy
          -> reference/slopes/baseline calibration
          -> obstacle-aware costs -> certified route -> visualization
```

This is one preset of the composable pipeline, not a mandatory sequence. A
resolved stage graph skips disabled features and reuses valid supplied artifacts.
VLM is optional: independent extra traversability queries, manual obstacle names
and planner-only operation remain available. Sequential model stages are the
default for this offline workflow,
releasing each required GPU model before loading the next.

### Second delivery: live mapping, planning after completion

Retain the causal stream, prompt expiration, fresh SAM masks, cancellation and
warm sessions. Add typed positive/negative evidence, independent extra SAM query
execution and full pose/source metadata required by the planner. Extra SAM image
queries run on every selected semantic frame even with no active VLM concepts.
Plan on an immutable finalized snapshot using the same planner API.
Do not assume old mapped-only stream files satisfy the full-recording adapter.

Decide explicitly whether support comes from per-image SAM during the stream or
a support-video pass after completion. Start with the after-completion adapter
and recorded RGB correspondence; enabling per-image support is a separate
inference policy with its own validation.

### Later delivery: incremental replanning

Publish immutable/versioned map prefixes and explicit selected-frame sequences.
Use a snapshot-aware adapter instead of the existing contiguous full-frame
assumptions. Cost/reference/calibration identity is fixed within a snapshot;
updated estimates create a new revision and invalidate its old route.

Run CPU planning independently from perception and publish a route only for the
matching snapshot. Keep the previous route visibly marked stale or withhold it
when the snapshot changes. Native/Python global optimality is within the selected
snapshot, not a claim about unseen future geometry.

## 7. Resource and lifecycle ownership

Only orchestration owns Worker, ModelSession and GPU allocation coordination.
Visualizers invoke start/stop/unload through that service. Offline workers and
live sessions must share allocation coordination rather than two unrelated locks.
Do not start batch inference on a GPU occupied by a persistent live session.

Keep separate model interpreters and checkpoints. Retain the VLM branch's
Qwen3-VL-4B Transformers backend; no model or engine migration is part of this
merge. Keep CPU BLAS limits and the planner's >900-frame exclusivity rules.

Build required worker roles from the enabled stage graph. Slope-only and planner
jobs over saved arrays require no GPU workers; 2D VLM/SAM-only jobs require no
LingBot worker; mapping with both analysis features off requires no native search
library. Persistent sessions retain only explicitly requested resident roles;
disabled model roles are unloaded while idle unless the user explicitly chooses
to retain them for subsequent jobs.

The standalone stream defaults to cache window 64; the browser selects 8.
Persist this distinction because cache size affects geometry. Preserve the
bounded SAM text cache and fresh per-image state. Expose memory/load/cache
telemetry through orchestration, while retaining the existing cancellation
boundary and completed-prefix artifacts.

## 8. Cache identity and output layout

Use a new external output directory and a versioned run schema for integration.
Do not rewrite old receipts or source hashes to make historical results current.
Keep archived viewers/manifests available for old runs.

Stage identities should bind the inputs actually consumed:

- Geometry: input/preprocessing/reconstruction source/model/settings.
- Support: geometry/RGB correspondence and support model/settings.
- Proposals: RGB/instruction/target/model/sampling and class-tagged preset.
- VLM SAM masks: typed proposal list, RGB, SAM model/threshold/settings and
  source-frame activation schedule.
- Extra SAM traversability masks: their independent prompt list, RGB, SAM
  model/threshold/settings; no dependency on VLM instruction, target or TTL.
- Manual non-traversability masks: their independent prompt list and SAM identity.
- Steepness: geometry/reference, same-frame grouping, neighborhood config and
  slope source identity; independent of route enablement and VLM instruction.
- Costs: geometry, all consumed positive/negative evidence, fusion/calibration
  policy, reference/H/config, slope policy, slope identity only when consumed,
  and cost code.
- Routes: cost identity, endpoint policy and Python/native search identity.

Store requested and resolved features/stage modes in the manifest. Disabled
semantic channels have explicit identities so obstacle-aware caches cannot become
planner-only caches by omission. Enabling/disabling slopes changes costs/routes,
but route off does not invalidate independently generated slopes or geometry.
Steepness output is optional; planning-disabled output has no new route receipt.

Relocation changes existing identities, so require fresh integration artifacts.
Within the new schema, an instruction change need not force reconstruction; a
viewer-only change need not invalidate costs. Native binary/source identity is
part of route provenance. Atomically publish a completion receipt last, after
all referenced files have been written and verified.

```text
external-batch/
  manifest.json
  allocation locks / reports
  runs/<category>/<video>/
    entry.json / config.json / status.json
    geometry/predictions/ + receipt
    sam/support/ + receipt
    vlm/proposals.json + instruction/target identity
    sam/vlm_traversable/ or sam/vlm_non_traversable/ + receipt
    sam/extra_traversability/ + independent receipt
    sam/manual_non_traversability/ + receipt
    map/<snapshot-id>/
      points.npy / source_rgb.npy
      base_support_mask.npy / sam_traversability_mask.npy
      vlm_traversability_mask.npy / vlm_non_traversability_mask.npy
      manual_non_traversability_mask.npy
      support_mask.npy / obstacle_mask.npy  # Derived; fusion policy recorded
      frame_ids.npy / pixel correspondence
      cameras.npy / timestamps.npy / provenance.json
    planning/<snapshot-id>/<policy-id>/
      reference.npy / calibration.json
      slope.npy / quality.npy       # Only if steepness enabled; own receipt
      cost.npy                     # Only if costs requested; own receipt
      route.npz / report.json       # Only if route requested; own receipt
    visualization/ + logs/
```

## 9. Implementation sequence and reviewable commits

| Phase | Deliverable | Completion gate |
| --- | --- | --- |
| 0. Capture baselines | Pin both branch commits, record external model commits/environment versions, run each existing suite and save small baseline artifacts | Known passing baseline or explicit pre-existing failures |
| 1. Mechanical merge | New integration branch from planning, merge vlm_improvements, reconcile README, retain planning scripts/CI/ignore rules | Both code families present; no unresolved merge conflict; no model changes |
| 2. Component extraction | Migrate model adapters, orchestration, pure planner interfaces and views by family; retain CLI wrappers | Existing behavior passes after each family move; dependency boundaries checked |
| 3. Artifact contracts | Add source frame/RGB identity, VLM target and matching presets, parallel independent SAM queries, per-source/class masks, standalone APIs and map snapshots | Row/mask/pose correspondence, target/source isolation and optional-artifact tests pass |
| 4. Configurable planner | Separate slope/cost/search APIs; slope-disabled policy; positive/negative fusion in CostMap; calibration/restoration/native parity | Default parity, extra path support, negative-overlap priority and native certification tests pass |
| 5. Composable offline orchestrator | Independent feature switches, compute/reuse/off stage graph, receipts, resource coordination, restart semantics | Standalone and supported combinations produce only requested verified artifacts |
| 6. Unified visualizer | Independent feature and layer controls; source/obstacle/cost/steepness, camera, routes and stale-result handling | Viewer consumes matching optional artifacts and never silently enables features |
| 7. Live finalization | Causal stream support-evidence adapter and planning after EOF | Warm reuse, cancellation, frame identity and finalized route validation pass |
| 8. Optional replanning | Immutable prefixes and revision-aware asynchronous planning | No stale route appears current; snapshot-local certification passes |

Mechanical merge preparation should use an isolated worktree. Do not switch or
stash the user's dirty current checkout. Suggested future branch name:
integration/path-planning-vlm. Recheck the pinned refs before implementation.

Resolve README manually around the six components. Keep planning CPU CI as the
working basis and add discovery of plugins/small_obstacles/tests until migration
finishes. Retain .gitattributes source normalization and planning .gitignore
rules; add new output patterns without removing tracked historical evidence.
Treat .gitmodules as stale declarations until an explicit external-checkout
policy is documented; no upstream model checkout changes are required by merge.

## 10. Validation matrix

### Component and dependency tests

- Run all 18 planning test modules and all nine VLM-branch plugin test modules.
- Compile the native library with existing precision flags and test parity.
- Planner import/use works with NumPy/SciPy and native code only: no Torch,
  Transformers, SAM, LingBot or Viser required.
- Orchestrator/visualizer imports do not initialize CUDA or model weights.
- Compatibility CLI help/argument forwarding and subprocess paths work after moves.
- Each standalone CLI runs from supplied inputs without launching other model,
  analysis or visualization stages.
- Slope-only imports/runs require no native search library; planning with reused
  costs requires no model weights; slope-disabled planning never calls slope fitting.

### Evidence and geometry tests

- Exact point-row order, float32 XYZ, raw RGB and source-frame correspondence.
- RGB hash/dimension mismatch is rejected; missing/failed masks differ from empty.
- Sparse frame numbering and source timestamps do not become renumbered silently.
- Official camera/depth convention parity; no extra inversion or axis transform.
- Floor and obstacle masks cannot overwrite or masquerade as one another.
- Original support-video and obstacle-image preprocessing contracts both hold.
- VLM target/preset/instruction identity propagates into proposals, SAM source
  masks, map fusion, costs and routes. Reusing strings under the opposite target
  is rejected; ambiguous/mismatched response classes are not guessed.
- Positive/negative masks remain separated even for the same prompt string and
  shared image embedding/text features. Sources remain independently identifiable.

### Planner behavior tests

- All-false valid obstacle mask matches original costs, edges and path outcomes.
- A support row flagged as obstacle becomes unavailable and joins obstacle queries.
- Closest blocked beneath-camera support fails rather than selecting farther ground.
- A thin obstacle crossing an edge blocks it even if both endpoints are feasible.
- Mixed support/obstacle labels preserve the raw reference/calibration evidence.
- Adding flags cannot create nodes/edges for fixed reference/H/configuration.
- Python/native paths and costs agree for detours, no path, coincident nodes and
  fixed-start nearest-reachable partial routes.
- Missing reference/calibration yields the existing explicit no-route states.
- Slope-disabled costs omit the slope term and Python/native predicates both
  omit node inclination/edge grade checks; support/gap/obstacle checks remain.
- Steep geometry that fails the default policy exercises the disabled policy,
  with its distinct certification/report identity and no fabricated slope values.
- Slope-aware and disabled calibration/cost/route caches cannot cross-reuse.
- A valid extra SAM path mask outside base floor labels expands candidate support
  and can receive finite cost when geometric conditions pass; a rendered green
  overlay alone does not satisfy this test.
- VLM traversable target contributes support; non-traversable target contributes
  exclusions. The same synthetic mask exercises both interpretations distinctly.
- A path/positive mask overlapping a negative mask is blocked in costs, support
  indexes, endpoint selection and Python/native edge search.
- Positive masks cannot force finite cost past support/gap/clearance checks or
  enabled slope limits. No valid depth means no invented point rows.
- Disabling all additional semantic sources preserves the planner baseline.
- Reference and calibration source are explicit when base support is absent;
  invalid reference evidence gives an explicit no-route outcome.

### Workflow and UI tests

- Staged manual-obstacle integration before VLM model smoke tests.
- Real indoor and outdoor recordings using matching baseline/combined geometry.
- Cold/warm sessions, settings changes, pending-request drain and failure cleanup.
- Cancellation preserves only verified completed artifacts.
- Changed evidence invalidates costs/routes; old snapshot routes are withheld.
- Source RGB survives semantic coloring; display sampling never reaches planner.
- Camera updates remain available between map updates.
- Separate batch/live allocation owners cannot collide on the same GPU.
- Exercise all four slope/planner on/off combinations, including slopes-only and
  slope-disabled routes, and cost-map-only operation.
- Exercise each compute/reuse/off mode with valid, absent and stale inputs;
  reject dependency conflicts before allocating workers.
- Geometry-only, 2D VLM-only, manual SAM-only, obstacle-map-only, standalone
  planner and full combined runs produce only their requested stage artifacts.
- Feature changes start a new run/derived job; visibility changes do not restart
  processing. Disabled stages do not leave old layers/receipts appearing current.
- Turn on slopes or planning later against saved geometry without rerunning
  reconstruction or unrelated model stages.
- Extra SAM traversability prompts run with VLM off, empty, delayed, expired or
  failed; they remain in the next frame's query set and produce their own mask.
  A required VLM failure is still recorded and withholds incomplete combined
  certification rather than being hidden by successful extra segmentation.
- VLM queries and extra queries share one fresh image embedding per image-query
  stage. VLM TTL/max-prompts never expires/evicts independent fixed queries.
- Changing extra prompts invalidates their masks and downstream costs/routes but
  not geometry, VLM proposals or independent support masks. Changing VLM target
  invalidates its masks and dependent costs/routes, not independent extra masks.
- Combine non-traversable VLM + independent path queries, traversable VLM + path
  queries, extra-SAM-only mapping/planning and each slope/planner on/off mode.

GPU validation records environment versions, models, selected source frames,
prompt activation, thresholds, peak memory, geometry/map rates and mask/route
differences. Unit tests use synthetic/fake model responses and require no weight
downloads. Timing results from previous branches are historical baselines only.

## 11. Acceptance criteria for the initial combined release

1. Six component owners are explicit in directories, imports and documentation.
2. Planner core can run independently on aligned arrays; its default slope-aware
   policy retains the existing rules.
3. One offline command generates matching selected semantic sources on one
   reconstruction and produces certified routing or an explicit no-route outcome.
4. All-false obstacle evidence preserves baseline behavior; flagged obstacles
   affect both node eligibility and whole-edge clearance.
5. The default policy preserves reference/H calibration and blocked-nearest
   endpoint behavior; slope-disabled calibration is separately identified, and
   each policy freezes its baseline calibration before obstacle filtering.
6. Python and native certification agree, including partial routes.
7. Viewer shows source RGB, obstacle evidence, slopes, costs, camera and route
   from the same snapshot and hides stale results.
8. Models remain in their separate environments with centralized lifecycle and
   allocation coordination.
9. Original command entry points remain usable during migration.
10. Both regression suites and new cross-component tests pass; inherited CI is
    corrected to run the actual tests.
11. All components expose independent stage interfaces and can be composed by
    the orchestrator with compute/reuse/off modes and explicit dependency checks.
12. All four steepness/planning on/off combinations are supported. The default
    slope-aware policy preserves existing behavior; slope-disabled planning is
    explicit, tested and certified under its own policy.
13. Feature enablement and layer visibility are separate UI controls. Disabled
    stages load no unnecessary workers and publish no misleading old results.
14. Optional analysis can be added to saved geometry without rerunning upstream
    pipelines, and every reused input is validated against its consumed identity.
15. VLM target can be traversable or non-traversable with class-tagged instruction
    presets and the same explicit interpretation throughout the artifact chain.
16. Independent extra SAM prompts such as path run alongside VLM queries and
    without VLM, retaining separate configuration, masks, status and provenance.
17. CostMap consumes positive/negative evidence: positive sources expand support
    candidates, negative sources veto overlaps and affect node/edge clearance,
    and original geometric checks remain active under the selected slope policy.
18. Independent extra prompts are never dropped by VLM TTL, empty proposals or
    its query limit; cost/native restoration and route certification use the same
    fused evidence and report any missing required source.

Open implementation decisions: final import/package naming within the existing
path-planning folder; exact supported environment/checkpoint pins; and whether
incremental replanning is included after the initial offline/finalized-stream
release. These do not prevent the initial integration design above.

## Implementation resolution

The numerical package is at root path_planner/; path-planning/ preserves legacy
entry points, the native build location and regression tests. The six first-party
component packages and standalone/composed CLIs are implemented. Finalized causal
streams support optional analysis and routing; continuous prefix replanning is
not included. Model inference remains in separately supplied environments.
