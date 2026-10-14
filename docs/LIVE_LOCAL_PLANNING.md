# Live local planning in the visualizer

The `full-integration` worktree now provides **Framewise local planning** in
the existing visualizer. It updates bounded local terrain, costs, a reachable
camera-relative forward goal and a connected route while video processing runs.
It skips final-map exact analysis at EOF. The complete model pipeline has not
been tested after this integration, at the user's request.

**Final-map fast planning** is also available: it archives aligned projected
packets during reconstruction, then builds an approximate global grid capped at
512 x 512 cells after EOF. It chooses observed ground near recording endpoints
and validates its route against raw witnesses. It avoids dense per-point
slope/cost stages and exact global search. Resolution adapts to trajectory bounds.
Missing ground or disconnected components produce unavailable or explicitly
partial results. Exact analysis remains **Final-map exact analysis (slow)**.

Fast modes now use **Slope exclusion**: sparse 5 x 5 source-image plane estimates,
with a 9 x 9 fallback for undefined neighborhoods, only at retained ground
representatives. They exclude unknown or >=30-degree ground without a dense
per-point slope stage, KD slope refiner or slope cost penalty. SAM support and
extra traversability queries remain independent of this switch. Exact analysis
retains its original per-point stages separately.

A focused twelve-packet saved-data check produced ten valid connected live
updates after reference bootstrap and a validated fast-final route in about
1.1 seconds on CPU. This is a numerical component check, not a timing or success
claim for a complete recording/model pipeline.

## Start a run

Open the visualizer on port 8083. In Settings:

1. Select **Framewise local planning**.
2. Keep **Paced live sampling (2 FPS)** and **Map every: Selected frame** for an
   initial live run. Late input frames may be skipped; pending planning packets
   are bounded to one. Offline frame-count sampling is still available.
3. Select **Auto GPU with CPU fallback**. The GPU validator uses the LingBot
   Python environment; fallback and its reason are shown in Overview.
4. Enable SAM support or positive traversability sources. Enable **Cost map**
   and **Path planning**. Slopes remain independently selectable.
5. Press **Start processing** in Overview.

**Auto visible-ground lookahead** is on by default. In demo mode it extends the
camera-relative goal range and shifts the bounded window when the nearest
observed ground lies beyond the configured range. It remains capped at 48 support
radii; the grid capacity stays fixed. Disable it (CLI `--fixed-lookahead`) for a
fixed bound. Strict camera-ground requirements never adapt away missing support.

**Focus planned path** moves the view to the current route, or a previous trace
when no current route exists, and disables camera following. Magenta route lines
have a small display-only ground-normal offset to avoid point-cloud occlusion;
planning coordinates and validation are unchanged. Grey is historical, not a
currently valid route. The Overview explains why no current route is available.

The default **Visible ground (demo)** start chooses observed eligible ground;
it does not claim a verified connection from a robot position. **Strict camera
ground** instead requires support below the current camera and refuses a blocked
nearest observation. Either mode requires a validated graph connection to the
retained path; no straight connector is drawn across unknown or blocked space.

Visible-ground mode considers up to four observed components within one search
budget and requires at least half a support radius of forward progress from its
start. This avoids retaining a tiny isolated initial route. Anchors may associate
with nearby reachable samples within the original connection radius, rather
than requiring an identical grid cell; the resulting short connector must still
pass raw support, slope and swept-obstacle validation. Persistent vetoed anchors
are excluded. Searches also consider retained reachable anchors just behind the
camera, rather than always starting on a new visible patch. Raw geometry
failures provide edge/node diagnostics and trigger bounded replanning on the
current revision. A genuine loss of observed connection remains unavailable.

In Prompts, choose VLM `traversable` or `non_traversable`. Independent extra SAM
queries such as `path` work with VLM enabled or skipped. Positive masks are
unioned; VLM/manual negative masks veto overlaps. Disabling slopes removes only
their gates and penalties. Cost-only and slope-only computation do not start a
route search. **Final-map fast planning** uses the same independent switches;
**Final-map exact analysis (slow)** remains available separately.

## What the viewer shows

- Magenta: currently validated connected suffix.
- Yellow: observed reachable forward goal.
- Grey: retained inferred/proposed prefix; not confirmed robot execution.
- Local costs/slopes: separate bounded terrain overlay, selectable in Colour.
- Terrain green: eligible; grey: uncertain slope/height/ground; orange: steep;
  red: semantic or geometry obstacle; muted red: clearance; purple: failed route
  validation. Ctrl/Cmd-click terrain for the exclusion reason, slope and source
  frame. Higher finite costs use green-to-amber, rather than obstacle red.
- Status: calibration readiness, missing ground/heading/connection, rejected
  geometry, backend/fallback, revision and processing time.

Arrival of a newer aligned packet immediately hides an older active route.
Only a result matching the latest packet becomes active. Failed or unavailable
updates retain historical state but do not display an old route as current.

## Calibration and evidence

Startup requires three consistent observed-ground plane fits within the latest
five complete packets. The reference and support radius are then fixed for the
run. Radius can be supplied in model units; otherwise it is twenty times the
observed ground pixel spacing. Automatic clearance is an observed camera-to-ground
height proxy, not a calibrated robot body. Supply an explicit relative clearance
height when appropriate. A fresh run closes the old coordinate/reference epoch.

If the whole visible ground fails the plane-support requirement, fast modes try
bounded near-field ground crops. Each crop retains the same plane-quality gates,
and three distinct source frames still must agree within five degrees. A crop is
not counted as another observation of its parent frame. Radius estimation keeps
the whole-ground pixel spacing rather than shrinking to the cropped near-field
spacing. Waiting status exposes suitable-frame counts and calibration diagnostics.

The rolling grid has 200 x 200 cells at radius/8 resolution, biased ahead of the
camera. Compatible observations refresh their actual source-point representatives
and source-frame angles together, reducing seams between frozen older coordinates
and new observations. Different height layers are not averaged. Native routing
can connect over up to four cells within the support-radius scale; every crossed
cell must be observed and eligible, including diagonal corner cells. Raw support
and upward clearance are validated afterward. Raw evidence is limited to twenty
processed packets. Input queues cannot grow with video length.

Geometry-only height disagreements are rejected on that update, then may
recover when a later observation confirms the new surface. They do not become
permanent semantic negatives. Newer compatible ground evidence can correct an
older untyped floor label at the same height. VLM/manual negatives remain vetoes.
Elevated geometry requires two distinct newer observations with depth rays through
the old point ending farther away before recovery; positive ground labels alone
do not clear it. Historical untyped raw samples require the same two-ray evidence
before removal from validation. Source identity and the latest confirmation time
remain separate. Per-update diagnostic events are saved under
`local_planning/events.jsonl` and include independent exclusion counts.

GPU validation uses FP64 support-count, continuous support-gap and upward-swept
obstacle predicates. Slopes use the inexpensive cached source-frame screen; the
full slope fitter is not invoked in fast modes. CPU validation remains available. These are relative
geometry checks; metric robot dimensions, execution feedback and kinematic
constraints are not inferred.

**Record dense map and frame evidence** is off by default in framewise mode.
The accumulated bounded display and current RGB/mask preview remain available.
Enable recording when complete raw frame/map artifacts are needed. Exact mode
keeps its original recording/finalized-analysis behavior.

## Component ownership

`path_planner/local_grid.py`, `local_route.py` and `gpu_geometry.py/.cu` own
numerical costs, connected proposals and validation. `orchestrator/live_planning.py`
and `local_evidence.py` own packet lifetime, reference bootstrap, rolling fusion,
bounded scheduling and the optional GPU worker. `visualizer/local_planning.py`
renders arrays and validity. LingBot, VLM and SAM keep their existing model
adapters and source-separated masks.

`path_planner/final_grid.py` owns recording-endpoint grid routing;
`orchestrator/fast_final.py` owns the direct packet archive and final-grid build.
Final fast mode screens final retained representatives once, rather than fitting
every archived packet's dense points. Demo initialization skips unproductive
nearby islands within the nearest observed-ground neighborhood; strict start
requirements stay unchanged. A candidate with failed edges can retain a useful
prefix only after that prefix separately passes raw validation. It is explicitly
partial and never crosses the failed edge. Final fast mode never invokes
`orchestrator.analysis.costs` or the exact route search. Its raw packet archive is needed for final validation even when optional
dense PLY/full RGB-depth recording is off.

Install optional GPU dependencies in the GPU worker environment:

```bash
.venv-gpu-validation/bin/python -m pip install -r requirements/gpu-planning.txt
```

The native grid-search library is compiled once from `local_grid_search.cpp`
with `g++` and cached by source hash under `~/.cache/embodiedai/native/`.

Focused checks cover numerical joins, negative-veto retention, nearest blocked
ground, cost-only mode, diagonal barriers and pending-packet supersession. No
complete video/model pipeline test was run after integration.

## SAM-only fix checks, 2026-10-14

Saved projected evidence from IMG_5700 and IMG_5702 was replanned without model
inference. Eligible cells increased from 5,410 to 18,659 and from 5,878 to 28,268.
Undefined-slope exclusions fell to one and two cells respectively. Both recordings
returned raw-validated **partial** routes: four vertices for the junction and
seventeen for the straight path. Complete routes remain unavailable where raw
support/geometry is inconsistent. Replanning took about 8.0 and 14.6 seconds;
these are full-recording final-map stages, not per-frame planning times.

A separate twelve-update local test used the straight clip's recorded fixed
reference, radius and height to isolate planning from reference bootstrap. At
16-radius lookahead, nine updates were ready; one was geometry-rejected, one had
no goal ahead and one lost connection. Warm updates took about 61–106 ms; the
first GPU worker start took about 619 ms. This is not an end-to-end FPS claim or
a guarantee of continuous routes. A 12-radius lookahead excluded the earliest
observed ground in the first three updates. The UI range remains independently
adjustable; increase it toward 16 when the camera sees ground only farther ahead.

The same independently validated prefix fallback is now available during live
planning (`partial_ready`), while preserving its retained-path anchor. The previous
validated trace is stored for grey display when the newest revision is pending or
unavailable; it is never relabeled as current. Thirty-one focused numerical and
viewer checks cover these behaviors without model inference.

The subsequent IMG_5511 run had no path because its ground reference never
initialized. Its saved last-frame whole-ground plane support was 76.3%, below the
85% requirement; a near-field crop had 99.05% support. A single-frame diagnostic
with that supplied observed local reference and adaptive lookahead produced a
raw-validated ten-point route. This isolates planning on saved evidence and does
not verify multi-frame startup on the complete recording. The visualizer's saved
preview labels that limited scope explicitly.

The earlier [SAM-only baseline analysis](SAM_ONLY_PLANNING_ANALYSIS.md) and its
figures are retained for comparison. New saved-data artifacts are under
`/home/jovyan/EmbodiedAI-gpu-validation/sam-planning-fix-validation/`.

## Lower reconstruction layers

Fast live and final planning now enable **Remove lower ground copies** by
default. Within each observed grid column, the filter selects an actual source
point from the highest sufficiently supported SAM-positive ground cluster below
the camera. It does not average stacked floors or select arbitrary high objects.
If the camera is underneath the previously selected surface, lower observed
ground can remain usable, preserving legitimate underpasses.

A replacement height needs two distinct source frames before the old lower
copies are excluded from raw planning support and the displayed map. Filtering
is restricted to confirmed columns in the observed window; semantic negative
evidence continues to veto traversability. Original reconstruction files remain
intact. **Show raw reconstruction layers** restores the unfiltered cloud for
comparison. This handles duplicate ground in planning; it does not repair
LingBot's camera poses or reconstruction itself.

When most of the remaining inferred demo path lies below corrected ground,
**Replan after ground-layer correction (demo)** archives that path under
`local_planning/retired_paths/` and starts a new planning epoch on the corrected
surface. It never draws a vertical connecting bridge. Disable this option to
retain the previous anchor; strict planning never restarts automatically.
CLI equivalents are `--keep-lower-layers` and `--keep-layer-path-anchor`.

Thirty-eight focused checks cover stacked surfaces, camera-height changes,
distinct-frame confirmation, negative vetoes and path-epoch retirement. A saved
IMG_5511 single-frame component check with a supplied recorded ground reference
returned a raw-validated 22-vertex route in about 720 ms, including cold GPU
worker startup. This is not a complete recording or model-pipeline test.

## Forward frontier and continued demo planning

The IMG_5511 run `81cbd6a6f75a` initially produced a four-point route, then
reported `connection_lost` from revision 16 through EOF. It still had 13,433
eligible cells at EOF; the retained old route had left the planning window.
Lower-layer filtering alone did not resolve this stale inferred anchor.

Auto visible-ground lookahead now targets the furthest reachable observed
frontier, within a 48-support-radius bound, using a fixed 224-square grid with
radius/4 cells. The window includes near ground and the frontier together.
Coarser cells improve far-field source density without changing raw support,
negative veto, clearance or grade validation. Detours use the same obstacle
exclusions. Shared predecessor chains join at their common ancestor, avoiding
an unnecessary return to the original start before proceeding forward.

Connected updates retain the exact shared prefix. When two distinct source
frames report a lost connection, **Continue on new visible ground (demo)** can
validate a fresh local segment. Only a successful raw validation retires the old
inferred route into `retired_paths/` and starts a new epoch. No connecting bridge
is fabricated. Strict mode and `--keep-disconnected-path-anchor` retain the
failure instead. A robot-position connection requires execution/localization
feedback; a video-only demo cannot guarantee one through reconstruction drift.

A bounded confirmed-ground atlas preserves display filtering outside the
rolling window; it never supplies planning support after raw witnesses expire.
The default **Show full LingBot map** view preserves the complete accumulated
reconstruction. Disabling it hides only SAM-positive ground below a confirmed
surface; untyped RGB geometry, upper ground and semantic negatives remain.
Planner layer filtering is independent of this display choice. This changes
display, not the saved raw map.

Goals require support in the current source frame, rather than merely appearing
in the retained old map. Forty-four focused checks pass, including forward range, a raw-validated obstacle
detour, distinct-frame recovery, failed-recovery preservation, strict-mode
behavior and filtering beyond the local window. On the saved IMG_5511 frame,
radius/8 and radius/6 grids returned validated paths extending 18.8 and 21.8
support radii ahead; radius/4 extended the validated prefix to 23.6 radii.
Cold checks took about 0.8–0.9 seconds including GPU worker startup. The supplied
reference and prior recorded disconnection isolate planning from model inference;
these are component checks, not a complete recording test or end-to-end FPS.

## Accumulated-map display regression

The briefly introduced current-frame RGB / cleaned-ground-history composite
discarded older untyped RGB geometry whenever the scene advanced, creating
disappearing objects and an inconsistent map display. It has been removed.
IMG_5511 runs `81cbd6a6f75a` and `529ba005fbd8` have bit-identical saved depth,
depth confidence, intrinsics, extrinsics and accumulated 200,000-point display
clouds. That confirms the newly reported inconsistency was introduced by the
display composition rather than a change to LingBot in these two runs. It does
not establish that the raw reconstruction is free of drift or duplicate layers.
Forty-seven focused checks pass, including preservation of accumulated objects,
semantic negatives, upper ground and unconfirmed columns during display updates.

## Tiny path appearing only at EOF

IMG_5511 runs `a418c5d7ebdc` and `eb2edc07c2e8` repeatedly reported
`no_goal_ahead` despite thousands of eligible cells. Routing searched a stale
near-camera component of 148 cells while goals required current-frame support.
The final five-point route also began on older ground behind the currently
visible patch. Demo seeds now require currently observed support, prioritize
current starts over retained-anchor starts, and search retained path connections
separately. Strict camera-ground association remains unchanged.

Completed planning results were also discarded whenever capture submitted a
newer packet before processing finished. At two FPS this could hide every
validated update taking more than 500 ms until EOF. Completed validated paths
now commit their connected prefix and remain visible as grey history while
newer evidence is pending. Delayed results never become current active routes;
the UI reports the lag in mapped frames. Capacity-one pending work still skips
intermediate packets rather than building a queue.

A raw rejection at the old anchor now counts as lost continuity, so the existing
demo-only recovery can restart after two distinct failing frames. Successful
raw validation is required before archiving that path. Strict mode and disabled
demo recovery preserve the anchor and the failure.

Raw evidence and its GPU index can be reused for prefix/retry checks only within
the same terrain update, slope array and original evidence bounding box. A new
frame, changed context or expanded box forces a fresh collection/index. Positive
and negative masks, support, obstacle clearance and grade predicates are
unchanged. Cached GPU checks matched fresh checks for supported full and partial
routes and an intentionally invalid route.

Fifty-three focused checks pass, including stale start islands, delayed route
publication, raw-rejected anchors and cache bounds/frame invalidation. A saved
IMG_5511 fused final grid plus current-frame raw evidence now yields a validated
42-point prefix instead of the displayed five-point route. This diagnostic uses
the recorded reference and does not replay the complete live history. On the
first twelve archived IMG_5702 projected packets, a 64-point validated partial
path appeared on the third update after calibration; six updates produced routes
and four later updates remained geometry-rejected. Warm median processing time
fell from about 527 to 398 ms with evidence/index reuse. These are saved-packet
component checks without model inference, not a complete live-pipeline test or
a guarantee of uninterrupted routes.
