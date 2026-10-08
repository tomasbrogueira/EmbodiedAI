# Short-horizon corridor demonstration

The 5506 replay plans the continuing mission **forward along the recorded corridor**.
The mission direction is fixed in the world frame. Recorded camera turns do not
silently change that objective. A temporary reachable frontier provides a local
planning target; reaching it or ending the recording does not complete the mission.

For each saved observation, the demonstration admits only that frame and its
predecessors to a fresh surface costmap. It follows a simulated ground agent,
recomputes a supported eight-neighbour Dijkstra route and checks each edge and
shortcut with the complete circular robot footprint. The saved full route is
internal. The video and inset show its next **at most 2 metres of 3D arc length**,
with an interpolated endpoint when the horizon falls inside a segment. A shorter
supported route is displayed at its actual length.

The presentation uses full portrait footage with a small map inset. On-screen
text is limited to the scale marker and the final awaiting-observations status.
Explanations, assumptions and provenance are retained in the saved receipts.

The camera's first underfoot floor is unobserved. The simulated robot is therefore
initialized on the first observed forward floor patch that passes the footprint
checks. Subsequent simulated horizontal positions add the recorded camera
displacement to that initial position. No unsupported camera-to-agent connection
is created. This placement and movement are explicit research assumptions.

At recording EOF, the final next-step path remains saved and visible in a paused
hold. Export status becomes complete; mission status stays active and execution
awaits observations. The hold neither creates new observations nor extrapolates
robot movement. Unknown or blocked cells cannot provide invented continuation.

## Scope

This is an **observation-prefix map/planner replay using cached geometry**.
LingBot's cached initialization can depend on later initialization frames, and
the up/scale reference is a fixed offline preset from the corrected review. The
demo consequently does not establish causal live perception, inference latency,
physical metric calibration, free space or robot control. The geometry-only
research profile and its assumptions remain the same as the earlier 5506 review.
Saved SAM3 masks provide visual surface evidence; they do not choose these routes.

## Files and interfaces

- `src/build_navigation_replay.py` verifies saved sources and constructs prefix
  costmaps plus per-frame routes in a new output directory.
- `src/pipeline_common/navigation_replay.py` implements checked frontier planning
  from the supplied supported ground position and exact arc-length truncation.
- `src/export_navigation_replay_demo.py` admits the local replay, source video,
  geometry, reference plan and masks, then exports an annotated movie.
- `src/export_segmentation_route_recording.py --replanning-replay ...` admits the
  replay separately from the original final-map `--research-plan` contract.

The original final-map plan loader, physical-calibration gate and projection
function retain their contracts. New artifacts identify `research_navigation_replay`,
`observation_prefix_map_replay`, `cached_geometry_replay`, the persistent mission,
the exact source frame/time and the last admitted observation index. Source and
costmap hashes are retained for review. Every output is separate from original
runs and published video/map pairs.

## Multiple recordings

The builder accepts any saved recording identity, with at most 256 sampled
observations. Each recording must supply its own identity-bound corrected
manifest and research reference plan. The approved camera-height assumption is
1.5 m; the level-floor declaration belongs only to 5506. Other recordings retain
the assumed upright first-camera up direction independently of the terrain
plane, so slopes and steps are not flattened. These remain research assumptions.

```text
python src/build_navigation_replay.py --geometry ORIGINAL_CACHE_GEOMETRY \
  --reference-plan REFERENCE_PLAN --sequence SEQUENCE \
  --geometry-manifest CORRECTED_MANIFEST --output FRESH_REPLAY_DIRECTORY
```

The default input is the original saved archive. The shared
`numeric_for_derivation` helper applies only its declared derivation. To admit
an already corrected archive, add `--geometry-kind corrected_w2c`; its hash and
numeric fingerprint are checked and its poses are never inverted again.

Early missing-support observations produce `awaiting_support` rows with null
ground position and empty paths. Initialization retries a currently observed
forward patch; recorded displacement starts at the actual initialization
observation. Subsequent loss of support never relocates the agent. Missing
ground/scale references also produce route-free unknown-only display grids.
Missing up may use a marked upright-first-camera display basis, creating no floor.

Explicit unverified-pose copies must declare `pose_correction_verified=false`
and `method=saved_pose_passthrough_unverified_display_only`. Every frame then
has `status=pose_unverified`, null ground position and empty paths, even if a
planar fit would otherwise appear traversable. The continuing mission stays
active and awaits evidence.

The optional `--batch-config CONFIG --output FRESH_BATCH_DIRECTORY` uses the
schema below, with paths relative to the config. Each recording gets its own
directory and receipt. Failed admissions retain errors without modifying saved
inputs. No assumptions are copied between recordings.

```json
{
  "schema_version": 1,
  "artifact_kind": "research_navigation_replay_batch_config",
  "research_illustration": true,
  "recordings": [{
    "recording_id": "outdoor_IMG_5518",
    "geometry": "clip/original_cache/geometry.npz",
    "reference_plan": "clip/reference/research_plan.json",
    "sequence": "clip/sequence.json",
    "geometry_manifest": "clip/derived/geometry/manifest.json"
  }]
}
```

Every row retains exact `accepted_prefix_points` and `occupied_prefix_voxels`,
including zero-occupancy prefixes. Batch completion describes artifact production
and never completes a navigation mission. This is CPU processing of cached
arrays; it executes no model inference or robot commands.
