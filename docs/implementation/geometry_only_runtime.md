# Geometry-only and the common mapping runtime

The four public entry points are implemented with `semantic_mapping_v1`, schema
1. Exactly `geometry_only`, `ground_surface`, `fixed_hazards`, `qwen_hazards` are
registered lazily. Model imports and weights are unnecessary for common CPU
checks. The common runner owns sequence validation, geometry cache, scheduler,
per-concept fusion, robot costmap, plans, evaluator and comparison gates.

## Reproduce the tiny model-free comparison

From the repository root, using Python with NumPy and Pillow:

```bash
PYTHONPATH=src:VLM_evaluation/src python -m unittest discover -s tests/pipeline_common
PYTHONPATH=src:VLM_evaluation/src python -m unittest discover -s tests/pipelines
PYTHONPATH=src python -m pipeline_common.fixture --output outputs/pipeline_fixture_new
```

Choose a fresh output directory each time. On Windows, use a semicolon in
`PYTHONPATH` instead of a colon. The fixture generator creates three RGB frames,
explicit synthetic capture times, a metric analytic floor, an independent
robot-decision reference and a visibly synthetic circular robot profile. It
runs all four IDs through the actual run/evaluate/compare CLI files. Fixture
providers are never production fallback. Their model-call and text-query counts
are zero: the masks are analytically injected software evidence.

Generated `comparison/comparison.json` and `comparison/metrics.csv` are the
reviewable comparison. No automatic winner is produced. All fixture results
are software checks, not real-model accuracy, navigation or runtime evidence.

## Public CLI and a small KTH geometry run

The server supervisor must first check current CPU/RAM/disk/GPU allocation and
processes, use an isolated environment and one visible device, and preserve
completed component evaluations, credentials and caches. This implementation
performs no server connection or download. Existing source gitlinks remain
unchanged. Pipeline 1 requires the pinned LingBot source revision
`849e690bb086103637e44b1e91878d9d43a8bf0c`, local official `lingbot-map.pt`,
compatible Torch/Torchvision and `requirements/geometry_only_gpu.txt`. SAM3 and
Qwen weights are not needed for this pipeline. Checkpoint SHA256 and source/code
identity are recorded; a declared source/checkpoint identity mismatch fails.

Make a server-specific copy of `configs/pipelines/geometry_only.json` with
`geometry.source_root` and `geometry.checkpoint` set to existing absolute server
paths. Freeze the exact checkpoint SHA256 as `geometry.checkpoint_sha256` after
verification. Reuse the server's approved Torch/CUDA stack in the new isolated
environment; the earlier CUDA 12.8 example is not a requirement to alter the
completed hazard environment. This small command selects at most eight frames:

```bash
python src/prepare_sequence.py --input data/videos/indoor/IMG_5506.mp4 \
  --config configs/experiments/prepare_video_smoke.json \
  --output outputs/geometry_smoke_sequence
python src/run_pipeline.py --pipeline geometry_only \
  --sequence outputs/geometry_smoke_sequence/sequence.json \
  --config configs/pipelines/geometry_only.server.json \
  --output outputs/geometry_smoke
python src/evaluate_pipeline.py --run outputs/geometry_smoke \
  --output outputs/geometry_smoke_evaluation
PYTHONPATH=src python -m pipeline_common.viewer --run outputs/geometry_smoke
```

The video path is an actual restored repository recording; transfer is the
supervisor's responsibility. Another preset video may be substituted and must
receive its own sequence ID/output. Evaluation has no reference in this command,
so quality/accuracy/path-validity metrics are unavailable. Raw points are
`map/raw_surfaces.ply`; voxel evidence and verified W2C poses are numeric NPZ.
The optional viewer loads these saved artifacts only.

Metric scale, independently verified up/gravity, and the actual robot profile
are currently absent. The default config intentionally produces raw mapping
and an empty physical costmap, with one `blocked_inputs` outcome per requested
plan. An internal requested start/goal in unknown reconstruction units is never
interpreted as a physical safe route. To enable physical checks, supply an
external scale `{meters_per_unit,verified:true,source}`, up
`{vector,verified:true,source}`, complete versioned robot limits, anchored
support height and a frozen metric planning/evaluation grid. No test-reference
scale fitting is supported.

## Configuration and input bridge

`sequence.json` binds ordered frames to encoded-file SHA256 and decoded uint8
RGB C-order SHA256, named recipes, dimensions, unique IDs, monotonic known capture
timestamps and provenance. Video timestamps are decoder-reported presentation
times in the video timeline, not wall-clock recording dates. Folder timing must
be supplied or stays unknown. Numeric unknown timestamps are `-1`.

Run configs require contract/version. `pipeline_id` optionally validates the
chosen ID. Pipeline-owned settings can be in `pipeline` or `pipeline_config`,
or the appropriate entry of `pipelines`. Common settings are `geometry`,
`fusion`, `robot`, `planning`, `goals`, `evaluation_grid`, `hardware_budget`,
`semantic_keyframe_ids`, `runtime`. Resolved defaults/digests are saved once,
then augmented by verified producing geometry identity. Fake providers require
both CLI `--fixture` and explicit fixture sequence/config provenance.

Geometry cache reuse uses `--geometry-cache <dir>` and verifies actual source
frames, timestamps, transforms, model/checkpoint/code/config, calibration, pose
revision, archive bytes, arrays, processed pixels and reprojection. Points,
optical depth and camera translations receive the same scale. LingBot's official
depth/pose unprojection is reused. Source EXIF/crop/resize/padding is recorded;
artificial padding and cropped-out pixels contribute no mapping coverage.

Adapters use dataclasses in `pipeline_common.contracts` and synchronous
`create_adapter(config)`, `observe(FramePacket) -> SemanticFrame`, `close()`.
`pipeline_common.input_bridge.to_model_input(frame)` returns `(record,data_root)`.
`validate_model_input` returns the validated path; `load_model_rgb` supplies
pixels decoded from one owned, hashed byte snapshot. `input_contract_id` selects
generic runtime validation in both legacy readers. Benchmark `validate_frame`
still accepts only genuine RELLIS/COCO records. No fabricated dataset identity
is introduced. Every processed handoff is exact RGB PNG; JPEG cannot satisfy it.

`query_count` is actual SAM text execution count, or null with an explicit
`query_count_reason` for a crash/unauditable execution. Model calls and known
counts remain separate. Partial successful queries fuse independently; failed
partial masks remain diagnostic. Unknown nouns retain stable unmapped IDs.

## Mapping, planning and evaluation meaning

All required common artifacts are written. `run.json.status=complete` is the
atomic completion marker; directory existence has no completion meaning.
Completed/nonempty output directories cannot be overwritten. Every input has a
disposition; unavailable geometry/model inputs and failed attempts remain
explicit. Missing semantic models preserve successfully reconstructed raw maps.

Geometry mapping is surface-only: no free-space rays are inferred from missing
depth. Sparse missing voxels remain unknown. Geometry weights are capped per
frame/voxel. Semantic evidence is independent per concept, capped across
correlated pixels/aliases, and idempotent on identical replay. Only successful
queries contribute positive/observed evidence. Omitted concepts are unqueried.
Expiry makes semantics stale without clearing geometry; changed poses/cache
identity are rejected rather than silently mixed. The frozen
`fusion.semantic_threshold` defaults to 0.5 and controls semantic policy input.

The planner uses anchored lower support, slope/roughness, step/drop-off edges,
circular footprint, observed-shell height clearance and deterministic grid
search. Stacked floor/ceiling shells are kept distinct. Semantics cannot repair
geometry violations or missing support. Plans require XYZ endpoints in the
declared map frame and are internal static 2.5D diagnostics; unseen overhangs,
dynamics and real robot control are not certified.

The evaluator uses independent hashed assets and frozen coverage/ROI. Missing
predictions are unknown, missing references/empty denominators are null with
reasons, and geometry-only semantic quality is `not_applicable`. Robot reference
policy must match the exact robot; map/grid frame disagreements are rejected.
Common metric IDs/CSV and comparability fingerprints follow `shared_contract.md`.
Incompatible geometry, scale/up, fusion, robot, goals, ROI, mode, scheduling,
hardware budget, keyframes or semantic evaluation threshold refuse comparison.
Surface/hazard diagnostics are not averaged into one score.

`quality_replay` freezes geometry and semantic keyframes. `paced_runtime` exposes
one in-flight semantic observation plus one latest pending frame, immutable
frame/geometry joins, supersession journals, actual call start/completion and
mapped replay capture/fusion clocks. **The current production geometry adapter
returns a sequence batch.** These scheduling runs are explicitly recorded as
`cached_geometry_scheduler_replay`, even when the batch was reconstructed in
the same run. Live end-to-end latency, semantic update rate, and real evidence
age stay unavailable. A streamed geometry producer and joint residency run
remain required to establish actual end-to-end real-time capability. Requested
playback rate is never measured FPS. Audited model loading is separate from
synchronized call timing; unaudited/unsynchronized GPU durations are unavailable.
The common resource collector samples actual process RSS and reads actual
PyTorch allocator peaks across the run, with explicit scope, sampling limits
and unavailable states. Staged model residency is recorded; these observations
do not demonstrate simultaneous LingBot/SAM/Qwen fit. Component peaks are never
summed. The final completion marker is published after memory/artifact writes.

## Remaining validation

Real LingBot loading, checkpoint compatibility, GPU memory/latency, geometry
quality and the preset-video visual smoke remain server work. SAM/Qwen adapter
validation is owned by sessions 2–4. Physical planning needs the actual robot,
external scale/up and independent navigation references. The saved component
`hazard_prompt_v1` experiment remains unchanged. No training, tuning of weights,
bulk labeling, bulk downloads, deployment or remote workloads occurred here.
