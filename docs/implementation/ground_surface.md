# Ground-surface adapter

`ground_surface` records candidate ground or floor evidence from one frozen SAM3
phrase on the exact LingBot processed RGB grid. The common package owns geometry,
voxel projection/fusion, scheduling, robot costs, planning and evaluation. Strong
surface evidence cannot repair slope, step/drop-off, footprint, support or
clearance failures. An empty successful query supplies no positive surface
evidence; it never creates free space or declares an obstacle.

## Configuration and identity

The common runner passes the resolved `pipeline` section to `create_adapter`.
The named run files retain the common geometry-only starting settings and frozen
visualization ROI, with no fabricated robot profile. Replace local checkpoint
paths for an authorized server run, or use the validated replay helper below to
prepare the surface section from a completed common geometry run.
The three public configurations are:

| File under `configs/pipelines/` | Exact phrase | Recorded config ID |
|---|---|---|
| `ground_surface_outdoor.json` | `ground` | `ground_surface_outdoor_v1` |
| `ground_surface_indoor.json` | `floor` | `ground_surface_indoor_v1` |
| `ground_surface_path_legacy.json` | `Path` | `ground_surface_path_legacy_ablation_v1` |

Every query retains its original phrase and has `concept_id=ground_surface`,
`role=candidate_surface`, and `mapping_version=ground_surface_v1`. New runs use
`adapter_id=ground_surface_sam3_v1`. The proposed starting terms have no measured
advantage over `Path`. Choose a term on scene-disjoint development data and save
the frozen config before held-out evaluation.

`development_prompt_variants(config, sequence_manifest)` declares all three
variants against the identical frozen development-frame digest and SAM settings.
It rejects test sequences and does not inspect labels or choose a winner. Run
each variant with the same geometry cache, keyframes, independent reference,
fusion, robot and evaluator inputs. Preserve the resulting reports as a prompt
ablation; do not select a term using test labels or extend this surface adapter
into the hazard vocabulary.

## Reuse, alignment and failures

This adapter extends the previous `src/path_mapping` baseline through the common
LingBot geometry/cache bridge. It uses the existing compatible, checkpoint/code
verified `traversability_hazard_segmentation.sam3_adapter` processor to retain
individual instance masks and scores instead of discarding them into a dense
aggregate. Loader policy metadata is retained as component provenance; the
surface semantic policy is separately identified as `single_candidate_surface_v1`.
Runtime queries do not use hazard annotations, reference-present prompts or the
hazard controller.

The common bridge verifies lossless RGB PNG pixels, encoded-file SHA256 and
decoded uint8 RGB SHA256, dimensions and processed-grid identity. Both file-backed
model readers use explicit `input_contract_id=semantic_mapping_v1`; a phone/video
frame is never disguised as RELLIS/COCO. Original image identity and crop/resize
transform remain in each semantic record. The adapter does no independent crop
or resize and never substitutes a newer frame's pose.

One `segment_image` invocation and at most one actual text query occur per selected
frame. Successful empty queries are `ok`; an image/reset/query failure is `error`.
If a backend crashes before returning an audited count, `query_count` remains
null with an explicit reason; it is not rewritten to zero.
Valid instances returned before a query failure remain diagnostic under an
errored query. The shared fuser ignores them. No all-query union supplies fusion.
Overlapping valid instances remain separately inspectable while common fusion
caps correlated pixels/instances to one contribution per frame/concept/voxel.
SAM scores are uncalibrated evidence weights, not traversability probabilities.

Importing the adapter and constructing its settings loads no Torch/model.
Checkpoint loading is lazy and local-only. The first observation records loading
separately from the SAM invocation in `adapter_provenance.timing`; real calls use
explicit CUDA synchronization. First calls are marked cold; warmup is not claimed.
These fields do not by themselves establish live end-to-end speed or joint memory.
`close()` releases the owned SAM reader and is idempotent.

## Legacy Path artifacts

The old exact-`Path` CLI, `sam3_lingbot_path_v1` identity, scores, voxels and viewer
remain intact. New `Path` runs are explicitly named ablations and do not relabel
old files as `ground` or `floor` observations.

`inspect_legacy_path_archive(geometry_path, scores_path,
expected_geometry_fingerprint=...)` reads the old schema without modifying it.
It uses the existing geometry/score loaders, verifies grid/array/RGB/pose/order
fingerprints and exact case-sensitive `Path`, and records SHA256 of both archive
files. Old score arrays are maximum-over-instance aggregates. Original instances,
query statuses and checkpoint hashes cannot be recovered; the audit explicitly
declines to feed these aggregates into new per-instance semantic fusion. Replay
them through the unchanged legacy CLI when that legacy representation is required.

## CPU checks and common artifacts

From the repository root with NumPy and Pillow installed:

```bash
PYTHONPATH=src:VLM_evaluation/src python -m unittest discover \
  -s tests/pipelines -p test_ground_surface.py -v
PYTHONPATH=src python -m unittest discover -s tests/path_mapping -v
```

The tests use the actual shared dataclasses, hash bridge, native SAM3 fixture
processor, common fuser and common planner. They cover all three named configs,
original phrase/role/identity, PNG/hash/grid rejection, successful-empty failures,
partial diagnostics, actual query counts, overlap/dedup/distinct-frame support,
legacy archive rejection and cleanup. A geometry conflict test supplies an extreme
candidate-surface prior to a steep support slope and still receives `blocked`.
Fixtures execute no learned model and demonstrate no segmentation or navigation
accuracy.

The adapter and offline replay helper have 35 CPU checks (19 adapter, 16 helper).
The bounded KTH pilot runner has 38 separate operational checks for resource
accounting, protected scopes, preservation, resume and diagnostics. These checks
establish software behavior and are separate from real-model evidence.

An inspectable actual-adapter fixture is saved under
`outputs/ground_surface_fixture_adapter_v1/`. Its `run/` uses session 1's analytic
sequence and verified common geometry cache, native fixture SAM3 processor, shared
fuser/planner and the common artifact layout. Its `evaluation/` uses session 1's
independent synthetic robot reference; `evaluation_without_references/` keeps
quality metrics null. `comparison/` matches the other three common fixture
pipelines with no comparison differences and no automatic ranking. These are
CPU software artifacts only; the processor executes no learned weights.

Use the public common runner/evaluator for the same layout:

```bash
python src/run_pipeline.py --pipeline ground_surface \
  --sequence <sequence-dir>/sequence.json \
  --config configs/pipelines/ground_surface_outdoor.json \
  --geometry-cache <verified-common-cache-dir> --output <new-run-dir>
python src/evaluate_pipeline.py --run <new-run-dir> --output <evaluation-dir>
```

For a synthetic sequence, the common runner requires both `--fixture` and an
explicit fixture run config; it selects its named fake provider, never a real
model fallback. Adapter tests additionally route the native fixture SAM processor
through the actual runner. The same map/concept/contribution, semantic mask,
planning and evaluator schemas apply. Without independent references, quality
metrics remain null/unavailable. Without verified scale/up and an actual robot
profile, physical planning remains `blocked_inputs`.

## Completed bounded KTH pilot

On 2026-10-02, pipeline 2 completed the authorized KTH pilot in
`/home/jovyan/EmbodiedAI-pipelines/outputs/kth_pipeline2_pilot_v5_20261002`:
22 stages, five model subprocesses, exit code 0. Pipeline 2 owned server setup
and browser/GPU execution; root orchestrated and reviewed. The isolated
`.venv-pipelines` used Python 3.12.14, Torch 2.10.0+cu126, torchvision 0.25.0+cu126
and NumPy 1.26.4, with four CPU threads and one assigned H100 MIG 2g.20gb device.
All runs are explicitly nonfixture and record `models_executed=true`.

Only indoor `IMG_5508.mp4` and outdoor `IMG_5513.mp4` were used. The smoke selected
the first two adjacent indoor frames. The two development clips each selected
nine frames at 2 fps; LingBot used an eight-frame scale window, followed by the
streaming frame. SAM replay used the exact saved 518×518 RGB PNGs and geometry
cache, with matching input/archive/geometry/grid identities.

| Clip | Frames | SAM phrase | Map voxels | Retained SAM instances | Nonzero positive-evidence voxels | Voxels meeting evidence threshold 0.5 |
|---|---:|---|---:|---:|---:|---:|
| Adjacent indoor geometry smoke | 2 | — | 706 | — | — | — |
| Indoor geometry + surface replay | 9 | `floor` | 2,980 | 1 | 13 | 0 |
| Outdoor geometry + surface replay | 9 | `ground` | 2,233 | 11 | 227 | 181 |

Both surface runs have nine `ok` frames, nine `ok` queries and one actual text
execution per frame. The indoor instance occurs in one frame; the other eight
queries succeed with empty results. Semantic row counts (2,980 indoor, 2,233
outdoor) include observed coverage with zero positive weight: 2,967 indoor rows
and 2,006 outdoor rows. They are not counts of detected floor/ground voxels.
The 13 indoor positive voxels each have one positive frame contribution, and
their evidence scores are below the frozen 0.5 threshold. Evidence is an
uncalibrated statistic, not a safety probability.

The inspected indoor frames show a stairway. Sparse `floor` output on this clip
does not establish performance on flat indoor floors; no reference labels were
provided for the stairs or other visible surfaces.

Internal depth/pose/reprojection checks pass on all valid geometry pixels;
maximum reprojection residuals are below 0.0002 pixels. This checks geometric
consistency and alignment, not accuracy against the real scene. Each clip's
geometry/surface comparison is compatible with no differences and no automatic
ranking. Independent references are absent; semantic/safety/navigation quality
metrics remain null. Scale/up are unverified, `robot=null`, and physical planning
is `blocked_inputs` in every run.

The saved static viewers are available in authenticated Jupyter:
[indoor floor report](https://gpu1.eecs.kth.se/user/tombro/files/EmbodiedAI-pipelines/outputs/kth_pipeline2_pilot_v5_20261002/indoor_nine_frames/ground_surface_report/report.html)
and [outdoor ground report](https://gpu1.eecs.kth.se/user/tombro/files/EmbodiedAI-pipelines/outputs/kth_pipeline2_pilot_v5_20261002/outdoor_nine_frames/ground_surface_report/report.html).
Both direct viewers were inspected in the browser. Proof images are saved in
`outputs/kth_pipeline2_execution/indoor_floor_viewer.jpg`,
`outdoor_ground_viewer.jpg` and `outdoor_ground_evidence.jpg`.
The canonical local audit is
`outputs/kth_pipeline2_execution/v5_evidence/logs/pilot_v5_final_evidence.json`;
supplemental semantic counts are in
`outputs/kth_pipeline2_execution/pilot_v5_semantic_evidence.json` and the direct
contribution audit `.codex-local/handoffs/pipeline2_result_review.json`.
The metadata/log evidence ZIP is `outputs/kth_pipeline2_execution/pilot_v5_evidence.zip`,
SHA256 `43ade17a41d3286d8046292baea8e856e3b83f5a9fefe5c24e4c56abe18d8693`.
The complete PNGs, instance masks, geometry/maps and viewer HTML remain on KTH.

## Recorded KTH commands

The resource-checked runner used the two explicit recordings. Its local
`.codex-local/kth-server/pipeline2_pilot.py` source was deployed as
`scripts/kth/pipeline2_pilot.py`:

```bash
cd /home/jovyan/EmbodiedAI-pipelines
PYTHON=/home/jovyan/EmbodiedAI-pipelines/.venv-pipelines/bin/python
PILOT=/home/jovyan/EmbodiedAI-pipelines/outputs/kth_pipeline2_pilot_v5_20261002
"$PYTHON" scripts/kth/pipeline2_pilot.py \
  --indoor-video /home/jovyan/EmbodiedAI/data/videos/indoor/IMG_5508.mp4 \
  --outdoor-video /home/jovyan/EmbodiedAI/data/videos/outdoor/IMG_5513.mp4 \
  --output "$PILOT"
```

Within that runner, the indoor saved-geometry replay used the verified helper
and public common interfaces below. Outdoor replay used `outdoor_nine_frames`
and `--environment outdoor`, which selected the frozen `ground` prompt.

```bash
CLIP="$PILOT/indoor_nine_frames"
"$PYTHON" docs/implementation/prepare_ground_surface_replay.py \
  --geometry-run "$CLIP/geometry_only_run" \
  --sequence "$CLIP/sequence/sequence.json" --environment indoor \
  --output "$CLIP/ground_surface.config.json"
"$PYTHON" src/run_pipeline.py --pipeline ground_surface \
  --sequence "$CLIP/sequence/sequence.json" \
  --config "$CLIP/ground_surface.config.json" \
  --geometry-cache "$CLIP/geometry_only_run/geometry/cache" \
  --output "$CLIP/ground_surface_run"
"$PYTHON" src/evaluate_pipeline.py --run "$CLIP/ground_surface_run" \
  --output "$CLIP/ground_surface_evaluation"
"$PYTHON" src/inspect_pipeline_results.py --run "$CLIP/ground_surface_run" \
  --report "$CLIP/ground_surface_evaluation/report.json" \
  --output "$CLIP/ground_surface_report"
```

These are recorded commands with completed output paths. Preserve them; a new
attempt requires new config/run/evaluation/report paths and fresh resource checks.
The helper verifies existing pinned SAM assets offline and preserves the frozen
geometry, sequence, keyframes, robot, ROI, fusion and evaluator settings. The
output retains individual masks in `semantics/masks/*.npz`, original phrases and
provenance in `semantics/frames.jsonl`, and query/pose contributions in
`map/contributions.jsonl`.

This pilot demonstrates real pinned model loading and aligned saved-cache replay
on two short development clips. It supplies no controlled `Path` prompt ablation
or segmentation/reconstruction accuracy, verified metric scale/up, actual robot
parameters, independent video references, live end-to-end throughput, joint
model residency validation or navigation/controller validation. The completed
KTH hazard image benchmark remains separate evidence and is preserved unchanged.
