# Six components and independent pipelines

This branch combines `path-planning-cleanup` (`708053a0`) and `vlm_improvements`
(`7f9a3a58`). Qwen remains Qwen3-VL-4B through Transformers. External model code,
weights and environments are separate from the numerical planner.

| Component | Public interface | Responsibility |
| --- | --- | --- |
| Path planner | `path_planner`, `python -m path_planner` | Pure numerical reference/slopes/costs/endpoints/Python and native route certification |
| VLM | `vlm`, `python -m vlm propose` | Typed traversable/non-traversable concept discovery |
| Orchestrator | `python -m orchestrator`; `orchestrator.streaming` | Model lifecycle, scheduling, evidence projection, artifact IO and composition |
| SAM | `python -m sam support` / `segment` | Base support and independently typed image-query segmentation |
| LingBot-MAP | `python -m lingbot` | Offline spread-anchor reconstruction; causal adapter in `lingbot.stream` |
| Visualizer | `python -m visualizer`; `visualizer.live_demo` | Artifact-only views and live processing controls |

The old `path-planning/`, `plugins/small_obstacles/` and PLY viewer commands remain
compatibility entry points. The first-party planner package is `path_planner/`;
`path-planning/` retains the native source/build location and regression tests.

## Semantic evidence and costs

VLM has an explicit `traversable` or `non_traversable` target. Each instruction
preset stores its class. Extra manual SAM traversability prompts such as `path`
are separate from generated concepts, remain active when VLM is off/empty/busy,
and do not share its TTL or prompt limit. SAM image features are computed once
per current image, and the unions remain separate by source and class.

```
positive = base floor/ground OR independent SAM positives OR VLM positives
negative = VLM negatives OR independent manual negatives
eligible support = positive AND NOT negative
observed obstacles = NOT positive OR negative
```

These masks affect finite costs, support neighborhoods, clearance and whole-edge
checks. Negative overlap wins. A positive label is support evidence; it does not
bypass geometric checks. Raw RGB, source masks, source frame/pixel IDs and camera
timestamps remain available. Planning uses every canonical point row; display
sampling is independent.

The default cost is `1 + slope penalty + obstacle clearance penalty`. Explicit
`slope_policy: disabled` removes both node/edge slope predicates and the slope
penalty, retaining support, gap and obstacle clearance. It does not fabricate
zero slopes and produces distinct costs/route provenance. Ground direction,
radius and camera-height clearance calibration remain required. With base support
off, supply a reference/radius/clearance height or explicitly select traversability
evidence for robust reference/calibration. Missing calibration/reference produces
an explicit no-route result.

## Environments

Keep existing LingBot, SAM and VLM model environments and checkpoints. Clone their
upstream sources into ignored `lingbot-map/` and `sam3/` directories. Install
`requirements/core.txt` for CPU analysis; the official projection helper also
uses Torch. Install `requirements/visualizer.txt` for views and OpenCV for video
sampling/live orchestration. VLM dependencies remain in
`plugins/small_obstacles/requirements-vlm.txt`.

Choose each model interpreter in configuration `runtime`. Optional
`lingbot_checkpoint` and `sam_checkpoint` select local weights. The controller
never initializes model weights itself. A shared user-wide GPU lock coordinates
offline jobs and persistent live sessions. Unload live models before starting
an offline model stage.

## Compose a run

Edit `examples/combined.json` for your input, new external output, model
interpreters and indoor `floor` or outdoor `ground` support prompt. Then:

```bash
python -m orchestrator run --config my-run.json --check
python -m orchestrator run --config my-run.json
python -m visualizer /absolute/path/to/new-combined-run
```

The `--check` command validates dependencies without launching workers or creating
output. Stage modes are `compute`, `reuse`, `off`. Disabled stages do not load
models or create analysis/route outputs. Required stages are not silently enabled.
Use a fresh output directory for each composition; reuse immutable stage artifacts
explicitly through `inputs`.

| Feature choice | Analysis settings | Planning settings |
| --- | --- | --- |
| Mapping only | steepness off, costs off | enabled false |
| Steepness only | steepness compute/reuse, costs off | enabled false |
| Cost-map only | costs compute/reuse, steepness chosen independently | enabled false; slope policy matches selected evidence |
| Route without slope checks | steepness off, costs compute/reuse | enabled true, slope_policy disabled |
| Slope-aware route | steepness compute/reuse, costs compute/reuse | enabled true, slope_policy enforce |

To analyze saved geometry, use `inputs.snapshot` with all model stages off, as in
`examples/reuse-without-slopes.json`. `inputs.steepness` and `inputs.costs` supply
reuse artifacts; their snapshot, implementation and policy identities must match.
Changed masks require a newly projected snapshot. Mixed computed/reused semantic
sources are independently validated and combined without enabling disabled ones.

### VLM + extra independent SAM path

```json
{
  "vlm": {"mode": "compute", "target": "non_traversable"},
  "sam": {
    "vlm_queries": "compute",
    "extra_traversability": {"mode": "compute", "prompts": ["path"]}
  }
}
```

This is a fragment to add to a complete run config. Set VLM target to traversable
to obtain another positive source. With VLM off, set `vlm_queries` off; extra SAM
queries still run. `manual_non_traversability` supports direct negative queries.

## Standalone stages

Each command consumes saved inputs and does not invoke unrelated stages:

```bash
# Geometry only, in the LingBot environment.
python -m lingbot --video_path VIDEO --output_folder GEOMETRY \
  --model_path lingbot-map/lingbot-map.pt --mode streaming \
  --num_scale_frames 8 --save_predictions --no_render --no_original_video

# Concept discovery only, in the VLM environment.
python -m vlm propose --lingbot-dir PREDICTIONS --output proposals.json \
  --target non_traversable

# Typed concepts alongside fixed positive path queries, in the SAM environment.
python -m sam segment --lingbot-dir PREDICTIONS --prompts proposals.json \
  --extra-traversability-prompts path --output-dir MASKS --frame-step 2

# Independent positive SAM queries without VLM.
python -m sam segment --lingbot-dir PREDICTIONS \
  --extra-traversability-prompts path --output-dir PATH_MASKS --frame-step 2

# Original floor/ground video support segmentation.
python -m sam support --video_path VIDEO --output_folder SUPPORT --prompt ground

# Projection only. Mask selection must match selected geometry frames exactly.
python -m orchestrator project --predictions PREDICTIONS --semantic-dir MASKS \
  --support-dir SUPPORT/VIDEO_masks --video-name VIDEO --support-prompt ground \
  --source-fps 30 --output SNAPSHOT

# Analysis/search stages independently. Reference/radius can be inferred from
# reliable base support; explicit values below are placeholders, not metric values.
python -m path_planner steepness --snapshot SNAPSHOT --output SLOPES
python -m path_planner costs --snapshot SNAPSHOT --slopes SLOPES --output COSTS
python -m path_planner route --costs COSTS --output ROUTE \
  --library path-planning/corridor_astar_feasible.so
```

VLM/SAM image-only jobs need RGB, not depth, in saved `frame_*.npz` inputs.
The composed runner can also sample video into RGB-only inputs when geometry is
off and projection is disabled. Standalone numerical APIs accept ordinary arrays.

## Live processing and controls

```bash
python -m visualizer.live_demo --port 8083 \
  --lingbot-python /path/to/lingbot/python \
  --vlm-python /path/to/vlm/python --sam-python /path/to/sam/python
```

The browser offers the VLM target, editable instruction/presets, independent extra
SAM traversability queries, base support segmentation, slope analysis, cost map
and route switches. Freeze computation settings at Start; visibility/color
controls can change during viewing. Models are reusable between healthy runs;
unload releases the allocation lock and GPU memory.

The causal CLI provides the same features:

```bash
python -m orchestrator.streaming --video-path VIDEO --output-dir NEW_RUN \
  --lingbot-python /path/to/lingbot/python --vlm-python /path/to/vlm/python \
  --sam-python /path/to/sam/python --vlm-target non_traversable \
  --extra-traversability-prompts path --support-prompt ground \
  --steepness --cost-map --plan --save-frames
```

Use `--no-vlm` for independent SAM queries; `--no-sam` for geometry/proposals only.
Omit `--steepness` for slope-disabled costs/routes. Analysis runs on the immutable
finalized map after processing, not continuously while the graph changes. The
offline spread-anchor and live causal reconstructions retain separate policies.
Continuous prefix replanning is a later extension, not part of this release.

## Validation

```bash
pip install -r requirements/test.txt
g++ -O3 -std=c++17 -ffp-contract=off -shared -fPIC \
  path-planning/corridor_astar_feasible.cpp -o path-planning/corridor_astar_feasible.so
export PYTHONPATH="$PWD:$PWD/path-planning:$PWD/orchestrator"
export DATASET_REPO="$PWD"
export DATASET_NATIVE_LIBRARY="$PWD/path-planning/corridor_astar_feasible.so"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -B -m unittest discover -s path-planning -v
python -B -m unittest discover -s plugins/small_obstacles/tests -v
python -B -m unittest discover -s tests/integration -v
```

Tests use CPU tensors, synthetic geometry and fake model responses; no weights
are downloaded. The official LingBot geometry source is required for projection
parity tests. CI checks it out at a pinned revision. Real model inference depends
on the separately installed model environments and available GPU memory.
