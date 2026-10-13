# Pre-implementation framewise benchmarks

The implementation plan is saved in Git. These are experimental CPU-only
benchmarks outside the application, not an integrated real-time feature.
Production Python/C++ code, model weights and the running server were unchanged.

## Inputs and reproduction

Experiments used the real-model saved IMG_5701 stream (56 mapped frames,
approximately 26 seconds of source video) and six mapped IMG_5516 frames tested
under both VLM-negative and VLM-positive semantic configurations. The latter
reuse the same geometry; this is two clips and three semantic configurations,
not three independent recordings. Independent SAM `path` masks are present in
the IMG_5516 runs.

All benchmark inputs were existing checksum-verified map artifacts. Calibration
was supplied from the saved profiles for timing comparisons. This is not a
causal online calibration test or a metric robot-body demonstration. The
prototype starts on nearest visible ground; strict ground association under the
camera is reported separately and was unavailable in the per-frame grids.

CPU allowance: 24 cores. NumPy/SciPy/BLAS were restricted to one thread. Native
grid search used `g++ -O3 -std=c++17 -shared -fPIC`. No GPU inference was run.
Scripts, source hashes, settings, timings and JSON results are saved at:

`/home/jovyan/EmbodiedAI-gpu-validation/framewise-benchmarks/`

Example reproduction, from the integration checkout:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-integration/bin/python \
  /home/jovyan/EmbodiedAI-gpu-validation/framewise-benchmarks/refinement.py
```

`manifest.json` fingerprints the experimental scripts and native source/library.
`round1/` preserves the original exploratory scripts; `round2/` preserves the
temporal experiment source. Per-frame samples and aggregate reports accompany
each experiment. Projection, decode, inference, transport and large exports are
excluded from these CPU planning measurements.

## Original dense per-frame work

On four IMG_5701 frames, approximately 102,000–112,000 point rows each:

- Original robust per-point slope estimation: **4.41–5.09 seconds**.
- Original cost-map construction: **259–295 ms**.
- Combined median: approximately **5.07 seconds**, before route search.

Running that complete dense analysis on each incoming frame is unsuitable for
the intended live workflow.

## Compact grid and native search

Experiments used a 200 x 200 grid (40,000 maximum vertices), without filling
unknown terrain. Two resolutions were tested: the supplied slope radius divided
by four or eight, in reconstruction units. Negative evidence blocked its cell
and was never averaged into positive support. Unknown/ambiguous ground stayed
unavailable; diagonal moves could not cut blocked corners.

Observed same-frame native search medians were approximately **0.3–0.8 ms**.
A fully open synthetic 40,000-cell grid completed in **5.65 ms**. Disconnected
walls, diagonal blocked corners and search deadline termination were also tested.

Direct 3 x 3 PCA fitting at every image point still cost hundreds of milliseconds:
the initial full CPU pipeline was approximately **340–440 ms**. Applying a small
PCA fit to the grid surface instead substantially reduced work. The finer grid
performed better in raw geometry checks. Coarse cell centroids could introduce
unsupported routes; the final experiment uses original observed point
representatives rather than invented averaged positions.

## Best tested same-frame approach

The selected experimental combination was:

1. Build a conservative grid at `radius / 8`, with original observed point
   representatives and separate positive/negative evidence.
2. Estimate slope on small grid neighbourhoods for fast candidate generation.
3. Run native bounded search for a reachable observed goal ahead.
4. Apply the original robust slope-fitting formula only to proposed route
   points, using their own frame's geometry. Block/replan rejected nodes, with
   a bounded number of attempts.
5. Validate support count, support gaps, edge grade and swept obstacle clearance
   on the raw evidence inside the route corridor.

| Saved-data configuration | Updates | Median CPU | P95 CPU | Publishable routes |
|---|---:|---:|---:|---:|
| IMG_5701 base support + VLM-negative channel | 56 | 53.0 ms | 66.5 ms | 53 |
| IMG_5516 VLM-negative + independent SAM positives | 6 | 54.2 ms | 60.0 ms | 4 |
| IMG_5516 VLM-positive + independent SAM positives | 6 | 58.6 ms | 65.4 ms | 4 |

These CPU figures include terrain/cost construction, raw KD indexing,
candidate search, selective robust slope checks, corridor extraction and raw
geometry validation. Frame data is already projected and resident in memory.
The slope-enabled prototype does not replace missing slopes with zero.

The selective robust slope checks matched the previously computed original
same-frame values and finite/NaN pattern exactly at every queried point in
these runs. A synthetic repeated-point case exercised fallback to distinct
neighbours. The raw fast grid estimate alone disagreed with an original slope
veto in one positive-VLM case; selective refinement rejected three nodes and
withheld the remaining invalid result. No published experimental route bypassed
that original route-node slope veto.

Across the 68 configuration-specific updates, 61 experimental routes passed
the publishability checks. The rest were unavailable or rejected; the
benchmark did not force goals or paths into invalid terrain. These counts are
not navigation accuracy or success rates: there is no labeled ground truth or
robot execution feedback.

Corridor-only raw geometry validation was compared against whole-frame evidence
on **32 cases**, with matching results. The corridor bounds retain every point
that can influence the tested support-radius and swept-clearance predicates.
This is not a claim that every grid cell or grid route is equivalent to the
original dense point graph or its global optimum.

## Temporal retention and connected-path findings

Per-frame grids did not supply a valid start directly under the camera in this
test. This is important: a path beginning on visible ground ahead cannot be
presented as a verified connection from the robot's current physical position.

A fixed-size temporal grid improved strict current-ground association on
IMG_5701 to **38 of 56 updates** in the slope-enabled cases. Its terrain/search
work was around **42 ms P95** at the finer resolution. However, rebuilding
raw validation structures from 20 retained frames exceeded **1 second P95**.
That naive history-rebuild approach is rejected for the live loop.

Indexing raw evidence by spatial tiles reduced this temporal experiment to
approximately **212 ms P95** overall, with a fixed 20-frame evidence limit.
The prototype retained at most 2,331,785 source rows and selected at most
298,576 rows for a route corridor. Exact extracted subsets and the full-context
geometry predicate matched in **five temporal test cases**. This supports using
indexed local evidence, but still misses the initial 100 ms combined CPU target.
The temporal experiment did not include selective robust route-node refinement.

The connected-extension experiment used the same anchor and explicitly
validated connectors. It obtained one initial proposed path and **no accepted
extensions over the remaining 55 updates**. It did not draw artificial joins.
Single-frame terrain loses the previous anchor as viewpoint/geometry changes;
blindly appending from a previous target is insufficient. A short ahead horizon
also provided no observed forward route in this camera view; a larger horizon
permitted the initial path. Lookahead, visible-ground range and units must be
configured consistently.

Synthetic splice tests preserved the prefix and rejected mismatched joins, but
those simple tests do not establish reliable real-video path continuity. The
recorded camera is not controlled by the proposed path, so its motion is not
proof of executed progress along it.

## Implementation decisions

- **Use:** a bounded grid, observed point representatives, native bounded
  search, cheap grid slope proposals, selective robust refinement, and indexed
  corridor evidence.
- **Reject:** fitting every raw point, image-wide PCA as the default hot path,
  averaged grid points without raw validation, full history reconstruction and
  independent frame routes presented as one continuous path.
- **Required next:** reliable temporal terrain fusion, stable retained anchors,
  validated current-ground association, source-aware witness retention and
  replacing an invalid future suffix rather than always extending its endpoint.
- **Still unproven:** live connected navigation, measured end-to-end latency,
  online reference/scale bootstrap, real robot footprint/kinematics and long-run
  behaviour. The experiments have not established production real-time planning.

The numerical same-frame work is now fast enough to justify implementing the
live architecture. The next engineering focus is temporal continuity and
evidence lifecycle, followed by paced real-model replay. It is not another
whole-cloud shortest-path optimization.
