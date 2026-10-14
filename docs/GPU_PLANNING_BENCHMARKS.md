# GPU geometry validation benchmark

Tested on 2026-10-14, outside the application. GPU support and obstacle
validation substantially improved the expensive cases and matched the CPU
decisions in all tested cases. GPU contention reduces the benefit, especially
for small cases. This is a geometric validation benchmark, not an implemented
live planner or an end-to-end navigation benchmark.

Production code, the existing server and its configuration were unchanged.
All experimental model workers were closed after testing.

## Hardware and implementation

The accessible device was an H100 MIG slice: 20,096 MiB reported memory and
26 multiprocessors. Tests used CuPy 13.6.0 with custom CUDA kernels, FP64
coordinates and no fast-math; fused multiply-add was disabled in the custom
predicates. Extra dependencies were installed in an isolated directory outside
the application, leaving the model and server environments unchanged.

The GPU builds a sorted three-dimensional voxel index over the raw witnesses.
It batches three original geometric checks:

1. At least six distinct eligible support points within the support radius of
   every route vertex. Duplicate observations cannot satisfy this requirement
   by themselves.
2. Ground coverage along every route edge, using the same sample spacing and
   conservative support-distance padding as the CPU predicate.
3. Clearance of the entire edge swept upward by the supplied body height,
   including end caps and the obstacle margin.

Slope-enabled edge-grade rejection remains the same. The GPU receives the
identical support, negative and proxy-slope arrays used by the CPU validator;
negative evidence is retained even where it overlaps positive evidence.
Source-specific robust route-node slope fitting was performed when fixtures
were captured, but has not itself been ported to GPU.

## Inputs and agreement

Captured 22 raw geometric predicate cases from the chronological persistent-map
replay: 19 accepted and three rejected. They include IMG_5701 current-ground
replans and tail-extension experiments, plus IMG_5516 VLM-positive and
VLM-negative configurations with independent SAM positives. The largest
corridor contained 271,148 raw rows. Fixtures were selected to retain all
slope-enabled validation calls, small-clip calls and every fifth remaining
CPU validation call; they are not a random navigation sample.

Every GPU decision matched the original CPU decision in each of four conditions:
GPU isolated, models loaded but idle, continuously active models, and paced
model inference. Each case also received repeated GPU checks with no decision
drift. Fourteen controlled checks matched CPU decisions, covering supported
ground, unknown gaps, negative walls, overhead obstacles, steepness enabled/off,
unknown slopes, repeated points, empty evidence, overlapping positive/negative
masks, zero-length edges and obstacle-margin boundary perturbations.

This is agreement on tested inputs, not proof of global numerical equivalence,
navigation safety or accuracy on unseen terrain. The supplied reference,
radius and height are timing profiles; online calibration remains untested.

## Paired validation costs

Each CPU measurement includes rebuilding the raw support/obstacle structures
and checking the same candidate route. GPU timings distinguish raw upload,
voxel-index construction and resident route checks. CUDA events measure device
execution; synchronized wall time also includes host sample preparation, route
upload and result download. Input loading and corridor extraction are excluded
from both validators.

| Condition | CPU median | CPU P95 | GPU stage-cost median | GPU stage-cost P95 |
|---|---:|---:|---:|---:|
| Isolated GPU | 27.5 ms | 152.0 ms | 3.9 ms | 11.3 ms |
| All model weights loaded, idle | 25.7 ms | 152.3 ms | 4.0 ms | 12.2 ms |
| All models continuously infer | 26.5 ms | 156.4 ms | 49.9 ms | 69.4 ms |
| Models infer with application-like pacing | 26.3 ms | 155.5 ms | 20.6 ms | 46.4 ms |

**GPU stage cost** is the sum of measured upload and index-build time for each
fixture plus its median resident check time. Its P95 is across those 22 fixture
costs, not a P95 of complete live updates or all individual queries. Resident
checks were repeated five to nine times per fixture after warmup. Compilation,
initial CUDA setup and model loading are excluded from the table; setup/warmup
times are saved separately.

The final GPU index occupied at most 8,947,932 bytes (about 8.5 MiB). This
excludes temporary sorting buffers, allocator caches and CUDA context overhead.
It also describes corridor-sized indexes, not the full 20-frame witness store.

Loading model weights alone did not materially affect validation performance.
Continuous inference did: for small cases, uploading/rebuilding an index on the
busy GPU could cost more than the CPU validator. The expensive CPU cases still
benefited. Performance depends on witness count, route size, index reuse,
scheduling and available compute, not simply whether CUDA is enabled.

## Actual-model contention test

Started separate persistent workers for LingBot, SAM3 and Qwen3-VL-4B, using
the existing worker adapters and local checkpoints. Warmed all three before
measurement. SAM queried `ground`, independent `path`, and a negative `log`
channel. VLM performed actual generation with the saved non-traversability
instruction and a 128-token limit.

The continuous stress test issued inference without pacing in all three
processes. Across its approximately 19.3-second validation interval, 39 LingBot,
16 SAM and nine VLM calls overlapped the measurements. A resident large-case
validator performed 153 checks, with 63.7 ms median and 81.1 ms P95 latency.
That stress case was a rejected route; its repeated correct rejection is not
navigation progress. Retaining model allocation caches left only about 161 MiB
free at the final sample, so this is not a suitable default deployment policy.

## Paced combined-model test

The more representative run serialized geometry then SAM, as the current
application does, with a requested 4 Hz geometry cadence, SAM every second
geometry call and VLM in the background at a requested 0.5 Hz. It used the
existing budget-based CUDA cache policy with a 2 GiB free-memory trigger.
That trigger is not a guaranteed reserved-memory floor.

For the resident test, the index was already uploaded and built. Its accepted
current-ground fixture contained **261,005 raw rows and 93 route vertices**.
Ran **40 checks at 2 Hz over approximately 19.5 seconds**:

| Resident GPU validation | Measured latency |
|---|---:|
| Median | 27.9 ms |
| P95 | 40.6 ms |
| Maximum | 42.8 ms |
| P95 start-scheduling lag | 0.22 ms |

Every check matched the CPU's accepted result. This P95 is calculated from the
40 actual resident query wall times, unlike the fixture stage-cost table.

During that interval, 45 LingBot, 22 SAM and ten VLM inference calls overlapped
validation. Completion rates within the measured interval were:

| Worker | Completed calls per second | Median call time | P95 call time |
|---|---:|---:|---:|
| LingBot | 2.25 | 204 ms | 371 ms |
| SAM | 1.13 | 230 ms | 642 ms |
| VLM | 0.51 | 1,155 ms | 1,216 ms |

The requested geometry cadence was not achieved: 35 scheduled geometry slots
were dropped among overlapping calls. Minimum sampled free memory was about
0.85 GiB. All workers completed normally without an out-of-memory failure.

The models replayed six preprocessed RGB images repeatedly to measure load and
contention. LingBot maintained its causal cache, but this repeated sequence is
not a natural camera trajectory. Newly generated outputs were **not** used as
inputs to the saved-geometry validator. Thus worker completion rates are not
fresh map or connected-path rates, and the test does not measure capture-to-route
latency or actual navigation.

## Implementation decision

Proceed with an optional GPU backend for raw geometric validation, maintaining
the CPU backend and the same negative, distinct-support, gap and clearance rules.
Keep witness indexes resident and update them incrementally. Keep bounded native
graph search on CPU initially. Benchmark light-case CPU fallback and scheduling
under actual perception load before selecting thresholds.

A wholesale Rust rewrite is not supported by these measurements. The major gain
comes from batching geometry checks and replacing repeated CPU index rebuilds.
Models already use compiled CUDA operations, and grid search already uses native
C++. Rust may be useful for specific CPU work, but would not remove GPU contention
or fix ground association and retained-path anchor failures.

Remaining gates are GPU witness lifecycle over the full bounded map, source-aware
slope refinement cost, connected suffix/progress correctness, online calibration,
a paced replay using fresh model outputs and longer-duration memory/latency tests.
The [persistent-path test](PERSISTENT_PATH_BENCHMARKS.md) still documents unresolved
continuity failures; faster validation does not resolve them.

## Artifacts and reproduction

Everything executable is outside the application repository, under
`/home/jovyan/EmbodiedAI-gpu-validation/framewise-benchmarks/`:

- `gpu_geometry.cu`, `gpu_geometry.py`: experimental index and FP64 predicates.
- `gpu_prepare.py`, `gpu-fixtures.json`, `gpu-fixtures/`: original CPU inputs.
- `gpu_benchmark.py`, `gpu-guards-report.json`, `gpu-results/`: paired timings.
- `gpu_combined.py`, `gpu_paced.py`: model-load and inference tests.
- `gpu-results/combined-models/`, `gpu-results/paced-models/`: events, individual
  model logs/memory traces, decision samples and combined reports.
- `gpu-manifest.json`: hardware, versions, input/source/result hashes and settings.

The additional CuPy/fastrlock wheels are in the sibling
`/home/jovyan/EmbodiedAI-gpu-validation/gpu-benchmark-deps/` directory. They are
provided to these scripts through `PYTHONPATH`. CUDA NVRTC/runtime library paths
are supplied only to the benchmark processes. The recorded runner preserves the
exact environment for reproduction without changing application environments.

After the tests, GPU usage returned to 12 MiB and no compute processes were
listed. Server PID 13137 remained running on port 8083.
