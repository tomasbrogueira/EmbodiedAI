# Optional streaming memory and SAM improvements

Implemented on `vlm_improvements`, after the
[hardware audit](vlm_performance_analysis.md). These changes are in the optional
plugin; the original LingBot model, SAM3 source, orchestrator, proposal prompt,
batch runner and reference SAM wrapper remain unchanged. Changes are not yet
committed. The original 64-keyframe cache, four camera iterations and original
SAM path remain defaults.

The smaller cache allowed full selected sequences to complete on the available
19.625 GiB GPU partition. It changes reconstruction after cache eviction, so
this is an experimental setting rather than a validated accuracy improvement.
No ground-truth geometry or obstacle annotations were supplied.

## What changed

- `run_streaming.py` and `stream_worker.py` expose `--kv-cache-window` and pass
  it into the existing model's cache configuration. Eviction and inference use
  the original implementation. No model source is patched.
- `runtime_metrics.py` records device-wide free/used memory, each worker's own
  allocator usage and peaks, host RSS/cgroup memory, and the LingBot cache size.
  Workers sample into separate JSONL traces; the controller creates no extra
  CUDA context. `status.json` and Viser display the sampled GPU peak and minimum
  free memory. Errors retain memory evidence and pending geometry indices.
- `--cuda-cache-policy budget` releases unused PyTorch allocator blocks after
  a completed request if device free memory is below 2 GiB. It preserves live
  weights, geometry caches and text features. The threshold triggers cleanup;
  it does not reserve 2 GiB or prevent a larger transient allocation.
- `sam_runtime.py` adds an opt-in, per-model LRU cache for exact text prompts.
  SAM still embeds each mapped image and grounds every active prompt on that
  image. Instance masks retain the original confidence, interpolation and
  Boolean threshold rules, are unioned on the GPU, then transferred to CPU
  once. No image embedding or mask is reused across views.
- `--worker-threads 4` caps Torch and BLAS threads in each model worker.
  The audit found little isolated inference benefit; this avoids unnecessary
  thread oversubscription on a container with 24 CPU equivalents.
- Separate tools generate matching-frame causal/windowed geometry references,
  compare predictions after global alignment, and verify masks and full map
  payloads against the original implementations.

The implementation uses the SAM processor's existing private grounding method.
Source hashes are recorded in the benchmark directory; repeat equivalence checks
when upgrading SAM or PyTorch.

## Complete-video measurements

All four runs retained four camera iterations, width 518, scale initialization
of two frames, keyframe interval two, `--input-fps 4`, `--map-step 2`, and
`--vlm-interval 2`. Every run used cache window 8 and four worker threads, saved
mapped predictions/masks, and processed to EOF without playback pacing or
dropped selected frames. There are **three distinct videos**, with the log clip
run twice to compare allocator policies. The 29.85–29.97 FPS inputs use integer
stride 7, retaining approximately 4.26–4.28 source frames per video second.

| Video / run | Geometry / map frames | SAM frames | Processing s | Recorded wall s | Geometry Hz | Map Hz | Sampled peak GiB | Min sampled free GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Outdoor IMG_5520, retain | 124 / 62 | 0 | 32.28 | 65.08 | 3.84 | 1.92 | 17.961 | 1.664 |
| Indoor IMG_5522, original SAM path, retain | 96 / 48 | 0 | 24.97 | 56.28 | 3.84 | 1.92 | 17.938 | 1.687 |
| Outdoor IMG_5516, cached SAM, retain | 50 / 25 | 21 | 18.00 | 50.96 | 2.78 | 1.39 | 19.287 | 0.338 |
| Outdoor IMG_5516, cached SAM, budget | 50 / 25 | 21 | 18.36 | 49.72 | 2.72 | 1.36 | 18.705 | 0.920 |

Processing time includes decoding after initialization, worker transfers,
inference, projection, PLY/display updates, saved artifacts and the final VLM
drain. Recorded wall time additionally includes model initialization; final
status is written before worker shutdown. These are complete runner rates,
not isolated GPU inference rates. `--save-frames` adds artifact-writing costs.

Both log runs produced **3,400,657 points**, including **124,895 obstacle-marked
points**. Each made five valid VLM calls. The budget run recorded 20 SAM text
cache hits and one miss across 21 active SAM frames. It released unused allocator
blocks after 61 worker requests. Compared with retention, the single measured
budget run increased minimum sampled free memory by 0.582 GiB and increased
processing time by 2.0%. This is one comparison, not a repeated statistical
benchmark or an isolated measurement of the SAM optimization's throughput gain.

The longest video produced **9,675,416 points**, but Qwen proposed no obstacles.
The indoor clip also generated no active proposals, despite a bag having appeared
in earlier batch evaluation. Those runs validate geometry/memory completion,
not obstacle discovery or active SAM memory. The log run supplies the positive
segmentation check. Proposal scheduling and model classification can still miss
objects; this change does not tune the prompt or establish hazard recall.

Memory traces sample every 100 ms and include request-boundary snapshots.
Device-wide peaks are lower bounds on actual transient peaks. Allocator peaks
refer to individual workers and must not be summed across different times.
Background tracing starts after each worker loads; already loaded workers can
observe later workers' loading. The 0.920 GiB observed headroom remains limited,
particularly if more prompts, higher resolution, or longer context are requested.

The original no-drop 64-keyframe full-pipeline replay had failed after 22 geometry
frames. This prototype completed 124 frames on the longest clip and 50 with
active SAM on the log clip. It demonstrates completion for these inputs, not
reliability on all videos or indefinitely long streams. It has not reached
30 FPS or the roughly 4.28 geometry / 2.14 map Hz input schedule with active SAM.

## Geometry comparison

References decode exactly the same source indices and verify RGB hashes before
running inference. One reference uses the original persistent causal loop with
cache 64; the other calls the original windowed inference helper with 32
keyframes, eight-keyframe overlap and the original median scale alignment.
Reference inference runs with only LingBot resident because the full reference
cache does not fit alongside Qwen and SAM on this partition.

For each mapped frame, up to 2,048 deterministic same-pixel correspondences are
selected from points passing both models' original finite/confidence/depth
filters. A single global similarity transform removes coordinate origin,
rotation and scale differences. The reported residual is Euclidean point
separation divided by the reference pixel's depth; it is a prediction difference,
not a metric accuracy score or percentage of incorrect points.

| Candidate cache 8 / reference | Matched mapped frames | Frames with all prediction arrays bit exact | Median aligned separation / depth | 95th percentile |
| --- | ---: | ---: | ---: | ---: |
| IMG_5520 / causal cache 64 | 62 | 9 | 3.60% | 10.24% |
| IMG_5520 / original windowed 32 | 62 | 0 | 4.50% | 12.51% |
| IMG_5516 / causal cache 64 | 25 | 9 | 3.41% | 7.34% |
| IMG_5516 / original windowed 32 | 25 | 6 | 3.41% | 7.34% |
| IMG_5516 budget / retain, both cache 8 | 25 | 25 | numerical alignment roundoff | numerical alignment roundoff |

The first nine mapped predictions match the causal 64 reference before the
smaller cache first evicts history. Later differences are real. The memory
policy itself left all 25 saved log prediction arrays bit identical, and both
log runs retained the same masks and markings.

The longest run ends with ten patch keyframes: the configured eight recent
keyframes plus two initialization frames. Its original cache also retains 318
older special tokens; the total recorded cache is approximately 1.93 GiB.
Older special tokens continue to accumulate, so this bounds patch context rather
than all memory. The existing retained-sequence limit remains 1,024 frames.
On the longest run the last ten geometry worker batches averaged 216 ms; the
standalone causal-64 reference averaged about 394 ms over those same late
batches. This comparison excludes the rest of the pipeline and is not a
complete-run speedup factor.

**Decision:** retain cache 64 as the default. Offer cache 8 explicitly for this
machine and review its reconstruction before using it where geometry accuracy
matters. The measurements do not establish equal map quality. Quantizing or
offloading Qwen, or correcting the LingBot key-cache dtype upstream, are separate
next steps if retaining larger context is required.

## Verification

- **22 unit tests passed.** They cover frame/mask alignment, original geometry
  filters and projection, streaming state, bounded text caching, current-image
  mask use, device versus process memory accounting, allocator policy, error
  evidence, and similarity alignment. Compilation also passed.
- **43 real-GPU SAM cases passed bit-exact comparison** against the original
  `SamSegmenter.masks()` plus CPU union: all 25 mapped frames of the budget log
  run, plus one frame from each of all 18 existing video maps. Every saved log
  mask also matched its independently recomputed reference. This covers 17
  nonempty cases and 26 empty masks; it is not all-frame validation of 18 videos.
- Original mapping of the saved predictions/masks produced byte-identical PLY
  vertex payloads for the longest run and both log runs. Counts, frame indices,
  RGB hashes, native Boolean mask shape and mask pixel counts matched. This
  establishes projection/export equivalence for the supplied geometry, not
  equivalence of smaller-cache geometry to the original reconstruction.
- Hashes and Git comparisons confirm that the original model/demo, SAM
  processor, proposal prompt, SAM reference wrapper, batch runner, orchestrator
  and geometry helper remained unchanged.

Raw configurations, statuses, events, worker logs, memory traces, mapped NPZs,
PLYs, reference predictions and verification JSONs are under
[`data/vlm_maps/cache_trial/`](../data/vlm_maps/cache_trial/). A compact
[`summary.json`](../data/vlm_maps/cache_trial/summary.json),
[`summary.csv`](../data/vlm_maps/cache_trial/summary.csv) and
[`runtime plot`](../data/vlm_maps/cache_trial/runtime.png) summarize the runs.
These artifacts are local and include large point clouds; they are not committed.

## Running and inspecting it

The tested command and option table are in
[`plugins/small_obstacles/STREAMING.md`](../plugins/small_obstacles/STREAMING.md#tested-memory-constrained-configuration).
Use a fresh output directory. To select the experimental path, add:

```text
--kv-cache-window 8 --worker-threads 4 --cuda-cache-policy budget --sam-text-cache
```

Keep `--save-frames` when inspecting masks or comparing predictions. Observe
`status.json` during a run for rates, frame counts, drops and sampled memory;
`events.jsonl` contains per-frame timings, VLM responses and cache measurements.
The PLY accumulates all valid mapped observations; the display uses the existing
bounded 200,000-point sample. No temporal or voxel fusion is added.

To inspect the completed budget run with Viser, use an available port:

```bash
/opt/conda/envs/orchestrator-visualizer/bin/python \
  plugins/small_obstacles/view_stream.py \
  data/vlm_maps/cache_trial/log_window8_budget_cached_sam --port 8083
```

In this JupyterHub workspace, use `/user/jorisk/proxy/8083/` under the existing
JupyterHub origin. Localhost works on the server or through a port forward.
The earlier Viser viewers on 8080 and 8082 were left running.

For independent checks, run `verify_streaming_run.py --role map` in the LingBot
environment or `--role sam --all-videos` in the SAM environment, supplying
`--run RUN --output NEW_REPORT.json`. Geometry references use
`benchmark_geometry.py --run RUN --output NEW_DIRECTORY`; differences use
`compare_geometry.py --candidate RUN/predictions --reference REFERENCE_DIRECTORY
--output NEW_REPORT.json`. All verification tools operate on separate outputs.
