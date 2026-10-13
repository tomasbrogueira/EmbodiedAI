# Implementation plan for faster optional VLM mapping

This plan follows the measured [performance analysis](vlm_performance_analysis.md).
All changes belong to the optional plugin on `vlm_improvements`. Keep the original
orchestrator, model source checkouts, and `run_existing_pipeline.py` intact.
The first optional improvements are now implemented: configurable patch cache,
per-worker/device memory telemetry, allocator release policy, and SAM text caching
with GPU union. Complete-video and equivalence tests are recorded in the
[bounded-cache results](vlm_bounded_cache_results.md). The smaller cache remains
experimental because geometry changes. The phases below are the original plan;
their broader acceptance criteria and remaining optimizations are still pending.

## Goals and constraints

- Preserve the existing RGB preprocessing, native mask alignment, official
  unprojection, pose convention, confidence/depth filters, pixel step, and full
  observation history. Keep path planning outside this plugin.
- Process fixed, identical source frame indices when comparing performance.
  Report geometry Hz, semantic map Hz, VLM discovery Hz, frame drops, and latency
  independently. A higher rate obtained by dropping more frames is a scheduling
  tradeoff, not a model speedup.
- Keep four camera iterations and full-resolution per-frame SAM as the reference.
  Changing context, resolution, weights, discovery frequency, or tracking needs
  a separate quality evaluation and explicit configuration.
- Support bounded memory and predictable latency beyond the short demo. A useful
  first target is sustaining the existing approximately 4.28 geometry Hz and
  2.14 semantic Hz sampling schedule with one active prompt. This is a target,
  not a demonstrated capability. Thirty-FPS full-model inference is not supported
  by current measurements.

## Phase 1: Make measurements and memory failures actionable

**Priority: immediate. Estimated effort: 1–2 engineering days.**

Touch plugin `run_streaming.py` and `stream_worker.py`; add `runtime_metrics.py`
and `benchmark_streaming.py`. Keep the read-only `profile_runtime.py` probe.

1. Record source frame index, geometry sequence index, mapped status, RGB hash,
   active prompts, and proposal image timestamp for each operation. Add the
   time a proposal becomes available and is first applied to a map frame.
2. Measure decode including skipped-frame `grab()`, preprocessing, IPC write/read,
   worker queue wait, inference, return transfer, SAM image/text/grounding/union,
   projection, PLY, display sampling/export, and optional artifact compression.
   Separate startup, first inference, steady-state, playback pacing, and EOF drain.
3. Add per-worker GPU allocated/reserved/peak memory and device free memory at
   scale initialization and regular intervals. Record host RSS, CPU quota, and
   cgroup memory/cache. Account for all workers and CUDA contexts; isolated
   per-model allocator counters do not measure the whole device.
4. Add a fixed-input benchmark mode with no playback pacing or frame dropping.
   Use a recorded VLM proposal schedule or fixed prompt manifest when isolating
   geometry/SAM changes. Preserve the full live mode as a separate benchmark:
   asynchronous completion otherwise changes the prompt schedule between runs.
5. Reproduce the recorded allocator assertion on a bounded replay, preserving
   partial output and memory telemetry. Compare supported allocator settings in
   separate runs, before CUDA initialization. Do not infer that changing the
   allocator fixes a resident-memory budget larger than the device.
6. Expose per-role CPU thread settings, initially benchmark 4–8 intra-op and 1–4
   inter-op threads plus a small OpenCV pool. Set BLAS/OpenMP limits before child
   imports. Treat this as resource control: no large inference gain was measured.

**Acceptance:** every source frame is accounted for; stage totals explain the
critical path without double-counting concurrent VLM work; failure reports include
the last complete frame and device memory. The benchmark emits JSON/CSV plus
median/p90/p95 latency and records all settings and software versions.

## Phase 2: Bound the GPU working set and optimize LingBot cache handling

**Priority: blocking sustained operation. Estimated effort: 3–5 days for the
adapter/prototype and tests; backend compatibility may add time.**

Add `lingbot_runtime.py` / `cache_adapter.py`, selected only by the plugin worker.
Reuse existing model entry points. Do not modify the model checkout in place.

1. Calculate the planned resident working set from model allocations, cache
   dimensions/dtypes, backend page-pool sizing, and temporary buffers. Require
   headroom for SAM and Qwen inference, not just their idle weights.
2. Implement an optional compact cache adapter that keeps aggregator keys in the
   dtype actually used by autocast SDPA, after normalization/RoPE, before storage
   and append. Preserve query values, scale/recent/special-token order, eviction,
   and keyframe/non-keyframe behavior. Leave camera/depth head precision unchanged.
3. Avoid `torch.cat` promotion, full-cache `.clone()`, and holding all old tensors
   while replacing them. Use a reusable buffer/ring or compatible paged storage.
   The measured post-frame casting prototype is a correctness clue, not the
   implementation: it saves idle allocation but barely changes peak memory and
   slows inference.
4. In an isolated optional environment, test the existing `use_sdpa=False`
   FlashInfer backend against SDPA. Pin compatible Torch/CUDA/FlashInfer builds;
   verify its exact API and prebuilt kernel support on this driver/MIG device.
   Budget the default page-pool headroom before loading other models. If compact
   page sizing requires an adapter, keep it plugin-local and retain eager SDPA.
5. Benchmark short, growing, and full-cache streams, including non-keyframes,
   eviction boundaries, non-square supported shapes, and sequence reset. Never
   implement an unaligned cache reset that creates a new coordinate system in
   the middle of the saved map.
6. Decide the capacity path using measured all-worker peak memory:

| Option | Preserves model settings? | Decision |
| --- | --- | --- |
| Larger allocation, preferably at least 40 GB | Yes | Simplest capacity path if the machine owner can provide it; benchmark compute resources too. |
| Efficient cache plus Qwen/SAM memory reductions | Potentially, but precision may change | Preferred engineering experiment if remaining on the 20 GB partition. |
| Smaller patch cache, e.g. 16/24/32 | No; temporal context changes | Explicit optional mode only, with geometry comparison. |
| Existing windowed engine and alignment | Reuses baseline behavior if identical settings are used | Evaluate incremental integration and chunk latency; windows count keyframes, not source frames. |
| Weight/cache offloading or CPU VLM | Can preserve weights but changes latency | Capacity fallback; measure transfers and discovery delay before adoption. |

The original resident total is about 26.9 GiB; even BF16 stored keys leave about
22.8 GiB. Neither an allocator flag nor cache casting alone solves this.

**Acceptance:** no allocator failure on every dataset clip and a long stress
stream; record device-wide peak with all models active. Aim for at least 2 GiB
measured device headroom. Verify depth/confidence/pose outputs, original projection
results, and mask/color alignment against the reference on identical frames.
Attempt bit equality for storage-only changes; investigate every difference.
Kernel/backend changes require documented tolerances and geometry-quality checks.

## Phase 3: Apply the demonstrated SAM improvements

**Priority: high, small implementation. Estimated effort: 1–2 days.**

Add `sam_runtime.py` implementing an optional adapter around the existing model
and processor. `segment.py` and the batch runner remain a reference path.

1. Cache text encoder outputs using a bounded key containing checkpoint/model
   identity, tokenizer/config, device, dtype, and exact prompt. Cache only
   immutable language features; isolate per-image mutable processor state.
2. Reuse the image embedding exactly as today. Clear prior prompt geometry/results
   between concepts, inject cached language features, and invoke the existing
   grounding implementation.
3. OR the thresholded native-resolution instance masks on GPU across all prompts,
   including empty instance lists; transfer one H × W Boolean union to CPU.
   Preserve instance confidence threshold 0.5 and mask threshold 0.5.
4. Keep cache behavior behind an optional runtime flag and an eager fallback.
   Pin/test the processor API because the prototype uses its private grounding
   method. Return exactly the existing mask schema to projection.
5. Test batched text/grounding as a later substep only if profiling indicates a
   benefit. A joined sentence is not equivalent to separate object prompts;
   do not replace multi-prompt segmentation with a single comma-separated query.

**Acceptance:** bit-identical unions on broader multi-instance and multi-positive
cases, empty prompts, zero detections, varying image sizes, changing concept order,
cache eviction, and model reset. Evaluate sampled frames throughout all 18 clips.
Use the measured 7/22/58 ms query savings as a starting expectation, not an
end-to-end speed promise. Keep only changes with a repeatable improvement.

## Phase 4: Remove avoidable startup, IPC, and output latency

**Priority: medium; helps responsiveness and tails. Estimated effort: 2–3 days.**

Touch plugin controller/state/viewer files; add `frame_transport.py` and
`output_workers.py` where needed.

1. Import official geometry and initialize its CPU path before the first frame.
   Record initialization time explicitly rather than making it disappear from
   wall-clock reporting. The measured first projection costs 1.42 seconds.
2. Keep workers alive across sequential videos or in a reusable local service.
   Add explicit sequence-reset operations for LingBot cache/counters and controller
   prompt state; use separate output/map state per video. Models currently load
   once per run, but reload on the next video/run.
3. Benchmark bounded parallel CPU checkpoint loading if startup remains important.
   Limit concurrent threads and peak host/device memory. Report cold launch,
   warm-service first map, and shutdown separately.
4. Replace bulk disk NPZ IPC with bounded CPU shared-memory slots and JSON metadata
   containing source index, shape, dtype, generation ID, and ownership state.
   Support all three Python versions, stale results, worker failure, and cleanup.
   Retain disk transport as a debugging fallback. GPU IPC/environment unification
   is a later experiment, not a prerequisite.
5. Put display reservoir/export and optional diagnostic compression into bounded
   consumers. Display can keep only the latest snapshot; full-map observation
   writes must be lossless and apply backpressure rather than silently dropping.
   Preserve vertex order, native mask association, valid PLY header updates, and
   recoverable partial maps. Drain required writes before reporting completion.
6. Update Viser status only when changed, compute display bounds less frequently,
   and keep a bounded point cloud/message size. Verify through the actual Jupyter
   proxy as well as direct localhost; preserve existing port 8080/8082 viewers.

**Acceptance:** original mapper replay remains equivalent; no mixed frame IDs or
partial shared-buffer reads; bounded queues; full observations retained; viewer
remains responsive. Report the modest measured NPZ/PLY costs accurately. The
largest startup improvement will come from reusing loaded workers, not tmpfs.

## Phase 5: Test compilation and GPU scheduling

**Priority: after memory stability. Estimated effort: 2–4 days of experiments.**

1. Expose plugin-only compile flags. Reuse LingBot's existing `compile_model`
   targets and warm both keyframe and non-keyframe paths, then reset the cache
   before the actual video. Start with SAM's fixed 1008-resolution vision path
   via the existing `compile=True` builder option.
2. Measure compile time, recompiles, steady latency, and peak memory. Dynamic cache
   lengths and variable mask counts can create graph breaks or extra compiled
   variants. Warmup settings must match input shape and the two scale frames.
3. Set allocator configuration before CUDA initialization. The existing demo
   documents a Torch 2.8 expandable-segment/CUDA-graph interaction; the plugin
   must select and validate the correct allocator explicitly because adding a
   config flag will not automatically trigger the demo's CLI check.
4. Benchmark identical geometry/SAM work with Qwen resident but idle, then making
   identical recorded proposals. Compare asynchronous calls and controlled
   admission policies. Three GPU processes do not automatically improve GPU
   throughput; evaluate p95 map latency and proposal age, not just average Hz.
5. Test TF32 for eligible LingBot FP32 matmuls as an explicit precision option.
   Keep existing BF16 aggregator/autocast behavior; it is already enabled.

PyTorch's compile modes can trade Python/launch overhead for extra memory, so
compilation is gated on the memory budget, not assumed to solve it.
[PyTorch compile documentation](https://docs.pytorch.org/docs/2.8/generated/torch.compile.html).

**Acceptance:** a repeatable improvement on matched workloads, no memory failure,
no unbounded recompilation, and documented output differences/fallback behavior.
Do not combine several experimental changes before measuring them individually.

## Phase 6: Evaluate options that can change perception or reconstruction

**Priority: conditional. Estimated effort: 3–7 days, depending on engines and
ground-truth preparation.**

Keep these modes separate from optimizations expected to preserve outputs.

- Qwen prompts-only JSON: retain the hazard inclusion/exclusion rules, validate
  specific SAM-findable object nouns, and compare proposal recall/false positives
  and output tokens. Lowering the generation cap alone is not a speed improvement.
- Qwen 8/4-bit inference or a supported serving engine: prioritize memory savings,
  cap engine memory/KV allocation, and measure batch-one latency. Preserve the
  existing 4B BF16 model as reference. Previously weaker smaller-model proposals
  make a smaller model an accuracy decision rather than a default optimization.
- Smaller LingBot temporal cache or input size: compare scene alignment, depth,
  pose drift, and projected obstacle displacement. Keep every required frame;
  do not describe more frame dropping as faster inference.
- Smaller SAM encoder resolution: first establish architecture/output-shape
  support, then small-object boundary/recall quality. Changing only LingBot image
  size does not reduce SAM's 1008-pixel encoder workload.
- SAM video tracking between fresh detections: use the model's real temporal
  tracking API, not the previous image's pixel mask. Measure drift, new-object
  delay, disappearing objects, water/hole behavior, and reset conditions.
- TensorRT/export or hardware video decode: pursue only if prior profiling shows
  their stage is limiting throughput. An installed library is not an exported
  and validated model engine.

**Acceptance:** quality and latency evaluated together. The existing videos have
no sufficient ground truth for safety/recall claims; annotate representative
small hazards and terrain negatives and add pond/puddle/open-hole examples.
Any adopted tradeoff gets a named configuration and a reference fallback.

## Validation matrix and delivery order

| Dimension | Required checks |
| --- | --- |
| Source frames | Fixed frame IDs and RGB hashes; no drops for throughput comparisons. |
| Prompts | 0, 1, 3, 8; negative, positive, multiple instances, changing concepts; recorded and live VLM schedules. |
| Stream length | Short demo, all 18 clips, >162 retained-frame cache stress, declared sequence-limit behavior. |
| Geometry | Scale initialization, both keyframe paths, eviction boundaries, reset, original projection convention. |
| Runtime | Eager reference; each adapter separately; then combined; idle/live Qwen; cold/warm service. |
| Resources | Device-wide free/peak memory; per-worker allocator memory; host RSS/cache; CPU quota; output backlog. |
| Outputs | Native union shape, masks/colors, complete PLY observation count/order, partial-file validity. |
| Viewer | Bounded 200k display and stable direct/proxied WebSocket connection. |
| Quality | Mask equivalence where expected; geometry tolerances; annotated hazard precision/recall for changed models/settings. |

For significant benchmarks, run at least five measured repeats after warmup;
report medians and p90/p95 with frame/call counts. Do not extrapolate a single
positive log scene to every video or small-obstacle class.

Deliver four reviewable increments: (1) instrumentation and memory-capacity
decision, (2) cache adapter plus SAM adapter, (3) warm service/transport/output
work and selective compilation, (4) separately evaluated quality tradeoffs.
At every increment retain eager plugin operation and the original batch pipeline.
The initial instrumentation and SAM work are relatively small; a substantial
sustained speedup depends on solving the GPU cache/capacity bottleneck and has
not yet been demonstrated on this machine.
