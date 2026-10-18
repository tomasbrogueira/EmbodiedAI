# Performance analysis of the optional VLM mapping pipeline

Audit date: 2026-10-11. Workspace: `/home/jovyan/EmbodiedAI`, branch
`vlm_improvements`, starting commit `878c57b6`. The streaming additions are still
uncommitted. This audit adds diagnostic benchmarks and documentation; it does
not apply optimizations to the production pipeline or modify the original mapper,
LingBot model, SAM source, or batch runner.

**Subsequent implementation:** the optional smaller cache, memory telemetry,
allocator policy and SAM adapter are now implemented and tested. See the
[bounded-cache results](vlm_bounded_cache_results.md). The measurements below
describe the original audit, before those improvements.

## Findings that determine the implementation plan

There are useful improvements available, but no measured evidence yet for a
large throughput increase with unchanged model settings. The main obstacle is
the memory and attention cost of LingBot's growing cache on a small GPU partition.
SAM's repeated text processing is a smaller, demonstrated opportunity.

1. This workspace has **26 GPU SMs and approximately 19.63 GiB of usable device
   capacity**, rather than exclusive access to the entire H100 80 GB device.
2. LingBot's default streaming cache increases isolated frame time from roughly
   **153 ms early in a sequence to 500 ms when the patch cache fills**. Allocated
   memory reaches **15.16 GiB for LingBot alone**. Qwen and SAM cannot remain
   resident alongside that cache on this allocation.
3. A new replay of the short log video, without playback pacing or frame drops,
   **failed after 22 geometry frames**, with a CUDA allocator/NVML assertion.
   The previous successful live run also processed only 22 frames. Its success
   does not establish that the current implementation supports longer streams.
4. A diagnostic SAM adapter caching text features and unioning masks on the GPU
   saved about **7 / 22 / 58 ms of query time for 1 / 3 / 8 prompts**. Masks were
   bit identical to the original SAM wrapper on one sampled frame from each of
   all 18 videos. This is an equivalence check, not a full-video accuracy evaluation.
5. Reducing PyTorch threads did not materially accelerate isolated model
   inference. Reducing camera refinement from four iterations to one saved only
   about 3%, while changing reconstruction behavior.

The implementation sequence and acceptance criteria are in
[the implementation plan](vlm_performance_implementation_plan.md).

## Hardware actually available to this process

| Resource | Observed allocation | Consequence |
| --- | --- | --- |
| Physical GPU | NVIDIA H100 80 GB HBM3 | The physical model name overstates our available resources. |
| Visible CUDA device | One MIG device; 26 SMs; compute capability 9.0; 21,072,183,296 bytes | All three model processes share this partition. |
| MIG name | `nvidia-smi -L`: `1g.20gb`; PyTorch: `2g.20gb` | The labels disagree. Use the agreed memory capacity and SM count for budgeting. |
| Driver | 535.247.01; `nvidia-smi` reports CUDA 12.4 | Existing eager inference works; compilation/extensions still need compatibility tests. |
| Host CPU | Dual Xeon Platinum 8480C; 224 logical CPUs visible | Affinity and host CPU count are not the container's CPU entitlement. |
| CPU quota | `cpu.max = 2400000 100000`: 24 core equivalents | Three 112-thread model pools can oversubscribe the quota. |
| Container RAM | `memory.max = 68719476736`: 64 GiB | Host-reported 2 TiB is not available to this job. |
| RAM usage at inventory | About 56 GiB charged; about 47 GiB file cache, with much smaller anonymous memory | A high charged total is not all model RSS. Avoid large preload/compile peaks; monitor reclaimable cache separately. |
| Workspace filesystem | Local RAID mount; about 8.3 TiB free | Frame IPC is a serialization/copy cost, not an obvious storage bandwidth crisis. |
| `/dev/shm` | Large host tmpfs mount | Its mount size does not override the 64 GiB cgroup RAM limit. Use bounded allocations. |
| Video engines | MIG table exposes one decoder | OpenCV currently uses CPU decoding; hardware decoding is secondary on this dataset. |

MIG partitions compute and memory resources, so calling this a full-H100
benchmark would be misleading. See [NVIDIA's MIG resource description](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/introduction.html).
No partition, driver, affinity, or system configuration was changed during the audit.

## Installed libraries and backends

| Worker | Python / PyTorch | Other relevant installed packages | Missing optional packages |
| --- | --- | --- | --- |
| LingBot | 3.10.22 / 2.8.0+cu128 | torchvision 0.23; Triton 3.4; OpenCV 5.0; NumPy 2.2.6 | FlashInfer, external flash-attn |
| Qwen | 3.11.9 / 2.10.0+cu128 | transformers 5.3; torchvision 0.25; Triton 3.6; Accelerate 1.15 in a temporary package directory; TensorRT 10.14 | vLLM, bitsandbytes, external flash-attn, ONNX Runtime GPU |
| SAM | 3.12.15 / 2.10.0+cu128 | torchvision 0.25; Triton 3.6; timm 1.0.30 | FlashInfer, external flash-attn |
| Viewer | Separate CPU environment | Viser 1.1.1 | GPU inference is not needed here. |

The environments intentionally differ. Combining them into one process would
require dependency and ABI validation; it is not a quick performance fix.
`PYTHONNOUSERSITE=1` is required for the tested VLM environment to avoid a
conflicting user-site CPU PyTorch installation. Its Accelerate installation under
`/tmp/small-obstacles-vlm-deps` should become a reproducible plugin environment.

**Fused attention is already present.** Qwen reports `sdpa` for both text and
vision. A LingBot CUDA profile recorded 72 Flash Attention calls and 16 memory
efficient attention calls among 136 SDPA calls on one late streaming step.
Those counts include multiple parts of the model; they are not 136 distinct
layers. Missing the external `flash-attn` package does not imply unfused inference.
PyTorch selects SDPA implementations according to input/backend constraints:
[PyTorch SDPA documentation](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.scaled_dot_product_attention.html).

The active global streaming class is **`SDPAAttention`**, not the older
`CausalAttention` branch that creates an all-true attention mask. The active class
already invokes SDPA without that mask. Removing the mask in the older class
would not optimize this runner.

A small `torch.compile` CUDA kernel passed on the installed driver, taking
1.94 seconds on first invocation. This proves a basic compiler path works;
neither full model compilation nor CUDA graph replay has been validated.
CUDA 12 minor-version compatibility has limitations, especially for newer
features/PTX, so version mismatch alone neither proves a failure nor establishes
full compatibility. [NVIDIA compatibility documentation](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

FlashInfer is absent, although LingBot already contains a paged-cache backend.
The current FlashInfer documentation lists newer CUDA builds than the installed
cu128 environments. Select a compatible release and verify the project's API
usage in a separate environment; do not blindly install the latest package.
[FlashInfer installation requirements](https://docs.flashinfer.ai/installation.html).

## Measurement method and limits

The reproducible probe is
[profile_runtime.py](../plugins/small_obstacles/profile_runtime.py). Raw measurements,
hardware/library inventory, video metadata, code hashes, and failure logs are in
[`data/vlm_maps/performance_analysis`](../data/vlm_maps/performance_analysis/).

Model roles ran sequentially on the GPU, using the real checkpoints and a saved
518 × 518 RGB frame from `outdoor/IMG_5515`. CUDA was synchronized at diagnostic
stage boundaries. Model load used four CPU threads. Thread sweeps vary intra-op
threads with inter-op threads held at four; they do not recreate the complete
multi-worker scheduler. SAM timings use four measured repetitions after two
warmups; Qwen uses three after one; LingBot short tests use twelve after four.
Small differences below a few percent should not be treated as established gains.

The longer LingBot tests repeat the same image to measure cache cost over 162
retained frames, including two scale frames. They are compute/memory probes,
not natural trajectories or reconstruction-quality tests. SAM's eight-prompt
benchmark contains one positive concept and several negative concepts, so it
does not measure a scene with many positive instances. CPU decode measurements
finished before Qwen's measured generation sweep.

All 18 video headers and processed dimensions were inspected: the existing
preprocessing yields 518 × 518 frames for this dataset. At the default four-FPS
sampling target, clips retain 29–124 frames without drops; nine retain more than
50. Full streaming inference was not rerun on all 18 during this audit.

## Isolated LingBot measurements

Four camera iterations, two scale frames, keyframe interval two, SDPA, patch
cache window 64; all times include the worker's NPZ input/output and postprocessing.

| Cache stage | Median frame time | p90 | Allocated at end | Allocator reserved at end |
| --- | ---: | ---: | ---: | ---: |
| Early: sequence indices 6–21 | 152.7 ms | 167.1 ms | 4.92 GiB | 7.23 GiB |
| Middle: 62–81 | 289.0 ms | 327.3 ms | 10.60 GiB | 18.18 GiB |
| Filled patch cache: 142–161 | 500.4 ms | 547.3 ms | 15.16 GiB | 19.01 GiB |

Peak allocated memory in the long test was 16.46 GiB. Reserved memory is reusable
within that worker; another model process cannot use its reserved blocks directly.
Patch eviction bounds patch history, while retained special tokens still grow
with the sequence, and the current runner also has a 1024-frame sequence limit.

Short tests gave 148.4 ms at 112 threads and 148.4 ms at four threads. The forward
pass took 141.3 ms and original postprocessing about 0.8 ms. One camera iteration
took 144.0 ms at four threads: only about 3% faster. Keep four by default.

The active attention implementation repeatedly concatenates and clones cached
keys/values. Inspection of actual tensors found **FP32 keys and BF16 values**, even
though aggregator weights use BF16. The first layer's cache shape after scale
initialization is `[1, 16, 2, 1375, 64]`. These casts, copies, and a growing attention
sequence are more important than the sub-millisecond postprocess.

### BF16 stored-key diagnostic

Casting only aggregator cache storage to BF16 after each frame gave bit-identical
depth, confidence, intrinsics, and extrinsics on sixteen subsequent same-image
steps. At the end of the long test, allocated memory fell from 15.16 to **11.00
GiB**, but peak allocation remained **16.44 GiB**, and late frame time was **513.8
ms**, slightly slower than baseline.

This is not a production optimization: each new FP32 key causes concatenation
to promote the stored cache again, and the post-frame conversion adds copying.
A first implementation retaining `list(cache.items())` while replacing tensors
also failed with an allocator assertion; retaining only keys let the isolated
probe finish. The failed attempt is preserved in the artifact directory.

A useful adapter must preserve the effective SDPA inputs while avoiding promotion,
full-cache cloning, and simultaneous old/new allocations. The preliminary
equivalence result supports further work; it does not validate arbitrary cache
precision changes, other backends, or natural long sequences.

### Memory budget with all models

| Resident component | Measured isolated allocation |
| --- | ---: |
| LingBot with filled original patch cache | 15.16 GiB |
| Qwen after generation | 8.30 GiB |
| SAM after image/query inference | About 3.47 GiB |
| Sum, before additional process/context/workspace headroom | **26.93 GiB** |
| Visible device capacity | **19.63 GiB** |

These are individual measurements, not a simultaneous peak-memory measurement.
Nevertheless, the resident sum already exceeds capacity. Even the experimental
11-GiB LingBot storage plus Qwen/SAM totals about 22.8 GiB. Cache storage work alone
cannot make the unchanged 64-window configuration fit with all models resident.

FlashInfer's existing page pool also reserves substantial headroom: patch pages
are `scale + window + 16`, with additional special-token pages and headroom.
Enabling it unchanged can allocate a large pool immediately. Estimate its
actual bytes before enabling it alongside Qwen and SAM. Paged attention is a
candidate, not a guaranteed memory or speed solution.

## Isolated Qwen measurements

Model: `Qwen/Qwen3-VL-4B-Instruct`; 4,437,815,808 parameters; BF16 weights;
greedy generation; unchanged instruction; maximum 128 output tokens.

| Intra-op threads | Preprocessing median | Generation median |
| --- | ---: | ---: |
| 112 | 13.4 ms | 733.4 ms |
| 4 | 8.7 ms | 734.4 ms |
| 1 | 12.2 ms | 732.3 ms |
| 8 | 8.3 ms | 734.1 ms |

All sixteen calls produced the same response, with **587 input tokens and 35
output tokens**. Image grid `[1,32,32]` and spatial merge imply 256 image tokens.
Allocated memory was 8.30 GiB; peak 8.46 GiB. At this input, generation is about
48 output tokens/s including prefill, not a pure decode-only rate.

Thread limits help small preprocessing variability, not generation throughput.
The 128-token ceiling does not force 128 tokens of work: this response stops
at 35. Simply lowering the ceiling will not accelerate it unless output changes.

The factual `ground` sentence is not used by SAM/mapping. A prompts-only response
could reduce decoding work, but deleting that instruction could change hazard
recognition and must be evaluated on all 18 videos and additional water/hole
examples. Quantization is primarily a memory candidate here; its speed and
proposal quality must be measured. TensorRT being installed does not establish
an available, compatible Qwen-VL inference engine. Qwen's existing SDPA support
is documented in [the official Transformers model documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_vl).

VLM proposals already run asynchronously after the first call. Its image precedes
the instruction in the message, so caching a stable text prefix is not directly
applicable without changing message ordering and validating output behavior.

## Isolated SAM measurements

SAM's internal image encoder receives **1008 × 1008**, even though LingBot RGB
and returned masks are 518 × 518. The plugin already embeds each image once and
reuses that embedding for all prompts; repeated image encoding per prompt is
not the problem. CUDA preprocessing and mixed precision are also already used.

| Prompts | Image encoding median, 112 threads | Queries median | Total median |
| --- | ---: | ---: | ---: |
| 1 | About 91 ms | About 42 ms | About 133 ms |
| 3 | About 91 ms | About 125 ms | About 216 ms |
| 8 | About 91 ms | About 332 ms | About 424 ms |

Four/one/eight thread variants stayed close to these times. Each query currently
re-encodes its text, transfers every retained instance mask to CPU, and performs
the final union in NumPy. The diagnostic candidate stores text features on the
GPU and transfers only the final native-resolution Boolean union.

| Prompts | Matched baseline query median | Cached text + GPU union | Saving |
| --- | ---: | ---: | ---: |
| 1 | 43.3 ms | 36.5 ms | 6.8 ms |
| 3 | 129.1 ms | 107.0 ms | 22.1 ms |
| 8 | 338.5 ms | 280.8 ms | 57.7 ms |

These exclude image encoding. For eight prompts the estimated SAM total falls
by about 13%, not by the 17% query-only reduction. Effects on the complete pipeline
are smaller because geometry, decoding, and projection remain.

The comparison passed on eighteen sampled frames, with 28 concept queries and
four frames having nonempty unions, plus the repeated timing cases. It does not
cover every object, frame, shape, model revision, or positive multi-instance case.
Cache keys and tensor ownership need explicit handling in a production adapter.
Preserve thresholding each resized instance mask before OR; combining logits
before interpolation can change results.

SAM exposes `build_sam3_image_model(compile=True)` but the current plugin leaves
it disabled. Compile the fixed-shape vision path first and measure warmup,
steady latency, memory, graph breaks, and fallback. Smaller encoder resolutions
or tracking are separate accuracy-changing modes; especially for small obstacles,
they should not replace native-resolution per-frame SAM by default.

## CPU, transfer, persistence, and viewer costs

| Operation | Median | Notes |
| --- | ---: | --- |
| First projection | 1.419 s | Includes importing official geometry dependencies. |
| Warm original projection/filter/coloring | 25.7 ms | 185,494 valid points for this frame; 40 repetitions. |
| Whole-history display reservoir update | 5.8 ms | Existing bounded 200k sample plus new frame. |
| NPZ prediction write/read on workspace | 5.34 ms | 2,952,908-byte payload; warm page cache. |
| Same NPZ operation on `/dev/shm` | 4.31 ms | Small benefit; still serialization and copies. |
| PLY append on workspace | 3.56 ms | 2,782,410-byte point payload; buffered flush, not durable `fsync`. |
| PLY append on `/dev/shm` | 2.94 ms | A diagnostic, not a recommendation to keep full maps in RAM. |
| Decode each original source frame | 4.87 ms | 197 frames; first decode 272 ms; does not benchmark selected `grab()` scheduling. |

Moving transfer NPZs to tmpfs by itself saves about one millisecond for this
payload. A bounded shared-memory buffer can avoid serialization and more copies,
but is secondary to 150–500 ms geometry and 130–420 ms SAM. Do not present IPC
optimization as the main throughput breakthrough.

The controller also performs display export, optional frame/mask compression,
PLY writing, and flushed event logging on the inference critical path. Separate
bounded consumers can reduce latency tails. Never discard full map observations
to make display updates faster. The viewer repeatedly updates unchanged status
text and computes percentiles over the sampled cloud; only update changed state
and compute bounds less often. Retain the 200k-point display limit established
for the Jupyter proxy.

## Complete-run evidence and throughput interpretation

The previous `streaming_final` run on the 197-frame log video retained 22 geometry
frames, mapped 11, made three VLM calls, and dropped seven scheduled samples.
It used one camera iteration, playback pacing, and saved frame artifacts:
9.94 seconds processing; 49.46 seconds recorded wall time; 2.21 geometry Hz;
1.11 map Hz. First map appeared after 43.13 seconds, dominated by sequential
model loads. These rates include initialization, pacing, and the final VLM drain.

This audit's `replay_baseline` uses the full same clip with **four camera
iterations, no pacing, no frame dropping**, source stride seven, map step two,
VLM interval two source seconds, and saved artifacts. It failed on the next
geometry request after 22 completed geometry frames / 11 mapped frames. Its
exception was:

```text
NVML_SUCCESS == r INTERNAL ASSERT FAILED ... CUDACachingAllocator.cpp:1131
```

The exception is an allocator assertion, not a clean reported out-of-memory
exception. Memory pressure is strongly supported by the measured budget, but
the exact NVML/allocator failure needs a dedicated reproduction and telemetry.
The partial output and logs are preserved. This failed run is not a throughput
benchmark of a completed video.

A rough optimistic serial model for one prompt is:

```text
time per geometry frame ≈ geometry + (SAM + projection + mapped-output work) / map_step
```

Using early medians and ignoring VLM contention, decoder/IPC costs and startup:
153 + (133 + 26)/2 ≈ 233 ms, about 4.3 geometry Hz / 2.1 map Hz. With the filled
cache: 500 + (133 + 26)/2 ≈ 580 ms, about 1.7 / 0.86 Hz. These are arithmetic
estimates, not measured sustained rates, and the current full cache cannot fit
with all models resident. Eight prompts make the serial budget larger.

At 29.97 source FPS the four-FPS target yields 4.28 retained FPS because of integer
stride rounding. Preserving every original source frame and producing semantic
maps at that rate is well beyond current measurements. Reporting faster playback
by dropping more frames would not demonstrate faster inference.

## Recommended direction

First build accurate timing/memory accounting and make memory use predictable.
Then implement the small demonstrated SAM gain and remove avoidable startup and
output latency. Pursue compact cache storage and the existing paged backend,
followed by selective compilation, only with measured peak-memory and output
equivalence gates.

For unchanged 64-window model settings and all models resident, a larger GPU
allocation is the straightforward capacity solution. A 40-GB-or-larger MIG
allocation is worth requesting from the machine owner, but it was not changed
or benchmarked here and provides no guaranteed speed multiplier. Staying within
this partition will likely require a combination of efficient cache storage,
Qwen/SAM memory reductions or a validated smaller temporal window. These have
different accuracy and engineering tradeoffs and are listed separately in the
implementation plan.
