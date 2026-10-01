# Whole-frame deployment benchmark

Install the package as described in [setup.md](setup.md), run Notebook 00, then
open `notebooks/04_deployment_benchmark.ipynb`. Its top configuration selects
roots, a prepared run, one model and an optional settings JSON. Both real and
fake execution flags default to false. The real default is `clip_vit_b32`;
`qwen3_vl_4b` and `qwen3_5_4b` run separately. `qwen3_5_2b` is an explicitly
selected fallback, never an automatic substitute. `DEVICE` explicitly selects
one visible CUDA device and defaults to `cuda:0`.

The default public run contains 100 RELLIS frames: 20 development frames from
00000 and 80 test frames from 00001 through 00004. Its
[automatic material-policy references](automatic_references.md) replace bulk
manual labeling. Phone clips are an optional extension; their absence does not
block the public benchmark. Profiling consumes only RGB, planning-relevant SAM
masks and the common policy, never reference labels or semantic-ID masks.

## Execute one backend

Set `RUN_FAKE_BENCHMARK = True` for a small CPU fixture. It creates explicitly
marked fixture data and a separate profile beneath the configured external
roots; it needs no model adapter, downloads or GPU. Its results are software
checks and cannot be cited as measured model performance.

For a real profile, prepare `frames.jsonl` and `regions.jsonl`, install inference
dependencies, explicitly cache the chosen checkpoint through the inference
workflow, calibrate CLIP using development frames, and set
`RUN_BENCHMARK = True`. `CLIP_CALIBRATION_PATH` defaults to
`metadata/clip_calibration.json` within the prepared run, unless supplied in the
settings dictionary. `MODEL_CACHE_PATH` can override that dictionary's cache
path; otherwise the default is `<cache_root>/huggingface/hub`, matching Notebook
02. Relative model-cache paths resolve under the cache root; relative calibration
paths resolve under the prepared run. Notebook 04 forces offline checkpoint
loading. Missing weights, dependencies or valid frame coverage produce errors
instead of downloads. Run one backend per action. The runner calls `close()` only
on the backend it created and never directly clears another module's resources.
Real loading requires a dedicated idle process/kernel: it refuses measurable
nonzero allocated or reserved Torch memory before loading. Other modules may
remain resident in separate GPU processes; device-wide measurements include
them. Restart the benchmark kernel if its allocator baseline is nonzero.
Current peer adapters call `torch.cuda.empty_cache()` during close, clearing
unused allocator caches across the current process. The idle-kernel safeguard
protects measurable resident Torch allocations from this operation, but their
cleanup still needs a peer/integration fix before sharing one kernel with other
modules. The benchmark does not alter peer adapters or clear other GPU processes.

`configs/benchmark/default.json` records the fixed counts, seed, 6,000,000,000-byte
target and initial token/quantization settings. `robot_policy: null` asks the
backend to resolve its canonical `rellis_material_v1` policy. An optional settings JSON replaces
the whole settings dictionary: use the exact frozen settings from inference,
including a development-calibrated CLIP vocabulary/rejection threshold where
applicable. Actual settings/checkpoint/processor/quantization come from the loaded
backend when available; the notebook supplies `metadata/<model_key>.json` for
comparison if present. Missing actual fields and sidecar mismatches are explicitly
marked. Input settings alone are not evidence of actual preprocessing or
quantization. Review `metadata.comparison_metadata_complete` and
`summary.comparison_ready` before making a model comparison. Do not change
settings or offload mid-profile.

The public API is:

```python
selection = freeze_selection(run_dir, data_root, seed=0)
result = profile_backend(
    model_key, settings, selection, data_root=data_root,
    output_dir=unique_profile_dir, inference_metadata=metadata,
)
summary = result["summary"]
```

Tests can inject `backend_factory(model_key, settings)`, a clock and a memory
probe only with explicit `fixture=True`. Fixture budget verdicts remain null,
even with injected GPU statistics. The production loader uses the fixed `load_backend` / `predict_frame` /
`close` inference API. Frozen selection contains five development and 20 test
frames joined to every planning-relevant region, with stable IDs/timestamps,
source coverage and an input fingerprint. All selected-region flags are ignored
for profiling. Reuse the selection across models; insufficient frames fail and
are never padded with synthetic data.

## Timing, failures and memory

Loading is measured separately. Five development calls warm up the backend;
the first is also recorded as first-call/cold-start. None enters the inference
aggregate. Two repeats over the fixed test set give 40 measured frame calls.
Wall-clock time surrounds the entire `predict_frame`, including target/image
preparation, every region request and parsing. Synchronization before/after the
call prevents asynchronous CUDA work from escaping that interval; see
[PyTorch synchronization](https://docs.pytorch.org/docs/2.8/generated/torch.cuda.synchronize.html).

Profiles record region counts, median/p95 frame latency, calls with failures and
failed-region denominators. Missing, duplicate, unexpected or incorrect-frame
predictions count as failures. Failed attempts retain their time in all-attempt
statistics; successful-only statistics are also reported. Fatal GPU errors stop
that backend, preserve partial results and require explicit investigation before
another run. There are no automatic retries or memory-setting changes.

Allocator and device-wide memory have different meanings:

- **Allocator allocated/reserved peaks:** exact process allocator counters, with
  increments above the pre-load resident baseline so model weights are included.
  Baseline collection occurs after telemetry/CUDA context initialization, so
  that initialization overhead is excluded and recorded in `baseline_scope`.
  Counters do not cover CUDA allocations outside PyTorch. Resetting peaks affects
  the current process's measurement counters, so use a dedicated idle benchmark
  kernel. See [PyTorch memory limits](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html).
- **Device-wide used memory:** baseline, capacity, observed peak and incremental
  usage include all processes. The peak is sampled and is a lower bound; record
  the sampling interval. Other processes' changing allocations affect the delta.
  See [global memory information](https://docs.pytorch.org/docs/2.8/generated/torch.cuda.memory.mem_get_info.html).
- **Unavailable measurements:** JSON `null` plus a reason. CPU tests never imply
  zero GPU usage or a passing budget.

Separate loading/cold-start/warmed peaks are retained. The proposed component
target is 6 GB on a shared 20 GB GPU. Allocator measurements and sampled shared
device usage do not establish the final full-pipeline budget; the group must
replay under active component load.

Profiles live under `<run>/benchmark/<model_key>/<profile_id>/` and contain the
frozen selection, per-call samples, summary and configuration metadata. An
existing profile directory is refused. These artifacts do not overwrite
classification predictions or annotations. Retain actual real-source coverage,
the campus recording split limitation and fixture/metadata flags when reporting
results. The selected GPU's identity/runtime is recorded when available. No GPU
or real-data benchmark has been run as part of this local setup.
