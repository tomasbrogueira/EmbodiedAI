# Hazard SAM3 segmentation

`traversability_hazard_segmentation.load_segmenter(settings)` loads the official
SAM3 image model lazily. Importing the package or adapter loads no model library
or weights. `segment_image(frame, prompts, data_root)` accepts only the original
RGB/frame identity and original phrases; image-specific references do not enter
this adapter. `close()` releases its own references without clearing global CUDA
caches or closing caller-owned injected resources.

## Safe defaults and setup

Start from `configs/hazards/segmentation.json`. Model loading, downloads and fixture
mode are disabled by default. Prepare an **isolated Python 3.12+ environment**.
The proposed unified GPU stack is Torch 2.10.0/torchvision 0.25.0 CUDA 12.8,
NumPy 1.26.4, Pillow 11.3.0, Transformers 5.18.0, Accelerate 1.15.0,
bitsandbytes 0.50.2 and huggingface-hub 1.31.0. The four hazard fragments and
the pinned official SAM3 source belong in one selected isolated environment;
use the explicit hazard GPU profile in [setup](setup.md). An approved pre-cached
source checkout can replace online VCS installation. Model loading and GPU
compatibility remain untested; setup preparation does not establish either.

The [pinned official README](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/README.md)
requires Python >=3.12, PyTorch >=2.7 and CUDA >=12.6. Its installation example
uses Torch 2.10.0 with CUDA 12.8; [PyTorch's compatibility table](https://pytorch.org/get-started/previous-versions/)
pairs that release with torchvision 0.25.0 on Linux/Windows. SAM3's
[pinned package metadata](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/pyproject.toml)
requires NumPy >=1.26,<2. All active hazard fragments now share NumPy 1.26.4 and
the Torch pair above. The selected CUDA wheel must be checked against the target
driver; a portable server setup must inspect that server's driver separately.
Real SAM3 platform/dependency, checkpoint and GPU validation is pending;
CPU mocked tests do not establish it.
The adapter's 15 CPU mocked/provenance tests were actually run with Python
3.12.13, NumPy 1.26.4 and Pillow 11.3.0. Torch, torchvision, SAM3 and checkpoint
loading were mocked; their proposed versions were not installed or tested.

The code pin is `2345a4ad109ac29c569da749c91d84f10dc08c40`. Install from this
official Git VCS URL so its `direct_url.json` records the immutable commit, or
set `code_source_root` to the clean checkout containing the imported package.
The loader verifies that provenance and the official builder/processor Git blobs.
It refuses missing/mismatched provenance or source content.

The checkpoint is official `facebook/sam3/sam3.pt`, revision
`3c879f39826c281e95690f02c7821c4de09afae7`, 3,450,062,241 bytes, SHA256
`9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`.
These values were verified using the [official immutable metadata API](https://huggingface.co/api/models/facebook/sam3/revision/3c879f39826c281e95690f02c7821c4de09afae7?blobs=true),
without downloading weights. Access is manually gated. Use `cache_root` or
`TRAVERSABILITY_CACHE_ROOT` for an existing cache. Set `checkpoint_path` to a
POSIX-relative file under that root for local weights, or leave it null to resolve
the exact pre-cached revision with `local_files_only=True`. Absolute/escaping
checkpoint paths are rejected. Every actual file is checked against its official
size and SHA256 before loading. Downloading requires both `allow_downloads=True`
and `local_files_only=False`; it remains an explicit integration action.
Relative cache and code checkout roots resolve against `VLM_evaluation/`;
absolute runtime roots are retained, regardless of the caller's working directory.

Real runs additionally require `enable_model_loading=True` and one explicit
indexed device such as `cuda:0`. The wrapper selects that device context while
passing the literal `"cuda"` to the upstream builder, which otherwise does not
transfer a model when given `"cuda:0"`. Missing code, checkpoint, dependencies or
CUDA produce `SegmenterUnavailable` with a diagnostic error code. There is no
SAM2, Transformers grounding, random-weight or CPU real-model fallback.

Real processor calls scope CUDA inference mode and BF16 autocast to the selected
device, following the [pinned official image example](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/examples/sam3_image_predictor_example.ipynb).
The fused image layers produce BF16 tensors; their floating outputs are promoted
to float32 before NumPy capture. Fixture calls do not import Torch or enable CUDA
contexts. These settings describe execution; they do not establish GPU fit or
complete benchmark validation.

## Processor behavior and records

The [pinned processor](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/sam3/model/sam3_image_processor.py)
receives original PIL RGB without EXIF rotation or adapter resizing. Each image
gets one `set_image`; every usable original phrase gets `reset_all_prompts`
followed by `set_text_prompt`. Reset clears language/geometric/output state while
preserving image features. Frozen vague phrases yield explicit failed queries
without a broad SAM request. Distinct scoring aliases remain distinct prompts.

Starting settings are resolution 1008, confidence 0.5 and mask probability 0.5.
Upstream uses strict `>` confidence filtering, bilinear mask interpolation to
original dimensions, sigmoid, and an existing strict `>0.5` mask threshold.
Its misleadingly named `masks_logits` holds probabilities; the adapter copies
those to CPU and applies the configured strict mask threshold. Auditable settings
record both thresholds and postprocessing. Every validated returned instance and
score is copied before another query mutates state; query/frame unions retain
original RGB dimensions. Successful empty lists/no detections are empty masks.
Image errors and query errors remain errors, with partial validated pairs and
their unions retained diagnostically. SAM call counts include attempted text
calls, excluding reset failures and vague phrases.

The result contains `queries`, `frame` and `settings`. The diagnostic controller
adds exact condition/query IDs, scoring bookkeeping, asset paths, atomic records
and resume identity defined in `docs/hazard_interfaces.md`. Reference-present and
fixed-policy behavior belong to that controller. Runtime roots are omitted from
adapter metadata; checkpoint provenance, applied settings, policy/alias hashes,
software versions and fixture provenance remain auditable.

For CPU tests, `load_segmenter({"fixture": True, "processor": mock})` or
`Sam3Segmenter({"fixture": True}, processor=mock)` requires explicit fixture mode.
It records `fixture_id`, `fixture_identity`, and `execution_kind="fixture"`; such
outputs are synthetic evidence. No actual model dependencies or weights load.
The upstream loader's handling of missing checkpoint keys and import-time TF32/
visible-device-0 probing are recorded limitations awaiting real-run validation.

## Frozen selection and diagnostic controller

`freeze_selection(run_dir, *, data_root=None, policy_path=None,
test_frames_per_source=10, warmup_frames=5, seed=0)` uses explicit roots when
supplied, then `TRAVERSABILITY_DATA_ROOT` or the component's default data
directory. It reads only frames, references, dataset metadata and policy;
prediction contents cannot influence selection. Test IDs are ranked by seeded
SHA256 separately within RELLIS and COCO, ten each. Five development IDs are
ranked globally. Selection freezes image/reference asset hashes, policy/alias
identity and coverage. Freeze it after all data is prepared and before notebook
11 creates model metadata or predictions. First publication refuses existing
prediction/segmentation/profile artifacts. Insufficient counts remain incomplete;
changing frozen inputs or the selection requires a new run. `run_conditions` accepts explicit
data/run/cache roots and uses the same selection implementation internally.
Reference validation also checks source vocabularies, mask counts and dimensions,
hazard-mask membership in evaluable pixels, and absence eligibility. Exactly 5%
ignored RELLIS pixels remains eligible; lower coverage still preserves positives.
COCO uses exhaustive evaluable pixels for its five declared categories.

```python
from traversability_hazard_segmentation import run_conditions

# No loading by default. Enable only after preparing the isolated GPU setup.
summary = run_conditions("configs/hazards/segmentation.json")
```

For an enabled run, load that config as a dictionary, set `enable_real_run=True`,
and supply the runtime roots/local checkpoint settings. One owned SAM3 adapter
serves all four conditions with identical applied settings. Separate invocations
also validate requested settings and the actual adapter identity against all
existing conditions. Passing a segmenter
to `run_conditions` is fixture-only, including when resuming completed records.
References enter only the controller: reference-present uses that frame's names;
fixed-policy always uses all sixteen canonical phrases in policy order. VLM
conditions use saved original phrases, preserving water/puddle as two queries.
Unknown concrete phrases remain queries with null scoring bookkeeping; vague
phrases are explicit failures. Missing VLM rows leave a condition incomplete;
upstream errors produce frame errors, whereas successful empty lists produce
original-size empty successful masks. `completed_queries` counts successful
queries; `failed_queries` counts saved errors. A complete subset of conditions
is insufficient for `comparison_ready`; the comparison requires all four.

Artifacts follow `docs/hazard_interfaces.md`: exact-key query/frame JSONL files,
metadata, original-size single-channel instance/query-union/frame-union PNGs,
and the frozen selection. Paths in records are POSIX-relative to the run root.
Filesystem-safe asset names include a frame digest and invocation ID. Immutable
per-frame transaction journals under `.transactions/` commit assets and record
hashes before indexes are replaced atomically; frame indexes are published last.
Resume replays interrupted transactions without repeating model calls and
validates every completed asset, upstream prediction, model/settings and input
identity. Completed errors remain immutable. Missing masks are corruption errors.
Native inference record digests and latest attempts are checked before SAM calls;
an interrupted inference publication must be resumed first. Malformed Unicode
raw responses remain escaped error diagnostics through transactions and profiles.
Deliberate retries require a new run; no automatic error retry is performed.

Independent CPU fixtures are available from
`traversability_hazard_segmentation.fixtures.prepare_fixture(run_dir, data_root)`
using empty, separate roots and `FakeSegmenter()`. They freeze their selection
before generating marked fake predictions. Notebook 13 demonstrates the full
fixture workflow with cleared source outputs and all action switches disabled.
