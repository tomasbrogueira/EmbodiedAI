# Raw semantic inference

`traversability_inference` implements the fixed prediction schema in
[interfaces.md](interfaces.md). The default `rellis_material_v1` policy matches
[automatic references](automatic_references.md); choose `small_wheeled_v1`
explicitly for the broader floor/short-grass policy. Qwen accepts explicit custom
policy text; CLIP requires one of the exact frozen policies because its vocabulary
mapping is fixed. Only RGB and SAM masks enter predictions.

Use an isolated Python 3.11+ environment. The proposed GPU stack is PyTorch
2.13.0, torchvision 0.28.0, Transformers 5.18.0, Accelerate 1.15.0 and
bitsandbytes 0.50.2. These are documented-compatible candidates awaiting GPU
validation, based on the [Transformers release](https://github.com/huggingface/transformers/releases/tag/v5.18.0)
and [official PyTorch combinations](https://pytorch.org/get-started/previous-versions/).
Choose a CUDA wheel compatible with the machine's driver; for example:

```bash
python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements/inference.txt
```

Run these commands from `VLM_evaluation/`. If this component is not installed,
add its `src` directory to `PYTHONPATH` or
`sys.path`. The notebook handles this without changing shared setup files.
All production adapters require one explicitly selected CUDA device. Qwen uses
NF4 language weights, double quantization and BF16 compute when supported, FP16
otherwise; `model.visual` and `lm_head` stay floating point. Loading audits actual
module quantization, dtypes and placement, and refuses CPU/disk offload. No adapter
changes model automatically. The 6 GB component target remains **unmeasured**.

| Explicit model key | Checkpoint | Frozen revision |
|---|---|---|
| `qwen3_vl_4b` | Qwen/Qwen3-VL-4B-Instruct | `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| `qwen3_5_4b` | Qwen/Qwen3.5-4B | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| `qwen3_5_2b` | Qwen/Qwen3.5-2B, optional fallback | `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| `clip_vit_b32` | openai/clip-vit-base-patch32 | `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` |

Revisions were checked using official metadata for
[Qwen3-VL](https://huggingface.co/api/models/Qwen/Qwen3-VL-4B-Instruct),
[Qwen3.5-4B](https://huggingface.co/api/models/Qwen/Qwen3.5-4B),
[Qwen3.5-2B](https://huggingface.co/api/models/Qwen/Qwen3.5-2B) and
[CLIP](https://huggingface.co/api/models/openai/clip-vit-base-patch32).
The [pinned 4-bit quantizer](https://raw.githubusercontent.com/huggingface/transformers/v5.18.0/src/transformers/quantizers/quantizer_bnb_4bit.py)
consumes the exclusion list; the
[Qwen3.5 template](https://huggingface.co/Qwen/Qwen3.5-4B/raw/main/chat_template.jinja)
supports disabling thinking.

## Loading and inference

Imports never load models or download weights. Cache downloads are a separate,
explicit notebook step. Set `local_files_only=True` after downloading. Start with
[default.json](../configs/inference/default.json), adding machine-local paths:

```python
import json
from pathlib import Path
from traversability_inference import load_backend, run_inference

settings = json.loads(Path("configs/inference/default.json").read_text())
settings.update(cache_dir="/path/to/cache", local_files_only=True)
backend = load_backend("qwen3_5_4b", settings)
try:
    predictions = run_inference(
        "/path/to/run", "qwen3_5_4b", settings, "/path/to/data", backend=backend
    )
finally:
    backend.close()  # idempotent
```

Alternatively omit `backend`; the runner loads and closes its own adapter.
`backend.predict_frame(frame, regions, data_root)` also accepts benchmark-supplied
regions directly. The runner defaults to frozen `selected_for_classification`
regions; `region_ids=[...]` requests an explicit subset. It joins by IDs, preserves
the input manifests and their timestamps, and returns all accumulated predictions.
Preparation, inference, parsing and malformed backend results yield one `unknown`
error record per requested region, with a diagnostic code and available raw text.
Loading, invalid manifests and resume incompatibility raise before prediction
publication.

Preparation draws a deterministic one-pixel exterior magenta outline and crops
the original RGB bounding box with `ceil(20% * width/height)` on **each side**,
clipped to the image. Empty, color or misaligned masks fail. Qwen scene/crop
dimensions are independently aligned to the loaded processor and capped at
384/256 merged visual tokens (393,216/262,144 pixels at factor 32). Further
resizing is disabled and actual grids/placeholders are checked. Both Qwen families
use the same JSON-only policy prompt, disabled thinking, seed 0, greedy decoding,
concurrency one, 128 output tokens and at most 2,048 input-plus-output tokens.
Parsing rejects duplicate keys, invalid types/labels, extra keys, fences and prose.

## CLIP development calibration

Calibrate separately before loading a classification adapter:

```python
from traversability_inference import calibrate_clip

settings["calibration_path"] = "/path/to/run/metadata/clip_calibration.json"
calibration = calibrate_clip("/path/to/run", settings, "/path/to/data")
# Then use load_backend/run_inference with model_key="clip_vit_b32".
```

The helper scores only selected development regions with complete annotations,
valid masks and the matching robot profile. Successful examples from both
`traversable` and `non_traversable` references are required. Test references and
reference masks/fractions do not participate. The frozen crop vocabulary chooses
the highest normalized image/text cosine match; its semantic class maps through
the selected policy. Scores are uncalibrated diagnostics saved in `raw_response`.

The inclusive rejection threshold maximizes observed useful acceptance subject
to zero observed unsafe acceptance; equal optima prefer the higher threshold.
If useful acceptance cannot meet that constraint, the artifact explicitly records
`reject_all` and classification returns unknown. Provenance includes development
IDs, input/configuration fingerprints and observed counts. This gives no held-out
safety guarantee. Classification requires an explicit compatible calibration file;
changed scoring settings, weights, software or development inputs require a new
calibration.

## Outputs, resume and checks

The runner holds an exclusive per-model process lock and publishes immutable
`metadata/<model_key>.json` before `predictions/<model_key>.jsonl`. Metadata records
the actual adapter configuration and a fingerprint of all frozen frame/region
records and RGB/mask contents, including deterministic missing-file sentinels.
Local roots and requested subsets do not affect identity. Injected fixtures are
identified separately and cannot resume production model outputs.

Predictions are flushed to same-directory temporary files and atomically replaced
after every frame. Compatible completed subsets and errors survive resume;
abandoned temporary files are ignored. Incompatible metadata, orphaned outputs,
duplicate IDs and corrupt published JSONL stop the run. Use a fresh run directory
when intentionally changing inputs or experiment configuration.

Run CPU checks without Torch, Transformers, downloads or CUDA:

```bash
python -m pip install numpy Pillow pytest
python -m pytest tests/inference -q
```

These tests use temporary synthetic inputs and injected or mocked models, covering
preparation, strict parsing, ID joins, completeness, resume, interrupted publication,
annotation isolation, calibration boundaries and model-loading options. They do
not create measured experiment results. Follow
[02_vlm_inference.ipynb](../notebooks/02_vlm_inference.ipynb) for separate opt-in
download, calibration, loading, inference and cleanup steps. GPU loading, actual
processor/generation integration, latency and memory under shared component load
remain pending; no INESC server was contacted.
