# Traversability evaluation interfaces

Legacy region-classification contract. The current text-first hazard task uses
[hazard interfaces](hazard_interfaces.md) and [protocol](hazard_protocol.md).

The VLM produces raw current-frame semantic labels. Geometry, voxel fusion,
temporal consistency and planning belong to the wider pipeline.

The default public benchmark now uses [automatic dataset-policy references](automatic_references.md).
Manual selection/annotation is optional. Existing record formats remain compatible;
the revised policy, sampling and provenance rules in that document supersede the
earlier manual-labeling recipe below for the public benchmark.

## Paths and artifacts

`TRAVERSABILITY_DATA_ROOT` holds input images and aligned binary masks.
`TRAVERSABILITY_RUN_ROOT` holds generated runs. Defaults may be `data/` and
`runs/` relative to `VLM_evaluation/`, but both roots must be configurable.
Use POSIX relative paths in manifests and resolve them against the data root.
Do not store machine-specific absolute paths in portable manifests.

A prepared run contains:

```text
<run_root>/<run_name>/
  frames.jsonl
  regions.jsonl
  annotations.jsonl
  predictions/<model_key>.jsonl
  metadata/<model_key>.json
  evaluation/
  benchmark/
```

JSONL files contain one UTF-8 JSON object per line. IDs are stable and unique
within the run. Writes must preserve existing annotations and predictions.

## Frame record

Required fields:

```json
{
  "frame_id": "source:sequence:frame",
  "source": "rellis",
  "scene_id": "physical-location-id",
  "sequence_id": "00001",
  "timestamp_s": 0.0,
  "image_path": "rellis/00001/frame.png",
  "split": "test"
}
```

`timestamp_s` may be null for static inputs without verified timestamps. Do not
invent timestamps. Splits are `development` or `test`. RELLIS sequence splits
are held-out recordings within one campus, not independent geographic sites.

## Region record

```json
{
  "region_id": "source:sequence:frame:region",
  "frame_id": "source:sequence:frame",
  "mask_path": "prepared_masks/source/sequence/frame/region.png",
  "planning_relevant": true,
  "selected_for_classification": true
}
```

Masks are single-channel PNG files: zero outside the region and nonzero inside.
Mask dimensions must match the RGB image. Benchmark all planning-relevant
regions; classification uses the frozen selected subset.

## Annotation record

```json
{
  "region_id": "source:sequence:frame:region",
  "reference_label": null,
  "semantic_class": null,
  "hazard_type": null,
  "mask_quality": "unchecked",
  "annotation_status": "pending",
  "robot_profile_id": "small_wheeled_v1"
}
```

Labels are `traversable`, `non_traversable` or `unknown`. Missing annotations
remain null with status `pending`; they are not reference-unknown examples.
Completed annotations have status `complete` and a valid reference label.
Mask quality is `valid`, `mixed`, `broken` or `unchecked`. Primary classification
metrics use completed annotations with valid masks. Other masks are reported
separately. Join artifacts by ID, never by row order.

Proposed policy `small_wheeled_v1`: permit apparently dry floors, concrete,
asphalt, compact soil and short grass, conditional on geometric feasibility.
Avoid people, animals, liquids, mud, dense vegetation, fragile belongings,
cables and loose obstacles. Unclear material or target identity is unknown.
Freeze the policy before annotation. Numeric slope/clearance are not VLM claims.

## Prediction record

```json
{
  "region_id": "source:sequence:frame:region",
  "frame_id": "source:sequence:frame",
  "model_key": "qwen3_vl_4b",
  "label": "unknown",
  "semantic_class": null,
  "reason": "",
  "raw_response": "",
  "status": "ok",
  "error_code": null
}
```

Each requested region has exactly one prediction. Failed parsing or inference
returns unknown with `status: error` and a diagnostic error code; failures remain
in evaluation denominators and are reported separately. Do not convert a missing
reference label into an unknown label. Preserve raw responses before fusion.

Model keys: `clip_vit_b32`, `qwen3_vl_4b`, `qwen3_5_4b`; fallback
`qwen3_5_2b`. Metadata records exact checkpoint revision, preprocessing,
quantization, prompt, vocabulary/policy, decoding settings and software versions.

## Inference API

The `traversability_inference` package exports:

```python
def load_backend(model_key: str, settings: dict):
    """Return a loaded backend; importing the package loads no weights."""

# Backend methods:
def predict_frame(self, frame: dict, regions: list[dict], data_root):
    """Return prediction records for every supplied region."""

def close(self):
    """Release this backend's resources."""
```

`data_root` is a pathlib-compatible path. The backend prepares the outlined
scene and 20%-padded crop, runs requests and parses results inside `predict_frame`.
The benchmark times this whole method, including image preparation, every target
request and parsing. Loading is measured separately; warm-up is excluded.

Settings keys: `robot_profile_id`, `robot_policy`, `language_quantization_bits`,
`scene_visual_token_budget`, `crop_visual_token_budget`, `context_token_limit`,
`max_new_tokens`, `request_concurrency`, `seed`, `enable_thinking`.
Initial values: 4-bit language weights, scene 384/crop 256 visual tokens,
context 2048, output 128, concurrency 1, seed 0 and thinking disabled.
Record the actual processor settings because models tokenize images differently.
CLIP uses a frozen crop-based semantic vocabulary and development-set rejection
threshold. Do not invent a calibrated VLM confidence probability.

## Evaluation and resources

Report unsafe acceptance (predicted traversable / reference non-traversable),
useful acceptance (predicted traversable / reference traversable), predicted
unknown rate, reference-unknown acceptance and the three-way confusion table.
Publish numerator/denominator counts, errors, pending counts and per-source
results. Empty denominators produce null, not zero. Missing/duplicate predictions
must be detected. Development rows are not final test results.

Proposed classification sample: 100 public RELLIS keyframes, up to five regions each.
Use 20 development and 80 test frames with automatic references as described above.
The data recipe is in
`research/traversability/data_manifest.json`; own hazard clips remain unavailable.
Tests may use clearly identified synthetic fixtures, excluded from experiment
results. Real-data coverage must be reported explicitly.

Proposed component budget: 6 GB on a shared 20 GB GPU. This is unmeasured.
Profile one model at a time, using 20 fixed held-out frames, five development
warm-up frames and two measured repeats. Record regions per frame, median/p95
frame time, incremental peak memory, total peak memory and failures. A final
full-pipeline replay must verify the budget under active component load.

## Notebook behavior

Notebook order: setup, data/annotation, inference, classification, profiling.
Notebooks import reusable package functions and expose their configuration at
the top. They locate `VLM_evaluation/` without a Windows-only absolute path.
Downloads, model loading and GPU workloads are opt-in. Missing annotations,
hazard clips, dependencies or hardware produce clear actionable messages.
Keep notebook outputs cleared for version control.
