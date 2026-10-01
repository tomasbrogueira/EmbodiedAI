# Hazard prompt evaluation

The task is one RGB image → a list of visible concepts to avoid → SAM text
segmentation. The VLM receives no SAM masks, crops, depth or image-specific
reference labels. Evaluate objects and hazardous surfaces, such as water and mud.
Geometry and planning decide whether a detected region obstructs the robot.

This revision replaces the region-classification experiment. Keep its code and
results as legacy material; new runs use `hazard_prompt_v1`. Implement the new
notebooks 10–13 before preparing a new visualization session.

## Output and models

Return only short concrete noun phrases, once per visible type:

```json
{"prompts": ["person", "puddle", "fallen log"]}
```

An empty successful list means no hazards were identified. Errors, truncation or
invalid JSON remain failures. Ask for no counts, boxes, explanations or confidence.
Preserve the original phrases for SAM; use a frozen alias table only for scoring.
The common policy provides the benchmark inventory without revealing which
concepts occur in an image. Additional nouns remain unscored, with counts reported.
This measures a declared vocabulary, not unrestricted hazard discovery.

Compare the existing `qwen3_vl_4b` and `qwen3_5_4b` checkpoints. Prepare
`qwen3_5_2b` as an explicit fallback if a larger model fails the measured budget.
Reuse loading/quantization logic; replace scene-plus-crop inference with one
whole-image request. Initial settings: 4-bit language weights, at most 1024 visual
tokens, context limit 2048, output limit 256, concurrency 1, thinking disabled.
Record actual token counts and preprocessing. These are starting settings, not
validated accuracy or memory claims.

SAM3 supports text concepts and all matching instances. Use one text query per
phrase, a fixed image-model checkpoint and the same settings for every condition.
SAM/SAM2 require a separate grounding stage for this text-first interface.
[SAM3 implementation](https://github.com/facebookresearch/sam3),
[SAM2 interface](https://github.com/facebookresearch/sam2).

## Automatically labeled data

| Source | Development | Test | Purpose |
|---|---:|---:|---|
| RELLIS: 00000 / 00001–00004 | 20 | 80 | Outdoor obstacles and unsafe surfaces |
| COCO val2017, selected image IDs | 8 | 32 | People, cups, bottles, chairs and dogs |

Reuse the RELLIS RGB/ID imports; derive concept presence directly from semantic
IDs. SAM-first mask generation, region selection and mask-purity references are
unnecessary. Water and puddle share the `water` scoring group. Sky is not an
obstacle prompt. Void, grass and unspecified `object` are excluded from named
references and pixel evaluation. Presence means at least one annotated pixel;
keep tiny hazards and report their area. Absent-concept precision is scored only
where the declared annotation-coverage rule permits it.

For COCO, select 40 images deterministically from instance annotations: 32 with
at least one of the five target classes and 8 without those classes. Split them
as 6/26 positives and 2/6 target-negative controls for development/test. Balance
available classes and small targets, freeze image IDs before model inference,
and report missing strata instead of inventing examples. Only those five classes
are scored; a target-negative control need not be free of other hazards. Union
same-class instance masks, including crowds; do not report instance counts from
RELLIS. Download only selected COCO images, with explicit opt-in actions.

These datasets support object/material-policy references. They do not establish
that a cup is on the floor or that a route is physically safe. Phone hazard clips
and cable/spill cases remain optional pending inputs. No bulk manual annotation
or VLM-generated ground truth is required.
[RELLIS labels](https://raw.githubusercontent.com/unmannedlab/RELLIS-3D/main/benchmarks/SalsaNext/train/tasks/semantic/config/labels/rellis.yaml),
[COCO downloads](https://cocodataset.org/#download),
[COCO mask/download API](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py).

## Experiments

1. **VLM discovery on all frozen test images.** Report annotated concept recall
   (`TP / reference concepts`), annotation-supported precision (`TP / scored
   predicted concepts`), missed-concept counts by class and tiny-target area,
   parse/inference failures and unscored/vague phrases. Use image–concept pairs,
   not object instances. Report source and split separately, with numerators and
   denominators. Failed responses miss all annotated positives; missing prediction
   records block a comparison until the run is complete. Empty denominators are
   null. Never describe an unscored noun as a confirmed hallucination.
2. **SAM handoff on 20 fixed test images, 10 per source.** Compare each VLM's
   actual phrases, reference-present canonical phrases, and the full fixed
   16-concept policy list without a VLM. Freeze selection before predictions.
   Keep SAM checkpoint, image encoding, thresholds and postprocessing identical.
   Report annotated hazard-pixel recall/precision/IoU, failures and SAM query count.
   Pixel metrics use each source's scored concepts and evaluable pixels; export
   the all-query mask union separately for downstream use. The reference-present
   condition diagnoses segmentation, rather than supplying a VLM score or a
   guaranteed upper bound. The fixed-list baseline tests whether the VLM saves
   queries without losing coverage. These are combined VLM/SAM results.
3. **Deployment cost on the same 20 images.** Five fixed development warmups,
   two measured repeats, median/p95 time per image, failures, incremental and total
   GPU peaks, and prompt count. Profile VLM generation and SAM independently;
   record combined warm processing cost only from an actual combined run. Measure
   loading separately. The VLM target remains 6 decimal GB on a shared 20 GB GPU;
   SAM memory is additional. Joint residency and full-pipeline fit require their
   own measured replay. Separate-process peaks cannot establish joint fit.

Tune only on development data. Select a VLM using hazard recall, supported
precision, failures and measured cost. No model is the winner before measurements.
Temporal fusion and route consistency remain whole-pipeline evaluations.

## Setup

Use `VLM_evaluation/` and configurable external data/run/cache roots. The official
SAM3 path requires Python 3.12+, so prepare a compatible GPU environment during
integration. Checkpoint access may require the user's approval by Meta/Hugging
Face; support pre-cached weights and clear unavailable states. Do not download
weights/data, run GPU work or connect to a server during prompt implementation.
No visualization implementation is part of this revision.
