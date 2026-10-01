# Hazard prompt interfaces v1

Read [the experiment protocol](hazard_protocol.md) and
`configs/hazards/policy.json` first. All paths are relative to `VLM_evaluation/`.
Parallel sessions may read these files, but integration alone may revise them.

## Identity and storage

Task ID and schema: `hazard_prompt_v1`, version 1. Default run name:
`hazard_prompt_v1`. Data/run/cache roots remain configurable through the existing
`TRAVERSABILITY_*_ROOT` environment variables. Paths stored in artifacts are
POSIX-relative to the appropriate root; reject escapes, duplicates and identity
mismatches. Never resume a legacy region-classification artifact as a hazard run.

```text
<run_root>/<run_name>/
  frames.jsonl
  references.jsonl
  predictions/<model_key>.jsonl
  metadata/dataset.json
  metadata/<model_key>.json
  evaluation/hazard_report.json
  segmentation/selection.json
  segmentation/<condition_key>/queries.jsonl
  segmentation/<condition_key>/frames.jsonl
  segmentation/<condition_key>/metadata.json
  benchmark/<condition_key>/<profile_id>/{samples.jsonl,summary.json,metadata.json}
```

Frame schema reuses `frame_id`, `source`, `scene_id`, `sequence_id`,
`timestamp_s`, `image_path`, `split`; add `width`, `height`, `image_sha256`.
Use null for unverifiable timestamps. Dataset metadata records task/schema,
policy and alias hashes, source annotation provenance/hashes, exact selected IDs,
sampling, fixture status, coverage and a reproducible dataset fingerprint.
References never enter the inference backend or image prompt.

Canonical JSON hashes mean SHA256 of UTF-8 JSON with sorted keys, compact
separators, `ensure_ascii=False` and nonfinite values rejected. `policy_sha256`
always hashes original policy file bytes; `policy_hash` hashes the canonical
policy object; `alias_sha256`/`alias_hash` hash canonical `policy.aliases`.
The active evaluator explicitly binds these producer fields in
`configs/hazards/evaluation.json`; it does not equate different recipes.
Inference transaction/audit fingerprints use the same canonical separators but
`ensure_ascii=True`, including arbitrary raw responses and Unicode phrases.
Hazard SAM/profile JSON and canonical hashes escape nonscalar Unicode code points
from failed raw model text with UTF-8 `backslashreplace`; valid Unicode bytes and
all other hash recipes remain unchanged. Such raw text remains an explicit error.

Data metadata keeps per-frame `annotations`, all original/reference `assets`,
`coverage.expected_frames_by_source_split`, `selected_frame_ids` of materialized
frames, `sampling` and source selection/provenance. Its `dataset_fingerprint`
hashes `{metadata,frames,references}` with only `dataset_fingerprint` removed
from metadata and frame/reference rows in frame-ID order. New data preparation
publishes `observed_input_signature` before inference: canonical
`{frames,references,assets}` with ID-sorted rows and RGB/valid/concept-mask asset
hashes (source label/annotation files are bound by dataset metadata separately).
Unchanged pre-existing data artifacts remain readable; an absent prepared input
pin does not establish a final real comparison. Changing frozen inputs requires
a new run.

Reference records contain:

```json
{
  "frame_id": "rellis:00001:frame",
  "scored_concepts": ["tree", "pole", "water", "vehicle", "building", "log", "person", "fence", "bush", "barrier", "mud", "rubble"],
  "present_concepts": ["tree", "water"],
  "absence_scoring_eligible": true,
  "concept_masks": {"tree": "hazard_references/FRAME/tree.png", "water": "hazard_references/FRAME/water.png"},
  "concept_pixel_counts": {"tree": 200, "water": 40},
  "valid_mask_path": "hazard_references/FRAME/valid.png",
  "annotation_source": "dataset_policy",
  "reference_scope": "annotated_hazard_concepts",
  "status": "complete"
}
```

Reference mask assets live under the data root, have original RGB dimensions and
use single-channel binary PNG. On RELLIS, ignore IDs 0/3/9; absent concepts are
score-eligible only when ignored pixels are at most 5% of the image. Present
concepts remain scored on lower-coverage frames. This is an operational
annotation-supported rule. Unexpected IDs are validation errors. COCO references
are exhaustive within the five selected annotated categories, with crowd masks
included in unions; no location/floor predicate is inferred.

## Whole-image inference

Model output is exactly `{"prompts": ["person", "puddle"]}`. Saved records:

```json
{
  "task_id": "hazard_prompt_v1",
  "schema_version": 1,
  "frame_id": "rellis:00001:frame",
  "model_key": "qwen3_vl_4b",
  "prompts": ["person", "puddle"],
  "raw_response": "{\"prompts\":[\"person\",\"puddle\"]}",
  "status": "ok",
  "error_code": null
}
```

Maximum 32 phrases, 80 characters per phrase; no silent truncation. Deduplicate
only exact normalized repeats while retaining an auditable original response.
Preserve distinct phrases even if the scoring alias maps them to one concept.
Malformed JSON, extra schema keys, empty entries, generation truncation and
inference failures produce `status:error`, a diagnostic and no successful-empty
claim. Unmapped concrete nouns remain valid output and are reported unscored;
frozen vague terms are flagged as unusable, never matched to every hazard.

```python
load_backend(model_key: str, settings: dict) -> backend
backend.predict_image(frame: dict, data_root) -> prediction_record
backend.close() -> None
run_inference(run_dir, model_key, settings, data_root, *, backend=None) -> summary
```

Package: `traversability_hazard_inference`. Imports load no model libraries or
weights. `predict_image` includes RGB loading/preprocessing, generation and
parsing. No region argument or fake region records. Metadata records checkpoint
revision, prompt/policy/alias hashes, preprocessing, token budgets/counts,
quantization, device, generation settings, software and input fingerprint.
Resume only records with matching identity/settings/input hashes. Preserve raw
responses, completed records and errors; retry only through an explicit option.

Model metadata uses `configuration` for policy/alias hashes, model pins and
semantic settings, `inputs` for full dataset metadata, frozen frame rows and RGB
content identity, and `execution.kind` for real versus injected fixture identity.
`attempts[frame_id]` preserves records, digests and per-call diagnostics;
`prediction_digests` binds published rows. Successful raw output may retain a
terminal EOS token: parse only the audited `parse_response_text` when generation
diagnostics verify `terminal_eos`, `stop_reason:eos`, the token ID and `eos_text`,
and exact concatenation reconstructs `raw_response`. Failed/truncated responses
keep error status and empty `prompts`; their raw text remains diagnostic.

## Data and evaluation

Package `traversability_hazard_data` exports
`prepare_run(config, *, fixture=False)` and `validate_run(run_dir, data_root)`;
it creates frames/references/metadata with opt-in downloads only.

Package `traversability_hazard_evaluation` exports
`evaluate(config) -> report` and `export_report(report, output_dir) -> paths`.
Read JSONL directly; do not depend on inference/segmentation implementations.
Report schema records task/version, fixture marker, identity, validation, coverage,
model/source/split rows, per-concept TP/FN/scored-FP, rate objects
`{numerator, denominator, rate}`,
unscored/vague phrases, errors and pending data. Concept sets are deduplicated
through the frozen alias map. Precision TP/FP uses only absence-eligible frames;
recall uses all annotated positives. Keep these denominators separate.

SAM score rows join segmentation records to references; metrics use only query
masks mapped into that source's scored vocabulary and its valid pixels. Preserve
the separate all-query union and all unscored-query counts. Compute aggregate
pixel TP/FN/FP, recall, precision and IoU with null empty denominators. Failed
required queries count as empty predictions for recall, with failures explicit;
partial masks from failed queries remain diagnostic and are not scored as success;
missing records block comparison rather than disappear. Export per-concept mask
coverage for diagnosing omitted concepts versus poor segmentation.

## SAM handoff and cost

Packages `traversability_hazard_segmentation` and
`traversability_hazard_benchmark` export:

```python
load_segmenter(settings: dict) -> segmenter
segmenter.segment_image(frame: dict, prompts: list[str], data_root) -> result
segmenter.close() -> None
freeze_selection(run_dir, *, data_root=None, policy_path=None,
                 test_frames_per_source=10, warmup_frames=5, seed=0)
run_conditions(config, *, segmenter=None) -> summary
profile_backend(config, *, backend=None, segmenter=None) -> profile
```

An injected backend/segmenter is allowed only in explicitly marked fixture mode.
Conditions: `vlm__<model_key>`, `reference_present`, `fixed_policy`.
Reference-present uses that frame's reference concept names; fixed-policy uses
all 16 common canonical phrases, independent of per-image labels. References
are allowed only in the diagnostic controller, never inside VLM or SAM adapters.
Selection is deterministic from development/test frames, frozen and hashed
before prediction. Record image/reference identity, exact IDs and coverage;
changing any frozen input requires a new run/profile.

Explicit `data_root`/`policy_path` arguments take precedence over environment
defaults. Freeze first after complete data preparation and before any prediction
or model metadata; existing selections are verified on unchanged reuse. The
selection seal hashes canonical selection JSON with both equal `fingerprint`
and `selection_hash` fields removed. Its `identity` binds complete dataset
metadata, ID-sorted frame/reference records, policy/alias recipes and RGB/reference
asset hashes. `selected_inputs` also binds each selected image/reference digest.
The evaluator verifies both the selection seal and current input identities.

Query records use these exact keys:

```json
{
  "task_id": "hazard_prompt_v1", "schema_version": 1,
  "frame_id": "rellis:00001:frame", "condition_key": "vlm__qwen3_vl_4b",
  "query_id": "vlm__qwen3_vl_4b:rellis:00001:frame:q000",
  "phrase": "person", "canonical_concept": "person",
  "mask_paths": [], "scores": [], "returned_instance_count": 0,
  "union_mask_path": "segmentation/CONDITION/masks/QUERY_union.png",
  "status": "ok", "error_code": null
}
```

Query IDs include condition, frame ID and zero-based `qNNN` phrase index. Mask
paths resolve against the run directory, rather than the data root; generated
asset filenames must be filesystem-safe. Scores align with instance-mask paths.
An unmapped concept is null. A failed query preserves partial masks, if any,
but has error status; missing files cannot masquerade as empty detections.

Frame records have task/schema, `frame_id`, `condition_key`, `query_ids`,
`requested_queries`, `completed_queries`, `failed_queries`, `sam_query_count`,
`upstream_status`, `union_mask_path`, `status`, `error_code`. Successful empty
lists produce an original-size empty mask and status ok. An upstream VLM failure
remains a frame error even with zero requested queries. Vague/unusable phrases
produce explicit failed query records rather than broad SAM matches. Copy query
outputs to CPU before the next query can mutate SAM's state.

`completed_queries` counts successful queries; `failed_queries` counts saved
failed queries. Both are terminal records for resume. The evaluator also supports
an explicitly configured attempted-count adapter for external saved producers.

Encode each image once within each measured condition. Reset text/geometric
prompt state between phrases while preserving that image encoding. Keep original
phrases: no GT-assisted rewriting. Use one pinned SAM3 image checkpoint, input
resolution, confidence threshold and mask threshold across conditions. Record
actual settings and code/checkpoint revisions; gate unapproved/missing weights
with clear messages. SAM2 is not a transparent text-prompt fallback.

Actual SAM settings use `checkpoint_repository` and `checkpoint_revision`, plus
`code_revision`, image resolution, confidence/mask thresholds and postprocessing.
Condition/profile metadata records `sam_settings` and adapter provenance; every
condition must use equal actual settings.

The backend result returns `queries`, `frame` and auditable settings. Query
results carry phrase, CPU mask arrays, scores and status/error; the frame result
carries frame ID, CPU union mask and status/error. The runner adds canonical
bookkeeping, condition/query identity, asset paths and publishes saved records.
Profile synchronized whole-image calls with five development warmups and two
test repeats. Report per-stage times, median/p95, prompt counts, failures,
load time, memory baselines/peaks/measurement limitations and fixture status.
Do not infer a combined p95 or joint VRAM peak by summing separate summaries.
Never clear other components' global CUDA caches to satisfy a budget.

`profile_backend` returns `output_dir`, `metadata`, `summary` and `samples`.
The summary fields below are under its `summary` key.

Profile summaries use task/schema, `fixture`, `execution_kind`, `component`
(`vlm`, `sam`, `combined`), `condition_key`, `complete`, `comparison_ready`,
`measured_frames`, `expected_measured_frames`, `unique_test_frames`,
`failed_calls`, `sam_query_count`, `latency:{median_s,p95_s}` and `memory`.
Memory fields retain allocated/reserved/device-used peak byte values and their
incremental counterparts, baselines and unavailable reasons. Metadata pins
selection/model/settings identity; samples identify frame/repeat, elapsed time,
status/errors, actual SAM calls and telemetry. No real-model comparison-ready
claim is permitted for synthetic fixtures.

Profiles seal full `identity` in `identity_sha256`; metadata seals samples and
summary in `samples_fingerprint`/`summary_fingerprint`. Measured
`(frame_id,repeat)` slots are globally unique. Warmups repeat after resume and
are unique within `session_id`; every session with measured calls first records
its fixed development warmups. Unattempted upstream-blocked SAM calls have null
elapsed time and `call_attempted:false`. Invalid synchronized timings have null
elapsed time and `timing_valid:false` with an explicit error. Catastrophic query
failures may have null actual SAM counts with retained failure diagnostics.
Latency quantiles use all valid measured attempts and linear interpolation;
unavailable calls remain in completeness/failure counters. Fixtures provide
software verification only and never establish accuracy or resource readiness.
