# Saved hazard evaluation

`traversability_hazard_evaluation` evaluates whole-image noun phrases and optional
fixed-SAM artifacts on CPU. It reads JSON/JSONL directly and imports no inference,
data, segmentation or model package. Legacy region outputs are rejected.

From `VLM_evaluation/`, use an existing isolated Python 3.12+ environment with
`requirements/hazard_evaluation.txt`, then:

```powershell
$env:PYTHONPATH = "$PWD/src"
python -m traversability_hazard_evaluation --config configs/hazards/evaluation.json
python -m traversability_hazard_evaluation --config configs/hazards/evaluation.json --output-dir runs/hazard_prompt_v1/evaluation
python -m unittest discover -s tests/hazard_evaluation -p "test_*.py" -v
python -m pytest tests/hazard_evaluation --import-mode=importlib -q
```

The first command writes nothing. Exit 2 indicates validation errors or incomplete
discovery. No dataset/model download, GPU action or server connection is involved.
`12_hazard_evaluation.ipynb` exposes the same configuration and disables export by
default; its saved outputs and execution counts are cleared.

```python
from traversability_hazard_evaluation import evaluate, export_report
report = evaluate(config)                  # dict in, JSON-serializable dict out
paths = export_report(report, output_dir)   # filename -> absolute path string
```

Explicit `data_root` and `run_root` override `TRAVERSABILITY_DATA_ROOT` and
`TRAVERSABILITY_RUN_ROOT`, then component-relative `data/` and `runs/` defaults.
`component_root` or `TRAVERSABILITY_REPO_ROOT` can identify the component or parent
checkout. Artifact paths must be safe POSIX-relative paths, including through
symlinks: reference masks resolve under the data root; SAM masks under the run.
Export only replaces the evaluator's named outputs in the supplied directory.

## Counts and completeness

Rows in `report["concepts"]["rows"]` separate model, source and split, with `all`
source aggregates within each split. Use test rows for final comparison. Recall
counts every completed annotated positive, including absence-ineligible images.
Precision has its own `precision_tp` and `scored_fp`, restricted to absence-eligible
images and source-scored concepts. All rates retain numerator/denominator; an
empty denominator has a null rate. Tiny concepts have image area fraction
**0 < area < 0.001**. One annotated pixel is still a positive; exact threshold
objects remain positives outside the tiny stratum.

Exact frozen aliases use case folding and collapsed whitespace. Water and puddle
merge once per image. Original raw-response phrases, saved prompts, alias merges,
exact normalized repeats and unscored/vague phrases remain auditable. Unknown
nouns and out-of-source concepts are unscored, not confirmed hallucinations;
broad phrases never match hazards. No instance counts or VLM judge are used.

Saved errors and valid empty lists both miss positives; only errors count as
response failures. `failure_rate` is saved errors / valid saved response records;
record coverage separately uses every requested frame. Missing/duplicate/invalid
predictions block completeness and receive zero matches in flagged diagnostic
recall. Partial precision is provisional, **not** a conservative lower bound.
Pending/missing references never become negative labels. Invalid unjoined truth
is reported separately rather than silently treated as an empty reference.

`comparison` separates discovery, SAM and deployment completeness. Fixtures never
produce real comparison readiness. Resource-informed selection readiness requires
compatible saved VLM/combined measurements for each model. `winner` is always null:
the evaluator publishes evidence, without choosing a model.

## Metadata bindings

The generic reader supports explicit external metadata layouts. Defaults expect
root `task_id`, `schema_version`, `fixture`, `policy_hash`, `alias_hash` and
`dataset_fingerprint`. Dataset metadata also needs `selected_frame_ids`,
`coverage.expected_frames`, `source_annotations` by source and `sampling`. Model
metadata needs `model_key` and `requested_frame_ids`. Every requested frame must
have a saved prediction, even when its reference is pending.

The active checked-in configuration binds the actual producer layouts explicitly:

| Artifact | Active field bindings |
|---|---|
| Data | `policy_sha256`, `alias_sha256`, `selected_frame_ids`, `annotations`, `coverage.expected_frames_by_source_split` |
| Inference | `configuration` hashes/model revision/settings; `inputs.dataset` fingerprint/fixture; `inputs.frames` requested IDs; `execution.kind` |
| Selection/SAM/profile | `policy_sha256`, root alias/fingerprint identity, actual SAM `checkpoint_repository` / `checkpoint_revision` |

The evaluator verifies complete frozen inference input rows/RGB bytes, committed
attempt records and inference's ASCII-escaped transaction fingerprints. It reads
successful EOS-bearing raw responses only through exact verified generation and
parse audits. Saved original phrases must match original raw phrases after exact
normalized repeat removal; scoring aliases never authorize prompt rewriting.

Adapt layouts explicitly with JSON pointers, for example:

```json
{"metadata_bindings": {
  "model": {"policy_hash": "/identity/policy_sha256", "requested_frame_ids": "/inputs/frame_ids"},
  "dataset": {"selected_frame_ids": "/selection/frame_ids"}
}}
```

Supported binding kinds are `dataset`, `model`, `selection`,
`segmentation_metadata`, `deployment_metadata` and `deployment_samples`. Missing
required identity fields are errors. Canonical hashes use UTF-8, sorted keys,
compact JSON separators and `ensure_ascii=False`; aliases hash `policy.aliases`.
`hash_recipes.policy="file_bytes"` adapts producers hashing policy file bytes.
The frozen policy contents themselves remain pinned independently.
Declared `run_name` and `policy_id` are checked when present; explicit bindings
make those optional fields required. `sam_settings` bindings adapt checkpoint and
revision pins inside the saved SAM settings object.

`dataset_fingerprint_recipe="declared"` checks equality across stages but cannot
prove the producer's unspecified fingerprint recipe. A real final comparison
additionally requires either `expected_input_signature` pinned from the frozen
run, or `dataset_fingerprint_recipe="frames_references_assets_v1"`. That recipe
hashes canonical JSON of `{frames, references, assets}`: ID-sorted records and a
sorted relative-path -> file-SHA256 map covering RGB, concept and valid masks.
The current value is exported as `identity.observed_input_signature`; freeze it
before predictions, not retrospectively after modifying references. Integration
must align the producer recipe or supply the explicit pin.

The active `hazard_data_v1` recipe independently recomputes the producer's
canonical `{metadata,frames,references}` fingerprint (excluding only the
`dataset_fingerprint` field), validates all declared original/reference asset
hashes and annotation provenance, then verifies the prepared data metadata's
`observed_input_signature`. This pin is created by notebook 10 before notebook 11
inference. Existing unchanged artifacts without a pin retain declared identity
limitations until a separately pre-inference signature was explicitly pinned.

## Optional SAM and measured costs

Enable `segmentation.enabled` and select condition strings (`vlm__MODEL`,
`reference_present`, `fixed_policy`), or objects with `condition_key` and optional
relative `metadata_path`, `frames_path`, `queries_path`. Selection defaults to
`segmentation/selection.json`, with root `test_frame_ids`, `warmup_frame_ids`,
`selection_hash` and common identity fields. Its default coverage is ten test
frames per source and five development warmups; fixture configs override counts.
Condition metadata needs matching `selection_hash` and identical nonempty
`sam_settings`. `completed_queries_semantics` defaults to `successful`; use
`attempted` when a producer includes saved failed queries in that counter.
`selection_hash_recipe="canonical_selection"` verifies the canonical selection
object with its bound hash field removed. Include the frozen
`observed_input_signature` (adaptable as binding `selection.input_signature`) to
verify selection input identity. The default `declared` recipe validates saved
hash equality but leaves final SAM comparison readiness unverified.

The active `canonical_selection_dual` recipe removes both equal `fingerprint`
and `selection_hash` fields before hashing, and independently checks the full
selection `identity` and `selected_inputs` against current data. Notebook 10 must
freeze the complete selection after preparation and before predictions; explicit
data/policy roots remain authoritative.

Pixel scoring unions only successful queries mapped into the source vocabulary,
then intersects prediction and reference unions with the valid-pixel mask.
Required failed queries act as empty predictions; partial failed masks are never
scored. Missing records block completeness. An unavailable reference makes the
affected aggregate metrics unavailable instead of shrinking its denominator.
The all-query frame union is validated and its path preserved separately. Its
membership for failed partial queries is not inferred. Successful empty lists
require original-size empty masks. Per-concept exports separate omitted prompts,
query failures and achieved pixel coverage. Reference-present is a diagnostic;
fixed-policy is the no-VLM baseline; VLM/SAM rows are combined results.

`deployment.profiles` accepts relative summary paths or objects with
`summary_path`, optional `metadata_path` and `samples_path`. Profile metadata pins
stage/condition, execution kind, selection and actual model/SAM settings; bindings
adapt nesting. Preserve warm latency, loading/cold statistics, memory attribution,
query counts and unavailable reasons independently. Combined values require a
saved actual combined profile. Separate p95 values or GPU peaks are never summed.
Synthetic measurements do not establish accuracy, memory fit or model preference.
Profile pins default to `model_key`, `checkpoint_revision`, `settings` and
`sam_settings` (with `checkpoint` and `revision`). Explicit sample checking uses
`phase` (`warmup`/`measured`), `frame_id`, zero-based `repeat` (null for warmup),
`elapsed_s`, `status` and `sam_query_count`; `deployment_samples` bindings adapt
those names. Missing/duplicate frame-repeat calls block completeness. Resource
selection requires measured latency and at least one reported total GPU peak;
unavailable telemetry remains visible without supplying selection evidence.

Native profiles automatically validate their saved samples alongside journal and
summary seals. Repeated development warmups are grouped by `session_id`; measured
frame/repeat slots remain unique across resume. Null blocked/failed timings and
unknown actual query counts require explicit failure/measurement diagnostics and
retain completeness/failure counters. The evaluator recomputes quantiles from
valid measured samples, checks summary counters and preserves loading/phase memory
without summing separate processes. A complete fixture profile is never measured
model latency or memory evidence.

Exports include `hazard_report.json`, coverage JSON, concept/per-concept/missed
CSV tables, phrase/error JSONL, SAM tables/frame audits and stage-cost CSV. No
visualization, temporal, geometry or physical-safety logic is implemented.
