# Classification evaluation

`traversability_evaluation` reads the fixed JSONL artifacts directly, joins by IDs
and imports no data, inference or benchmark module. It evaluates the frozen
`rellis_material_v1` semantic-material policy, conditional on geometry; it does
not establish physical traversability. [Automatic references](automatic_references.md)
use independent RELLIS image annotations; unresolved grass/void stay pending.

Primary regions are selected, completely annotated and annotation-valid, with
readable nonempty single-channel PNG masks aligned to RGB. Pending and missing
annotations are excluded and counted. Completed `unknown` references are included.
Mixed/broken/unchecked masks and physical defects are reported separately.

| Metric | Numerator | Denominator |
|---|---|---|
| Unsafe acceptance | Reference non-traversable predicted traversable | Reference non-traversable |
| Useful acceptance | Reference traversable predicted traversable | Reference traversable |
| Predicted unknown rate | Effective unknown predictions | All primary regions |
| Reference-unknown acceptance | Reference unknown predicted traversable | Reference unknown |

Every rate stores `{numerator, denominator, rate}`; zero denominators give `null`.
Prediction errors count as unknown in these denominators and as separate failures.
Missing primary predictions block all splits of that configuration's scores and
confusion matrices and expose missing IDs, so omission cannot improve scores. Invalid identity,
duplicates, or incompatible manifests also block comparison. Confusion rows are
references and columns predictions, ordered traversable/non-traversable/unknown.
The separate **annotated-hazard-overlap acceptance** diagnostic is traversable
predictions among selected masks containing annotated avoided pixels divided by
all such masks, including mixed masks; it requires the reference-transfer sidecar.
The default pixel-count field is `avoided_pixels`, adaptable through
`avoided_pixel_count_field`. Null counts make this diagnostic unavailable; they
do not invalidate otherwise eligible primary references.

From `VLM_evaluation/` (Python 3.11+), install only CPU dependencies in an isolated
environment, set the roots/run/model keys in the config, then run:

```powershell
python -m pip install -r requirements/evaluation.txt
$env:PYTHONPATH = "$PWD/src"
python -m traversability_evaluation --config configs/evaluation/default.json --export
python -m unittest discover -s tests/evaluation -p "test_*.py"
```

The thin, output-free `03_classification_evaluation.ipynb` exposes roots, run/model
keys and optional metadata expectations before validation. `evaluate(config)`
returns a report even for incomplete artifacts. `export_report(report, output_dir=None,
plots=False)` writes compact JSON/CSV plus an HTML review gallery only beneath the
destination run's `evaluation/`; plots require optional matplotlib. Generated
results remain outside Git. `render_review_gallery(report, limit=20)` returns HTML.

To compare configurations of one model, use `comparisons` entries with `run_name`,
`model_key` and `configuration_label`. Optional `expected_metadata` maps JSON
pointers to expected values; `expected_metadata_sha256` pins canonical metadata.
`metadata_fields` adapts model/policy JSON pointers for alternate metadata layouts.
Built-in inference fingerprints and frozen input identity are verified; alternate
layouts require explicit checks or a pinned hash. The report retains fingerprints.

Coverage separates planned/public manifest/readable frames, expected annotation
records from selected IDs, pending/excluded annotations and eligible masks. The
default public target is 20 development frames from recording 00000 and 20 test
frames each from 00001–00004 (100 total); six unrecorded
phone hazard clips remain optional/pending. Public results are provisional for
that hazard coverage. Development rows are tuning evidence. RELLIS test recordings
share one campus; neighboring regions are not independent scenes. Fixture reports
are marked synthetic and never real experiment coverage.

Error exports group recognition, target grounding, policy and segmentation; suggested
causes remain tentative until explicit review. Optional audit labels can be read
from `review_labels_path` and are never overwritten by export. Each review JSONL
record has `{case_id, category, note}`; a nonempty note records evidence for a
confirmed category. A gallery prioritizes
unsafe acceptances, mask defects and prediction failures, capped at 20 cases; no bulk
manual labeling or another VLM judge is required. This module has no temporal,
smoothing, path or voxel metric. Serialization and image APIs follow the official
[Python JSON](https://docs.python.org/3/library/json.html) and
[Pillow image](https://pillow.readthedocs.io/en/stable/reference/Image.html) documentation.
