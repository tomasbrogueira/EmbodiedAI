# Image-level hazard references

Use an isolated Python 3.12+ environment from `VLM_evaluation/`:

```console
python -m pip install -r requirements/hazard_data.txt
python -m pip install --no-deps -e .
python -m unittest discover -s tests/hazard_data -v
```

Open `notebooks/10_hazard_data.ipynb`. Its default action prepares and validates
11 explicitly synthetic images (five RELLIS-schema frames and six COCO-schema
images) in a separate fixture namespace. These are CPU checks, not the 140-image
experiment. No downloads occur by default. Package imports load no model libraries.

For direct use, `prepare_run(config, *, fixture=False) -> Path` accepts a mapping
or JSON configuration path; `validate_run(run_dir, data_root) -> dict` returns
identity, fixture status, frame/reference counts, coverage and the fingerprint.
Invalid artifacts raise `ValueError` or `FileNotFoundError` with a diagnostic.

Integration also supports `fixture_profile="protocol"` with `fixture=True`:
100 synthetic RELLIS-schema images and 40 synthetic COCO-schema images, enough
for the fixed 20-test/5-warmup SAM replay. The default fixture remains 11 images.
Use [the offline workflow](hazard_integration.md) to connect the actual producers;
synthetic quota coverage is never real accuracy or GPU-memory evidence.

New runs publish `selected_frame_ids` and `observed_input_signature` before
predictions. The signature hashes sorted frames/references and their RGB/valid/
concept-mask asset hashes; original annotations remain additionally protected by
the dataset fingerprint. Unchanged older runs keep their published identities.
After all intended real inputs are complete, notebook 10 freezes the SAM test and
development selection before notebook 11. Missing sources stay fillable until
downstream artifacts exist.

```python
import json
from pathlib import Path
from traversability_hazard_data import prepare_run, validate_run

config = json.loads(Path("configs/hazards/data.json").read_text(encoding="utf-8"))
config.update(data_root="/external/hazard-data", run_root="/external/hazard-runs")
run_dir = prepare_run(config)  # Missing real inputs remain pending.
print(validate_run(run_dir, config["data_root"])["coverage"])
```

Explicit roots override `TRAVERSABILITY_REPO_ROOT`, `TRAVERSABILITY_DATA_ROOT`,
`TRAVERSABILITY_RUN_ROOT` and `TRAVERSABILITY_CACHE_ROOT`; remaining defaults are
the component and its `data/`, `runs/`, `.cache/` directories. Relative configured
roots and `policy_path` resolve against the component; COCO annotation/image paths
resolve against the data root. The notebook finds the component from the checkout,
component or notebook directory on Windows/Linux; `REPO_OVERRIDE` can identify
either the checkout or component. Choose new run names when completed inputs change.

## Sources and selection

RELLIS uses existing original RGB and semantic-ID files, including legacy imports.
Set `rellis.source_dir` for an extracted publisher tree, or leave it null for
`raw/rellis_rgb`, `raw/rellis_semantic` and existing imported `rellis` files.
Require stem-aligned RGB/ID pairs with equal original dimensions. Each sequence
`00000`–`00004` needs at least 20 annotated pairs: naturally sort stems, then select
20 uniformly spaced endpoint-inclusive frames. `00000` is development; the other
four sequences are test. Short/missing sequences remain pending. Stable IDs use
source/sequence/stem, and unverifiable timestamps are null.

Use the [official RELLIS documentation](https://github.com/unmannedlab/RELLIS-3D),
[RGB archive](https://drive.google.com/file/d/1F3Leu0H_m6aPVpZITragfreO_SGtL2yV/view)
and [original semantic-ID archive](https://drive.google.com/file/d/16URBUQn_VOGvUqfms-0I8HHKMtjPHsu5/view).
Obtain/extract them separately if needed; this module performs no RELLIS downloads.
Check sequence coverage, RGB/ID alignment, raster dimensions and IDs against the
[frozen policy](../configs/hazards/policy.json). Do not use LiDAR learning labels,
color masks, raw recordings or majority voting.

COCO reads `instances_val2017.json`, resolving person/cup/bottle/chair/dog IDs by
category name. Selection uses decoded annotation masks, never model predictions or
bounding boxes. Development selects six target-positive/two target-negative images;
test selects 26/six from the remaining candidates. Seed 0 is fixed. For each split,
first cover one stratum per class: a small instance if available in the remaining
pool, otherwise any positive. Then maximize class exposure scores
`sum(1 / (1 + selected_count))`, followed by
the same score across small-target classes. A small instance has decoded area
below 1,024 pixels. Break ties by ascending SHA-256 of
`hazard_prompt_v1:coco:0:<image_id>`, then numeric image ID. Negative controls use
that ordering. Missing classes, small-target strata and quotas are reported.
Freeze exact IDs before requiring RGB availability; never replace a frozen image
because its RGB is missing. Keep all scored instances in each selected image,
unioning same-category polygons/RLE, including crowds, through the
[official COCO mask API](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py).

The notebook's `DOWNLOAD_COCO = True` is an explicit real-data action: download the
annotation archive, call real preparation to freeze IDs, fetch only those images
using their official `coco_url`, then prepare again. The optional helpers are
`download_coco_annotations(config, allow_download=True)` and
`download_selected_coco_images(run_dir, data_root, cache_root, allow_download=True)`.
See [official COCO downloads](https://cocodataset.org/#download). Setting
`RUN_REAL_PREPARATION = True` alone processes available local files without network
access. No helper runs during imports or ordinary preparation.

## Reference meaning and artifacts

These are whole-image annotated object/material references under the common
policy, not physical traversability or unrestricted hazard ground truth.
RELLIS presence is at least one annotated pixel; water unions IDs 6/31. Ignore
IDs 0/3/9 in pixel metrics; sky ID7 is valid background, never a prompt. Reject
unknown IDs. Score absence only when ignored pixels occupy at most 5%; positive
concepts remain usable above that threshold. Retain tiny positives and record
their counts, image-area fractions and the strict marker `0 < fraction < 0.001`.
COCO keeps every target with positive decoded mask area, scores only its five
categories, and makes every pixel evaluable and absence eligible; a target-negative
image can contain other hazards. Infer no floor/contact
predicate. Cables, independent spill/floor-contact cases, phone clips and physical
safety remain pending optional inputs. See the [protocol](hazard_protocol.md).

Each run publishes exactly:

```text
<run_root>/<run_name>/frames.jsonl
<run_root>/<run_name>/references.jsonl
<run_root>/<run_name>/metadata/dataset.json
<data_root>/rellis/<sequence>/{rgb,semantic}/<original filename>
<data_root>/coco/val2017/<original filename>
<data_root>/coco/annotations/instances_val2017_<annotation SHA-256>.json
<data_root>/hazard_references/<run_name>/<SHA-256 of frame_id>/valid.png
<data_root>/hazard_references/<run_name>/<SHA-256 of frame_id>/<concept>.png
```

Fixture source files use `fixtures/hazard_prompt_v1/sources/` beneath their data
root; fixture RGB/annotation imports prefix the source layouts above with
`fixtures/hazard_prompt_v1/imports/`. The notebook additionally places that data
root beneath `DATA_ROOT/fixtures/hazard_prompt_v1` and uses the separate run name
`hazard_prompt_v1_fixture`.

RGB/reference asset paths in records are portable POSIX-relative paths under the
data root. Masks are single-channel binary `{0,255}` PNGs at original dimensions;
only present concepts need concept masks. Reference records follow the
[frozen interface](hazard_interfaces.md); area/tiny/coverage diagnostics live in
metadata. Original RGB bytes are hashed without resizing/re-encoding. Metadata
records task/schema, `fixture`, original annotation hashes/provenance, exact IDs,
selection parameters, source/split coverage, missing strata and
`dataset_fingerprint`. `annotations[frame_id]` records source kind, relative path,
original-byte `sha256` and provenance; COCO entries also record numeric `image_id`.
`selection.rellis` freezes stems/frame IDs and original RGB/ID hashes per sequence.
`selection.coco` freezes numeric image IDs, splits, category/small-target coverage
and the annotation hash; `source_images.coco` retains each selected original
image-table entry, keyed by its numeric ID as a string. RELLIS frame IDs are
`rellis:<sequence>:<stem>`; COCO IDs are `coco:val2017:<12-digit image_id>`.
`assets` maps every published RGB, annotation and reference path to its byte hash.
`policy_sha256` hashes original policy bytes;
`alias_sha256` hashes sorted compact UTF-8 JSON of the alias table. The fingerprint
excludes absolute roots and volatile timestamps.

Preparation validates joins, dimensions, binary values, categories, counts,
hashes, path escapes and split separation. Identical repeated preparation preserves
completed outputs. Missing frozen assets and previously absent strata may be filled
before downstream artifacts exist. Predictions, evaluation, segmentation,
benchmark or model metadata freeze additions. Changed completed RGB, annotations,
policy, references or selection require a new run; legacy runs are refused and
legacy outputs preserved. Missing inputs produce actionable pending/partial
coverage without fabricated samples. References must never enter the VLM prompt.
