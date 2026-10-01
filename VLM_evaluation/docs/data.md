# Data and annotations

Run from `VLM_evaluation/`. Use an isolated Python environment and install
`requirements/data.txt` plus this component's editable package. Open
`notebooks/01_data_and_annotations.ipynb`.
Set `TRAVERSABILITY_DATA_ROOT` and `TRAVERSABILITY_RUN_ROOT`, or pass explicit
roots to `resolve_roots`; defaults are component-relative `data/` and `runs/`.
Keep datasets, masks, checkpoints and generated runs outside Git.

## Download/import

The notebook reads exact public URLs from `research/traversability/data_manifest.json`.
Downloads are disabled until a resource key is added to `DOWNLOAD_KEYS`; archives
already obtained from those links go in `IMPORT_ARCHIVES`. Transfers and extraction
use temporary files, reject unsafe paths, and preserve completed destinations.
Google Drive needs optional `gdown==6.4.1`; quota/access failures explain browser
import and the publisher's retry guidance. No bulk download runs on import.

`import_rellis` accepts an existing publisher tree or the safely extracted archives.
The current contract defaults to 20 development frames from `00000` and 20 test
frames from each of `00001`–`00004`: 100 public frames. Sampling is uniform over
naturally sorted eligible stems. Short/missing recordings remain pending; RELLIS
timestamps stay null unless verified. These are recordings on one campus.

`configs/data/legacy_phone_100.json` retains the earlier recipe: 10 development
RELLIS frames, 60 test RELLIS frames, and 10 phone frames at each A/B/C. Use a
separate run name and its `rellis_recipe`. Without phone clips, coverage is 70/100;
its `small_wheeled_v1` references remain optional manual annotations.

`import_tum` reads only RGB from `rgb.txt` for `freiburg1_floor` and
`freiburg3_walking_xyz`, retaining recorded timestamps in supplementary runs.
`import_phone_video` needs optional `av>=16,<20`; import two five-keyframe clips
per location, with A development and B/C test. Decoded RGB stays unrotated;
presentation timestamps and display rotation are recorded separately. Phone clips
remain pending and do not establish hazard accuracy until actually recorded.
After import, set notebook `MASK_TARGET="phone"` to generate/import aligned masks,
freeze their subset and use the optional viewer for that separate phone run.

## Generate/import masks

`import_sam_masks(run_dir, data_root, records)` accepts producer rows with
`region_id`, `frame_id`, `mask_path` (or `source_mask_path`) and optional
`planning_relevant`, `image_sha256` and `generator`. Import paths may be absolute;
published paths are always relative. Masks must be nonempty single-channel binary
PNGs matching the original RGB dimensions. Semantic/reference origins and known
semantic-aid paths are rejected. Every original planning mask is retained.

Alternatively, explicitly call `load_sam2_generator` with a local checkpoint, then
`generate_sam2_regions`. Follow the [official SAM2 installation instructions](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md):
Python ≥3.10, torch ≥2.5.1, torchvision ≥0.20.1; Windows guidance recommends Ubuntu
under WSL. `SAM2_BUILD_CUDA=0` omits the optional extension. Official instructions
were checked on 2026-10-01 at revision `2b90b9f5ceec907a1c18123530e92e794ad901a4`.
The configurable starter pair is `sam2.1_hiera_tiny.pt` and
`configs/sam2.1/sam2.1_hiera_t.yaml`. The notebook can explicitly download that
single [official checkpoint](https://github.com/facebookresearch/sam2/blob/main/checkpoints/download_ckpts.sh).
Loading never downloads weights. Runtime versions, actual source revision when
available, settings, hashes and preprocessing are recorded; no SAM2 accuracy or
GPU resource result is claimed by the CPU tests.

`freeze_selection(..., seed=0)` deduplicates selection candidates by identical
binary geometry, retaining the lowest stable ID. Sort by area then ID, divide
into contiguous thirds (remainder goes to earlier thirds), and use seeded SHA-256
ranking to pick 2 small, 1 medium and 2 large masks. Fill shortages from remaining
candidates; use all if at most five unique masks exist. No human confirmation is
required. All planning masks remain available for profiling; classification uses
only frozen selected IDs.

## Annotate and resume

For the default run, `derive_references` implements
[automatic references](automatic_references.md). It validates original RELLIS IDs
and requires 95% non-void coverage and 95% single-class dominance. Ground-dominated
masks with any annotated avoided pixels stay mixed/pending; grass, void and missing
aids remain null/pending. These are semantic-material-policy proxies, conditional
on geometry. Semantic IDs, labels and fractions never enter inference records.

Human annotation is optional: `annotation_viewer` shows original RGB, overlay and
mask with label and quality controls. Optional `ipywidgets>=8,<9` enables widgets;
`annotation_viewer(..., use_widgets=False)` returns a session with
`pending_region_ids`, `show()` and `save()`. Direct `show_region` and
`save_annotation` also work. `unknown` is explicit; missing labels stay pending.

Rerun with the same roots, run name and settings to resume. Completed annotations
are immutable, preparation fills only missing pending rows, and locked atomic
saves preserve prior files on failure. Changed policy, keyframe sample, RGB,
frozen masks or selection require a new run. Newly available recordings/clips can
be appended without changing existing frozen frames.

## Export and CPU check

Runs contain the fixed `frames.jsonl`, `regions.jsonl`, `annotations.jsonl`.
`metadata/data.json`, `masks.json`, `selection.json`, optional `sam2.json` and
`reference_transfer.jsonl` hold coverage/provenance. Join by IDs. Inference reads
RGB and SAM regions; reference provenance is for evaluation only.
`coverage_summary` reports actual frames, masks and completed/valid selected
references. `export_run` validates the freeze and creates an immutable snapshot;
portable RGB/mask paths still resolve against the original configured data root.
Use a new export destination when annotations change.

`create_cpu_fixture` demonstrates import, mask validation, stratified freezing,
annotation persistence and resume with synthetic data, excluded from real coverage.
With CPU dependencies and pytest installed, run `python -m pytest tests/data -q`.
No datasets, checkpoints or GPU are needed.
