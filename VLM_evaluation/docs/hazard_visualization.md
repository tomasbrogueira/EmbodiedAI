# Saved hazard visualization

Notebook [14](../notebooks/14_hazard_visualization.ipynb) views existing
`<run_root>/<run_name>/evaluation/hazard_report.json` on CPU with Matplotlib and
ipywidgets. It runs no evaluation, producer, model, download or server. The
frozen `hazard_prompt_v1` schema 1 and source vocabularies remain authoritative;
legacy region reports are rejected. Missing inputs produce informative empty
states, without implicitly loading a fixture.

Install `requirements/hazard_visualization.txt` only into the chosen isolated
CPU environment, or select `--with-module hazard_visualization` in
`scripts/setup_env.py`. Notebook support remains the separate `--notebooks`
option. No GPU dependency or pin is changed.

## Configuration and views

The registered `hazard-visualization-config` cell reads
`configs/hazards/visualization.json`. Set `run_name` and optional explicit
`data_root` / `run_root`. Otherwise `TRAVERSABILITY_DATA_ROOT` and
`TRAVERSABILITY_RUN_ROOT` apply, followed by component-relative `data/` and
`runs/`. `component_root` or `TRAVERSABILITY_REPO_ROOT` accepts the component or
its parent checkout. Relative roots resolve against the component on Windows
and Linux; historical absolute paths in the saved report never redirect reads.

Four tabs provide:

- Discovery: model/source/split selectors, test by default, saved recall,
  supported precision, tiny recall and response failure rates with their own
  numerators/denominators, record/reference coverage, and per-concept recall.
  Selecting `all` uses only the saved aggregate rows. Rates are never averaged
  and source rows are never added to aggregates. Development stays separate.
- Image explorer: exact source/split/frame/model joins, available SAM conditions
  and queries, aggregate or per-concept references, opacity, RGB/reference/scored
  overlays with original coordinates and stable legend colors. Saved original
  phrases, canonical/source-scored/unscored categories, raw responses, response
  errors, missed annotated concepts and SAM audits are shown. Expand the details
  to see full long responses and all diagnostics. Missing/invalid records and
  response failures differ from successful empty lists. Unknown/out-of-source
  phrases remain unscored, without a hallucination judgment.
- SAM handoff: direct pixel recall/precision/IoU and frame/query failure rates,
  saved/expected/reference frame counts, and per-concept prompt coverage,
  successful-query coverage and pixel recall. VLM conditions are combined
  VLM+SAM results; `fixed_policy` is the no-VLM baseline; `reference_present`
  is a diagnostic control. With all conditions selected, the compact coverage
  plot uses the explicitly labeled first saved condition; select another
  condition to inspect its coverage. Prompt/query coverage remains visible when
  pixel metrics are unavailable.
- Deployment: explicitly select `(component, condition_key, profile_id)` tuples;
  nothing selects a newest profile. Native profile IDs come from the saved
  summary path. Isolated VLM, isolated SAM and actual combined profiles stay
  separate, retaining validity/completeness/readiness, median/p95, timed and
  protocol sample counts, failures/query counts and resource provenance.
  Allocator allocated/reserved and sampled shared-device usage are distinct.
  Baselines, increments, unavailable reasons, phase memory, loading, budget
  provenance and full original profile rows remain in details and export
  sidecars. Invalid profiles with missing component identity show `unavailable`.
  Missing native profiles use the validated summary path's planned condition,
  prefixed `expected:`, to distinguish selector entries that share a profile ID.
  This preserves every pending row without supplying measured component,
  condition, validity or resource evidence; observed row fields stay unchanged.
  No isolated stage p95/peak values are added.

Report `concepts.rows` has nested `metrics`; `concepts.per_concept` and SAM rows
have direct rate objects. The report already embeds the named evaluator exports;
duplicate CSV/JSONL exports are unnecessary. Frames/references and any native
prediction JSONL present are indexed once. RGB and reference masks are read
lazily under the data root; SAM instance/query/frame masks under the selected
run. Duplicate identities, path escapes (including symlinks), mismatched joins,
wrong vocabularies and wrong dimensions are rejected. Missing frame/reference
files leave report tables usable. Missing optional SAM/profiles explain which
saved stage is needed and never start it.

## Scored and diagnostic masks

There is no saved scored-union PNG. `full_union_path` is the all-query diagnostic.
`reconstruct_overlay` unions only successful, saved, source-scored queries that
the evaluator did not mark invalid/missing/failed, validates their original-size
instance/union masks, and intersects with the annotation valid mask. It compares
reconstructed TP/FN/FP with the saved frame counts; discrepancies make the scored
overlay unavailable. A failed query does not remove another valid successful
query. Evaluator-invalid frames and unavailable references cannot supply a
scored overlay. No report metric is recomputed or replaced.

All-query, selected-query and failed-partial masks require explicit diagnostic
selection, with a different legend/title. They are never scored. Ignored pixels
are gray. Source vocabulary governs reference unions/per-concept masks. Masks
are never resized. Null rates/memory stay unavailable; provisional counts,
validation errors, completeness and comparison readiness remain visible.
Every fixture figure carries a conspicuous synthetic-data label. Scripted CPU
fixture latency is not hardware evidence. The viewer makes no model-winner,
joint GPU fit, safety, floor-contact, temporal stability or path-quality claim.

## Public headless API and exports

```python
from traversability_hazard_visualization import (
    load_run, reconstruct_overlay, create_viewer, render_discovery,
    render_image, render_segmentation, render_deployment, export_figure,
)
saved = load_run(config)                    # read/index once; SavedRun
saved.status()                             # fixture, errors, coverage, readiness
figure = render_discovery(saved, source="coco", split="test")
image = render_image(saved, frame_id, model="qwen3_vl_4b",
                     condition="vlm__qwen3_vl_4b", opacity=0.45)
handoff = render_segmentation(saved, source="rellis", condition="fixed_policy")
cost = render_deployment(saved, [("combined", "vlm__qwen3_vl_4b", "my_profile")])
widgets = create_viewer(saved)              # notebook-only; imports ipywidgets lazily
```

`SHOW_WIDGETS=False` uses the notebook's non-widget render path; `FRAME_ID`,
`MODEL`, `CONDITION` and `PROFILE_IDS` supply explicit saved selections. The public
render helpers return ordinary Matplotlib figures and work with `MPLBACKEND=Agg`.
If ipywidgets is missing, the notebook reports that dependency and displays a
headless discovery figure. A notebook kernel/IPython still needs notebook support.

Exports default off. Set `config['export_enabled']=True` and invoke
`export_figure(saved, figure, 'view.png', enabled=True)` (or `.svg`). In the
notebook, additionally set `EXPORT=True` for the static figure collection or
press the explicit per-tab export button. Only safe basenames are accepted,
only the configured `output_dir` is used, and existing files are never overwritten.
The default is component-relative `visualization_outputs/`; input/source
directory overlap is rejected. No UI selector writes artifacts. Each export
includes a `.png.json` / `.svg.json` evidence sidecar with run/source/split,
counts, selected profile IDs/rows, fixture/provisional status, readiness and full
image diagnostics. Figures include the same identities/status in captions.

## Verification performed

The existing isolated CPU environment was used: Python 3.12.13, NumPy 1.26.4,
Pillow 12.3.0, Matplotlib 3.11.2, nbformat 5.11.1, nbclient 0.11.0,
ipykernel 7.4.0. Only missing visualization packages were added: ipywidgets
8.1.9, jupyterlab-widgets 3.0.17 and widgetsnbextension 4.0.16.
The headless notebook and ipywidgets dependencies were available after that
installation. JupyterLab itself is absent in this environment; it was not
installed because this task does not launch a frontend or server.

Focused CPU tests cover source-scored/all-query separation, retained success
beside failed partial queries, ignored pixels, evaluator exclusions, mismatched
counts/instance unions, missing assets, dimensions, path containment, exact
joins, duplicate identities, null/provisional/fixture behavior, explicit profile
selection, opt-in non-overwriting exports, relocated roots and widget callbacks.
Headless source notebook executions cover no-report and supplied saved fixtures,
with a model/producer import guard and input-byte snapshots.
The final focused visualization/setup/notebook plus evaluator regression run
passed **144 tests and 104 subtests**, with one native symlink check skipped.
The selected environment's 52 packages passed `uv pip check`; `git diff --check`
also passed. The full repository/GPU suite was not run for this viewer change.

Safe Python-mode copies of notebooks 00/10/11/12/13/14 passed twice, once with
missing report/data and once with explicitly supplied existing full fixture
roots for 14. `--with-fixture` was not used: no experiments were regenerated.
For portable reproduction with your own saved roots:

```console
python scripts/check_notebooks.py --execution python --output-root /new/checks
python scripts/check_notebooks.py --execution python --output-root /new/saved-checks \
  --visualization-run-root /saved/runs --visualization-data-root /saved/data \
  --visualization-run-name hazard_prompt_v1
python -B -m pytest --import-mode=importlib tests/hazard_visualization tests/setup -q
```

The existing integrated replay's 80 SAM frame audits all reconstructed with
matching saved pixel counts. Hashes of all 3,442 fixture files remained unchanged.
Saved fixture notebook execution/rendering imported no Torch/Transformers/SAM or
producer packages. Charts and overlays were rendered and inspected, including
response failure, successful empty output, partial query failure, unscored terms,
ignored pixels, diagnostic masks, missing report and explicit profile states.
Widget construction/callbacks were checked headlessly; an interactive browser
frontend and a registered Jupyter kernel were not launched. Native symlink
creation is unavailable under this Windows account; that check is skipped.
Real-data correctness and real hardware measurements remain pending saved inputs.
