# Connected hazard workflow

The active task/schema/run identity is `hazard_prompt_v1`, version 1. Read the
[protocol](hazard_protocol.md), [contracts](hazard_interfaces.md) and
[setup](setup.md) before real runs. The producer path is data → whole-image
inference → discovery evaluation → SAM controls/cost → evaluation again.

Prepare every intended RELLIS/COCO input before predictions. New data metadata
pins the evaluable input signature, selected IDs, file-byte policy hash and
canonical alias hash. Freeze the deterministic 20-test/5-development SAM
selection with `freeze_selection(run_dir, data_root=...)` before model metadata
or predictions. Notebook 10 does this after complete real-data preparation.
Incomplete data and unchanged older data artifacts are preserved; completing
or changing inputs after downstream publication requires a new run.

Notebook 11 uses the same frozen common policy for both Qwen 4B models. It opens
RGB/frames/dataset identity, without opening reference or source-label assets.
The eight-field prediction schema distinguishes a successful empty list from
JSON, truncation and inference errors. Original raw suffixes retain terminal EOS;
audited diagnostics identify the parseable JSON suffix. Resume validates model,
settings, RGB and audit identities. Legacy region artifacts are rejected.

Run notebook 12 for discovery, then notebook 13 for the fixed four conditions.
References enter only reference-present diagnostic queries. SAM receives each
original phrase, resets query state while retaining the image embedding, and
copies instance outputs before another query can mutate processor state.
Separate condition/profile invocations reject changed SAM identities. Five
development warmups and two repeats remain separate from model loading.

Finally enable segmentation and explicit deployment profile paths in notebook
12's evaluation configuration. Native metadata pointers/hash recipes are in
`configs/hazards/evaluation.json`. Reports retain failures, unscored phrases,
source/split denominators, per-concept and source-valid pixel scores, and
independently measured stage profiles. Successful-query unions determine scores;
all-query unions remain downstream artifacts. Isolated profile peaks/p95 values
never establish joint residency or combined p95.

## Offline CPU replay

From `VLM_evaluation/`, use an isolated Python with the CPU hazard data/evaluation
fragments, pytest, nbformat, nbclient and ipykernel. Refresh this environment's
editable install if its checkout moved:
`python -m pip install --no-build-isolation --no-deps -e .`.
The full replay is explicit and writes only to a new/empty requested directory:

```console
python scripts/run_hazard_fixture.py --output-root ../.codex-local/hazard_fixture_check
```

It prepares 140 synthetic RGB/source-annotation pairs through `prepare_run`,
freezes the same 20-test/5-warmup schedule, uses `run_inference` for both marked
fake VLMs, saves fake SAM masks for all four conditions, and evaluates discovery,
pixel metrics and seven synthetic profiles (two VLM, four SAM, one actual fake
combined replay). Scripted time and unavailable GPU telemetry are explicitly
synthetic. Real accuracy/memory evidence and comparison readiness stay false.

The workflow exercises valid empty output, misses, an unscored term, malformed
JSON, truncation, a failed query with diagnostic masks, low annotation coverage,
missing-record comparison blocking, changed-input rejection across every stage
and legacy scoring/resume rejection. It restores temporary mutations. Compatible
replay is explicit with `--resume`; altered inputs/settings still fail. Outputs
include `fixture_summary.json`, `evaluation_config.json`, the native run artifacts
and evaluator JSON/CSV/JSONL exports.

For combined pytest suites, use importlib mode and expose the independent test
helper directories along with `src` (path separator `;` on Windows, `:` on Linux):

```powershell
$env:PYTHONPATH = ((Resolve-Path src).Path, (Resolve-Path tests/hazard_data).Path, (Resolve-Path tests/hazard_evaluation).Path) -join [IO.Path]::PathSeparator
python -B -m pytest --import-mode=importlib tests
```

Safe headless notebook copies:

```console
python scripts/check_notebooks.py --execution python --output-root ../.codex-local/hazard_notebook_checks
python scripts/check_notebooks.py --execution python --with-fixture --output-root ../.codex-local/hazard_notebook_fixtures
```

Checker switches disable downloads and real model actions; source outputs remain
cleared. Windows accounts without symlink permission skip native symlink tests.
Rerun those cases on a capable host before trusting that host's path containment.

Integration verification on 2026-10-01: the full CPU suite passed 603 tests and
476 subtests, with four native symlink checks skipped on Windows. The connected
140-image synthetic workflow and seven fake profiles produced zero evaluator
validation errors. Safe/fixture notebook copies passed headlessly and source
outputs stayed cleared. These checks establish software behavior only.

## Later consumers and pending measurements

Use the public packages and saved joins rather than re-parsing console output:
`traversability_hazard_data`, `traversability_hazard_inference`,
`traversability_hazard_evaluation`, `traversability_hazard_segmentation` and
`traversability_hazard_benchmark`. RGB and truth masks resolve against the data
root; SAM instance/query/frame masks resolve against the run directory. IDs are
opaque and joined exactly. Profile identity includes component, condition,
profile ID, pinned selection, models and settings.

Future visualization should read the evaluator exports (`hazard_report.json`,
concept/pixel CSVs, phrase/error/frame audits), source RGB/frame/reference joins,
SAM query/frame rows and distinct all-query masks, and native profile samples/
summaries. Preserve fixture markers, status and null unavailable values.
Visualization is deferred to a new session.

Actual 140-image inputs and gated SAM3 checkpoint access are still required.
Follow the exact pinned-source/checkpoint instructions in [setup](setup.md).
Real GPU checkpoint loading, Qwen/SAM output quality, timing, memory peaks,
6 decimal GB VLM target and combined residency are unmeasured. Cables, independent
spills/floor contact, phone clips, physical safety and temporal/depth/fusion/
planning remain pending. Future server preparation must follow the INESC rules;
this integration performs no connection or job.
