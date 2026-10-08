# Repository readiness — 9 October 2026

The local repository preparation is complete. The project owner authorized
publication to `main` after review. The starting Git commit is
`d763b12`. Existing component changes
and research work have been preserved. No server connection or job change was
made.

Start with the [README](../README.md), then the
[team workflow](team_workflow.md), [configuration guide](pipeline_configuration.md)
and [run storage guide](run_storage.md). These explain the four owner assignments,
shared interfaces, common recordings, timing scopes and Sunday handoff.

## Prepared changes

| Area | Result |
|---|---|
| Collaboration | Owner-specific adapter/config guidance, shared-stage boundaries, contribution checks and a common per-pipeline report template |
| Meeting presets | Exact lowercase `floor` preset for Tomás and an opt-in, versioned 32-phrase surface/hazard policy for Florian; historical presets remain available |
| Run storage | Fresh attempt reservation, unique output-root naming, lifecycle/failure records, atomic artifact writes and requested/resolved configuration snapshots |
| Reproducibility | Input/config/source hashes, model identities, explicit timing scope, cache references and documented relocation limits |
| Tests | Independent temporary video-export fixtures, subprocess source paths, CPU dependency setup and Linux/Windows GitHub checks |
| Cleanup | Private server-monitor/batch tests kept locally, OS metadata removed from publication, and bulk outputs/environments/caches ignored |
| Available results | A compact, hashed historical Tomás pilot with original metadata and three representative images |

The route-viewer checksum allowlist now includes the reviewed current route
exporter and retains the historical exporter pin. Exporter behavior and running
workloads were preserved. A Qwen CPU test now supplies a drive-qualified absolute
cache path on Windows and checks the processor and model cache paths exactly;
production model-loading code and checkpoint pins were preserved.

Git keeps Python sources in LF form and preserves committed JSON bytes so
encoded policy checksums survive Windows/Linux checkouts. Historical report
files also bypass line-ending conversion.

## Verification

The top-level CPU suite passed against an exact copy of the publishable source
files: **500 tests and 430 subtests passed; nine Linux-only process tests were
skipped on Windows**. The copy contained no local outputs, private helpers,
model submodule checkouts or recorded videos. This is a source-copy check, not
an actual Git clone. Tests generate their analytic inputs and do not load model
weights.

Checks used Python 3.12.13, NumPy 1.26.4, Pillow 11.3.0, OpenCV 4.11.0.86 and
pytest 9.1.1 in a separate CPU environment. Shared runtime/storage tests and the
four-pipeline run/evaluate/compare fixture also passed.

The separate VLM component covered **658 distinct CPU tests: 653 passed and
five Windows symlink checks were skipped**. This coverage comes from completed
checks of the initial directories and the remaining five directories after the
Qwen test-input correction, with overlapping tests counted only once. Setup
checks passed with the already declared notebook dependencies (`nbformat`
5.11.1 and `nbclient` 0.11.0) installed only in the isolated test environment.
Subtest totals from overlapping component runs are not summed. These checks
include synthetic notebook profiles and do not load actual model weights.

Independent reviews found no remaining findings in the storage changes, richer
vocabulary contract, source-pin correction or curated pilot. Reviewed local
links and heading anchors in the setup/workflow documents resolve. Both
model submodules are clean at their recorded revisions:

- LingBot wrapper: `849e690bb086103637e44b1e91878d9d43a8bf0c`.
- SAM wrapper: `e467ecef8efdc86eaa806d27a99f749b45e4853f`.

All 18 recovered recordings match their original Git blobs at
`a13bc91bc1519c748934dd9df5dc5f0e45deb3a3`. The published
[dataset manifest](../data/videos/manifest.json) records their SHA256 hashes and
byte sizes, totaling 261,164,118 bytes. An ordinary clone includes the videos.

The publication inventory contains the 18 intended project recordings and no
individual file above 100 MiB. Python syntax, selected credential-token patterns,
ignored-output leakage and the 13 published evidence hashes passed inspection.
Historical report files retain their original bytes. These checks do not replace
review of future additions.

The checked-in [CPU workflow](../.github/workflows/cpu-tests.yml) is configured
for Linux and Windows. The local checks above precede hosted CI; consult
[GitHub Actions](https://github.com/tomasbrogueira/EmbodiedAI/actions) for the
published commit's results, including the Linux-specific process checks.

## Ready for teammate work

Each owner can start from the shared CPU fixture, use a frozen prepared sequence
and work within their adapter and versioned configuration. Changes to preparation,
geometry, fusion, scheduling, planning and evaluation require coordination.
Retries use new attempt directories; bulk runs stay outside Git. Later reports
use new dated directories rather than overwriting the
[available pilot](../reports/2026-10-02-pipeline2-pilot/README.md).

The following research work remains explicit:

- João fixes and verifies the reported LingBot integration problem on his
  working setup; this preparation does not establish its cause or resolve it.
- Owners evaluate the 18 recordings and report runtime, failures, advantages,
  limitations and concrete good/bad examples. The richer vocabulary is an
  unmeasured starting policy.
- Joris's assignment and small-VLM choice remain tentative. The current Qwen
  adapter proposes hazards; a generic surface-word proposer needs an agreed
  model and explicit semantic-role policy.
- Independent video annotations, a frozen development/test split, metric scale,
  verified up direction and a robot profile remain team decisions where needed.
  Their absence keeps the corresponding quality/navigation metrics unavailable.
- The historical pilot uses staged cached-geometry inference. It does not prove
  live end-to-end performance or that all models fit together on one GPU.

The [external setup guide](local_setup.md) describes explicit preparation of
model sources, weights, environments and caches outside the repository on PCs
or servers. Importing code, preparing sequences and running the shared pipelines
do not automatically download model assets; missing local assets fail clearly.

The soft deadline remains Sunday 11 October. The team sets the hard deadline at
that meeting. These research limits are recorded alongside the reviewed code
and clearly labeled current evidence.
