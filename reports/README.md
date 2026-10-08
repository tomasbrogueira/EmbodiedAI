# Reviewed result snapshots

Keep small, reviewed results here so teammates can inspect them after a clone.
Raw runs live under ignored `outputs/`; checkpoints, caches, generated videos
and full geometry archives stay outside Git. The [run storage guide](../docs/run_storage.md)
describes the complete run layout and attempt lifecycle.

| Snapshot | Contents | Scope |
|---|---|---|
| [Pipeline 2 pilot, 2 Oct 2026](2026-10-02-pipeline2-pilot/README.md) | Original configs, metadata, evaluation reports, audited counts and three screenshots | Nine indoor `floor` frames and nine outdoor `ground` frames; real GPU inference, staged replay |

For each future snapshot, include the input recording/frame selection, pipeline
and configuration identities, code/checkpoint revisions, hardware and timing
boundaries, exact-source file hashes, representative examples, and measured
advantages and limitations. Preserve the original attempt and its records.
Use a new dated directory for later results; do not overwrite earlier snapshots.

State which artifacts were omitted and where the complete run is available.
A curated metadata snapshot is not an executable geometry cache or complete run.
Do not convert an unavailable metric into zero, mix synthetic fixtures with
model results, or report cached-stage timing as live end-to-end performance.

Before publishing, inspect screenshots and records for access tokens and local
secrets, verify the SHA256 manifest against the copied bytes, and check that
the report's counts agree with the original semantic/evaluation records.
