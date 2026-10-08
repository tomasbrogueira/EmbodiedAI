# Run storage and provenance

Run each recording and pipeline into a fresh attempt directory. Keep failed
attempts for diagnosis, and use a new directory for every retry. The runner
refuses every existing output path, including an empty folder, a running attempt,
and a completed run. It never resumes or merges an earlier attempt.

## Choose an attempt directory

Run commands from the repository root in the environment for your pipeline.
The example assumes an already prepared sequence and approved model settings;
see the [team workflow](team_workflow.md) for setup and pipeline ownership.

```powershell
python src/run_pipeline.py --pipeline ground_surface `
  --sequence outputs/sequences/recording_01/sequence.json `
  --config configs/pipelines/ground_surface_floor.json `
  --output-root outputs/runs/recording_01/floor_v1
```

`--output-root` creates a new child such as
`20261008T140000.123456Z_ground_surface_<uuid>`. Names use UTC and the pipeline
ID; the CLI prints the actual attempt path. Keep the parent meaningful:
`outputs/runs/<recording-id>/<experiment-id>/`. Share that full path and the
`attempt_id` when discussing results. The UUID in the directory name and the
metadata's `attempt_id` are independent unique identifiers.

For scripts that already choose a unique directory, use
`--output outputs/runs/recording_01/floor_v1/attempt_001`. The Python API remains
`run(pipeline, sequence_path, config_path, output, fixture=False, cache=None)`
with the last two parameters passed by keyword. `--output` and `--output-root`
are mutually exclusive. Creating the attempt directory is the reservation: one
invocation owns it, and a concurrent loser cannot update its files. Create the
parent in advance if useful; let the runner create the final attempt directory.

Use `--geometry-cache <cache-directory>` only for a compatible, validated cache.
Without it, geometry is reconstructed into this attempt's `geometry/cache/`.
Cache reuse validates the sequence, processed grid, settings and geometry
fingerprints. Do not repoint a run to a different cache with the same name.

## Actual layout

These paths are relative to the attempt directory. Some are absent when the run
fails before their stage; their absence is diagnostic, not a completed result.

```text
<attempt>/
  run.json                    # lifecycle, attempt/config/provenance IDs
  config.requested.json       # requested config snapshot, if readable
  config.resolved.json        # defaults and geometry identity frozen together
  provenance.json             # source hashes, recording, declared hardware, software
  failure.json                # escaped exception type, stage and traceback, if raised
  events.jsonl                # geometry/semantic/planning/cleanup diagnostics
  frames.jsonl                # one disposition per required frame
  summary.json                # counts, timing scope, resource measurements, limitations
  geometry/
    manifest.json             # geometry identity and cache reference
    cache/                    # only when reconstructing inside this attempt
      manifest.json
      geometry.npz
      processed_frames/<index>.png # aligned lossless RGB required by adapters
    camera_trajectory.npz     # when processed geometry frames exist
  semantics/
    frames.jsonl              # queries, errors, per-frame adapter provenance
    masks/<sha256>.npz        # numeric masks; referenced by relative path and hash
  map/
    voxels.npz
    semantic_evidence.npz
    concepts.json
    contributions.jsonl
    manifest.json
    raw_surfaces.ply          # when processed geometry frames exist
  planning/
    costmap.npz
    manifest.json
    plans.jsonl
```

JSON, JSONL and NPZ writers finish a private temporary file in the destination
directory, flush it, then replace the destination atomically. Serialization or
replacement failures leave the previous destination intact and clean up that
temporary file. Transient Windows replacement conflicts have a short, bounded
retry; persistent permission errors remain explicit. This protects each file;
it is not a transaction over the entire
attempt. Files created through these writers use the temporary file's permissions
(owner-only on POSIX); sharing permissions are managed by the owner separately.
No crash durability guarantee is made for an abrupt power loss or filesystem
failure. A forced process kill can leave an unfinished attempt or temporary file.

## Read lifecycle and diagnostics

`run.json` is authoritative for readiness:

| Status | Meaning |
| --- | --- |
| `running` | Validation, reconstruction or semantic processing is underway. |
| `finalizing` | Artifacts or resource measurements are still being written. |
| `complete` | Finalization succeeded and at least one required frame succeeded. |
| `failed` | A processing, validation, cleanup or finalization error occurred, or no frame succeeded. |
| `unavailable` | A required model/dependency/input capability was unavailable; compatible raw geometry may still exist. |

`complete` permits individual frame errors and dropped/skipped semantics; inspect
`summary.json` counts and `frames.jsonl` before claiming full coverage. Likewise,
planning availability is separate from run completion: a complete raw map may
have `blocked_inputs` planning when scale, up direction or robot inputs are
unknown. The maps describe observed surfaces and do not establish free space or
physical navigation safety.

`stage` identifies validation, geometry, semantics, finalizing, resources or
finished. Start/finish timestamps use UTC. Handled errors that still permit
artifact finalization are recorded in events and frame rows. Escaped exceptions,
including an orderly `KeyboardInterrupt`, produce `failed` metadata and
`failure.json` when the filesystem permits it. The original exception remains
authoritative if saving diagnostics also fails. Cleanup errors preserve earlier
processing diagnostics, and events are checkpointed after geometry and before
artifact finalization. Files already written remain available.

CLI exit codes are `0` for complete, `2` for a returned failed/unavailable attempt,
and `1` for an escaped ordinary exception. An interrupted Python process may use
its platform's interruption exit code. A run left running/finalizing after a
forced termination must be inspected manually; the runner does not reclaim it.

## IDs and what provenance captures

| Field | Identity |
| --- | --- |
| `attempt_id` | Random UUID for this execution; changes on retry. |
| `requested_config_digest` | Canonical SHA256 of the parsed requested config. |
| `initial_config_digest` | Resolved config digest before reconstructed geometry identity is added. |
| `config_digest` | Final resolved config, including geometry identity; matches `config.resolved.json`. |
| `provenance_id` | Final config/geometry, sequence, project code, input hashes, software, hardware budget and declared provenance. |

Canonical config hashing uses sorted compact JSON with non-finite numbers
prohibited, excluding only the config's own `config_digest` field. Attempt IDs,
timestamps and output names are kept outside resolved config. Repeating the same
frozen inputs/settings/code in the same software environment gives stable config
and provenance IDs while attempt IDs differ. Changing Python/NumPy/platform or
source-file bytes can change `provenance_id`; config whitespace alone does not
change `requested_config_digest`.

The provenance ID hashes `contract_id`, `schema_version`,
`requested_config_digest`, `sequence`, `hardware_budget`, `declared`, `software`,
`project_code`, `config_digest`, `geometry_identity`, and an `inputs`-derived `source_hashes`
object. Machine-local input locator paths and execution timestamps are excluded
from this recipe. `run.json` and `provenance.json` store the final provenance ID.
Early failures may have only the IDs known at that stage.

`provenance.json` copies the sequence's `input_provenance`, input file hashes,
declared `hardware_budget`, optional config `provenance`, and actual Python,
NumPy and platform versions. Geometry manifests and semantic rows carry their
existing model, checkpoint, code and preprocessing identities. Keep the full
sequence manifest externally with the prepared images: it records selection,
frame hashes, dimensions and timestamp provenance.

`project_code` fingerprints the project Python files on disk when validation
finishes: `src/pipeline_common/`, `src/path_mapping/`, `src/pipelines/`, root
`src/*.py` CLIs and `VLM_evaluation/src/`. It stores repo-relative paths, file-byte
SHA256 values, count and aggregate hash. Recipe
`sha256_project_python_sources_v1` hashes the sorted list of
`{"path":...,"sha256":...}` records with the same canonical JSON function.
Tests, data, outputs, caches and vendored `src/lingbot-map/` and `src/sam3-robot/`
are excluded; upstream code/checkpoint pins remain in model provenance. A shared
code edit changes `provenance_id` even when settings and comparison gates match.
Keep the source checkout stable during an evaluation: this fingerprint records
disk contents, not a source archive or a snapshot of already imported modules.
The repository's `.gitattributes` keeps Python sources in LF form on Windows and
Linux, preserving their file-byte hashes across fresh clones. Historical files
under `reports/` retain their exact bytes; do not normalize or reserialize a
published snapshot to match a new checkout or code revision.

Before a real evaluation, put useful facts into config `provenance`, for example
`owner`, `experiment_id`, `recording_id`, repository commit and local-change note,
GPU model, driver/CUDA/library versions, and links to the external recording/cache.
These declarations are preserved and hashed; the runner does not independently
verify them. Use verified checkpoint/code pins and the same frozen sequence
manifests across all four pipelines. `hardware_budget` is a declaration, not a
hardware measurement. Actual process resource samples and their limitations live
in `summary.json`; missing measurements remain explicitly unavailable.

Report both total staged run duration and audited model-call timing with their
scope, GPU synchronization and cache-reuse facts. Batch LingBot reconstruction
and paced cached replay do not establish live end-to-end real-time operation or
simultaneous model fit. Synthetic `--fixture` runs validate integration only.

## Move or publish results

Mask paths are relative to the attempt and carry hashes. Geometry's additive
`cache_reference` is either `{"kind":"run_relative","path":"geometry/cache"}`
or an explicit external location. A complete run with an internal cache can move
as a directory without changing those references. The legacy absolute
`cache_path` remains for existing replay/export readers: after moving a run, pass
their explicit cache override where supported. These older readers do not all
automatically follow `cache_reference`. External cache artifacts must be kept
separately and identified by the recorded geometry/input fingerprints.

For Git review, curate a small result directory under `reports/` with a
short report, `run.json`, requested/resolved configs, `provenance.json`,
`summary.json`, selected small metrics and representative images. Include the
recording ID, pipeline/config IDs, attempt ID, fixture/model execution facts,
runtime scope, advantages/limitations and good/bad examples. Record an external
artifact URI, geometry fingerprint and hashes needed to retrieve bulk artifacts.
Check configs, tracebacks and manifests for local paths, credentials and personal
details before publishing.

Large geometry caches, masks, point clouds, generated exports, environments,
weights and bulk logs stay in ignored output storage or an external archive.
A summary-only publication is not a replayable full run: label omitted artifacts
explicitly and keep the original attempt intact. Write evaluations and derived
exports to separate directories named with the source attempt ID; retries and
derived outputs should preserve their parent run identity.

The common/geometry-only transfer bundle (`python -m pipeline_common.bundle` with
`PYTHONPATH=src`) also refuses an existing archive or companion manifest. It
includes code/tests and this guide, with file hashes; it is not a results archive
or the full four-pipeline distribution.
