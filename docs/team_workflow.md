# Team workflow and evaluation protocol

The 8 October 2026 meeting agreed on four pretrained pipelines and one shared
video comparison. The soft deadline is **Sunday 11 October 2026**. At that
meeting, decide what to merge and publish and set the hard deadline for the
remaining work. Joris's assignment is tentative.

## Owners and current scope

| Owner | Pipeline ID | Primary work |
| --- | --- | --- |
| João | geometry_only | LingBot-only reconstruction, current LingBot issue and its working environment |
| Tomás | ground_surface | SAM3 with exact lowercase **floor**, on the same geometry |
| Florian | fixed_hazards | A larger robust fixed SAM vocabulary with explicit concept roles |
| Joris, tentative | qwen_hazards | Small VLM phrase proposal followed by SAM3 |

Each owner checks their adapter, versioned configs and tests, evaluates the
shared recordings and reports runtime, advantages, limitations and good/bad
examples. Review pipeline configuration before using a historical preset; hazard
labels and candidate surfaces have different meanings.

Shared preparation, contracts, geometry, fusion, scheduling, storage, planning
and evaluation need coordinated changes. Pipeline authors should not independently
fork these stages. Agree on the affected files before parallel edits. Preserve
existing research, component evaluations, submodule changes and server runs.
João's LingBot fix should arrive with the working source/environment details and
a small reproducible check; avoid speculative upstream fixes elsewhere.

## Shared videos and frozen experiments

Use the [same 18 recordings](../data/videos/README.md), five indoor and thirteen
outdoor. Prepare a recording once, then share its sequence manifest and selected
images. Record source hash, selection settings, frame IDs, timestamps and
timestamp provenance. Video presentation timestamps do not establish wall-clock
camera capture latency; folders with unknown timestamps remain explicit.

Agree on scene/sequence-disjoint development and test videos before choosing
words or thresholds. The repo does not currently supply a frozen 18-video split
or independent video reference annotations. Record the chosen split in each
preparation config and publish the agreed list. Do not tune on test outcomes.

For a matched **quality_replay**, freeze the same sequence, geometry cache,
processed RGB grid, semantic keyframes, calibration, fusion settings, robot
profile, goals, evaluation ROI and declared hardware budget. Change only the
declared semantic policy/model variant. Keep config digests and model revisions.
Use the common evaluator/comparer; an incompatible comparison reports differences
and must not become a combined ranking.

The historical RELLIS/COCO component evaluation has a different protocol and
scope. Keep its reports unchanged. It does not substitute for these videos.

## Runtime scopes

Report timing boundaries, hardware, loading/warmup, synchronization, completed
and failed frames, attempted/executed queries, and measurement scope together.

| Scope | What it supports |
| --- | --- |
| CPU fixture | Software contracts and CLI behavior; synthetic execution |
| Saved-image Qwen/SAM component measurement | Those adapter calls under the recorded component profile |
| Staged quality replay | Offline geometry/semantic/fusion/planning stage diagnostics |
| Cached geometry scheduler replay | Queue/drop behavior at replay timing, with geometry already reconstructed |
| Actual streamed end-to-end run | Full capture/decode through geometry, semantics, fusion and map/planning update, only when connected and measured |

Current LingBot integration reconstructs a sequence batch. Selecting
**paced_runtime** exercises semantic scheduling against staged geometry; the
common report keeps live end-to-end latency/update-rate/evidence-age claims
unavailable. It also cannot establish simultaneous three-model GPU residency.
Do not add component p95s or separate memory peaks to claim whole-pipeline speed.

For everyone, report wall-clock duration, selected/processed/failed frames and
available stage median/p95 with counts. Report loading separately and declare
whether a cold first call is included. State preprocessing resolution, GPU
visibility, host/GPU memory collection scope and whether geometry was reused.
If GPU timing is asynchronous, use the adapter's synchronization evidence.

Use the current metric IDs and availability rules in
[evaluation.py](../src/pipeline_common/evaluation.py). Unknown/unavailable values
remain null with reasons; geometry-only semantics are not applicable.
Keep failed attempts in failure-rate denominators.

## Quality and examples

Without independent references, use visual examples and internal diagnostics,
and label accuracy metrics unavailable. Predictions from SAM or LingBot cannot
be their own ground truth. Metric planning additionally requires measured scale,
verified up direction and a real robot profile; a returned internal path is
not independent navigation success.

Useful common diagnostics include geometry validity/reprojection, semantic
query counts, failures, coverage/unknown regions and plan statuses. If independent
coverage exists, use common semantic/robot/navigation metrics with the required
reference protocol. Propose extra metrics to the team with a definition, units,
scope and shared feasibility; do not silently add a pipeline-specific score to
the combined comparison.

For each good/bad example, record the video, frame ID or timestamp, run/config
identity and artifact reference. Describe the observed behavior and practical
effect. Include the aligned RGB/mask and map view when useful; distinguish an
observed issue from a suspected cause. Retain poor examples and failures.

## Per-pipeline report template

Copy this into a reviewed Markdown report, replacing every placeholder:

~~~markdown
# <Pipeline ID> — <owner> — <date>
- Variant/config digest:
- Commit and model/checkpoint revisions:
- Dataset/split/sequence manifests:
- Hardware, resolution, selected frames and cache reuse:
- Run/evaluation IDs and bulk artifact locations:

## Results
| Video | Run ID | Status | Processed/required frames | Runtime and scope | Evaluation/report |
| --- | --- | --- | --- | --- | --- |
| <recording> | <attempt> | <status> | <counts> | <duration; median/p95 if measured> | <link> |

## Advantages
<Observed strengths with examples and relevant conditions.>

## Limitations and failures
<Observed weaknesses, missing measurements, calibration/reference limits.>

## Good and bad examples
| Kind | Video/frame or time | Observation | Artifact | Suspected cause, if any |
| --- | --- | --- | --- | --- |
| <good/bad> | <identity> | <behavior and impact> | <link> | <clearly marked hypothesis> |

## Runtime
<Loading/warmup, stage boundaries/counts/failures, median/p95, hardware and
memory scope. State explicitly whether end-to-end and real-time were measured.>

## Verification and blockers
<Checks run, real-model coverage, unresolved issues and needed team decisions.>
~~~

## Publishing results

Full runs and evaluations live outside Git in the ignored run/result locations.
See [run storage](run_storage.md) for immutable attempts and artifact references.

For the reviewable push, include only deliberate compact summaries, provenance
and selected examples under the reviewed **reports/** area. The
[2 October pilot publication](../reports/2026-10-02-pipeline2-pilot/README.md) preserves
available historical floor/ground evidence. Future reports should use a distinct
dated directory with the per-pipeline template above. Link to externally retained
bulk data and include IDs/hashes sufficient to trace a report to its run and
checkpoints. The storage guide also describes the fields needed in a curated
publication; a small summary is not a replayable full run.

Publish available Tomás results with honest scope now; add later results as new
reviewed reports when they arrive. Preserve original runs and existing jobs.
Check example rights, size and accidental personal/credential paths before
publishing. The project owner approves the GitHub push after review.

## Shared-server work

These INESC rules apply only to Aquila and INESC compute hosts. Before any
connection or remote command, read
**C:\Users\tomas\.ssh\INESC_SERVER_RULES.md** in full, together with project
instructions. Aquila is SSH/status only. Compute aliases use ProxyJump with the
local key; keep agent forwarding disabled.

Immediately before every job, inspect CPU load/available cores, RAM, disk bytes
and inodes, and GPU processes. An occupied GPU at zero utilization is still
occupied. Use at most one currently free GPU, selected by full UUID through
**~/.local/bin/inesc-run**, inside a named tmux/screen session on the compute host.
Fit workers and library threads within the requested budget; do not bypass
launcher refusals.

Keep home below 30 GB. Put bulk data, environments, caches, checkpoints and logs
under **/data/tmpC/tomasbrogueira**. Record host, session, command, resource budget
and log path; scratch is not a backup. Preserve
**/home/tomasbrogueira/SERVER_RULES.md** references in remote AGENTS.md/CLAUDE.md,
including projects outside shared home. Do not change other users' work or launch
unrequested jobs. KTH requires its own current resource checks and allocation;
historical KTH capacity is not present availability.

This repository preparation performs no server connection, job launch or job
modification.
