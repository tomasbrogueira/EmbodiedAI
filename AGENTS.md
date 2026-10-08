# Repository agent guidance

Read `README.md`, `CONTRIBUTING.md` and `docs/team_workflow.md` before changing
shared pipeline behavior. Preserve unrelated local work. Check which files
other sessions are editing before working in a shared checkout.

The public pipeline IDs are `geometry_only`, `ground_surface`, `fixed_hazards`
and `qwen_hazards`. Preserve the `semantic_mapping_v1` frame/grid/geometry
identities and the adapter interface in `src/pipeline_common/contracts.py`.
Keep per-query surface/hazard roles, original phrases, errors and actual model
execution counts. Coordinate changes to common interfaces and their consumers.

Use CPU fixtures and focused checks first. Model downloads, remote workloads,
training and bulk dataset evaluations require task authorization; they are not
cleanup side effects. Do not interrupt existing jobs. João owns the current
LingBot integration fix; do not rewrite upstream internals during repo cleanup.

Keep generated runs and temporary files outside Git. Create a fresh attempt for
each run, preserve failures, and publish small reviewed snapshots under
`reports/` with source hashes and timing scope. Never label fixture or cached
stage timing as real-model live performance. See `docs/run_storage.md`.

Model submodules have separate histories. Preserve their pinned revisions and
record dependency changes deliberately. Source/config/doc changes are not
permission to push; follow the user's publication instructions.

<!-- BEGIN INESC SERVER RULES -->
## INESC-ID shared-server use

These rules apply only to Aquila and its INESC-ID compute servers.
Before connecting or running any remote command, read
`C:\Users\tomas\.ssh\INESC_SERVER_RULES.md` in full.
Aquila (`aquila.inesc-id.pt`) is SSH/status only. Never run workloads, installs,
agents, editor backends, or persistent sessions there. Compute aliases already
use ProxyJump with the local key and agent forwarding disabled.
Check current CPU load, RAM, disk space, and GPU processes before every job.
Use tmux/screen on a compute host and `~/.local/bin/inesc-run` for workloads.
Use at most one free GPU. Keep home below 30 GB. Use
`/data/tmpC/tomasbrogueira` for bulk files, environments, caches, and logs.
Never change other users' work, override safeguards, or launch unrequested jobs.
Preserve `/home/tomasbrogueira/SERVER_RULES.md` references in remote project
AGENTS.md and CLAUDE.md files, including projects outside the shared home.
<!-- END INESC SERVER RULES -->
