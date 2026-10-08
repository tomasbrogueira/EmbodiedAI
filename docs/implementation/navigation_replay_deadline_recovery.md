# Optional replay deadline recovery candidate

This is a local, isolated candidate. The running controller, installed support,
frozen source ZIP, original recordings, and published pairs are unchanged. The
candidate is not needed while the 900-second replay stages succeed. It must not
be uploaded or launched without a concrete timeout and an explicit recovery
decision.

Candidate files:

- `.codex-local/kth-server/navigation_all18_20261008_deadline.py`
- `.codex-local/kth-server/pipeline2_cpu_stage_20261008_deadline.py`
- `tests/test_navigation_deadline_recovery.py`

The supervisor candidate differs from its exact frozen ZIP member only in two
literal replacements: `not 1 <= max_runtime <= 900` becomes `<= 3600`, and the
matching validation message states 3600. Its SHA-256 is
`58925c4682dbfaef0d95c780bd1902d40d3a0cee4c44293eea47c8de386ac986`.
Ownership, inherited leases, parent-death handling, descriptor checks, process
identity checks, TERM/KILL cleanup, and cleanup reporting are unchanged. There
is no shim, monkey patch, or in-memory modification.

The controller accepts `--replay-timeout 900..3600`; 2400 is the suggested
recovery bound. Every other stage stays at 900 seconds. The actual requested
bound and supervisor hash appear in each new stage receipt. The batch identity
also binds the replay deadline, other-stage deadline, supervisor admission cap,
and supervisor hash. Resume requires an exact identity, including that policy.
The four-core affinity, four-thread environment, CUDA-disabled execution, and
pre-stage resource checks remain in place.

## Required evidence before migration

`--migrate-execution-tools OLD_BUNDLE_SHA` requires `--resume --retry-failed`
and `--previous-bundle` pointing to the exact retained original ZIP. The old
batch must have finished with `complete_with_failures`, cover the same ordered
18 clips, and contain a failed current replay attempt with
`stop_reason=whole_cpu_stage_deadline`, `max_runtime_seconds=900`, and verified
reaping of the helper, helper group, and owned descendants. Running,
interrupted, resource-waiting, subset, and non-timeout batches are rejected.
The candidate accepts only this existing batch and subsequent resumes with
committed migration history; it cannot start a fresh batch.

The controller first acquires the existing native-controller lock and batch
lock. It then acquires the shared execution lease for the whole migration
audit and installation. An active or draining job prevents migration before
support installation or receipt changes.

Admission checks both ZIP hashes, exact manifests, the complete installed old
support inventory, old receipt identity, all five canonical source-input
closures per clip, source-video provenance, all five completed stage receipts
for every completed clip, completed stage output inventories and supervision
hashes, all retained attempts, both completed videos, previous delivery hash,
and the original published-pairs baseline. The new support must be a fresh
versioned path. The allowed bundle diff is exactly:

- `tools/navigation_all18_20261008.py`
- `tools/pipeline2_cpu_stage.py`

All pipeline and test members, bundle metadata, and file names remain exact.
Additional supervisor changes are rejected byte-for-byte. The controller
invoked for migration must itself match the new bundle member.
The qualifying timeout must also match the exact original replay CLI, saved
adapter specification, old supervisor hash, helper command, inherited
descriptors, and native/controller/execution lease paths. A replay-named report
from another command cannot qualify. Output paths inside the original or
published-pairs trees, including descendants, are rejected before any write.

## Preservation and recovery behavior

The same batch directory is retained. Completed clip records and their old
attempt paths remain unchanged and are skipped. Failed clips receive fresh
attempt directories; partial attempts and old support are retained. A new
`migration_history/execution_tools_<time>/` directory stores the exact previous
receipt, a copy of the previous delivery ZIP, and an exclusive-create audit
event binding old/new identities, manifests, source closure, retained output
closure, deadline evidence, and both tool hashes. The new master receipt refers
to that event by hash. Resume and finalization recheck the history, old support,
and retained attempts. Gallery and delivery ZIP are regenerated after retry;
the archive inside migration history preserves the prior delivery.

The original receipt is changed only after admission and fresh installation
succeed. A rejected admission leaves it unchanged. An interrupted installation
or uncommitted history directory is retained for review; no automatic deletion
or rollback occurs. If interruption precedes receipt commit, use another fresh
support path after inspecting the retained preparation. If commit succeeded,
resume with the new exact bundle, support, and deadline, without the migration
flag. No route geometry, model cache, semantic mask, planner source, robot
assumption, or pose admission policy changes in this recovery.

## Review and use

Local checks:

```powershell
python -B -m unittest discover -s tests -p test_navigation_deadline_recovery.py -v
```

The source bundle is prepared directly from the immutable original ZIP, never
from current workspace pipeline sources. It retains the original 42 member
names and replaces only the two canonical tool members plus their manifest
hashes. The local `create_recovery_bundle` function requires a fresh output
file. The optional recovery ZIP and its receipt contain the resulting exact
hashes. They have not been uploaded or used.

Only after the required finished-batch evidence and a recovery decision, a
reviewed deployment can stage the candidate controller byte-for-byte from its
new ZIP into a separate file outside old support. The following invocation is
a template, not a performed action. `OLD_ZIP`, `NEW_ZIP`, `NEW_SHA`, and
`CANDIDATE_CONTROLLER` must be replaced with their verified staged paths/hash.

```bash
ROOT=/home/jovyan/EmbodiedAI-pipelines
"$ROOT/.venv-pipelines/bin/python" -B "$CANDIDATE_CONTROLLER" \
  --root "$ROOT" --bundle "$NEW_ZIP" --bundle-sha "$NEW_SHA" \
  --previous-bundle "$OLD_ZIP" \
  --migrate-execution-tools 9b065ac58b607d4e985aec02acb6f0cbf47df9289de3765be99627b271fec560 \
  --output "$ROOT/outputs/evaluation_navigation_all18_20261008" \
  --support "$ROOT/.viewer_support/navigation-all18-20261008-deadline-v1" \
  --resume --retry-failed --replay-timeout 2400
```

This candidate does not add parallel clips, a new model run, or any change to
the active batch.
