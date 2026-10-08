# Contributing

Coordinate work by pipeline owner and keep changes focused. The
[team workflow](docs/team_workflow.md) records the current assignments and shared
evaluation protocol; the [repository guide](docs/repository_guide.md) identifies
the common interfaces.

## Before changing code

Check the current working tree and preserve unrelated local work. Use a branch
per change, such as **codex/ground-surface-evaluation**, and identify the files
you intend to edit. Separate checkouts/worktrees help parallel work; uncommitted
files do not automatically appear in a new worktree.

Keep the four pipeline IDs and the **semantic_mapping_v1** handoffs compatible.
A semantic adapter returns **SemanticFrame** for the exact **FramePacket**
identity and processed RGB grid. Retain per-concept roles, original phrases,
errors and actual model-call counts. Link interface changes to
[contracts.py](src/pipeline_common/contracts.py) and coordinate every affected
consumer before merging.

Discuss changes to common preparation, geometry, fusion, planning, scheduling,
storage and evaluation with the team. New words or thresholds should normally
be a versioned pipeline config/policy change. Keep historical presets and report
which variant produced a result. Model submodules have their own histories;
coordinate source/pin changes, especially João's LingBot fix.

## Check and review

From the repository root in the CPU environment:

~~~bash
python -m pip install -r requirements/path_mapping_cpu.txt -r requirements/test.txt
python -m pytest -q tests
~~~

Use a fresh output directory for the [CPU fixture](README.md#start-with-a-cpu-fixture)
when a change affects shared handoffs or the CLI. Run focused production-adapter
fixture checks for changed semantics. These checks establish software behavior;
record real-model tests separately with actual hardware, weights and timing scope.

The [CPU workflow](.github/workflows/cpu-tests.yml) runs the top-level checks on
Linux and Windows for pushes and pull requests, using Python 3.12 and the test
requirements. Check the current commit's results in
[GitHub Actions](https://github.com/tomasbrogueira/EmbodiedAI/actions). The workflow
follows [GitHub's Python testing guidance](https://docs.github.com/en/actions/tutorials/build-and-test-code/python).

Python sources use LF line endings through `.gitattributes`, so byte-level source
pins stay stable across checkouts. JSON keeps its committed bytes because
versioned policies can pin their encoded-file hash. Recompute the declared hash
when publishing an intentional policy revision. Preserve files under `reports/` byte-for-byte:
their publication manifests bind the original evidence and screenshots.

The separate [VLM_evaluation setup](VLM_evaluation/docs/setup.md) governs its
component tests and additional dependencies.

A review should explain the problem, resulting behavior, affected configs and
checks, plus any unrun model validation. Include relevant success/failure examples.
Do not combine unrelated cleanup with research or evaluation changes.

## What belongs in Git

Commit source, configs, tests, docs and the intentionally included project
recordings. Keep full runs, caches, checkpoints, extracted frames, renders,
credentials and machine-specific settings outside Git. The ignore rules protect
common local locations; check newly added files explicitly.

Publish small reviewed result summaries with provenance and stable links to bulk
artifacts as described in [team workflow](docs/team_workflow.md#publishing-results).
Do not rewrite source runs to fit a publication format. Preserve older reports
and failures. The project owner authorizes publication to GitHub after review.
