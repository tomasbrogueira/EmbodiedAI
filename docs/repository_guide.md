# Repository guide

Start at the [top-level README](../README.md). The common runner is the integration
path for comparing the four pipelines; older mapping tools remain available for
replaying and inspecting existing artifacts.

## Structure

| Location | Purpose |
| --- | --- |
| src/run_pipeline.py | Shared pipeline CLI and lifecycle |
| src/prepare_sequence.py | Generic video/frame preparation |
| src/evaluate_pipeline.py, src/compare_pipelines.py | Common reporting and comparison gates |
| src/pipelines/ | One semantic adapter per stable pipeline ID |
| src/pipeline_common/ | Contracts, input bridge, geometry cache, fusion, scheduler, storage, planner and evaluator |
| configs/pipelines/ | Versioned pipeline settings and policy presets |
| src/path_mapping/, src/fuse_path_into_map.py | Original staged mapper and reusable geometry/archive machinery |
| src/lingbot-map/, src/sam3-robot/ | Pinned source components; coordinate submodule changes |
| src/export_*.py, src/serve_*.py | Artifact rendering, recording export and local viewing tools; inspect each CLI's help |
| tests/ | Mapping/adapters/CLI software checks |
| requirements/ | Scoped dependency lists; see its README |
| [reports/](../reports/README.md) | Curated small result publications and provenance; bulk runs remain external |
| data/videos/ | The 18 intentionally versioned recordings |
| data/prepared/, runs/, outputs/, output/, results/, tmp/ | Local generated data and artifacts, ignored by Git |
| VLM_evaluation/ | Supported component evaluation subsystem reused by semantic adapters; separate environment/protocol |
| docs/implementation/ | Historical handoff specifications and scoped implementation notes |
| research/, notebooks/, proposal.tex, docs/pipeline_math.tex | Research material, analysis and illustrations; preserve their provenance |

The preparation does not remove legacy SAM2/CLIP work in VLM_evaluation.
It remains component/research material outside the current four-pipeline comparison.

## Interfaces and extension points

| Need | Current source of truth |
| --- | --- |
| Pipeline IDs and typed frame/query/instance records | [contracts.py](../src/pipeline_common/contracts.py) |
| Adapter selection and fixture-only substitution | [pipelines/__init__.py](../src/pipelines/__init__.py), [fixture_adapters.py](../src/pipeline_common/fixture_adapters.py) |
| Verified file-backed model handoff | [input_bridge.py](../src/pipeline_common/input_bridge.py) |
| Sequence identities, hashes and timestamps | [sequence.py](../src/pipeline_common/sequence.py) |
| Geometry cache, calibration and aligned RGB | [geometry.py](../src/pipeline_common/geometry.py) |
| Run config resolution and stage orchestration | [runtime.py](../src/pipeline_common/runtime.py) |
| Artifact layout, statuses and attempts | [run storage](run_storage.md) |
| Prompts, roles, thresholds and checkpoint settings | [pipeline configuration](pipeline_configuration.md) |
| Comparable metrics and reference requirements | [evaluation.py](../src/pipeline_common/evaluation.py) and [team protocol](team_workflow.md) |

Each pipeline exports **create_adapter(config)** and returns the shared
**SemanticFrame** through **observe(frame)**; **close()** releases its resources.
Keep model loading lazy so CPU checks can import common code. Preserve exact
grid/hash/frame identities and per-query status; a missing detection, failed
query and unqueried concept have different meanings. Model scores are evidence
weights, not calibrated robot safety probabilities.

Adapters produce observations. The common modules own fusion, scheduling,
planning and evaluation. When extending a pipeline, add a named config/policy
and meaningful adapter checks rather than copying those common stages.

## Prepare a shared sequence

Save a JSON preparation config outside the source tree, for example under
ignored outputs/configs/. For a bounded development smoke on IMG_5506:

~~~json
{
  "contract_id": "semantic_mapping_v1",
  "schema_version": 1,
  "sequence_id": "IMG_5506",
  "split": "development",
  "fixture": false,
  "sample_fps": 2.0,
  "stride": 1,
  "max_frames": 8
}
~~~

Pass that file to **src/prepare_sequence.py --config** as shown in the README.
This example selects at most eight frames; it does not define the final full-video
protocol. Freeze the agreed sampling and limits before comparison. **sample_fps**
applies to videos. **stride** additionally subsamples video or naturally ordered
folder inputs. Frame folders can provide **timestamps_ns** (one per selected
image) and **timestamp_provenance**; otherwise their capture timing is unknown.
Use a stable recording/sequence ID and the agreed development/test designation.
Choose a new output path for each preparation; existing paths are refused.

## Run and report flow

Preparation writes a portable sequence manifest beside its selected images.
Geometry establishes the processed grid and cache identity. Semantics attach to
that grid, then fusion writes geometry and per-concept evidence with source
contributions. Planning uses the declared robot/calibration inputs. Evaluation
reads completed artifacts into a separate report directory; comparison reads
those evaluation directories.

See [run storage](run_storage.md) for the actual file inventory and cache transfer
rules. Moving only a run directory can leave a referenced cache unavailable.
Publish sufficient referenced assets or a self-contained bundle when someone
else needs to reproduce the report.

## Parallel ownership

The [meeting assignments](team_workflow.md#owners-and-current-scope) are the
default code ownership boundaries. Coordinate changes to common modules and
exporters before editing them. During this preparation, a separate active chat
owns planning/rendering changes; documentation and adapter cleanup must preserve
that work. Research/profile exports can illustrate an existing reconstruction;
they do not establish physical robot performance.

Historical specs may contain stricter requirements or older missing-feature
claims. Compare them with current source and checks before implementing anything.
