# Pipeline configuration and meeting presets

The agreed comparison uses the same 18 project videos for all four pipelines.
João owns geometry only; Tomás uses SAM3 with exactly lowercase **`floor`**;
Florian uses a larger fixed SAM3 vocabulary. Joris's small-VLM-to-SAM assignment
remains tentative. Record runtime, advantages, limitations, and good and bad
examples for each pipeline. The soft deadline is Sunday **11 October 2026**;
the hard deadline is to be decided that Sunday.

These are reproducible starting configurations, not measured model results.
The local checks exercise CPU fixtures and shared software contracts. They do
not establish SAM3 quality, LingBot correctness, GPU runtime or navigation
safety on the project videos. The larger vocabulary has not yet been shown to
be more robust.

## Choose a preset

The registry remains exactly `geometry_only`, `ground_surface`, `fixed_hazards`
and `qwen_hazards`. Preset variants do not introduce additional pipeline IDs.

| Owner / purpose | Preset in `configs/pipelines/` | Pipeline ID | Meaning |
|---|---|---|---|
| João | `geometry_only.json` | `geometry_only` | Common LingBot geometry; no SAM/Qwen semantic queries |
| Tomás, meeting baseline | `ground_surface_floor.json` | `ground_surface` | One exact `floor` query on every selected frame, including outdoor scenes |
| Historical indoor option | `ground_surface_indoor.json` | `ground_surface` | Exact `floor`, historical indoor selection provenance |
| Historical outdoor option | `ground_surface_outdoor.json` | `ground_surface` | Exact `ground` |
| Historical ablation | `ground_surface_path_legacy.json` | `ground_surface` | Exact case-sensitive `Path` |
| Florian, larger starting vocabulary | `fixed_hazards_rich_vocabulary.json` | `fixed_hazards` | 32 frozen phrases with explicit surface/hazard roles |
| Historical hazard benchmark | `fixed_hazards.json` | `fixed_hazards` | Original 16 hazard phrases; model loading disabled by default |
| Current Qwen alternatives | `qwen_hazards_qwen3_5_4b.json`, `qwen_hazards_qwen3_vl_4b.json` | `qwen_hazards` | One selected 4B Qwen proposes hazard phrases; model loading disabled by default |

The meeting floor preset retains `ground_surface_indoor_v1` as the identity of
the exact phrase policy. Its selection basis is
`meeting_exact_floor_baseline_unmeasured`, its environment is `mixed`, and its
configuration digest differs from the historical indoor preset. There is no
automatic indoor/outdoor prompt switching. Uppercase `Floor`, a prompt list,
or a new free-form prompt is rejected by the single-surface adapter.

The floor and richer presets include run-level geometry settings and enable
SAM loading. Their local paths still need to identify already available model
assets. The Qwen and historical fixed-hazard files are semantic presets rather
than fully supplied real-video experiments: merge their `pipeline` section
into the frozen common experiment configuration.

## What the richer vocabulary means

The historical `fixed_hazards` vocabulary comes from the
[visible-avoid policy](../VLM_evaluation/configs/hazards/policy.json). Its 16
canonical queries are concrete hazards used by the RELLIS/COCO image-component
evaluation: `tree`, `pole`, `water`, `vehicle`, `building`, `log`, `person`,
`fence`, `bush`, `barrier`, `mud`, `rubble`, `cup`, `bottle`, `chair`, `dog`.
That historical policy remains strict and unchanged. It cannot be reordered,
prefiltered or extended by setting `prompts` alone.

The new [versioned mapping policy](../configs/pipelines/fixed_vocabulary_surface_hazards_v1.policy.json)
is explicitly selected with `vocabulary_mode: "explicit_concepts_v1"`.
It preserves those 16 queries and adds the following groups in this order:

| Phrases | Recorded role | Recorded concept |
|---|---|---|
| `floor`, `ground`, `path`, `sidewalk`, `pavement`, `asphalt`, `concrete`, `dirt road`, `gravel` | `candidate_surface` | `ground_surface` for all nine |
| `stairs`, `step`, `wall`, `rock`, `cable`, `bicycle`, `bench` | `hazard` | The original phrase for each |

All 32 original strings are passed to SAM3 unchanged, after one image encoding
per selected frame. `dirt road` is one phrase. The lowercase `path` in this
policy is distinct from the historical case-sensitive `Path` ablation.
The policy's `aliases` record lexical provenance; they do not add SAM queries
or rewrite the original phrases.

Surface synonyms share one concept so the common fuser caps correlated
evidence within a frame instead of multiplying support by the number of
surface words. The fuser still retains individual query/instance identities
and masks. Hazard concepts keep their hazard role. A candidate surface is
only evidence of a visible surface; gravel, a road or a floor can still fail
geometry and robot checks. Empty detections do not prove free space or safety.
No benchmark annotation mappings are declared by the new policy, so it is
not a replacement policy for the historical RELLIS/COCO benchmark.

## Configuration fields

| Field | Interpretation |
|---|---|
| Top-level `contract_id`, `schema_version` | `semantic_mapping_v1`, integer `1`; shared packet/output identity |
| Top-level `pipeline_id` and `pipeline.pipeline_id` | Same registered pipeline ID |
| `pipeline` | The adapter settings block used by the common runner |
| `pipeline_config` / `pipelines.<id>` | Supported resolved/combined forms; the selected `pipelines.<id>` takes precedence over `pipeline_config`, which takes precedence over `pipeline` |
| `pipeline.config_id` | Human-readable preset identity; the runner records the resolved configuration digest |
| `ground_surface.prompt` | Exactly one of `floor`, `ground`, `Path` |
| `ground_surface.prompt_selection` | Frozen development-selection provenance, not a measurement of prompt quality |
| `fixed_hazards.vocabulary_mode` | Default `legacy_hazards_v1`; custom policies require explicit `explicit_concepts_v1` |
| `fixed_hazards.policy_path` | Repository-relative or absolute JSON path; required for explicit vocabulary mode |
| Policy `canonical_prompts` / `concepts` | Same ordered inventory; each concept contains exactly `phrase`, `concept_id`, `role` |
| Policy `role` | `candidate_surface` or `hazard`; required for every custom phrase |
| `policy_hash`, `policy_file_sha256`, `alias_hash` | Canonical JSON policy hash, encoded-file hash, canonical JSON alias hash; configured values are checked against the file |
| `sam3` / `sam_settings` | SAM settings in `ground_surface` / `fixed_hazards` and `qwen_hazards`, respectively |
| `enable_model_loading` | Explicit local model loading gate; Qwen and historical fixed presets start disabled |
| `confidence_threshold` | Official SAM3 processor detection threshold; default `0.5`, not a calibrated safety probability |
| `mask_probability_threshold` | Per-pixel mask threshold; default `0.5`, strictly greater than comparison |
| `resolution` | SAM3 image processor resolution, `1008` in these presets; returned masks must match the full shared processed RGB grid |
| `fusion.semantic_threshold` | Separate downstream evidence threshold; not a SAM mask threshold or traversability probability |
| `fixture` | Boolean execution identity; config, sequence and CLI must agree |

An explicit policy accepts 1–32 unique phrases of at most 80 characters each.
Its policy ID must differ from the historical ID and its inventory must align
with the concept records. One concept cannot have conflicting roles.
Historical canonical hazard phrases/concept IDs cannot be assigned a surface
role. Changing vocabulary or roles changes the policy hash, observation
identity and emitted mapping version. The supplied richer preset pins all
three hashes; edit the policy under a new version and recompute the hashes
when creating a new experiment. Git preserves JSON bytes across platforms;
retain the policy's exact encoded bytes while its file checksum is pinned.

The adapters return `SemanticFrame` for the input `FramePacket`, retaining its
frame, processed grid, decoded RGB, geometry fingerprint and capture timestamp
identities. Each `QueryRecord` preserves original phrases and status; each
`InstanceRecord` retains a separate boolean mask and raw score. Failed masks
are diagnostic. A load failure reports failed queries with zero executions;
an uncounted backend crash reports unavailable execution counts rather than an
invented zero. The actual model metadata and policy hashes remain in provenance.

## Prepare real runs

Use an environment that already contains the required model dependencies and
assets. No configuration downloads weights or substitutes another checkpoint.
The [local setup guide](local_setup.md) describes external pinned sources,
environments, checkpoints and caches. Copy a preset to an external local config
and override absolute runtime roots rather than modifying the shared template.
When higher-precedence `pipeline_config` or `pipelines.<id>` blocks are present,
coordinate the overrides in the selected adapter block as described above.

* LingBot needs the intended local checkout and checkpoint, or a complete,
  verified common geometry cache for the exact sequence/settings. The path
  `models/lingbot-map.pt` is a local configuration placeholder, not a supplied
  checkpoint. João's existing LingBot integration blocker must be resolved and
  verified on his device; these presets do not fix upstream model internals or
  establish the cause of that failure.
* All participating models must use the same one indexed CUDA device, such as
  `cuda:0`, also declared in the common geometry/hardware configuration.
* SAM3 is the same pinned official image model in all presets: code revision
  `2345a4ad109ac29c569da749c91d84f10dc08c40`, checkpoint revision
  `3c879f39826c281e95690f02c7821c4de09afae7`, file `sam3.pt`, size
  `3450062241`, SHA-256
  `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`.
  The reader verifies the official source and checkpoint, requiring a compatible
  Python/Torch/CUDA environment; see [SAM3 setup](../VLM_evaluation/docs/setup.md).
* Configure an existing absolute `cache_root` and optional absolute
  `code_source_root` for the executor. A non-null `checkpoint_path` must be a
  POSIX-relative path under `cache_root`, such as `models/sam3.pt`; absolute
  checkpoint paths, backslashes and parent components are rejected.
  SAM relative roots are resolved
  by the component under `VLM_evaluation`, whereas the fixed adapter's relative
  `policy_path` is resolved from the repository root. Using absolute runtime
  cache/source roots avoids ambiguity. The preset combination `cache_root: "../.cache"`,
  `checkpoint_path: "models/sam3.pt"` means the repository's
  `.cache/models/sam3.pt`. A null checkpoint path uses the pinned offline cache
  lookup; it does not authorize downloading. `allow_downloads` is always false
  and `local_files_only` true.
* Freeze the same sequence frames, geometry settings, semantic keyframes,
  fusion settings, evaluation region and runtime policy for comparisons.
  Verified metric scale, an up direction and an actual robot profile are
  required before interpreting plans as physical navigation.

The basic CLI shape, run from the repository root after setting local asset
paths and preparing a real `sequence.json`, is:

```powershell
python src/run_pipeline.py --pipeline ground_surface --sequence output/study/video01/sequence.json --config configs/pipelines/ground_surface_floor.json --output output/study/video01/floor
python src/run_pipeline.py --pipeline fixed_hazards --sequence output/study/video01/sequence.json --config configs/pipelines/fixed_hazards_rich_vocabulary.json --output output/study/video01/rich
```

These commands reconstruct geometry unless `--geometry-cache` points to a
verified common cache. Reuse requires **exactly matching geometry settings**;
copy the geometry block from the completed geometry run's resolved config
instead of combining different presets blindly. For Tomás's existing indoor
floor replay, the [ground-surface preparation helper](implementation/prepare_ground_surface_replay.py)
already verifies the sequence/cache, preserves common experiment fields and
checks local SAM assets without running models:

```powershell
python docs/implementation/prepare_ground_surface_replay.py --geometry-run output/study/video01/geometry_only --sequence output/study/video01/sequence.json --environment indoor --cache-root C:/existing/sam-cache --output output/study/video01/floor.replay.json
```

The helper's indoor choice produces exactly `floor`; its historical indoor
selection metadata differs from the meeting preset. For a full mixed-scene
meeting comparison, keep the `ground_surface_floor.json` adapter block and
the same frozen common fields. When producing resolved replay configs, update
all applicable `pipeline`, `pipeline_config` and `pipelines.<id>` blocks to
avoid an old higher-precedence block overriding the intended preset. The helper
reports asset verification separately from GPU readiness and never establishes
GPU speed. Nothing in this document authorizes new shared-server jobs; existing
server rules and running work remain unchanged.

## Qwen and the tentative small-VLM branch

The current Qwen adapter supports only `qwen3_5_4b` and `qwen3_vl_4b`; these
are alternatives within `qwen_hazards`. The adapter uses the visible-avoid
hazard policy, maps every successful phrase to a **hazard** role, and sends the
original words to SAM3. Canonical aliases map to historical concepts; concrete
unknown phrases have stable unmapped concept IDs. A successful empty list is
recorded as no proposed hazards, not as a safe scene.

Real integration needs the chosen pinned cached Qwen revision/processor,
working 4-bit CUDA dependencies, compatible BF16 compute and the shared SAM3
assets. Malformed/truncated output or unverified generation completion blocks
the SAM handoff. `skip_sam` avoids image encoding for successful empty output;
`encode_image` performs image encoding with zero text queries. The adapter
records separate synchronized Qwen/SAM call timing and model loading timing.

Joris still needs an agreed model/checkpoint, phrase-selection policy and
semantic role mapping. The current 4B hazard branch is not an implemented
generic small-VLM surface-word proposer. Another small VLM or surface discovery
requires an explicit adapter/policy change with the same shared output contract;
do not rename a checkpoint or silently reuse hazard roles as surface evidence.

## Extend and verify an owner's adapter

For Tomás, use the exact floor preset without adding prompt selection logic.
For Florian, create a new versioned explicit policy and preset, preserving
original phrases and setting every role/concept explicitly. Surface synonyms
can share `ground_surface`; for example:

```json
{"phrase": "painted walkway", "concept_id": "ground_surface", "role": "candidate_surface"}
```

Keep an ordered matching `canonical_prompts` list, aliases containing each
original phrase, the explicit vocabulary mode and checked hashes. Aliases do
not count as extra prompts. Adapt model-specific loading inside the owned
adapter while keeping `observe(FramePacket) -> SemanticFrame` and `close()`.
Do not put planners, geometry changes, reference-label access or model fallback
inside a semantic adapter. Any change to shared contracts or common experiment
behavior needs coordination with the common-code owner.

For CPU verification from the repository root, use the environment prepared in
the [external CPU setup](local_setup.md#choose-external-storage-and-check-the-cpu-path).
Its PowerShell commands define `$cpuPython` as the external CPU interpreter:

```powershell
$env:PYTHONPATH = 'src;VLM_evaluation/src'
& $cpuPython -m pytest -q tests/pipelines/test_ground_surface.py tests/pipelines/test_fixed_hazards.py tests/pipelines/test_qwen_hazards.py tests/pipelines/test_meeting_presets.py
```

The tests inject explicitly identified fixture processors into the production
adapters and check original phrase order, masks, roles, counts, provenance and
error behavior. Ordinary `run_pipeline.py --fixture` instead selects the common
synthetic provider. It checks the shared lifecycle/output layout but does not
exercise a production SAM vocabulary or Qwen model. CPU timings from either
fixture path must remain labeled synthetic. Report actual per-video model and
end-to-end timing separately when the real runs are available; cached-geometry
semantic replay is not an end-to-end live-camera measurement.
