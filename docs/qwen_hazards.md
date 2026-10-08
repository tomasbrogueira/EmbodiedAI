# Qwen-guided hazard mapping

`qwen_hazards` is one architecture with two separately selected 4B variants:

| Model key | Configuration |
|---|---|
| `qwen3_5_4b` | `configs/pipelines/qwen_hazards_qwen3_5_4b.json` |
| `qwen3_vl_4b` | `configs/pipelines/qwen_hazards_qwen3_vl_4b.json` |

The pretrained Qwen and SAM3 components are composed directly. There is no
training, alternative model fallback, automatic switching, controller/reference
dependency or adapter queue. Import and factory calls are model-free. Actual
loading is lazy, local-cache-only and disabled in the checked-in configurations.
The common runner owns geometry, scheduling, voxel fusion, planning and evaluation.

## Interface and observations

`pipelines.qwen_hazards.create_adapter(config)` accepts the common runner's
resolved pipeline section. Its synchronous `observe(FramePacket)` returns the
actual `pipeline_common.contracts.SemanticFrame`; `close()` releases only resources
loaded by this adapter. Private injected backends/clocks require explicit fixture
mode and stable provider identities. They remain caller-owned.

Qwen receives the whole LingBot-processed RGB grid through
`pipeline_common.input_bridge.to_model_input`. The bridge verifies lossless RGB
PNG bytes, decoded pixels and equality with the packet before inference. It is
called again after Qwen and SAM. Generic frames use the explicit
`input_contract_id=semantic_mapping_v1` readers. No RELLIS/COCO identity is invented,
and benchmark annotation validation stays strict. Qwen's input contains no depth,
poses, SAM masks, reference assets or image-specific annotation names.

Successful discovery sends original phrases unchanged and in order to SAM on
the same file/grid. Water aliases map to `water`, people aliases to `person`,
fallen-log aliases to `log`; canonical IDs and mapping version
`visible_avoid_concepts_v1` match fixed hazards. Additional concrete nouns have
stable `unmapped:<sha256-of-normalized-phrase>` IDs. Original words, model/policy
identity, source RGB/transform, both file/pixel hashes, grid, geometry fingerprint,
pose revision and captured timestamp remain in provenance.

Individual SAM instances and scores are retained; overlap between concepts
remains independent evidence. Mixed successful/failed queries become `partial`.
Failed-query partial masks remain diagnostic; the common fuser accepts only
successful queries. The legacy all-query union is never semantic input. SAM scores
are uncalibrated detector evidence weights; Qwen supplies no confidence probability.
Successful queries with no detections and successful empty Qwen discovery do not
prove traversability. Omitted Qwen concepts remain unqueried.

Qwen error/truncation/invalid JSON/unverified stop prevents SAM calls. Raw decoded
response, terminal-EOS/parser diagnostics, actual generation token counts when
available and the unchanged backend metadata snapshot are copied per call. A
preprocessing wrapper explicitly names `input_space=lingbot_processed_rgb`, the
source/grid/transform identities and actual Qwen resize dimensions. The legacy
metadata wording “original whole RGB image” refers to the supplied processed
image. These inputs differ from the earlier original-camera-image benchmark.

`empty_discovery_policy=skip_sam` is the default: successful empty discovery makes
zero SAM method/encoding/text calls. `encode_image` explicitly reproduces the
legacy empty-prompt image call. A failed legacy image encode stays `error`.
Saved method counts are `qwen_predict_image` and `sam3_segment_image`; actual SAM
text executions come from validated frame/per-query counters. Unexpected crashes
or unauditable joins/counters produce `query_count=null` with
`query_count_reason`; the evaluator reports that total as unavailable and retains
the known subtotal. Invalid masks do not erase otherwise audited execution counts.

Per-stage monotonic start/completion/duration records include failures and cold
loading. Genuine model calls synchronize the selected CUDA device without
resetting global peak counters. `timing.model_calls_ms` / `semantic_call_ms` is the
sum of the actual Qwen and optional SAM calls in that observation, excluding
loading and bridge/output checks. `qwen_call_ms`, `sam3_call_ms`, `loading_ms` and
GPU synchronization status are separate. The common runner consumes the
multiple-model timing hook. The complete `observe` boundary includes loading and
validation; it is saved separately by the core. Null timing is not a measured zero.

## Configuration and validation

Both configs pin the existing checkpoint revisions, deterministic generation,
language-only NF4 4-bit weights, BF16 floating vision/compute, concurrency one,
1024 visual tokens, 2048 context tokens, 256 output tokens and thinking disabled.
An unsupported BF16 device is an explicit failure. SAM config dictionaries match
`fixed_hazards.json` exactly: same official code/checkpoint, 1008 resolution,
confidence/mask thresholds and postprocessing. Loading records retain actual
software, dtype, quantization and checkpoint audits; requested settings do not
establish successful loading, accuracy or a performance bound. Both providers
must match the complete common policy/alias hashes, not just nominal policy names.

Run the CPU conformance and integration suite from the repository root:

```text
python -m unittest discover -s tests/pipelines -p test_qwen_hazards.py -v
```

On this Windows workspace, `python` is available at
`C:\Users\tomas\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`.
29 tests passed with Python 3.12.14, NumPy 2.3.5 and Pillow 12.3.0. These are CPU
fixtures; they do not validate the separately pinned real GPU stack. The tests
exercise both variants, original/unknown phrases, failures/empties, immutable
hash/grid/pose joins, actual legacy RGB/SAM generic readers, partial masks,
call counts, owned cleanup, the actual common fuser and bounded scheduler.
A delayed frame retains its captured geometry after newer geometry advances;
the superseded pending frame is journaled, and mapped capture age uses the
correct frame. Synthetic/unknown capture timestamps cannot establish real age.

Saved evidence is under
`.codex-local/pipeline4_qwen_validation/adapter_cli_v2/`: both variant configs
through the actual CLI argument parsers, common runtime/map/planner/evaluator,
plus fixed hazards on identical geometry/robot/ROI settings. Fixture provider
injection exercises this production adapter implementation; ordinary shared
`--fixture` runs use the common fake registry instead. Each Qwen has 9 text
queries across 3 images; fixed hazards has 48. Water/person/log meanings reach
the map, geometry arrays match exactly, and comparison is compatible with no
automatic ranking. `adapter_cli_crash_v1/` retains one failed SAM frame per Qwen,
an unavailable text-count total and a failure rate of 1/3. Missing-reference
reports preserve null quality metrics. All such results are explicitly synthetic.

Legacy suites also passed: hazard inference 70 tests (2 platform symlink skips),
hazard segmentation 52, and hazard benchmark 31. No legacy reports were rewritten.

## Public and prepared KTH commands

The public entry point is identical for both variants:

```text
python src/run_pipeline.py --pipeline qwen_hazards --sequence <sequence.json> --config configs/pipelines/qwen_hazards_qwen3_5_4b.json --output <fresh-run-dir>
python src/evaluate_pipeline.py --run <fresh-run-dir> --output <separate-evaluation-dir>
```

Loading-disabled configs intentionally cannot perform real inference. The
following is a prepared small KTH replay, **not run in this session**. The current
authorization is for the supervising chat's pipeline 1 run. Qwen GPU validation
requires a separately authorized job, the pinned isolated Python 3.12+ GPU stack,
existing checkpoints/access and a common sequence of at most 3 frames with a
verified shared geometry cache. Use one selected Qwen per process/run; change
`QWEN_KEY` for a separate second variant run.

```bash
cd /home/jovyan/EmbodiedAI
export PYTHONPATH="$PWD/src:$PWD/VLM_evaluation/src"
# Select the already prepared sequence, pipeline-1 common config and cache.
: "${SEQUENCE_DIR:?Set the directory containing sequence.json}"
: "${COMMON_CONFIG:?Set the common experiment or baseline resolved JSON config}"
: "${GEOMETRY_CACHE:?Set the verified cache used by the other pipelines}"
: "${TRAVERSABILITY_CACHE_ROOT:?Set the existing absolute checkpoint cache}"
QWEN_KEY=qwen3_5_4b

# Check current allocation/processes before the authorized job.
uptime
free -h
df -h "$PWD" "$TRAVERSABILITY_CACHE_ROOT"
nvidia-smi

SMOKE_ROOT="$(mktemp -d /tmp/qwen-hazards-smoke.XXXXXX)"
python - "$QWEN_KEY" "$COMMON_CONFIG" "$SEQUENCE_DIR" "$TRAVERSABILITY_CACHE_ROOT" "$SMOKE_ROOT/config.json" <<'PY'
import json, sys
from pathlib import Path
key, common, sequence, cache, target = sys.argv[1:]
assert key in {'qwen3_5_4b', 'qwen3_vl_4b'}
assert len(json.loads((Path(sequence)/'sequence.json').read_text())['frames']) <= 3
config = json.loads(Path(common).read_text())
options = json.loads(Path(f'configs/pipelines/qwen_hazards_{key}.json').read_text())['pipeline']
options['enable_model_loading'] = True
options['qwen_settings']['cache_dir'] = str(Path(cache)/'huggingface/hub')
options['sam_settings'].update(enable_model_loading=True, cache_root=cache)
config.update(pipeline_id='qwen_hazards', fixture=False, pipeline=options, pipeline_config=options)
Path(target).write_text(json.dumps(config, indent=2)+'\n')
PY
python src/run_pipeline.py --pipeline qwen_hazards \
  --sequence "$SEQUENCE_DIR/sequence.json" --config "$SMOKE_ROOT/config.json" \
  --geometry-cache "$GEOMETRY_CACHE" --output "$SMOKE_ROOT/run"
python src/evaluate_pipeline.py --run "$SMOKE_ROOT/run" --output "$SMOKE_ROOT/evaluation"
```

Set `sam_settings.code_source_root` only when the installed SAM3 needs a verified
clean official checkout rather than its pinned VCS installation. Preserve caches,
credentials and completed runs; no downloads or installs are implied by this
command. Paced scheduling can be selected through the common experiment config,
but batch/cached LingBot replay remains a staged measurement.

Real Qwen/SAM/LingBot execution, joint GPU residency, growing-map host/GPU memory,
latency, independent video quality, verified physical scale/up and the real robot
profile remain unvalidated. Missing scale/profile blocks physical planning; missing
independent references leaves visual/operational results. Historical component
medians exclude geometry, scheduling/fusion/planning and cannot choose a winner
or establish whole-pipeline speed/fit.
