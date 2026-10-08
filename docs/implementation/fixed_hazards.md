# Fixed hazard vocabulary runtime adapter

`fixed_hazards` queries the original frozen policy once per shared semantic
keyframe. Every observation contains sixteen separate concepts, in this order:
tree, pole, water, vehicle, building, log, person, fence, bush, barrier, mud,
rubble, cup, bottle, chair and dog. Runtime does not read annotations, benchmark
selection, reference-present prompts, predictions or evaluator output.

`src/pipelines/fixed_hazards.py` exports `create_adapter(config)` and synchronous
`observe(FramePacket) -> SemanticFrame`, plus `close()`. The shared registry
passes the resolved `pipeline` section with its explicit fixture flag. Loading
is lazy through `traversability_hazard_segmentation.load_segmenter(settings)`;
one `segment_image(record, prompts, data_root)` call submits the full vocabulary.
The existing pinned SAM3 wrapper encodes that image once, resets prompt state
and executes one original text query per concept. The common runner controls
keyframes, worker scheduling, geometry, fusion, planning and evaluation.

The adapter consumes `pipeline_common.input_bridge.to_model_input(frame)`.
Its lossless processed PNG must match both the encoded-file hash and decoded
uint8 RGB hash and the immutable processed grid. No private resize, crop or
dataset identity is added. The shared `input_contract_id=semantic_mapping_v1`
route accepts generic video frames; legacy RELLIS/COCO reference validation stays
strict. Input provenance is copied and checked after inference so a mutable
backend cannot rewrite the source, transform, frame or clock identity.

Query/instance IDs identify observations, not tracked objects. Original phrases
and versioned canonical concept IDs remain separate fields. Overlapping masks
and multiple instances survive. Raw SAM scores are nullable, uncalibrated
evidence weights. The common fuser applies robot concept costs only after the
common geometric checks. Missing or invalid geometry abstains from projection
while raw masks remain saved.

All successful queries, including sixteen successful empty detections, produce
`ok`. Mixed outcomes produce `partial`; all failed queries or image-encoding
failure produce `error`. Failed queries can contain validated partial masks:
these stay in diagnostic instances with `fusion_eligible=false`, and the common
fuser accepts only `ok` queries in an `ok`/`partial` frame. Neither legacy union
is used as fusion input. Successful empties do not clear geometry or certify
safety. Requested, successful and failed queries are separate from the actual
text execution count; image/reset failures execute zero text queries, while an
attempted text call that fails still counts one.
Model-loading failure is an explicit unavailable error with zero calls and zero
text executions, and the adapter does not retry loading automatically. An
unexpected backend crash preserves the known adapter call and a null text count
with its reason; it creates error diagnostics for every requested concept. The
common evaluator keeps the total query metric unavailable when any execution
count is unknown and preserves failed frames in the failure-rate denominator.

Policy identity includes version, canonical JSON digest, encoded policy-file
SHA256, alias digest and named hash recipes. Actual SAM settings, checkpoint and
code verification, explicit device, thresholds, mask processing, software and
fixture identity are retained. The configuration pins the same SAM checkpoint,
resolution 1008, confidence 0.5 and strict mask probability 0.5 as both Qwen
variants. Real execution uses the existing wrapper's scoped BF16 inference and
float32 capture. Loading time is saved separately in adapter provenance.
Genuine SAM call timing synchronizes the explicit selected device before and
after the call. Synchronization failure makes that adapter timing null with an
explicit reason, rather than reporting a cheap asynchronous call.
Only an adapter-loaded segmenter is closed; injected fixtures remain borrowed.
Cleanup errors propagate while wrapper references are released.

## Configuration and commands

`configs/pipelines/fixed_hazards.json` is a pipeline settings template. Model
loading is disabled by default, weights must already be cached, and downloads
are refused. The pipeline template must be combined with the same frozen common
experiment settings used by the geometry and ground-surface runs. Its defaults
do not invent metric scale, gravity, robot dimensions or independent references.

```text
python src/run_pipeline.py --pipeline fixed_hazards --sequence <sequence-dir>/sequence.json --config <resolved-fixed-config.json> --geometry-cache <verified-common-cache> --output <new-run-dir>
python src/evaluate_pipeline.py --run <new-run-dir> --output <new-evaluation-dir>
```

Only an independently supplied evaluator reference enables covered quality
metrics. Omitting it preserves null/unavailable accuracy, safety and navigation
metrics. The completed `hazard_prompt_v1` static-image results remain component
evidence; they do not validate these processed video grids or robot navigation.

## CPU conformance

From the repository root, with NumPy and Pillow available and both local source
roots on `PYTHONPATH`:

```text
python -m unittest discover -s tests/pipelines -p test_fixed_hazards.py -v
python -m unittest discover -s VLM_evaluation/tests/hazard_segmentation -v
```

The adapter tests run the real existing SAM wrapper with a mocked CPU processor,
not fake successful real inference. They cover the exact inventory and one image
call, overlapping water/person/log instances, successful empties, image/reset/
text failures and counts, wrong grids/hashes/pixels, generic input, immutable
provenance, actual checkpoint metadata and borrowed/owned cleanup. Shared
VoxelFuser checks preserve all sixteen registry entries, separate evidence at an
overlapping voxel and frame/query traceability, while excluding failed partial
masks and invalid geometry. A shared planner check confirms an observed clearance
violation stays blocked despite favorable surface evidence.

The integration test calls the actual run/evaluate/compare CLI functions for
geometry_only, ground_surface and fixed_hazards on one common fixture sequence
and verified geometry cache. Geometry/ground use the common explicit fake
providers; fixed_hazards uses this actual adapter with a marked mock processor.
It verifies the required common artifact layout, sixteen query records and
numeric masks, sixteen actual fixture text calls, compatible fingerprints and
null quality metrics when references are absent. The standard `--fixture` CLI
uses common fake providers and is not evidence that pretrained SAM executed.

## Small KTH real-model replay preparation

These are prepared commands, not an executed or authorized pipeline-3 server
job. The supervisor handles transfer, fresh CPU/RAM/disk/GPU/allocation checks
and any separately authorized execution in an isolated environment. Use the
same indexed visible GPU and the existing pre-cached pinned SAM checkpoint;
preserve the completed hazard environment and runs. First complete/review the
shared foundation and geometry smoke. Historical MIG capacity does not establish
current headroom or joint LingBot/SAM residency.

The following concrete names assume the supervisor's two-frame geometry smoke
uses `outputs/kth_smoke/sequence/sequence.json` and
`outputs/kth_smoke/geometry_only`. Set `TRAVERSABILITY_CACHE_ROOT` to the verified
existing SAM cache before execution. If it is not the Hugging Face snapshot
layout, set `checkpoint_path` to the existing file's POSIX-relative path inside
that cache in the generated config; the loader verifies size/SHA256.

```bash
cd /home/jovyan/EmbodiedAI
python - <<'PY'
import json
from pathlib import Path
base = Path('outputs/kth_smoke/geometry_only')
assert json.loads((base / 'run.json').read_text())['status'] == 'complete'
sequence = json.loads(Path('outputs/kth_smoke/sequence/sequence.json').read_text())
assert sequence['fixture'] is False and 1 <= len(sequence['frames']) <= 2
common = json.loads((base / 'config.resolved.json').read_text())
assert common['fixture'] is False and common['mode'] == 'quality_replay'
section = json.loads(Path('configs/pipelines/fixed_hazards.json').read_text())['pipeline']
section['enable_model_loading'] = True
common.update(pipeline_id='fixed_hazards', pipeline=section, pipeline_config=section)
common.pop('config_digest', None)
common.pop('pipelines', None)
target = Path('outputs/kth_smoke/fixed_hazards.run.json')
assert not target.exists()
target.write_text(json.dumps(common, indent=2) + '\n')
PY
python src/run_pipeline.py --pipeline fixed_hazards --sequence outputs/kth_smoke/sequence/sequence.json --config outputs/kth_smoke/fixed_hazards.run.json --geometry-cache outputs/kth_smoke/geometry_only/geometry/cache --output outputs/kth_smoke/fixed_hazards
python src/evaluate_pipeline.py --run outputs/kth_smoke/fixed_hazards --output outputs/kth_smoke/fixed_hazards_evaluation
```

Inspect frame/error/query records, per-instance overlays on persisted processed
RGB, separate concept contributions and geometry abstentions. A cached short
quality replay checks model loading, identity and mapping handoff. It does not
measure live end-to-end latency, joint residency peaks, temporal safety, accuracy
without independent references, or physical robot feasibility. The common
implementation currently stages batch geometry before semantic scheduling;
paced scheduling remains a staged diagnostic until genuine streaming geometry
and resource telemetry are available.
