# Real-model validation on the 20 GB allocation

Tested 2026-10-13 on H100 MIG `2g.20gb` (19.625 GiB). An isolated
`.venv-gpu-validation` environment uses Torch 2.8.0+cu126, torchvision 0.23.0,
Transformers 5.3.0 and the existing cached LingBot, Qwen3-VL-4B and SAM3 weights.
The shared environment was not modified.

## Results

| Check | Result |
| --- | --- |
| Offline LingBot reconstruction | 12 real frames reconstructed and restored to original source order |
| VLM non-traversable target | Proposed `fallen log` and `log` |
| Base SAM video support | All 12 `ground` masks generated |
| VLM-query SAM and independent `path` | Six mapped-frame mask sets produced; sources remained separate |
| Canonical map | 803,503 original point rows, with source RGB, labels and pixel/frame correspondence |
| Automatic reference and steepness | Completed |
| Default camera-height calibration | `missing_clearance_calibration`; route correctly withheld |
| Explicit diagnostic clearance | 160,632 finite-cost rows and a certified 15-point partial route |
| Non-traversable cost veto | All 53,786 flagged point rows had infinite cost, including 301 positive/negative overlaps |
| Independent path contribution | 44,052 path rows outside base support; 1,139 received finite costs under the diagnostic profile |
| VLM traversable target | Proposed `gravel path`, `concrete slab` and `grass` |
| Mixed computation and reuse | New positive VLM masks combined with bit-identically reused independent path masks |
| Slope-disabled cost profile | Completed with 593,767 finite-cost rows |
| Cached SAM grouped union | Bit-identical to reference masks and warm replay; cache hits recorded |
| Viewer artifact loading | Source, traversability, steepness, cost and certified partial-route layers loaded |
| Regression suites after fix | 158 planner + 44 plugin + 19 integration = 221 passing tests |

Models were run sequentially, not kept resident together. Device memory sampled
every second reached 15.414 GiB used, including approximately 3.17 GiB of existing
allocations. This is a sampled device peak, not an exact transient peak. No OOM
occurred. Cached SAM parity testing separately recorded 4.414 GiB maximum Torch
allocated memory.

## Bug found and fixed

The official LingBot postprocessor squeezes geometry arrays but leaves
`frame_type` and `is_keyframe` as `[1, N]` arrays. The original offline wrapper
rejected that shape after inference. The adapter now normalizes these two known
metadata fields before restoring source order; malformed geometry still fails.
A regression test reproduces the actual output shape. The successful GPU output
retains all twelve frame flags in original source order.

## Clearance and route interpretation

The three-second test clip did not provide baseline-valid beneath-camera support
for height calibration. The default pipeline completed with an explicit no-route
result. It did not invent a camera height or discard that requirement.

To exercise cost construction and native certification on the real masks, a
separate diagnostic run explicitly supplied clearance height equal to the
estimated radius (`0.017742744483910833` reconstruction units). Its partial route
is certified against that declared graph and its intended goal is unreachable.
This is not a camera-calibrated or metric body-clearance result.

## Reproduction and artifacts

All video input, complete run configurations, logs, masks, predictions, receipts,
memory samples and the machine-readable summary are under:

`/home/jovyan/EmbodiedAI-gpu-validation/`

Key runs:

- `combined-real-models-v2/`: real reconstruction, VLM, both SAM stages, steepness,
  and the default calibration/no-route outcome.
- `positive-vlm-and-reused-path/`: traversable VLM target and independent reuse.
- `diagnostic-clearance-plan/`: explicitly supplied clearance and certified route.
- `positive-costs-no-slopes/`: positive evidence with slope checks disabled.
- `cached-sam-real-validation.json`: reference/warm grouped mask parity.
- `gpu-validation-summary.json`: aggregate results and implementation limits.

Run the saved JSON configurations through `python -m orchestrator run --config`,
changing each output to a new directory. Use the GPU interpreter for model stages
and the CPU integration interpreter for orchestration/analysis. The environments
and weight paths are recorded in each configuration.

This validation covers a 12-frame clip sampled at 4 FPS from IMG_5516. It does not
establish full-video accuracy, precision/recall, long-stream memory bounds or
three-model persistent-session capacity. Those need separate workload tests.
