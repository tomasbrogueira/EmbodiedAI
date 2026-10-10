# Workflow scripts

Run LingBot → SAM3 → semantic point cloud using the isolated KTH setup.
Follow the [setup notes](../../KTH_SETUP_FIXES.md) first. Scripts activate the
three environments under `.setup/envs/` automatically; they install nothing.

## Run

~~~bash
cd /home/jovyan/EmbodiedAI-non-vibed
bash orchestrator/scripts/run_pipeline.sh \
  .setup/verification/IMG_5522_32f.mp4 floor .setup/runs/my-run
~~~

Arguments: `VIDEO PROMPT NEW_RUN_DIR [--view]`. Add `--view` to open the viewer
after processing. Use a **new output directory** for every run.

Outputs: `lingbot/` predictions, `sam3/` masks, `semantic_map.ply`, and
`logs/` inside the run directory. Processing stops on the first failed stage.

## Individual commands

Run with `bash orchestrator/scripts/<script>`. All support `--help`.

| Script | Arguments |
| --- | --- |
| `run_lingbot.sh` | `VIDEO NEW_OUTPUT_DIR` |
| `run_sam3.sh` | `VIDEO PROMPT NEW_OUTPUT_DIR` |
| `build_map.sh` | `VIDEO PROMPT LINGBOT_DIR SAM_MASK_DIR NEW_OUTPUT_PLY` |
| `view_map.sh` | `PLY [VIEWER_OPTIONS...]` |

Relative paths start at the repository root. Quote spaces; prompts must be
nonempty and contain no slashes or newlines. Map inputs are the exact
`lingbot/<video_stem>/` and `sam3/<video_stem>_masks/` directories.
Existing output directories and PLY files are never overwritten.

## View a result

~~~bash
bash orchestrator/scripts/view_map.sh .setup/runs/my-run/semantic_map.ply
~~~

Open the [KTH viewer](https://gpu1.eecs.kth.se/user/tombro/vscode/proxy/8080/?websocket=wss://gpu1.eecs.kth.se/user/tombro/vscode/proxy/8080).
Keep the terminal running; press Ctrl+C to stop. The default port is 8080.

## Settings and verification

Processing settings follow the root README. The approved exception is LingBot's
`--no_render`: predictions are saved, but its video renderer is skipped.
The PLY viewer still works. No dependencies or existing Python files were changed.

Verified on KTH on **2026-10-10** with the 32-frame clip and `floor` prompt:
exit 0, 32 predictions, 32 masks, and a 3,732,781-point PLY. The viewer was tested
separately with its default 500,000-point sample. Shell syntax, argument checks,
overwrite protection and failure propagation also passed.

Results and logs: `.setup/verification/scripts smoke 20261010/`.
Long videos, other prompts, segmentation accuracy, and LingBot video rendering
were not verified.
