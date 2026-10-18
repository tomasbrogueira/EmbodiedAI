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

## Settings

The wrappers use the current path-planning/reconstruct.py (streaming, eight spread initialization views, original frame order) and path-planning/generate_masks.py (existing masks with CPU video storage). Map filters are unchanged. They install nothing and preserve new-output checks and failure propagation. See the [path-planning guide](../../path-planning/README.md) for costs, routes, visualization and tests. Model inference is not part of the CPU regression suite.
