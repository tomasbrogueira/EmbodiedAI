# VLM masks with the original mapping pipeline

[run_existing_pipeline.py](run_existing_pipeline.py) is a separate optional runner. It calls the existing LingBot batch reconstruction and `orchestrator/main.py` without changing either implementation. The older pipeline and plugin entry points remain available.

## Settings and frame identity

The runner uses the existing README's reconstruction settings: windowed mode, window size 32, keyframe interval 2, two scale frames, eight overlap keyframes, and SDPA. Rendering is disabled to measure reconstruction and semantic mapping without the unrelated rendering stage; image extraction defaults are unchanged.

Mapping uses the original settings: frame step 2, pixel step 1, minimum depth confidence 2.5, maximum depth 25, patch size 14, and RGB marking color 255,0,0.

VLM and SAM both consume precisely the saved frames selected by the mapper (`frame index % 2 == 0`). This runner intentionally replaces the plugin's older every-30th-frame discovery default. VLM still processes one image per call and collects a sequence-wide concept list. Every selected frame is segmented with all discovered concepts. The new plugin uses SAM3 image inference rather than the original single-prompt video tracker; matching the sampling and mapping settings does not make these two inference modes identical.

A read-only symlink view of the selected predictions prevents list-position sampling from selecting different frames when frame numbers have gaps. `selection.json` records the exact common frame list. The mask adapter checks the source RGB hash, spatial dimensions, manifest frame list and that the original mapper's patch-14 preprocessing leaves the aligned mask unchanged. It exports filenames such as `IMG_5522_obstacles_frame000002.npz` for the existing mapper. Original predictions and SAM outputs are not modified.

## Run all videos and measure time

From the repository root, with the interpreters for the existing environments:

```bash
python plugins/small_obstacles/run_existing_pipeline.py run \
  --videos-dir data/videos \
  --output-dir data/vlm_maps/my_run \
  --reconstruct \
  --lingbot-python /path/to/lingbot-env/bin/python \
  --vlm-python /path/to/vlm-env/bin/python \
  --sam-python /path/to/sam3-env/bin/python \
  --mapper-python /path/to/orchestrator-env/bin/python
```

Use `--video-path data/videos/indoor/IMG_5522.mp4` instead of `--videos-dir` for one video. Use `--lingbot-root data/lingbot_output` instead of `--reconstruct` to reuse existing per-video predictions; this excludes reconstruction time. `--dry-run` prints the selection settings and planned scene output paths without running models or writing files.

Each run requires new scene output directories. Outputs are grouped by the input video's parent directory and stem. Reconstruction outputs live within the new run's scene folder, preventing overwrite of earlier geometry.

```text
RUN_ROOT/indoor/IMG_5522/
  lingbot/IMG_5522/frame_*.npz
  selected_predictions/frame_*.npz  # read-only source view
  selection.json
  prompts.json
  masks/frame_*.npz
  mapper_masks/IMG_5522_obstacles_frame*.npz
  obstacle_map.ply
  timings.json
  lingbot.log
  vlm.log
  sam.log
  adapt_masks.log
  mapping.log
RUN_ROOT/batch_timings.json
```

Model stdout/stderr streams to the console and individual stage logs. `timings.json` records the running stage before it starts, then its exit status and elapsed wall time on completion. `batch_timings.json`, `timing_summary.csv` and `index.html` update after each scene, including failed scenes. Run `python plugins/small_obstacles/timing_report.py RUN_ROOT` to refresh the report during an active stage. The runner continues to later videos after a scene failure and returns a failure status if any scene failed.

## Interpreting timing

Stage wall times include process startup, imports, model loading, inference, transfers and file IO. Models are loaded independently per video, and stages run sequentially to keep GPU memory usage bounded. These are end-to-end processing costs, not isolated model latency. Rendering and interactive viewing are excluded.

The runner also parses the existing LingBot log's reported inference duration and FPS when available. That is separate from full reconstruction-stage wall time. VLM/SAM stage rates divide selected-frame count by stage wall time. SAM time depends strongly on the sequence-wide prompt count; an empty list avoids loading SAM and still writes empty masks. Mask-adapter/mapping rates are IO and geometry-processing rates, not GPU model Hz.

The original LingBot batch program can record a scene failure while exiting zero; the runner checks its `batch_results.json` as well as its process return code. This avoids treating partial reconstructions as completed scenes.

The mapper exports XYZ/RGB PLY; the merged binary masks retain explicit obstacle observations. No scene-level semantic fusion, path planning, or extra traversability model is added.
# Compare exported maps in Viser

Run the separate viewer in the existing visualization environment:

```bash
conda activate orchestrator-visualizer
python plugins/small_obstacles/view_maps.py data/vlm_maps/timed_run --port 8081
```

Forward port 8081 using your JupyterHub proxy or VS Code. Choose Map A and Map B
from the dropdowns, then select side-by-side, a single map, or an overlay. Overlay
comparison requires maps in shared coordinates; different videos are generally
independent reconstructions. Side-by-side views center each map for display and
retain its scale. Reset the camera after changing maps.

The viewer shows accumulated whole-video point clouds, sampling up to 200,000
points per map by default. `--max-points` adjusts the display limit without editing
the files. Two 500,000-point clouds can exceed the JupyterHub proxy's default
10 MiB WebSocket message limit, causing repeated reconnects. The default keeps
the initial pair below that limit. Neutral geometry hides RGB and obstacle markings; original RGB beneath
marked points cannot be recovered from the PLY alone. Red color is not a reliable
semantic label because naturally red surfaces can share that color. The original
per-frame masks remain the authoritative segmentation output.

You can also pass two or more explicit PLY paths, including baseline exports,
instead of a directory. The original mapper and its single-map viewer are unchanged.
