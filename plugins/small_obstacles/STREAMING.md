# Incremental video mapping

`run_streaming.py` is a separate optional runner on `vlm_improvements`. It accepts
a video directly and writes an accumulated obstacle map while frames are being
processed. The existing batch pipeline, SAM3 code, LingBot model code and original
mapper remain unchanged.

```bash
python -m plugins.small_obstacles.run_streaming \
  --video-path data/videos/outdoor/IMG_5515.mp4 \
  --output-dir data/vlm_maps/my_live_run \
  --lingbot-python /opt/conda/envs/lingbot-map/bin/python \
  --vlm-python /opt/conda/bin/python \
  --sam-python /opt/conda/envs/sam3/bin/python \
  --viewer-python /opt/conda/envs/orchestrator-visualizer/bin/python \
  --input-fps 4 --map-step 2 --vlm-interval 2 \
  --realtime --serve --port 8082
```

Run the controller in an environment containing NumPy, OpenCV, and the original
geometry dependencies; the existing `lingbot-map` environment works. Each model
worker uses its own Python environment. Install the VLM requirements into the VLM
environment first. In this particular workspace Accelerate was installed under
`/tmp/small-obstacles-vlm-deps`, so the tested command uses:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=/tmp/small-obstacles-vlm-deps \
  /opt/conda/envs/lingbot-map/bin/python -m plugins.small_obstacles.run_streaming \
  --video-path data/videos/outdoor/IMG_5515.mp4 \
  --output-dir data/vlm_maps/my_live_run \
  --lingbot-python /opt/conda/envs/lingbot-map/bin/python \
  --vlm-python /opt/conda/bin/python \
  --sam-python /opt/conda/envs/sam3/bin/python \
  --viewer-python /opt/conda/envs/orchestrator-visualizer/bin/python \
  --realtime --vlm-interval 2 --serve --port 8082
```

Use a new output directory for each run. Add `--camera-iterations 1` to select
the tested faster configuration; the default retains the original four camera
refinement iterations. One iteration changes reconstruction behavior and needs
quality evaluation before deployment.

## Tested memory-constrained configuration

Smaller-cache tests completed three distinct full videos on this machine without
dropping selected frames. The command below selects the configuration tested
with active SAM segmentation on the log video. It retains the original
four camera iterations, image resolution, keyframe interval and mapping filters.
The smaller cache changes temporal context and therefore changes geometry;
it is opt-in, with the original 64-keyframe setting still the default.

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=/tmp/small-obstacles-vlm-deps \
  /opt/conda/envs/lingbot-map/bin/python -m plugins.small_obstacles.run_streaming \
  --video-path data/videos/outdoor/IMG_5516.mp4 \
  --output-dir data/vlm_maps/my_bounded_run \
  --lingbot-python /opt/conda/envs/lingbot-map/bin/python \
  --vlm-python /opt/conda/bin/python \
  --sam-python /opt/conda/envs/sam3/bin/python \
  --input-fps 4 --map-step 2 --vlm-interval 2 \
  --kv-cache-window 8 --worker-threads 4 \
  --cuda-cache-policy budget --sam-text-cache --save-frames
```

This command processes the entire selected sequence as quickly as possible.
For paced playback, add `--realtime`; that mode can drop late samples. For a live
viewer, add `--serve --port 8083` and
`--viewer-python /opt/conda/envs/orchestrator-visualizer/bin/python`.
Port 8083 avoids the existing demo viewer on 8082; use the corresponding proxy
route or port forward.

On the log clip this configuration processed 50 geometry frames and 25 map
updates in 18.36 seconds (2.72 geometry Hz, 1.36 map Hz), with 0.92 GiB minimum
sampled free GPU memory. Loading is separate. This is not a 30 FPS result or an
indefinite-stream memory guarantee. See the
[results and geometry comparison](../../docs/vlm_bounded_cache_results.md).

| Option | Default | Effect |
| --- | --- | --- |
| `--kv-cache-window` | `64` | Existing model's recent patch keyframe window; `8` reduces context and memory. |
| `--worker-threads` | `0` | Keep environment defaults; `4` caps each worker's Torch and BLAS threads. |
| `--cuda-cache-policy` | `retain` | `budget` releases unused allocator blocks after a request when device free memory is low; `release` always releases unused blocks. |
| `--cuda-cache-min-free-gib` | `2` | Trigger for `budget`; transient free memory can still fall below this threshold. |
| `--sam-text-cache` | disabled | Bounded per-model text-feature cache and GPU union, with one final CPU mask transfer. |
| `--sam-text-cache-size` | `16` | Maximum cached exact prompt strings. |
| `--memory-sample-interval` | `0.1` | Background memory trace period in seconds; `0` disables background traces but keeps request snapshots. |

The patch cache also retains the two initial scale frames. Older special tokens
continue to accumulate under the original model's eviction behavior, so memory
is not strictly constant for arbitrarily long input. Cache bytes, patch keyframe
count, and special-token count are recorded. SAM reuses only text features;
every mapped image still gets fresh image features and masks.

## Processing and scheduling

- Decode one selected frame at a time; do not preload the video. Preprocessing
  matches the original video loader's OpenCV resize/crop, normally width 518 and
  dimensions divisible by 14.
- Load LingBot, Qwen and SAM once and keep them resident in separate workers.
  This requires enough GPU memory for all three models together. The tested
  H100 MIG 20 GB partition successfully ran all three on a short, 22-frame
  retained stream. A later no-drop replay failed after 22 completed frames with
  a CUDA allocator assertion. The growing cache does not fit alongside Qwen
  and SAM on longer streams at these settings; see the measured
  [performance analysis](../../docs/vlm_performance_analysis.md) and
  [implementation plan](../../docs/vlm_performance_implementation_plan.md).
  The optional smaller-cache configuration above completed the subsequent
  no-drop tests; the 64-keyframe default has not been made viable for all three
  resident models on this partition.
- Use the existing LingBot model's causal `forward` loop with its KV cache:
  two initial scale frames, keyframe interval 2, sliding patch cache 64 and SDPA.
  Reuse the original demo's postprocessing, including its pose convention.
  Unlike the batch runner, this uses a persistent causal stream rather than
  overlapping windows and their later alignment. It therefore does not produce
  identical reconstruction to the previous windowed exports. The sequence limit
  defaults to 1024 retained frames; increase `--max-frame-num` for longer clips,
  with appropriate memory and reconstruction-quality checks.
- Block once for the initial VLM proposal. Later proposals run asynchronously,
  with at most one request outstanding and no growing proposal queue. The
  interval is measured in source-video seconds, not wall time. An outstanding
  call can delay the next requested refresh.
- Retain recently proposed object concepts for five source-video seconds by
  default. Expiration uses the proposal image timestamp, so a delayed result
  does not become artificially fresh. Equivalent `fallen log`/`detached log`/
  `log` proposals become one `log` query. Other exact prompt strings remain
  distinct; at most eight active concepts are passed to SAM.
- SAM computes a new image embedding and a new union mask on every mapped frame.
  Only concept names are reused. A mask is never reused on another camera view.
  Empty active concepts bypass SAM inference and produce an all-false mask.
- Project using the original official geometry helper and mask preprocessing.
  Preserve pixel step 1, confidence >2.5, depth >0 and <25, patch size 14, finite
  geometry filtering and red markings. Tests compare the resulting observations
  directly against the unchanged original mapper.
- Append all valid observations to a binary PLY on disk, updating its fixed-width
  vertex count. This avoids repeatedly loading or rewriting the whole map.
  The live display uses a bounded random reservoir over all observations,
  normally 200,000 points, including earlier portions of the map. Point files
  and event logs grow on disk; display memory is bounded.

`--input-fps 4` uses the original extraction convention of rounding the source
FPS ratio to an integer stride. The actual retained rate can differ: the tested
29.97 FPS video uses stride 7, or approximately 4.28 FPS. `--input-fps 0` retains
every source frame. `--map-step 2` maps every second retained geometry frame;
`--map-step 1` runs SAM/mapping on each retained frame and costs more.

Without `--realtime`, the selected frames are processed in sequence as quickly
as possible, with no further frame dropping and no playback pacing. With
`--realtime`, video playback is paced after initialization; when processing is
late, the decoder advances to the latest scheduled sample. This bounds playback
lag by dropping stale samples, and the drop count is recorded. It does not make
the models faster or guarantee a particular update rate. Large gaps can also
affect geometry quality and miss short-lived hazards.

## Viewing and outputs

To control a run from the browser, the optional `live_demo.py` launcher presents
a video selector and **Start processing** / **Stop processing** buttons. It starts
the tested cache-8 configuration when requested, shows the accumulated map and
current RGB/mask overlay, and retains each run in a fresh output directory. Models
load on the first Start and stay resident for subsequent runs in the same service.

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=/tmp/small-obstacles-vlm-deps \
  /opt/conda/envs/orchestrator-visualizer/bin/python \
  plugins/small_obstacles/live_demo.py --port 8083
```

In this workspace, open `/user/jorisk/proxy/8083/` on the existing JupyterHub host.
The default is the outdoor log video, `IMG_5516.mp4`. Output runs are saved under
`data/vlm_maps/live_demo/`. The launcher handles one run at a time; stopping
finishes the current model request and retains partial output. Run it when other
model jobs are idle. The Start button applies the experimental settings above,
including their geometry tradeoff; the original streaming defaults are unchanged.

After the workspace runtime restarted, the viewer and extra dependencies were
restored under `/home/jovyan/.local/share/lingbot-runtime/`. To use those persistent
packages with the current base interpreter:

```bash
bash plugins/small_obstacles/start_live_demo.sh --port 8080
```

The restarted service uses port 8080, matching the earlier working viewer link:
`http://localhost:8080/` through the existing access setup, or
`/user/jorisk/proxy/8080/` on the JupyterHub host.

The launcher checks free GPU memory before a cold start; below 18.5 GiB it displays
an explanation and keeps Start available for retry. Loaded sessions reuse their
existing allocation and skip this cold-start threshold. This is a startup check,
not a guarantee against transient memory pressure. `--initial-run RUN` displays
an explicitly labeled saved preview until a fresh run starts. The current
runtime uses Torch 2.10 for all models; the earlier timing/geometry
benchmarks used Torch 2.8 for LingBot. Two consecutive full log-video runs now
passed on the restored runtime, including worker reuse and unchanged-map checks;
see the [persistent-session results](../../docs/vlm_persistent_session_results.md).
The fixed camera
setup sets position before look-at, because Viser shifts its target when position
changes.

The browser service owns one persistent session containing LingBot, Qwen, and
SAM workers. Each video resets LingBot's aggregator and camera-head KV history,
frame counters, prompt history, trajectory, and output map. SAM's current-image
features are released between videos; its bounded text cache can remain. Pending
VLM requests are drained before the next run. Model weights stay on the GPU until
**Unload models / free GPU** or service shutdown. Changing model settings reloads
the session; a failed run discards its workers before retry. This persistence is
within the running service, so restarting the service still requires one cold load.

The viewer's right panel is docked at 520 pixels, with **Overview**, **Prompts**
and **Settings** tabs. Overview keeps live counts, rates, elapsed/first-map time,
GPU readings, current queries, pose and side-by-side RGB/mask previews together.
Compact spacing and preview sizing fit the ordinary overview at 1280×720 and
1440×900 without scrolling. The panel can still be resized by dragging its edge.
Prompt editing and model unloading are on their own tabs.

**Follow camera** is enabled initially and uses a third-person view behind and
above the reconstructed camera. The default vertical field of view is 80°.
Settings contains **Follow distance**, **Follow height** and **Field of view (°)**;
larger distance or field of view reduces the zoom. Disable the checkbox to
inspect freely; re-enable it to resume following, even after the video completes.
This changes the viewer only. Camera axes, an orange viewing frustum and forward
direction, and a cyan trajectory remain visible. The marker updates on every geometry frame,
including frames that do not produce a semantic map update. Position and rotation
are relative to the first camera pose, and use the same world coordinates as the
original mapper. Scale is in model units, not calibrated meters. This is the
estimated camera pose; a robot-body pose needs a camera-to-body calibration.
Pose metadata is saved as `camera_pose` in `status.json`, and is also displayed by
`view_stream.py` for saved runs containing it.

### Offline frame-count mode

In **Settings**, choose **Processing mode → Offline frame count**, enter
**Frames to process** (2–1024), then press Start in Overview. For example, 100
selects 100 unique source frames evenly spaced across the entire video, including
its first and last frames. If fewer frames are available, all available frames
are selected and Overview shows the actual target count. The original default
remains approximately 4 input FPS without dropping selected frames.

Offline runs still build the map incrementally in temporal order using the same
causal LingBot reconstruction, VLM refresh schedule, SAM mask union and original
projection. Processing waits for each selected frame rather than keeping pace
with video time. **Map every** defaults to every second selected frame as before;
choose **Selected frame** to segment and project every selected geometry frame.
More input frames can increase scene coverage but do not guarantee better quality.

Overview shows completed/target geometry frames, map updates, elapsed time and
rates. Each run gets a separate output directory; config records `frame_count`
and `map_step`, status records requested and available target counts, and events
record actual source indices. The model cache and other model settings stay the
same, and loaded weights are reused. The 1024-frame viewer limit follows the
current model's configured sequence limit; it does not imply indefinite GPU
memory support.

The separate command-line runner accepts `--frame-count 100` (optionally
`--map-step 1`). This overrides `--input-fps`, and cannot be combined with
`--realtime` or the smoke-run prefix limit `--max-frames`. A premature decode
failure reports a failed run instead of claiming the requested count completed.

### Editing and testing prompts

In Overview, keep **Prompt source → VLM proposals**, then open the **Prompts** tab.
The **VLM instruction** field starts with the current built-in hazard instruction.
Edit it, enter a name in **Save as**, and press **Save prompt**. The **Saved prompt**
selector reloads named presets or restores **Built-in default**. Saving the same
name replaces that preset; earlier runs keep their original instruction.
Presets are stored as JSON files under `data/vlm_maps/live_demo/prompt_presets/`
(or the selected output root), and survive service restarts.

Press **Start processing** to test the current editor contents, including unsaved
edits. Controls are locked during processing so one run uses one instruction.
Prompt changes apply to the next run and reuse loaded models. Preserve the JSON
response contract: an object containing a `prompts` list of concrete, short object
names; use an empty list when no qualifying object is visible. Each run saves the
verbatim instruction, its SHA-256 hash and preset/draft label in `config.json`.
`events.jsonl` retains Qwen's responses and actual SAM queries, and `status.json`
shows the mode, label and query count. Compare runs on the same selected video;
asynchronous VLM refresh timing can still vary between runs.

To supply SAM queries directly, select **Prompt source → Manual SAM prompts**
in Overview, then open **Prompts**. Enter names in **SAM object names**, one per line or separated by commas, for
example `log`, `plastic bottle`, and `pond`. Up to eight distinct names are used
on every mapped frame, without VLM discovery or concept expiration. SAM computes
fresh current-frame masks and unions them before the same map projection.
Manual mode makes zero Qwen calls and does not load Qwen in a fresh session.
If Qwen is already resident from a previous VLM run, its weights remain available
for switching back; **Unload models / free GPU** clears the session.

The standalone optional runner supports the same choices:

```bash
# Add either of these to the existing run_streaming command:
--vlm-instruction-file my_prompt.txt --vlm-prompt-name "My hazard prompt"
--sam-prompts "log" "plastic bottle" "pond"
```

The UTF-8 instruction file is read once at run start. `--sam-prompts` bypasses
Qwen; its exact normalized query list is saved as `manual_sam_prompts`. Geometry
sampling, SAM thresholds, union behavior and the original mapping filters are
the same in both modes. No core LingBot or original batch files are changed.

`--serve` starts a Viser server on port 8082 and leaves it available after video
EOF. Model workers close and release GPU memory when mapping finishes. Press
Ctrl+C to stop the viewer. To view a completed run separately:

```bash
conda activate orchestrator-visualizer
python plugins/small_obstacles/view_stream.py data/vlm_maps/my_live_run --port 8082
```

Use the same access route that worked for the existing Viser viewer, changing
the port to 8082. In this remote JupyterHub workspace the route is
`/user/jorisk/proxy/8082/` under your existing JupyterHub origin. Localhost works
only on the server itself or through an established port forward. The complete
PLY also opens in the original `orchestrator/view_ply.py` or the comparison viewer.

```text
RUN/
  config.json           # model settings and exact VLM instruction
  status.json           # atomic live summary, loads, FPS, frames and drops
  events.jsonl          # proposal responses and per-frame timings/provenance
  memory_lingbot.jsonl  # sampled shared-device use and this worker's allocator
  memory_vlm.jsonl      # separate worker allocator counters
  memory_sam.jsonl      # sampler peaks are lower bounds on transient peaks
  obstacle_map.ply      # every mapped observation, appended incrementally
  display.npz           # atomic bounded sample of the whole accumulated map
  lingbot.log
  vlm.log
  sam.log
  viewer.log            # if --serve
  work/                 # a fixed set of reusable worker transfer files
  predictions/          # optional --save-frames: mapped frames only
  masks/                # optional --save-frames: current-frame masks and RGB hashes
```

The optional saved predictions/masks keep original source frame numbers. These
are selected online samples, rather than a complete batch reconstruction.
An invalid proposal is logged and counted; it retains prior, unexpired concepts
instead of inventing an empty successful response. Model worker errors fail the
run and preserve the partial map and logs.

## Timing and accuracy limits

`status.json` separates model loading, first map latency and processing time.
Geometry Hz counts all retained frames; mapping Hz counts selected semantic
frames. Worker timings include their input/output transfer files and inference.
Per-frame capture-to-map latency starts when the selected image is decoded, so
it does not include skipped frames, video-to-camera delay or model loading.
The final aggregate includes waiting for any last outstanding VLM response.

The initial benchmark records are saved in `data/vlm_maps/STREAMING_REPORT.md`;
the subsequent memory-constrained benchmarks are in
`data/vlm_maps/cache_trial/` and the
[bounded-cache report](../../docs/vlm_bounded_cache_results.md).
Three distinct videos were processed completely in the latter tests, and SAM
equivalence was checked on one additional frame from each of all 18 videos.
This is not an all-video real-time quality evaluation.
The existing prompt still has known classification limits. Sparse discovery can
miss an object between VLM calls, a later proposal does not relabel already
written history, and the new causal geometry / optional faster camera head
need quality comparison against the windowed baseline. Maps still concatenate
observations without voxel or semantic fusion. No path planning is added.
