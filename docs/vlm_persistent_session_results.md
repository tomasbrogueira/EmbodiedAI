# Persistent model session and live camera pose

Validated on 2026-10-11, on the `vlm_improvements` branch. The optional browser
service owns persistent LingBot, Qwen and SAM workers across video runs. The
original model implementations, batch runner, and map projection are unchanged.
The service was restarted on port 8080 and left running with all three models loaded.

## Consecutive full-video test

Two consecutive runs of `data/videos/outdoor/IMG_5516.mp4` used the same GPU worker
processes: LingBot 46821, Qwen 46943, SAM 47043. Both processed 50 geometry frames
and 25 semantic map updates with no dropped selected frames. Settings were the
browser demo's experimental cache window 8, four camera refinements, target input
4 FPS (actual stride 7 at 29.97 FPS), map step 2, SAM text caching, and budget
allocator policy. The original streaming cache default remains 64.

| Measurement | Cold run | Warm run |
| --- | ---: | ---: |
| First map update from Start | 38.21 s | 1.50 s |
| Total recorded wall time | 53.15 s | 16.96 s |
| Processing time after initialization | 18.74 s | 16.58 s |
| Geometry update rate | 2.67 Hz | 3.02 Hz |
| Semantic map update rate | 1.33 Hz | 1.51 Hz |
| LingBot model initialization | 12.47 s | 0 s |
| Qwen model initialization | 9.42 s | 0 s |
| SAM model initialization | 9.23 s | 0 s |

The principal benefit is avoiding initialization between runs. These two
measurements do not establish a sustained inference speedup or 30 FPS operation.
First-map time includes worker startup, sequence setup, initial geometry, VLM
proposal and SAM/map processing. Recorded wall time excludes final sequence
cleanup; readiness for another run was checked separately through the browser.

All 25 saved prediction files were bit-identical between the cold and warm runs,
including camera parameters, depth, confidence and RGB. Replaying the warm run's
saved union masks through the unchanged original mapper produced an identical
PLY vertex payload: 3,400,657 points, including 124,895 obstacle-marked points.
The RGB/mask hashes matched on every mapped frame.

At each run boundary, LingBot's sequence cache reported zero bytes, zero patch
keyframes and zero older special tokens. SAM current-image state was released;
model weights and its bounded text cache remained. After cleanup the shared GPU
used approximately 15.12 GiB with 4.50 GiB free. Each video receives fresh prompt
state, camera trajectory and output files. An outstanding VLM request is drained
before reuse. A failed run discards workers, and changed model settings reload
the session.

The browser's **Unload models / free GPU** button closes the workers when idle.
**Stop processing** requests cancellation at a frame boundary after the current
model request finishes and retains partial output. Service shutdown unloads the
models. Restarting the service requires another first cold load. The standalone
command-line runner still closes its workers after its single video completes.

## Camera position and orientation

The live map contains camera axes, an orange viewing frustum and forward line,
and a cyan trajectory. The pose updates on every geometry frame, including the
frames between semantic map updates. The panel reports translation from the
first camera and the viewing direction in map coordinates. Saved pose metadata
also includes relative rotation as a rotation vector in degrees. Camera
transforms use the same inverse extrinsic
operation as the original official depth-unprojection helper. A principal-pixel
regression test checks that the camera origin and forward direction agree with
that helper.

Coordinates are in the reconstruction's model-relative scale. They are not
calibrated meters or a robot-body pose. A camera-to-body transform would be
needed to show a robot chassis accurately. The final trajectory contains all 50
geometry observations and resets for a new run. `status.json` stores the latest
pose and trajectory; the separate saved-run viewer also renders this metadata.

## Validation and artifacts

All 28 plugin tests passed, covering session reuse, active-run protection,
settings changes, failure cleanup, both LingBot history caches, fresh-process
geometry imports, pose/projection agreement, mask provenance and mapping.
The real browser test used the authenticated JupyterHub HTTP/WebSocket proxy,
reported no JavaScript errors, and verified the position controls were visible.
Its largest received WebSocket message was 3,158,475 bytes.

The restored runtime uses Torch 2.10.0 with CUDA 12.8 on the H100 MIG partition;
earlier performance reports used Torch 2.8 for LingBot. This validation covers
two full log-video runs on the restored runtime, rather than a new all-video
accuracy evaluation.

- [Browser checks and full statuses](../data/vlm_maps/live_demo/persistent_pose_verification.json)
- [Prediction and original-mapper checks](../data/vlm_maps/live_demo/persistent_geometry_verification.json)
- [Viewer screenshot](../data/vlm_maps/live_demo/persistent_pose_viewer.png)
- Cold output: `data/vlm_maps/live_demo/IMG_5516_671d1626e66e/`
- Warm output: `data/vlm_maps/live_demo/IMG_5516_718b53a59f91/`

Open `http://localhost:8080/` through the existing viewer access setup, or use
`/user/jorisk/proxy/8080/` on the existing JupyterHub host. Select a video and
press **Start processing**. Relaunch with
`bash plugins/small_obstacles/start_live_demo.sh --port 8080` if needed.
