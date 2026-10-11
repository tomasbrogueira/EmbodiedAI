# Offline frame-count mode and third-person following

Validated on 2026-10-11 on `vlm_improvements`. The optional live viewer on port
8080 now supports fixed-count offline sampling. The original pipeline and model
implementations are unchanged. Sampling mode, count and mapping frequency are
snapshotted at Start; these controls are disabled during a run.

## Use

Open the viewer, select Settings → Processing mode → Offline frame count,
enter Frames to process, and Start from Overview. The viewer accepts 2–1024
frames, capped at the video's available source frames. Samples are unique and
uniform across the full sequence, including the first and last frame. They are
decoded in temporal order and processed without pacing or dropping. The existing
causal geometry, asynchronous VLM refresh, fresh SAM mask union and original
projection still build the map incrementally. Models remain loaded between runs.
Default sampling remains approximately 4 FPS; default mapping remains every
second selected frame. Map every → Selected frame enables SAM and projection on
each geometry frame. CLI equivalents are `--frame-count 100` and optionally
`--map-step 1`. Fixed count rejects `--realtime` and `--max-frames` combinations.

The Overview records actual/target geometry counts, map updates, elapsed time,
processing rates and prompt information. Each run gets its own output directory,
config, source-index events, masks, predictions, and full-resolution PLY map.

## Measured full-video run

Both runs used `outdoor/IMG_5516.mp4`, Qwen → multiple SAM prompts → union mask →
unchanged map projection, four camera iterations, cache 8 and mapping step 2.
The earlier standard run is a reference, not a controlled benchmark of geometry
quality. Timing includes asynchronous proposal work; wall time also includes
model loading and final cleanup.

| Measurement | Standard sampling reference | Offline frame count |
| --- | ---: | ---: |
| Geometry frames | 50 | 100 |
| Map updates | 25 | 50 |
| Processing seconds | 18.43 | 32.88 |
| Total seconds including loading | 51.94 | 69.07 |
| Dropped selected frames | 0 | 0 |
| VLM calls | 5 | 6 |
| Point observations | 3,400,657 | 6,816,638 |
| Obstacle-marked observations | 124,895 | 247,078 |

The new run selected exactly 100 unique source indices spanning the full video.
The existing mapper accumulates observations without voxel fusion, so more point
observations alone do not demonstrate better reconstruction or obstacle accuracy.
Denser sampling changes geometry inputs while keeping model/projection settings
fixed. The 1024-frame sequence limit does not guarantee GPU capacity for every
input; memory usage also depends on image dimensions and temporal context.

- [100-frame map](../data/vlm_maps/live_demo/IMG_5516_c0c48a4bcee1/obstacle_map.ply)
- [Earlier 50-frame map](../data/vlm_maps/live_demo/IMG_5516_81bdbcd150ef/obstacle_map.ply)
- [Offline/browser verification](../data/vlm_maps/live_demo/offline_frame_count_verification.json)
- [Live third-person map screenshot](../data/vlm_maps/live_demo/offline_third_person_live.png)
- [Settings screenshot](../data/vlm_maps/live_demo/offline_frame_count_settings.png)

## Camera and validation

Follow camera now uses a viewpoint behind and above the estimated camera, with
an 80° default vertical field of view. The camera marker and trajectory remain
visible. Settings provides distance, height and field-of-view controls. Controls
apply even at the final frame; pausing allows free navigation, and resuming
restores the latest third-person pose. This changes only the viewport.

All 44 plugin tests passed. New decoder checks cover exact full-video sampling,
long processing delays without dropping, requests above available counts,
unchanged standard/prefix sampling, premature EOF and incompatible modes.
Browser checks verified all camera controls, free navigation/resume, a completed
100-frame VLM run, exact source indices, 96 distinct live
viewpoints and the final third-person transform. The ordinary Overview fit at
1280×720 without scrolling, and no JavaScript errors were reported. LingBot,
Qwen and SAM remain resident on the updated port-8080 service.
