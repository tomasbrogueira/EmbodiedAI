# Full VLM/SAM/LingBot map run

Completed all 18 videos: 7,796 reconstructed frames and 3,903 shared VLM/SAM/mapping frames. Batch wall time: 93.9 minutes.

The new runner feeds merged masks into the unchanged original mapper. Original reconstruction and mapping scripts were not edited. Frame step 2, pixel step 1, confidence >2.5, depth <25, patch size 14 and red marking color were preserved. Reconstruction used windowed mode, window size 32, keyframe interval 2, two scale frames, eight overlap keyframes and SDPA. Rendering was disabled. Each scene was reconstructed afresh into this run folder; existing outputs were preserved.

| Video | Input frames | Shared frames | LingBot (s) | VLM (s) | SAM (s) | Mapping (s) | Total (min) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [IMG_5506](indoor/IMG_5506/obstacle_map.ply) | 320 | 160 | 114.9 | 99.6 | 2.1 | 9.4 | 3.81 |
| [IMG_5507](indoor/IMG_5507/obstacle_map.ply) | 340 | 170 | 126.6 | 93.4 | 2.2 | 9.7 | 3.91 |
| [IMG_5508](indoor/IMG_5508/obstacle_map.ply) | 499 | 250 | 176.2 | 129.0 | 3.0 | 13.5 | 5.42 |
| [IMG_5522](indoor/IMG_5522/obstacle_map.ply) | 672 | 336 | 208.1 | 166.5 | 88.8 | 16.2 | 8.08 |
| [IMG_5523](indoor/IMG_5523/obstacle_map.ply) | 655 | 328 | 197.6 | 174.1 | 3.9 | 16.5 | 6.62 |
| [IMG_5509](outdoor/IMG_5509/obstacle_map.ply) | 321 | 161 | 117.6 | 98.0 | 2.0 | 7.9 | 3.80 |
| [IMG_5510](outdoor/IMG_5510/obstacle_map.ply) | 339 | 170 | 125.8 | 112.0 | 53.3 | 9.3 | 5.05 |
| [IMG_5511](outdoor/IMG_5511/obstacle_map.ply) | 620 | 310 | 218.5 | 180.5 | 3.8 | 13.8 | 7.02 |
| [IMG_5512](outdoor/IMG_5512/obstacle_map.ply) | 489 | 245 | 174.8 | 150.6 | 48.1 | 12.2 | 6.49 |
| [IMG_5513](outdoor/IMG_5513/obstacle_map.ply) | 323 | 162 | 117.5 | 95.8 | 2.4 | 8.6 | 3.78 |
| [IMG_5514](outdoor/IMG_5514/obstacle_map.ply) | 550 | 275 | 197.1 | 163.7 | 3.3 | 13.5 | 6.36 |
| [IMG_5515](outdoor/IMG_5515/obstacle_map.ply) | 197 | 99 | 77.2 | 77.7 | 35.6 | 5.6 | 3.30 |
| [IMG_5516](outdoor/IMG_5516/obstacle_map.ply) | 349 | 175 | 129.5 | 115.9 | 45.3 | 8.6 | 5.03 |
| [IMG_5517](outdoor/IMG_5517/obstacle_map.ply) | 223 | 112 | 86.2 | 85.5 | 39.5 | 6.2 | 3.65 |
| [IMG_5518](outdoor/IMG_5518/obstacle_map.ply) | 357 | 179 | 113.6 | 112.5 | 46.6 | 8.9 | 4.74 |
| [IMG_5519](outdoor/IMG_5519/obstacle_map.ply) | 382 | 191 | 118.7 | 113.6 | 2.4 | 9.7 | 4.12 |
| [IMG_5520](outdoor/IMG_5520/obstacle_map.ply) | 862 | 431 | 266.2 | 235.2 | 5.1 | 20.0 | 8.88 |
| [IMG_5521](outdoor/IMG_5521/obstacle_map.ply) | 298 | 149 | 97.1 | 89.4 | 35.2 | 7.5 | 3.86 |

All stage times are process wall times, including imports, model loading, inference and IO. Total also includes mask adaptation and setup. Models ran sequentially and were loaded per video. An empty prompt list bypasses SAM model loading. Thus SAM rates on empty scenes measure file handling, not segmentation inference. The separately reported LingBot inference FPS is available in the timing CSV/JSON and is computational throughput, not input-video Hz or a live map update rate.

VLM discovers one global prompt list per video and SAM applies that list to every shared frame. Different descriptions such as log/fallen log/detached log are separate queries, then combined with a boolean union. These exports concatenate frame observations without semantic fusion; marked observations are not unique objects or unique physical points. PLY contains XYZ/RGB, while binary masks retain the explicit semantic observations.

## Verification and qualitative limits

All 18 pipelines completed. Exact frame-ID agreement was checked between the original mapper selection, VLM observations and SAM manifest. Every merged mask and mapper-compatible file exists. Every PLY payload length matches its XYZ/RGB vertex declaration, and vertex counts match the original mapper logs. Ten plugin tests passed, including a direct original-mapper projection of the adapted masks.

Positive checks include the floor bag in IMG_5522 and the logs in IMG_5515/5516/5517. In IMG_5515 frame 60, the log mask has 4,298 pixels; 1,852 survive the original geometry filters, matching the mapper log. [Preview](outdoor/IMG_5515/preview_000060.jpg).

The dense evaluation revealed remaining proposal/mask issues: fabric in IMG_5522 also selects chair upholstery and clothing; the stone-block class can select fixed blocks; red object in IMG_5512 and IMG_5518 violates the intended specific-object rule and has unverified identity. These outputs were retained, with the instruction fixed throughout the timed run. A successful export establishes pipeline execution, not hazard-detection accuracy. There are no ground-truth annotations, and water/open-hole recall remains unvalidated.

[IMG_5522 example](indoor/IMG_5522/preview_000632.jpg) · [IMG_5518 example](outdoor/IMG_5518/preview_000238.jpg)

## Artifacts

[Timing dashboard](index.html) · [Timing CSV](timing_summary.csv) · [Batch JSON](batch_timings.json) · [Verified results](verified_results.json)

Each scene folder contains raw predictions, exact frame selection, VLM responses/prompts, merged masks, mapper-compatible masks, logs, timings and obstacle_map.ply. [Runner usage](../../../plugins/small_obstacles/EXISTING_PIPELINE.md).
