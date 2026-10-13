Improved prompt: paired VLM evaluation and SAM review

The deployed instruction is in `plugins/small_obstacles/propose.py`; this run's exact substituted instruction is saved in [instruction.txt](instruction.txt). The core mapper and SAM implementation were unchanged.

The revised prompt requires visible loose/detached/displaced objects on potentially traversable ground, or identifiable water/open holes. It explicitly excludes normal grass and maintained gravel, installed path slabs/steps, intact paving stones, and closed flush covers. It rejects vague debris labels and favors short conventional object names for SAM.

All 18 supplied videos were rerun with Qwen/Qwen3-VL-4B-Instruct at the original 2-second sampling interval plus near-end samples: 150 frames, including 46 indoor frames and 104 outdoor frames. Sampling, model, and preprocessing were kept the same. These videos were also used to tune the prompt; this is a paired development comparison, not a held-out benchmark. There are no annotated ground-truth masks or hazard labels, so no precision, recall, or IoU is claimed.

| Output statistic | Previous prompt | Improved prompt |
| --- | ---: | ---: |
| Videos | 18 | 18 |
| Sampled frames | 150 | 150 |
| Invalid JSON responses | 0 | 0 |
| Frames with proposals | 16 | 13 |
| Prompt applications | 23 | 13 |
| Previously detected log frames still proposed | 9 | 9 |

SAM3 was run at confidence threshold 0.5 for every new nonempty proposal frame. All 13 applications produce a nonempty mask. Saved per-prompt masks were checked to union exactly to the saved merged masks; all linked preview and mask files were verified. SAM consumed the saved evaluation JPEGs, so these are evaluation masks, not newly projected map outputs.

Visual findings

- All nine log frames in IMG_5515/5516/5517 retain useful masks. The labels vary between `log`, `fallen log`, and `detached log`; the masks visually follow the logs rather than the surrounding path.
- IMG_5515 at 4 seconds now proposes only `log`. The false mask on installed path steps from `broken concrete slab` is gone.
- IMG_5517 at 2 seconds now proposes only `fallen log`. The broad maintained-path mask from `gravel debris` is gone.
- IMG_5512 at 16 seconds now returns an empty list. The previous broad vegetation mask from `overgrown grass` is gone.
- IMG_5519 at 2 seconds now returns an empty list. The previous closed-cover mask is gone.
- IMG_5518 at 4 seconds no longer proposes a planting-bed `soil mound`.
- IMG_5510 at 6, 8, 10, and 11.11 seconds consistently proposes `stone block`. SAM now finds the displaced block in all four frames, whereas `tilted stone block` produced empty masks in the first two frames previously.

Remaining limits and tradeoffs

The simpler `stone block` class also selects upright installed blocks in IMG_5510 at 6 and 8 seconds. Text-only class segmentation cannot enforce the VLM's loose/displaced distinction for each instance. Some nearby stone fragments are included, but the new VLM no longer requests the separate `small stones`/`small rocks` concepts. Individual-fragment recall has not been measured. This matters for the plugin's small-object focus, and the current result should not be described as solving all false positives or misses.

All indoor frames still return empty proposals. The earlier possible floor bag/fabric in IMG_5522 near the end is unresolved. Partial logs at later timestamps remain omitted. There are no useful positive water or open-hole examples in the reviewed sampled frames, so those categories remain unvalidated.

An additional ten-frame trial added an explicit instruction to find smaller objects beside larger ones. It recovered a `rock` proposal at IMG_5510 at 8 seconds, but also reintroduced a planting-bed `soil pile` proposal at IMG_5518 at 4 seconds. That candidate was not deployed; the trial responses and exact instruction are saved in [small_object_trial.json](small_object_trial.json).

Artifacts

- [VLM results for all 18 videos](index.html)
- [All new SAM previews in one folder](../improved_proposal_masks/index.html)
- [Combined preview image](../improved_proposal_masks/all_proposals.jpg)
- [Machine-readable paired statistics](metrics.json)
- [Previous masks and review](../proposal_masks/analysis.md)

The revised prompt reduces the observed false terrain/infrastructure proposals while preserving the nine earlier log detections. The remaining fixed-block instance leakage would need instance grounding or filtering to resolve reliably; changing the prompt alone has not established that capability.
