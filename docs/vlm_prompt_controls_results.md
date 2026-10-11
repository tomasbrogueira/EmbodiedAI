# Viewer prompt controls and direct SAM mode

Implemented and validated on 2026-10-11 in the optional `vlm_improvements`
plugin. The live service is available on port 8080. The original batch pipeline,
Qwen proposer, SAM implementation and original mapper remain unchanged.

## Controls

- **Prompt source → VLM proposals:** edit **VLM instruction**, enter **Save as**,
  and click **Save prompt**. **Saved prompt** reloads presets or the built-in
  default. Unsaved edits can also be tested with Start and are labeled as a draft.
- **Prompt source → Manual SAM prompts:** enter concrete names in **SAM object
  names**, separated by newlines or commas. Up to eight distinct names are used
  on every map update. Qwen discovery and proposal expiration are bypassed.
- Changes apply to the next run. Prompt controls are disabled during processing,
  and each run has a fresh output directory.

Named VLM presets live in `data/vlm_maps/live_demo/prompt_presets/` and survive
service restarts. Saving an existing name updates that preset. Every run records
its exact instruction, preset/draft label, SHA-256 hash, mode and manual query
list in `config.json`, so a later preset edit cannot alter an earlier experiment.
Raw VLM replies and per-frame SAM queries remain in `events.jsonl`.

Manual mode starts only LingBot and SAM in a fresh session. Switching to VLM mode
adds Qwen while reusing the other workers. Switching back to manual mode makes
zero Qwen calls; previously loaded weights remain resident for future reuse.
The Unload button releases the whole session. The same current-frame SAM masks
are unioned and passed through the same original projection in both modes.

The optional standalone runner accepts `--vlm-instruction-file FILE` with an
optional `--vlm-prompt-name NAME`, or `--sam-prompts "log" "plastic bottle"`.
The original built-in instruction remains the default.

## Full-video browser checks

Three consecutive runs of the same outdoor log clip, `IMG_5516.mp4`, completed
with 50 geometry frames, 25 semantic map updates, and no dropped selected frames
per run. These are three runs of one video, not three different scenes.

| Run | Queries | Worker behavior |
| --- | --- | --- |
| Manual SAM | `log`, `plastic bottle` on all 25 mapped frames; zero Qwen calls | Loaded LingBot PID 60764 and SAM PID 60872 only |
| Saved “Ground hazards” | Five Qwen queries using the built-in hazard instruction | Reused LingBot/SAM, added Qwen PID 61379 |
| Saved “Loose logs only” | Five Qwen queries using the custom log-only instruction | Reused all three workers; all load times zero |

The browser reloaded both presets after refreshing the page and checked their
exact contents. The run snapshots matched the editor text and corresponding
hashes. The final warm run reached its first map update in 1.35 seconds and
recorded 16.65 seconds total wall time. These are measurements from this test,
not a sustained throughput guarantee. No browser JavaScript errors were reported.

All 25 saved prediction files were bit-identical across all three runs. Replaying
the manual and custom-VLM union masks through the unchanged original mapper
produced bit-identical PLY vertex payloads and matching RGB/mask hashes. Each map
contained 3,400,657 points and 124,895 obstacle-marked points. The additional
plastic-bottle query added no marked points in this clip; differing instructions
and queries can legitimately produce the same final mask for a scene.

All 35 plugin tests passed. New checks cover persisted presets and immutable run
snapshots, instruction-file loading, invalid instructions and queries, safe preset
filenames, mode switching without unnecessary model reloads, and the direct SAM
controller path. The controller test checks that manual queries remain active
beyond the ordinary VLM prompt TTL and that no Qwen worker is constructed.
The real browser test also verifies the Save callback completes: the viewer uses
a reentrant lock because Viser synchronously invokes load callbacks when Save
changes the selected preset.

## Artifacts

- [Browser checks and complete run statuses](../data/vlm_maps/live_demo/prompt_modes_verification.json)
- [Geometry and original-mapper checks](../data/vlm_maps/live_demo/prompt_modes_geometry_verification.json)
- [Viewer with prompt editor](../data/vlm_maps/live_demo/prompt_modes_viewer.png)
- Manual output: `data/vlm_maps/live_demo/IMG_5516_8e0d6727ed68/`
- Built-in preset output: `data/vlm_maps/live_demo/IMG_5516_03f4271f2856/`
- Log-only preset output: `data/vlm_maps/live_demo/IMG_5516_ced2d966f7f8/`

See [viewer usage](../plugins/small_obstacles/STREAMING.md#editing-and-testing-prompts)
for the controls and response format. Preserve JSON with a `prompts` list of short
concrete object names when editing a VLM instruction.
