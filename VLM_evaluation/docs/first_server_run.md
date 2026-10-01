# First server run

Start with a one-image GPU smoke test, then run the frozen evaluation. The local
CPU checks do not establish model loading, GPU compatibility, latency or VRAM.

## Prepare the host

Read `/home/tomasbrogueira/SERVER_RULES.md` and the local INESC server rules.
Aquila is the gateway only. Select a compute host using current status/load
checks; inspect its CPU load, free RAM, disk bytes/inodes and GPU processes.
Use one currently free GPU by its full UUID. Recheck before every job.

Keep the checkout, environment, data, runs, caches and logs under
`/data/tmpC/tomasbrogueira`. Before preparing a remote checkout, preserve or
create its local AGENTS.md/CLAUDE.md references to the server rules above.
Run installs, tests, downloads and notebooks inside tmux/screen through
`~/.local/bin/inesc-run`. Use an approved Python 3.12 interpreter; no system
package or driver changes. Keep the requested CPU/thread/RAM limits within
the launcher rules. `--mem-gb` limits host RAM, not GPU memory.

## Environment and access

Clone/update the published GitHub revision on the compute host. From its
`VLM_evaluation/` directory, preview setup after setting `VLM_DRIVER_VERSION`
to the driver actually reported there. Select the CUDA wheel for that host;
the cu128 proposal below must pass its driver check.

Use `git clone --filter=blob:none` for a new checkout: old video blobs remain in
repository history and need not be downloaded. The current source tree excludes
videos. Download public evaluation data on the compute host through the guarded
launcher; transfer private recordings separately into its ignored data root.

```bash
~/.local/bin/inesc-run --cpus 1 --mem-gb 4 --hours 1 -- \
  python3.12 scripts/setup_env.py \
  --env-dir /data/tmpC/tomasbrogueira/envs/vlm-hazard \
  --data-root /data/tmpC/tomasbrogueira/data/vlm-hazard \
  --run-root /data/tmpC/tomasbrogueira/runs/vlm-hazard \
  --cache-root /data/tmpC/tomasbrogueira/cache/vlm-hazard \
  --notebooks --with-module hazard_data --with-module hazard_evaluation \
  --with-module hazard_visualization --cuda --gpu-profile hazard \
  --torch-index-url https://download.pytorch.org/whl/cu128 \
  --driver-version "$VLM_DRIVER_VERSION" --dry-run
```

Install the reviewed plan in this isolated environment using the same command
without `--dry-run`. Keep model/data downloads explicit. Obtain Hugging Face
access to the pinned SAM3 checkpoint and authenticate without saving tokens
in the repository or notebooks. All model caches use the configured cache
root; the default Hugging Face hub directory is `huggingface/hub` below it.
See [setup](setup.md) for pinned revisions, offline preparation and kernel use.

## Smoke test and evaluation

1. Run notebook 00 and a small CPU fixture/check before loading models.
2. On one RGB image, load each VLM separately, check its parsed hazard phrases,
   record peak memory, then close it. Use a separate smoke run, not the final
   frozen benchmark. Load SAM3 separately and segment a concrete phrase.
3. Inspect the first minutes for RAM/GPU pressure. Test actual combined
   VLM/SAM residency only after isolated stages succeed; retain headroom for
   the wider pipeline. The 6 decimal GB VLM target remains unmeasured.
4. Prepare and freeze the real 100 RELLIS + 40 COCO inputs in notebook 10.
   Run 11, 12, 13, rerun 12 with saved SAM/profiles enabled, then view 14.
   Keep both VLMs sequential; compare identical saved SAM settings and inputs.

Launch authenticated Jupyter through `scripts/start_notebook.py` inside the
guarded compute-host session. It binds to localhost; use a local SSH tunnel
to that compute host. Keep one active model kernel and close completed jobs.
Record host, session, revision, CPU/RAM request, GPU UUID and log paths.
