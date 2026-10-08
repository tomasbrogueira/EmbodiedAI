# Shared project recordings

These are Tomás and João's project recordings, intentionally included in Git as
the common input for all four mapping pipelines. Downloaded datasets, extracted
frames, model files and generated outputs remain excluded.

| Directory | Recordings | Count |
| --- | --- | --- |
| indoor/ | IMG_5506–IMG_5508, IMG_5522–IMG_5523 | 5 |
| outdoor/ | IMG_5509–IMG_5521 | 13 |
| Total | Original MP4 files | 18 |

The original files were recovered from commit
**a13bc91bc1519c748934dd9df5dc5f0e45deb3a3**. On 9 October, all 18 recovered files
were verified byte-for-byte against their original Git blobs.
[manifest.json](manifest.json) records each file's SHA256, byte size and original
Git blob identity. The inventory contains **261,164,118 bytes** in total; the largest file
is outdoor/IMG_5514.mp4 at **49,440,145 bytes** (about 47.15 MiB). Every recording
is below GitHub's 50 MiB warning threshold and 100 MiB regular-Git file limit.
[GitHub's size documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
defines those limits.

The root ignore rules include only these named recordings, the manifest and this guide.
Place prepared images under data/prepared/ or another ignored data directory;
place overlays, renders and run exports in the ignored run/output locations.
Agree on new recordings before deliberately extending the inclusion list.

An ordinary clone of the published repository includes these MP4 files; no
separate video download, Git LFS fetch or model setup is needed. Validate the
SHA256 manifest after transferring videos to another host. Model code, weights
and caches follow the separate [external setup guide](../../docs/local_setup.md).

## Shared evaluation use

Prepare each recording once using the public preparation CLI, then share the
sequence manifest and its images. Freeze sampling, frame selection, timestamps,
geometry cache and common experiment settings. Keep recording IDs stable across
all four pipelines. See [pipeline configuration](../../docs/pipeline_configuration.md)
for preparation settings and [team workflow](../../docs/team_workflow.md)
for the development/test split and reporting protocol.

No independent reference annotations are supplied with these recordings, and no
frozen development/test list is currently supplied for all 18. Agree and record
that split before choosing words or thresholds. No training is performed.

Runs can supply reconstructions, masks, semantic maps, visual examples and
supported operational diagnostics. Accuracy requires separately provided
independent references; metric planning additionally needs calibrated scale/up
and a real robot profile. The existing two-clip pilot is historical evidence,
not an evaluation of every video.

For presentations, report good and bad examples with video/frame or timestamp,
run/config ID, and an artifact link. Preserve failures and state measurement
scope when reporting runtime.
