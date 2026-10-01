# Whole-image hazard inference

`traversability_hazard_inference` produces one `hazard_prompt_v1` prediction per
requested RGB frame. Each image receives the same frozen 16-concept policy, with
additional visible concrete nouns allowed. The input contains the whole RGB
image and common policy; frame identifiers and source-specific labels do not
enter the prompt. See the frozen [protocol](hazard_protocol.md) and
[interfaces](hazard_interfaces.md).

## Isolated setup

Use the dedicated Python 3.12 unified hazard environment described in
[setup](setup.md), selecting `--cuda --gpu-profile hazard` explicitly.
`requirements/hazard_inference.txt` and the SAM3 fragment now share Torch
2.10.0/torchvision 0.25.0, NumPy 1.26.4 and Hugging Face Hub 1.31.0.
The NumPy pin satisfies SAM3's `<2` requirement; the Hub pin satisfies
[Transformers 5.18's declared requirements](https://github.com/huggingface/transformers/blob/v5.18.0/setup.py).
This proposed GPU stack has not been installed or model/GPU tested. Legacy
region inference retains its separate environment and requirements.

The pinned release sources are [PyTorch 2.10.0](https://github.com/pytorch/pytorch/releases/tag/v2.10.0),
[TorchVision 0.25.0](https://github.com/pytorch/vision/releases/tag/v0.25.0),
[Transformers 5.18.0](https://github.com/huggingface/transformers/releases/tag/v5.18.0),
[Accelerate 1.15.0](https://github.com/huggingface/accelerate/blob/v1.15.0/src/accelerate/__init__.py)
and [bitsandbytes 0.50.2](https://github.com/bitsandbytes-foundation/bitsandbytes/releases/tag/0.50.2).
bitsandbytes' [versioned installation guide](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/docs/source/installation.mdx)
describes supported CUDA builds and GPU capabilities.

Add `VLM_evaluation/src` to `PYTHONPATH`, or use the notebook's source-path
bootstrap. Package import is lazy: model libraries, weights and CUDA are accessed
only by the explicit load step. No hazard data, evaluation or segmentation
package is required for inference. Provide an existing run containing
`frames.jsonl` and `metadata/dataset.json`, plus the RGB files under the data root.
The runner reads those two input files directly. Dataset identity must declare
the task/schema, fixture status and dataset fingerprint from the data contract.

Configure external locations with `TRAVERSABILITY_DATA_ROOT`,
`TRAVERSABILITY_RUN_ROOT` and `TRAVERSABILITY_CACHE_ROOT`. The notebook also
accepts explicit root overrides; `TRAVERSABILITY_REPO_ROOT` may identify either
the parent checkout or `VLM_evaluation`. Stored artifact paths are POSIX-relative
to their applicable root, and paths escaping a root are rejected.

## Notebook usage

Open `notebooks/11_hazard_inference.ipynb`. Its checked-in outputs are cleared and
all expensive actions default to disabled. Running the default notebook inspects
configuration and required input-file availability without downloading,
initializing CUDA or creating output directories.

1. Choose the roots, prepared `RUN_NAME`, one explicit `MODEL_KEY` and CUDA
   `settings['device']`. Inspect the normalized configuration and pinned revision.
2. For an explicit download, set `DOWNLOAD_WEIGHTS=True` and run only the download
   cell. It fetches the selected pinned checkpoint into the configured cache.
   Skip this cell for pre-cached/offline weights. Model loading keeps
   `local_files_only=True`; a missing cache produces an actionable error.
3. Set `LOAD_MODEL=True` and run the separate load cell. It loads only the selected
   model on the selected CUDA device. A second load is rejected while this
   notebook still owns a backend.
4. Set `RUN_INFERENCE=True` and run the inference cell. `RETRY_ERRORS=False`
   preserves completed errors on resume; enable it explicitly to retry errors.
   Run or rerun the inference cell only with the model/settings used at load.
5. Run the close cell after inference or interruption and before changing models.
   The notebook owns its supplied backend; its close step releases that backend's
   references. Cleanup is safe to repeat and does not clear global CUDA caches.

Compare `qwen3_vl_4b` and `qwen3_5_4b` sequentially on the same frozen frame
manifest. Close the first backend, select the second key and repeat the load and
inference steps. `qwen3_5_2b` is available only by explicitly selecting that key;
it is never substituted automatically after an error.

| Model key | Checkpoint | Frozen revision |
|---|---|---|
| `qwen3_vl_4b` | `Qwen/Qwen3-VL-4B-Instruct` | `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| `qwen3_5_4b` | `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| `qwen3_5_2b` | `Qwen/Qwen3.5-2B` | `15852e8c16360a2fea060d615a32b45270f8a8fc` |

The [Qwen3-VL checkpoint configuration](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/blob/ebb281ec70b05090aa6165b016eac8ec08e71b17/config.json)
and [Qwen3.5 checkpoint configuration](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json)
identify their conditional-generation classes. The backend retains all processor
fields required by the versioned [Qwen3-VL implementation](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)
and [Qwen3.5 implementation](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py),
including multimodal token-type IDs.

## Artifacts and resume

`load_backend(model_key, settings)` returns a backend with
`predict_image(frame, data_root)` and `close()`. The convenience runner is
`run_inference(run_dir, model_key, settings, data_root, *, backend=None)`; a
caller-supplied backend remains caller-owned. Explicit checkpoint preparation is
`traversability_hazard_inference.download.download_weights(model_key, settings)`.
Injected CPU backends are allowed only in clearly marked fixture runs.

The runner saves `metadata/<model_key>.json` before atomically publishing
`predictions/<model_key>.jsonl`. The prediction file contains exactly the
contract's eight core keys: task/schema identity, frame/model identity, original
`prompts`, full `raw_response`, `status` and `error_code`. Stable frame IDs and
timestamps remain in the unchanged frame manifest. Metadata contains exact
model/settings/policy identity and fingerprints of the frame records and RGB
hashes, together with actual preprocessing, token counts, placement and
quantization audits. Per-frame diagnostics are retained in
`attempts[frame_id][*].diagnostics`, alongside each complete attempted prediction
and its digest; retry history remains auditable. `raw_response` preserves the
full decoded generation, including its terminal EOS token. Parsing excludes only
that verified terminal EOS token.

Resume requires compatible task/version, model/settings/policy and inputs.
Completed predictions and errors are preserved unless error retry is explicitly
selected. Duplicated identities, orphaned output, incompatible metadata and
legacy artifacts are rejected. Use a newly prepared run for changed frozen input;
do not relabel an existing result as a new configuration. A previously missing
RGB file becoming available changes the input fingerprint and requires a newly
prepared run.

The model must return exactly `{"prompts": ["person", "puddle"]}`. A valid empty
list has `status:ok`. Invalid JSON, extra keys, wrong types, empty phrases, phrases
over 80 characters, more than 32 phrases, generation truncation and inference
exceptions have `status:error` and a diagnostic. Exact repeats are deduplicated
after case/whitespace normalization; original phrase strings and distinct alias
variants remain available for SAM. Unmapped nouns stay valid. Frozen vague terms
are preserved as unusable diagnostics rather than assigned a hazard match.

## Real limitations

Initial limits are 1024 merged visual tokens, 2048 input-plus-output tokens and
256 new tokens, with one whole-image request, concurrency one, greedy decoding
and thinking disabled. Actual aligned resizing and processed token counts are
recorded; small hazards can become difficult to see at this visual cap. Context
and output limits produce explicit failures rather than silently shortening
inputs or responses.

The candidate quantization is NF4 language linear weights with nested statistics
and floating vision/output-head components. Transformers' pinned
[4-bit quantizer](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/quantizers/quantizer_bnb_4bit.py)
uses `llm_int8_skip_modules` for 4-bit exclusions. Nonlinear language components
remain floating; actual dtypes and initialized
[bitsandbytes quantization state](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/nn/modules.py)
are audited. Qwen3.5's recurrent calculations use floating state; its official
[tests](https://github.com/huggingface/transformers/blob/v5.18.0/tests/models/qwen3_5/test_modeling_qwen3_5.py)
identify quantized-cache generation as unsupported. The wrapper uses ordinary
cache behavior, one explicitly selected CUDA device and no CPU/disk offload.

CPU fixtures validate contracts and failure handling only. No real checkpoint
loading, quantized CUDA inference, latency, accuracy, GPU memory or combined
VLM/SAM residency was validated in this implementation session. The 6 decimal GB
VLM target is unmeasured. Objects and hazardous surfaces are identified by policy;
geometry, path relevance and physical safety require downstream work.
