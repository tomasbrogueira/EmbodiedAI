"""Explicitly invoked pinned weight download; no model imports."""

from .configuration import MODEL_SPECS, normalize_settings


def download_weights(model_key, settings):
    resolved = normalize_settings(model_key, settings)
    if resolved["fixture"]:
        raise ValueError("Synthetic CPU fixtures do not download weights")
    from huggingface_hub import snapshot_download

    spec = MODEL_SPECS[model_key]
    return snapshot_download(
        repo_id=spec["checkpoint"], revision=spec["revision"],
        cache_dir=resolved["cache_dir"], local_files_only=False,
        allow_patterns=["*.json", "*.jinja", "*.safetensors", "merges.txt", "vocab.json"],
    )
