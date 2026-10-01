"""Whole-image hazard prompts; importing this package loads no model libraries."""

__all__ = ["load_backend", "run_inference", "download_weights"]


def load_backend(model_key: str, settings: dict):
    """Load only the explicitly selected CUDA Qwen backend."""
    from .qwen import QwenBackend
    return QwenBackend(model_key, settings)


def run_inference(run_dir, model_key, settings, data_root, *, backend=None):
    """Run all frozen frames, preserving compatible completed rows and errors."""
    from .runner import run_inference as run
    return run(run_dir, model_key, settings, data_root, backend=backend)


def download_weights(model_key, settings):
    """Explicit opt-in download, separate from package import and model loading."""
    from .download import download_weights as download
    return download(model_key, settings)
