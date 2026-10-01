"""Raw semantic inference. Importing this package loads no model libraries or weights."""

from .configuration import normalize_settings

__all__ = ["load_backend", "run_inference", "calibrate_clip"]


def load_backend(model_key, settings):
    """Load exactly the explicitly selected model; no automatic fallback."""
    resolved = normalize_settings(model_key, settings)
    if model_key == "clip_vit_b32":
        from .clip import CLIPBackend
        return CLIPBackend(model_key, resolved)
    from .qwen import QwenBackend
    return QwenBackend(model_key, resolved)


def run_inference(run_dir, model_key, settings, data_root, *, backend=None, region_ids=None):
    """Classify frozen regions and atomically preserve compatible accumulated outputs."""
    from .runner import run_inference as run
    return run(run_dir, model_key, settings, data_root, backend=backend, region_ids=region_ids)


def calibrate_clip(run_dir, settings, data_root, *, backend=None):
    """Fit CLIP rejection on eligible development annotations only."""
    from .calibration import calibrate_clip as calibrate
    return calibrate(run_dir, settings, data_root, backend=backend)
