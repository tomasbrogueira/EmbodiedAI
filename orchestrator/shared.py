"""Shared file contracts; importing this module does not load any models."""

import hashlib
import json
from pathlib import Path

import numpy as np

from orchestrator.io_utils import frame_number, load_rgb


def frames(directory, step=1, limit=None):
    if step <= 0 or (limit is not None and limit <= 0):
        raise ValueError("Frame step and frame limit must be positive")
    paths = sorted(Path(directory).glob("frame_*.npz"), key=frame_number)[::step]
    if not paths:
        raise FileNotFoundError(f"No frame_*.npz predictions in {directory}")
    return paths if limit is None else paths[:limit]


def read_rgb(path):
    with np.load(path, allow_pickle=False) as data:
        return load_rgb(data["images"])


def rgb_digest(rgb):
    return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()


from vlm.prompts import normalize_prompts, parse_proposal, load_prompts

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
