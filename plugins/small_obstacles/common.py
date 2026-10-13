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


def normalize_prompts(prompts):
    if not isinstance(prompts, list):
        raise ValueError("prompts must be a JSON list of short strings")
    result = []
    for prompt in prompts:
        if not isinstance(prompt, str):
            raise ValueError("Every prompt must be a string")
        prompt = " ".join(prompt.strip().lower().split())
        if not prompt or len(prompt) > 100 or len(prompt.split()) > 8:
            raise ValueError(f"Expected a short, nonempty object phrase: {prompt!r}")
        if prompt not in result:
            result.append(prompt)
    return result


def parse_proposal(text):
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    value = json.loads(text)
    if not isinstance(value, dict) or "prompts" not in value:
        raise ValueError('VLM response must be an object containing "prompts"')
    return normalize_prompts(value["prompts"])


def load_prompts(path):
    value = json.loads(Path(path).read_text())
    return normalize_prompts(value if isinstance(value, list) else value["prompts"])


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
