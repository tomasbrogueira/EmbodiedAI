"""Prompt parsing and semantic class validation without model imports."""
import json
from pathlib import Path

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


def parse_proposal(text, target=None):
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    value = json.loads(text)
    if not isinstance(value, dict) or "prompts" not in value:
        raise ValueError('VLM response must be an object containing "prompts"')
    if target is not None:
        validate_target(target)
        if "target" in value and value["target"] != target:
            raise ValueError("VLM response target disagrees with the configured target")
    return normalize_prompts(value["prompts"])


def load_prompts(path):
    value = json.loads(Path(path).read_text())
    return normalize_prompts(value if isinstance(value, list) else value["prompts"])



TARGETS = ("traversable", "non_traversable")

def validate_target(target):
    if target not in TARGETS:
        raise ValueError("target must be traversable or non_traversable")
    return target
