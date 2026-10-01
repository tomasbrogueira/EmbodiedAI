"""Resolve configuration against the component, never the notebook cwd."""

from collections.abc import Mapping
import os
from pathlib import Path

from .storage import data_path, read_json, safe_name

TASK_ID = "hazard_prompt_v1"
SCHEMA_VERSION = 1
SEQUENCES = [f"{i:05d}" for i in range(5)]


def component_root():
    return Path(__file__).resolve().parents[2]


def resolve_config(config):
    if isinstance(config, (str, os.PathLike)):
        path = Path(config).expanduser()
        config = read_json(path if path.is_absolute() else component_root() / path)
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping or JSON configuration path")
    cfg = dict(config)
    if cfg.get("fixture_profile", "small") not in {"small", "protocol"}:
        raise ValueError("fixture_profile must be small or protocol")
    if cfg.get("task_id", TASK_ID) != TASK_ID or type(cfg.get("schema_version", 1)) is not int or cfg.get("schema_version", 1) != 1:
        raise ValueError("Configuration task/schema mismatch")
    if type(cfg.get("seed", 0)) is not int or cfg.get("seed", 0) != 0:
        raise ValueError("Frozen hazard sampling requires seed 0")
    repo = Path(cfg.get("repo_root") or os.environ.get("TRAVERSABILITY_REPO_ROOT") or component_root()).expanduser().resolve()
    if not (repo / "docs/hazard_interfaces.md").is_file() and (repo / "VLM_evaluation/docs/hazard_interfaces.md").is_file():
        repo = repo / "VLM_evaluation"
    if not (repo / "docs/hazard_interfaces.md").is_file():
        raise ValueError("repo_root must be the VLM_evaluation component")
    cfg.update(task_id=TASK_ID, schema_version=1, seed=0, repo_root=repo,
               run_name=safe_name(cfg.get("run_name", TASK_ID)))
    for name, default in (("data_root", "data"), ("run_root", "runs"), ("cache_root", ".cache")):
        path = Path(cfg.get(name) or os.environ.get("TRAVERSABILITY_" + name.upper()) or default).expanduser()
        cfg[name] = (path if path.is_absolute() else repo / path).resolve()
    policy = Path(cfg.get("policy_path") or "configs/hazards/policy.json").expanduser()
    cfg["policy_path"] = (policy if policy.is_absolute() else repo / policy).resolve()
    rellis = dict(cfg.get("rellis") or {})
    if rellis.get("sequences", SEQUENCES) != SEQUENCES or rellis.get("frames_per_sequence", 20) != 20 or type(rellis.get("frames_per_sequence", 20)) is not int:
        raise ValueError("Frozen RELLIS recipe requires sequences 00000–00004 and 20 frames each")
    source = rellis.get("source_dir")
    if source:
        source = Path(source).expanduser()
        rellis["source_dir"] = (source if source.is_absolute() else repo / source).resolve()
    else:
        rellis["source_dir"] = None
    cfg["rellis"] = rellis
    coco = dict(cfg.get("coco") or {})
    for key, expected in (("positive_counts", {"development": 6, "test": 26}),
                          ("negative_counts", {"development": 2, "test": 6}),
                          ("small_instance_pixels", 1024)):
        if coco.get(key, expected) != expected:
            raise ValueError(f"Frozen COCO sampling changed: {key}")
    for name, default in (("annotation_path", "raw/coco/annotations/instances_val2017.json"),
                          ("image_dir", "raw/coco/val2017")):
        value = coco.get(name) or default
        path = Path(value).expanduser()
        if path.is_absolute():
            coco[name] = path.resolve()
        else:
            coco[name] = data_path(cfg["data_root"], path.as_posix(), must_exist=False)
    cfg["coco"] = coco
    return cfg
