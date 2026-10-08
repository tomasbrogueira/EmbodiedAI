"""Exactly four lazy production adapters; no model imports at registration."""
from importlib import import_module
from pipeline_common.contracts import PIPELINE_IDS

def create_adapter(pipeline_id, config):
    if pipeline_id not in PIPELINE_IDS: raise ValueError(f"Unknown pipeline {pipeline_id}")
    try: module=import_module(f"pipelines.{pipeline_id}")
    except ModuleNotFoundError as error:
        if error.name==f"pipelines.{pipeline_id}": raise RuntimeError(f"Production adapter unavailable: {pipeline_id}") from error
        raise
    return module.create_adapter(config)
