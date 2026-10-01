"""Frozen whole-RGB hazard/SAM APIs; importing loads no model libraries."""

from .selection import freeze_selection


def load_segmenter(settings):
    from .sam3_adapter import load_segmenter as load
    return load(settings)


def run_conditions(config, *, segmenter=None):
    from .controller import run_conditions as run
    return run(config, segmenter=segmenter)


__all__ = ["load_segmenter", "freeze_selection", "run_conditions"]
