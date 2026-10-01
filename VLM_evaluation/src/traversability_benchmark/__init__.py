"""Portable setup and explicit whole-frame deployment profiling."""

from .environment import RootPaths, environment_report, resolve_roots
from .fixtures import FakeBackend, prepare_fixture
from .memory import NullMemoryProbe, TorchMemoryProbe
from .profiling import aggregate_samples, profile_backend
from .selection import freeze_selection

__all__ = [
    "RootPaths", "resolve_roots", "environment_report", "freeze_selection",
    "profile_backend", "aggregate_samples", "NullMemoryProbe", "TorchMemoryProbe",
    "FakeBackend", "prepare_fixture",
]
