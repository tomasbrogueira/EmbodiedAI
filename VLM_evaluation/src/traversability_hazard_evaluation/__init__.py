"""CPU-only evaluation of frozen hazard discovery and fixed-SAM artifacts."""

from .core import evaluate
from .reporting import export_report

__all__ = ["evaluate", "export_report"]
