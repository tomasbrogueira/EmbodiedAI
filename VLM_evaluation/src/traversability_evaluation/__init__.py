"""Standalone CPU classification evaluation of frozen JSONL artifacts."""

from .core import evaluate
from .reporting import comparison_html, export_report, render_review_gallery

__all__ = ["evaluate", "export_report", "render_review_gallery", "comparison_html"]
