"""Compatibility entry point; implementation lives in visualizer/view_stream.py."""
import sys
from pathlib import Path
import importlib
import runpy
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if __name__ == "__main__":
    runpy.run_module('visualizer.view_stream', run_name="__main__")
else:
    _implementation = importlib.import_module('visualizer.view_stream')
    globals().update({k: v for k, v in vars(_implementation).items() if not k.startswith("__")})
    sys.modules[__name__] = _implementation
