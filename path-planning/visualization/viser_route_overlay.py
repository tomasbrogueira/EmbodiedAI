"""Compatibility entry point; implementation lives in visualizer/viser_route_overlay.py."""
import sys
from pathlib import Path
import importlib
import runpy
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if __name__ == "__main__":
    runpy.run_module('visualizer.viser_route_overlay', run_name="__main__")
else:
    _implementation = importlib.import_module('visualizer.viser_route_overlay')
    globals().update({k: v for k, v in vars(_implementation).items() if not k.startswith("__")})
    sys.modules[__name__] = _implementation
