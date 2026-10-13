# Support direct compatibility-script execution without environment mutation.
if __package__ in (None, ''):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
"""Compatibility entry point; implementation lives in visualizer/view_ply.py."""
import sys
from pathlib import Path
import importlib
import runpy
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if __name__ == "__main__":
    runpy.run_module('visualizer.view_ply', run_name="__main__")
else:
    _implementation = importlib.import_module('visualizer.view_ply')
    globals().update({k: v for k, v in vars(_implementation).items() if not k.startswith("__")})
    sys.modules[__name__] = _implementation
