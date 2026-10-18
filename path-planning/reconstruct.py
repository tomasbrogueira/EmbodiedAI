"""Compatibility entry point; implementation lives in lingbot/offline.py."""
import sys
from pathlib import Path
import importlib
import runpy
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if __name__ == "__main__":
    runpy.run_module('lingbot.offline', run_name="__main__")
else:
    _implementation = importlib.import_module('lingbot.offline')
    globals().update({k: v for k, v in vars(_implementation).items() if not k.startswith("__")})
    sys.modules[__name__] = _implementation
