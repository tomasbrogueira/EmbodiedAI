"""Compatibility entry point; implementation lives in orchestrator/profile_runtime.py."""
import sys
from pathlib import Path
import importlib
import runpy
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if __name__ == "__main__":
    runpy.run_module('orchestrator.profile_runtime', run_name="__main__")
else:
    _implementation = importlib.import_module('orchestrator.profile_runtime')
    globals().update({k: v for k, v in vars(_implementation).items() if not k.startswith("__")})
    sys.modules[__name__] = _implementation
