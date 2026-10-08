"""Give subprocess contract checks the same source roots as pytest itself."""
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
source_roots = [str(ROOT / 'src'), str(ROOT / 'VLM_evaluation' / 'src')]
existing = os.environ.get('PYTHONPATH')
if existing:
    source_roots.append(existing)
os.environ['PYTHONPATH'] = os.pathsep.join(source_roots)
