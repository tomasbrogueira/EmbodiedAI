"""Atomic, versioned artifacts shared by independent pipeline stages."""
import hashlib
import json
from pathlib import Path
import numpy as np

SCHEMA_VERSION = 2


def json_value(value):
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def identity(value):
    return hashlib.sha256(json.dumps(json_value(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(json_value(value), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def save_array(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp.npy')
    np.save(temporary, value, allow_pickle=False)
    temporary.replace(path)


def publish(directory, kind, metadata):
    """Publish last; readers accept only a complete receipt with matching files."""
    directory = Path(directory)
    files = {p.name: digest(p) for p in sorted(directory.iterdir())
             if p.is_file() and p.name != 'receipt.json' and not '.tmp' in p.name}
    value = dict(schema_version=SCHEMA_VERSION, kind=kind, metadata=json_value(metadata), files=files)
    value['identity'] = identity(value)
    write_json(directory / 'receipt.json', value)
    return value


def verify(directory, kind=None):
    directory = Path(directory)
    value = json.loads((directory / 'receipt.json').read_text())
    saved = value.pop('identity')
    if value.get('schema_version') != SCHEMA_VERSION or identity(value) != saved:
        raise ValueError('Artifact receipt identity/schema mismatch')
    if kind is not None and value.get('kind') != kind:
        raise ValueError(f'Expected {kind} artifact')
    for name, expected in value['files'].items():
        if Path(name).name != name or digest(directory / name) != expected:
            raise ValueError(f'Artifact changed or has an invalid filename: {name}')
    return value | {'identity': saved}


def code_identity():
    """Bind first-party implementation and native source, excluding UI-only code."""
    from orchestrator.paths import ROOT
    sources = {}
    for component in ('path_planner', 'orchestrator', 'sam', 'vlm', 'lingbot'):
        for file in sorted((ROOT / component).rglob('*.py')):
            sources[str(file.relative_to(ROOT))] = digest(file)
    native = ROOT / 'path-planning/corridor_astar_feasible.cpp'
    sources[str(native.relative_to(ROOT))] = digest(native)
    return identity(sources)
