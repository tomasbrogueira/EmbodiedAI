"""Artifact-only view of source, traversability, slopes, costs and certified routes."""
import json
from pathlib import Path
import numpy as np
from orchestrator.artifacts import verify, code_identity
from orchestrator.analysis import load_snapshot
from orchestrator.semantics import colors


def load_view(directory, max_points=200000):
    directory = Path(directory)
    if (directory / 'status.json').exists():
        state = json.loads((directory / 'status.json').read_text())
        outputs = state.get('outputs', {})
        if 'map' not in outputs:
            raise ValueError('This run has no 3D snapshot; use the 2D stage artifacts')
        snapshot = Path(outputs['map'])
    else:
        snapshot, outputs, state = directory, {}, {}
    receipt, data = load_snapshot(snapshot)
    rows = np.linspace(0, len(data['points']) - 1, min(max_points, len(data['points'])), dtype=np.int64)
    valid = np.isfinite(data['points'][rows]).all(axis=1)
    rows = rows[valid]
    xyz = np.asarray(data['points'][rows])
    source = np.asarray(data['source_rgb'][rows])
    modes = {'Source': source, 'Traversability': colors(source, data['support_mask'][rows], data['obstacle_mask'][rows])}
    note, route = [], np.empty((0, 3))
    from visualizer.dataset_viewer import palette
    for name, kind, field in (('steepness', 'steepness', 'slope'), ('costs', 'costs', 'cost')):
        if name not in outputs:
            continue
        stage = Path(outputs[name])
        try:
            current = verify(stage, kind)
            meta = current['metadata']
            if meta['snapshot_identity'] != receipt['identity'] or meta['code_identity'] != code_identity():
                raise ValueError('snapshot or implementation mismatch')
            if meta.get('status') != 'ok':
                note.append(f'{name}: {meta.get("status")}')
                continue
            values = np.load(stage / f'{field}.npy', mmap_mode='r')
            if values.shape != (len(data['points']),):
                raise ValueError('array alignment mismatch')
            if kind == 'steepness':
                modes['Steepness'] = palette(values[rows], (0., np.pi / 2))
            else:
                finite = values[np.isfinite(values)]
                limits = np.percentile(finite, [5, 95]) if len(finite) else (0., 1.)
                modes['Cost map'] = palette(values[rows], limits)
                if meta['slope_policy'] == 'disabled':
                    note.append('Slope checks disabled')
        except (OSError, ValueError, KeyError) as exc:
            note.append(f'{name} withheld: {exc}')
    report = {}
    if 'route' in outputs:
        try:
            stage = Path(outputs['route'])
            current = verify(stage, 'route')
            report = current['metadata']
            costs_receipt = verify(outputs['costs'], 'costs')
            if report['snapshot_identity'] != receipt['identity'] or report['cost_identity'] != costs_receipt['identity'] or report['code_identity'] != code_identity():
                raise ValueError('route snapshot/cost/implementation mismatch')
            if report.get('optimal') and report['status'] in ('ok', 'partial'):
                with np.load(stage / 'route.npz', allow_pickle=False) as saved:
                    ids, positions = saved['ids'], saved['positions']
                    values = np.load(Path(outputs['costs']) / 'cost.npy', mmap_mode='r')
                    if ids.ndim != 1 or ids.dtype.kind not in 'iu' or not len(ids) or np.any(ids < 0) or np.any(ids >= len(data['points'])):
                        raise ValueError('invalid route IDs')
                    if not np.array_equal(positions, data['points'][ids]) or not np.isfinite(values[ids]).all():
                        raise ValueError('invalid route correspondence/availability')
                    route = positions
        except (OSError, ValueError, KeyError) as exc:
            note.append(f'route withheld: {exc}')
    return dict(points=xyz, modes=modes, route=route, cameras=np.asarray(data['cameras']),
                report=report, notes=note, state=state, snapshot_identity=receipt['identity'])


def main(argv=None):
    import argparse
    import time
    import viser
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8094)
    parser.add_argument('--max-points', type=int, default=200000)
    args = parser.parse_args(argv)
    view = load_view(args.directory, args.max_points)
    server = viser.ViserServer(host=args.host, port=args.port)
    options = tuple(view['modes'])
    selector = server.gui.add_dropdown('Colour', options=options,
                                        initial_value='Cost map' if 'Cost map' in options else 'Traversability')
    server.gui.add_markdown('Green: positive traversability evidence. Red: non-traversable evidence; vetoes overlaps. '
                            'Gray cost: unavailable. Steepness spans 0–90° relative to ground. '
                            'Visibility controls do not change feature computation.\n\n' + '\n\n'.join(view['notes']))
    cloud = server.scene.add_point_cloud('/map', points=view['points'], colors=view['modes'][selector.value], point_size=.02)
    @selector.on_update
    def change(_):
        cloud.colors = view['modes'][selector.value]
    for name, points, color in (('Camera trajectory', view['cameras'], (0, 220, 255)),
                                ('Route', view['route'], (240, 0, 220))):
        if len(points) > 1:
            handle = server.scene.add_line_segments('/' + name.replace(' ', '_'),
                points=np.stack((points[:-1], points[1:]), axis=1).astype(np.float32), colors=color, line_width=3.)
            toggle = server.gui.add_checkbox(name, initial_value=True)
            toggle.on_update(lambda event, h=handle: setattr(h, 'visible', event.target.value))
    if len(view['route']) and view['report'].get('status') == 'partial':
        server.scene.add_point_cloud('/partial_endpoint', points=view['route'][-1:], colors=(255, 150, 0), point_size=.1)
    server.gui.add_markdown(json.dumps(view['report'], indent=2))
    while True:
        time.sleep(.5)
