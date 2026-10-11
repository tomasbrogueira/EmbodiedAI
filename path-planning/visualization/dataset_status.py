"""Live read-only batch overview, including completed runs and later retries."""
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import html
import json
import time

SOURCE_ROOT = Path(__file__).resolve().parents[1]
FEASIBILITY_PENDING = 'Cost colors are not ready for this map version. The gray preview does not mean the whole map is blocked.'


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def cost_map_hash(manifest):
    try:
        return hashlib.sha256((Path(manifest['repo']) / 'path-planning/cost_map.py').read_bytes()).hexdigest()
    except (OSError, KeyError, TypeError):
        return None


def adapter_hash(name):
    try:
        return hashlib.sha256((SOURCE_ROOT / name).read_bytes()).hexdigest()
    except OSError:
        return None


def reviewed_output(output, state, report, current_hash, current_prepare_hash, current_association_hash):
    """Withhold cached feasibility results whose saved source is no longer current."""
    data = output / 'data'
    provenance = read(data / 'provenance.json')
    saved_hash = provenance.get('source_hashes', {}).get('path-planning/cost_map.py')
    cached = (data / 'cost.npy').exists() or (output / 'route.npz').exists() or bool(report.get('route_points'))
    stale = cached and (not current_hash or saved_hash != current_hash or not current_prepare_hash
                        or provenance.get('prepare_adapter_sha256') != current_prepare_hash
                        or not current_association_hash
                        or provenance.get('calibration_association_sha256') != current_association_hash)
    if stale:
        failed = state.get('status') == 'error' or state.get('stage') == 'error'
        message = FEASIBILITY_PENDING + ((' Processing error: ' + str(state.get('message') or state.get('error', ''))) if failed else '')
        state = state | dict(stage='updated_feasibility', status='error' if failed else 'pending', message=message)
        report = report | dict(status='updated_feasibility_pending', optimal=False, correspondence_verified=False,
            selected={'start': None, 'goal': None}, start=None, goal=None, supported_count=None,
            segment_label='', finite_cost_rows=None, cost_range=None, route_points=0, total_cost=None, reason_counts={})
    elif state.get('stage') != 'complete' and report.get('status') in ('ok', 'degenerate_route', 'partial'):
        # Files publish separately: a new cost receipt must not revive the previous route.
        report = report | dict(status='pending', optimal=False, selected={'start': None, 'goal': None},
            start=None, goal=None, supported_count=None, segment_label='', route_points=0, total_cost=None)
    return state, report, stale


def outcome(state, report):
    if state.get('status') == 'error' or state.get('stage') == 'error':
        return 'errors'
    if state.get('stage') == 'complete':
        verified = (report.get('optimal') is True and report.get('correspondence_verified') is True
                    and (report.get('route_points') or 0) > 0)
        if (verified and report.get('status') == 'partial' and report.get('goal_reached') is False
                and report.get('optimality_scope') == 'nearest_reachable_endpoint'):
            return 'verified partial routes'
        if verified and report.get('status') in ('ok', 'degenerate_route'):
            return 'verified routes'
        return 'completed without route'
    return 'running' if state.get('status') == 'running' else 'pending'


def endpoint_label(endpoint):
    if not endpoint:
        return '—'
    mode = endpoint.get('association_mode', 'beneath_camera').replace('_', ' ')
    distance = endpoint.get('camera_distance')
    return mode + (f' ({distance:.5g} map units from camera)' if distance is not None else '')


def render(manifest_path, viewer_url=None):
    root = manifest_path.resolve().parent
    from dataset_resources import write
    manifest = read(manifest_path)
    current_hash = cost_map_hash(manifest)
    current_prepare_hash = adapter_hash('dataset_prepare.py')
    current_association_hash = adapter_hash('dataset_path.py')
    rows, results = [], []
    outcomes = {'verified routes': 0, 'verified partial routes': 0, 'completed without route': 0, 'running': 0, 'pending': 0, 'errors': 0}
    for entry in manifest['videos']:
        out = Path(entry['output'])
        state, report = read(out / 'status.json'), read(out / 'report.json')
        state, report, stale = reviewed_output(out, state, report, current_hash, current_prepare_hash, current_association_hash)
        selected = report.get('selected') or {}
        start, goal = selected.get('start') or report.get('start'), selected.get('goal') or report.get('goal')
        result = dict(video=entry['relative_path'], prompt=entry['prompt'], frames=entry['frame_count'],
                      stage=state.get('stage', 'queued'), status=state.get('status', 'pending'),
                      message=state.get('message') or state.get('error', ''),
                      supported=report.get('supported_count'), associated=report.get('supported_count'), start=start, goal=goal,
                      segment_label=report.get('segment_label', 'supported video segment' if start and goal else ''),
                      total_rows=report.get('total_rows'), finite_rows=report.get('finite_cost_rows'),
                      route_points=report.get('route_points'), route_cost=report.get('total_cost'),
                      optimal=report.get('optimal', False), correspondence=report.get('correspondence_verified', False),
                      rejection_counts=report.get('reason_counts', {}), stale_costs=stale)
        if report.get('status') == 'missing_reference':
            result['reference'] = read(out / 'geometry.json').get('reference_report', {})
        for name in ('reconstruction.log', 'masks.log', 'dataset_prepare.log', 'dataset_path.log'):
            if (out / name).exists():
                with (out / name).open('rb') as stream:
                    stream.seek(max(0, (out / name).stat().st_size - 1600))
                    result[name] = stream.read().decode('utf-8', errors='replace')
        results.append(result)
        segment = f"{start['time_seconds']:.2f}–{goal['time_seconds']:.2f} s" if start and goal else '—'
        if result['segment_label']:
            segment = result['segment_label'] + ' · ' + segment
        associations = f"Start: {endpoint_label(start)}; end: {endpoint_label(goal)}" if start and goal else '—'
        support = f"{result['supported']} / {entry['frame_count']}" if result['supported'] is not None else '—'
        fields = [entry['relative_path'], entry['prompt'], result['stage'] + ' · ' + result['status'],
                  support, segment, associations, str(result['route_points'] or '—'), result['message']]
        rows.append('<tr>' + ''.join('<td>' + html.escape(value) + '</td>' for value in fields) + '</tr>')
        outcomes[outcome(state, report)] += 1
    counts = dict(Counter(result['stage'] for result in results))
    summary = {'updated': time.time(), 'videos': len(results), 'stages': counts, 'outcomes': outcomes,
               'memory_bytes': int(Path('/sys/fs/cgroup/memory.current').read_text()), 'results': results}
    write(root / 'review.json', summary)
    title = 'Dataset path review'
    page = '''<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="20">
<title>Dataset path review</title><style>body{font:15px system-ui;margin:32px;color:#223;max-width:1500px}
table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #dde;text-align:left}
th{background:#eef3f8}td:last-child{max-width:360px}a{color:#067}</style>'''
    page += '<h1>' + title + f'</h1><p>{len(results)} recordings · floor indoors, ground outdoors · automatic video-associated endpoints</p>'
    page += '<p>' + ' · '.join(f'<strong>{number}</strong> {name}' for name, number in outcomes.items()) + '</p>'
    if viewer_url:
        page += '<p><a href="' + html.escape(viewer_url, quote=True) + '">Open the interactive Viser map selector</a></p>'
    page += '<p>Each route uses original points, costs and connection rules. Visible-ground endpoints refer to ground observed in the associated frame; camera-to-ground offsets are not route edges. A missing route is reported explicitly. Times refer to the original video. This overview refreshes every 20 seconds.</p>'
    page += '<table><thead><tr>' + ''.join('<th>'+x+'</th>' for x in ['Video','SAM','State','Associated poses','Selected segment','Endpoint associations','Route points','Details']) + '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>'
    temporary = root / 'review.tmp.html'
    temporary.write_text(page)
    temporary.replace(root / 'review.html')
    (root / 'progress-summary.html').write_text('<meta charset="utf-8"><meta http-equiv="refresh" content="20"><pre>' + html.escape(json.dumps(summary, indent=2)) + '</pre>')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--viewer-url', help='Optional link to the running viewer')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    while True:
        render(args.manifest, args.viewer_url)
        if args.once:
            break
        time.sleep(20)
