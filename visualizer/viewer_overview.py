"""Compact, escaped live overview and frame previews for the optional viewer."""
import base64
from html import escape
import io

import numpy as np
from PIL import Image


def formatted(value, unit='', digits=2):
    if value is None:
        return '—'
    if isinstance(value, float):
        value = f'{value:.{digits}f}'
    elif isinstance(value, int):
        value = f'{value:,}'
    return escape(str(value))+unit


def overview_html(data, directory=None, roles=(), message='', error=None):
    pose = data.get('camera_pose') or {}
    mode = 'Manual SAM' if data.get('proposal_mode') == 'manual' else 'VLM proposals'
    metrics = [
        ('Video time', formatted(data.get('source_time', data.get('last_geometry_source_time')), ' s')),
        ('Elapsed', formatted(data.get('wall_seconds'), ' s')),
        ('Geometry frames', formatted(data.get('geometry_frames'))+
            (' / '+formatted(data['target_geometry_frames']) if data.get('target_geometry_frames') else '')),
        ('Map updates', formatted(data.get('mapped_frames'))),
        ('Geometry rate', formatted(data.get('geometry_hz'), ' Hz')),
        ('Map rate', formatted(data.get('mapping_hz'), ' Hz')),
        ('Total points', formatted(data.get('points'))),
        ('Marked points', formatted(data.get('marked_points'))),
        ('GPU peak', formatted(data.get('gpu_sampled_peak_gib'), ' GiB')),
        ('GPU minimum free', formatted(data.get('gpu_min_sampled_free_gib'), ' GiB')),
        ('First map', formatted(data.get('first_map_wall_seconds'), ' s')),
        ('VLM calls', formatted(data.get('vlm_calls'))),
    ]
    cells = ''.join(f'<div class="overview-metric"><span>{label}</span><strong>{value}</strong></div>'
                    for label, value in metrics)
    def vector(key):
        value = pose.get(key)
        return '—' if value is None else ', '.join(f'{v:.3f}' for v in value)
    state = escape(str(data.get('status', 'ready')))
    loading = data.get('loading_model')
    if loading:
        state += ' · loading '+escape(loading)
    queries = ', '.join(data.get('prompts') or data.get('manual_sam_prompts') or []) or 'none'
    saved = '' if directory is None else str(directory)
    details = [
        ('Sampling', (f'Offline · requested {data["requested_frame_count"]:,} · no dropping'
            if data.get('requested_frame_count') else 'FPS sampling · '+('paced playback' if data.get('realtime') else 'no dropping'))),
        ('Models', ', '.join(roles)+' · kept loaded' if roles else 'not loaded / loading'),
        ('Prompt', mode+' · '+str(data.get('vlm_prompt_name', 'Built-in default'))),
        ('SAM queries', queries),
        ('Relative position', vector('relative_position')+' · model units'),
        ('Viewing direction', vector('view_direction_world')),
        ('Relative rotation', vector('relative_rotation_degrees')+' °'),
    ]
    rows = ''.join(f'<div class="overview-detail"><span>{label}</span><strong>{escape(value)}</strong></div>'
                   for label, value in details)
    notice = f'<div class="overview-notice">{escape(message)}</div>' if message else ''
    problem = error or data.get('error')
    failure = f'<div class="overview-error">{escape(str(problem))}</div>' if problem else ''
    return f'''<style>
    .live-overview {{font-size:12px;line-height:1.4;}}
    .overview-title {{font-size:14px;font-weight:650;margin-bottom:6px;}}
    .overview-grid {{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:2px 12px;margin:6px 0;}}
    .overview-metric {{display:flex;justify-content:space-between;gap:6px;padding:3px 0;border-bottom:1px solid #8882;}}
    .overview-metric span,.overview-detail span {{color:#777;}}
    .overview-detail {{display:grid;grid-template-columns:110px minmax(0,1fr);gap:8px;margin:3px 0;}}
    .overview-detail strong {{font-weight:500;overflow-wrap:anywhere;}}
    .overview-notice {{color:#555;font-size:11px;margin:5px 0;}}
    .overview-error {{color:#b42318;overflow-wrap:anywhere;}}
    .overview-saved {{font-size:10px;color:#777;overflow-wrap:anywhere;margin-top:5px;}}
    @media(max-height:800px) {{
      .live-overview {{font-size:11px;line-height:1.3;}}
      .overview-metric {{padding:2px 0;}}
      .overview-detail {{margin:2px 0;}}
      .live-previews img {{height:90px!important;}}
    }}
    </style><section class="live-overview" data-testid="live-overview">
    <div class="overview-title">Status: {state}</div>{notice}{failure}
    <div class="overview-grid">{cells}</div>{rows}
    <div class="overview-saved">Saved run: {escape(saved) if saved else '—'}</div></section>'''


def previews_html(rgb, mask, source_time):
    overlay = rgb.copy()
    overlay[mask] = (rgb[mask].astype(np.float32)*.45+np.array([255,0,0])*.55).astype(np.uint8)
    images = []
    for label, frame in [('Current RGB', rgb), ('Obstacle mask', overlay)]:
        image = Image.fromarray(frame)
        image.thumbnail((280, 180))
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=82)
        encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
        images.append(f'<figure style="margin:0;min-width:0"><figcaption style="font-size:11px;margin-bottom:3px">{label}</figcaption>'
            f'<img alt="{label}" src="data:image/jpeg;base64,{encoded}" '
            'style="display:block;width:100%;height:clamp(80px,16vh,150px);object-fit:contain;border-radius:5px;background:#eee"></figure>')
    return '<div class="live-previews" data-testid="live-previews" style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin-top:5px">'+''.join(images)+f'</div><div style="font-size:10px;color:#777;margin-top:3px">Mapped frame at {source_time:.2f} s · red marks obstacle-mask observations</div>'
