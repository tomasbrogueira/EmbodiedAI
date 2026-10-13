"""Generate a CSV and HTML timing summary for an existing-pipeline run."""
import argparse
import csv
from datetime import datetime, timezone
import html
import io
import json
from pathlib import Path


def report(root):
    rows = []
    for path in sorted(root.glob('*/*/timings.json')):
        data = json.loads(path.read_text())
        stages = {s['name']:s for s in data['stages']}
        row = dict(scene=path.parent.relative_to(root).as_posix(),status=data['status'],
                   frames=data.get('prediction_frame_count',''),selected_frames=data.get('selected_frame_count',''),
                   prompts=len(data.get('prompts',[])),total_seconds=round(data.get('total_seconds',0),2))
        for name in ('lingbot','vlm','sam','adapt_masks','mapping'):
            stage=stages.get(name,{})
            row[name+'_seconds']=round(stage['elapsed_seconds'],2) if 'elapsed_seconds' in stage else ''
        row['lingbot_inference_seconds']=stages.get('lingbot',{}).get('reported_inference_seconds','')
        row['lingbot_inference_fps']=stages.get('lingbot',{}).get('reported_inference_fps','')
        row['current_stage']=next((s['name'] for s in data['stages'] if s['status']=='running'),'')
        row['error']=data.get('error','')
        rows.append(row)
    if not rows: return
    buffer=io.StringIO()
    writer=csv.DictWriter(buffer,fieldnames=list(rows[0]),lineterminator='\n')
    writer.writeheader();writer.writerows(rows)
    (root/'timing_summary.csv').write_text(buffer.getvalue())
    fields=['scene','status','frames','selected_frames','prompts','lingbot_seconds','vlm_seconds','sam_seconds','mapping_seconds','total_seconds','lingbot_inference_fps','current_stage']
    cards=[]
    for row in rows:
        cells=[]
        for field in fields:
            value=html.escape(str(row[field]))
            if field=='scene':value=f'<a href="{row["scene"]}/timings.json">{value}</a>'
            cells.append('<td>'+value+'</td>')
        scene=row['scene']
        links=f'<a href="{scene}/obstacle_map.ply">PLY</a> | <a href="{scene}/selection.json">Selected frames</a> | <a href="{scene}/prompts.json">Prompts</a>'
        cards.append('<tr>'+''.join(cells)+'<td>'+links+'</td></tr>')
    elapsed=sum(r['total_seconds'] for r in rows)
    complete=sum(r['status']=='complete' for r in rows)
    failed=sum(r['status']=='failed' for r in rows)
    page='<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="15"><title>VLM map timing</title><style>body{font-family:sans-serif;margin:24px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}td{white-space:nowrap}</style>'
    page+=f'<h1>VLM map processing</h1><p>Updated {datetime.now(timezone.utc).isoformat()}. Complete: {complete}; failed: {failed}. Recorded completed-scene wall time: {elapsed:.1f}s.</p>'
    page+='<p>Stage times include startup, model loading and IO. LingBot inference FPS is reported separately by the original implementation. Rendering is excluded. VLM, SAM and mapping use the same every-second saved-frame selection.</p>'
    page+='<p><a href="timing_summary.csv">Download timing CSV</a> | <a href="batch_timings.json">Batch JSON</a></p><table><tr>'+''.join('<th>'+html.escape(f)+'</th>' for f in fields)+'<th>Artifacts</th></tr>'+''.join(cards)+'</table>'
    if (root/'REPORT.md').exists():
        page+='<p><a href="REPORT.md">Final report and validation findings</a> | <a href="verified_results.json">Verified export statistics</a></p>'
    (root/'index.html').write_text(page)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root',type=Path)
    args=parser.parse_args()
    report(args.run_root)
