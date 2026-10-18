"""Persistent GPU geometry validator, with a JSON-lines protocol."""
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np

def main():
    try:
        from .gpu_geometry import GPUIndex
        print(json.dumps(dict(status='ready')),flush=True)
    except Exception as exc:
        print(json.dumps(dict(status='error',error=str(exc))),flush=True); return
    index=None
    for line in sys.stdin:
        request=json.loads(line)
        if request.get('op')=='close': break
        try:
            start=time.perf_counter()
            with np.load(request['input'],allow_pickle=False) as data:
                reuse=request.get('op')=='route'
                if reuse:
                    if index is None:raise ValueError('No geometry index loaded for route check')
                else:
                    index=None
                    index=GPUIndex(data['points'],data['positive'],data['negative'],data['slopes'],float(data['radius']),bool(data['enforce']))
                valid,metrics=index.check(data['route'],data['normal'],float(data['height']),bool(data['enforce']))
            print(json.dumps(dict(status='ok',valid=valid,metrics=metrics,
                upload_ms=0. if reuse else index.upload_ms,index_ms=0. if reuse else index.build_ms,index_reused=reuse,
                wall_ms=(time.perf_counter()-start)*1000)),flush=True)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            print(json.dumps(dict(status='error',error=str(exc))),flush=True)

if __name__=='__main__': main()
