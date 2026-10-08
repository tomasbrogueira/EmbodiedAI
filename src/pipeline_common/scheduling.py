"""One synchronous adapter, immutable in-flight joins, one latest pending frame."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from copy import deepcopy
import time
from .contracts import validate_semantic

def immutable_frame(frame):
    def own(array):
        value=array.copy(); value.setflags(write=False); return value
    geometry=frame.geometry
    if geometry is not None:
        geometry=replace(geometry,scale=deepcopy(geometry.scale),**{key:own(getattr(geometry,key)) for key in
            ("points","depth","validity","confidence","intrinsics","world_to_camera")})
    return replace(frame,rgb=own(frame.rgb),geometry=geometry,timestamp_provenance=deepcopy(frame.timestamp_provenance),
        source_rgb_identity=deepcopy(frame.source_rgb_identity),source_to_processed=deepcopy(frame.source_to_processed))

class LatestPendingWorker:
    """No hidden adapter scheduling; at most one running and one pending job."""
    def __init__(self,adapter):
        self.adapter=adapter; self.pool=ThreadPoolExecutor(max_workers=1)
        self.inflight=None; self.pending=None; self.events=[]; self.closed=False
    def _start(self,frame,capture_monotonic_ns):
        def execute():
            started=time.monotonic_ns()
            try: return {"result":self.adapter.observe(frame),"started_monotonic_ns":started,"completed_monotonic_ns":time.monotonic_ns()}
            except Exception as error: return {"error":error,"started_monotonic_ns":started,"completed_monotonic_ns":time.monotonic_ns()}
        future=self.pool.submit(execute)
        self.inflight=(frame,capture_monotonic_ns,future)
    def submit(self,frame,capture_monotonic_ns=None):
        if self.closed: raise RuntimeError("Worker closed")
        owned=immutable_frame(frame)
        if self.inflight is None: self._start(owned,capture_monotonic_ns)
        else:
            if self.pending is not None:
                self.events.append({"event":"queue_drop","frame_id":self.pending[0].frame_id,
                    "reason":"superseded_latest_pending","monotonic_ns":time.monotonic_ns()})
            self.pending=(owned,capture_monotonic_ns)
    def poll(self,block=False):
        if self.inflight is None: return None
        frame,capture,future=self.inflight
        if not block and not future.done(): return None
        record=None
        try:
            record=future.result()
            if "result" in record: validate_semantic(record["result"],frame)
            record.update(frame=frame,mapped_capture_monotonic_ns=capture)
        except Exception as error:
            record=record or {"started_monotonic_ns":time.monotonic_ns(),"completed_monotonic_ns":time.monotonic_ns()}
            record.pop("result",None)
            record.update(frame=frame,error=error,mapped_capture_monotonic_ns=capture)
        self.inflight=None
        if self.pending is not None:
            pending=self.pending;self.pending=None;self._start(*pending)
        return record
    def drain(self):
        while self.inflight is not None: yield self.poll(block=True)
    def close(self):
        self.closed=True;self.pool.shutdown(wait=True)

def mapped_capture(frame,source_origin_ns,replay_origin_ns,rate):
    if frame.timestamp_ns is None or frame.timestamp_provenance.get("kind") in {"unknown","synthetic"}: return None
    return replay_origin_ns+int((frame.timestamp_ns-source_origin_ns)/rate)
