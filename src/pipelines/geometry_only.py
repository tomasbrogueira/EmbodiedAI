"""Geometry baseline: explicitly absent semantic evidence and zero models."""
from pipeline_common.contracts import SemanticFrame

class GeometryOnlyAdapter:
    def observe(self, frame):
        return SemanticFrame(sequence_id=frame.sequence_id,frame_id=frame.frame_id,
            processed_grid_id=frame.processed_grid_id,decoded_rgb_sha256=frame.decoded_rgb_sha256,
            geometry_fingerprint=frame.geometry.geometry_fingerprint if frame.geometry else None,
            adapter_provenance={"adapter_id":"geometry_only","models_executed":False},
            status="not_applicable",timestamp_ns=frame.timestamp_ns,model_calls={"sam3":0,"qwen":0})
    def close(self): pass

def create_adapter(config): return GeometryOnlyAdapter()
