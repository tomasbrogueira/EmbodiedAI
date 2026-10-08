"""Explicit fake providers, never selected for real inference or a real sequence."""
import time
import numpy as np
from .contracts import SemanticFrame,QueryRecord,InstanceRecord
from pipelines.geometry_only import GeometryOnlyAdapter

class FixtureAdapter:
    def __init__(self,pipeline,config): self.pipeline=pipeline;self.config=config
    def observe(self,frame):
        started=time.monotonic_ns();time.sleep(self.config.get("delay_s",0))
        mask=np.zeros(frame.rgb.shape[:2],bool)
        if self.pipeline=="ground_surface": mask[:]=True; phrase=self.config.get("prompt","ground");concept="ground_surface";role="candidate_surface"
        else: mask[3:6,5:8]=True;phrase="water";concept="hazard.water.v1";role="hazard"
        query=QueryRecord(query_id=f"{frame.frame_id}:q0",original_phrase=phrase,concept_id=concept,role=role,
            instances=[InstanceRecord(observation_id=f"{frame.frame_id}:q0:i0",mask=mask,score=.9,score_meaning="synthetic_evidence_weight")])
        return SemanticFrame(sequence_id=frame.sequence_id,frame_id=frame.frame_id,processed_grid_id=frame.processed_grid_id,
            decoded_rgb_sha256=frame.decoded_rgb_sha256,geometry_fingerprint=frame.geometry.geometry_fingerprint,
            adapter_provenance={"adapter_id":self.pipeline,"fixture":True,"models_executed":False},status="ok",queries=[query],
            model_calls={"sam3":0,"qwen":0},query_count=0,timestamp_ns=frame.timestamp_ns,
            started_monotonic_ns=started,completed_monotonic_ns=time.monotonic_ns())
    def close(self): pass

def create_fixture_adapter(pipeline,config,sequence):
    if not sequence.get("fixture") or config.get("fixture") is not True: raise ValueError("Fake providers require explicit fixture config and sequence")
    return GeometryOnlyAdapter() if pipeline=="geometry_only" else FixtureAdapter(pipeline,config)
