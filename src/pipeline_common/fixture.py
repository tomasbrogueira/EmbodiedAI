"""Create a tiny analytic scene/reference, then run four real CLI paths on CPU."""
from pathlib import Path
import argparse
import subprocess
import sys
import numpy as np
from PIL import Image
from .contracts import PIPELINE_IDS,CONTRACT_ID
from .io import *
from .sequence import prepare
from .runtime import resolve_config
from .geometry import geometry_cache

def create_fixture(output):
    output=Path(output).resolve()
    if output.exists() and any(output.iterdir()): raise ValueError("Fixture output must be empty")
    inputs=output/"source_images";inputs.mkdir(parents=True)
    v,u=np.mgrid[:12,:16]
    for i in range(3): Image.fromarray(np.stack([u*12+10,v*15+10,np.full_like(u,90+i)],-1).astype(np.uint8)).save(inputs/f"{i}.png")
    prep={"contract_id":CONTRACT_ID,"schema_version":1,"sequence_id":"independent_analytic_fixture_v1","split":"test","fixture":True,"max_frames":3,"timestamps_ns":[0,100000000,200000000],"timestamp_provenance":{"clock":"synthetic_fixture_capture","kind":"synthetic"}}
    write_json(output/"prepare.json",prep);sequence=prepare(inputs,prep,output/"sequence")
    robot={"version":"synthetic_circle_robot_v1","fixture":True,"footprint_radius":.1,"height":.5,"clearance":.05,"max_slope_degrees":25.,"max_step":.15,"min_support_points":3,"max_roughness":.03,"unknown_rule":"blocked","semantic_costs":{"hazard.water.v1":{"cost":1.,"blocked":True}}}
    grid={"shape":[4,6],"origin":[-1.5,-1.,0.],"resolution":.5,"projection_basis":[[1.,0.,0.],[0.,1.,0.]],"map_frame":"fixture_world"}
    config={"contract_id":CONTRACT_ID,"schema_version":1,"fixture":True,"mode":"quality_replay","protocol_id":"semantic_mapping_eval_v1",
        "geometry":{"map_frame":"fixture_world","scale":{"verified":True,"source":"independent analytic fixture metre construction","meters_per_unit":1.},"up":{"vector":[0,0,1],"verified":True,"source":"independent analytic fixture gravity"}},
        "fusion":{"voxel_size":.25,"origin":[0,0,0],"semantic_expiry_ns":None,"negative_evidence":True},"robot":robot,
        "planning":{"resolution":.5,"origin":[-1.5,-1.],"shape":[4,6],"support_height":0.},
        "goals":[{"request_id":"across_floor","start":[-.75,-.25,0.],"goal":[.75,-.25,0.],"frame":"fixture_world"}],
        "evaluation_grid":grid,"hardware_budget":{"device":"CPU fixture","max_visible_gpus":0,"measured":False},
        "semantic_keyframe_ids":[r["frame_id"] for r in sequence["frames"]],"pipeline_config":{}}
    write_json(output/"run.json",config)
    resolved=resolve_config(config,"geometry_only",sequence,True)
    packets,manifest=geometry_cache(output/"sequence/sequence.json",sequence,resolved["geometry"],output/"geometry_cache",fixture=True)
    # This target is authored analytically before predictions: an infinite flat
    # plane with an independently specified synthetic excluded water rectangle.
    decisions=np.full((4,6),2,np.uint8);decisions[2,2]=1
    write_npz(output/"reference/decisions.npz",decision_state=decisions,valid_mask=np.ones((4,6),bool),ignore_mask=np.zeros((4,6),bool))
    from .evaluation import fingerprint
    reference={"contract_id":CONTRACT_ID,"schema_version":1,"reference_id":"independent_analytic_robot_fixture_v1","reference_protocol_id":"analytic_reference_v1",
        "sequence":resolved["sequence"],"fixture":True,"provenance":{"independent":True,"source":"analytic infinite floor and externally specified excluded synthetic water region; no evaluated model outputs"},
        "taxonomy":{"version":"fixture_v1","concept_ids":["ground_surface","hazard.water.v1"]},
        "coordinates":{"map_frame":"fixture_world","units":"metres","axes":["x","y","z"],"up":[0,0,1],"alignment":{"verified":True,"source":"analytic fixture"},"scale_provenance":{"verified":True,"source":"analytic fixture"},"pose_provenance":{"verified":True,"source":"analytic fixture"}},
        "assets":[{"asset_id":"decisions","path":"decisions.npz","sha256":file_sha256(output/"reference/decisions.npz")}],
        "kinds":{"robot_decision":{"asset_id":"decisions","decision_key":"decision_state","valid_key":"valid_mask","ignore_key":"ignore_mask","grid":grid,"coverage":{"description":"all 24 independently specified synthetic cells"},"robot_reference_policy":{"independent":True,"version":"synthetic_robot_reference_v1","robot_fingerprint":fingerprint(robot),"description":"analytic floor support; the externally defined synthetic excluded region is not robot-safe"}}}}
    write_json(output/"reference/reference.json",reference)
    return output

def check_fixture(output):
    output=create_fixture(output);repo=Path(__file__).parents[2]
    evaluations=[]
    for pipeline in PIPELINE_IDS:
        run_dir=output/"runs"/pipeline;eval_dir=output/"evaluations"/pipeline
        subprocess.run([sys.executable,str(repo/"src/run_pipeline.py"),"--pipeline",pipeline,"--sequence",str(output/"sequence/sequence.json"),"--config",str(output/"run.json"),"--geometry-cache",str(output/"geometry_cache"),"--output",str(run_dir),"--fixture"],check=True,cwd=repo)
        subprocess.run([sys.executable,str(repo/"src/evaluate_pipeline.py"),"--run",str(run_dir),"--reference",str(output/"reference/reference.json"),"--output",str(eval_dir)],check=True,cwd=repo)
        evaluations.append(eval_dir)
    arguments=[sys.executable,str(repo/"src/compare_pipelines.py")]
    for directory in evaluations: arguments.extend(["--evaluation",str(directory)])
    subprocess.run(arguments+["--output",str(output/"comparison")],check=True,cwd=repo)
    return output

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",required=True)
    args=parser.parse_args();print(check_fixture(args.output))
