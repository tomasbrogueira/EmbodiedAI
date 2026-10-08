"""Inspect a completed raw surface map and camera poses without loading models."""
import argparse
from pathlib import Path
import time
import numpy as np
from .io import read_json

def show(run_dir,host="127.0.0.1",port=8080):
    run_dir=Path(run_dir).resolve();run=read_json(run_dir/"run.json")
    if run.get("status") not in {"complete","unavailable"}: raise ValueError("Only finished raw maps can be viewed")
    try:
        import viser
        from viser.transforms import SO3
    except ImportError as error: raise RuntimeError("Viewer requires optional viser; raw_surfaces.ply is also portable") from error
    with np.load(run_dir/"map/voxels.npz",allow_pickle=False) as data: centers=data["centers"]
    manifest=read_json(run_dir/"map/manifest.json")
    server=viser.ViserServer(host=host,port=port)
    try:
        if manifest.get("up"): server.scene.set_up_direction(manifest["up"])
        cloud=server.scene.add_point_cloud("/observed_surfaces",points=centers,colors=(145,175,210),point_size=manifest.get("voxel_size",.05))
        trajectory_path=run_dir/"geometry/camera_trajectory.npz"
        if trajectory_path.is_file():
            with np.load(trajectory_path,allow_pickle=False) as data: poses=data["world_to_camera"];intrinsics=data["intrinsics"]
            for index,(w2c,k) in enumerate(zip(poses,intrinsics)):
                c2w=np.linalg.inv(w2c)
                server.scene.add_camera_frustum(f"/cameras/{index:06d}",fov=float(2*np.arctan(max(1,k[1,2]*2)/2/k[1,1])),aspect=float(k[0,2]/max(k[1,2],1)),scale=.1,
                    wxyz=SO3.from_matrix(c2w[:3,:3]).wxyz,position=c2w[:3,3],color=(100,200,150))
        server.gui.add_markdown(f"**Observed surfaces — {manifest.get('units')}**\n\nPhysical planning: {run.get('planning_status','unavailable')}. Missing surfaces remain unknown.")
        print(f"Raw map viewer: http://{host}:{port}",flush=True)
        while True: time.sleep(.5)
    except KeyboardInterrupt: pass
    finally: server.stop()

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run",required=True);parser.add_argument("--host",default="127.0.0.1");parser.add_argument("--port",type=int,default=8080)
    args=parser.parse_args();show(args.run,args.host,args.port)
