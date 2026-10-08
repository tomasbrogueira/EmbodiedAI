"""Verified LingBot depth/pose geometry cache; imports models only on inference."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from path_mapping.runner import (validate_geometry, reprojection_diagnostic,
    geometry_fingerprint, GEOMETRY_KEYS)
from .contracts import FramePacket, GeometryFrame
from .io import *

def _orientation_matrix(orientation,w,h):
    matrices={1:[[1,0,0],[0,1,0],[0,0,1]],2:[[-1,0,w-1],[0,1,0],[0,0,1]],
        3:[[-1,0,w-1],[0,-1,h-1],[0,0,1]],4:[[1,0,0],[0,-1,h-1],[0,0,1]],
        5:[[0,1,0],[1,0,0],[0,0,1]],6:[[0,-1,h-1],[1,0,0],[0,0,1]],
        7:[[0,-1,h-1],[-1,0,w-1],[0,0,1]],8:[[0,1,0],[-1,0,w-1],[0,0,1]]}
    if orientation not in matrices: raise ValueError("Unknown EXIF orientation")
    return np.array(matrices[orientation],float)

def preprocessing(paths,image_size,fixture=False):
    outputs=[]; records=[]
    for path in paths:
        with Image.open(path) as original:
            w0,h0=original.size; orientation=original.getexif().get(274,1)
            image=ImageOps.exif_transpose(original)
            if image.mode=="RGBA": image=Image.alpha_composite(Image.new("RGBA",image.size,(255,255,255,255)),image)
            image=image.convert("RGB"); w,h=image.size
            nw=w if fixture else image_size
            nh=h if fixture else round(h*(nw/w)/14)*14
            if nh<1: raise ValueError("Image too short for LingBot preprocessing")
            resized=image if fixture else image.resize((nw,nh),Image.Resampling.BICUBIC)
            top=0 if fixture else max(0,(nh-image_size)//2)
            ch=nh if fixture else min(nh,image_size)
            if top or ch!=nh: resized=resized.crop((0,top,nw,top+ch))
            rgb=np.array(resized,dtype=np.uint8)
            # Upstream ToTensor uint8 -> float32 /255, followed by *255 -> uint8.
            if not fixture: rgb=(rgb.astype(np.float32)/np.float32(255)*np.float32(255)).astype(np.uint8)
        transform=np.array([[nw/w,0,(nw/w-1)/2],[0,nh/h,(nh/h-1)/2-top],[0,0,1]])@_orientation_matrix(orientation,w0,h0)
        records.append({"source_shape":[h0,w0],"oriented_shape":[h,w],"resized_shape":[nh,nw],
            "crop_xywh":[0,top,nw,ch],"exif_orientation":orientation,"matrix":transform.tolist(),
            "recipe":"synthetic_identity_v1" if fixture else "lingbot_crop_bicubic_patch14_pixel_centers_v1",
            "coverage":"cropped_out_source_pixels_excluded"})
        outputs.append(rgb)
    maxh=max(a.shape[0] for a in outputs); maxw=max(a.shape[1] for a in outputs)
    padded=[]
    for rgb,record in zip(outputs,records):
        top=(maxh-rgb.shape[0])//2; left=(maxw-rgb.shape[1])//2
        pad=np.full((maxh,maxw,3),255,np.uint8);pad[top:top+rgb.shape[0],left:left+rgb.shape[1]]=rgb
        matrix=np.array(record["matrix"]);matrix[:2,2]+=[left,top]
        record.update(matrix=matrix.tolist(),pad_ltrb=[left,top,maxw-rgb.shape[1]-left,maxh-rgb.shape[0]-top],processed_shape=[maxh,maxw])
        padded.append(pad)
    return np.stack(padded),records

def _code_identity(source):
    from path_mapping.models import _source_revision
    upstream=Path(source)/"lingbot_map/utils/load_fn.py"
    package=Path(source)/"lingbot_map"
    code_digest=digest_json({p.relative_to(package).as_posix():file_sha256(p) for p in sorted(package.rglob("*.py"))}) if package.is_dir() else None
    return {"source_revision":_source_revision(Path(source)),"upstream_python_code_sha256":code_digest,
        "adapter_sha256":file_sha256(Path(__file__).parents[1]/"path_mapping/models.py"),
        "geometry_bridge_sha256":file_sha256(__file__),
        "preprocessing_source_sha256":file_sha256(upstream) if upstream.is_file() else None}

def geometry_settings_identity(settings, fixture):
    selected={k:v for k,v in settings.items() if k not in {"cache_dir"}}
    source=Path(settings.get("source_root","src/lingbot-map"))
    checkpoint=Path(settings.get("checkpoint",""))
    selected["actual_code"]=_code_identity(source)
    if settings.get("source_revision") and selected["actual_code"]["source_revision"]!=settings["source_revision"]:
        raise ValueError("LingBot source revision mismatch against declared pin")
    if not fixture:
        declared=settings.get("checkpoint_sha256")
        if checkpoint.is_file():
            actual=file_sha256(checkpoint)
            if declared is not None and declared!=actual: raise ValueError("LingBot checkpoint identity mismatch")
            selected["checkpoint_sha256"]=actual
        elif not declared:
            raise FileNotFoundError("Local LingBot checkpoint missing; cached reuse requires its declared SHA256")
    selected["fixture"]=fixture
    return selected

def synthetic_geometry(images):
    s,h,w,_=images.shape; v,u=np.mgrid[:h,:w]; f=12.0
    depth=np.full((s,h,w),2.0,np.float32)
    intrinsic=np.tile(np.array([[f,0,(w-1)/2],[0,f,(h-1)/2],[0,0,1]],np.float32),(s,1,1))
    rotation=np.diag([1,-1,-1]).astype(np.float32)
    extrinsic=np.tile(np.column_stack([rotation,[0,0,2]]),(s,1,1)).astype(np.float32)
    camera=np.stack(((u-(w-1)/2)*2/f,(v-(h-1)/2)*2/f,np.full_like(u,2)),axis=-1)
    points=(camera-[0,0,2])@rotation
    return {"images":images,"world_points":np.tile(points,(s,1,1,1)).astype(np.float32),
        "world_points_conf":np.full((s,h,w),2,np.float32),"depth":depth,
        "extrinsic":extrinsic,"intrinsic":intrinsic}

def geometry_cache(sequence_path,sequence,settings,cache_dir,*,fixture=False,reuse=False):
    sequence_path=Path(sequence_path).resolve();cache_dir=Path(cache_dir).resolve()
    if fixture!=sequence["fixture"]: raise ValueError("Geometry fixture identity mismatch")
    paths=[safe_path(sequence_path.parent,r["image_path"]) for r in sequence["frames"]]
    expected_images,transforms=preprocessing(paths,settings.get("image_size",518),fixture)
    expected={"contract_id":"semantic_mapping_v1","schema_version":1,
        "sequence_digest":sequence["manifest_digest"],"settings":geometry_settings_identity(settings,fixture),"transforms":transforms}
    expected_id=digest_json(expected)
    manifest_path=cache_dir/"manifest.json"
    if reuse:
        manifest=read_json(manifest_path)
        if manifest.get("status")!="complete" or manifest.get("input_identity")!=expected or manifest.get("input_fingerprint")!=expected_id:
            raise ValueError("Geometry cache identity mismatch (frames/code/settings/calibration/timestamps)")
        if file_sha256(cache_dir/"geometry.npz")!=manifest.get("archive_sha256"): raise ValueError("Geometry cache archive tampering")
        with np.load(cache_dir/"geometry.npz",allow_pickle=False) as data: geometry={k:data[k] for k in GEOMETRY_KEYS}
    else:
        if cache_dir.exists() and any(cache_dir.iterdir()): raise ValueError("New geometry cache must be empty")
        cache_dir.mkdir(parents=True)
        if fixture: geometry=synthetic_geometry(expected_images);model={"fixture":True,"models_executed":False}
        else:
            from path_mapping.models import reconstruct
            geometry,model=reconstruct(paths,checkpoint=Path(settings["checkpoint"]),source_root=Path(settings.get("source_root","src/lingbot-map")),device=settings.get("device","cuda:0"),image_size=settings.get("image_size",518),keyframe_interval=settings.get("keyframe_interval",1))
        validate_geometry(geometry,allow_single_frame=True)
        if not np.array_equal(geometry["images"],expected_images): raise ValueError("LingBot preprocessing differs from documented transform/grid")
        scale=settings.get("scale",{}); factor=scale.get("meters_per_unit",1.0)
        if not np.isfinite(factor) or factor<=0: raise ValueError("Invalid metric scale")
        geometry={k:v.copy() for k,v in geometry.items()}
        for key in ("world_points","depth"): geometry[key]*=factor
        geometry["extrinsic"][:,:,3]*=factor
        write_npz(cache_dir/"geometry.npz",**geometry)
        manifest={**expected,"input_identity":expected,"input_fingerprint":expected_id,
            "model":model,"archive_sha256":file_sha256(cache_dir/"geometry.npz")}
    validate_geometry(geometry,allow_single_frame=True)
    if not np.array_equal(geometry["images"],expected_images): raise ValueError("Cache processed RGB/source identity mismatch")
    alignment=reprojection_diagnostic(geometry,settings.get("min_confidence",1.5))
    # Reprojection alone cannot distinguish wrong positive depth: check optical z.
    for xyz,depth,pose,confidence in zip(geometry["world_points"],geometry["depth"],geometry["extrinsic"],geometry["world_points_conf"]):
        z=(xyz@pose[:,:3].T+pose[:,3])[...,2];depth=depth.reshape(z.shape)
        valid=np.isfinite(xyz).all(-1)&np.isfinite(depth)&(depth>0)&np.isfinite(confidence)&(confidence>=settings.get("min_confidence",1.5))&(confidence>0)
        if not np.allclose(z[valid],depth[valid],rtol=1e-4,atol=1e-5): raise ValueError("Optical-axis depth/pose unprojection mismatch")
    fingerprint=geometry_fingerprint(geometry)
    if reuse and fingerprint!=manifest.get("geometry_fingerprint"): raise ValueError("Cache geometry fingerprint mismatch")
    scale=settings.get("scale",{});up=settings.get("up",{})
    units="metres" if scale.get("verified") is True and scale.get("source") and scale.get("meters_per_unit") else "reconstruction_units"
    up_vector=up.get("vector") if up.get("verified") is True and up.get("source") else None
    if up_vector is not None:
        up_vector=np.asarray(up_vector,float)
        if up_vector.shape!=(3,) or not np.isfinite(up_vector).all() or np.linalg.norm(up_vector)<1e-9: raise ValueError("Invalid up calibration")
        up_vector=(up_vector/np.linalg.norm(up_vector)).tolist()
    processed=cache_dir/"processed_frames";processed.mkdir(exist_ok=True); packets=[]
    grid_id=digest_json({"input_fingerprint":expected_id,"geometry":fingerprint,"shape":list(geometry["images"].shape[1:3])})
    for index,(row,rgb,transform) in enumerate(zip(sequence["frames"],geometry["images"],transforms)):
        path=processed/f"{index:06d}.png"
        if not reuse: Image.fromarray(rgb).save(path)
        with Image.open(path) as image:
            if image.format!="PNG" or not np.array_equal(np.asarray(image),rgb): raise ValueError("Processed persisted RGB tampering")
        depth=geometry["depth"][index].reshape(rgb.shape[:2]); points=geometry["world_points"][index];confidence=geometry["world_points_conf"][index]
        valid=np.isfinite(points).all(-1)&np.isfinite(depth)&(depth>0)&np.isfinite(confidence)&(confidence>0)&(confidence>=settings.get("min_confidence",1.5))
        left,top,right,bottom=transform["pad_ltrb"]
        if top: valid[:top]=False
        if bottom: valid[-bottom:]=False
        if left: valid[:,:left]=False
        if right: valid[:,-right:]=False
        pose=np.eye(4);pose[:3]=geometry["extrinsic"][index]
        gf=GeometryFrame(points=points,depth=depth,validity=valid,confidence=confidence,
            intrinsics=geometry["intrinsic"][index],world_to_camera=pose,processed_grid_id=grid_id,
            geometry_fingerprint=fingerprint,map_frame=settings.get("map_frame","lingbot_world"),units=units,
            up=tuple(up_vector) if up_vector else None,scale=scale,pose_revision=settings.get("pose_revision","initial"))
        for array in (rgb,points,depth,confidence,valid,pose,gf.intrinsics): array.setflags(write=False)
        packets.append(FramePacket(sequence_id=sequence["sequence_id"],frame_id=row["frame_id"],timestamp_ns=row["timestamp_ns"],timestamp_provenance=row["timestamp_provenance"],rgb=rgb,image_path=path,encoded_file_sha256=file_sha256(path),decoded_rgb_sha256=rgb_sha256(rgb),processed_grid_id=grid_id,source_rgb_identity={"image_path":row["image_path"],"encoded_file_sha256":row["encoded_file_sha256"],"decoded_rgb_sha256":row["decoded_rgb_sha256"]},source_to_processed=transform,geometry=gf))
    manifest.update(status="complete",geometry_fingerprint=fingerprint,processed_grid_id=grid_id,
        map_frame=settings.get("map_frame","lingbot_world"),units=units,up=up_vector,scale=scale,
        axes="OpenCV W2C: x right, y down, z forward",depth_kind="optical_axis",pose_revision=settings.get("pose_revision","initial"),alignment=alignment,free_space_capability=False,unknown_timestamp_array_value=-1)
    if not reuse: write_json(manifest_path,manifest)
    return packets,manifest
