"""Ordered local inputs with honest clock and two distinct image identities."""
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
from .contracts import CONTRACT_ID
from .io import *

def validate_sequence(path):
    path = Path(path).resolve(); data = read_json(path)
    if data.get("contract_id") != CONTRACT_ID or type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError("Sequence contract/schema mismatch")
    if not isinstance(data.get("sequence_id"), str) or not data["sequence_id"] or data.get("split") not in ("development", "test") or type(data.get("fixture")) is not bool:
        raise ValueError("Invalid sequence identity/split/fixture")
    if digest_json({k:v for k,v in data.items() if k != "manifest_digest"}) != data.get("manifest_digest"):
        raise ValueError("Sequence manifest digest mismatch")
    if data.get("encoded_hash_recipe") != ENCODED_HASH_RECIPE or data.get("decoded_hash_recipe") != RGB_HASH_RECIPE:
        raise ValueError("Sequence hash recipes mismatch")
    rows = data.get("frames"); ids=set(); previous=None
    if not isinstance(rows,list) or not rows: raise ValueError("Sequence must have frames")
    for row in rows:
        if not isinstance(row.get("frame_id"),str) or not row["frame_id"] or row["frame_id"] in ids: raise ValueError("Missing/duplicate frame ID")
        ids.add(row["frame_id"])
        if "timestamp_ns" not in row or not isinstance(row.get("timestamp_provenance"),dict): raise ValueError("Missing capture clock/provenance")
        timestamp=row["timestamp_ns"]
        if timestamp is not None:
            if type(timestamp) is not int or not 0<=timestamp<=np.iinfo(np.int64).max or (previous is not None and timestamp <= previous): raise ValueError("Known timestamps must strictly increase within int64")
            previous=timestamp
        image_path=safe_path(path.parent,row["image_path"])
        if file_sha256(image_path) != row.get("encoded_file_sha256"): raise ValueError("Sequence encoded hash mismatch")
        with Image.open(image_path) as image: rgb=np.asarray(image.convert("RGB"))
        if list(rgb.shape[:2]) != [row.get("height"),row.get("width")] or rgb_sha256(rgb) != row.get("decoded_rgb_sha256"):
            raise ValueError("Sequence decoded pixel/grid mismatch")
    return data

def prepare(input_path, config, output):
    from path_mapping.runner import collect_frames
    input_path=Path(input_path).resolve(); output=Path(output).resolve()
    if output.exists() and any(output.iterdir()): raise ValueError("Sequence output must be empty")
    if config.get("contract_id") != CONTRACT_ID or config.get("schema_version") != 1: raise ValueError("Preparation config contract/schema mismatch")
    if config.get("split") not in ("development","test") or type(config.get("fixture",False)) is not bool: raise ValueError("Invalid prepare split/fixture")
    stride=config.get("stride",1); maximum=config.get("max_frames",8)
    if type(stride) is not int or stride<1 or type(maximum) is not int or maximum<1: raise ValueError("Invalid selection limits")
    output.mkdir(parents=True); images=output/"images"; images.mkdir()
    sources=[]; timestamps=[]; provenances=[]
    if input_path.is_dir():
        paths=collect_frames(input_path,stride=stride,max_frames=maximum)
        provided=config.get("timestamps_ns")
        if provided is not None and len(provided)!=len(paths): raise ValueError("Provide one timestamp per selected image")
        for i,path in enumerate(paths):
            destination=images/f"{i:06d}{path.suffix.lower()}"; shutil.copyfile(path,destination)
            sources.append((path,destination)); timestamps.append(provided[i] if provided is not None else None)
            provenances.append(config.get("timestamp_provenance", {"clock":"unknown", "kind":"unknown", "reason":"frame folders have no capture timing"}))
        provenance={"kind":"generic_frame_folder","path":str(input_path),"ordering":"natural_filename_order", "selection":{"stride":stride,"max_frames":maximum}}
    else:
        import cv2
        capture=cv2.VideoCapture(str(input_path)); rate=config.get("sample_fps",2.0)
        if not np.isfinite(rate) or rate<=0: raise ValueError("Invalid sample_fps")
        fps=capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or not np.isfinite(fps) or fps<=0: raise ValueError("Video unavailable or unknown FPS")
        interval=max(1,round(fps/rate))*stride; index=0; last=None
        try:
            while len(sources)<maximum:
                ok,bgr=capture.read()
                if not ok: break
                if index%interval==0:
                    pts=capture.get(cv2.CAP_PROP_POS_MSEC)
                    ts=int(round(pts*1e6)) if np.isfinite(pts) and pts>=0 and (last is None or pts>last) else None
                    last=pts
                    destination=images/f"{index:09d}.png"; Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).save(destination)
                    sources.append((input_path,destination)); timestamps.append(ts)
                    provenances.append({"clock":"video_presentation_timeline", "kind":"decoder_reported_pts" if ts is not None else "unknown", "source_frame_index":index,"backend":capture.getBackendName()})
                index+=1
        finally: capture.release()
        provenance={"kind":"generic_video","path":str(input_path),"encoded_video_sha256":file_sha256(input_path),"source_fps":fps,"timestamp_limit":"OpenCV decoder reported presentation times, not wall-clock capture"}
    rows=[]
    for i,((source,path),timestamp,clock) in enumerate(zip(sources,timestamps,provenances)):
        with Image.open(path) as image:
            rgb=np.asarray(image.convert("RGB")); orientation=image.getexif().get(274,1)
        rows.append({"frame_id":f"frame_{i:06d}","timestamp_ns":timestamp,"timestamp_provenance":clock,
            "image_path":path.relative_to(output).as_posix(),"encoded_file_sha256":file_sha256(path),"decoded_rgb_sha256":rgb_sha256(rgb),"width":rgb.shape[1],"height":rgb.shape[0],"exif_orientation":orientation,"source_path":str(source)})
    data={"contract_id":CONTRACT_ID,"schema_version":1,"sequence_id":config["sequence_id"],"split":config["split"],"fixture":config.get("fixture",False),"frames":rows,"input_provenance":provenance,"encoded_hash_recipe":ENCODED_HASH_RECIPE,"decoded_hash_recipe":RGB_HASH_RECIPE,"unknown_timestamp_array_value":-1}
    data["manifest_digest"]=digest_json(data); write_json(output/"sequence.json",data)
    validate_sequence(output/"sequence.json"); return data
