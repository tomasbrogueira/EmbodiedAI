"""Common CLI lifecycle, frozen settings, map/planner and auditable dispositions."""
from __future__ import annotations
from dataclasses import fields
from pathlib import Path
from datetime import datetime, timezone
import copy
import os
import platform
import sys
import time
import traceback
import uuid
import numpy as np
from .contracts import *
from .io import *
from .sequence import validate_sequence
from .geometry import geometry_cache
from .scheduling import LatestPendingWorker,mapped_capture

def new_attempt_path(output_root, pipeline):
    """Choose a fresh name; run() exclusively reserves it before writing."""
    if pipeline not in PIPELINE_IDS: raise ValueError("Unknown pipeline ID")
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return Path(output_root).resolve()/f"{stamp}_{pipeline}_{uuid.uuid4().hex}"

def _reserve_output(output):
    output.parent.mkdir(parents=True,exist_ok=True)
    try:
        output.mkdir()
    except FileExistsError:
        if output.is_dir() and (output/"run.json").is_file():
            try: completed=read_json(output/"run.json").get("status")=="complete"
            except (ValueError,OSError): completed=False
            if completed: raise ValueError("Refusing to overwrite completed run") from None
        raise ValueError("Run output already exists; choose a fresh attempt directory") from None

def _failure(output, identity, error):
    """Preserve the original exception and any earlier diagnostic artifacts."""
    detail={"event":"failure","stage":identity.get("stage"),"error_type":type(error).__name__,
        "error":str(error),"traceback":traceback.format_exc()}
    identity.update(status="failed",reason=f"{detail['stage']}:{type(error).__name__}:{error}",
        finished_at_utc=datetime.now(timezone.utc).isoformat())
    for path,value in ((output/"failure.json",detail),(output/"run.json",identity)):
        try: write_json(path,value)
        except Exception as diagnostic_error:
            error.add_note(f"Could not persist {path.name}: {diagnostic_error}")

def resolve_config(config,pipeline,sequence,fixture):
    config=copy.deepcopy(config)
    if config.get("contract_id")!=CONTRACT_ID or type(config.get("schema_version")) is not int or config["schema_version"]!=1:
        raise ValueError("Run config contract/schema mismatch")
    if pipeline not in PIPELINE_IDS: raise ValueError("Unknown pipeline ID")
    if config.get("pipeline_id",pipeline)!=pipeline: raise ValueError("Config pipeline identity mismatch")
    if type(config.get("fixture",False)) is not bool or fixture!=sequence["fixture"] or config.get("fixture",False)!=fixture: raise ValueError("Fixture config/sequence/CLI identity mismatch")
    adapter_section=config.get("pipeline") if isinstance(config.get("pipeline"),dict) else None
    config.update(pipeline_id=pipeline,fixture=fixture)
    config.setdefault("protocol_id","semantic_mapping_eval_v1")
    config.setdefault("mode","quality_replay")
    if config["mode"] not in {"quality_replay","paced_runtime"}: raise ValueError("Invalid replay mode")
    config["sequence"]={k:sequence[k] for k in ("sequence_id","split","manifest_digest")}
    config["sequence"]["frame_count"]=len(sequence["frames"])
    config.setdefault("geometry",{})
    defaults={"image_size":518,"keyframe_interval":1,"min_confidence":1.5,"source_root":"src/lingbot-map","device":"cuda:0","map_frame":"lingbot_world","pose_revision":"initial","scale":{},"up":{}}
    config["geometry"]={**defaults,**config["geometry"]}
    config["fusion"]={"voxel_size":.05,"origin":[0,0,0],"semantic_expiry_ns":None,"negative_evidence":True,"semantic_threshold":.5,**config.get("fusion",{})}
    threshold=config["fusion"]["semantic_threshold"]
    if type(threshold) not in (int,float) or not np.isfinite(threshold) or not 0<=threshold<=1: raise ValueError("Invalid frozen semantic evidence threshold")
    config.setdefault("robot",None);config.setdefault("planning",{});config.setdefault("goals",[])
    config.setdefault("evaluation_grid",None);config.setdefault("hardware_budget",{"device":"cuda:0","max_visible_gpus":1,"measured":False})
    config.setdefault("semantic_keyframe_ids",[r["frame_id"] for r in sequence["frames"]])
    known={r["frame_id"] for r in sequence["frames"]};keys=config["semantic_keyframe_ids"]
    if len(keys)!=len(set(keys)) or not set(keys)<=known: raise ValueError("Invalid frozen keyframe IDs")
    config["runtime"]={"playback_rate":1.0,"cadence_frames":1,"pending_capacity":1,"expiry_ns":None,**config.get("runtime",{})}
    rt=config["runtime"]
    if rt["pending_capacity"]!=1 or type(rt["cadence_frames"]) is not int or rt["cadence_frames"]<1 or not np.isfinite(rt["playback_rate"]) or rt["playback_rate"]<=0:
        raise ValueError("Invalid common scheduler settings")
    if rt["expiry_ns"] is not None:
        if config["fusion"]["semantic_expiry_ns"] not in (None,rt["expiry_ns"]): raise ValueError("Runtime/fusion expiry disagreement")
        config["fusion"]["semantic_expiry_ns"]=rt["expiry_ns"]
    # Adapter settings are pipeline-owned; support configs as standalone sections.
    adapter=config.get("pipeline_config",adapter_section or config.get("semantic",{}))
    if isinstance(config.get("pipelines"),dict): adapter=config["pipelines"].get(pipeline,adapter)
    config["pipeline_config"]={**adapter,"fixture":fixture,"pipeline_id":pipeline}
    if not fixture and pipeline!="geometry_only":
        model_devices=[config["pipeline_config"].get(key,{}).get("device","cuda:0") for key in
            (("qwen_settings","sam_settings") if pipeline=="qwen_hazards" else (("sam3",) if pipeline=="ground_surface" else ("sam_settings",)))]
        if any(device!=config["geometry"]["device"] for device in model_devices): raise ValueError("All common/pipeline models must use one explicitly matching visible device")
    config["config_digest"]=digest_json({k:v for k,v in config.items() if k!="config_digest"})
    return config

def semantic_record(result,output):
    meta={field.name:getattr(result,field.name) for field in fields(result) if field.name!="queries"}
    if result.query_count is None and not meta.get("query_count_reason"):
        meta["query_count_reason"]=result.adapter_provenance.get("query_counts",{}).get("unavailable_reason") or str(result.error) or "adapter declared count unauditable"
    meta["queries"]=[]
    for qindex,query in enumerate(result.queries):
        q={field.name:getattr(query,field.name) for field in fields(query) if field.name!="instances"};q["instances"]=[]
        for iindex,instance in enumerate(query.instances):
            mask_id=digest_json({"frame":result.frame_id,"query":query.query_id,"instance":instance.observation_id})
            relative=f"semantics/masks/{mask_id}.npz";write_npz(Path(output)/relative,mask=instance.mask)
            q["instances"].append({"observation_id":instance.observation_id,"score":instance.score,
                "score_meaning":instance.score_meaning,"processed_grid_id":result.processed_grid_id,
                "mask_path":relative,"mask_key":"mask","mask_sha256":file_sha256(Path(output)/relative),"metadata":instance.metadata})
        meta["queries"].append(q)
    return meta

def _status_frame(frame,status,error=None):
    return SemanticFrame(sequence_id=frame.sequence_id,frame_id=frame.frame_id,
        processed_grid_id=frame.processed_grid_id,decoded_rgb_sha256=frame.decoded_rgb_sha256,
        geometry_fingerprint=frame.geometry.geometry_fingerprint if frame.geometry else None,
        adapter_provenance={"common_scheduler":True},status=status,error=error,timestamp_ns=frame.timestamp_ns,
        query_count=None if status=="error" else 0,query_count_reason="adapter result unavailable; text executions unauditable" if status=="error" else None)

def _run(pipeline,sequence_path,config_path,output,*,identity,config,fixture=False,cache=None):
    from .fusion import VoxelFuser
    from .planning import build_costmap,plan_requests
    output=Path(output).resolve()
    sequence=validate_sequence(sequence_path);resolved=resolve_config(config,pipeline,sequence,fixture)
    for sub in ("geometry","semantics/masks","map","planning"): (output/sub).mkdir(parents=True,exist_ok=True)
    identity.update({"contract_id":CONTRACT_ID,"schema_version":1,"protocol_id":resolved["protocol_id"],
        "pipeline_id":pipeline,"fixture":fixture,"mode":resolved["mode"],"status":"running",
        "sequence":resolved["sequence"],"requested_config_digest":digest_json(config),
        "initial_config_digest":resolved["config_digest"],"stage":"geometry",
        "capabilities":{"semantic":pipeline!="geometry_only","free_space":False,"physical_navigation_validated":False}})
    write_json(output/"run.json",identity);write_json(output/"config.resolved.json",resolved)
    provenance={"contract_id":CONTRACT_ID,"schema_version":1,
        "requested_config_digest":identity["requested_config_digest"],"sequence":resolved["sequence"],
        "recording":sequence.get("input_provenance",{}),"hardware_budget":resolved["hardware_budget"],
        "declared":resolved.get("provenance",{}),"software":{"python":sys.version,"numpy":np.__version__,"platform":platform.platform()},
        "project_code":project_code_identity(),
        "inputs":{"config":{"path":str(Path(config_path).resolve()),"sha256":file_sha256(config_path)},
            "sequence":{"path":str(Path(sequence_path).resolve()),"sha256":file_sha256(sequence_path)}},
        "location_note":"Input paths are machine-local locators; hashes and identities are authoritative."}
    write_json(output/"provenance.json",provenance)
    events=[]; semantic_rows={};dispositions={};timing={"geometry_stage_ms":[],"semantic_call_ms":[],"semantic_evidence_age_ms":[],"semantic_observe_attempt_ms":[]};synchronization=[]
    fuser=VoxelFuser(resolved["fusion"]);adapter=None;worker=None;started=time.monotonic_ns();packets=[];manifest={};unavailable=None;origin=None
    geometry_started=time.monotonic_ns()
    try:
        packets,manifest=geometry_cache(sequence_path,sequence,resolved["geometry"],cache or output/"geometry/cache",fixture=fixture,reuse=cache is not None)
        duration=(time.monotonic_ns()-geometry_started)/1e6
        events.append({"event":"geometry_stage","duration_ms":duration,"scope":"cache_load" if cache else "sequence_batch_reconstruction","cached":cache is not None})
        write_jsonl(output/"events.jsonl",events)
        # Batch sequence timing is not a collection of per-frame latencies.
        resolved["geometry_identity"]={"geometry_fingerprint":manifest["geometry_fingerprint"],"input_fingerprint":manifest["input_fingerprint"],"processed_grid_id":manifest["processed_grid_id"],"scale":manifest["scale"],"up":manifest["up"],"units":manifest["units"],"pose_revision":manifest["pose_revision"]}
        resolved["config_digest"]=digest_json({k:v for k,v in resolved.items() if k!="config_digest"})
        write_json(output/"config.resolved.json",resolved)
        cache_path=Path(cache or output/"geometry/cache").resolve()
        cache_reference={"kind":"run_relative","path":cache_path.relative_to(output).as_posix()} if cache_path.is_relative_to(output) else {"kind":"external","path":str(cache_path)}
        write_json(output/"geometry/manifest.json",{**manifest,"cache_path":str(cache_path),"cache_reference":cache_reference,"reuse":cache is not None})
        for frame in packets: fuser.add_geometry(frame)
        identity.update(stage="semantics",geometry_identity=resolved["geometry_identity"],config_digest=resolved["config_digest"])
        write_json(output/"run.json",identity)
        if fixture:
            from .fixture_adapters import create_fixture_adapter
            adapter=create_fixture_adapter(pipeline,resolved["pipeline_config"],sequence)
        else:
            from pipelines import create_adapter
            adapter=create_adapter(pipeline,resolved["pipeline_config"])
        def consume(record):
            frame=record["frame"]
            if "error" in record: result=_status_frame(frame,"error",str(record["error"]))
            else: result=record["result"]
            validate_semantic(result,frame)
            if result.status in {"ok","partial"}: fuser.add_semantics(result,frame)
            fusion_done=time.monotonic_ns()
            measured=result.adapter_provenance.get("timing",{})
            if pipeline!="geometry_only":
                attempt_ms=(record["completed_monotonic_ns"]-record["started_monotonic_ns"])/1e6
                timing["semantic_observe_attempt_ms"].append(attempt_ms)
                audited_call=measured.get("semantic_call_ms",measured.get("sam_call_ms",measured.get("model_calls_ms")))
                if audited_call is not None or fixture: timing["semantic_call_ms"].append(audited_call if audited_call is not None else attempt_ms)
                stages=result.adapter_provenance.get("stage_timestamps",{})
                actual_stages=[stages[k] for k in ("qwen","sam3") if k in stages]
                audited_sync=bool(actual_stages) and all(s.get("synchronized") is True for s in actual_stages)
                synchronization.append(bool(measured.get("gpu_synchronized",measured.get("synchronized",audited_sync))))
                if measured.get("loading_ms") is not None: events.append({"event":"model_loading","frame_id":frame.frame_id,"duration_ms":measured["loading_ms"],"scope":"audited_adapter_loading"})
            capture=record.get("mapped_capture_monotonic_ns")
            if capture is not None and pipeline!="geometry_only": timing["semantic_evidence_age_ms"].append((fusion_done-capture)/1e6)
            semantic_rows[frame.frame_id]=semantic_record(result,output)
            dispositions[frame.frame_id]={"frame_id":frame.frame_id,"status":"error" if result.status=="error" else "ok", "semantic_status":result.status,"geometry_status":"ok","timestamp_ns":frame.timestamp_ns,"processed_grid_id":frame.processed_grid_id,"error":result.error}
            events.append({"event":"semantic_fusion","frame_id":frame.frame_id,"status":result.status,
                "fusion_monotonic_ns":fusion_done,"mapped_capture_monotonic_ns":capture})
        if resolved["mode"]=="quality_replay":
            for frame in packets:
                if pipeline!="geometry_only" and frame.frame_id not in resolved["semantic_keyframe_ids"]:
                    semantic_rows[frame.frame_id]=semantic_record(_status_frame(frame,"skipped","not_a_frozen_semantic_keyframe"),output)
                    dispositions[frame.frame_id]={"frame_id":frame.frame_id,"status":"ok","semantic_status":"skipped","geometry_status":"ok"};continue
                call_start=time.monotonic_ns()
                try: result=adapter.observe(frame);record={"frame":frame,"result":result}
                except Exception as error: record={"frame":frame,"error":error}
                record.update(started_monotonic_ns=call_start,completed_monotonic_ns=time.monotonic_ns());consume(record)
        else:
            worker=LatestPendingWorker(adapter);origin=time.monotonic_ns()
            known=[p.timestamp_ns for p in packets if p.timestamp_ns is not None];source_origin=known[0] if known else 0
            for index,frame in enumerate(packets):
                capture=mapped_capture(frame,source_origin,origin,resolved["runtime"]["playback_rate"])
                availability=origin+int((frame.timestamp_ns-source_origin)/resolved["runtime"]["playback_rate"]) if frame.timestamp_ns is not None else None
                if availability is not None:
                    remaining=(availability-time.monotonic_ns())/1e9
                    if remaining>0: time.sleep(remaining)
                record=worker.poll()
                if record: consume(record)
                if pipeline=="geometry_only" or (frame.frame_id in resolved["semantic_keyframe_ids"] and index%resolved["runtime"]["cadence_frames"]==0): worker.submit(frame,capture)
                else:
                    semantic_rows[frame.frame_id]=semantic_record(_status_frame(frame,"skipped","runtime_cadence"),output)
                    dispositions[frame.frame_id]={"frame_id":frame.frame_id,"status":"ok","semantic_status":"skipped","geometry_status":"ok"}
            for record in worker.drain(): consume(record)
            events.extend(worker.events)
            write_jsonl(output/"events.jsonl",events)
            for drop in worker.events:
                frame=next(p for p in packets if p.frame_id==drop["frame_id"])
                semantic_rows[frame.frame_id]=semantic_record(_status_frame(frame,"skipped",drop["reason"]),output)
                dispositions[frame.frame_id]={"frame_id":frame.frame_id,"status":"ok","semantic_status":"queue_dropped","geometry_status":"ok"}
    except (FileNotFoundError,ImportError,RuntimeError) as error:
        unavailable=str(error);identity.update(status="unavailable",reason=unavailable)
        events.append({"event":"unavailable","error":unavailable,"error_type":type(error).__name__,"stage":identity["stage"]})
    except Exception as error:
        unavailable=str(error);identity.update(status="failed",reason=unavailable)
        events.append({"event":"failed","error":unavailable,"error_type":type(error).__name__,"stage":identity["stage"]})
    finally:
        active_error=sys.exc_info()[1];cleanup_interrupt=None
        for resource in (worker,adapter):
            if resource is not None:
                try: resource.close()
                except Exception as error:
                    unavailable=unavailable or str(error)
                    identity.update(status="failed",reason=unavailable)
                    events.append({"event":"cleanup_error","error":str(error),"error_type":type(error).__name__})
                except BaseException as error:
                    if active_error is not None: active_error.add_note(f"Cleanup interrupted: {error}")
                    else: cleanup_interrupt=cleanup_interrupt or error
        try: write_jsonl(output/"events.jsonl",events)
        except Exception as diagnostic_error:
            primary_error=active_error or cleanup_interrupt
            if primary_error is None: raise
            primary_error.add_note(f"Could not persist events.jsonl: {diagnostic_error}")
        if cleanup_interrupt is not None: raise cleanup_interrupt
    identity.update(stage="finalizing");write_json(output/"run.json",{**identity,"status":"finalizing"})
    for row in sequence["frames"]:
        if row["frame_id"] not in dispositions:
            dispositions[row["frame_id"]]={"frame_id":row["frame_id"],"status":"error","geometry_status":"unavailable" if not packets else "ok","semantic_status":"unavailable","reason":unavailable or "absent required attempt"}
    for frame in packets:
        if frame.frame_id not in semantic_rows: semantic_rows[frame.frame_id]=semantic_record(_status_frame(frame,"error",unavailable),output)
    for row in sequence["frames"]:
        if row["frame_id"] not in semantic_rows:
            semantic_rows[row["frame_id"]]={"contract_id":CONTRACT_ID,"schema_version":1,"sequence_id":sequence["sequence_id"],"frame_id":row["frame_id"],
                "processed_grid_id":None,"decoded_rgb_sha256":None,"geometry_fingerprint":None,"adapter_provenance":{"availability":"unavailable","reason":"processed geometry frame unavailable"},
                "status":"error","queries":[],"model_calls":{},"query_count":0,"timestamp_ns":row["timestamp_ns"],"error":unavailable}
    write_jsonl(output/"frames.jsonl",[dispositions[r["frame_id"]] for r in sequence["frames"]])
    write_jsonl(output/"semantics/frames.jsonl",[semantic_rows[r["frame_id"]] for r in sequence["frames"] if r["frame_id"] in semantic_rows])
    capture_now=max((p.timestamp_ns for p in packets if p.timestamp_ns is not None),default=0)
    if resolved["mode"]=="paced_runtime" and packets and origin is not None:
        capture_now=max(capture_now,source_origin+int((time.monotonic_ns()-origin)*resolved["runtime"]["playback_rate"]))
    voxels,evidence,concepts,contributions,map_metadata=fuser.export(now_timestamp_ns=capture_now)
    write_npz(output/"map/voxels.npz",**voxels);write_npz(output/"map/semantic_evidence.npz",**evidence)
    write_json(output/"map/concepts.json",concepts);write_jsonl(output/"map/contributions.jsonl",contributions)
    write_json(output/"map/manifest.json",{"contract_id":CONTRACT_ID,"schema_version":1,**map_metadata,"map_frame":manifest.get("map_frame"),"units":manifest.get("units"),"up":manifest.get("up"),"scale":manifest.get("scale"),"geometry_fingerprint":manifest.get("geometry_fingerprint"),"free_space_capability":False})
    planning_config={"planning":resolved["planning"],"robot":resolved["robot"]}
    semantic_points={}
    for cindex,concept in enumerate(concepts):
        positive=(evidence["concept_row"]==cindex)&(evidence["evidence_score"]>=resolved["fusion"]["semantic_threshold"])
        semantic_points[concept["concept_id"]]={"role":concept["role"],"points":voxels["centers"][evidence["voxel_row"][positive]]}
    costmap,planning_meta=build_costmap(fuser.observed_points(),planning_config,units=manifest.get("units","reconstruction_units"),up=manifest.get("up"),scale=manifest.get("scale"),semantic_points=semantic_points,terrain_voxels=voxels,voxel_size=map_metadata["voxel_size"])
    plans=plan_requests(costmap,planning_meta,resolved["goals"],manifest.get("map_frame",resolved["geometry"]["map_frame"]))
    write_npz(output/"planning/costmap.npz",**costmap);write_json(output/"planning/manifest.json",planning_meta);write_jsonl(output/"planning/plans.jsonl",plans)
    if packets:
        from path_mapping.artifacts import _write_ply
        points=fuser.observed_points();_write_ply(output/"map/raw_surfaces.ply",points,np.full(points.shape,145,np.uint8))
        write_npz(output/"geometry/camera_trajectory.npz",world_to_camera=np.stack([p.geometry.world_to_camera for p in packets]),intrinsics=np.stack([p.geometry.intrinsics for p in packets]),timestamp_ns=np.array([p.timestamp_ns if p.timestamp_ns is not None else -1 for p in packets],np.int64))
    if not (output/"geometry/manifest.json").exists(): write_json(output/"geometry/manifest.json",{"contract_id":CONTRACT_ID,"schema_version":1,"status":"unavailable","reason":unavailable})
    events.append({"event":"planning","availability":planning_meta["availability"],"reason":planning_meta.get("reason")})
    write_jsonl(output/"events.jsonl",events)
    successful=sum(d["status"]=="ok" for d in dispositions.values());elapsed=(time.monotonic_ns()-started)/1e9
    summary={"contract_id":CONTRACT_ID,"schema_version":1,"fixture":fixture,"models_executed":not fixture and ((cache is None and bool(packets)) or any(sum(r.get("model_calls",{}).values()) for r in semantic_rows.values())),
        "required_frame_attempts":len(sequence["frames"]),"counts":{"frames":len(sequence["frames"]),"successful_frames":successful,"failed_frames":len(sequence["frames"])-successful,"voxels":len(voxels["centers"]),"semantic_rows":len(evidence["voxel_row"]),"queue_drops":sum(e.get("event")=="queue_drop" for e in events)},
        "run_duration_s":elapsed,"timing_samples":timing,"timing_metadata":{"semantic_call_ms":{"clock":"monotonic_ns","timing_boundaries":"audited model calls excluding loading; fixture software calls explicitly synthetic","scope":"fixture_cpu" if fixture else "staged_replay","gpu_work":not fixture,"synchronized":all(synchronization) if synchronization else None},"semantic_evidence_age_ms":{"clock":"mapped_capture_monotonic_ns","timing_boundaries":"mapped replay availability to semantic fusion","scope":"cached_geometry_scheduler_replay"}},
        "eligible_replay_duration_s":None,"planning":planning_meta,"operational_metrics":{},
        "measurement_scope":"cached_geometry_scheduler_replay" if resolved["mode"]=="paced_runtime" else "staged_quality_replay",
        "geometry_diagnostics":{"alignment":manifest.get("alignment"),"valid_surface_pixels":sum(int(p.geometry.validity.sum()) for p in packets),"processed_pixels":sum(p.rgb.shape[0]*p.rgb.shape[1] for p in packets)},
        "limitations":["2.5D static planning is an internal diagnostic","surface-only mapping emits no free-space evidence","batch LingBot reconstruction means paced scheduling is staged; live end-to-end speed and joint model memory remain unavailable"]}
    write_json(output/"summary.json",summary)
    if unavailable is None:
        identity.update(status="complete" if successful else "failed",planning_status=planning_meta["availability"])
        if not successful:
            import json
            problems=" ".join(json.dumps(r.get("error"),default=str) for r in semantic_rows.values()).casefold()
            availability_codes=("unavailable","checkpoint_missing","dependency_missing","model_loading_disabled","cuda_device","not cached","missing requested model")
            if any(code in problems for code in availability_codes): identity["status"]="unavailable"
            identity["reason"]="all required semantic observations failed; inspect explicit frame errors"
    identity.update(geometry_identity=resolved.get("geometry_identity"),config_digest=resolved["config_digest"],elapsed_s=elapsed)
    provenance.update(config_digest=resolved["config_digest"],geometry_identity=resolved.get("geometry_identity"))
    provenance_identity={key:provenance.get(key) for key in ("contract_id","schema_version",
        "requested_config_digest","sequence","hardware_budget","declared","software","project_code","config_digest","geometry_identity")}
    provenance_identity["source_hashes"]={key:value["sha256"] for key,value in provenance["inputs"].items()}
    identity["provenance_id"]=digest_json(provenance_identity)
    provenance["provenance_id"]=identity["provenance_id"];write_json(output/"provenance.json",provenance)
    write_json(output/"run.json",{**identity,"status":"finalizing"})
    return identity

def run(pipeline,sequence_path,config_path,output,*,fixture=False,cache=None):
    from .resources import ResourceCollector
    output=Path(output).resolve()
    _reserve_output(output)
    identity={"contract_id":CONTRACT_ID,"schema_version":1,"pipeline_id":pipeline,"fixture":fixture,
        "attempt_id":uuid.uuid4().hex,"status":"running","stage":"validation",
        "started_at_utc":datetime.now(timezone.utc).isoformat(),"artifact_root":"."}
    collector=None
    try:
        write_json(output/"run.json",identity)
        config=read_json(config_path);write_json(output/"config.requested.json",config)
        configured_device=config.get("geometry",{}).get("device","cuda:0")
        collector=ResourceCollector(device=configured_device if isinstance(configured_device,str) and configured_device.startswith("cuda:") else "cuda:0")
        collector.start()
        result=_run(pipeline,sequence_path,config_path,output,fixture=fixture,cache=cache,identity=identity,config=config)
        identity.update(stage="resources")
        metrics=collector.stop()
        summary=read_json(output/"summary.json");summary["operational_metrics"].update(metrics)
        write_json(output/"summary.json",summary)
        result.update(stage="finished",finished_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(output/"run.json",result)
        return result
    except BaseException as error:
        _failure(output,identity,error)
        raise
    finally:
        if collector is not None:
            try: collector.stop()
            except Exception: pass  # The original failure remains authoritative.
