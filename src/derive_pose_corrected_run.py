"""Derive a separately identified map from saved depth and C2W pose evidence.

No model is loaded. Original inputs are admitted by the unchanged export loader
and retained byte-for-byte. Mask arrays retain their exact processed RGB grid;
only their derived geometry association changes with explicit provenance.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import numpy as np
import cv2
from export_pipeline_video import load_export_inputs, load_npz_checked, sha256
from path_mapping.runner import GEOMETRY_KEYS, geometry_fingerprint, validate_geometry, reprojection_diagnostic
from pipeline_common.io import write_json,write_jsonl,write_npz,digest_json
from pipeline_common.contracts import GeometryFrame,FramePacket,SemanticFrame,QueryRecord,InstanceRecord
from pipeline_common.fusion import VoxelFuser
from pipeline_common.planning import build_costmap


def pose_track_evidence(g):
    """Independent cross-frame visual feature evidence, not self-projection."""
    images=g['images'];E=g['extrinsic'];K=g['intrinsic'];D=g['depth'].reshape(images.shape[:-1])
    n=len(images); pairs=sorted(set((i,i+1) for i in np.linspace(0,n-2,min(5,n-1),dtype=int)))
    rows=[]
    for i,j in pairs:
        gray=[cv2.cvtColor(images[t],cv2.COLOR_RGB2GRAY) for t in (i,j)]
        xy=cv2.goodFeaturesToTrack(gray[0],maxCorners=800,qualityLevel=.02,minDistance=8)
        if xy is None:continue
        dest,st,_=cv2.calcOpticalFlowPyrLK(*gray,xy,None,winSize=(31,31),maxLevel=4)
        back,bst,_=cv2.calcOpticalFlowPyrLK(gray[1],gray[0],dest,None,winSize=(31,31),maxLevel=4)
        ok=(st[:,0]>0)&(bst[:,0]>0)&(np.linalg.norm(back-xy,axis=(1,2))<.7)
        xy,dest=xy[:,0][ok],dest[:,0][ok]
        ix=np.rint(xy).astype(int);d=D[i,ix[:,1],ix[:,0]]
        valid=(g['world_points_conf'][i,ix[:,1],ix[:,0]]>=1.5)&np.isfinite(d)&(d>0)
        xy,d,dest=xy[valid],d[valid],dest[valid]
        if len(xy)<30:continue
        local=np.column_stack([xy,np.ones(len(xy))])@np.linalg.inv(K[i]).T*d[:,None]
        row={'frames':[int(i),int(j)],'matched_features':len(xy),
             'median_pixel_motion':float(np.median(np.linalg.norm(xy-dest,axis=1)))}
        for name in ('saved_w2c','candidate_c2w'):
            if name=='saved_w2c': world=(local-E[i,:,3])@E[i,:,:3];cam=world@E[j,:,:3].T+E[j,:,3]
            else:world=local@E[i,:,:3].T+E[i,:,3];cam=(world-E[j,:,3])@E[j,:,:3]
            p=cam@K[j].T;error=np.linalg.norm(p[:,:2]/p[:,2:]-dest,axis=1)
            row[name]={'median_error_pixels':float(np.median(error)),'p90_error_pixels':float(np.percentile(error,90))}
        rows.append(row)
    decisive=[row for row in rows if row['median_pixel_motion']>3 and
              row['candidate_c2w']['median_error_pixels']<8 and
              row['saved_w2c']['median_error_pixels']>3*max(row['candidate_c2w']['median_error_pixels'],1.)]
    contradict=[row for row in rows if row['median_pixel_motion']>3 and
                row['candidate_c2w']['median_error_pixels']>2*max(row['saved_w2c']['median_error_pixels'],1.)]
    return {'method':'forward_backward_KLT_saved_depth_cross_frame_reprojection',
            'pairs':rows,'decisive_c2w_pairs':len(decisive),'contradictory_pairs':len(contradict),
            'correction_supported':len(decisive)>=2 and not contradict,
            'physical_calibration_verified':False}


def corrected_numeric(g):
    """Invert decoded C2W poses and unproject unchanged optical depth directly."""
    out={k:v.copy() for k,v in g.items()}
    e=g['extrinsic'].astype(np.float64); R=e[:,:,:3];t=e[:,:,3]
    w2c=np.empty_like(e);w2c[:,:,:3]=R.transpose(0,2,1);w2c[:,:,3]=-np.einsum('sji,sj->si',R,t)
    h,w=g['images'].shape[1:3];y,x=np.meshgrid(np.arange(h),np.arange(w),indexing='ij')
    pixels=np.stack([x,y,np.ones((h,w))],axis=-1)
    local=np.einsum('sij,hwj->shwi',np.linalg.inv(g['intrinsic'].astype(float)),pixels)*g['depth'].reshape(len(e),h,w,1)
    world=np.einsum('sij,shwj->shwi',R,local)+t[:,None,None,:]
    out['extrinsic']=w2c.astype(g['extrinsic'].dtype)
    out['world_points']=world.astype(g['world_points'].dtype)
    validate_geometry(out,allow_single_frame=True)
    return out


CORRECTED_METHOD = 'decode_saved_pose_as_c2w_invert_to_w2c_unproject_saved_optical_depth'
UNVERIFIED_DISPLAY_METHOD = 'saved_pose_passthrough_unverified_display_only'


def numeric_for_derivation(g, provenance):
    """Reproduce the admitted numeric derivation, including legacy verified copies."""
    method = provenance.get('method', CORRECTED_METHOD)
    verified = provenance.get('pose_correction_verified', True)
    if method == CORRECTED_METHOD and verified is True:
        return corrected_numeric(g)
    if method == UNVERIFIED_DISPLAY_METHOD and verified is False:
        return {key: value.copy() for key, value in g.items()}
    raise ValueError('Unsupported or contradictory pose derivation provenance')


def derivation_policy(evidence, *, allow_unverified_display_copy=False):
    """A display fallback cannot turn missing pose evidence into a correction."""
    if evidence.get('correction_supported') is True:
        return {'method': CORRECTED_METHOD, 'pose_correction_verified': True,
                'planning_admitted': True}
    if not allow_unverified_display_copy:
        raise ValueError('Saved C2W correction lacks sufficient independent cross-frame evidence: '+str(evidence))
    return {'method': UNVERIFIED_DISPLAY_METHOD, 'pose_correction_verified': False,
            'planning_admitted': False}


def derive(run,sequence,cache,output, *, allow_unverified_display_copy=False):
    run,sequence,cache,output=map(lambda x:Path(x).resolve(),(run,sequence,cache,output))
    if output.exists() or output.is_relative_to(run) or output.is_relative_to(cache):raise ValueError('Fresh separate derived output required')
    data=load_export_inputs(run,sequence,cache_dir=cache)
    g=load_npz_checked(cache/'geometry.npz',max_bytes=1024**3,names=GEOMETRY_KEYS)
    evidence=pose_track_evidence(g)
    policy=derivation_policy(evidence,allow_unverified_display_copy=allow_unverified_display_copy)
    corrected=numeric_for_derivation(g,policy);fingerprint=geometry_fingerprint(corrected)
    output.mkdir(parents=True)
    dst=output/'derived_run';dst.mkdir()
    newcache=dst/'geometry/cache';newcache.mkdir(parents=True)
    identity=deepcopy(data['geometry'])
    revision=('saved_c2w_interpretation_corrected_v1' if policy['pose_correction_verified']
              else 'saved_pose_unverified_display_only_v1')
    provenance={'artifact_kind':'derived_pose_convention_correction','source_run':str(run),
        'source_geometry_fingerprint':data['geometry']['geometry_fingerprint'],
        'source_archive_sha256':data['geometry']['archive_sha256'],
        'source_manifest_sha256':sha256(run/'geometry/manifest.json'),
        **policy,
        'models_executed':False,'mask_pixels_changed':False,'rgb_pixels_changed':False,
        'depth_changed':False,'intrinsics_changed':False,'cross_frame_evidence':evidence,
        'helper_sha256':sha256(__file__)}
    write_npz(newcache/'geometry.npz',**corrected)
    identity.update(geometry_fingerprint=fingerprint,archive_sha256=sha256(newcache/'geometry.npz'),
        pose_revision=revision,cache_path=str(newcache),alignment=reprojection_diagnostic(corrected,1.5),
        derivation=provenance)
    write_json(newcache/'manifest.json',identity);write_json(dst/'geometry/manifest.json',identity)
    shutil.copytree(cache/'processed_frames',newcache/'processed_frames')
    poses=np.broadcast_to(np.eye(4),(len(g['images']),4,4)).copy();poses[:,:3]=corrected['extrinsic']
    write_npz(dst/'geometry/camera_trajectory.npz',world_to_camera=poses,intrinsics=corrected['intrinsic'],
        timestamp_ns=np.array([row['timestamp_ns'] if row['timestamp_ns'] is not None else -1 for row in data['frames']],dtype=np.int64))
    config=deepcopy(data['config']);run_meta=deepcopy(data['run'])
    for record in (config,run_meta):
        record['geometry_identity'].update(geometry_fingerprint=fingerprint,pose_revision=revision)
        record['derivation']=provenance
    config['config_digest']=digest_json({k:v for k,v in config.items() if k!='config_digest'})
    run_meta.update(config_digest=config['config_digest'],models_executed=False)
    write_json(dst/'config.resolved.json',config);write_json(dst/'run.json',run_meta)
    shutil.copy2(run/'frames.jsonl',dst/'frames.jsonl')
    fuser=VoxelFuser(config['fusion']);semantic_rows=[];mask_receipts=[]
    for i,frame in enumerate(data['frames']):
        xyz=corrected['world_points'][i];dep=corrected['depth'][i].reshape(xyz.shape[:-1]);conf=corrected['world_points_conf'][i]
        valid=np.isfinite(xyz).all(-1)&np.isfinite(dep)&(dep>0)&np.isfinite(conf)&(conf>=identity['settings']['min_confidence'])&(conf>0)
        l,t,r,b=frame['transform']['pad_ltrb']
        if t:valid[:t]=False
        if b:valid[-b:]=False
        if l:valid[:,:l]=False
        if r:valid[:,-r:]=False
        gf=GeometryFrame(xyz,dep,valid,conf,corrected['intrinsic'][i],poses[i],identity['processed_grid_id'],fingerprint,
                         map_frame=identity['map_frame'],units=identity['units'],up=identity['up'],scale=identity['scale'],pose_revision=revision)
        packet=FramePacket(data['sequence']['sequence_id'],frame['frame_id'],frame['timestamp_ns'],frame['source']['timestamp_provenance'],
                           frame['rgb'],newcache/'processed_frames'/f'{i:06d}.png',sha256(newcache/'processed_frames'/f'{i:06d}.png'),
                           (frame['semantic'] or {}).get('decoded_rgb_sha256'),identity['processed_grid_id'],
                           {k:frame['source'][k] for k in ('image_path','encoded_file_sha256','decoded_rgb_sha256')},frame['transform'],gf)
        fuser.add_geometry(packet)
        row=deepcopy(frame['semantic'])
        if row is None:continue
        row['geometry_fingerprint']=fingerprint
        row['derived_geometry_binding']={'source_geometry_fingerprint':data['geometry']['geometry_fingerprint'],
                                         'mask_inference_geometry_unchanged':True,
                                         'reason':('identical processed RGB/masks re-associated with corrected XYZ'
                                                   if policy['pose_correction_verified'] else
                                                   'identical processed RGB/masks and saved XYZ retained for unverified display only')}
        queries=[]
        for query in row['queries']:
            instances=[]
            for instance in query['instances']:
                mask=instance.pop('mask');src=run/instance['mask_path'];target=dst/instance['mask_path']
                target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():shutil.copy2(src,target)
                if sha256(target)!=instance['mask_sha256']:raise ValueError('Copied mask pixels differ')
                mask_receipts.append({'source':str(src),'derived':str(target),'sha256':instance['mask_sha256']})
                instances.append(InstanceRecord(instance['observation_id'],mask,instance.get('score'),instance.get('score_meaning','unavailable'),instance.get('processed_grid_id'),instance.get('metadata',{})))
            queries.append(QueryRecord(**{k:query[k] for k in ('query_id','original_phrase','concept_id','role','status')},
                error=query.get('error'),instances=instances,score_metadata=query.get('score_metadata',{}),mapping_version=query.get('mapping_version','v1')))
        sem=SemanticFrame(**{k:row[k] for k in ('sequence_id','frame_id','processed_grid_id','decoded_rgb_sha256','geometry_fingerprint','adapter_provenance','status')},
                          queries=queries,model_calls={},query_count=len(queries),timestamp_ns=row['timestamp_ns'],error=row.get('error'))
        fuser.add_semantics(sem,packet);semantic_rows.append(row)
    write_jsonl(dst/'semantics/frames.jsonl',semantic_rows)
    voxels,semantics,concepts,contributions,meta=fuser.export(now_timestamp_ns=max(f['timestamp_ns'] or 0 for f in data['frames']))
    meta.update(geometry_fingerprint=fingerprint,derivation=provenance)
    write_npz(dst/'map/voxels.npz',**voxels);write_npz(dst/'map/semantic_evidence.npz',**semantics)
    write_json(dst/'map/manifest.json',meta);write_json(dst/'map/concepts.json',concepts);write_jsonl(dst/'map/contributions.jsonl',contributions)
    arrays,planning=build_costmap(np.empty((0,3)),config,units=identity['units'],up=identity['up'],scale=identity['scale'])
    write_npz(dst/'planning/costmap.npz',**arrays);write_json(dst/'planning/manifest.json',planning);write_jsonl(dst/'planning/plans.jsonl',[])
    # Admit the complete derived closure through the same unchanged strict loader.
    check=load_export_inputs(dst,sequence,cache_dir=newcache)
    for path,digest in data['input_sha256'].items():
        if sha256(path)!=digest:raise ValueError('Original input changed during derivation')
    write_json(output/'derivation_receipt.json',{'status':'complete',**provenance,
        'geometry_fingerprint':fingerprint,'mask_copies':mask_receipts,'original_input_sha256':data['input_sha256'],
        'derived_input_sha256':check['input_sha256'],'voxels':len(voxels['centers'])})
    return dst


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('run','sequence','cache','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--allow-unverified-display-copy',action='store_true',
        help='Preserve original poses/XYZ for display only when correction evidence is insufficient; planning must remain blocked')
    args=parser.parse_args();print(derive(args.run,args.sequence,args.cache,args.output,
        allow_unverified_display_copy=args.allow_unverified_display_copy))
