"""Real CLI/cache/scheduler regressions without loading any model."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from pipeline_common.fixture import create_fixture,check_fixture
from pipeline_common.sequence import validate_sequence
from pipeline_common.runtime import resolve_config,run
from pipeline_common.geometry import geometry_cache,preprocessing,synthetic_geometry
from pipeline_common.scheduling import LatestPendingWorker
from pipeline_common.fixture_adapters import FixtureAdapter
from pipeline_common.contracts import SemanticFrame
from pipeline_common.io import read_json,write_json,digest_json,file_sha256

class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.fixture=create_fixture(self.root/"fixture")
        self.sequence=validate_sequence(self.fixture/"sequence/sequence.json")
        self.config=read_json(self.fixture/"run.json")
        self.resolved=resolve_config(self.config,"geometry_only",self.sequence,True)
        self.packets,self.manifest=geometry_cache(self.fixture/"sequence/sequence.json",self.sequence,self.resolved["geometry"],self.fixture/"geometry_cache",fixture=True,reuse=True)

    def test_all_four_real_cli_paths_compare_same_frozen_inputs(self):
        result=check_fixture(self.root/"all_four")
        comparison=read_json(result/"comparison/comparison.json")
        self.assertEqual(comparison["status"],"compatible")
        for name in ("geometry_only","ground_surface","fixed_hazards","qwen_hazards"):
            completed=read_json(result/"runs"/name/"run.json");self.assertEqual(completed["status"],"complete")
            rows=(result/"runs"/name/"frames.jsonl").read_text().splitlines();self.assertEqual(len(rows),3)
        with np.load(result/"runs/geometry_only/map/semantic_evidence.npz",allow_pickle=False) as arrays:
            self.assertEqual(arrays["voxel_row"].size,0)
        semantics=[json.loads(line) for line in (result/"runs/geometry_only/semantics/frames.jsonl").read_text().splitlines()]
        self.assertTrue(all(r["status"]=="not_applicable" and r["query_count"]==0 for r in semantics))

    def test_manifest_hash_file_pixels_order_and_timestamps_tampering(self):
        path=self.fixture/"sequence/sequence.json";original=read_json(path)
        for change in ("pixels","digest","timestamp","duplicate","path"):
            altered=copy.deepcopy(original)
            if change=="pixels": altered["frames"][0]["decoded_rgb_sha256"]="0"*64
            if change=="digest": altered["manifest_digest"]="0"*64
            if change=="timestamp": altered["frames"][1]["timestamp_ns"]=0
            if change=="duplicate": altered["frames"][1]["frame_id"]=altered["frames"][0]["frame_id"]
            if change=="path": altered["frames"][0]["image_path"]="../outside.png"
            if change!="digest": altered["manifest_digest"]=digest_json({k:v for k,v in altered.items() if k!="manifest_digest"})
            write_json(path,altered)
            with self.subTest(change=change),self.assertRaises(ValueError): validate_sequence(path)
        write_json(path,original)

    def test_cache_archive_metadata_and_source_identity_refused(self):
        manifest_path=self.fixture/"geometry_cache/manifest.json";original=read_json(manifest_path)
        modified=copy.deepcopy(original);modified["input_identity"]["settings"]["pose_revision"]="tampered"
        write_json(manifest_path,modified)
        with self.assertRaisesRegex(ValueError,"identity"): geometry_cache(self.fixture/"sequence/sequence.json",self.sequence,self.resolved["geometry"],self.fixture/"geometry_cache",fixture=True,reuse=True)
        write_json(manifest_path,original)
        archive=self.fixture/"geometry_cache/geometry.npz";archive.write_bytes(archive.read_bytes()+b"tampered")
        with self.assertRaisesRegex(ValueError,"tampering"): geometry_cache(self.fixture/"sequence/sequence.json",self.sequence,self.resolved["geometry"],self.fixture/"geometry_cache",fixture=True,reuse=True)

    def test_declared_source_revision_mismatch_fails_before_model_loading(self):
        settings={**self.resolved["geometry"],"source_revision":"0"*40}
        with self.assertRaisesRegex(ValueError,"source revision mismatch"):
            geometry_cache(self.fixture/"sequence/sequence.json",self.sequence,settings,self.root/"wrong_revision",fixture=True)

    def test_fixed_vocabulary_sam_device_matches_geometry_before_loading(self):
        config=copy.deepcopy(self.config);config["fixture"]=False
        sequence={**self.sequence,"fixture":False}
        config["geometry"]["device"]="cuda:0"
        config["pipeline_config"]={"sam_settings":{"device":"cuda:1"}}
        with self.assertRaisesRegex(ValueError,"matching visible device"):
            resolve_config(config,"fixed_hazards",sequence,False)
        config["pipeline_config"]["sam_settings"]["device"]="cuda:0"
        self.assertEqual(resolve_config(config,"fixed_hazards",sequence,False)["pipeline_config"]["sam_settings"]["device"],"cuda:0")

    def test_padding_is_excluded_and_crop_matrix_has_pixel_center_offset(self):
        folder=self.root/"shapes";folder.mkdir();first=folder/"first.png";second=folder/"second.png"
        Image.new("RGB",(28,14),(50,70,90)).save(first);Image.new("RGB",(28,42),(30,50,70)).save(second)
        images,transforms=preprocessing([first,second],28)
        self.assertEqual(images.shape,(2,28,28,3))
        self.assertEqual(transforms[0]["pad_ltrb"],[0,7,0,7])
        self.assertEqual(transforms[1]["crop_xywh"],[0,7,28,28])
        np.testing.assert_allclose(transforms[1]["matrix"],[[1,0,0],[0,1,-7],[0,0,1]])

    def test_scaled_rotated_pose_up_and_crop_geometry_agree(self):
        settings={**self.resolved["geometry"],"scale":{"meters_per_unit":2,"verified":True,"source":"independent fixture"},"up":{"vector":[0,1,0],"verified":True,"source":"independent fixture"}}
        packets,manifest=geometry_cache(self.fixture/"sequence/sequence.json",self.sequence,settings,self.root/"scaled",fixture=True)
        np.testing.assert_allclose(packets[0].geometry.points,self.packets[0].geometry.points*2)
        np.testing.assert_allclose(packets[0].geometry.depth,self.packets[0].geometry.depth*2)
        np.testing.assert_allclose(packets[0].geometry.world_to_camera[:3,3],self.packets[0].geometry.world_to_camera[:3,3]*2)
        self.assertEqual(packets[0].geometry.up,(0,1,0));self.assertEqual(manifest["units"],"metres")
        # Legacy unprojection validation with a rigid rotated camera and crop grid.
        source=self.root/"source.png";Image.new("RGB",(28,42),(30,50,70)).save(source)
        processed,_=preprocessing([source],28);geometry=synthetic_geometry(processed)
        angle=.3;rotation=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
        oldpose=geometry["extrinsic"][0].copy();newpose=oldpose[:,:3]@rotation;translation=np.array([.3,-.2,.1])
        geometry["world_points"][0]=(geometry["world_points"][0]-translation)@rotation
        geometry["extrinsic"][0,:,:3]=newpose;geometry["extrinsic"][0,:,3]=oldpose[:,3]+oldpose[:,:3]@translation
        from path_mapping.runner import reprojection_diagnostic
        self.assertLess(reprojection_diagnostic(geometry)["max_error_pixels"],1e-4)

    def test_scheduler_latest_drop_immutable_join_and_actual_call_timing(self):
        adapter=FixtureAdapter("fixed_hazards",{"delay_s":.02});worker=LatestPendingWorker(adapter);self.addCleanup(worker.close)
        frame=self.packets[0];mutable=replace(frame,rgb=frame.rgb.copy(),timestamp_provenance={"kind":"synthetic"})
        worker.submit(mutable);worker.submit(self.packets[1]);worker.submit(self.packets[2])
        mutable.rgb[:]=0;mutable.timestamp_provenance["kind"]="changed"
        time.sleep(.12);first=worker.poll()
        self.assertEqual(first["frame"].frame_id,frame.frame_id)
        self.assertEqual(first["frame"].timestamp_provenance["kind"],"synthetic")
        self.assertLess((first["completed_monotonic_ns"]-first["started_monotonic_ns"])/1e6,100)
        self.assertEqual([event["frame_id"] for event in worker.events],[self.packets[1].frame_id])
        remaining=list(worker.drain());self.assertEqual(remaining[0]["frame"].frame_id,self.packets[2].frame_id)

    def test_scheduler_rejects_wrong_inflight_frame_identity(self):
        other=self.packets[1]
        class WrongAdapter:
            def observe(self,frame): return FixtureAdapter("fixed_hazards",{}).observe(other)
        worker=LatestPendingWorker(WrongAdapter());self.addCleanup(worker.close);worker.submit(self.packets[0]);record=worker.poll(block=True)
        self.assertIn("identity mismatch",str(record["error"]))

    def test_unknown_physical_inputs_preserve_raw_map_and_block_every_request(self):
        config=copy.deepcopy(self.config);config["geometry"]["scale"]={};config["geometry"]["up"]={};config["robot"]=None
        path=self.root/"raw.json";write_json(path,config)
        result=run("geometry_only",self.fixture/"sequence/sequence.json",path,self.root/"raw",fixture=True)
        self.assertEqual(result["status"],"complete");self.assertEqual(result["planning_status"],"blocked_inputs")
        self.assertGreater((self.root/"raw/map/raw_surfaces.ply").stat().st_size,100)
        plans=[json.loads(s) for s in (self.root/"raw/planning/plans.jsonl").read_text().splitlines()]
        self.assertEqual(len(plans),1);self.assertEqual(plans[0]["status"],"blocked_inputs")
        with self.assertRaisesRegex(ValueError,"completed"): run("geometry_only",self.fixture/"sequence/sequence.json",path,self.root/"raw",fixture=True)

    def test_semantic_unavailability_does_not_discard_valid_geometry(self):
        with patch("pipeline_common.fixture_adapters.create_fixture_adapter",side_effect=RuntimeError("missing requested model")):
            result=run("fixed_hazards",self.fixture/"sequence/sequence.json",self.fixture/"run.json",self.root/"unavailable",fixture=True,cache=self.fixture/"geometry_cache")
        self.assertEqual(result["status"],"unavailable")
        with np.load(self.root/"unavailable/map/voxels.npz",allow_pickle=False) as data: self.assertGreater(len(data["centers"]),0)
        self.assertEqual(len((self.root/"unavailable/frames.jsonl").read_text().splitlines()),3)
        self.assertEqual(len((self.root/"unavailable/semantics/frames.jsonl").read_text().splitlines()),3)

    def test_paced_adapter_unavailability_preserves_raw_map_before_scheduler_starts(self):
        config=copy.deepcopy(self.config);config["mode"]="paced_runtime"
        path=self.root/"paced_missing.json";write_json(path,config)
        with patch("pipeline_common.fixture_adapters.create_fixture_adapter",side_effect=RuntimeError("missing requested model")):
            result=run("fixed_hazards",self.fixture/"sequence/sequence.json",path,self.root/"paced_missing",fixture=True,cache=self.fixture/"geometry_cache")
        self.assertEqual(result["status"],"unavailable")
        with np.load(self.root/"paced_missing/map/voxels.npz",allow_pickle=False) as data:
            self.assertGreater(len(data["centers"]),0)
        self.assertEqual(len((self.root/"paced_missing/frames.jsonl").read_text().splitlines()),3)

    def test_expiry_and_paced_drop_dispositions_remain_auditable(self):
        config=copy.deepcopy(self.config);config["mode"]="paced_runtime";config["runtime"]={"expiry_ns":1,"playback_rate":100};config["pipeline_config"]={"delay_s":.02}
        path=self.root/"paced.json";write_json(path,config)
        result=run("fixed_hazards",self.fixture/"sequence/sequence.json",path,self.root/"paced",fixture=True,cache=self.fixture/"geometry_cache")
        self.assertEqual(result["status"],"complete")
        summary=read_json(self.root/"paced/summary.json");self.assertEqual(summary["measurement_scope"],"cached_geometry_scheduler_replay")
        self.assertEqual(summary["counts"]["frames"],3);self.assertEqual(summary["counts"]["queue_drops"],1)
        with np.load(self.root/"paced/map/semantic_evidence.npz",allow_pickle=False) as arrays: self.assertEqual(len(arrays["voxel_row"]),0)

    def test_finalization_error_writes_failed_marker(self):
        config=copy.deepcopy(self.config);config["robot"]["semantic_costs"]={"hazard.water.v1":{"cost":"invalid"}}
        path=self.root/"badplan.json";write_json(path,config)
        with self.assertRaises(ValueError): run("fixed_hazards",self.fixture/"sequence/sequence.json",path,self.root/"badplan",fixture=True,cache=self.fixture/"geometry_cache")
        self.assertEqual(read_json(self.root/"badplan/run.json")["status"],"failed")

if __name__=="__main__": unittest.main()
