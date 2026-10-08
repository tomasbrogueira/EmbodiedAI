"""Narrow recording-count, source-coordinate, native capture and packaging checks."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
import json
import os
import sys
from pathlib import Path
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

import serve_lingbot_recording as capture
import export_native_recording as package


class CaptureAdmissionTests(unittest.TestCase):
    def test_exact_expected_count_rejects_short_or_long_saved_sequences(self):
        for actual in (57,59):
            with patch.object(capture,"load_export_inputs",return_value={"frames":[{}]*actual}), \
                    patch.object(capture,"load_npz_checked") as read_arrays:
                with self.assertRaisesRegex(ValueError,"exact complete"):
                    capture.prepare_inputs("run","sequence",expected_frames=58)
                read_arrays.assert_not_called()
        for invalid in (True,0,59):
            with self.assertRaisesRegex(ValueError,"exact expected count"):
                capture.prepare_inputs("run","sequence",expected_frames=invalid)

    def test_58_saved_frames_keep_original_arrays_and_bound_native_points(self):
        n,h,w = 58,20,20
        points = np.ones((n,h,w,3),np.float32)
        points[0,0,1]=np.inf
        original = points.copy()
        extrinsic = np.tile(np.column_stack([np.eye(3),np.zeros(3)]),(n,1,1)).astype(np.float32)
        arrays = {"images":np.full((n,h,w,3),128,np.uint8),"world_points":points,
                  "world_points_conf":np.full((n,h,w),1.5,np.float32),"depth":np.ones((n,h,w),np.float32),
                  "extrinsic":extrinsic,"intrinsic":np.tile(np.eye(3),(n,1,1)).astype(np.float32)}
        frames = [{"frame_id":f"f{i}","timestamp_ns":i*500000000,"source":{},
                   "transform":{"pad_ltrb":[1,0,0,0]}} for i in range(n)]
        data = {"frames":frames,"cache":Path("cache"),"geometry":{"archive_sha256":"digest"},"config":{}}
        with patch.object(capture,"load_export_inputs",return_value=data), \
                patch.object(capture,"load_npz_checked",return_value=arrays), \
                patch.object(capture,"sha256",return_value="digest"), \
                patch.object(capture,"cumulative_states",return_value=[]), \
                patch.object(capture,"MAX_POINTS",580):
            admitted = capture.prepare_inputs("run","sequence",expected_frames=n)
        np.testing.assert_array_equal(points,original)
        np.testing.assert_array_equal(admitted["native_pred"]["extrinsic"],extrinsic)
        self.assertTrue(np.isnan(admitted["native_pred"]["world_points"][:,:,0]).all())
        self.assertTrue(np.isnan(admitted["native_pred"]["world_points"][0,0,1]).all())
        self.assertEqual(admitted["valid_point_counts"],[379]+[380]*57)
        factor = admitted["native_downsample"]
        self.assertLessEqual(sum((count+factor-1)//factor for count in admitted["valid_point_counts"]),580)
        self.assertLess(admitted["native_threshold"],1.5)

    def test_first_camera_side_view_does_not_reverse_or_mutate_xyz(self):
        points = np.array([[x,y,z] for x in (-2.,2.) for y in (-1.,1.) for z in (5.,10.)])
        saved = points.copy()
        viewer = types.SimpleNamespace(all_steps=[0],vis_pts_list=[points],
                                       cam_dict={"R":{0:np.eye(3)},"t":{0:np.zeros(3)}})
        pose = capture.native_display_pose(viewer)
        np.testing.assert_array_equal(points,saved)
        self.assertLess(pose["position"][2],pose["look_at"][2])
        self.assertEqual(pose["display_yaw_degrees"],12)
        np.testing.assert_allclose(pose["up_direction"],[0,-1,0])
        self.assertFalse(pose["geometry_changed"])

    def test_registry_tracks_native_buttons_stored_only_in_callback_closures(self):
        gui = types.SimpleNamespace(add_button=lambda *_:types.SimpleNamespace(disabled=False),
                                    add_slider=lambda *_:types.SimpleNamespace(disabled=False))
        original = gui.add_button
        controls = capture.GuiControls(gui)
        next_step = gui.add_button("Next Step")
        overview = gui.add_button("Overview")
        self.assertEqual(controls.handles,[next_step,overview])
        controls.restore_methods()
        self.assertIs(gui.add_button,original)

    def test_semantic_centers_have_one_fixed_bounded_selection_without_mutating_evidence(self):
        states = [{"positive_indices":np.array([[i,0,0] for i in range(5)]),
                   "hazard_indices":np.array([[i,1,0] for i in range(5)])}]
        before = states[0]["positive_indices"].copy()
        viewer = types.SimpleNamespace(server=types.SimpleNamespace(scene=types.SimpleNamespace()))
        with patch.object(capture,"MAX_SEMANTIC_CENTERS",3):
            overlays = capture.SavedOverlays(viewer,{"states":states,"frames":[{}]},semantics=True)
        self.assertEqual(len(overlays.semantic_selected),3)
        self.assertEqual(overlays.semantic_sampling["union_cells"],10)
        self.assertFalse(overlays.semantic_sampling["saved_evidence_changed"])
        np.testing.assert_array_equal(states[0]["positive_indices"],before)

    def serve_wrapper(self,run,*,parent_change=False):
        args = types.SimpleNamespace(max_runtime=.2,parent_pipe_fd=None,controller_lock_fd=None,
                                     source_root=Path("isolated/vendor/lingbot-map"),port=8891,
                                     execution_lock_path=Path("isolated/outputs/.pipeline2_video_execution.lock"),
                                     output=Path("unused"))
        with patch.object(capture,"NativeLease") as lease,patch.object(capture,"_serve",side_effect=run), \
                patch.object(capture.signal,"SIGHUP",1,create=True),patch.object(capture.signal,"signal"), \
                patch.object(capture.os,"getppid",side_effect=([42,43,43,43] if parent_change else lambda:42)):
            return capture.serve(args),lease

    def test_deadline_covers_preparation_before_any_browser_is_open(self):
        def preparing(_args,stop,deadline,ready):
            self.assertFalse(ready.is_set())
            self.assertTrue(stop.wait(.8))
            self.assertGreaterEqual(time.monotonic(),deadline)
            return "stopped"
        result,lease = self.serve_wrapper(preparing)
        self.assertEqual(result,"stopped")
        lease.return_value.acquire.assert_called_once()
        lease.return_value.close.assert_called_once()

    def test_parent_death_stops_before_browser_or_render(self):
        def waiting(_args,stop,deadline,ready):
            self.assertTrue(stop.wait(.15))
            self.assertLess(time.monotonic(),deadline)
            return "parent_gone"
        result,_ = self.serve_wrapper(waiting,parent_change=True)
        self.assertEqual(result,"parent_gone")

    def test_cli_is_png_only_and_expected_count_is_mandatory(self):
        base = ["--source-root","source","--run","run","--sequence","sequence","--output","output"]
        with redirect_stderr(StringIO()),patch.object(capture,"serve") as serve:
            for arguments in (base,base+["--expected-frames","59"],base+["--expected-frames","58","--encode-video"]):
                with self.assertRaises(SystemExit):
                    capture.main(arguments)
            serve.assert_not_called()

    def test_native_capture_disables_all_registered_controls_and_has_no_rgb_fallback(self):
        controls = [types.SimpleNamespace(disabled=False) for _ in range(3)]
        nodes = [types.SimpleNamespace(visible=True)]
        timestep = types.SimpleNamespace(value=0,disabled=False)
        threshold = types.SimpleNamespace(disabled=True)
        viewer = types.SimpleNamespace(gui_timestep=timestep,fourd=False,frame_nodes=nodes,
                                       vis_threshold_slider=threshold,
                                       recording_controls=types.SimpleNamespace(handles=controls+[timestep,threshold]),
                                       server=types.SimpleNamespace(flush=lambda:None),update_frame_visibility=lambda:None)
        camera = types.SimpleNamespace(position=np.array([0,0,-1]),look_at=np.zeros(3),up_direction=np.array([0,-1,0]),
                                       wxyz=np.array([1,0,0,0]),fov=.8)
        client = types.SimpleNamespace(camera=camera,client_id=1)
        data = {"frames":[{"frame_id":"f0","timestamp_ns":0}],"states":[{"research_path":np.empty((0,3))}]}
        overlays = types.SimpleNamespace(tick=lambda:None,research=False)
        manifest = {"outputs_sha256":{}}
        def no_render(*_,**__):
            self.assertTrue(all(handle.disabled for handle in controls))
            return None
        with tempfile.TemporaryDirectory() as directory,patch.object(capture,"get_native_render",side_effect=no_render):
            with self.assertRaisesRegex(RuntimeError,"RGB fallback is forbidden"):
                capture.capture_native(viewer,client,data,Path(directory),manifest,lambda:None,
                                       threading.Event(),time.monotonic()+3,overlays)
        self.assertEqual(manifest["capture"]["status"],"failed")
        self.assertTrue(all(not handle.disabled for handle in controls))


class PackagingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write(self,path,value):
        path.write_text(json.dumps(value),encoding="utf-8")
        if path.name=="manifest.json" and path.parent.name=="segmentation":
            (path.parent/"manifest.sha256").write_text(package.sha256(path)+"  manifest.json\n")

    def fixture(self,n=2):
        native,segmentation = self.root/"native",self.root/"segmentation"
        native.mkdir();segmentation.mkdir()
        (native/"native_frames").mkdir();(segmentation/"segmentation_frames").mkdir()
        original = self.root/"original.txt";original.write_text("immutable saved input")
        identity = {key:"saved" for key in package.IDENTITY}
        identity.update(scale={},up=None,units="reconstruction_units")
        rows = [{"frame_id":f"f{i}","timestamp_ns":i*500000000,
                 "timestamp_provenance":{"clock":"video_presentation_timeline","source_frame_index":i*15},
                 "disposition":{"status":"ok"},"semantic_status":"ok",
                 "queries":[{"prompt":"ground","status":"ok","instances":0,"error":None}]} for i in range(n)]
        timeline = package.sampled_timeline([{"timestamp_ns":row["timestamp_ns"],
                                             "source":{"timestamp_provenance":row["timestamp_provenance"]}} for row in rows],
                                           video_fps=20,end_hold_seconds=2)
        common = {"schema_version":1,"status":"complete","run_dir":str(self.root/"saved_run"),
                  "sequence_path":str(self.root/"sequence/sequence.json"),"geometry_identity":identity,
                  "original_planning":{"availability":"blocked_inputs"},"research_plan":{},
                  "coverage":rows,"input_sha256":{str(original.resolve()):package.sha256(original)},"outputs_sha256":{}}
        native_manifest,segmentation_manifest = deepcopy(common),deepcopy(common)
        native_manifest["capture"]={"status":"complete","renderer":package.RENDERER,
                                    "frames":[{"frame_id":row["frame_id"],"timestamp_ns":row["timestamp_ns"],
                                               "visible_frame_indices":list(range(i+1)),"research_route_visible":False}
                                              for i,row in enumerate(rows)]}
        segmentation_manifest.update(sampled_frame_count=n,encoded_frame_count=sum(timeline["repeat_counts"]))
        for row,count in zip(segmentation_manifest["coverage"],timeline["repeat_counts"]):
            row["encoded_repeat_count"]=count
        native_config = {"source_revision":package.SOURCE_REVISION,"device":"cpu","model":None,"sky_segmentation":False,
                         "saved_xyz_changed":False,"saved_extrinsic":"world_to_camera_unchanged","native_use_point_map":True,
                         "adapter_sha256":package.sha256(Path(capture.__file__)),"expected_sampled_frames":n,
                         "capture_output":"native_WebGL_PNG_only","actual_cuda_available":False,"native_cameras_hidden":True,
                         "actual_bind_host":"127.0.0.1","actual_bind_port":8891,"port":8891,
                         "actual_cpu_affinity":[0,1,2,3],"actual_display_point_count":123,"timeline":timeline}
        segmentation_config = {"frames_only":False,"rgb_view":"source",
                               "mask_projection":"saved_source_to_processed_center_affine_nearest_no_extrapolation","timeline":timeline}
        self.write(native/"viewer_config.json",native_config)
        native_manifest["outputs_sha256"]["viewer_config.json"]=package.sha256(native/"viewer_config.json")
        self.write(segmentation/"export_config.json",segmentation_config)
        segmentation_manifest["outputs_sha256"]["export_config.json"]=package.sha256(segmentation/"export_config.json")
        for index in range(n):
            for directory,manifest,subdirectory in ((native,native_manifest,"native_frames"),
                                                    (segmentation,segmentation_manifest,"segmentation_frames")):
                relative = f"{subdirectory}/{index:06d}.png"
                Image.new("RGB",(800,800),(index%255,23,42)).save(directory/relative)
                manifest["outputs_sha256"][relative]=package.sha256(directory/relative)
        (segmentation/"segmentation.mp4").write_bytes(b"saved bounded movie fixture")
        segmentation_manifest["outputs_sha256"]["segmentation.mp4"]=package.sha256(segmentation/"segmentation.mp4")
        self.write(native/"manifest.json",native_manifest);self.write(segmentation/"manifest.json",segmentation_manifest)
        return native,segmentation,native_manifest,segmentation_manifest

    def load(self,native,segmentation,count):
        return package.load_recording(native,segmentation,package.sha256(native/"manifest.json"),expected_frames=count)

    def test_complete_58_frame_recording_joins_and_preserves_native_pixels(self):
        native,segmentation,_,_ = self.fixture(58)
        data = self.load(native,segmentation,58)
        self.assertEqual(len(data["native_pixels"]),58)
        self.assertEqual(data["native"]["coverage"][-1]["timestamp_ns"],57*500000000)
        panels = package.panels(data,57)
        np.testing.assert_array_equal(np.asarray(panels["native_map"])[104:904,:800],data["native_pixels"][57])
        self.assertEqual(panels["combined"].size,(1600,1000))

    def test_short_recording_and_future_cumulative_frame_are_rejected(self):
        native,segmentation,native_manifest,_ = self.fixture()
        with self.assertRaisesRegex(ValueError,"frame counts differ"):
            self.load(native,segmentation,3)
        native_manifest["capture"]["frames"][0]["visible_frame_indices"]=[0,1]
        self.write(native/"manifest.json",native_manifest)
        with self.assertRaisesRegex(ValueError,"future cumulative frame"):
            self.load(native,segmentation,2)

    def test_missing_empty_query_or_oversized_runtime_point_count_is_rejected(self):
        native,segmentation,_,segmentation_manifest = self.fixture()
        segmentation_manifest["coverage"][0]["queries"] = []
        self.write(segmentation/"manifest.json",segmentation_manifest)
        with self.assertRaisesRegex(ValueError,"query status or empty result"):
            self.load(native,segmentation,2)

    def test_changed_settings_and_expired_admission_deadline_are_rejected(self):
        native,segmentation,_,_ = self.fixture()
        with self.assertRaisesRegex(RuntimeError,"whole-stage deadline"):
            package.load_recording(native,segmentation,package.sha256(native/"manifest.json"),expected_frames=2,
                                   stage_deadline=time.monotonic()-1)
        (segmentation/"export_config.json").write_text("{}")
        with self.assertRaisesRegex(ValueError,"settings hash differs"):
            self.load(native,segmentation,2)

    def test_query_identity_and_actual_native_point_cap_are_joined(self):
        native,segmentation,native_manifest,segmentation_manifest = self.fixture()
        native_manifest["coverage"][0]["queries"][0]["query_id"]="native_query"
        segmentation_manifest["coverage"][0]["queries"][0]["query_id"]="other_query"
        self.write(native/"manifest.json",native_manifest);self.write(segmentation/"manifest.json",segmentation_manifest)
        with self.assertRaisesRegex(ValueError,"query status or empty result"):
            self.load(native,segmentation,2)
        config_path = native/"viewer_config.json"
        config = json.loads(config_path.read_text());config["actual_display_point_count"]=300001
        self.write(config_path,config)
        native_manifest["outputs_sha256"]["viewer_config.json"]=package.sha256(config_path)
        self.write(native/"manifest.json",native_manifest)
        with self.assertRaisesRegex(ValueError,"point-count runtime evidence"):
            self.load(native,segmentation,2)

    def test_recorded_no_path_cannot_claim_a_rendered_route(self):
        native,segmentation,native_manifest,segmentation_manifest = self.fixture()
        research = {"planning_basis":"geometry_only","semantic_guidance":False,"status":"no_path"}
        native_manifest["research_plan"]=segmentation_manifest["research_plan"]=research
        native_manifest["capture"]["frames"][-1]["research_route_visible"]=True
        self.write(native/"manifest.json",native_manifest);self.write(segmentation/"manifest.json",segmentation_manifest)
        with self.assertRaisesRegex(ValueError,"unavailable or nonfinal research route"):
            self.load(native,segmentation,2)

    def test_gallery_uses_native_player_names_and_rejects_changed_html(self):
        directory = self.root/"gallery_recording";directory.mkdir()
        manifest = {"status":"complete","label":"Recording","players":{},"outputs_sha256":{}}
        for kind,title in package.PLAYERS:
            page = directory/(kind+".html");page.write_text(title)
            poster = directory/(kind+"_first.jpg");Image.new("RGB",(10,10)).save(poster)
            manifest["players"][kind]={"html":page.name,"first_thumbnail":poster.name}
            for path in (page,poster):
                manifest["outputs_sha256"][path.name]=package.sha256(path)
        self.write(directory/"manifest.json",manifest)
        (directory/"manifest.sha256").write_text(package.sha256(directory/"manifest.json")+"  manifest.json\n")
        card = package.gallery_card(directory,page_dir=self.root)
        self.assertIn("native_map.html",card)
        self.assertNotIn("voxel_map.html",card)
        (directory/"native_map.html").write_text("changed")
        with self.assertRaisesRegex(ValueError,"player path/hash"):
            package.gallery_card(directory)


@unittest.skipUnless(sys.platform.startswith("linux"),"actual inherited flock/pipe proof requires Linux")
class LinuxControllerDescriptors(unittest.TestCase):
    def descriptor_fixture(self,root):
        import fcntl
        read_fd,write_fd = os.pipe()
        lock_fd = os.open(root/"controller.lock",os.O_CREAT|os.O_RDWR,0o600)
        fcntl.flock(lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return read_fd,write_fd,lock_fd

    def close_owned(self,*descriptors):
        for descriptor in descriptors:
            try:
                os.close(descriptor)
            except OSError:
                pass

    def test_inherited_pipe_and_held_lock_validate_unheld_lock_fails(self):
        import fcntl
        with tempfile.TemporaryDirectory() as temporary:
            read_fd,write_fd,lock_fd = self.descriptor_fixture(Path(temporary))
            try:
                args = types.SimpleNamespace(parent_pipe_fd=read_fd,controller_lock_fd=lock_fd)
                self.assertEqual(capture.inherited_controller(args),read_fd)
                fcntl.flock(lock_fd,fcntl.LOCK_UN)
                with self.assertRaisesRegex(ValueError,"not already held"):
                    capture.inherited_controller(args)
            finally:
                self.close_owned(read_fd,write_fd,lock_fd)

    def test_controller_pipe_eof_stops_before_browser_connection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            read_fd,write_fd,lock_fd = self.descriptor_fixture(root)
            os.close(write_fd)
            args = types.SimpleNamespace(max_runtime=.3,parent_pipe_fd=read_fd,controller_lock_fd=lock_fd,
                                         source_root=root/"vendor/lingbot-map",port=8891,
                                         execution_lock_path=root/"outputs/.pipeline2_video_execution.lock",output=root/"unused")
            def waiting(_args,stop,deadline,ready):
                self.assertTrue(stop.wait(.15))
                self.assertLess(time.monotonic(),deadline)
                return "controller_pipe_eof"
            try:
                with patch.object(capture,"NativeLease"),patch.object(capture,"_serve",side_effect=waiting):
                    self.assertEqual(capture.serve(args),"controller_pipe_eof")
                for descriptor in (read_fd,lock_fd):
                    with self.assertRaises(OSError):
                        os.fstat(descriptor)
            finally:
                self.close_owned(read_fd,lock_fd)


if __name__=="__main__":
    unittest.main()
