"""Projection display checks: exact RGB placement, preserved masks and no fake route."""
import unittest
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from unittest import mock

import numpy as np
from PIL import Image

import export_segmentation_route_recording as export
from export_pipeline_video import render_segmentation


def projection(segments=(), heads=(), status="no_route"):
    return {"source_segments":np.asarray(segments,float).reshape(-1,2,2),
            "source_arrowheads":np.asarray(heads,float).reshape(-1,3,2),
            "report":{"status":status,"visible_segments":len(segments),"arrow_count":len(heads)}}


class PlacementTests(unittest.TestCase):
    def test_portrait_resize_uses_exact_saved_mask_rectangle_and_pixel_centres(self):
        rect = export.source_image_rectangle((1920,1080))
        self.assertEqual(rect,(259,110,281,500))
        np.testing.assert_allclose(export.map_source_pixels([[539.5,959.5]],(1920,1080),rect),
                                   [[399.,359.5]])
        np.testing.assert_allclose(export.map_source_pixels([[-.5,-.5]],(1920,1080),rect),
                                   [[258.5,109.5]])

    def test_overlay_is_clipped_to_actual_rgb_and_keeps_away_pixels_intact(self):
        before = Image.new("RGB",(800,800),(31,62,93))
        route = projection([[[0,100],[199,100]]],[[[180,100],[160,85],[160,115]]],"visible")
        after = export.draw_projected_route(before.copy(),route,(200,200))
        pixels, original = np.asarray(after),np.asarray(before)
        left,top,width,height = export.source_image_rectangle((200,200))
        outside = np.ones((800,800),bool)
        outside[top:top+height,left:left+width] = False
        np.testing.assert_array_equal(pixels[outside],original[outside])
        self.assertGreater(np.count_nonzero(np.any(pixels!=original,axis=2)),100)
        np.testing.assert_array_equal(pixels[top+20,left+20],original[top+20,left+20])

    def test_no_route_overlay_is_a_byte_identical_noop(self):
        image = Image.new("RGB",(800,800),(31,62,93))
        self.assertEqual(image.tobytes(),export.draw_projected_route(image.copy(),projection(),(200,200)).tobytes())

    def test_coverage_distinguishes_visible_route_line_from_direction_arrows(self):
        frame = {"frame_id":"f0","timestamp_ns":0,"source":{},"disposition":{},"semantic":None}
        line = projection([[[0,0],[1,1]]],(),"visible")
        coverage = export.frame_coverage(frame,10,line)
        self.assertTrue(coverage["research_path_drawn"])
        self.assertEqual(coverage["route_projection"]["arrow_count"],0)
        self.assertFalse(export.frame_coverage(frame,10,projection())["research_path_drawn"])

    def test_mask_rgb_survives_direction_overlay_and_input_pixels_are_immutable(self):
        rgb = np.full((200,200,3),180,dtype=np.uint8)
        mask = np.ones((200,200),bool)
        frame = {"source_rgb":rgb,"rgb":rgb.copy(),"frame_id":"f0","timestamp_ns":0,
                 "source":{},"disposition":{},"transform":{"matrix":np.eye(3).tolist(),"pad_ltrb":[0,0,0,0]},
                 "semantic":{"status":"ok","queries":[{"query_id":"floor","original_phrase":"floor",
                    "status":"ok","role":"candidate_surface","instances":[{"mask":mask}]}]}}
        data = {"run":{},"frames":[frame],"research_plan":{"status":"ok"},
                "route_projections":[projection([[[100,170],[100,30]]],[[[100,50],[92,67],[108,67]]],"visible")]}
        original_rgb, original_mask = rgb.copy(),mask.copy()
        baseline = render_segmentation(data,frame,0)
        output = export.render_frame(data,frame,0)
        left,top,_,_ = export.source_image_rectangle(rgb.shape[:2])
        np.testing.assert_array_equal(np.asarray(output)[top+70,left+70],np.asarray(baseline)[top+70,left+70])
        np.testing.assert_array_equal(rgb,original_rgb)
        np.testing.assert_array_equal(mask,original_mask)
        self.assertTrue(np.any(np.asarray(output)[top+75:top+425,left+220:left+280]
                               !=np.asarray(baseline)[top+75:top+425,left+220:left+280]))


class ReplanningReplayTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.plan_path=self.root/"research_plan.json"
        identity={key:"identity-"+key for key in export.REPLAY_IDENTITY_KEYS}
        self.plan={"status":"ok","source":identity.copy(),"assumptions":{"metres_per_native_unit":2.},
                   "validated_path":[[-.5,0,2],[1,0,2]]}
        self.plan_path.write_text(json.dumps(self.plan),encoding="utf-8")
        transform={"matrix":np.eye(3).tolist(),"source_shape":[100,100],"processed_shape":[100,100],"pad_ltrb":[0,0,0,0]}
        rgb=np.full((100,100,3),180,np.uint8)
        self.frames=[{"frame_id":"f"+str(i),"timestamp_ns":i*500_000_000,"rgb":rgb.copy(),"source_rgb":rgb.copy(),
                      "transform":deepcopy(transform),"source":{},"disposition":{},"semantic":None} for i in range(2)]
        self.data={"geometry":{**identity,"archive_sha256":"archive-hash","settings":{"min_confidence":1.5}},
                   "research_plan":deepcopy(self.plan),"frames":self.frames,"input_sha256":{},"cache":self.root,
                   "map":{"voxel_size":.05},"run":{}}
        self.replay={"schema_version":1,"artifact_kind":"research_navigation_replay","research_illustration":True,
            "source":{**identity,"research_plan_sha256":export.sha256(self.plan_path)},
            "mission":{"id":"continue-corridor","policy":"persistent_forward_corridor","state":"active","completed":False},
            "display_horizon_assumed_m":2.,"perception_scope":"cached_geometry_replay","frame_scope":export.REPLAY_SCOPE,
            "frames":[{"frame_id":"f0","timestamp_ns":0,"observed_through_index":0,"status":"ok",
                       "path_points":[[-.5,0,2],[1,0,2]],"display_path_points":[[-.5,0,2],[.5,0,2]],
                       "navigation_state":"following_local_plan","diagnostics":{}},
                      {"frame_id":"f1","timestamp_ns":500_000_000,"observed_through_index":1,"status":"ok",
                       "path_points":[[-.4,.1,2],[.8,.1,2]],"display_path_points":[[-.4,.1,2],[.4,.1,2]],
                       "navigation_state":"following_local_plan","diagnostics":{"replanned_from_current_start":True}}],
            "terminal_state":{"mission_state":"active","execution_state":"awaiting_observations","mission_completed":False,
                              "reason":"recording_ended","retained_display_path_points":[[-.4,.1,2],[.4,.1,2]]}}

    def admit(self,replay=None,**kwargs):
        path=self.root/"replay.json"
        path.write_text(json.dumps(self.replay if replay is None else replay),encoding="utf-8")
        return export.load_replanning_replay(self.data,path,research_plan_path=self.plan_path,**kwargs)

    def test_independent_admission_keeps_mission_active_and_registers_input_hash(self):
        admitted=self.admit()
        self.assertEqual(admitted["terminal_state"]["execution_state"],"awaiting_observations")
        self.assertFalse(admitted["mission"]["completed"])
        self.assertEqual(self.data["input_sha256"][str(self.root/"replay.json")],export.sha256(self.root/"replay.json"))
        self.assertEqual(self.data["research_plan"],self.plan)

    def test_rejects_source_frame_time_cutoff_and_plan_hash_mismatches(self):
        cases=[]
        changed=deepcopy(self.replay);changed["source"]["map_frame"]="another-map";cases.append(changed)
        changed=deepcopy(self.replay);changed["source"]["research_plan_sha256"]="changed";cases.append(changed)
        changed=deepcopy(self.replay);changed["frames"].reverse();cases.append(changed)
        changed=deepcopy(self.replay);changed["frames"][0]["timestamp_ns"]=False;cases.append(changed)
        changed=deepcopy(self.replay);changed["frames"][1]["observed_through_index"]=2;cases.append(changed)
        changed=deepcopy(self.replay);changed["frames"].pop();cases.append(changed)
        for changed in cases:
            with self.subTest(change=changed):
                with self.assertRaises(ValueError):self.admit(changed)

    def test_short_prefix_allows_interpolation_but_rejects_excess_and_shortcuts(self):
        self.admit()  # The first display endpoint interpolates the full edge.
        cases=[]
        changed=deepcopy(self.replay);changed["display_horizon_assumed_m"]=2.1;cases.append(changed)
        changed=deepcopy(self.replay);changed["frames"][0]["display_path_points"]=[[-.5,0,2],[.6,0,2]];cases.append(changed)
        changed=deepcopy(self.replay);changed["frames"][0]["display_path_points"]=[[-.5,0,2],[.4,.1,2]];cases.append(changed)
        changed=deepcopy(self.replay);changed["frames"][0]["display_path_points"][0][0]=float("nan");cases.append(changed)
        for changed in cases:
            with self.subTest(change=changed):
                with self.assertRaises(ValueError):self.admit(changed)

    def test_awaiting_support_requires_no_route_and_eof_cannot_complete_mission(self):
        changed=deepcopy(self.replay)
        changed["frames"][0].update(status="awaiting_support",path_points=[],display_path_points=[],navigation_state="awaiting_support")
        self.admit(changed)
        changed["frames"][0].update(status="awaiting_observation",navigation_state="awaiting_observations")
        self.admit(changed)
        changed["frames"][0]["display_path_points"]=[[-.5,0,2],[.5,0,2]]
        with self.assertRaises(ValueError):self.admit(changed)
        changed=deepcopy(self.replay);changed["terminal_state"]["mission_completed"]=True
        with self.assertRaises(ValueError):self.admit(changed)
        changed=deepcopy(self.replay);changed["terminal_state"]["retained_display_path_points"][0][0]=-.3
        with self.assertRaisesRegex(ValueError,"last admitted"):self.admit(changed)

    def test_projection_uses_each_replan_and_only_display_horizon_not_full_route(self):
        self.admit()
        intrinsic=np.tile(np.array([[100.,0,50],[0,100.,50],[0,0,1.]]),(2,1,1))
        arrays={"intrinsic":intrinsic,"extrinsic":np.tile(np.column_stack((np.eye(3),np.zeros(3))),(2,1,1)),
                "depth":np.full((2,100,100),3.),"world_points_conf":np.full((2,100,100),2.)}
        def digest(path):
            return export.BASE_HELPER_SHA256 if Path(path).name=="export_segmentation_recording.py" else "archive-hash"
        with mock.patch.object(export,"sha256",side_effect=digest),mock.patch.object(export,"load_npz_checked",return_value=arrays):
            export.prepare_route_projection(self.data)
        projections=self.data["route_projections"]
        np.testing.assert_allclose(projections[0]["source_segments"],[[[25,50],[75,50]]])
        np.testing.assert_allclose(projections[1]["source_segments"],[[[30,55],[70,55]]])
        self.assertEqual(projections[0]["report"]["display_path_length_assumed_m"],2.)
        self.assertEqual(projections[1]["report"]["mission_id"],"continue-corridor")
        self.assertEqual(projections[1]["report"]["observed_through_index"],1)
        self.assertTrue(projections[1]["report"]["planner_recomputed_route"])
        np.testing.assert_array_equal(self.data["research_plan"]["validated_path"],self.plan["validated_path"])

    def test_last_frame_hold_says_paused_active_and_static_caption_is_unchanged(self):
        self.admit()
        self.data["route_projections"]=[projection(),projection()]
        for projected in self.data["route_projections"]:projected["report"]["navigation_status"]="awaiting_support"
        with mock.patch.object(export,"render_segmentation",return_value=Image.new("RGB",(800,800))),mock.patch.object(export,"_text") as text:
            export.render_frame(self.data,self.frames[-1],1)
        messages=[call.args[2] for call in text.call_args_list]
        self.assertIn("Recording paused / mission active - awaiting observations.",messages)
        self.assertTrue(any("Masks + next steps" in message for message in messages))
        self.data.pop("replanning_replay")
        with mock.patch.object(export,"render_segmentation",return_value=Image.new("RGB",(800,800))),mock.patch.object(export,"_text") as text:
            export.render_frame(self.data,self.frames[-1],1)
        self.assertIn("Saved camera projection; final-map research route, not live guidance.",[call.args[2] for call in text.call_args_list])


if __name__ == "__main__":
    unittest.main()
