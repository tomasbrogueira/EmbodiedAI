"""CPU research illustration checks; no models, GPU or calibrated claims."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from path_mapping.runner import geometry_fingerprint
from pipeline_common import planning as common
from pipeline_research_plan import (DEFAULT_ASSUMPTIONS, export_research_plan,
                                    plan_illustration, validate_assumptions)


def terrain(omit=(), wall=False, elevated=0.):
    points = [[x + ox, y + oy, elevated] for y in range(9) for x in range(9)
              if (y, x) not in omit for oy in (.15, .5, .85) for ox in (.15, .5, .85)]
    if wall:
        points += [[4.5, y + oy, z] for y in range(9) for oy in (.15, .5, .85)
                   for z in np.linspace(.1, 1.5, 20)]
    return np.asarray(points, float).reshape(-1, 3)


def assumptions(policy="camera_projected"):
    result = deepcopy(DEFAULT_ASSUMPTIONS)
    result.update(grid_resolution_m=1., endpoint_policy=policy,
                  up_vector_native=[0., 0., 1.],
                  ground_estimation={"mode": "explicit_plane", "normal_native": [0., 0., 1.],
                                     "point_native": [0., 0., 0.]})
    result["robot"].update(footprint_radius=.1, clearance=.05, height=1.,
                           max_step=.2, max_roughness=.03, min_support_points=9)
    return result


def plan(points, config=None, cameras=None):
    cameras = np.asarray([[1.5, 4.5, 1.], [7.5, 4.5, 1.]] if cameras is None else cameras)
    rotation = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    return plan_illustration(points, cameras, np.tile(rotation, (len(cameras), 1, 1)),
                             assumptions() if config is None else config,
                             {"map_frame": "test_world", "original_scale": {}, "original_up": None,
                              "original_planning_status": "blocked_inputs"})


class ResearchPlanningTests(unittest.TestCase):
    def test_flat_route_retains_native_coordinates_and_camera_track_separately(self):
        points = terrain(); original = points.copy()
        result, arrays = plan(points)
        self.assertEqual(result["status"], "ok")
        np.testing.assert_allclose(result["path_points"][0], [1.5, 4.5, 0.])
        np.testing.assert_allclose(result["path_points"][-1], [7.5, 4.5, 0.])
        self.assertFalse(result["camera_trajectory_is_planned_route"])
        self.assertFalse(result["calibration_verified"])
        self.assertFalse(result["safety_validated"])
        self.assertFalse(result["clearance_certified"])
        self.assertEqual(result["source"]["original_planning_status"], "blocked_inputs")
        self.assertEqual(result["source"]["original_scale"], {})
        self.assertIsNone(result["source"]["original_up"])
        self.assertTrue(all(arrays["decision_state"][tuple(cell)] == common.TRAVERSABLE
                            for cell in result["path_cells"]))
        np.testing.assert_array_equal(points, original)

    def test_unknown_gap_is_not_bridged(self):
        points = terrain(omit={(y, 4) for y in range(9)})
        result, arrays = plan(points)
        self.assertEqual(result["status"], "no_path")
        self.assertEqual(result["path_points"], [])
        self.assertIn("no connected", result["reason"])
        self.assertTrue(np.any(arrays["decision_state"] == common.UNKNOWN))

    def test_unsupported_camera_endpoints_report_missing_support_without_claiming_component_failure(self):
        result, arrays = plan(terrain(), cameras=[[-2.,4.5,1.],[12.,4.5,1.]])
        self.assertEqual(result["status"], "no_path")
        self.assertEqual(result["path_points"], [])
        self.assertIn("Camera projections lack observed", result["reason"])
        self.assertTrue(np.any(arrays["decision_state"] == common.TRAVERSABLE))

    def test_observed_wall_blocks_camera_route(self):
        result, arrays = plan(terrain(wall=True))
        self.assertEqual(result["status"], "no_path")
        self.assertEqual(result["path_points"], [])
        self.assertTrue(np.any(arrays["decision_state"] == common.BLOCKED))

    def test_demo_endpoints_stay_in_one_observed_component_and_are_labeled(self):
        result, arrays = plan(terrain(omit={(y, 4) for y in range(9)}),
                              assumptions("camera_projected_then_largest_observed_component"))
        self.assertEqual(result["status"], "ok")
        self.assertIn("automatic_demonstration", result["endpoints"]["selection"])
        cells = result["path_cells"]
        self.assertTrue(all(arrays["decision_state"][tuple(cell)] == common.TRAVERSABLE for cell in cells))
        self.assertTrue(all(abs(a[0]-b[0])+abs(a[1]-b[1]) == 1 for a,b in zip(cells,cells[1:])))
        self.assertTrue(all(point[0] < 4 or point[0] > 5 for point in result["path_points"]))
        self.assertEqual(result["camera_trajectory_points"], [[1.5,4.5,1.],[7.5,4.5,1.]])

    def test_elevated_surface_does_not_replace_declared_ground_anchor(self):
        result, arrays = plan(terrain(elevated=2.), assumptions("largest_observed_component"))
        self.assertEqual(result["status"], "no_path")
        self.assertFalse(np.any(arrays["decision_state"] == common.TRAVERSABLE))

    def test_metric_assumption_changes_checks_but_exports_original_native_points(self):
        config = assumptions();config["metres_per_native_unit"] = 2.
        config["grid_resolution_m"] = 2.
        config["robot"].update(footprint_radius=.2, clearance=.1, height=2., max_step=.4, max_roughness=.06)
        result, _ = plan(terrain(), config)
        self.assertEqual(result["status"], "ok")
        np.testing.assert_allclose(result["path_points"][0], [1.5,4.5,0.])
        np.testing.assert_allclose(result["path_points"][-1], [7.5,4.5,0.])
        self.assertEqual(result["diagnostics"]["planning_check_metadata"]["units"], "assumed_metres")

    def test_estimated_plane_and_up_remain_assumed_and_deterministic(self):
        config=deepcopy(DEFAULT_ASSUMPTIONS); config["grid_resolution_m"]=1.
        config["ground_estimation"].update(min_inliers=20, plane_tolerance_m=.02)
        first,_=plan(terrain(),config);second,_=plan(terrain(),config)
        self.assertEqual(first,second)
        np.testing.assert_allclose(first["assumed_up_vector"], [0,0,1],atol=1e-10)
        self.assertTrue(first["ground_plane"]["assumption"])
        self.assertFalse(first["ground_plane"]["floor_identity_verified"])
        self.assertFalse(first["ground_plane"]["gravity_measured"])

    def test_missing_ground_and_grid_limit_export_reason_without_route(self):
        config=deepcopy(DEFAULT_ASSUMPTIONS)
        wall=np.asarray([[0,y,z] for y in np.linspace(0,5,20) for z in np.linspace(0,2,20)])
        result,arrays=plan(wall,config)
        self.assertEqual(result["status"],"blocked_inputs")
        self.assertIsNone(arrays)
        self.assertIn("No sufficiently supported",result["reason"])
        config=assumptions();config["max_grid_cells"]=1
        result,arrays=plan(terrain(),config)
        self.assertEqual(result["status"],"blocked_inputs")
        self.assertIsNone(arrays)
        self.assertIn("grid size",result["reason"])

    def test_common_verified_input_gate_is_never_patched_or_satisfied_by_assumptions(self):
        original=common._settings
        plan(terrain())
        self.assertIs(common._settings,original)
        arrays,meta=common.build_costmap(terrain(),{},up=[0,0,1],scale={"assumed":True})
        self.assertEqual(meta["availability"],"blocked_inputs")
        self.assertIn("not verified",meta["reason"])
        self.assertEqual(arrays["decision_state"].shape,(0,0))

    def test_explicit_research_opt_in_and_unknown_blocking_are_required(self):
        for change in ({"research_illustration":False},{"metres_per_native_unit":0.},{"max_grid_cells":True}):
            config=assumptions();config.update(change)
            with self.assertRaises(ValueError):validate_assumptions(config)
        config=assumptions();config["robot"]["unknown_rule"]="free"
        with self.assertRaises(ValueError):validate_assumptions(config)


class ResearchCalibrationTests(unittest.TestCase):
    def config(self, *, explicit_reference=False, height=None):
        config = assumptions()
        config.pop("up_vector_native")
        config["level_reference"] = {
            "recording_id": "indoor_IMG_5506", "geometry_fingerprint": "a" * 64,
            "declaration": "user_declared_level_floor", "evidence": "Independent synthetic reference floor declared level",
        }
        if explicit_reference:
            config["level_reference"].update(normal_native=[0., 0., 1.], point_native=[0., 0., 0.])
        if height is not None:
            config["camera_height_reference"] = {
                "recording_id": "indoor_IMG_5506", "geometry_fingerprint": "a" * 64,
                "height_metres": height, "declaration": "user_assumed_camera_height",
                "evidence": "Explicit physical camera-height assumption for this recording",
            }
        return config

    def calculate(self, points, config, *, cameras=None, pitch=26.):
        cameras = np.asarray([[1.5, 4.5, 1.], [7.5, 4.5, 1.]] if cameras is None else cameras, float)
        angle = np.radians(pitch)
        rotation = np.array([[1., 0., 0.], [0., -np.sin(angle), -np.cos(angle)],
                             [0., np.cos(angle), -np.sin(angle)]])
        source = {"recording_id": "indoor_IMG_5506", "geometry_fingerprint": "a" * 64,
                  "run_dir": "/saved/indoor_IMG_5506/ground_surface_run", "original_scale": {},
                  "original_up": None, "original_planning_status": "blocked_inputs", "map_frame": "test_world"}
        return plan_illustration(points, cameras, np.tile(rotation, (len(cameras), 1, 1)), config, source)

    def test_declared_level_reference_corrects_phone_pitch_without_flattening_measurements(self):
        config = self.config()
        original = deepcopy(config)
        result, arrays = self.calculate(terrain(), config)
        baseline = deepcopy(config)
        baseline.pop("level_reference")
        tilted, _ = self.calculate(terrain(), baseline)
        self.assertEqual(tilted["status"], "no_path")
        self.assertEqual(result["status"], "ok")
        np.testing.assert_allclose(result["assumed_up_vector"], [0., 0., 1.], atol=1e-12)
        np.testing.assert_allclose(arrays["slope_degrees"][np.isfinite(arrays["slope_degrees"])], 0., atol=1e-10)
        correction = result["research_calibration"]["level_reference"]
        self.assertAlmostEqual(correction["angular_correction_degrees"], 26.)
        self.assertEqual(correction["declared_reference_grade_percent"], 0.)
        self.assertFalse(correction["local_terrain_measurements_forced_flat"])
        self.assertFalse(correction["geometry_up_verified"])
        self.assertFalse(result["calibration_verified"])
        self.assertEqual(result["source"]["original_scale"], {})
        self.assertIsNone(result["source"]["original_up"])
        self.assertEqual(config, original)

    def test_real_thirty_degree_ramp_remains_blocked_against_independent_level_reference(self):
        config = self.config(explicit_reference=True)
        angle = np.radians(30.)
        points = terrain()
        points[:, 2] = np.tan(angle) * points[:, 0]
        config["ground_estimation"] = {"mode": "explicit_plane", "normal_native": [-np.sin(angle), 0., np.cos(angle)],
                                       "point_native": [0., 0., 0.]}
        cameras = [[1.5, 4.5, np.tan(angle) * 1.5 + 1.], [7.5, 4.5, np.tan(angle) * 7.5 + 1.]]
        result, arrays = self.calculate(points, config, cameras=cameras)
        self.assertEqual(result["status"], "no_path")
        np.testing.assert_allclose(arrays["slope_degrees"][np.isfinite(arrays["slope_degrees"])], 30., atol=1e-10)
        self.assertTrue(np.any(arrays["decision_state"] == common.BLOCKED))
        self.assertEqual(result["assumptions"]["robot"]["max_slope_degrees"], 25.)

    def test_camera_height_derives_scale_and_preserves_physical_robot_and_native_route(self):
        config = self.config(height=2.)
        config["grid_resolution_m"] = 2.
        robot = deepcopy(config["robot"])
        result, arrays = self.calculate(terrain(), config)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["assumptions"]["metres_per_native_unit"], 2.)
        self.assertEqual(result["input_assumptions"]["metres_per_native_unit"], 1.)
        self.assertEqual(result["assumptions"]["robot"], robot)
        self.assertEqual(result["assumptions"]["robot"]["footprint_radius"], .1)
        self.assertEqual(arrays["resolution"][0], 2.)
        np.testing.assert_allclose(result["path_points"][0], [1.5, 4.5, 0.])
        np.testing.assert_allclose(result["path_points"][-1], [7.5, 4.5, 0.])
        calibration = result["research_calibration"]["camera_height_reference"]
        self.assertEqual(calibration["first_camera_height_native"], 1.)
        self.assertFalse(calibration["geometry_scale_verified"])
        self.assertTrue(calibration["height_is_assumed"])

    def test_camera_height_uses_up_ray_not_arbitrary_plane_centroid(self):
        config = self.config(height=1.5)
        config.pop("level_reference")
        config["up_vector_native"] = [0., 0., 1.]
        angle = np.radians(26.)
        points = terrain()
        points[:, 2] = np.tan(angle) * points[:, 1]
        config["ground_estimation"] = {"mode": "explicit_plane", "normal_native": [0., -np.sin(angle), np.cos(angle)],
                                       "point_native": [0., 4., 4. * np.tan(angle)]}
        z = 4.5 * np.tan(angle) + 1.
        result, _ = self.calculate(points, config, cameras=[[1.5, 4.5, z], [7.5, 4.5, z]])
        calibration = result["research_calibration"]["camera_height_reference"]
        self.assertAlmostEqual(calibration["first_camera_height_native"], 1.)
        self.assertAlmostEqual(result["assumptions"]["metres_per_native_unit"], 1.5)

    def test_calibration_binding_and_conflicting_up_are_refused(self):
        for key, value in (("recording_id", "indoor_other"), ("geometry_fingerprint", "b" * 64)):
            config = self.config()
            config["level_reference"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "another recording or geometry"):
                self.calculate(terrain(), config)
        config = self.config()
        config["up_vector_native"] = [0., 0., 1.]
        with self.assertRaisesRegex(ValueError, "do not silently override"):
            validate_assumptions(config)

    def test_absent_ground_does_not_fabricate_camera_height_scale(self):
        config = self.config(height=1.5)
        config.pop("level_reference")
        config["ground_estimation"] = deepcopy(DEFAULT_ASSUMPTIONS["ground_estimation"])
        wall = np.asarray([[0., y, z] for y in np.linspace(0, 5, 20) for z in np.linspace(0, 2, 20)])
        result, arrays = self.calculate(wall, config, pitch=0.)
        self.assertEqual(result["status"], "blocked_inputs")
        self.assertIsNone(arrays)
        self.assertEqual(result["assumptions"]["metres_per_native_unit"], 1.)
        self.assertEqual(result["research_calibration"]["camera_height_reference"]["availability"], "unavailable")


class ResearchExportTests(unittest.TestCase):
    def fixture(self, directory):
        run=directory/'original';cache=run/'geometry/cache';cache.mkdir(parents=True)
        points=terrain().reshape(1,27,27,3).astype(np.float32)
        geometry={"images":np.zeros((1,27,27,3),np.uint8),"world_points":points,
                  "world_points_conf":np.full((1,27,27),2.,np.float32),
                  "depth":np.ones((1,27,27),np.float32),
                  "extrinsic":np.asarray([[[1,0,0,0],[0,-1,0,0],[0,0,-1,1]]],np.float32),
                  "intrinsic":np.eye(3,dtype=np.float32)[None]}
        archive=cache/'geometry.npz';np.savez_compressed(archive,**geometry)
        fingerprint=geometry_fingerprint(geometry)
        manifest={"status":"complete","geometry_fingerprint":fingerprint,
                  "archive_sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),
                  "map_frame":"test_world","units":"reconstruction_units","scale":{},"up":None,
                  "settings":{"min_confidence":1.5},"cache_path":str(cache)}
        (run/'geometry/manifest.json').write_text(json.dumps(manifest))
        (run/'run.json').write_text(json.dumps({"status":"complete","fixture":True,
            "planning_status":"blocked_inputs","geometry_identity":{"geometry_fingerprint":fingerprint}}))
        (run/'map').mkdir();(run/'map/manifest.json').write_text(json.dumps({"geometry_fingerprint":fingerprint,"voxel_size":.05,"origin":[0,0,0]}))
        return run,archive

    def bind_derived_sequence(self, run, *, admitted=True):
        metadata=json.loads((run/'run.json').read_text())
        metadata['sequence']={'sequence_id':'outdoor_IMG_5513','manifest_digest':'a'*64}
        (run/'run.json').write_text(json.dumps(metadata))
        manifest=json.loads((run/'geometry/manifest.json').read_text())
        manifest['sequence_digest']='a'*64
        manifest['derivation']={'method':'test_identity_binding', 'planning_admitted':admitted}
        (run/'geometry/manifest.json').write_text(json.dumps(manifest))

    def test_nested_derived_recording_uses_immutable_sequence_and_keeps_derivation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);nested=root/'attempt_01'/'derived';nested.mkdir(parents=True)
            run,_=self.fixture(nested);self.bind_derived_sequence(run)
            result,_=export_research_plan(run,root/'research',assumptions('largest_observed_component'))
            self.assertEqual(result['source']['recording_id'],'outdoor_IMG_5513')
            self.assertEqual(result['source']['derived_geometry']['method'],'test_identity_binding')

    def test_derived_sequence_digest_mismatch_is_rejected_before_export(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run,_=self.fixture(root);self.bind_derived_sequence(run)
            manifest=json.loads((run/'geometry/manifest.json').read_text())
            manifest['sequence_digest']='b'*64
            (run/'geometry/manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'sequence digest differ'):
                export_research_plan(run,root/'research',assumptions())
            self.assertFalse((root/'research').exists())

    def test_display_only_pose_never_exports_static_reference_route(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run,_=self.fixture(root);self.bind_derived_sequence(run,admitted=False)
            result,_=export_research_plan(run,root/'research',assumptions('largest_observed_component'))
            self.assertEqual(result['status'],'blocked_inputs')
            self.assertEqual(result['path_points'],[])
            self.assertIsNone(result['endpoints'])
            self.assertFalse((root/'research/research_costmap.npz').exists())

    def test_export_is_separate_numeric_and_preserves_every_original_byte(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run,archive=self.fixture(root)
            original={q.relative_to(run):q.read_bytes() for q in run.rglob('*') if q.is_file()}
            result,path=export_research_plan(run,root/'research',assumptions('largest_observed_component'))
            self.assertEqual(result['status'],'ok')
            self.assertEqual(json.loads(path.read_text()),result)
            self.assertEqual(original,{q.relative_to(run):q.read_bytes() for q in run.rglob('*') if q.is_file()})
            with np.load(root/'research/research_costmap.npz',allow_pickle=False) as arrays:
                self.assertTrue(all(arrays[key].dtype!=object for key in arrays.files))
            with self.assertRaises(FileExistsError):export_research_plan(run,root/'research',assumptions())
            with self.assertRaisesRegex(ValueError,'separate'):export_research_plan(run,run/'research',assumptions())

    def test_tampered_or_incomplete_source_is_refused_before_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run,archive=self.fixture(root)
            archive.write_bytes(archive.read_bytes()+b'tampered')
            with self.assertRaisesRegex(ValueError,'archive hash'):export_research_plan(run,root/'research',assumptions())
            self.assertFalse((root/'research').exists())
            (run/'run.json').write_text(json.dumps({'status':'failed'}))
            with self.assertRaisesRegex(ValueError,'completed'):export_research_plan(run,root/'research',assumptions())
            self.assertFalse((root/'research').exists())


if __name__=='__main__':
    unittest.main()
