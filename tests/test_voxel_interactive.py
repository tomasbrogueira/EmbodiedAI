"""CPU-only diagnostic overlay checks; no native server or producer imports."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

import export_lingbot_voxel_interactive as export
from export_pipeline_video import load_npz_checked


def costmap(shape=(2, 3)):
    height = np.full(shape, .05)
    slope = np.full(shape, 10.)
    valid = np.ones(shape, dtype=bool)
    slope[0, 0], slope[0, 2], valid[0, 2] = 30., np.nan, False
    return {"support_height": height, "slope_degrees": slope, "terrain_slope_valid": valid,
            "projection_basis": np.eye(3), "origin": np.zeros(2), "resolution": np.array([1.]), "up": np.array([0., 0., 1.])}


def inputs():
    indices = np.array([[4, 4, 0], [14, 4, 0], [24, 4, 0], [4, 4, 30], [34, 4, 0]])
    return indices, (indices + .5) * .1


def classify(grid=None, indices=None, centers=None, **kwargs):
    if indices is None:
        indices, centers = inputs()
    values = dict(factor=1., voxel_size=.1, assumed_up=[0., 0., 1.])
    values.update(kwargs)
    return export.classify_voxel_slopes(indices, centers, grid or costmap(), **values)


class ClassificationTests(unittest.TestCase):
    def test_red_unknown_and_ceiling_filter_preserve_native_positions(self):
        indices, centers = inputs()
        before = centers.copy()
        grid = costmap()
        grid_before = deepcopy(grid)
        result = classify(grid, indices, centers)
        np.testing.assert_array_equal(result["points"], centers[[0, 2]])
        np.testing.assert_array_equal(result["colors"], [[235, 65, 65], [245, 174, 50]])
        np.testing.assert_array_equal(centers, before)
        for key in grid:
            np.testing.assert_array_equal(grid[key], grid_before[key])
        self.assertEqual(result["facts"]["near_raw_support_voxel_count"], 3)
        self.assertEqual(result["facts"]["full_red_voxel_count"], 1)
        self.assertEqual(result["facts"]["full_amber_voxel_count"], 1)
        self.assertFalse(result["facts"]["within_limit_means_free"])
        self.assertFalse(result["facts"]["physical_exclusions_overridden"])

    def test_strict_slope_threshold_and_no_raw_support_means_no_color(self):
        grid = costmap()
        grid["slope_degrees"][0, 0] = 25.
        grid["support_height"][0, 2] = np.nan
        self.assertEqual(len(classify(grid)["points"]), 0)

    def test_scaled_rotated_basis_projects_only_for_classification(self):
        grid = costmap()
        grid["projection_basis"] = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
        grid["up"] = np.array([0., 1., 0.])
        grid["support_height"][:] = .1
        centers = np.array([[.25, .05, .25], [.25, 1.5, .25]])
        result = classify(grid, np.array([[0, 0, 0], [0, 1, 0]]), centers, factor=2., assumed_up=[0., 2., 0.])
        np.testing.assert_array_equal(result["points"], centers[:1])
        self.assertEqual(result["facts"]["support_height_tolerance_assumed_metres"], .4)

    def test_cap_is_deterministic_under_input_permutation_with_full_counts(self):
        n = 8000
        grid = costmap((1, n))
        grid["slope_degrees"][:] = 40.
        grid["terrain_slope_valid"][:] = True
        indices = np.stack((np.arange(n) * 10 + 4, np.full(n, 4), np.zeros(n, int)), axis=1)
        centers = (indices + .5) * .1
        first = classify(grid, indices, centers)
        reverse = classify(grid, indices[::-1], centers[::-1])
        self.assertEqual(first["facts"]["full_red_voxel_count"], n)
        self.assertEqual(first["facts"]["selected_point_count"], 6000)
        np.testing.assert_array_equal(first["points"], reverse["points"])
        self.assertEqual(first["facts"]["selection_sha256"], reverse["facts"]["selection_sha256"])

    def test_invalid_projection_shape_angles_and_limits_are_rejected(self):
        changes = [lambda g: g.update(projection_basis=np.ones((3, 3))),
                   lambda g: g.update(up=np.array([0., 1., 0.])),
                   lambda g: g.update(terrain_slope_valid=np.ones((2, 3), int)),
                   lambda g: g["slope_degrees"].__setitem__((0, 0), np.nan),
                   lambda g: g["slope_degrees"].__setitem__((0, 0), 91.),
                   lambda g: g.update(support_height=np.ones((1, 250001))),
                   lambda g: g.update(resolution=np.array([0.]))]
        for change in changes:
            with self.subTest(change=change):
                grid = costmap()
                change(grid)
                with self.assertRaises(ValueError):
                    classify(grid)
        for kwargs in ({"factor": True}, {"factor": None}, {"voxel_size": -1}, {"max_points": 6001}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                classify(**kwargs)


class SourceTests(unittest.TestCase):
    def fixture(self, root, *, blocked=False):
        run = (root / "surface_run").resolve()
        research_dir = (root / "research").resolve()
        (run / "map").mkdir(parents=True)
        (run / "geometry").mkdir()
        research_dir.mkdir()
        (run / "run.json").write_text('{"status":"complete"}')
        (run / "geometry/manifest.json").write_text('{"status":"complete"}')
        saved_map = {"origin": [0., 0., 0.], "voxel_size": .1, "geometry_fingerprint": "geometry-fixture"}
        (run / "map/manifest.json").write_text(json.dumps(saved_map))
        indices, centers = inputs()
        np.savez_compressed(run / "map/voxels.npz", voxel_indices=indices, centers=centers)
        research = {"source": {"run_dir": str(run), "run_json_sha256": export.sha256(run / "run.json"),
                    "geometry_manifest_sha256": export.sha256(run / "geometry/manifest.json"),
                    "voxel_map_sha256": export.sha256(run / "map/voxels.npz"),
                    "voxel_map_manifest_sha256": export.sha256(run / "map/manifest.json"),
                    "slope_point_source": "saved_fused_LingBot_map_voxel_centers",
                    "voxel_origin_native": saved_map["origin"], "voxel_size_native": .1},
                    "safety_validated": False, "status": "blocked_inputs" if blocked else "no_path",
                    "reason": "No fitted ground" if blocked else "No path in this fixture",
                    "assumed_up_vector": [0., 0., 1.],
                    "slope_reference": {"gravity_measured": False, "up_native": [0., 0., 1.], "source": "explicit_assumed_up_vector"},
                    "assumptions": {"metres_per_native_unit": 1., "robot": {"max_slope_degrees": 25.}},
                    "diagnostics": {"planning_check_metadata": {"max_slope_degrees": 25., "terrain_estimation": {
                        "method": "observed_lower_voxel_two_scale_huber_heightfield_v1"}}}}
        if not blocked:
            np.savez_compressed(research_dir / "research_costmap.npz", **costmap())
            research.update(costmap_file="research_costmap.npz", costmap_sha256=export.sha256(research_dir / "research_costmap.npz"))
        path = research_dir / "research_plan.json"
        path.write_text(json.dumps(research))
        data = {"run_dir": run, "map": saved_map, "geometry": {"geometry_fingerprint": "geometry-fixture"},
                "research_plan": deepcopy(research), "input_sha256": {str(path): export.sha256(path)}}
        data["research_plan"]["validated_path"] = np.empty((0, 3))
        return path, data

    def test_exact_source_hashes_and_no_path_still_allow_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            path, data = self.fixture(Path(directory))
            result = export.load_voxel_slope_overlay(path, data, load_npz_checked)
            self.assertEqual(result["facts"]["availability"], "available")
            self.assertEqual(result["facts"]["selected_point_count"], 2)
            self.assertEqual(len(result["input_sha256"]), 6)

    def test_box_coverage_provenance_and_research_references_are_displayed(self):
        with tempfile.TemporaryDirectory() as directory:
            path, data = self.fixture(Path(directory))
            raw = json.loads(path.read_text())
            raw["diagnostics"]["planning_check_metadata"]["terrain_estimation"]["method"] = "observed_lower_voxel_two_scale_huber_heightfield_v2_box_coverage"
            raw["research_calibration"] = {
                "level_reference": {"availability": "applied", "declared_reference_grade_percent": 0.},
                "camera_height_reference": {"availability": "applied", "camera_height_metres": 1.5}}
            path.write_text(json.dumps(raw))
            data["research_plan"] = deepcopy(raw)
            data["research_plan"]["validated_path"] = np.empty((0, 3))
            data["input_sha256"][str(path)] = export.sha256(path)
            result = export.load_voxel_slope_overlay(path, data, load_npz_checked)
            self.assertIn("Reference floor declared level (0%)", result["facts"]["legend"])
            self.assertIn("camera height 1.5 m", result["facts"]["legend"])
            self.assertEqual(result["facts"]["research_calibration"], raw["research_calibration"])
            np.testing.assert_array_equal(result["points"], classify()["points"])

    def test_unknown_terrain_method_is_rejected_even_with_matching_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            path, data = self.fixture(Path(directory))
            raw = json.loads(path.read_text())
            raw["diagnostics"]["planning_check_metadata"]["terrain_estimation"]["method"] = "unidentified_estimator"
            path.write_text(json.dumps(raw))
            data["research_plan"] = deepcopy(raw)
            data["research_plan"]["validated_path"] = np.empty((0, 3))
            data["input_sha256"][str(path)] = export.sha256(path)
            with self.assertRaisesRegex(ValueError, "provenance"):
                export.load_voxel_slope_overlay(path, data, load_npz_checked)

    def test_tampered_raw_voxels_or_costmap_are_rejected(self):
        for target in ("voxel", "costmap"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                path, data = self.fixture(Path(directory))
                tamper = data["run_dir"] / "map/voxels.npz" if target == "voxel" else path.parent / "research_costmap.npz"
                with tamper.open("ab") as stream:
                    stream.write(b"tampered")
                with self.assertRaisesRegex(ValueError, "checksum differs"):
                    export.load_voxel_slope_overlay(path, data, load_npz_checked)

    def test_rehashed_voxel_centers_still_must_match_native_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            path, data = self.fixture(Path(directory))
            indices, centers = inputs()
            centers[0, 2] += .1
            np.savez_compressed(data["run_dir"] / "map/voxels.npz", voxel_indices=indices, centers=centers)
            data["research_plan"]["source"]["voxel_map_sha256"] = export.sha256(data["run_dir"] / "map/voxels.npz")
            raw = {k: v for k, v in data["research_plan"].items() if k != "validated_path"}
            path.write_text(json.dumps(raw))
            data["input_sha256"][str(path)] = export.sha256(path)
            with self.assertRaisesRegex(ValueError, "native grid"):
                export.load_voxel_slope_overlay(path, data, load_npz_checked)

    def test_blocked_no_plane_has_truthful_empty_overlay_and_verified_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path, data = self.fixture(Path(directory), blocked=True)
            result = export.load_voxel_slope_overlay(path, data, load_npz_checked)
            self.assertEqual(result["facts"]["availability"], "unavailable")
            self.assertEqual(result["facts"]["reason"], "No fitted ground")
            self.assertEqual(len(result["points"]), 0)
            self.assertIn("No reliable ground", result["facts"]["legend"])
            self.assertEqual(len(result["input_sha256"]), 5)
            with (data["run_dir"] / "map/voxels.npz").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum differs"):
                export.load_voxel_slope_overlay(path, data, load_npz_checked)

    def test_missing_costmap_for_unblocked_result_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path, data = self.fixture(Path(directory), blocked=True)
            data["research_plan"]["status"] = "ok"
            path.write_text(json.dumps({k: v for k, v in data["research_plan"].items() if k != "validated_path"}))
            data["input_sha256"][str(path)] = export.sha256(path)
            with self.assertRaisesRegex(ValueError, "explicitly blocked"):
                export.load_voxel_slope_overlay(path, data, load_npz_checked)


class SceneTests(unittest.TestCase):
    def test_only_final_adds_native_points_and_serialized_label_to_cleanup(self):
        calls = []
        scene = SimpleNamespace(add_point_cloud=lambda *a, **kw: calls.append(("points", a, kw)) or object(),
                                add_label=lambda *a, **kw: calls.append(("label", a, kw)) or object())
        viewer = SimpleNamespace(server=SimpleNamespace(scene=scene))
        overlays = SimpleNamespace(handles=["existing semantic handle"])
        payload = classify()
        overlay = export.FinalVoxelSlopeOverlay(viewer, payload, 3, [1., 2., 3.])
        overlay.tick(0, overlays)
        overlay.tick(1, overlays)
        self.assertEqual(calls, [])
        overlay.tick(2, overlays)
        overlay.tick(2, overlays)
        self.assertEqual([call[0] for call in calls], ["points", "label"])
        self.assertEqual(len(overlays.handles), 3)
        np.testing.assert_array_equal(calls[0][2]["points"], payload["points"].astype(np.float32))
        self.assertEqual(calls[1][1][1], export.SLOPE_LEGEND)

    def test_frozen_segmentation_verifier_and_original_exporter_are_preserved(self):
        import ast
        original = Path(export.__file__).with_name("export_lingbot_interactive.py")
        self.assertEqual(export.sha256(original), "c937a2233f16d6c3852dd16bb848cf2e677c900e660591f4c4fa7c1bc8393a35")
        def function(path):
            return next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "verify_segmentation")
        revised=function(Path(export.__file__))
        # The sole admission change permits two separately reviewed route versions.
        admissions=[node for node in ast.walk(revised) if isinstance(node,ast.Compare)
                    and any(isinstance(op,ast.NotIn) for op in node.ops)
                    and any(isinstance(item,ast.Name) and item.id=='ROUTE_SEGMENTATION_SHA'
                            for item in ast.walk(node))]
        self.assertEqual(len(admissions),1)
        admission=admissions[0]
        self.assertEqual([node.id for node in admission.comparators[0].elts],
                         ['SEGMENTATION_SHA','ROUTE_SEGMENTATION_SHA','LEGACY_ROUTE_SEGMENTATION_SHA'])
        admission.ops=[ast.NotEq()];admission.comparators=[ast.Name(id='SEGMENTATION_SHA',ctx=ast.Load())]
        self.assertEqual(ast.dump(function(original)), ast.dump(revised))
        self.assertEqual(export.SEGMENTATION_SHA, "d084e18f9546316788a78393580b6f3916929a3f442ede75f4ab5b2a0c86d8c7")
        self.assertEqual(export.LEGACY_ROUTE_SEGMENTATION_SHA,
                         "95463f4271695c3b156080079a29aa83f568d4721c9a999c3a9c075f1f097e6a")
        self.assertEqual(export.sha256(Path(export.__file__).with_name('export_segmentation_route_recording.py')),
                         export.ROUTE_SEGMENTATION_SHA)


if __name__ == "__main__":
    unittest.main()
