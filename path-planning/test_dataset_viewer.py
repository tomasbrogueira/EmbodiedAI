"""Read-only viewer fixtures for valid empty and non-traversable maps."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

try:
    import viser
except ModuleNotFoundError as error:
    if error.name != "viser":
        raise
    sys.modules["viser"] = types.ModuleType("viser")  # These fixtures do not start a GUI.

from visualization.dataset_viewer import colour_view, load_view, palette, signature
from visualization.dataset_status import adapter_hash, outcome, reviewed_output


class ViewerOutcomeTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parent
        self.temporary = tempfile.TemporaryDirectory(prefix="viewer-outcome-test-", dir=root)
        self.output = Path(self.temporary.name).resolve()
        assert self.output.is_relative_to(root)
        self.addCleanup(self.temporary.cleanup)
        self.data = self.output / "data"
        self.data.mkdir()
        self.video = {"output": str(self.output)}
        self.report = {"status": "missing_reference", "selected": {"start": None, "goal": None}}
        self.cameras = np.array([[1., 2., 3.], [4., 5., 6.]])
        np.save(self.data / "cameras.npy", self.cameras)
        np.save(self.data / "reference.npy", np.full(3, np.nan))
        self.reference = {"reason": "insufficient_reference_frames", "windows_checked": 10}
        (self.data / "provenance.json").write_text(json.dumps({"reference_report": self.reference}))

    def map(self, points):
        np.save(self.data / "points.npy", np.asarray(points))
        np.save(self.data / "cost.npy", np.full(len(points), np.inf))

    def test_empty_context_keeps_cameras_and_reference_diagnostics(self):
        self.map(np.empty((0, 3)))
        view = load_view(self.video, self.report)
        self.assertTrue(view["empty_map"])
        self.assertEqual(view["xyz"].shape, (0, 3))
        self.assertTrue(np.array_equal(view["cameras"], self.cameras))
        self.assertTrue(np.array_equal(view["box"], self.cameras))
        self.assertEqual(view["reference_report"], self.reference)
        self.assertFalse(view["reference_available"])
        self.assertEqual(len(view["route"]), 0)

    def test_unavailable_map_is_grey_without_route(self):
        self.map([[0., 0., 0.], [1., 0., 0.]])
        self.report["status"] = "no_supported_pose"
        np.savez(self.output / "route.npz", ids=[0], positions=[[0., 0., 0.]])
        view = load_view(self.video, self.report)
        self.assertFalse(view["empty_map"])
        self.assertEqual(view["finite_sample"], 0)
        self.assertTrue(np.all(view["cost_colours"] == [150, 155, 165]))
        self.assertEqual(len(view["route"]), 0)

    def test_malformed_and_nonfinite_maps_remain_errors(self):
        for points in (np.empty(0), np.array([[np.nan, 0., 0.]])):
            self.map(points)
            with self.assertRaises(ValueError):
                load_view(self.video, self.report)

    def test_finite_costs_use_heatmap_and_unavailable_points_are_grey(self):
        colours = palette(np.array([-1., 0., .5, 1., 100., np.inf, np.nan]), [0., 1.])
        np.testing.assert_array_equal(colours[:2], [[30, 70, 240], [30, 70, 240]])
        np.testing.assert_array_equal(colours[3:5], [[235, 50, 40], [235, 50, 40]])
        np.testing.assert_array_equal(colours[5], [150, 155, 165])
        np.testing.assert_array_equal(colours[6], [150, 155, 165])

    def test_steepness_uses_angles_even_when_cost_is_infinite(self):
        self.map([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
        np.save(self.data / "slope.npy", [0., np.pi / 4, np.pi / 2])
        view = load_view(self.video, self.report)
        self.assertTrue(np.all(view["cost_colours"] == [150, 155, 165]))
        np.testing.assert_array_equal(view["steepness_colours"], palette(np.array([0., 45., 90.]), [0., 90.]))
        self.assertEqual(view["slope_sample"], 3)

    def test_invalid_or_unknown_angles_are_grey(self):
        self.map(np.zeros((5, 3)))
        np.save(self.data / "slope.npy", [-.1, np.pi, np.nan, np.inf, -np.inf])
        view = load_view(self.video, self.report)
        self.assertEqual(view["slope_sample"], 0)
        self.assertTrue(np.all(view["steepness_colours"] == [150, 155, 165]))
        self.assertIn("unknown slope", colour_view(view, "Steepness")[1])

    def test_steepness_rejects_misaligned_or_malformed_rows(self):
        self.map(np.zeros((3, 3)))
        for slopes in ([0., 1.], np.zeros((3, 1)), np.array(["flat", "steep", "unknown"])):
            np.save(self.data / "slope.npy", slopes)
            with self.assertRaisesRegex(ValueError, "Slopes do not match"):
                load_view(self.video, self.report)

    def test_geometry_colours_remain_available_while_costs_are_stale(self):
        self.partial_route()
        np.save(self.data / "slope.npy", [0., np.pi / 4, np.pi / 2])
        rgb = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8)
        np.save(self.data / "colours.npy", rgb)
        view = load_view(self.video, self.report, stale_costs=True)
        np.testing.assert_array_equal(colour_view(view, "Source")[0], rgb)
        self.assertEqual(view["slope_sample"], 3)
        self.assertEqual(view["finite_sample"], 0)
        self.assertEqual(len(view["route"]), 0)

    def test_colour_selection_preserves_camera_and_route_arrays(self):
        self.partial_route()
        np.save(self.data / "slope.npy", [0., np.pi / 4, np.pi / 2])
        view = load_view(self.video, self.report)
        route, cameras = view["route"].copy(), view["cameras"].copy()
        for mode, field, text in (("Source", "source", "recorded appearance"),
                                  ("Cost map", "cost_colours", "costly but traversable"),
                                  ("Steepness", "steepness_colours", "90° vertical")):
            colours, legend = colour_view(view, mode)
            self.assertIs(colours, view[field])
            self.assertIn(text, legend)
            np.testing.assert_array_equal(view["route"], route)
            np.testing.assert_array_equal(view["cameras"], cameras)

    def test_slope_file_becoming_available_changes_refresh_signature(self):
        path = self.data / "slope.npy"
        before = signature([path])
        np.save(path, [0.])
        self.assertNotEqual(before, signature([path]))

    def partial_route(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
        self.map(points)
        np.save(self.data / "cost.npy", [1., 2., 1.])
        self.report.update(status="partial", optimal=True, correspondence_verified=True,
                           goal_reached=False, optimality_scope="nearest_reachable_endpoint", route_points=2,
                           selected={"start": {"id": 0, "frame_id": 0}, "goal": {"id": 2, "frame_id": 1}},
                           reached={"id": 1, "ground_point": points[1].tolist()})
        np.savez(self.output / "route.npz", ids=np.array([0, 1]), positions=points[:2])
        return points

    def test_partial_route_retains_intended_goal_and_camera_interval(self):
        points = self.partial_route()
        view = load_view(self.video, self.report)
        self.assertTrue(view["partial"])
        self.assertEqual(view["segment"], (0, 1))
        np.testing.assert_array_equal(view["cameras"], self.cameras)
        np.testing.assert_array_equal(view["route"], points[:2])
        np.testing.assert_array_equal(view["endpoints"][-1]["ground"], points[2])
        self.assertEqual(view["endpoints"][-1]["label"], "goal")

    def test_partial_route_requires_proven_unreachable_goal(self):
        self.partial_route()
        self.report["goal_reached"] = True
        with self.assertRaisesRegex(ValueError, "unreachable goal"):
            load_view(self.video, self.report)

    def test_partial_route_cannot_change_start_or_claim_wrong_endpoint(self):
        points = self.partial_route()
        for ids in ([1], [0, 2]):
            np.savez(self.output / "route.npz", ids=np.array(ids), positions=points[ids])
            with self.assertRaisesRegex(ValueError, "fixed start"):
                load_view(self.video, self.report)

    def test_verified_route_cannot_contain_unavailable_point(self):
        self.partial_route()
        np.save(self.data / "cost.npy", [1., np.inf, 1.])
        with self.assertRaisesRegex(ValueError, "unavailable point"):
            load_view(self.video, self.report)

    def test_partial_routes_are_counted_separately(self):
        self.partial_route()
        self.assertEqual(outcome({"stage": "complete"}, self.report), "verified partial routes")
        self.report["correspondence_verified"] = False
        self.assertEqual(outcome({"stage": "complete"}, self.report), "completed without route")

    def test_partial_route_is_withheld_during_rebuild(self):
        self.partial_route()
        (self.data / "provenance.json").write_text(json.dumps({
            "source_hashes": {"path-planning/cost_map.py": "current"},
            "prepare_adapter_sha256": "prepare", "calibration_association_sha256": "association"}))
        state, report, stale = reviewed_output(self.output, {"stage": "path", "status": "running"},
                                               self.report, "current", "prepare", "association")
        self.assertFalse(stale)
        self.assertEqual(report["status"], "pending")
        self.assertEqual(len(load_view(self.video, report)["route"]), 0)

    def test_calibration_code_changes_withhold_old_costs_and_route(self):
        self.partial_route()
        for saved in ({}, {"prepare_adapter_sha256": "old", "calibration_association_sha256": "association"},
                      {"prepare_adapter_sha256": "prepare", "calibration_association_sha256": "old"}):
            with self.subTest(saved=saved):
                (self.data / "provenance.json").write_text(json.dumps({
                    "source_hashes": {"path-planning/cost_map.py": "current"}, **saved}))
                state, report, stale = reviewed_output(self.output, {"stage": "complete", "status": "partial"},
                                                       self.report, "current", "prepare", "association")
                self.assertTrue(stale)
                self.assertEqual(state["stage"], "updated_feasibility")
                self.assertIsNone(report["selected"]["start"])
                view = load_view(self.video, report, stale)
                self.assertEqual(len(view["route"]), 0)
                self.assertEqual(view["finite_sample"], 0)
                self.assertTrue(np.all(view["cost_colours"] == [150, 155, 165]))

    def test_current_calibration_receipt_allows_verified_partial_route(self):
        self.partial_route()
        (self.data / "provenance.json").write_text(json.dumps({
            "source_hashes": {"path-planning/cost_map.py": "current"},
            "prepare_adapter_sha256": "prepare", "calibration_association_sha256": "association"}))
        _, report, stale = reviewed_output(self.output, {"stage": "complete", "status": "partial"},
                                           self.report, "current", "prepare", "association")
        self.assertFalse(stale)
        self.assertEqual(len(load_view(self.video, report, stale)["route"]), 2)

    def test_adapter_hash_refreshes_when_source_changes(self):
        script = self.output / "dataset_prepare.py"
        with patch("visualization.dataset_status.SOURCE_ROOT", self.output):
            self.assertIsNone(adapter_hash(script.name))
            script.write_text("first calibration")
            first = adapter_hash(script.name)
            script.write_text("second calibration")
            self.assertNotEqual(first, adapter_hash(script.name))


if __name__ == "__main__":
    unittest.main()
