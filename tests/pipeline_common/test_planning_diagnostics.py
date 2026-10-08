"""Independent composition checks for additive planner rejection diagnostics."""
import json
from io import BytesIO
import unittest

import numpy as np

from pipeline_common.planning import (
    BLOCKED, TRAVERSABLE, UNKNOWN, DIAGNOSTIC_MASK_NAMES, build_costmap,
)


def fixture(*, mixed=False):
    shape, resolution = (7, 7), .1
    config = {
        "planning": {"resolution": resolution, "origin": [0., 0.],
                     "shape": list(shape), "support_height": 0.},
        "robot": {"version": "diagnostic_fixture_v1", "fixture": True,
                  "footprint_radius": .1, "height": .35, "clearance": .02,
                  "max_slope_degrees": 25., "max_step": .08,
                  "max_roughness": .03, "min_support_points": 6,
                  "unknown_rule": "blocked", "semantic_costs": {}},
    }
    points = []
    for y in range(shape[0]):
        for x in range(shape[1]):
            if mixed and (y, x) in {(1, 5), (5, 1)}:
                continue
            for oy in (.15, .5, .85):
                for ox in (.15, .5, .85):
                    height = .15 if mixed and x >= 5 else 0.
                    if mixed and (y, x, oy, ox) == (3, 3, .5, .5):
                        height = .12
                    point = [(x + ox) * resolution, (y + oy) * resolution, height]
                    points.append(point)
                    if mixed and (y, x) == (2, 2):
                        points.append([point[0], point[1], height + .2])
    if mixed:
        # Pure vertical support in its own missing-floor cell is a wall, not a
        # horizontal floor patch. Several independent Y samples keep it a line.
        points.extend([[.55, y, z] for y in (.115, .15, .185)
                       for z in np.linspace(0., .8, 17)])
    return np.asarray(points), config


def build(points, config, **kwargs):
    return build_costmap(points, config, up=[0., 0., 1.],
                         scale={"verified": True, "source": "synthetic metres"},
                         **kwargs)


class PlanningDiagnosticTests(unittest.TestCase):
    def test_independent_reasons_compose_physical_and_support_decisions(self):
        points, config = fixture(mixed=True)
        arrays, metadata = build(points, config)
        for name in DIAGNOSTIC_MASK_NAMES:
            self.assertEqual(arrays[name].dtype, np.bool_)
            self.assertEqual(arrays[name].shape, (7, 7))
            self.assertEqual(metadata["diagnostic_counts"][name],
                             int(np.count_nonzero(arrays[name])))
        for reason in ("wall", "roughness", "clearance", "step"):
            self.assertTrue(arrays[reason + "_blocked_mask"].any(), reason)
        union = np.logical_or.reduce([arrays[name] for name in (
            "wall_blocked_mask", "roughness_blocked_mask",
            "clearance_blocked_mask", "step_blocked_mask", "slope_blocked_mask")])
        np.testing.assert_array_equal(union, arrays["physical_blocked_mask"])
        np.testing.assert_array_equal(arrays["physical_blocked_mask"],
                                      arrays["geometry_state"] == BLOCKED)
        np.testing.assert_array_equal(arrays["supported_mask"],
                                      arrays["geometry_state"] == TRAVERSABLE)
        np.testing.assert_array_equal(arrays["eligible_support_mask"],
                                      arrays["candidate_mask"] & ~union)
        np.testing.assert_array_equal(arrays["raw_candidate_mask"], arrays["candidate_mask"])
        self.assertTrue(np.any(arrays["roughness_blocked_mask"] &
                               arrays["clearance_blocked_mask"]))
        self.assertEqual(arrays["geometry_state"][5, 1], UNKNOWN)
        json.dumps(metadata, allow_nan=False)

    def test_circle_footprint_recomposition_preserves_unknown_and_obstacles(self):
        points, config = fixture(mixed=True)
        arrays, metadata = build(points, config)
        base = arrays["geometry_state"]
        expected_unknown, expected_blocked = np.zeros((7, 7), bool), np.zeros((7, 7), bool)
        # The .12m disk intersects these nine .1m boxes, including diagonal
        # boxes whose nearest corner is sqrt(2)*.05m away. No robot shrinking.
        for y in range(7):
            for x in range(7):
                for yy in range(y - 1, y + 2):
                    for xx in range(x - 1, x + 2):
                        if not (0 <= yy < 7 and 0 <= xx < 7):
                            expected_unknown[y, x] = True
                        elif base[yy, xx] == UNKNOWN:
                            expected_unknown[y, x] = True
                        elif base[yy, xx] == BLOCKED:
                            expected_blocked[y, x] = True
        np.testing.assert_array_equal(arrays["footprint_unknown_mask"], expected_unknown)
        np.testing.assert_array_equal(arrays["footprint_blocked_mask"], expected_blocked)
        expected_state = base.copy()
        expected_state[expected_unknown] = UNKNOWN
        expected_state[expected_blocked | (base == BLOCKED)] = BLOCKED
        np.testing.assert_array_equal(arrays["decision_state"], expected_state)
        rejected = (base == TRAVERSABLE) & (expected_unknown | expected_blocked)
        self.assertTrue(rejected.any())
        self.assertEqual(metadata["diagnostic_counts"]["supported_cells_rejected_by_footprint"],
                         int(np.count_nonzero(rejected)))
        self.assertTrue(np.isinf(arrays["costs"][expected_state != TRAVERSABLE]).all())

    def test_semantic_policy_does_not_change_geometric_reason_masks(self):
        points, config = fixture()
        plain, _ = build(points, config)
        config["robot"]["semantic_costs"] = {"excluded": {"blocked": True}}
        policy, metadata = build(points, config, semantic_points={
            "excluded": {"points": [[.35, .35, 0.]], "role": "hazard"}})
        for name in DIAGNOSTIC_MASK_NAMES:
            np.testing.assert_array_equal(plain[name], policy[name])
        self.assertEqual(plain["decision_state"][3, 3], TRAVERSABLE)
        self.assertEqual(policy["decision_state"][3, 3], BLOCKED)
        self.assertFalse(policy["physical_blocked_mask"][3, 3])
        self.assertEqual(metadata["diagnostic_counts"]["physical_blocked_mask"], 0)

    def test_blocked_input_schema_and_numeric_npz_roundtrip(self):
        points, config = fixture()
        arrays, metadata = build_costmap(points, config, units="reconstruction_units")
        self.assertEqual(metadata["availability"], "blocked_inputs")
        for name in DIAGNOSTIC_MASK_NAMES:
            self.assertEqual(arrays[name].shape, (0, 0))
            self.assertEqual(arrays[name].dtype, np.bool_)
            self.assertEqual(metadata["diagnostic_counts"][name], 0)
        arrays, _ = build(points, config)
        stream = BytesIO()
        np.savez_compressed(stream, **arrays)
        stream.seek(0)
        with np.load(stream, allow_pickle=False) as saved:
            for name in DIAGNOSTIC_MASK_NAMES:
                np.testing.assert_array_equal(arrays[name], saved[name])


if __name__ == "__main__":
    unittest.main()
