"""Small relative-geometry examples; original rows, no planner or body model."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from cost_map import build_cost_map, obstacle_distance_squared
from slope import estimate_slope


def floor_points(angle=0.0):
    x, y = np.meshgrid(np.arange(-6, 7) * .01, np.arange(-6, 7) * .01)
    return np.column_stack((x.ravel(), y.ravel(), x.ravel() * np.tan(angle)))


def row_at(points, x=0., y=0.):
    return int(np.argmin(np.linalg.norm(points[:, :2] - [x, y], axis=1)))


class CostMapTests(unittest.TestCase):
    def build(self, points, slope=None, reference=(0, 0, 1), **changes):
        config = dict(radius=.081, support_mask=np.ones(len(points), dtype=bool))
        config.update(changes)
        if slope is None:
            slope = np.zeros(len(points))
        return build_cost_map(points, slope, reference, config)

    def test_flat_floor_cost_and_symmetric_connections(self):
        points = floor_points()
        costs = self.build(points)
        a, b = row_at(points), row_at(points, .02)
        assert_allclose(costs.cost, 1 + np.exp(-1))
        self.assertIn(b, costs.neighbours(a))
        self.assertIn(a, costs.neighbours(b))
        self.assertNotIn(a, costs.neighbours(a))

    def test_existing_slope_on_ramp_and_excessive_slope(self):
        angle = np.deg2rad(20)
        points = floor_points(angle)
        slope = estimate_slope(points, [0, 0, 1], {"radius": .026})
        centre, other = row_at(points), row_at(points, .02)
        self.assertAlmostEqual(slope[centre], angle)
        costs = self.build(points, slope)
        assert_allclose(costs.cost[centre],
                        1 + (angle / (np.pi / 6)) ** 2 + np.exp(-1))
        self.assertIn(other, costs.neighbours(centre))
        steep = self.build(floor_points(np.deg2rad(35)),
                           np.full(len(points), np.deg2rad(35)))
        self.assertTrue(np.isinf(steep.cost).all())

    def test_missing_reference_support_and_slopes_are_unknown(self):
        points = floor_points()
        cases = [self.build(points, reference=None),
                 self.build(points, support_mask=np.zeros(len(points), bool)),
                 self.build(points, np.full(len(points), np.nan)),
                 build_cost_map(points, np.zeros(len(points)), [0, 0, 1],
                                {"radius": .081})]
        for costs in cases:
            self.assertTrue(np.isinf(costs.cost).all())
            self.assertEqual(len(costs.neighbours(row_at(points))), 0)

    def test_unknown_floor_slope_is_not_an_obstacle(self):
        points = floor_points()
        slope = np.zeros(len(points))
        centre, nearby = row_at(points), row_at(points, .01)
        slope[centre] = np.nan
        costs = self.build(points, slope)
        self.assertTrue(np.isinf(costs.cost[centre]))
        assert_allclose(costs.cost[nearby], 1 + np.exp(-1))

    def test_obstacle_distance_raises_cost_and_contact_is_infeasible(self):
        floor = floor_points()
        centre = row_at(floor)
        mask = np.r_[np.ones(len(floor), bool), False]
        near = self.build(np.vstack((floor, [0, 0, -.02])), support_mask=mask)
        far = self.build(np.vstack((floor, [0, 0, -.06])), support_mask=mask)
        assert_allclose(near.cost[centre], 1 + np.exp(-.02 / .081))
        self.assertLess(far.cost[centre], near.cost[centre])
        contact = self.build(np.vstack((floor, [0, 0, .001])), support_mask=mask)
        self.assertTrue(np.isinf(contact.cost[[centre, len(floor)]]).all())

    def test_isolated_labels_and_duplicates_do_not_supply_support(self):
        floor = floor_points()
        points = np.vstack((floor, np.tile([1., 1., 0.], (20, 1))))
        costs = self.build(points)
        self.assertTrue(np.isfinite(costs.cost[:len(floor)]).all())
        self.assertTrue(np.isinf(costs.cost[len(floor):]).all())
        # Five distinct labelled positions cannot become six through duplication.
        few = np.tile(floor[:5], (10, 1))
        self.assertTrue(np.isinf(self.build(few).cost).all())

    def test_floor_hole_prevents_connection_between_finite_endpoints(self):
        floor = floor_points()
        points = floor[np.abs(floor[:, 0]) >= .03]
        costs = self.build(points, connection_radius=.07)
        a, b = row_at(points, -.03), row_at(points, .03)
        self.assertTrue(np.isfinite(costs.cost[[a, b]]).all())
        self.assertNotIn(b, costs.neighbours(a))
        self.assertNotIn(a, costs.neighbours(b))

    def test_thin_barrier_between_samples_blocks_both_directions(self):
        floor = floor_points()
        a, b = row_at(floor, -.02), row_at(floor, .02)
        clear = self.build(floor, obstacle_margin=.0005)
        self.assertIn(b, clear.neighbours(a))
        # This obstacle lies between edge samples and misses every floor row.
        points = np.vstack((floor, [.0013, 0, .0002]))
        mask = np.r_[np.ones(len(floor), bool), False]
        blocked = self.build(points, support_mask=mask, obstacle_margin=.0005)
        self.assertTrue(np.isfinite(blocked.cost[:-1]).all())
        self.assertNotIn(b, blocked.neighbours(a))
        self.assertNotIn(a, blocked.neighbours(b))

    def test_vertical_connection_is_rejected(self):
        floor = floor_points()
        points = np.vstack((floor, floor + [0, 0, .03]))
        costs = self.build(points)
        a, b = row_at(floor), row_at(floor) + len(floor)
        self.assertTrue(np.isfinite(costs.cost[[a, b]]).all())
        self.assertNotIn(b, costs.neighbours(a))

    def test_overhead_log_blocks_nodes_but_ground_thickness_is_retained(self):
        floor = np.vstack((floor_points(), floor_points() + [0, 0, -.003]))
        points = np.vstack((floor, [0, 0, .04]))
        mask = np.r_[np.ones(len(floor), bool), False]
        costs = self.build(points, support_mask=mask)
        lower = row_at(floor_points()) + len(floor_points())
        self.assertTrue(np.isinf(costs.cost[[row_at(floor), lower]]).all())
        clear = self.build(floor)
        self.assertTrue(np.isfinite(clear.cost).all())
        assert_array_equal(costs.points, points)

    def test_overhead_obstacle_between_nodes_blocks_entire_edge(self):
        floor = floor_points()
        a, b = row_at(floor, -.02), row_at(floor, .02)
        points = np.vstack((floor, [.0013, 0, .04]))
        costs = self.build(points, support_mask=np.r_[np.ones(len(floor), bool), False],
                           obstacle_margin=.0005)
        self.assertTrue(np.isfinite(costs.cost[:-1]).all())
        self.assertNotIn(b, costs.neighbours(a))
        self.assertNotIn(a, costs.neighbours(b))
        self.assertFalse(costs.obstacle_free(points[a], points[b]))

    def test_ceiling_and_obstacles_below_floor_do_not_block_upward_clearance(self):
        floor = floor_points()
        points = np.vstack((floor, [0, 0, .10], [0, 0, -.02]))
        costs = self.build(points, support_mask=np.r_[np.ones(len(floor), bool), False, False])
        centre, other = row_at(floor), row_at(floor, .02)
        self.assertTrue(np.isfinite(costs.cost[:len(floor)]).all())
        self.assertIn(other, costs.neighbours(centre))

    def test_exact_swept_edge_interior_boundaries_and_coincident_points(self):
        a, b, up = np.array([0., 0., 0.]), np.array([2., 0., .5]), np.array([0., 0., 1.])
        samples = np.array([[1, .03, .75], [1, 0, 1.3], [-.04, 0, .5],
                            [0, 0, -.05], [2, 0, 1.55], [1, 0, 2.]])
        expected = [0.03 ** 2, .05 ** 2 / (1 + .25 ** 2), .04 ** 2,
                    .05 ** 2, .05 ** 2, .75 ** 2 / (1 + .25 ** 2)]
        assert_allclose(obstacle_distance_squared(samples, a, b, up, 1.), expected)
        assert_allclose(obstacle_distance_squared(samples, b, a, up, 1.), expected)
        assert_allclose(obstacle_distance_squared(np.array([[0., 0., .5], [0., 0., 1.1]]),
                                                 a, a, up, 1.), [0., .01])

    def test_zero_margin_detects_exact_obstacle_and_retains_clear_coincident_rows(self):
        floor = floor_points()
        centre = row_at(floor)
        points = np.vstack((floor, floor[centre], [0, 0, .04]))
        mask = np.r_[np.ones(len(points) - 1, bool), False]
        blocked = self.build(points, support_mask=mask, obstacle_margin=0.)
        self.assertTrue(np.isinf(blocked.cost[[centre, len(floor)]]).all())
        clear = self.build(points[:-1], obstacle_margin=0.)
        self.assertIn(len(floor), clear.neighbours(centre))
        self.assertIn(centre, clear.neighbours(len(floor)))

    def test_bounding_box_corner_does_not_hide_a_later_obstacle(self):
        floor = floor_points()
        # First obstacle is nearer in the compressed Chebyshev index but lies
        # outside the round margin. The later one is inside the swept volume.
        points = np.vstack((floor, [.0035, .0035, .04], [.0039, 0, .04]))
        mask = np.r_[np.ones(len(floor), bool), False, False]
        full = self.build(points, support_mask=mask)
        small = self.build(points, support_mask=mask, max_neighbors=8)
        self.assertTrue(np.isinf(full.cost[row_at(floor)]))
        assert_allclose(full.cost, small.cost)

    def test_spatial_index_matches_all_obstacles_for_nodes_and_edges(self):
        floor = floor_points()
        rng = np.random.default_rng(917)
        obstacles = rng.uniform([-.08, -.08, -.04], [.08, .08, .14], (120, 3))
        points = np.vstack((floor, obstacles))
        costs = self.build(points, support_mask=np.r_[np.ones(len(floor), bool),
                                                     np.zeros(len(obstacles), bool)], max_neighbors=8)
        margin2, height = costs.config["obstacle_margin"] ** 2, costs.config["clearance_height"]
        for row, position in enumerate(floor):
            blocked = np.any(obstacle_distance_squared(obstacles, position, position,
                                                        costs.reference, height) <= margin2)
            self.assertEqual(np.isinf(costs.cost[row]), blocked)
        for a, b in rng.uniform(-.07, .07, (60, 2, 3)):
            clear = np.all(obstacle_distance_squared(obstacles, a, b, costs.reference, height) > margin2)
            self.assertEqual(costs.obstacle_free(a, b), clear)

    def test_original_rows_duplicates_nonfinite_and_inputs_unchanged(self):
        floor = floor_points()
        centre = row_at(floor)
        points = np.vstack((floor, floor[centre], [np.nan, 0, 0], [np.inf, 0, 0]))
        slope = np.r_[np.zeros(len(floor) + 1), np.nan, np.nan]
        mask = np.ones(len(points), bool)
        config = {"radius": .081, "support_mask": mask}
        original, original_slope, original_mask = points.copy(), slope.copy(), mask.copy()
        costs = build_cost_map(points, slope, [0, 0, 1], config)
        self.assertEqual(costs.cost.shape, (len(points),))
        self.assertTrue(np.isinf(costs.cost[-2:]).all())
        self.assertIn(len(floor), costs.neighbours(centre))
        self.assertIn(centre, costs.neighbours(len(floor)))
        order = np.random.default_rng(4).permutation(len(points))
        reordered = self.build(points[order], slope[order], support_mask=mask[order])
        assert_allclose(reordered.cost, costs.cost[order])
        moved_centre = int(np.flatnonzero(order == centre)[0])
        assert_array_equal(np.sort(order[reordered.neighbours(moved_centre)]),
                           costs.neighbours(centre))
        assert_array_equal(points, original)
        assert_array_equal(slope, original_slope)
        assert_array_equal(mask, original_mask)
        self.assertEqual(set(config), {"radius", "support_mask"})

    def test_small_batches_retain_late_obstacles_and_all_neighbours(self):
        floor = floor_points()
        # Put the obstacle after several batches of duplicated floor observations.
        points = np.vstack((np.tile(floor, (3, 1)), [0, 0, .002]))
        mask = np.r_[np.ones(len(points) - 1, bool), False]
        full = self.build(points, support_mask=mask)
        small = self.build(points, support_mask=mask, max_neighbors=8)
        assert_allclose(small.cost, full.cost)
        self.assertTrue(np.isinf(small.cost[row_at(floor)]))
        row = row_at(floor, .03)
        self.assertGreater(len(full.neighbours(row)), 8)
        assert_array_equal(small.neighbours(row), full.neighbours(row))

    def test_uniform_scale_preserves_every_cost_and_neighbour_id(self):
        points = np.vstack((floor_points(), [.021, .013, .025], [0, 0, .001]))
        mask = np.r_[np.ones(len(points) - 2, bool), False, False]
        row = row_at(points, .03)
        explicit = dict(clearance_scale=.055, connection_radius=.041,
                        max_gap=.021, obstacle_margin=.003)
        for distances in ({}, explicit):
            base = self.build(points, support_mask=mask, **distances)
            for scale in (.001, 37.):
                with self.subTest(distances=distances, scale=scale):
                    scaled = self.build(points * scale, support_mask=mask,
                                        radius=.081 * scale,
                                        **{key: value * scale for key, value in distances.items()})
                    assert_allclose(scaled.cost, base.cost, rtol=1e-13)
                    assert_array_equal(scaled.neighbours(row), base.neighbours(row))

    def test_rigid_transform_with_reference_preserves_costs_and_connections(self):
        points = np.vstack((floor_points(), [.017, .013, .025]))
        mask = np.r_[np.ones(len(points) - 1, bool), False]
        original = self.build(points, support_mask=mask)
        angle = .7
        rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                             [-np.sin(angle), 0, np.cos(angle)]])
        transformed = self.build(points @ rotation.T + [.4, -.3, .2],
                                 reference=rotation[:, 2], support_mask=mask)
        assert_allclose(transformed.cost, original.cost)
        row = row_at(points)
        assert_array_equal(transformed.neighbours(row), original.neighbours(row))

    def test_invalid_config(self):
        points = floor_points()
        for change in ({"radius": 0}, {"radius": np.nan}, {"radius": True},
                       {"w_slope": -1}, {"w_clearance": np.inf},
                       {"clearance_scale": 0}, {"max_slope": np.pi / 2},
                       {"max_neighbors": 7}, {"max_neighbors": 8.5},
                       {"min_support": 5}, {"min_support": 6.5},
                       {"clearance_height": 0}, {"clearance_height": np.inf},
                       {"height": 1.8}, {"frames": []},
                       {"support_mask": np.ones(len(points), dtype=int)},
                       {"support_mask": np.ones(len(points) - 1, dtype=bool)}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.build(points, **change)
        with self.assertRaises(ValueError):
            build_cost_map(points, np.zeros(len(points)), [0, 0, 1], {})


if __name__ == "__main__":
    unittest.main()
