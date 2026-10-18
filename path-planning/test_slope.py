"""Small geometric checks for dense point-wise inclination and local quality."""
import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from slope import estimate_slope


def plane(degrees=0, size=31, noise=0):
    rng = np.random.default_rng(17)
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    xy = np.c_[x.ravel(), y.ravel()]
    if noise:
        xy += rng.uniform(-.003, .003, xy.shape)
    angle = np.deg2rad(degrees)
    normal = np.array([-np.sin(angle), 0, np.cos(angle)])
    return np.c_[xy, xy[:, 0] * np.tan(angle)] + rng.normal(0, noise, (len(xy), 1)) * normal


def slopes(points, **kwargs):
    return estimate_slope(points, [0, 0, 1], {'radius': .45}, **kwargs)


class SlopeTests(unittest.TestCase):
    def test_exact_and_noisy_flat_ramps(self):
        for degrees in (0, 20, 30):
            for noise in (0, .003, .01):
                with self.subTest(degrees=degrees, noise=noise):
                    p = plane(degrees, noise=noise)
                    s, q = slopes(p, return_quality=True)
                    self.assertGreaterEqual(np.isfinite(s).mean(), .99)
                    interior = (np.abs(p[:, :2]) < .65).all(axis=1)
                    error = np.abs(np.degrees(s[interior]) - degrees)
                    self.assertLess(np.nanpercentile(error, 95), 2 if noise else 1e-5)
                    self.assertTrue(((q >= 0) & (q <= 1)).all())

    def test_sparse_outliers_and_query_quality(self):
        p = plane(30, noise=.002)
        rng = np.random.default_rng(21)
        outliers = rng.choice(len(p), 20, replace=False)
        p[outliers] += [.0, .0, .15]
        s, q = slopes(p, return_quality=True)
        clean = np.ones(len(p), bool); clean[outliers] = False
        clean &= (np.abs(p[:, :2]) < .65).all(axis=1)
        self.assertGreater(np.isfinite(s[clean]).mean(), .99)
        self.assertLess(np.nanpercentile(np.abs(np.degrees(s[clean])-30), 95), 2)
        self.assertLess(np.median(q[outliers]), .1)
        self.assertGreater(np.median(q[clean]), .5)

    def test_mixed_surfaces_and_separate_layers(self):
        p = plane()
        # Parallel layers share inclination, but this says nothing about support.
        s, q = slopes(np.vstack((p, p+[0, 0, .02])), return_quality=True)
        centres = [len(p)//2, len(p)+len(p)//2]
        assert_allclose(s[centres], 0, atol=np.deg2rad(.1))
        assert_allclose(slopes(np.vstack((p, p+[0, 0, 1]))), 0, atol=1e-7)
        # Symmetric orthogonal sheets have no unique normal at their intersection.
        mixed = np.vstack((p, p[:, [2, 1, 0]]))
        s, q = estimate_slope(mixed, [0, 0, 1], {'radius': .45, 'neighbors': 2048}, return_quality=True)
        self.assertTrue(np.isnan(s[len(p)//2]))
        self.assertEqual(q[len(p)//2], 0)
        away = np.argmin(np.linalg.norm(p-[.8, 0, 0], axis=1))
        self.assertLess(s[away], 1e-7)

    def test_sparse_line_isotropic_and_narrow_strip(self):
        triangle = np.array([[0, 0, 0], [.1, 0, 0], [0, .1, 0]])
        line = np.c_[np.linspace(-1, 1, 41), np.zeros((41, 2))]
        for p in (np.vstack((triangle, triangle+[10, 0, 0])), line):
            s, q = slopes(p, return_quality=True)
            self.assertTrue(np.isnan(s).all()); self.assertFalse(q.any())
        isotropic = np.vstack((np.zeros(3), .1*np.eye(3), -.1*np.eye(3)))
        self.assertTrue(np.isnan(slopes(isotropic)[0]))
        strip = plane(); strip[:, 1] *= .01
        s, q = slopes(strip, return_quality=True)
        self.assertTrue(np.isfinite(s).all())
        assert_allclose(s, 0, atol=1e-7)
        self.assertLess(q[len(strip)//2], .02)

    def test_duplicates_do_not_supply_or_crowd_out_support(self):
        p = plane()
        self.assertTrue(np.isnan(slopes(np.tile(p[:5], (50, 1)))).all())
        duplicate = np.vstack((p, np.repeat(p[[len(p)//2]], 300, axis=0)))
        s, q = slopes(duplicate, return_quality=True)
        original, oq = slopes(p, return_quality=True)
        assert_allclose(s[:len(p)], original, atol=1e-7)
        assert_allclose(q[:len(p)], oq)
        assert_allclose(s[len(p):], original[len(p)//2], atol=1e-7)

    def test_units_rotation_order_and_inputs(self):
        p = plane(30, noise=.002)
        ref = np.array([0., 0., 2.]); config = {'radius': .45}
        original = p.copy(); base, bq = estimate_slope(p, ref, config, return_quality=True)
        rotation, _ = np.linalg.qr([[1., 2., 3.], [3., -2., 1.], [2., 1., -3.]])
        order = np.random.default_rng(4).permutation(len(p))
        for scale in (.001, 37.):
            s, q = estimate_slope(scale*p[order]@rotation.T+[4, -2, 7], -7*rotation[:, 2], {'radius': .45*scale}, return_quality=True)
            assert_allclose(s, base[order], atol=1e-8)
            assert_allclose(q, bq[order], atol=1e-7)
        assert_array_equal(p, original); assert_array_equal(ref, [0, 0, 2])
        self.assertEqual(config, {'radius': .45})

    def test_nonfinite_and_empty_rows(self):
        p = np.vstack((plane(), [np.nan, 0, 0], [0, np.inf, 0]))
        s, q = slopes(p, return_quality=True)
        self.assertEqual(s.shape, (len(p),)); self.assertTrue(np.isnan(s[-2:]).all())
        self.assertFalse(q[-2:].any()); assert_allclose(s[:-2], 0, atol=1e-7)
        self.assertEqual(slopes(np.empty((0, 3))).shape, (0,))

    def test_partially_filled_neighbour_batches(self):
        p = np.vstack((.1*plane(30, size=5)[:20], plane()+[10, 0, 0]))
        outputs = [estimate_slope(p, [0, 0, 1], {'radius': .3, 'neighbors': k}, return_quality=True) for k in (64, 128)]
        for first, second in zip(*outputs):
            assert_allclose(first[:20], second[:20])
        assert_allclose(outputs[0][0][:20], np.deg2rad(30), atol=1e-7)

    def test_invalid_reference_and_config(self):
        for normal in ([0, 0, 0], [0, 1], [0, 0, np.nan]):
            with self.assertRaises(ValueError): estimate_slope(plane(), normal, {'radius': .45})
        for config in ({}, {'radius': 0}, {'radius': True}, {'radius': np.inf}, {'radius': .45, 'min_samples': 5}, {'radius': .45, 'neighbors': 5}, {'radius': .45, 'neighbors': 6.5}, {'radius': .45, 'max_residual_ratio': .1}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                estimate_slope(plane(), [0, 0, 1], config)


if __name__ == '__main__': unittest.main()
