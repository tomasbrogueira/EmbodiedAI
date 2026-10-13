"""Each original observation retains its own frame's local inclination."""
import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from slope import estimate_slope
from test_slope import plane


class FrameSlopeTests(unittest.TestCase):
    def test_parallel_preserves_shuffled_rows_duplicates_nans_and_progress(self):
        points = np.vstack((plane(20, noise=.003), plane(30), plane()[:5],
                            np.repeat(plane()[[0]], 10, axis=0), [[np.nan, 0, 0]]))
        ids = np.r_[np.full(961, 10), np.full(961, 20), np.full(15, 30), 40]
        order = np.random.default_rng(9).permutation(len(points))
        points, ids = points[order], ids[order]
        expected = estimate_slope(points, [0,0,1], {'radius':.45}, frame_ids=ids, return_quality=True)
        updates = []
        actual = estimate_slope(points, [0,0,1], {'radius':.45}, frame_ids=ids,
            workers=4, return_quality=True, progress=lambda done,total: updates.append((done,total)))
        for a, b in zip(expected, actual):
            assert_array_equal(a, b)
        self.assertEqual(updates[-1], (len(points),len(points)))
        self.assertEqual(updates, sorted(updates))

    def test_offsets_and_different_angles_are_independent(self):
        for angle in (0, 20, 30):
            a, b = plane(angle), plane(30)+[0, 0, .02]
            p = np.vstack((a, b)); ids = np.repeat([7, 9001], len(a))
            s = estimate_slope(p, [0, 0, 1], {'radius': .45}, frame_ids=ids)
            assert_allclose(s[:len(a)], np.deg2rad(angle), atol=1e-7)
            assert_allclose(s[len(a):], np.deg2rad(30), atol=1e-7)

    def test_unsupported_frame_does_not_invalidate_supported_frame(self):
        a = plane()
        for b in (np.array([[0, 0, .02]]), np.tile(a[:5]+[0, 0, .02], (40, 1)), np.c_[np.linspace(-.1, .1, 11), np.zeros(11), np.full(11, .02)]):
            p = np.vstack((a, b)); ids = np.r_[np.full(len(a), 7), np.full(len(b), 9001)]
            s, q = estimate_slope(p, [0, 0, 1], {'radius': .45}, frame_ids=ids, return_quality=True)
            assert_allclose(s[:len(a)], 0, atol=1e-7)
            self.assertTrue(np.isnan(s[len(a):]).all()); self.assertFalse(q[len(a):].any())

    def test_single_frame_and_permuted_rows(self):
        p = plane(20, noise=.003); ids = np.full(len(p), 42)
        base, bq = estimate_slope(p, [0, 0, 1], {'radius': .45}, return_quality=True)
        order = np.random.default_rng(9).permutation(len(p)); copy = p.copy()
        s, q = estimate_slope(p[order], [0, 0, 1], {'radius': .45}, frame_ids=ids[order], return_quality=True)
        assert_allclose(s, base[order]); assert_allclose(q, bq[order]); assert_array_equal(p, copy)

    def test_invalid_frame_ids(self):
        p = plane(); ids = np.zeros(len(p), dtype=int)
        for invalid in (ids[:-1], ids.astype(float), ids[:, None]):
            with self.assertRaises(ValueError):
                estimate_slope(p, [0, 0, 1], {'radius': .45}, frame_ids=invalid)

    def test_invalid_worker_count(self):
        for workers in (-1, True, 1.5):
            with self.subTest(workers=workers), self.assertRaises(ValueError):
                estimate_slope(plane(), [0,0,1], {'radius':.45}, workers=workers)


if __name__ == '__main__': unittest.main()
