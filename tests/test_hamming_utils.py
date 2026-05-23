# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for the standalone Hamming distance utility functions in
faiss/utils/hamming.h. These are dynamic-dispatch entry points that aren't
exercised by IndexBinary tests, so they otherwise have no per-SIMD-level
coverage.

Each test computes a result at the current SIMD level (the level set by the
@for_all_simd_levels decorator) and compares against a NumPy popcount-xor
reference. The reference is level-independent, so any per-level dispatch
drift surfaces as an assertion failure.

Threshold-predicate convention: hamming_distance/common.h uses `h <= ht`
for hamming_count_thres, match_hamming_thres, crosshamming_count_thres --
the tests match that.
"""

import unittest

import faiss
import numpy as np

from common_faiss_tests import for_all_simd_levels


def _popcount_xor(a, b):
    """Reference Hamming distance via NumPy; bit-exact across platforms."""
    return np.unpackbits(np.bitwise_xor(a, b), axis=-1).sum(axis=-1).astype(
        np.int32)


def _make_codes(n, ncodes, seed):
    rs = np.random.RandomState(seed)
    return rs.randint(256, size=(n, ncodes), dtype=np.uint8)


# Threshold values chosen so each (ncodes, ht) pair produces a non-trivial
# number of matches over random uint8 codes (mean Hamming distance is
# 4 * ncodes; std grows like sqrt(2 * ncodes)). Picking ht slightly below the
# mean produces tens-to-hundreds of matches at the test sizes used here.
# A vacuous (zero-match) test would pass against a no-op implementation.
_THRESHOLD_CASES = [(8, 28), (16, 60), (32, 120)]


@for_all_simd_levels
class TestHammingUtils(unittest.TestCase):
    """Direct tests for hamming.h standalone entry points."""

    def test_hammings_all_pairs(self):
        # faiss.hammings: all-pairs Hamming distances.
        # ncodes covers SIMD-aligned (8, 16, 32, 64) and tail (24, 40) paths.
        for ncodes in (8, 16, 32, 24, 40, 64):
            with self.subTest(ncodes=ncodes):
                na, nb = 23, 47
                a = _make_codes(na, ncodes, seed=1 + ncodes)
                b = _make_codes(nb, ncodes, seed=2 + ncodes)
                dis = np.empty((na, nb), dtype=np.int32)
                faiss.hammings(
                    faiss.swig_ptr(a), faiss.swig_ptr(b),
                    na, nb, ncodes, faiss.swig_ptr(dis))
                expected = _popcount_xor(a[:, None, :], b[None, :, :])
                np.testing.assert_array_equal(dis, expected)

    def test_hammings_knn_mc(self):
        for ncodes in (8, 16, 32):
            with self.subTest(ncodes=ncodes):
                na, nb, k = 17, 200, 8
                a = _make_codes(na, ncodes, seed=10 + ncodes)
                b = _make_codes(nb, ncodes, seed=11 + ncodes)
                D = np.empty((na, k), dtype=np.int32)
                I = np.empty((na, k), dtype=np.int64)
                faiss.hammings_knn_mc(
                    faiss.swig_ptr(a), faiss.swig_ptr(b),
                    na, nb, k, ncodes,
                    faiss.swig_ptr(D), faiss.swig_ptr(I))

                full = _popcount_xor(a[:, None, :], b[None, :, :])
                # Distance-at-returned-index agrees with the reference.
                returned = np.take_along_axis(full, I, axis=1)
                np.testing.assert_array_equal(D, returned)
                # Each returned distance is no greater than the true k-th
                # smallest, so the result really is a top-k (with ties
                # broken in some impl-defined way).
                kth = np.partition(full, k - 1, axis=1)[:, k - 1:k]
                self.assertTrue(np.all(D <= kth))
                # IDs within a row are unique.
                for i in range(na):
                    self.assertEqual(len(np.unique(I[i])), k)

    def test_hamming_count_thres(self):
        for ncodes, ht in _THRESHOLD_CASES:
            with self.subTest(ncodes=ncodes, ht=ht):
                n1, n2 = 33, 71
                a = _make_codes(n1, ncodes, seed=20 + ncodes)
                b = _make_codes(n2, ncodes, seed=21 + ncodes)
                count = np.zeros(1, dtype=np.uint64)
                faiss.hamming_count_thres(
                    faiss.swig_ptr(a), faiss.swig_ptr(b),
                    n1, n2, ht, ncodes, faiss.swig_ptr(count))
                expected = int((_popcount_xor(
                    a[:, None, :], b[None, :, :]) <= ht).sum())
                # Sanity: thresholds chosen so the count is non-trivial
                # (otherwise the test passes vacuously).
                self.assertGreater(expected, 0)
                self.assertEqual(int(count[0]), expected)

    def test_match_hamming_thres(self):
        for ncodes, ht in _THRESHOLD_CASES:
            with self.subTest(ncodes=ncodes, ht=ht):
                n1, n2 = 25, 60
                a = _make_codes(n1, ncodes, seed=30 + ncodes)
                b = _make_codes(n2, ncodes, seed=31 + ncodes)
                full = _popcount_xor(a[:, None, :], b[None, :, :])
                expected_count = int((full <= ht).sum())
                self.assertGreater(expected_count, 0)
                # match_hamming_thres writes idx as alternating (i, j) int64
                # pairs (see hamming_distance/common.h match_hamming_thres_impl)
                # and dis as one int32 per match. Allocate generously.
                idx = np.empty(2 * expected_count + 4, dtype=np.int64)
                dis = np.empty(expected_count + 4, dtype=np.int32)
                got = faiss.match_hamming_thres(
                    faiss.swig_ptr(a), faiss.swig_ptr(b),
                    n1, n2, ht, ncodes,
                    faiss.swig_ptr(idx), faiss.swig_ptr(dis))
                self.assertEqual(got, expected_count)
                # Decode the (i, j) pairs.
                i_idx = idx[0:2 * got:2]
                j_idx = idx[1:2 * got:2]
                dis = dis[:got]
                # Every reported pair is within threshold and matches the
                # reference distance at (i, j).
                self.assertTrue(np.all(dis <= ht))
                np.testing.assert_array_equal(dis, full[i_idx, j_idx])
                # Every below-threshold pair in the reference appears in the
                # output exactly once.
                ref_pairs = set(zip(*np.where(full <= ht)))
                got_pairs = set(zip(i_idx.tolist(), j_idx.tolist()))
                self.assertEqual(ref_pairs, got_pairs)

    def test_crosshamming_count_thres(self):
        for ncodes, ht in _THRESHOLD_CASES:
            with self.subTest(ncodes=ncodes, ht=ht):
                n = 80
                dbs = _make_codes(n, ncodes, seed=40 + ncodes)
                count = np.zeros(1, dtype=np.uint64)
                faiss.crosshamming_count_thres(
                    faiss.swig_ptr(dbs), n, ht, ncodes,
                    faiss.swig_ptr(count))
                full = _popcount_xor(dbs[:, None, :], dbs[None, :, :])
                # crosshamming counts unordered pairs i < j.
                triu = np.triu(full <= ht, k=1)
                expected = int(triu.sum())
                self.assertGreater(expected, 0)
                self.assertEqual(int(count[0]), expected)

    def test_generalized_hammings_knn_hc(self):
        # Generalized Hamming distance: number of mismatched bytes.
        for code_size in (8, 16, 32):
            with self.subTest(code_size=code_size):
                na, nb, k = 13, 80, 5
                a = _make_codes(na, code_size, seed=50 + code_size)
                b = _make_codes(nb, code_size, seed=51 + code_size)
                D = np.empty((na, k), dtype=np.int32)
                I = np.empty((na, k), dtype=np.int64)
                heap = faiss.int_maxheap_array_t()
                heap.k = k
                heap.nh = na
                heap.ids = faiss.swig_ptr(I)
                heap.val = faiss.swig_ptr(D)
                faiss.generalized_hammings_knn_hc(
                    heap, faiss.swig_ptr(a), faiss.swig_ptr(b),
                    nb, code_size, True)

                # Reference distance: number of bytes that differ.
                ref_dist = code_size - (
                    a[:, None, :] == b[None, :, :]).sum(axis=-1).astype(
                        np.int32)
                # Distance at returned index matches the reference.
                returned = np.take_along_axis(ref_dist, I, axis=1)
                np.testing.assert_array_equal(D, returned)
                # Each returned distance is no greater than the true k-th
                # smallest -- catches an impl that returns I=[0..k-1].
                kth = np.partition(ref_dist, k - 1, axis=1)[:, k - 1:k]
                self.assertTrue(np.all(D <= kth))
                # ordered=True was passed: distances within each row are
                # non-decreasing.
                self.assertTrue(np.all(np.diff(D, axis=1) >= 0))
                # IDs are unique within a row.
                for i in range(na):
                    self.assertEqual(len(np.unique(I[i])), k)


if __name__ == "__main__":
    unittest.main()
