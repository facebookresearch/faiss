# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for BinaryInvertedListScanner::scan_codes, the bulk entry point a
caller uses to fold an inverted list into a heap it owns.

Each test compares against a NumPy popcount-xor reference, which is
level-independent, so per-level dispatch drift surfaces as a failure. The
@for_all_simd_levels decorator runs every case at each available level.
"""

import unittest

import faiss
import numpy as np

from common_faiss_tests import for_all_simd_levels

# with_HammingComputer dispatches on the code size: 4, 8, 16, 20, 32 and 64
# each get their own computer, and anything else lands on the generic one.
# scan_codes runs the same loop for all of them, so the semantic cases below
# cover every computer rather than only the 20-byte one.
WIDTHS = [4, 8, 16, 20, 24, 32, 64]
CODE_SIZE = 20
NEUTRAL = np.iinfo(np.int32).max


def _hamming(query, codes):
    return np.unpackbits(codes ^ query, axis=1).sum(1).astype("int32")


def _expected(query, codes, k, radius=None):
    dis = _hamming(query, codes)
    if radius is not None:
        dis = dis[dis < radius]
    return sorted(dis.tolist())[:k]


@for_all_simd_levels
class TestBinaryIVFScanner(unittest.TestCase):
    def make_scanner(self, rs, query, store_pairs=False, cs=CODE_SIZE):
        dim = cs * 8
        quantizer = faiss.IndexBinaryFlat(dim)
        quantizer.add(self.random_codes(rs, 1, cs))
        ivf = faiss.IndexBinaryIVF(quantizer, dim, 1)
        ivf.is_trained = True
        scanner = ivf.get_InvertedListScanner(store_pairs)
        scanner.set_query(faiss.swig_ptr(query))
        scanner.set_list(0, 0)
        # The index and quantizer must outlive the scanner.
        self.keep_alive = (ivf, quantizer)
        return scanner

    def random_codes(self, rs, n, cs=CODE_SIZE):
        return rs.randint(0, 256, size=(n, cs)).astype("uint8")

    def sequential_ids(self, n):
        return (np.arange(n) * 3 + 100).astype("int64")

    def run_scan(self, scanner, codes, ids, slice_sizes, k, radius=None):
        """Seeds the heap, scans each slice into it, then drops the slots the
        scan never filled. Seeding every slot with a radius makes the heap top
        reject any code at or beyond it."""
        simi = np.full(k, NEUTRAL if radius is None else radius, dtype="int32")
        idxi = np.full(k, -1, dtype="int64")
        offset = 0
        for count in slice_sizes:
            scanner.scan_codes(
                count,
                faiss.swig_ptr(codes[offset:]),
                faiss.swig_ptr(ids[offset:]),
                faiss.swig_ptr(simi),
                faiss.swig_ptr(idxi),
                k,
            )
            offset += count
        keep = idxi >= 0
        order = np.argsort(simi[keep], kind="stable")
        return simi[keep][order].tolist(), idxi[keep][order].tolist()

    def test_unseeded_heap_keeps_the_k_nearest(self):
        """Every code width, so the shared scan loop is covered for each
        HammingComputer specialization and for the generic one."""
        rs = np.random.RandomState(1234)
        n, k = 2000, 10
        for cs in WIDTHS:
            query = self.random_codes(rs, 1, cs)
            codes = self.random_codes(rs, n, cs)
            ids = self.sequential_ids(n)

            got, _ = self.run_scan(
                self.make_scanner(rs, query, cs=cs), codes, ids, [n], k
            )
            self.assertEqual(got, _expected(query, codes, k), f"code_size={cs}")

    def test_seeded_heap_bounds_every_width(self):
        """The radius seed must behave the same for every computer."""
        rs = np.random.RandomState(1235)
        n, k = 1500, 12
        for cs in WIDTHS:
            query = self.random_codes(rs, 1, cs)
            codes = self.random_codes(rs, n, cs)
            ids = self.sequential_ids(n)
            radius = int(cs * 8 * 0.42)  # well below the mean of cs*4

            got, _ = self.run_scan(
                self.make_scanner(rs, query, cs=cs),
                codes,
                ids,
                [n],
                k,
                radius,
            )
            self.assertEqual(
                got, _expected(query, codes, k, radius), f"code_size={cs}"
            )

    def test_seeded_heap_applies_the_radius_and_k_together(self):
        rs = np.random.RandomState(21)
        n = 4000
        query = self.random_codes(rs, 1)
        codes, ids = self.random_codes(rs, n), self.sequential_ids(n)

        for k, radius in [(64, 60), (5, 70)]:
            got, _ = self.run_scan(
                self.make_scanner(rs, query), codes, ids, [n], k, radius
            )
            self.assertEqual(got, _expected(query, codes, k, radius), f"k={k}")

    def test_results_accumulate_across_calls(self):
        rs = np.random.RandomState(7)
        n, k, radius = 3000, 16, 75
        query = self.random_codes(rs, 1)
        codes, ids = self.random_codes(rs, n), self.sequential_ids(n)

        for layout in [[1, n - 1], [0, n], [1000, 0, 1000, 1, 999]]:
            got, _ = self.run_scan(
                self.make_scanner(rs, query), codes, ids, layout, k, radius
            )
            self.assertEqual(
                got, _expected(query, codes, k, radius), str(layout)
            )

    def test_the_seeded_bound_is_exclusive(self):
        rs = np.random.RandomState(31)
        # An all-zero query, so a code with p bits set sits at distance p.
        query = np.zeros((1, CODE_SIZE), dtype="uint8")
        codes = np.zeros((2, CODE_SIZE), dtype="uint8")
        codes[0][0], codes[1][0] = 0x0F, 0x1F  # 4 and 5 bits set
        ids = self.sequential_ids(2)

        got, _ = self.run_scan(
            self.make_scanner(rs, query), codes, ids, [2], 10, 5
        )
        self.assertEqual(got, [4])

        # The same rule at its limit.
        got, _ = self.run_scan(
            self.make_scanner(rs, query), codes, ids, [2], 10, 0
        )
        self.assertEqual(got, [])
