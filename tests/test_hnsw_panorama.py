# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Panorama paper: https://www.arxiv.org/pdf/2510.00566"""

import unittest
import tempfile
import os
import numpy as np

import faiss
from common_faiss_tests import for_all_simd_levels
from faiss.contrib.datasets import SyntheticDataset


@for_all_simd_levels
class TestIndexHNSWFlatPanorama(unittest.TestCase):
    def generate_data(self, d, nt, nb, nq, seed=1234):
        ds = SyntheticDataset(d, nt, nb, nq, seed=seed)
        return ds.get_train(), ds.get_database(), ds.get_queries()

    def compute_ground_truth(self, xb, xq, k):
        d = xb.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(xb)
        D, I = index.search(xq, k)
        return D, I

    def compute_recall(self, gt_I, test_I):
        nq, k = gt_I.shape
        recalls = [np.isin(gt_I[i], test_I[i]).sum() for i in range(nq)]
        return sum(recalls) / (nq * k)

    def test_different_panorama_levels(self):
        d = 256
        nb = 500
        nt = 700
        nq = 50
        k = 5

        _, xb, xq = self.generate_data(d, nt, nb, nq, seed=456)

        _, gt_I = self.compute_ground_truth(xb, xq, k)

        for nlevels in [4, 8, 16]:
            with self.subTest(nlevels=nlevels):
                index = faiss.IndexHNSWFlatPanorama(d, 32, nlevels)
                index.hnsw.efSearch = 64

                index.add(xb)

                _, I = index.search(xq, k)

                recall = self.compute_recall(gt_I, I)
                self.assertGreaterEqual(recall, 0.80)

    def test_uneven_dimension_division(self):
        test_cases = [(65, 4), (63, 8), (100, 7), (960, 128), (964, 128)]

        nb, nt, nq, k = 1000, 700, 20, 5

        for d, nlevels in test_cases:
            with self.subTest(d=d, nlevels=nlevels):
                _, xb, xq = self.generate_data(d, nt, nb, nq, seed=789)

                index = faiss.IndexHNSWFlatPanorama(d, 32, nlevels)
                index.hnsw.efSearch = 64
                index.add(xb)

                _, I = index.search(xq, k)

                _, gt_I = self.compute_ground_truth(xb, xq, k)

                recall = self.compute_recall(gt_I, I)
                self.assertGreaterEqual(
                    recall,
                    0.80,
                    f"Recall too low for d={d}, nlevels={nlevels}: "
                    f"{recall:.4f}",
                )

    def test_io(self):
        d = 964
        nb = 500
        nt = 700
        nq = 10
        k = 5
        nlevels = 128

        _, xb, xq = self.generate_data(d, nt, nb, nq, seed=2024)

        index = faiss.IndexHNSWFlatPanorama(d, 16, nlevels)
        index.add(xb)

        D_before, I_before = index.search(xq, k)

        fname = tempfile.mktemp(suffix=".index")
        try:
            faiss.write_index(index, fname)
            loaded_index = faiss.read_index(fname)

            self.assertIsInstance(loaded_index, faiss.IndexHNSWFlatPanorama)
            self.assertEqual(loaded_index.num_panorama_levels, nlevels)

            D_after, I_after = loaded_index.search(xq, k)

            np.testing.assert_array_equal(I_before, I_after)
            np.testing.assert_array_equal(D_before, D_after)
        finally:
            if os.path.exists(fname):
                os.unlink(fname)

    def test_clone(self):
        d = 64
        nb = 500
        nt = 700
        nq = 10
        k = 5

        _, xb, xq = self.generate_data(d, nt, nb, nq, seed=2025)

        index = faiss.IndexHNSWFlatPanorama(d, 16, 8)
        index.add(xb)

        D_before, I_before = index.search(xq, k)

        cloned_index = faiss.clone_index(index)

        self.assertIsInstance(cloned_index, faiss.IndexHNSWFlatPanorama)
        self.assertEqual(cloned_index.num_panorama_levels, 8)

        D_after, I_after = cloned_index.search(xq, k)

        np.testing.assert_array_equal(I_before, I_after)
        np.testing.assert_array_equal(D_before, D_after)

    def test_factory(self):
        d = 64

        index1 = faiss.index_factory(d, "HNSW32_FlatPanorama")
        self.assertIsInstance(index1, faiss.IndexHNSWFlatPanorama)
        self.assertEqual(index1.hnsw.nb_neighbors(0), 2 * 32)
        self.assertEqual(index1.num_panorama_levels, 8)  # default

        index2 = faiss.index_factory(d, "HNSW16_FlatPanorama12")
        self.assertIsInstance(index2, faiss.IndexHNSWFlatPanorama)
        self.assertEqual(index2.hnsw.nb_neighbors(0), 2 * 16)
        self.assertEqual(index2.num_panorama_levels, 12)

    def test_reset(self):
        d = 64
        nb = 500
        nt = 700

        _, xb, _ = self.generate_data(d, nt, nb, 10, seed=999)

        index = faiss.IndexHNSWFlatPanorama(d, 16, 8)
        index.add(xb)

        self.assertGreater(index.cum_sums.size(), 0)

        index.reset()

        self.assertEqual(index.ntotal, 0)
        self.assertEqual(index.cum_sums.size(), 0)

    def test_add_after_search(self):
        d = 64
        nb_initial = 300
        nt_initial = 500
        nb_add = 200
        nt_add = 300
        nq = 10
        k = 5

        _, xb_initial, xq = self.generate_data(
            d, nt_initial, nb_initial, nq, seed=1111
        )
        _, xb_add, _ = self.generate_data(d, nt_add, nb_add, 1, seed=2222)

        index = faiss.IndexHNSWFlatPanorama(d, 16, 8)

        index.add(xb_initial)

        index.search(xq, k)

        index.add(xb_add)

        self.assertEqual(
            index.cum_sums.size(),
            (nb_initial + nb_add) * (index.num_panorama_levels + 1),
        )

        _, I_after = index.search(xq, k)

        xb_all = np.vstack([xb_initial, xb_add])

        _, gt_I = self.compute_ground_truth(xb_all, xq, k)

        recall = self.compute_recall(gt_I, I_after)
        self.assertGreaterEqual(recall, 0.80)

    def test_permute_entries(self):
        d = 64
        nb = 500
        nt = 700
        nq = 10
        k = 5

        _, xb, xq = self.generate_data(d, nt, nb, nq, seed=3333)

        index = faiss.IndexHNSWFlatPanorama(d, 16, 8)
        index.add(xb)

        D_before, I_before = index.search(xq, k)

        perm = np.arange(nb - 1, -1, -1, dtype=np.int64)

        cum_sums_before = faiss.vector_to_array(index.cum_sums).copy()

        index.permute_entries(perm)

        cum_sums_after = faiss.vector_to_array(index.cum_sums)
        for i in range(nb):
            src = perm[i]
            for j in range(index.num_panorama_levels + 1):
                expected = cum_sums_before[
                    src * (index.num_panorama_levels + 1) + j
                ]
                actual = cum_sums_after[i * (index.num_panorama_levels + 1) + j]
                self.assertEqual(
                    actual,
                    expected,
                    f"cum_sums not permuted correctly at i={i}, j={j}",
                )

        D_after, I_after = index.search(xq, k)

        np.testing.assert_allclose(
            D_before, D_after, err_msg="Distance changed after permutation"
        )

        for i in range(nq):
            for j in range(k):
                if I_before[i, j] >= 0:
                    old_id = I_before[i, j]
                    new_id = np.where(perm == old_id)[0][0]
                    self.assertEqual(
                        I_after[i, j],
                        new_id,
                        f"Permuted ID mismatch at position ({i}, {j})",
                    )

    def test_id_selector_range(self):
        d = 128
        nb = 1000
        nt = 1500
        nq = 10
        k = 10

        _, xb, xq = self.generate_data(d, nt, nb, nq, seed=321)

        index = faiss.IndexHNSWFlatPanorama(d, 32, 8)
        index.hnsw.efSearch = 64
        index.add(xb)

        selector = faiss.IDSelectorRange(200, 600)
        params = faiss.SearchParametersHNSW()
        params.sel = selector

        _, I = index.search(xq, k, params=params)

        # HNSW may return < k results, padded with -1
        for i in range(nq):
            for j in range(k):
                if I[i, j] != -1:
                    self.assertGreaterEqual(I[i, j], 200)
                    self.assertLess(I[i, j], 600)

    def test_panorama_stats(self):
        d, nb, nq, k = 128, 10000, 10, 10

        rs = np.random.RandomState(42)
        xb = rs.rand(nb, d).astype("float32")
        xq = rs.rand(nq, d).astype("float32")

        nt = faiss.omp_get_max_threads()
        faiss.omp_set_num_threads(1)

        try:
            ratios = []
            nlevels_list = [1, 4, 8, 16]
            for nlevels in nlevels_list:
                faiss.cvar.indexPanorama_stats.reset()

                index = faiss.IndexHNSWFlatPanorama(d, 32, nlevels)
                index.hnsw.efSearch = 64
                index.add(xb)
                index.search(xq, k)

                ratio = faiss.cvar.indexPanorama_stats.ratio_dims_scanned
                ratios.append(ratio)

            for i in range(1, len(ratios)):
                self.assertLessEqual(
                    ratios[i],
                    ratios[i - 1],
                    f"ratio should decrease: nlevels={nlevels_list[i]} "
                    f"ratio={ratios[i]:.4f} > prev={ratios[i-1]:.4f}",
                )
            self.assertLess(ratios[-1], 1.0, "Some pruning should occur")
        finally:
            faiss.omp_set_num_threads(nt)
