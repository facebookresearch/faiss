# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Panorama paper: https://www.arxiv.org/pdf/2510.00566"""

import unittest

import faiss
import numpy as np
from common_faiss_tests import for_all_simd_levels
from faiss.contrib.datasets import SyntheticDataset
from faiss.contrib.evaluation import check_ref_knn_with_draws


@for_all_simd_levels
class TestIndexFlatPanorama(unittest.TestCase):
    METRICS = [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]

    def generate_data(self, d, nt, nb, nq, seed=42):
        ds = SyntheticDataset(d, nt, nb, nq, seed=seed)
        return ds.get_train(), ds.get_database(), ds.get_queries()

    def create_flat(self, d, xb=None, metric=faiss.METRIC_L2):
        index = faiss.IndexFlat(d, metric)
        if xb is not None:
            index.add(xb)
        return index

    def create_panorama(
        self, d, nlevels, xb=None, batch_size=128, metric=faiss.METRIC_L2
    ):
        index = faiss.IndexFlatPanorama(d, metric, nlevels, batch_size)
        if xb is not None:
            index.add(xb)
        return index

    def assert_search_results_equal(
        self,
        D_regular,
        I_regular,
        D_panorama,
        I_panorama,
    ):
        # The reference IndexFlat kernel is SIMD-dispatched while Panorama
        # computes a * a + b * b - 2 * a * b at baseline, so the two round
        # differently and near-tied neighbors can swap: compare tie-aware.
        check_ref_knn_with_draws(
            D_regular, I_regular, D_panorama, I_panorama, rtol=1e-5
        )

    def assert_range_results_equal(
        self,
        lims_regular,
        D_regular,
        I_regular,
        lims_panorama,
        D_panorama,
        I_panorama,
        nq,
    ):
        np.testing.assert_array_equal(
            lims_regular, lims_panorama, err_msg="Different result counts"
        )

        for i in range(nq):
            n_results = lims_regular[i + 1] - lims_regular[i]
            if n_results > 0:
                ids_reg = I_regular[lims_regular[i] : lims_regular[i + 1]]
                dist_reg = D_regular[lims_regular[i] : lims_regular[i + 1]]
                ids_pan = I_panorama[lims_panorama[i] : lims_panorama[i + 1]]
                dist_pan = D_panorama[lims_panorama[i] : lims_panorama[i + 1]]

                sort_reg, sort_pan = np.argsort(ids_reg), np.argsort(ids_pan)
                np.testing.assert_array_equal(
                    ids_reg[sort_reg], ids_pan[sort_pan]
                )
                np.testing.assert_allclose(
                    dist_reg[sort_reg], dist_pan[sort_pan], rtol=1e-5
                )

    def test_range_search(self):
        d, nb, nt, nq, nlevels = 128, 50000, 75000, 10, 8
        _, xb, xq = self.generate_data(d, nt, nb, nq, seed=123)

        for metric in self.METRICS:
            index_regular = self.create_flat(d, xb, metric=metric)
            index_panorama = self.create_panorama(d, nlevels, xb, metric=metric)

            for radius in [0.5, 1.0, 2.0, 5.0]:
                with self.subTest(metric=metric, radius=radius):
                    lims_reg, D_reg, I_reg = index_regular.range_search(
                        xq, radius
                    )
                    lims_pan, D_pan, I_pan = index_panorama.range_search(
                        xq, radius
                    )

                    self.assert_range_results_equal(
                        lims_reg, D_reg, I_reg, lims_pan, D_pan, I_pan, nq
                    )

    def test_different_n_levels(self):
        d, nb, nt, nq, k = 128, 25000, 40000, 10, 15
        _, xb, xq = self.generate_data(d, nt, nb, nq, seed=456)

        for metric in self.METRICS:
            index_base = self.create_flat(d, xb, metric=metric)
            D_base, I_base = index_base.search(xq, k)

            nt = faiss.omp_get_max_threads()
            faiss.omp_set_num_threads(1)

            prev_ratio_dims_scanned = float("inf")
            for nlevels in [1, 2, 4, 8, 16, 32]:
                with self.subTest(metric=metric, nlevels=nlevels):
                    faiss.cvar.indexPanorama_stats.reset()
                    index = self.create_panorama(d, nlevels, xb, metric=metric)
                    D, I = index.search(xq, k)
                    self.assert_search_results_equal(D_base, I_base, D, I)

                    stats = faiss.cvar.indexPanorama_stats
                    ratio_dims_scanned = stats.ratio_dims_scanned
                    self.assertLess(ratio_dims_scanned, prev_ratio_dims_scanned)
                    prev_ratio_dims_scanned = ratio_dims_scanned

            faiss.omp_set_num_threads(nt)

    def test_uneven_dimension_division(self):
        test_cases = [(65, 4), (63, 8), (100, 7)]

        for metric in self.METRICS:
            for d, nlevels in test_cases:
                with self.subTest(metric=metric, d=d, nlevels=nlevels):
                    nb, nt, nq, k = 5000, 7000, 10, 5
                    _, xb, xq = self.generate_data(d, nt, nb, nq, seed=789)

                    index_regular = self.create_flat(d, xb, metric=metric)
                    index_panorama = self.create_panorama(
                        d, nlevels, xb, metric=metric
                    )

                    D_regular, I_regular = index_regular.search(xq, k)
                    D_panorama, I_panorama = index_panorama.search(xq, k)

                    self.assert_search_results_equal(
                        D_regular, I_regular, D_panorama, I_panorama
                    )

    def test_id_selector_range(self):
        d, nb, nt, nq, nlevels, k = 128, 80000, 120000, 500, 8, 20
        _, xb, xq = self.generate_data(d, nt, nb, nq, seed=321)

        for metric in self.METRICS:
            with self.subTest(metric=metric):
                index_regular = self.create_flat(d, xb, metric=metric)
                index_panorama = self.create_panorama(
                    d, nlevels, xb, metric=metric
                )

                params = faiss.SearchParameters()
                params.sel = faiss.IDSelectorRange(10000, 50000)

                D_regular, I_regular = index_regular.search(
                    xq, k, params=params
                )
                D_panorama, I_panorama = index_panorama.search(
                    xq, k, params=params
                )

                np.testing.assert_array_equal(I_regular, I_panorama)
                np.testing.assert_allclose(D_regular, D_panorama, rtol=1e-5)

    def test_selector_excludes_all(self):
        d, nb, nt, nq, nlevels, k = 32, 300, 400, 5, 4, 10
        _, xb, xq = self.generate_data(d, nt, nb, nq, seed=999)

        for metric in self.METRICS:
            with self.subTest(metric=metric):
                index_panorama = self.create_panorama(
                    d, nlevels, xb, metric=metric
                )

                params = faiss.SearchParameters()
                params.sel = faiss.IDSelectorRange(nb + 100, nb + 200)

                D, I = index_panorama.search(xq, k, params=params)
                self.assertTrue(np.all(I == -1))

    def test_batch_boundaries(self):
        d, nq, k = 128, 10, 15
        np.random.seed(1234)
        xq = np.random.rand(nq, d).astype("float32")

        for metric in self.METRICS:
            probe_index = self.create_panorama(d, nlevels=8, metric=metric)
            batch_size = probe_index.batch_size

            test_sizes = [
                batch_size - 1,
                batch_size,
                batch_size * 2,
                batch_size * 5 - 1,
            ]
            for nb in test_sizes:
                with self.subTest(nb=nb, metric=metric):
                    np.random.seed(987)
                    xb = np.random.rand(nb, d).astype("float32")

                    index_regular = self.create_flat(d, xb, metric=metric)
                    index_panorama = self.create_panorama(
                        d, nlevels=8, xb=xb, metric=metric
                    )

                    D_regular, I_regular = index_regular.search(xq, k)
                    D_panorama, I_panorama = index_panorama.search(xq, k)

                    self.assert_search_results_equal(
                        D_regular, I_regular, D_panorama, I_panorama
                    )

    def test_batch_size(self):
        d, nb, nt, nq, nlevels = 128, 10000, 15000, 10, 8
        _, xb, xq = self.generate_data(d, nt, nb, nq, seed=4242)

        for metric in self.METRICS:
            index_regular = self.create_flat(d, xb, metric=metric)

            for k, bs in [(1, 1), (100, 4096)]:
                with self.subTest(metric=metric, k=k, batch_size=bs):
                    index_panorama = self.create_panorama(
                        d, nlevels, xb=xb, batch_size=bs, metric=metric
                    )
                    D_regular, I_regular = index_regular.search(xq, k)
                    D_panorama, I_panorama = index_panorama.search(xq, k)
                    self.assert_search_results_equal(
                        D_regular, I_regular, D_panorama, I_panorama
                    )

    def test_range_search_edge_cases(self):
        # Both indexes are empty: the only search of a Panorama index with
        # ntotal == 0.
        d, nb, nt, nq, nlevels = 32, 500, 700, 10, 4
        _, _, xq = self.generate_data(d, nt, nb, nq, seed=777)

        for metric in self.METRICS:
            with self.subTest(metric=metric):
                index_regular = self.create_flat(d, metric=metric)
                index_panorama = self.create_panorama(d, nlevels, metric=metric)

                lims_reg, _, _ = index_regular.range_search(xq, 1000.0)
                lims_pan, _, _ = index_panorama.range_search(xq, 1000.0)

                np.testing.assert_array_equal(lims_reg, lims_pan)

    def test_add_search_add_search(self):
        d, nlevels, k = 32, 4, 5
        np.random.seed(555)

        for metric in self.METRICS:
            with self.subTest(metric=metric):
                index_regular = self.create_flat(d, metric=metric)
                index_panorama = self.create_panorama(d, nlevels, metric=metric)

                xb1 = np.random.rand(200, d).astype("float32")
                index_regular.add(xb1)
                index_panorama.add(xb1)

                xq1 = np.random.rand(10, d).astype("float32")

                D_reg_1, I_reg_1 = index_regular.search(xq1, k)
                D_pan_1, I_pan_1 = index_panorama.search(xq1, k)
                self.assert_search_results_equal(
                    D_reg_1, I_reg_1, D_pan_1, I_pan_1
                )

                xb2 = np.random.rand(300, d).astype("float32")
                index_regular.add(xb2)
                index_panorama.add(xb2)

                xq2 = np.random.rand(10, d).astype("float32")
                D_reg_2, I_reg_2 = index_regular.search(xq2, k)
                D_pan_2, I_pan_2 = index_panorama.search(xq2, k)
                self.assert_search_results_equal(
                    D_reg_2, I_reg_2, D_pan_2, I_pan_2
                )

    def test_reconstruct(self):
        d, nb, nt, nq, nlevels = 964, 1000, 15000, 10, 128
        _, xb, _ = self.generate_data(d, nt, nb, nq, seed=2025)

        index_panorama = self.create_panorama(d, nlevels, xb)

        idx = 123
        v_panorama = index_panorama.reconstruct(idx)
        np.testing.assert_array_equal(xb[idx], v_panorama)

        start_idx, n_vectors = 120, 10
        vn_panorama = index_panorama.reconstruct_n(start_idx, n_vectors)
        np.testing.assert_array_equal(
            xb[start_idx : start_idx + n_vectors], vn_panorama
        )

    def test_remove_ids_then_add(self):
        d, nb, nt, nq, nlevels, k = 964, 50000, 0, 10, 128, 15
        _, xb, xq = self.generate_data(d, nt, nb, nq, seed=2026)

        xb1 = xb[: nb // 2]
        xb2 = xb[nb // 2 :]

        index_regular = self.create_flat(d, xb1)
        index_panorama = self.create_panorama(d, nlevels, xb1)

        ids_to_remove = np.arange(0, nb, 2, dtype=np.int64)

        nremove_regular = index_regular.remove_ids(ids_to_remove)
        nremove_panorama = index_panorama.remove_ids(ids_to_remove)

        self.assertEqual(nremove_regular, nremove_panorama)

        D_reg_1, I_reg_1 = index_regular.search(xq, k)
        D_pan_1, I_pan_1 = index_panorama.search(xq, k)
        self.assert_search_results_equal(D_reg_1, I_reg_1, D_pan_1, I_pan_1)

        index_regular.add(xb2)
        index_panorama.add(xb2)

        D_reg_2, I_reg_2 = index_regular.search(xq, k)
        D_pan_2, I_pan_2 = index_panorama.search(xq, k)
        self.assert_search_results_equal(D_reg_2, I_reg_2, D_pan_2, I_pan_2)

    def test_merge_from(self):
        d, nb, nt, nq, nlevels, k, batch_size = 964, 50000, 0, 10, 128, 15, 16
        _, xb, xq = self.generate_data(d, nt, nb, nq, seed=2027)

        split = nb // 2
        xb1 = xb[:split]
        xb2 = xb[split:]

        index1_regular = self.create_flat(d, xb1)
        index2_regular = self.create_flat(d, xb2)

        index1_panorama = self.create_panorama(d, nlevels, xb1, batch_size)
        index2_panorama = self.create_panorama(
            d, nlevels * 2, xb2, batch_size // 2
        )

        index1_regular.merge_from(index2_regular, 0)
        index1_panorama.merge_from(index2_panorama, 0)

        self.assertEqual(index2_regular.ntotal, index2_panorama.ntotal)

        D_regular, I_regular = index1_regular.search(xq, k)
        D_panorama, I_panorama = index1_panorama.search(xq, k)
        self.assert_search_results_equal(
            D_regular, I_regular, D_panorama, I_panorama
        )

    def test_permute_entries(self):
        d, nb, nt, nq, nlevels, k = 964, 50000, 0, 20, 128, 10
        _, xb, xq = self.generate_data(d, nt, nb, nq, seed=2028)

        index_regular = self.create_flat(d, xb)
        index_panorama = self.create_panorama(d, nlevels, xb)

        np.random.seed(1234)
        perm = np.random.permutation(nb).astype(np.int64)

        index_regular.permute_entries(perm)
        index_panorama.permute_entries(perm)

        D_regular, I_regular = index_regular.search(xq, k)
        D_panorama, I_panorama = index_panorama.search(xq, k)

        self.assert_search_results_equal(
            D_regular, I_regular, D_panorama, I_panorama
        )

    def test_serialization(self):
        d, nb, nt, nq, nlevels, k = 964, 10000, 15000, 100, 128, 20
        _, xb, xq = self.generate_data(d, nt, nb, nq, seed=2024)

        for metric in self.METRICS:
            with self.subTest(metric=metric):
                index = self.create_panorama(d, nlevels, xb, metric=metric)

                D_before, I_before = index.search(xq, k)
                index_after = faiss.deserialize_index(
                    faiss.serialize_index(index)
                )
                D_after, I_after = index_after.search(xq, k)

                np.testing.assert_array_equal(I_before, I_after)
                np.testing.assert_array_equal(D_before, D_after)

    def test_ratio_dims_scanned(self):
        d, nb, nq, k = 128, 500000, 10, 1

        # Setup: All vectors in the dataset are [1, 1, ..., 1], except for
        # one, which is [0, 0, ..., 0]. The queries are [0, 0, ..., 0]. This
        # ensures that pruning happens as early as it can. As a result, the
        # ratio of dimensions scanned should be approximately (1 / nlevels).
        xb = np.ones((nb, d)).astype("float32")
        xb[-1] = 0
        xq = np.zeros((nq, d)).astype("float32")

        for metric in self.METRICS:
            with self.subTest(metric=metric):
                index_base = self.create_flat(d, xb, metric=metric)
                D_base, I_base = index_base.search(xq, k)

                nt = faiss.omp_get_max_threads()
                faiss.omp_set_num_threads(1)

                ratios = []
                nlevels_list = [1, 2, 4, 8, 16, 32]
                for nlevels in nlevels_list:
                    with self.subTest(nlevels=nlevels):
                        faiss.cvar.indexPanorama_stats.reset()
                        index = self.create_panorama(
                            d, nlevels, xb, metric=metric
                        )
                        D, I = index.search(xq, k)
                        self.assert_search_results_equal(D_base, I_base, D, I)

                        ratios.append(
                            faiss.cvar.indexPanorama_stats.ratio_dims_scanned
                        )

                expected_ratios = [1 / nlevels for nlevels in nlevels_list]
                np.testing.assert_allclose(ratios, expected_ratios, atol=1e-3)

                faiss.omp_set_num_threads(nt)
