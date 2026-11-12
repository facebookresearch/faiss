# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import faiss
import numpy as np
from common_faiss_tests import for_all_simd_levels
from faiss.contrib.datasets import SyntheticDataset
from faiss.contrib.evaluation import check_ref_knn_with_draws


@for_all_simd_levels
class TestIndexRefinePanorama(unittest.TestCase):
    METRICS = [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]

    def generate_data(self, d, nt, nb, nq, seed=42):
        ds = SyntheticDataset(d, nt, nb, nq, seed=seed)
        return ds.get_train(), ds.get_database(), ds.get_queries()

    def create_flat(
        self, d, base_factory, xt=None, xb=None, metric=faiss.METRIC_L2
    ):
        base_index = faiss.index_factory(d, base_factory, metric)
        index = faiss.IndexRefineFlat(base_index)
        if xt is not None:
            index.train(xt)
        if xb is not None:
            index.add(xb)
        return index

    def create_panorama(
        self,
        d,
        base_factory,
        nlevels,
        xt=None,
        xb=None,
        metric=faiss.METRIC_L2,
    ):
        base_index = faiss.index_factory(d, base_factory, metric)
        refine_index = faiss.index_factory(
            d,
            f"Flat{metric == faiss.METRIC_L2 and 'L2' or 'IP'}"
            f"Panorama{nlevels}_1",
            metric,
        )
        index = faiss.IndexRefinePanorama(base_index, refine_index)
        if xt is not None:
            index.train(xt)
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
        check_ref_knn_with_draws(
            D_regular, I_regular, D_panorama, I_panorama, rtol=1e-5, atol=1e-4
        )

    def test_refine_panorama_correctness(self):
        d = 128
        nt = 1000
        nb = 5000
        nq = 100
        k = 10
        n_levels = 8

        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=2025)

        for metric in self.METRICS:
            with self.subTest(metric=metric):
                base_cfg = "Flat"
                index_flat = self.create_flat(
                    d, base_cfg, xt=xt, xb=xb, metric=metric
                )
                index_pan = self.create_panorama(
                    d, base_cfg, n_levels, xt=xt, xb=xb, metric=metric
                )

                for k_factor in [1, 64, 1024]:
                    params = faiss.IndexRefineSearchParameters(
                        k_factor=float(k_factor)
                    )

                    D_flat, I_flat = index_flat.search(xq, k, params=params)
                    D_pano, I_pano = index_pan.search(xq, k, params=params)

                    np.testing.assert_array_equal(I_flat, I_pano)
                    np.testing.assert_allclose(D_flat, D_pano, rtol=1e-5)

    def test_different_n_levels(self):
        d, nb, nt, nq, k = 128, 25000, 40000, 10, 15
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=456)

        base_cfg = "Flat"

        for metric in self.METRICS:
            index_base = self.create_flat(
                d, base_cfg, xt=xt, xb=xb, metric=metric
            )
            D_base, I_base = index_base.search(xq, k)

            params = faiss.IndexRefineSearchParameters(k_factor=1000)

            nt_threads = faiss.omp_get_max_threads()
            faiss.omp_set_num_threads(1)

            prev_ratio_dims_scanned = float("inf")
            for nlevels in [1, 2, 4, 8, 16, 32]:
                with self.subTest(metric=metric, nlevels=nlevels):
                    faiss.cvar.indexPanorama_stats.reset()
                    index = self.create_panorama(
                        d, base_cfg, nlevels, xt=xt, xb=xb, metric=metric
                    )
                    D, I = index.search(xq, k, params=params)
                    self.assert_search_results_equal(D_base, I_base, D, I)

                    stats = faiss.cvar.indexPanorama_stats
                    ratio_dims_scanned = stats.ratio_dims_scanned
                    self.assertLess(ratio_dims_scanned, prev_ratio_dims_scanned)
                    prev_ratio_dims_scanned = ratio_dims_scanned

            faiss.omp_set_num_threads(nt_threads)

    def test_uneven_dimension_division(self):
        test_cases = [(65, 4), (63, 8), (100, 7), (960, 128), (964, 128)]

        for metric in self.METRICS:
            for d, nlevels in test_cases:
                with self.subTest(metric=metric, d=d, nlevels=nlevels):
                    nb, nt, nq, k = 5000, 7000, 10, 5
                    xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=789)

                    base_cfg = "Flat"
                    index_regular = self.create_flat(
                        d, base_cfg, xt=xt, xb=xb, metric=metric
                    )
                    index_panorama = self.create_panorama(
                        d, base_cfg, nlevels, xt=xt, xb=xb, metric=metric
                    )

                    D_regular, I_regular = index_regular.search(xq, k)
                    D_panorama, I_panorama = index_panorama.search(xq, k)

                    self.assert_search_results_equal(
                        D_regular, I_regular, D_panorama, I_panorama
                    )

    def test_id_selector_range(self):
        d, nb, nt, nq, nlevels, k = 128, 80000, 120000, 500, 8, 20
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=321)

        base_cfg = "Flat"

        for metric in self.METRICS:
            with self.subTest(metric=metric):
                index_regular = self.create_flat(
                    d, base_cfg, xt=xt, xb=xb, metric=metric
                )
                index_panorama = self.create_panorama(
                    d, base_cfg, nlevels, xt=xt, xb=xb, metric=metric
                )

                base_params = faiss.SearchParameters()
                base_params.sel = faiss.IDSelectorRange(10000, 50000)
                params = faiss.IndexRefineSearchParameters(
                    k_factor=1, base_index_params=base_params
                )

                D_regular, I_regular = index_regular.search(
                    xq, k, params=params
                )
                D_panorama, I_panorama = index_panorama.search(
                    xq, k, params=params
                )

                np.testing.assert_array_equal(I_regular, I_panorama)
                np.testing.assert_allclose(D_regular, D_panorama, rtol=1e-5)

    def test_add_search_add_search(self):
        d, nlevels, k = 32, 4, 5
        np.random.seed(555)

        base_cfg = "Flat"

        for metric in self.METRICS:
            with self.subTest(metric=metric):
                index_regular = self.create_flat(d, base_cfg, metric=metric)
                index_panorama = self.create_panorama(
                    d, base_cfg, nlevels, metric=metric
                )

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

    def test_serialization(self):
        d, nb, nt, nq, nlevels, k = 128, 10000, 15000, 100, 8, 20
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=2024)

        base_cfg = "Flat"

        for metric in self.METRICS:
            with self.subTest(metric=metric):
                index = self.create_panorama(
                    d, base_cfg, nlevels, xt=xt, xb=xb, metric=metric
                )

                D_before, I_before = index.search(xq, k)
                index_after = faiss.deserialize_index(
                    faiss.serialize_index(index)
                )
                D_after, I_after = index_after.search(xq, k)

                np.testing.assert_array_equal(I_before, I_after)
                np.testing.assert_array_equal(D_before, D_after)

    def test_ratio_dims_scanned(self):
        d, nb, nq, k = 128, 500000, 10, 1

        # Setup: All vectors in the dataset are [-1, -1, ..., -1], except for
        # one, which is [1, 1, ..., 1]. The queries are [1, 1, ..., 1]. This
        # ensures that pruning happens as early as it can. As a result, the
        # ratio of dimensions scanned should be approximately (1 / nlevels).
        xb = -np.ones((nb, d)).astype("float32")
        xb[-1] = 1
        xq = np.ones((nq, d)).astype("float32")

        for metric in self.METRICS:
            with self.subTest(metric=metric):
                index_base = faiss.IndexFlat(d, metric)
                index_base.add(xb)
                D_base, I_base = index_base.search(xq, k)

                nt = faiss.omp_get_max_threads()
                faiss.omp_set_num_threads(1)

                # Force k_base = nb
                k_factor = nb // k
                params = faiss.IndexRefineSearchParameters(
                    k_factor=float(k_factor)
                )

                ratios = []
                nlevels_list = [1, 2, 4, 8, 16, 32]
                for nlevels in nlevels_list:
                    with self.subTest(nlevels=nlevels):
                        metric_str = "L2" if metric == faiss.METRIC_L2 else "IP"
                        factory_str = f"Flat{metric_str}Panorama{nlevels}_1"
                        refine_index = faiss.index_factory(
                            d, factory_str, metric
                        )
                        refine_index.add(xb)
                        index = faiss.IndexRefinePanorama(
                            index_base, refine_index
                        )

                        faiss.cvar.indexPanorama_stats.reset()
                        D, I = index.search(xq, k, params=params)
                        self.assert_search_results_equal(D_base, I_base, D, I)

                        ratios.append(
                            faiss.cvar.indexPanorama_stats.ratio_dims_scanned
                        )

                expected_ratios = [1 / nlevels for nlevels in nlevels_list]

                # Extra low tolerance for point-wise Panorama refinement
                np.testing.assert_allclose(ratios, expected_ratios, atol=1e-5)

                faiss.omp_set_num_threads(nt)
