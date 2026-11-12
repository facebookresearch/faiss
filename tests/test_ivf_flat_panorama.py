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
class TestIndexIVFFlatPanorama(unittest.TestCase):
    METRICS = [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]

    def generate_data(self, d, nt, nb, nq, seed=42):
        ds = SyntheticDataset(d, nt, nb, nq, seed=seed)
        return ds.get_train(), ds.get_database(), ds.get_queries()

    def create_ivf_flat(
        self,
        d,
        nlist,
        xt,
        xb=None,
        nprobe=None,
        metric=faiss.METRIC_L2,
    ):
        quantizer = (
            faiss.IndexFlatL2(d)
            if metric == faiss.METRIC_L2
            else faiss.IndexFlatIP(d)
        )
        index = faiss.IndexIVFFlat(quantizer, d, nlist, metric)
        index.train(xt)
        if xb is not None:
            index.add(xb)
        if nprobe is not None:
            index.nprobe = nprobe
        return index

    def create_panorama(
        self,
        d,
        nlist,
        nlevels,
        xt,
        xb=None,
        nprobe=None,
        metric=faiss.METRIC_L2,
        batch_size=faiss.Panorama.kDefaultBatchSize,
    ):
        quantizer = (
            faiss.IndexFlatL2(d)
            if metric == faiss.METRIC_L2
            else faiss.IndexFlatIP(d)
        )
        index = faiss.IndexIVFFlatPanorama(
            quantizer, d, nlist, nlevels, metric, True, batch_size
        )
        index.train(xt)
        if xb is not None:
            index.add(xb)
        if nprobe is not None:
            index.nprobe = nprobe
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

    def assert_range_results_equal(
        self,
        lims_regular,
        D_regular,
        I_regular,
        lims_panorama,
        D_panorama,
        I_panorama,
        nq,
        otol=1e-3,
        rtol=1e-4,
    ):
        """Compare range search results with tolerance for boundary
        differences."""
        total_matches = total_regular = 0

        for i in range(nq):
            ids_reg = I_regular[lims_regular[i] : lims_regular[i + 1]]
            ids_pan = I_panorama[lims_panorama[i] : lims_panorama[i + 1]]
            total_regular += len(ids_reg)

            common = np.intersect1d(ids_reg, ids_pan)
            total_matches += len(common)

            if len(common) > 0:
                # Extract and sort distances for common IDs
                mask_reg, mask_pan = np.isin(ids_reg, common), np.isin(
                    ids_pan, common
                )
                dist_reg = D_regular[lims_regular[i] : lims_regular[i + 1]]
                dist_pan = D_panorama[lims_panorama[i] : lims_panorama[i + 1]]

                sort_reg, sort_pan = np.argsort(ids_reg[mask_reg]), np.argsort(
                    ids_pan[mask_pan]
                )
                np.testing.assert_allclose(
                    dist_reg[mask_reg][sort_reg],
                    dist_pan[mask_pan][sort_pan],
                    rtol=rtol,
                    err_msg=f"Query {i}: distance mismatch",
                )

        if total_regular > 0:
            overlap_rate = total_matches / total_regular
            self.assertGreater(
                overlap_rate,
                1 - otol,
                f"Range overlap {overlap_rate:.6f} not > {1-otol:.3f}",
            )

    def validate_and_compare_range_results(
        self, metric, radius, lims_reg, D_reg, I_reg, lims_pan, D_pan, I_pan, nq
    ):
        if metric == faiss.METRIC_L2:
            self.assertTrue(
                np.all(D_pan <= radius),
                f"Some L2 distances exceed radius={radius}",
            )
        else:
            self.assertTrue(
                np.all(D_pan >= radius),
                f"Some IP distances below radius={radius}",
            )
        self.assert_range_results_equal(
            lims_reg, D_reg, I_reg, lims_pan, D_pan, I_pan, nq
        )

    def test_range_search(self):
        d, nb, nt, nq, nlist, nlevels = 128, 50000, 75000, 500, 256, 8
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=123)

        for metric in self.METRICS:
            index_regular = self.create_ivf_flat(
                d, nlist, xt, xb, nprobe=64, metric=metric
            )
            index_panorama = self.create_panorama(
                d, nlist, nlevels, xt, xb, nprobe=64, metric=metric
            )

            for radius in [0.5, 1.0, 2.0, 5.0]:
                with self.subTest(metric=metric, radius=radius):
                    lims_reg, D_reg, I_reg = index_regular.range_search(
                        xq, radius
                    )
                    lims_pan, D_pan, I_pan = index_panorama.range_search(
                        xq, radius
                    )
                    self.validate_and_compare_range_results(
                        metric,
                        radius,
                        lims_reg,
                        D_reg,
                        I_reg,
                        lims_pan,
                        D_pan,
                        I_pan,
                        nq,
                    )

    def test_different_n_levels(self):
        d, nb, nt, nq, nlist, k = 128, 25000, 40000, 300, 128, 15
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=456)

        for metric in self.METRICS:
            index_base = self.create_ivf_flat(
                d, nlist, xt, xb, nprobe=32, metric=metric
            )
            D_base, I_base = index_base.search(xq, k)

            for nlevels in [1, 2, 4, 8, 16, 32]:
                with self.subTest(metric=metric, nlevels=nlevels):
                    index = self.create_panorama(
                        d, nlist, nlevels, xt, xb, nprobe=32, metric=metric
                    )
                    D, I = index.search(xq, k)
                    self.assert_search_results_equal(D_base, I_base, D, I)

    def test_uneven_dimension_division(self):
        test_cases = [(65, 4), (63, 8), (100, 7), (960, 128), (964, 128)]

        for metric in self.METRICS:
            for d, nlevels in test_cases:
                with self.subTest(metric=metric, d=d, nlevels=nlevels):
                    nb, nt, nq, nlist, k = 5000, 700, 20, 16, 5
                    xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=789)

                    index_regular = self.create_ivf_flat(
                        d, nlist, xt, xb, nprobe=4, metric=metric
                    )
                    index_panorama = self.create_panorama(
                        d, nlist, nlevels, xt, xb, nprobe=4, metric=metric
                    )

                    D_regular, I_regular = index_regular.search(xq, k)
                    D_panorama, I_panorama = index_panorama.search(xq, k)

                    self.assert_search_results_equal(
                        D_regular, I_regular, D_panorama, I_panorama
                    )

    def test_id_selector_range(self):
        d, nb, nt, nq, nlist, nlevels, k = 128, 80000, 120000, 500, 512, 8, 20
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=321)

        for metric in self.METRICS:
            with self.subTest(metric=metric):
                index_regular = self.create_ivf_flat(
                    d, nlist, xt, xb, nprobe=64, metric=metric
                )
                index_panorama = self.create_panorama(
                    d, nlist, nlevels, xt, xb, nprobe=64, metric=metric
                )

                params = faiss.SearchParametersIVF()
                params.sel = faiss.IDSelectorRange(10000, 50000)

                D_regular, I_regular = index_regular.search(
                    xq, k, params=params
                )
                D_panorama, I_panorama = index_panorama.search(
                    xq, k, params=params
                )

                np.testing.assert_array_equal(I_regular, I_panorama)
                np.testing.assert_allclose(D_regular, D_panorama, rtol=1e-4)

    def test_selector_excludes_all(self):
        d, nb, nt, nq, nlist, nlevels, k = 32, 300, 400, 5, 8, 4, 10
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=999)

        for metric in self.METRICS:
            with self.subTest(metric=metric):
                index_panorama = self.create_panorama(
                    d, nlist, nlevels, xt, xb, nprobe=nlist, metric=metric
                )

                params = faiss.SearchParametersIVF()
                params.sel = faiss.IDSelectorRange(nb + 100, nb + 200)

                D, I = index_panorama.search(xq, k, params=params)
                self.assertTrue(np.all(I == -1))

    def test_batch_boundaries(self):
        d, nlist, nlevels, nt, nq, k = 128, 64, 8, 10000, 200, 15
        np.random.seed(987)
        xt = np.random.rand(nt, d).astype("float32")
        xq = np.random.rand(nq, d).astype("float32")

        for metric in self.METRICS:
            for nb in [5000, 50000, 75000]:
                with self.subTest(metric=metric, nb=nb):
                    xb = np.random.rand(nb, d).astype("float32")

                    index_regular = self.create_ivf_flat(
                        d, nlist, xt, xb, nprobe=16, metric=metric
                    )
                    index_panorama = self.create_panorama(
                        d, nlist, nlevels, xt, xb, nprobe=16, metric=metric
                    )

                    D_regular, I_regular = index_regular.search(xq, k)
                    D_panorama, I_panorama = index_panorama.search(xq, k)

                    self.assert_search_results_equal(
                        D_regular, I_regular, D_panorama, I_panorama
                    )

    def test_single_vector_per_cluster(self):
        d, nb, nt, nq, nlist, nlevels, k = 32, 20, 100, 5, 16, 4, 3
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=1313)

        for metric in self.METRICS:
            with self.subTest(metric=metric):
                quantizer1 = (
                    faiss.IndexFlatL2(d)
                    if metric == faiss.METRIC_L2
                    else faiss.IndexFlatIP(d)
                )
                quantizer2 = (
                    faiss.IndexFlatL2(d)
                    if metric == faiss.METRIC_L2
                    else faiss.IndexFlatIP(d)
                )
                index_regular = faiss.IndexIVFFlat(quantizer1, d, nlist, metric)
                index_panorama = faiss.IndexIVFFlatPanorama(
                    quantizer2, d, nlist, nlevels, metric
                )

                index_regular.cp.min_points_per_centroid = 1
                index_panorama.cp.min_points_per_centroid = 1

                index_regular.train(xt)
                index_regular.add(xb)
                index_regular.nprobe = nlist

                index_panorama.train(xt)
                index_panorama.add(xb)
                index_panorama.nprobe = nlist

                D_regular, I_regular = index_regular.search(xq, k)
                D_panorama, I_panorama = index_panorama.search(xq, k)

                self.assert_search_results_equal(
                    D_regular, I_regular, D_panorama, I_panorama
                )

    def test_range_search_edge_cases(self):
        d, nb, nt, nq, nlist, nlevels = 32, 500, 700, 10, 8, 4
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=777)

        for metric in self.METRICS:
            index_regular = self.create_ivf_flat(
                d, nlist, xt, xb, nprobe=nlist, metric=metric
            )
            index_panorama = self.create_panorama(
                d, nlist, nlevels, xt, xb, nprobe=nlist, metric=metric
            )

            for radius in [0.01, 0.1, 100.0, 1000.0]:
                with self.subTest(metric=metric, radius=radius):
                    lims_reg, D_reg, I_reg = index_regular.range_search(
                        xq, radius
                    )
                    lims_pan, D_pan, I_pan = index_panorama.range_search(
                        xq, radius
                    )
                    self.validate_and_compare_range_results(
                        metric,
                        radius,
                        lims_reg,
                        D_reg,
                        I_reg,
                        lims_pan,
                        D_pan,
                        I_pan,
                        nq,
                    )

    def test_add_search_add_search(self):
        d, nt, nlist, nlevels, k = 32, 500, 8, 4, 5
        np.random.seed(555)
        xt = np.random.rand(nt, d).astype("float32")

        for metric in self.METRICS:
            with self.subTest(metric=metric):
                index_regular = self.create_ivf_flat(
                    d, nlist, xt, metric=metric
                )
                index_panorama = self.create_panorama(
                    d, nlist, nlevels, xt, metric=metric
                )

                xb1 = np.random.rand(200, d).astype("float32")
                index_regular.add(xb1)
                index_panorama.add(xb1)

                xq1 = np.random.rand(10, d).astype("float32")
                index_regular.nprobe = 4
                index_panorama.nprobe = 4

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

    def test_update_vectors(self):
        d, nb, nt, nq, nlist, nlevels, k = 128, 40000, 60000, 400, 256, 8, 15
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=1414)

        for metric in self.METRICS:
            with self.subTest(metric=metric):
                index_regular = self.create_ivf_flat(
                    d, nlist, xt, nprobe=32, metric=metric
                )
                index_panorama = self.create_panorama(
                    d, nlist, nlevels, xt, nprobe=32, metric=metric
                )
                for index in index_regular, index_panorama:
                    index.make_direct_map()
                    index.add(xb)

                update_ids = np.array(
                    [100, 1000, 5000, 10000, 20000, 30000], dtype=np.int64
                )
                xb_new = np.random.rand(len(update_ids), d).astype("float32")
                index_regular.update_vectors(update_ids, xb_new)
                index_panorama.update_vectors(update_ids, xb_new)

                D_reg, I_reg = index_regular.search(xq, k)
                D_pan, I_pan = index_panorama.search(xq, k)
                self.assert_search_results_equal(D_reg, I_reg, D_pan, I_pan)

    def test_serialization(self):
        d, nb, nt, nq, nlist, nlevels, k = 128, 10000, 15000, 100, 128, 8, 20
        xt, xb, xq = self.generate_data(d, nt, nb, nq, seed=2024)

        for metric in self.METRICS:
            with self.subTest(metric=metric):
                index = self.create_panorama(
                    d, nlist, nlevels, xt, xb, nprobe=32, metric=metric
                )

                D_before, I_before = index.search(xq, k)
                index_after = faiss.deserialize_index(
                    faiss.serialize_index(index)
                )
                D_after, I_after = index_after.search(xq, k)

                np.testing.assert_array_equal(I_before, I_after)
                np.testing.assert_array_equal(D_before, D_after)

    def test_read_legacy_format(self):
        """Indexes serialized in the legacy "IwPn"/"ilpn" format (which does
        not store batch_size and implies the legacy value of 128) must keep
        deserializing correctly, whatever the current default batch_size is.

        The current code always writes the explicit-batch_size format
        ("IwP2"/"ilp2"), so a legacy stream is reconstructed here by
        transforming a serialized index: swap the fourccs and drop the two
        8-byte batch_size fields.
        """
        d, nlist, nlevels, nb, nq, k = 32, 4, 8, 2000, 10, 10
        legacy_bs = 128
        rng = np.random.RandomState(123)
        xb = rng.rand(nb, d).astype("float32")
        xq = rng.rand(nq, d).astype("float32")

        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlatPanorama(
            quantizer, d, nlist, nlevels, faiss.METRIC_L2, True, legacy_bs
        )
        index.train(xb)
        index.add(xb)
        index.nprobe = nlist
        D_ref, I_ref = index.search(xq, k)

        buf = faiss.serialize_index(index).tobytes()
        # layout: "IwP2" | ivf header | n_levels (8) | batch_size (8) |
        #         "ilp2" | nlist (8) | code_size (8) | n_levels (8) |
        #         batch_size (8) | inverted lists data
        self.assertEqual(buf[:4], b"IwP2")
        self.assertEqual(buf.count(b"ilp2"), 1)
        p = buf.index(b"ilp2")
        legacy = (
            b"IwPn"
            + buf[4 : p - 8]  # drop the IVF-level batch_size field
            + b"ilpn"
            + buf[p + 4 : p + 28]  # nlist, code_size, n_levels
            + buf[p + 36 :]  # drop the invlist-level batch_size field
        )

        index_legacy = faiss.deserialize_index(
            np.frombuffer(legacy, dtype=np.uint8)
        )
        self.assertIsInstance(index_legacy, faiss.IndexIVFFlatPanorama)
        self.assertEqual(index_legacy.batch_size, legacy_bs)
        index_legacy.nprobe = nlist
        D_legacy, I_legacy = index_legacy.search(xq, k)
        np.testing.assert_array_equal(I_ref, I_legacy)
        np.testing.assert_array_equal(D_ref, D_legacy)

    def test_ratio_dims_scanned(self):
        d, nb, nq, nlist, k = 128, 500000, 1, 1, 1

        # Setup: All vectors in the dataset are [1, 1, ..., 1], except for
        # one, which is [0, 0, ..., 0]. The query is also [0, 0, ..., 0]. This
        # ensures that pruning happens as early as it can. As a result, the
        # ratio of dimensions scanned should be approximately (1 / nlevels).
        xb = np.ones((nb, d)).astype("float32")
        xb[-1] = 0
        xt = xb
        xq = np.zeros((nq, d)).astype("float32")

        for metric in self.METRICS:
            with self.subTest(metric=metric):
                index_base = self.create_ivf_flat(
                    d, nlist, xt, xb, nprobe=1, metric=metric
                )
                D_base, I_base = index_base.search(xq, k)

                nt = faiss.omp_get_max_threads()
                faiss.omp_set_num_threads(1)

                ratios = []
                nlevels_list = [1, 2, 4, 8, 16, 32]
                for nlevels in nlevels_list:
                    with self.subTest(nlevels=nlevels):
                        faiss.cvar.indexPanorama_stats.reset()
                        index = self.create_panorama(
                            d,
                            nlist,
                            nlevels,
                            xt,
                            xb,
                            nprobe=1,
                            metric=metric,
                            batch_size=128,
                        )
                        D, I = index.search(xq, k)
                        self.assert_search_results_equal(D_base, I_base, D, I)

                        ratios.append(
                            faiss.cvar.indexPanorama_stats.ratio_dims_scanned
                        )

                expected_ratios = [1 / nlevels for nlevels in nlevels_list]
                np.testing.assert_allclose(ratios, expected_ratios, atol=1e-3)

                faiss.omp_set_num_threads(nt)
