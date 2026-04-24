# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""Tests for TurboQuant (IndexIVFTurboQ).

Mirrors test_rabitq.py structure: construction, factory strings,
recall quality, serialization, IVF operations, and edge cases.
Both L2 and IP metrics are tested throughout.
"""

import unittest

import faiss
import numpy as np
from faiss.contrib import datasets


def compute_expected_turboq_code_size(d, nb_bits):
    """Compute expected code size for TurboQ.

    Layout: [MSE bit planes] [QJL sign bits] [TurboQFactors(norm, gamma)]
    """
    return (d + 7) // 8 * nb_bits + 8


def do_test_serde(description, metric=faiss.METRIC_L2):
    ds = datasets.SyntheticDataset(64, 1000, 100, 20)
    index = faiss.index_factory(ds.d, description, metric)
    index.train(ds.get_train())
    index.add(ds.get_database())
    Dref, Iref = index.search(ds.get_queries(), 10)

    b = faiss.serialize_index(index)
    index2 = faiss.deserialize_index(b)
    Dnew, Inew = index2.search(ds.get_queries(), 10)

    np.testing.assert_equal(Dref, Dnew)
    np.testing.assert_equal(Iref, Inew)

    b2 = faiss.serialize_index(index2)
    index3 = faiss.deserialize_index(b2)
    Dnew3, Inew3 = index3.search(ds.get_queries(), 10)
    np.testing.assert_equal(Dref, Dnew3)
    np.testing.assert_equal(Iref, Inew3)


class TestTurboQ(unittest.TestCase):
    """Consolidated tests for TurboQuant."""

    # ==================== Construction Tests ====================

    def test_valid_nb_bits_range(self):
        d = 128
        for nb_bits in range(1, 6):
            for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
                with self.subTest(nb_bits=nb_bits, metric=metric):
                    quantizer = faiss.IndexFlat(d, metric)
                    index = faiss.IndexIVFTurboQ(
                        quantizer, d, 1, metric, True, nb_bits
                    )
                    self.assertEqual(index.d, d)
                    self.assertEqual(index.turboq.nb_bits, nb_bits)

    def test_invalid_nb_bits(self):
        d = 128
        for nb_bits in [0, 6, 9]:
            with self.subTest(nb_bits=nb_bits):
                with self.assertRaises(RuntimeError):
                    faiss.index_factory(
                        d, f"RR{d},IVF1,TurboQ{nb_bits}", faiss.METRIC_L2
                    )

    def test_code_size_formula(self):
        d = 128
        for nb_bits in range(1, 6):
            with self.subTest(nb_bits=nb_bits):
                quantizer = faiss.IndexFlat(d, faiss.METRIC_L2)
                index = faiss.IndexIVFTurboQ(
                    quantizer, d, 1, faiss.METRIC_L2, True, nb_bits
                )
                expected = compute_expected_turboq_code_size(d, nb_bits)
                self.assertEqual(index.code_size, expected)

    # ==================== Factory String Tests ====================

    def test_factory_default(self):
        index = faiss.index_factory(128, "RR128,TurboQ")
        ivf = faiss.extract_index_ivf(index)
        self.assertEqual(
            ivf.code_size, compute_expected_turboq_code_size(128, 2)
        )

    def test_factory_multibit(self):
        for nb_bits in [1, 2, 3, 4, 5]:
            with self.subTest(nb_bits=nb_bits):
                index = faiss.index_factory(128, f"RR128,TurboQ{nb_bits}")
                ivf = faiss.extract_index_ivf(index)
                self.assertEqual(
                    ivf.code_size,
                    compute_expected_turboq_code_size(128, nb_bits),
                )

    def test_factory_random_rotation(self):
        ds = datasets.SyntheticDataset(64, 150, 200, 10)
        for nb_bits in [2, 4]:
            with self.subTest(nb_bits=nb_bits):
                index = faiss.index_factory(
                    ds.d, f"RR{ds.d},TurboQ{nb_bits}r"
                )
                index.train(ds.get_train())
                index.add(ds.get_database())
                D, I = index.search(ds.get_queries(), 5)
                self.assertTrue(np.all(I >= 0))

    def test_factory_ivf(self):
        for nlist in [1, 16]:
            for nb_bits in [2, 4]:
                with self.subTest(nlist=nlist, nb_bits=nb_bits):
                    index = faiss.index_factory(
                        128, f"RR128,IVF{nlist},TurboQ{nb_bits}"
                    )
                    ivf = faiss.extract_index_ivf(index)
                    self.assertEqual(ivf.nlist, nlist)
                    self.assertEqual(
                        ivf.code_size,
                        compute_expected_turboq_code_size(128, nb_bits),
                    )

    def test_factory_standalone(self):
        index = faiss.index_factory(128, "RR128,TurboQ2")
        ivf = faiss.extract_index_ivf(index)
        self.assertEqual(ivf.nlist, 1)

    def test_factory_end_to_end(self):
        ds = datasets.SyntheticDataset(64, 150, 200, 10)
        for nb_bits in [2, 4]:
            for suffix in [f"TurboQ{nb_bits}", f"IVF16,TurboQ{nb_bits}"]:
                factory_str = f"RR{ds.d},{suffix}"
                with self.subTest(factory=factory_str):
                    index = faiss.index_factory(
                        ds.d, factory_str, faiss.METRIC_INNER_PRODUCT
                    )
                    index.train(ds.get_train())
                    index.add(ds.get_database())
                    D, I = index.search(ds.get_queries(), 5)
                    self.assertEqual(D.shape, (ds.nq, 5))
                    self.assertTrue(np.all(I >= 0))

    # ==================== Basic Operations Tests ====================

    def test_basic_operations(self):
        ds = datasets.SyntheticDataset(128, 300, 500, 20)
        k = 10

        for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
            for nb_bits in [2, 4]:
                with self.subTest(metric=metric, nb_bits=nb_bits):
                    index = faiss.index_factory(
                        ds.d,
                        f"RR{ds.d},IVF1,TurboQ{nb_bits}",
                        metric,
                    )
                    index.train(ds.get_train())
                    index.add(ds.get_database())
                    D, I = index.search(ds.get_queries(), k)

                    self.assertTrue(index.is_trained)
                    self.assertEqual(index.ntotal, ds.nb)
                    self.assertEqual(D.shape, (ds.nq, k))
                    self.assertEqual(I.shape, (ds.nq, k))
                    self.assertTrue(np.all(I >= 0))
                    self.assertTrue(np.all(I < ds.nb))
                    self.assertTrue(np.all(np.isfinite(D)))

    # ==================== Recall Tests ====================

    def test_recall_quality(self):
        for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
            metric_str = "L2" if metric == faiss.METRIC_L2 else "IP"
            ds = datasets.SyntheticDataset(
                128, 500, 2000, 50, metric=metric_str
            )
            I_gt = ds.get_groundtruth(10)

            for nb_bits in [2, 4]:
                with self.subTest(metric=metric_str, nb_bits=nb_bits):
                    index = faiss.index_factory(
                        ds.d,
                        f"RR{ds.d},IVF1,TurboQ{nb_bits}",
                        metric,
                    )
                    index.train(ds.get_train())
                    index.add(ds.get_database())
                    _, I = index.search(ds.get_queries(), 10)
                    recall = (
                        faiss.eval_intersection(I, I_gt) / (ds.nq * 10)
                    )
                    self.assertGreater(
                        recall,
                        0.10,
                        f"TurboQ{nb_bits} {metric_str}: recall={recall:.3f}",
                    )

    def test_recall_monotonic_improvement(self):
        for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
            metric_str = "L2" if metric == faiss.METRIC_L2 else "IP"
            ds = datasets.SyntheticDataset(
                128, 500, 2000, 50, metric=metric_str
            )
            I_gt = ds.get_groundtruth(10)

            with self.subTest(metric=metric_str):
                recalls = {}
                for nb_bits in [2, 3, 4]:
                    index = faiss.index_factory(
                        ds.d,
                        f"RR{ds.d},IVF1,TurboQ{nb_bits}",
                        metric,
                    )
                    index.train(ds.get_train())
                    index.add(ds.get_database())
                    _, I = index.search(ds.get_queries(), 10)
                    recalls[nb_bits] = (
                        faiss.eval_intersection(I, I_gt) / (ds.nq * 10)
                    )

                tolerance = 0.03
                self.assertGreaterEqual(recalls[3], recalls[2] - tolerance)
                self.assertGreaterEqual(recalls[4], recalls[3] - tolerance)

    # ==================== QJL Projection Type Tests ====================

    def test_both_qjl_projections_recall(self):
        """Both FWHT and Random Rotation give reasonable recall."""
        for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
            metric_str = "L2" if metric == faiss.METRIC_L2 else "IP"
            ds = datasets.SyntheticDataset(
                128, 500, 2000, 50, metric=metric_str
            )
            I_gt = ds.get_groundtruth(10)

            for suffix in ["TurboQ2", "TurboQ2r"]:
                with self.subTest(metric=metric_str, qjl=suffix):
                    index = faiss.index_factory(
                        ds.d,
                        f"RR{ds.d},IVF1,{suffix}",
                        metric,
                    )
                    index.train(ds.get_train())
                    index.add(ds.get_database())
                    _, I = index.search(ds.get_queries(), 10)
                    recall = (
                        faiss.eval_intersection(I, I_gt)
                        / (ds.nq * 10)
                    )
                    self.assertGreater(
                        recall,
                        0.10,
                        f"{suffix} {metric_str}: recall={recall:.3f}",
                    )

    def test_rr_qjl_ivf(self):
        """Random Rotation QJL works with multi-list IVF."""
        ds = datasets.SyntheticDataset(128, 300, 500, 20)
        for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
            with self.subTest(metric=metric):
                index = faiss.index_factory(
                    ds.d,
                    f"RR{ds.d},IVF16,TurboQ2r",
                    metric,
                )
                index.train(ds.get_train())
                index.add(ds.get_database())
                params = faiss.IVFTurboQSearchParameters()
                params.nprobe = 4
                D, I = index.search(
                    ds.get_queries(), 10, params=params
                )
                self.assertTrue(np.all(I >= 0))
                self.assertTrue(np.all(np.isfinite(D)))

    def test_rr_qjl_with_qb(self):
        """Integer popcount path works with Random Rotation QJL."""
        ds = datasets.SyntheticDataset(
            128, 500, 2000, 50, metric="IP"
        )
        index = faiss.index_factory(
            ds.d,
            f"RR{ds.d},IVF1,TurboQ2r",
            faiss.METRIC_INNER_PRODUCT,
        )
        index.train(ds.get_train())
        index.add(ds.get_database())

        sp0 = faiss.IVFTurboQSearchParameters()
        sp0.nprobe = 1
        sp0.qb = 0
        D0, I0 = index.search(ds.get_queries(), 10, params=sp0)

        for qb in [4, 8]:
            sp = faiss.IVFTurboQSearchParameters()
            sp.nprobe = 1
            sp.qb = qb
            D, I = index.search(ds.get_queries(), 10, params=sp)
            recall = np.mean(
                [
                    len(np.intersect1d(I[i], I0[i])) / 10.0
                    for i in range(len(I))
                ]
            )
            self.assertGreater(
                recall,
                0.8,
                f"TurboQ2r qb={qb}: recall {recall:.3f} "
                f"too low vs float path",
            )

    # ==================== Query Quantization Tests ====================

    def test_qb_integer_mse(self):
        """qb parameter enables integer popcount MSE path."""
        for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
            metric_str = "L2" if metric == faiss.METRIC_L2 else "IP"
            ds = datasets.SyntheticDataset(
                128, 500, 2000, 50, metric=metric_str
            )

            with self.subTest(metric=metric_str):
                index = faiss.index_factory(
                    ds.d,
                    f"RR{ds.d},IVF1,TurboQ2",
                    metric,
                )
                index.train(ds.get_train())
                index.add(ds.get_database())

                sp0 = faiss.IVFTurboQSearchParameters()
                sp0.nprobe = 1
                sp0.qb = 0
                D0, I0 = index.search(ds.get_queries(), 10, params=sp0)

                for qb in [4, 8]:
                    sp = faiss.IVFTurboQSearchParameters()
                    sp.nprobe = 1
                    sp.qb = qb
                    D, I = index.search(ds.get_queries(), 10, params=sp)
                    recall = np.mean(
                        [
                            len(np.intersect1d(I[i], I0[i])) / 10.0
                            for i in range(len(I))
                        ]
                    )
                    self.assertGreater(
                        recall,
                        0.8,
                        f"qb={qb}: recall {recall:.3f} too low vs float path",
                    )

    # ==================== IVF Tests ====================

    def test_ivf_basic_operations(self):
        ds = datasets.SyntheticDataset(128, 300, 500, 20)

        for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
            for nb_bits in [2, 4]:
                with self.subTest(metric=metric, nb_bits=nb_bits):
                    index = faiss.index_factory(
                        ds.d,
                        f"RR{ds.d},IVF16,TurboQ{nb_bits}",
                        metric,
                    )
                    index.train(ds.get_train())
                    index.add(ds.get_database())

                    params = faiss.IVFTurboQSearchParameters()
                    params.nprobe = 4
                    D, I = index.search(
                        ds.get_queries(), 10, params=params
                    )

                    self.assertTrue(index.is_trained)
                    self.assertEqual(index.ntotal, ds.nb)
                    self.assertEqual(D.shape, (ds.nq, 10))
                    self.assertTrue(np.all(I >= 0))
                    self.assertTrue(np.all(np.isfinite(D)))

    def test_ivf_nprobe_improves_recall(self):
        for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
            metric_str = "L2" if metric == faiss.METRIC_L2 else "IP"
            ds = datasets.SyntheticDataset(
                128, 500, 2000, 50, metric=metric_str
            )
            I_gt = ds.get_groundtruth(10)

            for nb_bits in [2, 4]:
                with self.subTest(metric=metric_str, nb_bits=nb_bits):
                    index = faiss.index_factory(
                        ds.d,
                        f"RR{ds.d},IVF32,TurboQ{nb_bits}",
                        metric,
                    )
                    index.train(ds.get_train())
                    index.add(ds.get_database())

                    recalls = {}
                    for nprobe in [1, 2, 4, 8]:
                        params = faiss.IVFTurboQSearchParameters()
                        params.nprobe = nprobe
                        _, I = index.search(
                            ds.get_queries(), 10, params=params
                        )
                        recalls[nprobe] = (
                            faiss.eval_intersection(I, I_gt) / (ds.nq * 10)
                        )

                    self.assertGreaterEqual(recalls[2], recalls[1])
                    self.assertGreaterEqual(recalls[4], recalls[2])
                    self.assertGreaterEqual(recalls[8], recalls[4])

    # ==================== Serialization Tests ====================

    def test_serde_turboq2_l2(self):
        do_test_serde("RR64,IVF1,TurboQ2", faiss.METRIC_L2)

    def test_serde_turboq2_ip(self):
        do_test_serde("RR64,IVF1,TurboQ2", faiss.METRIC_INNER_PRODUCT)

    def test_serde_turboq4(self):
        do_test_serde("RR64,IVF1,TurboQ4")

    def test_serde_ivf_turboq(self):
        do_test_serde("RR64,IVF16,TurboQ2")

    def test_serde_random_rotation(self):
        do_test_serde("RR64,IVF1,TurboQ2r")

    def test_serde_random_rotation_4bit(self):
        do_test_serde("RR64,IVF1,TurboQ4r")

    def test_serialization_all_configs(self):
        ds = datasets.SyntheticDataset(64, 150, 200, 10)

        for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
            for nb_bits in [1, 2, 4]:
                with self.subTest(metric=metric, nb_bits=nb_bits):
                    index1 = faiss.index_factory(
                        ds.d,
                        f"RR{ds.d},IVF1,TurboQ{nb_bits}",
                        metric,
                    )
                    index1.train(ds.get_train())
                    index1.add(ds.get_database())
                    D1, I1 = index1.search(ds.get_queries(), 5)

                    index_bytes = faiss.serialize_index(index1)
                    index2 = faiss.deserialize_index(index_bytes)

                    self.assertEqual(index2.d, ds.d)
                    self.assertEqual(index2.ntotal, ds.nb)
                    self.assertTrue(index2.is_trained)

                    D2, I2 = index2.search(ds.get_queries(), 5)
                    np.testing.assert_array_equal(I1, I2)
                    np.testing.assert_allclose(D1, D2, rtol=1e-5)

    # ==================== Edge Cases ====================

    def test_non_power_of_2_dims(self):
        for d in [200, 300]:
            with self.subTest(d=d):
                ds = datasets.SyntheticDataset(d, 150, 500, 10)
                index = faiss.index_factory(
                    d,
                    f"RR{d},IVF1,TurboQ2",
                    faiss.METRIC_INNER_PRODUCT,
                )
                index.train(ds.get_train())
                index.add(ds.get_database())
                D, I = index.search(ds.get_queries(), 5)
                self.assertTrue(np.all(I >= 0))
                self.assertTrue(np.all(I < ds.nb))
                self.assertTrue(np.all(np.isfinite(D)))

    def test_by_residual_false(self):
        d = 128
        for nlist in [1, 16]:
            quantizer = faiss.IndexFlat(d, faiss.METRIC_L2)
            index = faiss.IndexIVFTurboQ(
                quantizer, d, nlist, faiss.METRIC_L2
            )
            self.assertFalse(index.by_residual)
