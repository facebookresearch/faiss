# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import faiss
from faiss.contrib import datasets

from common_faiss_tests import for_all_simd_levels, NoneSIMDLevel


def compute_expected_code_size(d, nb_bits):
    ex_bits = nb_bits - 1
    # For 1-bit: use SignBitFactors (8 bytes) for non-IVF
    # For multi-bit: use SignBitFactorsWithError (12 bytes)
    base_size = (d + 7) // 8 + (8 if ex_bits == 0 else 12)
    if ex_bits > 0:
        # ex-bit codes + ExtraBitsFactors
        ex_size = (d * ex_bits + 7) // 8 + 8
        return base_size + ex_size
    return base_size


def _create_fastscan_index(
    d,
    metric,
    use_ivf=False,
    nlist=16,
    nprobe=4,
    nb_bits=1,
):
    if use_ivf:
        quantizer = faiss.IndexFlat(d, metric)
        index = faiss.IndexIVFRaBitQFastScan(
            quantizer, d, nlist, metric, 32, True, nb_bits
        )
        index.nprobe = nprobe
    else:
        index = faiss.IndexRaBitQFastScan(d, metric, 32, nb_bits)
    return index


@for_all_simd_levels
class TestRaBitQFastScan(unittest.TestCase):
    NLIST = 16
    NPROBE = 4

    def _create_index(self, d, metric, use_ivf=False, nlist=None, nb_bits=1):
        return _create_fastscan_index(
            d,
            metric,
            use_ivf=use_ivf,
            nlist=nlist or self.NLIST,
            nprobe=self.NPROBE,
            nb_bits=nb_bits,
        )

    def _create_baseline(self, d, metric, use_ivf=False, nlist=None):
        if use_ivf:
            nlist = nlist or self.NLIST
            quantizer = faiss.IndexFlat(d, metric)
            index = faiss.IndexIVFRaBitQ(quantizer, d, nlist, metric)
            index.nprobe = self.NPROBE
        else:
            index = faiss.IndexRaBitQ(d, metric)
        return index

    def test_comparison_vs_baseline(self):
        ds = datasets.SyntheticDataset(128, 4096, 4096, 100)
        k = 10

        for use_ivf in [False, True]:
            for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
                with self.subTest(use_ivf=use_ivf, metric=metric):
                    index_flat = faiss.IndexFlat(ds.d, metric)
                    index_flat.add(ds.get_database())
                    _, I_gt = index_flat.search(ds.get_queries(), k)

                    index_base = self._create_baseline(ds.d, metric, use_ivf)
                    index_base.train(ds.get_train())
                    index_base.add(ds.get_database())
                    _, I_base = index_base.search(ds.get_queries(), k)

                    index_fs = self._create_index(ds.d, metric, use_ivf)
                    index_fs.train(ds.get_train())
                    index_fs.add(ds.get_database())
                    _, I_fs = index_fs.search(ds.get_queries(), k)

                    eval_base = faiss.eval_intersection(
                        I_base[:, :k], I_gt[:, :k]
                    ) / (ds.nq * k)
                    eval_fs = faiss.eval_intersection(
                        I_fs[:, :k], I_gt[:, :k]
                    ) / (ds.nq * k)

                    np.testing.assert_(abs(eval_base - eval_fs) < 0.05)
                    if not use_ivf and metric == faiss.METRIC_L2:
                        np.testing.assert_(eval_fs > 0.55)

    def test_encode_decode_consistency(self):
        for use_ivf in [False, True]:
            with self.subTest(use_ivf=use_ivf):
                nlist = 32
                ds = datasets.SyntheticDataset(64, 1000, 1000, 0)
                test_vectors = ds.get_database()[:100]

                if use_ivf:
                    index_fs = self._create_index(
                        ds.d, faiss.METRIC_L2, True, nlist
                    )
                    index_fs.train(ds.get_train())
                    index_fs.add(test_vectors)
                    decoded_fs = index_fs.reconstruct_n(0, len(test_vectors))

                    index_base = self._create_baseline(
                        ds.d, faiss.METRIC_L2, True, nlist
                    )
                    index_base.train(ds.get_train())
                    index_base.add(test_vectors)
                    decoded_base = index_base.reconstruct_n(
                        0, len(test_vectors)
                    )
                else:
                    index_fs = faiss.IndexRaBitQFastScan(ds.d, faiss.METRIC_L2)
                    index_fs.train(ds.get_train())

                    codes_fs = np.empty(
                        (len(test_vectors), index_fs.code_size), dtype=np.uint8
                    )
                    index_fs.compute_codes(
                        faiss.swig_ptr(codes_fs),
                        len(test_vectors),
                        faiss.swig_ptr(test_vectors),
                    )
                    decoded_fs = index_fs.sa_decode(codes_fs)

                    index_base = faiss.IndexRaBitQ(ds.d, faiss.METRIC_L2)
                    index_base.train(ds.get_train())
                    codes_base = index_base.sa_encode(test_vectors)
                    decoded_base = index_base.sa_decode(codes_base)

                err_fs = np.mean(
                    np.sum((test_vectors - decoded_fs) ** 2, axis=1)
                )
                err_base = np.mean(
                    np.sum((test_vectors - decoded_base) ** 2, axis=1)
                )

                np.testing.assert_(abs(err_fs - err_base) < 0.01)

    def test_serialization(self):
        for use_ivf in [False, True]:
            for nb_bits in [1, 2, 4]:
                with self.subTest(use_ivf=use_ivf, nb_bits=nb_bits):
                    ds = datasets.SyntheticDataset(128, 1000, 200, 20)

                    index = self._create_index(
                        128,
                        faiss.METRIC_L2,
                        use_ivf,
                        nb_bits=nb_bits,
                    )
                    index.train(ds.get_train())
                    index.add(ds.get_database())

                    Dref, Iref = index.search(ds.get_queries(), 10)

                    b = faiss.serialize_index(index)
                    index2 = faiss.deserialize_index(b)
                    if use_ivf:
                        index2.nprobe = self.NPROBE

                    Dnew, Inew = index2.search(ds.get_queries(), 10)

                    np.testing.assert_array_equal(Dref, Dnew)
                    np.testing.assert_array_equal(Iref, Inew)

                b2 = faiss.serialize_index(index2)
                index3 = faiss.deserialize_index(b2)
                if use_ivf:
                    index3.nprobe = self.NPROBE
                Dnew3, Inew3 = index3.search(ds.get_queries(), 10)
                np.testing.assert_array_equal(Dref, Dnew3)
                np.testing.assert_array_equal(Iref, Inew3)

    def test_memory_management(self):
        """Adds in chunks that end mid-block."""
        for use_ivf in [False, True]:
            with self.subTest(use_ivf=use_ivf):
                ds = datasets.SyntheticDataset(128, 2000, 2000, 50)

                index = self._create_index(ds.d, faiss.METRIC_L2, use_ivf)
                index.train(ds.get_train())

                chunk_size = 500
                for i in range(0, ds.nb, chunk_size):
                    end_idx = min(i + chunk_size, ds.nb)
                    index.add(ds.get_database()[i:end_idx])

                np.testing.assert_equal(index.ntotal, ds.nb)

                _, I = index.search(ds.get_queries(), 5)

                I_gt = ds.get_groundtruth(5)
                recall = faiss.eval_intersection(I[:, :5], I_gt[:, :5])
                recall /= ds.nq * 5
                np.testing.assert_(recall > 0.1)

    def test_thread_safety(self):
        """L2 distances must be clamped to >= 0."""
        for use_ivf in [False, True]:
            with self.subTest(use_ivf=use_ivf):
                nq = 300 if use_ivf else 500
                ds = datasets.SyntheticDataset(64, 2000, 2000, nq)

                index = self._create_index(ds.d, faiss.METRIC_L2, use_ivf)
                index.train(ds.get_train())
                index.add(ds.get_database())

                k = 10
                distances, labels = index.search(ds.get_queries(), k)

                np.testing.assert_(np.all(distances >= 0))
                np.testing.assert_(np.all(labels >= 0))
                np.testing.assert_(np.all(labels < ds.nb))

    def test_query_quantization_bits(self):
        ds = datasets.SyntheticDataset(64, 2000, 2000, 50)
        k = 10

        index = faiss.IndexRaBitQFastScan(ds.d, faiss.METRIC_L2)
        index.train(ds.get_train())
        index.add(ds.get_database())

        I_gt = ds.get_groundtruth(k)

        for qb in [4, 6, 8]:
            index.qb = qb
            _, I = index.search(ds.get_queries(), k)
            recall = faiss.eval_intersection(I[:, :k], I_gt[:, :k])
            recall /= ds.nq * k
            np.testing.assert_(recall > 0.4)

    def test_invalid_parameters(self):
        with np.testing.assert_raises(Exception):
            faiss.IndexRaBitQFastScan(0, faiss.METRIC_L2)

        try:
            faiss.IndexRaBitQFastScan(64, faiss.METRIC_Lp)
            self.fail("Should have raised exception for invalid metric")
        except RuntimeError:
            pass

    def test_nprobe_variations(self):
        nlist = 32
        ds = datasets.SyntheticDataset(64, 1000, 1000, 50)
        k = 10
        I_gt = ds.get_groundtruth(k)

        for nprobe in [1, 4, 8, 16]:
            with self.subTest(nprobe=nprobe):
                quantizer1 = faiss.IndexFlat(ds.d, faiss.METRIC_L2)
                index_base = faiss.IndexIVFRaBitQ(
                    quantizer1, ds.d, nlist, faiss.METRIC_L2
                )
                index_base.train(ds.get_train())
                index_base.add(ds.get_database())

                params_base = faiss.IVFRaBitQSearchParameters()
                params_base.nprobe = nprobe
                params_base.qb = 8
                params_base.centered = False
                _, I_base = index_base.search(
                    ds.get_queries(), k, params=params_base
                )

                quantizer2 = faiss.IndexFlat(ds.d, faiss.METRIC_L2)
                index_fs = faiss.IndexIVFRaBitQFastScan(
                    quantizer2, ds.d, nlist, faiss.METRIC_L2, 32
                )
                index_fs.qb = 8
                index_fs.centered = False
                index_fs.train(ds.get_train())
                index_fs.add(ds.get_database())

                params_fs = faiss.IVFSearchParameters()
                params_fs.nprobe = nprobe
                _, I_fs = index_fs.search(ds.get_queries(), k, params=params_fs)

                eval_base = faiss.eval_intersection(
                    I_base[:, :k], I_gt[:, :k]
                ) / (ds.nq * k)
                eval_fs = faiss.eval_intersection(I_fs[:, :k], I_gt[:, :k]) / (
                    ds.nq * k
                )

                np.testing.assert_(abs(eval_base - eval_fs) < 0.01)

    def _do_test_search_implementation(self, impl):
        nlist, nprobe = 32, 8
        ds = datasets.SyntheticDataset(128, 2048, 2048, 100)
        k = 10

        I_gt = ds.get_groundtruth(k)

        quantizer1 = faiss.IndexFlat(ds.d, faiss.METRIC_L2)
        index_base = faiss.IndexIVFRaBitQ(
            quantizer1, ds.d, nlist, faiss.METRIC_L2
        )
        index_base.qb = 8
        index_base.nprobe = nprobe
        index_base.train(ds.get_train())
        index_base.add(ds.get_database())

        params_base = faiss.IVFRaBitQSearchParameters()
        params_base.nprobe = nprobe
        params_base.qb = 8
        params_base.centered = False
        _, I_base = index_base.search(ds.get_queries(), k, params=params_base)

        eval_base = faiss.eval_intersection(I_base[:, :k], I_gt[:, :k]) / (
            ds.nq * k
        )

        quantizer2 = faiss.IndexFlat(ds.d, faiss.METRIC_L2)
        index_fs = faiss.IndexIVFRaBitQFastScan(
            quantizer2, ds.d, nlist, faiss.METRIC_L2, 32
        )
        index_fs.qb = 8
        index_fs.centered = False
        index_fs.nprobe = nprobe
        index_fs.implem = impl
        index_fs.train(ds.get_train())
        index_fs.add(ds.get_database())

        params_fs = faiss.IVFSearchParameters()
        params_fs.nprobe = nprobe
        _, I_fs = index_fs.search(ds.get_queries(), k, params=params_fs)

        eval_fs = faiss.eval_intersection(I_fs[:, :k], I_gt[:, :k]) / (
            ds.nq * k
        )

        np.testing.assert_(abs(eval_base - eval_fs) < 0.05)

    def test_search_implem_10(self):
        self._do_test_search_implementation(impl=10)

    def test_search_implem_14(self):
        self._do_test_search_implementation(impl=14)

    def test_search_with_parameters(self):
        nlist, nprobe, nq = 64, 8, 500
        ds = datasets.SyntheticDataset(128, 2048, 2048, nq)
        k = 10

        quantizer = faiss.IndexFlat(ds.d, faiss.METRIC_L2)
        index = faiss.IndexIVFRaBitQFastScan(
            quantizer, ds.d, nlist, faiss.METRIC_L2, 32
        )
        index.qb = 8
        index.centered = False
        index.nprobe = nprobe
        index.train(ds.get_train())
        index.add(ds.get_database())

        params = faiss.IVFSearchParameters()
        params.nprobe = nprobe

        _, I = faiss.search_with_parameters(index, ds.get_queries(), k, params)

        I_gt = ds.get_groundtruth(k)
        recall = faiss.eval_intersection(I, I_gt) / (nq * k)
        self.assertGreater(recall, 0.4)

    def test_ivf_large_bbs_aux_offsets(self):
        """nlist=1 puts all vectors in one list spanning many bbs=64 blocks, so
        sub-blocks b>=1 need the idx_base % bbs aux offset; bbs=64 recall must
        match bbs=32."""
        d, nlist, nprobe, k = 64, 1, 1, 10
        ds = datasets.SyntheticDataset(d, 2000, 2000, 100)
        I_gt = ds.get_groundtruth(k)

        def recall_for_bbs(bbs):
            quantizer = faiss.IndexFlat(d, faiss.METRIC_L2)
            index = faiss.IndexIVFRaBitQFastScan(
                quantizer, d, nlist, faiss.METRIC_L2, bbs, True, 1
            )
            index.qb = 8
            index.nprobe = nprobe
            index.train(ds.get_train())
            index.add(ds.get_database())
            _, I = index.search(ds.get_queries(), k)
            return faiss.eval_intersection(I[:, :k], I_gt[:, :k]) / (ds.nq * k)

        recall_bbs32 = recall_for_bbs(32)
        recall_bbs64 = recall_for_bbs(64)
        np.testing.assert_(
            abs(recall_bbs32 - recall_bbs64) < 0.02,
            f"bbs=64 recall ({recall_bbs64:.3f}) diverged from bbs=32 "
            f"({recall_bbs32:.3f}); aux offsets likely wrong for bbs>32",
        )


@for_all_simd_levels
class TestIVFRaBitQFastScanFiltering(unittest.TestCase):
    NLIST = 32
    NPROBE = 8

    def _create_index(self, d, metric, nb_bits=1):
        quantizer = faiss.IndexFlat(d, metric)
        index = faiss.IndexIVFRaBitQFastScan(
            quantizer, d, self.NLIST, metric, 32, True, nb_bits
        )
        index.nprobe = self.NPROBE
        return index

    def _do_test_filter(
        self, selector_type="batch", metric=faiss.METRIC_L2, nb_bits=1
    ):
        d, k, nb = 32, 10, 1000
        ds = datasets.SyntheticDataset(d, 2000, nb, 20, metric=metric)
        index = self._create_index(d, metric, nb_bits=nb_bits)
        index.train(ds.get_train())
        index.add(ds.get_database())

        rs = np.random.RandomState(123)
        if selector_type == "batch":
            subset = rs.choice(nb, nb // 3, replace=False).astype("int64")
            sel = faiss.IDSelectorBatch(subset)
            allowed = set(subset.tolist())
        elif selector_type == "empty":
            sel = faiss.IDSelectorBatch(np.array([], dtype="int64"))
            allowed = set()
        else:
            raise ValueError(f"Unknown selector type: {selector_type}")

        params = faiss.SearchParametersIVF(sel=sel, nprobe=self.NPROBE)
        _, labels = index.search(ds.get_queries(), k, params=params)

        for q in range(ds.nq):
            for j in range(k):
                idx = int(labels[q, j])
                if idx >= 0:
                    self.assertIn(
                        idx,
                        allowed,
                        f"Query {q}, rank {j}: id {idx} not allowed",
                    )

        if selector_type == "empty":
            np.testing.assert_array_equal(labels, -1)
        else:
            self.assertGreater(np.sum(labels >= 0), 0)

    def test_batch_filter_l2_1bit(self):
        self._do_test_filter("batch", nb_bits=1)

    def test_batch_filter_ip_1bit(self):
        self._do_test_filter("batch", faiss.METRIC_INNER_PRODUCT, nb_bits=1)

    def test_empty_selector_l2_1bit(self):
        self._do_test_filter("empty", nb_bits=1)

    def test_batch_filter_l2_multibit(self):
        self._do_test_filter("batch", nb_bits=2)


@for_all_simd_levels
class TestMultiBitRaBitQFastScan(unittest.TestCase):
    def test_code_size_formula(self):
        d = 128
        for nb_bits in range(1, 10):
            index = faiss.IndexRaBitQFastScan(d, faiss.METRIC_L2, 32, nb_bits)
            expected_size = compute_expected_code_size(d, nb_bits)
            self.assertEqual(index.code_size, expected_size)

    def test_ivf_construction(self):
        d, nlist = 128, 16
        for nb_bits in [1, 2, 4, 8]:
            quantizer = faiss.IndexFlat(d, faiss.METRIC_L2)
            index = faiss.IndexIVFRaBitQFastScan(
                quantizer, d, nlist, faiss.METRIC_L2, 32, True, nb_bits
            )
            self.assertEqual(index.rabitq.nb_bits, nb_bits)
            expected = compute_expected_code_size(d, nb_bits)
            self.assertEqual(index.code_size, expected)

    def test_ivf_basic_operations(self):
        for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
            metric_str = "L2" if metric == faiss.METRIC_L2 else "IP"
            ds = datasets.SyntheticDataset(
                128, 1000, 1000, 20, metric=metric_str
            )
            for nb_bits in [2, 4, 8]:
                with self.subTest(metric=metric, nb_bits=nb_bits):
                    quantizer = faiss.IndexFlat(ds.d, metric)
                    index = faiss.IndexIVFRaBitQFastScan(
                        quantizer, ds.d, 16, metric, 32, True, nb_bits
                    )
                    index.nprobe = 4
                    index.train(ds.get_train())
                    index.add(ds.get_database())
                    _, I = index.search(ds.get_queries(), 10)

                    self.assertTrue(np.all(I >= 0))

    def test_recall_monotonic_improvement(self):
        for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
            metric_str = "L2" if metric == faiss.METRIC_L2 else "IP"
            ds = datasets.SyntheticDataset(
                128, 500, 1000, 50, metric=metric_str
            )
            I_gt = ds.get_groundtruth(10)

            for qb in [1, 4, 8]:
                with self.subTest(metric=metric, qb=qb):
                    recalls = {}
                    for nb_bits in [1, 2, 4, 8]:
                        index = faiss.IndexRaBitQFastScan(
                            ds.d, metric, 32, nb_bits
                        )
                        index.qb = qb
                        index.train(ds.get_train())
                        index.add(ds.get_database())
                        _, I = index.search(ds.get_queries(), 10)
                        recalls[nb_bits] = faiss.eval_intersection(I, I_gt) / (
                            ds.nq * 10
                        )

                    tolerance = 0.03
                    self.assertGreaterEqual(recalls[2], recalls[1] - tolerance)
                    self.assertGreaterEqual(recalls[4], recalls[2] - tolerance)
                    self.assertGreaterEqual(recalls[8], recalls[4] - tolerance)
                    self.assertGreater(recalls[8], 0.75)

    def test_ivf_recall_improves_with_bits(self):
        d, nlist, nprobe = 128, 16, 8
        ds = datasets.SyntheticDataset(d, 1000, 1000, 50)
        I_gt = ds.get_groundtruth(10)

        recalls = {}
        for nb_bits in [1, 2, 4, 8]:
            quantizer = faiss.IndexFlat(d, faiss.METRIC_L2)
            index = faiss.IndexIVFRaBitQFastScan(
                quantizer, d, nlist, faiss.METRIC_L2, 32, True, nb_bits
            )
            index.nprobe = nprobe
            index.train(ds.get_train())
            index.add(ds.get_database())
            _, I = index.search(ds.get_queries(), 10)
            recalls[nb_bits] = faiss.eval_intersection(I, I_gt) / (ds.nq * 10)

        self.assertGreater(recalls[2], recalls[1])
        self.assertGreater(recalls[4], recalls[2])
        self.assertGreater(recalls[8], recalls[4])

    def test_comparison_vs_rabitq(self):
        ds = datasets.SyntheticDataset(128, 500, 1000, 50)
        k = 10

        index_flat_l2 = faiss.IndexFlat(ds.d, faiss.METRIC_L2)
        index_flat_l2.add(ds.get_database())
        _, I_f_l2 = index_flat_l2.search(ds.get_queries(), k)

        index_flat_ip = faiss.IndexFlat(ds.d, faiss.METRIC_INNER_PRODUCT)
        index_flat_ip.add(ds.get_database())
        _, I_f_ip = index_flat_ip.search(ds.get_queries(), k)

        for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
            I_f = I_f_l2 if metric == faiss.METRIC_L2 else I_f_ip
            for nb_bits in [2, 4, 8]:
                for qb in [1, 8]:
                    with self.subTest(metric=metric, nb_bits=nb_bits, qb=qb):
                        index_rbq = faiss.IndexRaBitQ(ds.d, metric, nb_bits)
                        index_rbq.qb = qb
                        index_rbq.train(ds.get_train())
                        index_rbq.add(ds.get_database())
                        _, I_rbq = index_rbq.search(ds.get_queries(), k)

                        index_fs = faiss.IndexRaBitQFastScan(
                            ds.d, metric, 32, nb_bits
                        )
                        index_fs.qb = qb
                        index_fs.train(ds.get_train())
                        index_fs.add(ds.get_database())
                        _, I_fs = index_fs.search(ds.get_queries(), k)

                        eval_rbq = faiss.eval_intersection(
                            I_rbq[:, :k], I_f[:, :k]
                        ) / (ds.nq * k)
                        eval_fs = faiss.eval_intersection(
                            I_fs[:, :k], I_f[:, :k]
                        ) / (ds.nq * k)

                        np.testing.assert_(abs(eval_rbq - eval_fs) < 0.05)

    def test_ivf_vs_ivfrabitq_equivalence(self):
        """SIMD path must return identical results to NONE on the same index."""
        d, nlist, nprobe, k = 128, 16, 4, 10
        ds = datasets.SyntheticDataset(d, 1000, 1000, 50)

        if not faiss.SIMDConfig.is_simd_level_available(faiss.SIMDLevel_NONE):
            self.skipTest("SIMDLevel.NONE not available")

        for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
            for nb_bits in [2, 4, 8]:
                with self.subTest(metric=metric, nb_bits=nb_bits):
                    quantizer1 = faiss.IndexFlat(d, metric)
                    index_rbq = faiss.IndexIVFRaBitQ(
                        quantizer1, d, nlist, metric, True, nb_bits
                    )
                    index_rbq.nprobe = nprobe
                    index_rbq.train(ds.get_train())
                    index_rbq.add(ds.get_database())
                    _, I_rbq_simd = index_rbq.search(ds.get_queries(), k)

                    quantizer2 = faiss.IndexFlat(d, metric)
                    index_fs = faiss.IndexIVFRaBitQFastScan(
                        quantizer2, d, nlist, metric, 32, True, nb_bits
                    )
                    index_fs.nprobe = nprobe
                    index_fs.train(ds.get_train())
                    index_fs.add(ds.get_database())
                    _, I_fs_simd = index_fs.search(ds.get_queries(), k)

                    with NoneSIMDLevel():
                        _, I_rbq_none = index_rbq.search(ds.get_queries(), k)
                        _, I_fs_none = index_fs.search(ds.get_queries(), k)

                    np.testing.assert_array_equal(I_rbq_simd, I_rbq_none)
                    np.testing.assert_array_equal(I_fs_simd, I_fs_none)

    def test_ivf_conversion_constructor_matches_direct_fastscan(self):
        d, nlist, nprobe, k = 64, 16, 4, 10
        ds = datasets.SyntheticDataset(d, 700, 300, 20)

        for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
            for nb_bits in [1, 4]:
                with self.subTest(metric=metric, nb_bits=nb_bits):
                    quantizer = faiss.IndexFlat(d, metric)
                    index_rbq = faiss.IndexIVFRaBitQ(
                        quantizer, d, nlist, metric, True, nb_bits
                    )
                    index_rbq.nprobe = nprobe
                    index_rbq.train(ds.get_train())
                    index_rbq.add(ds.get_database())

                    index_converted = faiss.IndexIVFRaBitQFastScan(
                        index_rbq, 32
                    )

                    direct_quantizer = faiss.clone_index(index_rbq.quantizer)
                    index_direct = faiss.IndexIVFRaBitQFastScan(
                        direct_quantizer, d, nlist, metric, 32, True, nb_bits
                    )
                    index_direct.nprobe = nprobe
                    index_direct.qb = index_rbq.qb
                    index_direct.centered = index_converted.centered
                    index_direct.train(ds.get_train())
                    index_direct.add(ds.get_database())

                    D_converted, I_converted = index_converted.search(
                        ds.get_queries(), k
                    )
                    D_direct, I_direct = index_direct.search(
                        ds.get_queries(), k
                    )

                    self.assertEqual(index_converted.ntotal, index_rbq.ntotal)
                    np.testing.assert_array_equal(I_converted, I_direct)
                    np.testing.assert_allclose(
                        D_converted, D_direct, rtol=0, atol=1e-5
                    )

    def test_serialization(self):
        ds = datasets.SyntheticDataset(64, 150, 200, 10)
        qb = 4

        for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
            for nb_bits in [1, 2, 4, 8, 9]:
                with self.subTest(metric=metric, nb_bits=nb_bits):
                    index1 = faiss.IndexRaBitQFastScan(
                        ds.d, metric, 32, nb_bits
                    )
                    index1.qb = qb
                    index1.train(ds.get_train())
                    index1.add(ds.get_database())
                    D1, I1 = index1.search(ds.get_queries(), 5)

                    index_bytes = faiss.serialize_index(index1)
                    index2 = faiss.deserialize_index(index_bytes)
                    index2.qb = qb
                    D2, I2 = index2.search(ds.get_queries(), 5)

                    self.assertEqual(index2.rabitq.nb_bits, nb_bits)
                    np.testing.assert_array_equal(I1, I2)
                    np.testing.assert_allclose(D1, D2, rtol=1e-5)

                    index_bytes2 = faiss.serialize_index(index2)
                    index3 = faiss.deserialize_index(index_bytes2)
                    index3.qb = qb
                    D3, I3 = index3.search(ds.get_queries(), 5)
                    np.testing.assert_array_equal(I1, I3)
                    np.testing.assert_allclose(D1, D3, rtol=1e-5)

    def test_ivf_serialization(self):
        ds = datasets.SyntheticDataset(64, 1000, 500, 20)
        nlist = 16

        for nb_bits in [1, 2, 4, 8]:
            for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
                with self.subTest(nb_bits=nb_bits, metric=metric):
                    quantizer = faiss.IndexFlat(ds.d, metric)
                    index1 = faiss.IndexIVFRaBitQFastScan(
                        quantizer, ds.d, nlist, metric, 32, True, nb_bits
                    )
                    index1.nprobe = 4
                    index1.train(ds.get_train())
                    index1.add(ds.get_database())
                    D1, I1 = index1.search(ds.get_queries(), 10)

                    index_bytes = faiss.serialize_index(index1)
                    index2 = faiss.deserialize_index(index_bytes)
                    D2, I2 = index2.search(ds.get_queries(), 10)

                    self.assertEqual(index2.rabitq.nb_bits, nb_bits)
                    np.testing.assert_array_equal(I1, I2)
                    np.testing.assert_allclose(D1, D2, rtol=1e-5)

                    index_bytes2 = faiss.serialize_index(index2)
                    index3 = faiss.deserialize_index(index_bytes2)
                    D3, I3 = index3.search(ds.get_queries(), 10)
                    np.testing.assert_array_equal(I1, I3)
                    np.testing.assert_allclose(D1, D3, rtol=1e-5)

    def test_ivf_reconstruction(self):
        d, nlist = 64, 8
        ds = datasets.SyntheticDataset(d, 500, 100, 0)

        for nb_bits in [1, 2, 4, 8]:
            for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
                with self.subTest(nb_bits=nb_bits, metric=metric):
                    quantizer = faiss.IndexFlat(d, metric)
                    index = faiss.IndexIVFRaBitQFastScan(
                        quantizer, d, nlist, metric, 32, True, nb_bits
                    )
                    index.train(ds.get_train())
                    test_vectors = ds.get_database()
                    index.add(test_vectors)

                    reconstructed = index.reconstruct_n(0, len(test_vectors))
                    errors = np.sum((test_vectors - reconstructed) ** 2, axis=1)
                    avg_error = np.mean(errors)

                    self.assertLess(avg_error, 15.0)

    def test_encode_decode_roundtrip(self):
        d, nlist = 64, 8

        for use_ivf in [False, True]:
            for nb_bits in [1, 2, 4]:
                for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
                    with self.subTest(
                        use_ivf=use_ivf, nb_bits=nb_bits, metric=metric
                    ):
                        metric_str = "L2" if metric == faiss.METRIC_L2 else "IP"
                        ds = datasets.SyntheticDataset(
                            d, 500, 100, 10, metric=metric_str
                        )

                        index = _create_fastscan_index(
                            d,
                            metric,
                            use_ivf=use_ivf,
                            nlist=nlist,
                            nb_bits=nb_bits,
                        )
                        if use_ivf:
                            index.nprobe = nlist
                        index.train(ds.get_train())
                        index.add(ds.get_database())

                        xb = ds.get_database()
                        _, I = index.search(xb, 1)

                        self_retrieval_count = sum(
                            1 for i in range(len(xb)) if I[i, 0] == i
                        )
                        self_retrieval_rate = self_retrieval_count / len(xb)
                        self.assertGreater(self_retrieval_rate, 0.5)

    def test_factory_construction(self):
        ds = datasets.SyntheticDataset(64, 1000, 200, 10)

        for prefix, cls in [
            ("", faiss.IndexRaBitQFastScan),
            ("IVF16,", faiss.IndexIVFRaBitQFastScan),
        ]:
            for suffix, nb_bits, bbs in [
                ("", 1, 32),
                ("4", 4, 32),
                ("_64", 1, 64),
                ("4_64", 4, 64),
            ]:
                factory_str = f"{prefix}RaBitQfs{suffix}"
                with self.subTest(factory_str=factory_str):
                    index = faiss.index_factory(ds.d, factory_str)
                    self.assertIsInstance(index, cls)
                    self.assertEqual(index.rabitq.nb_bits, nb_bits)
                    self.assertEqual(index.bbs, bbs)

                    index.train(ds.get_train())
                    index.add(ds.get_database())
                    _, I = index.search(ds.get_queries(), 5)
                    self.assertTrue(np.all(I >= 0))


class TestRaBitQFastScanSearchParams(unittest.TestCase):
    def test_higher_qb_improves_recall(self):
        d = 64
        nlist = 16
        nprobe = 4
        k = 10
        ds = datasets.SyntheticDataset(d, 5000, 5000, 50)

        index_flat = faiss.IndexFlatL2(d)
        index_flat.add(ds.get_database())
        _, I_gt = index_flat.search(ds.get_queries(), k)

        # Build IVF RaBitQ FastScan index with default qb=8
        quantizer = faiss.IndexFlat(d, faiss.METRIC_L2)
        index = faiss.IndexIVFRaBitQFastScan(
            quantizer, d, nlist, faiss.METRIC_L2, 32, True
        )
        index.nprobe = nprobe
        index.train(ds.get_train())
        index.add(ds.get_database())

        params_qb1 = faiss.IVFRaBitQSearchParameters()
        params_qb1.nprobe = nprobe
        params_qb1.qb = 1
        _, I_qb1 = index.search(ds.get_queries(), k, params=params_qb1)

        params_qb4 = faiss.IVFRaBitQSearchParameters()
        params_qb4.nprobe = nprobe
        params_qb4.qb = 4
        _, I_qb4 = index.search(ds.get_queries(), k, params=params_qb4)

        recall_qb1 = np.mean(
            [len(np.intersect1d(I_qb1[i], I_gt[i])) / k for i in range(ds.nq)]
        )
        recall_qb4 = np.mean(
            [len(np.intersect1d(I_qb4[i], I_gt[i])) / k for i in range(ds.nq)]
        )

        self.assertGreater(
            recall_qb4,
            recall_qb1,
            f"qb=4 recall ({recall_qb4:.3f}) should be higher "
            f"than qb=1 recall ({recall_qb1:.3f})",
        )


if __name__ == "__main__":
    unittest.main()
