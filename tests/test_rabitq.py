# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import faiss
from faiss.contrib import datasets
from faiss.contrib.evaluation import check_ref_knn_with_draws

from common_faiss_tests import for_all_simd_levels, NoneSIMDLevel


# Exercise SIMD codepaths, maintain multiple of 16.
TEST_DIM = 512 * 2 + 256 + 128 + 64 + 16  # 1488
TEST_N = 4096


# based on https://gist.github.com/mdouze/0b2386c31d7fb8b20ae04f3fcbbf4d9d
class ReferenceRabitQ:
    """Exact translation of the paper
    https://dl.acm.org/doi/pdf/10.1145/3654970
    This is both a quantizer and serves to store the codes
    """

    def __init__(self, d, Bq=4):
        self.d = d
        self.Bq = Bq

    def train(self, xtrain, P):
        self.centroid = xtrain.mean(0)
        self.P = P

    def rotation(self, x):
        return x @ self.P

    def inv_rotation(self, x):
        return x @ self.P.T

    def add(self, Or):
        Orc = Or - self.centroid
        self.O_norms = np.sqrt((Orc**2).sum(1))
        O = Orc / self.O_norms[:, None]

        # 3.1.3
        self.Xbarb = (self.inv_rotation(Orc) > 0).astype("int8")  # 0, 1
        # here the encoded vectors are stored as an int array for simplicity
        # but in the real code it would be as a packed uint8 array
        # reconstruct to compute <o, obar>
        Obar = self.rotation((2 * self.Xbarb - 1) / np.sqrt(self.d))
        self.o_Obar = (O * Obar).sum(1)  # store dot products

    def distances(self, Qr):
        """compute distance estimates for the queries to the stored vectors"""
        d = self.d
        Bq = self.Bq

        # preproc Qr
        Qrc = Qr - self.centroid
        Qrc_norms = np.sqrt((Qrc**2).sum(1))[:, None]
        Q = Qrc
        Qprime = self.inv_rotation(Q)

        # quantize queries to Bq bits
        mins, maxes = Qprime.min(axis=1)[:, None], Qprime.max(axis=1)[:, None]
        Delta = (maxes - mins) / (2**Bq - 1)

        # we'll use a non-randomized for the comparison purposes
        qbar = np.round((Qprime - mins) / Delta)
        # in the real implementation, this would be re-ordered
        # in least-to most-significant bit
        # dot product matrix, integers -- this is the expensive operation
        dp = (qbar[:, None, :] * self.Xbarb[None, :, :]).sum(2)

        # roll back the normalizations to get <xbar, qbar> (eq 19-20)
        sum_X = self.Xbarb.sum(1)
        sum_Q = qbar.sum(1)[:, None]
        sD = np.sqrt(d)
        xbar_qbar = 2 * Delta / sD * dp
        xbar_qbar += 2 * mins / sD * sum_X
        xbar_qbar -= Delta / sD * sum_Q
        xbar_qbar -= sD * mins

        # <xbar, qbar> is close to <xbar, q'> thm 3.3
        # <xbar, q'> = <obar, q>  eq 17

        # <obar, q> / <obar, o> estimates <q, o> (thm 3.2)
        q_o = xbar_qbar / self.o_Obar

        # eq 1-2 to de-normalize and get distances
        dis2_q_o = self.O_norms**2 + Qrc_norms**2 - 2 * self.O_norms * q_o

        return dis2_q_o


@for_all_simd_levels
class TestRaBitQ(unittest.TestCase):
    def do_comparison_vs_pq_test(self, metric_type=faiss.METRIC_L2):
        ds = datasets.SyntheticDataset(TEST_DIM, TEST_N, TEST_N, 100)
        k = 10

        index_flat = faiss.IndexFlat(ds.d, metric_type)
        index_flat.add(ds.get_database())
        _, I_f = index_flat.search(ds.get_queries(), k)

        def eval_I(I):
            return faiss.eval_intersection(I, I_f) / I_f.ravel().shape[0]

        for random_rotate in [False, True]:

            # PQ{D/4}x4fs, also 1 bit per query dimension
            index_pq = faiss.IndexPQFastScan(ds.d, ds.d // 4, 4, metric_type)
            # Share a single quantizer (much faster, minimal recall change)
            index_pq.pq.train_type = faiss.ProductQuantizer.Train_shared
            if random_rotate:
                rrot = faiss.RandomRotationMatrix(ds.d, ds.d)
                rrot.init(123)

                index_pq = faiss.IndexPreTransform(rrot, index_pq)
            index_pq.train(ds.get_train())
            index_pq.add(ds.get_database())

            _, I_pq = index_pq.search(ds.get_queries(), k)
            loss_pq = 1 - eval_I(I_pq)
            np.testing.assert_(loss_pq < 0.4, f"{loss_pq}")

            index_rbq = faiss.IndexRaBitQ(ds.d, metric_type)
            if random_rotate:
                rrot = faiss.RandomRotationMatrix(ds.d, ds.d)
                rrot.init(123)

                index_rbq = faiss.IndexPreTransform(rrot, index_rbq)
            index_rbq.train(ds.get_train())
            index_rbq.add(ds.get_database())

            def test(params, ratio_threshold, index, rotate, loss_pq_val):
                _, I_rbq = index.search(ds.get_queries(), k, params=params)

                loss_rbq = 1 - eval_I(I_rbq)
                np.testing.assert_(loss_rbq < loss_pq_val * ratio_threshold)

            for qb in [1, 2, 3, 4, 8]:
                for centered in [False, True]:
                    params = faiss.RaBitQSearchParameters(
                        qb=qb, centered=centered
                    )
                    ratio_threshold = 2.1 ** (1 / qb)
                    test(
                        params,
                        ratio_threshold,
                        index_rbq,
                        random_rotate,
                        loss_pq,
                    )

    def test_comparison_vs_pq_L2(self):
        self.do_comparison_vs_pq_test(faiss.METRIC_L2)

    def test_comparison_vs_pq_IP(self):
        self.do_comparison_vs_pq_test(faiss.METRIC_INNER_PRODUCT)

    def test_comparison_vs_ref_L2(self):
        ds = datasets.SyntheticDataset(TEST_DIM, TEST_N, TEST_N, 1)

        ref_rbq = ReferenceRabitQ(ds.d, Bq=8)
        ref_rbq.train(ds.get_train(), np.identity(ds.d))
        ref_rbq.add(ds.get_database())

        index_rbq = faiss.IndexRaBitQ(ds.d, faiss.METRIC_L2)
        index_rbq.qb = 8
        index_rbq.train(ds.get_train())
        index_rbq.add(ds.get_database())

        ref_dis = ref_rbq.distances(ds.get_queries())
        mean_dist = ref_dis.mean()

        dc = index_rbq.get_distance_computer()
        xq = ds.get_queries()

        dc.set_query(faiss.swig_ptr(xq[0]))
        for j in range(ds.nb):
            upd_dis = dc(j)
            np.testing.assert_(
                abs(ref_dis[0][j] - upd_dis) < mean_dist * 0.00001,
                f"{j} {ref_dis[0][j]} {upd_dis}",
            )

    def test_self_match_nonneg_distance(self):
        d = 64
        rs = np.random.RandomState(10996)
        xb = rs.randn(200, d).astype("float32")

        index = faiss.IndexRaBitQ(d, faiss.METRIC_L2)
        index.train(xb)
        index.add(xb)

        D, I = index.search(xb[:5], 5)
        np.testing.assert_(np.all(D >= 0), f"negative D: {D[D < 0]}")
        for i in range(5):
            self.assertIn(i, I[i])

    def test_serde_rabitq(self):
        do_test_serde("RaBitQ")


@for_all_simd_levels
class TestIVFRaBitQ(unittest.TestCase):
    def do_comparison_vs_pq_test(self, metric_type=faiss.METRIC_L2):
        nlist = 64
        nprobe = 8
        nq = 1000
        ds = datasets.SyntheticDataset(TEST_DIM, TEST_N, TEST_N, nq)
        k = 10

        d = ds.d
        xb = ds.get_database()
        xt = ds.get_train()
        xq = ds.get_queries()

        quantizer = faiss.IndexFlat(d, metric_type)
        index_flat = faiss.IndexIVFFlat(quantizer, d, nlist, metric_type)
        index_flat.train(xt)
        index_flat.add(xb)
        _, I_f = index_flat.search(
            xq, k, params=faiss.IVFSearchParameters(nprobe=nprobe)
        )

        def eval_I(I):
            return faiss.eval_intersection(I, I_f) / I_f.ravel().shape[0]

        for random_rotate in [False, True]:
            quantizer = faiss.IndexFlat(d, metric_type)
            index_rbq = faiss.IndexIVFRaBitQ(quantizer, d, nlist, metric_type)
            if random_rotate:
                rrot = faiss.RandomRotationMatrix(d, d)
                rrot.init(123)

                index_rbq = faiss.IndexPreTransform(rrot, index_rbq)
            index_rbq.train(xt)
            index_rbq.add(xb)

            # PQ{D/4}x4fs, also 1 bit per query dimension,
            # reusing quantizer from index_rbq.
            index_pq = faiss.IndexIVFPQFastScan(
                quantizer, d, nlist, d // 4, 4, metric_type
            )
            # Share a single quantizer (much faster, minimal recall change)
            index_pq.pq.train_type = faiss.ProductQuantizer.Train_shared
            if random_rotate:
                rrot = faiss.RandomRotationMatrix(d, d)
                rrot.init(123)

                index_pq = faiss.IndexPreTransform(rrot, index_pq)
            index_pq.train(xt)
            index_pq.add(xb)

            _, I_pq = index_pq.search(
                xq, k, params=faiss.IVFPQSearchParameters(nprobe=nprobe)
            )
            loss_pq = 1 - eval_I(I_pq)
            np.testing.assert_(loss_pq < 0.25, f"{loss_pq}")

            def test(params, ratio_threshold, index, rotate, loss_pq_val):
                _, I_rbq = index.search(xq, k, params=params)

                loss_rbq = 1 - eval_I(I_rbq)
                np.testing.assert_(loss_rbq < loss_pq_val * ratio_threshold)

            for qb in [1, 2, 3, 4, 8]:
                for centered in [False, True]:
                    params = faiss.IVFRaBitQSearchParameters(
                        nprobe=nprobe, qb=qb, centered=centered
                    )
                    ratio_threshold = 2 ** (1 / qb)
                    test(
                        params,
                        ratio_threshold,
                        index_rbq,
                        random_rotate,
                        loss_pq,
                    )

    def test_comparison_vs_pq_L2(self):
        self.do_comparison_vs_pq_test(faiss.METRIC_L2)

    def test_comparison_vs_pq_IP(self):
        self.do_comparison_vs_pq_test(faiss.METRIC_INNER_PRODUCT)

    def test_comparison_vs_ref_L2(self):
        # SIMD path must return identical results to NONE on the same index.
        ds = datasets.SyntheticDataset(TEST_DIM, TEST_N, TEST_N, 100)

        k = 10
        nlist = 200

        index_flat = faiss.IndexFlat(ds.d, faiss.METRIC_L2)
        index_rbq = faiss.IndexIVFRaBitQ(
            index_flat, ds.d, nlist, faiss.METRIC_L2
        )
        index_rbq.qb = 4
        index_rbq.train(ds.get_train())
        index_rbq.add(ds.get_database())

        for nprobe in 1, 4, 16:
            params = faiss.IVFRaBitQSearchParameters()
            params.qb = index_rbq.qb
            params.nprobe = nprobe
            _, I_simd, _ = faiss.search_with_parameters(
                index_rbq, ds.get_queries(), k, params, output_stats=True
            )

            with NoneSIMDLevel():
                _, I_none, _ = faiss.search_with_parameters(
                    index_rbq, ds.get_queries(), k, params, output_stats=True
                )

            np.testing.assert_array_equal(I_simd, I_none)

    def test_serde_ivfrabitq(self):
        do_test_serde("IVF16,RaBitQ")


@for_all_simd_levels
class TestRaBitQuantizerEncodeDecode(unittest.TestCase):
    def test_encode_decode_L2(self):
        # rabitq must precisely reconstruct a vector,
        #   which consists of +A and -A values
        d = 16
        seed = 123
        rs = np.random.RandomState(seed)

        ampl = 100
        n = 10
        vec = (2 * rs.randint(0, 2, d * n) - 1).astype(np.float32) * ampl
        vec = np.reshape(vec, (n, d))

        quantizer = faiss.RaBitQuantizer(d, faiss.METRIC_L2)
        vec_q = quantizer.compute_codes(vec)
        vec_rec = quantizer.decode(vec_q)
        np.testing.assert_equal(vec, vec_rec)


def create_index_rabitq_with_rotation(d, metric, nb_bits, qb=0):
    index_rbq = faiss.IndexRaBitQ(d, metric, nb_bits)
    index_rbq.qb = qb
    rrot = faiss.RandomRotationMatrix(d, d)
    rrot.init(123)
    return faiss.IndexPreTransform(rrot, index_rbq)


def create_index_ivf_rabitq_with_rotation(d, metric, nb_bits, qb=0):
    quantizer = faiss.IndexFlat(d, metric)
    index_rbq = faiss.IndexIVFRaBitQ(quantizer, d, 16, metric, True, nb_bits)
    index_rbq.qb = qb
    index_rbq.nprobe = 4
    rrot = faiss.RandomRotationMatrix(d, d)
    rrot.init(123)
    return faiss.IndexPreTransform(rrot, index_rbq)


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


def do_test_serde(description):
    ds = datasets.SyntheticDataset(32, 1000, 100, 20)

    index = faiss.index_factory(ds.d, description)
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


@for_all_simd_levels
class TestMultiBitRaBitQ(unittest.TestCase):
    def test_invalid_nb_bits(self):
        with self.assertRaises(RuntimeError):
            faiss.IndexRaBitQ(128, faiss.METRIC_L2, 0)
        with self.assertRaises(RuntimeError):
            faiss.IndexRaBitQ(128, faiss.METRIC_L2, 10)

    def test_code_size_formula(self):
        d = 128
        for nb_bits in range(1, 10):
            index = faiss.IndexRaBitQ(d, faiss.METRIC_L2, nb_bits)
            expected_size = compute_expected_code_size(d, nb_bits)
            self.assertEqual(index.code_size, expected_size)

    def test_recall_monotonic_improvement(self):
        for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
            metric_str = "L2" if metric == faiss.METRIC_L2 else "IP"
            ds = datasets.SyntheticDataset(
                128, 500, 1000, 50, metric=metric_str
            )
            I_gt = ds.get_groundtruth(10)

            for qb in [0, 4, 8]:
                with self.subTest(metric=metric, qb=qb):
                    recalls = {}
                    for nb_bits in [1, 2, 4, 8]:
                        index = create_index_rabitq_with_rotation(
                            ds.d, metric, nb_bits, qb=qb
                        )
                        index.train(ds.get_train())
                        index.add(ds.get_database())
                        _, I = index.search(ds.get_queries(), 10)
                        recalls[nb_bits] = faiss.eval_intersection(I, I_gt) / (
                            ds.nq * 10
                        )
                        self.assertGreater(recalls[nb_bits], 0.10)

                    tolerance = 0.03
                    self.assertGreaterEqual(recalls[2], recalls[1] - tolerance)
                    self.assertGreaterEqual(recalls[4], recalls[2] - tolerance)
                    self.assertGreaterEqual(recalls[8], recalls[4] - tolerance)
                    self.assertGreater(recalls[8], 0.75)

    def test_serialization(self):
        ds = datasets.SyntheticDataset(64, 150, 200, 10)
        qb = 4

        for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
            for nb_bits in [1, 9]:
                with self.subTest(metric=metric, nb_bits=nb_bits):
                    index1 = create_index_rabitq_with_rotation(
                        ds.d, metric, nb_bits, qb=qb
                    )
                    index1.train(ds.get_train())
                    index1.add(ds.get_database())
                    D1, I1 = index1.search(ds.get_queries(), 5)

                    index_bytes = faiss.serialize_index(index1)
                    index2 = faiss.deserialize_index(index_bytes)

                    params = faiss.RaBitQSearchParameters()
                    params.qb = qb
                    params.centered = False
                    D2, I2 = index2.search(ds.get_queries(), 5, params=params)

                    np.testing.assert_array_equal(I1, I2)
                    np.testing.assert_allclose(D1, D2, rtol=1e-5)

                    index_bytes2 = faiss.serialize_index(index2)
                    index3 = faiss.deserialize_index(index_bytes2)
                    D3, I3 = index3.search(ds.get_queries(), 5, params=params)
                    np.testing.assert_array_equal(I1, I3)
                    np.testing.assert_allclose(D1, D3, rtol=1e-5)

    def test_ivf_basic_operations(self):
        ds = datasets.SyntheticDataset(128, 300, 500, 20)

        for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
            for nb_bits in [1, 2, 4, 8]:
                for qb in [0, 4, 8]:
                    with self.subTest(metric=metric, nb_bits=nb_bits, qb=qb):
                        index = create_index_ivf_rabitq_with_rotation(
                            ds.d, metric, nb_bits, qb=qb
                        )
                        index.train(ds.get_train())
                        index.add(ds.get_database())
                        D, I = index.search(ds.get_queries(), 10)

                        self.assertTrue(np.all(I >= 0))
                        self.assertTrue(np.all(np.isfinite(D)))

    def test_ivf_nprobe_improves_recall(self):
        for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
            metric_str = "L2" if metric == faiss.METRIC_L2 else "IP"
            ds = datasets.SyntheticDataset(
                128, 500, 1000, 50, metric=metric_str
            )
            I_gt = ds.get_groundtruth(10)

            for nb_bits in [1, 4, 8]:
                with self.subTest(metric=metric, nb_bits=nb_bits):
                    quantizer = faiss.IndexFlat(ds.d, metric)
                    index_rbq = faiss.IndexIVFRaBitQ(
                        quantizer, ds.d, 32, metric, True, nb_bits
                    )
                    rrot = faiss.RandomRotationMatrix(ds.d, ds.d)
                    rrot.init(123)
                    index = faiss.IndexPreTransform(rrot, index_rbq)
                    index.train(ds.get_train())
                    index.add(ds.get_database())

                    recalls = {}
                    for nprobe in [1, 2, 4, 8]:
                        index_rbq.nprobe = nprobe
                        _, I = index.search(ds.get_queries(), 10)
                        recalls[nprobe] = faiss.eval_intersection(I, I_gt) / (
                            ds.nq * 10
                        )

                    self.assertGreaterEqual(recalls[2], recalls[1])
                    self.assertGreaterEqual(recalls[4], recalls[2])
                    self.assertGreaterEqual(recalls[8], recalls[4])

    def test_ivf_serialization(self):
        ds = datasets.SyntheticDataset(64, 150, 200, 10)
        qb = 4

        for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
            for nb_bits in [1, 9]:
                with self.subTest(metric=metric, nb_bits=nb_bits):
                    index1 = create_index_ivf_rabitq_with_rotation(
                        ds.d, metric, nb_bits, qb=qb
                    )
                    index1.train(ds.get_train())
                    index1.add(ds.get_database())
                    D1, I1 = index1.search(ds.get_queries(), 5)

                    index_bytes = faiss.serialize_index(index1)
                    index2 = faiss.deserialize_index(index_bytes)

                    params = faiss.IVFRaBitQSearchParameters()
                    params.qb = qb
                    params.centered = False
                    params.nprobe = 4
                    D2, I2 = index2.search(ds.get_queries(), 5, params=params)

                    np.testing.assert_array_equal(I1, I2)
                    np.testing.assert_allclose(D1, D2, rtol=1e-5)

                    index_bytes2 = faiss.serialize_index(index2)
                    index3 = faiss.deserialize_index(index_bytes2)
                    D3, I3 = index3.search(ds.get_queries(), 5, params=params)
                    np.testing.assert_array_equal(I1, I3)
                    np.testing.assert_allclose(D1, D3, rtol=1e-5)

    def test_query_quantization_levels(self):
        ds = datasets.SyntheticDataset(128, 300, 500, 20)

        for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
            for nb_bits in [1, 2, 4, 8]:
                with self.subTest(metric=metric, nb_bits=nb_bits):
                    index = create_index_rabitq_with_rotation(
                        ds.d, metric, nb_bits, qb=0
                    )
                    index.train(ds.get_train())
                    index.add(ds.get_database())

                    for qb in [0, 1, 2, 4, 6, 8]:
                        params = faiss.RaBitQSearchParameters()
                        params.qb = qb
                        params.centered = False
                        D, I = index.search(ds.get_queries(), 10, params=params)

                        self.assertTrue(np.all(I >= 0))
                        self.assertTrue(np.all(np.isfinite(D)))

    def test_factory(self):
        for prefix, cls in [
            ("", faiss.IndexRaBitQ),
            ("IVF16,", faiss.IndexIVFRaBitQ),
        ]:
            for nb_bits in [1, 2, 4, 8]:
                suffix = str(nb_bits) if nb_bits > 1 else ""
                index = faiss.index_factory(128, f"{prefix}RaBitQ{suffix}")
                self.assertIsInstance(index, cls)
                self.assertEqual(index.rabitq.nb_bits, nb_bits)

    def test_degenerate_centroid_distance(self):
        """A doc equal to its centroid hits the zero-residual early return of
        quantize_ex_bits (f_rescale_ex must be 0, f_add_ex metric-dependent).
        """
        d = 128
        nlist = 16
        rs = np.random.RandomState(42)
        xt = rs.randn(2000, d).astype("float32")
        xt /= np.linalg.norm(xt, axis=1, keepdims=True)

        query = rs.randn(1, d).astype("float32")
        query /= np.linalg.norm(query)

        for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
            for nb_bits in [2, 3, 4]:
                with self.subTest(metric=metric, nb_bits=nb_bits):
                    quantizer = faiss.IndexFlat(d, metric)
                    index = faiss.IndexIVFRaBitQ(
                        quantizer, d, nlist, metric, True, nb_bits
                    )
                    index.train(xt)
                    index.nprobe = nlist
                    centroid = index.quantizer.reconstruct(0).reshape(1, d)
                    index.add(centroid)

                    D, I = index.search(query, 1)

                    if metric == faiss.METRIC_INNER_PRODUCT:
                        true_dist = (query @ centroid.T).item()
                    else:
                        true_dist = float(np.sum((query - centroid) ** 2))

                    np.testing.assert_allclose(
                        D[0, 0],
                        true_dist,
                        atol=0.15,
                        err_msg=f"nb_bits={nb_bits}",
                    )

    def test_zero_residual_coordinate(self):
        """Test that an exactly-zero residual encodes like a negative one.

        The sign bit uses `residual > 0`, so a zero residual must flip its
        ex-bits too, or it decodes to the largest magnitude instead of the
        smallest.
        """
        ds = datasets.SyntheticDataset(64, 2000, 1, 1)
        zero_dims = np.arange(0, ds.d, 4)

        for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
            for nb_bits in [2, 4, 8, 9]:
                with self.subTest(metric=metric, nb_bits=nb_bits):
                    index = faiss.IndexRaBitQ(ds.d, metric, nb_bits)
                    index.train(ds.get_train())

                    # Anchor on the center: the residual must be exactly
                    # zero, not merely small.
                    center = faiss.vector_to_array(index.center)
                    x = ds.get_database().copy()
                    x[0, zero_dims] = center[zero_dims]

                    x_neg = x.copy()
                    x_neg[0, zero_dims] = np.nextafter(
                        x[0, zero_dims], np.float32(-np.inf)
                    )

                    index.add(x)
                    Dref, Iref = index.search(ds.get_queries(), 1)
                    index.reset()
                    index.add(x_neg)
                    Dnew, Inew = index.search(ds.get_queries(), 1)

                    check_ref_knn_with_draws(Dref, Iref, Dnew, Inew)


if __name__ == "__main__":
    unittest.main()
