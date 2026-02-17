# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import faiss
from faiss.contrib import datasets


def random_rotation(d, seed=123):
    rs = np.random.RandomState(seed)
    Q, _ = np.linalg.qr(rs.randn(d, d))
    return Q


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
        # centering & normalization
        Orc = Or - self.centroid
        self.O_norms = np.sqrt((Orc**2).sum(1))  # need to store the norms
        O = Orc / self.O_norms[:, None]

        # 3.1.3
        self.Xbarb = (self.inv_rotation(Orc) > 0).astype("int8")  # 0, 1
        # here the encoded vectors are stored as an int array for simplicity
        # but in the real code it would be as a packed uint8 array
        # self.Xbarb = np.packbits(self.inv_rotation(Orc) > 0, axis=1)
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

        # article mentioned a randomized variant
        # qbar = np.floor((Qprime - mins) / Delta + rs.rand(nq, d))

        # we'll use a non-randomized for the comparison purposes
        qbar = np.round((Qprime - mins) / Delta)
        # in the real implementation, this would be re-ordered
        # in least-to most-significant bit
        # dot product matrix, integers -- this is the expensive operation
        dp = (qbar[:, None, :] * self.Xbarb[None, :, :]).sum(2)

        # the operations below roll back the normalizations to get the distance
        # estimates. it is likely that they could be merged
        # or some of them could be left out because we are interested only
        # in top-k compute <xbar, qbar> (eq 19-20)
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


class ReferenceIVFRabitQ:
    """straightforward IVF implementation"""

    def __init__(self, d, nlist, Bq=4):
        self.d = d
        self.nlist = nlist
        self.invlists = [ReferenceRabitQ(d, Bq) for _ in range(nlist)]
        self.quantizer = None
        self.nprobe = 1

    def train(self, xtrain, P):
        if self.quantizer is None:
            km = faiss.Kmeans(self.d, self.nlist, niter=10)
            km.train(xtrain)
            centroids = km.centroids
            self.quantizer = faiss.IndexFlatL2(self.d)
            self.quantizer.add(centroids)
        else:
            centroids = self.quantizer.reconstruct_n()
        # Override the RabitQ train() to use a common random rotation
        #  and force centroids from the coarse quantizer
        for list_no, rq in enumerate(self.invlists):
            rq.centroid = centroids[list_no]
            rq.P = P

    def add(self, x):
        _, keys = self.quantizer.search(x, 1)
        keys = keys.ravel()
        n_per_invlist = np.bincount(keys, minlength=self.nlist)
        order = np.argsort(keys)
        i0 = 0
        for list_no, rab in enumerate(self.invlists):
            i1 = i0 + n_per_invlist[list_no]
            rab.list_size = i1 - i0
            if i1 > i0:
                ids = order[i0:i1]
                rab.ids = ids
                rab.add(x[ids])
            i0 = i1

    def search(self, x, k):
        nq = len(x)
        nprobe = self.nprobe
        D = np.zeros((nq, k), dtype="float32")
        I = np.zeros((nq, k), dtype=int)
        D[:] = np.nan
        I[:] = -1
        _, Ic = self.quantizer.search(x, nprobe)

        for qno, xq in enumerate(x):
            # naive top-k implemetation with a full sort
            q_dis = []
            q_ids = []
            for probe in range(nprobe):
                rab = self.invlists[Ic[qno, probe]]
                if rab.list_size == 0:
                    continue
                # we cannot exploit the batch version
                # of the queries (in this form)
                dis = rab.distances(xq[None, :])
                q_ids.append(rab.ids)
                q_dis.append(dis.ravel())
            q_dis = np.hstack(q_dis)
            q_ids = np.hstack(q_ids)
            o = q_dis.argsort()
            kq = min(k, len(q_dis))
            D[qno, :kq] = q_dis[o[:kq]]
            I[qno, :kq] = q_ids[o[:kq]]
        return D, I


class TestRaBitQ(unittest.TestCase):
    def do_comparison_vs_pq_test(self, metric_type=faiss.METRIC_L2):
        ds = datasets.SyntheticDataset(TEST_DIM, TEST_N, TEST_N, 100)
        k = 10

        index_flat = faiss.IndexFlat(ds.d, metric_type)
        index_flat.train(ds.get_train())
        index_flat.add(ds.get_database())
        _, I_f = index_flat.search(ds.get_queries(), k)

        def eval_I(I):
            return faiss.eval_intersection(I, I_f) / I_f.ravel().shape[0]

        print()

        for random_rotate in [False, True]:

            # PQ{D/4}x4fs, also 1 bit per query dimension
            index_pq = faiss.IndexPQFastScan(ds.d, ds.d // 4, 4, metric_type)
            # Share a single quantizer (much faster, minimal recall change)
            index_pq.pq.train_type = faiss.ProductQuantizer.Train_shared
            if random_rotate:
                # wrap with random rotations
                rrot = faiss.RandomRotationMatrix(ds.d, ds.d)
                rrot.init(123)

                index_pq = faiss.IndexPreTransform(rrot, index_pq)
            index_pq.train(ds.get_train())
            index_pq.add(ds.get_database())

            _, I_pq = index_pq.search(ds.get_queries(), k)
            loss_pq = 1 - eval_I(I_pq)
            print(f"{random_rotate=:1}, {loss_pq=:5.3f}")
            np.testing.assert_(loss_pq < 0.4, f"{loss_pq}")

            index_rbq = faiss.IndexRaBitQ(ds.d, metric_type)
            if random_rotate:
                # wrap with random rotations
                rrot = faiss.RandomRotationMatrix(ds.d, ds.d)
                rrot.init(123)

                index_rbq = faiss.IndexPreTransform(rrot, index_rbq)
            index_rbq.train(ds.get_train())
            index_rbq.add(ds.get_database())

            def test(params, ratio_threshold, index, rotate, loss_pq_val):
                _, I_rbq = index.search(ds.get_queries(), k, params=params)

                # ensure that RaBitQ and PQ are relatively close
                loss_rbq = 1 - eval_I(I_rbq)
                print(
                    f"{rotate=:1}, {params.qb=}, {params.centered=:1}: "
                    f"{loss_rbq=:5.3f} = loss_pq * "
                    f"{loss_rbq / loss_pq_val:5.3f}"
                    f" < {ratio_threshold=:.2f}"
                )

                np.testing.assert_(loss_rbq < loss_pq_val * ratio_threshold)

            for qb in [1, 2, 3, 4, 8]:
                print()
                for centered in [False, True]:
                    params = faiss.RaBitQSearchParameters(
                        qb=qb, centered=centered
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

    def test_comparison_vs_ref_L2_rrot(self, rrot_seed=123):
        ds = datasets.SyntheticDataset(TEST_DIM, TEST_N, TEST_N, 1)

        ref_rbq = ReferenceRabitQ(ds.d, Bq=8)
        ref_rbq.train(ds.get_train(), random_rotation(ds.d, rrot_seed))
        ref_rbq.add(ds.get_database())

        index_rbq = faiss.IndexRaBitQ(ds.d, faiss.METRIC_L2)
        index_rbq.qb = 8

        # wrap with random rotations
        rrot = faiss.RandomRotationMatrix(ds.d, ds.d)
        rrot.init(rrot_seed)

        index_cand = faiss.IndexPreTransform(rrot, index_rbq)
        index_cand.train(ds.get_train())
        index_cand.add(ds.get_database())

        ref_dis = ref_rbq.distances(ds.get_queries())

        dc = index_cand.get_distance_computer()
        xq = ds.get_queries()

        # ensure that the correlation coefficient is very high
        dc_dist = [0] * ds.nb

        dc.set_query(faiss.swig_ptr(xq[0]))
        for j in range(ds.nb):
            dc_dist[j] = dc(j)

        corr = np.corrcoef(dc_dist, ref_dis[0])[0, 1]
        print(corr)
        np.testing.assert_(corr > 0.9)

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

    def test_serde_rabitq(self):
        do_test_serde("RaBitQ")


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

        print()

        for random_rotate in [False, True]:
            quantizer = faiss.IndexFlat(d, metric_type)
            index_rbq = faiss.IndexIVFRaBitQ(quantizer, d, nlist, metric_type)
            if random_rotate:
                # wrap with random rotations
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
                # wrap with random rotations
                rrot = faiss.RandomRotationMatrix(d, d)
                rrot.init(123)

                index_pq = faiss.IndexPreTransform(rrot, index_pq)
            index_pq.train(xt)
            index_pq.add(xb)

            _, I_pq = index_pq.search(
                xq, k, params=faiss.IVFPQSearchParameters(nprobe=nprobe)
            )
            loss_pq = 1 - eval_I(I_pq)

            print(f"{random_rotate=:1}, {loss_pq=:5.3f}")
            np.testing.assert_(loss_pq < 0.25, f"{loss_pq}")

            def test(params, ratio_threshold, index, rotate, loss_pq_val):
                _, I_rbq = index.search(xq, k, params=params)

                # ensure that RaBitQ and PQ are relatively close
                loss_rbq = 1 - eval_I(I_rbq)
                print(
                    f"{rotate=:1}, {params.qb=}, {params.centered=:1}: "
                    f"{loss_rbq=:5.3f} = loss_pq * "
                    f"{loss_rbq / loss_pq_val:5.3f}"
                    f" < {ratio_threshold=:.2f}"
                )

                np.testing.assert_(loss_rbq < loss_pq_val * ratio_threshold)

            for qb in [1, 2, 3, 4, 8]:
                print()
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
        ds = datasets.SyntheticDataset(TEST_DIM, TEST_N, TEST_N, 100)

        k = 10
        nlist = 200
        ref_rbq = ReferenceIVFRabitQ(ds.d, nlist, Bq=4)
        ref_rbq.train(ds.get_train(), np.identity(ds.d))
        ref_rbq.add(ds.get_database())

        index_flat = faiss.IndexFlat(ds.d, faiss.METRIC_L2)
        index_rbq = faiss.IndexIVFRaBitQ(
            index_flat, ds.d, nlist, faiss.METRIC_L2
        )
        index_rbq.qb = 4
        index_rbq.train(ds.get_train())
        index_rbq.add(ds.get_database())

        for nprobe in 1, 4, 16:
            ref_rbq.nprobe = nprobe
            _, Iref = ref_rbq.search(ds.get_queries(), k)
            r_ref_k = faiss.eval_intersection(
                Iref[:, :k], ds.get_groundtruth()[:, :k]
            ) / (ds.nq * k)
            print(f"{nprobe=} k-recall@10={r_ref_k}")

            params = faiss.IVFRaBitQSearchParameters()
            params.qb = index_rbq.qb
            params.nprobe = nprobe
            _, Inew, _ = faiss.search_with_parameters(
                index_rbq, ds.get_queries(), k, params, output_stats=True
            )
            r_new_k = faiss.eval_intersection(
                Inew[:, :k], ds.get_groundtruth()[:, :k]
            ) / (ds.nq * k)
            print(f"{nprobe=} k-recall@10={r_new_k}")

            np.testing.assert_almost_equal(r_ref_k, r_new_k, 3)

    def test_comparison_vs_ref_L2_rrot(self):
        ds = datasets.SyntheticDataset(128, 4096, 4096, 100)

        k = 10
        nlist = 200
        rrot_seed = 123

        ref_rbq = ReferenceIVFRabitQ(ds.d, nlist, Bq=4)
        ref_rbq.train(ds.get_train(), random_rotation(ds.d, rrot_seed))
        ref_rbq.add(ds.get_database())

        index_flat = faiss.IndexFlat(ds.d, faiss.METRIC_L2)
        index_rbq = faiss.IndexIVFRaBitQ(
            index_flat, ds.d, nlist, faiss.METRIC_L2
        )
        index_rbq.qb = 4

        # wrap with random rotations
        rrot = faiss.RandomRotationMatrix(ds.d, ds.d)
        rrot.init(rrot_seed)

        index_cand = faiss.IndexPreTransform(rrot, index_rbq)
        index_cand.train(ds.get_train())
        index_cand.add(ds.get_database())

        for nprobe in 1, 4, 16:
            ref_rbq.nprobe = nprobe
            _, Iref = ref_rbq.search(ds.get_queries(), k)
            r_ref_k = faiss.eval_intersection(
                Iref[:, :k], ds.get_groundtruth()[:, :k]
            ) / (ds.nq * k)
            print(f"{nprobe=} k-recall@10={r_ref_k}")

            params = faiss.IVFRaBitQSearchParameters()
            params.qb = index_rbq.qb
            params.nprobe = nprobe
            _, Inew, _ = faiss.search_with_parameters(
                index_cand, ds.get_queries(), k, params, output_stats=True
            )
            r_new_k = faiss.eval_intersection(
                Inew[:, :k], ds.get_groundtruth()[:, :k]
            ) / (ds.nq * k)
            print(f"{nprobe=} k-recall@10={r_new_k}")

            np.testing.assert_almost_equal(r_ref_k, r_new_k, 2)

    def test_serde_ivfrabitq(self):
        do_test_serde("IVF16,RaBitQ")


class TestRaBitQuantizerEncodeDecode(unittest.TestCase):
    def do_test_encode_decode(self, d, metric):
        # rabitq must precisely reconstruct a vector,
        #   which consists of +A and -A values

        seed = 123
        rs = np.random.RandomState(seed)

        ampl = 100
        n = 10
        vec = (2 * rs.randint(0, 2, d * n) - 1).astype(np.float32) * ampl
        vec = np.reshape(vec, (n, d))

        quantizer = faiss.RaBitQuantizer(d, metric)

        # encode and decode
        vec_q = quantizer.compute_codes(vec)
        vec_rec = quantizer.decode(vec_q)

        # verify
        np.testing.assert_equal(vec, vec_rec)

    def test_encode_decode_L2(self):
        self.do_test_encode_decode(16, faiss.METRIC_L2)

    def test_encode_decode_IP(self):
        self.do_test_encode_decode(16, faiss.METRIC_INNER_PRODUCT)


# ==============================================================================
# Multi-bit RaBitQ Test Suite
# ==============================================================================


def create_index_rabitq_with_rotation(d, metric, nb_bits, qb=0, rrot_seed=123):
    """Helper: Create IndexRaBitQ with random rotation."""
    index_rbq = faiss.IndexRaBitQ(d, metric, nb_bits)
    index_rbq.qb = qb
    rrot = faiss.RandomRotationMatrix(d, d)
    rrot.init(rrot_seed)
    return faiss.IndexPreTransform(rrot, index_rbq)


def create_index_ivf_rabitq_with_rotation(
    d, metric, nb_bits, nlist=16, qb=0, nprobe=4, rrot_seed=123
):
    """Helper: Create IndexIVFRaBitQ with random rotation."""
    quantizer = faiss.IndexFlat(d, metric)
    index_rbq = faiss.IndexIVFRaBitQ(
        quantizer, d, nlist, metric, True, nb_bits
    )
    index_rbq.qb = qb
    index_rbq.nprobe = nprobe
    rrot = faiss.RandomRotationMatrix(d, d)
    rrot.init(rrot_seed)
    return faiss.IndexPreTransform(rrot, index_rbq)


def compute_expected_code_size(d, nb_bits):
    """Helper: Compute expected code size based on formula."""
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
    """Shared helper: Test serialize/deserialize preserves search results."""
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


class TestMultiBitRaBitQ(unittest.TestCase):
    """Consolidated tests for multi-bit RaBitQ.

    Tests IndexRaBitQ and IndexIVFRaBitQ for construction, basic operations,
    recall, serialization, IVF operations, query quantization, and factory.
    """

    # ==================== Construction Tests ====================

    def test_valid_nb_bits_range(self):
        """Test that nb_bits 1-9 are valid for IndexRaBitQ."""
        d = 128
        for nb_bits in range(1, 10):
            for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
                index = faiss.IndexRaBitQ(d, metric, nb_bits)
                self.assertEqual(index.d, d)
                self.assertEqual(index.metric_type, metric)
                self.assertEqual(index.rabitq.nb_bits, nb_bits)

    def test_invalid_nb_bits(self):
        """Test that invalid nb_bits values raise errors."""
        with self.assertRaises(RuntimeError):
            faiss.IndexRaBitQ(128, faiss.METRIC_L2, 0)
        with self.assertRaises(RuntimeError):
            faiss.IndexRaBitQ(128, faiss.METRIC_L2, 10)

    def test_code_size_formula(self):
        """Test that code sizes match expected formula for all nb_bits."""
        d = 128
        for nb_bits in range(1, 10):
            index = faiss.IndexRaBitQ(d, faiss.METRIC_L2, nb_bits)
            expected_size = compute_expected_code_size(d, nb_bits)
            self.assertEqual(index.code_size, expected_size)

    def test_ivf_construction(self):
        """Test IndexIVFRaBitQ construction with valid/invalid nb_bits."""
        d, nlist = 64, 16
        # Valid nb_bits
        for nb_bits in range(1, 10):
            quantizer = faiss.IndexFlat(d, faiss.METRIC_L2)
            index = faiss.IndexIVFRaBitQ(
                quantizer, d, nlist, faiss.METRIC_L2, True, nb_bits
            )
            self.assertEqual(index.rabitq.nb_bits, nb_bits)
            expected = compute_expected_code_size(d, nb_bits)
            self.assertEqual(index.code_size, expected)

        # Invalid nb_bits
        quantizer = faiss.IndexFlat(d, faiss.METRIC_L2)
        with self.assertRaises(RuntimeError):
            faiss.IndexIVFRaBitQ(
                quantizer, d, nlist, faiss.METRIC_L2, True, 0
            )
        with self.assertRaises(RuntimeError):
            faiss.IndexIVFRaBitQ(
                quantizer, d, nlist, faiss.METRIC_L2, True, 10
            )

    # ==================== Basic Operations Tests ====================

    def test_basic_operations(self):
        """Test train/add/search pipeline for various configurations."""
        ds = datasets.SyntheticDataset(128, 300, 500, 20)
        k = 10

        for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
            for nb_bits in [1, 2, 4, 8]:
                for qb in [0, 4, 8]:
                    with self.subTest(metric=metric, nb_bits=nb_bits, qb=qb):
                        index = create_index_rabitq_with_rotation(
                            ds.d, metric, nb_bits, qb=qb
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
        """Test that recall is reasonable for various configurations."""
        for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
            metric_str = 'L2' if metric == faiss.METRIC_L2 else 'IP'
            ds = datasets.SyntheticDataset(
                128, 500, 1000, 50, metric=metric_str
            )
            I_gt = ds.get_groundtruth(10)

            for nb_bits in [1, 2, 4, 8]:
                for qb in [0, 8]:
                    with self.subTest(metric=metric, nb_bits=nb_bits, qb=qb):
                        index = create_index_rabitq_with_rotation(
                            ds.d, metric, nb_bits, qb=qb
                        )
                        index.train(ds.get_train())
                        index.add(ds.get_database())
                        _, I = index.search(ds.get_queries(), 10)
                        recall = faiss.eval_intersection(
                            I, I_gt
                        ) / (ds.nq * 10)
                        self.assertGreater(recall, 0.10)

    def test_recall_monotonic_improvement(self):
        """Test that recall improves with more bits."""
        for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
            metric_str = 'L2' if metric == faiss.METRIC_L2 else 'IP'
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
                        recalls[nb_bits] = faiss.eval_intersection(
                            I, I_gt
                        ) / (ds.nq * 10)

                    # Monotonic improvement with tolerance
                    tolerance = 0.03
                    self.assertGreaterEqual(recalls[2], recalls[1] - tolerance)
                    self.assertGreaterEqual(recalls[4], recalls[2] - tolerance)
                    self.assertGreaterEqual(recalls[8], recalls[4] - tolerance)
                    self.assertGreater(recalls[8], 0.75)

    # ==================== Serialization Tests ====================

    def test_serialization(self):
        """Test serialize/deserialize preserves search results."""
        ds = datasets.SyntheticDataset(64, 150, 200, 10)

        for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
            for nb_bits in [1, 2, 4, 8, 9]:
                for qb in [0, 4, 8]:
                    with self.subTest(metric=metric, nb_bits=nb_bits, qb=qb):
                        index1 = create_index_rabitq_with_rotation(
                            ds.d, metric, nb_bits, qb=qb
                        )
                        index1.train(ds.get_train())
                        index1.add(ds.get_database())
                        D1, I1 = index1.search(ds.get_queries(), 5)

                        index_bytes = faiss.serialize_index(index1)
                        index2 = faiss.deserialize_index(index_bytes)

                        self.assertEqual(index2.d, ds.d)
                        self.assertEqual(index2.ntotal, ds.nb)
                        self.assertTrue(index2.is_trained)

                        params = faiss.RaBitQSearchParameters()
                        params.qb = qb
                        params.centered = False
                        D2, I2 = index2.search(
                            ds.get_queries(), 5, params=params
                        )

                        np.testing.assert_array_equal(I1, I2)
                        np.testing.assert_allclose(D1, D2, rtol=1e-5)

    # ==================== IVF Tests ====================

    def test_ivf_basic_operations(self):
        """Test IVF train/add/search pipeline."""
        ds = datasets.SyntheticDataset(128, 300, 500, 20)

        for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
            for nb_bits in [1, 2, 4, 8]:
                for qb in [0, 4, 8]:
                    with self.subTest(metric=metric, nb_bits=nb_bits, qb=qb):
                        index = create_index_ivf_rabitq_with_rotation(
                            ds.d, metric, nb_bits, nlist=16, qb=qb, nprobe=4
                        )
                        index.train(ds.get_train())
                        index.add(ds.get_database())
                        D, I = index.search(ds.get_queries(), 10)

                        self.assertTrue(index.is_trained)
                        self.assertEqual(index.ntotal, ds.nb)
                        self.assertEqual(D.shape, (ds.nq, 10))
                        self.assertTrue(np.all(I >= 0))
                        self.assertTrue(np.all(np.isfinite(D)))

    def test_ivf_nprobe_improves_recall(self):
        """Test that higher nprobe improves recall."""
        for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
            metric_str = 'L2' if metric == faiss.METRIC_L2 else 'IP'
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
                        recalls[nprobe] = faiss.eval_intersection(
                            I, I_gt
                        ) / (ds.nq * 10)

                    self.assertGreaterEqual(recalls[2], recalls[1])
                    self.assertGreaterEqual(recalls[4], recalls[2])
                    self.assertGreaterEqual(recalls[8], recalls[4])

    def test_ivf_serialization(self):
        """Test IVF serialization preserves results."""
        ds = datasets.SyntheticDataset(64, 150, 200, 10)

        for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
            for nb_bits in [1, 2, 4, 8, 9]:
                for qb in [0, 4, 8]:
                    with self.subTest(metric=metric, nb_bits=nb_bits, qb=qb):
                        index1 = create_index_ivf_rabitq_with_rotation(
                            ds.d, metric, nb_bits, nlist=16, qb=qb, nprobe=4
                        )
                        index1.train(ds.get_train())
                        index1.add(ds.get_database())
                        D1, I1 = index1.search(ds.get_queries(), 5)

                        index_bytes = faiss.serialize_index(index1)
                        index2 = faiss.deserialize_index(index_bytes)

                        self.assertEqual(index2.d, ds.d)
                        self.assertEqual(index2.ntotal, ds.nb)

                        params = faiss.IVFRaBitQSearchParameters()
                        params.qb = qb
                        params.centered = False
                        params.nprobe = 4
                        D2, I2 = index2.search(
                            ds.get_queries(), 5, params=params
                        )

                        np.testing.assert_array_equal(I1, I2)
                        np.testing.assert_allclose(D1, D2, rtol=1e-5)

    # ==================== Query Quantization Tests ====================

    def test_query_quantization_levels(self):
        """Test that all qb values produce valid results."""
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
                        D, I = index.search(
                            ds.get_queries(), 10, params=params
                        )

                        self.assertEqual(D.shape, (ds.nq, 10))
                        self.assertTrue(np.all(I >= 0))
                        self.assertTrue(np.all(np.isfinite(D)))

    # ==================== Index Factory Tests ====================

    def test_factory_default_nb_bits(self):
        """Test that 'RaBitQ' creates 1-bit index by default."""
        index = faiss.index_factory(128, "RaBitQ")
        self.assertIsInstance(index, faiss.IndexRaBitQ)
        self.assertEqual(index.rabitq.nb_bits, 1)

    def test_factory_multibit(self):
        """Test 'RaBitQ{nb_bits}' creates correct multi-bit indexes."""
        for nb_bits in [2, 4, 8]:
            index = faiss.index_factory(128, f"RaBitQ{nb_bits}")
            self.assertIsInstance(index, faiss.IndexRaBitQ)
            self.assertEqual(index.rabitq.nb_bits, nb_bits)
            self.assertEqual(
                index.code_size, compute_expected_code_size(128, nb_bits)
            )

    def test_factory_ivf(self):
        """Test IVF factory with default and multi-bit."""
        # Default 1-bit
        index = faiss.index_factory(128, "IVF16,RaBitQ")
        self.assertIsInstance(index, faiss.IndexIVFRaBitQ)
        self.assertEqual(index.rabitq.nb_bits, 1)

        # Multi-bit
        for nb_bits in [2, 4, 8]:
            index = faiss.index_factory(128, f"IVF16,RaBitQ{nb_bits}")
            self.assertIsInstance(index, faiss.IndexIVFRaBitQ)
            self.assertEqual(index.rabitq.nb_bits, nb_bits)

    def test_factory_end_to_end(self):
        """Test complete workflow: factory creation, train, add, search."""
        ds = datasets.SyntheticDataset(64, 150, 200, 10)

        for nb_bits in [1, 4]:
            # Non-IVF
            factory_str = f"RaBitQ{nb_bits}" if nb_bits > 1 else "RaBitQ"
            index = faiss.index_factory(ds.d, factory_str)
            index.train(ds.get_train())
            index.add(ds.get_database())
            D, I = index.search(ds.get_queries(), 5)
            self.assertEqual(D.shape, (ds.nq, 5))
            self.assertTrue(np.all(I >= 0))

            # IVF
            ivf_str = (
                f"IVF16,RaBitQ{nb_bits}" if nb_bits > 1 else "IVF16,RaBitQ"
            )
            ivf_index = faiss.index_factory(ds.d, ivf_str)
            ivf_index.train(ds.get_train())
            ivf_index.add(ds.get_database())
            D_ivf, I_ivf = ivf_index.search(ds.get_queries(), 5)
            self.assertEqual(D_ivf.shape, (ds.nq, 5))
            self.assertTrue(np.all(I_ivf >= 0))


class TestRaBitQStats(unittest.TestCase):
    """Test RaBitQStats tracking for multi-bit two-stage search."""

    INDEX_TYPES = [
        "IndexRaBitQ",
        "IndexIVFRaBitQ",
    ]

    @classmethod
    def setUpClass(cls):
        cls.stats_available = hasattr(faiss, 'cvar') and hasattr(
            faiss.cvar, 'rabitq_stats'
        )
        if cls.stats_available:
            cls.rabitq_stats = faiss.cvar.rabitq_stats

    def test_stats_reset_and_skip_percentage(self):
        """Test that stats can be reset and skip_percentage works."""
        if not self.stats_available:
            self.skipTest("rabitq_stats not available in Python bindings")
        self.rabitq_stats.reset()
        self.assertEqual(self.rabitq_stats.n_1bit_evaluations, 0)
        self.assertEqual(self.rabitq_stats.n_multibit_evaluations, 0)
        self.assertEqual(self.rabitq_stats.skip_percentage(), 0.0)

    def test_stats_collected_multibit_all_index_types(self):
        """Test that stats are collected for all multi-bit index types."""
        if not self.stats_available:
            self.skipTest("rabitq_stats not available in Python bindings")
        ds = datasets.SyntheticDataset(384, 50000, 50000, 10)
        nlist = 16

        for index_type in self.INDEX_TYPES:
            for nb_bits in [2, 4]:
                with self.subTest(index_type=index_type, nb_bits=nb_bits):
                    self.rabitq_stats.reset()

                    if index_type == "IndexRaBitQ":
                        index = faiss.IndexRaBitQ(
                            ds.d, faiss.METRIC_L2, nb_bits
                        )
                    elif index_type == "IndexIVFRaBitQ":
                        quantizer = faiss.IndexFlat(ds.d, faiss.METRIC_L2)
                        index = faiss.IndexIVFRaBitQ(
                            quantizer, ds.d, nlist, faiss.METRIC_L2,
                            True, nb_bits
                        )
                        index.nprobe = 4
                    else:
                        raise ValueError(f"Unknown index type: {index_type}")

                    index.train(ds.get_train())
                    index.add(ds.get_database())
                    index.search(ds.get_queries(), 10)

                    self.assertGreater(
                        self.rabitq_stats.n_1bit_evaluations, 0
                    )
                    self.assertGreater(
                        self.rabitq_stats.n_multibit_evaluations, 0
                    )
                    # For multi-bit, filtering should skip some candidates
                    self.assertLess(
                        self.rabitq_stats.n_multibit_evaluations,
                        self.rabitq_stats.n_1bit_evaluations,
                    )
                    skip_pct = self.rabitq_stats.skip_percentage()
                    self.assertGreater(skip_pct, 0.0)
                    self.assertLessEqual(skip_pct, 100.0)

                    n_1bit = self.rabitq_stats.n_1bit_evaluations
                    n_multibit = self.rabitq_stats.n_multibit_evaluations
                    print(
                        f"{index_type} nb_bits={nb_bits}: "
                        f"n_1bit={n_1bit}, "
                        f"n_multibit={n_multibit}, "
                        f"skip={skip_pct:.1f}%"
                    )

if __name__ == "__main__":
    unittest.main()
