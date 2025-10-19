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
            np.testing.assert_(loss_pq < 0.25, f"{loss_pq}")

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

    def do_test_serde(self, description):
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

    def test_serde_rabitq(self):
        self.do_test_serde("RaBitQ")


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

    def do_test_serde(self, description):
        ds = datasets.SyntheticDataset(32, 1000, 100, 20)

        xt = ds.get_train()
        xb = ds.get_database()

        index = faiss.index_factory(ds.d, description)
        index.train(xt)
        index.add(xb)

        Dref, Iref = index.search(ds.get_queries(), 10)

        b = faiss.serialize_index(index)
        index2 = faiss.deserialize_index(b)

        Dnew, Inew = index2.search(ds.get_queries(), 10)

        np.testing.assert_equal(Dref, Dnew)
        np.testing.assert_equal(Iref, Inew)

    def test_serde_ivfrabitq(self):
        self.do_test_serde("IVF16,RaBitQ")


class TestRaBitQFastScan(unittest.TestCase):
    def do_comparison_vs_rabitq_test(
        self, metric_type=faiss.METRIC_L2, bbs=32
    ):
        """Test IndexRaBitQFastScan produces similar results to IndexRaBitQ"""
        ds = datasets.SyntheticDataset(128, 4096, 4096, 100)
        k = 10

        # IndexRaBitQ baseline
        index_rbq = faiss.IndexRaBitQ(ds.d, metric_type)
        index_rbq.train(ds.get_train())
        index_rbq.add(ds.get_database())
        _, I_rbq = index_rbq.search(ds.get_queries(), k)

        # IndexRaBitQFastScan
        index_rbq_fs = faiss.IndexRaBitQFastScan(ds.d, metric_type, bbs)
        index_rbq_fs.train(ds.get_train())
        index_rbq_fs.add(ds.get_database())
        _, I_rbq_fs = index_rbq_fs.search(ds.get_queries(), k)

        index_flat = faiss.IndexFlat(ds.d, metric_type)
        index_flat.train(ds.get_train())
        index_flat.add(ds.get_database())
        _, I_f = index_flat.search(ds.get_queries(), k)

        # Evaluate against ground truth
        eval_rbq = faiss.eval_intersection(I_rbq[:, :k], I_f[:, :k])
        eval_rbq /= ds.nq * k
        eval_rbq_fs = faiss.eval_intersection(I_rbq_fs[:, :k], I_f[:, :k])
        eval_rbq_fs /= ds.nq * k

        print(
            f"RaBitQ baseline is {eval_rbq}, "
            f"RaBitQFastScan is {eval_rbq_fs}"
        )

        # FastScan should be similar to baseline
        np.testing.assert_(abs(eval_rbq - eval_rbq_fs) < 0.05)

    def test_comparison_vs_rabitq_L2(self):
        self.do_comparison_vs_rabitq_test(faiss.METRIC_L2)

    def test_comparison_vs_rabitq_IP(self):
        self.do_comparison_vs_rabitq_test(faiss.METRIC_INNER_PRODUCT)

    def test_encode_decode_consistency(self):
        """Test that encoding and decoding operations are consistent"""
        ds = datasets.SyntheticDataset(128, 1000, 0, 0)  # No queries/db needed

        # Test with IndexRaBitQFastScan
        index_rbq_fs = faiss.IndexRaBitQFastScan(ds.d, faiss.METRIC_L2)
        index_rbq_fs.train(ds.get_train())

        # Test encode/decode on a subset of training data
        test_vectors = ds.get_train()[:100]

        # Test compute_codes and sa_decode
        # This tests that factors are properly embedded in codes
        codes = np.empty(
            (len(test_vectors), index_rbq_fs.code_size), dtype=np.uint8
        )
        index_rbq_fs.compute_codes(
            faiss.swig_ptr(codes),
            len(test_vectors),
            faiss.swig_ptr(test_vectors)
        )

        # sa_decode should work directly with embedded codes
        decoded_fs = index_rbq_fs.sa_decode(codes)

        # Check reconstruction error for FastScan
        distances_fs = np.sum((test_vectors - decoded_fs) ** 2, axis=1)
        avg_distance_fs = np.mean(distances_fs)
        print(f"Average FastScan reconstruction error: {avg_distance_fs}")

        # Compare with original IndexRaBitQ on the SAME dataset
        index_rbq = faiss.IndexRaBitQ(ds.d, faiss.METRIC_L2)
        index_rbq.train(ds.get_train())

        # Encode with original RaBitQ (correct API - returns encoded array)
        codes_orig = index_rbq.sa_encode(test_vectors)

        # Decode with original RaBitQ
        decoded_orig = index_rbq.sa_decode(codes_orig)

        # Check reconstruction error for original
        distances_orig = np.sum((test_vectors - decoded_orig) ** 2, axis=1)
        avg_distance_orig = np.mean(distances_orig)
        print(
            f"Average original RaBitQ reconstruction error: "
            f"{avg_distance_orig}"
        )

        # Print comparison
        print(
            f"Error difference (FastScan - Original): "
            f"{avg_distance_fs - avg_distance_orig}"
        )

        # FastScan should have similar reconstruction error to original RaBitQ
        np.testing.assert_(
            abs(avg_distance_fs - avg_distance_orig) < 0.01
        )  # Should be nearly identical

    def test_query_quantization_bits(self):
        """Test different query quantization bit settings"""
        ds = datasets.SyntheticDataset(64, 2000, 2000, 50)
        k = 10

        index_rbq_fs = faiss.IndexRaBitQFastScan(ds.d, faiss.METRIC_L2)
        index_rbq_fs.train(ds.get_train())
        index_rbq_fs.add(ds.get_database())

        # Test different qb values
        results = {}
        for qb in [4, 6, 8]:
            index_rbq_fs.qb = qb
            _, I = index_rbq_fs.search(ds.get_queries(), k)
            results[qb] = I

        # All should produce reasonable results
        index_flat = faiss.IndexFlat(ds.d, faiss.METRIC_L2)
        index_flat.train(ds.get_train())
        index_flat.add(ds.get_database())
        _, I_f = index_flat.search(ds.get_queries(), k)

        for qb in [4, 6, 8]:
            eval_qb = faiss.eval_intersection(results[qb][:, :k], I_f[:, :k])
            eval_qb /= ds.nq * k
            print(f"Query quantization qb={qb} recall: {eval_qb}")
            np.testing.assert_(eval_qb > 0.4)  # Should be reasonable

    def test_small_dataset(self):
        """Test on a small dataset to ensure basic functionality"""
        d = 32
        n = 100
        nq = 10

        rs = np.random.RandomState(123)
        xb = rs.rand(n, d).astype(np.float32)
        xq = rs.rand(nq, d).astype(np.float32)

        index_rbq_fs = faiss.IndexRaBitQFastScan(d, faiss.METRIC_L2)
        index_rbq_fs.train(xb)
        index_rbq_fs.add(xb)

        k = 5
        distances, labels = index_rbq_fs.search(xq, k)

        # Check output shapes and validity
        np.testing.assert_equal(distances.shape, (nq, k))
        np.testing.assert_equal(labels.shape, (nq, k))

        # Check that labels are valid indices
        np.testing.assert_(np.all(labels >= 0))
        np.testing.assert_(np.all(labels < n))

        # Check that distances are non-negative (for L2)
        np.testing.assert_(np.all(distances >= 0))

        # Quick recall check against exact search
        index_flat = faiss.IndexFlat(d, faiss.METRIC_L2)
        index_flat.train(xb)
        index_flat.add(xb)
        _, I_f = index_flat.search(xq, k)

        # Evaluate recall
        recall = faiss.eval_intersection(labels[:, :k], I_f[:, :k])
        recall /= (nq * k)
        print(f"Small dataset recall: {recall:.3f}")
        np.testing.assert_(
            recall > 0.4
        )  # Should be reasonable for small dataset

    def test_comparison_vs_pq_fastscan(self):
        """Compare RaBitQFastScan to PQFastScan as a performance baseline"""
        ds = datasets.SyntheticDataset(128, 4096, 4096, 100)
        k = 10

        # PQFastScan baseline
        index_pq_fs = faiss.IndexPQFastScan(ds.d, 16, 4, faiss.METRIC_L2)
        index_pq_fs.train(ds.get_train())
        index_pq_fs.add(ds.get_database())
        _, I_pq_fs = index_pq_fs.search(ds.get_queries(), k)

        # RaBitQFastScan
        index_rbq_fs = faiss.IndexRaBitQFastScan(ds.d, faiss.METRIC_L2)
        index_rbq_fs.train(ds.get_train())
        index_rbq_fs.add(ds.get_database())
        _, I_rbq_fs = index_rbq_fs.search(ds.get_queries(), k)

        index_flat = faiss.IndexFlat(ds.d, faiss.METRIC_L2)
        index_flat.train(ds.get_train())
        index_flat.add(ds.get_database())
        _, I_f = index_flat.search(ds.get_queries(), k)

        # Evaluate both against ground truth
        eval_pq_fs = faiss.eval_intersection(I_pq_fs[:, :k], I_f[:, :k])
        eval_pq_fs /= ds.nq * k
        eval_rbq_fs = faiss.eval_intersection(I_rbq_fs[:, :k], I_f[:, :k])
        eval_rbq_fs /= ds.nq * k

        print(
            f"PQFastScan is {eval_pq_fs}, "
            f"RaBitQFastScan is {eval_rbq_fs}"
        )

        # RaBitQFastScan should have reasonable performance similar to regular
        # RaBitQ
        np.testing.assert_(eval_rbq_fs > 0.55)

    def test_serialization(self):
        """Test serialization and deserialization of RaBitQFastScan"""
        ds = datasets.SyntheticDataset(64, 1000, 100, 20)

        index_rbq_fs = faiss.IndexRaBitQFastScan(ds.d, faiss.METRIC_L2)
        index_rbq_fs.train(ds.get_train())
        index_rbq_fs.add(ds.get_database())

        Dref, Iref = index_rbq_fs.search(ds.get_queries(), 10)

        # Serialize and deserialize
        b = faiss.serialize_index(index_rbq_fs)
        index2 = faiss.deserialize_index(b)

        Dnew, Inew = index2.search(ds.get_queries(), 10)

        # Results should be identical
        np.testing.assert_array_equal(Dref, Dnew)
        np.testing.assert_array_equal(Iref, Inew)

    def test_memory_management(self):
        """Test that memory is managed correctly during operations"""
        ds = datasets.SyntheticDataset(128, 2000, 2000, 50)

        index_rbq_fs = faiss.IndexRaBitQFastScan(ds.d, faiss.METRIC_L2)
        index_rbq_fs.train(ds.get_train())

        # Add data in chunks to test memory management
        chunk_size = 500
        for i in range(0, ds.nb, chunk_size):
            end_idx = min(i + chunk_size, ds.nb)
            chunk_data = ds.get_database()[i:end_idx]
            index_rbq_fs.add(chunk_data)

        # Verify total count
        np.testing.assert_equal(index_rbq_fs.ntotal, ds.nb)

        # Test search still works and produces reasonable recall
        _, I = index_rbq_fs.search(ds.get_queries(), 5)
        np.testing.assert_equal(I.shape, (ds.nq, 5))

        # Compare against ground truth to verify recall
        index_flat = faiss.IndexFlat(ds.d, faiss.METRIC_L2)
        index_flat.train(ds.get_train())
        index_flat.add(ds.get_database())
        _, I_f = index_flat.search(ds.get_queries(), 5)

        # Calculate recall - should be reasonable despite multiple add() calls
        recall = faiss.eval_intersection(I[:, :5], I_f[:, :5])
        recall /= (ds.nq * 5)

        # If embedded factors are corrupted by multiple add() calls,
        # recall will be very low
        np.testing.assert_(
            recall > 0.1,
            f"Recall too low: {recall:.3f} - suggests multiple "
            f"add() calls corrupted embedded factors"
        )

    def test_invalid_parameters(self):
        """Test proper error handling for invalid parameters"""
        # Invalid dimension
        with np.testing.assert_raises(Exception):
            faiss.IndexRaBitQFastScan(0, faiss.METRIC_L2)

        # Invalid metric (should only support L2 and IP)
        try:
            faiss.IndexRaBitQFastScan(64, faiss.METRIC_Lp)
            np.testing.assert_(
                False, "Should have raised exception for invalid metric"
            )
        except RuntimeError:
            pass  # Expected

    def test_thread_safety(self):
        """Test that parallel operations work correctly"""
        ds = datasets.SyntheticDataset(64, 2000, 2000, 500)

        index_rbq_fs = faiss.IndexRaBitQFastScan(ds.d, faiss.METRIC_L2)
        index_rbq_fs.train(ds.get_train())
        index_rbq_fs.add(ds.get_database())

        # Search with multiple threads (implicitly tested through OpenMP)
        k = 10
        distances, labels = index_rbq_fs.search(ds.get_queries(), k)

        # Basic sanity checks
        np.testing.assert_equal(distances.shape, (ds.nq, k))
        np.testing.assert_equal(labels.shape, (ds.nq, k))
        np.testing.assert_(np.all(distances >= 0))
        np.testing.assert_(np.all(labels >= 0))
        np.testing.assert_(np.all(labels < ds.nb))

    def test_factory_construction(self):
        """Test that RaBitQFastScan can be constructed via factory method"""
        ds = datasets.SyntheticDataset(64, 500, 500, 20)

        # Test RaBitQFastScan (non-IVF) factory construction
        index_flat = faiss.index_factory(ds.d, "RaBitQfs")
        np.testing.assert_(isinstance(index_flat, faiss.IndexRaBitQFastScan))
        index_flat.train(ds.get_train())
        index_flat.add(ds.get_database())

        # Test basic search
        _, I_flat = index_flat.search(ds.get_queries(), 5)
        np.testing.assert_equal(I_flat.shape, (ds.nq, 5))

        # Test with custom batch size
        index_custom = faiss.index_factory(ds.d, "RaBitQfs_64")
        np.testing.assert_(isinstance(index_custom, faiss.IndexRaBitQFastScan))
        np.testing.assert_equal(index_custom.bbs, 64)


class TestIVFRaBitQFastScan(unittest.TestCase):
    def do_comparison_vs_ivfrabitq_test(self, metric_type=faiss.METRIC_L2):
        """Test IVFRaBitQFastScan produces similar results to IVFRaBitQ"""
        nlist = 64
        nprobe = 8
        nq = 500
        ds = datasets.SyntheticDataset(128, 2048, 2048, nq)
        k = 10

        d = ds.d
        xb = ds.get_database()
        xt = ds.get_train()
        xq = ds.get_queries()

        # Ground truth for evaluation
        index_flat = faiss.IndexFlat(d, metric_type)
        index_flat.train(xt)
        index_flat.add(xb)
        _, I_f = index_flat.search(xq, k)

        # Test different combinations of centered and qb values
        for centered in [False, True]:
            for qb in [1, 4, 8]:
                # IndexIVFRaBitQ baseline
                quantizer = faiss.IndexFlat(d, metric_type)
                index_ivf_rbq = faiss.IndexIVFRaBitQ(
                    quantizer, d, nlist, metric_type
                )
                index_ivf_rbq.qb = qb
                index_ivf_rbq.nprobe = nprobe
                index_ivf_rbq.train(xt)
                index_ivf_rbq.add(xb)

                rbq_params = faiss.IVFRaBitQSearchParameters()
                rbq_params.nprobe = nprobe
                rbq_params.qb = qb
                rbq_params.centered = centered
                _, I_ivf_rbq = index_ivf_rbq.search(xq, k, params=rbq_params)

                # IndexIVFRaBitQFastScan
                quantizer_fs = faiss.IndexFlat(d, metric_type)
                index_ivf_rbq_fs = faiss.IndexIVFRaBitQFastScan(
                    quantizer_fs, d, nlist, metric_type, 32
                )
                index_ivf_rbq_fs.qb = qb
                index_ivf_rbq_fs.centered = centered
                index_ivf_rbq_fs.nprobe = nprobe
                index_ivf_rbq_fs.train(xt)
                index_ivf_rbq_fs.add(xb)

                rbq_fs_params = faiss.IVFRaBitQFastScanSearchParameters()
                rbq_fs_params.nprobe = nprobe
                rbq_fs_params.qb = qb
                rbq_fs_params.centered = centered
                _, I_ivf_rbq_fs = index_ivf_rbq_fs.search(
                    xq, k, params=rbq_fs_params
                )

                # Evaluate against ground truth
                eval_ivf_rbq = faiss.eval_intersection(
                    I_ivf_rbq[:, :k], I_f[:, :k]
                )
                eval_ivf_rbq /= ds.nq * k
                eval_ivf_rbq_fs = faiss.eval_intersection(
                    I_ivf_rbq_fs[:, :k], I_f[:, :k]
                )
                eval_ivf_rbq_fs /= ds.nq * k

                # Performance gap should be within 1 percent
                recall_gap = abs(eval_ivf_rbq - eval_ivf_rbq_fs)
                np.testing.assert_(
                    recall_gap < 0.01,
                    f"Performance gap too large for centered={centered}, "
                    f"qb={qb}: {recall_gap:.4f}"
                )

    def test_comparison_vs_ivfrabitq_L2(self):
        self.do_comparison_vs_ivfrabitq_test(faiss.METRIC_L2)

    def test_comparison_vs_ivfrabitq_IP(self):
        self.do_comparison_vs_ivfrabitq_test(faiss.METRIC_INNER_PRODUCT)

    def test_encode_decode_consistency(self):
        """Test that encoding and decoding operations are consistent"""
        nlist = 32
        ds = datasets.SyntheticDataset(64, 1000, 1000, 0)  # No queries needed

        d = ds.d
        xt = ds.get_train()
        xb = ds.get_database()

        # Test with IndexIVFRaBitQFastScan
        quantizer = faiss.IndexFlat(d, faiss.METRIC_L2)
        index_ivf_rbq_fs = faiss.IndexIVFRaBitQFastScan(
            quantizer, d, nlist, faiss.METRIC_L2, 32  # bbs=32 for FastScan
        )
        index_ivf_rbq_fs.qb = 8
        index_ivf_rbq_fs.centered = False
        index_ivf_rbq_fs.train(xt)

        # Add vectors to the index
        test_vectors = xb[:100]
        index_ivf_rbq_fs.add(test_vectors)

        # Reconstruct the vectors using reconstruct_n
        decoded_fs = index_ivf_rbq_fs.reconstruct_n(0, len(test_vectors))

        # Check reconstruction error for FastScan
        distances_fs = np.sum((test_vectors - decoded_fs) ** 2, axis=1)
        avg_distance_fs = np.mean(distances_fs)

        # Compare with original IndexIVFRaBitQ
        quantizer_orig = faiss.IndexFlat(d, faiss.METRIC_L2)
        index_ivf_rbq = faiss.IndexIVFRaBitQ(
            quantizer_orig, d, nlist, faiss.METRIC_L2
        )
        index_ivf_rbq.qb = 8
        index_ivf_rbq.train(xt)
        index_ivf_rbq.add(test_vectors)

        # Reconstruct with original IVFRaBitQ
        decoded_orig = index_ivf_rbq.reconstruct_n(0, len(test_vectors))

        # Check reconstruction error for original
        distances_orig = np.sum((test_vectors - decoded_orig) ** 2, axis=1)
        avg_distance_orig = np.mean(distances_orig)

        # FastScan should have similar reconstruction error to original
        np.testing.assert_(
            abs(avg_distance_fs - avg_distance_orig) < 0.01
        )

    def test_nprobe_variations(self):
        """Test different nprobe values comparing with IVFRaBitQ"""
        nlist = 32
        ds = datasets.SyntheticDataset(64, 1000, 1000, 50)
        k = 10

        d = ds.d
        xb = ds.get_database()
        xt = ds.get_train()
        xq = ds.get_queries()

        # Ground truth
        index_flat = faiss.IndexFlat(d, faiss.METRIC_L2)
        index_flat.train(xt)
        index_flat.add(xb)
        _, I_f = index_flat.search(xq, k)

        # Test different nprobe values
        for nprobe in [1, 4, 8, 16]:
            # IndexIVFRaBitQ baseline
            quantizer_rbq = faiss.IndexFlat(d, faiss.METRIC_L2)
            index_ivf_rbq = faiss.IndexIVFRaBitQ(
                quantizer_rbq, d, nlist, faiss.METRIC_L2
            )
            index_ivf_rbq.qb = 8
            index_ivf_rbq.nprobe = nprobe
            index_ivf_rbq.train(xt)
            index_ivf_rbq.add(xb)

            rbq_params = faiss.IVFRaBitQSearchParameters()
            rbq_params.nprobe = nprobe
            rbq_params.qb = 8
            rbq_params.centered = False
            _, I_rbq = index_ivf_rbq.search(xq, k, params=rbq_params)

            # IndexIVFRaBitQFastScan
            quantizer_fs = faiss.IndexFlat(d, faiss.METRIC_L2)
            index_ivf_rbq_fs = faiss.IndexIVFRaBitQFastScan(
                quantizer_fs, d, nlist, faiss.METRIC_L2, 32
            )
            index_ivf_rbq_fs.qb = 8
            index_ivf_rbq_fs.centered = False
            index_ivf_rbq_fs.nprobe = nprobe
            index_ivf_rbq_fs.train(xt)
            index_ivf_rbq_fs.add(xb)

            rbq_fs_params = faiss.IVFRaBitQFastScanSearchParameters()
            rbq_fs_params.nprobe = nprobe
            rbq_fs_params.qb = 8
            rbq_fs_params.centered = False
            _, I_fs = index_ivf_rbq_fs.search(xq, k, params=rbq_fs_params)

            # Evaluate against ground truth
            eval_rbq = faiss.eval_intersection(I_rbq[:, :k], I_f[:, :k])
            eval_rbq /= ds.nq * k
            eval_fs = faiss.eval_intersection(I_fs[:, :k], I_f[:, :k])
            eval_fs /= ds.nq * k

            # Performance gap should be within 1 percent
            performance_gap = abs(eval_rbq - eval_fs)
            np.testing.assert_(
                performance_gap < 0.01,
                f"Performance gap too large for nprobe={nprobe}: "
                f"{performance_gap:.4f}"
            )

    def test_serialization(self):
        """Test serialization and deserialization of IVFRaBitQFastScan"""
        # Use similar parameters to non-IVF test but with IVF structure
        nlist = 4  # Small number of centroids for simpler test
        ds = datasets.SyntheticDataset(64, 1000, 100, 20)  # Match dataset size

        d = ds.d
        xb = ds.get_database()
        xt = ds.get_train()
        xq = ds.get_queries()

        # Create index similar to non-IVF but with IVF structure
        quantizer = faiss.IndexFlat(d, faiss.METRIC_L2)
        index_ivf_rbq_fs = faiss.IndexIVFRaBitQFastScan(
            quantizer, d, nlist, faiss.METRIC_L2
        )
        index_ivf_rbq_fs.train(xt)
        index_ivf_rbq_fs.add(xb)

        # Set reasonable search parameters
        index_ivf_rbq_fs.nprobe = 2  # Use fewer probes for stability

        # Test search before serialization
        Dref, Iref = index_ivf_rbq_fs.search(xq, 10)

        # Serialize and deserialize
        b = faiss.serialize_index(index_ivf_rbq_fs)
        index2 = faiss.deserialize_index(b)

        # Set same search parameters on deserialized index
        index2.nprobe = 2

        # Test search after deserialization
        Dnew, Inew = index2.search(xq, 10)

        # Results should be identical
        np.testing.assert_array_equal(Dref, Dnew)
        np.testing.assert_array_equal(Iref, Inew)

    def test_memory_management(self):
        """Test that memory is managed correctly during operations"""
        nlist = 16
        ds = datasets.SyntheticDataset(64, 1000, 1000, 50)

        d = ds.d
        xb = ds.get_database()
        xt = ds.get_train()

        quantizer = faiss.IndexFlat(d, faiss.METRIC_L2)
        index_ivf_rbq_fs = faiss.IndexIVFRaBitQFastScan(
            quantizer, d, nlist, faiss.METRIC_L2
        )
        index_ivf_rbq_fs.train(xt)

        # Add data in chunks to test memory management
        chunk_size = 250
        for i in range(0, ds.nb, chunk_size):
            end_idx = min(i + chunk_size, ds.nb)
            chunk_data = xb[i:end_idx]
            index_ivf_rbq_fs.add(chunk_data)

        # Verify total count
        np.testing.assert_equal(index_ivf_rbq_fs.ntotal, ds.nb)

        # Test search still works
        search_params = faiss.IVFRaBitQFastScanSearchParameters(nprobe=4)
        _, I = index_ivf_rbq_fs.search(
            ds.get_queries(), 5, params=search_params
        )
        np.testing.assert_equal(I.shape, (ds.nq, 5))

    def test_thread_safety(self):
        """Test parallel operations work correctly via OpenMP

        OpenMP parallelization is triggered when n * nprobe > 1000
        in compute_LUT (see IndexIVFRaBitQFastScan.cpp line 339).
        With nq=300 and nprobe=4: 300 * 4 = 1200 > 1000.

        This test verifies:
        1. OpenMP threshold is exceeded to trigger parallel execution
        2. Thread-safe operations produce correct results
        3. No race conditions occur with query_factors_storage
        """
        import os

        # Verify OpenMP is available
        omp_num_threads = os.environ.get("OMP_NUM_THREADS")
        if omp_num_threads and int(omp_num_threads) == 1:
            # Skip this test if OpenMP is explicitly disabled
            return

        nlist = 16
        ds = datasets.SyntheticDataset(64, 1000, 1000, 300)

        quantizer = faiss.IndexFlat(ds.d, faiss.METRIC_L2)
        index_ivf_rbq_fs = faiss.IndexIVFRaBitQFastScan(
            quantizer, ds.d, nlist, faiss.METRIC_L2
        )
        index_ivf_rbq_fs.qb = 8
        index_ivf_rbq_fs.centered = False
        index_ivf_rbq_fs.nprobe = 4
        index_ivf_rbq_fs.train(ds.get_train())
        index_ivf_rbq_fs.add(ds.get_database())

        # Create search parameters
        params = faiss.IVFRaBitQFastScanSearchParameters()
        params.nprobe = 4
        params.qb = 8
        params.centered = False

        # Search with multiple queries
        # n * nprobe = 300 * 4 = 1200 > 1000, triggering OpenMP parallel loop
        k = 10
        distances, labels = index_ivf_rbq_fs.search(
            ds.get_queries(), k, params=params
        )

        # Basic sanity checks
        np.testing.assert_equal(distances.shape, (ds.nq, k))
        np.testing.assert_equal(labels.shape, (ds.nq, k))
        np.testing.assert_(np.all(distances >= 0))
        np.testing.assert_(np.all(labels >= 0))
        np.testing.assert_(np.all(labels < ds.nb))

    def test_factory_construction(self):
        """Test that IVF index can be constructed via factory method"""
        nlist = 16
        ds = datasets.SyntheticDataset(64, 500, 500, 20)

        # Test IVFRaBitQFastScan factory construction
        index = faiss.index_factory(ds.d, f"IVF{nlist},RaBitQfs")
        np.testing.assert_(isinstance(index, faiss.IndexIVFRaBitQFastScan))
        index.train(ds.get_train())
        index.add(ds.get_database())

        # Test basic search
        _, I = index.search(ds.get_queries(), 5)
        np.testing.assert_equal(I.shape, (ds.nq, 5))

        # Test IVF with custom batch size
        index_custom = faiss.index_factory(
            ds.d, f"IVF{nlist},RaBitQfs_64"
        )
        np.testing.assert_(
            isinstance(index_custom, faiss.IndexIVFRaBitQFastScan)
        )
        np.testing.assert_equal(index_custom.bbs, 64)

    def do_test_search_implementation(self, impl):
        """Helper to test a specific search implementation"""
        nlist = 32
        nprobe = 8
        ds = datasets.SyntheticDataset(128, 2048, 2048, 100)
        k = 10

        d = ds.d
        xb = ds.get_database()
        xt = ds.get_train()
        xq = ds.get_queries()

        # Ground truth for evaluation
        index_flat = faiss.IndexFlat(d, faiss.METRIC_L2)
        index_flat.train(xt)
        index_flat.add(xb)
        _, I_f = index_flat.search(xq, k)

        # Baseline: IndexIVFRaBitQ
        quantizer_baseline = faiss.IndexFlat(d, faiss.METRIC_L2)
        index_ivf_rbq = faiss.IndexIVFRaBitQ(
            quantizer_baseline, d, nlist, faiss.METRIC_L2
        )
        index_ivf_rbq.qb = 8
        index_ivf_rbq.nprobe = nprobe
        index_ivf_rbq.train(xt)
        index_ivf_rbq.add(xb)

        rbq_params = faiss.IVFRaBitQSearchParameters()
        rbq_params.nprobe = nprobe
        rbq_params.qb = 8
        rbq_params.centered = False
        _, I_rbq = index_ivf_rbq.search(xq, k, params=rbq_params)

        # Evaluate baseline against ground truth
        eval_baseline = faiss.eval_intersection(I_rbq[:, :k], I_f[:, :k])
        eval_baseline /= ds.nq * k

        # Test IndexIVFRaBitQFastScan with specific implementation
        quantizer_fs = faiss.IndexFlat(d, faiss.METRIC_L2)
        index_ivf_rbq_fs = faiss.IndexIVFRaBitQFastScan(
            quantizer_fs, d, nlist, faiss.METRIC_L2, 32
        )
        index_ivf_rbq_fs.qb = 8
        index_ivf_rbq_fs.centered = False
        index_ivf_rbq_fs.nprobe = nprobe
        index_ivf_rbq_fs.implem = impl

        index_ivf_rbq_fs.train(xt)
        index_ivf_rbq_fs.add(xb)

        # Create search parameters
        params = faiss.IVFRaBitQFastScanSearchParameters()
        params.nprobe = nprobe
        params.qb = 8
        params.centered = False

        # Perform search
        _, I_impl = index_ivf_rbq_fs.search(xq, k, params=params)

        # Evaluate against ground truth
        eval_impl = faiss.eval_intersection(I_impl[:, :k], I_f[:, :k])
        eval_impl /= ds.nq * k

        # Basic sanity checks
        np.testing.assert_equal(I_impl.shape, (ds.nq, k))

        # FastScan should perform similarly to baseline (within 5% gap)
        recall_gap = abs(eval_baseline - eval_impl)
        np.testing.assert_(
            recall_gap < 0.05,
            f"Recall gap too large for search_implem_{impl}: "
            f"baseline={eval_baseline:.4f}, impl={eval_impl:.4f}, "
            f"gap={recall_gap:.4f}"
        )

    def test_search_implem_10(self):
        self.do_test_search_implementation(impl=10)

    def test_search_implem_12(self):
        self.do_test_search_implementation(impl=12)

    def test_search_implem_14(self):
        self.do_test_search_implementation(impl=14)

    def test_search_with_parameters(self):
        """Test IndexIVFRaBitQFastScan with search_with_parameters

        This tests the code path through search_with_parameters which
        performs explicit coarse quantization before calling
        search_preassigned.
        """
        nlist = 64
        nprobe = 8
        nq = 500
        ds = datasets.SyntheticDataset(128, 2048, 2048, nq)
        k = 10

        d = ds.d
        xb = ds.get_database()
        xt = ds.get_train()
        xq = ds.get_queries()

        quantizer = faiss.IndexFlat(d, faiss.METRIC_L2)

        index = faiss.IndexIVFRaBitQFastScan(
            quantizer, d, nlist, faiss.METRIC_L2, 32
        )
        index.qb = 8
        index.centered = False
        index.nprobe = nprobe

        index.train(xt)
        index.add(xb)

        params = faiss.IVFRaBitQFastScanSearchParameters()
        params.nprobe = nprobe
        params.qb = 8
        params.centered = False

        distances, labels = faiss.search_with_parameters(index, xq, k, params)

        self.assertEqual(distances.shape, (nq, k))
        self.assertEqual(labels.shape, (nq, k))
        self.assertGreater(np.sum(labels >= 0), 0)

        index_flat = faiss.IndexFlat(d, faiss.METRIC_L2)
        index_flat.add(xb)
        _, gt_labels = index_flat.search(xq, k)
        recall = faiss.eval_intersection(labels, gt_labels) / (nq * k)
        # With nlist=64 and nprobe=8, recall should be reasonable
        self.assertGreater(recall, 0.4)


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
