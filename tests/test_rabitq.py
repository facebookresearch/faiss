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

                rbq_fs_params = faiss.IVFSearchParameters()
                rbq_fs_params.nprobe = nprobe
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

            rbq_fs_params = faiss.IVFSearchParameters()
            rbq_fs_params.nprobe = nprobe
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
        search_params = faiss.IVFSearchParameters()
        search_params.nprobe = 4
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
        params = faiss.IVFSearchParameters()
        params.nprobe = 4

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

        k = 5
        _, I = index.search(ds.get_queries(), k)
        np.testing.assert_equal(I.shape, (ds.nq, k))

        quantizer = faiss.IndexFlat(ds.d, faiss.METRIC_L2)
        index_ivf_rbq = faiss.IndexIVFRaBitQ(
            quantizer, ds.d, nlist, faiss.METRIC_L2
        )
        index_ivf_rbq.train(ds.get_train())
        index_ivf_rbq.add(ds.get_database())
        _, I_rbq = index_ivf_rbq.search(ds.get_queries(), k)

        recall = faiss.eval_intersection(I[:, :k], I_rbq[:, :k])
        recall /= (ds.nq * k)
        print(f"IVFRaBitQFastScan vs IVFRaBitQ recall: {recall:.3f}")
        np.testing.assert_(
            recall > 0.95,
            f"Recall too low: {recall:.3f} - should be close to baseline"
        )

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
        params = faiss.IVFSearchParameters()
        params.nprobe = nprobe

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

        params = faiss.IVFSearchParameters()
        params.nprobe = nprobe

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


# ==============================================================================
# Multi-bit RaBitQ Test Suite
# ==============================================================================


def create_test_dataset(d=128, nb=1000, nq=100, nt=500, metric='L2'):
    """Helper: Create synthetic dataset for testing."""
    return datasets.SyntheticDataset(d, nt, nb, nq, metric=metric)


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


def compute_recall_at_k(I_gt, I_pred):
    """Helper: Compute recall@k metric."""
    nq, k = I_gt.shape
    recall = 0.0
    for i in range(nq):
        gt_set = set(I_gt[i])
        pred_set = set(I_pred[i])
        recall += len(gt_set & pred_set) / k
    return recall / nq


def compute_expected_code_size(d, nb_bits):
    """Helper: Compute expected code size based on formula."""
    ex_bits = nb_bits - 1
    # For 1-bit: use BaseFactorsData (8 bytes)
    # For multi-bit: use FactorsData (12 bytes)
    base_size = (d + 7) // 8 + (8 if ex_bits == 0 else 12)
    if ex_bits > 0:
        ex_size = (d * ex_bits + 7) // 8 + 8  # ex-bit codes + ExFactorsData
        return base_size + ex_size
    return base_size


class TestMultiBitRaBitQConstruction(unittest.TestCase):
    """Test construction and parameter validation for multi-bit RaBitQ."""

    def test_valid_nb_bits_range(self):
        """Test that nb_bits 1-9 are valid."""
        d = 128
        for nb_bits in range(1, 10):
            for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
                index = faiss.IndexRaBitQ(d, metric, nb_bits)
                self.assertEqual(index.d, d)
                self.assertEqual(index.metric_type, metric)
                self.assertEqual(index.rabitq.nb_bits, nb_bits)
                self.assertFalse(index.is_trained)

    def test_invalid_nb_bits_zero(self):
        """Test that nb_bits=0 raises error."""
        with self.assertRaises(RuntimeError):
            faiss.IndexRaBitQ(128, faiss.METRIC_L2, 0)

    def test_invalid_nb_bits_too_large(self):
        """Test that nb_bits=10 raises error."""
        with self.assertRaises(RuntimeError):
            faiss.IndexRaBitQ(128, faiss.METRIC_L2, 10)

    def test_code_size_formula(self):
        """Test that code sizes match expected formula for all nb_bits."""
        d = 128
        for nb_bits in range(1, 10):
            index = faiss.IndexRaBitQ(d, faiss.METRIC_L2, nb_bits)
            expected_size = compute_expected_code_size(d, nb_bits)
            self.assertEqual(
                index.code_size,
                expected_size,
                f"Code size mismatch for nb_bits={nb_bits}",
            )

    def test_ivf_construction_valid_nb_bits(self):
        """Test IndexIVFRaBitQ construction with valid nb_bits."""
        d = 64
        nlist = 16
        for nb_bits in range(1, 10):
            quantizer = faiss.IndexFlat(d, faiss.METRIC_L2)
            index = faiss.IndexIVFRaBitQ(
                quantizer, d, nlist, faiss.METRIC_L2, True, nb_bits
            )
            self.assertEqual(index.rabitq.nb_bits, nb_bits)
            self.assertEqual(index.d, d)
            expected_size = compute_expected_code_size(d, nb_bits)
            self.assertEqual(index.code_size, expected_size)

    def test_ivf_construction_invalid_nb_bits(self):
        """Test that IndexIVFRaBitQ rejects invalid nb_bits."""
        d = 64
        nlist = 16
        quantizer = faiss.IndexFlat(d, faiss.METRIC_L2)

        with self.assertRaises(RuntimeError):
            faiss.IndexIVFRaBitQ(quantizer, d, nlist, faiss.METRIC_L2, True, 0)

        with self.assertRaises(RuntimeError):
            faiss.IndexIVFRaBitQ(
                quantizer, d, nlist, faiss.METRIC_L2, True, 10
            )


class TestMultiBitRaBitQBasicOperations(unittest.TestCase):
    """Test basic train/add/search operations for all combinations."""

    def do_test_basic_operations(self, metric, nb_bits, qb):
        """Test train/add/search pipeline works correctly."""
        ds = create_test_dataset(d=128, nb=500, nq=20, nt=300)
        k = 10

        # Create index with rotation
        index = create_index_rabitq_with_rotation(ds.d, metric, nb_bits, qb=qb)

        # Train
        index.train(ds.get_train())
        self.assertTrue(index.is_trained)

        # Add
        index.add(ds.get_database())
        self.assertEqual(index.ntotal, ds.nb)

        # Search
        D, I = index.search(ds.get_queries(), k)

        # Assert: Result shapes are correct
        self.assertEqual(D.shape, (ds.nq, k))
        self.assertEqual(I.shape, (ds.nq, k))

        # Assert: Indices are valid
        self.assertTrue(np.all(I >= 0))
        self.assertTrue(np.all(I < ds.nb))

        # Assert: Distances are finite
        self.assertTrue(np.all(np.isfinite(D)))


# Programmatically generate test methods for better test granularity
def add_basic_operations_test(metric, nb_bits, qb):
    """Helper to add a basic operations test method to the test class."""
    metric_str = "L2" if metric == faiss.METRIC_L2 else "IP"
    test_name = f"test_basic_ops_{metric_str}_nb{nb_bits}_qb{qb}"
    setattr(
        TestMultiBitRaBitQBasicOperations,
        test_name,
        lambda self: self.do_test_basic_operations(metric, nb_bits, qb),
    )


# Generate tests for all combinations of metric, nb_bits, and qb
for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
    for nb_bits in [1, 2, 4, 8]:
        for qb in [0, 4, 8]:
            add_basic_operations_test(metric, nb_bits, qb)


class TestMultiBitRaBitQRecall(unittest.TestCase):
    """Test search quality using recall metric."""

    def do_test_recall_quality(self, metric, nb_bits, qb):
        """Test that recall is reasonable (better than random)."""
        metric_str = 'L2' if metric == faiss.METRIC_L2 else 'IP'
        ds = create_test_dataset(
            d=128, nb=1000, nq=50, nt=500, metric=metric_str
        )
        k = 10

        # Ground truth
        I_gt = ds.get_groundtruth(10)

        # RaBitQ search
        index = create_index_rabitq_with_rotation(ds.d, metric, nb_bits, qb=qb)
        index.train(ds.get_train())
        index.add(ds.get_database())
        _, I = index.search(ds.get_queries(), k)

        # Compute recall
        recall = compute_recall_at_k(I_gt, I)

        # Assert: Recall is better than random (random ~= k/nb = 0.01)
        self.assertGreater(
            recall,
            0.10,
            f"Recall {recall:.3f} too low for metric={metric}, "
            f"nb_bits={nb_bits}, qb={qb}",
        )

    def test_recall_all_combinations(self):
        """Test recall for strategic subset of combinations."""
        # Test subset: nb_bits=[1,2,4,8], qb=[0,8]
        for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
            for nb_bits in [1, 2, 4, 8]:
                for qb in [0, 8]:
                    with self.subTest(metric=metric, nb_bits=nb_bits, qb=qb):
                        self.do_test_recall_quality(metric, nb_bits, qb)

    def do_test_recall_monotonic_improvement(self, metric, qb):
        """Test that recall improves with more bits: 2>1, 4>2, 8>4."""
        metric_str = 'L2' if metric == faiss.METRIC_L2 else 'IP'
        ds = create_test_dataset(
            d=128, nb=1000, nq=50, nt=500, metric=metric_str
        )
        k = 10

        # Ground truth
        I_gt = ds.get_groundtruth(10)

        # Test different nb_bits
        recalls = {}
        for nb_bits in [1, 2, 4, 8]:
            index = create_index_rabitq_with_rotation(
                ds.d, metric, nb_bits, qb=qb
            )
            index.train(ds.get_train())
            index.add(ds.get_database())
            _, I = index.search(ds.get_queries(), k)
            recalls[nb_bits] = compute_recall_at_k(I_gt, I)

        # Assert: Monotonic improvement with tolerance for variance
        tolerance = 0.03
        self.assertGreaterEqual(
            recalls[2],
            recalls[1] - tolerance,
            f"2-bit recall {recalls[2]:.3f} should be >= "
            f"1-bit {recalls[1]:.3f} (metric={metric}, qb={qb})",
        )
        self.assertGreaterEqual(
            recalls[4],
            recalls[2] - tolerance,
            f"4-bit recall {recalls[4]:.3f} should be >= "
            f"2-bit {recalls[2]:.3f} (metric={metric}, qb={qb})",
        )
        self.assertGreaterEqual(
            recalls[8],
            recalls[4] - tolerance,
            f"8-bit recall {recalls[8]:.3f} should be >= "
            f"4-bit {recalls[4]:.3f} (metric={metric}, qb={qb})",
        )

        # Assert: 8-bit achieves high recall
        self.assertGreater(
            recalls[8],
            0.75,
            f"8-bit recall {recalls[8]:.3f} should be > 0.75 "
            f"(metric={metric}, qb={qb})",
        )

    def test_monotonic_improvement_all_qb(self):
        """Test monotonic improvement for both metrics and qb values."""
        for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
            for qb in [0, 4, 8]:
                with self.subTest(metric=metric, qb=qb):
                    self.do_test_recall_monotonic_improvement(metric, qb)


class TestMultiBitRaBitQSerialization(unittest.TestCase):
    """Test serialization/deserialization preserves behavior."""

    def do_test_serialization(self, metric, nb_bits, qb):
        """Test that serialize/deserialize preserves search results."""
        ds = create_test_dataset(d=64, nb=200, nq=10, nt=150)
        k = 5

        # Create and populate index
        index1 = create_index_rabitq_with_rotation(
            ds.d, metric, nb_bits, qb=qb
        )
        index1.train(ds.get_train())
        index1.add(ds.get_database())

        # Search before serialization
        D1, I1 = index1.search(ds.get_queries(), k)

        # Serialize and deserialize
        index_bytes = faiss.serialize_index(index1)
        index2 = faiss.deserialize_index(index_bytes)

        # Assert: Parameters preserved
        self.assertEqual(index2.d, ds.d)
        self.assertEqual(index2.ntotal, ds.nb)
        self.assertTrue(index2.is_trained)

        # Search after deserialization using search parameters
        params = faiss.RaBitQSearchParameters()
        params.qb = qb
        params.centered = False
        D2, I2 = index2.search(ds.get_queries(), k, params=params)

        # Assert: Results are identical
        np.testing.assert_array_equal(
            I1, I2, err_msg=f"Indices mismatch for nb_bits={nb_bits}, qb={qb}"
        )
        np.testing.assert_allclose(
            D1,
            D2,
            rtol=1e-5,
            err_msg=f"Distances mismatch for nb_bits={nb_bits}, qb={qb}",
        )

    def test_serialization_all_nb_bits(self):
        """Test serialization for all nb_bits values."""
        # Test all nb_bits including edge cases (1, 9)
        for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
            for nb_bits in [1, 2, 4, 8, 9]:
                for qb in [0, 4, 8]:
                    with self.subTest(metric=metric, nb_bits=nb_bits, qb=qb):
                        self.do_test_serialization(metric, nb_bits, qb)


class TestMultiBitIndexIVFRaBitQ(unittest.TestCase):
    """Test IndexIVFRaBitQ with multi-bit support."""

    def do_test_ivf_basic_operations(self, metric, nb_bits, qb):
        """Test IVF train/add/search pipeline."""
        ds = create_test_dataset(d=128, nb=500, nq=20, nt=300)
        k = 10
        nlist = 16

        # Create IVF index with rotation
        index = create_index_ivf_rabitq_with_rotation(
            ds.d, metric, nb_bits, nlist=nlist, qb=qb, nprobe=4
        )

        # Train
        index.train(ds.get_train())
        self.assertTrue(index.is_trained)

        # Add
        index.add(ds.get_database())
        self.assertEqual(index.ntotal, ds.nb)

        # Search
        D, I = index.search(ds.get_queries(), k)

        # Assert: Result shapes are correct
        self.assertEqual(D.shape, (ds.nq, k))
        self.assertEqual(I.shape, (ds.nq, k))

        # Assert: Indices are valid
        self.assertTrue(np.all(I >= 0))
        self.assertTrue(np.all(I < ds.nb))

        # Assert: Distances are finite
        self.assertTrue(np.all(np.isfinite(D)))

    def test_ivf_all_combinations(self):
        """Test IVF for subset of combinations."""
        for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
            for nb_bits in [1, 2, 4, 8]:
                for qb in [0, 4, 8]:
                    with self.subTest(metric=metric, nb_bits=nb_bits, qb=qb):
                        self.do_test_ivf_basic_operations(metric, nb_bits, qb)

    def do_test_ivf_nprobe_improves_recall(self, metric, nb_bits):
        """Test that higher nprobe improves recall."""
        metric_str = 'L2' if metric == faiss.METRIC_L2 else 'IP'
        ds = create_test_dataset(
            d=128, nb=1000, nq=50, nt=500, metric=metric_str
        )
        k = 10
        nlist = 32

        # Ground truth
        I_gt = ds.get_groundtruth(10)

        # Create IVF index
        quantizer = faiss.IndexFlat(ds.d, metric)
        index_rbq = faiss.IndexIVFRaBitQ(
            quantizer, ds.d, nlist, metric, True, nb_bits
        )
        rrot = faiss.RandomRotationMatrix(ds.d, ds.d)
        rrot.init(123)
        index = faiss.IndexPreTransform(rrot, index_rbq)

        index.train(ds.get_train())
        index.add(ds.get_database())

        # Test different nprobe values
        recalls = {}
        for nprobe in [1, 2, 4, 8]:
            index_rbq.nprobe = nprobe
            _, I = index.search(ds.get_queries(), k)
            recalls[nprobe] = compute_recall_at_k(I_gt, I)

        # Assert: Monotonic improvement with nprobe
        self.assertGreaterEqual(recalls[2], recalls[1])
        self.assertGreaterEqual(recalls[4], recalls[2])
        self.assertGreaterEqual(recalls[8], recalls[4])

    def test_nprobe_effect(self):
        """Test nprobe effect for both metrics and selected nb_bits."""
        for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
            for nb_bits in [1, 4, 8]:
                with self.subTest(metric=metric, nb_bits=nb_bits):
                    self.do_test_ivf_nprobe_improves_recall(metric, nb_bits)

    def do_test_ivf_serialization(self, metric, nb_bits, qb):
        """Test IVF serialization preserves results."""
        ds = create_test_dataset(d=64, nb=200, nq=10, nt=150)
        k = 5
        nlist = 16

        # Create and populate IVF index
        index1 = create_index_ivf_rabitq_with_rotation(
            ds.d, metric, nb_bits, nlist=nlist, qb=qb, nprobe=4
        )
        index1.train(ds.get_train())
        index1.add(ds.get_database())

        # Search before serialization
        D1, I1 = index1.search(ds.get_queries(), k)

        # Serialize and deserialize
        index_bytes = faiss.serialize_index(index1)
        index2 = faiss.deserialize_index(index_bytes)

        # Assert: Parameters preserved
        self.assertEqual(index2.d, ds.d)
        self.assertEqual(index2.ntotal, ds.nb)
        self.assertTrue(index2.is_trained)

        # Search after deserialization using search parameters
        params = faiss.IVFRaBitQSearchParameters()
        params.qb = qb
        params.centered = False
        params.nprobe = 4
        D2, I2 = index2.search(ds.get_queries(), k, params=params)

        # Assert: Results are identical
        np.testing.assert_array_equal(I1, I2)
        np.testing.assert_allclose(D1, D2, rtol=1e-5)

    def test_ivf_serialization(self):
        """Test IVF serialization for multiple configurations."""
        for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
            for nb_bits in [1, 2, 4, 8, 9]:
                for qb in [0, 4, 8]:
                    with self.subTest(metric=metric, nb_bits=nb_bits, qb=qb):
                        self.do_test_ivf_serialization(metric, nb_bits, qb)


class TestMultiBitRaBitQQueryQuantization(unittest.TestCase):
    """Test various query quantization levels."""

    def do_test_query_quantization_levels(self, metric, nb_bits):
        """Test that all qb values produce valid results."""
        ds = create_test_dataset(d=128, nb=500, nq=20, nt=300)
        k = 10

        # Create and train index once
        index = create_index_rabitq_with_rotation(ds.d, metric, nb_bits, qb=0)
        index.train(ds.get_train())
        index.add(ds.get_database())

        # Test all qb values using search parameters
        for qb in [0, 1, 2, 4, 6, 8]:
            params = faiss.RaBitQSearchParameters()
            params.qb = qb
            params.centered = False
            D, I = index.search(ds.get_queries(), k, params=params)

            # Assert: Valid results for all qb values
            self.assertEqual(D.shape, (ds.nq, k))
            self.assertEqual(I.shape, (ds.nq, k))
            self.assertTrue(np.all(I >= 0))
            self.assertTrue(np.all(I < ds.nb))
            self.assertTrue(np.all(np.isfinite(D)))

    def test_all_qb_values(self):
        """Test all qb values for both metrics and selected nb_bits."""
        for metric in [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT]:
            for nb_bits in [1, 2, 4, 8]:
                with self.subTest(metric=metric, nb_bits=nb_bits):
                    self.do_test_query_quantization_levels(metric, nb_bits)


class TestMultiBitRaBitQIndexFactory(unittest.TestCase):
    """Test index factory support for multi-bit RaBitQ."""

    def test_factory_default_nb_bits(self):
        """Test that 'RaBitQ' creates 1-bit index by default."""
        index = faiss.index_factory(128, "RaBitQ")
        self.assertIsInstance(index, faiss.IndexRaBitQ)
        self.assertEqual(index.rabitq.nb_bits, 1)
        expected_size = compute_expected_code_size(128, 1)
        self.assertEqual(index.code_size, expected_size)

    def test_factory_multibit_specifications(self):
        """Test that 'RaBitQ{nb_bits}' creates correct multi-bit indexes."""
        d = 128
        for nb_bits in [2, 4, 8]:
            factory_str = f"RaBitQ{nb_bits}"
            index = faiss.index_factory(d, factory_str)
            self.assertIsInstance(index, faiss.IndexRaBitQ)
            self.assertEqual(index.rabitq.nb_bits, nb_bits)
            expected_size = compute_expected_code_size(d, nb_bits)
            self.assertEqual(index.code_size, expected_size)

    def test_factory_ivf_default_nb_bits(self):
        """Test that 'IVF{nlist},RaBitQ' creates 1-bit IVF index."""
        nlist = 16
        index = faiss.index_factory(128, f"IVF{nlist},RaBitQ")
        self.assertIsInstance(index, faiss.IndexIVFRaBitQ)
        self.assertEqual(index.rabitq.nb_bits, 1)
        expected_size = compute_expected_code_size(128, 1)
        self.assertEqual(index.code_size, expected_size)

    def test_factory_ivf_multibit_specifications(self):
        """Test that 'IVF{nlist},RaBitQ{nb_bits}' creates multi-bit indexes."""
        d = 128
        nlist = 16
        for nb_bits in [2, 4, 8]:
            factory_str = f"IVF{nlist},RaBitQ{nb_bits}"
            index = faiss.index_factory(d, factory_str)
            self.assertIsInstance(index, faiss.IndexIVFRaBitQ)
            self.assertEqual(index.rabitq.nb_bits, nb_bits)
            expected_size = compute_expected_code_size(d, nb_bits)
            self.assertEqual(index.code_size, expected_size)

    def test_factory_end_to_end(self):
        """Test complete workflow: factory creation, train, add, search."""
        ds = create_test_dataset(d=64, nb=200, nq=10, nt=150)
        k = 5

        # Test both non-IVF and IVF with multi-bit
        for nb_bits in [1, 4]:
            # Non-IVF
            factory_str = f"RaBitQ{nb_bits}" if nb_bits > 1 else "RaBitQ"
            index = faiss.index_factory(ds.d, factory_str)
            index.train(ds.get_train())
            index.add(ds.get_database())
            D, I = index.search(ds.get_queries(), k)

            self.assertEqual(D.shape, (ds.nq, k))
            self.assertEqual(I.shape, (ds.nq, k))
            self.assertTrue(np.all(I >= 0))
            self.assertTrue(np.all(I < ds.nb))

            # IVF
            ivf_str = (
                f"IVF16,RaBitQ{nb_bits}" if nb_bits > 1 else "IVF16,RaBitQ"
            )
            ivf_index = faiss.index_factory(ds.d, ivf_str)
            ivf_index.train(ds.get_train())
            ivf_index.add(ds.get_database())
            D_ivf, I_ivf = ivf_index.search(ds.get_queries(), k)

            self.assertEqual(D_ivf.shape, (ds.nq, k))
            self.assertEqual(I_ivf.shape, (ds.nq, k))
            self.assertTrue(np.all(I_ivf >= 0))
            self.assertTrue(np.all(I_ivf < ds.nb))
