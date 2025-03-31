# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import faiss
import numpy as np

from faiss.contrib import datasets


def random_rotation(d, seed=123):
    rs = np.random.RandomState(seed)
    Q, _ = np.linalg.qr(rs.randn(d, d))
    return Q


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
        ds = datasets.SyntheticDataset(128, 4096, 4096, 100)
        k = 10

        # PQ 8-to-1
        index_pq = faiss.IndexPQ(ds.d, 16, 8, metric_type)
        index_pq.train(ds.get_train())
        index_pq.add(ds.get_database())
        _, I_pq = index_pq.search(ds.get_queries(), k)

        index_rbq = faiss.IndexRaBitQ(ds.d, metric_type)
        index_rbq.train(ds.get_train())
        index_rbq.add(ds.get_database())
        _, I_rbq = index_rbq.search(ds.get_queries(), k)

        # try quantized query
        rbq_params = faiss.RaBitQSearchParameters(qb=8)
        _, I_rbq_q8 = index_rbq.search(ds.get_queries(), k, params=rbq_params)

        rbq_params = faiss.RaBitQSearchParameters(qb=4)
        _, I_rbq_q4 = index_rbq.search(ds.get_queries(), k, params=rbq_params)

        index_flat = faiss.IndexFlat(ds.d, metric_type)
        index_flat.train(ds.get_train())
        index_flat.add(ds.get_database())
        _, I_f = index_flat.search(ds.get_queries(), k)

        # ensure that RaBitQ and PQ are relatively close
        eval_pq = faiss.eval_intersection(I_pq[:, :k], I_f[:, :k])
        eval_pq /= ds.nq * k
        eval_rbq = faiss.eval_intersection(I_rbq[:, :k], I_f[:, :k])
        eval_rbq /= ds.nq * k
        eval_rbq_q8 = faiss.eval_intersection(I_rbq_q8[:, :k], I_f[:, :k])
        eval_rbq_q8 /= ds.nq * k
        eval_rbq_q4 = faiss.eval_intersection(I_rbq_q4[:, :k], I_f[:, :k])
        eval_rbq_q4 /= ds.nq * k

        print(
            f"PQ is {eval_pq}, "
            f"RaBitQ is {eval_rbq}, "
            f"q8 RaBitQ is {eval_rbq_q8}, "
            f"q4 RaBitQ is {eval_rbq_q4}"
        )

        np.testing.assert_(abs(eval_pq - eval_rbq) < 0.05)
        np.testing.assert_(abs(eval_pq - eval_rbq_q8) < 0.05)
        np.testing.assert_(abs(eval_pq - eval_rbq_q4) < 0.05)
        np.testing.assert_(eval_pq > 0.55)

    def test_comparison_vs_pq_L2(self):
        self.do_comparison_vs_pq_test(faiss.METRIC_L2)

    def test_comparison_vs_pq_IP(self):
        self.do_comparison_vs_pq_test(faiss.METRIC_INNER_PRODUCT)

    def test_comparison_vs_ref_L2_rrot(self, rrot_seed=123):
        ds = datasets.SyntheticDataset(128, 4096, 4096, 1)

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
        ds = datasets.SyntheticDataset(128, 4096, 4096, 1)

        ref_rbq = ReferenceRabitQ(ds.d, Bq=8)
        ref_rbq.train(ds.get_train(), np.identity(ds.d))
        ref_rbq.add(ds.get_database())

        index_rbq = faiss.IndexRaBitQ(ds.d, faiss.METRIC_L2)
        index_rbq.qb = 8
        index_rbq.train(ds.get_train())
        index_rbq.add(ds.get_database())

        ref_dis = ref_rbq.distances(ds.get_queries())

        dc = index_rbq.get_distance_computer()
        xq = ds.get_queries()

        dc.set_query(faiss.swig_ptr(xq[0]))
        for j in range(ds.nb):
            upd_dis = dc(j)
            # print(f"{j} {ref_dis[0][j]} {upd_dis}")
            np.testing.assert_(abs(ref_dis[0][j] - upd_dis) < 0.001)

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
    def test_comparison_vs_ref_L2(self):
        ds = datasets.SyntheticDataset(128, 4096, 4096, 100)

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
            Dref, Iref = ref_rbq.search(ds.get_queries(), k)
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
            Dref, Iref = ref_rbq.search(ds.get_queries(), k)
            r_ref_k = faiss.eval_intersection(
                Iref[:, :k], ds.get_groundtruth()[:, :k]
            ) / (ds.nq * k)
            print(f"{nprobe=} k-recall@10={r_ref_k}")

            params = faiss.IVFRaBitQSearchParameters()
            params.qb = index_rbq.qb
            params.nprobe = nprobe
            Dnew, Inew, stats2 = faiss.search_with_parameters(
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
