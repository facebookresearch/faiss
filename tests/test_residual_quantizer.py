# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import faiss
import unittest

from faiss.contrib import datasets


def pairwise_distances(a, b):
    anorms = (a ** 2).sum(1)
    bnorms = (b ** 2).sum(1)
    return anorms.reshape(-1, 1) + bnorms - 2 * a @ b.T


def beam_search_encode_step_ref(cent, residuals, codes, L):
    """ Reference beam search implementation
    encodes a residual table.
    """
    K, d = cent.shape
    n, beam_size, d2 = residuals.shape
    assert d == d2
    n2, beam_size_2, m = codes.shape
    assert n2 == n and beam_size_2 == beam_size

    # compute all possible new residuals
    cent_distances = pairwise_distances(residuals.reshape(n * beam_size, d), cent)
    cent_distances = cent_distances.reshape(n, beam_size, K)

    # TODO write in vector form

    if beam_size * K <= L:
        # then keep all the results
        new_beam_size = beam_size * K
        new_codes = np.zeros((n, beam_size, K, m + 1), dtype=int)
        new_residuals = np.zeros((n, beam_size, K, d), dtype='float32')
        for i in range(n):
            new_codes[i, :, :, :-1] = codes[i]
            new_codes[i, :, :, -1] = np.arange(K)
            new_residuals[i] = residuals[i].reshape(1, d) - cent.reshape(K, d)
        new_codes = new_codes.reshape(n, new_beam_size, m + 1)
        new_residuals = new_residuals.reshape(n, new_beam_size, d)
        new_distances = cent_distances.reshape(n, new_beam_size)
    else:
        # keep top-L results
        new_beam_size = L
        new_codes = np.zeros((n, L, m + 1), dtype=int)
        new_residuals = np.zeros((n, L, d), dtype='float32')
        new_distances = np.zeros((n, L), dtype='float32')
        for i in range(n):
            cd = cent_distances[i].ravel()
            jl = np.argsort(cd)[:L]    # TODO argpartition
            js = jl // K     # input beam index
            ls = jl % K      # centroid index
            new_codes[i, :, :-1] = codes[i, js, :]
            new_codes[i, :, -1] = ls
            new_residuals[i, :, :] = residuals[i, js, :] - cent[ls, :]
            new_distances[i, :] = cd[jl]

    return new_codes, new_residuals, new_distances


def beam_search_encode_step(cent, residuals, codes, L, assign_index=None):
    """ Wrapper of the C++ function with the same interface """
    K, d = cent.shape
    n, beam_size, d2 = residuals.shape
    assert d == d2
    n2, beam_size_2, m = codes.shape
    assert n2 == n and beam_size_2 == beam_size

    assert L <= beam_size * K

    new_codes = np.zeros((n, L, m + 1), dtype='int32')
    new_residuals = np.zeros((n, L, d), dtype='float32')
    new_distances = np.zeros((n, L), dtype='float32')

    sp = faiss.swig_ptr
    codes = np.ascontiguousarray(codes, dtype='int32')
    faiss.beam_search_encode_step(
        d, K, sp(cent), n, beam_size, sp(residuals),
        m, sp(codes), L, sp(new_codes), sp(new_residuals), sp(new_distances),
        assign_index
    )

    return new_codes, new_residuals, new_distances


class TestBeamSearch(unittest.TestCase):

    def do_test(self, K=70, L=10, use_assign_index=False):
        """ compare C++ beam search with reference python implementation """
        d = 32
        n = 500
        L = 10 # beam size

        rs = np.random.RandomState(123)
        x = rs.rand(n, d).astype('float32')

        cent = rs.rand(K, d).astype('float32')

        # first quant step --> input beam size is 1
        codes = np.zeros((n, 1, 0), dtype=int)
        residuals = x.reshape(n, 1, d)

        assign_index = faiss.IndexFlatL2(d) if use_assign_index else None

        ref_codes, ref_residuals, ref_distances = beam_search_encode_step_ref(
            cent, residuals, codes, L
        )

        new_codes, new_residuals, new_distances = beam_search_encode_step(
            cent, residuals, codes, L, assign_index
        )

        np.testing.assert_array_equal(new_codes, ref_codes)
        np.testing.assert_array_equal(new_residuals, ref_residuals)
        np.testing.assert_allclose(new_distances, ref_distances, rtol=1e-5)

        # second quant step:
        K = 50
        cent = rs.rand(K, d).astype('float32')

        codes, residuals = ref_codes, ref_residuals

        ref_codes, ref_residuals, ref_distances = beam_search_encode_step_ref(
            cent, residuals, codes, L
        )

        new_codes, new_residuals, new_distances = beam_search_encode_step(
            cent, residuals, codes, L
        )

        np.testing.assert_array_equal(new_codes, ref_codes)
        np.testing.assert_array_equal(new_residuals, ref_residuals)
        np.testing.assert_allclose(new_distances, ref_distances, rtol=1e-5)

    def test_beam_search(self):
        self.do_test()

    def test_beam_search_assign_index(self):
        self.do_test(use_assign_index=True)

    def test_small_beam(self):
        self.do_test(L=1)

    def test_small_beam_2(self):
        self.do_test(L=2)


def eval_codec(q, xb):
    codes = q.compute_codes(xb)
    decoded = q.decode(codes)
    return ((xb - decoded) ** 2).sum()


class TestResidualQuantizer(unittest.TestCase):

    def test_training(self):
        """check that the error is in the same ballpark as PQ """
        ds = datasets.SyntheticDataset(32, 3000, 1000, 0)

        xt = ds.get_train()
        xb = ds.get_database()

        rq = faiss.ResidualQuantizer(ds.d, 4, 6)
        rq.verbose
        rq.verbose = True
        #
        rq.train_type = faiss.ResidualQuantizer.Train_default
        rq.cp.verbose
        # rq.cp.verbose = True
        rq.train(xt)
        err_rq = eval_codec(rq, xb)

        pq = faiss.ProductQuantizer(ds.d, 4, 6)
        pq.train(xt)
        err_pq = eval_codec(pq, xb)

        # in practice RQ is often better than PQ but it does not the case here, so just check
        # that we are within some factor.
        print(err_pq, err_rq)
        self.assertLess(err_rq, err_pq * 1.2)

    def test_beam_size(self):
        """ check that a larger beam gives a lower error """
        ds = datasets.SyntheticDataset(32, 3000, 1000, 0)

        xt = ds.get_train()
        xb = ds.get_database()

        rq0 = faiss.ResidualQuantizer(ds.d, 4, 6)
        rq0.train_type = faiss.ResidualQuantizer.Train_default
        rq0.max_beam_size = 2
        rq0.train(xt)
        err_rq0 = eval_codec(rq0, xb)

        rq1 = faiss.ResidualQuantizer(ds.d, 4, 6)
        rq1.train_type = faiss.ResidualQuantizer.Train_default
        rq1.max_beam_size = 10
        rq1.train(xt)
        err_rq1 = eval_codec(rq1, xb)

        self.assertLess(err_rq1, err_rq0)


class TestIndexResidual(unittest.TestCase):

    def test_io(self):
        ds = datasets.SyntheticDataset(32, 1000, 100, 0)

        xt = ds.get_train()
        xb = ds.get_database()

        ir = faiss.IndexResidual(ds.d, 3, 4)
        ir.rq.train_type = faiss.ResidualQuantizer.Train_default
        ir.train(xt)
        ref_codes = ir.sa_encode(xb)

        b = faiss.serialize_index(ir)
        ir2 = faiss.deserialize_index(b)
        codes2 = ir2.sa_encode(xb)

        np.testing.assert_array_equal(ref_codes, codes2)

    def test_factory(self):

        index = faiss.index_factory(5, "RQ2x16_3x8_6x4")

        np.testing.assert_array_equal(
            faiss.vector_to_array(index.rq.nbits),
            np.array([16, 16, 8, 8, 8, 4, 4, 4, 4, 4, 4])
        )

    def test_search_decompress(self):
        ds = datasets.SyntheticDataset(32, 1000, 1000, 100)

        xt = ds.get_train()
        xb = ds.get_database()

        ir = faiss.IndexResidual(ds.d, 3, 4)
        ir.rq.train_type = faiss.ResidualQuantizer.Train_default
        ir.train(xt)
        ir.add(xb)

        D, I = ir.search(ds.get_queries(), 10)
        gt = ds.get_groundtruth()

        recalls = {
            rank: (I[:, :rank] == gt[:, :1]).sum() / len(gt)
            for rank in [1, 10, 100]
        }
        # recalls are {1: 0.05, 10: 0.37, 100: 0.37}
        self.assertGreater(recalls[10], 0.35)


class TestIVFResidualCoarseQuantizer(unittest.TestCase):

    def test_IVF_resiudal(self):
        ds = datasets.SyntheticDataset(32, 3000, 1000, 100)

        xt = ds.get_train()
        xb = ds.get_database()

        gt = ds.get_groundtruth(1)

        # RQ 2x6 = 12 bits = 4096 centroids
        quantizer = faiss.ResidualCoarseQuantizer(ds.d, 2, 6)
        rq = quantizer.rq
        rq.train_type = faiss.ResidualQuantizer.Train_default
        index = faiss.IndexIVFFlat(quantizer, ds.d, 1 << rq.tot_bits)
        index.quantizer_trains_alone
        index.quantizer_trains_alone = True

        index.train(xt)
        index.add(xb)

        # make sure that increasing the nprobe increases accuracy

        index.nprobe = 10
        D, I = index.search(ds.get_queries(), 10)
        r10 = (I == gt[None, :]).sum() / ds.nq

        index.nprobe = 40
        D, I = index.search(ds.get_queries(), 10)
        r40 = (I == gt[None, :]).sum() / ds.nq

        self.assertGreater(r40, r10)

        # make sure that decreasing beam factor decreases accuracy
        quantizer.beam_factor
        quantizer.beam_factor = 1.0
        index.nprobe = 10
        D, I = index.search(ds.get_queries(), 10)
        r10_narrow_beam = (I == gt[None, :]).sum() / ds.nq

        self.assertGreater(r10, r10_narrow_beam)

    def test_factory(self):
        ds = datasets.SyntheticDataset(16, 500, 1000, 100)

        index = faiss.index_factory(ds.d, "IVF1024(RCQ2x5),Flat")
        index.train(ds.get_train())
        index.add(ds.get_database())

        Dref, Iref = index.search(ds.get_queries(), 10)

        b = faiss.serialize_index(index)
        index2 = faiss.deserialize_index(b)

        Dnew, Inew = index2.search(ds.get_queries(), 10)

        np.testing.assert_equal(Dref, Dnew)
        np.testing.assert_equal(Iref, Inew)

    def test_ivfsq(self):
        ds = datasets.SyntheticDataset(32, 3000, 1000, 100)

        xt = ds.get_train()
        xb = ds.get_database()

        gt = ds.get_groundtruth(1)

        # RQ 2x5 = 10 bits = 1024 centroids
        index = faiss.index_factory(ds.d, "IVF1024(RCQ2x5),SQ8")
        quantizer = faiss.downcast_index(index.quantizer)
        rq = quantizer.rq
        rq.train_type = faiss.ResidualQuantizer.Train_default

        index.train(xt)
        index.add(xb)

        # make sure that increasing the nprobe increases accuracy

        index.nprobe = 10
        D, I = index.search(ds.get_queries(), 10)
        r10 = (I == gt[None, :]).sum() / ds.nq

        index.nprobe = 40
        D, I = index.search(ds.get_queries(), 10)
        r40 = (I == gt[None, :]).sum() / ds.nq

        self.assertGreater(r40, r10)

    def test_rcq_LUT(self):
        ds = datasets.SyntheticDataset(32, 3000, 1000, 100)

        xt = ds.get_train()
        xb = ds.get_database()

        # RQ 2x5 = 10 bits = 1024 centroids
        index = faiss.index_factory(ds.d, "IVF1024(RCQ2x5),SQ8")

        quantizer = faiss.downcast_index(index.quantizer)
        rq = quantizer.rq
        rq.train_type = faiss.ResidualQuantizer.Train_default

        index.train(xt)
        index.add(xb)
        index.nprobe = 10

        # set exact centroids as coarse quantizer
        all_centroids = quantizer.reconstruct_n(0, quantizer.ntotal)
        q2 = faiss.IndexFlatL2(32)
        q2.add(all_centroids)
        index.quantizer = q2
        Dref, Iref = index.search(ds.get_queries(), 10)
        index.quantizer = quantizer

        # search with LUT
        quantizer.set_beam_factor(-1)
        Dnew, Inew = index.search(ds.get_queries(), 10)

        np.testing.assert_array_almost_equal(Dref, Dnew, decimal=5)
        np.testing.assert_array_equal(Iref, Inew)


class TestAdditiveQuantizerWithLUT(unittest.TestCase):

    def test_RCQ_knn(self):
        ds = datasets.SyntheticDataset(32, 1000, 0, 123)
        xt = ds.get_train()
        xq = ds.get_queries()

        # RQ 3+4+5 = 12 bits = 4096 centroids
        rcq = faiss.index_factory(ds.d, "RCQ1x3_1x4_1x5")
        rcq.train(xt)

        aq = rcq.rq

        cents = rcq.reconstruct_n(0, rcq.ntotal)

        sp = faiss.swig_ptr

        # test norms computation

        norms_ref = (cents ** 2).sum(1)
        norms = np.zeros(1 << aq.tot_bits, dtype="float32")
        aq.compute_centroid_norms(sp(norms))

        np.testing.assert_array_almost_equal(norms, norms_ref, decimal=5)

        # test IP search

        Dref, Iref = faiss.knn(
            xq, cents, 10,
            metric=faiss.METRIC_INNER_PRODUCT
        )

        Dnew = np.zeros_like(Dref)
        Inew = np.zeros_like(Iref)

        aq.knn_exact_inner_product(len(xq), sp(xq), 10, sp(Dnew), sp(Inew))

        np.testing.assert_array_almost_equal(Dref, Dnew, decimal=5)
        np.testing.assert_array_equal(Iref, Inew)

        # test L2 search

        Dref, Iref = faiss.knn(xq, cents, 10, metric=faiss.METRIC_L2)

        Dnew = np.zeros_like(Dref)
        Inew = np.zeros_like(Iref)

        aq.knn_exact_L2(len(xq), sp(xq), 10, sp(Dnew), sp(Inew), sp(norms))

        np.testing.assert_array_equal(Iref, Inew)
        np.testing.assert_array_almost_equal(Dref, Dnew, decimal=5)
