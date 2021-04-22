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
