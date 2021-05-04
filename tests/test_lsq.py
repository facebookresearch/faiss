# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import faiss
import unittest

from faiss.contrib import datasets


def construct_sparse_matrix(codes, K):
    n, M = codes.shape
    B = np.zeros((n, M * K), dtype=np.float32)
    for i in range(n):
        for j in range(M):
            code = codes[i, j]
            B[i, j * K + code] = 1
    return B


def update_codebooks_ref(x, codes, K, lambd):
    n, d = x.shape
    M = codes.shape[1]

    B = construct_sparse_matrix(codes, K)
    reg = np.identity(M * K) * float(lambd)
    reg = reg.astype(np.float32)

    # C = (B'B + lambd * I)^(-1)B'X
    bb = np.linalg.inv(B.T @ B + reg)
    C = bb @ B.T @ x
    C = C.reshape(M, K, d)

    return C


class TestUpdateCodebooks(unittest.TestCase):

    def test_update_codebooks(self):
        d = 32
        n = 1000
        M = 4
        nbits = 8
        K = (1 << nbits)

        # set a larger value to make the updating process more stable
        lambd = 1e-2

        rs = np.random.RandomState(123)
        x = rs.rand(n, d).astype(np.float32)
        codes = rs.randint(0, K, (n, M)).astype(np.int32)

        lsq = faiss.LocalSearchQuantizer(d, M, nbits)
        lsq.lambd = lambd
        lsq.train(x)  # just for allocating memory for codebooks

        codebooks = faiss.vector_float_to_array(lsq.codebooks)
        codebooks = codebooks.reshape(M, K, d).copy()

        lsq.update_codebooks(faiss.swig_ptr(x), faiss.swig_ptr(codes), n)
        new_codebooks = faiss.vector_float_to_array(lsq.codebooks)
        new_codebooks = new_codebooks.reshape(M, K, d).copy()

        ref_codebooks = update_codebooks_ref(x, codes, K, lambd)

        np.testing.assert_allclose(new_codebooks, ref_codebooks, atol=1e-3)


def eval_codec(q, xb):
    codes = q.compute_codes(xb)
    decoded = q.decode(codes)
    return ((xb - decoded) ** 2).sum()


class TestLocalSearchQuantizer(unittest.TestCase):

    def test_training(self):
        """check that the error is in the same ballpark as PQ."""
        ds = datasets.SyntheticDataset(32, 3000, 3000, 0)

        xt = ds.get_train()
        xb = ds.get_database()

        M = 4
        nbits = 4

        lsq = faiss.LocalSearchQuantizer(ds.d, M, nbits)
        lsq.train(xt)
        err_lsq = eval_codec(lsq, xb)

        pq = faiss.ProductQuantizer(ds.d, M, nbits)
        pq.train(xt)
        err_pq = eval_codec(pq, xb)

        rq = faiss.ResidualQuantizer(ds.d, M, nbits)
        rq.train(xt)
        err_rq = eval_codec(rq, xb)

        print(err_lsq, err_pq, err_rq)
        self.assertLess(err_lsq, err_pq)
        self.assertLess(err_lsq, err_rq)
