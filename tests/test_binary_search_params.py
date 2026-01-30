# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import unittest
import numpy as np
import faiss


class TestBinarySearchParams(unittest.TestCase):

    def do_test_with_param(
            self, index, ps_params, params, d=64, nb=500, nq=20, k=5):
        """
        Test equivalence between setting
        1. param_name = value with ParameterSpace
        2. pass in a SearchParameters with param_name = value
        """
        rs = np.random.RandomState(123)
        xb = rs.randint(256, size=(nb, d // 8), dtype="uint8")
        xq = rs.randint(256, size=(nq, d // 8), dtype="uint8")
        if hasattr(index, "is_trained") and not index.is_trained:
            xt = rs.randint(256, size=(nb, d // 8), dtype="uint8")
            index.train(xt)

        index.add(xb)

        I0, D0 = index.search(xq, k)

        Dnew, Inew = index.search(xq, k, params=params)

        # make sure the parameter does indeed change the result...
        self.assertFalse(np.all(Inew == I0))

        for param_name, value in ps_params.items():
            faiss.ParameterSpace().set_index_parameter(
                index, param_name, value)
        Dref, Iref = index.search(xq, k)

        np.testing.assert_array_equal(Iref, Inew)
        np.testing.assert_array_equal(Dref, Dnew)

    def test_nprobe(self):
        d = 64
        quantizer = faiss.IndexBinaryFlat(d)
        index = faiss.IndexBinaryIVF(quantizer, d, 32)
        self.do_test_with_param(
                index,
                {"nprobe": 3},
                faiss.SearchParametersIVF(nprobe=3),
                d=d, nb=500, nq=20, k=5)

    def test_efSearch(self):
        d = 64
        index = faiss.IndexBinaryHNSW(d, 32)
        self.do_test_with_param(
            index,
            {"efSearch": 4},
            faiss.SearchParametersHNSW(efSearch=4),
            d=d, nb=500, nq=20, k=5)

    def test_quantizer_hnsw(self):
        d = 64
        quantizer = faiss.IndexBinaryHNSW(d, 32)
        index = faiss.IndexBinaryIVF(quantizer, d, 32)
        self.do_test_with_param(
            index,
            {"quantizer_efSearch": 5, "nprobe": 10},
            faiss.SearchParametersIVF(
                nprobe=10,
                quantizer_params=faiss.SearchParametersHNSW(
                    efSearch=5)
            ),
            d=d, nb=500, nq=20, k=5)

    def test_max_codes(self):
        d = 64
        nb = 1000
        nq = 20
        k = 10

        rs = np.random.RandomState(123)
        xb = rs.randint(256, size=(nb, d // 8), dtype="uint8")
        xq = rs.randint(256, size=(nq, d // 8), dtype="uint8")

        quantizer = faiss.IndexBinaryFlat(d)
        index = faiss.IndexBinaryIVF(quantizer, d, 32)
        index.train(xb)
        index.add(xb)

        stats = faiss.cvar.indexIVF_stats
        stats.reset()
        D0, I0 = index.search(
            xq, k,
            params=faiss.SearchParametersIVF(nprobe=8)
        )
        ndis0 = stats.ndis
        target_ndis = ndis0 // nq
        for q in range(nq):
            stats.reset()
            Dq, Iq = index.search(
                xq[q:q + 1], k,
                params=faiss.SearchParametersIVF(
                    nprobe=8, max_codes=target_ndis
                )
            )
            self.assertLessEqual(stats.ndis, target_ndis)
            if stats.ndis < target_ndis:
                np.testing.assert_equal(I0[q], Iq[0])

    def test_efSearch(self):
        d = 64
        nb = 500
        nq = 20
        k = 5

        rs = np.random.RandomState(123)
        xb = rs.randint(256, size=(nb, d // 8), dtype="uint8")
        xq = rs.randint(256, size=(nq, d // 8), dtype="uint8")

        index = faiss.IndexBinaryHNSW(d, 32)
        index.add(xb)

        D1, I1 = index.search(
            xq, k,
            params=faiss.SearchParametersHNSW(efSearch=4)
        )

        D2, I2 = index.search(
            xq, k,
            params=faiss.SearchParametersHNSW(efSearch=16)
        )

        self.assertFalse(np.all(I1 == I2))

    def test_nprobe(self):
        d = 64
        nb = 500
        nq = 20
        k = 5

        rs = np.random.RandomState(123)
        xb = rs.randint(256, size=(nb, d // 8), dtype="uint8")
        xq = rs.randint(256, size=(nq, d // 8), dtype="uint8")

        quantizer = faiss.IndexBinaryFlat(d)
        index = faiss.IndexBinaryIVF(quantizer, d, 32)
        index.train(xb)
        index.add(xb)

        D1, I1 = index.search(
            xq, k,
            params=faiss.SearchParametersIVF(nprobe=1)
        )

        D2, I2 = index.search(
            xq, k,
            params=faiss.SearchParametersIVF(nprobe=8)
        )

        self.assertFalse(np.all(I1 == I2))
