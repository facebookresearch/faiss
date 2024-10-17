# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import numpy as np

import faiss
import unittest



class TestProductQuantizer(unittest.TestCase):

    def test_pq(self):
        d = 64
        n = 2000
        cs = 4
        np.random.seed(123)
        x = np.random.random(size=(n, d)).astype('float32')
        pq = faiss.ProductQuantizer(d, cs, 8)
        pq.train(x)
        codes = pq.compute_codes(x)
        x2 = pq.decode(codes)
        diff = ((x - x2)**2).sum()

        # diff= 4418.0562
        self.assertGreater(5000, diff)

        pq10 = faiss.ProductQuantizer(d, cs, 10)
        assert pq10.code_size == 5
        pq10.verbose = True
        pq10.cp.verbose = True
        pq10.train(x)
        codes = pq10.compute_codes(x)

        x10 = pq10.decode(codes)
        diff10 = ((x - x10)**2).sum()
        self.assertGreater(diff, diff10)

    def do_test_codec(self, nbit):
        pq = faiss.ProductQuantizer(16, 2, nbit)

        # simulate training
        rs = np.random.RandomState(123)
        centroids = rs.rand(2, 1 << nbit, 8).astype('float32')
        faiss.copy_array_to_vector(centroids.ravel(), pq.centroids)

        idx = rs.randint(1 << nbit, size=(100, 2))
        # can be encoded exactly
        x = np.hstack((
            centroids[0, idx[:, 0]],
            centroids[1, idx[:, 1]]
        ))

        # encode / decode
        codes = pq.compute_codes(x)
        xr = pq.decode(codes)
        assert np.all(xr == x)

        # encode w/ external index
        assign_index = faiss.IndexFlatL2(8)
        pq.assign_index = assign_index
        codes2 = np.empty((100, pq.code_size), dtype='uint8')
        pq.compute_codes_with_assign_index(
            faiss.swig_ptr(x), faiss.swig_ptr(codes2), 100)
        assert np.all(codes == codes2)

    def test_codec(self):
        for i in range(16):
            self.do_test_codec(i + 1)


class TestPQTransposedCentroids(unittest.TestCase):

    def do_test(self, d, dsub):
        M = d // dsub
        pq = faiss.ProductQuantizer(d, M, 8)
        xt = faiss.randn((max(1000, pq.ksub * 50), d), 123)
        pq.cp.niter = 4    # to avoid timeouts in tests
        pq.train(xt)

        codes = pq.compute_codes(xt)

        # enable transposed centroids table to speedup compute_codes()
        pq.sync_transposed_centroids()
        codes_transposed = pq.compute_codes(xt)

        # disable transposed centroids table
        pq.clear_transposed_centroids()
        codes_cleared = pq.compute_codes(xt)

        assert np.all(codes == codes_transposed)
        assert np.all(codes == codes_cleared)

    def test_dsub2(self):
        self.do_test(16, 2)

    def test_dsub5(self):
        self.do_test(20, 5)

    def test_dsub2_odd(self):
        self.do_test(18, 2)

    def test_dsub4(self):
        self.do_test(32, 4)

    def test_dsub4_odd(self):
        self.do_test(36, 4)


class TestPQTables(unittest.TestCase):

    def do_test(self, d, dsub, nbit=8, metric=None):
        if metric is None:
            self.do_test(d, dsub, nbit, faiss.METRIC_INNER_PRODUCT)
            self.do_test(d, dsub, nbit, faiss.METRIC_L2)
            return
        # faiss.cvar.distance_compute_blas_threshold = 1000000

        M = d // dsub
        pq = faiss.ProductQuantizer(d, M, nbit)
        xt = faiss.randn((max(1000, pq.ksub * 50), d), 123)
        pq.cp.niter = 4    # to avoid timeouts in tests
        pq.train(xt)

        centroids = faiss.vector_to_array(pq.centroids)
        centroids = centroids.reshape(pq.M, pq.ksub, pq.dsub)

        nx = 100
        x = faiss.randn((nx, d), 555)

        ref_tab = np.zeros((nx, M, pq.ksub), "float32")

        # computation of tables in numpy
        for sq in range(M):
            i0, i1 = sq * dsub, (sq + 1) * dsub
            xsub = x[:, i0:i1]
            centsq = centroids[sq, :, :]
            if metric == faiss.METRIC_INNER_PRODUCT:
                ref_tab[:, sq, :] = xsub @ centsq.T
            elif metric == faiss.METRIC_L2:
                xsub3 = xsub.reshape(nx, 1, dsub)
                cent3 = centsq.reshape(1, pq.ksub, dsub)
                ref_tab[:, sq, :] = ((xsub3 - cent3) ** 2).sum(2)
            else:
                assert False

        sp = faiss.swig_ptr

        new_tab = np.zeros((nx, M, pq.ksub), "float32")
        if metric == faiss.METRIC_INNER_PRODUCT:
            pq.compute_inner_prod_tables(nx, sp(x), sp(new_tab))
        elif metric == faiss.METRIC_L2:
            pq.compute_distance_tables(nx, sp(x), sp(new_tab))
        else:
            assert False

        # compute sdc tables in numpy
        cent1 = np.expand_dims(centroids, axis=2)  # [M, ksub, 1, dsub]
        cent2 = np.expand_dims(centroids, axis=1)  # [M, 1, ksub, dsub]
        ref_sdc_tab = ((cent1 - cent2) ** 2).sum(3)

        pq.compute_sdc_table()
        new_sdc_tab = faiss.vector_to_array(pq.sdc_table)
        new_sdc_tab = new_sdc_tab.reshape(M, pq.ksub, pq.ksub)

        np.testing.assert_array_almost_equal(ref_tab, new_tab, decimal=5)
        np.testing.assert_array_almost_equal(ref_sdc_tab, new_sdc_tab, decimal=5)

    def test_dsub2(self):
        self.do_test(16, 2)

    def test_dsub5(self):
        self.do_test(20, 5)

    def test_dsub2_odd(self):
        self.do_test(18, 2)

    def test_dsub4(self):
        self.do_test(32, 4)

    def test_dsub4_odd(self):
        self.do_test(36, 4)

    # too slow
    #def test_12bit(self):
    #    self.do_test(32, 4, nbit=12)

    def test_4bit(self):
        self.do_test(32, 4, nbit=4)
