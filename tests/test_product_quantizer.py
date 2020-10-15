# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import numpy as np

import faiss
import unittest

from common import get_dataset_2




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

        # print("diff=", diff)
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
            print("Testing nbits=%d" % (i + 1))
            self.do_test_codec(i + 1)


"""
class TestPQTables(unittest.TestCase):

    def do_test(self, d, dsub, nbit, metric):
        M = d // dsub
        pq = faiss.ProductQuantizer(d, M, nbit)
        pq.train(faiss.randn((1000, d), 123))

        assert pq.dsub == 2
        nx = 100
        x = faiss.randn((nx, d), 555)
        sp = faiss.swig_ptr
        ref_tab = np.zeros((nx, M, pq.ksub), "float32")

        new_tab = fast_scan.AlignedTableFloat32(nx * M * pq.ksub)
        # new_tab = np.zeros((nx, M, pq.ksub), "float32")
        pq.compute_inner_prod_tables(nx, sp(x), sp(ref_tab))

        fast_scan.compute_inner_prod_tables(pq, nx, sp(x), new_tab.get())

        new_tab = fast_scan.AlignedTable_to_array(new_tab)
        new_tab = new_tab.reshape(nx, M, pq.ksub)

        np.testing.assert_array_equal(ref_tab, new_tab)
"""

