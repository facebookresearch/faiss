# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD+Patents license found in the
# LICENSE file in the root directory of this source tree.

#! /usr/bin/env python2

import libfb.py.mkl  # noqa

import numpy as np

from libfb import testutil

import faiss


class EvalIVFPQAccuracy(testutil.BaseFacebookTestCase):

    def get_dataset(self):
        d = 128
        nb = 100000
        nt = 15000
        nq = 2000
        np.random.seed(123)

        # generate points in a low-dim subspace to make the resutls
        # look better :-)
        d1 = 16
        q, r = np.linalg.qr(np.random.randn(d, d))
        qc = q[:d1, :]
        def make_mat(n):
            return np.dot(
                np.random.random(size=(nb, d1)), qc).astype('float32')

        return (make_mat(nt), make_mat(nb), make_mat(nq))

    def test_IndexIVFPQ(self):
        (xt, xb, xq) = self.get_dataset()
        d = xt.shape[1]

        dev_no = 0
        usePrecomputed = True

        res = faiss.StandardGpuResources()

        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = dev_no

        gt_index = faiss.GpuIndexFlatL2(res, d, flat_config)
        gt_index.add(xb)
        D, gt_nns = gt_index.search(xq, 1)

        coarse_quantizer = faiss.IndexFlatL2(d)
        ncentroids = int(np.sqrt(xb.shape[0])) * 4

        index = faiss.IndexIVFPQ(coarse_quantizer, d, ncentroids, 32, 8)
        # add implemented on GPU but not train
        index.train(xt)

        ivfpq_config = faiss.GpuIndexIVFPQConfig()
        ivfpq_config.device = dev_no
        ivfpq_config.usePrecomputedTables = usePrecomputed

        gpuIndex = faiss.GpuIndexIVFPQ(res, index, ivfpq_config)
        gpuIndex.setNumProbes(64)
        index.add(xb)

        D, nns = index.search(xq, 10)
        n_ok = (nns == gt_nns).sum()
        nq = xq.shape[0]
        print ncentroids, n_ok, nq

        self.assertGreater(n_ok, nq * 0.2)

    def test_mm(self):
        # trouble with MKL+fbmake that appears only at runtime. Check it here
        x = np.random.random(size=(100, 20)).astype('float32')
        mat = faiss.PCAMatrix(20, 10)
        mat.train(x)
        mat.apply_py(x)
