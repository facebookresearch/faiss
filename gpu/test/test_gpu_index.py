# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD+Patents license found in the
# LICENSE file in the root directory of this source tree.

#! /usr/bin/env python2

import time
import unittest
import numpy as np
import faiss


class EvalIVFPQAccuracy(unittest.TestCase):

    def get_dataset(self, small_one=False):
        if not small_one:
            d = 128
            nb = 100000
            nt = 15000
            nq = 2000
        else:
            d = 32
            nb = 1000
            nt = 1000
            nq = 200
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


    def test_mm(self):
        # trouble with MKL+fbmake that appears only at runtime. Check it here
        x = np.random.random(size=(100, 20)).astype('float32')
        mat = faiss.PCAMatrix(20, 10)
        mat.train(x)
        mat.apply_py(x)

    def do_cpu_to_gpu(self, index_key):
        ts = []
        ts.append(time.time())
        (xt, xb, xq) = self.get_dataset(small_one=True)
        nb, d = xb.shape

        index = faiss.index_factory(d, index_key)
        if index.__class__ == faiss.IndexIVFPQ:
            # speed up test
            index.pq.cp.niter = 2
            index.do_polysemous_training = False
        ts.append(time.time())

        index.train(xt)
        ts.append(time.time())

        # adding some ids because there was a bug in this case
        index.add_with_ids(xb, np.arange(nb) * 3 + 12345)
        ts.append(time.time())

        index.nprobe = 4
        D, Iref = index.search(xq, 10)
        ts.append(time.time())

        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        ts.append(time.time())

        gpu_index.setNumProbes(4)

        D, Inew = gpu_index.search(xq, 10)
        ts.append(time.time())
        print 'times:', [t - ts[0] for t in ts]

        self.assertGreaterEqual((Iref == Inew).sum(), Iref.size)

        if faiss.get_num_gpus() == 1:
            return

        for shard in False, True:

            # test on just 2 GPUs
            res = [faiss.StandardGpuResources() for i in range(2)]
            co = faiss.GpuMultipleClonerOptions()
            co.shard = shard

            gpu_index = faiss.index_cpu_to_gpu_multiple_py(res, index, co)

            faiss.GpuParameterSpace().set_index_parameter(
                gpu_index, 'nprobe', 4)

            D, Inew = gpu_index.search(xq, 10)

            self.assertGreaterEqual((Iref == Inew).sum(), Iref.size)

    def test_cpu_to_gpu_IVFPQ(self):
        self.do_cpu_to_gpu('IVF128,PQ4')

    def test_cpu_to_gpu_IVFFlat(self):
        self.do_cpu_to_gpu('IVF128,Flat')

    def test_set_gpu_param(self):
        index = faiss.index_factory(12, "PCAR8,IVF10,PQ4")
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        faiss.GpuParameterSpace().set_index_parameter(gpu_index, "nprobe", 3)




if __name__ == '__main__':
    unittest.main()
