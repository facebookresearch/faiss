# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import unittest
import numpy as np
import faiss

from faiss.contrib.datasets import SyntheticDataset
from faiss.contrib.evaluation import check_ref_knn_with_draws

class TestShardedFlat(unittest.TestCase):

    @unittest.skipIf(faiss.get_num_gpus() < 2, "multiple GPU only test")
    def test_sharded(self):
        d = 32
        nb = 1000
        nq = 200
        k = 10
        rs = np.random.RandomState(123)
        xb = rs.rand(nb, d).astype('float32')
        xq = rs.rand(nq, d).astype('float32')

        index_cpu = faiss.IndexFlatL2(d)

        assert faiss.get_num_gpus() > 1

        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.use_raft = False
        index = faiss.index_cpu_to_all_gpus(index_cpu, co, ngpu=2)

        index.add(xb)
        D, I = index.search(xq, k)

        index_cpu.add(xb)
        D_ref, I_ref = index_cpu.search(xq, k)

        assert np.all(I == I_ref)

        del index
        index2 = faiss.index_cpu_to_all_gpus(index_cpu, co, ngpu=2)
        D2, I2 = index2.search(xq, k)

        assert np.all(I2 == I_ref)

        try:
            index2.add(xb)
        except RuntimeError:
            pass
        else:
            raise AssertionError("errpr: call should fail but isn't failing")

    @unittest.skipIf(faiss.get_num_gpus() < 2, "multiple GPU only test")
    def do_test_sharded_ivf(self, index_key):
        ds = SyntheticDataset(32, 8000, 10000, 100)
        index = faiss.index_factory(ds.d, index_key)
        if 'HNSW' in index_key:
            # make a bit more reproducible...
            faiss.ParameterSpace().set_index_parameter(
                index, 'quantizer_efSearch', 40)
        index.train(ds.get_train())
        index.add(ds.get_database())
        Dref, Iref = index.search(ds.get_queries(), 10)
        index.nprobe = 8
        Dref8, Iref8 = index.search(ds.get_queries(), 10)
        index.nprobe = 1
        print("REF checksum", faiss.checksum(Iref))

        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.common_ivf_quantizer = True
        co.use_raft = False
        index = faiss.index_cpu_to_all_gpus(index, co, ngpu=2)

        index.quantizer  # make sure there is indeed a quantizer
        print("QUANT", faiss.downcast_index(index.quantizer))
        Dnew, Inew = index.search(ds.get_queries(), 10)
        np.testing.assert_array_equal(Iref, Inew)
        np.testing.assert_array_almost_equal(Dref, Dnew, decimal=4)

        # the nprobe is taken from the sub-indexes
        faiss.GpuParameterSpace().set_index_parameter(index, 'nprobe', 8)
        Dnew8, Inew8 = index.search(ds.get_queries(), 10)
        np.testing.assert_array_equal(Iref8, Inew8)
        np.testing.assert_array_almost_equal(Dref8, Dnew8, decimal=4)

        index.reset()
        index.add(ds.get_database())

        Dnew8, Inew8 = index.search(ds.get_queries(), 10)
        # np.testing.assert_array_equal(Iref8, Inew8)
        self.assertLess((Iref8 != Inew8).sum(), Iref8.size * 0.003)
        np.testing.assert_array_almost_equal(Dref8, Dnew8, decimal=4)

    def test_sharded_IVFSQ(self):
        self.do_test_sharded_ivf("IVF128,SQ8")

    def test_sharded_IVF_HNSW(self):
        self.do_test_sharded_ivf("IVF1000_HNSW,Flat")

    def test_binary_clone(self, ngpu=1, shard=False):
        ds = SyntheticDataset(64, 1000, 1000, 200)
        tobinary = faiss.index_factory(ds.d, "LSHrt")
        tobinary.train(ds.get_train())
        index = faiss.IndexBinaryFlat(ds.d)
        xb = tobinary.sa_encode(ds.get_database())
        xq = tobinary.sa_encode(ds.get_queries())
        index.add(xb)
        Dref, Iref = index.search(xq, 5)

        co = faiss.GpuMultipleClonerOptions()
        co.shard = shard
        co.use_raft = False

        # index2 = faiss.index_cpu_to_all_gpus(index, ngpu=ngpu)
        res = faiss.StandardGpuResources()
        index2 = faiss.GpuIndexBinaryFlat(res, index)

        Dnew, Inew = index2.search(xq, 5)
        check_ref_knn_with_draws(Dref, Iref, Dnew, Inew)

    def test_binary_clone_replicas(self):
        self.test_binary_clone(ngpu=2, shard=False)

    def test_binary_clone_shards(self):
        self.test_binary_clone(ngpu=2, shard=True)


# This class also has a multi-GPU test within
class EvalIVFPQAccuracy(unittest.TestCase):
    def get_dataset(self, small_one=False):
        if not small_one:
            d = 128
            nb = 100000
            nt = 15000
            nq = 2000
        else:
            d = 32
            nb = 10000
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

        # adding some ids because there was a bug in this case;
        # those need to be cast to idx_t(= int64_t), because
        # on windows the numpy int default is int32
        ids = (np.arange(nb) * 3 + 12345).astype('int64')
        index.add_with_ids(xb, ids)
        ts.append(time.time())

        index.nprobe = 4
        Dref, Iref = index.search(xq, 10)
        ts.append(time.time())

        res = faiss.StandardGpuResources()
        co = faiss.GpuClonerOptions()
        co.use_raft = False
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index, co)
        ts.append(time.time())

        # Validate the layout of the memory info
        mem_info = res.getMemoryInfo()

        assert type(mem_info) == dict
        assert type(mem_info[0]['FlatData']) == tuple
        assert type(mem_info[0]['FlatData'][0]) == int
        assert type(mem_info[0]['FlatData'][1]) == int

        gpu_index.nprobe = 4

        Dnew, Inew = gpu_index.search(xq, 10)
        ts.append(time.time())
        print('times:', [t - ts[0] for t in ts])

        # Give us some margin of error
        self.assertGreaterEqual((Iref == Inew).sum(), Iref.size - 50)

        if faiss.get_num_gpus() == 1:
            return

        for shard in False, True:

            # test on just 2 GPUs
            res = [faiss.StandardGpuResources() for i in range(2)]
            co = faiss.GpuMultipleClonerOptions()
            co.shard = shard
            co.use_raft = False

            gpu_index = faiss.index_cpu_to_gpu_multiple_py(res, index, co)

            faiss.GpuParameterSpace().set_index_parameter(
                gpu_index, 'nprobe', 4)

            Dnew, Inew = gpu_index.search(xq, 10)

            # 0.99: allow some tolerance in results otherwise test
            # fails occasionally (not reproducible)
            self.assertGreaterEqual((Iref == Inew).sum(), Iref.size * 0.99)

    def test_cpu_to_gpu_IVFPQ(self):
        self.do_cpu_to_gpu('IVF128,PQ4')

    def test_cpu_to_gpu_IVFFlat(self):
        self.do_cpu_to_gpu('IVF128,Flat')

    def test_set_gpu_param(self):
        index = faiss.index_factory(12, "PCAR8,IVF10,PQ4")
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        faiss.GpuParameterSpace().set_index_parameter(gpu_index, "nprobe", 3)
