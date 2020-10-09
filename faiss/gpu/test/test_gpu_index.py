# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import time
import unittest
import numpy as np
import faiss
import math

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

        # adding some ids because there was a bug in this case
        index.add_with_ids(xb, np.arange(nb) * 3 + 12345)
        ts.append(time.time())

        index.nprobe = 4
        Dref, Iref = index.search(xq, 10)
        ts.append(time.time())

        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        ts.append(time.time())

        # Validate the layout of the memory info
        mem_info = res.getMemoryInfo()

        assert type(mem_info) == dict
        assert type(mem_info[0]['FlatData']) == tuple
        assert type(mem_info[0]['FlatData'][0]) == int
        assert type(mem_info[0]['FlatData'][1]) == int

        gpu_index.setNumProbes(4)

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


class ReferencedObject(unittest.TestCase):

    d = 16
    xb = np.random.rand(256, d).astype('float32')
    nlist = 128

    d_bin = 256
    xb_bin = np.random.randint(256, size=(10000, d_bin // 8)).astype('uint8')
    xq_bin = np.random.randint(256, size=(1000, d_bin // 8)).astype('uint8')

    def test_proxy(self):
        index = faiss.IndexReplicas()
        for _i in range(3):
            sub_index = faiss.IndexFlatL2(self.d)
            sub_index.add(self.xb)
            index.addIndex(sub_index)
        assert index.d == self.d
        index.search(self.xb, 10)

    def test_resources(self):
        # this used to crash!
        index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0,
                                       faiss.IndexFlatL2(self.d))
        index.add(self.xb)

    def test_flat(self):
        index = faiss.GpuIndexFlat(faiss.StandardGpuResources(),
                                   self.d, faiss.METRIC_L2)
        index.add(self.xb)

    def test_ivfflat(self):
        index = faiss.GpuIndexIVFFlat(
            faiss.StandardGpuResources(),
            self.d, self.nlist, faiss.METRIC_L2)
        index.train(self.xb)

    def test_ivfpq(self):
        index_cpu = faiss.IndexIVFPQ(
            faiss.IndexFlatL2(self.d),
            self.d, self.nlist, 2, 8)
        # speed up test
        index_cpu.pq.cp.niter = 2
        index_cpu.do_polysemous_training = False
        index_cpu.train(self.xb)

        index = faiss.GpuIndexIVFPQ(
            faiss.StandardGpuResources(), index_cpu)
        index.add(self.xb)

    def test_binary_flat(self):
        k = 10

        index_ref = faiss.IndexBinaryFlat(self.d_bin)
        index_ref.add(self.xb_bin)
        D_ref, I_ref = index_ref.search(self.xq_bin, k)

        index = faiss.GpuIndexBinaryFlat(faiss.StandardGpuResources(),
                                         self.d_bin)
        index.add(self.xb_bin)
        D, I = index.search(self.xq_bin, k)

        for d_ref, i_ref, d_new, i_new in zip(D_ref, I_ref, D, I):
            # exclude max distance
            assert d_ref.max() == d_new.max()
            dmax = d_ref.max()

            # sort by (distance, id) pairs to be reproducible
            ref = [(d, i) for d, i in zip(d_ref, i_ref) if d < dmax]
            ref.sort()

            new = [(d, i) for d, i in zip(d_new, i_new) if d < dmax]
            new.sort()

            assert ref == new

    def test_stress(self):
        # a mixture of the above, from issue #631
        target = np.random.rand(50, 16).astype('float32')

        index = faiss.IndexReplicas()
        size, dim = target.shape
        num_gpu = 4
        for _i in range(num_gpu):
            config = faiss.GpuIndexFlatConfig()
            config.device = 0   # simulate on a single GPU
            sub_index = faiss.GpuIndexFlatIP(faiss.StandardGpuResources(), dim, config)
            index.addIndex(sub_index)

        index = faiss.IndexIDMap(index)
        ids = np.arange(size)
        index.add_with_ids(target, ids)



class TestShardedFlat(unittest.TestCase):

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
            assert False, "this call should fail!"


class TestGPUKmeans(unittest.TestCase):

    def test_kmeans(self):
        d = 32
        nb = 1000
        k = 10
        rs = np.random.RandomState(123)
        xb = rs.rand(nb, d).astype('float32')

        km1 = faiss.Kmeans(d, k)
        obj1 = km1.train(xb)

        km2 = faiss.Kmeans(d, k, gpu=True)
        obj2 = km2.train(xb)

        print(obj1, obj2)
        assert np.allclose(obj1, obj2)


class TestAlternativeDistances(unittest.TestCase):

    def do_test(self, metric, metric_arg=0):
        res = faiss.StandardGpuResources()
        d = 32
        nb = 1000
        nq = 100

        rs = np.random.RandomState(123)
        xb = rs.rand(nb, d).astype('float32')
        xq = rs.rand(nq, d).astype('float32')

        index_ref = faiss.IndexFlat(d, metric)
        index_ref.metric_arg = metric_arg
        index_ref.add(xb)
        Dref, Iref = index_ref.search(xq, 10)

        # build from other index
        index = faiss.GpuIndexFlat(res, index_ref)
        Dnew, Inew = index.search(xq, 10)
        np.testing.assert_array_equal(Inew, Iref)
        np.testing.assert_allclose(Dnew, Dref, rtol=1e-6)

        #  build from scratch
        index = faiss.GpuIndexFlat(res, d, metric)
        index.metric_arg = metric_arg
        index.add(xb)

        Dnew, Inew = index.search(xq, 10)
        np.testing.assert_array_equal(Inew, Iref)

    def test_L1(self):
        self.do_test(faiss.METRIC_L1)

    def test_Linf(self):
        self.do_test(faiss.METRIC_Linf)

    def test_Lp(self):
        self.do_test(faiss.METRIC_Lp, 0.7)


class TestGpuRef(unittest.TestCase):

    def test_gpu_ref(self):
        # this crashes
        dim = 256
        training_data = np.random.randint(256, size=(10000, dim // 8)).astype('uint8')
        centroids = 330

        def create_cpu(dim):
            quantizer = faiss.IndexBinaryFlat(dim)
            return faiss.IndexBinaryIVF(quantizer, dim, centroids)

        def create_gpu(dim):
            gpu_quantizer = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(dim))

            index = create_cpu(dim)
            index.clustering_index = gpu_quantizer
            index.dont_dealloc_me = gpu_quantizer
            return index

        index = create_gpu(dim)

        index.verbose = True
        index.cp.verbose = True

        index.train(training_data)

class TestInterleavedIVFPQLayout(unittest.TestCase):
    def test_interleaved(self):
        res = faiss.StandardGpuResources()
        d = 128
        nb = 5000
        nq = 50

        rs = np.random.RandomState(123)
        xb = rs.rand(nb, d).astype('float32')
        xq = rs.rand(nq, d).astype('float32')

        nlist = int(math.sqrt(nb))
        sub_q = 16
        bits_per_code = 8
        nprobe = 4

        config = faiss.GpuIndexIVFPQConfig()
        config.alternativeLayout = True
        idx_gpu = faiss.GpuIndexIVFPQ(res, d, nlist, sub_q, bits_per_code, faiss.METRIC_L2, config)
        q = faiss.IndexFlatL2(d)
        idx_cpu = faiss.IndexIVFPQ(q, d, nlist, sub_q, bits_per_code, faiss.METRIC_L2)

        idx_gpu.train(xb)
        idx_gpu.add(xb)
        idx_cpu.train(xb)
        idx_cpu.add(xb)

        idx_gpu.nprobe = nprobe
        idx_cpu.nprobe = nprobe

        # Try without precomputed codes
        d_g, i_g = idx_gpu.search(xq, 10)
        d_c, i_c = idx_cpu.search(xq, 10)
        self.assertGreaterEqual((i_g == i_c).sum(), i_g.size - 10)
        self.assertTrue(np.allclose(d_g, d_c))

        # Try with precomputed codes (different kernel)
        idx_gpu.setPrecomputedCodes(True)
        d_g, i_g = idx_gpu.search(xq, 10)
        d_c, i_c = idx_cpu.search(xq, 10)
        self.assertGreaterEqual((i_g == i_c).sum(), i_g.size - 10)
        self.assertTrue(np.allclose(d_g, d_c))

    def test_copy_to_cpu(self):
        res = faiss.StandardGpuResources()
        d = 128
        nb = 5000
        nq = 50

        rs = np.random.RandomState(234)
        xb = rs.rand(nb, d).astype('float32')
        xq = rs.rand(nq, d).astype('float32')

        nlist = int(math.sqrt(nb))
        sub_q = 16
        bits_per_code = 8
        nprobe = 4

        config = faiss.GpuIndexIVFPQConfig()
        config.alternativeLayout = True
        idx_gpu = faiss.GpuIndexIVFPQ(res, d, nlist, sub_q, bits_per_code, faiss.METRIC_L2, config)
        q = faiss.IndexFlatL2(d)
        idx_cpu = faiss.IndexIVFPQ(q, d, nlist, sub_q, bits_per_code, faiss.METRIC_L2)

        idx_gpu.train(xb)
        idx_gpu.add(xb)

        idx_gpu.copyTo(idx_cpu)

        idx_gpu.nprobe = nprobe
        idx_cpu.nprobe = nprobe

        # Try without precomputed codes
        d_g, i_g = idx_gpu.search(xq, 10)
        d_c, i_c = idx_cpu.search(xq, 10)
        self.assertGreaterEqual((i_g == i_c).sum(), i_g.size - 10)
        self.assertTrue(np.allclose(d_g, d_c))

        # Try with precomputed codes (different kernel)
        idx_gpu.setPrecomputedCodes(True)
        d_g, i_g = idx_gpu.search(xq, 10)
        d_c, i_c = idx_cpu.search(xq, 10)
        self.assertGreaterEqual((i_g == i_c).sum(), i_g.size - 10)
        self.assertTrue(np.allclose(d_g, d_c))

    def test_copy_to_gpu(self):
        res = faiss.StandardGpuResources()
        d = 128
        nb = 5000
        nq = 50

        rs = np.random.RandomState(567)
        xb = rs.rand(nb, d).astype('float32')
        xq = rs.rand(nq, d).astype('float32')

        nlist = int(math.sqrt(nb))
        sub_q = 16
        bits_per_code = 8
        nprobe = 4

        config = faiss.GpuIndexIVFPQConfig()
        config.alternativeLayout = True
        idx_gpu = faiss.GpuIndexIVFPQ(res, d, nlist, sub_q, bits_per_code, faiss.METRIC_L2, config)
        q = faiss.IndexFlatL2(d)
        idx_cpu = faiss.IndexIVFPQ(q, d, nlist, sub_q, bits_per_code, faiss.METRIC_L2)

        idx_cpu.train(xb)
        idx_cpu.add(xb)

        idx_gpu.copyFrom(idx_cpu)

        idx_gpu.nprobe = nprobe
        idx_cpu.nprobe = nprobe

        # Try without precomputed codes
        d_g, i_g = idx_gpu.search(xq, 10)
        d_c, i_c = idx_cpu.search(xq, 10)
        self.assertGreaterEqual((i_g == i_c).sum(), i_g.size - 10)
        self.assertTrue(np.allclose(d_g, d_c))

        # Try with precomputed codes (different kernel)
        idx_gpu.setPrecomputedCodes(True)
        d_g, i_g = idx_gpu.search(xq, 10)
        d_c, i_c = idx_cpu.search(xq, 10)
        self.assertGreaterEqual((i_g == i_c).sum(), i_g.size - 10)
        self.assertTrue(np.allclose(d_g, d_c))

def make_t(num, d, clamp=False, seed=None):
    rs = None
    if seed is None:
        rs = np.random.RandomState(123)
    else:
        rs = np.random.RandomState(seed)

    x = rs.rand(num, d).astype(np.float32)
    if clamp:
        x = (x * 255).astype('uint8').astype('float32')
    return x

class TestBruteForceDistance(unittest.TestCase):
    def test_bf_input_types(self):
        d = 33
        k = 5
        nb = 1000
        nq = 10

        xs = make_t(nb, d)
        qs = make_t(nq, d)

        res = faiss.StandardGpuResources()

        # Get ground truth using IndexFlat
        index = faiss.IndexFlatL2(d)
        index.add(xs)
        ref_d, ref_i = index.search(qs, k)

        out_d = np.empty((nq, k), dtype=np.float32)
        out_i = np.empty((nq, k), dtype=np.int64)

        # Try f32 data/queries, i64 out indices
        params = faiss.GpuDistanceParams()
        params.k = k
        params.dims = d
        params.vectors = faiss.swig_ptr(xs)
        params.numVectors = nb
        params.queries = faiss.swig_ptr(qs)
        params.numQueries = nq
        params.outDistances = faiss.swig_ptr(out_d)
        params.outIndices = faiss.swig_ptr(out_i)

        faiss.bfKnn(res, params)

        self.assertTrue(np.allclose(ref_d, out_d, atol=1e-5))
        self.assertGreaterEqual((out_i == ref_i).sum(), ref_i.size)

        # Try int32 out indices
        out_i32 = np.empty((nq, k), dtype=np.int32)
        params.outIndices = faiss.swig_ptr(out_i32)
        params.outIndicesType = faiss.IndicesDataType_I32

        faiss.bfKnn(res, params)
        self.assertEqual((out_i32 == ref_i).sum(), ref_i.size)

        # Try float16 data/queries, i64 out indices
        xs_f16 = xs.astype(np.float16)
        qs_f16 = qs.astype(np.float16)
        xs_f16_f32 = xs_f16.astype(np.float32)
        qs_f16_f32 = qs_f16.astype(np.float32)
        index.reset()
        index.add(xs_f16_f32)
        ref_d_f16, ref_i_f16 = index.search(qs_f16_f32, k)

        params.vectors = faiss.swig_ptr(xs_f16)
        params.vectorType = faiss.DistanceDataType_F16
        params.queries = faiss.swig_ptr(qs_f16)
        params.queryType = faiss.DistanceDataType_F16

        out_d_f16 = np.empty((nq, k), dtype=np.float32)
        out_i_f16 = np.empty((nq, k), dtype=np.int64)

        params.outDistances = faiss.swig_ptr(out_d_f16)
        params.outIndices = faiss.swig_ptr(out_i_f16)
        params.outIndicesType = faiss.IndicesDataType_I64

        faiss.bfKnn(res, params)

        self.assertGreaterEqual((out_i_f16 == ref_i_f16).sum(), ref_i_f16.size - 5)
        self.assertTrue(np.allclose(ref_d_f16, out_d_f16, atol = 2e-3))

if __name__ == '__main__':
    unittest.main()
