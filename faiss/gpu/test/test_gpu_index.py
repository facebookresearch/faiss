# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import math
import time
import unittest
import numpy as np
import faiss
from faiss.contrib import datasets
from faiss.contrib import ivf_tools
from faiss.contrib.evaluation import knn_intersection_measure

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



class TestShardedFlat(unittest.TestCase):

    @unittest.skipIf(faiss.get_num_gpus() < 2, "Relevant for multiple GPU only.")
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


class TestIVFSearchPreassigned(unittest.TestCase):
    def test_ivfflat_search_preassigned(self):
        res = faiss.StandardGpuResources()
        d = 50
        nb = 50000
        nq = 100
        nlist = 128
        nprobe = 10
        k = 50

        idx_gpu = faiss.GpuIndexIVFFlat(res, d, nlist)
        idx_gpu.nprobe = nprobe

        rs = np.random.RandomState(567)
        xb = rs.rand(nb, d).astype('float32')
        xq = rs.rand(nq, d).astype('float32')

        idx_gpu.train(xb)
        idx_gpu.add(xb)

        # Search separately using the same quantizer
        q_d, q_i = idx_gpu.quantizer.search(xq, nprobe)

        preassigned_d, preassigned_i = ivf_tools.search_preassigned(
            idx_gpu, xq, k, q_i, q_d)

        # Search using the standard API
        d, i = idx_gpu.search(xq, k)

        # The two results should be exactly the same
        self.assertEqual((d == preassigned_d).sum(), d.size)
        self.assertEqual((i == preassigned_i).sum(), i.size)

    def test_ivfpq_search_preassigned(self):
        res = faiss.StandardGpuResources()
        d = 64
        nb = 50000
        nq = 100
        nlist = 128
        nprobe = 5
        k = 50

        idx_gpu = faiss.GpuIndexIVFPQ(res, d, nlist, 4, 8)
        idx_gpu.nprobe = nprobe

        rs = np.random.RandomState(567)
        xb = rs.rand(nb, d).astype('float32')
        xq = rs.rand(nq, d).astype('float32')

        idx_gpu.train(xb)
        idx_gpu.add(xb)

        # Search separately using the same quantizer
        q_d, q_i = idx_gpu.quantizer.search(xq, nprobe)

        preassigned_d, preassigned_i = ivf_tools.search_preassigned(
            idx_gpu, xq, k, q_i, q_d)

        # Search using the standard API
        d, i = idx_gpu.search(xq, k)

        # The two results should be exactly the same
        self.assertEqual((d == preassigned_d).sum(), d.size)
        self.assertEqual((i == preassigned_i).sum(), i.size)

    def test_ivfsq_search_preassigned(self):
        res = faiss.StandardGpuResources()
        d = 64
        nb = 50000
        nq = 100
        nlist = 128
        nprobe = 5
        k = 50

        idx_gpu = faiss.GpuIndexIVFScalarQuantizer(
            res, d, nlist,
            faiss.ScalarQuantizer.QT_6bit,
            faiss.METRIC_INNER_PRODUCT)
        idx_gpu.nprobe = nprobe

        rs = np.random.RandomState(567)
        xb = rs.rand(nb, d).astype('float32')
        xq = rs.rand(nq, d).astype('float32')

        idx_gpu.train(xb)
        idx_gpu.add(xb)

        # Search separately using the same quantizer
        q_d, q_i = idx_gpu.quantizer.search(xq, nprobe)

        preassigned_d, preassigned_i = ivf_tools.search_preassigned(
            idx_gpu, xq, k, q_i, q_d)

        # Search using the standard API
        d, i = idx_gpu.search(xq, k)

        # The two results should be exactly the same
        self.assertEqual((d == preassigned_d).sum(), d.size)
        self.assertEqual((i == preassigned_i).sum(), i.size)


class TestIVFPluggableCoarseQuantizer(unittest.TestCase):
    def test_ivfflat_cpu_coarse(self):
        res = faiss.StandardGpuResources()
        d = 128
        nb = 5000
        nq = 100
        nlist = 10
        nprobe = 3

        q = faiss.IndexFlatL2(d)
        idx_cpu = faiss.IndexIVFFlat(q, d, nlist)

        rs = np.random.RandomState(567)
        xb = rs.rand(nb, d).astype('float32')
        xq = rs.rand(nq, d).astype('float32')

        idx_cpu.train(xb)
        idx_cpu.add(xb)

        # construct a GPU index using the same trained coarse quantizer
        # from the CPU index
        idx_gpu = faiss.GpuIndexIVFFlat(res, q, d, nlist, faiss.METRIC_L2)
        assert(idx_gpu.is_trained)
        idx_gpu.add(xb)

        k = 20

        idx_cpu.nprobe = nprobe
        idx_gpu.nprobe = nprobe

        d_g, i_g = idx_gpu.search(xq, k)
        d_c, i_c = idx_cpu.search(xq, k)
        self.assertGreaterEqual((i_g == i_c).sum(), i_g.size * 0.9)
        self.assertTrue(np.allclose(d_g, d_c, rtol=5e-5, atol=5e-5))

    def test_ivfsq_cpu_coarse(self):
        res = faiss.StandardGpuResources()
        d = 128
        nb = 5000
        nq = 100
        nlist = 10
        nprobe = 3
        use_residual = True
        qtype = faiss.ScalarQuantizer.QT_8bit

        q = faiss.IndexFlatL2(d)
        idx_cpu = faiss.IndexIVFScalarQuantizer(
            q, d, nlist, qtype, faiss.METRIC_L2, use_residual)

        rs = np.random.RandomState(567)
        xb = rs.rand(nb, d).astype('float32')
        xq = rs.rand(nq, d).astype('float32')

        idx_cpu.train(xb)
        idx_cpu.add(xb)

        # construct a GPU index using the same trained coarse quantizer
        # from the CPU index
        idx_gpu = faiss.GpuIndexIVFScalarQuantizer(
            res, q, d, nlist, qtype, faiss.METRIC_L2, use_residual)
        assert(not idx_gpu.is_trained)
        idx_gpu.train(xb)
        idx_gpu.add(xb)

        k = 20

        idx_cpu.nprobe = nprobe
        idx_gpu.nprobe = nprobe

        d_g, i_g = idx_gpu.search(xq, k)
        d_c, i_c = idx_cpu.search(xq, k)

        self.assertGreaterEqual(knn_intersection_measure(i_c, i_g), 0.9)

        self.assertTrue(np.allclose(d_g, d_c, rtol=5e-5, atol=5e-5))

    def test_ivfpq_cpu_coarse(self):
        res = faiss.StandardGpuResources()
        d = 32
        nb = 50000
        nq = 20
        nlist_lvl_1 = 10
        nlist_lvl_2 = 1000
        nprobe_lvl_1 = 3
        nprobe_lvl_2 = 10

        rs = np.random.RandomState(567)
        coarse_centroids = rs.rand(nlist_lvl_2, d).astype('float32')

        # Construct an IVFFlat index for usage as a coarse quantizer
        idx_coarse_cpu = faiss.IndexIVFFlat(
            faiss.IndexFlatL2(d), d, nlist_lvl_1)
        idx_coarse_cpu.set_direct_map_type(faiss.DirectMap.Hashtable)
        idx_coarse_cpu.nprobe = nprobe_lvl_1

        idx_coarse_cpu.train(coarse_centroids)
        idx_coarse_cpu.add(coarse_centroids)
        idx_coarse_cpu.make_direct_map()

        assert(idx_coarse_cpu.ntotal == nlist_lvl_2)

        idx_cpu = faiss.IndexIVFPQ(
            idx_coarse_cpu, d, nlist_lvl_2, 4, 8)

        xb = rs.rand(nb, d).astype('float32')
        idx_cpu.train(xb)
        idx_cpu.add(xb)
        idx_cpu.nprobe = nprobe_lvl_2

        # construct a GPU index using the same trained coarse quantizer
        # from the CPU index
        idx_gpu = faiss.GpuIndexIVFPQ(
            res, idx_coarse_cpu, d, nlist_lvl_2, 4, 8)
        assert(not idx_gpu.is_trained)

        idx_gpu.train(xb)
        idx_gpu.add(xb)
        idx_gpu.nprobe = nprobe_lvl_2

        k = 10

        # precomputed codes also utilize the coarse quantizer
        for use_precomputed in [False, True]:
            idx_gpu.setPrecomputedCodes(use_precomputed)

            xq = rs.rand(nq, d).astype('float32')
            d_g, i_g = idx_gpu.search(xq, k)
            d_c, i_c = idx_cpu.search(xq, k)

            self.assertGreaterEqual(knn_intersection_measure(i_c, i_g), 0.9)


class TestInterleavedIVFPQLayout(unittest.TestCase):
    def test_interleaved(self):
        res = faiss.StandardGpuResources()

        for bits_per_code in [4, 5, 6, 8]:
            d = 128
            nb = 10000
            nq = 20

            rs = np.random.RandomState(123)
            xb = rs.rand(nb, d).astype('float32')
            xq = rs.rand(nq, d).astype('float32')

            nlist = int(math.sqrt(nb))
            sub_q = 16
            nprobe = 16

            config = faiss.GpuIndexIVFPQConfig()
            config.interleavedLayout = True
            idx_gpu = faiss.GpuIndexIVFPQ(res, d, nlist, sub_q, bits_per_code, faiss.METRIC_L2, config)
            q = faiss.IndexFlatL2(d)
            idx_cpu = faiss.IndexIVFPQ(q, d, nlist, sub_q, bits_per_code, faiss.METRIC_L2)

            idx_gpu.train(xb)
            idx_gpu.add(xb)
            idx_gpu.copyTo(idx_cpu)

            idx_gpu.nprobe = nprobe
            idx_cpu.nprobe = nprobe

            k = 20

            # Try without precomputed codes
            d_g, i_g = idx_gpu.search(xq, k)
            d_c, i_c = idx_cpu.search(xq, k)
            self.assertGreaterEqual((i_g == i_c).sum(), i_g.size * 0.9)
            self.assertTrue(np.allclose(d_g, d_c, rtol=5e-5, atol=5e-5))

            # Try with precomputed codes (different kernel)
            idx_gpu.setPrecomputedCodes(True)
            d_g, i_g = idx_gpu.search(xq, k)
            d_c, i_c = idx_cpu.search(xq, k)
            self.assertGreaterEqual((i_g == i_c).sum(), i_g.size * 0.9)
            self.assertTrue(np.allclose(d_g, d_c, rtol=5e-5, atol=5e-5))

    def test_copy_to_cpu(self):
        res = faiss.StandardGpuResources()

        for bits_per_code in [4, 5, 6, 8]:
            d = 128
            nb = 10000
            nq = 20

            rs = np.random.RandomState(234)
            xb = rs.rand(nb, d).astype('float32')
            xq = rs.rand(nq, d).astype('float32')

            nlist = int(math.sqrt(nb))
            sub_q = 16
            bits_per_code = 8
            nprobe = 4

            config = faiss.GpuIndexIVFPQConfig()
            config.interleavedLayout = True
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
            self.assertGreaterEqual((i_g == i_c).sum(), i_g.size * 0.9)
            self.assertTrue(np.allclose(d_g, d_c))

            # Try with precomputed codes (different kernel)
            idx_gpu.setPrecomputedCodes(True)
            d_g, i_g = idx_gpu.search(xq, 10)
            d_c, i_c = idx_cpu.search(xq, 10)
            self.assertGreaterEqual((i_g == i_c).sum(), i_g.size * 0.9)
            self.assertTrue(np.allclose(d_g, d_c))

    def test_copy_to_gpu(self):
        res = faiss.StandardGpuResources()

        for bits_per_code in [4, 5, 6, 8]:
            d = 128
            nb = 10000
            nq = 20

            rs = np.random.RandomState(567)
            xb = rs.rand(nb, d).astype('float32')
            xq = rs.rand(nq, d).astype('float32')

            nlist = int(math.sqrt(nb))
            sub_q = 16
            bits_per_code = 8
            nprobe = 4

            config = faiss.GpuIndexIVFPQConfig()
            config.interleavedLayout = True
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
            self.assertGreaterEqual((i_g == i_c).sum(), i_g.size * 0.9)
            self.assertTrue(np.allclose(d_g, d_c))

            # Try with precomputed codes (different kernel)
            idx_gpu.setPrecomputedCodes(True)
            d_g, i_g = idx_gpu.search(xq, 10)
            d_c, i_c = idx_cpu.search(xq, 10)
            self.assertGreaterEqual((i_g == i_c).sum(), i_g.size * 0.9)
            self.assertTrue(np.allclose(d_g, d_c))


# Make sure indices are properly stored in the IVF lists
class TestIVFIndices(unittest.TestCase):
    def test_indices_ivfflat(self):
        res = faiss.StandardGpuResources()
        d = 128
        nb = 5000
        nlist = 10

        rs = np.random.RandomState(567)
        xb = rs.rand(nb, d).astype('float32')
        xb_indices_base = np.arange(nb, dtype=np.int64)

        # Force values to not be representable in int32
        xb_indices = (xb_indices_base + 4294967296).astype('int64')

        config = faiss.GpuIndexIVFFlatConfig()
        idx = faiss.GpuIndexIVFFlat(res, d, nlist, faiss.METRIC_L2, config)
        idx.train(xb)
        idx.add_with_ids(xb, xb_indices)

        _, I = idx.search(xb[10:20], 5)
        self.assertTrue(np.array_equal(xb_indices[10:20], I[:, 0]))

        # Store values using 32-bit indices instead
        config.indicesOptions = faiss.INDICES_32_BIT
        idx = faiss.GpuIndexIVFFlat(res, d, nlist, faiss.METRIC_L2, config)
        idx.train(xb)
        idx.add_with_ids(xb, xb_indices)

        _, I = idx.search(xb[10:20], 5)
        # This will strip the high bit
        self.assertTrue(np.array_equal(xb_indices_base[10:20], I[:, 0]))

    def test_indices_ivfpq(self):
        res = faiss.StandardGpuResources()
        d = 128
        nb = 5000
        nlist = 10
        M = 4
        nbits = 8

        rs = np.random.RandomState(567)
        xb = rs.rand(nb, d).astype('float32')
        xb_indices_base = np.arange(nb, dtype=np.int64)

        # Force values to not be representable in int32
        xb_indices = (xb_indices_base + 4294967296).astype('int64')

        config = faiss.GpuIndexIVFPQConfig()
        idx = faiss.GpuIndexIVFPQ(res, d, nlist, M, nbits,
                                  faiss.METRIC_L2, config)
        idx.train(xb)
        idx.add_with_ids(xb, xb_indices)

        _, I = idx.search(xb[10:20], 5)
        self.assertTrue(np.array_equal(xb_indices[10:20], I[:, 0]))

        # Store values using 32-bit indices instead
        config.indicesOptions = faiss.INDICES_32_BIT
        idx = faiss.GpuIndexIVFPQ(res, d, nlist, M, nbits,
                                  faiss.METRIC_L2, config)
        idx.train(xb)
        idx.add_with_ids(xb, xb_indices)

        _, I = idx.search(xb[10:20], 5)
        # This will strip the high bit
        self.assertTrue(np.array_equal(xb_indices_base[10:20], I[:, 0]))

    def test_indices_ivfsq(self):
        res = faiss.StandardGpuResources()
        d = 128
        nb = 5000
        nlist = 10
        qtype = faiss.ScalarQuantizer.QT_4bit

        rs = np.random.RandomState(567)
        xb = rs.rand(nb, d).astype('float32')
        xb_indices_base = np.arange(nb, dtype=np.int64)

        # Force values to not be representable in int32
        xb_indices = (xb_indices_base + 4294967296).astype('int64')

        config = faiss.GpuIndexIVFScalarQuantizerConfig()
        idx = faiss.GpuIndexIVFScalarQuantizer(res, d, nlist, qtype,
                                               faiss.METRIC_L2, True, config)
        idx.train(xb)
        idx.add_with_ids(xb, xb_indices)

        _, I = idx.search(xb[10:20], 5)
        self.assertTrue(np.array_equal(xb_indices[10:20], I[:, 0]))

        # Store values using 32-bit indices instead
        config.indicesOptions = faiss.INDICES_32_BIT
        idx = faiss.GpuIndexIVFScalarQuantizer(res, d, nlist, qtype,
                                               faiss.METRIC_L2, True, config)
        idx.train(xb)
        idx.add_with_ids(xb, xb_indices)

        _, I = idx.search(xb[10:20], 5)
        # This will strip the high bit
        self.assertTrue(np.array_equal(xb_indices_base[10:20], I[:, 0]))


class TestSQ_to_gpu(unittest.TestCase):

    def test_sq_cpu_to_gpu(self):
        res = faiss.StandardGpuResources()
        index = faiss.index_factory(32, "SQfp16")
        index.add(np.random.rand(1000, 32).astype(np.float32))
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        self.assertIsInstance(gpu_index, faiss.GpuIndexFlat)


class TestInvalidParams(unittest.TestCase):

    def test_indices_ivfpq(self):
        res = faiss.StandardGpuResources()
        d = 128
        nb = 5000
        nlist = 10
        M = 4
        nbits = 8

        rs = np.random.RandomState(567)
        xb = rs.rand(nb, d).astype('float32')
        xb_indices_base = np.arange(nb, dtype=np.int64)

        # Force values to not be representable in int32
        xb_indices = (xb_indices_base + 4294967296).astype('int64')

        config = faiss.GpuIndexIVFPQConfig()
        idx = faiss.GpuIndexIVFPQ(res, d, nlist, M, nbits,
                                  faiss.METRIC_L2, config)
        idx.train(xb)
        idx.add_with_ids(xb, xb_indices)

        # invalid k (should be > 0)
        k = -5
        idx.setNumProbes(3)
        self.assertRaises(AssertionError, idx.search, xb[10:20], k)

        # invalid nprobe (should be > 0)
        self.assertRaises(RuntimeError, idx.setNumProbes, 0)
        self.assertRaises(RuntimeError, idx.setNumProbes, -3)

        k = 5
        idx.nprobe = -3
        self.assertRaises(RuntimeError, idx.search, xb[10:20], k)

        # valid params
        k = 5
        idx.setNumProbes(3)
        _, I = idx.search(xb[10:20], k)
        self.assertTrue(np.array_equal(xb_indices[10:20], I[:, 0]))


class TestLSQIcmEncoder(unittest.TestCase):

    @staticmethod
    def eval_codec(q, xb):
        codes = q.compute_codes(xb)
        decoded = q.decode(codes)
        return ((xb - decoded) ** 2).sum()

    def subtest_gpu_encoding(self, ngpus):
        """check that the error is in the same as cpu."""
        ds = datasets.SyntheticDataset(32, 1000, 1000, 0)

        xt = ds.get_train()
        xb = ds.get_database()

        M = 4
        nbits = 8

        lsq = faiss.LocalSearchQuantizer(ds.d, M, nbits)
        lsq.train(xt)
        err_cpu = self.eval_codec(lsq, xb)

        lsq = faiss.LocalSearchQuantizer(ds.d, M, nbits)
        lsq.train(xt)
        lsq.icm_encoder_factory = faiss.GpuIcmEncoderFactory(ngpus)
        err_gpu = self.eval_codec(lsq, xb)

        # 13804.411 vs 13814.794, 1 gpu
        print(err_gpu, err_cpu)
        self.assertLess(err_gpu, err_cpu * 1.05)

    def test_one_gpu(self):
        self.subtest_gpu_encoding(1)

    def test_multiple_gpu(self):
        ngpu = faiss.get_num_gpus()
        self.subtest_gpu_encoding(ngpu)


if __name__ == '__main__':
    unittest.main()
