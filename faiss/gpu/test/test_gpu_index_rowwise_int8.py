# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import faiss
import numpy as np


def i8ptr(a):
    return faiss.cast_integer_to_int8_ptr(a.ctypes.data)


def get_int8(index):
    codes = np.empty((index.ntotal, index.d), dtype="int8")
    scales = np.empty(index.ntotal, dtype="float32")
    index.get_int8(0, index.ntotal, i8ptr(codes), faiss.swig_ptr(scales))
    return codes, scales


def search_int8(index, codes, scales, k):
    codes = np.ascontiguousarray(codes, dtype="int8")
    scales = np.ascontiguousarray(scales, dtype="float32")
    n = codes.shape[0]
    D = np.empty((n, k), dtype="float32")
    I = np.empty((n, k), dtype="int64")
    index.search_int8(
        n, i8ptr(codes), faiss.swig_ptr(scales), k,
        faiss.swig_ptr(D), faiss.swig_ptr(I))
    return D, I


class TestGpuIndexRowwiseInt8(unittest.TestCase):

    def setUp(self):
        rs = np.random.RandomState(1234)
        self.d = 128
        # nb is not a multiple of 64: exercises the padding rows
        self.xb = (rs.randn(5000, self.d)
                   * rs.uniform(0.1, 3, (5000, 1))).astype("float32")
        self.xq = (rs.randn(300, self.d)
                   * rs.uniform(0.1, 3, (300, 1))).astype("float32")
        self.res = faiss.StandardGpuResources()

    def compare(self, Dg, Ig, Dc, Ic):
        # same neighbours (ties aside) and same distances up to rounding
        same = (np.sort(Ig, 1) == np.sort(Ic, 1)).mean()
        self.assertGreater(same, 0.999)
        np.testing.assert_allclose(
            Dg, Dc, rtol=1e-5, atol=1e-5 * np.abs(Dc).max())

    def check_vs_cpu(self, metric):
        k = 10
        cpu = faiss.IndexRowwiseInt8(self.d, metric)
        cpu.add(self.xb)
        Dc, Ic = cpu.search(self.xq, k)

        # GPU index built from the CPU one
        gpu = faiss.index_cpu_to_gpu(self.res, 0, cpu)
        self.assertIsInstance(gpu, faiss.GpuIndexRowwiseInt8)
        Dg, Ig = gpu.search(self.xq, k)
        self.compare(Dg, Ig, Dc, Ic)

        # GPU index filled directly: encoding identical to the CPU one
        gpu2 = faiss.GpuIndexRowwiseInt8(self.res, self.d, metric)
        gpu2.add(self.xb)
        c1, s1 = get_int8(cpu)
        c2, s2 = get_int8(gpu2)
        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(s1, s2)

        # int8 queries
        cq = faiss.IndexRowwiseInt8(self.d, metric)
        cq.add(self.xq)
        qc, qs = get_int8(cq)
        D8, I8 = search_int8(gpu, qc, qs, k)
        np.testing.assert_array_equal(D8, Dg)
        np.testing.assert_array_equal(I8, Ig)

    def test_vs_cpu_L2(self):
        self.check_vs_cpu(faiss.METRIC_L2)

    def test_vs_cpu_IP(self):
        self.check_vs_cpu(faiss.METRIC_INNER_PRODUCT)

    def test_tiling(self):
        cpu = faiss.IndexRowwiseInt8(self.d)
        cpu.add(self.xb)
        ref = faiss.GpuIndexRowwiseInt8(self.res, cpu)
        Dr, Ir = ref.search(self.xq, 7)
        cfg = faiss.GpuIndexRowwiseInt8Config()
        cfg.tileQueries = 64        # several query tiles
        cfg.tileBank = 1024         # several bank tiles + merge
        tiled = faiss.GpuIndexRowwiseInt8(self.res, cpu, cfg)
        Dt, It = tiled.search(self.xq, 7)
        self.compare(Dt, It, Dr, Ir)

    def test_copy_back(self):
        cpu = faiss.IndexRowwiseInt8(self.d)
        cpu.add(self.xb)
        gpu = faiss.index_cpu_to_gpu(self.res, 0, cpu)
        back = faiss.index_gpu_to_cpu(gpu)
        self.assertIsInstance(back, faiss.IndexRowwiseInt8)
        np.testing.assert_array_equal(
            faiss.vector_to_array(back.codes),
            faiss.vector_to_array(cpu.codes))

    def test_reconstruct(self):
        cpu = faiss.IndexRowwiseInt8(self.d)
        cpu.add(self.xb)
        gpu = faiss.index_cpu_to_gpu(self.res, 0, cpu)
        np.testing.assert_array_equal(
            gpu.reconstruct_n(10, 20), cpu.reconstruct_n(10, 20))

    def test_small_and_empty(self):
        gpu = faiss.GpuIndexRowwiseInt8(self.res, self.d)
        D, I = gpu.search(self.xq[:4], 5)            # empty index
        self.assertTrue(np.all(I == -1))
        gpu.add(self.xb[:3])                          # k > ntotal
        D, I = gpu.search(self.xq[:4], 5)
        self.assertTrue(np.all(I[:, :3] >= 0))
        self.assertTrue(np.all(I[:, 3:] == -1))


if __name__ == "__main__":
    unittest.main()
