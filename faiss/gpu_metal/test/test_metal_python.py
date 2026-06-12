# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import unittest
import faiss


class TestMetalFlat(unittest.TestCase):

    def test_get_num_gpus(self):
        n = faiss.get_num_gpus()
        self.assertGreaterEqual(n, 0)
        if n == 0:
            self.skipTest("No Metal device")

    def test_standard_gpu_resources(self):
        if faiss.get_num_gpus() == 0:
            self.skipTest("No Metal device")
        res = faiss.StandardGpuResources()
        self.assertIsNotNone(res)

    def test_profiler_and_sync(self):
        faiss.gpu_profiler_start()
        faiss.gpu_sync_all_devices()
        faiss.gpu_profiler_stop()

    def test_flat_l2(self):
        if faiss.get_num_gpus() == 0:
            self.skipTest("No Metal device")
        d, nb, nq, k = 64, 1000, 10, 5
        np.random.seed(1234)
        xb = np.random.randn(nb, d).astype(np.float32)
        xq = np.random.randn(nq, d).astype(np.float32)

        cpu_index = faiss.IndexFlatL2(d)
        cpu_index.add(xb)

        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

        D_gpu, I_gpu = gpu_index.search(xq, k)
        D_cpu, I_cpu = cpu_index.search(xq, k)
        np.testing.assert_allclose(D_gpu, D_cpu, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(I_gpu, I_cpu)

    def test_flat_ip(self):
        if faiss.get_num_gpus() == 0:
            self.skipTest("No Metal device")
        d, nb, nq, k = 32, 500, 5, 3
        np.random.seed(5678)
        xb = np.random.randn(nb, d).astype(np.float32)
        xq = np.random.randn(nq, d).astype(np.float32)

        cpu_index = faiss.IndexFlatIP(d)
        cpu_index.add(xb)

        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

        D_gpu, I_gpu = gpu_index.search(xq, k)
        D_cpu, I_cpu = cpu_index.search(xq, k)
        np.testing.assert_allclose(D_gpu, D_cpu, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(I_gpu, I_cpu)

    def test_flat_gpu_to_cpu(self):
        if faiss.get_num_gpus() == 0:
            self.skipTest("No Metal device")
        d, nb, nq, k = 32, 200, 5, 3
        np.random.seed(9999)
        xb = np.random.randn(nb, d).astype(np.float32)
        xq = np.random.randn(nq, d).astype(np.float32)

        cpu_index = faiss.IndexFlatL2(d)
        cpu_index.add(xb)

        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        cpu_back = faiss.index_gpu_to_cpu(gpu_index)

        self.assertEqual(cpu_back.ntotal, nb)
        D_orig, I_orig = cpu_index.search(xq, k)
        D_back, I_back = cpu_back.search(xq, k)
        np.testing.assert_array_almost_equal(D_back, D_orig)
        np.testing.assert_array_equal(I_back, I_orig)


class TestMetalIVFFlat(unittest.TestCase):

    def test_ivfflat_l2(self):
        if faiss.get_num_gpus() == 0:
            self.skipTest("No Metal device")
        d, nb, nq, k = 64, 1000, 10, 5
        nlist, nprobe = 8, 4
        np.random.seed(1234)
        xb = np.random.randn(nb, d).astype(np.float32)
        xq = np.random.randn(nq, d).astype(np.float32)

        quantizer = faiss.IndexFlatL2(d)
        cpu_index = faiss.IndexIVFFlat(quantizer, d, nlist)
        cpu_index.nprobe = nprobe
        cpu_index.train(xb)
        cpu_index.add(xb)

        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

        D_gpu, I_gpu = gpu_index.search(xq, k)
        D_cpu, I_cpu = cpu_index.search(xq, k)
        np.testing.assert_allclose(D_gpu, D_cpu, rtol=1e-4, atol=1e-4)
        np.testing.assert_array_equal(I_gpu, I_cpu)

    def test_ivfflat_ip(self):
        if faiss.get_num_gpus() == 0:
            self.skipTest("No Metal device")
        d, nb, nq, k = 64, 1000, 10, 5
        nlist, nprobe = 8, 4
        np.random.seed(5678)
        xb = np.random.randn(nb, d).astype(np.float32)
        xq = np.random.randn(nq, d).astype(np.float32)

        quantizer = faiss.IndexFlatIP(d)
        cpu_index = faiss.IndexIVFFlat(
            quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        cpu_index.nprobe = nprobe
        cpu_index.train(xb)
        cpu_index.add(xb)

        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

        D_gpu, I_gpu = gpu_index.search(xq, k)
        D_cpu, I_cpu = cpu_index.search(xq, k)
        np.testing.assert_allclose(D_gpu, D_cpu, rtol=1e-4, atol=1e-4)
        np.testing.assert_array_equal(I_gpu, I_cpu)

    def test_ivfflat_reset(self):
        if faiss.get_num_gpus() == 0:
            self.skipTest("No Metal device")
        d, nb = 32, 500
        nlist = 4
        np.random.seed(42)
        xb = np.random.randn(nb, d).astype(np.float32)

        quantizer = faiss.IndexFlatL2(d)
        cpu_index = faiss.IndexIVFFlat(quantizer, d, nlist)
        cpu_index.train(xb)
        cpu_index.add(xb)

        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        self.assertEqual(gpu_index.ntotal, nb)

        gpu_index.reset()
        self.assertEqual(gpu_index.ntotal, 0)

    def test_ivfflat_gpu_to_cpu_roundtrip(self):
        if faiss.get_num_gpus() == 0:
            self.skipTest("No Metal device")
        d, nb, nq, k = 64, 800, 5, 5
        nlist, nprobe = 8, 4
        np.random.seed(300)
        xb = np.random.randn(nb, d).astype(np.float32)
        xq = np.random.randn(nq, d).astype(np.float32)

        quantizer = faiss.IndexFlatL2(d)
        cpu_index = faiss.IndexIVFFlat(quantizer, d, nlist)
        cpu_index.nprobe = nprobe
        cpu_index.train(xb)
        cpu_index.add(xb)

        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        cpu_back = faiss.index_gpu_to_cpu(gpu_index)

        self.assertEqual(cpu_back.ntotal, nb)
        D_orig, I_orig = cpu_index.search(xq, k)
        D_back, I_back = cpu_back.search(xq, k)
        np.testing.assert_allclose(D_back, D_orig, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(I_back, I_orig)


class TestMetalIVFPQ(unittest.TestCase):

    def test_ivfpq_l2(self):
        if faiss.get_num_gpus() == 0:
            self.skipTest("No Metal device")
        d, nb, nq, k = 64, 2000, 10, 5
        nlist, nprobe, M = 8, 4, 8
        np.random.seed(1234)
        xb = np.random.randn(nb, d).astype(np.float32)
        xq = np.random.randn(nq, d).astype(np.float32)

        quantizer = faiss.IndexFlatL2(d)
        cpu_index = faiss.IndexIVFPQ(quantizer, d, nlist, M, 8)
        cpu_index.nprobe = nprobe
        cpu_index.train(xb)
        cpu_index.add(xb)

        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

        D_gpu, I_gpu = gpu_index.search(xq, k)
        D_cpu, I_cpu = cpu_index.search(xq, k)
        matches = sum(
            1 for q in range(nq) for j in range(k)
            if I_gpu[q, j] in set(I_cpu[q])
        )
        self.assertGreater(matches, nq * k // 2)

    def test_ivfpq_ip(self):
        if faiss.get_num_gpus() == 0:
            self.skipTest("No Metal device")
        d, nb, nq, k = 64, 2000, 10, 5
        nlist, nprobe, M = 8, 4, 8
        np.random.seed(5678)
        xb = np.random.randn(nb, d).astype(np.float32)
        xq = np.random.randn(nq, d).astype(np.float32)

        quantizer = faiss.IndexFlatIP(d)
        cpu_index = faiss.IndexIVFPQ(
            quantizer, d, nlist, M, 8, faiss.METRIC_INNER_PRODUCT)
        cpu_index.nprobe = nprobe
        cpu_index.train(xb)
        cpu_index.add(xb)

        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

        D_gpu, I_gpu = gpu_index.search(xq, k)
        D_cpu, I_cpu = cpu_index.search(xq, k)
        matches = sum(
            1 for q in range(nq) for j in range(k)
            if I_gpu[q, j] in set(I_cpu[q])
        )
        self.assertGreater(matches, nq * k // 2)

    def test_ivfpq_reset(self):
        if faiss.get_num_gpus() == 0:
            self.skipTest("No Metal device")
        d, nb = 32, 1000
        nlist, M = 4, 8
        np.random.seed(42)
        xb = np.random.randn(nb, d).astype(np.float32)

        quantizer = faiss.IndexFlatL2(d)
        cpu_index = faiss.IndexIVFPQ(quantizer, d, nlist, M, 8)
        cpu_index.train(xb)
        cpu_index.add(xb)

        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        self.assertEqual(gpu_index.ntotal, nb)

        gpu_index.reset()
        self.assertEqual(gpu_index.ntotal, 0)

    def test_ivfpq_gpu_to_cpu_roundtrip(self):
        if faiss.get_num_gpus() == 0:
            self.skipTest("No Metal device")
        d, nb, nq, k = 64, 1500, 5, 5
        nlist, nprobe, M = 8, 4, 8
        np.random.seed(300)
        xb = np.random.randn(nb, d).astype(np.float32)
        xq = np.random.randn(nq, d).astype(np.float32)

        quantizer = faiss.IndexFlatL2(d)
        cpu_index = faiss.IndexIVFPQ(quantizer, d, nlist, M, 8)
        cpu_index.nprobe = nprobe
        cpu_index.train(xb)
        cpu_index.add(xb)

        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        cpu_back = faiss.index_gpu_to_cpu(gpu_index)

        self.assertEqual(cpu_back.ntotal, nb)
        D_orig, I_orig = cpu_index.search(xq, k)
        D_back, I_back = cpu_back.search(xq, k)
        np.testing.assert_allclose(D_back, D_orig, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(I_back, I_orig)


if __name__ == "__main__":
    unittest.main()
