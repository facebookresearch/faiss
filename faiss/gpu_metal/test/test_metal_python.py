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


class TestMetalBinaryFlat(unittest.TestCase):

    def setUp(self):
        if faiss.get_num_gpus() == 0:
            self.skipTest("No Metal device")

    def test_clone_search_and_lifetime(self):
        import gc

        rng = np.random.default_rng(1234)
        xb = rng.integers(0, 256, (1537, 5), dtype=np.uint8)
        xq = rng.integers(0, 256, (9, 5), dtype=np.uint8)
        cpu = faiss.IndexBinaryFlat(40)
        cpu.add(xb)
        resources = faiss.StandardGpuResources()
        gpu = faiss.index_binary_cpu_to_gpu(resources, 0, cpu)
        del resources, cpu
        gc.collect()

        distances = np.unpackbits(xq[:, None, :] ^ xb[None, :, :], axis=2).sum(axis=2)
        order = np.argsort(distances, axis=1, kind="stable")[:, :512]
        expected = np.take_along_axis(distances, order, axis=1)
        for _ in range(2):
            d, labels = gpu.search(xq, 512)
            self.assertEqual(d.dtype, np.int32)
            self.assertEqual(labels.dtype, np.int64)
            np.testing.assert_array_equal(d, expected)
            np.testing.assert_array_equal(labels, order)
        np.testing.assert_array_equal(gpu.reconstruct(17), xb[17])

        back = faiss.index_binary_gpu_to_cpu(gpu)
        gpu.reset()
        self.assertEqual(back.ntotal, len(xb))
        np.testing.assert_array_equal(back.reconstruct(17), xb[17])
        gpu.add(xb[:7])
        d, labels = gpu.search(xq, 20)
        self.assertTrue(np.all(labels[:, 7:] == -1))
        self.assertTrue(np.all(d[:, 7:] == np.iinfo(np.int32).max))

    def test_cpu_fallback_and_invalid_device(self):
        xb = np.arange(200, dtype=np.uint8).reshape(100, 2)
        cpu = faiss.IndexBinaryFlat(16)
        cpu.add(xb)
        resources = faiss.StandardGpuResources()
        with self.assertRaises(RuntimeError):
            faiss.index_binary_cpu_to_gpu(resources, 1, cpu)
        gpu = faiss.index_binary_cpu_to_gpu(resources, 0, cpu)
        expected_d, expected_i = cpu.search(xb[:3], 513)
        d, labels = gpu.search(xb[:3], 513)
        np.testing.assert_array_equal(d, expected_d)
        np.testing.assert_array_equal(labels, expected_i)


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
            quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT
        )
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


class TestMetalAllGpusHelpers(unittest.TestCase):

    def test_index_cpu_to_all_gpus(self):
        if faiss.get_num_gpus() == 0:
            self.skipTest("No Metal device")
        d, nb, nq, k = 32, 500, 5, 3
        np.random.seed(77)
        xb = np.random.randn(nb, d).astype(np.float32)
        xq = np.random.randn(nq, d).astype(np.float32)

        cpu_index = faiss.IndexFlatL2(d)
        cpu_index.add(xb)
        gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)

        D_gpu, I_gpu = gpu_index.search(xq, k)
        D_cpu, I_cpu = cpu_index.search(xq, k)
        np.testing.assert_allclose(D_gpu, D_cpu, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(I_gpu, I_cpu)

    def test_kmeans_gpu(self):
        if faiss.get_num_gpus() == 0:
            self.skipTest("No Metal device")
        d, n, k = 16, 2000, 10
        np.random.seed(88)
        x = np.random.randn(n, d).astype(np.float32)

        km_cpu = faiss.Kmeans(d, k, niter=5, seed=1)
        km_cpu.train(x)
        km_gpu = faiss.Kmeans(d, k, niter=5, seed=1, gpu=True)
        km_gpu.train(x)
        np.testing.assert_allclose(
            km_gpu.centroids, km_cpu.centroids, rtol=1e-4, atol=1e-4
        )


if __name__ == "__main__":
    unittest.main()
