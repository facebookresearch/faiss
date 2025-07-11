import faiss
import unittest
import numpy as np


@unittest.skipIf(not hasattr(faiss, "MetalIndexFlat"), "Metal not supported")
class TestMetal(unittest.TestCase):
    def test_create_metal_indices(self):
        index = faiss.MetalIndexFlat(128)
        self.assertIsNotNone(index)

        quantizer = faiss.IndexFlatL2(128)
        index = faiss.MetalIndexIVFFlat(quantizer, 128, 16)
        self.assertIsNotNone(index)

        index = faiss.MetalIndexIVFPQ(quantizer, 128, 16, 8, 8)
        self.assertIsNotNone(index)

        index = faiss.MetalIndexHNSW(128, 32)
        self.assertIsNotNone(index)

    def test_search(self):
        d = 64
        nb = 1000
        nq = 100
        k = 10

        xt = np.random.rand(nb, d).astype("float32")
        xq = np.random.rand(nq, d).astype("float32")

        index = faiss.MetalIndexFlat(d)
        index.add(xt)

        D, I = index.search(xq, k)

        self.assertEqual(I.shape, (nq, k))
        self.assertEqual(D.shape, (nq, k))

    def test_ivfflat(self):
        d = 64
        nb = 1000
        nq = 100
        k = 10
        nlist = 16
        nprobe = 4

        xt = np.random.rand(nb, d).astype("float32")
        xq = np.random.rand(nq, d).astype("float32")

        quantizer = faiss.IndexFlatL2(d)
        index = faiss.MetalIndexIVFFlat(quantizer, d, nlist)
        index.nprobe = nprobe
        index.add(xt)

        D, I = index.search(xq, k)

        self.assertEqual(I.shape, (nq, k))
        self.assertEqual(D.shape, (nq, k))

    def test_ivfflat_search(self):
        d = 64
        nb = 1000
        nq = 100
        k = 10
        nlist = 16
        nprobe = 4

        xt = np.random.rand(nb, d).astype("float32")
        xq = np.random.rand(nq, d).astype("float32")

        quantizer = faiss.IndexFlatL2(d)
        cpu_index = faiss.IndexIVFFlat(quantizer, d, nlist)
        cpu_index.nprobe = nprobe
        cpu_index.add(xt)
        D_cpu, I_cpu = cpu_index.search(xq, k)

        metal_index = faiss.MetalIndexIVFFlat(quantizer, d, nlist)
        metal_index.nprobe = nprobe
        metal_index.add(xt)
        D_metal, I_metal = metal_index.search(xq, k)

        self.assertTrue(np.allclose(D_cpu, D_metal))
        self.assertTrue(np.array_equal(I_cpu, I_metal))

    def test_ivfpq(self):
        d = 64
        nb = 1000
        nq = 100
        k = 10
        nlist = 16
        M = 8
        nbits = 8
        nprobe = 4

        xt = np.random.rand(nb, d).astype("float32")
        xq = np.random.rand(nq, d).astype("float32")

        quantizer = faiss.IndexFlatL2(d)
        cpu_index = faiss.IndexIVFPQ(quantizer, d, nlist, M, nbits)
        cpu_index.nprobe = nprobe
        cpu_index.train(xt)
        cpu_index.add(xt)
        D_cpu, I_cpu = cpu_index.search(xq, k)

        metal_index = faiss.MetalIndexIVFPQ(quantizer, d, nlist, M, nbits)
        metal_index.nprobe = nprobe
        metal_index.train(xt)
        metal_index.add(xt)
        D_metal, I_metal = metal_index.search(xq, k)

        self.assertTrue(np.allclose(D_cpu, D_metal))
        self.assertTrue(np.array_equal(I_cpu, I_metal))
