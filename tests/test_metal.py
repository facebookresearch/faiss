import faiss
import unittest


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
