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
