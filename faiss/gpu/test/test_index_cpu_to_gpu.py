import numpy as np
import unittest
import faiss


class TestMoveToGpu(unittest.TestCase):
    def test_index_cpu_to_gpu(self):
        dimension = 128
        n = 2500
        db_vectors = np.random.random((n, dimension)).astype('float32')
        code_size = 16
        res = faiss.StandardGpuResources()
        index_pq = faiss.IndexPQ(dimension, code_size, 6)
        index_pq.train(db_vectors)
        index_pq.add(db_vectors)
        self.assertRaisesRegex(Exception, ".*not implemented.*",
                               faiss.index_cpu_to_gpu, res, 0, index_pq)

    def test_index_cpu_to_gpu_does_not_throw_with_index_flat(self):
        dimension = 128
        n = 100
        db_vectors = np.random.random((n, dimension)).astype('float32')
        res = faiss.StandardGpuResources()
        index_flat = faiss.IndexFlatL2(dimension)
        index_flat.add(db_vectors)
        try:
            faiss.index_cpu_to_gpu(res, 0, index_flat)
        except Exception:
            self.fail("index_cpu_to_gpu() threw an unexpected exception.")
