# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import faiss
import numpy as np


class TestHNSWRaBitQ(unittest.TestCase):
    def make_data(self, d=32, nt=400, nb=600, nq=20):
        rs = np.random.RandomState(123)
        xt = rs.randn(nt, d).astype("float32")
        xb = rs.randn(nb, d).astype("float32")
        xq = rs.randn(nq, d).astype("float32")
        return xt, xb, xq

    def make_index(self, metric=faiss.METRIC_L2, nb_bits=3):
        xt, xb, xq = self.make_data()
        if metric == faiss.METRIC_INNER_PRODUCT:
            faiss.normalize_L2(xt)
            faiss.normalize_L2(xb)
            faiss.normalize_L2(xq)
        index = faiss.IndexHNSWRaBitQ(xt.shape[1], 8, nb_bits, metric)
        index.hnsw.efConstruction = 40
        index.hnsw.efSearch = 32
        index.train(xt)
        index.add(xb)
        return index, xb, xq

    def test_staged_search_finalize_clone_and_io(self):
        index, xb, xq = self.make_index(nb_bits=3)

        faiss.cvar.rabitq_hnsw_stats.reset()
        Dref, Iref = index.search(xq, 10)
        stats = faiss.cvar.rabitq_hnsw_stats
        self.assertGreater(stats.n_1bit, 0)
        self.assertGreaterEqual(stats.n_1bit, stats.n_refine)
        self.assertTrue(np.all(Iref >= 0))
        self.assertTrue(np.all(np.isfinite(Dref)))

        # Clones and serialized indexes are search-only and do not retain the
        # exact FP32 build storage.
        cloned = faiss.clone_index(index)
        self.assertIsInstance(cloned, faiss.IndexHNSWRaBitQ)
        Dclone, Iclone = cloned.search(xq, 10)
        np.testing.assert_array_equal(Iclone, Iref)
        np.testing.assert_array_equal(Dclone, Dref)

        loaded = faiss.deserialize_index(faiss.serialize_index(index))
        self.assertIsInstance(loaded, faiss.IndexHNSWRaBitQ)
        Dloaded, Iloaded = loaded.search(xq, 10)
        np.testing.assert_array_equal(Iloaded, Iref)
        np.testing.assert_array_equal(Dloaded, Dref)

        index.finalize()
        with self.assertRaises(RuntimeError):
            index.add(xb[:1])

    def test_one_bit_fallback(self):
        index, _, xq = self.make_index(nb_bits=1)
        self.assertFalse(index.hnsw.is_rabitq)
        faiss.cvar.rabitq_hnsw_stats.reset()
        D, I = index.search(xq, 10)
        self.assertTrue(np.all(I >= 0))
        self.assertTrue(np.all(np.isfinite(D)))
        self.assertEqual(faiss.cvar.rabitq_hnsw_stats.n_1bit, 0)
        self.assertEqual(faiss.cvar.rabitq_hnsw_stats.n_refine, 0)

    def test_inner_product_and_reset(self):
        index, xb, xq = self.make_index(
            metric=faiss.METRIC_INNER_PRODUCT, nb_bits=3
        )
        D, I = index.search(xq, 10)
        self.assertTrue(np.all(I >= 0))
        self.assertTrue(np.all(np.isfinite(D)))
        self.assertTrue(np.all(D[:, :-1] >= D[:, 1:]))

        index.reset()
        self.assertEqual(index.ntotal, 0)
        index.add(xb)
        self.assertEqual(index.ntotal, len(xb))

    def test_factory(self):
        default_index = faiss.index_factory(32, "HNSW8,RaBitQ")
        self.assertIsInstance(default_index, faiss.IndexHNSWRaBitQ)
        default_storage = faiss.downcast_index(default_index.storage)
        self.assertIsInstance(default_storage, faiss.IndexRaBitQ)
        self.assertEqual(default_storage.rabitq.nb_bits, 4)

        one_bit = faiss.index_factory(32, "HNSW8,RaBitQ1")
        self.assertIsInstance(one_bit, faiss.IndexHNSWRaBitQ)
        self.assertFalse(one_bit.hnsw.is_rabitq)


if __name__ == "__main__":
    unittest.main()
