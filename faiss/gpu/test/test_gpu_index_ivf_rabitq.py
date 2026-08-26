# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import faiss
import numpy as np


@unittest.skipIf(
    "CUVS" not in faiss.get_compile_options(), "only if cuVS is compiled in"
)
class TestGpuIndexIVFRaBitQ(unittest.TestCase):
    def test_build_search_and_reset(self):
        d, nlist, nb, nq, k = 32, 16, 2048, 16, 10
        rng = np.random.RandomState(123)
        xb = rng.random_sample((nb, d)).astype("float32")
        xq = rng.random_sample((nq, d)).astype("float32")

        resources = faiss.StandardGpuResources()
        config = faiss.GpuIndexIVFRaBitQConfig()
        config.use_cuvs = True
        config.bitsPerDim = 3
        config.nprobe = nlist

        index = faiss.GpuIndexIVFRaBitQ(
            resources, d, nlist, faiss.METRIC_L2, config
        )
        index.add(xb)

        self.assertTrue(index.is_trained)
        self.assertEqual(index.ntotal, nb)

        distances, labels = index.search(xq, k)
        self.assertEqual(distances.shape, (nq, k))
        self.assertEqual(labels.shape, (nq, k))
        self.assertTrue(np.all(labels >= 0))
        self.assertTrue(np.all(labels < nb))

        with self.assertRaises(RuntimeError):
            index.add(xb[:1])

        index.reset()
        self.assertFalse(index.is_trained)
        self.assertEqual(index.ntotal, 0)

        index.add(xb)
        self.assertTrue(index.is_trained)
        self.assertEqual(index.ntotal, nb)

    def test_rejects_inner_product(self):
        resources = faiss.StandardGpuResources()
        config = faiss.GpuIndexIVFRaBitQConfig()
        config.use_cuvs = True

        with self.assertRaises(RuntimeError):
            faiss.GpuIndexIVFRaBitQ(
                resources, 32, 16, faiss.METRIC_INNER_PRODUCT, config
            )
