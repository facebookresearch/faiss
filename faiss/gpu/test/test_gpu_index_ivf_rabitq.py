# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import faiss
import numpy as np


@unittest.skipIf(
    "CUVS" not in faiss.get_compile_options() or faiss.get_num_gpus() == 0,
    "requires cuVS support and a CUDA device",
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
        self.assertTrue(np.all(np.isfinite(distances)))
        self.assertTrue(np.all(labels >= 0))
        self.assertTrue(np.all(labels < nb))
        self.assertTrue(np.all(distances[:, :-1] <= distances[:, 1:]))

        exact_index = faiss.IndexFlatL2(d)
        exact_index.add(xb)
        _, exact_labels = exact_index.search(xq, k)
        recall = np.mean(
            [
                len(set(exact_labels[i]).intersection(labels[i])) / k
                for i in range(nq)
            ]
        )
        self.assertGreaterEqual(recall, 0.5)

        params = faiss.SearchParametersIVFRaBitQ()
        params.nprobe = nlist
        params.searchMode = faiss.IVFRaBitQSearchMode_LUT16
        param_distances, param_labels = index.search(xq, k, params=params)
        self.assertTrue(np.all(np.isfinite(param_distances)))
        self.assertTrue(np.all((param_labels >= 0) & (param_labels < nb)))
        self.assertTrue(
            np.all(param_distances[:, :-1] <= param_distances[:, 1:])
        )

        params.nprobe = 0
        with self.assertRaises(RuntimeError):
            index.search(xq, k, params=params)

        params.nprobe = 16385
        with self.assertRaises(RuntimeError):
            index.search(xq, k, params=params)

        with self.assertRaises(RuntimeError):
            index.search(xq, k, params=faiss.SearchParameters())

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

    def test_rejects_invalid_configuration(self):
        resources = faiss.StandardGpuResources()

        for bits_per_dim in (0, 10):
            config = faiss.GpuIndexIVFRaBitQConfig()
            config.use_cuvs = True
            config.bitsPerDim = bits_per_dim
            with self.assertRaises(RuntimeError):
                faiss.GpuIndexIVFRaBitQ(
                    resources, 32, 16, faiss.METRIC_L2, config
                )

        config = faiss.GpuIndexIVFRaBitQConfig()
        config.use_cuvs = True
        with self.assertRaises(RuntimeError):
            faiss.GpuIndexIVFRaBitQ(resources, 32, 0, faiss.METRIC_L2, config)

        for field in (
            "kmeansNIterations",
            "maxTrainPointsPerCluster",
            "streamingBatchSize",
        ):
            config = faiss.GpuIndexIVFRaBitQConfig()
            config.use_cuvs = True
            setattr(config, field, 0)
            with self.assertRaises(RuntimeError):
                faiss.GpuIndexIVFRaBitQ(
                    resources, 32, 16, faiss.METRIC_L2, config
                )
