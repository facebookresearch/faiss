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
    def assert_valid_results(self, distances, labels, nb):
        self.assertTrue(np.all(np.isfinite(distances)))
        self.assertTrue(np.all((labels >= 0) & (labels < nb)))
        self.assertTrue(np.all(distances[:, :-1] <= distances[:, 1:]))

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
        self.assert_valid_results(distances, labels, nb)

        exact_index = faiss.IndexFlatL2(d)
        exact_index.add(xb)
        _, exact_labels = exact_index.search(xq, k)
        for mode in (
            faiss.IVFRaBitQSearchMode_LUT16,
            faiss.IVFRaBitQSearchMode_LUT32,
            faiss.IVFRaBitQSearchMode_QUANT4,
            faiss.IVFRaBitQSearchMode_QUANT8,
        ):
            with self.subTest(search_mode=mode):
                params = faiss.SearchParametersIVFRaBitQ()
                params.nprobe = nlist
                params.searchMode = mode
                distances, labels = index.search(xq, k, params=params)
                self.assert_valid_results(distances, labels, nb)
                recall = np.mean(
                    [
                        len(set(exact_labels[i]).intersection(labels[i])) / k
                        for i in range(nq)
                    ]
                )
                self.assertGreaterEqual(recall, 0.5)

        # Exercise the inherited Float16 query dispatch through Python wrappers.
        rounded_queries = xq.astype("float16")
        float_distances, float_labels = index.search(
            rounded_queries.astype("float32"), k
        )
        half_distances, half_labels = index.search(
            rounded_queries, k, numeric_type=faiss.Float16
        )
        np.testing.assert_array_equal(half_labels, float_labels)
        np.testing.assert_allclose(half_distances, float_distances, rtol=1e-6)

        with self.assertRaises(RuntimeError):
            index.add(xb[:1])
        with self.assertRaises(RuntimeError):
            index.add_with_ids(xb[:1], np.array([42], dtype="int64"))
        self.assertEqual(index.ntotal, nb)
        with self.assertRaises(RuntimeError):
            faiss.index_gpu_to_cpu(index)
        with self.assertRaises(RuntimeError):
            faiss.serialize_index(index)

        index.reset()
        self.assertFalse(index.is_trained)
        self.assertEqual(index.ntotal, 0)
        with self.assertRaises(RuntimeError):
            index.search(xq, k)
        with self.assertRaises(RuntimeError):
            index.add_with_ids(xb[:1], np.array([42], dtype="int64"))
        with self.assertRaises(RuntimeError):
            index.train(xb, numeric_type=faiss.Float16)
        with self.assertRaises(RuntimeError):
            index.add(xb, numeric_type=faiss.Float16)
        self.assertFalse(index.is_trained)
        self.assertEqual(index.ntotal, 0)

        # train() builds the complete GPU-native index, just like first add().
        index.train(xb)
        self.assertTrue(index.is_trained)
        self.assertEqual(index.ntotal, nb)
        distances, labels = index.search(xq, k)
        self.assert_valid_results(distances, labels, nb)

    def test_supported_bit_endpoints(self):
        d, nb, nlist = 32, 1024, 8
        rng = np.random.RandomState(234)
        xb = rng.random_sample((nb, d)).astype("float32")
        xq = rng.random_sample((8, d)).astype("float32")
        resources = faiss.StandardGpuResources()
        for bits_per_dim in (1, 9):
            with self.subTest(bits_per_dim=bits_per_dim):
                config = faiss.GpuIndexIVFRaBitQConfig()
                config.use_cuvs = True
                config.bitsPerDim = bits_per_dim
                config.nprobe = nlist
                index = faiss.GpuIndexIVFRaBitQ(
                    resources, d, nlist, faiss.METRIC_L2, config
                )
                index.add(xb)
                distances, labels = index.search(xq, 10)
                self.assert_valid_results(distances, labels, nb)

    def test_host_streaming_build(self):
        d, nb, nlist = 32, 1024, 8
        rng = np.random.RandomState(345)
        xb = rng.random_sample((nb, d)).astype("float32")
        xq = rng.random_sample((8, d)).astype("float32")
        resources = faiss.StandardGpuResources()
        config = faiss.GpuIndexIVFRaBitQConfig()
        config.use_cuvs = True
        config.nprobe = nlist
        config.forceStreaming = True
        config.streamingBatchSize = 256
        index = faiss.GpuIndexIVFRaBitQ(
            resources, d, nlist, faiss.METRIC_L2, config
        )
        index.train(xb)
        self.assertTrue(index.is_trained)
        self.assertEqual(index.ntotal, nb)
        distances, labels = index.search(xq, 10)
        self.assert_valid_results(distances, labels, nb)

    def test_rejects_streaming_batch_smaller_than_cluster(self):
        d, nb, nlist = 32, 1024, 8
        rng = np.random.RandomState(346)
        xb = rng.random_sample((nb, d)).astype("float32")
        resources = faiss.StandardGpuResources()
        config = faiss.GpuIndexIVFRaBitQConfig()
        config.use_cuvs = True
        config.forceStreaming = True
        config.streamingBatchSize = 1
        index = faiss.GpuIndexIVFRaBitQ(
            resources, d, nlist, faiss.METRIC_L2, config
        )
        with self.assertRaisesRegex(RuntimeError, "max cluster size"):
            index.train(xb)
        self.assertFalse(index.is_trained)
        self.assertEqual(index.ntotal, 0)
        index.reset()

    def test_rejects_invalid_build_and_search_inputs(self):
        d, nb, nlist = 32, 1024, 8
        rng = np.random.RandomState(456)
        xb = rng.random_sample((nb, d)).astype("float32")
        xq = rng.random_sample((2, d)).astype("float32")
        resources = faiss.StandardGpuResources()
        config = faiss.GpuIndexIVFRaBitQConfig()
        config.use_cuvs = True
        config.nprobe = nlist
        index = faiss.GpuIndexIVFRaBitQ(
            resources, d, nlist, faiss.METRIC_L2, config
        )
        for n in (0, nlist - 1):
            with self.subTest(training_rows=n), self.assertRaises(RuntimeError):
                index.train(xb[:n])
        with self.assertRaises(RuntimeError):
            index.train_c(nb, None)
        self.assertFalse(index.is_trained)
        self.assertEqual(index.ntotal, 0)
        index.train(xb)

        with self.assertRaises(RuntimeError):
            index.search(xq, 10, params=faiss.SearchParameters())
        params = faiss.SearchParametersIVFRaBitQ()
        for nprobe in (0, 16385):
            params.nprobe = nprobe
            with self.subTest(nprobe=nprobe), self.assertRaises(RuntimeError):
                index.search(xq, 10, params=params)
        params.nprobe = nlist
        params.searchMode = -1
        with self.assertRaises(RuntimeError):
            index.search(xq, 10, params=params)
        params.searchMode = faiss.IVFRaBitQSearchMode_QUANT4
        selector = faiss.IDSelectorRange(0, nb // 2)
        params.sel = selector
        with self.assertRaises(RuntimeError):
            index.search(xq, 10, params=params)
        params.sel = None
        params.nprobe = nlist + 1
        distances, labels = index.search(xq, 10, params=params)
        self.assert_valid_results(distances, labels, nb)
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
        for field, value in (
            ("bitsPerDim", 0),
            ("bitsPerDim", 10),
            ("kmeansNIterations", 0),
            ("maxTrainPointsPerCluster", 0),
            ("streamingBatchSize", 0),
            ("nprobe", 0),
            ("nprobe", 16385),
            ("searchMode", -1),
            ("use_cuvs", False),
        ):
            with self.subTest(field=field, value=value):
                config = faiss.GpuIndexIVFRaBitQConfig()
                config.use_cuvs = True
                setattr(config, field, value)
                with self.assertRaises(RuntimeError):
                    faiss.GpuIndexIVFRaBitQ(
                        resources, 32, 16, faiss.METRIC_L2, config
                    )
        config = faiss.GpuIndexIVFRaBitQConfig()
        config.use_cuvs = True
        for dim, nlist in ((0, 16), (-1, 16), (32, 0), (32, 2**32)):
            with self.subTest(dim=dim, nlist=nlist):
                with self.assertRaises(RuntimeError):
                    faiss.GpuIndexIVFRaBitQ(
                        resources, dim, nlist, faiss.METRIC_L2, config
                    )
