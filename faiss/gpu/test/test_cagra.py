# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license in the LICENSE file.

"""Python migration contract; CAGRA search/interop is exercised in C++ tests.

Dataset has no Python binding yet. CPU HNSW storage and serialization remain
available, but GPU CAGRA construction requires a C++ Dataset.
"""

import unittest
import numpy as np
import faiss


@unittest.skipIf("CUVS" not in faiss.get_compile_options(), "requires cuVS")
class TestCagraDatasetBoundary(unittest.TestCase):
    def test_dataset_is_cpp_only(self):
        self.assertFalse(hasattr(faiss, "Dataset"))
        self.assertFalse(hasattr(faiss.GpuIndexCagra, "updateDataset"))

    def test_pointer_build_requires_dataset(self):
        res = faiss.StandardGpuResources()
        for dtype, numeric_type in ((np.float32, faiss.Float32),
                                    (np.float16, faiss.Float16),
                                    (np.int8, faiss.Int8)):
            with self.subTest(dtype=dtype):
                index = faiss.GpuIndexCagra(res, 20)
                xb = np.zeros((64, 20), dtype=dtype)
                for method in (index.train, index.add):
                    with self.assertRaisesRegex(RuntimeError, "Dataset"):
                        method(xb, numeric_type=numeric_type)
                self.assertEqual(index.ntotal, 0)

    def test_cpu_hnsw_stays_independent(self):
        xb = np.random.RandomState(7).random_sample((128, 20)).astype("float32")
        cpu = faiss.IndexHNSWCagra(20, 16)
        cpu.add(xb)
        expected_d, expected_i = cpu.search(xb[:8], 4)
        loaded = faiss.deserialize_index(faiss.serialize_index(cpu))
        actual_d, actual_i = loaded.search(xb[:8], 4)
        np.testing.assert_array_equal(actual_i, expected_i)
        np.testing.assert_allclose(actual_d, expected_d)
        res = faiss.StandardGpuResources()
        with self.assertRaisesRegex(RuntimeError, "Dataset"):
            faiss.index_cpu_to_gpu(res, 0, loaded)

    def test_idmap_does_not_bypass_dataset_requirement(self):
        res = faiss.StandardGpuResources()
        cagra = faiss.GpuIndexCagra(res, 20)
        index = faiss.IndexIDMap(cagra)
        xb = np.zeros((64, 20), dtype="float32")
        with self.assertRaisesRegex(RuntimeError, "Dataset"):
            index.train(xb)


@unittest.skipIf(
    "CUVS" not in faiss.get_compile_options(), "only if cuVS is compiled in"
)
class TestCagraConfig(unittest.TestCase):
    """Config surface for the multi-GPU build, checkable without 2 GPUs."""

    def test_multi_gpu_knobs_survive_swig(self):
        devices = faiss.Int32Vector()
        devices.push_back(0)
        devices.push_back(1)

        all_neighbors = faiss.AllNeighborsCagraConfig()
        all_neighbors.n_clusters = 16
        all_neighbors.overlap_factor = 3
        all_neighbors.ivf_pq_search_batch_size = 8192

        config = faiss.GpuIndexCagraConfig()
        config.devices = devices
        config.all_neighbors_params = all_neighbors

        self.assertEqual(config.devices.size(), 2)
        self.assertEqual(config.devices.at(1), 1)
        stored = config.all_neighbors_params
        self.assertEqual(
            (
                stored.n_clusters,
                stored.overlap_factor,
                stored.ivf_pq_search_batch_size,
            ),
            (16, 3, 8192),
        )
