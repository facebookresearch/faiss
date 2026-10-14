# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license in the LICENSE file.

"""Dataset is C++ only; binary graph/search coverage lives in C++ tests."""
import unittest
import numpy as np
import faiss


@unittest.skipIf("CUVS" not in faiss.get_compile_options(), "requires cuVS")
class TestBinaryCagraDatasetBoundary(unittest.TestCase):
    def test_pointer_build_requires_dataset(self):
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexBinaryCagra(res, 160)
        xb = np.zeros((64, 20), dtype="uint8")
        for method in (index.train, index.add):
            with self.assertRaisesRegex(RuntimeError, "Dataset"):
                method(xb)
        self.assertFalse(hasattr(index, "updateDataset"))
        self.assertEqual(index.ntotal, 0)

    def test_cpu_hnsw_serialization_and_clone_boundary(self):
        xb = np.random.RandomState(7).randint(0, 256, (128, 20)).astype("uint8")
        cpu = faiss.IndexBinaryHNSWCagra(160, 16)
        cpu.add(xb)
        expected_d, expected_i = cpu.search(xb[:8], 4)
        loaded = faiss.deserialize_index_binary(faiss.serialize_index_binary(cpu))
        actual_d, actual_i = loaded.search(xb[:8], 4)
        np.testing.assert_array_equal(actual_i, expected_i)
        np.testing.assert_array_equal(actual_d, expected_d)
        res = faiss.StandardGpuResources()
        with self.assertRaisesRegex(RuntimeError, "Dataset"):
            faiss.index_binary_cpu_to_gpu(res, 0, loaded)

    def test_idmap_does_not_bypass_dataset_requirement(self):
        res = faiss.StandardGpuResources()
        cagra = faiss.GpuIndexBinaryCagra(res, 160)
        index = faiss.IndexBinaryIDMap(cagra)
        xb = np.zeros((64, 20), dtype="uint8")
        with self.assertRaisesRegex(RuntimeError, "Dataset"):
            index.add_with_ids(xb, np.arange(64, dtype="int64"))
