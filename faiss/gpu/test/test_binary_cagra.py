# @lint-ignore-every LICENSELINT
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) 2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
import unittest

import faiss

from faiss.contrib import evaluation





# @unittest.skipIf(
#     "CUVS" not in faiss.get_compile_options(),
#     "only if cuVS is compiled in")
# class TestInterop(unittest.TestCase):

#     def do_interop(self):
#         d = 64 * 8
#         k = 12

#         res = faiss.StandardGpuResources()

#         index = faiss.GpuIndexBinaryCagra(res, d)
#         xb = np.random.randint(
#             low=0, high=256, size=(1000000, d // 8), dtype=np.uint8)
#         index.train(xb)
#         queries = np.random.randint(
#             low=0, high=256, size=(1000, d // 8), dtype=np.uint8)
#         Dnew, Inew = index.search(queries, k)

#         cpu_index = faiss.index_binary_gpu_to_cpu(index)
#         Dref, Iref = cpu_index.search(queries, k)

#         evaluation.check_ref_knn_with_draws(Dref, Iref, Dnew, Inew, k)

#         deserialized_index = faiss.deserialize_index_binary(
#             faiss.serialize_index_binary(cpu_index))

#         gpu_index = faiss.index_binary_cpu_to_gpu(res, 0, deserialized_index)
#         Dnew2, Inew2 = gpu_index.search(queries, k)

#         evaluation.check_ref_knn_with_draws(Dnew2, Inew2, Dnew, Inew, k)

#     def test_interop(self):
#         self.do_interop()


# @unittest.skipIf(
#     "CUVS" not in faiss.get_compile_options(),
#     "only if cuVS is compiled in")
# class TestComputeGT(unittest.TestCase):

#     def do_compute_GT(self, build_algo=None):
#         d = 64 * 8
#         k = 12
#         flat_index = faiss.IndexBinaryFlat(d)
#         xb = np.random.randint(
#             low=0, high=256, size=(1000000, d // 8), dtype=np.uint8)
#         flat_index.add(xb)
#         queries = np.random.randint(
#             low=0, high=256, size=(1000, d // 8), dtype=np.uint8)
#         Dref, Iref = flat_index.search(queries, k)

#         res = faiss.StandardGpuResources()

#         # Configure the index with specified build algorithm
#         config = faiss.GpuIndexCagraConfig()
#         if build_algo is not None:
#             config.build_algo = build_algo
#             if build_algo == faiss.graph_build_algo_NN_DESCENT:
#                 config.nn_descent_niter = 20

#         index = faiss.GpuIndexBinaryCagra(res, d, config)
#         index.train(xb)
#         Dnew, Inew = index.search(queries, k)

#         evaluation.check_ref_knn_with_draws(Dref, Iref, Dnew, Inew, k)

#     def test_compute_GT_nn_descent(self):
#         self.do_compute_GT(faiss.graph_build_algo_NN_DESCENT)

#     def test_compute_GT_iterative_search(self):
#         self.do_compute_GT(faiss.graph_build_algo_ITERATIVE_SEARCH)


# @unittest.skipIf(
#     "CUVS" not in faiss.get_compile_options(),
#     "only if cuVS is compiled in")
# class TestIndexBinaryIDMap(unittest.TestCase):
#     """Test IndexBinaryIDMap functionality with GpuIndexBinaryCagra"""

#     def test_add_with_ids(self):
#         d = 128 * 8
#         k = 10
#         n = 10000

#         res = faiss.StandardGpuResources()

#         # Create GpuIndexBinaryCagra with IDMap
#         index_gpu = faiss.GpuIndexBinaryCagra(res, d)
#         index_idmap = faiss.IndexBinaryIDMap(index_gpu)

#         # Generate data with custom IDs
#         xb = np.random.randint(
#             low=0, high=256, size=(n, d // 8), dtype=np.uint8)
#         ids = np.arange(1000000, 1000000 + n).astype(np.int64)

#         # Add with IDs
#         index_idmap.add_with_ids(xb, ids)

#         # Search
#         nq = 100
#         xq = np.random.randint(
#             low=0, high=256, size=(nq, d // 8), dtype=np.uint8)
#         D, I = index_idmap.search(xq, k)

#         # Verify returned IDs are in our ID range
#         self.assertTrue(np.all(I >= 1000000))
#         self.assertTrue(np.all(I < 1000000 + n))

#         # Test searching for exact vectors
#         D_exact, I_exact = index_idmap.search(xb[:10], 1)
#         expected_ids = ids[:10].reshape(-1, 1)
#         np.testing.assert_array_equal(I_exact, expected_ids)
#         np.testing.assert_array_equal(D_exact, np.zeros((10, 1)))

# @unittest.skipIf(
#     "CUVS" not in faiss.get_compile_options(),
#     "only if cuVS is compiled in")
# class TestLargeScale(unittest.TestCase):
#     """Test GpuIndexBinaryCagra with large datasets"""

#     def test_large_dataset(self):
#         # Use 1M vectors for faster test, but can be increased to 10M
#         n = 10000000  # Change to 10000000 for 10M test
#         d = 256 * 8  # 256 bytes per vector
#         k = 100
#         nq = 1000

#         res = faiss.StandardGpuResources()
        
#         # Create index
#         index = faiss.GpuIndexBinaryCagra(res, d)
        
#         # Generate data in batches to avoid memory issues
#         batch_size = 10000000
#         for i in range(0, n, batch_size):
#             batch_n = min(batch_size, n - i)
#             xb_batch = np.random.randint(
#                 low=0, high=256, size=(batch_n, d // 8), dtype=np.uint8)
#             if i == 0:
#                 index.train(xb_batch)
#             else:
#                 # For binary CAGRA, we need to retrain with all data
#                 # This is a limitation - in practice, train on a subset
#                 pass
        
#         # For actual testing, train on subset and verify search works
#         xb_train = np.random.randint(
#             low=0, high=256, size=(100000, d // 8), dtype=np.uint8)
#         index.train(xb_train)
        
#         # Test search
#         xq = np.random.randint(
#             low=0, high=256, size=(nq, d // 8), dtype=np.uint8)
#         D, I = index.search(xq, k)
        
#         # Basic sanity checks
#         self.assertEqual(D.shape, (nq, k))
#         self.assertEqual(I.shape, (nq, k))
#         self.assertTrue(np.all(I >= 0))
#         self.assertTrue(np.all(I < 100000))
        
#         # Check distances are sorted
#         for i in range(nq):
#             self.assertTrue(np.all(D[i, :-1] <= D[i, 1:]))


@unittest.skipIf(
    "CUVS" not in faiss.get_compile_options(),
    "only if cuVS is compiled in")
class TestIndexBinaryHNSWInterop(unittest.TestCase):
    """Test interoperability between GpuIndexBinaryCagra and IndexBinaryHNSW"""

    def test_copy_to_hnsw(self):
        d = 128 * 8
        n = 1000000
        k = 32
        nq = 1000

        res = faiss.StandardGpuResources()
        
        # Create and train GpuIndexBinaryCagra
        index_gpu = faiss.GpuIndexBinaryCagra(res, d)
        xb = np.random.randint(
            low=0, high=256, size=(n, d // 8), dtype=np.uint8)
        index_gpu.train(xb)
        
        # Search with GPU index
        xq = np.random.randint(
            low=0, high=256, size=(nq, d // 8), dtype=np.uint8)
        
        
        # Copy to CPU (creates IndexBinaryIVF)
        index_cpu = faiss.index_binary_gpu_to_cpu(index_gpu)
        # D_gpu, I_gpu = index_cpu.search(xq, k)
        
        # Create HNSW index and add same data
        # index_hnsw = faiss.IndexBinaryHNSW(d, 32)
        # index_hnsw.hnsw.efConstruction = 200
        # index_hnsw.hnsw.efSearch = 128
        # index_hnsw.add(xb)
        
        # # Search with HNSW
        # D_hnsw, I_hnsw = index_hnsw.search(xq, k)
        
        # evaluation.check_ref_knn_with_draws(D_hnsw, I_hnsw, D_gpu, I_gpu, k)

    # def test_serialization(self):
    #     d = 64 * 8
    #     n = 1000000
    #     k = 10

    #     res = faiss.StandardGpuResources()
        
    #     # Create and train index
    #     index_gpu = faiss.GpuIndexBinaryCagra(res, d)
    #     xb = np.random.randint(
    #         low=0, high=256, size=(n, d // 8), dtype=np.uint8)
    #     index_gpu.train(xb)
        
    #     # Convert to CPU
    #     index_cpu = faiss.index_binary_gpu_to_cpu(index_gpu)
        
    #     # Serialize and deserialize
    #     data = faiss.serialize_index_binary(index_cpu)
    #     index_cpu2 = faiss.deserialize_index_binary(data)
        
    #     # Convert back to GPU
    #     index_gpu2 = faiss.index_binary_cpu_to_gpu(res, 0, index_cpu2)
        
    #     # Test search produces same results
    #     xq = np.random.randint(
    #         low=0, high=256, size=(100, d // 8), dtype=np.uint8)
    #     D1, I1 = index_gpu.search(xq, k)
    #     D2, I2 = index_gpu2.search(xq, k)
        
    #     np.testing.assert_array_equal(D1, D2)
    #     np.testing.assert_array_equal(I1, I2)


# @unittest.skipIf(
#     "CUVS" not in faiss.get_compile_options(),
#     "only if cuVS is compiled in")
# class TestAdditionalFunctionality(unittest.TestCase):
#     """Test additional important functionality"""

#     def test_different_dimensions(self):
#         """Test with various vector dimensions"""
#         res = faiss.StandardGpuResources()
        
#         # Test different dimensions (must be multiple of 64 bits = 8 bytes)
#         dims = [64 * 8, 128 * 8, 256 * 8, 512 * 8]
        
#         for d in dims:
#             index = faiss.GpuIndexBinaryCagra(res, d)
#             n = 1000
#             xb = np.random.randint(
#                 low=0, high=256, size=(n, d // 8), dtype=np.uint8)
#             index.train(xb)
            
#             # Search
#             xq = np.random.randint(
#                 low=0, high=256, size=(10, d // 8), dtype=np.uint8)
#             D, I = index.search(xq, 5)
            
#             self.assertEqual(D.shape, (10, 5))
#             self.assertEqual(I.shape, (10, 5))

#     def test_empty_index(self):
#         """Test behavior with empty index"""
#         d = 128 * 8
#         res = faiss.StandardGpuResources()
        
#         index = faiss.GpuIndexBinaryCagra(res, d)
        
#         # Try to search empty index (should fail gracefully)
#         xq = np.random.randint(
#             low=0, high=256, size=(10, d // 8), dtype=np.uint8)
        
#         # This should raise an exception since index is not trained
#         with self.assertRaises(RuntimeError):
#             D, I = index.search(xq, 5)

#     def test_reconstruction(self):
#         """Test if reconstruction is supported"""
#         d = 64 * 8
#         n = 1000
#         res = faiss.StandardGpuResources()
        
#         # Create index with store_dataset=True
#         config = faiss.GpuIndexCagraConfig()
#         config.store_dataset = True
        
#         index = faiss.GpuIndexBinaryCagra(res, d, config)
#         xb = np.random.randint(
#             low=0, high=256, size=(n, d // 8), dtype=np.uint8)
#         index.train(xb)
        
#         # Try reconstruction (may not be supported)
#         try:
#             recons = index.reconstruct(0)
#             # If supported, verify dimension
#             self.assertEqual(len(recons), d // 8)
#         except RuntimeError:
#             # Reconstruction might not be supported
#             pass

#     def test_config_parameters(self):
#         """Test various configuration parameters"""
#         d = 128 * 8
#         res = faiss.StandardGpuResources()
        
#         # Test different graph degrees
#         configs = [
#             (32, 64),  # (graph_degree, intermediate_graph_degree)
#             (64, 128),
#             (128, 256),
#         ]
        
#         for graph_degree, intermediate_degree in configs:
#             config = faiss.GpuIndexCagraConfig()
#             config.graph_degree = graph_degree
#             config.intermediate_graph_degree = intermediate_degree
#             config.build_algo = faiss.graph_build_algo_NN_DESCENT
            
#             index = faiss.GpuIndexBinaryCagra(res, d, config)
            
#             n = 5000
#             xb = np.random.randint(
#                 low=0, high=256, size=(n, d // 8), dtype=np.uint8)
#             index.train(xb)
            
#             # Verify search works
#             xq = np.random.randint(
#                 low=0, high=256, size=(10, d // 8), dtype=np.uint8)
#             D, I = index.search(xq, 5)
            
#             self.assertEqual(D.shape, (10, 5))
