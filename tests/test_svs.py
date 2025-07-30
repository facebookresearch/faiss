# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import unittest
import tempfile
import faiss


class TestSVSAdapter(unittest.TestCase):
    """Test the FAISS-SVS adapter layer integration"""

    target_class = faiss.IndexSVS

    def setUp(self):
        self.d = 32
        self.nb = 1000
        self.nq = 100
        np.random.seed(1234)
        self.xb = np.random.random((self.nb, self.d)).astype('float32')
        self.xq = np.random.random((self.nq, self.d)).astype('float32')

    def test_svs_construction(self):
        """Test construction and basic properties"""
        # Test default construction
        index = self.target_class(self.d)
        self.assertEqual(index.d, self.d)
        self.assertTrue(index.is_trained)
        self.assertEqual(index.ntotal, 0)
        self.assertEqual(index.metric_type, faiss.METRIC_L2)

        index_ip = self.target_class(self.d, faiss.METRIC_INNER_PRODUCT)
        self.assertEqual(index_ip.metric_type, faiss.METRIC_INNER_PRODUCT)

    def test_svs_add_search_interface(self):
        """Test FAISS add/search interface compatibility"""
        index = self.target_class(self.d)

        # Test add interface
        index.add(self.xb)
        self.assertEqual(index.ntotal, self.nb)

        # Test search interface
        k = 4
        D, I = index.search(self.xq, k)
        self.assertEqual(D.shape, (self.nq, k))
        self.assertEqual(I.shape, (self.nq, k))
        self.assertTrue(np.all(I >= 0))
        self.assertTrue(np.all(I < self.nb))

        # Test reset
        index.reset()
        self.assertEqual(index.ntotal, 0)

    def test_svs_metric_types(self):
        """Test different metric types are handled correctly"""
        # L2 metric
        index_l2 = self.target_class(self.d, faiss.METRIC_L2)
        index_l2.add(self.xb)
        D_l2, _ = index_l2.search(self.xq[:10], 4)

        index_ip = self.target_class(self.d, faiss.METRIC_INNER_PRODUCT)
        index_ip.add(self.xb)
        D_ip, _ = index_ip.search(self.xq[:10], 4)

        # Results should be different (testing adapter forwards metric correctly)
        self.assertFalse(np.array_equal(D_l2, D_ip))

    def test_svs_serialization(self):
        """Test FAISS serialization system works with SVS indices"""
        index = self.target_class(self.d)

        index.num_threads = 2

        index.add(self.xb)
        D_before, I_before = index.search(self.xq, 4)

        with tempfile.NamedTemporaryFile() as f:
            faiss.write_index(index, f.name)
            loaded = faiss.read_index(f.name)

        # Verify adapter layer preserves type and parameters
        self.assertIsInstance(loaded, self.target_class)
        self.assertEqual(loaded.d, self.d)
        self.assertEqual(loaded.ntotal, self.nb)
        self.assertEqual(loaded.metric_type, index.metric_type)
        self.assertEqual(loaded.num_threads, index.num_threads)

        # Verify functionality is preserved
        D_after, I_after = loaded.search(self.xq, 4)
        np.testing.assert_array_equal(I_before, I_after)
        np.testing.assert_allclose(D_before, D_after, rtol=1e-6)

    def test_svs_error_handling(self):
        """Test that FAISS error handling works with SVS indices"""
        index = self.target_class(self.d)

        # Test search before adding data
        with self.assertRaises(RuntimeError):
            index.search(self.xq, 4)

        # Test wrong dimension
        wrong_dim_data = np.random.random((100, self.d + 1)).astype('float32')
        with self.assertRaises(AssertionError):
            index.add(wrong_dim_data)

    def test_svs_fourcc_handling(self):
        """Test that FAISS I/O system handles SVS fourccs correctly"""
        # Create and populate index
        index = self.target_class(self.d)
        index.add(self.xb[:100])  # Smaller dataset for speed

        # Test round-trip serialization preserves exact type
        with tempfile.NamedTemporaryFile() as f:
            faiss.write_index(index, f.name)
            loaded = faiss.read_index(f.name)

            # Verify exact type preservation (fourcc working correctly)
            self.assertEqual(type(loaded), self.target_class)

    def test_svs_batch_operations(self):
        """Test that batch operations work correctly through adapter"""
        index = self.target_class(self.d)

        # Add in multiple batches
        batch_size = 250
        for i in range(0, self.nb, batch_size):
            end_idx = min(i + batch_size, self.nb)
            index.add(self.xb[i:end_idx])

        self.assertEqual(index.ntotal, self.nb)

        # Verify search still works after batch operations
        D, _ = index.search(self.xq, 4)
        self.assertEqual(D.shape, (self.nq, 4))


class TestSVSAdapterLVQ4x4(TestSVSAdapter):
    """Repeat all tests for SVSLVQ4x4 variant"""
    target_class = faiss.IndexSVSLVQ4x4

class TestSVSAdapterLVQ4x8(TestSVSAdapter):
    """Repeat all tests for SVSLVQ4x8 variant"""
    target_class = faiss.IndexSVSLVQ4x8

class TestSVSAdapterFlat(TestSVSAdapter):
    """Repeat all tests for SVSFlat variant"""
    target_class = faiss.IndexSVSFlat


class TestSVSVamanaParameters(unittest.TestCase):
    """Test Vamana-specific parameter forwarding and persistence for SVS Vamana variants"""

    target_class = faiss.IndexSVS

    def setUp(self):
        self.d = 32
        self.nb = 500  # Smaller dataset for parameter tests
        self.nq = 50
        np.random.seed(1234)
        self.xb = np.random.random((self.nb, self.d)).astype('float32')
        self.xq = np.random.random((self.nq, self.d)).astype('float32')

    def test_vamana_parameter_setting(self):
        """Test that all Vamana parameters can be set and retrieved"""
        index = self.target_class(self.d)

        # Set non-default values for all parameters
        index.num_threads = 4
        index.graph_max_degree = 32
        index.alpha = 1.5
        index.search_window_size = 20
        index.search_buffer_capacity = 25
        index.construction_window_size = 80
        index.max_candidate_pool_size = 150
        index.prune_to = 30
        index.use_full_search_history = False

        # Verify all parameters are set correctly
        self.assertEqual(index.num_threads, 4)
        self.assertEqual(index.graph_max_degree, 32)
        self.assertAlmostEqual(index.alpha, 1.5, places=6)
        self.assertEqual(index.search_window_size, 20)
        self.assertEqual(index.search_buffer_capacity, 25)
        self.assertEqual(index.construction_window_size, 80)
        self.assertEqual(index.max_candidate_pool_size, 150)
        self.assertEqual(index.prune_to, 30)
        self.assertEqual(index.use_full_search_history, False)

    def test_vamana_parameter_defaults(self):
        """Test that Vamana parameters have correct default values"""
        index = self.target_class(self.d)

        # Verify default values match C++ header
        self.assertEqual(index.num_threads, 1)
        self.assertEqual(index.graph_max_degree, 64)
        self.assertAlmostEqual(index.alpha, 1.2, places=6)
        self.assertEqual(index.search_window_size, 10)
        self.assertEqual(index.search_buffer_capacity, 10)
        self.assertEqual(index.construction_window_size, 40)
        self.assertEqual(index.max_candidate_pool_size, 200)
        self.assertEqual(index.prune_to, 60)
        self.assertEqual(index.use_full_search_history, True)

    def test_vamana_parameter_serialization(self):
        """Test that all Vamana parameters are preserved through serialization"""
        index = self.target_class(self.d)

        # Set distinctive non-default values
        index.num_threads = 8
        index.graph_max_degree = 48
        index.alpha = 1.8
        index.search_window_size = 15
        index.search_buffer_capacity = 18
        index.construction_window_size = 60
        index.max_candidate_pool_size = 180
        index.prune_to = 45
        index.use_full_search_history = False

        # Add data and train
        index.add(self.xb)

        # Serialize and deserialize
        with tempfile.NamedTemporaryFile() as f:
            faiss.write_index(index, f.name)
            loaded = faiss.read_index(f.name)

        # Verify all parameters are preserved
        self.assertIsInstance(loaded, self.target_class)
        self.assertEqual(loaded.num_threads, 8)
        self.assertEqual(loaded.graph_max_degree, 48)
        self.assertAlmostEqual(loaded.alpha, 1.8, places=6)
        self.assertEqual(loaded.search_window_size, 15)
        self.assertEqual(loaded.search_buffer_capacity, 18)
        self.assertEqual(loaded.construction_window_size, 60)
        self.assertEqual(loaded.max_candidate_pool_size, 180)
        self.assertEqual(loaded.prune_to, 45)
        self.assertEqual(loaded.use_full_search_history, False)

        # Verify results are unaffected
        D_before, I_before = index.search(self.xq, 4)
        D_after, I_after = loaded.search(self.xq, 4)
        np.testing.assert_array_equal(I_before, I_after)
        np.testing.assert_allclose(D_before, D_after, rtol=1e-6)


class TestSVSVamanaParametersLVQ4x4(TestSVSVamanaParameters):
    """Repeat Vamana parameter tests for SVSLVQ4x4 variant"""
    target_class = faiss.IndexSVSLVQ4x4


class TestSVSVamanaParametersLVQ4x8(TestSVSVamanaParameters):
    """Repeat Vamana parameter tests for SVSLVQ4x8 variant"""
    target_class = faiss.IndexSVSLVQ4x8

if __name__ == '__main__':
    unittest.main()
