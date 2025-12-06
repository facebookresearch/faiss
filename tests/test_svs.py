# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Portions Copyright 2025 Intel Corporation
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

_SKIP_SVS = "SVS" not in faiss.get_compile_options().split()
_SKIP_REASON = "SVS support not compiled in"

# Check if LVQ/LeanVec support is available
_SKIP_SVS_LL = _SKIP_SVS or not faiss.IndexSVSVamana.is_lvq_leanvec_enabled()
_SKIP_SVS_LL_REASON = "LVQ/LeanVec support not available on this platform or build configuration"


@unittest.skipIf(_SKIP_SVS, _SKIP_REASON)
class TestSVSAdapter(unittest.TestCase):
    """Test the FAISS-SVS adapter layer integration"""

    target_class = None  # set in setUpClass

    def _create_instance(self) -> "faiss.IndexSVSVamana | faiss.IndexSVSFlat":
        """Create an instance of the SVS index"""
        self.assertIsNotNone(self.target_class, "target_class must be configured in setUpClass()")
        return self.target_class(self.d, 64)

    @classmethod
    def setUpClass(cls):
        # need to configure target_class here to avoid issues when SVS support is not compiled in
        cls.target_class = faiss.IndexSVSVamana

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
        index = self._create_instance()
        self.assertEqual(index.d, self.d)
        self.assertTrue(index.is_trained)
        self.assertEqual(index.ntotal, 0)
        self.assertEqual(index.metric_type, faiss.METRIC_L2)

        index_ip = self._create_instance()
        index_ip.metric_type = faiss.METRIC_INNER_PRODUCT
        self.assertEqual(index_ip.metric_type, faiss.METRIC_INNER_PRODUCT)

    def test_svs_add_search_remove_interface(self):
        """Test FAISS add/search/remove_ids interface compatibility"""
        index = self._create_instance()

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

        # Test remove
        ids = np.arange(index.ntotal)
        toremove = np.ascontiguousarray(ids[0:200:3])
        sel = faiss.IDSelectorArray(50, faiss.swig_ptr(toremove[:50]))
        nremove = index.remove_ids(sel)
        nremove += index.remove_ids(toremove[50:])

        self.assertEqual(nremove, len(toremove))

        # remove more to trigger cleanup
        toremove = np.ascontiguousarray(ids[200:800])
        nremove = index.remove_ids(toremove)
        self.assertEqual(nremove, len(toremove))

        # Test reset
        index.reset()
        self.assertEqual(index.ntotal, 0)

    def test_svs_search_selected(self):
        """Test FAISS search with IDSelector interface compatibility"""
        index = self._create_instance()

        # Test add interface
        index.add(self.xb)
        self.assertEqual(index.ntotal, self.nb)

        # Create selector to select a subset of ids
        min = self.nb // 5
        max = self.nb * 4 // 5
        sel = faiss.IDSelectorRange(min, max)  # select ids in [100, 200)
        params = faiss.SearchParameters(sel=sel)

        # Test search interface
        k = 10
        D, I = index.search(self.xq, k, params=params)
        self.assertEqual(D.shape, (self.nq, k))
        self.assertEqual(I.shape, (self.nq, k))
        self.assertTrue(np.all(I >= min))
        self.assertTrue(np.all(I < max))

    def test_svs_range_search(self):
        """Test FAISS range_search interface compatibility"""
        index = self._create_instance()

        # Test add interface
        index.add(self.xb)
        self.assertEqual(index.ntotal, self.nb)

        # Test search interface
        range = 0.1
        lims, D, I = index.range_search(self.xq, range)
        self.assertEqual(D.shape, I.shape)
        self.assertTrue(np.all(D <= range))
        self.assertTrue(np.all(I >= 0))
        self.assertTrue(np.all(I < self.nb))

    def test_svs_range_search_ip(self):
        """Test FAISS range_search interface compatibility"""
        index = self._create_instance()
        index.metric_type = faiss.METRIC_INNER_PRODUCT
        index.alpha = 0.95

        # Test add interface
        index.add(self.xb)
        self.assertEqual(index.ntotal, self.nb)

        # Test search interface
        range = 10
        lims, D, I = index.range_search(self.xq, range)
        self.assertEqual(D.shape, I.shape)
        self.assertTrue(np.all(D >= range))
        self.assertTrue(np.all(I >= 0))
        self.assertTrue(np.all(I < self.nb))

    def test_svs_range_search_selected(self):
        """Test FAISS add/search/remove_ids interface compatibility"""
        index = self._create_instance()

        # Test add interface
        index.add(self.xb)
        self.assertEqual(index.ntotal, self.nb)

        # Create selector to select a subset of ids
        min = self.nb // 5
        max = self.nb * 4 // 5
        sel = faiss.IDSelectorRange(min, max)  # select ids in [100, 200)
        params = faiss.SearchParameters(sel=sel)

        # Test search interface
        radius = 0.1
        lims, D, I = index.range_search(self.xq, radius, params=params)
        self.assertEqual(D.shape, I.shape)
        self.assertTrue(np.all(D <= radius))
        self.assertTrue(np.all(I >= min))
        self.assertTrue(np.all(I < max))

        # Test reset
        index.reset()
        self.assertEqual(index.ntotal, 0)

    def test_svs_metric_types(self):
        """Test different metric types are handled correctly"""
        # L2 metric
        index_l2 = self._create_instance()
        index_l2.metric_type = faiss.METRIC_L2
        index_l2.add(self.xb)
        D_l2, _ = index_l2.search(self.xq[:10], 4)

        index_ip = self._create_instance()
        index_ip.metric_type = faiss.METRIC_INNER_PRODUCT
        index_ip.alpha = 0.95
        index_ip.add(self.xb)
        D_ip, _ = index_ip.search(self.xq[:10], 4)

        # Results should be different (testing adapter forwards metric correctly)
        self.assertFalse(np.array_equal(D_l2, D_ip))

    def test_svs_serialization(self):
        """Test FAISS serialization system works with SVS indices"""
        index = self._create_instance()

        index.add(self.xb)
        D_before, I_before = index.search(self.xq, 4)

        loaded = faiss.deserialize_index(faiss.serialize_index(index))
        # Verify adapter layer preserves type and parameters
        self.assertIsInstance(loaded, self.target_class)
        self.assertEqual(loaded.d, self.d)
        self.assertEqual(loaded.ntotal, self.nb)
        self.assertEqual(loaded.metric_type, index.metric_type)

        # Verify functionality is preserved
        D_after, I_after = loaded.search(self.xq, 4)
        np.testing.assert_array_equal(I_before, I_after)
        np.testing.assert_allclose(D_before, D_after, rtol=1e-6)

    def test_svs_error_handling(self):
        """Test that FAISS error handling works with SVS indices"""
        index = self._create_instance()

        # Test wrong dimension
        wrong_dim_data = np.random.random((100, self.d + 1)).astype('float32')
        with self.assertRaises(AssertionError):
            index.add(wrong_dim_data)

    def test_svs_fourcc_handling(self):
        """Test that FAISS I/O system handles SVS fourccs correctly"""
        # Create and populate index
        index = self._create_instance()
        index.add(self.xb[:100])  # Smaller dataset for speed

        # Test round-trip serialization preserves exact type
        loaded = faiss.deserialize_index(faiss.serialize_index(index))

        # Verify exact type preservation (fourcc working correctly)
        self.assertEqual(type(loaded), self.target_class)

    def test_svs_batch_operations(self):
        """Test that batch operations work correctly through adapter"""
        index = self._create_instance()

        # Add in multiple batches
        batch_size = 250
        for i in range(0, self.nb, batch_size):
            end_idx = min(i + batch_size, self.nb)
            index.add(self.xb[i:end_idx])

        self.assertEqual(index.ntotal, self.nb)

        # Verify search still works after batch operations
        D, _ = index.search(self.xq, 4)
        self.assertEqual(D.shape, (self.nq, 4))

@unittest.skipIf(_SKIP_SVS, _SKIP_REASON)
class TestSVSFactory(unittest.TestCase):
    """Test that SVS factory works correctly"""

    def test_svs_factory_flat(self):
        index = faiss.index_factory(32, "SVSFlat")
        self.assertEqual(index.d, 32)

    def test_svs_factory_vamana(self):
        index = faiss.index_factory(32, "SVSVamana64")
        self.assertEqual(index.d, 32)
        self.assertEqual(index.graph_max_degree, 64)
        self.assertEqual(index.metric_type, faiss.METRIC_L2)
        self.assertEqual(index.storage_kind, faiss.SVS_FP32)

    def test_svs_factory_fp16(self):
        index = faiss.index_factory(256, "SVSVamana16,FP16")
        self.assertEqual(index.d, 256)
        self.assertEqual(index.graph_max_degree, 16)
        self.assertEqual(index.storage_kind, faiss.SVS_FP16)

    def test_svs_factory_sqi8(self):
        index = faiss.index_factory(64, "SVSVamana24,SQI8")
        self.assertEqual(index.d, 64)
        self.assertEqual(index.graph_max_degree, 24)
        self.assertEqual(index.storage_kind, faiss.SVS_SQI8)

@unittest.skipIf(_SKIP_SVS_LL, _SKIP_SVS_LL_REASON)
class TestSVSFactoryLVQLeanVec(unittest.TestCase):
    """Test that SVS factory works correctly for LVQ and LeanVec"""

    def test_svs_factory_lvq(self):
        index = faiss.index_factory(16, "SVSVamana32,LVQ4x8")
        self.assertEqual(index.d, 16)
        self.assertEqual(index.graph_max_degree, 32)
        self.assertEqual(index.storage_kind, faiss.SVS_LVQ4x8)

    def test_svs_factory_leanvec(self):
        index = faiss.index_factory(128, "SVSVamana48,LeanVec4x4_64")
        self.assertEqual(index.d, 128)
        self.assertEqual(index.graph_max_degree, 48)
        self.assertEqual(index.storage_kind, faiss.SVS_LeanVec4x4)
        self.assertEqual(index.leanvec_d, 64)


@unittest.skipIf(_SKIP_SVS, _SKIP_REASON)
class TestSVSAdapterFP16(TestSVSAdapter):
    """Repeat all tests for SVS Float16 variant"""
    def _create_instance(self):
        idx = self.target_class(self.d, 64)
        idx.storage_kind = faiss.SVS_FP16
        return idx

@unittest.skipIf(_SKIP_SVS, _SKIP_REASON)
class TestSVSAdapterSQI8(TestSVSAdapter):
    """Repeat all tests for SVS SQ int8 variant"""
    def _create_instance(self):
        idx = self.target_class(self.d, 64)
        idx.storage_kind = faiss.SVS_SQI8
        return idx

@unittest.skipIf(_SKIP_SVS_LL, _SKIP_SVS_LL_REASON)
class TestSVSAdapterLVQ4x0(TestSVSAdapter):
    """Repeat all tests for SVSLVQ4x0 variant"""

    @classmethod
    def setUpClass(cls):
        cls.target_class = faiss.IndexSVSVamanaLVQ

    def _create_instance(self):
        idx = self.target_class(self.d, 64)
        idx.storage_kind = faiss.SVS_LVQ4x0
        return idx

@unittest.skipIf(_SKIP_SVS_LL, _SKIP_SVS_LL_REASON)
class TestSVSAdapterLVQ4x4(TestSVSAdapter):
    """Repeat all tests for SVSLVQ4x4 variant"""

    @classmethod
    def setUpClass(cls):
        cls.target_class = faiss.IndexSVSVamanaLVQ

    def _create_instance(self):
        idx = self.target_class(self.d, 64)
        idx.storage_kind = faiss.SVS_LVQ4x4
        return idx

@unittest.skipIf(_SKIP_SVS_LL, _SKIP_SVS_LL_REASON)
class TestSVSAdapterLVQ4x8(TestSVSAdapter):
    """Repeat all tests for SVSLVQ4x8 variant"""

    @classmethod
    def setUpClass(cls):
        cls.target_class = faiss.IndexSVSVamanaLVQ

    def _create_instance(self):
        idx = self.target_class(self.d, 64)
        idx.storage_kind = faiss.SVS_LVQ4x8
        return idx

class TestSVSAdapterFlat(TestSVSAdapter):
    """Repeat all tests for SVSFlat variant"""

    @classmethod
    def setUpClass(cls):
        cls.target_class = faiss.IndexSVSFlat

    def _create_instance(self):
        return self.target_class(self.d)

    def test_svs_metric_types(self):
        """Test different metric types are handled correctly"""
        # L2 metric
        index_l2 = self._create_instance()
        index_l2.metric_type = faiss.METRIC_L2
        index_l2.add(self.xb)
        D_l2, _ = index_l2.search(self.xq[:10], 4)

        index_ip = self._create_instance()
        index_ip.metric_type = faiss.METRIC_INNER_PRODUCT
        index_ip.add(self.xb)
        D_ip, _ = index_ip.search(self.xq[:10], 4)

        # Results should be different (testing adapter forwards metric correctly)
        self.assertFalse(np.array_equal(D_l2, D_ip))

    # The fowlloing tests are expected to fail for IndexSVSFlat as it doesn't support yet
    @unittest.expectedFailure
    def test_svs_search_selected(self):
        return super().test_svs_search_selected()

    @unittest.expectedFailure
    def test_svs_range_search(self):
        return super().test_svs_range_search()

    @unittest.expectedFailure
    def test_svs_range_search_ip(self):
        return super().test_svs_range_search_ip()

    @unittest.expectedFailure
    def test_svs_range_search_selected(self):
        return super().test_svs_range_search_selected()

    @unittest.expectedFailure
    def test_svs_add_search_remove_interface(self):
        super().test_svs_add_search_remove_interface()

    @unittest.expectedFailure
    def test_svs_batch_operations(self):
        super().test_svs_batch_operations()


@unittest.skipIf(_SKIP_SVS, _SKIP_REASON)
class TestSVSVamanaParameters(unittest.TestCase):
    """Test Vamana-specific parameter forwarding and persistence for SVS Vamana variants"""

    @classmethod
    def setUpClass(cls):
        cls.target_class = faiss.IndexSVSVamana

    def _create_instance(self):
        """Create an instance of the SVS Vamana index"""
        return self.target_class(self.d ,64)

    def setUp(self):
        self.d = 32
        self.nb = 500  # Smaller dataset for parameter tests
        self.nq = 50
        np.random.seed(1234)
        self.xb = np.random.random((self.nb, self.d)).astype('float32')
        self.xq = np.random.random((self.nq, self.d)).astype('float32')

    def test_vamana_parameter_setting(self):
        """Test that all Vamana parameters can be set and retrieved"""
        index = self._create_instance()

        # Set non-default values for all parameters
        index.graph_max_degree = 32
        index.alpha = 1.5
        index.search_window_size = 20
        index.search_buffer_capacity = 25
        index.construction_window_size = 80
        index.max_candidate_pool_size = 150
        index.prune_to = 30
        index.use_full_search_history = False

        # Verify all parameters are set correctly
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
        index = self._create_instance()

        # Verify default values match C++ header
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
        index = self._create_instance()

        # Set distinctive non-default values
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
        loaded = faiss.deserialize_index(faiss.serialize_index(index))

        # Verify all parameters are preserved
        self.assertIsInstance(loaded, self.target_class)
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


@unittest.skipIf(_SKIP_SVS, _SKIP_REASON)
class TestSVSVamanaParametersFP16(TestSVSVamanaParameters):
    """Repeat Vamana parameter tests for SVS Float16 variant"""
    def _create_instance(self):
        idx = self.target_class(self.d, 64)
        idx.storage_kind = faiss.SVS_FP16
        return idx

@unittest.skipIf(_SKIP_SVS, _SKIP_REASON)
class TestSVSVamanaParametersSQI8(TestSVSVamanaParameters):
    """Repeat Vamana parameter tests for SVS SQ int8 variant"""
    def _create_instance(self):
        idx = self.target_class(self.d, 64)
        idx.storage_kind = faiss.SVS_SQI8
        return idx

@unittest.skipIf(_SKIP_SVS_LL, _SKIP_SVS_LL_REASON)
class TestSVSVamanaParametersLVQ4x0(TestSVSVamanaParameters):
    """Repeat Vamana parameter tests for SVSLVQ4x0 variant"""

    @classmethod
    def setUpClass(cls):
        cls.target_class = faiss.IndexSVSVamanaLVQ

    def _create_instance(self):
        idx = self.target_class(self.d, 64)
        idx.storage_kind = faiss.SVS_LVQ4x0
        return idx

@unittest.skipIf(_SKIP_SVS_LL, _SKIP_SVS_LL_REASON)
class TestSVSVamanaParametersLVQ4x4(TestSVSVamanaParameters):
    """Repeat Vamana parameter tests for SVSLVQ4x4 variant"""

    @classmethod
    def setUpClass(cls):
        cls.target_class = faiss.IndexSVSVamanaLVQ

    def _create_instance(self):
        idx = self.target_class(self.d, 64)
        idx.storage_kind = faiss.SVS_LVQ4x4
        return idx

@unittest.skipIf(_SKIP_SVS_LL, _SKIP_SVS_LL_REASON)
class TestSVSVamanaParametersLVQ4x8(TestSVSVamanaParameters):
    """Repeat Vamana parameter tests for SVSLVQ4x8 variant"""

    @classmethod
    def setUpClass(cls):
        cls.target_class = faiss.IndexSVSVamanaLVQ

    def _create_instance(self):
        idx = self.target_class(self.d, 64)
        idx.storage_kind = faiss.SVS_LVQ4x8
        return idx

if __name__ == '__main__':
    unittest.main()
