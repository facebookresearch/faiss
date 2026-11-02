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
import os
import tempfile

_SKIP_SVS = "SVS" not in faiss.get_compile_options().split()
_SKIP_REASON = "SVS support not compiled in"

_SKIP_SVS_LL = _SKIP_SVS or not faiss.IndexSVSVamana.is_lvq_leanvec_enabled()
_SKIP_SVS_LL_REASON = (
    "LVQ/LeanVec support not available on this platform or build configuration"
)


@unittest.skipIf(_SKIP_SVS, _SKIP_REASON)
class TestSVSAdapter(unittest.TestCase):
    target_class = None  # set in setUpClass

    def _create_instance(self) -> "faiss.IndexSVSVamana | faiss.IndexSVSFlat":
        return self.target_class(self.d, 64)

    @classmethod
    def setUpClass(cls):
        # need to configure target_class here to avoid issues when
        # SVS support is not compiled in
        cls.target_class = faiss.IndexSVSVamana
        cls.d = 32
        cls.nb = 500
        cls.nq = 50
        np.random.seed(1234)
        cls.xb = np.random.random((cls.nb, cls.d)).astype("float32")
        cls.xq = np.random.random((cls.nq, cls.d)).astype("float32")

    def test_svs_add_search_remove_interface(self):
        index = self._create_instance()

        index.add(self.xb)
        self.assertEqual(index.ntotal, self.nb)

        k = 4
        _, I = index.search(self.xq, k)
        self.assertTrue(np.all(I >= 0))
        self.assertTrue(np.all(I < self.nb))

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

        index.reset()
        self.assertEqual(index.ntotal, 0)

    def test_svs_search_selected(self):
        index = self._create_instance()

        index.add(self.xb)

        min = self.nb // 5
        max = self.nb * 4 // 5
        sel = faiss.IDSelectorRange(min, max)
        params = faiss.SearchParameters(sel=sel)

        k = 10
        _, I = index.search(self.xq, k, params=params)
        self.assertTrue(np.all(I >= min))
        self.assertTrue(np.all(I < max))

    def test_svs_range_search_ip(self):
        index = self._create_instance()
        index.metric_type = faiss.METRIC_INNER_PRODUCT
        index.alpha = 0.95

        index.add(self.xb)

        range = 10
        lims, D, I = index.range_search(self.xq, range)
        self.assertTrue(np.all(D >= range))
        self.assertTrue(np.all(I >= 0))
        self.assertTrue(np.all(I < self.nb))

    def test_svs_range_search_selected(self):
        index = self._create_instance()

        index.add(self.xb)

        min = self.nb // 5
        max = self.nb * 4 // 5
        sel = faiss.IDSelectorRange(min, max)
        params = faiss.SearchParameters(sel=sel)

        radius = 4.0
        lims, D, I = index.range_search(self.xq, radius, params=params)
        self.assertGreater(len(I), 0)
        self.assertTrue(np.all(D <= radius))
        self.assertTrue(np.all(I >= min))
        self.assertTrue(np.all(I < max))

    def test_svs_serialization(self):
        index = self._create_instance()

        index.add(self.xb)
        D_before, I_before = index.search(self.xq, 4)

        loaded = faiss.deserialize_index(faiss.serialize_index(index))
        self.assertEqual(type(loaded), self.target_class)
        self.assertEqual(loaded.d, self.d)
        self.assertEqual(loaded.ntotal, self.nb)
        self.assertEqual(loaded.metric_type, index.metric_type)

        D_after, I_after = loaded.search(self.xq, 4)
        np.testing.assert_array_equal(I_before, I_after)
        np.testing.assert_allclose(D_before, D_after, rtol=1e-6)

    def test_svs_error_handling(self):
        index = self._create_instance()

        wrong_dim_data = np.random.random((100, self.d + 1)).astype("float32")
        with self.assertRaises(AssertionError):
            index.add(wrong_dim_data)

    def test_svs_batch_operations(self):
        index = self._create_instance()

        batch_size = 250
        for i in range(0, self.nb, batch_size):
            end_idx = min(i + batch_size, self.nb)
            index.add(self.xb[i:end_idx])

        self.assertEqual(index.ntotal, self.nb)

        index.search(self.xq, 4)


@unittest.skipIf(_SKIP_SVS, _SKIP_REASON)
class TestSVSStoreVectors(unittest.TestCase):
    """Test the store_vectors opt-out for the fp32 reconstruct() copy"""

    @classmethod
    def setUpClass(cls):
        cls.d = 32
        cls.nb = 500
        cls.nq = 10
        np.random.seed(1234)
        cls.xb = np.random.random((cls.nb, cls.d)).astype("float32")
        cls.xq = np.random.random((cls.nq, cls.d)).astype("float32")

    def test_stores_vectors_by_default(self):
        index = faiss.IndexSVSVamana(self.d, 64)
        self.assertTrue(index.store_vectors)
        index.add(self.xb)
        self.assertEqual(index.stored_vectors.size(), self.nb * self.d)
        np.testing.assert_array_equal(index.reconstruct(0), self.xb[0])

    def test_store_vectors_disabled_in_constructor(self):
        index = faiss.IndexSVSVamana(
            self.d, 64, faiss.METRIC_L2, faiss.SVS_FP32, False, False
        )
        self.assertFalse(index.store_vectors)
        index.add(self.xb)
        self.assertEqual(index.ntotal, self.nb)
        self.assertEqual(index.stored_vectors.size(), 0)
        with self.assertRaises(RuntimeError):
            index.reconstruct(0)

    def test_store_vectors_disabled_by_attribute(self):
        """Search must be unaffected by dropping the fp32 copy"""
        reference = faiss.IndexSVSVamana(self.d, 64)
        reference.add(self.xb)
        _, I_ref = reference.search(self.xq, 4)

        index = faiss.IndexSVSVamana(self.d, 64)
        index.store_vectors = False
        index.add(self.xb)
        self.assertEqual(index.stored_vectors.size(), 0)

        _, I = index.search(self.xq, 4)
        np.testing.assert_array_equal(I_ref, I)

    def test_store_vectors_disabled_survives_serialization(self):
        index = faiss.IndexSVSVamana(self.d, 64)
        index.store_vectors = False
        index.add(self.xb)

        loaded = faiss.deserialize_index(faiss.serialize_index(index))
        self.assertEqual(loaded.ntotal, self.nb)
        self.assertEqual(loaded.stored_vectors.size(), 0)
        with self.assertRaises(RuntimeError):
            loaded.reconstruct(0)


@unittest.skipIf(_SKIP_SVS, _SKIP_REASON)
class TestSVSFactory(unittest.TestCase):
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

    def test_svs_factory_sq8(self):
        index = faiss.index_factory(64, "SVSVamana24,SQ8")
        self.assertEqual(index.d, 64)
        self.assertEqual(index.graph_max_degree, 24)
        self.assertEqual(index.storage_kind, faiss.SVS_SQ8)


@unittest.skipIf(_SKIP_SVS_LL, _SKIP_SVS_LL_REASON)
class TestSVSFactoryLVQLeanVec(unittest.TestCase):
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
class TestSVSIVFCoarseQuantizerFactory(unittest.TestCase):
    def test_ivf_svsvamana_storage(self):
        for storage in ("SQ8", "FP16", "FP32"):
            index = faiss.index_factory(
                32, f"IVF256_SVSVamana32_{storage},Flat"
            )
            self.assertEqual(index.d, 32)
            self.assertEqual(index.nlist, 256)

    def test_ivf_svsvamana_invalid_storage(self):
        with self.assertRaises(RuntimeError):
            faiss.index_factory(32, "IVF256_SVSVamana32_INVALID,Flat")


@unittest.skipIf(_SKIP_SVS, _SKIP_REASON)
class TestSVSAdapterFP16(TestSVSAdapter):
    def _create_instance(self):
        idx = self.target_class(self.d, 64)
        idx.storage_kind = faiss.SVS_FP16
        return idx


@unittest.skipIf(_SKIP_SVS, _SKIP_REASON)
class TestSVSAdapterSQ8(TestSVSAdapter):
    def _create_instance(self):
        idx = self.target_class(self.d, 64)
        idx.storage_kind = faiss.SVS_SQ8
        return idx


@unittest.skipIf(_SKIP_SVS_LL, _SKIP_SVS_LL_REASON)
class TestSVSAdapterLVQ4x0(TestSVSAdapter):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.target_class = faiss.IndexSVSVamanaLVQ

    def _create_instance(self):
        idx = self.target_class(self.d, 64)
        idx.storage_kind = faiss.SVS_LVQ4x0
        return idx


@unittest.skipIf(_SKIP_SVS_LL, _SKIP_SVS_LL_REASON)
class TestSVSAdapterLVQ4x4(TestSVSAdapter):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.target_class = faiss.IndexSVSVamanaLVQ

    def _create_instance(self):
        idx = self.target_class(self.d, 64)
        idx.storage_kind = faiss.SVS_LVQ4x4
        return idx


@unittest.skipIf(_SKIP_SVS_LL, _SKIP_SVS_LL_REASON)
class TestSVSAdapterLVQ4x8(TestSVSAdapter):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.target_class = faiss.IndexSVSVamanaLVQ

    def _create_instance(self):
        idx = self.target_class(self.d, 64)
        idx.storage_kind = faiss.SVS_LVQ4x8
        return idx


class TestSVSAdapterFlat(TestSVSAdapter):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.target_class = faiss.IndexSVSFlat

    def _create_instance(self):
        return self.target_class(self.d)

    def test_svs_metric_types(self):
        index_l2 = self._create_instance()
        index_l2.metric_type = faiss.METRIC_L2
        index_l2.add(self.xb)
        D_l2, _ = index_l2.search(self.xq[:10], 4)

        index_ip = self._create_instance()
        index_ip.metric_type = faiss.METRIC_INNER_PRODUCT
        index_ip.add(self.xb)
        D_ip, _ = index_ip.search(self.xq[:10], 4)

        # the distances differ only if the adapter forwards the metric
        self.assertFalse(np.array_equal(D_l2, D_ip))

    # IndexSVSFlat has no selector, range search or remove_ids support
    @unittest.expectedFailure
    def test_svs_search_selected(self):
        return super().test_svs_search_selected()

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
    @classmethod
    def setUpClass(cls):
        cls.target_class = faiss.IndexSVSVamana
        cls.d = 32
        cls.nb = 500
        cls.nq = 50
        np.random.seed(1234)
        cls.xb = np.random.random((cls.nb, cls.d)).astype("float32")
        cls.xq = np.random.random((cls.nq, cls.d)).astype("float32")

    def _create_instance(self):
        return self.target_class(self.d, 64)

    def test_vamana_parameter_defaults(self):
        index = self._create_instance()

        self.assertEqual(index.graph_max_degree, 64)
        self.assertAlmostEqual(index.alpha, 1.2, places=6)
        self.assertEqual(index.search_window_size, 10)
        self.assertEqual(index.search_buffer_capacity, 10)
        self.assertEqual(index.construction_window_size, 40)
        self.assertEqual(index.max_candidate_pool_size, 200)
        self.assertEqual(index.prune_to, 60)
        self.assertEqual(index.use_full_search_history, True)

    def test_vamana_parameter_serialization(self):
        index = self._create_instance()

        index.graph_max_degree = 48
        index.alpha = 1.8
        index.search_window_size = 15
        index.search_buffer_capacity = 18
        index.construction_window_size = 60
        index.max_candidate_pool_size = 180
        index.prune_to = 45
        index.use_full_search_history = False

        index.add(self.xb)

        loaded = faiss.deserialize_index(faiss.serialize_index(index))

        self.assertEqual(loaded.graph_max_degree, 48)
        self.assertAlmostEqual(loaded.alpha, 1.8, places=6)
        self.assertEqual(loaded.search_window_size, 15)
        self.assertEqual(loaded.search_buffer_capacity, 18)
        self.assertEqual(loaded.construction_window_size, 60)
        self.assertEqual(loaded.max_candidate_pool_size, 180)
        self.assertEqual(loaded.prune_to, 45)
        self.assertEqual(loaded.use_full_search_history, False)


@unittest.skipIf(_SKIP_SVS, _SKIP_REASON)
class TestSVSVamanaSetIndexParameter(unittest.TestCase):
    """Test the Vamana parameters exposed through ParameterSpace"""

    target_class = None  # set in setUpClass

    @classmethod
    def setUpClass(cls):
        # need to configure target_class here to avoid issues when
        # SVS support is not compiled in
        cls.target_class = faiss.IndexSVSVamana
        cls.d = 64
        cls.nb = 1000
        cls.nq = 50
        np.random.seed(1234)
        cls.xb = np.random.random((cls.nb, cls.d)).astype("float32")
        cls.xq = np.random.random((cls.nq, cls.d)).astype("float32")
        ref = faiss.IndexFlatL2(cls.d)
        ref.add(cls.xb)
        _, cls.gt = ref.search(cls.xq, 4)

    def setUp(self):
        self.ps = faiss.ParameterSpace()

    def _create_instance(self):
        return self.target_class(self.d, 64)

    def _storage_kind(self):
        """A storage kind the index under test can be switched to"""
        return faiss.SVS_FP16

    def _build_parameters(self):
        """(name, value) pairs covering every build-time parameter"""
        return [
            ("graph_max_degree", 32),
            ("prune_to", 28),
            ("alpha", 1.5),
            ("construction_window_size", 80),
            ("max_candidate_pool_size", 150),
            ("use_full_search_history", 0),
            ("storage_kind", self._storage_kind()),
            ("is_static", 1),
        ]

    def _recall(self, index):
        """Fraction of the exact 4 nearest neighbours the index returns"""
        _, I = index.search(self.xq, self.gt.shape[1])
        found = sum(len(set(a) & set(b)) for a, b in zip(I, self.gt))
        return found / self.gt.size

    def test_build_parameters(self):
        index = self._create_instance()
        for name, value in self._build_parameters():
            self.ps.set_index_parameter(index, name, value)

        self.assertEqual(index.graph_max_degree, 32)
        self.assertEqual(index.prune_to, 28)
        self.assertAlmostEqual(index.alpha, 1.5, places=6)
        self.assertEqual(index.construction_window_size, 80)
        self.assertEqual(index.max_candidate_pool_size, 150)
        self.assertFalse(index.use_full_search_history)
        self.assertEqual(index.storage_kind, self._storage_kind())
        self.assertTrue(index.is_static)

        # the values are accepted by the SVS backend
        index.add(self.xb)
        self.assertEqual(index.ntotal, self.nb)
        _, I = index.search(self.xq, 4)
        self.assertTrue((I >= 0).all())

    def test_build_parameters_rejected_after_build(self):
        index = self._create_instance()
        index.add(self.xb)
        for name, value in self._build_parameters():
            with self.assertRaises(RuntimeError):
                self.ps.set_index_parameter(index, name, value)

    def test_search_parameters(self):
        index = self._create_instance()
        index.add(self.xb)

        # search parameters are tunable after the index has been built
        self.ps.set_index_parameters(
            index, "search_window_size=4,search_buffer_capacity=4"
        )
        narrow_recall = self._recall(index)

        self.ps.set_index_parameter(index, "search_window_size", 128)
        self.ps.set_index_parameter(index, "search_buffer_capacity", 160)
        self.assertEqual(index.search_window_size, 128)
        self.assertEqual(index.search_buffer_capacity, 160)

        # the values reach the SVS backend: a wider search window explores
        # more of the graph and finds more of the true neighbours
        self.assertGreater(self._recall(index), narrow_recall)

    def test_store_vectors(self):
        index = self._create_instance()
        index.add(self.xb)

        self.ps.set_index_parameter(index, "store_vectors", 0)
        self.assertFalse(index.store_vectors)
        index.add(self.xb)  # drops the copy of the vectors

        # re-enabling would leave the copy misaligned with the ids
        with self.assertRaises(RuntimeError):
            self.ps.set_index_parameter(index, "store_vectors", 1)

    def test_leanvec_d(self):
        """leanvec_d applies to IndexSVSVamanaLeanVec only"""
        index = self.target_class(self.d, 64)
        with self.assertRaises(RuntimeError):
            self.ps.set_index_parameter(index, "leanvec_d", 32)

    def test_invalid_values_rejected(self):
        index = self._create_instance()
        with self.assertRaises(RuntimeError):
            self.ps.set_index_parameter(index, "graph_max_degree", -1)
        with self.assertRaises(RuntimeError):
            self.ps.set_index_parameter(index, "graph_max_degree", 1.5)
        with self.assertRaises(RuntimeError):
            # larger than 2**64: must not overflow the size_t cast
            self.ps.set_index_parameter(index, "graph_max_degree", 1e30)
        with self.assertRaises(RuntimeError):
            self.ps.set_index_parameter(index, "storage_kind", faiss.SVS_count)
        # a parameter of another index type is still an error
        with self.assertRaises(RuntimeError):
            self.ps.set_index_parameter(index, "efSearch", 32)

    def test_coarse_quantizer_parameters(self):
        index = faiss.index_factory(self.d, "IVF16_SVSVamana32,Flat")
        self.ps.set_index_parameter(index, "quantizer_search_window_size", 24)
        self.ps.set_index_parameter(index, "quantizer_graph_max_degree", 48)
        quantizer = faiss.downcast_index(index.quantizer)
        self.assertEqual(quantizer.search_window_size, 24)
        self.assertEqual(quantizer.graph_max_degree, 48)


@unittest.skipIf(_SKIP_SVS_LL, _SKIP_SVS_LL_REASON)
class TestSVSVamanaSetIndexParameterLeanVec(TestSVSVamanaSetIndexParameter):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.target_class = faiss.IndexSVSVamanaLeanVec

    def _storage_kind(self):
        return faiss.SVS_LeanVec4x8

    def _create_instance(self):
        # LeanVec has to be trained before vectors can be added
        index = self.target_class(self.d, 64)
        index.train(self.xb)
        return index

    def test_leanvec_d(self):
        """leanvec_d is accepted, but only before train()"""
        index = self.target_class(self.d, 64)
        self.ps.set_index_parameter(index, "leanvec_d", 32)
        self.assertEqual(index.leanvec_d, 32)

        index.train(self.xb)
        # train() has consumed leanvec_d, so it is too late to change it
        with self.assertRaises(RuntimeError):
            self.ps.set_index_parameter(index, "leanvec_d", 16)

        index.add(self.xb)
        self.assertEqual(index.ntotal, self.nb)


###############################################################################
# SVS IVF Tests
###############################################################################


@unittest.skipIf(_SKIP_SVS, _SKIP_REASON)
class TestSVSIVFAdapter(unittest.TestCase):
    target_class = None

    @classmethod
    def setUpClass(cls):
        cls.target_class = faiss.IndexSVSIVF

    def setUp(self):
        self.d = 32
        self.nb = 1000
        self.nq = 100
        self.nlist = 4
        np.random.seed(1234)
        self.xb = np.random.random((self.nb, self.d)).astype("float32")
        self.xq = np.random.random((self.nq, self.d)).astype("float32")

    def _create_instance(self):
        idx = self.target_class(self.d, self.nlist)
        idx.num_threads = 4
        return idx

    def test_ivf_construction(self):
        index = self._create_instance()
        self.assertEqual(index.d, self.d)
        self.assertFalse(index.is_trained)
        self.assertEqual(index.ntotal, 0)
        self.assertEqual(index.metric_type, faiss.METRIC_L2)

    def test_ivf_throws_add_without_train(self):
        index = self._create_instance()
        with self.assertRaises(RuntimeError):
            index.add(self.xb)

    def test_ivf_serialization(self):
        index = self._create_instance()
        index.train(self.xb)

        extra = np.random.random((200, self.d)).astype("float32")
        index.add(extra)

        D_before, I_before = index.search(self.xq, 4)

        loaded = faiss.deserialize_index(faiss.serialize_index(index))
        self.assertIsInstance(loaded, self.target_class)
        self.assertEqual(loaded.d, self.d)
        self.assertEqual(loaded.num_centroids, self.nlist)

        D_after, I_after = loaded.search(self.xq, 4)
        # IVF search may produce different ID orderings for results
        # at similar distances, so only check distances are close.
        np.testing.assert_allclose(D_before, D_after, rtol=1e-4)

    def test_ivf_remove_ids(self):
        index = self._create_instance()
        index.train(self.xb)

        extra = np.random.random((200, self.d)).astype("float32")
        index.add(extra)
        before = index.ntotal

        toremove = np.arange(0, 50, dtype=np.int64)
        sel = faiss.IDSelectorArray(len(toremove), faiss.swig_ptr(toremove))
        nremoved = index.remove_ids(sel)
        self.assertEqual(nremoved, len(toremove))
        self.assertEqual(index.ntotal, before - nremoved)

    def test_ivf_reset(self):
        index = self._create_instance()
        index.train(self.xb)

        extra = np.random.random((100, self.d)).astype("float32")
        index.add(extra)

        index.reset()
        self.assertEqual(index.ntotal, 0)
        self.assertFalse(index.is_trained)


@unittest.skipIf(_SKIP_SVS, _SKIP_REASON)
class TestSVSIVFAdapterFP16(TestSVSIVFAdapter):
    def _create_instance(self):
        idx = self.target_class(
            self.d, self.nlist, faiss.METRIC_L2, faiss.SVS_FP16
        )
        idx.num_threads = 4
        return idx


@unittest.skipIf(_SKIP_SVS, _SKIP_REASON)
class TestSVSIVFAdapterSQ8(TestSVSIVFAdapter):
    def _create_instance(self):
        idx = self.target_class(
            self.d, self.nlist, faiss.METRIC_L2, faiss.SVS_SQ8
        )
        idx.num_threads = 4
        return idx


@unittest.skipIf(_SKIP_SVS_LL, _SKIP_SVS_LL_REASON)
class TestSVSIVFAdapterLVQ4x4(TestSVSIVFAdapter):
    @classmethod
    def setUpClass(cls):
        cls.target_class = faiss.IndexSVSIVFLVQ

    def _create_instance(self):
        idx = self.target_class(self.d, self.nlist)
        idx.storage_kind = faiss.SVS_LVQ4x4
        idx.num_threads = 4
        return idx


@unittest.skipIf(_SKIP_SVS_LL, _SKIP_SVS_LL_REASON)
class TestSVSIVFAdapterLVQ4x8(TestSVSIVFAdapter):
    @classmethod
    def setUpClass(cls):
        cls.target_class = faiss.IndexSVSIVFLVQ

    def _create_instance(self):
        idx = self.target_class(self.d, self.nlist)
        idx.storage_kind = faiss.SVS_LVQ4x8
        idx.num_threads = 4
        return idx


@unittest.skipIf(_SKIP_SVS_LL, _SKIP_SVS_LL_REASON)
class TestSVSIVFAdapterLVQ8x0(TestSVSIVFAdapter):
    @classmethod
    def setUpClass(cls):
        cls.target_class = faiss.IndexSVSIVFLVQ

    def _create_instance(self):
        idx = self.target_class(self.d, self.nlist)
        idx.storage_kind = faiss.SVS_LVQ8x0
        idx.num_threads = 4
        return idx


@unittest.skipIf(_SKIP_SVS_LL, _SKIP_SVS_LL_REASON)
class TestSVSIVFAdapterLeanVec4x4(TestSVSIVFAdapter):
    @classmethod
    def setUpClass(cls):
        cls.target_class = faiss.IndexSVSIVFLeanVec

    def _create_instance(self):
        idx = self.target_class(
            self.d, self.nlist, faiss.METRIC_L2, 0, faiss.SVS_LeanVec4x4
        )
        idx.num_threads = 4
        return idx


@unittest.skipIf(_SKIP_SVS, _SKIP_REASON)
class TestSVSIVFFactory(unittest.TestCase):
    def test_svs_factory_ivf(self):
        index = faiss.index_factory(32, "SVSIVF4")
        self.assertEqual(index.d, 32)
        self.assertIsInstance(index, faiss.IndexSVSIVF)
        self.assertEqual(index.num_centroids, 4)
        self.assertEqual(index.storage_kind, faiss.SVS_FP32)

    def test_svs_factory_ivf_fp16(self):
        index = faiss.index_factory(32, "SVSIVF4,FP16")
        self.assertEqual(index.d, 32)
        self.assertIsInstance(index, faiss.IndexSVSIVF)
        self.assertEqual(index.storage_kind, faiss.SVS_FP16)

    def test_svs_factory_ivf_sq8(self):
        index = faiss.index_factory(64, "SVSIVF8,SQ8")
        self.assertEqual(index.d, 64)
        self.assertIsInstance(index, faiss.IndexSVSIVF)
        self.assertEqual(index.storage_kind, faiss.SVS_SQ8)


@unittest.skipIf(_SKIP_SVS_LL, _SKIP_SVS_LL_REASON)
class TestSVSIVFFactoryLVQLeanVec(unittest.TestCase):
    def test_svs_factory_ivf_lvq(self):
        index = faiss.index_factory(16, "SVSIVF4,LVQ4x8")
        self.assertEqual(index.d, 16)
        self.assertIsInstance(index, faiss.IndexSVSIVFLVQ)
        self.assertEqual(index.storage_kind, faiss.SVS_LVQ4x8)

    def test_svs_factory_ivf_leanvec(self):
        index = faiss.index_factory(128, "SVSIVF8,LeanVec4x4_64")
        self.assertEqual(index.d, 128)
        self.assertIsInstance(index, faiss.IndexSVSIVFLeanVec)
        self.assertEqual(index.storage_kind, faiss.SVS_LeanVec4x4)
        self.assertEqual(index.leanvec_d, 64)


@unittest.skipIf(_SKIP_SVS, _SKIP_REASON)
class TestSVSIVFParameters(unittest.TestCase):
    def setUp(self):
        self.d = 32
        self.nb = 5000
        self.nq = 50
        self.nlist = 100
        np.random.seed(1234)
        self.xb = np.random.random((self.nb, self.d)).astype("float32")
        self.xq = np.random.random((self.nq, self.d)).astype("float32")

    def test_ivf_parameter_defaults(self):
        index = faiss.IndexSVSIVF(self.d, self.nlist)

        self.assertEqual(index.num_centroids, self.nlist)
        self.assertEqual(index.n_probes, 10)
        self.assertAlmostEqual(index.k_reorder, 1.0, places=6)
        self.assertEqual(index.num_iterations, 10)
        self.assertEqual(index.minibatch_size, 10000)
        self.assertEqual(index.seed, 42)

    def test_ivf_parameter_serialization(self):
        index = faiss.IndexSVSIVF(self.d, self.nlist)

        index.n_probes = 5
        index.k_reorder = 3.0
        index.num_iterations = 15
        index.minibatch_size = 128
        index.seed = 77
        index.num_threads = 4

        index.train(self.xb)
        extra = np.random.random((100, self.d)).astype("float32")
        index.add(extra)

        loaded = faiss.deserialize_index(faiss.serialize_index(index))

        self.assertIsInstance(loaded, faiss.IndexSVSIVF)
        self.assertEqual(loaded.num_centroids, self.nlist)
        self.assertEqual(loaded.n_probes, 5)
        self.assertAlmostEqual(loaded.k_reorder, 3.0, places=6)
        self.assertEqual(loaded.num_iterations, 15)
        self.assertEqual(loaded.minibatch_size, 128)
        self.assertEqual(loaded.seed, 77)

        D_before, I_before = index.search(self.xq, 4)
        D_after, I_after = loaded.search(self.xq, 4)
        # IVF search may produce different ID orderings for results
        # at similar distances, so only check distances are close.
        np.testing.assert_allclose(D_before, D_after, rtol=1e-4)


@unittest.skipIf(_SKIP_SVS_LL, _SKIP_SVS_LL_REASON)
class TestSVSIVFLeanVecOOD(unittest.TestCase):
    """Test out-of-distribution training for IVF LeanVec SVS indices"""

    def setUp(self):
        self.d = 64
        self.n = 5000
        self.nlist = 100
        self.leanvec_d = 32
        np.random.seed(1234)
        self.x = np.random.rand(self.n, self.d).astype("float32")
        self.tq = np.random.rand(500, self.d).astype("float32")

    def _create_instance(self):
        idx = faiss.IndexSVSIVFLeanVec(
            self.d,
            self.nlist,
            faiss.METRIC_L2,
            self.leanvec_d,
            faiss.SVS_LeanVec4x8,
        )
        idx.num_threads = 4
        return idx

    def test_ivf_leanvec_ood_training_wrong_dim(self):
        idx = self._create_instance()
        wrong_dim = np.random.rand(500, self.d + 1).astype("float32")
        with self.assertRaises(AssertionError):
            idx.train(self.x, xq_train=wrong_dim)

    def test_ivf_leanvec_ood_training_wrong_type(self):
        idx = self._create_instance()
        with self.assertRaises(TypeError):
            idx.train(self.x, xq_train=self.tq, numeric_type=faiss.Float16)

    def test_ivf_leanvec_ood_serialization(self):
        idx = self._create_instance()
        idx.train(self.x, xq_train=self.tq)

        xq = np.random.rand(50, self.d).astype("float32")
        D_before, I_before = idx.search(xq, 4)

        loaded = faiss.deserialize_index(faiss.serialize_index(idx))
        self.assertIsInstance(loaded, faiss.IndexSVSIVFLeanVec)

        D_after, I_after = loaded.search(xq, 4)
        np.testing.assert_allclose(D_before, D_after, rtol=1e-4)


@unittest.skipIf(_SKIP_SVS_LL, _SKIP_SVS_LL_REASON)
class TestSVSLeanVecResetRetrain(unittest.TestCase):
    """Test that reset() properly clears training_data so retrain works."""

    def _make_data(self, d, n):
        return np.random.rand(n, d).astype("float32")

    def test_ivf_leanvec_ood_reset_retrain(self):
        d, n = 64, 5000
        X = self._make_data(d, n)
        Xq = self._make_data(d, 100)
        idx = faiss.index_factory(d, "SVSIVF100,LeanVec4x8_32")
        idx.num_threads = 4

        idx.train(X, xq_train=Xq)
        self.assertTrue(idx.is_trained)

        idx.reset()
        self.assertFalse(idx.is_trained)

        idx.train(X, xq_train=Xq)
        self.assertTrue(idx.is_trained)
        self.assertEqual(idx.ntotal, n)

    def test_vamana_leanvec_ood_reset_retrain(self):
        d = 64
        X = self._make_data(d, 1000)
        Xq = self._make_data(d, 100)
        idx = faiss.IndexSVSVamanaLeanVec(
            d, 64, faiss.METRIC_INNER_PRODUCT, 32, faiss.SVS_LeanVec4x8
        )

        idx.train(X, xq_train=Xq)
        self.assertTrue(idx.is_trained)

        idx.reset()
        self.assertFalse(idx.is_trained)

        idx.train(X, xq_train=Xq)
        self.assertTrue(idx.is_trained)


@unittest.skipIf(_SKIP_SVS, _SKIP_REASON)
class TestSVSStaticIVF(unittest.TestCase):
    def setUp(self):
        self.d = 32
        self.nb = 1000
        self.nq = 100
        self.nlist = 4
        np.random.seed(1234)
        self.xb = np.random.random((self.nb, self.d)).astype("float32")
        self.xq = np.random.random((self.nq, self.d)).astype("float32")

    def _create_static(self):
        idx = faiss.IndexSVSIVF(
            self.d, self.nlist, faiss.METRIC_L2, faiss.SVS_FP32, True
        )
        idx.num_threads = 4
        return idx

    def test_static_ivf_add_throws(self):
        index = self._create_static()
        index.train(self.xb)
        extra = np.random.random((100, self.d)).astype("float32")
        with self.assertRaises(RuntimeError):
            index.add(extra)

    def test_static_ivf_remove_throws(self):
        index = self._create_static()
        index.train(self.xb)
        ids = np.arange(0, 10, dtype=np.int64)
        sel = faiss.IDSelectorArray(len(ids), faiss.swig_ptr(ids))
        with self.assertRaises(RuntimeError):
            index.remove_ids(sel)

    def test_static_ivf_serialization(self):
        index = self._create_static()
        index.train(self.xb)

        D_before, I_before = index.search(self.xq, 4)

        loaded = faiss.deserialize_index(faiss.serialize_index(index))
        self.assertIsInstance(loaded, faiss.IndexSVSIVF)
        self.assertTrue(loaded.is_static)
        self.assertEqual(loaded.ntotal, self.nb)

        D_after, I_after = loaded.search(self.xq, 4)
        np.testing.assert_allclose(D_before, D_after, rtol=1e-4)

    def test_static_ivf_reset(self):
        index = self._create_static()
        index.train(self.xb)
        self.assertTrue(index.is_trained)

        index.reset()
        self.assertEqual(index.ntotal, 0)
        self.assertFalse(index.is_trained)

        index.train(self.xb)
        self.assertTrue(index.is_trained)
        self.assertEqual(index.ntotal, self.nb)


@unittest.skipIf(_SKIP_SVS_LL, _SKIP_SVS_LL_REASON)
class TestSVSStaticIVFLVQ(unittest.TestCase):
    def setUp(self):
        self.d = 32
        self.nb = 1000
        self.nq = 100
        self.nlist = 4
        np.random.seed(1234)
        self.xb = np.random.random((self.nb, self.d)).astype("float32")
        self.xq = np.random.random((self.nq, self.d)).astype("float32")

    def _create_static_lvq(self):
        idx = faiss.IndexSVSIVFLVQ(
            self.d, self.nlist, faiss.METRIC_L2, faiss.SVS_LVQ4x4, True
        )
        idx.num_threads = 4
        return idx

    def test_static_ivf_lvq_serialization(self):
        index = self._create_static_lvq()
        index.train(self.xb)

        D_before, _ = index.search(self.xq, 4)

        loaded = faiss.deserialize_index(faiss.serialize_index(index))
        self.assertIsInstance(loaded, faiss.IndexSVSIVFLVQ)
        self.assertTrue(loaded.is_static)

        D_after, _ = loaded.search(self.xq, 4)
        np.testing.assert_allclose(D_before, D_after, rtol=1e-4)


@unittest.skipIf(_SKIP_SVS_LL, _SKIP_SVS_LL_REASON)
class TestSVSStaticIVFLeanVec(unittest.TestCase):
    def setUp(self):
        self.d = 64
        self.nb = 5000
        self.nq = 100
        self.nlist = 10
        np.random.seed(1234)
        self.xb = np.random.random((self.nb, self.d)).astype("float32")
        self.xq = np.random.random((self.nq, self.d)).astype("float32")

    def _create_static_leanvec(self):
        idx = faiss.IndexSVSIVFLeanVec(
            self.d, self.nlist, faiss.METRIC_L2, 0, faiss.SVS_LeanVec4x4, True
        )
        idx.num_threads = 4
        return idx

    def test_static_ivf_leanvec_serialization(self):
        index = self._create_static_leanvec()
        index.train(self.xb)

        D_before, _ = index.search(self.xq, 4)

        loaded = faiss.deserialize_index(faiss.serialize_index(index))
        self.assertIsInstance(loaded, faiss.IndexSVSIVFLeanVec)
        self.assertTrue(loaded.is_static)

        D_after, _ = loaded.search(self.xq, 4)
        np.testing.assert_allclose(D_before, D_after, rtol=1e-4)


@unittest.skipIf(_SKIP_SVS, _SKIP_REASON)
class TestSVSStaticVamana(unittest.TestCase):
    def setUp(self):
        self.d = 32
        self.nb = 1000
        self.nq = 100
        self.degree = 32
        np.random.seed(1234)
        self.xb = np.random.random((self.nb, self.d)).astype("float32")
        self.xq = np.random.random((self.nq, self.d)).astype("float32")

    def _create_static(self):
        return faiss.IndexSVSVamana(
            self.d, self.degree, faiss.METRIC_L2, faiss.SVS_FP32, True
        )

    def test_static_second_add_throws(self):
        index = self._create_static()
        index.add(self.xb)
        extra = np.random.random((100, self.d)).astype("float32")
        with self.assertRaises(RuntimeError):
            index.add(extra)

    def test_static_remove_throws(self):
        index = self._create_static()
        index.add(self.xb)
        ids = np.arange(0, 10, dtype=np.int64)
        sel = faiss.IDSelectorArray(len(ids), faiss.swig_ptr(ids))
        with self.assertRaises(RuntimeError):
            index.remove_ids(sel)

    def test_static_serialization(self):
        index = self._create_static()
        index.add(self.xb)

        D_before, _ = index.search(self.xq, 4)

        loaded = faiss.deserialize_index(faiss.serialize_index(index))
        self.assertIsInstance(loaded, faiss.IndexSVSVamana)
        self.assertTrue(loaded.is_static)
        self.assertEqual(loaded.ntotal, self.nb)

        D_after, _ = loaded.search(self.xq, 4)
        np.testing.assert_allclose(D_before, D_after, rtol=1e-4)

    def test_static_memmapping(self):
        index = self._create_static()
        index.add(self.xb)

        D_before, _ = index.search(self.xq, 4)

        fd, fname = tempfile.mkstemp()
        os.close(fd)

        loaded = None
        try:
            faiss.write_index(index, fname)
            loaded = faiss.read_index(fname, faiss.IO_FLAG_MMAP_IFC)
            self.assertIsInstance(loaded, faiss.IndexSVSVamana)
            self.assertTrue(loaded.is_static)
            self.assertEqual(loaded.ntotal, self.nb)

            D_after, _ = loaded.search(self.xq, 4)
            np.testing.assert_allclose(D_before, D_after, rtol=1e-4)
        finally:
            del loaded
            if os.path.exists(fname):
                try:
                    os.unlink(fname)
                except OSError:
                    pass

    def test_static_reset_and_readd(self):
        index = self._create_static()
        index.add(self.xb)
        self.assertTrue(index.is_trained)

        index.reset()
        self.assertEqual(index.ntotal, 0)
        self.assertFalse(index.is_trained)

        index.add(self.xb)
        self.assertTrue(index.is_trained)
        self.assertEqual(index.ntotal, self.nb)

    def test_static_reconstruct(self):
        index = self._create_static()
        index.add(self.xb)
        recons = index.reconstruct(0)
        np.testing.assert_array_equal(recons, self.xb[0])


@unittest.skipIf(_SKIP_SVS_LL, _SKIP_SVS_LL_REASON)
class TestSVSStaticVamanaLVQ(unittest.TestCase):
    def setUp(self):
        self.d = 32
        self.nb = 1000
        self.nq = 100
        self.degree = 32
        np.random.seed(1234)
        self.xb = np.random.random((self.nb, self.d)).astype("float32")
        self.xq = np.random.random((self.nq, self.d)).astype("float32")

    def _create_static_lvq(self):
        return faiss.IndexSVSVamanaLVQ(
            self.d, self.degree, faiss.METRIC_L2, faiss.SVS_LVQ4x4, True
        )

    def test_static_lvq_serialization(self):
        index = self._create_static_lvq()
        index.add(self.xb)

        D_before, _ = index.search(self.xq, 4)

        loaded = faiss.deserialize_index(faiss.serialize_index(index))
        self.assertIsInstance(loaded, faiss.IndexSVSVamanaLVQ)
        self.assertTrue(loaded.is_static)

        D_after, _ = loaded.search(self.xq, 4)
        np.testing.assert_allclose(D_before, D_after, rtol=1e-4)

    def test_static_lvq_memmapping(self):
        index = self._create_static_lvq()
        index.add(self.xb)

        D_before, _ = index.search(self.xq, 4)

        fd, fname = tempfile.mkstemp()
        os.close(fd)

        loaded = None
        try:
            faiss.write_index(index, fname)
            loaded = faiss.read_index(fname, faiss.IO_FLAG_MMAP_IFC)
            self.assertIsInstance(loaded, faiss.IndexSVSVamana)
            self.assertTrue(loaded.is_static)

            D_after, _ = loaded.search(self.xq, 4)
            np.testing.assert_allclose(D_before, D_after, rtol=1e-4)
        finally:
            del loaded
            if os.path.exists(fname):
                try:
                    os.unlink(fname)
                except OSError:
                    pass


@unittest.skipIf(_SKIP_SVS_LL, _SKIP_SVS_LL_REASON)
class TestSVSStaticVamanaLeanVec(unittest.TestCase):
    def setUp(self):
        self.d = 64
        self.nb = 5000
        self.nq = 100
        self.degree = 32
        np.random.seed(1234)
        self.xb = np.random.random((self.nb, self.d)).astype("float32")
        self.xq = np.random.random((self.nq, self.d)).astype("float32")

    def _create_static_leanvec(self):
        return faiss.IndexSVSVamanaLeanVec(
            self.d, self.degree, faiss.METRIC_L2, 0, faiss.SVS_LeanVec4x4, True
        )

    def test_static_leanvec_ood(self):
        """train_with_queries followed by add() builds the static OOD index"""
        index = self._create_static_leanvec()
        index.train(self.xb, xq_train=self.xq)
        index.add(self.xb)
        self.assertTrue(index.is_static)
        self.assertTrue(index.is_trained)

        index.search(self.xq, 4)

    def test_static_leanvec_serialization(self):
        index = self._create_static_leanvec()
        index.train(self.xb)
        index.add(self.xb)

        D_before, _ = index.search(self.xq, 4)

        loaded = faiss.deserialize_index(faiss.serialize_index(index))
        self.assertIsInstance(loaded, faiss.IndexSVSVamanaLeanVec)
        self.assertTrue(loaded.is_static)

        D_after, _ = loaded.search(self.xq, 4)
        np.testing.assert_allclose(D_before, D_after, rtol=1e-4)

    def test_static_leanvec_memmapping(self):
        index = self._create_static_leanvec()
        index.train(self.xb)
        index.add(self.xb)

        D_before, _ = index.search(self.xq, 4)

        fd, fname = tempfile.mkstemp()
        os.close(fd)

        loaded = None
        try:
            faiss.write_index(index, fname)
            loaded = faiss.read_index(fname, faiss.IO_FLAG_MMAP_IFC)
            self.assertIsInstance(loaded, faiss.IndexSVSVamana)
            self.assertTrue(loaded.is_static)

            D_after, _ = loaded.search(self.xq, 4)
            np.testing.assert_allclose(D_before, D_after, rtol=1e-4)
        finally:
            del loaded
            if os.path.exists(fname):
                try:
                    os.unlink(fname)
                except OSError:
                    pass


if __name__ == "__main__":
    unittest.main()
