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

"""Binary CAGRA Dataset construction, ownership, and CPU interchange."""
import gc
import unittest
import weakref

import numpy as np
import faiss
from faiss.contrib import evaluation


@unittest.skipIf("CUVS" not in faiss.get_compile_options(), "requires cuVS")
class TestBinaryCagraDataset(unittest.TestCase):
    def setUp(self):
        self.res = faiss.StandardGpuResources()
        # Match the C++ Dataset fixture's seed, size, and byte generation.
        self.xb = np.random.RandomState(4321).randint(
            0, 2**32, (512, 19), dtype=np.uint32
        ).astype("uint8")
        self.host = faiss.Dataset.viewHost(self.xb)

    def make_index(self, build_algo=None):
        config = faiss.GpuIndexCagraConfig()
        config.graph_degree = 16
        config.intermediate_graph_degree = 32
        config.build_algo = (
            faiss.graph_build_algo_NN_DESCENT
            if build_algo is None else build_algo
        )
        return faiss.GpuIndexBinaryCagra(self.res, self.xb.shape[1] * 8, config)

    def check_search(self, index):
        distances, labels = index.search(self.xb[:8], 1)
        self.assertTrue(np.all((labels >= 0) & (labels < len(self.xb))))
        self_hits = labels[:, 0] == np.arange(8)
        # Preserve the C++ Dataset regression's accepted self-hit threshold.
        self.assertGreaterEqual(np.count_nonzero(self_hits), 6)
        np.testing.assert_array_equal(distances[self_hits], 0)
        # Check actual Hamming distances, including any non-self result.
        expected = np.unpackbits(
            self.xb[labels[:, 0]] ^ self.xb[:8], axis=1
        ).sum(axis=1)
        np.testing.assert_array_equal(distances[:, 0], expected)

    def test_device_build_and_search(self):
        for algorithm in (
            faiss.graph_build_algo_NN_DESCENT,
            faiss.graph_build_algo_ITERATIVE_SEARCH,
        ):
            with self.subTest(algorithm=algorithm):
                storage = faiss.Dataset.makeDevicePadded(self.res, self.host, 0)
                storage_ref = weakref.ref(storage)
                index = self.make_index(algorithm)
                index.train(storage)
                del storage
                self.assertIsNotNone(storage_ref())
                self.assertTrue(index.is_trained)
                self.assertEqual(index.ntotal, len(self.xb))
                self.assertTrue(index.isSearchReady())
                self.check_search(index)
                index.reset()
                self.assertIsNone(storage_ref())

    def test_host_graph_attachment_and_reset(self):
        index = self.make_index()
        source = self.host.view()
        source_ref = weakref.ref(source)
        index.train(source)
        del source
        self.assertIsNotNone(source_ref())
        self.assertTrue(index.is_trained)
        self.assertFalse(index.isSearchReady())
        with self.assertRaisesRegex(RuntimeError, "updateDataset"):
            index.search(self.xb[:8], 1)
        cpu = faiss.IndexBinaryHNSWCagra()
        cpu.base_level_only = True
        index.copyTo(cpu)
        np.testing.assert_array_equal(cpu.reconstruct_n(), self.xb)

        storage = faiss.Dataset.makeDevicePadded(self.res, self.host, 0)
        storage_ref = weakref.ref(storage)
        index.updateDataset(storage.view())
        del storage
        self.assertIsNone(source_ref())
        self.assertIsNotNone(storage_ref())
        self.assertTrue(index.isSearchReady())
        with self.assertRaises(RuntimeError):
            index.updateDataset(self.host)
        index.train(self.host)
        index.add(self.host)
        self.assertIsNotNone(storage_ref())
        self.check_search(index)
        index.reset()
        self.assertIsNone(storage_ref())
        self.assertFalse(index.is_trained)
        self.assertFalse(index.isSearchReady())
        self.assertEqual(index.ntotal, 0)
        storage = faiss.Dataset.makeDevicePadded(self.res, self.host, 0)
        storage_ref = weakref.ref(storage)
        index.add(storage)
        del storage
        self.assertIsNotNone(storage_ref())
        self.check_search(index)
        index.reset()
        self.assertIsNone(storage_ref())

    def test_idmap_dataset(self):
        ids = np.arange(len(self.xb), dtype="int64") * 7 + 100
        for map_class in (faiss.IndexBinaryIDMap, faiss.IndexBinaryIDMap2):
            with self.subTest(map_class=map_class.__name__):
                storage = faiss.Dataset.makeDevicePadded(self.res, self.host, 0)
                owner_ref = weakref.ref(storage)
                cagra = self.make_index()
                index = map_class(cagra)
                index.add_with_ids(storage, ids)
                index.train(self.host)  # Preserve the device Dataset owner.
                if isinstance(index, faiss.IndexBinaryIDMap2):
                    index.check_consistency()
                expected_d, positions = cagra.search(self.xb[:8], 4)
                actual_d, labels = index.search(self.xb[:8], 4)
                np.testing.assert_array_equal(labels, ids[positions])
                np.testing.assert_array_equal(actual_d, expected_d)
                del storage
                self.assertIsNotNone(owner_ref())
                index.reset()
                self.assertIsNone(owner_ref())
                self.assertFalse(index.is_trained)
                self.assertFalse(cagra.is_trained)
                self.assertEqual(index.ntotal, 0)

    def test_cpu_serialization_and_optional_destination_import(self):
        index = self.make_index()
        index.train(self.host)
        cpu = faiss.IndexBinaryHNSWCagra()
        cpu.base_level_only = True
        index.copyTo(cpu)
        self.addCleanup(faiss.omp_set_num_threads, faiss.omp_get_max_threads())
        faiss.omp_set_num_threads(2)
        expected_d, expected_i = cpu.search(self.xb[:8], 4)
        self.assertTrue(np.all(expected_i >= 0))
        loaded = faiss.deserialize_index_binary(faiss.serialize_index_binary(cpu))
        actual_d, actual_i = loaded.search(self.xb[:8], 4)
        evaluation.check_ref_knn_with_draws(
            expected_d, expected_i, actual_d, actual_i
        )
        graph_only = faiss.index_binary_cpu_to_gpu(self.res, 0, loaded)
        self.assertTrue(graph_only.thisown)
        self.assertTrue(graph_only.is_trained)
        self.assertEqual(graph_only.ntotal, len(self.xb))
        self.assertFalse(graph_only.isSearchReady())

        # The owning allocation exposes all 32 physical bytes to verify that
        # copyFrom fills logical columns and zeroes every padding byte.
        sentinel = np.full((512, 32), 0xa5, dtype="uint8")
        sentinel_host = faiss.Dataset.viewHost(sentinel)
        allocation = faiss.Dataset.makeDevicePadded(self.res, sentinel_host, 0)
        destination = faiss.Dataset.viewDevicePadded(
            allocation.getData(), 512, 19, faiss.UInt8, 32, 0
        )
        imported = self.make_index()
        address = int(allocation.getData())
        imported.copyFrom(loaded, destination)
        self.assertEqual(int(allocation.getData()), address)
        self.assertTrue(allocation.ownsData())
        self.assertFalse(destination.ownsData())
        self.assertTrue(imported.isSearchReady())
        np.testing.assert_array_equal(loaded.reconstruct_n(), self.xb)
        expected_physical = np.zeros_like(sentinel)
        expected_physical[:, :19] = self.xb
        np.testing.assert_array_equal(allocation.copyToHost(), expected_physical)

        owner = faiss.Dataset.makeDevicePadded(self.res, self.host, 0)
        owner_ref = weakref.ref(owner)
        graph_only.copyFrom(loaded, owner)
        del owner
        self.assertIsNotNone(owner_ref())
        with self.assertRaises(RuntimeError):
            graph_only.copyFrom(loaded, self.host)
        self.assertIsNotNone(owner_ref())
        self.check_search(graph_only)
        graph_only.copyFrom(loaded)
        self.assertIsNone(owner_ref())
        self.assertFalse(graph_only.isSearchReady())

        source_ref = weakref.ref(loaded)
        descriptor_ref = weakref.ref(destination)
        index.reset()
        del loaded, cpu, destination
        gc.collect()
        self.assertIsNone(source_ref())
        self.assertIsNotNone(descriptor_ref())
        self.check_search(imported)  # Raw pointer storage remains explicitly retained.
        with self.assertRaisesRegex(RuntimeError, "updateDataset"):
            graph_only.search(self.xb[:8], 1)
        exported = faiss.IndexBinaryHNSWCagra()
        with self.assertRaisesRegex(RuntimeError, "Dataset"):
            graph_only.copyTo(exported)
        attached = faiss.Dataset.viewDevice(
            allocation.getData(), 512, 19, faiss.UInt8, 32, 0
        )
        graph_only.updateDataset(attached)
        self.check_search(graph_only)
        exported.base_level_only = True
        graph_only.copyTo(exported)
        np.testing.assert_array_equal(exported.reconstruct_n(), self.xb)
        graph_only.reset()
        imported.reset()
        self.assertIsNone(descriptor_ref())

    def test_pointer_and_base_build_require_dataset(self):
        index = self.make_index()
        for method in (index.train, index.add):
            with self.assertRaisesRegex(TypeError, "Dataset"):
                method(self.xb)
        for method in (faiss.IndexBinary.train, faiss.IndexBinary.add):
            with self.assertRaisesRegex(RuntimeError, "Dataset"):
                method(index, self.xb)
        self.assertEqual(index.ntotal, 0)

    def test_idmap_does_not_bypass_dataset_requirement(self):
        cagra = self.make_index()
        index = faiss.IndexBinaryIDMap(cagra)
        with self.assertRaisesRegex(RuntimeError, "Dataset"):
            index.add_with_ids(self.xb, np.arange(len(self.xb), dtype="int64"))
