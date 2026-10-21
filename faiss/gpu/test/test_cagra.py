# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license in the LICENSE file.

"""CAGRA Dataset workflows and Python ownership lifetimes."""

import gc
import unittest
import weakref

import numpy as np
import faiss


@unittest.skipIf("CUVS" not in faiss.get_compile_options(), "requires cuVS")
class TestDataset(unittest.TestCase):
    def test_host_views_preserve_types_strides_and_borrowing(self):
        for dtype, numeric_type in ((np.float32, faiss.Float32),
                                    (np.float16, faiss.Float16),
                                    (np.int8, faiss.Int8),
                                    (np.uint8, faiss.UInt8)):
            with self.subTest(dtype=dtype):
                allocation = np.arange(27, dtype=dtype).reshape(3, 9)
                values = allocation[:, :5]
                values_ref = weakref.ref(values)
                source = faiss.Dataset.viewHost(values)
                self.assertEqual(int(source.getData()), values.ctypes.data)
                self.assertEqual(source.getNumRows(), 3)
                self.assertEqual(source.getDim(), 5)
                self.assertEqual(source.getRowStride(), 9)
                self.assertEqual(source.getNumericType(), numeric_type)
                self.assertFalse(source.ownsData())
                self.assertFalse(source.isDevice())
                packed = source.copyToHost()
                self.assertEqual(packed.dtype, values.dtype)
                self.assertTrue(packed.flags.c_contiguous)
                np.testing.assert_array_equal(packed, values)

                raw = faiss.Dataset.viewHost_c(
                    faiss.cast_integer_to_void_ptr(values.ctypes.data),
                    3, 5, numeric_type, 9)
                np.testing.assert_array_equal(raw.copyToHost(), values)
                del raw  # Raw pointer views require an explicitly retained allocation.
                source_ref = weakref.ref(source)
                descriptor = source.view()
                del source, values
                gc.collect()
                self.assertIsNotNone(source_ref())
                self.assertIsNotNone(values_ref())
                np.testing.assert_array_equal(descriptor.copyToHost(), packed)
                del descriptor
                gc.collect()
                self.assertIsNone(source_ref())
                self.assertIsNone(values_ref())

    def test_host_view_rejects_implicit_conversion_or_column_strides(self):
        for values in (np.zeros((3, 5), dtype="float64"),
                       np.zeros(15, dtype="float32"),
                       np.zeros((3, 10), dtype="float32")[:, ::2],
                       np.zeros((3, 5), dtype="float32")[::-1]):
            with self.subTest(shape=values.shape, strides=values.strides):
                with self.assertRaises((TypeError, ValueError)):
                    faiss.Dataset.viewHost(values)

    def test_padded_owners_and_device_views(self):
        res = faiss.StandardGpuResources()
        for dtype, numeric_type in ((np.float32, faiss.Float32),
                                    (np.float16, faiss.Float16),
                                    (np.int8, faiss.Int8),
                                    (np.uint8, faiss.UInt8)):
            with self.subTest(dtype=dtype):
                allocation = np.arange(27, dtype=dtype).reshape(3, 9)
                values = allocation[:, :5]
                source = faiss.Dataset.viewHost(values)
                owner = faiss.Dataset.makeDevicePadded(res, source, 0)
                self.assertTrue(owner.ownsData())
                self.assertTrue(owner.isDevice())
                self.assertTrue(owner.isCagraCompatible())
                self.assertEqual(owner.getDevice(), 0)
                packed = owner.copyToHost()
                self.assertEqual(packed.dtype, values.dtype)
                np.testing.assert_array_equal(packed, values)
                del source
                stride = owner.getRowStride()
                standard = faiss.Dataset.viewDevice(
                    owner.getData(), 3, 5, numeric_type, stride, 0)
                padded = faiss.Dataset.viewDevicePadded(
                    faiss.cast_integer_to_void_ptr(int(owner.getData())),
                    3, 5, numeric_type, stride, 0)
                for descriptor in (standard, padded):
                    self.assertFalse(descriptor.ownsData())
                    self.assertEqual(int(descriptor.getData()), int(owner.getData()))
                    np.testing.assert_array_equal(descriptor.copyToHost(), values)
                physical = faiss.Dataset.viewDevice(
                    owner.getData(), 3, stride, numeric_type, stride, 0)
                np.testing.assert_array_equal(physical.copyToHost()[:, 5:], 0)
                del standard, padded, descriptor, physical
                descriptor = owner.view()
                owner_ref = weakref.ref(owner)
                del owner
                gc.collect()
                self.assertIsNotNone(owner_ref())
                self.assertFalse(descriptor.ownsData())
                np.testing.assert_array_equal(descriptor.copyToHost(), values)
                del descriptor
                gc.collect()
                self.assertIsNone(owner_ref())


@unittest.skipIf("CUVS" not in faiss.get_compile_options(), "requires cuVS")
class TestCagraDataset(unittest.TestCase):
    @staticmethod
    def config():
        config = faiss.GpuIndexCagraConfig()
        config.graph_degree = 32
        config.intermediate_graph_degree = 64
        config.build_algo = faiss.graph_build_algo_NN_DESCENT
        return config

    def check_search(self, index, database, queries, numeric_type):
        k = 8
        params = faiss.SearchParametersCagra()
        params.itopk_size = 128
        distances, labels = index.search(
            queries, k, params=params, numeric_type=numeric_type)
        self.assertTrue(np.all((labels >= 0) & (labels < len(database))))
        float_database = database.astype("float32")
        float_queries = queries.astype("float32")
        _, reference = faiss.knn(float_queries, float_database, k)
        recall = np.mean([len(set(found) & set(expected)) / k
                          for found, expected in zip(labels, reference)])
        self.assertGreaterEqual(recall, 0.98)
        expected_distances = np.sum(
            (float_queries[:, None, :] - float_database[labels]) ** 2, axis=2)
        np.testing.assert_allclose(
            distances, expected_distances,
            rtol=2e-3 if numeric_type == faiss.Float16 else 6e-5, atol=1e-5)

    def check_workflow(self, dtype, numeric_type):
        random = np.random.RandomState(1732)
        rows, dim = 1024, 19
        if dtype == np.int8:
            values = random.randint(-50, 51, (rows, dim)).astype(dtype)
            queries = random.randint(-50, 51, (32, dim)).astype(dtype)
        else:
            values = random.normal(size=(rows, dim)).astype(dtype)
            queries = random.normal(size=(32, dim)).astype(dtype)
        # Host gaps may contain arbitrary values; keep the whole allocation alive.
        allocation = np.full((rows, dim + 4), 37, dtype=dtype)
        allocation[:, :dim] = values
        source = faiss.Dataset.viewHost(allocation[:, :dim])
        res = faiss.StandardGpuResources()
        storage = faiss.Dataset.makeDevicePadded(res, source, 0)
        index = faiss.GpuIndexCagra(res, dim, faiss.METRIC_L2, self.config())
        descriptor_ref = weakref.ref(source)
        index.train(source)
        del source
        gc.collect()
        self.assertIsNotNone(descriptor_ref())
        self.assertTrue(index.is_trained)
        self.assertEqual(index.ntotal, rows)
        self.assertFalse(index.isSearchReady())
        with self.assertRaisesRegex(RuntimeError, "Dataset"):
            index.search(queries, 8, numeric_type=numeric_type)
        graph = faiss.vector_to_array(index.get_knngraph())
        descriptor = storage.view()
        index.updateDataset(descriptor)
        self.assertIsNone(descriptor_ref())
        descriptor_ref = weakref.ref(descriptor)
        del descriptor
        self.assertIsNotNone(descriptor_ref())
        self.assertTrue(index.isSearchReady())
        np.testing.assert_array_equal(faiss.vector_to_array(index.get_knngraph()), graph)
        self.check_search(index, values, queries, numeric_type)
        np.testing.assert_array_equal(storage.copyToHost(), values)
        np.testing.assert_array_equal(allocation[:, dim:], 37)
        with self.assertRaises(RuntimeError):
            index.train(storage)
        self.assertIsNotNone(descriptor_ref())
        index.reset()
        self.assertIsNone(descriptor_ref())
        self.assertFalse(index.isSearchReady())
        self.assertFalse(index.is_trained)
        self.assertEqual(index.ntotal, 0)
        index.add(storage)
        self.check_search(index, values, queries, numeric_type)

        cpu = faiss.IndexHNSWCagra()
        cpu.base_level_only = True
        index.copyTo(cpu)
        loaded = faiss.deserialize_index(faiss.serialize_index(cpu))
        np.testing.assert_array_equal(loaded.reconstruct_n(0, rows).astype(dtype), values)
        cpu_storage = faiss.downcast_index(loaded.storage)
        cpu_codes = faiss.vector_to_array(cpu_storage.codes)
        with self.assertRaisesRegex(RuntimeError, "Dataset"):
            faiss.index_cpu_to_gpu(res, 0, loaded)

        # Allocate all physical columns with a sentinel, including future padding.
        stride = storage.getRowStride()
        sentinel = np.full((rows, stride), 37, dtype=dtype)
        sentinel_view = faiss.Dataset.viewHost(sentinel)
        output_owner = faiss.Dataset.makeDevicePadded(res, sentinel_view, 0)
        output = faiss.Dataset.viewDevicePadded(
            output_owner.getData(), rows, dim, numeric_type, stride, 0)
        self.assertFalse(output.ownsData())
        imported = faiss.GpuIndexCagra(res, dim, faiss.METRIC_L2, self.config())
        imported.copyFrom(loaded, output)
        output_ref = weakref.ref(output)
        del output
        self.assertIsNotNone(output_ref())
        self.assertTrue(imported.isSearchReady())
        filled = output_owner.copyToHost()
        self.assertEqual(filled.dtype, values.dtype)
        np.testing.assert_array_equal(filled[:, :dim], values)
        np.testing.assert_array_equal(filled[:, dim:], 0)
        np.testing.assert_array_equal(faiss.vector_to_array(cpu_storage.codes), cpu_codes)
        with self.assertRaises(RuntimeError):
            imported.copyFrom(loaded, faiss.Dataset.viewHost(values))
        self.assertIsNotNone(output_ref())
        self.check_search(imported, values, queries, numeric_type)
        imported.copyFrom(loaded)
        self.assertIsNone(output_ref())
        self.assertFalse(imported.isSearchReady())
        imported.updateDataset(storage)

        graph_only = faiss.GpuIndexCagra(res, dim, faiss.METRIC_L2, self.config())
        graph_only.copyFrom(loaded)
        del cpu_storage, loaded, cpu
        gc.collect()
        self.assertFalse(graph_only.isSearchReady())
        self.assertTrue(graph_only.is_trained)
        self.assertEqual(graph_only.ntotal, rows)
        np.testing.assert_array_equal(
            faiss.vector_to_array(graph_only.get_knngraph()),
            faiss.vector_to_array(imported.get_knngraph()))
        with self.assertRaisesRegex(RuntimeError, "Dataset"):
            graph_only.search(queries, 8, numeric_type=numeric_type)
        empty_cpu = faiss.IndexHNSWCagra()
        with self.assertRaisesRegex(RuntimeError, "Dataset"):
            graph_only.copyTo(empty_cpu)
        graph_only.updateDataset(storage)
        storage_ref = weakref.ref(storage)
        del storage
        gc.collect()
        self.assertIsNotNone(storage_ref())
        self.check_search(index, values, queries, numeric_type)
        self.check_search(imported, values, queries, numeric_type)
        del index
        imported.reset()
        self.assertIsNotNone(storage_ref())
        self.check_search(graph_only, values, queries, numeric_type)
        del graph_only
        gc.collect()
        self.assertIsNone(storage_ref())

    def test_float32_workflow(self):
        self.check_workflow(np.float32, faiss.Float32)

    def test_float16_workflow(self):
        self.check_workflow(np.float16, faiss.Float16)

    def test_int8_workflow(self):
        self.check_workflow(np.int8, faiss.Int8)

    def test_arrays_and_idmap_do_not_bypass_dataset_requirement(self):
        res = faiss.StandardGpuResources()
        cagra = faiss.GpuIndexCagra(res, 20, faiss.METRIC_L2, self.config())
        values = np.zeros((64, 20), dtype="float32")
        for method in (cagra.train, cagra.add):
            with self.assertRaisesRegex(TypeError, "Dataset"):
                method(values)
        index = faiss.IndexIDMap(cagra)
        with self.assertRaisesRegex(RuntimeError, "Dataset"):
            index.train(values)


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
