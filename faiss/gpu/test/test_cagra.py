# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import faiss

from faiss.contrib import datasets, evaluation
import numpy as np


@unittest.skipIf(
    "CUVS" not in faiss.get_compile_options(), "only if cuVS is compiled in"
)
class TestComputeGT(unittest.TestCase):

    def do_compute_GT(self, metric, numeric_type):
        d = 64
        k = 12

        if numeric_type == faiss.Int8:
            data_base_nt = np.random.randint(-128, 128, size=(10000, d), dtype=np.int8)
            data_query_nt = np.random.randint(-128, 128, size=(100, d), dtype=np.int8)
            data_base = data_base_nt.astype(np.float32)
            data_query = data_query_nt.astype(np.float32)
        else:
            ds = datasets.SyntheticDataset(d, 0, 10000, 100)
            data_base = ds.get_database()  # fp32
            data_query = ds.get_queries()  # fp32
            # Normalize for inner product to avoid duplicate neighbors
            if metric == faiss.METRIC_INNER_PRODUCT:
                # Normalize database vectors
                data_base = data_base / np.linalg.norm(data_base, axis=1, keepdims=True)
                # Normalize query vectors
                data_query = data_query / np.linalg.norm(
                    data_query, axis=1, keepdims=True
                )
            if numeric_type == faiss.Float16:
                data_base_nt = data_base.astype(np.float16)
                data_query_nt = data_query.astype(np.float16)
            elif numeric_type == faiss.Float32:
                data_base_nt = data_base
                data_query_nt = data_query

        Dref, Iref = faiss.knn(data_query, data_base, k, metric)

        res = faiss.StandardGpuResources()

        # attempt to set custom IVF-PQ params
        cagraIndexConfig = faiss.GpuIndexCagraConfig()
        cagraIndexConfig.graph_degree = 32
        cagraIndexConfig.intermediate_graph_degree = 64
        cagraIndexIVFPQConfig = faiss.IVFPQBuildCagraConfig()
        cagraIndexIVFPQConfig.kmeans_trainset_fraction = 0.5
        cagraIndexConfig.ivf_pq_params = cagraIndexIVFPQConfig
        cagraIndexConfig.build_algo = faiss.graph_build_algo_IVF_PQ

        index = faiss.GpuIndexCagra(res, d, metric, cagraIndexConfig)
        index.train(data_base_nt, numeric_type=numeric_type)
        Dnew, Inew = index.search(data_query_nt, k, numeric_type=numeric_type)

        evaluation.check_ref_knn_with_draws(Dref, Iref, Dnew, Inew, k)

    def test_compute_GT_L2(self):
        self.do_compute_GT(faiss.METRIC_L2, faiss.Float32)

    def test_compute_GT_IP(self):
        self.do_compute_GT(faiss.METRIC_INNER_PRODUCT, faiss.Float32)

    def test_compute_GT_L2_FP16(self):
        self.do_compute_GT(faiss.METRIC_L2, faiss.Float16)

    @unittest.skip(
        "GPU CAGRA inner-product + FP16 search deadlocks on CUDA 12.9; "
        "root cause TBD."
    )
    def test_compute_GT_IP_FP16(self):
        self.do_compute_GT(faiss.METRIC_INNER_PRODUCT, faiss.Float16)

    def test_compute_GT_L2_Int8(self):
        self.do_compute_GT(faiss.METRIC_L2, faiss.Int8)

    def test_compute_GT_IP_Int8(self):
        self.do_compute_GT(faiss.METRIC_INNER_PRODUCT, faiss.Int8)


@unittest.skipIf(
    "CUVS" not in faiss.get_compile_options(), "only if cuVS is compiled in"
)
class TestInterop(unittest.TestCase):

    def do_interop(self, metric, numeric_type):
        d = 64
        k = 12
        if numeric_type == faiss.Int8:
            data_base_nt = np.random.randint(-128, 128, size=(10000, d), dtype=np.int8)
            data_query_nt = np.random.randint(-128, 128, size=(100, d), dtype=np.int8)
            data_base = data_base_nt.astype(np.float32)
            data_query = data_query_nt.astype(np.float32)
        else:
            ds = datasets.SyntheticDataset(d, 0, 10000, 100)
            data_base = ds.get_database()  # fp32
            data_query = ds.get_queries()  # fp32
            if numeric_type == faiss.Float16:
                data_base_nt = data_base.astype(np.float16)
                data_query_nt = data_query.astype(np.float16)
            elif numeric_type == faiss.Float32:
                data_base_nt = data_base
                data_query_nt = data_query

        res = faiss.StandardGpuResources()

        index = faiss.GpuIndexCagra(res, d, metric)
        index.train(data_base_nt, numeric_type=numeric_type)
        Dnew, Inew = index.search(data_query_nt, k, numeric_type=numeric_type)

        cpu_index = faiss.index_gpu_to_cpu(index)
        # cpu index always search in fp32
        Dref, Iref = cpu_index.search(data_query, k)

        evaluation.check_ref_knn_with_draws(Dref, Iref, Dnew, Inew, k)

        deserialized_index = faiss.deserialize_index(faiss.serialize_index(cpu_index))

        gpu_index = faiss.index_cpu_to_gpu(res, 0, deserialized_index)
        Dnew2, Inew2 = gpu_index.search(data_query_nt, k, numeric_type=numeric_type)

        evaluation.check_ref_knn_with_draws(Dnew2, Inew2, Dnew, Inew, k)

    def test_interop_L2(self):
        self.do_interop(faiss.METRIC_L2, faiss.Float32)

    def test_interop_IP(self):
        self.do_interop(faiss.METRIC_INNER_PRODUCT, faiss.Float32)

    def test_interop_L2_FP16(self):
        self.do_interop(faiss.METRIC_L2, faiss.Float16)

    def test_interop_IP_FP16(self):
        self.do_interop(faiss.METRIC_INNER_PRODUCT, faiss.Float16)

    def test_interop_L2_Int8(self):
        self.do_interop(faiss.METRIC_L2, faiss.Int8)

    def test_interop_IP_Int8(self):
        self.do_interop(faiss.METRIC_INNER_PRODUCT, faiss.Int8)

    def test_base_level_only_range_search(self):
        d = 32
        nb = 1000
        nq = 10
        ds = datasets.SyntheticDataset(d, 0, nb, nq)
        data_base = ds.get_database()
        data_query = ds.get_queries()

        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexCagra(res, d, faiss.METRIC_L2)
        index.train(data_base, numeric_type=faiss.Float32)

        cpu_index = faiss.index_gpu_to_cpu(index)
        cpu_index.base_level_only = True
        cpu_index.num_base_level_search_entrypoints = 8

        radius = np.float32(1e9)
        lims, _, _ = cpu_index.range_search(data_query, radius)
        counts = lims[1:] - lims[:-1]
        self.assertTrue(np.all(counts > 0))


@unittest.skipIf(
    "CUVS" not in faiss.get_compile_options(), "only if cuVS is compiled in"
)
class TestIDMapCagra(unittest.TestCase):

    def do_IDMapCagra(self, metric, numeric_type):
        d = 64
        k = 12
        if numeric_type == faiss.Int8:
            data_base_nt = np.random.randint(-128, 128, size=(10000, d), dtype=np.int8)
            data_query_nt = np.random.randint(-128, 128, size=(100, d), dtype=np.int8)
            data_base = data_base_nt.astype(np.float32)
            data_query = data_query_nt.astype(np.float32)
        else:
            ds = datasets.SyntheticDataset(d, 0, 10000, 100)
            data_base = ds.get_database()  # fp32
            data_query = ds.get_queries()  # fp32
            if numeric_type == faiss.Float16:
                data_base_nt = data_base.astype(np.float16)
                data_query_nt = data_query.astype(np.float16)
            elif numeric_type == faiss.Float32:
                data_base_nt = data_base
                data_query_nt = data_query

        Dref, Iref = faiss.knn(data_query, data_base, k, metric)

        res = faiss.StandardGpuResources()

        index = faiss.GpuIndexCagra(res, d, metric)
        idMapIndex = faiss.IndexIDMap(index)
        idMapIndex.train(data_base_nt, numeric_type=numeric_type)
        ids = np.array([i for i in range(10000)])
        idMapIndex.add_with_ids(data_base_nt, ids, numeric_type=numeric_type)
        Dnew, Inew = idMapIndex.search(data_query_nt, k, numeric_type=numeric_type)

        evaluation.check_ref_knn_with_draws(Dref, Iref, Dnew, Inew, k)

    def test_IDMapCagra_L2(self):
        self.do_IDMapCagra(faiss.METRIC_L2, faiss.Float32)

    def test_IDMapCagra_IP(self):
        self.do_IDMapCagra(faiss.METRIC_INNER_PRODUCT, faiss.Float32)

    def test_IDMapCagra_L2_FP16(self):
        self.do_IDMapCagra(faiss.METRIC_L2, faiss.Float16)

    def test_IDMapCagra_IP_FP16(self):
        self.do_IDMapCagra(faiss.METRIC_INNER_PRODUCT, faiss.Float16)

    def test_IDMapCagra_L2_Int8(self):
        self.do_IDMapCagra(faiss.METRIC_L2, faiss.Int8)

    def test_IDMapCagra_IP_Int8(self):
        self.do_IDMapCagra(faiss.METRIC_INNER_PRODUCT, faiss.Int8)


@unittest.skipIf(
    "CUVS" not in faiss.get_compile_options(),
    "only if cuVS is compiled in")
@unittest.skipIf(
    faiss.get_num_gpus() < 2,
    "need at least 2 GPUs for multi-GPU test")
class TestMultiGpuCagra(unittest.TestCase):

    def test_multi_gpu_build_and_search(self):
        ds = datasets.SyntheticDataset(128, 0, 100_000, 1000)
        xb = ds.get_database()
        xq = ds.get_queries()
        k = 10

        gt_index = faiss.IndexFlatL2(ds.d)
        gt_index.add(xb)
        Dref, Iref = gt_index.search(xq, k)

        num_gpus = min(faiss.get_num_gpus(), 4)
        resources = faiss.GpuResourcesVector()
        devices = faiss.Int32Vector()
        res_list = []
        for i in range(num_gpus):
            res = faiss.StandardGpuResources()
            res_list.append(res)
            resources.push_back(res)
            devices.push_back(i)

        config = faiss.GpuIndexCagraConfig()
        config.graph_degree = 32
        index = faiss.GpuIndexCagra(res_list[0], ds.d, faiss.METRIC_L2, config)
        index.trainMultiGpu(ds.nb, faiss.swig_ptr(xb), resources, devices, 0, 2)

        cpu_index = faiss.IndexHNSWCagra()
        index.copyTo(cpu_index)
        self.assertEqual(cpu_index.ntotal, ds.nb)

        cpu_index.hnsw.efSearch = 128
        Dnew, Inew = cpu_index.search(xq, k)

        recall = np.mean([
            len(set(Inew[i]) & set(Iref[i])) / k for i in range(ds.nq)
        ])
        self.assertGreater(recall, 0.80,
                           f"Multi-GPU recall@{k} too low: {recall:.4f}")

        # Serialization roundtrip
        data = faiss.serialize_index(cpu_index)
        loaded = faiss.deserialize_index(data)
        loaded.hnsw.efSearch = 128
        Dnew2, Inew2 = loaded.search(xq, k)
        np.testing.assert_array_equal(Inew, Inew2)

    def test_all_neighbors_build(self):
        ds = datasets.SyntheticDataset(32, 0, 50_000, 100)
        xb = ds.get_database()
        xq = ds.get_queries()
        k = 10

        gt_index = faiss.IndexFlatL2(ds.d)
        gt_index.add(xb)
        Dref, Iref = gt_index.search(xq, k)

        num_gpus = min(faiss.get_num_gpus(), 4)
        devices = faiss.Int32Vector()
        for i in range(num_gpus):
            devices.push_back(i)

        res = faiss.StandardGpuResources()
        config = faiss.GpuIndexCagraConfig()
        config.graph_degree = 32
        config.intermediate_graph_degree = 48
        index = faiss.GpuIndexCagra(
            res, ds.d, faiss.METRIC_L2, config)
        index.trainAllNeighbors(
            ds.nb, faiss.swig_ptr(xb), devices,
            0, 0, True, 0)

        cpu_index = faiss.IndexHNSWCagra()
        cpu_index.base_level_only = True
        index.copyTo(cpu_index)
        self.assertEqual(cpu_index.ntotal, ds.nb)

        cpu_index.hnsw.efSearch = 128
        Dnew, Inew = cpu_index.search(xq, k)

        recall = np.mean([
            len(set(Inew[i]) & set(Iref[i])) / k
            for i in range(ds.nq)
        ])
        self.assertGreater(
            recall, 0.70,
            f"all_neighbors recall@{k} too low: {recall:.4f}")
