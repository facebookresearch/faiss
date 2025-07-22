# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import faiss

from faiss.contrib import datasets, evaluation
import numpy as np

@unittest.skipIf(
    "CUVS" not in faiss.get_compile_options(),
    "only if cuVS is compiled in")
class TestComputeGT(unittest.TestCase):

    def do_compute_GT(self, metric, numeric_type):
        d = 64
        k = 12
        ds = datasets.SyntheticDataset(d, 0, 10000, 100)
        
        # Get the data
        database = ds.get_database()
        queries = ds.get_queries()
        
        # Normalize for inner product to avoid duplicate neighbors
        if metric == faiss.METRIC_INNER_PRODUCT:
            # Normalize database vectors
            database = database / np.linalg.norm(database, axis=1, keepdims=True)
            # Normalize query vectors
            queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        
        Dref, Iref = faiss.knn(queries, database, k, metric)

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
<<<<<<< HEAD
        index.train(database)
        Dnew, Inew = index.search(queries, k)
=======
        database = ds.get_database().astype(np.float16) if numeric_type == faiss.Float16  else ds.get_database()
        index.train(database, numeric_type=numeric_type)
        queries = ds.get_queries().astype(np.float16) if numeric_type == faiss.Float16 else ds.get_queries()
        Dnew, Inew = index.search(queries, k, numeric_type=numeric_type)
>>>>>>> 2954f1f51809a10a8762a117098eae48fac2b56e
        
        evaluation.check_ref_knn_with_draws(Dref, Iref, Dnew, Inew, k)

    def test_compute_GT_L2(self):
        self.do_compute_GT(faiss.METRIC_L2, faiss.Float32)

    def test_compute_GT_IP(self):
<<<<<<< HEAD
        self.do_compute_GT(faiss.METRIC_INNER_PRODUCT) 
=======
        self.do_compute_GT(faiss.METRIC_INNER_PRODUCT, faiss.Float32)
>>>>>>> 2954f1f51809a10a8762a117098eae48fac2b56e

    def test_compute_GT_L2_FP16(self):
        self.do_compute_GT(faiss.METRIC_L2, faiss.Float16)

<<<<<<< HEAD
    def do_compute_GT(self, metric):
        d = 64
        k = 12
        ds = datasets.SyntheticDataset(d, 0, 10000, 100)
        Dref, Iref = faiss.knn(ds.get_queries(), ds.get_database(), k, metric)

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
        fp16_data = ds.get_database().astype(np.float16)
        index.train(fp16_data, faiss.Float16)
        fp16_queries = ds.get_queries().astype(np.float16)
        Dnew, Inew = index.search(fp16_queries, k, numeric_type=faiss.Float16)
        
        evaluation.check_ref_knn_with_draws(Dref, Iref, Dnew, Inew, k)

    def test_compute_GT_L2(self):
        self.do_compute_GT(faiss.METRIC_L2)

    def test_compute_GT_IP(self):
        self.do_compute_GT(faiss.METRIC_INNER_PRODUCT)
=======
    def test_compute_GT_IP_FP16(self):
        self.do_compute_GT(faiss.METRIC_INNER_PRODUCT, faiss.Float16)
>>>>>>> 2954f1f51809a10a8762a117098eae48fac2b56e

@unittest.skipIf(
    "CUVS" not in faiss.get_compile_options(),
    "only if cuVS is compiled in")
class TestInterop(unittest.TestCase):

    def do_interop(self, metric, numeric_type):
        d = 64
        k = 12
        ds = datasets.SyntheticDataset(d, 0, 10000, 100)

        res = faiss.StandardGpuResources()

        index = faiss.GpuIndexCagra(res, d, metric)
        database = ds.get_database().astype(np.float16) if numeric_type == faiss.Float16 else ds.get_database()
        index.train(database, numeric_type=numeric_type)
        queries = ds.get_queries().astype(np.float16) if numeric_type == faiss.Float16 else ds.get_queries()
        Dnew, Inew = index.search(queries, k, numeric_type=numeric_type)

        cpu_index = faiss.index_gpu_to_cpu(index)
        # cpu index always search in fp32
        Dref, Iref = cpu_index.search(ds.get_queries(), k)
        
        evaluation.check_ref_knn_with_draws(Dref, Iref, Dnew, Inew, k)

        deserialized_index = faiss.deserialize_index(
            faiss.serialize_index(cpu_index))

        gpu_index = faiss.index_cpu_to_gpu(res, 0, deserialized_index)
        Dnew2, Inew2 = gpu_index.search(queries, k, numeric_type=numeric_type)

        evaluation.check_ref_knn_with_draws(Dnew2, Inew2, Dnew, Inew, k)

    def test_interop_L2(self):
        self.do_interop(faiss.METRIC_L2, faiss.Float32)

    def test_interop_IP(self):
        self.do_interop(faiss.METRIC_INNER_PRODUCT, faiss.Float32)

    def test_interop_L2_FP16(self):
        self.do_interop(faiss.METRIC_L2, faiss.Float16)

    def test_interop_IP_FP16(self):
        self.do_interop(faiss.METRIC_INNER_PRODUCT, faiss.Float16)


@unittest.skipIf(
    "CUVS" not in faiss.get_compile_options(),
    "only if cuVS is compiled in")
class TestIDMapCagra(unittest.TestCase):

    def do_IDMapCagra(self, metric, numeric_type):
        d = 64
        k = 12
        ds = datasets.SyntheticDataset(d, 0, 10000, 100)
        Dref, Iref = faiss.knn(ds.get_queries(), ds.get_database(), k, metric)

        res = faiss.StandardGpuResources()

        index = faiss.GpuIndexCagra(res, d, metric)
        idMapIndex = faiss.IndexIDMap(index)
        database = ds.get_database().astype(np.float16) if numeric_type == faiss.Float16 else ds.get_database()
        idMapIndex.train(database, numeric_type=numeric_type)
        ids = np.array([i for i in range(10000)])
        idMapIndex.add_with_ids(database, ids, numeric_type=numeric_type)
        queries = ds.get_queries().astype(np.float16) if numeric_type == faiss.Float16 else ds.get_queries()
        Dnew, Inew = idMapIndex.search(queries, k, numeric_type=numeric_type)

        evaluation.check_ref_knn_with_draws(Dref, Iref, Dnew, Inew, k)

    def test_IDMapCagra_L2(self):
        self.do_IDMapCagra(faiss.METRIC_L2, faiss.Float32)

    def test_IDMapCagra_IP(self):
        self.do_IDMapCagra(faiss.METRIC_INNER_PRODUCT, faiss.Float32)

    def test_IDMapCagra_L2_FP16(self):
        self.do_IDMapCagra(faiss.METRIC_L2, faiss.Float16)

    def test_IDMapCagra_IP_FP16(self):
        self.do_IDMapCagra(faiss.METRIC_INNER_PRODUCT, faiss.Float16)
