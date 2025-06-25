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

    def do_compute_GT(self, metric, fp16):
        d = 64
        k = 12
        ds = datasets.SyntheticDataset(d, 0, 10000, 100)
        Dref, Iref = faiss.knn(ds.get_queries(), ds.get_database(), k, metric)

        res = faiss.StandardGpuResources()

        # attempt to set custom IVF-PQ params
        cagraIndexConfig = faiss.GpuIndexCagraConfig()
        cagraIndexIVFPQConfig = faiss.IVFPQBuildCagraConfig()
        cagraIndexIVFPQConfig.kmeans_trainset_fraction = 0.1
        cagraIndexConfig.ivf_pq_params = cagraIndexIVFPQConfig
        cagraIndexConfig.build_algo = faiss.graph_build_algo_IVF_PQ

        index = faiss.GpuIndexCagra(res, d, metric, cagraIndexConfig)
        database = ds.get_database().astype(np.float16) if fp16 else ds.get_database()
        index.train(database)
        queries = ds.get_queries().astype(np.float16) if fp16 else ds.get_queries()
        Dnew, Inew = index.search(queries, k)
        
        evaluation.check_ref_knn_with_draws(Dref, Iref, Dnew, Inew, k)

    def test_compute_GT_L2(self):
        self.do_compute_GT(faiss.METRIC_L2, False)

    def test_compute_GT_IP(self):
        self.do_compute_GT(faiss.METRIC_INNER_PRODUCT, False)

    def test_compute_GT_L2_FP16(self):
        self.do_compute_GT(faiss.METRIC_L2, True)

    def test_compute_GT_IP_FP16(self):
        self.do_compute_GT(faiss.METRIC_INNER_PRODUCT, True)

@unittest.skipIf(
    "CUVS" not in faiss.get_compile_options(),
    "only if cuVS is compiled in")
class TestInterop(unittest.TestCase):

    def do_interop(self, metric, fp16):
        d = 64
        k = 12
        ds = datasets.SyntheticDataset(d, 0, 10000, 100)

        res = faiss.StandardGpuResources()

        index = faiss.GpuIndexCagra(res, d, metric)
        database = ds.get_database().astype(np.float16) if fp16 else ds.get_database()
        index.train(database)
        queries = ds.get_queries().astype(np.float16) if fp16 else ds.get_queries()
        Dnew, Inew = index.search(queries, k)

        cpu_index = faiss.index_gpu_to_cpu(index)
        # cpu index always search in fp32
        Dref, Iref = cpu_index.search(ds.get_queries(), k)
        
        evaluation.check_ref_knn_with_draws(Dref, Iref, Dnew, Inew, k)

        deserialized_index = faiss.deserialize_index(
            faiss.serialize_index(cpu_index))

        gpu_index = faiss.index_cpu_to_gpu(res, 0, deserialized_index)
        Dnew2, Inew2 = gpu_index.search(queries, k)

        evaluation.check_ref_knn_with_draws(Dnew2, Inew2, Dnew, Inew, k)

    def test_interop_L2(self):
        self.do_interop(faiss.METRIC_L2, False)

    def test_interop_IP(self):
        self.do_interop(faiss.METRIC_INNER_PRODUCT, False)

    def test_interop_L2_FP16(self):
        self.do_interop(faiss.METRIC_L2, True)

    def test_interop_IP_FP16(self):
        self.do_interop(faiss.METRIC_INNER_PRODUCT, True)


@unittest.skipIf(
    "CUVS" not in faiss.get_compile_options(),
    "only if cuVS is compiled in")
class TestIDMapCagra(unittest.TestCase):

    def do_IDMapCagra(self, metric, fp16):
        d = 64
        k = 12
        ds = datasets.SyntheticDataset(d, 0, 10000, 100)
        Dref, Iref = faiss.knn(ds.get_queries(), ds.get_database(), k, metric)

        res = faiss.StandardGpuResources()

        index = faiss.GpuIndexCagra(res, d, metric)
        idMapIndex = faiss.IndexIDMap(index)
        database = ds.get_database().astype(np.float16) if fp16 else ds.get_database()
        idMapIndex.train(database)
        ids = [i for i in range(10000)]
        idMapIndex.add_with_ids(database, ids)
        queries = ds.get_queries().astype(np.float16) if fp16 else ds.get_queries()
        Dnew, Inew = idMapIndex.search(queries, k)

        evaluation.check_ref_knn_with_draws(Dref, Iref, Dnew, Inew, k)

    def test_IDMapCagra_L2(self):
        self.do_IDMapCagra(faiss.METRIC_L2, False)

    def test_IDMapCagra_IP(self):
        self.do_IDMapCagra(faiss.METRIC_INNER_PRODUCT, False)

    def test_IDMapCagra_L2_FP16(self):
        self.do_IDMapCagra(faiss.METRIC_L2, True)

    def test_IDMapCagra_IP_FP16(self):
        self.do_IDMapCagra(faiss.METRIC_INNER_PRODUCT, True)
