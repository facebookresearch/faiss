# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import faiss

from faiss.contrib import datasets, evaluation


@unittest.skipIf(
    "RAFT" not in faiss.get_compile_options(),
    "only if RAFT is compiled in")
class TestComputeGT(unittest.TestCase):

    def do_compute_GT(self, metric):
        d = 64
        k = 12
        ds = datasets.SyntheticDataset(d, 0, 10000, 100)
        Dref, Iref = faiss.knn(ds.get_queries(), ds.get_database(), k, metric)

        res = faiss.StandardGpuResources()

        index = faiss.GpuIndexCagra(res, d, metric)
        index.train(ds.get_database())
        Dnew, Inew = index.search(ds.get_queries(), k)
        
        evaluation.check_ref_knn_with_draws(Dref, Iref, Dnew, Inew, k)

    def test_compute_GT_L2(self):
        self.do_compute_GT(faiss.METRIC_L2)

    def test_compute_GT_IP(self):
        self.do_compute_GT(faiss.METRIC_INNER_PRODUCT)

@unittest.skipIf(
    "RAFT" not in faiss.get_compile_options(),
    "only if RAFT is compiled in")
class TestInterop(unittest.TestCase):

    def do_interop(self, metric):
        d = 64
        k = 12
        ds = datasets.SyntheticDataset(d, 0, 10000, 100)

        res = faiss.StandardGpuResources()

        index = faiss.GpuIndexCagra(res, d, metric)
        index.train(ds.get_database())
        Dnew, Inew = index.search(ds.get_queries(), k)

        cpu_index = faiss.index_gpu_to_cpu(index)
        Dref, Iref = cpu_index.search(ds.get_queries(), k)
        
        evaluation.check_ref_knn_with_draws(Dref, Iref, Dnew, Inew, k)

        deserialized_index = faiss.deserialize_index(
            faiss.serialize_index(cpu_index))

        gpu_index = faiss.index_cpu_to_gpu(res, 0, deserialized_index)
        Dnew2, Inew2 = gpu_index.search(ds.get_queries(), k)

        evaluation.check_ref_knn_with_draws(Dnew2, Inew2, Dnew, Inew, k)

    def test_interop_L2(self):
        self.do_interop(faiss.METRIC_L2)

    def test_interop_IP(self):
        self.do_interop(faiss.METRIC_INNER_PRODUCT)
