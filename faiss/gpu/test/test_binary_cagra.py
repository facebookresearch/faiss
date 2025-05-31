# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import faiss

from faiss.contrib import datasets, evaluation


@unittest.skipIf(
    "CUVS" not in faiss.get_compile_options(),
    "only if cuVS is compiled in")
class TestComputeGT(unittest.TestCase):

    def do_compute_GT(self):
        d = 64 * 8
        k = 12
        ds = datasets.SyntheticDataset(d, 0, 1000000, 10000)
        flat_index = faiss.IndexBinaryFlat(d)
        flat_index.add(ds.get_database())
        Dref, Iref = flat_index.search(ds.get_queries(), k)

        res = faiss.StandardGpuResources()

        index = faiss.GpuIndexBinaryCagra(res, d)
        index.train(ds.get_database())
        Dnew, Inew = index.search(ds.get_queries(), k)
        
        evaluation.check_ref_knn_with_draws(Dref, Iref, Dnew, Inew, k)
    
    def test_compute_GT(self):
        self.do_compute_GT()


# @unittest.skipIf(
#     "CUVS" not in faiss.get_compile_options(),
#     "only if cuVS is compiled in")
class TestInterop(unittest.TestCase):

    def do_interop(self):
        d = 64 * 8
        k = 12
        ds = datasets.SyntheticDataset(d, 0, 100000, 1000)

        res = faiss.StandardGpuResources()

        index = faiss.GpuIndexBinaryCagra(res, d)
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
    
    def test_interop(self):
        self.do_interop()
