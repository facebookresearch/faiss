# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import unittest
import numpy as np
import faiss
from faiss.contrib.datasets import SyntheticDataset


@unittest.skipIf(
    "CUVS" not in faiss.get_compile_options(),
    "only if CUVS is compiled in")
class TestBfKnn(unittest.TestCase):

    def test_large_k_search(self):
        k = 10_000
        ds = SyntheticDataset(32, 100_000, 100_000, 1000)
        res = faiss.StandardGpuResources()
        config = faiss.GpuIndexFlatConfig()
        config.use_cuvs = True
        index_gpu = faiss.GpuIndexFlatL2(res, ds.d, config)

        index_gpu.add(ds.get_database())

        # Try larger than 2048
        _, I = index_gpu.search(ds.get_queries(), k)
        np.testing.assert_equal(I.shape, (ds.nq, k))


    def test_bfKnn(self):

        ds = SyntheticDataset(32, 0, 4321, 1234)

        Dref, Iref = faiss.knn(ds.get_queries(), ds.get_database(), 12)

        res = faiss.StandardGpuResources()

        # Faiss internal implementation
        Dnew, Inew = faiss.knn_gpu(
            res, ds.get_queries(), ds.get_database(), 12, use_cuvs=False)
        np.testing.assert_allclose(Dref, Dnew, atol=1e-4)
        np.testing.assert_array_equal(Iref, Inew)

        # cuVS version
        Dnew, Inew = faiss.knn_gpu(
            res, ds.get_queries(), ds.get_database(), 12, use_cuvs=True)
        np.testing.assert_allclose(Dref, Dnew, atol=1e-4)
        np.testing.assert_array_equal(Iref, Inew)

    def test_IndexFlat(self):
        ds = SyntheticDataset(32, 0, 4000, 1234)

        # add only first half of database
        xb = ds.get_database()
        index = faiss.IndexFlatL2(ds.d)
        index.add(xb[:2000])
        Dref, Iref = index.search(ds.get_queries(), 13)

        res = faiss.StandardGpuResources()
        co = faiss.GpuClonerOptions()
        co.use_cuvs = True
        index_gpu = faiss.index_cpu_to_gpu(res, 0, index, co)
        Dnew, Inew = index_gpu.search(ds.get_queries(), 13)
        np.testing.assert_allclose(Dref, Dnew, atol=1e-5)
        np.testing.assert_array_equal(Iref, Inew)

        # add rest of database
        index.add(xb[2000:])
        Dref, Iref = index.search(ds.get_queries(), 13)

        index_gpu.add(xb[2000:])
        Dnew, Inew = index_gpu.search(ds.get_queries(), 13)
        np.testing.assert_allclose(Dref, Dnew, atol=1e-4)
        np.testing.assert_array_equal(Iref, Inew)

        # copy back to CPU
        index2 = faiss.index_gpu_to_cpu(index_gpu)
        Dnew, Inew = index2.search(ds.get_queries(), 13)
        np.testing.assert_allclose(Dref, Dnew, atol=1e-4)
        np.testing.assert_array_equal(Iref, Inew)
