# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import unittest
import numpy as np
import faiss
from faiss.contrib.datasets import SyntheticDataset


@unittest.skipIf(
    "RAFT" not in faiss.get_compile_options(),
    "only if RAFT is compiled in")
class TestBfKnn(unittest.TestCase):

    def test_bfKnn(self):

        ds = SyntheticDataset(32, 0, 4321, 1234)

        Dref, Iref = faiss.knn(ds.get_queries(), ds.get_database(), 12)

        res = faiss.StandardGpuResources()

        # Faiss internal implementation
        Dnew, Inew = faiss.knn_gpu(
            res, ds.get_queries(), ds.get_database(), 12, use_raft=False)
        np.testing.assert_allclose(Dref, Dnew, atol=1e-5)
        np.testing.assert_array_equal(Iref, Inew)

        # RAFT version
        Dnew, Inew = faiss.knn_gpu(
            res, ds.get_queries(), ds.get_database(), 12, use_raft=True)
        np.testing.assert_allclose(Dref, Dnew, atol=1e-5)
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
        co.use_raft = True
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
