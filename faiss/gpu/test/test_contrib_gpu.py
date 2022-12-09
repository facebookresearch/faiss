# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import faiss
import numpy as np

from common_faiss_tests import get_dataset_2

from faiss.contrib import datasets, evaluation, ivf_tools
from faiss.contrib.exhaustive_search import knn_ground_truth, \
    range_ground_truth


class TestComputeGT(unittest.TestCase):

    def test_compute_GT(self):
        d = 64
        xt, xb, xq = get_dataset_2(d, 0, 10000, 100)

        index = faiss.IndexFlatL2(d)
        index.add(xb)
        Dref, Iref = index.search(xq, 10)

        # iterator function on the matrix

        def matrix_iterator(xb, bs):
            for i0 in range(0, xb.shape[0], bs):
                yield xb[i0:i0 + bs]

        Dnew, Inew = knn_ground_truth(xq, matrix_iterator(xb, 1000), 10)

        np.testing.assert_array_equal(Iref, Inew)
        np.testing.assert_almost_equal(Dref, Dnew, decimal=4)

    def do_test_range(self, metric):
        ds = datasets.SyntheticDataset(32, 0, 1000, 10)
        xq = ds.get_queries()
        xb = ds.get_database()
        D, I = faiss.knn(xq, xb, 10, metric=metric)
        threshold = float(D[:, -1].mean())

        index = faiss.IndexFlat(32, metric)
        index.add(xb)
        ref_lims, ref_D, ref_I = index.range_search(xq, threshold)

        new_lims, new_D, new_I = range_ground_truth(
            xq, ds.database_iterator(bs=100), threshold,
            metric_type=metric)

        evaluation.test_ref_range_results(
            ref_lims, ref_D, ref_I,
            new_lims, new_D, new_I
        )

    def test_range_L2(self):
        self.do_test_range(faiss.METRIC_L2)

    def test_range_IP(self):
        self.do_test_range(faiss.METRIC_INNER_PRODUCT)


class TestBigBatchSearch(unittest.TestCase):

    def do_test(self, factory_string):
        ds = datasets.SyntheticDataset(32, 2000, 4000, 1000)
        k = 10
        index = faiss.index_factory(ds.d, factory_string)
        index.train(ds.get_train())
        index.add(ds.get_database())
        index.nprobe = 5
        Dref, Iref = index.search(ds.get_queries(), k)
        res = faiss.StandardGpuResources()

        def pairwise_distances(xq, xb, metric=faiss.METRIC_L2):
            return faiss.pairwise_distance_gpu(
                res, xq, xb, metric=faiss.METRIC_L2)

        def knn_function(xq, xb, k, metric=faiss.METRIC_L2):
            return faiss.knn_gpu(res, xq, xb, k, metric=faiss.METRIC_L2)

        for method in "pairwise_distances", "knn_function":
            Dnew, Inew = ivf_tools.big_batch_search(
                index, ds.get_queries(),
                k, method=method,
                pairwise_distances=pairwise_distances,
                knn=knn_function
            )
            self.assertLess((Inew != Iref).sum() / Iref.size, 1e-4)
            np.testing.assert_almost_equal(Dnew, Dref, decimal=4)

    def test_Flat(self):
        self.do_test("IVF64,Flat")

    def test_PQ(self):
        self.do_test("IVF64,PQ4np")


class TestBigBatchSearchMultiGPU(unittest.TestCase):

    @unittest.skipIf(faiss.get_num_gpus() < 2, "multiple GPU only test")
    def do_test(self, factory_string):
        ds = datasets.SyntheticDataset(32, 2000, 4000, 1000)
        k = 10
        index = faiss.index_factory(ds.d, factory_string)
        index.train(ds.get_train())
        index.add(ds.get_database())
        index.nprobe = 5
        Dref, Iref = index.search(ds.get_queries(), k)
        ngpu = faiss.get_num_gpus()
        res = [faiss.StandardGpuResources() for _ in range(ngpu)]

        def knn_function(xq, xb, k, metric=faiss.METRIC_L2, thread_id=None):
            return faiss.knn_gpu(
                res[thread_id], xq, xb, k,
                metric=faiss.METRIC_L2, device=thread_id
            )

        Dnew, Inew = ivf_tools.big_batch_search(
            index, ds.get_queries(),
            k, method="knn_function",
            knn=knn_function,
            threaded=8,
            computation_threads=ngpu
        )
        self.assertLess((Inew != Iref).sum() / Iref.size, 1e-4)
        np.testing.assert_almost_equal(Dnew, Dref, decimal=4)

    def test_Flat(self):
        self.do_test("IVF64,Flat")
