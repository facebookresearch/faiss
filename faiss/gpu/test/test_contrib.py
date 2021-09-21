# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import faiss
import unittest
import numpy as np

from faiss.contrib import datasets
from faiss.contrib.exhaustive_search import knn_ground_truth, range_ground_truth
from faiss.contrib import evaluation


from common_faiss_tests import get_dataset_2


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
