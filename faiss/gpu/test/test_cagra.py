# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import faiss
import numpy as np

from common_faiss_tests import get_dataset_2

from faiss.contrib import datasets, evaluation, big_batch_search
from faiss.contrib.exhaustive_search import knn_ground_truth, \
    range_ground_truth, range_search_gpu, \
    range_search_max_results, exponential_query_iterator


class TestComputeGT(unittest.TestCase):

    def do_compute_GT(self, metric):
        d = 64
        xt, xb, xq = get_dataset_2(d, 0, 10000, 100)

        index = faiss.GpuIndexCagra(d)
        index.train(xb)
        Dref, Iref = index.search(xq, 10)

        # iterator function on the matrix

        def matrix_iterator(xb, bs):
            for i0 in range(0, xb.shape[0], bs):
                yield xb[i0:i0 + bs]

        Dnew, Inew = knn_ground_truth(xq, matrix_iterator(xb, 1000), 10)

        np.testing.assert_array_equal(Iref, Inew)
        np.testing.assert_almost_equal(Dref, Dnew, decimal=4)

    def test_compute_GT_L2(self):
        self.do_compute_GT(faiss.METRIC_L2)

    def test_range_IP(self):
        self.do_compute_GT(faiss.METRIC_INNER_PRODUCT)
