# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import unittest

import faiss

# noqa E741
# translation of test_knn.lua

from common_faiss_tests import Randu10k

ev = Randu10k()

d = ev.d



# Parameters for LSH
nbits = d

# Parameters for indexes involving PQ
M = int(d / 8)  # for PQ: #subquantizers
nbits_per_index = 8  # for PQ


class IndexAccuracyLowPrecision(unittest.TestCase):
    def test_IndexFlatIP(self):
        q = faiss.IndexFlatIP(d)  # Ask inner product
        res = ev.launch("FLAT / IP", q)
        e = ev.evalres(res)
        assert e[1] > 0.99

    

    # Approximate search module: PQ with inner product distance
    def test_IndexPQ_ip(self):
        q = faiss.IndexPQ(d, M, nbits_per_index, faiss.METRIC_INNER_PRODUCT)
        res = ev.launch("FLAT / PQ IP", q)
        e = ev.evalres(res)
        # should give 0.070  0.230  0.260
        # (same result as regular PQ on normalized distances)
        assert e[10] > 0.2    