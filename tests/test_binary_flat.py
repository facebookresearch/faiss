# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Binary flat tests. Ported from test_binary_flat.cpp."""

import unittest

import numpy as np

import faiss

faiss.omp_set_num_threads(4)


class TestBinaryFlat(unittest.TestCase):
    def test_accuracy(self):
        d = 64
        nb = 1000
        nq = 200
        k = 5

        rs = np.random.RandomState(123)
        database = rs.randint(0, 256, size=(nb, d // 8)).astype(np.uint8)
        queries = rs.randint(0, 256, size=(nq, d // 8)).astype(np.uint8)

        index = faiss.IndexBinaryFlat(d)
        index.add(database)

        dis, _nns = index.search(queries, k)

        for i in range(nq):
            dist_min = (
                np.unpackbits(
                    np.bitwise_xor(database, queries[i]), axis=1
                )
                .sum(axis=1)
                .min()
            )
            self.assertEqual(int(dist_min), int(dis[i, 0]))
