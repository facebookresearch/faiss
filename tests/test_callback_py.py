# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import numpy as np
import faiss


class TestCallbackPy(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_timeout(self) -> None:
        n = 1000
        k = 100
        d = 128
        niter = 1_000_000_000

        x = np.random.rand(n, d).astype('float32')
        index = faiss.IndexFlat(d)

        cp = faiss.ClusteringParameters()
        cp.niter = niter
        cp.verbose = False

        kmeans = faiss.Clustering(d, k, cp)

        with self.assertRaises(RuntimeError):
            with faiss.TimeoutGuard(0.010):
                kmeans.train(x, index)
