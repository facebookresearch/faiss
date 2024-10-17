# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import faiss
import numpy as np


class TestGpuIndexIvfflat(unittest.TestCase):
    def test_reconstruct_n(self):
        index = faiss.index_factory(4, "IVF10,Flat")
        x = np.random.RandomState(123).rand(10, 4).astype('float32')
        index.train(x)
        index.add(x)
        res = faiss.StandardGpuResources()
        res.noTempMemory()
        config = faiss.GpuIndexIVFFlatConfig()
        config.use_raft = False
        index2 = faiss.GpuIndexIVFFlat(res, index, config)
        recons = index2.reconstruct_n(0, 10)

        np.testing.assert_array_equal(recons, x)
