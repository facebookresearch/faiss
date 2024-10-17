# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import faiss
import unittest

from common_faiss_tests import get_dataset_2


class TestIndexRowwiseMinmax(unittest.TestCase):
    def compare_train_vs_train_inplace(self, factory_key):
        d = 96
        nb = 1000
        nq = 0
        nt = 2000

        xt, x, _ = get_dataset_2(d, nt, nb, nq)

        assert x.size > 0

        codec = faiss.index_factory(d, factory_key)

        # use the regular .train()
        codec.train(xt)
        codes_train = codec.sa_encode(x)

        decoded = codec.sa_decode(codes_train)

        # use .train_inplace()
        xt_cloned = np.copy(xt)
        codec.train_inplace(xt_cloned)
        codes_train_inplace = codec.sa_encode(x)

        # compare .train and .train_inplace codes
        n_diff = (codes_train != codes_train_inplace).sum()
        self.assertEqual(n_diff, 0)

        # make sure that the array used for .train_inplace got affected
        n_diff_xt = (xt_cloned != xt).sum()
        self.assertNotEqual(n_diff_xt, 0)

        # make sure that the reconstruction error is not crazy
        reconstruction_err = ((x - decoded) ** 2).sum()

        self.assertLess(reconstruction_err, 0.6)

    def test_fp32(self) -> None:
        self.compare_train_vs_train_inplace("MinMax,SQ8")

    def test_fp16(self) -> None:
        self.compare_train_vs_train_inplace("MinMaxFP16,SQ8")
