# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import numpy as np
import faiss

from common import make_binary_dataset


class TestBinaryIVF(unittest.TestCase):

    def test_nprobe(self):
        """Test in case of nprobe > nlist."""
        d = 32
        nq = 100
        nt = 100
        nb = 2000
        xt, xb, xq = make_binary_dataset(d, nt, nb, nq)

        # nlist = 10
        index = faiss.index_binary_factory(d, "BIVF10")

        # When nprobe >= nlist, it is equivalent to an IndexFlat.

        index.train(xt)
        index.add(xb)
        index.nprobe = 2048
        k = 5

        # test kNN search
        ref_D, ref_I = index.search(xq, k)

        gt_index = faiss.index_binary_factory(d, "BFlat")
        gt_index.add(xb)
        D, I = gt_index.search(xq, k)

        print(D[0], ref_D[0])
        print(I[0], ref_I[0])
        assert np.all(D == ref_D)
        # assert np.all(I == ref_I)  # id may be different

        # test range search
        thresh = 5   # *squared* distance
        ref_lims, ref_D, ref_I = index.range_search(xq, thresh)
        lims, D, I = gt_index.range_search(xq, thresh)
        assert np.all(lims == ref_lims)
        assert np.all(D == ref_D)
        # assert np.all(I == ref_I)  # id may be different
