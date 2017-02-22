
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

#! /usr/bin/env python2

"""this is a basic test script that works with fbmake to check if
some simple indices work"""

import libfb.py.mkl  # noqa

import numpy as np
import pdb
from libfb import testutil

import faiss


class EvalIVFPQAccuracy(testutil.BaseFacebookTestCase):

    def get_dataset(self):
        d = 64
        nb = 1000
        nt = 1500
        nq = 200
        np.random.seed(123)
        xb = np.random.random(size=(nb, d)).astype('float32')
        xt = np.random.random(size=(nt, d)).astype('float32')
        xq = np.random.random(size=(nq, d)).astype('float32')

        return (xt, xb, xq)

    def test_IndexIVFPQ(self):
        (xt, xb, xq) = self.get_dataset()
        d = xt.shape[1]

        gt_index = faiss.IndexFlatL2(d)
        gt_index.add(xb)
        D, gt_nns = gt_index.search(xq, 1)

        coarse_quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQ(coarse_quantizer, d, 25, 16, 8)
        index.train(xt)
        index.add(xb)
        index.nprobe = 5
        D, nns = index.search(xq, 10)
        n_ok = (nns == gt_nns).sum()
        nq = xq.shape[0]

        self.assertGreater(n_ok, nq * 0.4)
