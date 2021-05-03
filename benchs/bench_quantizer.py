# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import faiss
import time
import numpy as np
from faiss.contrib.datasets import DatasetSIFT1M


def eval_codec(q, xb):
    codes = q.compute_codes(xb)
    decoded = q.decode(codes)
    return ((xb - decoded) ** 2).sum() / xb.shape[0]


todo = sys.argv[1:]
ds = DatasetSIFT1M()

xq = ds.get_queries()
xb = ds.get_database()
gt = ds.get_groundtruth()
xt = ds.get_train()

nb, d = xb.shape
nq, d = xq.shape
nt, d = xt.shape

M = 4
nbits = 8

if 'lsq' in todo:
    lsq = faiss.LocalSearchQuantizer(d, M, nbits)
    lsq.log_level = 2  # show detailed training progress
    lsq.train(xt)
    err_lsq = eval_codec(lsq, xb)
    print('lsq:', err_lsq)

if 'pq' in todo:
    pq = faiss.ProductQuantizer(d, M, nbits)
    pq.train(xt)
    err_pq = eval_codec(pq, xb)
    print('pq:', err_pq)

if 'rq' in todo:
    rq = faiss.ResidualQuantizer(d, M, nbits)
    rq.train_type = faiss.ResidualQuantizer.Train_default
    rq.verbose = True
    rq.train(xt)
    err_rq = eval_codec(rq, xb)
    print('rq:', err_rq)
