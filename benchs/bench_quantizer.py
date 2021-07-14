# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import faiss
import time
import numpy as np

try:
    from faiss.contrib.datasets_fb import DatasetSIFT1M, DatasetDeep1B
except ImportError:
    from faiss.contrib.datasets import DatasetSIFT1M, DatasetDeep1B


def eval_codec(q, xq, xb, gt):
    t0 = time.time()
    codes = q.compute_codes(xb)
    t1 = time.time()
    xb_decoded = q.decode(codes)
    recons_err = ((xb - xb_decoded) ** 2).sum() / xb.shape[0]
    err_compat = np.linalg.norm(xb - xb_decoded, axis=1).mean()  # for compatibility with the codec benchmarks
    xq_decoded = q.decode(q.compute_codes(xq))
    D, I = faiss.knn(xq_decoded, xb_decoded, 1)
    recall = (I[:, 0] == gt[:, 0]).sum() / nq
    print(
        f"\tencode time: {t1 - t0:.3f} reconstruction error: {recons_err:.3f} "
        f"1-recall@1: {recall:.4f} recons_err_compat {err_compat:.3f}")


def eval_quantizer(q, xq, xb, gt, xt, variants=[(None, None)]):
    t0 = time.time()
    q.train(xt)
    t1 = time.time()
    train_t = t1 - t0
    print(f'\ttraining time: {train_t:.3f} s')
    for name, val in variants:
        if name is not None:
            print(f"{name}={val}")
            getattr(q, name)  # make sure field exists
            setattr(q, name, val)
        eval_codec(q, xq, xb, gt)


todo = sys.argv[1:]

if len(todo) > 0 and "deep1B" in todo[0]:
    ds = DatasetDeep1B(10**6)
    del todo[0]
else:
    ds = DatasetSIFT1M()

xq = ds.get_queries()
xb = ds.get_database()
gt = ds.get_groundtruth()
xt = ds.get_train(maxtrain=10**5)

nb, d = xb.shape
nq, d = xq.shape
nt, d = xt.shape

M = 8
nbits = 8

if len(todo) > 0:
    if "x" in todo[0]:
        if todo[0] == "8x8":
            M, nbits = 8, 8
        elif todo[0] == "6x10":
            M, nbits = 6, 10
        elif todo[0] == "5x12":
            M, nbits = 5, 12
        elif todo[0] == "10x6":
            M, nbits = 10, 6
        else:
            assert False
        del todo[0]

print(f"eval on {M}x{nbits}")

# fastest to slowest

if 'pq' in todo:
    pq = faiss.ProductQuantizer(d, M, nbits)
    print("===== PQ")
    eval_quantizer(pq, xq, xb, gt, xt)

if 'rq' in todo:
    print("===== RQ")
    rq = faiss.ResidualQuantizer(d, M, nbits, )
    # rq.train_type = faiss.ResidualQuantizer.Train_default
    # rq.verbose = True
    variants = [("max_beam_size", i) for i in (1, 2, 4, 8, 16, 32)]
    eval_quantizer(rq, xq, xb, gt, xt, variants=variants)

if 'lsq' in todo:
    print("===== LSQ")
    lsq = faiss.LocalSearchQuantizer(d, M, nbits)
    lsq.verbose = True
    variants = [("encode_ils_iters", i) for i in (2, 3, 4, 8, 16)]
    eval_quantizer(lsq, xq, xb, gt, xt, variants=variants)
