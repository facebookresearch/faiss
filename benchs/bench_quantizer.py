# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import faiss
import time
import numpy as np

try:
    from faiss.contrib.datasets_fb import \
        DatasetSIFT1M, DatasetDeep1B, DatasetBigANN
except ImportError:
    from faiss.contrib.datasets import \
        DatasetSIFT1M, DatasetDeep1B, DatasetBigANN


def eval_codec(q, xq, xb, gt):
    t0 = time.time()
    codes = q.compute_codes(xb)
    t1 = time.time()
    xb_decoded = q.decode(codes)
    recons_err = ((xb - xb_decoded) ** 2).sum() / xb.shape[0]
    # for compatibility with the codec benchmarks
    err_compat = np.linalg.norm(xb - xb_decoded, axis=1).mean()
    xq_decoded = q.decode(q.compute_codes(xq))
    D, I = faiss.knn(xq_decoded, xb_decoded, 1)
    recall = (I[:, 0] == gt[:, 0]).sum() / nq
    print(
        f"\tencode time: {t1 - t0:.3f} reconstruction error: {recons_err:.3f} "
        f"1-recall@1: {recall:.4f} recons_err_compat {err_compat:.3f}")


def eval_quantizer(q, xq, xb, gt, xt, variants=None):
    if variants is None:
        variants = [(None, None)]
    t0 = time.time()
    q.train(xt)
    t1 = time.time()
    train_t = t1 - t0
    print(f'\ttraining time: {train_t:.3f} s')
    for name, val in variants:
        if name is not None:
            print(f"{name}={val}")

            if isinstance(q, faiss.ProductAdditiveQuantizer):
                for i in range(q.nsplits):
                    subq = faiss.downcast_Quantizer(q.subquantizer(i))
                    getattr(subq, name)
                    setattr(subq, name, val)
            else:
                getattr(q, name)  # make sure field exists
                setattr(q, name, val)

        eval_codec(q, xq, xb, gt)


todo = sys.argv[1:]

if len(todo) > 0 and "deep1M" in todo[0]:
    ds = DatasetDeep1B(10**6)
    del todo[0]
elif len(todo) > 0 and "bigann1M" in todo[0]:
    ds = DatasetBigANN(nb_M=1)
    del todo[0]
else:
    ds = DatasetSIFT1M()

if len(todo) > 0:
    if todo[0].count("x") == 1:
        M, nbits = [int(x) for x in todo[0].split("x")]
        del todo[0]
    elif todo[0].count("x") == 2:
        nsplits, Msub, nbits = [int(x) for x in todo[0].split("x")]
        M = nsplits * Msub
        del todo[0]

maxtrain = max(100 << nbits, 10**5)
print(f"eval on {M}x{nbits} maxtrain={maxtrain}")

xq = ds.get_queries()
xb = ds.get_database()
gt = ds.get_groundtruth()

xt = ds.get_train(maxtrain=maxtrain)

nb, d = xb.shape
nq, d = xq.shape
nt, d = xt.shape


# fastest to slowest

if 'lsq-gpu' in todo:
    lsq = faiss.LocalSearchQuantizer(d, M, nbits)
    ngpus = faiss.get_num_gpus()
    lsq.icm_encoder_factory = faiss.GpuIcmEncoderFactory(ngpus)
    lsq.verbose = True
    eval_quantizer(lsq, xb, xt, 'lsq-gpu')

if 'pq' in todo:
    pq = faiss.ProductQuantizer(d, M, nbits)
    print("===== PQ")
    eval_quantizer(pq, xq, xb, gt, xt)

if 'opq' in todo:
    d2 = ((d + M - 1) // M) * M
    print("OPQ d2=", d2)
    opq = faiss.OPQMatrix(d, M, d2)
    opq.train(xt)
    xq2 = opq.apply(xq)
    xb2 = opq.apply(xb)
    xt2 = opq.apply(xt)
    pq = faiss.ProductQuantizer(d2, M, nbits)
    print("===== PQ")
    eval_quantizer(pq, xq2, xb2, gt, xt2)

if 'prq' in todo:
    print(f"===== PRQ{nsplits}x{Msub}x{nbits}")
    prq = faiss.ProductResidualQuantizer(d, nsplits, Msub, nbits)
    variants = [("max_beam_size", i) for i in (1, 2, 4, 8, 16, 32)]
    eval_quantizer(prq, xq, xb, gt, xt, variants=variants)

if 'plsq' in todo:
    print(f"===== PLSQ{nsplits}x{Msub}x{nbits}")
    plsq = faiss.ProductLocalSearchQuantizer(d, nsplits, Msub, nbits)
    variants = [("encode_ils_iters", i) for i in (2, 3, 4, 8, 16)]
    eval_quantizer(plsq, xq, xb, gt, xt, variants=variants)

if 'rq' in todo:
    print("===== RQ")
    rq = faiss.ResidualQuantizer(d, M, nbits, )
    rq.max_beam_size
    rq.max_beam_size = 30   # for compatibility with older runs
    # rq.train_type = faiss.ResidualQuantizer.Train_default
    # rq.verbose = True
    variants = [("max_beam_size", i) for i in (1, 2, 4, 8, 16, 32)]
    eval_quantizer(rq, xq, xb, gt, xt, variants=variants)

if 'rq_lut' in todo:
    print("===== RQ")
    rq = faiss.ResidualQuantizer(d, M, nbits, )
    rq.max_beam_size
    rq.max_beam_size = 30   # for compatibility with older runs
    rq.use_beam_LUT
    rq.use_beam_LUT = 1
    # rq.train_type = faiss.ResidualQuantizer.Train_default
    # rq.verbose = True
    variants = [("max_beam_size", i) for i in (1, 2, 4, 8, 16, 32, 64)]
    eval_quantizer(rq, xq, xb, gt, xt, variants=variants)

if 'lsq' in todo:
    print("===== LSQ")
    lsq = faiss.LocalSearchQuantizer(d, M, nbits)
    variants = [("encode_ils_iters", i) for i in (2, 3, 4, 8, 16)]
    eval_quantizer(lsq, xq, xb, gt, xt, variants=variants)
