# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
import faiss
import time
from datasets import load_sift1M

def evaluate(index, xq, gt, k):
    nq = xq.shape[0]
    t0 = time.time()
    D, I = index.search(xq, k)  # noqa: E741
    t1 = time.time()
    recalls = {}
    i = 1
    while i <= k:
        recalls[i] = (I[:, :i] == gt[:nq, :1]).sum() / float(nq)
        i *= 10

    return (t1 - t0) * 1000.0 / nq, recalls

xb, xq, xt, gt = load_sift1M()
nq, d = xq.shape

k = 50

quantizer = faiss.IndexFlat(d)
index = faiss.IndexIVFFlat(quantizer, d, 1024)
index.train(xt)
index.add(xb)

for pmode in 0,4:
    print("pmode = ",pmode)
    for nprobe in 16,32,64:
        for nq in 100,1000,10000:
            try:
                index.parallel_mode = pmode
                index.nprobe = nprobe
                sub_xq = xq[:nq]
                t, r = evaluate(index, sub_xq, gt, k)
                print("\tnprobe=%d,nq=%d; %7.3f ms per query, R@1 %.4f" % (nprobe, nq, t, r[1]))
            except Exception as e:
                print(e)