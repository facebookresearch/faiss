#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function

import time
import numpy as np
import faiss
from datasets import load_sift1M, evaluate

NUM_TRAIN_RUNS = 5

print("load data")
xb, xq, xt, gt = load_sift1M()
nq, d = xq.shape

train_times = []
for run in range(NUM_TRAIN_RUNS):
    index = faiss.IndexPQ(d, 16, 8)
    index.do_polysemous_training = True
    index.verbose = (run == 0)

    print("train run %d/%d" % (run + 1, NUM_TRAIN_RUNS))

    t0 = time.time()
    index.train(xt)
    t1 = time.time()
    elapsed = t1 - t0
    train_times.append(elapsed)
    print("  Training time: %.2f s" % elapsed)

times = np.array(train_times)
print("\nTraining time over %d runs: "
      "median %.2f s, mean %.2f s, std %.2f s, min %.2f s, max %.2f s"
      % (NUM_TRAIN_RUNS, np.median(times), np.mean(times),
         np.std(times), np.min(times), np.max(times)))

print("\nadd vectors to index")

index.add(xb)

nt = 1
faiss.omp_set_num_threads(1)


print("PQ baseline", end=' ')
index.search_type = faiss.IndexPQ.ST_PQ
t, r = evaluate(index, xq, gt, 1)
print("\t %7.3f ms per query, R@1 %.4f" % (t, r[1]))

for ht in 64, 62, 58, 54, 50, 46, 42, 38, 34, 30:
    print("Polysemous", ht, end=' ')
    index.search_type = faiss.IndexPQ.ST_polysemous
    index.polysemous_ht = ht
    t, r = evaluate(index, xq, gt, 1)
    print("\t %7.3f ms per query, R@1 %.4f" % (t, r[1]))

