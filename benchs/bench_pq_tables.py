#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import os
import numpy as np
import faiss

os.system("grep -m1 'model name' < /proc/cpuinfo")

def format_tab(x):
    return "\n".join("\t".join("%g" % xi for xi in row) for row in x)


def run_bench(d, dsub, nbit=8, metric=None):

    M = d // dsub
    pq = faiss.ProductQuantizer(d, M, nbit)
    pq.train(faiss.randn((max(1000, pq.ksub * 50), d), 123))


    sp = faiss.swig_ptr

    times = []
    nrun = 100

    print(f"d={d} dsub={dsub} ksub={pq.ksub}", end="\t")
    res = []
    for nx in 1, 10, 100:
        x = faiss.randn((nx, d), 555)

        times = []
        for run in range(nrun):
            t0 = time.time()
            new_tab = np.zeros((nx, M, pq.ksub), "float32")
            if metric == faiss.METRIC_INNER_PRODUCT:
                pq.compute_inner_prod_tables(nx, sp(x), sp(new_tab))
            elif metric == faiss.METRIC_L2:
                pq.compute_distance_tables(nx, sp(x), sp(new_tab))
            else:
                assert False
            t1 = time.time()
            if run >= nrun // 5: # the rest is considered warmup
                times.append((t1 - t0))
        times = np.array(times) * 1000

        print(f"nx={nx}: {np.mean(times):.3f} ms (± {np.std(times):.4f})",
               end="\t")
        res.append(times.mean())
    print()
    return res

# for have_threads in True, False:
for have_threads in False, True:

    if have_threads:
        # good config for Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz
        nthread = 32
    else:
        nthread = 1

    faiss.omp_set_num_threads(nthread)

    for metric in faiss.METRIC_INNER_PRODUCT, faiss.METRIC_L2:
        print("============= nthread=", nthread, "metric=", metric)
        allres = []
        for dsub in 2, 4, 8:
            for nbit in 4, 8:
                for M in 8, 20:
                    res = run_bench(M * dsub, dsub, nbit, metric)
                    allres.append(res)
        allres = np.array(allres)
        print("formated result:")
        print(format_tab(allres))
