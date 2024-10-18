# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import faiss
import time
import os
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt

try:
    from faiss.contrib.datasets_fb import \
        DatasetSIFT1M, DatasetDeep1B, DatasetBigANN
except ImportError:
    from faiss.contrib.datasets import \
        DatasetSIFT1M, DatasetDeep1B, DatasetBigANN


# ds = DatasetDeep1B(10**6)
# ds = DatasetBigANN(nb_M=1)
ds = DatasetSIFT1M()

xq = ds.get_queries()
xb = ds.get_database()
gt = ds.get_groundtruth()

xt = ds.get_train()

nb, d = xb.shape
nq, d = xq.shape
nt, d = xt.shape

k = 1
AQ = faiss.AdditiveQuantizer


def eval_recall(index, name):
    t0 = time.time()
    D, I = index.search(xq, k=k)
    t = time.time() - t0
    speed = t * 1000 / nq
    qps = 1000 / speed

    corrects = (gt == I).sum()
    recall = corrects / nq
    print(
        f'\tnprobe {index.nprobe:3d}, Recall@{k}: '
        f'{recall:.6f}, speed: {speed:.6f} ms/query'
    )

    return recall, qps


def eval_and_plot(name, rescale_norm=True, plot=True):
    index = faiss.index_factory(d, name)
    index_path = f"indices/{name}.faissindex"

    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
    else:
        faiss.omp_set_num_threads(mp.cpu_count())
        index.train(xt)
        index.add(xb)
        faiss.write_index(index, index_path)

    # search params
    if hasattr(index, 'rescale_norm'):
        index.rescale_norm = rescale_norm
        name += f"(rescale_norm={rescale_norm})"
    faiss.omp_set_num_threads(1)

    data = []
    print(f"======{name}")
    for nprobe in 1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128:
        index.nprobe = nprobe
        recall, qps = eval_recall(index, name)
        data.append((recall, qps))

    if plot:
        data = np.array(data)
        plt.plot(data[:, 0], data[:, 1], label=name)  # x - recall, y - qps


M, nlist = 32, 1024

# just for warmup...
# eval_and_plot(f"IVF{nlist},PQ{M}x4fs", plot=False)

# benchmark
plt.figure(figsize=(8, 6), dpi=80)

# PQ
eval_and_plot(f"IVF{nlist},PQ{M}x4fs")
eval_and_plot(f"IVF{nlist},PQ{M}x4fsr")

# AQ, by_residual
eval_and_plot(f"IVF{nlist},LSQ{M-2}x4fsr_Nlsq2x4")
eval_and_plot(f"IVF{nlist},RQ{M-2}x4fsr_Nrq2x4")
eval_and_plot(f"IVF{nlist},LSQ{M-2}x4fsr_Nlsq2x4", rescale_norm=False)
eval_and_plot(f"IVF{nlist},RQ{M-2}x4fsr_Nrq2x4", rescale_norm=False)

# AQ, no by_residual
eval_and_plot(f"IVF{nlist},LSQ{M-2}x4fs_Nlsq2x4")
eval_and_plot(f"IVF{nlist},RQ{M-2}x4fs_Nrq2x4")

plt.title("Indices on SIFT1M")
plt.xlabel("Recall@1")
plt.ylabel("QPS")
plt.legend(bbox_to_anchor=(1.02, 0.1), loc='upper left', borderaxespad=0)
plt.savefig("bench_ivf_fastscan.png", bbox_inches='tight')
