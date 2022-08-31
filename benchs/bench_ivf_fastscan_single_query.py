# Copyright (c) Facebook, Inc. and its affiliates.
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
ds = DatasetBigANN(nb_M=50)
# ds = DatasetSIFT1M()

xq = ds.get_queries()
xb = ds.get_database()
gt = ds.get_groundtruth()

xt = ds.get_train()

nb, d = xb.shape
nq, d = xq.shape
nt, d = xt.shape

print('the dimension is {}, {}'.format(nb, d))

k = 64


def eval_recall(index, name, single_query=False):
    t0 = time.time()
    D, I = index.search(xq, k=k)

    t = time.time() - t0
    if single_query:
        t0 = time.time()
        for row in range(nq):
            Ds, Is = index.search(xq[row:row + 1], k=k)
            D[row, :] = Ds
            I[row, :] = Is
        t = time.time() - t0
    speed = t * 1000 / nq
    qps = 1000 / speed

    corrects = (gt[:, :1] == I[:, :k]).sum()
    recall = corrects / nq
    print(
        f'\tnprobe {index.nprobe:3d}, 1Recall@{k}: '
        f'{recall:.6f}, speed: {speed:.6f} ms/query'
    )

    return recall, qps


def eval_and_plot(
        name, rescale_norm=True, plot=True, single_query=False,
        implem=None, num_threads=1):
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
    if implem is not None and hasattr(index, 'implem'):
        index.implem = implem
        name += f"(implem={implem})"
    if single_query:
        name += f"(single_query={single_query})"
    if num_threads > 1:
        name += f"(num_threads={num_threads})"

    faiss.omp_set_num_threads(num_threads)

    data = []
    print(f"======{name}")
    for nprobe in 1, 4, 8, 16, 32, 64, 128, 256:
        index.nprobe = nprobe
        recall, qps = eval_recall(index, name, single_query=single_query)
        data.append((recall, qps))

    if plot:
        data = np.array(data)
        plt.plot(data[:, 0], data[:, 1], label=name)  # x - recall, y - qps


M, nlist = 64, 4096

# just for warmup...
# eval_and_plot(f"IVF{nlist},PQ{M}x4fs", plot=False)

# benchmark
plt.figure(figsize=(8, 6), dpi=80)

eval_and_plot(f"IVF{nlist},PQ{M}x4fs", num_threads=8)
eval_and_plot(f"IVF{nlist},PQ{M}x4fs", single_query=True, implem=0, num_threads=8)
eval_and_plot(f"IVF{nlist},PQ{M}x4fs", single_query=True, implem=14, num_threads=8)
eval_and_plot(f"IVF{nlist},PQ{M}x4fs", single_query=True, implem=15, num_threads=8)

plt.title("Indices on Bigann50M")
plt.xlabel("1Recall@{}".format(k))
plt.ylabel("QPS")
plt.legend(bbox_to_anchor=(1.02, 0.1), loc='upper left', borderaxespad=0)
plt.savefig("bench_ivf_fastscan.png", bbox_inches='tight')
