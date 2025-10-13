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
        DatasetGIST1M
except ImportError:
    from faiss.contrib.datasets import \
        DatasetGIST1M

ds = DatasetGIST1M()

xq = ds.get_queries()
xb = ds.get_database()
gt = ds.get_groundtruth()

xt = ds.get_train()

nb, d = xb.shape
nq, d = xq.shape
nt, d = xt.shape

k = 1
nlevels = 8


def get_ivf_index(index):
    """Get the underlying IVF index, handling IndexPreTransform wrapper."""
    if isinstance(index, faiss.IndexPreTransform):
        return faiss.downcast_index(index.index)
    return index


def eval_recall(index, name, nprobe_val):
    ivf_index = get_ivf_index(index)
    t0 = time.time()
    _, I = index.search(xq, k=k)
    t = time.time() - t0
    speed = t * 1000 / nq
    qps = 1000 / speed

    corrects = (gt == I).sum()
    recall = corrects / nq
    print(
        f'\tnprobe {nprobe_val:3d}, Recall@{k}: '
        f'{recall:.6f}, speed: {speed:.6f} ms/query'
    )

    return recall, qps


def eval_and_plot(name, plot=True):
    index = faiss.index_factory(d, name)
    index_path = f"indices/{name}.faissindex"

    faiss.omp_set_num_threads(mp.cpu_count())
    index.train(xt)
    index.add(xb)
    faiss.write_index(index, index_path)

    faiss.omp_set_num_threads(1)

    # Get the underlying IVF index for setting nprobe
    ivf_index = get_ivf_index(index)

    data = []
    print(f"======{name}")
    for nprobe in 1, 2, 4, 6, 8, 12:
        ivf_index.nprobe = nprobe
        recall, qps = eval_recall(index, name, nprobe)
        data.append((recall, qps))

    if plot:
        data = np.array(data)
        plt.plot(data[:, 0], data[:, 1], label=name)  # x - recall, y - qps


nlist = 1024

plt.figure(figsize=(8, 6), dpi=80)

# IVFFlat
eval_and_plot(f"IVF{nlist},Flat")

# IVFFlatPanorama (with PCAR transform to concentrate energy in early dimensions)
eval_and_plot(f"PCAR{d},IVF{nlist},FlatPanorama{nlevels}")

plt.title("Indices on GIST1M")
plt.xlabel("Recall@1")
plt.ylabel("QPS")
plt.legend(bbox_to_anchor=(1.02, 0.1), loc='upper left', borderaxespad=0)
plt.savefig("bench_ivf_flat_panorama.png", bbox_inches='tight')

