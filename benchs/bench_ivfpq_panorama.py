# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing as mp
import time

import faiss
import matplotlib.pyplot as plt
import numpy as np

try:
    from faiss.contrib.datasets_fb import DatasetGIST1M
except ImportError:
    from faiss.contrib.datasets import DatasetGIST1M

ds = DatasetGIST1M()

SUBSET = 0.1  # Set to 1.0 for full dataset

xq = ds.get_queries()
xb_full = ds.get_database()
nb_full = xb_full.shape[0]
nb = int(nb_full * SUBSET)
xb = xb_full[:nb].copy()
del xb_full

gt = ds.get_groundtruth() if SUBSET == 1.0 else None
xt = ds.get_train()[:max(nb // 2, 50000)]

nb, d = xb.shape
nq = xq.shape[0]
nt = xt.shape[0]

k = 10

if gt is None:
    print(f"Computing ground truth for {SUBSET*100:.0f}% subset ({nb} vectors)...")
    flat = faiss.IndexFlatL2(d)
    flat.add(xb)
    _, gt = flat.search(xq, k)
else:
    gt = gt[:, :k]

print(f"Database: {nb} x {d}, Queries: {nq}, Train: {nt}")

M_values = [960, 480, 240]
nbits = 8
nlist = 128
n_levels = 16


def get_ivf_index(index):
    if isinstance(index, faiss.IndexPreTransform):
        return faiss.downcast_index(index.index)
    return index


def eval_recall(index, nprobe_val):
    faiss.cvar.indexPanorama_stats.reset()
    t0 = time.time()
    _, I = index.search(xq, k=k)
    t = time.time() - t0
    speed = t * 1000 / nq
    qps = 1000 / speed

    corrects = (gt == I).sum()
    recall = corrects / (nq * k)
    ratio_dims_scanned = faiss.cvar.indexPanorama_stats.ratio_dims_scanned
    print(
        f"\tnprobe {nprobe_val:3d}, Recall@{k}: "
        f"{recall:.6f}, speed: {speed:.6f} ms/query, QPS: {qps:.1f}, "
        f"dims scanned: {ratio_dims_scanned * 100:.1f}%"
    )

    return recall, qps


def build_index(name):
    index = faiss.index_factory(d, name)

    faiss.omp_set_num_threads(mp.cpu_count())
    index.train(xt)
    index.add(xb)

    return index


def eval_and_plot(name, label=None):
    index = build_index(name)
    ivf_index = get_ivf_index(index)

    faiss.omp_set_num_threads(1)

    data = []
    print(f"====== {label or name}")
    for nprobe in nprobes:
        ivf_index.nprobe = nprobe
        recall, qps = eval_recall(index, nprobe)
        data.append((recall, qps))

    data = np.array(data)
    plt.plot(data[:, 0], data[:, 1], "o-", label=label or name)


nprobes = [1, 2, 4, 8, 16, 32, 64]

plt.figure(figsize=(10, 7), dpi=80)

for M in M_values:
    eval_and_plot(
        f"IVF{nlist},PQ{M}x{nbits}",
        label=f"IVFPQ (M={M})",
    )
    eval_and_plot(
        f"PCA{d},IVF{nlist},PQ{M}x{nbits}Panorama{n_levels}",
        label=f"PCA + IVFPQPanorama (M={M})",
    )

plt.title(f"IVFPQ Panorama on GIST1M (nlist={nlist})")
plt.xlabel(f"Recall@{k}")
plt.ylabel("QPS")
plt.yscale("log")
plt.legend(bbox_to_anchor=(1.02, 0.1), loc="upper left", borderaxespad=0)
plt.savefig("bench_ivfpq_panorama.png", bbox_inches="tight")
print("\nBenchmark complete! Plot saved to bench_ivfpq_panorama.png")
