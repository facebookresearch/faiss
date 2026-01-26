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

nq = 10
xq = ds.get_queries()[:nq]
xb = ds.get_database()
gt = ds.get_groundtruth()[:nq]

xt = ds.get_train()

nb, d = xb.shape
nt, d = xt.shape

k = 10
gt = gt[:, :k]


def eval_qps(index):
    faiss.cvar.indexPanorama_stats.reset()
    t0 = time.time()
    _, I = index.search(xq, k=k)
    t = time.time() - t0
    speed = t * 1000 / nq  # ms/query
    qps = 1000 / speed

    corrects = (gt == I).sum()
    recall = corrects / (nq * k)
    ratio_dims_scanned = faiss.cvar.indexPanorama_stats.ratio_dims_scanned
    print(
        f"\tRecall@{k}: {recall:.6f}, speed: {speed:.6f} ms/query, "
        f"dims scanned: {ratio_dims_scanned * 100:.2f}%"
    )
    return recall, qps


def build_index(name):
    index = faiss.index_factory(d, name)

    faiss.omp_set_num_threads(mp.cpu_count())
    index.train(xt)
    index.add(xb)

    faiss.omp_set_num_threads(1)
    return index


nlevels = 8
batch_size = 512

plt.figure(figsize=(8, 6), dpi=80)

names = [
    "Flat",
    f"PCA{d},FlatL2Panorama{nlevels}_{batch_size}",
]

labels = []
qps_values = []

for name in names:
    print(f"======{name}")
    index = build_index(name)
    recall, qps = eval_qps(index)
    labels.append(f"{name}\n(r@{recall:.3f})")
    qps_values.append(qps)

x = np.arange(len(names))
plt.bar(x, qps_values, color=['#1f77b4', '#ff7f0e'])
speedup = qps_values[1] / qps_values[0]
ax = plt.gca()
ax.text(
	x[1],
	qps_values[1] * 1.01,
	f"{speedup:.2f}x",
	ha="center",
	va="bottom",
)
plt.xticks(x, labels, rotation=0)
plt.ylabel("QPS")
plt.title("Flat Indexes on GIST1M")

plt.tight_layout()
plt.savefig("bench_flat_l2_panorama.png", bbox_inches="tight")
