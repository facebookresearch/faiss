# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing as mp
import time
import math

import faiss
import matplotlib.pyplot as plt
import numpy as np

try:
    from faiss.contrib.datasets_fb import DatasetGIST1M
except ImportError:
    from faiss.contrib.datasets import DatasetGIST1M


ds = DatasetGIST1M()

nq = 100
xq = ds.get_queries()[:nq]
xb = ds.get_database()
gt = ds.get_groundtruth()[:nq]

xt = ds.get_train()

nb, d = xb.shape
nt, d = xt.shape

k = 10
gt = gt[:, :k]


def eval_once(index, queries, params=None):
    t0 = time.time()
    D, I = index.search(queries, k=k, params=params)
    t = time.time() - t0
    speed_ms = t * 1000 / nq
    qps = 1000.0 / speed_ms
    corrects = (gt == I).sum()
    recall = corrects / (nq * k)
    return recall, qps


def build_refine_indexes(factory_string, n_levels):
    base_index = faiss.index_factory(d, factory_string)
    faiss.omp_set_num_threads(mp.cpu_count())

    base_index.train(xt)
    base_index.add(xb)

    refine_pano = faiss.index_factory(d, f"PCA{d},FlatL2Panorama{n_levels}_1")
    refine_pano.train(xt)
    refine_pano.add(xb)

    faiss.omp_set_num_threads(1)

    # Refine wrappers
    idx_flat = faiss.IndexRefineFlat(base_index, faiss.swig_ptr(xb))
    idx_pano = faiss.IndexRefinePanorama(base_index, refine_pano)

    return base_index, idx_flat, idx_pano


factory = "IVF256,PQ60x4fs"
nlevels = 8

base_index, idx_flat, idx_pano = build_refine_indexes(factory, nlevels)

# Parameter sweeps
nprobe_list = [4, 16, 64, 256]
kfactor_list = [1, 8, 64, 256, 1024]

print(f"Benchmark on GIST1M with base '{factory}', k={k}, nq={nq}")
print("nprobe  k_factor   recall_flat   qps_flat   recall_pano   qps_pano   dims_scanned(%)  speedup(x)")

faiss.omp_set_num_threads(1)

# Visualization at a fixed nprobe
plt.figure(figsize=(8, 5), dpi=300)
qps_f_list, qps_p_list = [], []
fixed_nprobe = 16

for nprobe in nprobe_list:
    base_index.nprobe = nprobe

    for kf in kfactor_list:
        params = faiss.IndexRefineSearchParameters(k_factor=kf)

        # Flat refinement
        recall_f, qps_f = eval_once(idx_flat, xq, params=params)

        # Panorama refinement
        faiss.cvar.indexPanorama_stats.reset()
        recall_p, qps_p = eval_once(idx_pano, xq, params=params)
        dims_pct = faiss.cvar.indexPanorama_stats.ratio_dims_scanned * 100.0

        speedup = qps_p / qps_f

        print(
            f"{nprobe:6d}  {kf:7.1f}   "
            f"{recall_f:11.6f}  {qps_f:9.2f}   "
            f"{recall_p:11.6f}  {qps_p:9.2f}   "
            f"{dims_pct:15.2f}  {speedup:9.2f}x"
        )

        if nprobe == fixed_nprobe:
            qps_f_list.append(qps_f)
            qps_p_list.append(qps_p)

            # Draw speedup and recall
            plt.plot([kf, kf], [qps_f, qps_p], 'k--', linewidth=1, alpha=0.7)
            mid_y = (qps_f * qps_p) ** 0.5
            plt.text(kf + 10, mid_y, f"{speedup:.2f}x\nr={recall_p:.2f}", 
            ha="left", va="center", fontsize=8)

plt.plot(kfactor_list, qps_f_list, label="RefineFlat")
plt.plot(kfactor_list, qps_p_list, label=f"RefineFlatPanorama({nlevels})")
plt.yscale("log")
plt.ylim(bottom=100)
plt.xlim(right=kfactor_list[-1] * 1.075)
plt.xlabel("k_factor (k_base = k * k_factor)")
plt.ylabel("QPS")
plt.title(f"GIST1M, base={factory}, nprobe={fixed_nprobe}, nlevels={nlevels}, k={k}")
plt.legend()
plt.tight_layout()
plt.savefig("bench_refine_panorama.png", bbox_inches="tight")
