# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing as mp
import time

import faiss
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import block_diag
from sklearn.decomposition import PCA

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
nprobes = [1, 2, 4, 8, 16, 32, 64]


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


def eval_index(index, label):
    ivf_index = get_ivf_index(index)

    faiss.omp_set_num_threads(1)

    data = []
    print(f"====== {label}")
    for nprobe in nprobes:
        ivf_index.nprobe = nprobe
        recall, qps = eval_recall(index, nprobe)
        data.append((recall, qps))

    data = np.array(data)
    plt.plot(data[:, 0], data[:, 1], "o-", label=label)


def build_ivfpq(M):
    """Build vanilla IVFPQ (no transform) via index_factory."""
    index = faiss.index_factory(d, f"IVF{nlist},PQ{M}x{nbits}")
    faiss.omp_set_num_threads(mp.cpu_count())
    index.train(xt)
    index.add(xb)
    return index


def make_pca_level_rotation_transform(xt, n_levels, seed=77):
    """Build a fused PCA + per-level random rotation as a LinearTransform.

    FAISS LinearTransform applies: y = A_stored @ x + b  (column-vector)
    We want: y = R_block @ P @ (x - mean)
      1. Center x
      2. PCA project (P @ x_centered)
      3. Per-level rotation (R_block @ z_pca)

    So: A_stored = R_block @ P,  b = -A_stored @ mean
    """
    pca = PCA(n_components=d)
    pca.fit(xt)

    P = pca.components_.astype(np.float32)  # (d, d)
    mean = pca.mean_.astype(np.float32)     # (d,)

    block_size = d // n_levels
    rng = np.random.RandomState(seed)
    blocks = []
    for _ in range(n_levels):
        H = rng.randn(block_size, block_size).astype(np.float32)
        Q, R = np.linalg.qr(H)
        Q *= np.sign(np.diag(R))[:, None]
        blocks.append(Q)
    A = block_diag(*blocks).astype(np.float32)  # (d, d)

    combined = A @ P  # (d, d)  -- rotation AFTER PCA

    lt = faiss.LinearTransform(d, d, True)
    faiss.copy_array_to_vector(combined.ravel(), lt.A)
    faiss.copy_array_to_vector(-(combined @ mean).ravel(), lt.b)
    lt.is_trained = True
    lt.have_bias = True

    return lt


def build_ivfpq_panorama(M, n_levels):
    """Build PCA + LevelRotation + IVFPQPanorama."""
    lt = make_pca_level_rotation_transform(xt, n_levels)

    quantizer = faiss.IndexFlatL2(d)
    ivfpq_pano = faiss.IndexIVFPQPanorama(
        quantizer, d, nlist, M, nbits, n_levels,
    )

    index = faiss.IndexPreTransform(lt, ivfpq_pano)

    faiss.omp_set_num_threads(mp.cpu_count())
    index.train(xt)
    index.add(xb)

    return index


plt.figure(figsize=(10, 7), dpi=80)

for M in M_values:
    ivfpq = build_ivfpq(M)
    eval_index(ivfpq, label=f"IVFPQ (M={M})")
    del ivfpq

    pano = build_ivfpq_panorama(M, n_levels)
    eval_index(pano, label=f"PCA+Rot + IVFPQPanorama (M={M})")
    del pano

plt.title(f"IVFPQ Panorama on GIST1M (nlist={nlist})")
plt.xlabel(f"Recall@{k}")
plt.ylabel("QPS")
plt.yscale("log")
plt.legend(bbox_to_anchor=(1.02, 0.1), loc="upper left", borderaxespad=0)
plt.savefig("bench_ivfpq_panorama.png", bbox_inches="tight")
print("\nBenchmark complete! Plot saved to bench_ivfpq_panorama.png")
