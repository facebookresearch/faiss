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

ALPHA = 8
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


def compute_level_energies(variances, n_levels, block_size):
    """Sum per-dimension variances into per-level total energies."""
    return np.array([
        np.sum(variances[l * block_size : (l + 1) * block_size])
        for l in range(n_levels)
    ])


def find_n_spill(variances, level_start, block_size, max_energy_per_level, d):
    """Find the smallest number of extra dimensions to spill into.

    After a random rotation over (block_size + n_spill) dims, each dim gets
    uniform expected energy.  The level's expected energy becomes:
        block_size * total_subspace_energy / (block_size + n_spill)

    Returns the smallest n_spill >= 1 where this is <= max_energy_per_level,
    or all remaining dims if the cap can't be reached.
    """
    level_end = level_start + block_size
    max_extra = d - level_end
    if max_extra == 0:
        return 0

    total = np.sum(variances[level_start:level_end])
    for n in range(1, max_extra + 1):
        total += variances[level_end + n - 1]
        if block_size * total / (block_size + n) <= max_energy_per_level:
            return n

    return max_extra


def random_orthogonal(size, rng):
    """Haar-distributed random orthogonal matrix via QR of Gaussian."""
    H = rng.randn(size, size).astype(np.float32)
    Q, R = np.linalg.qr(H)
    Q *= np.sign(np.diag(R))[:, None]
    return Q


def build_energy_spill_rotation(eigenvalues, n_levels, block_size,
                                alpha, seed=42):
    """Orthogonal matrix that caps per-level energy via localized rotations.

    Iterates over levels sequentially.  When a level's effective energy
    exceeds alpha * avg_energy_per_level, applies a random rotation spanning
    that level plus enough subsequent dimensions to bring the expected level
    energy down to the cap.

    Variances are tracked analytically: after each rotation the dims in the
    rotated subspace are set to uniform expected variance.

    Returns (spill_rotation, effective_variances).
    """
    d = len(eigenvalues)
    total_energy = float(np.sum(eigenvalues))
    max_energy_per_level = alpha * total_energy / n_levels

    variances = eigenvalues.astype(np.float32).copy()
    spill_matrix = np.eye(d, dtype=np.float32)
    rng = np.random.RandomState(seed)

    for level in range(n_levels):
        start = level * block_size
        end = start + block_size
        level_energy = float(np.sum(variances[start:end]))

        if level_energy <= max_energy_per_level:
            continue

        n_spill = find_n_spill(
            variances, start, block_size, max_energy_per_level, d,
        )
        if n_spill == 0:
            continue

        sub_end = end + n_spill
        Q = random_orthogonal(block_size + n_spill, rng)

        full_Q = np.eye(d, dtype=np.float32)
        full_Q[start:sub_end, start:sub_end] = Q
        spill_matrix = full_Q @ spill_matrix

        avg_var = float(np.sum(variances[start:sub_end])) / (block_size + n_spill)
        variances[start:sub_end] = avg_var

    return spill_matrix, variances


def build_level_equalization_rotation(d, n_levels, block_size, seed=77):
    """Block-diagonal random rotation for within-level energy equalization."""
    rng = np.random.RandomState(seed)
    blocks = [random_orthogonal(block_size, rng) for _ in range(n_levels)]
    return block_diag(*blocks).astype(np.float32)


def print_energy_diagnostics(eigenvalues, effective_variances, n_levels,
                             block_size, alpha):
    """Print per-level energy before/after the spill transform."""
    before = compute_level_energies(eigenvalues, n_levels, block_size)
    after = compute_level_energies(effective_variances, n_levels, block_size)
    total = float(np.sum(eigenvalues))
    cap = alpha * total / n_levels


def make_pca_level_rotation_transform(xt, n_levels, alpha=ALPHA, seed=77):
    """Build PCA + energy-spill + per-level rotation as one LinearTransform.

    Pipeline:  y = R_eq @ R_spill @ P @ (x - mean)
      1. Center + PCA project           (P, mean)
      2. Energy spill across levels      (R_spill)
      3. Within-level equalization       (R_eq, block-diagonal)

    Stored as:  A = R_eq @ R_spill @ P,  b = -A @ mean
    """
    dim = xt.shape[1]
    block_size = dim // n_levels

    pca = PCA(n_components=dim)
    pca.fit(xt)
    P = pca.components_.astype(np.float32)
    mean = pca.mean_.astype(np.float32)
    eigenvalues = pca.explained_variance_.astype(np.float32)

    R_spill, effective_variances = build_energy_spill_rotation(
        eigenvalues, n_levels, block_size, alpha, seed=seed,
    )
    print_energy_diagnostics(
        eigenvalues, effective_variances, n_levels, block_size, alpha,
    )

    R_eq = build_level_equalization_rotation(
        dim, n_levels, block_size, seed=seed + 1,
    )

    combined = (R_eq @ R_spill @ P).astype(np.float32)

    lt = faiss.LinearTransform(dim, dim, True)
    faiss.copy_array_to_vector(combined.ravel(), lt.A)
    faiss.copy_array_to_vector(-(combined @ mean).ravel(), lt.b)
    lt.is_trained = True
    lt.have_bias = True

    return lt


def build_ivfpq_panorama(M, n_levels, alpha=ALPHA):
    """Build PCA + EnergySpill + LevelRotation + IVFPQPanorama."""
    lt = make_pca_level_rotation_transform(xt, n_levels, alpha=alpha)

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
    eval_index(pano, label=f"PCA+Spill+Rot + IVFPQPanorama (M={M})")
    del pano

plt.title(f"IVFPQ Panorama on GIST1M (nlist={nlist})")
plt.xlabel(f"Recall@{k}")
plt.ylabel("QPS")
plt.yscale("log")
plt.legend(bbox_to_anchor=(1.02, 0.1), loc="upper left", borderaxespad=0)
plt.savefig("bench_ivfpq_panorama.png", bbox_inches="tight")
print("\nBenchmark complete! Plot saved to bench_ivfpq_panorama.png")
