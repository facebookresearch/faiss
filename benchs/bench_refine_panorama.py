# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing as mp
import time

import faiss
import matplotlib.pyplot as plt

try:
    from faiss.contrib.datasets_fb import DatasetGIST1M, DatasetSIFT1M
except ImportError:
    from faiss.contrib.datasets import DatasetGIST1M, DatasetSIFT1M


k = 10
# Single-threaded so the comparison is apples-to-apples.
faiss.omp_set_num_threads(1)


def eval_once(index, queries, gt, params=None):
    nq = len(queries)
    t0 = time.time()
    _, I = index.search(queries, k=k, params=params)
    t = time.time() - t0
    qps = nq / t
    corrects = (gt == I).sum()
    recall = corrects / (nq * k)
    return recall, qps


def build_refine_indexes(d, xt, xb, factory_string, n_levels):
    base_index = faiss.index_factory(d, factory_string)
    faiss.omp_set_num_threads(mp.cpu_count())

    base_index.train(xt)
    base_index.add(xb)

    refine_pano = faiss.index_factory(d, f"PCA{d},FlatL2Panorama{n_levels}_1")
    refine_pano.train(xt)
    refine_pano.add(xb)

    faiss.omp_set_num_threads(1)

    idx_flat = faiss.IndexRefineFlat(base_index, faiss.swig_ptr(xb))
    idx_pano = faiss.IndexRefinePanorama(base_index, refine_pano)

    return base_index, idx_flat, idx_pano


def benchmark_dataset(
    name,
    ds,
    factory,
    nlevels,
    nq=100,
    nprobe_list=(4, 16, 64, 256),
    kfactor_list=(1, 8, 64, 256, 1024),
    fixed_nprobe=16,
):
    xq = ds.get_queries()[:nq]
    xb = ds.get_database()
    gt = ds.get_groundtruth()[:nq, :k]
    xt = ds.get_train()
    nb, d = xb.shape

    print(f"\n{'=' * 72}")
    print(
        f"Benchmark on {name} (d={d}, nb={nb}, nq={nq}) "
        f"with base '{factory}', nlevels={nlevels}, k={k}"
    )
    print(f"{'=' * 72}")
    print(
        f"{'nprobe':>6}  {'k_fac':>7}   "
        f"{'recall_flat':>11}  {'qps_flat':>9}   "
        f"{'recall_pano':>11}  {'qps_pano':>9}   "
        f"{'dims(%)':>8}  speedup"
    )

    base_index, idx_flat, idx_pano = build_refine_indexes(
        d, xt, xb, factory, nlevels
    )

    plt.figure(figsize=(8, 5), dpi=300)
    qps_f_list, qps_p_list = [], []

    for nprobe in nprobe_list:
        base_index.nprobe = nprobe
        for kf in kfactor_list:
            params = faiss.IndexRefineSearchParameters(k_factor=float(kf))

            recall_f, qps_f = eval_once(idx_flat, xq, gt, params=params)

            faiss.cvar.indexPanorama_stats.reset()
            recall_p, qps_p = eval_once(idx_pano, xq, gt, params=params)
            dims_pct = faiss.cvar.indexPanorama_stats.ratio_dims_scanned * 100.0

            speedup = qps_p / qps_f

            print(
                f"{nprobe:6d}  {kf:7.1f}   "
                f"{recall_f:11.6f}  {qps_f:9.2f}   "
                f"{recall_p:11.6f}  {qps_p:9.2f}   "
                f"{dims_pct:7.2f}%  {speedup:6.2f}x"
            )

            if nprobe == fixed_nprobe:
                qps_f_list.append(qps_f)
                qps_p_list.append(qps_p)
                plt.plot([kf, kf], [qps_f, qps_p],
                         "k--", linewidth=1, alpha=0.7)
                mid_y = (qps_f * qps_p) ** 0.5
                plt.text(
                    kf * 1.05, mid_y,
                    f"{speedup:.2f}x\nr={recall_p:.2f}",
                    ha="left", va="center", fontsize=8,
                )

    if qps_f_list:
        plt.plot(kfactor_list, qps_f_list, marker="o", label="RefineFlat")
        plt.plot(
            kfactor_list, qps_p_list, marker="o",
            label=f"RefineFlatPanorama({nlevels})",
        )
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("k_factor (k_base = k * k_factor)")
        plt.ylabel("QPS")
        plt.title(
            f"{name}, base={factory}, nprobe={fixed_nprobe}, "
            f"nlevels={nlevels}, k={k}"
        )
        plt.legend()
        plt.tight_layout()
        out = f"bench_refine_panorama_{name}.png"
        plt.savefig(out, bbox_inches="tight")
        print(f"\nSaved plot to {out}")
    plt.close()


if __name__ == "__main__":
    # SIFT1M: low-dim (128), use few Panorama levels (level width 64).
    benchmark_dataset(
        "SIFT1M",
        DatasetSIFT1M(),
        factory="IVF256,PQ32x4fs",
        nlevels=2,
    )

    # GIST1M: high-dim (960), use many Panorama levels (level width 120).
    benchmark_dataset(
        "GIST1M",
        DatasetGIST1M(),
        factory="IVF256,PQ60x4fs",
        nlevels=8,
    )
