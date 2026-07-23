# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import multiprocessing as mp
import time

import faiss
import matplotlib.pyplot as plt
import numpy as np

try:
    from faiss.contrib.datasets_fb import DatasetSIFT1M, DatasetGIST1M
except ImportError:
    from faiss.contrib.datasets import DatasetSIFT1M, DatasetGIST1M

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="gist1m", choices=["sift1m", "gist1m"])
parser.add_argument(
    "--nq",
    default="1000",
    help="comma-separated query counts to sweep "
    "(query blocking needs a real batch to help)",
)
parser.add_argument(
    "--threads",
    default="1",
    help="comma-separated search thread counts to sweep "
    "(index build always uses all cores)",
)
parser.add_argument(
    "--query-block-size",
    default="0",
    help="comma-separated block sizes to sweep for the Panorama index; "
    "0 or 1 selects the original query-at-a-time path; the first value is "
    "the reference for the vs-Pano speedup in the summary",
)
parser.add_argument(
    "--repeat",
    type=int,
    default=1,
    help="timed repetitions per configuration, reporting the fastest; "
    "raise this for small --nq where a single run is noisy",
)
args = parser.parse_args()

query_block_sizes = [int(v) for v in args.query_block_size.split(",")]
nq_values = [int(v) for v in args.nq.split(",")]
thread_values = [int(v) for v in args.threads.split(",")]

if args.dataset == "sift1m":
    ds = DatasetSIFT1M()
else:
    ds = DatasetGIST1M()

max_nq = max(nq_values)
xq_all = ds.get_queries()[:max_nq]
xb = ds.get_database()
gt_all = ds.get_groundtruth()[:max_nq]

xt = ds.get_train()

nb, d = xb.shape
nt, d = xt.shape

k = 10
gt_all = gt_all[:, :k]


def eval_qps(index, xq, gt):
    nq = len(xq)
    faiss.cvar.indexPanorama_stats.reset()
    t = np.inf
    for _ in range(args.repeat):
        t0 = time.time()
        _, I = index.search(xq, k=k)
        t = min(t, time.time() - t0)
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
    return index


nlevels = 16 if args.dataset == "gist1m" else 8
batch_size = 512

pano_name = f"PCA{d},FlatL2Panorama{nlevels}_{batch_size}"

# Both indexes are built once; only nq, the thread count, and the global
# query-block toggle change between runs, so results must match.
print("======building Flat")
flat_index = build_index("Flat")
print(f"======building {pano_name}")
pano_index = build_index(pano_name)


def qbs_tag(qbs):
    return "baseline" if qbs <= 1 else f"block={qbs}"


rows = []  # one entry per (nq, threads): flat QPS + per-block-size results
for nq in nq_values:
    xq = xq_all[:nq]
    gt = gt_all[:nq]
    for nthr in thread_values:
        faiss.omp_set_num_threads(nthr)
        print(f"====== nq={nq} threads={nthr}")
        print("---Flat")
        flat_recall, flat_qps = eval_qps(flat_index, xq, gt)
        per_qbs = []
        for qbs in query_block_sizes:
            faiss.cvar.panorama_query_block_size = qbs
            print(f"---Pano {qbs_tag(qbs)}")
            recall, qps = eval_qps(pano_index, xq, gt)
            per_qbs.append((qbs, recall, qps))
        faiss.cvar.panorama_query_block_size = 0
        rows.append((nq, nthr, flat_recall, flat_qps, per_qbs))

# Report speedups: Panorama-vs-Flat, and each block size vs the first swept
# block size at the same nq/threads.
ref_tag = qbs_tag(query_block_sizes[0])
print("\n=== summary ===")
for nq, nthr, _, flat_qps, per_qbs in rows:
    ref_qps = per_qbs[0][2]
    print(f"nq={nq} threads={nthr}: Flat {flat_qps:.1f} QPS")
    for qbs, recall, qps in per_qbs:
        vs_flat = qps / flat_qps
        vs_ref = qps / ref_qps
        print(
            f"  Pano {qbs_tag(qbs):>10}: {qps:10.1f} QPS  "
            f"({vs_flat:.2f}x vs Flat, {vs_ref:.2f}x vs Pano {ref_tag})"
        )

# The bar chart only makes sense for a single nq/threads combination; sweeps
# rely on the summary table above.
if len(rows) == 1:
    nq, nthr, flat_recall, flat_qps, per_qbs = rows[0]
    labels = [f"Flat\n(r@{flat_recall:.3f})"]
    qps_values = [flat_qps]
    for qbs, recall, qps in per_qbs:
        labels.append(f"Pano\n{qbs_tag(qbs)}\n(r@{recall:.3f})")
        qps_values.append(qps)

    plt.figure(figsize=(8, 6), dpi=80)
    x = np.arange(len(qps_values))
    colors = ["#1f77b4"] + ["#ff7f0e"] * len(per_qbs)
    plt.bar(x, qps_values, color=colors)
    ax = plt.gca()
    # Annotate each Panorama bar with its speedup over plain Flat.
    for xi, qps in zip(x[1:], qps_values[1:]):
        ax.text(
            xi, qps * 1.01, f"{qps / flat_qps:.2f}x", ha="center", va="bottom"
        )
    plt.xticks(x, labels, rotation=0)
    plt.ylabel("QPS")
    dataset_label = args.dataset.upper()
    plt.title(f"Flat Indexes on {dataset_label}")

    plt.tight_layout()
    plt.savefig(f"bench_flat_l2_panorama_{args.dataset}.png", bbox_inches="tight")
