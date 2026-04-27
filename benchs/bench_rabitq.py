#!/usr/bin/env -S grimaldi --kernel faiss
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# fmt: off
# flake8: noqa

""":py"""
import statistics
import timeit
from collections import defaultdict

import faiss
from faiss.contrib.datasets import SyntheticDataset

""":py"""
# Dimensions to sweep. The rabitq SIMD kernel
# (bitwise_and_dot_product / bitwise_xor_dot_product / popcount) selects
# its widest tier based on size = d / 8 bytes:
#   d=256  -> size=32  -> 256-bit ymm only (no 512-bit work)
#   d=512  -> size=64  -> 512-bit zmm, 1 iteration per bit-plane
#   d=768  -> size=96  -> 512-bit zmm + 256-bit tail
#   d=1024 -> size=128 -> 512-bit zmm only, 2 iterations per bit-plane
# Sweeping these is useful for verifying the AVX512_SPR (vpopcntdq)
# specialization in faiss/utils/simd_impl/rabitq_avx512_spr.cpp and for
# profiling perf-record annotations across SIMD-width tiers.
DIMENSIONS = [256, 512, 768, 1024]
nlist: int = 1000
qb: int = 8
# Number of independent timing samples to take per (index, k, nprobe)
# combination. Each sample is itself an average over `trials=10` calls
# inside timeit, so total searches per row = ITERATIONS * 10. Using 3
# samples is enough to flag whether differences across dimensions are
# noise or real, while keeping the bench cheap.
ITERATIONS: int = 3
# This will contain <"index name", ([recalls],[speeds],[labels (the k)])>
recall_speed_data = defaultdict(lambda: [[], [], []])
# This will contain <"index name", ([recalls],[memory for this index])>
recall_memory_data = defaultdict(lambda: [[], []])

# Set when entering each per-d block below; used by helpers that close
# over the active dataset.
ds: SyntheticDataset = None  # type: ignore

""":py"""
# Helpers


def trials(index, xq, k):
    trials = 10
    result = timeit.timeit(
        stmt="index.search(xq, k)",
        number=trials,
        globals={"index": index, "xq": xq, "k": k},
    )
    return result / trials * 1000.0  # ms


def trials_ivf(index, xq, k, params=None):
    trials = 10
    result = timeit.timeit(
        stmt="search_with_parameters(index, xq, k, params)",
        number=trials,
        globals={
            "search_with_parameters": faiss.search_with_parameters,
            "index": index,
            "xq": xq,
            "k": k,
            "params": params,
        },
    )
    return result / trials * 1000.0  # ms


def compute_recall(ground_truth_I, predicted_I):
    n_queries, k = ground_truth_I.shape
    intersection = faiss.eval_intersection(ground_truth_I, predicted_I)
    recall = intersection / (n_queries * k)
    return recall


def repeated_trials(trials_fn, *args, n=ITERATIONS, **kwargs):
    """Run a trials function n times and return the list of per-iteration
    average speeds (each in ms). Each call to trials_fn is itself an
    average over multiple back-to-back searches, so the returned list
    contains n independent samples of that average.
    """
    return [trials_fn(*args, **kwargs) for _ in range(n)]


def summarize(samples):
    """Return (mean, median, stdev) over a list of timing samples in ms.
    stdev is the sample standard deviation (n-1); returns 0.0 for n==1
    since stdev is undefined.
    """
    mean = statistics.mean(samples)
    median = statistics.median(samples)
    stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return mean, median, stdev


def fmt_speed(samples):
    """Format a list of timing samples as 'mean=X median=Y stdev=Z'."""
    mean, median, stdev = summarize(samples)
    return f"mean={mean:.1f}ms median={median:.1f}ms stdev={stdev:.2f}ms"


def create_index(ds, factory_string):
    index = faiss.index_factory(ds.d, factory_string)
    index.train(ds.get_train())
    index.add(ds.get_database())
    return index


# pyre-ignore
def handle_index(prefix, index, ds, mem, k):
    gt_I = ds.get_groundtruth(k)
    _, I_res = index.search(ds.get_queries(), k)
    speed_samples = repeated_trials(trials, index, ds.get_queries(), k)
    mean_speed, _, _ = summarize(speed_samples)
    recall = compute_recall(gt_I, I_res)
    print(
        f"{prefix} recall@{k}: {recall}.  Speed: {fmt_speed(speed_samples)}.  Memory: {mem/1e6:.3f}MB"
    )
    recall_speed_data[prefix][0].append(recall)
    recall_speed_data[prefix][1].append(mean_speed)
    recall_speed_data[prefix][2].append(f"k={k}")
    recall_memory_data[prefix][0].append(recall)
    recall_memory_data[prefix][1].append(mem)


# pyre-ignore
def handle_ivf_index(prefix, index, ds, mem, k, params):
    gt_I = ds.get_groundtruth(k)
    for nprobe in 4, 16, 32:
        params.nprobe = nprobe
        _, I_res = faiss.search_with_parameters(index, ds.get_queries(), k, params)
        speed_samples = repeated_trials(
            trials_ivf, index, ds.get_queries(), k, params
        )
        mean_speed, _, _ = summarize(speed_samples)
        recall = compute_recall(gt_I, I_res)
        print(
            f"{prefix} nprobe={nprobe}: recall@{k}: {recall}.  Speed: {fmt_speed(speed_samples)}.  Memory: {mem/1e6:.3f}MB"
        )
        recall_speed_data[prefix][0].append(recall)
        recall_speed_data[prefix][1].append(mean_speed)
        recall_speed_data[prefix][2].append(f"k={k}, nprobe={nprobe}")
        recall_memory_data[prefix][0].append(recall)
        recall_memory_data[prefix][1].append(mem)


# pyre-ignore
def vary_k_nprobe_measuring_recall_and_memory(prefix, index, ds, mem):
    classname = type(index).__name__
    for k in (100,):
        if classname in [
            "IndexRaBitQ",
            "IndexPQFastScan",
            "IndexHNSWFlat",
            "IndexScalarQuantizer",
        ]:
            handle_index(prefix, index, ds, mem, k)
        elif classname in [
            "IndexIVFRaBitQ",
            "IndexPreTransform",
            "IndexIVFPQFastScan",
            "IndexIVFScalarQuantizer",
        ]:
            if (
                classname == "IndexIVFPQFastScan"
                or classname == "IndexIVFScalarQuantizer"
            ):
                params = faiss.IVFSearchParameters()
            else:
                params = faiss.IVFRaBitQSearchParameters()
                params.qb = qb
            handle_ivf_index(prefix, index, ds, mem, k, params)

""":py '605360559215064'"""
# RaBitQ kernels swept across dimensions. Each iteration rebuilds the
# dataset and the three rabitq index variants. Suffix _d{d} on the
# result key keeps the per-dimension series distinct in the plots.

for d in DIMENSIONS:
    print(f"\n========== d={d} ==========")
    # Dataset sized to keep the full 4-dimension sweep under ~10 minutes.
    # nq=1k is enough for stable timeit averages across 10 trials; nb=200k
    # keeps groundtruth (brute-force knn over xb) tractable at d=1024;
    # nt=100k still satisfies the IVF k-means training-points floor for
    # nlist=1000 (39 × 1000 = 39k minimum).
    ds = SyntheticDataset(d, 100_000, 200_000, 1_000)

    # IndexRaBitQ
    fac_s = "RaBitQ"
    non_ivf_rbq = faiss.index_factory(ds.d, fac_s)
    non_ivf_rbq.qb = qb
    non_ivf_rbq.train(ds.get_train())
    non_ivf_rbq.add(ds.get_database())
    mem = non_ivf_rbq.code_size * non_ivf_rbq.ntotal

    vary_k_nprobe_measuring_recall_and_memory(f"{fac_s}_d{d}", non_ivf_rbq, ds, mem)

    del non_ivf_rbq

    # IndexIVFRaBitQ with no random rotation
    fac_s = f"IVF{nlist},RaBitQ"
    rbq1 = faiss.index_factory(ds.d, fac_s)
    rbq1.qb = qb
    rbq1.train(ds.get_train())
    rbq1.add(ds.get_database())
    mem = rbq1.code_size * rbq1.ntotal

    vary_k_nprobe_measuring_recall_and_memory(f"{fac_s}_d{d}", rbq1, ds, mem)

    del rbq1

    # IndexIVFRaBitQ with random rotation
    fac_s = f"IVF{nlist},RaBitQ"
    rbq2 = faiss.index_factory(ds.d, fac_s)
    rbq2.qb = qb
    rrot = faiss.RandomRotationMatrix(ds.d, ds.d)
    rrot.init(123)
    index_pt = faiss.IndexPreTransform(rrot, rbq2)
    index_pt.train(ds.get_train())
    index_pt.add(ds.get_database())
    mem = rbq2.code_size * index_pt.ntotal

    vary_k_nprobe_measuring_recall_and_memory(
        f"{fac_s}_RROT_d{d}", index_pt, ds, mem
    )

    del index_pt

""":py '644702398382829'"""
# Non-rabitq baselines (SQ, PQfs, HNSW) below. These don't exercise the
# rabitq SIMD kernels, so we don't sweep dimensions for them; instead
# we pick one dimension and build them once. Change BASELINE_D if you
# want a different working point, or comment out the cells below if
# you only care about the rabitq sweep.
BASELINE_D = 256
ds = SyntheticDataset(BASELINE_D, 100_000, 200_000, 1_000)

# IndexScalarQuantizer

for M in [4, 6, 8]:
    fac_s = f"SQ{M}"
    sq = create_index(ds, fac_s)
    mem = sq.code_size * sq.ntotal
    vary_k_nprobe_measuring_recall_and_memory("Index" + fac_s, sq, ds, mem)

""":py '1347502839702520'"""
# IndexIVFScalarQuantizer

for M in [4, 6]:  # 8 seems to have no recall improvement in this dataset.
    fac_s = f"IVF{nlist},SQ{M}"
    sq = create_index(ds, fac_s)
    mem = sq.code_size * sq.ntotal
    vary_k_nprobe_measuring_recall_and_memory(fac_s, sq, ds, mem)

""":py '1350039419637535'"""
# PQFS

for m in [32, 64, 128]:
    fac_s = f"PQ{m}x4fs"
    pqfs = create_index(ds, fac_s)
    mem = pqfs.code_size * pqfs.ntotal
    vary_k_nprobe_measuring_recall_and_memory(fac_s, pqfs, ds, mem)
    del pqfs

""":py '2549074352105737'"""
# IVFPQFS

for m in [32, 64, 128]:
    fac_s = f"IVF{nlist},PQ{m}x4fs"
    ivf_pqfs = create_index(ds, fac_s)
    mem = ivf_pqfs.code_size * ivf_pqfs.ntotal
    vary_k_nprobe_measuring_recall_and_memory(fac_s, ivf_pqfs, ds, mem)
    del ivf_pqfs

""":py '3933359133572530'"""
# HNSW

for m in [8, 16, 32]:
    fac_s = f"HNSW{m}"
    index = create_index(ds, fac_s)
    storage = faiss.downcast_index(index.storage)
    mem = (
        storage.ntotal * storage.code_size
        + index.hnsw.neighbors.size() * 4
        + index.hnsw.offsets.size() * 8
    )
    vary_k_nprobe_measuring_recall_and_memory(fac_s, index, ds, mem)
    del index

""":py"""
import matplotlib.pyplot as plt
from adjustText import adjust_text


# Specific colors that stand out against each other for this many data points.
colors = [
    "black",
    "darkgray",
    "darkred",
    "red",
    "orange",
    "wheat",
    "olive",
    "yellow",
    "lime",
    "teal",
    "cyan",
    "skyblue",
    "royalblue",
    "navy",
    "darkviolet",
    "fuchsia",
    "deeppink",
    "pink",
]

""":py '1023372579245229'"""
slowest_speed = 0.0
for key, vals in recall_speed_data.items():
    for speed in vals[1]:
        slowest_speed = max(slowest_speed, speed)

plt.axis([0, 1.0, 0, slowest_speed + 100.0])  # [xmin, xmax, ymin, ymax]
for i, (key, vals) in enumerate(recall_speed_data.items()):
    recalls = vals[0]
    speeds = vals[1]
    plt.plot(
        recalls,
        speeds,
        linestyle=" ",
        marker="o",
        color=colors[i % len(colors)],
        label=key,
        markersize=15,
    )
    # Adding k and nprobe labels makes the diagram very busy, but can be enabled by uncommenting the following lines:
    # ks = vals[2]
    # texts = []
    # for i, (x_val, y_val) in enumerate(zip(recalls, speeds)):
    #     texts.append(plt.text(x_val, y_val, ks[i]))
    # # Adjust text labels
    # adjust_text(
    #     texts,
    #     arrowprops=dict(arrowstyle="-", color="black", lw=0.5),
    #     force_text=(0.1, 0.25),
    #     force_points=(0.2, 0.5),
    #     only_move={"points": "xy"},
    # )

plt.title("Recall vs Speed")
plt.xlabel("Recall")
plt.ylabel("Speed")
plt.legend()
plt.show()

""":py '1354989919068149'"""
largest_mem = 0.0
for key, vals in recall_memory_data.items():
    for mem in vals[1]:
        largest_mem = max(largest_mem, mem)

plt.ylim(1e6, 1e10)
plt.yscale("log", base=10)

for i, (key, vals) in enumerate(recall_memory_data.items()):
    recalls = vals[0]
    mems = vals[1]
    plt.plot(
        recalls,
        mems,
        linestyle=" ",
        marker="o",
        color=colors[i % len(colors)],
        label=key,
        markersize=10,
    )

    texts = []
    if i == 0:
        for j in range(min(2, len(recalls))):
            texts.append(plt.text(recalls[j], mems[j], "RaBitQ"))
    adjust_text(
        texts,
        arrowprops=dict(arrowstyle="-", color="black", lw=0.5),
        force_text=(0.5, 0.25),
        force_points=(1.0, 1.5),
        expand_points=(5.0, 10.0),
    )

plt.title("Recall vs Memory")
plt.xlabel("Recall")
plt.ylabel("Memory")
plt.legend()
plt.show()

""":py"""
