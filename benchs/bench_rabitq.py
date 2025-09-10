#!/usr/bin/env -S grimaldi --kernel faiss
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# fmt: off
# flake8: noqa

# NOTEBOOK_NUMBER: N7030784 (685760243832285)

""":py"""
import timeit
from collections import defaultdict

import faiss
from faiss.contrib.datasets import SyntheticDataset

""":py"""
ds: SyntheticDataset = SyntheticDataset(256, 1_000_000, 1_000_000, 10_000)
nlist: int = 1000
qb: int = 8
# This will contain <"index name", ([recalls],[speeds],[labels (the k)])>
recall_speed_data = defaultdict(lambda: [[], [], []])
# This will contain <"index name", ([recalls],[memory for this index])>
recall_memory_data = defaultdict(lambda: [[], []])

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


def create_index(ds, factory_string):
    index = faiss.index_factory(ds.d, factory_string)
    index.train(ds.get_train())
    index.add(ds.get_database())
    return index


# pyre-ignore
def handle_index(prefix, index, ds, mem, k):
    gt_I = ds.get_groundtruth(k)
    _, I_res = index.search(ds.get_queries(), k)
    avg_speed = trials(index, ds.get_queries(), k)
    recall = compute_recall(gt_I, I_res)
    print(
        f"{prefix} recall@{k}: {recall}.  Average speed: {avg_speed:.1f}ms.  Memory: {mem/1e6:.3f}MB"
    )
    recall_speed_data[prefix][0].append(recall)
    recall_speed_data[prefix][1].append(avg_speed)
    recall_speed_data[prefix][2].append(f"k={k}")
    recall_memory_data[prefix][0].append(recall)
    recall_memory_data[prefix][1].append(mem)


# pyre-ignore
def handle_ivf_index(prefix, index, ds, mem, k, params):
    gt_I = ds.get_groundtruth(k)
    for nprobe in 4, 16, 32:
        params.nprobe = nprobe
        _, I_res = faiss.search_with_parameters(index, ds.get_queries(), k, params)
        avg_speed = trials_ivf(index, ds.get_queries(), k, params)
        recall = compute_recall(gt_I, I_res)
        print(
            f"{prefix} nprobe={nprobe}: recall@{k}: {recall}.  Average speed: {avg_speed:.1f}ms.  Memory: {mem/1e6:.3f}MB"
        )
        recall_speed_data[prefix][0].append(recall)
        recall_speed_data[prefix][1].append(avg_speed)
        recall_speed_data[prefix][2].append(f"k={k}, nprobe={nprobe}")
        recall_memory_data[prefix][0].append(recall)
        recall_memory_data[prefix][1].append(mem)


# pyre-ignore
def vary_k_nprobe_measuring_recall_and_memory(prefix, index, ds, mem):
    classname = type(index).__name__
    for k in 1, 10, 100:
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
# IndexRaBitQ

fac_s = "RaBitQ"
non_ivf_rbq = faiss.index_factory(ds.d, fac_s)
non_ivf_rbq.qb = qb
non_ivf_rbq.train(ds.get_train())
non_ivf_rbq.add(ds.get_database())
mem = non_ivf_rbq.code_size * non_ivf_rbq.ntotal

vary_k_nprobe_measuring_recall_and_memory(fac_s, non_ivf_rbq, ds, mem)

del non_ivf_rbq

""":py '3928150077498381'"""
# IndexIVFRaBitQ with no random rotation

fac_s = f"IVF{nlist},RaBitQ"
rbq1 = faiss.index_factory(ds.d, fac_s)
rbq1.qb = qb
rbq1.train(ds.get_train())
rbq1.add(ds.get_database())
mem = rbq1.code_size * rbq1.ntotal

vary_k_nprobe_measuring_recall_and_memory(fac_s, rbq1, ds, mem)

del rbq1

""":py '1484145352968190'"""
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

vary_k_nprobe_measuring_recall_and_memory(fac_s + "_RROT", index_pt, ds, mem)

del index_pt

""":py '644702398382829'"""
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
        color=colors[i],
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
        color=colors[i],
        label=key,
        markersize=10,
    )

    texts = []
    if i == 0:
        texts.append(plt.text(recalls[0], mems[0], "RaBitQ"))
        texts.append(plt.text(recalls[1], mems[1], "RaBitQ"))
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
