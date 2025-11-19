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
    from faiss.contrib.datasets_fb import (
        DatasetSIFT1M,
        DatasetGIST1M,
        SyntheticDataset,
    )
except ImportError:
    from faiss.contrib.datasets import (
        DatasetSIFT1M,
        DatasetGIST1M,
        SyntheticDataset,
    )


def eval_recall(index, efSearch_val, xq, gt, k):
    """Evaluate recall and QPS for a given efSearch value."""
    t0 = time.time()
    _, I = index.search(xq, k=k)
    t = time.time() - t0
    speed = t * 1000 / len(xq)
    qps = 1000 / speed

    corrects = (gt == I).sum()
    recall = corrects / (len(xq) * k)
    print(
        f"\tefSearch {efSearch_val:3d}, Recall@{k}: "
        f"{recall:.6f}, speed: {speed:.6f} ms/query, QPS: {qps:.2f}"
    )

    return recall, qps


def get_hnsw_index(index):
    """Extract the underlying HNSW index from a PreTransform index."""
    if isinstance(index, faiss.IndexPreTransform):
        return faiss.downcast_index(index.index)
    return index


def eval_and_plot(name, ds, k=10, nlevels=8, plot_data=None):
    """Evaluate an index configuration and collect data for plotting."""
    xq = ds.get_queries()
    xb = ds.get_database()
    gt = ds.get_groundtruth()

    if hasattr(ds, "get_train"):
        xt = ds.get_train()
    else:
        # Use database as training data if no separate train set
        xt = xb

    nb, d = xb.shape
    nq, d = xq.shape
    gt = gt[:, :k]

    print(f"\n======{name} on {ds.__class__.__name__}======")
    print(f"Database: {nb} vectors, {d} dimensions")
    print(f"Queries: {nq} vectors")

    # Create index
    index = faiss.index_factory(d, name)

    faiss.omp_set_num_threads(mp.cpu_count())
    index.train(xt)
    index.add(xb)

    faiss.omp_set_num_threads(1)

    # Get the underlying HNSW index for setting efSearch
    hnsw_index = get_hnsw_index(index)

    data = []
    for efSearch in [16, 32, 64, 128, 256, 512]:
        hnsw_index.hnsw.efSearch = efSearch
        recall, qps = eval_recall(index, efSearch, xq, gt, k)
        data.append((recall, qps))

    if plot_data is not None:
        data = np.array(data)
        plot_data.append((name, data))


def benchmark_dataset(ds, dataset_name, k=10, nlevels=8, M=32):
    """Benchmark both regular HNSW and HNSW Panorama on a dataset."""
    d = ds.d

    plot_data = []

    # HNSW Flat (baseline)
    eval_and_plot(f"HNSW{M},Flat", ds, k=k, nlevels=nlevels, plot_data=plot_data)

    # HNSW Flat Panorama (with PCA to concentrate energy)
    eval_and_plot(
        f"PCA{d},HNSW{M},FlatPanorama{nlevels}",
        ds,
        k=k,
        nlevels=nlevels,
        plot_data=plot_data,
    )

    # Plot results
    plt.figure(figsize=(8, 6), dpi=80)
    for name, data in plot_data:
        plt.plot(data[:, 0], data[:, 1], marker="o", label=name)

    plt.title(f"HNSW Indexes on {dataset_name}")
    plt.xlabel(f"Recall@{k}")
    plt.ylabel("QPS")
    plt.yscale("log")
    plt.legend(bbox_to_anchor=(1.02, 0.1), loc="upper left", borderaxespad=0)
    plt.grid(True, alpha=0.3)

    output_file = f"bench_hnsw_flat_panorama_{dataset_name}.png"
    plt.savefig(output_file, bbox_inches="tight")
    print(f"Saved plot to {output_file}")
    plt.close()


if __name__ == "__main__":
    k = 10
    nlevels = 8
    M = 32

    # Test on 3 datasets with varying dimensionality:
    # SIFT1M (128d), GIST1M (960d), and Synthetic high-dim (2048d)
    datasets = [
        (DatasetSIFT1M(), "SIFT1M"),
        (DatasetGIST1M(), "GIST1M"),
        # Synthetic high-dimensional dataset: 2048d, 100k train, 1M database, 10k queries
        (SyntheticDataset(2048, 100000, 1000000, 10000), "Synthetic2048D"),
    ]

    for ds, name in datasets:
        print(f"\n{'='*60}")
        print(f"Benchmarking on {name}")
        print(f"{'='*60}")
        benchmark_dataset(ds, name, k=k, nlevels=nlevels, M=M)

    print("\n" + "="*60)
    print("All benchmarks completed!")
    print("="*60)
