# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Python counterpart of cpp/bench_codec_kmeans.cpp. The cpp file benchmarks
# only the k-means assignment step (IndexFlatL2 search over k centroids);
# this file mirrors that and additionally benchmarks full faiss.Kmeans
# training over the same d/k grid. Here `k` is the number of centroids
# (as in the cpp file), not a search result count.

import faiss
import numpy as np

from bench_utils import dataset_or_skip, params, rand_mat, require_attr


@params(d=[128, 256], k=[256, 1024, 4096], n=[10000])
def bench_kmeans_assignment(benchmark, d, k, n):
    x = rand_mat(n, d)
    centroids = rand_mat(k, d, seed=54321)
    index = faiss.IndexFlatL2(d)
    index.add(centroids)
    benchmark(index.search, x, 1)


@params(d=[128, 256], k=[256, 1024, 4096], n=[200000])
def bench_kmeans_train(benchmark, d, k, n):
    x = rand_mat(n, d)

    def fresh_kmeans():
        # niter=10 (not the faiss default 25): the cpp file only times a
        # single assignment step, and 25 iterations at d=256/k=4096
        # single-threaded is pathologically slow for a benchmark run.
        km = faiss.Kmeans(d, k, niter=10, verbose=False, seed=12345)
        return (km,), {}

    # rounds=1: training is expensive; each round trains a fresh Kmeans.
    benchmark.pedantic(
            lambda km: km.train(x), setup=fresh_kmeans, rounds=1)


# SuperKMeans vs vanilla faiss.Clustering on the same synthetic Gaussian
# mixture (k centers in [-1, 1]^d, ~n/k samples per center, sigma 0.1).
# Defaults: n=10000, d=128, k=64, niter=10, seed=42.
def _gaussian_mixture(n, d, k, seed=42):
    rng = np.random.RandomState(seed)
    centers = rng.uniform(-1.0, 1.0, size=(k, d)).astype("float32")
    cluster = np.arange(n) % k
    noise = rng.normal(0.0, 0.1, size=(n, d)).astype("float32")
    return centers[cluster] + noise


@params(d=[128], k=[64], n=[10000])
def bench_super_kmeans_train(benchmark, d, k, n):
    require_attr(faiss, "SuperKMeans")
    x = _gaussian_mixture(n, d, k)

    def fresh_super_kmeans():
        p = faiss.SuperKMeansParameters()
        p.seed = 42
        p.niter = 10
        p.verbose = False
        return (faiss.SuperKMeans(d, k, p),), {}

    benchmark.pedantic(
            lambda sc: sc.train(x), setup=fresh_super_kmeans, rounds=1)


@params(d=[128], k=[64], n=[10000])
def bench_clustering_train(benchmark, d, k, n):
    x = _gaussian_mixture(n, d, k)

    def fresh_clustering():
        cl = faiss.Clustering(d, k)
        cl.seed = 42
        cl.niter = 10
        cl.verbose = False
        return (cl, faiss.IndexFlatL2(d)), {}

    benchmark.pedantic(
            lambda cl, quant: cl.train(x, quant),
            setup=fresh_clustering,
            rounds=1)


# ---------------------------------------------------------------------------
# SIFT1M dataset variants: assignment step and full k-means training over
# the real learn set (nt=100000, d=128). Registered only when --data_dir
# points at a SIFT1M-layout directory.
# ---------------------------------------------------------------------------


@params(k=[256, 1024])
def bench_kmeans_assignment_sift1m(benchmark, data_dir, k):
    ds = dataset_or_skip(data_dir)
    xt = ds["xt"]
    # The first k learn vectors stand in for the centroids; as in the cpp
    # file, only the assignment step (IndexFlatL2 search) is timed.
    index = faiss.IndexFlatL2(xt.shape[1])
    index.add(xt[:k])
    benchmark(index.search, xt, 1)


@params(k=[256, 1024])
def bench_kmeans_train_sift1m(benchmark, data_dir, k):
    ds = dataset_or_skip(data_dir)
    xt = ds["xt"]

    def fresh_kmeans():
        # niter=10, as in the synthetic sweep above. faiss.Kmeans itself
        # subsamples to max_points_per_centroid=256 vectors per centroid,
        # which keeps the k=256 training round bounded.
        km = faiss.Kmeans(xt.shape[1], k, niter=10, verbose=False, seed=12345)
        return (km,), {}

    # rounds=1: each round trains a fresh Kmeans over the 100k learn set.
    benchmark.pedantic(lambda km: km.train(xt), setup=fresh_kmeans, rounds=1)
