/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Super K-Means Kernel Benchmarks
// Benchmarks the IndexFlatL2 search which uses the same BLAS-based distance
// computation path as the super k-means kernels.

#include <gflags/gflags.h>
#include <omp.h>

#include <benchmark/benchmark.h>
#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/SuperKMeans.h>
#include <faiss/utils/AlignedTable.h>
#include <faiss/utils/random.h>

#include "bench_cli_utils.h"
#include "bench_dataset_utils.h"

using namespace faiss;

DEFINE_uint32(iterations, 0, "iterations (0 = auto)");
BENCH_DEFINE_DATASET_FILE_FLAGS();
DEFINE_string(
        d,
        "",
        "comma-separated vector dimensions (default: 128,256; "
        "train group: 128)");
DEFINE_string(
        k,
        "",
        "comma-separated numbers of centroids (default: 256,1024,4096; "
        "train group: 64)");

// Benchmark the distance computation loop that super_kmeans uses
static void bench_kmeans_assignment(
        benchmark::State& state,
        int d,
        int k,
        int n) {
    AlignedTable<float> x(d * n);
    AlignedTable<float> centroids(d * k);
    float_rand(x.data(), d * n, 12345);
    float_rand(centroids.data(), d * k, 54321);

    IndexFlatL2 index(d);
    index.add(k, centroids.data());

    std::vector<float> distances(n);
    std::vector<int64_t> labels(n);

    omp_set_num_threads(1);
    for (auto _ : state) {
        index.search(n, x.data(), 1, distances.data(), labels.data());
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * (int64_t)n * k);
    state.SetBytesProcessed(state.iterations() * (n + k) * d * sizeof(float));
    state.counters["d"] = d;
    state.counters["k"] = k;
    state.counters["n"] = n;
}

// Synthetic Gaussian mixture (mirrors python bench_codec_kmeans.py
// _gaussian_mixture): k centers
// uniform in [-1, 1]^d, point i assigned to center i % k, sigma 0.1.
static std::vector<float> gaussian_mixture(int n, int d, int k, int seed) {
    std::vector<float> centers((size_t)k * d);
    float_rand(centers.data(), (size_t)k * d, seed);
    for (float& c : centers) {
        c = 2.0f * c - 1.0f; // [0, 1) -> [-1, 1)
    }
    std::vector<float> x((size_t)n * d);
    float_randn(x.data(), (size_t)n * d, seed + 1);
    for (int i = 0; i < n; i++) {
        const float* c = centers.data() + (size_t)(i % k) * d;
        float* xi = x.data() + (size_t)i * d;
        for (int j = 0; j < d; j++) {
            xi[j] = c[j] + 0.1f * xi[j];
        }
    }
    return x;
}

// Full SuperKMeans training; each iteration constructs and trains a
// fresh object (training is the timed region).
static void bench_super_kmeans_train(
        benchmark::State& state,
        int d,
        int k,
        int n) {
    std::vector<float> x = gaussian_mixture(n, d, k, 42);

    omp_set_num_threads(1);
    for (auto _ : state) {
        SuperKMeansParameters p;
        p.seed = 42;
        p.niter = 10;
        p.verbose = false;
        SuperKMeans skm(d, k, p);
        skm.train(n, x.data());
        benchmark::DoNotOptimize(skm.centroids[0]);
    }
    state.SetItemsProcessed(state.iterations() * (int64_t)n);
    state.counters["d"] = d;
    state.counters["k"] = k;
    state.counters["n"] = n;
}

// Vanilla Clustering + IndexFlatL2 baseline on the same data.
static void bench_clustering_train(
        benchmark::State& state,
        int d,
        int k,
        int n) {
    std::vector<float> x = gaussian_mixture(n, d, k, 42);

    omp_set_num_threads(1);
    for (auto _ : state) {
        Clustering clus(d, k);
        clus.seed = 42;
        clus.niter = 10;
        clus.verbose = false;
        IndexFlatL2 quantizer(d);
        clus.train(n, x.data(), quantizer);
        benchmark::DoNotOptimize(clus.centroids[0]);
    }
    state.SetItemsProcessed(state.iterations() * (int64_t)n);
    state.counters["d"] = d;
    state.counters["k"] = k;
    state.counters["n"] = n;
}

// SIFT1M variant: assignment of real train vectors against k centroids
// sampled (evenly strided) from the full train set.
static void bench_kmeans_assignment_dataset(
        benchmark::State& state,
        const float* x,
        int n,
        int nt,
        int d,
        int k) {
    std::vector<float> centroids((size_t)k * d);
    for (int i = 0; i < k; i++) {
        memcpy(centroids.data() + (size_t)i * d,
               x + ((size_t)i * nt / k) * d,
               d * sizeof(float));
    }

    IndexFlatL2 index(d);
    index.add(k, centroids.data());

    std::vector<float> distances(n);
    std::vector<int64_t> labels(n);

    omp_set_num_threads(1);
    for (auto _ : state) {
        index.search(n, x, 1, distances.data(), labels.data());
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * (int64_t)n * k);
    state.SetBytesProcessed(state.iterations() * (n + k) * d * sizeof(float));
    state.counters["d"] = d;
    state.counters["k"] = k;
    state.counters["n"] = n;
    state.counters["dataset"] = 1;
}

// SIFT1M variant: full SuperKMeans training on real train vectors
// (subsampled outside the timed loop; training is the timed region).
static void bench_super_kmeans_train_dataset(
        benchmark::State& state,
        const float* x,
        int n,
        int d,
        int k) {
    omp_set_num_threads(1);
    for (auto _ : state) {
        SuperKMeansParameters p;
        p.seed = 42;
        p.niter = 10;
        p.verbose = false;
        SuperKMeans skm(d, k, p);
        skm.train(n, x);
        benchmark::DoNotOptimize(skm.centroids[0]);
    }
    state.SetItemsProcessed(state.iterations() * (int64_t)n);
    state.counters["d"] = d;
    state.counters["k"] = k;
    state.counters["n"] = n;
    state.counters["dataset"] = 1;
}

// SIFT1M variant: vanilla Clustering + IndexFlatL2 baseline on the same
// real train vectors.
static void bench_clustering_train_dataset(
        benchmark::State& state,
        const float* x,
        int n,
        int d,
        int k) {
    omp_set_num_threads(1);
    for (auto _ : state) {
        Clustering clus(d, k);
        clus.seed = 42;
        clus.niter = 10;
        clus.verbose = false;
        IndexFlatL2 quantizer(d);
        clus.train(n, x, quantizer);
        benchmark::DoNotOptimize(clus.centroids[0]);
    }
    state.SetItemsProcessed(state.iterations() * (int64_t)n);
    state.counters["d"] = d;
    state.counters["k"] = k;
    state.counters["n"] = n;
    state.counters["dataset"] = 1;
}

int main(int argc, char** argv) {
    benchmarks::bench_init(
            &argc,
            &argv,
            "k-means assignment step via IndexFlatL2 search (the BLAS-based "
            "distance path used by the super k-means kernels)",
            "--d=128 --k=1024 --benchmark_filter='kmeans_assignment/.*'");

    std::vector<int> ks = benchmarks::int_list(FLAGS_k, {256, 1024, 4096});
    std::vector<int> dims = benchmarks::int_list(FLAGS_d, {128, 256});
    int n = 10000;

    for (int d : dims) {
        for (int k : ks) {
            std::string name = "kmeans_assignment/d:" + std::to_string(d) +
                    "/k:" + std::to_string(k) + "/n:" + std::to_string(n);
            auto* b = benchmark::RegisterBenchmark(
                    name.c_str(), bench_kmeans_assignment, d, k, n);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);
        }
    }

    // Full training: SuperKMeans vs vanilla Clustering on a Gaussian
    // mixture (defaults n=10000, d=128, k=64, niter=10, seed=42)
    std::vector<int> dims_train = benchmarks::int_list(FLAGS_d, {128});
    std::vector<int> ks_train = benchmarks::int_list(FLAGS_k, {64});
    for (int d : dims_train) {
        for (int k : ks_train) {
            std::string suffix = "/d:" + std::to_string(d) +
                    "/k:" + std::to_string(k) + "/n:" + std::to_string(n);

            auto* b = benchmark::RegisterBenchmark(
                    ("super_kmeans_train" + suffix).c_str(),
                    bench_super_kmeans_train,
                    d,
                    k,
                    n);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);

            b = benchmark::RegisterBenchmark(
                    ("clustering_train" + suffix).c_str(),
                    bench_clustering_train,
                    d,
                    k,
                    n);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);
        }
    }

    // SIFT1M-based benchmarks (if dataset available)
    static benchmarks::DatasetSIFT1M sift;
    if (benchmarks::dataset_available(FLAGS_data_dir) &&
        sift.load(
                FLAGS_data_dir,
                FLAGS_train_file,
                FLAGS_base_file,
                FLAGS_query_file,
                FLAGS_gt_file)) {
        const std::string ds = benchmarks::dataset_label(FLAGS_base_file);
        int sd = (int)sift.d;
        int snt = (int)sift.nt;
        // Subsample the real train set (SIFT1M nt=100k) to the synthetic
        // scale n=10000 to keep dataset-variant runtimes comparable.
        int sn = n < snt ? n : snt;

        for (int k : ks) {
            std::string name = ds +
                    "/kmeans_assignment/k:" + std::to_string(k) +
                    "/n:" + std::to_string(sn);
            auto* b = benchmark::RegisterBenchmark(
                    name.c_str(),
                    bench_kmeans_assignment_dataset,
                    sift.xt.data(),
                    sn,
                    snt,
                    sd,
                    k);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);
        }

        for (int k : ks_train) {
            std::string suffix =
                    "/k:" + std::to_string(k) + "/n:" + std::to_string(sn);

            auto* b = benchmark::RegisterBenchmark(
                    (ds + "/super_kmeans_train" + suffix).c_str(),
                    bench_super_kmeans_train_dataset,
                    sift.xt.data(),
                    sn,
                    sd,
                    k);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);

            b = benchmark::RegisterBenchmark(
                    (ds + "/clustering_train" + suffix).c_str(),
                    bench_clustering_train_dataset,
                    sift.xt.data(),
                    sn,
                    sd,
                    k);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);
        }
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
