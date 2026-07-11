/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Big Batch IVF Search Benchmarks
// Benchmarks large query batches with IVF indexes, measuring throughput
// when many queries are processed simultaneously (batch search).
//
// NOTE: The true "big batch" algorithm is faiss.contrib.big_batch_search — a
// Python-only contrib that, for a large query batch, first assigns all queries
// to their nprobe lists, then processes the index list-by-list with a
// reservoir/block strategy so each inverted list is scanned once for every
// query that probes it. There is NO C++ equivalent of that algorithm in faiss.
// The faithful reproduction lives in benchs/python/bench_index_ivf_big_batch.py;
// this C++ file measures ordinary large-batch IndexIVF::search throughput as
// the closest C++ analogue (the "reference search" baseline that
// big_batch_search is compared against).
//
// Default options: d=32, nlist=100, k=10, nprobe=5, and the S/M/L
// SyntheticDataset sizes (nb 4000/40000/400000, trained on nt = nb/2). The
// --factory_string flag allows an arbitrary-factory override.
//
// Design notes:
// - --threads defaults to 1 (not the OpenMP default of all cores).

#include <gflags/gflags.h>
#include <omp.h>

#include <memory>

#include <benchmark/benchmark.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/index_factory.h>
#include <faiss/utils/AlignedTable.h>
#include <faiss/utils/random.h>

#include "bench_cli_utils.h"
#include "bench_dataset_utils.h"

using namespace faiss;

DEFINE_uint32(iterations, 0, "iterations (0 = auto)");
DEFINE_uint32(threads, 1, "number of threads");
BENCH_DEFINE_DATASET_FILE_FLAGS();
DEFINE_string(
        nb,
        "",
        "comma-separated database sizes; synthetic groups train on nb/2 "
        "vectors (default: 4000,40000,400000; ivfpq group: 40000,400000)");
DEFINE_string(
        nq,
        "",
        "comma-separated query batch sizes "
        "(default: 1000,10000,100000; ivfpq group: 1000,10000; "
        "sift1m groups: 1000,10000 / 10000, capped to the dataset size)");
DEFINE_string(nprobe, "", "comma-separated nprobe values (default: 5,10)");
DEFINE_string(
        nlist,
        "",
        "comma-separated IVF list counts, applied to all groups "
        "(default: 100)");
DEFINE_string(k, "", "comma-separated knn k values (default: 10)");
DEFINE_string(
        factory_string,
        "",
        "index_factory override. May contain the "
        "literal '{nlist}' placeholder, substituted with each nlist value; "
        "if empty, uses the built-in IVFFlat/IVFPQ groups");

static void bench_big_batch_ivfflat(
        benchmark::State& state,
        int d,
        int nb,
        int nq,
        int nlist,
        int nprobe,
        int k) {
    int nt = std::max(nb / 2, 39 * nlist);
    AlignedTable<float> xt(d * nt);
    AlignedTable<float> xb((size_t)d * nb);
    AlignedTable<float> xq(d * nq);
    float_rand(xt.data(), d * nt, 12345);
    float_rand(xb.data(), (size_t)d * nb, 54321);
    float_rand(xq.data(), d * nq, 67890);

    IndexFlatL2 quantizer(d);
    IndexIVFFlat index(&quantizer, d, nlist);
    index.verbose = false;
    omp_set_num_threads(FLAGS_threads);
    index.train(nt, xt.data());
    index.add(nb, xb.data());
    index.nprobe = nprobe;

    std::vector<float> distances(nq * k);
    std::vector<int64_t> labels(nq * k);

    // Warmup
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    for (auto _ : state) {
        index.search(nq, xq.data(), k, distances.data(), labels.data());
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["nq"] = nq;
    state.counters["nlist"] = nlist;
    state.counters["nprobe"] = nprobe;
    state.counters["k"] = k;
    state.counters["threads"] = FLAGS_threads;
}

static void bench_big_batch_ivfpq(
        benchmark::State& state,
        int d,
        int nb,
        int nq,
        int nlist,
        int M,
        int nprobe,
        int k) {
    int nt = std::max(nb / 2, 39 * nlist);
    AlignedTable<float> xt(d * nt);
    AlignedTable<float> xb((size_t)d * nb);
    AlignedTable<float> xq(d * nq);
    float_rand(xt.data(), d * nt, 12345);
    float_rand(xb.data(), (size_t)d * nb, 54321);
    float_rand(xq.data(), d * nq, 67890);

    IndexFlatL2 quantizer(d);
    IndexIVFPQ index(&quantizer, d, nlist, M, 8);
    index.verbose = false;
    omp_set_num_threads(FLAGS_threads);
    index.train(nt, xt.data());
    index.add(nb, xb.data());
    index.nprobe = nprobe;

    std::vector<float> distances(nq * k);
    std::vector<int64_t> labels(nq * k);

    // Warmup
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    for (auto _ : state) {
        index.search(nq, xq.data(), k, distances.data(), labels.data());
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["nq"] = nq;
    state.counters["nlist"] = nlist;
    state.counters["M"] = M;
    state.counters["nprobe"] = nprobe;
    state.counters["k"] = k;
    state.counters["threads"] = FLAGS_threads;
}

// SIFT1M variant: accepts external data pointers, builds index once per call
static void bench_big_batch_ivfflat_dataset(
        benchmark::State& state,
        const float* xt,
        int nt,
        const float* xb,
        int nb,
        const float* xq,
        int nq,
        const int32_t* gt,
        size_t gt_k,
        int d,
        int nlist,
        int nprobe,
        int k) {
    IndexFlatL2 quantizer(d);
    IndexIVFFlat index(&quantizer, d, nlist);
    index.verbose = false;
    omp_set_num_threads(FLAGS_threads);
    index.train(nt, xt);
    index.add(nb, xb);
    index.nprobe = nprobe;

    std::vector<float> distances(nq * k);
    std::vector<int64_t> labels(nq * k);

    // Warmup
    index.search(nq, xq, k, distances.data(), labels.data());

    for (auto _ : state) {
        index.search(nq, xq, k, distances.data(), labels.data());
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["nq"] = nq;
    state.counters["nlist"] = nlist;
    state.counters["nprobe"] = nprobe;
    state.counters["k"] = k;
    state.counters["threads"] = FLAGS_threads;
    state.counters["dataset"] = 1;

    // Compute recall from last iteration's results
    double recall =
            benchmarks::compute_recall_at(labels.data(), gt, nq, k, gt_k);
    state.counters["recall"] = recall;
}

// SIFT1M variant for IVFPQ
static void bench_big_batch_ivfpq_dataset(
        benchmark::State& state,
        const float* xt,
        int nt,
        const float* xb,
        int nb,
        const float* xq,
        int nq,
        const int32_t* gt,
        size_t gt_k,
        int d,
        int nlist,
        int M,
        int nprobe,
        int k) {
    IndexFlatL2 quantizer(d);
    IndexIVFPQ index(&quantizer, d, nlist, M, 8);
    index.verbose = false;
    omp_set_num_threads(FLAGS_threads);
    index.train(nt, xt);
    index.add(nb, xb);
    index.nprobe = nprobe;

    std::vector<float> distances(nq * k);
    std::vector<int64_t> labels(nq * k);

    // Warmup
    index.search(nq, xq, k, distances.data(), labels.data());

    for (auto _ : state) {
        index.search(nq, xq, k, distances.data(), labels.data());
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["nq"] = nq;
    state.counters["nlist"] = nlist;
    state.counters["M"] = M;
    state.counters["nprobe"] = nprobe;
    state.counters["k"] = k;
    state.counters["threads"] = FLAGS_threads;
    state.counters["dataset"] = 1;

    // Compute recall from last iteration's results
    double recall =
            benchmarks::compute_recall_at(labels.data(), gt, nq, k, gt_k);
    state.counters["recall"] = recall;
}

// Arbitrary index_factory string (--factory_string override). Measures
// large-batch search throughput of any IVF factory key.
static void bench_big_batch_factory(
        benchmark::State& state,
        std::string factory_string,
        int d,
        int nb,
        int nq,
        int nprobe,
        int k) {
    int nt = nb / 2;
    AlignedTable<float> xt(d * nt);
    AlignedTable<float> xb((size_t)d * nb);
    AlignedTable<float> xq(d * nq);
    float_rand(xt.data(), d * nt, 12345);
    float_rand(xb.data(), (size_t)d * nb, 54321);
    float_rand(xq.data(), d * nq, 67890);

    std::unique_ptr<Index> index(index_factory(d, factory_string.c_str()));
    index->verbose = false;
    omp_set_num_threads(FLAGS_threads);
    index->train(nt, xt.data());
    index->add(nb, xb.data());
    auto* ivf = dynamic_cast<IndexIVF*>(index.get());
    int nlist = 0;
    if (ivf != nullptr) {
        ivf->nprobe = nprobe;
        nlist = (int)ivf->nlist;
    }

    std::vector<float> distances(nq * k);
    std::vector<int64_t> labels(nq * k);

    // Warmup
    index->search(nq, xq.data(), k, distances.data(), labels.data());

    for (auto _ : state) {
        index->search(nq, xq.data(), k, distances.data(), labels.data());
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["nq"] = nq;
    state.counters["nlist"] = nlist;
    state.counters["nprobe"] = nprobe;
    state.counters["k"] = k;
    state.counters["threads"] = FLAGS_threads;
}

int main(int argc, char** argv) {
    benchmarks::bench_init(
            &argc,
            &argv,
            "big-batch IVF search throughput (IVFFlat and IVFPQ indexes "
            "with large query batches)",
            "--nq=1000 --nprobe=5 --benchmark_filter='big_batch_ivfflat/.*'");

    int d = 32;
    int k = benchmarks::int_list(FLAGS_k, {10}).front();
    // Large query batches characteristic of big_batch_ivf
    std::vector<int> nqs =
            benchmarks::int_list(FLAGS_nq, {1000, 10000, 100000});
    std::vector<int> nbs =
            benchmarks::int_list(FLAGS_nb, {4000, 40000, 400000});
    std::vector<int> nprobes = benchmarks::int_list(FLAGS_nprobe, {5, 10});
    std::vector<int> nlists = benchmarks::int_list(FLAGS_nlist, {100});

    // --factory_string override: benchmark an arbitrary IVF factory key
    // over the S/M/L sizes instead of the built-in IVFFlat/IVFPQ groups.
    if (!FLAGS_factory_string.empty()) {
        for (int nlist : nlists) {
            // Substitute a literal "{nlist}" placeholder if present.
            std::string factory = FLAGS_factory_string;
            const std::string ph = "{nlist}";
            size_t pos = factory.find(ph);
            if (pos != std::string::npos) {
                factory.replace(pos, ph.size(), std::to_string(nlist));
            }
            for (int nb : nbs) {
                for (int nq : nqs) {
                    for (int nprobe : nprobes) {
                        std::string name =
                                "big_batch_factory/factory:" + factory +
                                "/nb:" + std::to_string(nb) +
                                "/nq:" + std::to_string(nq) +
                                "/nprobe:" + std::to_string(nprobe) +
                                "/k:" + std::to_string(k);
                        auto* b = benchmark::RegisterBenchmark(
                                name.c_str(),
                                bench_big_batch_factory,
                                factory,
                                d,
                                nb,
                                nq,
                                nprobe,
                                k);
                        if (FLAGS_iterations > 0)
                            b->Iterations(FLAGS_iterations);
                    }
                }
            }
        }
        benchmark::RunSpecifiedBenchmarks();
        benchmark::Shutdown();
        return 0;
    }

    for (int nb : nbs) {
        for (int nlist : nlists) {
            for (int nq : nqs) {
                for (int nprobe : nprobes) {
                    if (nprobe > nlist)
                        continue;
                    std::string name =
                            "big_batch_ivfflat/nb:" + std::to_string(nb) +
                            "/nq:" + std::to_string(nq) +
                            "/nlist:" + std::to_string(nlist) +
                            "/nprobe:" + std::to_string(nprobe);
                    auto* b = benchmark::RegisterBenchmark(
                            name.c_str(),
                            bench_big_batch_ivfflat,
                            d,
                            nb,
                            nq,
                            nlist,
                            nprobe,
                            k);
                    if (FLAGS_iterations > 0)
                        b->Iterations(FLAGS_iterations);
                }
            }
        }
    }

    // IVFPQ with large batches
    std::vector<int> nbs_pq = benchmarks::int_list(FLAGS_nb, {40000, 400000});
    std::vector<int> nqs_pq = benchmarks::int_list(FLAGS_nq, {1000, 10000});
    for (int nb : nbs_pq) {
        int M = 16;
        for (int nlist : nlists) {
            for (int nq : nqs_pq) {
                for (int nprobe : nprobes) {
                    std::string name =
                            "big_batch_ivfpq/nb:" + std::to_string(nb) +
                            "/nq:" + std::to_string(nq) +
                            "/nlist:" + std::to_string(nlist) +
                            "/M:" + std::to_string(M) +
                            "/nprobe:" + std::to_string(nprobe);
                    auto* b = benchmark::RegisterBenchmark(
                            name.c_str(),
                            bench_big_batch_ivfpq,
                            d,
                            nb,
                            nq,
                            nlist,
                            M,
                            nprobe,
                            k);
                    if (FLAGS_iterations > 0)
                        b->Iterations(FLAGS_iterations);
                }
            }
        }
    }

    // SIFT1M-based benchmarks (if dataset available). Big-batch search over
    // the real query set, with nq capped to the dataset's query count.
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
        int snb = (int)sift.nb;
        int snt = (int)sift.nt;

        // IVFFlat on the dataset
        std::vector<int> sift_nqs =
                benchmarks::int_list(FLAGS_nq, {1000, 10000});
        for (int snlist : nlists) {
            for (int nq : sift_nqs) {
                if ((size_t)nq > sift.nq)
                    continue;
                for (int nprobe : nprobes) {
                    std::string name = ds +
                            "/big_batch_ivfflat/nq:" + std::to_string(nq) +
                            "/nlist:" + std::to_string(snlist) +
                            "/nprobe:" + std::to_string(nprobe);
                    auto* b = benchmark::RegisterBenchmark(
                            name.c_str(),
                            bench_big_batch_ivfflat_dataset,
                            sift.xt.data(),
                            snt,
                            sift.xb.data(),
                            snb,
                            sift.xq.data(),
                            nq,
                            sift.gt.data(),
                            sift.gt_k,
                            sd,
                            snlist,
                            nprobe,
                            k);
                    if (FLAGS_iterations > 0)
                        b->Iterations(FLAGS_iterations);
                }
            }
        }

        // IVFPQ on the dataset
        int sift_M = 16;
        std::vector<int> sift_nqs_pq = benchmarks::int_list(FLAGS_nq, {10000});
        for (int snlist : nlists) {
            for (int nq : sift_nqs_pq) {
                if ((size_t)nq > sift.nq)
                    continue;
                for (int nprobe : nprobes) {
                    std::string name = ds +
                            "/big_batch_ivfpq/nq:" + std::to_string(nq) +
                            "/nlist:" + std::to_string(snlist) +
                            "/M:" + std::to_string(sift_M) +
                            "/nprobe:" + std::to_string(nprobe);
                    auto* b = benchmark::RegisterBenchmark(
                            name.c_str(),
                            bench_big_batch_ivfpq_dataset,
                            sift.xt.data(),
                            snt,
                            sift.xb.data(),
                            snb,
                            sift.xq.data(),
                            nq,
                            sift.gt.data(),
                            sift.gt_k,
                            sd,
                            snlist,
                            sift_M,
                            nprobe,
                            k);
                    if (FLAGS_iterations > 0)
                        b->Iterations(FLAGS_iterations);
                }
            }
        }
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
