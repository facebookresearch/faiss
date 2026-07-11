/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Residual Coarse Quantizer Search Benchmarks
// Benchmarks RCQ search at various batch sizes, nprobes, and beam factors.
//
// Design notes:
// - Uses Google Benchmark auto-iteration (use --iterations to pin).

#include <gflags/gflags.h>
#include <omp.h>

#include <memory>

#include <benchmark/benchmark.h>
#include <faiss/IndexAdditiveQuantizer.h>
#include <faiss/utils/AlignedTable.h>
#include <faiss/utils/random.h>

#include "bench_cli_utils.h"
#include "bench_dataset_utils.h"

using namespace faiss;

DEFINE_uint32(iterations, 0, "iterations (0 = auto)");
BENCH_DEFINE_DATASET_FILE_FLAGS();
DEFINE_string(
        batch,
        "",
        "comma-separated query batch sizes (default: 1,4,16,64)");
DEFINE_string(nprobe, "", "comma-separated nprobe values (default: 1,4,16)");
DEFINE_string(beam, "", "comma-separated beam factors (default: 1,4)");

static void bench_rcq_search(
        benchmark::State& state,
        int d,
        int batch_size,
        int nprobe,
        float beam_factor) {
    // the 16-bit first level needs >= 65536 training points for k-means
    int nt = 2 << 15; // 65536
    AlignedTable<float> xt(d * nt);
    float_rand(xt.data(), d * nt, 12345);

    ResidualCoarseQuantizer rq(d, {16, 8});
    rq.verbose = false;
    rq.rq.cp.min_points_per_centroid = 1;
    omp_set_num_threads(1);
    rq.train(nt, xt.data());

    AlignedTable<float> xq(d * batch_size);
    float_rand(xq.data(), d * batch_size, 12345);

    std::vector<float> distances(nprobe * batch_size);
    std::vector<int64_t> labels(nprobe * batch_size);

    SearchParametersResidualCoarseQuantizer params;
    params.beam_factor = beam_factor;

    // Warmup
    rq.search(
            batch_size,
            xq.data(),
            nprobe,
            distances.data(),
            labels.data(),
            &params);

    for (auto _ : state) {
        rq.search(
                batch_size,
                xq.data(),
                nprobe,
                distances.data(),
                labels.data(),
                &params);
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * batch_size);
    state.counters["d"] = d;
    state.counters["batch_size"] = batch_size;
    state.counters["nprobe"] = nprobe;
    state.counters["beam_factor"] = beam_factor;
}

// Dataset-backed RCQ: trained once on the real train vectors and reused
// across all dataset registrations (training the 16-bit first level is
// expensive).
static ResidualCoarseQuantizer* rcq_dataset_index(
        const float* xt,
        int nt,
        int d) {
    static std::unique_ptr<ResidualCoarseQuantizer> rq;
    if (!rq) {
        rq.reset(new ResidualCoarseQuantizer(d, {16, 8}));
        rq->verbose = false;
        omp_set_num_threads(1);
        rq->train(nt, xt);
    }
    return rq.get();
}

// SIFT1M variant: RCQ trained on the real train vectors, searched with real
// queries.
static void bench_rcq_search_dataset(
        benchmark::State& state,
        const float* xt,
        int nt,
        const float* xq,
        int d,
        int batch_size,
        int nprobe,
        float beam_factor) {
    ResidualCoarseQuantizer* rq = rcq_dataset_index(xt, nt, d);
    omp_set_num_threads(1);

    std::vector<float> distances(nprobe * batch_size);
    std::vector<int64_t> labels(nprobe * batch_size);

    SearchParametersResidualCoarseQuantizer params;
    params.beam_factor = beam_factor;

    // Warmup
    rq->search(
            batch_size, xq, nprobe, distances.data(), labels.data(), &params);

    for (auto _ : state) {
        rq->search(
                batch_size,
                xq,
                nprobe,
                distances.data(),
                labels.data(),
                &params);
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * batch_size);
    state.counters["d"] = d;
    state.counters["batch_size"] = batch_size;
    state.counters["nprobe"] = nprobe;
    state.counters["beam_factor"] = beam_factor;
    state.counters["dataset"] = 1;
}

int main(int argc, char** argv) {
    benchmarks::bench_init(
            &argc,
            &argv,
            "ResidualCoarseQuantizer search at various batch sizes, "
            "nprobes and beam factors",
            "--batch=16 --beam=4 --benchmark_filter='rcq_search/.*'");

    int d = 512;
    std::vector<int> batch_sizes =
            benchmarks::int_list(FLAGS_batch, {1, 4, 16, 64});
    std::vector<int> nprobes = benchmarks::int_list(FLAGS_nprobe, {1, 4, 16});
    std::vector<int> beam_factors = benchmarks::int_list(FLAGS_beam, {1, 4});

    for (int batch_size : batch_sizes) {
        for (int nprobe : nprobes) {
            for (int beam_factor_int : beam_factors) {
                float beam_factor = (float)beam_factor_int;
                std::string name =
                        "rcq_search/batch:" + std::to_string(batch_size) +
                        "/nprobe:" + std::to_string(nprobe) +
                        "/beam:" + std::to_string((int)beam_factor);
                auto* b = benchmark::RegisterBenchmark(
                        name.c_str(),
                        bench_rcq_search,
                        d,
                        batch_size,
                        nprobe,
                        beam_factor);
                if (FLAGS_iterations > 0)
                    b->Iterations(FLAGS_iterations);
            }
        }
    }

    // SIFT1M-based benchmarks (if dataset available). The 16-bit first level
    // needs >= 65536 training points, so skip datasets with smaller train
    // sets.
    static benchmarks::DatasetSIFT1M sift;
    if (benchmarks::dataset_available(FLAGS_data_dir) &&
        sift.load(
                FLAGS_data_dir,
                FLAGS_train_file,
                FLAGS_base_file,
                FLAGS_query_file,
                FLAGS_gt_file) &&
        sift.nt >= 65536) {
        const std::string ds = benchmarks::dataset_label(FLAGS_base_file);
        // Keep the dataset cross-product small: a few representative batch
        // and nprobe points, crossed with the beam-factor sweep.
        std::vector<int> sift_batches =
                benchmarks::int_list(FLAGS_batch, {1, 64});
        std::vector<int> sift_nprobes =
                benchmarks::int_list(FLAGS_nprobe, {1, 16});
        for (int batch_size : sift_batches) {
            if ((size_t)batch_size > sift.nq)
                continue;
            for (int nprobe : sift_nprobes) {
                for (int beam_factor_int : beam_factors) {
                    float beam_factor = (float)beam_factor_int;
                    std::string name = ds +
                            "/rcq_search/batch:" + std::to_string(batch_size) +
                            "/nprobe:" + std::to_string(nprobe) +
                            "/beam:" + std::to_string((int)beam_factor);
                    auto* b = benchmark::RegisterBenchmark(
                            name.c_str(),
                            bench_rcq_search_dataset,
                            sift.xt.data(),
                            (int)sift.nt,
                            sift.xq.data(),
                            (int)sift.d,
                            batch_size,
                            nprobe,
                            beam_factor);
                    if (FLAGS_iterations > 0)
                        b->Iterations(FLAGS_iterations);
                }
            }
        }
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
