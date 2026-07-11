/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Scalar Quantizer Benchmarks
//
// The distance benchmark measures SQDistanceComputer over all SQ types for
// BOTH METRIC_L2 and METRIC_INNER_PRODUCT. The accuracy benchmark measures
// reconstruction error and encode idempotence (ndiff) counters over all SQ
// types.
//
// Design notes:
//   * Auto-iterates by default (use --iterations to pin a fixed count).
//   * code_size_two/code_size_three counters and compile-options labels
//     are omitted; they add noise without actionable signal.

#include <gflags/gflags.h>
#include <omp.h>

#include <memory>

#include <benchmark/benchmark.h>
#include <faiss/MetricType.h>
#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/utils/AlignedTable.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>

#include "bench_cli_utils.h"
#include "bench_dataset_utils.h"

using namespace faiss;

DEFINE_uint32(iterations, 0, "iterations (0 = auto)");
BENCH_DEFINE_DATASET_FILE_FLAGS();
DEFINE_string(
        type,
        "",
        "comma-separated scalar quantizer types "
        "(default: QT_8bit,QT_4bit,QT_6bit,QT_8bit_uniform,QT_4bit_uniform,"
        "QT_fp16,QT_bf16,QT_8bit_direct,QT_8bit_direct_signed)");
DEFINE_string(d, "", "comma-separated vector dimensions (default: 128,768)");
DEFINE_uint32(n, 2000, "number of vectors");

static void bench_sq_encode(
        benchmark::State& state,
        ScalarQuantizer::QuantizerType type,
        int d) {
    int n = FLAGS_n;
    AlignedTable<float> x(d * n);
    float_rand(x.data(), d * n, 12345);

    ScalarQuantizer sq(d, type);
    omp_set_num_threads(1);
    sq.train(n, x.data());

    std::vector<uint8_t> codes(sq.code_size * n);
    state.counters["code_size"] = sq.code_size;

    for (auto _ : state) {
        sq.compute_codes(x.data(), codes.data(), n);
        benchmark::DoNotOptimize(codes[0]);
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * d * sizeof(float));
}

static void bench_sq_decode(
        benchmark::State& state,
        ScalarQuantizer::QuantizerType type,
        int d) {
    int n = FLAGS_n;
    AlignedTable<float> x(d * n);
    float_rand(x.data(), d * n, 12345);

    ScalarQuantizer sq(d, type);
    omp_set_num_threads(1);
    sq.train(n, x.data());

    std::vector<uint8_t> codes(sq.code_size * n);
    sq.compute_codes(x.data(), codes.data(), n);

    std::vector<float> decoded(d * n);

    for (auto _ : state) {
        sq.decode(codes.data(), decoded.data(), n);
        benchmark::DoNotOptimize(decoded[0]);
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * d * sizeof(float));
}

static void bench_sq_distance(
        benchmark::State& state,
        ScalarQuantizer::QuantizerType type,
        MetricType metric,
        int d) {
    int n = FLAGS_n;
    AlignedTable<float> x(d * n);
    float_rand(x.data(), d * n, 12345);

    ScalarQuantizer sq(d, type);
    omp_set_num_threads(1);
    sq.train(n, x.data());

    std::vector<uint8_t> codes(sq.code_size * n);
    sq.compute_codes(x.data(), codes.data(), n);

    std::unique_ptr<ScalarQuantizer::SQDistanceComputer> dc(
            sq.get_distance_computer(metric));

    // For each of the n vectors used as query, set_query then n
    // query_to_code calls — n*n distances per iteration.
    float result = 0;
    for (auto _ : state) {
        for (int i = 0; i < n; i++) {
            dc->set_query(x.data() + (size_t)i * d);
            for (int j = 0; j < n; j++) {
                result += dc->query_to_code(codes.data() + j * sq.code_size);
            }
        }
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * n * n);
}

// Accuracy diagnostic: reports reconstruction error and
// encode idempotence (ndiff between codes and re-encoded codes) as counters.
// The timed body is trivial — the interesting output is the counters.
static void bench_sq_accuracy(
        benchmark::State& state,
        ScalarQuantizer::QuantizerType type,
        int d) {
    int n = FLAGS_n;
    AlignedTable<float> x(d * n);
    float_rand(x.data(), d * n, 12345);

    ScalarQuantizer sq(d, type);
    omp_set_num_threads(1);
    sq.train(n, x.data());

    size_t code_size = sq.code_size;
    std::vector<uint8_t> codes(code_size * n);
    sq.compute_codes(x.data(), codes.data(), n);

    std::vector<float> x2(d * n);
    sq.decode(codes.data(), x2.data(), n);

    // Re-encode the decoded vectors to measure idempotence.
    std::vector<uint8_t> codes2(code_size * n);
    sq.compute_codes(x2.data(), codes2.data(), n);
    size_t ndiff = 0;
    for (size_t i = 0; i < codes.size(); i++) {
        if (codes[i] != codes2[i]) {
            ndiff++;
        }
    }

    double recons_error = fvec_L2sqr(x.data(), x2.data(), (size_t)n * d) / n;
    for (auto _ : state) {
        // keep the harness happy; the diagnostic is precomputed above.
        benchmark::DoNotOptimize(recons_error);
    }
    state.counters["code_size"] = sq.code_size;
    state.counters["sql2_recons_error"] = recons_error;
    state.counters["ndiff_for_idempotence"] = ndiff;
    state.counters["d"] = d;
}

// SIFT1M variant of the distance benchmark: train on the dataset's train
// vectors, encode a subset of the base vectors and scan them with a real
// query vector.
static void bench_sq_distance_dataset(
        benchmark::State& state,
        ScalarQuantizer::QuantizerType type,
        MetricType metric,
        const float* xt,
        int nt,
        const float* xb,
        int n,
        const float* query,
        int d) {
    ScalarQuantizer sq(d, type);
    omp_set_num_threads(1);
    sq.train(nt, xt);

    std::vector<uint8_t> codes(sq.code_size * n);
    sq.compute_codes(xb, codes.data(), n);

    std::unique_ptr<ScalarQuantizer::SQDistanceComputer> dc(
            sq.get_distance_computer(metric));
    dc->set_query(query);

    float result = 0;
    for (auto _ : state) {
        for (int i = 0; i < n; i++) {
            result += dc->query_to_code(codes.data() + i * sq.code_size);
        }
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.counters["d"] = d;
    state.counters["dataset"] = 1;
}

// SIFT1M variant of the accuracy diagnostic: reconstruction error and
// encode idempotence on a subset of the base vectors, trained on sift.xt.
static void bench_sq_accuracy_dataset(
        benchmark::State& state,
        ScalarQuantizer::QuantizerType type,
        const float* xt,
        int nt,
        const float* xb,
        int n,
        int d) {
    ScalarQuantizer sq(d, type);
    omp_set_num_threads(1);
    sq.train(nt, xt);

    size_t code_size = sq.code_size;
    std::vector<uint8_t> codes(code_size * n);
    sq.compute_codes(xb, codes.data(), n);

    std::vector<float> x2((size_t)d * n);
    sq.decode(codes.data(), x2.data(), n);

    // Re-encode the decoded vectors to measure idempotence.
    std::vector<uint8_t> codes2(code_size * n);
    sq.compute_codes(x2.data(), codes2.data(), n);
    size_t ndiff = 0;
    for (size_t i = 0; i < codes.size(); i++) {
        if (codes[i] != codes2[i]) {
            ndiff++;
        }
    }

    double recons_error = fvec_L2sqr(xb, x2.data(), (size_t)n * d) / n;
    for (auto _ : state) {
        // keep the harness happy; the diagnostic is precomputed above.
        benchmark::DoNotOptimize(recons_error);
    }
    state.counters["code_size"] = sq.code_size;
    state.counters["sql2_recons_error"] = recons_error;
    state.counters["ndiff_for_idempotence"] = ndiff;
    state.counters["d"] = d;
    state.counters["dataset"] = 1;
}

int main(int argc, char** argv) {
    benchmarks::bench_init(
            &argc,
            &argv,
            "scalar quantizer encode/decode/distance for all quantizer types "
            "(8/6/4-bit, uniform, fp16, bf16, 8-bit direct)",
            "--type=QT_8bit,QT_fp16 --d=128 --benchmark_filter='encode/.*'");

    struct SQDef {
        const char* name;
        ScalarQuantizer::QuantizerType type;
    };

    std::vector<SQDef> all_sq_types = {
            {"QT_8bit", ScalarQuantizer::QT_8bit},
            {"QT_4bit", ScalarQuantizer::QT_4bit},
            {"QT_6bit", ScalarQuantizer::QT_6bit},
            {"QT_8bit_uniform", ScalarQuantizer::QT_8bit_uniform},
            {"QT_4bit_uniform", ScalarQuantizer::QT_4bit_uniform},
            {"QT_fp16", ScalarQuantizer::QT_fp16},
            {"QT_bf16", ScalarQuantizer::QT_bf16},
            {"QT_8bit_direct", ScalarQuantizer::QT_8bit_direct},
            {"QT_8bit_direct_signed", ScalarQuantizer::QT_8bit_direct_signed},
    };

    std::vector<std::string> default_type_names;
    for (const auto& sq : all_sq_types) {
        default_type_names.push_back(sq.name);
    }
    std::vector<std::string> type_names =
            benchmarks::str_list(FLAGS_type, default_type_names);

    std::vector<SQDef> sq_types;
    for (const auto& name : type_names) {
        bool found = false;
        for (const auto& sq : all_sq_types) {
            if (name == sq.name) {
                sq_types.push_back(sq);
                found = true;
                break;
            }
        }
        if (!found) {
            fprintf(stderr,
                    "bench_codec_sq: unknown quantizer type '%s'\n",
                    name.c_str());
            return 1;
        }
    }

    std::vector<int> dims = benchmarks::int_list(FLAGS_d, {128, 768});

    struct MetricDef {
        const char* name;
        MetricType metric;
    };
    std::vector<MetricDef> metrics = {
            {"L2", METRIC_L2},
            {"IP", METRIC_INNER_PRODUCT},
    };

    for (auto& sq : sq_types) {
        for (int d : dims) {
            std::string suffix =
                    std::string("/") + sq.name + "/d:" + std::to_string(d);

            auto* b = benchmark::RegisterBenchmark(
                    ("encode" + suffix).c_str(), bench_sq_encode, sq.type, d);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);

            b = benchmark::RegisterBenchmark(
                    ("decode" + suffix).c_str(), bench_sq_decode, sq.type, d);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);

            for (const auto& m : metrics) {
                std::string dist_name = std::string("distance/metric:") +
                        m.name + "/" + sq.name + "/d:" + std::to_string(d);
                b = benchmark::RegisterBenchmark(
                        dist_name.c_str(),
                        bench_sq_distance,
                        sq.type,
                        m.metric,
                        d);
                if (FLAGS_iterations > 0)
                    b->Iterations(FLAGS_iterations);
            }

            b = benchmark::RegisterBenchmark(
                    ("accuracy" + suffix).c_str(),
                    bench_sq_accuracy,
                    sq.type,
                    d);
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
        // Scan/accuracy over a subset of the base vectors (matches the
        // synthetic n).
        int sn = sift.nb < FLAGS_n ? (int)sift.nb : (int)FLAGS_n;

        // Keep the dataset cross-product small: a few representative types
        // (overridable with --type).
        std::vector<std::string> sift_type_names = benchmarks::str_list(
                FLAGS_type, {"QT_8bit", "QT_4bit", "QT_fp16"});
        std::vector<SQDef> sift_sq_types;
        for (const auto& name : sift_type_names) {
            for (const auto& sq : all_sq_types) {
                if (name == sq.name) {
                    sift_sq_types.push_back(sq);
                    break;
                }
            }
        }

        for (auto& sq : sift_sq_types) {
            for (const auto& m : metrics) {
                std::string dist_name =
                        ds + "/distance/metric:" + m.name + "/" + sq.name;
                auto* b = benchmark::RegisterBenchmark(
                        dist_name.c_str(),
                        bench_sq_distance_dataset,
                        sq.type,
                        m.metric,
                        sift.xt.data(),
                        snt,
                        sift.xb.data(),
                        sn,
                        sift.xq.data(),
                        sd);
                if (FLAGS_iterations > 0)
                    b->Iterations(FLAGS_iterations);
            }

            auto* b = benchmark::RegisterBenchmark(
                    (ds + "/accuracy/" + sq.name).c_str(),
                    bench_sq_accuracy_dataset,
                    sq.type,
                    sift.xt.data(),
                    snt,
                    sift.xb.data(),
                    sn,
                    sd);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);
        }
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
