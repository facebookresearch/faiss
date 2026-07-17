/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Additive Quantizer / Codec Benchmarks
// Benchmarks ResidualQuantizer encode and decode operations.
//
// Design notes:
//   * Encodes a 1000-vector subset (throughput-equivalent per vector).
//   * The ils_iters, beam-LUT (rq_lut), OPQ and PRQ-beam variants are
//     covered by the python port (benchs/python/bench_codec_quantizers.py)
//     only.

#include <gflags/gflags.h>
#include <omp.h>

#include <benchmark/benchmark.h>
#include <faiss/IndexAdditiveQuantizer.h>
#include <faiss/impl/AdditiveQuantizer.h>
#include <faiss/impl/LocalSearchQuantizer.h>
#include <faiss/impl/ProductAdditiveQuantizer.h>
#include <faiss/impl/ResidualQuantizer.h>
#include <faiss/utils/AlignedTable.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>

#include "bench_cli_utils.h"
#include "bench_dataset_utils.h"

using namespace faiss;

DEFINE_uint32(iterations, 0, "iterations (0 = auto)");
BENCH_DEFINE_DATASET_FILE_FLAGS();
DEFINE_string(d, "", "comma-separated vector dimensions (default: 128,256)");
DEFINE_string(
        M,
        "",
        "comma-separated residual quantizer step counts (default: 8,16)");
DEFINE_string(
        nbits,
        "",
        "comma-separated bits per quantizer step (default: 8,4; "
        "prq_encode group: 8,4; lsq_encode/plsq_encode group: 4 — "
        "LSQ/PLSQ training at nbits=8 takes 1-3+ minutes single-threaded)");
DEFINE_string(
        beam,
        "",
        "comma-separated RQ max beam sizes for the rq_encode_beam "
        "benchmark (default: 1,2,4,8,16,32)");

static void bench_rq_encode(
        benchmark::State& state,
        int d,
        int M,
        int nbits,
        int n) {
    int nt = 10000;
    AlignedTable<float> xt(d * nt);
    AlignedTable<float> xb(d * n);
    float_rand(xt.data(), d * nt, 12345);
    float_rand(xb.data(), d * n, 54321);

    ResidualQuantizer rq(d, M, nbits);
    rq.verbose = false;
    // Pin beam size before training for reproducible quantizer state
    rq.max_beam_size = 30;
    omp_set_num_threads(1);
    rq.train(nt, xt.data());

    std::vector<uint8_t> codes(rq.code_size * n);

    // Warmup
    rq.compute_codes(xb.data(), codes.data(), n);

    for (auto _ : state) {
        rq.compute_codes(xb.data(), codes.data(), n);
        benchmark::DoNotOptimize(codes[0]);
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.counters["d"] = d;
    state.counters["M"] = M;
    state.counters["nbits"] = nbits;
    state.counters["n"] = n;
    state.counters["code_size"] = rq.code_size;
}

static void bench_rq_decode(
        benchmark::State& state,
        int d,
        int M,
        int nbits,
        int n) {
    int nt = 10000;
    AlignedTable<float> xt(d * nt);
    AlignedTable<float> xb(d * n);
    float_rand(xt.data(), d * nt, 12345);
    float_rand(xb.data(), d * n, 54321);

    ResidualQuantizer rq(d, M, nbits);
    rq.verbose = false;
    // Pin beam size before training for reproducible quantizer state
    rq.max_beam_size = 30;
    omp_set_num_threads(1);
    rq.train(nt, xt.data());

    std::vector<uint8_t> codes(rq.code_size * n);
    rq.compute_codes(xb.data(), codes.data(), n);

    std::vector<float> decoded(d * n);

    for (auto _ : state) {
        rq.decode(codes.data(), decoded.data(), n);
        benchmark::DoNotOptimize(decoded[0]);
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.counters["d"] = d;
    state.counters["M"] = M;
    state.counters["nbits"] = nbits;
    state.counters["n"] = n;
    state.counters["code_size"] = rq.code_size;
}

static void bench_rq_reconstruction_error(
        benchmark::State& state,
        int d,
        int M,
        int nbits,
        int n) {
    int nt = 10000;
    AlignedTable<float> xt(d * nt);
    AlignedTable<float> xb(d * n);
    float_rand(xt.data(), d * nt, 12345);
    float_rand(xb.data(), d * n, 54321);

    ResidualQuantizer rq(d, M, nbits);
    rq.verbose = false;
    // Pin beam size before training for reproducible quantizer state
    rq.max_beam_size = 30;
    omp_set_num_threads(1);
    rq.train(nt, xt.data());

    std::vector<uint8_t> codes(rq.code_size * n);
    std::vector<float> decoded(d * n);

    for (auto _ : state) {
        rq.compute_codes(xb.data(), codes.data(), n);
        rq.decode(codes.data(), decoded.data(), n);
        float err = fvec_L2sqr(xb.data(), decoded.data(), (size_t)n * d) / n;
        benchmark::DoNotOptimize(err);
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.counters["d"] = d;
    state.counters["M"] = M;
    state.counters["nbits"] = nbits;
    state.counters["n"] = n;
}

// Additive-quantizer encoders: train on 10000 vectors, time
// compute_codes of 1000 vectors. Shared timing loop for LSQ/PRQ/PLSQ.
static void run_aq_encode(
        benchmark::State& state,
        AdditiveQuantizer& aq,
        int d,
        int M,
        int nbits,
        int n) {
    int nt = 10000;
    AlignedTable<float> xt(d * nt);
    AlignedTable<float> xb(d * n);
    float_rand(xt.data(), d * nt, 12345);
    float_rand(xb.data(), d * n, 54321);

    aq.verbose = false;
    omp_set_num_threads(1);
    aq.train(nt, xt.data());

    std::vector<uint8_t> codes(aq.code_size * n);

    // Warmup
    aq.compute_codes(xb.data(), codes.data(), n);

    for (auto _ : state) {
        aq.compute_codes(xb.data(), codes.data(), n);
        benchmark::DoNotOptimize(codes[0]);
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.counters["d"] = d;
    state.counters["M"] = M;
    state.counters["nbits"] = nbits;
    state.counters["n"] = n;
    state.counters["code_size"] = aq.code_size;
}

static void bench_lsq_encode(
        benchmark::State& state,
        int d,
        int M,
        int nbits,
        int n) {
    LocalSearchQuantizer lsq(d, M, nbits);
    run_aq_encode(state, lsq, d, M, nbits, n);
}

static void bench_prq_encode(
        benchmark::State& state,
        int d,
        int M,
        int nbits,
        int n) {
    ProductResidualQuantizer prq(d, 2, M / 2, nbits);
    run_aq_encode(state, prq, d, M, nbits, n);
}

static void bench_plsq_encode(
        benchmark::State& state,
        int d,
        int M,
        int nbits,
        int n) {
    ProductLocalSearchQuantizer plsq(d, 2, M / 2, nbits);
    run_aq_encode(state, plsq, d, M, nbits, n);
}

// RQ encode over max_beam_size (beam
// sweep): trained with beam pinned to 30, then re-encoded at each
// beam size.
static void bench_rq_encode_beam(
        benchmark::State& state,
        int d,
        int M,
        int nbits,
        int beam,
        int n) {
    int nt = 10000;
    AlignedTable<float> xt(d * nt);
    AlignedTable<float> xb(d * n);
    float_rand(xt.data(), d * nt, 12345);
    float_rand(xb.data(), d * n, 54321);

    ResidualQuantizer rq(d, M, nbits);
    rq.verbose = false;
    // Pin beam size before training for reproducible quantizer state
    rq.max_beam_size = 30;
    omp_set_num_threads(1);
    rq.train(nt, xt.data());
    rq.max_beam_size = beam;

    std::vector<uint8_t> codes(rq.code_size * n);

    // Warmup
    rq.compute_codes(xb.data(), codes.data(), n);

    for (auto _ : state) {
        rq.compute_codes(xb.data(), codes.data(), n);
        benchmark::DoNotOptimize(codes[0]);
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.counters["d"] = d;
    state.counters["M"] = M;
    state.counters["nbits"] = nbits;
    state.counters["beam"] = beam;
    state.counters["n"] = n;
}

// SIFT1M variants: train the ResidualQuantizer on the dataset's train
// vectors and encode/decode a subset of the base vectors.
static void bench_rq_encode_dataset(
        benchmark::State& state,
        const float* xt,
        int nt,
        const float* xb,
        int n,
        int d,
        int M,
        int nbits) {
    ResidualQuantizer rq(d, M, nbits);
    rq.verbose = false;
    // Pin beam size before training for reproducible quantizer state
    rq.max_beam_size = 30;
    omp_set_num_threads(1);
    rq.train(nt, xt);

    std::vector<uint8_t> codes(rq.code_size * n);

    // Warmup
    rq.compute_codes(xb, codes.data(), n);

    for (auto _ : state) {
        rq.compute_codes(xb, codes.data(), n);
        benchmark::DoNotOptimize(codes[0]);
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.counters["d"] = d;
    state.counters["M"] = M;
    state.counters["nbits"] = nbits;
    state.counters["n"] = n;
    state.counters["code_size"] = rq.code_size;
    state.counters["dataset"] = 1;
}

static void bench_rq_decode_dataset(
        benchmark::State& state,
        const float* xt,
        int nt,
        const float* xb,
        int n,
        int d,
        int M,
        int nbits) {
    ResidualQuantizer rq(d, M, nbits);
    rq.verbose = false;
    // Pin beam size before training for reproducible quantizer state
    rq.max_beam_size = 30;
    omp_set_num_threads(1);
    rq.train(nt, xt);

    std::vector<uint8_t> codes(rq.code_size * n);
    rq.compute_codes(xb, codes.data(), n);

    std::vector<float> decoded((size_t)d * n);

    for (auto _ : state) {
        rq.decode(codes.data(), decoded.data(), n);
        benchmark::DoNotOptimize(decoded[0]);
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.counters["d"] = d;
    state.counters["M"] = M;
    state.counters["nbits"] = nbits;
    state.counters["n"] = n;
    state.counters["code_size"] = rq.code_size;
    state.counters["dataset"] = 1;
}

static void bench_rq_reconstruction_error_dataset(
        benchmark::State& state,
        const float* xt,
        int nt,
        const float* xb,
        int n,
        int d,
        int M,
        int nbits) {
    ResidualQuantizer rq(d, M, nbits);
    rq.verbose = false;
    // Pin beam size before training for reproducible quantizer state
    rq.max_beam_size = 30;
    omp_set_num_threads(1);
    rq.train(nt, xt);

    std::vector<uint8_t> codes(rq.code_size * n);
    std::vector<float> decoded((size_t)d * n);

    for (auto _ : state) {
        rq.compute_codes(xb, codes.data(), n);
        rq.decode(codes.data(), decoded.data(), n);
        float err = fvec_L2sqr(xb, decoded.data(), (size_t)n * d) / n;
        benchmark::DoNotOptimize(err);
    }
    state.SetItemsProcessed(state.iterations() * n);
    state.counters["d"] = d;
    state.counters["M"] = M;
    state.counters["nbits"] = nbits;
    state.counters["n"] = n;
    state.counters["dataset"] = 1;
}

int main(int argc, char** argv) {
    benchmarks::bench_init(
            &argc,
            &argv,
            "ResidualQuantizer codec: encode, decode and reconstruction "
            "error across M x nbits configurations",
            "--d=128 --M=8 --nbits=8 --benchmark_filter='rq_encode/.*'");

    int n = 1000;
    std::vector<int> dims = benchmarks::int_list(FLAGS_d, {128, 256});
    // M x nbits combinations (total bits = M * nbits)
    std::vector<int> Ms = benchmarks::int_list(FLAGS_M, {8, 16});
    std::vector<int> nbits_list = benchmarks::int_list(FLAGS_nbits, {8, 4});

    for (int d : dims) {
        for (int nbits : nbits_list) {
            for (int M : Ms) {
                std::string suffix = "/d:" + std::to_string(d) +
                        "/M:" + std::to_string(M) +
                        "/nbits:" + std::to_string(nbits);

                auto* b = benchmark::RegisterBenchmark(
                        ("rq_encode" + suffix).c_str(),
                        bench_rq_encode,
                        d,
                        M,
                        nbits,
                        n);
                if (FLAGS_iterations > 0)
                    b->Iterations(FLAGS_iterations);

                b = benchmark::RegisterBenchmark(
                        ("rq_decode" + suffix).c_str(),
                        bench_rq_decode,
                        d,
                        M,
                        nbits,
                        n);
                if (FLAGS_iterations > 0)
                    b->Iterations(FLAGS_iterations);

                b = benchmark::RegisterBenchmark(
                        ("rq_recons_error" + suffix).c_str(),
                        bench_rq_reconstruction_error,
                        d,
                        M,
                        nbits,
                        n);
                if (FLAGS_iterations > 0)
                    b->Iterations(FLAGS_iterations);
            }
        }
    }

    // Additive-quantizer encoders: d=128, M sweep; LSQ/PLSQ default to
    // nbits=4 (nbits=8 training takes minutes), PRQ to nbits={8,4}.
    std::vector<int> dims_aq = benchmarks::int_list(FLAGS_d, {128});
    std::vector<int> Ms_aq = benchmarks::int_list(FLAGS_M, {8, 16});
    std::vector<int> nbits_lsq = benchmarks::int_list(FLAGS_nbits, {4});
    std::vector<int> nbits_prq = benchmarks::int_list(FLAGS_nbits, {8, 4});

    for (int d : dims_aq) {
        for (int M : Ms_aq) {
            for (int nbits : nbits_lsq) {
                std::string suffix = "/d:" + std::to_string(d) +
                        "/M:" + std::to_string(M) +
                        "/nbits:" + std::to_string(nbits);

                auto* b = benchmark::RegisterBenchmark(
                        ("lsq_encode" + suffix).c_str(),
                        bench_lsq_encode,
                        d,
                        M,
                        nbits,
                        n);
                if (FLAGS_iterations > 0)
                    b->Iterations(FLAGS_iterations);

                b = benchmark::RegisterBenchmark(
                        ("plsq_encode" + suffix).c_str(),
                        bench_plsq_encode,
                        d,
                        M,
                        nbits,
                        n);
                if (FLAGS_iterations > 0)
                    b->Iterations(FLAGS_iterations);
            }

            for (int nbits : nbits_prq) {
                std::string suffix = "/d:" + std::to_string(d) +
                        "/M:" + std::to_string(M) +
                        "/nbits:" + std::to_string(nbits);

                auto* b = benchmark::RegisterBenchmark(
                        ("prq_encode" + suffix).c_str(),
                        bench_prq_encode,
                        d,
                        M,
                        nbits,
                        n);
                if (FLAGS_iterations > 0)
                    b->Iterations(FLAGS_iterations);
            }
        }
    }

    // RQ encode over max_beam_size: d=128, nbits=8 fixed.
    std::vector<int> beams =
            benchmarks::int_list(FLAGS_beam, {1, 2, 4, 8, 16, 32});
    for (int d : dims_aq) {
        for (int M : Ms_aq) {
            for (int beam : beams) {
                std::string suffix = "/d:" + std::to_string(d) +
                        "/M:" + std::to_string(M) +
                        "/beam:" + std::to_string(beam);

                auto* b = benchmark::RegisterBenchmark(
                        ("rq_encode_beam" + suffix).c_str(),
                        bench_rq_encode_beam,
                        d,
                        M,
                        8,
                        beam,
                        n);
                if (FLAGS_iterations > 0)
                    b->Iterations(FLAGS_iterations);
            }
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
        // Cap training at the synthetic scale (10000): RQ training cost
        // grows steeply with the train-set size and each registration
        // retrains, so the full 100K learn set would dominate the run.
        int snt = (int)sift.nt < 10000 ? (int)sift.nt : 10000;
        // Encode/decode a subset of the base vectors (matches the synthetic
        // n; codec throughput does not need all 1M).
        int sn = (int)sift.nb < n ? (int)sift.nb : n;

        // RQ codec on real data: M sweep, nbits=8 (train on sift.xt)
        std::vector<int> sift_Ms = benchmarks::int_list(FLAGS_M, {8, 16});
        std::vector<int> sift_nbits = benchmarks::int_list(FLAGS_nbits, {8});
        for (int M : sift_Ms) {
            for (int nbits : sift_nbits) {
                std::string suffix = "/M:" + std::to_string(M) +
                        "/nbits:" + std::to_string(nbits);

                auto* b = benchmark::RegisterBenchmark(
                        (ds + "/rq_encode" + suffix).c_str(),
                        bench_rq_encode_dataset,
                        sift.xt.data(),
                        snt,
                        sift.xb.data(),
                        sn,
                        sd,
                        M,
                        nbits);
                if (FLAGS_iterations > 0)
                    b->Iterations(FLAGS_iterations);

                b = benchmark::RegisterBenchmark(
                        (ds + "/rq_decode" + suffix).c_str(),
                        bench_rq_decode_dataset,
                        sift.xt.data(),
                        snt,
                        sift.xb.data(),
                        sn,
                        sd,
                        M,
                        nbits);
                if (FLAGS_iterations > 0)
                    b->Iterations(FLAGS_iterations);

                b = benchmark::RegisterBenchmark(
                        (ds + "/rq_recons_error" + suffix).c_str(),
                        bench_rq_reconstruction_error_dataset,
                        sift.xt.data(),
                        snt,
                        sift.xb.data(),
                        sn,
                        sd,
                        M,
                        nbits);
                if (FLAGS_iterations > 0)
                    b->Iterations(FLAGS_iterations);
            }
        }
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
