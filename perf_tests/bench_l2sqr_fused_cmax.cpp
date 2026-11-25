/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Benchmark for the fused L2 + top-1 kernel
// (faiss/utils/distances_fused/rvv.cpp,
// exhaustive_L2sqr_fused_cmax<RISCV_RVV>).
//
// The measured path is the production one: knn_L2sqr with k == 1 dispatches
// to Top1BlockResultHandler and tries the fused kernel before falling back to
// the BLAS tiles kernel. At the RISC-V baseline the fused kernel is a stub
// returning false, so the number below is the cost of the BLAS fallback;
// once the RVV kernel is implemented the same call gets faster.
//
// All parameters (d / nx / ny / threads / repeats) come from flags.

#include <chrono>
#include <cstdio>
#include <vector>

#include <benchmark/benchmark.h>
#include <gflags/gflags.h>
#include <omp.h>

#include <faiss/utils/distances.h>
#include <faiss/utils/distances_fused/distances_fused.h>
#include <faiss/utils/random.h>

using namespace faiss;

DEFINE_uint32(d, 12, "dimension of the vectors");
DEFINE_uint32(nx, 16384, "number of query vectors");
DEFINE_uint32(ny, 1024, "number of database vectors");
DEFINE_uint32(threads, 1, "OpenMP threads");
DEFINE_uint32(repeats, 5, "timed repetitions inside one benchmark run");

static void bench_knn_top1(benchmark::State& state) {
    const size_t d = FLAGS_d;
    const size_t nx = FLAGS_nx;
    const size_t ny = FLAGS_ny;

    std::vector<float> x(nx * d);
    std::vector<float> y(ny * d);
    std::vector<float> y_norms(ny);
    float_rand(x.data(), x.size(), 12345);
    float_rand(y.data(), y.size(), 67890);
    fvec_norms_L2sqr(y_norms.data(), y.data(), d, ny);

    std::vector<float> distances(nx);
    std::vector<int64_t> ids(nx);

    // Whether the fused kernel handles this dimensionality at all.
    int fused_applied = 0;
#if defined(__riscv) && defined(__riscv_vector)
    {
        // Probe on a scratch result set so the timed loop stays untouched.
        std::vector<float> dis_probe(nx);
        std::vector<int64_t> ids_probe(nx);
        Top1BlockResultHandler<CMax<float, int64_t>> probe(
                nx, dis_probe.data(), ids_probe.data());
        fused_applied =
                exhaustive_L2sqr_fused_cmax<SIMDLevel::RISCV_RVV>(
                        x.data(), y.data(), d, nx, ny, probe, y_norms.data())
                ? 1
                : 0;
    }
#endif
    std::printf("FUSED_APPLIED=%d\n", fused_applied);
    if (nx * d < static_cast<size_t>(distance_compute_blas_threshold)) {
        std::printf(
                "WARNING: nx*d=%zu < blas threshold=%d, the measured path is "
                "the sequential one, not the fused/BLAS kernel\n",
                nx * d,
                distance_compute_blas_threshold);
    }

    const uint32_t repeats = FLAGS_repeats > 0 ? FLAGS_repeats : 1;
    double total_ns = 0.0;
    size_t timed = 0;
    for (auto _ : state) {
        for (uint32_t r = 0; r < repeats; r++) {
            auto t0 = std::chrono::steady_clock::now();
            knn_L2sqr(
                    x.data(),
                    y.data(),
                    d,
                    nx,
                    ny,
                    size_t(1),
                    distances.data(),
                    ids.data(),
                    y_norms.data(),
                    nullptr);
            auto t1 = std::chrono::steady_clock::now();
            total_ns +=
                    std::chrono::duration<double, std::nano>(t1 - t0).count();
            timed++;
            benchmark::ClobberMemory();
        }
    }

    state.PauseTiming();
    const double ns_per_call = timed > 0 ? total_ns / double(timed) : 0.0;
    std::printf("FUSED_CMAX_NS=%.1f\n", ns_per_call);
    state.counters["nx"] = double(nx);
    state.counters["ny"] = double(ny);
    state.counters["d"] = double(d);
    state.ResumeTiming();
}

// Direct call into the per-SIMD specialization, so the RVV kernel cost can be
// read without the knn_L2sqr dispatch overhead. Informational, and RISC-V
// only: the specialization named here is the RVV one.
#if defined(__riscv) && defined(__riscv_vector)
static void bench_fused_only(benchmark::State& state) {
    const size_t d = FLAGS_d;
    const size_t nx = FLAGS_nx;
    const size_t ny = FLAGS_ny;

    std::vector<float> x(nx * d);
    std::vector<float> y(ny * d);
    std::vector<float> y_norms(ny);
    float_rand(x.data(), x.size(), 12345);
    float_rand(y.data(), y.size(), 67890);
    fvec_norms_L2sqr(y_norms.data(), y.data(), d, ny);

    std::vector<float> distances(nx);
    std::vector<int64_t> ids(nx);
    Top1BlockResultHandler<CMax<float, int64_t>> res(
            nx, distances.data(), ids.data());

    double total_ns = 0.0;
    size_t timed = 0;
    for (auto _ : state) {
        auto t0 = std::chrono::steady_clock::now();
        exhaustive_L2sqr_fused_cmax<SIMDLevel::RISCV_RVV>(
                x.data(), y.data(), d, nx, ny, res, y_norms.data());
        auto t1 = std::chrono::steady_clock::now();
        total_ns += std::chrono::duration<double, std::nano>(t1 - t0).count();
        timed++;
        benchmark::ClobberMemory();
    }
    state.PauseTiming();
    std::printf(
            "FUSED_ONLY_NS=%.1f\n", timed > 0 ? total_ns / double(timed) : 0.0);
    state.ResumeTiming();
}
#endif // defined(__riscv) && defined(__riscv_vector)

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);
    gflags::AllowCommandLineReparsing();
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    omp_set_num_threads(int(FLAGS_threads));

    benchmark::RegisterBenchmark("BM_KnnTop1", bench_knn_top1)
            ->Iterations(1)
            ->Unit(benchmark::kNanosecond);
#if defined(__riscv) && defined(__riscv_vector)
    benchmark::RegisterBenchmark("BM_FusedOnly", bench_fused_only)
            ->Iterations(1)
            ->Unit(benchmark::kNanosecond);
#endif

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
