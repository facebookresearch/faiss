/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Microbenchmarks for the SuperKMeans block_l2 kernel and the "multi-block
// fusion" sweep that reuses x across several centroids.
//
// block_l2 itself: one PDX block against one centroid (the metric used by the
// block_l2 optimization experiment).
//
// Sweep: the inner structure of SuperKMeans::run_iter_pruned — for one x row,
// accumulate squared L2 against k centroids over n_blocks PDX blocks. The
// "baseline" order is centroid-outer / block-inner (what the caller does now);
// "fused" is block-outer with 4 centroids batched so each x chunk is loaded
// once and reused across 4 y chunks (halves the load instruction count).

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include <faiss/impl/simd_dispatch.h>
#include <faiss/utils/simd_impl/super_kmeans_kernels.h>
#include <faiss/utils/simd_levels.h>

#ifdef COMPILE_SIMD_RISCV_RVV
#include <riscv_vector.h>
#endif

namespace {

std::vector<float> make_data(int n, std::uint32_t seed) {
    std::vector<float> v(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        std::uint32_t h = static_cast<std::uint32_t>(i) * 2654435761u + seed;
        v[static_cast<std::size_t>(i)] =
                static_cast<float>(h % 2000) / 1000.0f - 1.0f;
    }
    return v;
}

// Scalar reference on the same board, so speedups are board-relative.
// VLEN-sensitivity probe: plain LMUL variants of the same kernel structure.
// m1/m2/m4 VLMAX = LMUL*VLEN/32 lanes; the optimal LMUL depends on VLEN.
#ifdef COMPILE_SIMD_RISCV_RVV
template <int LMUL>
__attribute__((noinline))
float raw_block_l2(const float* x, const float* y, int n);

template <>
__attribute__((noinline))
float raw_block_l2<1>(const float* x, const float* y, int n) {
    const size_t vl = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t a0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t a1 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    int m = 0;
    const int ivl = static_cast<int>(vl);
    for (; m + 2 * ivl <= n; m += 2 * ivl) {
        vfloat32m1_t x0 = __riscv_vle32_v_f32m1(x + m, vl);
        vfloat32m1_t d0 = __riscv_vfsub_vv_f32m1(
                x0, __riscv_vle32_v_f32m1(y + m, vl), vl);
        a0 = __riscv_vfmacc_vv_f32m1(a0, d0, d0, vl);
        vfloat32m1_t x1 = __riscv_vle32_v_f32m1(x + m + ivl, vl);
        vfloat32m1_t d1 = __riscv_vfsub_vv_f32m1(
                x1, __riscv_vle32_v_f32m1(y + m + ivl, vl), vl);
        a1 = __riscv_vfmacc_vv_f32m1(a1, d1, d1, vl);
    }
    for (; m + ivl <= n; m += ivl) {
        vfloat32m1_t d0 = __riscv_vfsub_vv_f32m1(
                __riscv_vle32_v_f32m1(x + m, vl),
                __riscv_vle32_v_f32m1(y + m, vl), vl);
        a0 = __riscv_vfmacc_vv_f32m1(a0, d0, d0, vl);
    }
    if (m < n) {
        const size_t vlt = __riscv_vsetvl_e32m1(static_cast<size_t>(n - m));
        vfloat32m1_t d0 = __riscv_vfsub_vv_f32m1(
                __riscv_vle32_v_f32m1(x + m, vlt),
                __riscv_vle32_v_f32m1(y + m, vlt), vlt);
        a0 = __riscv_vfmacc_vv_f32m1(a0, d0, d0, vlt);
    }
    vfloat32m1_t acc = __riscv_vfadd_vv_f32m1(a0, a1, vl);
    vfloat32m1_t seed = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    return __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredusum_vs_f32m1_f32m1(acc, seed, vl));
}

template <>
__attribute__((noinline))
float raw_block_l2<2>(const float* x, const float* y, int n) {
    const size_t vl = __riscv_vsetvlmax_e32m2();
    vfloat32m2_t a0 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    vfloat32m2_t a1 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    int m = 0;
    const int ivl = static_cast<int>(vl);
    for (; m + 2 * ivl <= n; m += 2 * ivl) {
        vfloat32m2_t x0 = __riscv_vle32_v_f32m2(x + m, vl);
        vfloat32m2_t d0 = __riscv_vfsub_vv_f32m2(
                x0, __riscv_vle32_v_f32m2(y + m, vl), vl);
        a0 = __riscv_vfmacc_vv_f32m2(a0, d0, d0, vl);
        vfloat32m2_t x1 = __riscv_vle32_v_f32m2(x + m + ivl, vl);
        vfloat32m2_t d1 = __riscv_vfsub_vv_f32m2(
                x1, __riscv_vle32_v_f32m2(y + m + ivl, vl), vl);
        a1 = __riscv_vfmacc_vv_f32m2(a1, d1, d1, vl);
    }
    for (; m + ivl <= n; m += ivl) {
        vfloat32m2_t d0 = __riscv_vfsub_vv_f32m2(
                __riscv_vle32_v_f32m2(x + m, vl),
                __riscv_vle32_v_f32m2(y + m, vl), vl);
        a0 = __riscv_vfmacc_vv_f32m2(a0, d0, d0, vl);
    }
    if (m < n) {
        const size_t vlt = __riscv_vsetvl_e32m2(static_cast<size_t>(n - m));
        vfloat32m2_t d0 = __riscv_vfsub_vv_f32m2(
                __riscv_vle32_v_f32m2(x + m, vlt),
                __riscv_vle32_v_f32m2(y + m, vlt), vlt);
        a0 = __riscv_vfmacc_vv_f32m2(a0, d0, d0, vlt);
    }
    vfloat32m2_t acc = __riscv_vfadd_vv_f32m2(a0, a1, vl);
    vfloat32m1_t seed = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    return __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredusum_vs_f32m2_f32m1(acc, seed, vl));
}

template <>
__attribute__((noinline))
float raw_block_l2<4>(const float* x, const float* y, int n) {
    const size_t vl = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t a0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    vfloat32m4_t a1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    int m = 0;
    const int ivl = static_cast<int>(vl);
    for (; m + 2 * ivl <= n; m += 2 * ivl) {
        vfloat32m4_t x0 = __riscv_vle32_v_f32m4(x + m, vl);
        vfloat32m4_t d0 = __riscv_vfsub_vv_f32m4(
                x0, __riscv_vle32_v_f32m4(y + m, vl), vl);
        a0 = __riscv_vfmacc_vv_f32m4(a0, d0, d0, vl);
        vfloat32m4_t x1 = __riscv_vle32_v_f32m4(x + m + ivl, vl);
        vfloat32m4_t d1 = __riscv_vfsub_vv_f32m4(
                x1, __riscv_vle32_v_f32m4(y + m + ivl, vl), vl);
        a1 = __riscv_vfmacc_vv_f32m4(a1, d1, d1, vl);
    }
    for (; m + ivl <= n; m += ivl) {
        vfloat32m4_t d0 = __riscv_vfsub_vv_f32m4(
                __riscv_vle32_v_f32m4(x + m, vl),
                __riscv_vle32_v_f32m4(y + m, vl), vl);
        a0 = __riscv_vfmacc_vv_f32m4(a0, d0, d0, vl);
    }
    if (m < n) {
        const size_t vlt = __riscv_vsetvl_e32m4(static_cast<size_t>(n - m));
        vfloat32m4_t d0 = __riscv_vfsub_vv_f32m4(
                __riscv_vle32_v_f32m4(x + m, vlt),
                __riscv_vle32_v_f32m4(y + m, vlt), vlt);
        a0 = __riscv_vfmacc_vv_f32m4(a0, d0, d0, vlt);
    }
    vfloat32m4_t acc = __riscv_vfadd_vv_f32m4(a0, a1, vl);
    vfloat32m1_t seed = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    return __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredusum_vs_f32m4_f32m1(acc, seed, vl));
}

template <int LMUL>
void BM_RawLmul(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const std::vector<float> x = make_data(n, 12345u);
    const std::vector<float> y = make_data(n, 67890u);
    float sink = 0.0f;
    for (auto _ : state) {
        sink += raw_block_l2<LMUL>(x.data(), y.data(), n);
        benchmark::DoNotOptimize(sink);
    }
    state.SetItemsProcessed(state.iterations() * n);
}
#endif

void BM_BlockL2Scalar(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const std::vector<float> x = make_data(n, 12345u);
    const std::vector<float> y = make_data(n, 67890u);

    float sink = 0.0f;
    for (auto _ : state) {
        sink += faiss::detail::block_l2<faiss::SIMDLevel::NONE>(
                x.data(), y.data(), n);
        benchmark::DoNotOptimize(sink);
    }
    state.SetItemsProcessed(state.iterations() * n);
}

void BM_BlockL2(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const std::vector<float> x = make_data(n, 12345u);
    const std::vector<float> y = make_data(n, 67890u);

    float sink = 0.0f;
    faiss::with_simd_level([&]<faiss::SIMDLevel SL>() {
        for (auto _ : state) {
            sink += faiss::detail::block_l2<SL>(x.data(), y.data(), n);
            benchmark::DoNotOptimize(sink);
        }
    });

    state.SetItemsProcessed(state.iterations() * n);
}

// ---------------------------------------------------------------------------
// Sweep: one x row vs k centroids over n_blocks blocks of n_blk dims.
// Y is PDX layout: block b, centroid j at offset (b*k + j) * n_blk.
// ---------------------------------------------------------------------------

// Current caller order: for each centroid, walk all blocks calling block_l2.
void sweep_baseline(
        const float* x,
        const float* Ypdx,
        int n_blk,
        int n_blocks,
        int k,
        float* out) {
    faiss::with_simd_level([&]<faiss::SIMDLevel SL>() {
        for (int j = 0; j < k; ++j) {
            float s = 0.0f;
            for (int b = 0; b < n_blocks; ++b) {
                s += faiss::detail::block_l2<SL>(
                        x + static_cast<std::size_t>(b) * n_blk,
                        Ypdx + (static_cast<std::size_t>(b) * k + j) * n_blk,
                        n_blk);
            }
            out[j] += s;
        }
    });
}

#ifdef COMPILE_SIMD_RISCV_RVV

inline float reduce_m1(vfloat32m1_t v, size_t vl) {
    vfloat32m1_t seed = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    return __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredusum_vs_f32m1_f32m1(v, seed, vl));
}

// Fused order: block-outer, centroids batched by 4 so one x chunk is loaded
// once and reused against four y chunks.
void sweep_fused4(
        const float* x,
        const float* Ypdx,
        int n_blk,
        int n_blocks,
        int k,
        float* out) {
    const std::size_t vl = __riscv_vsetvlmax_e32m1();
    const int ivl = static_cast<int>(vl);
    for (int b = 0; b < n_blocks; ++b) {
        const float* xb = x + static_cast<std::size_t>(b) * n_blk;
        const float* yb = Ypdx + static_cast<std::size_t>(b) * k * n_blk;
        int j = 0;
        for (; j + 4 <= k; j += 4) {
            const float* y0p = yb + static_cast<std::size_t>(j + 0) * n_blk;
            const float* y1p = yb + static_cast<std::size_t>(j + 1) * n_blk;
            const float* y2p = yb + static_cast<std::size_t>(j + 2) * n_blk;
            const float* y3p = yb + static_cast<std::size_t>(j + 3) * n_blk;
            vfloat32m1_t a0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t a1 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t a2 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t a3 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            int m = 0;
            for (; m + ivl <= n_blk; m += ivl) {
                vfloat32m1_t xv = __riscv_vle32_v_f32m1(xb + m, vl);
                vfloat32m1_t d0 =
                        __riscv_vfsub_vv_f32m1(xv,
                                               __riscv_vle32_v_f32m1(y0p + m, vl),
                                               vl);
                a0 = __riscv_vfmacc_vv_f32m1(a0, d0, d0, vl);
                vfloat32m1_t d1 =
                        __riscv_vfsub_vv_f32m1(xv,
                                               __riscv_vle32_v_f32m1(y1p + m, vl),
                                               vl);
                a1 = __riscv_vfmacc_vv_f32m1(a1, d1, d1, vl);
                vfloat32m1_t d2 =
                        __riscv_vfsub_vv_f32m1(xv,
                                               __riscv_vle32_v_f32m1(y2p + m, vl),
                                               vl);
                a2 = __riscv_vfmacc_vv_f32m1(a2, d2, d2, vl);
                vfloat32m1_t d3 =
                        __riscv_vfsub_vv_f32m1(xv,
                                               __riscv_vle32_v_f32m1(y3p + m, vl),
                                               vl);
                a3 = __riscv_vfmacc_vv_f32m1(a3, d3, d3, vl);
            }
            if (m < n_blk) {
                const size_t vlt =
                        __riscv_vsetvl_e32m1(static_cast<size_t>(n_blk - m));
                vfloat32m1_t xv = __riscv_vle32_v_f32m1(xb + m, vlt);
                vfloat32m1_t d0 =
                        __riscv_vfsub_vv_f32m1(xv,
                                               __riscv_vle32_v_f32m1(y0p + m, vlt),
                                               vlt);
                a0 = __riscv_vfmacc_vv_f32m1(a0, d0, d0, vlt);
                vfloat32m1_t d1 =
                        __riscv_vfsub_vv_f32m1(xv,
                                               __riscv_vle32_v_f32m1(y1p + m, vlt),
                                               vlt);
                a1 = __riscv_vfmacc_vv_f32m1(a1, d1, d1, vlt);
                vfloat32m1_t d2 =
                        __riscv_vfsub_vv_f32m1(xv,
                                               __riscv_vle32_v_f32m1(y2p + m, vlt),
                                               vlt);
                a2 = __riscv_vfmacc_vv_f32m1(a2, d2, d2, vlt);
                vfloat32m1_t d3 =
                        __riscv_vfsub_vv_f32m1(xv,
                                               __riscv_vle32_v_f32m1(y3p + m, vlt),
                                               vlt);
                a3 = __riscv_vfmacc_vv_f32m1(a3, d3, d3, vlt);
            }
            out[j + 0] += reduce_m1(a0, vl);
            out[j + 1] += reduce_m1(a1, vl);
            out[j + 2] += reduce_m1(a2, vl);
            out[j + 3] += reduce_m1(a3, vl);
        }
        for (; j < k; ++j) {
            out[j] += faiss::detail::block_l2<faiss::SIMDLevel::RISCV_RVV>(
                    xb, yb + static_cast<std::size_t>(j) * n_blk, n_blk);
        }
    }
}
#endif // COMPILE_SIMD_RISCV_RVV

struct SweepData {
    int n_blk, n_blocks, k;
    std::vector<float> x, y;
    std::vector<float> out;
    SweepData(int nb, int nblk, int kk)
            : n_blk(nb), n_blocks(nblk), k(kk) {
        x = make_data(n_blocks * n_blk, 1u);
        y = make_data(k * n_blocks * n_blk, 2u);
        out.assign(k, 0.0f);
    }
};

void BM_BlockL2Sweep_baseline(benchmark::State& state) {
    const int n_blk = static_cast<int>(state.range(0));
    const int k = static_cast<int>(state.range(1));
    const int n_blocks = static_cast<int>(state.range(2));
    SweepData d(n_blk, n_blocks, k);
    for (auto _ : state) {
        std::fill(d.out.begin(), d.out.end(), 0.0f);
        sweep_baseline(d.x.data(), d.y.data(), n_blk, n_blocks, k, d.out.data());
        benchmark::DoNotOptimize(d.out.data());
    }
    state.SetItemsProcessed(
            state.iterations() * static_cast<int64_t>(k) * n_blocks * n_blk);
}

#ifdef COMPILE_SIMD_RISCV_RVV
// The fused sweep is the RVV-only path (SuperKMeans selects it via if constexpr
// on SIMDLevel::RISCV_RVV). Register nothing elsewhere rather than silently
// benchmarking the baseline under a "fused" name.
void BM_BlockL2Sweep_fused(benchmark::State& state) {
    const int n_blk = static_cast<int>(state.range(0));
    const int k = static_cast<int>(state.range(1));
    const int n_blocks = static_cast<int>(state.range(2));
    SweepData d(n_blk, n_blocks, k);
    for (auto _ : state) {
        std::fill(d.out.begin(), d.out.end(), 0.0f);
        sweep_fused4(
                d.x.data(), d.y.data(), n_blk, n_blocks, k, d.out.data());
        benchmark::DoNotOptimize(d.out.data());
    }
    state.SetItemsProcessed(
            state.iterations() * static_cast<int64_t>(k) * n_blocks * n_blk);
}
#endif // COMPILE_SIMD_RISCV_RVV

} // namespace

BENCHMARK(BM_BlockL2Scalar)
        ->Arg(16)
        ->Arg(32)
        ->Arg(64)
        ->Arg(128)
        ->Arg(256)
        ->Arg(512);

BENCHMARK(BM_BlockL2)
        ->Arg(16)
        ->Arg(32)
        ->Arg(64)
        ->Arg(128)
        ->Arg(256)
        ->Arg(512);

#ifdef COMPILE_SIMD_RISCV_RVV
BENCHMARK_TEMPLATE(BM_RawLmul, 1)->Arg(64);
BENCHMARK_TEMPLATE(BM_RawLmul, 2)->Arg(64);
BENCHMARK_TEMPLATE(BM_RawLmul, 4)->Arg(64);
#endif

// {n_blk, k, n_blocks}
BENCHMARK(BM_BlockL2Sweep_baseline)->Args({64, 64, 6});
BENCHMARK(BM_BlockL2Sweep_baseline)->Args({64, 256, 6});
#ifdef COMPILE_SIMD_RISCV_RVV
BENCHMARK(BM_BlockL2Sweep_fused)->Args({64, 64, 6});
BENCHMARK(BM_BlockL2Sweep_fused)->Args({64, 256, 6});
#endif

BENCHMARK_MAIN();
