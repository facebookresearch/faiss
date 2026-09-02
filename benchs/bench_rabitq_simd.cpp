/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <benchmark/benchmark.h>
#include <faiss/utils/AlignedTable.h>
#include <faiss/utils/rabitq_simd.h>
#include <faiss/utils/random.h>
#include <faiss/utils/simd_levels.h>

namespace faiss {

template <SIMDLevel SL>
bool check_simd_available(benchmark::State& state) {
    if constexpr (SL != SIMDLevel::NONE) {
        if (!SIMDConfig::is_simd_level_available(SL)) {
            state.SkipWithError(
                    "requested SIMD level is unavailable on this CPU");
            return false;
        }
    }
    return true;
}

const auto& randomData() {
    static auto data = [] {
        AlignedTable<uint8_t> x(10 << 20); // 10 MiB
        byte_rand(x.data(), x.size(), 456);
        return x;
    }();
    return data;
}

void bench_rabitq_generic(benchmark::State& state, auto distFn) {
    uint8_t qb = state.range(0);
    size_t d = state.range(1);
    size_t size = (d + 7) / 8;

    auto& x = randomData();
    AlignedTable<uint8_t> q(qb * size);
    byte_rand(q.data(), q.size(), 123);

    size_t n = x.size() / size;

    uint64_t sum = 0;
    size_t r = 0;
    for (auto _ : state) {
        ++r;
        for (size_t i = 0; i < n; ++i) {
            sum += distFn(q.data(), x.data() + i * size, size, qb);
        }
        benchmark::DoNotOptimize(sum);
    }
    state.SetItemsProcessed(n * r);
    state.SetBytesProcessed(r * x.size());
}

void bench_rabitq_sum(benchmark::State& state) {
    bench_rabitq_generic(
            state,
            [](const uint8_t*, const uint8_t* x, size_t size, size_t)
                    -> int64_t { return rabitq::popcount(x, size); });
}

void bench_rabitq_and_dot_product(benchmark::State& state) {
    bench_rabitq_generic(
            state,
            [](const uint8_t* q, const uint8_t* x, size_t size, size_t qb)
                    -> int64_t {
                return rabitq::bitwise_and_dot_product(q, x, size, qb);
            });
}

void bench_rabitq_xor_dot_product(benchmark::State& state) {
    bench_rabitq_generic(
            state,
            [](const uint8_t* q, const uint8_t* x, size_t size, size_t qb)
                    -> int64_t {
                return rabitq::bitwise_xor_dot_product(q, x, size, qb);
            });
}

void bench_rabitq_and_dot_product_with_sum(benchmark::State& state) {
    bench_rabitq_generic(
            state,
            [](const uint8_t* q, const uint8_t* x, size_t size, size_t qb)
                    -> int64_t {
                auto sum_q = rabitq::popcount(x, size);
                auto dp = rabitq::bitwise_and_dot_product(q, x, size, qb);
                // Synthetic operation using both inputs for benchmarking.
                return sum_q + dp;
            });
}

void bench_rabitq_and_dot_product_with_sum_fused(benchmark::State& state) {
    bench_rabitq_generic(
            state,
            [](const uint8_t* q, const uint8_t* x, size_t size, size_t qb)
                    -> int64_t {
                auto result = rabitq::bitwise_and_dot_product_with_popcount(
                        q, x, size, qb);
                // Synthetic operation using both inputs for benchmarking.
                return result.popcount + result.dot_product;
            });
}

template <SIMDLevel SL>
void bench_rabitq_rearrange_impl(benchmark::State& state) {
    if (!check_simd_available<SL>(state)) {
        return;
    }
    size_t qb = state.range(0);
    size_t d = state.range(1);
    size_t offset = (d + 7) / 8;

    AlignedTable<uint8_t> rotated_qq(d);
    byte_rand(rotated_qq.data(), rotated_qq.size(), 10996);
    // codes are qb-bit values, one per dimension
    const uint8_t code_mask = static_cast<uint8_t>((1u << qb) - 1);
    for (size_t i = 0; i < d; i++) {
        rotated_qq[i] &= code_mask;
    }
    AlignedTable<uint8_t> out(offset * qb);

    for (auto _ : state) {
        rabitq::rearrange_bit_planes<SL>(rotated_qq.data(), d, qb, out.data());
        benchmark::DoNotOptimize(out.data());
    }
    state.SetItemsProcessed(state.iterations() * d);
}

void bench_rabitq_rearrange_scalar(benchmark::State& state) {
    bench_rabitq_rearrange_impl<SIMDLevel::NONE>(state);
}

#if defined(__x86_64__)
void bench_rabitq_rearrange_avx2(benchmark::State& state) {
    bench_rabitq_rearrange_impl<SIMDLevel::AVX2>(state);
}
#endif

template <SIMDLevel SL>
void bench_rabitq_selected_float_sum(benchmark::State& state) {
    if (!check_simd_available<SL>(state)) {
        return;
    }
    const size_t d = state.range(0);
    AlignedTable<uint8_t> sign_bits((d + 7) / 8);
    AlignedTable<float> values(d);
    byte_rand(sign_bits.data(), sign_bits.size(), 721);
    float_rand(values.data(), values.size(), 812);

    for (auto _ : state) {
        float result = rabitq::selected_float_sum<SL>(
                sign_bits.data(), values.data(), d);
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations());
}

template <SIMDLevel SL>
void bench_rabitq_multibit_inner_product(benchmark::State& state) {
    if (!check_simd_available<SL>(state)) {
        return;
    }
    const size_t ex_bits = state.range(0);
    const size_t d = state.range(1);
    AlignedTable<uint8_t> sign_bits((d + 7) / 8);
    // Padding keeps the benchmark input valid for kernels that load a complete
    // machine word at the end of a bit-packed code.
    AlignedTable<uint8_t> extra_code((d * ex_bits + 7) / 8 + 8);
    AlignedTable<float> query(d);
    byte_rand(sign_bits.data(), sign_bits.size(), 913);
    byte_rand(extra_code.data(), extra_code.size(), 114);
    float_rand(query.data(), query.size(), 315);

    for (auto _ : state) {
        float result = rabitq::multibit::compute_inner_product<SL>(
                sign_bits.data(),
                extra_code.data(),
                query.data(),
                d,
                ex_bits,
                -3.25f);
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations());
}

const std::vector<int64_t> qbs{1, 2, 4, 8};
const std::vector<int64_t> dims{64, 100, 256, 512, 1000, 1024, 3072};
const std::vector<int64_t> scorer_dims{128, 768, 1537};

BENCHMARK(bench_rabitq_sum)->ArgsProduct({{0}, dims})->ArgNames({"qb", "d"});
BENCHMARK(bench_rabitq_and_dot_product)
        ->ArgsProduct({qbs, dims})
        ->ArgNames({"qb", "d"});
BENCHMARK(bench_rabitq_xor_dot_product)
        ->ArgsProduct({qbs, dims})
        ->ArgNames({"qb", "d"});
BENCHMARK(bench_rabitq_and_dot_product_with_sum)
        ->ArgsProduct({qbs, dims})
        ->ArgNames({"qb", "d"});
BENCHMARK(bench_rabitq_and_dot_product_with_sum_fused)
        ->ArgsProduct({qbs, dims})
        ->ArgNames({"qb", "d"});
BENCHMARK(bench_rabitq_rearrange_scalar)
        ->ArgsProduct({qbs, dims})
        ->ArgNames({"qb", "d"});
#if defined(__x86_64__)
BENCHMARK(bench_rabitq_rearrange_avx2)
        ->ArgsProduct({qbs, dims})
        ->ArgNames({"qb", "d"});

BENCHMARK_TEMPLATE(bench_rabitq_selected_float_sum, SIMDLevel::NONE)
        ->ArgsProduct({scorer_dims})
        ->ArgNames({"d"});
BENCHMARK_TEMPLATE(bench_rabitq_selected_float_sum, SIMDLevel::AVX2)
        ->ArgsProduct({scorer_dims})
        ->ArgNames({"d"});
BENCHMARK_TEMPLATE(bench_rabitq_selected_float_sum, SIMDLevel::AVX512)
        ->ArgsProduct({scorer_dims})
        ->ArgNames({"d"});
BENCHMARK_TEMPLATE(bench_rabitq_multibit_inner_product, SIMDLevel::NONE)
        ->ArgsProduct({{1, 3, 7}, scorer_dims})
        ->ArgNames({"extra_bits", "d"});
BENCHMARK_TEMPLATE(bench_rabitq_multibit_inner_product, SIMDLevel::AVX2)
        ->ArgsProduct({{1, 3, 7}, scorer_dims})
        ->ArgNames({"extra_bits", "d"});
BENCHMARK_TEMPLATE(bench_rabitq_multibit_inner_product, SIMDLevel::AVX512)
        ->ArgsProduct({{1, 3, 7}, scorer_dims})
        ->ArgNames({"extra_bits", "d"});
#endif

} // namespace faiss

BENCHMARK_MAIN();
