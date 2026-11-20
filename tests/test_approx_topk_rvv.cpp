/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Correctness tests for the RISC-V Vector (RVV) implementations of the
// approximate top-k kernels.
//
// Code under test:
//   - faiss/impl/approx_topk/rvv.cpp
//       HeapWithBucketsCMaxFloat<NBUCKETS, N, RISCV_RVV>::bs_addn
//       accum_tab_rvv_{store,add,finalize}, reached through
//       accum_and_*_tab<M, NK, RISCV_RVV>
//   - faiss/utils/approx_topk_hamming/approx_topk_hamming-rvv.h
//       HeapWithBucketsForHamming32<..., RISCV_RVV>
//   - faiss/utils/hamming_distance/hamming_computer-rvv.h
//       HammingComputer32_tpl<RISCV_RVV>::{hamming, hamming_batch}
//
// Every kernel is compared against its SIMDLevel::NONE scalar reference.
// The bucket scans reproduce the reference comparison chain exactly, so
// both heaps must match bit for bit (values *and* ids: the ids index into
// the distance array, so an equal value with a different id is a real
// divergence). The accum kernels add the same rows in the same order and
// are therefore also bit-identical.
//
// Independent oracles (std::min_element, sorted Hamming distances) check
// the cases where the bucketed result is exact, so a bug shared by both
// implementations cannot hide behind the parity checks.
//
// This TU is compiled only on RISC-V, with the same -march as the
// library's RVV sources (see tests/CMakeLists.txt).

#ifdef COMPILE_SIMD_RISCV_RVV

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include <faiss/impl/IDSelector.h>
#include <faiss/impl/ResidualQuantizer.h>
#include <faiss/impl/approx_topk/approx_topk.h>
#include <faiss/impl/approx_topk/rq_beam_search_tab-inl.h>
#include <faiss/impl/simdlib/simdlib.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/approx_topk_hamming/approx_topk_hamming-rvv.h>
#include <faiss/utils/approx_topk_hamming/approx_topk_hamming.h>
#include <faiss/utils/hamming.h>
#include <faiss/utils/hamming_distance/hamming_computer-generic.h>
#include <faiss/utils/hamming_distance/hamming_computer-rvv.h>
#include <faiss/utils/ordered_key_value.h>
#include <faiss/utils/random.h>
#include <faiss/utils/simd_levels.h>

// accum_and_*_tab<M, 4, RISCV_RVV> are explicitly instantiated in
// impl/approx_topk/rvv.cpp. Declare them extern so this TU links to that
// definition instead of emitting a second copy. NK values other than 4 are
// not instantiated in the library; the test TU includes the defining header
// and instantiates them locally, which is fine because NK does not change
// the native kernel.
namespace faiss {

#define FAISS_TEST_EXTERN_ACCUM_TAB(M)                                       \
    extern template void accum_and_store_tab<M, 4, SIMDLevel::RISCV_RVV>(    \
            size_t,                                                          \
            const float* __restrict,                                         \
            const uint64_t* __restrict,                                      \
            const int32_t* __restrict,                                       \
            size_t,                                                          \
            size_t,                                                          \
            size_t,                                                          \
            float* __restrict);                                              \
    extern template void accum_and_add_tab<M, 4, SIMDLevel::RISCV_RVV>(      \
            size_t,                                                          \
            const float* __restrict,                                         \
            const uint64_t* __restrict,                                      \
            const int32_t* __restrict,                                       \
            size_t,                                                          \
            size_t,                                                          \
            size_t,                                                          \
            float* __restrict);                                              \
    extern template void accum_and_finalize_tab<M, 4, SIMDLevel::RISCV_RVV>( \
            const float* __restrict,                                         \
            const uint64_t* __restrict,                                      \
            const int32_t* __restrict,                                       \
            size_t,                                                          \
            size_t,                                                          \
            size_t,                                                          \
            const float* __restrict,                                         \
            const float* __restrict,                                         \
            float* __restrict);

FAISS_TEST_EXTERN_ACCUM_TAB(1)
FAISS_TEST_EXTERN_ACCUM_TAB(2)
FAISS_TEST_EXTERN_ACCUM_TAB(3)
FAISS_TEST_EXTERN_ACCUM_TAB(4)
FAISS_TEST_EXTERN_ACCUM_TAB(5)
FAISS_TEST_EXTERN_ACCUM_TAB(6)
FAISS_TEST_EXTERN_ACCUM_TAB(7)
FAISS_TEST_EXTERN_ACCUM_TAB(8)

#undef FAISS_TEST_EXTERN_ACCUM_TAB

} // namespace faiss

// The level selector used by hamming_impl.h must keep RVV (its kernel is
// native) and fold the levels that have no bucket kernel.
static_assert(
        faiss::approx_topk_hamming_level_selector<
                faiss::SIMDLevel::RISCV_RVV>::value ==
        faiss::SIMDLevel::RISCV_RVV);
static_assert(
        faiss::approx_topk_hamming_level_selector<
                faiss::SIMDLevel::AVX512>::value == faiss::SIMDLevel::AVX2);
static_assert(
        faiss::approx_topk_hamming_level_selector<
                faiss::SIMDLevel::AVX512_SPR>::value == faiss::SIMDLevel::AVX2);
static_assert(
        faiss::approx_topk_hamming_level_selector<
                faiss::SIMDLevel::ARM_SVE>::value ==
        faiss::SIMDLevel::ARM_NEON);

namespace {

// ApproxTopK_mode_t is declared at global scope (approx_topk.h), next to the
// faiss namespace rather than inside it.
using ::ApproxTopK_mode_t;
using faiss::SIMDLevel;

constexpr SIMDLevel kRVV = SIMDLevel::RISCV_RVV;
constexpr SIMDLevel kNone = SIMDLevel::NONE;

// Non-negative, right-skewed values, like the RQ distance tables these
// kernels see in production.
std::vector<float> random_distances(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::exponential_distribution<float> dist(1.0f);
    std::vector<float> values(n);
    for (auto& v : values) {
        v = dist(rng);
    }
    return values;
}

template <typename T>
std::vector<T> random_bytes(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<T> bytes(n);
    for (auto& b : bytes) {
        b = static_cast<T>(rng());
    }
    return bytes;
}

// ---------------------------------------------------------------------------
// Float bucket scan (HeapWithBucketsCMaxFloat)
// ---------------------------------------------------------------------------

// Run the RVV and the NONE bucket scans on the same input and compare the
// reordered heaps element by element. k == 1 and single-element-per-bucket
// shapes are additionally checked against an exact reference, because then
// the approximate scan is not allowed to lose any candidate.
template <uint32_t NBUCKETS, uint32_t N>
void expect_bs_addn_parity(
        uint32_t beam_size,
        uint32_t n_per_beam,
        uint32_t k,
        const std::vector<float>& distances,
        bool check_exact = true) {
    using C = faiss::CMax<float, int>;

    ASSERT_EQ(distances.size(), size_t(beam_size) * n_per_beam);
    ASSERT_GE(k, 1u);
    ASSERT_LE(k, beam_size * n_per_beam);

    std::vector<float> rvv_val(k, C::neutral());
    std::vector<int32_t> rvv_ids(k, -1);
    std::vector<float> ref_val(k, C::neutral());
    std::vector<int32_t> ref_ids(k, -1);

    faiss::HeapWithBucketsCMaxFloat<NBUCKETS, N, kRVV>::bs_addn(
            beam_size,
            n_per_beam,
            distances.data(),
            k,
            rvv_val.data(),
            rvv_ids.data());
    faiss::HeapWithBucketsCMaxFloat<NBUCKETS, N, kNone>::bs_addn(
            beam_size,
            n_per_beam,
            distances.data(),
            k,
            ref_val.data(),
            ref_ids.data());

    faiss::heap_reorder<C>(k, rvv_val.data(), rvv_ids.data());
    faiss::heap_reorder<C>(k, ref_val.data(), ref_ids.data());

    EXPECT_EQ(rvv_val, ref_val);
    EXPECT_EQ(rvv_ids, ref_ids);

    if (!check_exact || rvv_val != ref_val || rvv_val[0] == C::neutral()) {
        return;
    }

    // k == 1 is exact: the heap must hold the global minimum. The id is
    // not necessarily the first occurrence (buckets are merged by bucket,
    // not by position), so only require that it points at an element
    // holding the minimum.
    if (k == 1) {
        const float best =
                *std::min_element(distances.begin(), distances.end());
        EXPECT_EQ(rvv_val[0], best);
        ASSERT_GE(rvv_ids[0], 0);
        EXPECT_LT(size_t(rvv_ids[0]), distances.size());
        EXPECT_EQ(distances[size_t(rvv_ids[0])], best);
    }

    // One element per bucket: every candidate is still in the heap, so the
    // result must be exactly the k smallest values.
    if (beam_size == 1 && n_per_beam == NBUCKETS && k <= NBUCKETS) {
        std::vector<float> sorted = distances;
        std::sort(sorted.begin(), sorted.end());
        std::vector<float> got(rvv_val.begin(), rvv_val.end());
        std::sort(got.begin(), got.end());
        EXPECT_TRUE(std::equal(got.begin(), got.end(), sorted.begin()));
    }
}

TEST(ApproxTopKRVV, BsAddnParityAllInstantiations) {
    // Every instantiation emitted by impl/approx_topk/rvv.cpp, with
    // k == 1 and k == 8 (the special-cased heap paths), other k values
    // (generic heap path), full blocks, and leftover elements.
    uint32_t seed = 100;

    auto run = [&]<uint32_t NB, uint32_t ND>(
                       uint32_t beam, uint32_t n, uint32_t k) {
        SCOPED_TRACE(
                "NB=" + std::to_string(NB) + " N=" + std::to_string(ND) +
                " beam=" + std::to_string(beam) + " n=" + std::to_string(n) +
                " k=" + std::to_string(k));
        expect_bs_addn_parity<NB, ND>(
                beam, n, k, random_distances(size_t(beam) * n, seed++));
    };

    run.template operator()<8, 3>(1, 256, 16);
    run.template operator()<8, 3>(4, 259, 8);
    run.template operator()<8, 3>(1, 8, 1);
    run.template operator()<8, 3>(1, 9, 1);
    run.template operator()<8, 3>(2, 255, 24);

    run.template operator()<8, 2>(1, 256, 16);
    run.template operator()<8, 2>(2, 128, 1);
    run.template operator()<8, 2>(2, 131, 4);
    run.template operator()<8, 2>(1, 64, 8);

    run.template operator()<16, 2>(1, 256, 16);
    run.template operator()<16, 2>(4, 259, 8);
    run.template operator()<16, 2>(1, 16, 1);
    run.template operator()<16, 2>(3, 33, 4);

    run.template operator()<16, 1>(1, 256, 8);
    run.template operator()<16, 1>(4, 66, 4);
    run.template operator()<16, 1>(1, 17, 1);

    run.template operator()<32, 2>(1, 256, 16);
    run.template operator()<32, 2>(4, 259, 8);
    run.template operator()<32, 2>(1, 32, 1);
    run.template operator()<32, 2>(3, 97, 4);
}

TEST(ApproxTopKRVV, BsAddnLeftoverSweep) {
    // Remainder shapes of the scan loop: no leftover, one leftover, and
    // several partial blocks.
    uint32_t seed = 1000;
    for (uint32_t n_per_beam : {32u, 33u, 63u, 64u, 65u, 96u, 97u, 130u}) {
        SCOPED_TRACE("n_per_beam=" + std::to_string(n_per_beam));
        const auto distances = random_distances(n_per_beam, seed++);
        expect_bs_addn_parity<8, 2>(1, n_per_beam, 8, distances);
        expect_bs_addn_parity<16, 2>(1, n_per_beam, 8, distances);
        expect_bs_addn_parity<32, 2>(1, n_per_beam, 8, distances);
    }
}

TEST(ApproxTopKRVV, BsAddnTies) {
    // Duplicated values exercise the documented tie rules: the bucket scan
    // replaces only on a strictly smaller candidate, and the merge keeps
    // the element already in the heap on equal values.
    uint32_t seed = 2000;
    for (uint32_t variant = 0; variant < 2; variant++) {
        SCOPED_TRACE("variant=" + std::to_string(variant));
        auto make = [&](size_t n) {
            auto d = random_distances(n, seed++);
            if (variant == 0) {
                for (size_t i = 0; i + 1 < n; i += 2) {
                    d[i + 1] = d[i];
                }
            } else {
                std::fill(d.begin(), d.end(), 0.5f);
            }
            return d;
        };
        expect_bs_addn_parity<8, 2>(1, 256, 16, make(256));
        expect_bs_addn_parity<8, 3>(4, 64, 8, make(4 * 64));
        expect_bs_addn_parity<16, 2>(2, 128, 16, make(2 * 128));
        expect_bs_addn_parity<32, 2>(1, 128, 32, make(128));
    }
}

TEST(ApproxTopKRVV, BsAddnBoundaryValues) {
    // 0, -0, +inf, FLT_MAX and denormals all take part in the comparison
    // chain. +inf and FLT_MAX lose against every finite value; the tests
    // pin down that the RVV scan orders them exactly like NONE.
    const float pattern[] = {
            0.0f,
            1.0f,
            -0.0f,
            std::numeric_limits<float>::max(),
            std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::denorm_min(),
            3.5f,
            2.5f};

    std::vector<float> distances(2 * 64);
    for (size_t i = 0; i < distances.size(); i++) {
        distances[i] = pattern[i % 8];
    }

    expect_bs_addn_parity<8, 2>(2, 64, 6, distances);
    expect_bs_addn_parity<16, 2>(2, 64, 8, distances);
}

TEST(ApproxTopKRVV, BsAddnNaN) {
    // NaN distances are outside the production contract (callers feed L2
    // residuals and cross-norm sums). Pin down that neither path crashes
    // and that each path is deterministic when run twice; the two paths
    // are deliberately not compared across each other.
    using C = faiss::CMax<float, int>;
    const uint32_t beam = 1, n_per_beam = 64, k = 4;

    std::mt19937 rng(3000);
    std::exponential_distribution<float> dist(1.0f);
    std::vector<float> distances(size_t(beam) * n_per_beam);
    for (size_t i = 0; i < distances.size(); i++) {
        distances[i] = (i % 16 == 7) ? std::nanf("") : dist(rng) * 0.01f;
    }

    auto run = [&]<SIMDLevel SL>() {
        std::vector<float> val(k, C::neutral());
        std::vector<int32_t> ids(k, -1);
        faiss::HeapWithBucketsCMaxFloat<8, 2, SL>::bs_addn(
                beam, n_per_beam, distances.data(), k, val.data(), ids.data());
        return std::make_pair(val, ids);
    };

    const auto rvv1 = run.template operator()<kRVV>();
    const auto rvv2 = run.template operator()<kRVV>();
    const auto ref1 = run.template operator()<kNone>();
    const auto ref2 = run.template operator()<kNone>();

    // memcmp compares NaN bit patterns as well.
    EXPECT_EQ(
            std::memcmp(
                    rvv1.first.data(), rvv2.first.data(), k * sizeof(float)),
            0);
    EXPECT_EQ(rvv1.second, rvv2.second);
    EXPECT_EQ(
            std::memcmp(
                    ref1.first.data(), ref2.first.data(), k * sizeof(float)),
            0);
    EXPECT_EQ(ref1.second, ref2.second);
}

// ---------------------------------------------------------------------------
// accum_and_*_tab
// ---------------------------------------------------------------------------

struct AccumTestSetup {
    size_t M = 0;
    size_t K = 0;
    size_t ldc = 0;
    size_t beam = 0;
    std::vector<float> cross_norms; // (M * K rows) x ldc
    std::vector<uint64_t> offsets;
    std::vector<int32_t> codes; // beam x M
    std::vector<float> distances_i;
    std::vector<float> cd_common;
};

AccumTestSetup make_accum_setup(
        size_t M,
        size_t K,
        size_t ldc,
        size_t beam,
        uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);

    AccumTestSetup s;
    s.M = M;
    s.K = K;
    s.ldc = ldc;
    s.beam = beam;

    s.cross_norms.resize(M * K * ldc);
    for (auto& v : s.cross_norms) {
        v = u(rng);
    }
    s.offsets.resize(M);
    for (size_t i = 0; i < M; i++) {
        s.offsets[i] = i * K;
    }
    s.codes.resize(beam * M);
    for (auto& c : s.codes) {
        c = int32_t(rng() % K);
    }
    s.distances_i.resize(beam);
    for (auto& v : s.distances_i) {
        v = u(rng);
    }
    s.cd_common.resize(K);
    for (auto& v : s.cd_common) {
        v = u(rng);
    }
    return s;
}

template <size_t M, size_t NK = 4>
void expect_accum_tab_parity(size_t K, size_t ldc, size_t beam, uint32_t seed) {
    const AccumTestSetup s = make_accum_setup(M, K, ldc, beam, seed);

    std::vector<float> rvv_accum(K, 0.0f);
    std::vector<float> ref_accum(K, 0.0f);

    faiss::accum_and_store_tab<M, NK, kRVV>(
            M,
            s.cross_norms.data(),
            s.offsets.data(),
            s.codes.data(),
            0,
            s.ldc,
            K,
            rvv_accum.data());
    faiss::accum_and_store_tab<M, NK, kNone>(
            M,
            s.cross_norms.data(),
            s.offsets.data(),
            s.codes.data(),
            0,
            s.ldc,
            K,
            ref_accum.data());
    EXPECT_EQ(rvv_accum, ref_accum) << "store";

    for (size_t b = 1; b < beam; b++) {
        faiss::accum_and_add_tab<M, NK, kRVV>(
                M,
                s.cross_norms.data(),
                s.offsets.data(),
                s.codes.data(),
                b,
                s.ldc,
                K,
                rvv_accum.data());
        faiss::accum_and_add_tab<M, NK, kNone>(
                M,
                s.cross_norms.data(),
                s.offsets.data(),
                s.codes.data(),
                b,
                s.ldc,
                K,
                ref_accum.data());
        EXPECT_EQ(rvv_accum, ref_accum) << "add b=" << b;
    }

    std::vector<float> rvv_out(beam * K, 0x1p-80f);
    std::vector<float> ref_out(beam * K, 0x1p-80f);
    for (size_t b = 0; b < beam; b++) {
        faiss::accum_and_finalize_tab<M, NK, kRVV>(
                s.cross_norms.data(),
                s.offsets.data(),
                s.codes.data(),
                b,
                s.ldc,
                K,
                s.distances_i.data(),
                s.cd_common.data(),
                rvv_out.data());
        faiss::accum_and_finalize_tab<M, NK, kNone>(
                s.cross_norms.data(),
                s.offsets.data(),
                s.codes.data(),
                b,
                s.ldc,
                K,
                s.distances_i.data(),
                s.cd_common.data(),
                ref_out.data());
        EXPECT_EQ(rvv_out, ref_out) << "finalize b=" << b;
    }

    // Plain scalar re-derivation, guarding against a bug shared by both
    // implementations. Not bit-exact by construction (the vector body
    // contracts 2 * r + c into an FMA while this form rounds separately),
    // so allow a handful of ULPs.
    for (size_t b = 0; b < beam; b++) {
        for (size_t k = 0; k < K; k++) {
            float reg = 0;
            for (size_t ij = 0; ij < M; ij++) {
                const size_t code = size_t(s.codes[b * M + ij]);
                reg += s.cross_norms[(s.offsets[ij] + code) * ldc + k];
            }
            const float expect = s.distances_i[b] + s.cd_common[k] + 2 * reg;
            const float got = rvv_out[b * K + k];
            const float scale = std::max(
                    std::abs(expect),
                    std::max(std::abs(got), std::abs(s.cd_common[k])));
            const float tol =
                    (M + 8) * scale * std::numeric_limits<float>::epsilon();
            EXPECT_LE(std::abs(got - expect), tol)
                    << "M=" << M << " K=" << K << " ldc=" << ldc << " b=" << b
                    << " k=" << k;
        }
    }
}

TEST(ApproxTopKRVV, AccumTabParity) {
    // M covers the instantiation set of rvv.cpp (1..8).
    expect_accum_tab_parity<1>(64, 64, 3, 10);
    expect_accum_tab_parity<2>(64, 64, 3, 11);
    expect_accum_tab_parity<3>(64, 64, 3, 12);
    expect_accum_tab_parity<4>(64, 64, 3, 13);
    expect_accum_tab_parity<5>(64, 64, 3, 14);
    expect_accum_tab_parity<6>(64, 64, 3, 15);
    expect_accum_tab_parity<7>(64, 64, 3, 16);
    expect_accum_tab_parity<8>(64, 64, 3, 17);

    // K not a multiple of the vector length: the tail block is handled
    // with a shorter vl.
    expect_accum_tab_parity<1>(40, 40, 2, 20);
    expect_accum_tab_parity<2>(40, 40, 2, 21);
    expect_accum_tab_parity<4>(20, 20, 2, 22);
    expect_accum_tab_parity<8>(17, 17, 2, 23);
    expect_accum_tab_parity<3>(33, 33, 2, 24);

    // ldc > K: rows are strided.
    expect_accum_tab_parity<2>(64, 80, 2, 30);
    expect_accum_tab_parity<5>(48, 96, 3, 31);

    // NK only shapes the generic (NONE) blocking; the RVV kernel strip-mines
    // K independently of NK, so other values must give the same results.
    expect_accum_tab_parity<1, 1>(64, 64, 2, 40);
    expect_accum_tab_parity<3, 1>(40, 40, 2, 41);
    expect_accum_tab_parity<4, 8>(64, 64, 2, 42);
    expect_accum_tab_parity<8, 8>(33, 33, 2, 43);
}

// ---------------------------------------------------------------------------
// Hamming top-k
// ---------------------------------------------------------------------------

template <uint32_t NBUCKETS, uint32_t N, typename HC_RVV, typename HC_NONE>
void expect_hamming_topk_parity(
        int code_size,
        uint32_t beam_size,
        uint32_t n_per_beam,
        uint32_t k,
        bool use_sel,
        uint32_t seed) {
    using C = faiss::CMax<int, int64_t>;

    ASSERT_GE(k, 1u);
    ASSERT_LE(k, beam_size * n_per_beam);

    for (uint32_t trial = 0; trial < 4; trial++) {
        SCOPED_TRACE(
                "trial=" + std::to_string(trial) +
                " beam=" + std::to_string(beam_size) + " n_per_beam=" +
                std::to_string(n_per_beam) + " k=" + std::to_string(k) +
                " sel=" + std::to_string(int(use_sel)));

        const size_t n = size_t(beam_size) * n_per_beam;
        std::vector<uint8_t> query =
                random_bytes<uint8_t>(size_t(code_size), seed + trial * 2);
        std::vector<uint8_t> vectors = random_bytes<uint8_t>(
                n * size_t(code_size), seed + trial * 2 + 1);

        HC_RVV hc_rvv(query.data(), code_size);
        HC_NONE hc_none(query.data(), code_size);

        faiss::IDSelectorRange sel(0, int64_t(n) / 2);
        const faiss::IDSelector* selector = use_sel ? &sel : nullptr;

        std::vector<int> rvv_val(k, C::neutral());
        std::vector<int64_t> rvv_ids(k, -1);
        std::vector<int> ref_val(k, C::neutral());
        std::vector<int64_t> ref_ids(k, -1);

        faiss::HeapWithBucketsForHamming32<
                C,
                NBUCKETS,
                N,
                HC_RVV,
                SIMDLevel::RISCV_RVV>::
                bs_addn(beam_size,
                        n_per_beam,
                        hc_rvv,
                        vectors.data(),
                        k,
                        rvv_val.data(),
                        rvv_ids.data(),
                        selector);
        faiss::HeapWithBucketsForHamming32<
                C,
                NBUCKETS,
                N,
                HC_NONE,
                SIMDLevel::NONE>::
                bs_addn(beam_size,
                        n_per_beam,
                        hc_none,
                        vectors.data(),
                        k,
                        ref_val.data(),
                        ref_ids.data(),
                        selector);

        faiss::heap_reorder<C>(k, rvv_val.data(), rvv_ids.data());
        faiss::heap_reorder<C>(k, ref_val.data(), ref_ids.data());

        EXPECT_EQ(rvv_val, ref_val);
        EXPECT_EQ(rvv_ids, ref_ids);

        // Each returned id must reproduce its distance and respect the
        // selector; a missing beam offset or bucket fixup shows up here.
        // With a selector the bucket contributions can in principle run
        // dry, in which case a slot legitimately stays at -1.
        for (size_t i = 0; i < k; i++) {
            if (rvv_ids[i] < 0) {
                EXPECT_TRUE(use_sel);
                continue;
            }
            ASSERT_LT(size_t(rvv_ids[i]), n);
            if (use_sel) {
                EXPECT_TRUE(sel.is_member(rvv_ids[i]));
            }
            EXPECT_EQ(
                    hc_none.hamming(
                            vectors.data() + size_t(rvv_ids[i]) * code_size),
                    rvv_val[i]);
        }

        // One element per bucket and no selector: the scan must be exact.
        if (!use_sel && beam_size == 1 && n_per_beam == NBUCKETS) {
            std::vector<int> all(n_per_beam);
            for (uint32_t i = 0; i < n_per_beam; i++) {
                all[i] =
                        hc_none.hamming(vectors.data() + size_t(i) * code_size);
            }
            std::sort(all.begin(), all.end());
            std::vector<int> got(rvv_val.begin(), rvv_val.end());
            std::sort(got.begin(), got.end());
            EXPECT_TRUE(std::equal(got.begin(), got.end(), all.begin()));
        }
    }
}

TEST(ApproxTopKRVV, Hamming32TopKParity) {
    // The 32-byte computer uses the RVV hamming_batch() fast path (the
    // selector-free path only); keep a few shapes for each instantiation
    // used by hamming_impl.h.
    using HC_RVV = faiss::HammingComputer32_tpl<kRVV>;
    using HC_NONE = faiss::HammingComputer32_tpl<kNone>;

    expect_hamming_topk_parity<8, 3, HC_RVV, HC_NONE>(32, 1, 64, 8, false, 50);
    expect_hamming_topk_parity<8, 3, HC_RVV, HC_NONE>(32, 2, 67, 12, true, 51);
    expect_hamming_topk_parity<8, 2, HC_RVV, HC_NONE>(32, 1, 64, 8, false, 52);
    expect_hamming_topk_parity<8, 2, HC_RVV, HC_NONE>(
            32, 2, 131, 16, false, 53);

    expect_hamming_topk_parity<16, 2, HC_RVV, HC_NONE>(
            32, 1, 64, 16, false, 54);
    expect_hamming_topk_parity<16, 2, HC_RVV, HC_NONE>(32, 2, 65, 16, true, 55);
    expect_hamming_topk_parity<16, 2, HC_RVV, HC_NONE>(32, 1, 16, 8, false, 56);

    expect_hamming_topk_parity<32, 2, HC_RVV, HC_NONE>(
            32, 1, 128, 32, false, 57);
    expect_hamming_topk_parity<32, 2, HC_RVV, HC_NONE>(
            32, 2, 133, 32, true, 58);
    expect_hamming_topk_parity<32, 2, HC_RVV, HC_NONE>(
            32, 1, 32, 16, false, 59);
}

TEST(ApproxTopKRVV, HammingTopKParityOtherCodeSizes) {
    // The RVV bucket scan is independent of the Hamming computer; the
    // 16/20/64/Default computers take the per-code hamming() path because
    // only the 32-byte computer defines hamming_batch().
    expect_hamming_topk_parity<
            16,
            2,
            faiss::HammingComputer16_tpl<kRVV>,
            faiss::HammingComputer16_tpl<kNone>>(16, 2, 65, 16, false, 60);
    expect_hamming_topk_parity<
            16,
            2,
            faiss::HammingComputer16_tpl<kRVV>,
            faiss::HammingComputer16_tpl<kNone>>(16, 1, 32, 8, true, 61);
    expect_hamming_topk_parity<
            8,
            2,
            faiss::HammingComputer20_tpl<kRVV>,
            faiss::HammingComputer20_tpl<kNone>>(20, 1, 64, 8, false, 62);
    expect_hamming_topk_parity<
            8,
            3,
            faiss::HammingComputer64_tpl<kRVV>,
            faiss::HammingComputer64_tpl<kNone>>(64, 2, 67, 12, false, 63);
    expect_hamming_topk_parity<
            16,
            2,
            faiss::HammingComputer64_tpl<kRVV>,
            faiss::HammingComputer64_tpl<kNone>>(64, 1, 48, 16, true, 64);
    expect_hamming_topk_parity<
            8,
            2,
            faiss::HammingComputerDefault_tpl<kRVV>,
            faiss::HammingComputerDefault_tpl<kNone>>(12, 2, 65, 8, false, 65);
    expect_hamming_topk_parity<
            8,
            2,
            faiss::HammingComputerDefault_tpl<kRVV>,
            faiss::HammingComputerDefault_tpl<kNone>>(48, 1, 64, 8, true, 66);
}

TEST(ApproxTopKRVV, HammingComputer32RVV) {
    using HC_RVV = faiss::HammingComputer32_tpl<kRVV>;
    using HC_NONE = faiss::HammingComputer32_tpl<kNone>;

    for (uint32_t n : {1u, 3u, 4u, 5u, 7u, 8u, 15u, 16u, 17u, 31u, 33u, 64u}) {
        SCOPED_TRACE("n=" + std::to_string(n));
        const auto query = random_bytes<uint8_t>(32, 70);
        const auto vectors = random_bytes<uint8_t>(size_t(n) * 32, 71);

        HC_RVV hc_rvv(query.data(), 32);
        HC_NONE hc_none(query.data(), 32);

        std::vector<uint32_t> batch(n, 0xdeadbeef);
        hc_rvv.hamming_batch(vectors.data(), n, batch.data());

        for (uint32_t i = 0; i < n; i++) {
            const uint8_t* code = vectors.data() + size_t(i) * 32;
            EXPECT_EQ(batch[i], uint32_t(hc_rvv.hamming(code)));
            EXPECT_EQ(batch[i], uint32_t(hc_none.hamming(code)));
        }

        // A second batch must reuse the cached query pattern unchanged.
        std::vector<uint32_t> batch2(n, 0xdeadbeef);
        hc_rvv.hamming_batch(vectors.data(), n, batch2.data());
        EXPECT_EQ(batch, batch2);
    }
}

TEST(ApproxTopKRVV, HammingComputerParity) {
    // Per-code hamming() must agree between the RVV and NONE computers for
    // every code size, including the sizes that only inherit the scalar
    // body.
    auto check = [&]<typename HC_RVV, typename HC_NONE>(int code_size) {
        const auto query = random_bytes<uint8_t>(size_t(code_size), 80);
        const auto vectors = random_bytes<uint8_t>(size_t(code_size) * 8, 81);
        HC_RVV hc_rvv(query.data(), code_size);
        HC_NONE hc_none(query.data(), code_size);
        for (size_t i = 0; i < 8; i++) {
            const uint8_t* code = vectors.data() + i * size_t(code_size);
            EXPECT_EQ(hc_rvv.hamming(code), hc_none.hamming(code));
        }
    };

    check.template operator()<
            faiss::HammingComputer16_tpl<kRVV>,
            faiss::HammingComputer16_tpl<kNone>>(16);
    check.template operator()<
            faiss::HammingComputer20_tpl<kRVV>,
            faiss::HammingComputer20_tpl<kNone>>(20);
    check.template operator()<
            faiss::HammingComputer32_tpl<kRVV>,
            faiss::HammingComputer32_tpl<kNone>>(32);
    check.template operator()<
            faiss::HammingComputer64_tpl<kRVV>,
            faiss::HammingComputer64_tpl<kNone>>(64);
    check.template operator()<
            faiss::HammingComputerDefault_tpl<kRVV>,
            faiss::HammingComputerDefault_tpl<kNone>>(12);
    check.template operator()<
            faiss::HammingComputerDefault_tpl<kRVV>,
            faiss::HammingComputerDefault_tpl<kNone>>(48);
}

TEST(ApproxTopKRVV, HammingKnnHCEndToEnd) {
    // Drive the production entry point (hamming_impl.h) at both levels.
    for (int code_size : {8, 16, 20, 32, 48, 64}) {
        SCOPED_TRACE("code_size=" + std::to_string(code_size));
        const size_t nh = 2, nb = 257, k = 16;
        const auto queries = random_bytes<uint8_t>(nh * size_t(code_size), 90);
        const auto database = random_bytes<uint8_t>(nb * size_t(code_size), 91);

        for (auto mode :
             {ApproxTopK_mode_t::APPROX_TOPK_BUCKETS_B8_D3,
              ApproxTopK_mode_t::APPROX_TOPK_BUCKETS_B8_D2,
              ApproxTopK_mode_t::APPROX_TOPK_BUCKETS_B16_D2,
              ApproxTopK_mode_t::APPROX_TOPK_BUCKETS_B32_D2}) {
            std::vector<int64_t> rvv_ids(nh * k);
            std::vector<int> rvv_val(nh * k);
            std::vector<int64_t> ref_ids(nh * k);
            std::vector<int> ref_val(nh * k);

            faiss::int_maxheap_array_t rvv = {
                    nh, k, rvv_ids.data(), rvv_val.data()};
            faiss::int_maxheap_array_t ref = {
                    nh, k, ref_ids.data(), ref_val.data()};

            faiss::hammings_knn_hc_fixSL<kRVV>(
                    &rvv,
                    queries.data(),
                    database.data(),
                    nb,
                    size_t(code_size),
                    1,
                    mode,
                    nullptr);
            faiss::hammings_knn_hc_fixSL<kNone>(
                    &ref,
                    queries.data(),
                    database.data(),
                    nb,
                    size_t(code_size),
                    1,
                    mode,
                    nullptr);

            EXPECT_EQ(rvv_val, ref_val);
            EXPECT_EQ(rvv_ids, ref_ids);
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

TEST(ApproxTopKRVV, ApproxTopKByModeParity) {
    using C = faiss::CMax<float, int>;
    const uint32_t beam = 4, n_per_beam = 512, k = 16;

    for (int mode_int = int(ApproxTopK_mode_t::EXACT_TOPK);
         mode_int <= int(ApproxTopK_mode_t::APPROX_TOPK_BUCKETS_B8_D2);
         mode_int++) {
        const auto mode = ApproxTopK_mode_t(mode_int);
        SCOPED_TRACE("mode=" + std::to_string(mode_int));
        const auto distances =
                random_distances(beam * n_per_beam, 100 + mode_int);

        std::vector<float> rvv_val(k, C::neutral());
        std::vector<int32_t> rvv_ids(k, -1);
        std::vector<float> ref_val(k, C::neutral());
        std::vector<int32_t> ref_ids(k, -1);

        faiss::approx_topk_by_mode<kRVV>(
                mode,
                beam,
                n_per_beam,
                distances.data(),
                k,
                rvv_val.data(),
                rvv_ids.data());
        faiss::approx_topk_by_mode<kNone>(
                mode,
                beam,
                n_per_beam,
                distances.data(),
                k,
                ref_val.data(),
                ref_ids.data());

        faiss::heap_reorder<C>(k, rvv_val.data(), rvv_ids.data());
        faiss::heap_reorder<C>(k, ref_val.data(), ref_ids.data());

        EXPECT_EQ(rvv_val, ref_val);
        EXPECT_EQ(rvv_ids, ref_ids);
    }
}

TEST(ApproxTopKRVV, DispatchSelectsRVV) {
    // with_simd_level_256bit() used to fold RISCV_RVV down to NONE. The
    // approx-top-k call sites now provide native RVV kernels, so the
    // selector must hand them RISCV_RVV.
    const auto level_256 =
            faiss::with_simd_level_256bit([]<SIMDLevel SL>() { return SL; });
    EXPECT_EQ(level_256, SIMDLevel::RISCV_RVV);

    const auto level_base =
            faiss::with_simd_level([]<SIMDLevel SL>() { return SL; });
    EXPECT_EQ(level_base, SIMDLevel::RISCV_RVV);
}

// ---------------------------------------------------------------------------
// End-to-end: ResidualQuantizer encode with every approx mode
// ---------------------------------------------------------------------------

TEST(ApproxTopKRVV, ResidualQuantizerApproxModes) {
    // The encode path is the main user of the float bucket scan and the
    // accum tables. A broken kernel still decodes to finite vectors, so
    // compare the reconstruction error against the EXACT_TOPK mode: the
    // bucketed search may lose candidates but cannot invent better ones.
    const size_t d = 16;
    const size_t nt = 512, nq = 64;

    std::vector<float> xt(nt * d), xq(nq * d);
    faiss::rand_smooth_vectors(nt, d, xt.data(), 1234);
    faiss::rand_smooth_vectors(nq, d, xq.data(), 5678);

    faiss::ResidualQuantizer rq(d, 2, 8); // M = 2, nbits = 8
    rq.max_beam_size = 5;
    rq.train(nt, xt.data());
    rq.use_beam_LUT = 1;

    const size_t code_size = rq.code_size;
    std::vector<uint8_t> codes(nq * code_size);
    std::vector<float> reconstructed(nq * d);

    auto encode_and_error = [&](ApproxTopK_mode_t mode) {
        rq.approx_topk_mode = mode;
        rq.compute_codes(xq.data(), codes.data(), nq);
        rq.decode(codes.data(), reconstructed.data(), nq);
        double err = 0;
        for (size_t i = 0; i < nq; i++) {
            double e2 = 0;
            for (size_t j = 0; j < d; j++) {
                const double diff =
                        double(reconstructed[i * d + j]) - xq[i * d + j];
                e2 += diff * diff;
            }
            err += e2;
        }
        return err / nq;
    };

    const double err_exact = encode_and_error(ApproxTopK_mode_t::EXACT_TOPK);
    ASSERT_GT(err_exact, 0.0);

    const std::pair<const char*, ApproxTopK_mode_t> modes[] = {
            {"B8_D3", ApproxTopK_mode_t::APPROX_TOPK_BUCKETS_B8_D3},
            {"B8_D2", ApproxTopK_mode_t::APPROX_TOPK_BUCKETS_B8_D2},
            {"B16_D2", ApproxTopK_mode_t::APPROX_TOPK_BUCKETS_B16_D2},
            {"B32_D2", ApproxTopK_mode_t::APPROX_TOPK_BUCKETS_B32_D2},
    };
    for (const auto& [name, mode] : modes) {
        const double err_approx = encode_and_error(mode);
        EXPECT_LT(err_approx, 4.0 * err_exact + 1e-3)
                << "mode=" << name << " err_exact=" << err_exact;
    }
}

} // namespace

#endif // COMPILE_SIMD_RISCV_RVV
