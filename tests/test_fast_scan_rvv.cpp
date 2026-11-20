/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * FastScan RVV correctness tests: validate the native RVV fast_scan path
 * (kernels + accumulation loops + handlers + end-to-end search) against
 * the SIMDLevel::NONE scalar reference.
 *
 * Scope — the code under test is the fast_scan RVV optimization work:
 *   - kernels_rvv.h pq4_kernel_qbs / kernel_accumulate_block (native RVV,
 *     runtime-vl-agnostic; DummyScaler and NormTableScaler)
 *   - kernels_simd256.h dispatch for KernelSL == RISCV_RVV
 *   - accumulate_loops.h pq4_accumulate_loop_qbs_fixed_scaler_simd
 *     (4step dispatch table, both scalers) and
 *     pq4_accumulate_loop_fixed_scaler (search_1, both scalers)
 *   - impl-riscv.cpp: accumulate_to_mem_impl<RISCV_RVV>, the knn/range/
 *     partial-range scanner factories and the RaBitQ scanner factories
 *   - end-to-end: IndexPQFastScan (bbs=32 and bbs>32), IndexIVFPQFastScan
 *     knn + range search, IDSelector, IndexRaBitQFastScan and
 *     IndexIVFRaBitQFastScan (1-bit and multi-bit) vs their non-fast
 *     baselines
 *
 * All parity checks compare RVV against the SIMDLevel::NONE scalar
 * reference (independent oracle). The integer fast-scan domain is exact:
 * identical codes + identical LUT bytes must give bit-identical uint16
 * accumulators — no tolerance needed anywhere except the float-domain
 * end-to-end tests.
 *
 * Build: this TU must be compiled for RISC-V with the vector extension
 * (same per-file flags as impl-riscv.cpp: -march=rv64gcv_zvfhmin). On
 * other targets the file compiles to nothing.
 */

#ifdef COMPILE_SIMD_RISCV_RVV

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQFastScan.h>
#include <faiss/IndexIVFRaBitQ.h>
#include <faiss/IndexIVFRaBitQFastScan.h>
#include <faiss/IndexPQ.h>
#include <faiss/IndexPQFastScan.h>
#include <faiss/IndexRaBitQ.h>
#include <faiss/IndexRaBitQFastScan.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/fast_scan/accumulate_loops.h>
#include <faiss/impl/fast_scan/decompose_qbs.h>
#include <faiss/impl/fast_scan/fast_scan.h>
#include <faiss/impl/fast_scan/kernels_simd256.h>
#include <faiss/impl/fast_scan/simd_result_handlers.h>
#include <faiss/impl/simdlib/simdlib.h>
#include <faiss/index_factory.h>
#include <faiss/utils/AlignedTable.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/quantize_lut.h>
#include <faiss/utils/random.h>
#include <faiss/utils/simd_levels.h>

namespace {

using faiss::SIMDLevel;

constexpr SIMDLevel kRVV = SIMDLevel::RISCV_RVV;
constexpr SIMDLevel kNone = SIMDLevel::NONE;

std::mt19937& test_rng() {
    static std::mt19937 rng(20261008);
    return rng;
}

uint8_t rand_u8() {
    return static_cast<uint8_t>(test_rng()() >> 24);
}

// ---------------------------------------------------------------------------
// T2: pq4 kernel parity — the exact production kernel instantiations
// ---------------------------------------------------------------------------

// Generic parity driver for the QBS kernel (bbs=32, NQ in {1,2,3,4}).
template <int NQ>
void check_qbs_kernel_parity(int nsq, size_t n_blocks, bool norm_scale) {
    const size_t bbs = 32;
    const size_t nb = n_blocks * bbs;

    faiss::AlignedTable<uint8_t> codes(nsq * nb / 2);
    for (size_t i = 0; i < codes.size(); i++)
        codes[i] = rand_u8();
    // keep nibble domain small enough that u16 accumulators do not wrap:
    // worst case sum = nsq/2 pairs * 2 lookups * 255 * 16 (=scale) is too
    // big; norm_scale tests use their own bound. For DummyScaler path
    // worst case = nsq * 255 <= 65535 for nsq <= 257 — fine.
    if (norm_scale) {
        // NormTableScaler multiplies u16 accumulators by scale; keep the
        // LUT bytes small so products stay in range.
        for (size_t i = 0; i < codes.size(); i++)
            codes[i] &= 0x0F;
    }

    // LUT: NQ queries x nsq sub-quantizers x 16 entries, packed per
    // (sq pair, q) as 32 bytes.
    const size_t lut_bytes = NQ * nsq * 16;
    faiss::AlignedTable<uint8_t> lut(lut_bytes);
    for (size_t i = 0; i < lut.size(); i++)
        lut[i] = norm_scale ? (uint8_t)(test_rng()() % 32) : rand_u8();

    std::vector<uint16_t> got(NQ * nb, 0x5a5a);
    std::vector<uint16_t> want(NQ * nb, 0x5a5a);

    using namespace faiss::simd_result_handlers;

    {
        StoreResultHandler<SIMDLevel::NONE> h(want.data(), nb);
        if (norm_scale) {
            faiss::NormTableScaler<SIMDLevel::NONE> scaler(3);
            faiss::pq4_kernel_qbs_256<NQ, kNone>(
                    nsq, codes.get(), lut.get(), h, scaler);
        } else {
            faiss::DummyScaler<kNone> scaler;
            faiss::pq4_kernel_qbs_256<NQ, kNone>(
                    nsq, codes.get(), lut.get(), h, scaler);
        }
    }
    {
        StoreResultHandler<SIMDLevel::RISCV_RVV> h(got.data(), nb);
        if (norm_scale) {
            faiss::NormTableScaler<SIMDLevel::RISCV_RVV> scaler(3);
            faiss::pq4_kernel_qbs_256<NQ, kRVV>(
                    nsq, codes.get(), lut.get(), h, scaler);
        } else {
            faiss::DummyScaler<kRVV> scaler;
            faiss::pq4_kernel_qbs_256<NQ, kRVV>(
                    nsq, codes.get(), lut.get(), h, scaler);
        }
    }

    // find first differing index for diagnosis
    size_t first_diff = SIZE_MAX;
    int ndiff = 0;
    for (size_t i = 0; i < got.size(); i++) {
        if (got[i] != want[i]) {
            ndiff++;
            if (first_diff == SIZE_MAX)
                first_diff = i;
        }
    }
    ASSERT_EQ(ndiff, 0) << "NQ=" << NQ << " nsq=" << nsq
                        << " norm=" << norm_scale
                        << " first_diff=" << first_diff << " got[first]="
                        << (first_diff < got.size() ? got[first_diff] : 0)
                        << " want[first]="
                        << (first_diff < want.size() ? want[first_diff] : 0)
                        << " total=" << got.size();
}

TEST(FastScanRVVKernel, QbsKernelParity) {
    // nsq must be even; cover typical PQ16/PQ32/PQ64 shapes and small ones.
    check_qbs_kernel_parity<1>(2, 1, false);
    check_qbs_kernel_parity<1>(8, 1, false);
    check_qbs_kernel_parity<1>(16, 1, false);
    check_qbs_kernel_parity<1>(32, 1, false);
    check_qbs_kernel_parity<2>(16, 1, false);
    check_qbs_kernel_parity<2>(32, 1, false);
    check_qbs_kernel_parity<3>(16, 1, false);
    check_qbs_kernel_parity<3>(32, 1, false);
    check_qbs_kernel_parity<4>(16, 1, false);
    check_qbs_kernel_parity<4>(32, 1, false);
    // NormTableScaler path (AQ fast-scan style, nscale=2)
    check_qbs_kernel_parity<1>(16, 1, true);
    check_qbs_kernel_parity<2>(32, 1, true);
    check_qbs_kernel_parity<4>(16, 1, true);
}

// Generic parity driver for the multi-BB search_1 kernel.
template <int NQ, int BB>
void check_block_kernel_parity(int nsq, bool norm_scale = false) {
    const size_t bbs = 32 * BB;

    faiss::AlignedTable<uint8_t> codes(nsq * bbs / 2);
    for (size_t i = 0; i < codes.size(); i++)
        codes[i] = rand_u8();
    if (norm_scale) {
        for (size_t i = 0; i < codes.size(); i++)
            codes[i] &= 0x0F;
    }

    const size_t lut_bytes = NQ * nsq * 16;
    faiss::AlignedTable<uint8_t> lut(lut_bytes);
    for (size_t i = 0; i < lut.size(); i++)
        lut[i] = norm_scale ? (uint8_t)(test_rng()() % 32) : rand_u8();

    std::vector<uint16_t> got(NQ * bbs, 0x5a5a);
    std::vector<uint16_t> want(NQ * bbs, 0x5a5a);

    using namespace faiss::simd_result_handlers;

    if (norm_scale) {
        StoreResultHandler<SIMDLevel::NONE> h(want.data(), bbs);
        faiss::NormTableScaler<kNone> scaler(3);
        faiss::kernel_accumulate_block<NQ, BB, kNone>(
                nsq, codes.get(), lut.get(), h, scaler);
    } else {
        StoreResultHandler<SIMDLevel::NONE> h(want.data(), bbs);
        faiss::DummyScaler<kNone> scaler;
        faiss::kernel_accumulate_block<NQ, BB, kNone>(
                nsq, codes.get(), lut.get(), h, scaler);
    }
    if (norm_scale) {
        StoreResultHandler<SIMDLevel::RISCV_RVV> h(got.data(), bbs);
        faiss::NormTableScaler<kRVV> scaler(3);
        faiss::kernel_accumulate_block<NQ, BB, kRVV>(
                nsq, codes.get(), lut.get(), h, scaler);
    } else {
        StoreResultHandler<SIMDLevel::RISCV_RVV> h(got.data(), bbs);
        faiss::DummyScaler<kRVV> scaler;
        faiss::kernel_accumulate_block<NQ, BB, kRVV>(
                nsq, codes.get(), lut.get(), h, scaler);
    }

    ASSERT_EQ(got, want) << "NQ=" << NQ << " BB=" << BB << " nsq=" << nsq
                         << " norm=" << norm_scale;
}

TEST(FastScanRVVKernel, BlockKernelParity) {
    check_block_kernel_parity<1, 1>(16);
    check_block_kernel_parity<1, 2>(16);
    check_block_kernel_parity<1, 2>(32);
    check_block_kernel_parity<1, 3>(16);
    check_block_kernel_parity<1, 4>(32);
    check_block_kernel_parity<2, 1>(16);
    check_block_kernel_parity<2, 2>(32);
    check_block_kernel_parity<3, 1>(16);
    check_block_kernel_parity<4, 1>(32);
    // NormTableScaler tail (AdditiveQuantizer norm pair), multi-BB path.
    check_block_kernel_parity<1, 1>(16, /*norm_scale=*/true);
    check_block_kernel_parity<1, 2>(16, /*norm_scale=*/true);
    check_block_kernel_parity<1, 4>(8, /*norm_scale=*/true);
    check_block_kernel_parity<2, 1>(16, /*norm_scale=*/true);
    check_block_kernel_parity<2, 2>(8, /*norm_scale=*/true);
    check_block_kernel_parity<4, 1>(8, /*norm_scale=*/true);
}

// ---------------------------------------------------------------------------
// T3: accumulation loop parity — QBS dispatch table + search_1 path
// ---------------------------------------------------------------------------

void check_qbs_loop_parity(
        int qbs,
        int nq,
        size_t nb_blocks,
        int nsq,
        bool norm_scale = false) {
    const size_t nb = nb_blocks * 32;

    faiss::AlignedTable<uint8_t> codes(nsq * nb / 2);
    for (size_t i = 0; i < codes.size(); i++)
        codes[i] = rand_u8();
    if (norm_scale) {
        // Keep the u16 accumulator products in range; the kernel wraps
        // identically in both implementations, but small values make a
        // mismatch easier to read in the failure output.
        for (size_t i = 0; i < codes.size(); i++)
            codes[i] &= 0x0F;
    }

    const size_t lut_bytes = nq * nsq * 16;
    faiss::AlignedTable<uint8_t> lut(lut_bytes);
    for (size_t i = 0; i < lut.size(); i++)
        lut[i] = norm_scale ? (uint8_t)(test_rng()() % 32) : rand_u8();

    std::vector<uint16_t> got(nq * nb, 0x5a5a);
    std::vector<uint16_t> want(nq * nb, 0x5a5a);

    using namespace faiss::simd_result_handlers;

    {
        StoreResultHandler<SIMDLevel::NONE> h(want.data(), nb);
        if (norm_scale) {
            faiss::NormTableScaler<kNone> scaler(3);
            faiss::pq4_accumulate_loop_qbs_fixed_scaler_simd<kNone>(
                    qbs,
                    nb,
                    nsq,
                    codes.get(),
                    lut.get(),
                    h,
                    scaler,
                    32 * nsq / 2);
        } else {
            faiss::DummyScaler<kNone> scaler;
            faiss::pq4_accumulate_loop_qbs_fixed_scaler_simd<kNone>(
                    qbs,
                    nb,
                    nsq,
                    codes.get(),
                    lut.get(),
                    h,
                    scaler,
                    32 * nsq / 2);
        }
    }
    {
        StoreResultHandler<SIMDLevel::RISCV_RVV> h(got.data(), nb);
        if (norm_scale) {
            faiss::NormTableScaler<kRVV> scaler(3);
            faiss::pq4_accumulate_loop_qbs_fixed_scaler_simd<kRVV>(
                    qbs,
                    nb,
                    nsq,
                    codes.get(),
                    lut.get(),
                    h,
                    scaler,
                    32 * nsq / 2);
        } else {
            faiss::DummyScaler<kRVV> scaler;
            faiss::pq4_accumulate_loop_qbs_fixed_scaler_simd<kRVV>(
                    qbs,
                    nb,
                    nsq,
                    codes.get(),
                    lut.get(),
                    h,
                    scaler,
                    32 * nsq / 2);
        }
    }

    ASSERT_EQ(got, want) << "qbs=0x" << std::hex << qbs << " nq=" << nq
                         << " norm=" << norm_scale;
}

TEST(FastScanRVVLoop, QbsLoopParityAllDispatchEntries) {
    // Every qbs value in the compile-time dispatch table of
    // pq4_accumulate_loop_qbs_fixed_scaler_simd (see accumulate_loops.h).
    const int table[] = {
            0x3333, 0x2333, 0x2233, 0x333, 0x2223, 0x233, 0x1223, 0x223,
            0x34,   0x133,  0x6,    0x33,  0x123,  0x222, 0x23,   0x5,
            0x13,   0x22,   0x4,    0x3,   0x21,   0x2,   0x1,
    };
    for (int qbs : table) {
        int nq = 0;
        for (int qi = qbs; qi; qi >>= 4)
            nq += qi & 15;
        check_qbs_loop_parity(qbs, nq, 2, 16);
    }
    // qbs not in the table -> generic fallthrough loop (runtime value)
    check_qbs_loop_parity(0x12, 3, 2, 16); // 1 + 2 = 3 queries, not listed
    check_qbs_loop_parity(0x1111, 4, 2, 8);
}

TEST(FastScanRVVLoop, QbsLoopParityNormTableScaler) {
    // AdditiveQuantizer fast-scan path: scaler.nscale == 2, the last
    // sub-quantizer pair is multiplied by scale_int. This exercises the
    // scaled tail inside the multi-step QBS loop, not just the kernel.
    const int table[] = {0x3333, 0x2333, 0x23, 0x5, 0x2, 0x1};
    for (int qbs : table) {
        int nq = 0;
        for (int qi = qbs; qi; qi >>= 4)
            nq += qi & 15;
        check_qbs_loop_parity(qbs, nq, 2, 16, /*norm_scale=*/true);
    }
    check_qbs_loop_parity(0x12, 3, 2, 8, /*norm_scale=*/true);
}

void check_search1_loop_parity(int nq, int bb, int nsq, bool norm_scale) {
    const size_t bbs = 32 * bb;

    faiss::AlignedTable<uint8_t> codes(nsq * bbs / 2);
    for (size_t i = 0; i < codes.size(); i++)
        codes[i] = rand_u8();
    if (norm_scale) {
        for (size_t i = 0; i < codes.size(); i++)
            codes[i] &= 0x0F;
    }

    faiss::AlignedTable<uint8_t> lut(nq * nsq * 16);
    for (size_t i = 0; i < lut.size(); i++)
        lut[i] = norm_scale ? (uint8_t)(test_rng()() % 32) : rand_u8();

    std::vector<uint16_t> got(nq * bbs, 0x5a5a);
    std::vector<uint16_t> want(nq * bbs, 0x5a5a);

    using namespace faiss::simd_result_handlers;

    if (norm_scale) {
        StoreResultHandler<SIMDLevel::NONE> h(want.data(), bbs);
        faiss::NormTableScaler<kNone> scaler(3);
        faiss::pq4_accumulate_loop_fixed_scaler<kNone>(
                nq,
                bbs,
                (int)bbs,
                nsq,
                codes.get(),
                lut.get(),
                h,
                scaler,
                32 * nsq / 2);
    } else {
        StoreResultHandler<SIMDLevel::NONE> h(want.data(), bbs);
        faiss::DummyScaler<kNone> scaler;
        faiss::pq4_accumulate_loop_fixed_scaler<kNone>(
                nq,
                bbs,
                (int)bbs,
                nsq,
                codes.get(),
                lut.get(),
                h,
                scaler,
                32 * nsq / 2);
    }
    if (norm_scale) {
        StoreResultHandler<SIMDLevel::RISCV_RVV> h(got.data(), bbs);
        faiss::NormTableScaler<kRVV> scaler(3);
        faiss::pq4_accumulate_loop_fixed_scaler<kRVV>(
                nq,
                bbs,
                (int)bbs,
                nsq,
                codes.get(),
                lut.get(),
                h,
                scaler,
                32 * nsq / 2);
    } else {
        StoreResultHandler<SIMDLevel::RISCV_RVV> h(got.data(), bbs);
        faiss::DummyScaler<kRVV> scaler;
        faiss::pq4_accumulate_loop_fixed_scaler<kRVV>(
                nq,
                bbs,
                (int)bbs,
                nsq,
                codes.get(),
                lut.get(),
                h,
                scaler,
                32 * nsq / 2);
    }

    ASSERT_EQ(got, want) << "nq=" << nq << " bbs=" << bbs
                         << " norm=" << norm_scale;
}

TEST(FastScanRVVLoop, Search1LoopParity) {
    // pq4_accumulate_loop_fixed_scaler: all (nq, bbs) instantiations
    // listed in its dispatch switch.
    struct Combo {
        int nq;
        int bb; // bbs = 32 * bb
    };
    const Combo combos[] = {
            {1, 1},
            {1, 2},
            {1, 3},
            {1, 4},
            {1, 5},
            {2, 1},
            {2, 2},
            {3, 1},
            {4, 1},
    };
    for (const auto& c : combos) {
        check_search1_loop_parity(c.nq, c.bb, 16, /*norm_scale=*/false);
    }
}

TEST(FastScanRVVLoop, Search1LoopParityNormTableScaler) {
    // NormTableScaler through the search_1 (bbs > 32) accumulation loop.
    struct Combo {
        int nq;
        int bb;
    };
    const Combo combos[] = {{1, 1}, {1, 2}, {1, 4}, {2, 2}, {4, 1}};
    for (const auto& c : combos) {
        check_search1_loop_parity(c.nq, c.bb, 16, /*norm_scale=*/true);
    }
}

// ---------------------------------------------------------------------------
// T4: accumulate_to_mem — public entry, exercises impl-riscv.cpp wiring
// ---------------------------------------------------------------------------

TEST(FastScanRVVEntry, AccumulateToMemParity) {
    // Same inputs through the runtime-dispatched public API at the current
    // level (RVV in a static RISC-V build) vs the NONE reference obtained
    // by direct template instantiation.
    const int nsq = 16;
    const size_t ntotal2 = 96; // 3 blocks of 32
    const int nq = 3;

    faiss::AlignedTable<uint8_t> codes(nsq * ntotal2 / 2);
    for (size_t i = 0; i < codes.size(); i++)
        codes[i] = rand_u8();
    faiss::AlignedTable<uint8_t> lut(nq * nsq * 16);
    for (size_t i = 0; i < lut.size(); i++)
        lut[i] = rand_u8();

    std::vector<uint16_t> got(nq * ntotal2, 0x5a5a);
    std::vector<uint16_t> want(nq * ntotal2, 0x5a5a);

    // NONE reference by explicit template instantiation (always present).
    faiss::accumulate_to_mem_impl<kNone>(
            nq, ntotal2, nsq, codes.get(), lut.get(), want.data());

    // RVV specialization directly (this TU is compiled for RVV).
    faiss::accumulate_to_mem_impl<kRVV>(
            nq, ntotal2, nsq, codes.get(), lut.get(), got.data());

    ASSERT_EQ(got, want);

    // Public runtime entry must agree with the RVV specialization in a
    // static RISC-V build (with_simd_level resolves to RVV).
    std::vector<uint16_t> via_api(nq * ntotal2, 0x5a5a);
    faiss::accumulate_to_mem(
            nq, ntotal2, nsq, codes.get(), lut.get(), via_api.data());
    ASSERT_EQ(via_api, got);
}

// ---------------------------------------------------------------------------
// T5: handler mask path — HeapHandler over RVV kernel vs scalar oracle
// ---------------------------------------------------------------------------

TEST(FastScanRVVHandler, HeapHandlerKnnParity) {
    // Run the production knn scanner (k>1 heap path + k=1 single path)
    // over random packed codes/LUT and compare with a scalar oracle that
    // consumes the same StoreResultHandler distances.
    const int nsq = 16;
    const size_t nb = 128;
    const int nq = 3;
    const size_t k = 10;

    faiss::AlignedTable<uint8_t> codes(nsq * nb / 2);
    for (size_t i = 0; i < codes.size(); i++)
        codes[i] = rand_u8();
    faiss::AlignedTable<uint8_t> lut(nq * nsq * 16);
    for (size_t i = 0; i < lut.size(); i++)
        lut[i] = rand_u8();

    // Distances via the NONE kernel (oracle input).
    // Distances via the NONE kernel (oracle input). qbs=0x12 encodes
    // 1+2 = 3 queries: the nibble sum MUST equal nq, otherwise the staged
    // FixedStorageHandler flush overruns the handler's query dimension.
    std::vector<uint16_t> dis(nq * nb);
    {
        using namespace faiss::simd_result_handlers;
        StoreResultHandler<SIMDLevel::NONE> h(dis.data(), nb);
        faiss::DummyScaler<kNone> scaler;
        faiss::pq4_accumulate_loop_qbs_fixed_scaler_simd<kNone>(
                0x12, nb, nsq, codes.get(), lut.get(), h, scaler, 32 * nsq / 2);
    }

    for (int impl = 0; impl < 2; impl++) {
        // impl even -> HeapHandler; impl odd -> ReservoirHandler
        std::vector<float> D1(nq * k, std::numeric_limits<float>::max());
        std::vector<float> D2(nq * k);
        std::vector<int64_t> I1(nq * k, -1), I2(nq * k, -1);

        // Production scanner: runtime dispatch (RVV in static build).
        auto scanner = faiss::make_fast_scan_knn_scanner(
                /*is_max=*/true,
                impl,
                nq,
                nb,
                k,
                D1.data(),
                I1.data(),
                /*sel=*/nullptr);
        scanner->handler()->begin(nullptr);
        scanner->accumulate_loop_qbs(
                0x12,
                nb,
                nsq,
                codes.get(),
                lut.get(),
                /*pq2x4_scale=*/0,
                32 * nsq / 2);
        scanner->handler()->end();

        // Oracle: top-k from the scalar distance matrix.
        for (int q = 0; q < nq; q++) {
            std::vector<std::pair<uint16_t, int64_t>> pairs;
            pairs.reserve(nb);
            for (size_t j = 0; j < nb; j++)
                pairs.emplace_back(dis[q * nb + j], (int64_t)j);
            std::sort(pairs.begin(), pairs.end());
            for (size_t j = 0; j < k; j++) {
                D2[q * k + j] = pairs[j].first;
                I2[q * k + j] = pairs[j].second;
            }
        }

        for (int q = 0; q < nq; q++) {
            for (size_t j = 0; j < k; j++) {
                ASSERT_EQ(I1[q * k + j], I2[q * k + j])
                        << "impl=" << impl << " q=" << q << " j=" << j;
                ASSERT_EQ((uint16_t)D1[q * k + j], D2[q * k + j])
                        << "impl=" << impl << " q=" << q << " j=" << j;
            }
        }
    }
}

TEST(FastScanRVVHandler, SingleResultKnnParity) {
    // k=1 path (SingleResultHandler) — the hottest search shape.
    const int nsq = 16;
    const size_t nb = 96;
    const int nq = 2;

    faiss::AlignedTable<uint8_t> codes(nsq * nb / 2);
    for (size_t i = 0; i < codes.size(); i++)
        codes[i] = rand_u8();
    faiss::AlignedTable<uint8_t> lut(nq * nsq * 16);
    for (size_t i = 0; i < lut.size(); i++)
        lut[i] = rand_u8();

    std::vector<uint16_t> dis(nq * nb);
    {
        using namespace faiss::simd_result_handlers;
        StoreResultHandler<SIMDLevel::NONE> h(dis.data(), nb);
        faiss::DummyScaler<kNone> scaler;
        faiss::pq4_accumulate_loop_qbs_fixed_scaler_simd<kNone>(
                0x2, nb, nsq, codes.get(), lut.get(), h, scaler, 32 * nsq / 2);
    }

    std::vector<float> D1(nq, std::numeric_limits<float>::max());
    std::vector<float> D2(nq);
    std::vector<int64_t> I1(nq, -1), I2(nq, -1);

    auto scanner = faiss::make_fast_scan_knn_scanner(
            true, 0, nq, nb, 1, D1.data(), I1.data(), nullptr);
    scanner->handler()->begin(nullptr);
    scanner->accumulate_loop_qbs(
            0x2, nb, nsq, codes.get(), lut.get(), 0, 32 * nsq / 2);
    scanner->handler()->end();

    for (int q = 0; q < nq; q++) {
        size_t best = 0;
        for (size_t j = 1; j < nb; j++)
            if (dis[q * nb + j] < dis[q * nb + best])
                best = j;
        I2[q] = best;
        D2[q] = dis[q * nb + best];
        ASSERT_EQ(I1[q], I2[q]) << "q=" << q;
        ASSERT_EQ((uint16_t)D1[q], (uint16_t)D2[q]) << "q=" << q;
    }
}

// ---------------------------------------------------------------------------
// T6: end-to-end — IndexPQFastScan / IndexIVFPQFastScan vs IndexFlat
// ---------------------------------------------------------------------------

TEST(FastScanRVVE2E, PQ4FastScanSearchAccuracy) {
    // Same shape as tests/test_fast_scan.py::test_PQ4_accuracy but in C++:
    // recall@1 against an exact Flat index.
    const int d = 32;
    const size_t nt = 2000, nb = 5000, nq = 200;

    std::vector<float> xt(nt * d), xb(nb * d), xq(nq * d);
    faiss::rand_smooth_vectors(nt, d, xt.data(), 1234);
    faiss::rand_smooth_vectors(nb, d, xb.data(), 4567);
    faiss::rand_smooth_vectors(nq, d, xq.data(), 5678);

    faiss::IndexFlatL2 flat(d);
    flat.add(nb, xb.data());
    std::vector<float> Dref(nq * 10);
    std::vector<int64_t> Iref(nq * 10);
    flat.search(nq, xq.data(), 10, Dref.data(), Iref.data());

    std::unique_ptr<faiss::Index> fs(
            faiss::index_factory(d, "PQ16x4fs", faiss::METRIC_L2));
    fs->train(nt, xt.data());
    fs->add(nb, xb.data());

    std::vector<float> Da(nq * 10);
    std::vector<int64_t> Ia(nq * 10);
    fs->search(nq, xq.data(), 10, Da.data(), Ia.data());

    size_t agree = 0;
    for (size_t i = 0; i < nq; i++)
        if (Ia[i * 10] == Iref[i * 10])
            agree++;

    // PQ16x4 approximation floor: recall@1 of ~0.55 is expected for this
    // data shape (the python-side equivalent sees ~0.6 with other seeds).
    // The integer fast-scan path itself is verified bit-exact against the
    // NONE oracle in the kernel tests above; a broken kernel collapses
    // recall to near 0, so 0.4 still guards against regressions.
    ASSERT_GT(agree, nq * 4 / 10) << "recall@1 = " << (double)agree / nq;
}

TEST(FastScanRVVE2E, IVFPQFastScanSearchAccuracy) {
    const int d = 64;
    const size_t nt = 1500, nb = 3000, nq = 100;

    std::vector<float> xt(nt * d), xb(nb * d), xq(nq * d);
    faiss::rand_smooth_vectors(nt, d, xt.data(), 4242);
    faiss::rand_smooth_vectors(nb, d, xb.data(), 31337);
    faiss::rand_smooth_vectors(nq, d, xq.data(), 777);

    faiss::IndexFlatL2 flat(d);
    flat.add(nb, xb.data());
    std::vector<float> Dref(nq * 10);
    std::vector<int64_t> Iref(nq * 10);
    flat.search(nq, xq.data(), 10, Dref.data(), Iref.data());

    std::unique_ptr<faiss::IndexIVFPQFastScan> ivf(
            dynamic_cast<faiss::IndexIVFPQFastScan*>(faiss::index_factory(
                    d, "IVF32,PQ16x4fs", faiss::METRIC_L2)));
    ivf->train(nt, xt.data());
    ivf->add(nb, xb.data());
    ivf->nprobe = 8;

    std::vector<float> Da(nq * 10);
    std::vector<int64_t> Ia(nq * 10);
    ivf->search(nq, xq.data(), 10, Da.data(), Ia.data());

    size_t agree10 = 0;
    for (size_t i = 0; i < nq; i++) {
        for (int j = 0; j < 10; j++) {
            int64_t id = Ia[i * 10 + j];
            for (int jj = 0; jj < 10; jj++) {
                if (id == Iref[i * 10 + jj]) {
                    agree10++;
                    break;
                }
            }
        }
    }
    // recall@10 with nprobe=8/32: PQ16x4 approximation floor ~0.27 here.
    // A broken kernel collapses this to near 0; 0.15 keeps margin.
    ASSERT_GT(agree10, nq * 10 * 15 / 100)
            << "recall@10 = " << (double)agree10 / (nq * 10);
}

TEST(FastScanRVVE2E, PQ4FastScanIPMetric) {
    const int d = 32;
    const size_t nt = 1000, nb = 2000, nq = 100;

    std::vector<float> xt(nt * d), xb(nb * d), xq(nq * d);
    faiss::rand_smooth_vectors(nt, d, xt.data(), 2026);
    faiss::rand_smooth_vectors(nb, d, xb.data(), 4096);
    faiss::rand_smooth_vectors(nq, d, xq.data(), 8192);

    faiss::IndexFlatIP flat(d);
    flat.add(nb, xb.data());
    std::vector<float> Dref(nq * 10);
    std::vector<int64_t> Iref(nq * 10);
    flat.search(nq, xq.data(), 10, Dref.data(), Iref.data());

    std::unique_ptr<faiss::Index> fs(
            faiss::index_factory(d, "PQ16x4fs", faiss::METRIC_INNER_PRODUCT));
    fs->train(nt, xt.data());
    fs->add(nb, xb.data());

    std::vector<float> Da(nq * 10);
    std::vector<int64_t> Ia(nq * 10);
    fs->search(nq, xq.data(), 10, Da.data(), Ia.data());

    size_t agree = 0;
    for (size_t i = 0; i < nq; i++)
        if (Ia[i * 10] == Iref[i * 10])
            agree++;
    ASSERT_GT(agree, nq * 5 / 10) << "recall@1 = " << (double)agree / nq;
}

// ---------------------------------------------------------------------------
// T7: unaligned codes robustness (mirrors test_pqfs_unaligned.cpp)
// ---------------------------------------------------------------------------

TEST(FastScanRVVEntry, UnalignedCodesSameResults) {
    // Mirrors test_pqfs_unaligned.cpp: the kernel's loadu must tolerate
    // codes pointers that are off alignment. Reference = the NONE-level
    // accumulate (its asserts are compiled out under NDEBUG; NONE loads
    // are byte-wise so alignment cannot perturb them).
    const int nsq = 8;
    const size_t ntotal2 = 64; // 2 blocks of 32
    const int nq = 2;

    faiss::AlignedTable<uint8_t> codes(nsq * ntotal2 / 2 + 1);
    for (size_t i = 0; i < codes.size(); i++)
        codes[i] = rand_u8();
    faiss::AlignedTable<uint8_t> lut(nq * nsq * 16);
    for (size_t i = 0; i < lut.size(); i++)
        lut[i] = rand_u8();

    const uint8_t* codes_unaligned = codes.get() + 1;
    ASSERT_FALSE(faiss::is_aligned_pointer(codes_unaligned));

    std::vector<uint16_t> got(nq * ntotal2);
    std::vector<uint16_t> want(nq * ntotal2);

    {
        using namespace faiss::simd_result_handlers;
        StoreResultHandler<SIMDLevel::NONE> h(want.data(), ntotal2);
        faiss::DummyScaler<kNone> scaler;
        faiss::accumulate<kNone>(
                nq,
                ntotal2,
                nsq,
                codes_unaligned,
                lut.get(),
                h,
                scaler,
                32 * nsq / 2);
    }
    {
        using namespace faiss::simd_result_handlers;
        StoreResultHandler<SIMDLevel::RISCV_RVV> h(got.data(), ntotal2);
        faiss::DummyScaler<kRVV> scaler;
        faiss::accumulate<kRVV>(
                nq,
                ntotal2,
                nsq,
                codes_unaligned,
                lut.get(),
                h,
                scaler,
                32 * nsq / 2);
    }
    ASSERT_EQ(got, want);
}

// ---------------------------------------------------------------------------
// T8: range scanners — make_range_scanner / make_partial_range_scanner
// (impl-riscv.cpp instantiates both for RISCV_RVV).
//
// RangeHandler compares the quantized distance against
// int(a * (radius - b)); with a = 1, b = 0 the keep rule is
//   is_max=true  (CMax): d <  thr      (L2-style: keep smaller distances)
//   is_max=false (CMin): d >  thr      (similarity-style)
// ---------------------------------------------------------------------------

void compute_none_qbs_distances(
        int qbs,
        int nq,
        size_t nb,
        int nsq,
        const uint8_t* codes,
        const uint8_t* lut,
        std::vector<uint16_t>& dis) {
    using namespace faiss::simd_result_handlers;
    dis.assign(nq * nb, 0xffff);
    StoreResultHandler<SIMDLevel::NONE> h(dis.data(), nb);
    faiss::DummyScaler<kNone> scaler;
    faiss::pq4_accumulate_loop_qbs_fixed_scaler_simd<kNone>(
            qbs, nb, nsq, codes, lut, h, scaler, 32 * nsq / 2);
}

void check_range_results(
        const faiss::RangeSearchResult& rres,
        int nq,
        size_t nb,
        const std::vector<uint16_t>& dis,
        bool is_max,
        uint16_t thr,
        const char* label) {
    for (int q = 0; q < nq; q++) {
        std::vector<int64_t> got, want;
        for (size_t i = rres.lims[q]; i < rres.lims[q + 1]; i++) {
            int64_t id = rres.labels[i];
            ASSERT_GE(id, 0) << label << " q=" << q;
            ASSERT_LT(id, (int64_t)nb) << label << " q=" << q;
            // a = 1, b = 0 -> the reported distance is the raw u16 distance
            EXPECT_FLOAT_EQ(rres.distances[i], (float)dis[q * nb + id])
                    << label << " q=" << q << " id=" << id;
            got.push_back(id);
        }
        for (size_t j = 0; j < nb; j++) {
            bool keep =
                    is_max ? (dis[q * nb + j] < thr) : (dis[q * nb + j] > thr);
            if (keep)
                want.push_back((int64_t)j);
        }
        std::sort(got.begin(), got.end());
        std::sort(want.begin(), want.end());
        EXPECT_EQ(got, want) << label << " q=" << q;
    }
}

void check_range_scanner_parity(bool is_max) {
    const int nsq = 16;
    const int nq = 6;
    const int qbs = 0x123; // 1 + 2 + 3 = 6 queries
    const size_t nb = 256;
    const uint16_t thr = 30000;
    const float radius = float(thr);

    faiss::AlignedTable<uint8_t> codes((size_t)nsq * nb / 2);
    for (size_t i = 0; i < codes.size(); i++)
        codes[i] = rand_u8();

    faiss::AlignedTable<uint8_t> lut_src((size_t)nq * nsq * 16);
    for (size_t i = 0; i < lut_src.size(); i++)
        lut_src[i] = rand_u8();
    faiss::AlignedTable<uint8_t> lut(lut_src.size());
    // qbs stage layout: each stage is packed with its own nq, which is what
    // pq4_accumulate_loop_qbs_fixed_scaler_simd (4-step path and the generic
    // fallback) consumes.
    ASSERT_EQ(faiss::pq4_pack_LUT_qbs(qbs, nsq, lut_src.get(), lut.get()), nq);

    std::vector<uint16_t> dis;
    compute_none_qbs_distances(qbs, nq, nb, nsq, codes.get(), lut.get(), dis);

    faiss::RangeSearchResult rres(nq);
    auto scanner = faiss::make_range_scanner(is_max, rres, radius, nb, nullptr);

    std::vector<float> norms(2 * nq);
    for (int q = 0; q < nq; q++) {
        norms[2 * q] = 1.0f;
        norms[2 * q + 1] = 0.0f;
    }

    // RangeHandler is instantiated with with_id_map=true, which
    // unconditionally dereferences q_map (the production IVF range paths set
    // it in search_implem_10/12 before accumulating). Provide the identity
    // map ourselves; it must outlive the accumulate calls.
    std::vector<int> q_map(nq);
    for (int q = 0; q < nq; q++)
        q_map[q] = q;

    auto* handler = scanner->handler();
    handler->q_map = q_map.data();
    handler->begin(norms.data());
    scanner->accumulate_loop_qbs(
            qbs, nb, nsq, codes.get(), lut.get(), 0, 32 * nsq / 2);
    handler->end();

    check_range_results(rres, nq, nb, dis, is_max, thr, "full");
}

TEST(FastScanRVVEntry, RangeScannerParity) {
    check_range_scanner_parity(/*is_max=*/true);
    check_range_scanner_parity(/*is_max=*/false);
}

void check_partial_range_scanner_parity(bool is_max) {
    const int nsq = 16;
    const int nq = 6;
    const size_t nb = 256;
    const uint16_t thr = 30000;
    const float radius = float(thr);

    faiss::AlignedTable<uint8_t> codes((size_t)nsq * nb / 2);
    for (size_t i = 0; i < codes.size(); i++)
        codes[i] = rand_u8();

    faiss::AlignedTable<uint8_t> lut_src((size_t)nq * nsq * 16);
    for (size_t i = 0; i < lut_src.size(); i++)
        lut_src[i] = rand_u8();
    faiss::AlignedTable<uint8_t> lut_full(lut_src.size());
    const int full_qbs = 0x123;
    ASSERT_EQ(
            faiss::pq4_pack_LUT_qbs(
                    full_qbs, nsq, lut_src.get(), lut_full.get()),
            nq);

    std::vector<uint16_t> dis;
    compute_none_qbs_distances(
            full_qbs, nq, nb, nsq, codes.get(), lut_full.get(), dis);

    std::vector<float> norms(2 * nq);
    for (int q = 0; q < nq; q++) {
        norms[2 * q] = 1.0f;
        norms[2 * q + 1] = 0.0f;
    }

    faiss::RangeSearchResult rres(nq);
    {
        // Two query slices into one partial result, as the parallel IVF
        // range-search path does per thread.
        faiss::RangeSearchPartialResult pres(&rres);
        struct Slice {
            int i0, i1, qbs;
        };
        const Slice slices[] = {{0, 2, 0x2}, {2, 6, 0x22}};
        for (const auto& s : slices) {
            const int snq = s.i1 - s.i0;
            faiss::AlignedTable<uint8_t> lut_slice((size_t)snq * nsq * 16);
            ASSERT_EQ(
                    faiss::pq4_pack_LUT_qbs(
                            s.qbs,
                            nsq,
                            lut_src.get() + (size_t)s.i0 * nsq * 16,
                            lut_slice.get()),
                    snq);

            auto scanner = faiss::make_partial_range_scanner(
                    is_max, pres, radius, nb, s.i0, s.i1, nullptr);

            // with_id_map=true dereferences q_map; production
            // (search_implem_12) maps the slice-local query number qc.qno
            // (0..snq-1) onto itself. The handler adds q0 itself.
            std::vector<int> q_map(snq);
            for (int i = 0; i < snq; i++)
                q_map[i] = i;

            auto* handler = scanner->handler();
            handler->q_map = q_map.data();
            handler->begin(norms.data() + 2 * s.i0);
            scanner->accumulate_loop_qbs(
                    s.qbs,
                    nb,
                    nsq,
                    codes.get(),
                    lut_slice.get(),
                    0,
                    32 * nsq / 2);
            handler->end();
        }
        pres.set_lims();
        rres.do_allocation();
        pres.copy_result();
    }

    check_range_results(rres, nq, nb, dis, is_max, thr, "partial");
}

TEST(FastScanRVVEntry, PartialRangeScannerParity) {
    check_partial_range_scanner_parity(/*is_max=*/true);
    check_partial_range_scanner_parity(/*is_max=*/false);
}

// ---------------------------------------------------------------------------
// T9: end-to-end — IVF range search, bbs > 32, IDSelector
// ---------------------------------------------------------------------------

TEST(FastScanRVVE2E, IVFPQFastScanRangeSearchMatchesKnn) {
    // Both paths run the same quantized distances on the same probed lists,
    // so the range result must be a prefix of the distance-sorted knn
    // result. This validates radius scaling + result collection without
    // depending on the PQ approximation error.
    const int d = 32;
    const size_t nt = 1000, nb = 2000, nq = 20;

    std::vector<float> xt(nt * d), xb(nb * d), xq(nq * d);
    faiss::rand_smooth_vectors(nt, d, xt.data(), 1234);
    faiss::rand_smooth_vectors(nb, d, xb.data(), 4567);
    faiss::rand_smooth_vectors(nq, d, xq.data(), 5678);

    std::unique_ptr<faiss::Index> ivf(
            faiss::index_factory(d, "IVF32,PQ16x4fs", faiss::METRIC_L2));
    ivf->train(nt, xt.data());
    ivf->add(nb, xb.data());
    auto* ivf_fs = dynamic_cast<faiss::IndexIVFFastScan*>(ivf.get());
    ASSERT_NE(ivf_fs, nullptr);
    ivf_fs->nprobe = 4;

    // All candidate distances from the same lists range search will scan.
    std::vector<float> Dall(nq * nb);
    std::vector<int64_t> Iall(nq * nb);
    ivf->search(nq, xq.data(), nb, Dall.data(), Iall.data());

    std::vector<float> finite;
    for (size_t j = 0; j < nb; j++) {
        if (Dall[j] < std::numeric_limits<float>::max() / 2)
            finite.push_back(Dall[j]);
    }
    ASSERT_FALSE(finite.empty());
    std::sort(finite.begin(), finite.end());
    const float radius = finite[finite.size() / 2];

    faiss::RangeSearchResult rres(nq);
    ivf->range_search(nq, xq.data(), radius, &rres);

    size_t total = 0;
    for (size_t q = 0; q < nq; q++) {
        std::vector<float> row(
                Dall.begin() + q * nb, Dall.begin() + (q + 1) * nb);
        std::sort(row.begin(), row.end());

        std::vector<float> got(
                rres.distances + rres.lims[q],
                rres.distances + rres.lims[q + 1]);
        std::sort(got.begin(), got.end());
        total += got.size();

        ASSERT_LE(got.size(), row.size()) << "q=" << q;
        for (size_t i = 0; i < got.size(); i++) {
            EXPECT_FLOAT_EQ(got[i], row[i]) << "q=" << q << " rank=" << i;
        }
    }
    EXPECT_GT(total, 0u);
}

TEST(FastScanRVVE2E, Bbs64MatchesBbs32) {
    // bbs > 32 selects the search_1 / multi-BB path (ScannerMixIn::
    // accumulate_loop -> kernel_accumulate_block). The accumulation is
    // exact, so bbs=64 must return bit-identical results to bbs=32.
    const int d = 32;
    const size_t nt = 1000, nb = 2048, nq = 100, k = 10;

    std::vector<float> xt(nt * d), xb(nb * d), xq(nq * d);
    faiss::rand_smooth_vectors(nt, d, xt.data(), 1234);
    faiss::rand_smooth_vectors(nb, d, xb.data(), 4567);
    faiss::rand_smooth_vectors(nq, d, xq.data(), 5678);

    faiss::IndexPQ pq(d, 16, 4);
    pq.train(nt, xt.data());

    faiss::IndexPQFastScan fs32(pq, 32);
    faiss::IndexPQFastScan fs64(pq, 64);
    fs32.add(nb, xb.data());
    fs64.add(nb, xb.data());

    std::vector<float> D32(nq * k), D64(nq * k);
    std::vector<int64_t> I32(nq * k), I64(nq * k);
    fs32.search(nq, xq.data(), k, D32.data(), I32.data());
    fs64.search(nq, xq.data(), k, D64.data(), I64.data());

    EXPECT_EQ(D32, D64);
    EXPECT_EQ(I32, I64);
}

TEST(FastScanRVVE2E, IDSelectorFiltersKnn) {
    const int d = 32;
    const size_t nt = 1000, nb = 2000, nq = 50, k = 10;

    std::vector<float> xt(nt * d), xb(nb * d), xq(nq * d);
    faiss::rand_smooth_vectors(nt, d, xt.data(), 1234);
    faiss::rand_smooth_vectors(nb, d, xb.data(), 4567);
    faiss::rand_smooth_vectors(nq, d, xq.data(), 5678);

    std::unique_ptr<faiss::Index> fs(
            faiss::index_factory(d, "PQ16x4fs", faiss::METRIC_L2));
    fs->train(nt, xt.data());
    fs->add(nb, xb.data());

    // Unfiltered ground truth over the same codes.
    std::vector<float> Dall(nq * nb);
    std::vector<int64_t> Iall(nq * nb);
    fs->search(nq, xq.data(), nb, Dall.data(), Iall.data());

    faiss::IDSelectorRange sel(0, (faiss::idx_t)(nb / 2), false);
    faiss::SearchParameters params;
    params.sel = &sel;

    std::vector<float> D(nq * k);
    std::vector<int64_t> I(nq * k);
    fs->search(nq, xq.data(), k, D.data(), I.data(), &params);

    for (int q = 0; q < nq; q++) {
        std::vector<float> want;
        for (size_t j = 0; j < nb && want.size() < k; j++) {
            int64_t id = Iall[q * nb + j];
            if (id >= 0 && sel.is_member(id))
                want.push_back(Dall[q * nb + j]);
        }
        ASSERT_EQ(want.size(), k) << "q=" << q;
        for (size_t j = 0; j < k; j++) {
            ASSERT_TRUE(sel.is_member(I[q * k + j])) << "q=" << q << " j=" << j;
            EXPECT_FLOAT_EQ(D[q * k + j], want[j]) << "q=" << q << " j=" << j;
        }
    }
}

// ---------------------------------------------------------------------------
// T10: RaBitQ scanners — rabitq_make_knn_scanner_impl &&
// rabitq_ivf_make_knn_scanner_impl (impl-riscv.cpp), single-bit and multi-bit
// ---------------------------------------------------------------------------

double recall_at_k(
        const std::vector<int64_t>& got,
        const std::vector<int64_t>& ref,
        size_t nq,
        size_t k) {
    size_t hit = 0;
    for (size_t q = 0; q < nq; q++) {
        for (size_t j = 0; j < k; j++) {
            int64_t id = got[q * k + j];
            for (size_t jj = 0; jj < k; jj++) {
                if (id == ref[q * k + jj]) {
                    hit++;
                    break;
                }
            }
        }
    }
    return double(hit) / double(nq * k);
}

TEST(FastScanRVVE2E, RaBitQFastScanMatchesBaseline) {
    const int d = 32;
    const size_t nt = 2000, nb = 3000, nq = 100, k = 10;

    std::vector<float> xt(nt * d), xb(nb * d), xq(nq * d);
    faiss::rand_smooth_vectors(nt, d, xt.data(), 4242);
    faiss::rand_smooth_vectors(nb, d, xb.data(), 31337);
    faiss::rand_smooth_vectors(nq, d, xq.data(), 777);

    faiss::IndexFlatL2 flat(d);
    flat.add(nb, xb.data());
    std::vector<float> Dref(nq * k);
    std::vector<int64_t> Iref(nq * k);
    flat.search(nq, xq.data(), k, Dref.data(), Iref.data());

    for (int nb_bits : {1, 2}) {
        faiss::IndexRaBitQ base(d, faiss::METRIC_L2, (uint8_t)nb_bits);
        base.train(nt, xt.data());
        base.add(nb, xb.data());

        faiss::IndexRaBitQFastScan fs(
                d, faiss::METRIC_L2, 32, (uint8_t)nb_bits);
        fs.train(nt, xt.data());
        fs.add(nb, xb.data());

        std::vector<float> D0(nq * k), D1(nq * k);
        std::vector<int64_t> I0(nq * k), I1(nq * k);
        base.search(nq, xq.data(), k, D0.data(), I0.data());
        fs.search(nq, xq.data(), k, D1.data(), I1.data());

        const double r_base = recall_at_k(I0, Iref, nq, k);
        const double r_fs = recall_at_k(I1, Iref, nq, k);
        EXPECT_NEAR(r_fs, r_base, 0.05)
                << "nb_bits=" << nb_bits << " base=" << r_base
                << " fastscan=" << r_fs;
    }
}

TEST(FastScanRVVE2E, IVFRaBitQFastScanMatchesBaseline) {
    const int d = 32;
    const size_t nt = 2000, nb = 3000, nq = 100, k = 10;
    const size_t nlist = 16, nprobe = 4;

    std::vector<float> xt(nt * d), xb(nb * d), xq(nq * d);
    faiss::rand_smooth_vectors(nt, d, xt.data(), 4242);
    faiss::rand_smooth_vectors(nb, d, xb.data(), 31337);
    faiss::rand_smooth_vectors(nq, d, xq.data(), 777);

    faiss::IndexFlatL2 flat(d);
    flat.add(nb, xb.data());
    std::vector<float> Dref(nq * k);
    std::vector<int64_t> Iref(nq * k);
    flat.search(nq, xq.data(), k, Dref.data(), Iref.data());

    for (int nb_bits : {1, 2}) {
        // Both indexes take ownership of the inverted lists only, so keep
        // the quantizers alive here.
        std::unique_ptr<faiss::Index> q0(new faiss::IndexFlatL2(d));
        faiss::IndexIVFRaBitQ base(
                q0.get(), d, nlist, faiss::METRIC_L2, true, (uint8_t)nb_bits);
        base.nprobe = nprobe;
        base.train(nt, xt.data());
        base.add(nb, xb.data());

        std::unique_ptr<faiss::Index> q1(new faiss::IndexFlatL2(d));
        faiss::IndexIVFRaBitQFastScan fs(
                q1.get(),
                d,
                nlist,
                faiss::METRIC_L2,
                32,
                true,
                (uint8_t)nb_bits);
        fs.nprobe = nprobe;
        fs.train(nt, xt.data());
        fs.add(nb, xb.data());

        std::vector<float> D0(nq * k), D1(nq * k);
        std::vector<int64_t> I0(nq * k), I1(nq * k);
        base.search(nq, xq.data(), k, D0.data(), I0.data());
        fs.search(nq, xq.data(), k, D1.data(), I1.data());

        const double r_base = recall_at_k(I0, Iref, nq, k);
        const double r_fs = recall_at_k(I1, Iref, nq, k);
        EXPECT_NEAR(r_fs, r_base, 0.05)
                << "nb_bits=" << nb_bits << " base=" << r_base
                << " fastscan=" << r_fs;
    }
}

} // namespace

#endif // COMPILE_SIMD_RISCV_RVV
