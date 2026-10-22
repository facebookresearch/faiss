/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Native RVV implementation of HeapWithBucketsForHamming32.
//
// Same design as impl/approx_topk/rvv.cpp: vectorize over all NBUCKETS of
// one bucket block with a runtime vsetvl, instead of the 8-lane simd8uint32
// grouping. The Hamming distances themselves are produced by the Hamming
// Computer (scalar or RVV, depending on the code size), then loaded into one
// RVV vector and reduced with unsigned compares + vmerge.
//
// Semantics match the generic template (and upstream faiss):
//   flag = candidate < current (unsigned; tie keeps current)
//   minValues = flag ? candidate : current
//   minIndices = flag ? candidateIdx : currentIdx
//   maxValues = flag ? current : candidate
//   maxIndices = flag ? currentIdx : candidateIdx
// The max side feeds the next depth; unlike the float kernel it uses the
// cmplt max (candidate vs current), not the fresh input.

#pragma once

#if defined(COMPILE_SIMD_RISCV_RVV)

#include <cstddef>
#include <cstdint>
#include <limits>

#include <riscv_vector.h>

#include <faiss/utils/approx_topk_hamming/approx_topk_hamming.h>
#include <faiss/utils/Heap.h>

namespace faiss {

namespace approx_topk_hamming_rvv {

// Per-LMUL uint32 operations. RVV vector types cannot be struct members or
// array elements, so these traits only carry aliases and static functions;
// all state lives in named locals in scan_chunk.
#define FAISS_RVV_HAMMING_OPS(NAME, LM, VU, VB, MB)                           \
    struct NAME {                                                             \
        using vu = VU;                                                        \
        using vb = VB;                                                        \
                                                                              \
        static inline size_t setvl(size_t avl) {                              \
            return __riscv_vsetvl_e32##LM(avl);                               \
        }                                                                     \
        static inline vu splat_u(uint32_t x, size_t vl) {                     \
            return __riscv_vmv_v_x_u32##LM(x, vl);                            \
        }                                                                     \
        static inline vu vid(size_t vl) {                                     \
            return __riscv_vid_v_u32##LM(vl);                                 \
        }                                                                     \
        static inline vu add_x(vu a, uint32_t x, size_t vl) {                 \
            return __riscv_vadd_vx_u32##LM(a, x, vl);                         \
        }                                                                     \
        static inline vu load_u(const uint32_t* p, size_t vl) {               \
            return __riscv_vle32_v_u32##LM(p, vl);                            \
        }                                                                     \
        static inline void store_u(uint32_t* p, vu v, size_t vl) {            \
            __riscv_vse32_v_u32##LM(p, v, vl);                                \
        }                                                                     \
        static inline vb lt(vu a, vu b, size_t vl) {                          \
            return __riscv_vmsltu_vv_u32##LM##_b##MB(a, b, vl);               \
        }                                                                     \
        /* mask ? t : f */                                                    \
        static inline vu select_u(vb m, vu t, vu f, size_t vl) {              \
            return __riscv_vmerge_vvm_u32##LM(f, t, m, vl);                   \
        }                                                                     \
    };

// Mask ratio is SEW/LMUL: m1 -> 32, m2 -> 16, m4 -> 8.
FAISS_RVV_HAMMING_OPS(rvv_e32m1, m1, vuint32m1_t, vbool32_t, 32)
FAISS_RVV_HAMMING_OPS(rvv_e32m2, m2, vuint32m2_t, vbool16_t, 16)
FAISS_RVV_HAMMING_OPS(rvv_e32m4, m4, vuint32m4_t, vbool8_t, 8)

#undef FAISS_RVV_HAMMING_OPS

// Chunk / LMUL policy, selected once per call from the hardware VLEN.
// Narrow (VLEN=128): m2/m4/m4 with B32 split into two chunks.
// Wide (VLEN>=256): m1/m2/m4 with B32 in a single m4 chunk.
template <uint32_t NBUCKETS, bool Wide>
struct rvv_plan;

template <>
struct rvv_plan<8, false> {
    static constexpr uint32_t kChunk = 8;
    using ops = rvv_e32m2;
};
template <>
struct rvv_plan<8, true> {
    static constexpr uint32_t kChunk = 8;
    using ops = rvv_e32m1;
};
template <>
struct rvv_plan<16, false> {
    static constexpr uint32_t kChunk = 16;
    using ops = rvv_e32m4;
};
template <>
struct rvv_plan<16, true> {
    static constexpr uint32_t kChunk = 16;
    using ops = rvv_e32m2;
};
template <>
struct rvv_plan<32, false> {
    static constexpr uint32_t kChunk = 16;
    using ops = rvv_e32m4;
};
template <>
struct rvv_plan<32, true> {
    static constexpr uint32_t kChunk = 32;
    using ops = rvv_e32m4;
};

// One depth of the bucket reduction.
template <typename O>
inline __attribute__((always_inline)) void hamming_step(
        typename O::vu& candidate_v,
        typename O::vu& candidate_i,
        typename O::vu& min_v,
        typename O::vu& min_i,
        const size_t vl) {
    const typename O::vb flag = O::lt(candidate_v, min_v, vl);

    const typename O::vu next_min_v = O::select_u(flag, candidate_v, min_v, vl);
    const typename O::vu next_min_i = O::select_u(flag, candidate_i, min_i, vl);
    const typename O::vu next_max_v = O::select_u(flag, min_v, candidate_v, vl);
    const typename O::vu next_max_i = O::select_u(flag, min_i, candidate_i, vl);

    min_v = next_min_v;
    min_i = next_min_i;
    candidate_v = next_max_v;
    candidate_i = next_max_i;
}

// Scan one chunk of buckets over all full NBUCKETS-sized blocks.
template <uint32_t NBUCKETS, uint32_t N, typename HammingComputerT, typename O>
inline void scan_chunk(
        const HammingComputerT& hc,
        const uint8_t* const binary_vectors,
        const IDSelector* sel,
        const size_t code_size,
        const uint32_t n_per_beam,
        const uint32_t beam_off,
        const uint32_t chunk_off,
        uint32_t (&min_v)[N][NBUCKETS],
        uint32_t (&min_i)[N][NBUCKETS],
        const size_t vl) {
    using vu = typename O::vu;

    const vu lane = O::vid(vl);
    const vu idx0 = O::add_x(lane, chunk_off, vl);

    [[maybe_unused]] vu v0, v1, v2;
    [[maybe_unused]] vu i0, i1, i2;
    constexpr uint32_t kMax = std::numeric_limits<int32_t>::max();
    v0 = O::splat_u(kMax, vl);
    i0 = idx0;
    if constexpr (N >= 2) {
        v1 = v0;
        i1 = idx0;
    }
    if constexpr (N >= 3) {
        v2 = v0;
        i2 = idx0;
    }

    const uint32_t nb = (n_per_beam / NBUCKETS) * NBUCKETS;

    uint32_t dist_buf[NBUCKETS];
    vu base = idx0;

    for (uint32_t ip = 0; ip < nb; ip += NBUCKETS) {
        bool any_valid = true;
        if (sel == nullptr) {
            const uint32_t first = beam_off + ip + chunk_off;
            if constexpr (requires(
                                  const HammingComputerT& h,
                                  const uint8_t* p,
                                  uint32_t m,
                                  uint32_t* o) { h.hamming_batch(p, m, o); }) {
                // Batch distance kernel (HammingComputer32 RVV): computes
                // all `vl` distances of this bucket block in one vectorized
                // SWAR pass instead of `vl` per-code calls.
                hc.hamming_batch(
                        binary_vectors + size_t(first) * code_size,
                        uint32_t(vl),
                        dist_buf);
            } else {
                for (uint32_t l = 0; l < vl; l++) {
                    const uint32_t idx = first + l;
                    dist_buf[l] = hc.hamming(
                            binary_vectors + size_t(idx) * code_size);
                }
            }
        } else {
            any_valid = false;
            const uint32_t first = beam_off + ip + chunk_off;
            for (uint32_t l = 0; l < vl; l++) {
                const uint32_t idx = first + l;
                if (sel->is_member(idx)) {
                    dist_buf[l] = hc.hamming(
                            binary_vectors + size_t(idx) * code_size);
                    any_valid = true;
                } else {
                    dist_buf[l] = kMax;
                }
            }
        }

        if (!any_valid) {
            base = O::add_x(base, NBUCKETS, vl);
            continue;
        }

        const vu fresh = O::load_u(dist_buf, vl);
        vu cand_v = fresh;
        vu cand_i = base;

        hamming_step<O>(cand_v, cand_i, v0, i0, vl);
        if constexpr (N >= 2) {
            hamming_step<O>(cand_v, cand_i, v1, i1, vl);
        }
        if constexpr (N >= 3) {
            hamming_step<O>(cand_v, cand_i, v2, i2, vl);
        }

        base = O::add_x(base, NBUCKETS, vl);
    }

    // Bucket ids already contain the in-block offset; add the beam base.
    i0 = O::add_x(i0, beam_off, vl);
    O::store_u(&min_v[0][chunk_off], v0, vl);
    O::store_u(&min_i[0][chunk_off], i0, vl);
    if constexpr (N >= 2) {
        i1 = O::add_x(i1, beam_off, vl);
        O::store_u(&min_v[1][chunk_off], v1, vl);
        O::store_u(&min_i[1][chunk_off], i1, vl);
    }
    if constexpr (N >= 3) {
        i2 = O::add_x(i2, beam_off, vl);
        O::store_u(&min_v[2][chunk_off], v2, vl);
        O::store_u(&min_i[2][chunk_off], i2, vl);
    }
}

template <uint32_t NBUCKETS, uint32_t N, typename HammingComputerT, typename P>
inline void bs_addn_impl(
        const uint32_t beam_size,
        const uint32_t n_per_beam,
        const HammingComputerT& hc,
        const uint8_t* const __restrict binary_vectors,
        const uint32_t k,
        int* const __restrict bh_val,
        int64_t* const __restrict bh_ids,
        const IDSelector* sel) {
    using C = CMax<int, int64_t>;
    using O = typename P::ops;

    static_assert(NBUCKETS % P::kChunk == 0, "chunk must divide NBUCKETS");

    const size_t vl = O::setvl(P::kChunk);
    const size_t code_size = size_t(hc.get_code_size());
    const uint32_t nb = (n_per_beam / NBUCKETS) * NBUCKETS;
    constexpr uint32_t kMax = std::numeric_limits<int32_t>::max();

    for (uint32_t beam_index = 0; beam_index < beam_size; beam_index++) {
        alignas(32) uint32_t min_values[N][NBUCKETS];
        alignas(32) uint32_t min_indices[N][NBUCKETS];

        const uint32_t beam_off = n_per_beam * beam_index;

        for (uint32_t chunk_off = 0; chunk_off < NBUCKETS;
             chunk_off += P::kChunk) {
            scan_chunk<NBUCKETS, N, HammingComputerT, O>(
                    hc,
                    binary_vectors,
                    sel,
                    code_size,
                    n_per_beam,
                    beam_off,
                    chunk_off,
                    min_values,
                    min_indices,
                    vl);
        }

        for (uint32_t p = 0; p < N; p++) {
            for (uint32_t b = 0; b < NBUCKETS; b++) {
                const uint32_t value = min_values[p][b];
                if (value >= kMax) {
                    continue;
                }
                const int64_t index = int64_t(min_indices[p][b]);
                if (C::cmp2(bh_val[0], int(value), bh_ids[0], index)) {
                    heap_replace_top<C>(
                            k, bh_val, bh_ids, int(value), index);
                }
            }
        }

        for (uint32_t ip = nb; ip < n_per_beam; ip++) {
            const int64_t index = ip + beam_off;
            if (!sel || sel->is_member(index)) {
                const int value = hc.hamming(
                        binary_vectors + size_t(index) * code_size);
                if (C::cmp(bh_val[0], value)) {
                    heap_replace_top<C>(k, bh_val, bh_ids, value, index);
                }
            }
        }
    }
}

} // namespace approx_topk_hamming_rvv

// Native RVV specialization of the Hamming top-k bucket scan.
template <uint32_t NBUCKETS, uint32_t N, typename HammingComputerT>
struct HeapWithBucketsForHamming32<
        CMax<int, int64_t>,
        NBUCKETS,
        N,
        HammingComputerT,
        SIMDLevel::RISCV_RVV> {
    static_assert(
            NBUCKETS > 0 && (NBUCKETS % 8) == 0,
            "Number of buckets needs to be 8, 16, 24, ...");
    static_assert(N >= 1 && N <= 3, "RVV Hamming bucket scan supports N 1..3");

    static void addn(
            const uint32_t n,
            const HammingComputerT& hc,
            const uint8_t* const __restrict binaryVectors,
            const uint32_t k,
            int* const __restrict bh_val,
            int64_t* const __restrict bh_ids,
            const IDSelector* sel = nullptr) {
        bs_addn(1, n, hc, binaryVectors, k, bh_val, bh_ids, sel);
    }

    static void bs_addn(
            const uint32_t beam_size,
            const uint32_t n_per_beam,
            const HammingComputerT& hc,
            const uint8_t* const __restrict binary_vectors,
            const uint32_t k,
            int* const __restrict bh_val,
            int64_t* const __restrict bh_ids,
            const IDSelector* sel = nullptr) {
        using namespace approx_topk_hamming_rvv;
        if (__riscv_vlenb() >= 32) {
            bs_addn_impl<NBUCKETS, N, HammingComputerT, rvv_plan<NBUCKETS, true>>(
                    beam_size,
                    n_per_beam,
                    hc,
                    binary_vectors,
                    k,
                    bh_val,
                    bh_ids,
                    sel);
        } else {
            bs_addn_impl<
                    NBUCKETS,
                    N,
                    HammingComputerT,
                    rvv_plan<NBUCKETS, false>>(
                    beam_size,
                    n_per_beam,
                    hc,
                    binary_vectors,
                    k,
                    bh_val,
                    bh_ids,
                    sel);
        }
    }
};

} // namespace faiss

#endif // COMPILE_SIMD_RISCV_RVV
