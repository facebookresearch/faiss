/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Native RISC-V RVV implementation of the uint16_t partitioning and histogram
// kernels used by IVF / k-means training, using RVV 1.0 intrinsics directly:
//
//  * bisection / counting: e16m4 (SEW/LMUL = 4 -> vbool4) with vcmp + vcpop,
//    32 elements per iteration on VLEN=128.
//  * compression: ratio-8 register groups so a single mask drives the 16-bit
//    value compressor and both id compressors:
//        values  e16m2 (vbool8)
//        id32    e32m4 (vbool8)
//        id64    e64m8 (vbool8)
//    16 elements per iteration on VLEN=128. vcompress preserves order, so
//    the output is identical to the scalar compressor.
//  * histograms: e16m8 blocks, one vmseq + vcpop per bin.
//
// n_eq handling for equal values follows the AVX2/NEON simdlib256 path:
// viota gives each equal lane its rank among the equal lanes of the current
// block, and only ranks < remaining n_eq are kept.

#ifdef COMPILE_SIMD_RISCV_RVV

#include <riscv_vector.h>

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <faiss/utils/ordered_key_value.h>
#include <faiss/utils/partitioning.h>

namespace faiss {

namespace {

/******************************************************************
 * Helpers
 ******************************************************************/

size_t rvv_vlmax_e16m8() {
    return __riscv_vsetvl_e16m8((size_t)-1);
}

// min / max over an aligned uint16_t array
[[maybe_unused]] void rvv_find_minimax(
        const uint16_t* vals,
        size_t n,
        uint16_t& smin,
        uint16_t& smax) {
    smin = 0xffff;
    smax = 0;
    if (n == 0) {
        return;
    }

    const size_t vl = rvv_vlmax_e16m8();
    if (n < vl) {
        for (size_t i = 0; i < n; i++) {
            smin = std::min(smin, vals[i]);
            smax = std::max(smax, vals[i]);
        }
        return;
    }

    vuint16m8_t vmin = __riscv_vmv_v_x_u16m8(0xffff, vl);
    vuint16m8_t vmax = __riscv_vmv_v_x_u16m8(0, vl);

    size_t i = 0;
    for (; i + vl <= n; i += vl) {
        vuint16m8_t v = __riscv_vle16_v_u16m8(vals + i, vl);
        vmin = __riscv_vminu_vv_u16m8(vmin, v, vl);
        vmax = __riscv_vmaxu_vv_u16m8(vmax, v, vl);
    }
    // reprocess the last full block so the tail stays vectorized (min/max
    // are idempotent, overlapping lanes are harmless)
    if (i < n) {
        vuint16m8_t v = __riscv_vle16_v_u16m8(vals + n - vl, vl);
        vmin = __riscv_vminu_vv_u16m8(vmin, v, vl);
        vmax = __riscv_vmaxu_vv_u16m8(vmax, v, vl);
    }

    // vredminu.vs / vredmaxu.vs take the scalar operand as an m1 vector
    vuint16m1_t seed_min = __riscv_vmv_v_x_u16m1((uint16_t)0xffff, 1);
    vuint16m1_t seed_max = __riscv_vmv_v_x_u16m1((uint16_t)0, 1);
    smin = __riscv_vmv_x_s_u16m1_u16(
            __riscv_vredminu_vs_u16m8_u16m1(vmin, seed_min, vl));
    smax = __riscv_vmv_x_s_u16m1_u16(
            __riscv_vredmaxu_vs_u16m8_u16m1(vmax, seed_max, vl));
}

// count values < thresh (CMax) or > thresh (CMin)
// e16m4 (32 lanes on VLEN=128): two 16-bit compares + vcpop per block.
template <class C>
size_t rvv_count_lt(const uint16_t* vals, size_t n, uint16_t thresh) {
    const size_t vl = __riscv_vsetvl_e16m4((size_t)-1);
    if (n < vl) {
        size_t lt = 0;
        for (size_t i = 0; i < n; i++) {
            if (C::cmp(thresh, vals[i])) {
                lt++;
            }
        }
        return lt;
    }

    vuint16m4_t vthr = __riscv_vmv_v_x_u16m4(thresh, vl);
    size_t lt = 0;
    size_t i = 0;
    for (; i + vl <= n; i += vl) {
        vuint16m4_t v = __riscv_vle16_v_u16m4(vals + i, vl);
        if constexpr (C::is_max) {
            lt += __riscv_vcpop_m_b4(
                    __riscv_vmsltu_vv_u16m4_b4(v, vthr, vl), vl);
        } else {
            lt += __riscv_vcpop_m_b4(
                    __riscv_vmsgtu_vv_u16m4_b4(v, vthr, vl), vl);
        }
    }

    for (; i < n; i++) {
        if (C::cmp(thresh, vals[i])) {
            lt++;
        }
    }
    return lt;
}

// count values == thresh
size_t rvv_count_eq(const uint16_t* vals, size_t n, uint16_t thresh) {
    const size_t vl = __riscv_vsetvl_e16m4((size_t)-1);
    if (n < vl) {
        size_t eq = 0;
        for (size_t i = 0; i < n; i++) {
            eq += (vals[i] == thresh);
        }
        return eq;
    }

    vuint16m4_t vthr = __riscv_vmv_v_x_u16m4(thresh, vl);
    size_t eq = 0;
    size_t i = 0;
    for (; i + vl <= n; i += vl) {
        vuint16m4_t v = __riscv_vle16_v_u16m4(vals + i, vl);
        eq += __riscv_vcpop_m_b4(__riscv_vmseq_vv_u16m4_b4(v, vthr, vl), vl);
    }

    for (; i < n; i++) {
        eq += (vals[i] == thresh);
    }
    return eq;
}

// Compress values / ids that satisfy the threshold, keeping at most n_eq
// equal values. `ids` is int32_t or int64_t.
template <class C, class IDT>
size_t rvv_compress_array(
        uint16_t* vals,
        IDT* ids,
        size_t n,
        uint16_t thresh,
        int64_t n_eq) {
    static_assert(
            sizeof(IDT) == 4 || sizeof(IDT) == 8, "ids must be 32 or 64 bit");

    size_t wp = 0;
    size_t i = 0;

    // ratio-8 groups: e16m2 / e32m4 / e64m8 share one vbool8 mask and one vl
    const size_t vlmax = __riscv_vsetvl_e16m2((size_t)-1);

    // Common case: no equal-value quota left (the bisection normally breaks
    // with q == n_lt, so n_eq_1 == 0). Specialize the loop so the equality
    // compare and the n_eq branch disappear from the hot path.
    if (n_eq == 0) {
        auto process_no_eq = [&](size_t vl) {
            vuint16m2_t v = __riscv_vle16_v_u16m2(vals + i, vl);
            vuint16m2_t vthr = __riscv_vmv_v_x_u16m2(thresh, vl);
            vbool8_t keep;
            if constexpr (C::is_max) {
                keep = __riscv_vmsltu_vv_u16m2_b8(v, vthr, vl);
            } else {
                keep = __riscv_vmsgtu_vv_u16m2_b8(v, vthr, vl);
            }
            size_t k = __riscv_vcpop_m_b8(keep, vl);

            vuint16m2_t vc = __riscv_vcompress_vm_u16m2(v, keep, vl);
            __riscv_vse16_v_u16m2(vals + wp, vc, vl);

            if constexpr (sizeof(IDT) == 4) {
                vuint32m4_t idv =
                        __riscv_vle32_v_u32m4((const uint32_t*)(ids + i), vl);
                vuint32m4_t idx = __riscv_viota_m_u32m4(keep, vl);
                idx = __riscv_vsll_vx_u32m4(idx, 2, vl);
                __riscv_vsoxei32_v_u32m4_m(
                        keep, (uint32_t*)(ids + wp), idx, idv, vl);
            } else {
                vuint64m8_t idv =
                        __riscv_vle64_v_u64m8((const uint64_t*)(ids + i), vl);
                vuint64m8_t idx = __riscv_viota_m_u64m8(keep, vl);
                idx = __riscv_vsll_vx_u64m8(idx, 3, vl);
                __riscv_vsoxei64_v_u64m8_m(
                        keep, (uint64_t*)(ids + wp), idx, idv, vl);
            }
            wp += k;
        };

        for (; i + vlmax <= n; i += vlmax) {
            process_no_eq(vlmax);
        }
        if (i < n) {
            process_no_eq(__riscv_vsetvl_e16m2(n - i));
        }
        return wp;
    }

    auto process = [&](size_t vl) {
        vuint16m2_t v = __riscv_vle16_v_u16m2(vals + i, vl);
        vuint16m2_t vthr = __riscv_vmv_v_x_u16m2(thresh, vl);

        vbool8_t eqmask = __riscv_vmseq_vv_u16m2_b8(v, vthr, vl);
        vbool8_t ltmask;
        if constexpr (C::is_max) {
            ltmask = __riscv_vmsltu_vv_u16m2_b8(v, vthr, vl);
        } else {
            ltmask = __riscv_vmsgtu_vv_u16m2_b8(v, vthr, vl);
        }

        vbool8_t keep;
        if (n_eq > 0) {
            size_t e = __riscv_vcpop_m_b8(eqmask, vl);
            if ((int64_t)e <= n_eq) {
                keep = __riscv_vmor_mm_b8(ltmask, eqmask, vl);
                n_eq -= (int64_t)e;
            } else {
                // keep the first n_eq equal lanes of this block
                vuint16m2_t rank = __riscv_viota_m_u16m2(eqmask, vl);
                vbool8_t take =
                        __riscv_vmsltu_vx_u16m2_b8(rank, (uint16_t)n_eq, vl);
                vbool8_t eqkeep = __riscv_vmand_mm_b8(eqmask, take, vl);
                keep = __riscv_vmor_mm_b8(ltmask, eqkeep, vl);
                n_eq = 0;
            }
        } else {
            keep = ltmask;
        }

        size_t k = __riscv_vcpop_m_b8(keep, vl);

        vuint16m2_t vc = __riscv_vcompress_vm_u16m2(v, keep, vl);
        __riscv_vse16_v_u16m2(vals + wp, vc, vl);

        // ids: masked scatter of the kept lanes to their compressed
        // positions (base + rank * sizeof(IDT), RVV indices are byte
        // offsets). This avoids vcompress on the id tail.
        if constexpr (sizeof(IDT) == 4) {
            vuint32m4_t idv =
                    __riscv_vle32_v_u32m4((const uint32_t*)(ids + i), vl);
            vuint32m4_t idx = __riscv_viota_m_u32m4(keep, vl);
            idx = __riscv_vsll_vx_u32m4(idx, 2, vl);
            __riscv_vsoxei32_v_u32m4_m(
                    keep, (uint32_t*)(ids + wp), idx, idv, vl);
        } else {
            vuint64m8_t idv =
                    __riscv_vle64_v_u64m8((const uint64_t*)(ids + i), vl);
            vuint64m8_t idx = __riscv_viota_m_u64m8(keep, vl);
            idx = __riscv_vsll_vx_u64m8(idx, 3, vl);
            __riscv_vsoxei64_v_u64m8_m(
                    keep, (uint64_t*)(ids + wp), idx, idv, vl);
        }

        wp += k;
    };

    for (; i + vlmax <= n; i += vlmax) {
        process(vlmax);
    }
    if (i < n) {
        size_t vl = __riscv_vsetvl_e16m2(n - i);
        process(vl);
    }

    assert(n_eq == 0);
    return wp;
}

/******************************************************************
 * Partitioning driver (same search as partitioning_simdlib256.h)
 ******************************************************************/

template <class C>
uint16_t rvv_partition_fuzzy_with_bounds(
        uint16_t* vals,
        typename C::TI* ids,
        size_t n,
        size_t q_min,
        size_t q_max,
        size_t* q_out,
        uint16_t s0i,
        uint16_t s1i) {
    if (q_min == 0) {
        if (q_out) {
            *q_out = 0;
        }
        return 0;
    }
    if (q_max >= n) {
        if (q_out) {
            *q_out = q_max;
        }
        return 0xffff;
    }
    if (s0i == s1i) {
        if (q_out) {
            *q_out = q_min;
        }
        return s0i;
    }

    // lower bound inclusive, upper exclusive
    size_t s0 = s0i, s1 = size_t(s1i) + 1;

    int thresh = 0;
    size_t n_eq = 0, n_lt = 0;
    size_t q = 0;

    for (int it = 0; it < 200; it++) {
        thresh = (s0 + s1) / 2;
        // n_eq is only needed while the threshold sits below the target
        // quantile, so count it lazily (one extra pass instead of one extra
        // compare + vcpop on every bisection pass).
        n_lt = rvv_count_lt<C>(vals, n, (uint16_t)thresh);

        if (n_lt <= q_min) {
            n_eq = rvv_count_eq(vals, n, (uint16_t)thresh);
            if (n_lt + n_eq >= q_min) {
                q = q_min;
                break;
            } else {
                if (C::is_max) {
                    s0 = thresh;
                } else {
                    s1 = thresh;
                }
            }
        } else if (n_lt <= q_max) {
            q = n_lt;
            break;
        } else {
            if (C::is_max) {
                s1 = thresh;
            } else {
                s0 = thresh;
            }
        }
    }

    // number of equal values to keep
    int64_t n_eq_1 = (int64_t)q - (int64_t)n_lt;

    if (n_eq_1 < 0) { // happens when > q elements are at lower bound
        assert(s0 + 1 == s1);
        q = q_min;
        if (C::is_max) {
            thresh--;
        } else {
            thresh++;
        }
        n_eq_1 = q;
    } else {
        assert((size_t)n_eq_1 <= n_eq);
    }

    size_t wp = rvv_compress_array<C>(vals, ids, n, (uint16_t)thresh, n_eq_1);

    assert(wp == q);
    if (q_out) {
        *q_out = q;
    }

    return (uint16_t)thresh;
}

template <class C>
uint16_t rvv_partition_fuzzy(
        uint16_t* vals,
        typename C::TI* ids,
        size_t n,
        size_t q_min,
        size_t q_max,
        size_t* q_out) {
    // Start the bisection from the full 16-bit range instead of first
    // computing the exact min/max. This removes one full-array scan; the
    // bisection may pay at most one extra step for the wider initial range.
    return rvv_partition_fuzzy_with_bounds<C>(
            vals, ids, n, q_min, q_max, q_out, 0, 0xffff);
}

/******************************************************************
 * Histograms
 ******************************************************************/

// bins of (x - min) >> shift, values outside the range are ignored
template <int NBINS>
void rvv_histogram_bounded(
        const uint16_t* data,
        int n,
        uint16_t min,
        int shift,
        int* hist) {
    memset(hist, 0, sizeof(*hist) * NBINS);
    if (n <= 0) {
        return;
    }

    // inclusive upper bound of (x - min) for values that land in a bin
    uint32_t vmax0 = std::min<uint32_t>(
            ((uint32_t)NBINS << shift) + (uint32_t)min, 65536u);
    uint16_t vmax = (uint16_t)(vmax0 - 1u - (uint32_t)min);

    const size_t vlmax = rvv_vlmax_e16m8();
    const uint16_t out_of_range = (uint16_t)NBINS;

    auto process = [&](size_t i, size_t vl) {
        vuint16m8_t v = __riscv_vle16_v_u16m8(data + i, vl);
        vuint16m8_t diff = __riscv_vsub_vx_u16m8(v, min, vl);
        vbool2_t valid = __riscv_vmsleu_vx_u16m8_b2(diff, vmax, vl);
        vuint16m8_t bin = __riscv_vsrl_vx_u16m8(diff, (size_t)shift, vl);
        vuint16m8_t obin = __riscv_vmv_v_x_u16m8(out_of_range, vl);
        bin = __riscv_vmerge_vvm_u16m8(obin, bin, valid, vl);

        for (int b = 0; b < NBINS; b++) {
            vbool2_t mb = __riscv_vmseq_vx_u16m8_b2(bin, (uint16_t)b, vl);
            hist[b] += (int)__riscv_vcpop_m_b2(mb, vl);
        }
    };

    size_t i = 0;
    for (; i + vlmax <= (size_t)n; i += vlmax) {
        process(i, vlmax);
    }
    if (i < (size_t)n) {
        size_t vl = __riscv_vsetvl_e16m8((size_t)n - i);
        process(i, vl);
    }
}

// shift < 0: raw value histogram, values outside [0, NBINS) are ignored
template <int NBINS>
void rvv_histogram_unbounded(const uint16_t* data, int n, int* hist) {
    memset(hist, 0, sizeof(*hist) * NBINS);
    if (n <= 0) {
        return;
    }

    const size_t vlmax = rvv_vlmax_e16m8();
    auto process = [&](size_t i, size_t vl) {
        vuint16m8_t v = __riscv_vle16_v_u16m8(data + i, vl);
        for (int b = 0; b < NBINS; b++) {
            vbool2_t mb = __riscv_vmseq_vx_u16m8_b2(v, (uint16_t)b, vl);
            hist[b] += (int)__riscv_vcpop_m_b2(mb, vl);
        }
    };

    size_t i = 0;
    for (; i + vlmax <= (size_t)n; i += vlmax) {
        process(i, vlmax);
    }
    if (i < (size_t)n) {
        size_t vl = __riscv_vsetvl_e16m8((size_t)n - i);
        process(i, vl);
    }
}

void rvv_simd_histogram_8(
        const uint16_t* data,
        int n,
        uint16_t min,
        int shift,
        int* hist) {
    if (shift < 0) {
        rvv_histogram_unbounded<8>(data, n, hist);
    } else {
        rvv_histogram_bounded<8>(data, n, min, shift, hist);
    }
}

void rvv_simd_histogram_16(
        const uint16_t* data,
        int n,
        uint16_t min,
        int shift,
        int* hist) {
    if (shift < 0) {
        rvv_histogram_unbounded<16>(data, n, hist);
    } else {
        rvv_histogram_bounded<16>(data, n, min, shift, hist);
    }
}

} // anonymous namespace

/******************************************************************
 * Entry points called from partitioning.cpp
 ******************************************************************/

template <>
uint16_t partition_fuzzy_simd<SIMDLevel::RISCV_RVV, CMax<uint16_t, int64_t>>(
        uint16_t* vals,
        int64_t* ids,
        size_t n,
        size_t q_min,
        size_t q_max,
        size_t* q_out) {
    return rvv_partition_fuzzy<CMax<uint16_t, int64_t>>(
            vals, ids, n, q_min, q_max, q_out);
}

template <>
uint16_t partition_fuzzy_simd<SIMDLevel::RISCV_RVV, CMin<uint16_t, int64_t>>(
        uint16_t* vals,
        int64_t* ids,
        size_t n,
        size_t q_min,
        size_t q_max,
        size_t* q_out) {
    return rvv_partition_fuzzy<CMin<uint16_t, int64_t>>(
            vals, ids, n, q_min, q_max, q_out);
}

template <>
uint16_t partition_fuzzy_simd<SIMDLevel::RISCV_RVV, CMax<uint16_t, int>>(
        uint16_t* vals,
        int* ids,
        size_t n,
        size_t q_min,
        size_t q_max,
        size_t* q_out) {
    return rvv_partition_fuzzy<CMax<uint16_t, int>>(
            vals, ids, n, q_min, q_max, q_out);
}

template <>
uint16_t partition_fuzzy_simd<SIMDLevel::RISCV_RVV, CMin<uint16_t, int>>(
        uint16_t* vals,
        int* ids,
        size_t n,
        size_t q_min,
        size_t q_max,
        size_t* q_out) {
    return rvv_partition_fuzzy<CMin<uint16_t, int>>(
            vals, ids, n, q_min, q_max, q_out);
}

template <>
void simd_histogram_8<SIMDLevel::RISCV_RVV>(
        const uint16_t* data,
        int n,
        uint16_t min,
        int shift,
        int* hist) {
    rvv_simd_histogram_8(data, n, min, shift, hist);
}

template <>
void simd_histogram_16<SIMDLevel::RISCV_RVV>(
        const uint16_t* data,
        int n,
        uint16_t min,
        int shift,
        int* hist) {
    rvv_simd_histogram_16(data, n, min, shift, hist);
}

} // namespace faiss

#endif // COMPILE_SIMD_RISCV_RVV
