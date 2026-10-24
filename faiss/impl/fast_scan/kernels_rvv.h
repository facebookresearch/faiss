/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#if !defined(__riscv) || !defined(__riscv_vector)
#error "kernels_rvv.h requires a RISC-V compiler with the vector extension"
#endif

/** Native RVV fast-scan kernels.
 *
 * Unlike kernels_simd256.h these implementations do not go through the
 * 256-bit simdlib abstraction and do not use any GNU VLS (vector_size)
 * types. The logical fast-scan block is still 32 database vectors with
 * 4-bit PQ codes (that is the on-disk format and is shared with the other
 * SIMD levels), but the computation is expressed with native RVV
 * intrinsics and a fixed vl that is valid for every VLEN >= 128:
 *
 *   - 32 x u8  : e8m2, vl = 32   (VLMAX = 32 at VLEN=128, 64 at VLEN=256)
 *   - 16 x u16 : e16m2, vl = 16
 *
 * At VLEN=128 the e8m2 group is fully used; at VLEN >= 256 the same code
 * is still correct (the group has spare lanes) and needs no compiler
 * support for VLS sub-vector extraction, which is where GCC < 14.3
 * miscompiles on VLEN > 128 boards.
 *
 * Covered paths:
 *   - pq4_kernel_qbs:           QBS path, one block of 32 vectors (bbs=32)
 *   - kernel_accumulate_block:  search_1 path, NQ x BB blocks of 32
 * Both the plain (DummyScaler) and the AdditiveQuantizer norm-table
 * scaled (NormTableScaler) variants are implemented natively.
 */

#include <cstdint>

#include <riscv_vector.h>

namespace faiss {
namespace rvv_fast_scan {

// Number of database vectors per block (fast-scan format constant).
constexpr size_t kBlockSize = 32;
// Number of u16 accumulator lanes per block (32 u8 = 16 u16).
constexpr size_t kAccuLanes = 16;

// Look up one 4-bit code per lane in a 32-entry LUT built from the two
// 16-entry tables of a sub-quantizer pair.
//
// tbl[j]  = distance of code j for the first sub-quantizer  (j = 0..15)
// tbl[j]  = distance of code j for the second sub-quantizer (j = 16..31)
// lane j picks table (j >= 16), i.e. index bit 4 is the table selector,
// exactly like AVX2 pshufb's per-128-bit-lane lookup.
inline vuint8m2_t lut_gather32(vuint8m2_t tbl, vuint8m2_t idx) {
    constexpr size_t vl = kBlockSize;
    vuint8m2_t lane = __riscv_vid_v_u8m2(vl);
    vuint8m2_t g = __riscv_vand_vx_u8m2(idx, 0x0f, vl);
    g = __riscv_vor_vv_u8m2(g, __riscv_vand_vx_u8m2(lane, 0x10, vl), vl);
    return __riscv_vrgather_vv_u8m2(tbl, g, vl);
}

// Build the 32-lane LUT vector from two 16-entry tables.
inline vuint8m2_t make_lut32(const uint8_t* lut) {
    constexpr size_t vl = kBlockSize;
    vuint8m1_t lo = __riscv_vle8_v_u8m1(lut, 16);
    vuint8m1_t hi = __riscv_vle8_v_u8m1(lut + 16, 16);
    vuint8m2_t tbl = __riscv_vlmul_ext_v_u8m1_u8m2(lo);
    vuint8m2_t hi2 = __riscv_vlmul_ext_v_u8m1_u8m2(hi);
    return __riscv_vslideup_vx_u8m2(tbl, hi2, 16, vl);
}

// Low / high nibble of each code byte, as gather indices.
inline vuint8m2_t code_low_nibble(vuint8m2_t c) {
    return __riscv_vand_vx_u8m2(c, 0x0f, kBlockSize);
}
inline vuint8m2_t code_high_nibble(vuint8m2_t c) {
    return __riscv_vand_vx_u8m2(
            __riscv_vsrl_vx_u8m2(c, 4, kBlockSize), 0x0f, kBlockSize);
}

// accu0 += res, accu1 += res >> 8, accu2 += res1, accu3 += res1 >> 8,
// mirroring simd16uint16(res) / simd16uint16(res) >> 8 in the generic
// kernel (32 u8 reinterpreted as 16 u16 lanes).
//
// Scaled == true is the NormTableScaler (AdditiveQuantizer) path:
//   accu0 += simd16uint16(res) * scale
//   accu1 += (simd16uint16(res) >> 8) * scale
// with u16 wraparound, exactly like scale_lo / scale_hi.
template <bool Scaled>
inline void accumulate_lookup(
        vuint8m2_t res0,
        vuint8m2_t res1,
        vuint16m2_t& accu0,
        vuint16m2_t& accu1,
        vuint16m2_t& accu2,
        vuint16m2_t& accu3,
        uint16_t scale = 0) {
    constexpr size_t vl = kAccuLanes;
    vuint16m2_t r0 = __riscv_vreinterpret_v_u8m2_u16m2(res0);
    vuint16m2_t r1 = __riscv_vreinterpret_v_u8m2_u16m2(res1);
    vuint16m2_t h0 = __riscv_vsrl_vx_u16m2(r0, 8, vl);
    vuint16m2_t h1 = __riscv_vsrl_vx_u16m2(r1, 8, vl);
    if constexpr (Scaled) {
        r0 = __riscv_vmul_vx_u16m2(r0, scale, vl);
        h0 = __riscv_vmul_vx_u16m2(h0, scale, vl);
        r1 = __riscv_vmul_vx_u16m2(r1, scale, vl);
        h1 = __riscv_vmul_vx_u16m2(h1, scale, vl);
    }
    accu0 = __riscv_vadd_vv_u16m2(accu0, r0, vl);
    accu1 = __riscv_vadd_vv_u16m2(accu1, h0, vl);
    accu2 = __riscv_vadd_vv_u16m2(accu2, r1, vl);
    accu3 = __riscv_vadd_vv_u16m2(accu3, h1, vl);
}

inline void clear_accumulators(
        vuint16m2_t& accu0,
        vuint16m2_t& accu1,
        vuint16m2_t& accu2,
        vuint16m2_t& accu3) {
    constexpr size_t vl = kAccuLanes;
    accu0 = __riscv_vmv_v_x_u16m2(0, vl);
    accu1 = __riscv_vmv_v_x_u16m2(0, vl);
    accu2 = __riscv_vmv_v_x_u16m2(0, vl);
    accu3 = __riscv_vmv_v_x_u16m2(0, vl);
}

// combine2x2: a = (a0, a1), b = (b0, b1) with 8-lane halves;
// returns (a0 + a1, b0 + b1) — the same contract as faiss::combine2x2 for
// the 256-bit type, expressed on the 16-lane e16m2 value.
inline vuint16m2_t combine2x2(vuint16m2_t a, vuint16m2_t b) {
    constexpr size_t vl = kAccuLanes;
    vuint16m2_t ah = __riscv_vslidedown_vx_u16m2(a, 8, vl);
    vuint16m2_t bh = __riscv_vslidedown_vx_u16m2(b, 8, vl);
    vuint16m2_t sa = __riscv_vadd_vv_u16m2(a, ah, vl);
    vuint16m2_t sb = __riscv_vadd_vv_u16m2(b, bh, vl);
    return __riscv_vslideup_vx_u16m2(sa, sb, 8, vl);
}

// accu0 -= accu1 << 8, then combine each pair; hand the 32-byte result to
// the (simdlib-typed) result handler. The union is used purely as a
// 32-byte container here; the stores are RVV intrinsics, no VLS operation
// is involved.
template <class SIMD16>
inline void finish_block(
        vuint16m2_t accu0,
        vuint16m2_t accu1,
        vuint16m2_t accu2,
        vuint16m2_t accu3,
        SIMD16& h0,
        SIMD16& h1) {
    constexpr size_t vl = kAccuLanes;
    accu0 = __riscv_vsub_vv_u16m2(
            accu0, __riscv_vsll_vx_u16m2(accu1, 8, vl), vl);
    accu2 = __riscv_vsub_vv_u16m2(
            accu2, __riscv_vsll_vx_u16m2(accu3, 8, vl), vl);
    vuint16m2_t dis0 = combine2x2(accu0, accu1);
    vuint16m2_t dis1 = combine2x2(accu2, accu3);
    __riscv_vse16_v_u16m2(reinterpret_cast<uint16_t*>(&h0), dis0, vl);
    __riscv_vse16_v_u16m2(reinterpret_cast<uint16_t*>(&h1), dis1, vl);
}

// QBS kernel for the non-scaled (DummyScaler) path.
//
// Semantics match kernels_simd256.h::pq4_kernel_qbs_256 exactly:
//   - codes: 32 bytes per sub-quantizer pair (low nibble = sq, high = sq+1)
//   - LUT:   (sq_pair, q) interleaved rows of 32 bytes
//   - accu[j] keeps the low / high byte sums, then accu0 -= accu1 << 8
//   - combine2x2 folds each 128-bit half and the handler receives dis0/dis1
template <int NQ, class Scaler, class ResultHandler>
inline void pq4_kernel_qbs(
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        ResultHandler& res,
        const Scaler& scaler) {
    if constexpr (NQ > 0) {
        constexpr int nscale = Scaler::nscale;
        // The RVV intrinsic types cannot be array elements, so iterate the
        // queries in the outer loop. codes are re-read for every query
        // (32 B per sub-quantizer pair, L1-resident).
        for (int q = 0; q < NQ; q++) {
            const uint8_t* cp = codes;
            const uint8_t* lp = LUT + (size_t)q * 32;

            vuint16m2_t accu0, accu1, accu2, accu3;
            clear_accumulators(accu0, accu1, accu2, accu3);

            for (int sq = 0; sq < nsq - nscale; sq += 2) {
                vuint8m2_t c = __riscv_vle8_v_u8m2(cp, kBlockSize);
                cp += 32;

                vuint8m2_t tbl = make_lut32(lp);
                lp += (size_t)NQ * 32;

                vuint8m2_t res0 = lut_gather32(tbl, code_low_nibble(c));
                vuint8m2_t res1 = lut_gather32(tbl, code_high_nibble(c));
                accumulate_lookup<false>(
                        res0, res1, accu0, accu1, accu2, accu3);
            }

            // AdditiveQuantizer norm-table pair (NormTableScaler).
            if constexpr (nscale > 0) {
                const uint16_t scale = (uint16_t)scaler.scale_int;
                for (int sq = 0; sq < nscale; sq += 2) {
                    vuint8m2_t c = __riscv_vle8_v_u8m2(cp, kBlockSize);
                    cp += 32;

                    vuint8m2_t tbl = make_lut32(lp);
                    lp += (size_t)NQ * 32;

                    vuint8m2_t res0 = lut_gather32(tbl, code_low_nibble(c));
                    vuint8m2_t res1 = lut_gather32(tbl, code_high_nibble(c));
                    accumulate_lookup<true>(
                            res0, res1, accu0, accu1, accu2, accu3, scale);
                }
            }

            typename ResultHandler::simd16uint16 h0, h1;
            finish_block(accu0, accu1, accu2, accu3, h0, h1);
            res.handle(q, 0, h0, h1);
        }
    }
}

// search_1 kernel (multi-BB) for the non-scaled (DummyScaler) path.
//
// Semantics match kernels_simd256.h::kernel_accumulate_block exactly:
//   - for each sub-quantizer pair: NQ LUT rows of 32 bytes, then BB code
//     blocks of 32 bytes
//   - handler receives (q, b, dis0, dis1)
//
// The generic kernel caches the NQ LUT rows across BB. Native RVV types
// cannot live in arrays, so the loops are restructured as q -> b -> sq;
// this re-reads 32 B of codes per query and 32 B of LUT per block, all
// L1-resident.
template <int NQ, int BB, class Scaler, class ResultHandler>
inline void kernel_accumulate_block(
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        ResultHandler& res,
        const Scaler& scaler) {
    if constexpr (NQ > 0 && BB > 0) {
        constexpr int nscale = Scaler::nscale;
        for (int q = 0; q < NQ; q++) {
            for (int b = 0; b < BB; b++) {
                const uint8_t* cp = codes + (size_t)b * 32;
                const uint8_t* lp = LUT + (size_t)q * 32;

                vuint16m2_t accu0, accu1, accu2, accu3;
                clear_accumulators(accu0, accu1, accu2, accu3);

                for (int sq = 0; sq < nsq - nscale; sq += 2) {
                    vuint8m2_t c = __riscv_vle8_v_u8m2(cp, kBlockSize);
                    // BB code blocks per sub-quantizer pair.
                    cp += (size_t)BB * 32;

                    vuint8m2_t tbl = make_lut32(lp);
                    // NQ LUT rows per sub-quantizer pair.
                    lp += (size_t)NQ * 32;

                    vuint8m2_t res0 = lut_gather32(tbl, code_low_nibble(c));
                    vuint8m2_t res1 = lut_gather32(tbl, code_high_nibble(c));
                    accumulate_lookup<false>(
                            res0, res1, accu0, accu1, accu2, accu3);
                }

                if constexpr (nscale > 0) {
                    const uint16_t scale = (uint16_t)scaler.scale_int;
                    for (int sq = 0; sq < nscale; sq += 2) {
                        vuint8m2_t c = __riscv_vle8_v_u8m2(cp, kBlockSize);
                        cp += (size_t)BB * 32;

                        vuint8m2_t tbl = make_lut32(lp);
                        lp += (size_t)NQ * 32;

                        vuint8m2_t res0 = lut_gather32(tbl, code_low_nibble(c));
                        vuint8m2_t res1 =
                                lut_gather32(tbl, code_high_nibble(c));
                        accumulate_lookup<true>(
                                res0, res1, accu0, accu1, accu2, accu3, scale);
                    }
                }

                typename ResultHandler::simd16uint16 h0, h1;
                finish_block(accu0, accu1, accu2, accu3, h0, h1);
                res.handle(q, b, h0, h1);
            }
        }
    }
}

} // namespace rvv_fast_scan
} // namespace faiss
