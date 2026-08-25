/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_RISCV_RVV

#include <faiss/impl/scalar_quantizer/codecs.h>
#include <faiss/impl/scalar_quantizer/distance_computers.h>
#include <faiss/impl/scalar_quantizer/quantizers.h>
#include <faiss/impl/scalar_quantizer/scanners.h>
#include <faiss/impl/scalar_quantizer/similarities.h>

#include <riscv_vector.h>
#include <cmath>

namespace faiss {

namespace scalar_quantizer {

/*************************************************************************
 * Marker specializations.
 *
 * Unlike x86/NEON sq-*.cpp files that expose a fixed 8-wide / 16-wide codec
 * interface (reconstruct_8_components / reconstruct_16_components), RVV is
 * variable-width: the native vector length is implementation-defined and
 * queried at runtime via __riscv_vsetvl. Forcing RVV into a fixed-width
 * codec would leave performance on the table on wider hardware.
 *
 * So the strategy here is: Codec / Quantizer / Similarity classes for
 * RISCV_RVV act as opaque TAG TYPES — they only need to be complete types
 * so that baseline's sq-dispatch.h can form template arguments like
 * `DCTemplate<QuantizerTemplate<Codec4bit<RISCV_RVV>, UNIFORM, RISCV_RVV>,
 *             SimilarityL2<RISCV_RVV>, RISCV_RVV>`.
 *
 * The real SIMD work lives in full DCTemplate specializations below.
 * Unspecialized combinations fall through to scalar via the fallback
 * `DCTemplate<Q, Sim, RISCV_RVV> : DCTemplate<Q, Sim, NONE>`.
 ************************************************************************/

template <>
struct Codec8bit<SIMDLevel::RISCV_RVV> : Codec8bit<SIMDLevel::NONE> {};

template <>
struct Codec4bit<SIMDLevel::RISCV_RVV> : Codec4bit<SIMDLevel::NONE> {};

template <>
struct Codec6bit<SIMDLevel::RISCV_RVV> : Codec6bit<SIMDLevel::NONE> {};

template <class Codec>
struct QuantizerTemplate<
        Codec,
        QuantizerTemplateScaling::UNIFORM,
        SIMDLevel::RISCV_RVV>
        : QuantizerTemplate<
                  Codec,
                  QuantizerTemplateScaling::UNIFORM,
                  SIMDLevel::NONE> {
    QuantizerTemplate(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate<
                      Codec,
                      QuantizerTemplateScaling::UNIFORM,
                      SIMDLevel::NONE>(d, trained) {}
};

template <class Codec>
struct QuantizerTemplate<
        Codec,
        QuantizerTemplateScaling::NON_UNIFORM,
        SIMDLevel::RISCV_RVV>
        : QuantizerTemplate<
                  Codec,
                  QuantizerTemplateScaling::NON_UNIFORM,
                  SIMDLevel::NONE> {
    QuantizerTemplate(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate<
                      Codec,
                      QuantizerTemplateScaling::NON_UNIFORM,
                      SIMDLevel::NONE>(d, trained) {}
};

template <>
struct QuantizerFP16<SIMDLevel::RISCV_RVV> : QuantizerFP16<SIMDLevel::NONE> {
    QuantizerFP16(size_t d, const std::vector<float>& trained)
            : QuantizerFP16<SIMDLevel::NONE>(d, trained) {}
};

template <>
struct QuantizerBF16<SIMDLevel::RISCV_RVV> : QuantizerBF16<SIMDLevel::NONE> {
    QuantizerBF16(size_t d, const std::vector<float>& trained)
            : QuantizerBF16<SIMDLevel::NONE>(d, trained) {}
};

template <>
struct Quantizer8bitDirect<SIMDLevel::RISCV_RVV>
        : Quantizer8bitDirect<SIMDLevel::NONE> {
    Quantizer8bitDirect(size_t d, const std::vector<float>& trained)
            : Quantizer8bitDirect<SIMDLevel::NONE>(d, trained) {}
};

template <>
struct Quantizer8bitDirectSigned<SIMDLevel::RISCV_RVV>
        : Quantizer8bitDirectSigned<SIMDLevel::NONE> {
    Quantizer8bitDirectSigned(size_t d, const std::vector<float>& trained)
            : Quantizer8bitDirectSigned<SIMDLevel::NONE>(d, trained) {}
};

template <>
struct SimilarityL2<SIMDLevel::RISCV_RVV> : SimilarityL2<SIMDLevel::NONE> {
    using SimilarityL2<SIMDLevel::NONE>::SimilarityL2;
};

template <>
struct SimilarityIP<SIMDLevel::RISCV_RVV> : SimilarityIP<SIMDLevel::NONE> {
    using SimilarityIP<SIMDLevel::NONE>::SimilarityIP;
};

/*************************************************************************
 * Fallback DCTemplate / DistanceComputerByte for RISCV_RVV.
 *
 * Inheriting from the NONE specialization means every (Quantizer, Similarity)
 * combination that does NOT have a hand-tuned RVV full specialization below
 * falls through to scalar code. Callers and the dispatcher don't know or care.
 ************************************************************************/

template <class Quantizer, class Similarity>
struct DCTemplate<Quantizer, Similarity, SIMDLevel::RISCV_RVV>
        : DCTemplate<Quantizer, Similarity, SIMDLevel::NONE> {
    using Base = DCTemplate<Quantizer, Similarity, SIMDLevel::NONE>;
    using Base::Base;
};

template <class Similarity>
struct DistanceComputerByte<Similarity, SIMDLevel::RISCV_RVV>
        : DistanceComputerByte<Similarity, SIMDLevel::NONE> {
    using Base = DistanceComputerByte<Similarity, SIMDLevel::NONE>;
    using Base::Base;
};

//  * Fast path — QT_4bit_uniform + L2
//  *
//  * 4-bit UNIFORM scaling: every component reconstructs as an affine function
//  * of the 4-bit code,
//  *     recon(c) = vmin + vdiff * (c + 0.5) / 15 = final_scale * c + bias
//  * where final_scale = vdiff / 15. L2 distance between two reconstructions
//  * therefore reduces to final_scale^2 * (q_c - c_c)^2 over integer codes,
//  * so we can stay in the int domain and pay one float multiply at the end.
//  *
//  * The RVV path pre-nibbles the query into q_lo / q_hi (even / odd lanes)
//  * once at set_query time and then processes native-VL-sized chunks of code
//  * without ever decoding to float.
//  ************************************************************************/

template <>
struct DCTemplate<
        QuantizerTemplate<
                Codec4bit<SIMDLevel::RISCV_RVV>,
                QuantizerTemplateScaling::UNIFORM,
                SIMDLevel::RISCV_RVV>,
        SimilarityL2<SIMDLevel::RISCV_RVV>,
        SIMDLevel::RISCV_RVV> : SQDistanceComputer {
    using Sim = SimilarityL2<SIMDLevel::RISCV_RVV>;

    size_t d;
    float vmin;
    float vdiff;
    float final_scale_sq;
    std::vector<uint8_t> q_lo;
    std::vector<uint8_t> q_hi;

    DCTemplate(size_t d_in, const std::vector<float>& trained)
            : d(d_in),
              vmin(trained[0]),
              vdiff(trained[1]),
              q_lo((d_in + 1) / 2, 0),
              q_hi((d_in + 1) / 2, 0) {
        const float final_scale = vdiff / 15.0f;
        final_scale_sq = final_scale * final_scale;
    }

    void set_query(const float* x) final {
        this->q = x;
        const float inv_scale = (vdiff == 0.0f) ? 0.0f : 15.0f / vdiff;
        for (size_t i = 0; i < d; i++) {
            float val = (x[i] - vmin) * inv_scale;
            int code = static_cast<int>(val);
            if (code < 0) {
                code = 0;
            }
            if (code > 15) {
                code = 15;
            }
            if (i % 2 == 0) {
                q_lo[i / 2] = static_cast<uint8_t>(code);
            } else {
                q_hi[i / 2] = static_cast<uint8_t>(code);
            }
        }
    }

    /// Squared integer-domain L2 between pre-nibbled q and packed 4-bit code.
    /// Returns the raw integer sum — caller multiplies by final_scale_sq.
    ///
    /// ID1: byte-domain loop over nb = ceil(d/2) code bytes with vsetvl
    /// hoisted (0 vsetvl inside the hot loop), a cross-iteration vector
    /// accumulator, and a SINGLE horizontal reduction after the loop.
    /// e8m2 fills all lanes (32B per iteration at VLEN=128).
    ///
    /// ID3: signed-domain squaring — |q-c|^2 == (q-c)^2, so the 3-op
    /// absdiff (vmaxu/vminu/vsub) per nibble stream is replaced by one
    /// widening subtract (vwsubu: zext operands, signed i16 result) and
    /// the square-accumulate becomes vwmacc (i16*i16 += i32). The i32
    /// accumulator cannot overflow for any realistic d, which removes
    /// the u16 strip-mine flush entirely — one flat loop.
    int64_t accumulate_int_l2(const uint8_t* code) const {
        const size_t nb = (d + 1) / 2; // bytes per code
        // Odd d: the last byte's high nibble is padding. The standard
        // encoder zeroes it and q_hi's tail slot is 0, but to stay exact
        // for any producer we process that byte's low nibble in scalar.
        const size_t nbv = (d & 1) ? (nb - 1) : nb; // vector-domain bytes
        int64_t acc = 0;
        size_t b = 0;

        // Hoist vsetvl: full VLMAX for e8m2, reused across the hot loop.
        const size_t vlb = __riscv_vsetvl_e8m2(nbv > 0 ? nbv : 1);
        const uint8_t* qlp = q_lo.data();
        const uint8_t* qhp = q_hi.data();

        vint32m8_t acc32 = __riscv_vmv_v_x_i32m8(0, vlb);

        // Hot loop: 3 loads, nibble split, two widening subtracts, two
        // fused square-accumulates. No vsetvl, no reduction inside.
        for (; b + vlb <= nbv; b += vlb) {
            vuint8m2_t packed = __riscv_vle8_v_u8m2(code + b, vlb);
            vuint8m2_t ql = __riscv_vle8_v_u8m2(qlp + b, vlb);
            vuint8m2_t qh = __riscv_vle8_v_u8m2(qhp + b, vlb);

            vuint8m2_t lo_nib = __riscv_vand_vx_u8m2(packed, 0x0F, vlb);
            vuint8m2_t hi_nib = __riscv_vsrl_vx_u8m2(packed, 4, vlb);

            // Signed differences in i16 (values in [-15, 15]); vwsubu
            // returns the 2's-complement bit pattern as u16 — reinterpret.
            vint16m4_t d_lo = __riscv_vreinterpret_v_u16m4_i16m4(
                    __riscv_vwsubu_vv_u16m4(ql, lo_nib, vlb));
            vint16m4_t d_hi = __riscv_vreinterpret_v_u16m4_i16m4(
                    __riscv_vwsubu_vv_u16m4(qh, hi_nib, vlb));

            acc32 = __riscv_vwmacc_vv_i32m8(acc32, d_lo, d_lo, vlb);
            acc32 = __riscv_vwmacc_vv_i32m8(acc32, d_hi, d_hi, vlb);
        }

        // Tail: fewer than vlb bytes left — one shorter-vl pass into the
        // same accumulator (only the first vt lanes are touched).
        if (b < nbv) {
            const size_t vt = __riscv_vsetvl_e8m2(nbv - b);

            vuint8m2_t packed = __riscv_vle8_v_u8m2(code + b, vt);
            vuint8m2_t ql = __riscv_vle8_v_u8m2(qlp + b, vt);
            vuint8m2_t qh = __riscv_vle8_v_u8m2(qhp + b, vt);

            vuint8m2_t lo_nib = __riscv_vand_vx_u8m2(packed, 0x0F, vt);
            vuint8m2_t hi_nib = __riscv_vsrl_vx_u8m2(packed, 4, vt);

            vint16m4_t d_lo = __riscv_vreinterpret_v_u16m4_i16m4(
                    __riscv_vwsubu_vv_u16m4(ql, lo_nib, vt));
            vint16m4_t d_hi = __riscv_vreinterpret_v_u16m4_i16m4(
                    __riscv_vwsubu_vv_u16m4(qh, hi_nib, vt));

            acc32 = __riscv_vwmacc_vv_i32m8(acc32, d_lo, d_lo, vt);
            acc32 = __riscv_vwmacc_vv_i32m8(acc32, d_hi, d_hi, vt);
        }

        // Single horizontal reduction over all vlb lanes.
        vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
        vint32m1_t red = __riscv_vredsum_vs_i32m8_i32m1(acc32, zero, vlb);
        acc += __riscv_vmv_x_s_i32m1_i32(red);

        // Odd d: last dimension lives in the low nibble of the last byte.
        if (d & 1) {
            int diff = int(q_lo[nb - 1]) - int(code[nb - 1] & 0x0F);
            acc += diff * diff;
        }
        return acc;
    }

    float query_to_code(const uint8_t* code) const final {
        return static_cast<float>(accumulate_int_l2(code)) * final_scale_sq;
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        // Not on the critical path for most workloads; reconstruct both
        // codes into nibbles scalar-style and compute squared distance.
        const uint8_t* c1 = codes + i * code_size;
        const uint8_t* c2 = codes + j * code_size;
        int64_t acc = 0;
        for (size_t k = 0; k < d; k++) {
            uint8_t a = (k % 2 == 0) ? (c1[k / 2] & 0x0F) : (c1[k / 2] >> 4);
            uint8_t b = (k % 2 == 0) ? (c2[k / 2] & 0x0F) : (c2[k / 2] >> 4);
            int diff = int(a) - int(b);
            acc += diff * diff;
        }
        return static_cast<float>(acc) * final_scale_sq;
    }

    void query_to_codes_batch_4(
            const uint8_t* code_0,
            const uint8_t* code_1,
            const uint8_t* code_2,
            const uint8_t* code_3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) const final {
        // Simple 4x unroll of the single-code path; good enough as a first
        // cut — gives ILP across the four independent accumulate loops.
        dis0 = static_cast<float>(accumulate_int_l2(code_0)) * final_scale_sq;
        dis1 = static_cast<float>(accumulate_int_l2(code_1)) * final_scale_sq;
        dis2 = static_cast<float>(accumulate_int_l2(code_2)) * final_scale_sq;
        dis3 = static_cast<float>(accumulate_int_l2(code_3)) * final_scale_sq;
    }
};

//  * Fast path — QT_4bit (NON_UNIFORM) + L2
//  *
//  * Per-dimension affine reconstruction:
//  *     recon_i(c) = vmin[i] + vdiff[i] * (c + 0.5) / 15
//  *                = rmin_i + a_i * c,  with a_i = vdiff[i] / 15,
//  *                  rmin_i = vmin[i] + 0.5 * a_i
//  * L2 contribution per dim: (q_i - recon_i)^2 = (e_i - a_i * c_i)^2
//  * where e_i = q_i - rmin_i is precomputed once per query in set_query.
//  *
//  * Unlike the UNIFORM variant (above) the scale a_i differs per dimension,
//  * so the sum cannot be pulled out of the integer domain — the kernel stays
//  * in f32 and evaluates one fused multiply-sub + one fused multiply-add per
//  * nibble stream.
//  *
//  * The packed 4-bit code interleaves dimensions (even dim -> low nibble,
//  * odd dim -> high nibble), so the constructor deinterleaves a_i / rmin_i
//  * into _lo/_hi streams once, and set_query does the same for e_i. The hot
//  * loop then works on contiguous streams: per 16-byte code chunk (32 dims,
//  * e8m1 -> f32m4) it unpacks nibbles, converts to f32 and accumulates
//  * squared differences. vsetvl is hoisted (0 explicit vsetvl in the loop).
//  *
//  * Padding: for odd d the last byte's high nibble is not a real dimension;
//  * a_hi/e_hi keep 0 in that slot so its contribution is exactly 0.
//  ************************************************************************/

template <>
struct DCTemplate<
        QuantizerTemplate<
                Codec4bit<SIMDLevel::RISCV_RVV>,
                QuantizerTemplateScaling::NON_UNIFORM,
                SIMDLevel::RISCV_RVV>,
        SimilarityL2<SIMDLevel::RISCV_RVV>,
        SIMDLevel::RISCV_RVV> : SQDistanceComputer {
    using Sim = SimilarityL2<SIMDLevel::RISCV_RVV>;

    size_t d;
    size_t half; // bytes per code = ceil(d/2)
    std::vector<float> a_lo, a_hi;       // a_i = vdiff[i]/15, deinterleaved
    std::vector<float> rmin_lo, rmin_hi; // vmin[i] + 0.5*a_i, deinterleaved
    std::vector<float> e_lo, e_hi;       // q_i - rmin_i, per query

    DCTemplate(size_t d_in, const std::vector<float>& trained)
            : d(d_in),
              half((d_in + 1) / 2),
              a_lo(half, 0.0f),
              a_hi(half, 0.0f),
              rmin_lo(half, 0.0f),
              rmin_hi(half, 0.0f),
              e_lo(half, 0.0f),
              e_hi(half, 0.0f) {
        const float* vmin = trained.data();
        const float* vdiff = trained.data() + d_in;
        for (size_t i = 0; i < d_in; i++) {
            float a = vdiff[i] / 15.0f;
            float rmin = vmin[i] + 0.5f * a;
            if (i % 2 == 0) {
                a_lo[i / 2] = a;
                rmin_lo[i / 2] = rmin;
            } else {
                a_hi[i / 2] = a;
                rmin_hi[i / 2] = rmin;
            }
        }
    }

    void set_query(const float* x) final {
        q = x;
        // e_i = q_i - rmin_i, deinterleaved. Padding slot (odd d) keeps 0.
        for (size_t i = 0; i < d; i++) {
            if (i % 2 == 0) {
                e_lo[i / 2] = x[i] - rmin_lo[i / 2];
            } else {
                e_hi[i / 2] = x[i] - rmin_hi[i / 2];
            }
        }
    }

    /// Full-precision vector L2 over the packed code.
    /// Hot loop body per byte chunk: unpack nibbles (e8), widen+convert to
    /// f32, then t = e - a*c (vfnmsac) and acc += t*t (vfmacc).
    float compute_l2(const uint8_t* code) const {
        const size_t nb = half; // total bytes to process
        size_t b = 0;

        // Hoist vsetvl: VLMAX for e8m1 (== f32 lanes per m4 group chunk)
        size_t vlb = __riscv_vsetvl_e8m1(nb);
        vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vlb);

        for (; b + vlb <= nb; b += vlb) {
            vuint8m1_t packed = __riscv_vle8_v_u8m1(code + b, vlb);
            vuint8m1_t lo = __riscv_vand_vx_u8m1(packed, 0x0F, vlb);
            vuint8m1_t hi = __riscv_vsrl_vx_u8m1(packed, 4, vlb);

            // Low-nibble stream (even dims)
            vfloat32m4_t clo = __riscv_vfcvt_f_xu_v_f32m4(
                    __riscv_vzext_vf4_u32m4(lo, vlb), vlb);
            vfloat32m4_t t_lo = __riscv_vle32_v_f32m4(e_lo.data() + b, vlb);
            t_lo = __riscv_vfnmsac_vv_f32m4(
                    t_lo,
                    __riscv_vle32_v_f32m4(a_lo.data() + b, vlb),
                    clo,
                    vlb);
            acc = __riscv_vfmacc_vv_f32m4(acc, t_lo, t_lo, vlb);

            // High-nibble stream (odd dims)
            vfloat32m4_t chi = __riscv_vfcvt_f_xu_v_f32m4(
                    __riscv_vzext_vf4_u32m4(hi, vlb), vlb);
            vfloat32m4_t t_hi = __riscv_vle32_v_f32m4(e_hi.data() + b, vlb);
            t_hi = __riscv_vfnmsac_vv_f32m4(
                    t_hi,
                    __riscv_vle32_v_f32m4(a_hi.data() + b, vlb),
                    chi,
                    vlb);
            acc = __riscv_vfmacc_vv_f32m4(acc, t_hi, t_hi, vlb);
        }

        // Tail: fewer than vlb bytes left; accumulates into the first
        // lanes of acc (safe: reduction below covers all vlb lanes).
        if (b < nb) {
            size_t vt = __riscv_vsetvl_e8m1(nb - b);

            vuint8m1_t packed = __riscv_vle8_v_u8m1(code + b, vt);
            vuint8m1_t lo = __riscv_vand_vx_u8m1(packed, 0x0F, vt);
            vuint8m1_t hi = __riscv_vsrl_vx_u8m1(packed, 4, vt);

            vfloat32m4_t clo = __riscv_vfcvt_f_xu_v_f32m4(
                    __riscv_vzext_vf4_u32m4(lo, vt), vt);
            vfloat32m4_t t_lo = __riscv_vle32_v_f32m4(e_lo.data() + b, vt);
            t_lo = __riscv_vfnmsac_vv_f32m4(
                    t_lo,
                    __riscv_vle32_v_f32m4(a_lo.data() + b, vt),
                    clo,
                    vt);
            acc = __riscv_vfmacc_vv_f32m4(acc, t_lo, t_lo, vt);

            vfloat32m4_t chi = __riscv_vfcvt_f_xu_v_f32m4(
                    __riscv_vzext_vf4_u32m4(hi, vt), vt);
            vfloat32m4_t t_hi = __riscv_vle32_v_f32m4(e_hi.data() + b, vt);
            t_hi = __riscv_vfnmsac_vv_f32m4(
                    t_hi,
                    __riscv_vle32_v_f32m4(a_hi.data() + b, vt),
                    chi,
                    vt);
            acc = __riscv_vfmacc_vv_f32m4(acc, t_hi, t_hi, vt);
        }

        // Horizontal reduce over all vlb lanes
        vfloat32m1_t red = __riscv_vfredusum_vs_f32m4_f32m1(
                acc, __riscv_vfmv_v_f_f32m1(0.0f, 1), vlb);
        return __riscv_vfmv_f_s_f32m1_f32(red);
    }

    float query_to_code(const uint8_t* code) const final {
        return compute_l2(code);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        // Not on the benchmark-critical path; scalar per-dim evaluation.
        // recon1 - recon2 = a_k * (c1_k - c2_k)
        const uint8_t* c1 = codes + i * code_size;
        const uint8_t* c2 = codes + j * code_size;
        float acc = 0;
        for (size_t k = 0; k < d; k++) {
            uint8_t n1 = (k % 2 == 0) ? (c1[k / 2] & 0x0F) : (c1[k / 2] >> 4);
            uint8_t n2 = (k % 2 == 0) ? (c2[k / 2] & 0x0F) : (c2[k / 2] >> 4);
            float a = (k % 2 == 0) ? a_lo[k / 2] : a_hi[k / 2];
            float diff = a * float(int(n1) - int(n2));
            acc += diff * diff;
        }
        return acc;
    }

    void query_to_codes_batch_4(
            const uint8_t* code_0,
            const uint8_t* code_1,
            const uint8_t* code_2,
            const uint8_t* code_3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) const final {
        dis0 = compute_l2(code_0);
        dis1 = compute_l2(code_1);
        dis2 = compute_l2(code_2);
        dis3 = compute_l2(code_3);
    }
};

//  * Fast path — QT_4bit (NON_UNIFORM) + IP
//  *
//  * Per-dimension affine reconstruction:
//  *     recon_i(c) = vmin[i] + vdiff[i] * (c + 0.5) / 15
//  *                = rmin_i + a_i * c,  with a_i = vdiff[i] / 15,
//  *                  rmin_i = vmin[i] + 0.5 * a_i
//  * The inner product against the query decomposes into a query-only
//  * constant plus a coefficient dot product over the integer codes:
//  *     IP(q, code) = sum_i q_i * recon_i(c_i)
//  *                 = sum_i q_i * rmin_i + sum_i (q_i * a_i) * c_i
//  *                 = K_q + sum_i b_i * c_i
//  * K_q and b_i = q_i * a_i depend only on the query, so set_query
//  * precomputes them once per query (amortized over all codes).
//  *
//  * The packed 4-bit code interleaves dimensions (even dim -> low nibble,
//  * odd dim -> high nibble), so the constructor deinterleaves a_i / rmin_i
//  * into _lo/_hi streams once and set_query does the same for b_i. The hot
//  * loop works on contiguous streams: per 16-byte code chunk (32 dims,
//  * e8m1 -> f32m4) it unpacks nibbles, converts to f32 and accumulates
//  * b * c into a single accumulator (1 load + 1 vfmacc per nibble stream —
//  * lighter than the L2 kernel). vsetvl is hoisted (0 explicit vsetvl in
//  * the loop).
//  *
//  * Padding: for odd d the last byte's high nibble is not a real dimension;
//  * b_hi keeps 0 in that slot so its contribution is exactly 0.
//  ************************************************************************/

template <>
struct DCTemplate<
        QuantizerTemplate<
                Codec4bit<SIMDLevel::RISCV_RVV>,
                QuantizerTemplateScaling::NON_UNIFORM,
                SIMDLevel::RISCV_RVV>,
        SimilarityIP<SIMDLevel::RISCV_RVV>,
        SIMDLevel::RISCV_RVV> : SQDistanceComputer {
    using Sim = SimilarityIP<SIMDLevel::RISCV_RVV>;

    size_t d;
    size_t half; // bytes per code = ceil(d/2)
    std::vector<float> a_lo, a_hi;       // a_i = vdiff[i]/15, deinterleaved
    std::vector<float> rmin_lo, rmin_hi; // vmin[i] + 0.5*a_i, deinterleaved
    std::vector<float> b_lo, b_hi;       // q_i * a_i, per query
    float k_q;                           // sum_i q_i * rmin_i, per query

    DCTemplate(size_t d_in, const std::vector<float>& trained)
            : d(d_in),
              half((d_in + 1) / 2),
              a_lo(half, 0.0f),
              a_hi(half, 0.0f),
              rmin_lo(half, 0.0f),
              rmin_hi(half, 0.0f),
              b_lo(half, 0.0f),
              b_hi(half, 0.0f),
              k_q(0.0f) {
        const float* vmin = trained.data();
        const float* vdiff = trained.data() + d_in;
        for (size_t i = 0; i < d_in; i++) {
            float a = vdiff[i] / 15.0f;
            float rmin = vmin[i] + 0.5f * a;
            if (i % 2 == 0) {
                a_lo[i / 2] = a;
                rmin_lo[i / 2] = rmin;
            } else {
                a_hi[i / 2] = a;
                rmin_hi[i / 2] = rmin;
            }
        }
    }

    void set_query(const float* x) final {
        q = x;
        // b_i = q_i * a_i, deinterleaved; K_q = sum_i q_i * rmin_i.
        // Padding slot (odd d) keeps b_hi = 0 so its contribution is 0.
        float acc = 0;
        for (size_t i = 0; i < d; i++) {
            if (i % 2 == 0) {
                b_lo[i / 2] = x[i] * a_lo[i / 2];
                acc += x[i] * rmin_lo[i / 2];
            } else {
                b_hi[i / 2] = x[i] * a_hi[i / 2];
                acc += x[i] * rmin_hi[i / 2];
            }
        }
        k_q = acc;
    }

    /// IP = K_q + sum_i b_i * c_i over the packed code.
    /// Hot loop body per byte chunk: unpack nibbles (e8), widen+convert
    /// to f32, then acc += b * c (vfmacc). Single accumulator; vsetvl
    /// hoisted outside the loop (0 explicit vsetvl inside).
    /// ID4 (kept): one scalar prefetch per chunk for the code stream.
    /// ID8: manual software pipelining — the packed-byte load of chunk
    /// k+1 is issued at the top of iteration k (prologue + rotate), so
    /// the load-use distance spans a whole iteration body instead of a
    /// couple of instructions. Zero extra instructions, +1 live m1 reg.
    float compute_ip(const uint8_t* code) const {
        const size_t nb = half; // total bytes to process
        size_t bpos = 0;

        // Hoist vsetvl: VLMAX for e8m1 (== f32 lanes per m4 group chunk)
        size_t vlb = __riscv_vsetvl_e8m1(nb);
        vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vlb);

        if (nb >= vlb) {
            // Prologue: preload chunk 0.
            vuint8m1_t packed = __riscv_vle8_v_u8m1(code, vlb);

            // Main loop: while a full NEXT chunk exists, issue its load
            // first, then process the current one.
            for (; bpos + 2 * vlb <= nb; bpos += vlb) {
                __builtin_prefetch(code + bpos + 256, 0, 0);
                vuint8m1_t packed_next =
                        __riscv_vle8_v_u8m1(code + bpos + vlb, vlb);

                vuint8m1_t lo = __riscv_vand_vx_u8m1(packed, 0x0F, vlb);
                vuint8m1_t hi = __riscv_vsrl_vx_u8m1(packed, 4, vlb);

                // Low-nibble stream (even dims): acc += b_lo * c_lo
                vfloat32m4_t clo = __riscv_vfcvt_f_xu_v_f32m4(
                        __riscv_vzext_vf4_u32m4(lo, vlb), vlb);
                acc = __riscv_vfmacc_vv_f32m4(
                        acc,
                        __riscv_vle32_v_f32m4(b_lo.data() + bpos, vlb),
                        clo,
                        vlb);

                // High-nibble stream (odd dims): acc += b_hi * c_hi
                vfloat32m4_t chi = __riscv_vfcvt_f_xu_v_f32m4(
                        __riscv_vzext_vf4_u32m4(hi, vlb), vlb);
                acc = __riscv_vfmacc_vv_f32m4(
                        acc,
                        __riscv_vle32_v_f32m4(b_hi.data() + bpos, vlb),
                        chi,
                        vlb);

                packed = packed_next; // rotate
            }

            // Epilogue: process the last full chunk (already loaded).
            {
                vuint8m1_t lo = __riscv_vand_vx_u8m1(packed, 0x0F, vlb);
                vuint8m1_t hi = __riscv_vsrl_vx_u8m1(packed, 4, vlb);

                vfloat32m4_t clo = __riscv_vfcvt_f_xu_v_f32m4(
                        __riscv_vzext_vf4_u32m4(lo, vlb), vlb);
                acc = __riscv_vfmacc_vv_f32m4(
                        acc,
                        __riscv_vle32_v_f32m4(b_lo.data() + bpos, vlb),
                        clo,
                        vlb);

                vfloat32m4_t chi = __riscv_vfcvt_f_xu_v_f32m4(
                        __riscv_vzext_vf4_u32m4(hi, vlb), vlb);
                acc = __riscv_vfmacc_vv_f32m4(
                        acc,
                        __riscv_vle32_v_f32m4(b_hi.data() + bpos, vlb),
                        chi,
                        vlb);

                bpos += vlb;
            }
        }

        // Tail: fewer than vlb bytes left; accumulates into the first
        // lanes of acc (safe: reduction below covers all vlb lanes).
        if (bpos < nb) {
            size_t vt = __riscv_vsetvl_e8m1(nb - bpos);

            vuint8m1_t packed = __riscv_vle8_v_u8m1(code + bpos, vt);
            vuint8m1_t lo = __riscv_vand_vx_u8m1(packed, 0x0F, vt);
            vuint8m1_t hi = __riscv_vsrl_vx_u8m1(packed, 4, vt);

            vfloat32m4_t clo = __riscv_vfcvt_f_xu_v_f32m4(
                    __riscv_vzext_vf4_u32m4(lo, vt), vt);
            acc = __riscv_vfmacc_vv_f32m4(
                    acc,
                    __riscv_vle32_v_f32m4(b_lo.data() + bpos, vt),
                    clo,
                    vt);

            vfloat32m4_t chi = __riscv_vfcvt_f_xu_v_f32m4(
                    __riscv_vzext_vf4_u32m4(hi, vt), vt);
            acc = __riscv_vfmacc_vv_f32m4(
                    acc,
                    __riscv_vle32_v_f32m4(b_hi.data() + bpos, vt),
                    chi,
                    vt);
        }

        // Horizontal reduce over all vlb lanes
        vfloat32m1_t red = __riscv_vfredusum_vs_f32m4_f32m1(
                acc, __riscv_vfmv_v_f_f32m1(0.0f, 1), vlb);
        return k_q + __riscv_vfmv_f_s_f32m1_f32(red);
    }

    float query_to_code(const uint8_t* code) const final {
        return compute_ip(code);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        // Not on the benchmark-critical path; scalar per-dim evaluation.
        // IP(recon(c1), recon(c2)) = sum_k recon1_k * recon2_k
        const uint8_t* c1 = codes + i * code_size;
        const uint8_t* c2 = codes + j * code_size;
        float acc = 0;
        for (size_t k = 0; k < d; k++) {
            uint8_t n1 = (k % 2 == 0) ? (c1[k / 2] & 0x0F) : (c1[k / 2] >> 4);
            uint8_t n2 = (k % 2 == 0) ? (c2[k / 2] & 0x0F) : (c2[k / 2] >> 4);
            float a = (k % 2 == 0) ? a_lo[k / 2] : a_hi[k / 2];
            float rmin = (k % 2 == 0) ? rmin_lo[k / 2] : rmin_hi[k / 2];
            float r1 = rmin + a * float(n1);
            float r2 = rmin + a * float(n2);
            acc += r1 * r2;
        }
        return acc;
    }

    void query_to_codes_batch_4(
            const uint8_t* code_0,
            const uint8_t* code_1,
            const uint8_t* code_2,
            const uint8_t* code_3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) const final {
        dis0 = compute_ip(code_0);
        dis1 = compute_ip(code_1);
        dis2 = compute_ip(code_2);
        dis3 = compute_ip(code_3);
    }
};

//  * Fast path — QT_4bit_uniform + IP
//  *
//  * 4-bit UNIFORM scaling: every component reconstructs as an affine
//  * function of the 4-bit code with SHARED scalar constants,
//  *     recon(c) = vmin + vdiff * (c + 0.5) / 15 = c0 + a * c
//  * with a = vdiff / 15 and c0 = vmin + 0.5 * a. The inner product
//  * against the query then folds into a query-only constant plus one
//  * uniformly-scaled integer-code dot product:
//  *     IP(q, code) = sum_i q_i * (c0 + a * c_i)
//  *                 = c0 * sum_i q_i  +  a * sum_i q_i * c_i
//  *                 = K_q + a * S
//  * K_q = c0 * sum(q) is precomputed once per query in set_query, so the
//  * hot loop only evaluates S = sum_i q_i * c_i. Unlike the NON_UNIFORM
//  * IP variant there is no per-dimension coefficient stream to build —
//  * the q streams ARE the raw query components (deinterleaved).
//  *
//  * The packed 4-bit code interleaves dimensions (even dim -> low nibble,
//  * odd dim -> high nibble), so set_query deinterleaves q into q_lo/q_hi
//  * once per query. Hot loop per 16-byte code chunk (32 dims, e8m1 ->
//  * f32m4): unpack nibbles, widen+convert to f32, acc += q * c (vfmacc,
//  * single accumulator). vsetvl hoisted (0 explicit vsetvl in the loop);
//  * software pipelining (next-chunk packed load issued at the top of the
//  * current iteration) + one scalar prefetch per chunk, both carried
//  * over from the board-validated qt_4bit_ip v6 kernel shape.
//  *
//  * Padding: for odd d the last byte's high nibble is not a real
//  * dimension; q_hi keeps 0 in that slot so its contribution is 0.
//  * vdiff == 0 degenerates naturally (a = 0 -> IP = K_q), no branch.
//  ************************************************************************/

template <>
struct DCTemplate<
        QuantizerTemplate<
                Codec4bit<SIMDLevel::RISCV_RVV>,
                QuantizerTemplateScaling::UNIFORM,
                SIMDLevel::RISCV_RVV>,
        SimilarityIP<SIMDLevel::RISCV_RVV>,
        SIMDLevel::RISCV_RVV> : SQDistanceComputer {
    using Sim = SimilarityIP<SIMDLevel::RISCV_RVV>;

    size_t d;
    size_t half; // bytes per code = ceil(d/2)
    float vmin;
    float vdiff;
    float a;  // vdiff / 15
    float c0; // vmin + 0.5 * a
    std::vector<float> q_lo, q_hi; // query components, deinterleaved
    float k_q;                     // c0 * sum_i q_i, per query

    DCTemplate(size_t d_in, const std::vector<float>& trained)
            : d(d_in),
              half((d_in + 1) / 2),
              vmin(trained[0]),
              vdiff(trained[1]),
              q_lo(half, 0.0f),
              q_hi(half, 0.0f),
              k_q(0.0f) {
        a = vdiff / 15.0f;
        c0 = vmin + 0.5f * a;
    }

    void set_query(const float* x) final {
        q = x;
        // Deinterleave query into per-nibble-stream layout and fold the
        // query-only constant K_q = c0 * sum(q). Padding slot (odd d)
        // keeps q_hi = 0 so the padding nibble contributes exactly 0.
        float sum = 0;
        for (size_t i = 0; i < d; i++) {
            sum += x[i];
            if (i % 2 == 0) {
                q_lo[i / 2] = x[i];
            } else {
                q_hi[i / 2] = x[i];
            }
        }
        k_q = c0 * sum;
    }

    /// S = sum_i q_i * c_i over the packed code (raw integer nibbles).
    /// Hot loop body per byte chunk: unpack nibbles (e8), widen+convert
    /// to f32, then acc += q * c (vfmacc). Single accumulator; vsetvl
    /// hoisted outside the loop (0 explicit vsetvl inside). Software
    /// pipelining: the packed-byte load of chunk k+1 is issued at the
    /// top of iteration k (prologue + rotate) so the load-use distance
    /// spans a whole iteration body. One scalar prefetch per chunk.
    float compute_qc_dot(const uint8_t* code) const {
        const size_t nb = half; // total bytes to process
        size_t bpos = 0;

        // Hoist vsetvl: VLMAX for e8m1 (== f32 lanes per m4 group chunk)
        size_t vlb = __riscv_vsetvl_e8m1(nb);
        // ID3: dual accumulators — one per nibble stream — break the
        // per-iteration serial chain of two dependent vfmacc ops into
        // two independent chains (f32 FMA latency > 1 iter issue time).
        vfloat32m4_t acc_lo = __riscv_vfmv_v_f_f32m4(0.0f, vlb);
        vfloat32m4_t acc_hi = __riscv_vfmv_v_f_f32m4(0.0f, vlb);

        if (nb >= vlb) {
            // Prologue: preload chunk 0.
            vuint8m1_t packed = __riscv_vle8_v_u8m1(code, vlb);

            // Main loop: while a full NEXT chunk exists, issue its load
            // first, then process the current one.
            for (; bpos + 2 * vlb <= nb; bpos += vlb) {
                // ID5: prefetch distance 256 -> 512 (8 chunks ahead) —
                // probe this kernel's sensitivity to SW prefetch depth.
                __builtin_prefetch(code + bpos + 512, 0, 0);
                vuint8m1_t packed_next =
                        __riscv_vle8_v_u8m1(code + bpos + vlb, vlb);

                vuint8m1_t lo = __riscv_vand_vx_u8m1(packed, 0x0F, vlb);
                vuint8m1_t hi = __riscv_vsrl_vx_u8m1(packed, 4, vlb);

                // ID4: manual scheduling — issue both q-stream loads
                // right after the nibble split, before the convert
                // chains, so each load has >=3 instructions of distance
                // to its consuming vfmacc (hides L1-hit load-use
                // latency on an in-order core).
                vfloat32m4_t vq_lo =
                        __riscv_vle32_v_f32m4(q_lo.data() + bpos, vlb);
                vfloat32m4_t vq_hi =
                        __riscv_vle32_v_f32m4(q_hi.data() + bpos, vlb);

                // Low-nibble stream (even dims): acc_lo += q_lo * c_lo
                // ID2: narrow widening chain — u8 -> u16m2 (vzext_vf2)
                // -> f32m4 (vfwcvt.f.xu.v): same 2 instructions as the
                // vzext_vf4+vfcvt chain but the integer intermediate is
                // half as wide (m2 vs m4), halving its register beats.
                vfloat32m4_t clo = __riscv_vfwcvt_f_xu_v_f32m4(
                        __riscv_vzext_vf2_u16m2(lo, vlb), vlb);
                acc_lo = __riscv_vfmacc_vv_f32m4(acc_lo, vq_lo, clo, vlb);

                // High-nibble stream (odd dims): acc_hi += q_hi * c_hi
                vfloat32m4_t chi = __riscv_vfwcvt_f_xu_v_f32m4(
                        __riscv_vzext_vf2_u16m2(hi, vlb), vlb);
                acc_hi = __riscv_vfmacc_vv_f32m4(acc_hi, vq_hi, chi, vlb);

                packed = packed_next; // rotate
            }

            // Epilogue: process the last full chunk (already loaded).
            {
                vuint8m1_t lo = __riscv_vand_vx_u8m1(packed, 0x0F, vlb);
                vuint8m1_t hi = __riscv_vsrl_vx_u8m1(packed, 4, vlb);

                vfloat32m4_t clo = __riscv_vfwcvt_f_xu_v_f32m4(
                        __riscv_vzext_vf2_u16m2(lo, vlb), vlb);
                acc_lo = __riscv_vfmacc_vv_f32m4(
                        acc_lo,
                        __riscv_vle32_v_f32m4(q_lo.data() + bpos, vlb),
                        clo,
                        vlb);

                vfloat32m4_t chi = __riscv_vfwcvt_f_xu_v_f32m4(
                        __riscv_vzext_vf2_u16m2(hi, vlb), vlb);
                acc_hi = __riscv_vfmacc_vv_f32m4(
                        acc_hi,
                        __riscv_vle32_v_f32m4(q_hi.data() + bpos, vlb),
                        chi,
                        vlb);

                bpos += vlb;
            }
        }

        // Tail: fewer than vlb bytes left; accumulates into the first
        // lanes of acc (safe: reduction below covers all vlb lanes).
        if (bpos < nb) {
            size_t vt = __riscv_vsetvl_e8m1(nb - bpos);

            vuint8m1_t packed = __riscv_vle8_v_u8m1(code + bpos, vt);
            vuint8m1_t lo = __riscv_vand_vx_u8m1(packed, 0x0F, vt);
            vuint8m1_t hi = __riscv_vsrl_vx_u8m1(packed, 4, vt);

            vfloat32m4_t clo = __riscv_vfwcvt_f_xu_v_f32m4(
                    __riscv_vzext_vf2_u16m2(lo, vt), vt);
            acc_lo = __riscv_vfmacc_vv_f32m4(
                    acc_lo,
                    __riscv_vle32_v_f32m4(q_lo.data() + bpos, vt),
                    clo,
                    vt);

            vfloat32m4_t chi = __riscv_vfwcvt_f_xu_v_f32m4(
                    __riscv_vzext_vf2_u16m2(hi, vt), vt);
            acc_hi = __riscv_vfmacc_vv_f32m4(
                    acc_hi,
                    __riscv_vle32_v_f32m4(q_hi.data() + bpos, vt),
                    chi,
                    vt);
        }

        // Merge the two chains, then one horizontal reduce over vlb lanes
        vfloat32m4_t acc = __riscv_vfadd_vv_f32m4(acc_lo, acc_hi, vlb);
        vfloat32m1_t red = __riscv_vfredusum_vs_f32m4_f32m1(
                acc, __riscv_vfmv_v_f_f32m1(0.0f, 1), vlb);
        return __riscv_vfmv_f_s_f32m1_f32(red);
    }

    float query_to_code(const uint8_t* code) const final {
        return k_q + a * compute_qc_dot(code);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        // Not on the benchmark-critical path; scalar per-dim evaluation.
        // IP(recon(c1), recon(c2)) = sum_k (c0 + a*n1) * (c0 + a*n2)
        const uint8_t* c1 = codes + i * code_size;
        const uint8_t* c2 = codes + j * code_size;
        float acc = 0;
        for (size_t k = 0; k < d; k++) {
            uint8_t n1 = (k % 2 == 0) ? (c1[k / 2] & 0x0F) : (c1[k / 2] >> 4);
            uint8_t n2 = (k % 2 == 0) ? (c2[k / 2] & 0x0F) : (c2[k / 2] >> 4);
            float r1 = c0 + a * float(n1);
            float r2 = c0 + a * float(n2);
            acc += r1 * r2;
        }
        return acc;
    }

    void query_to_codes_batch_4(
            const uint8_t* code_0,
            const uint8_t* code_1,
            const uint8_t* code_2,
            const uint8_t* code_3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) const final {
        dis0 = k_q + a * compute_qc_dot(code_0);
        dis1 = k_q + a * compute_qc_dot(code_1);
        dis2 = k_q + a * compute_qc_dot(code_2);
        dis3 = k_q + a * compute_qc_dot(code_3);
    }
};

// * Fast path — QT_6bit (NON_UNIFORM) + L2
//  *
//  * Per-dimension affine reconstruction:
//  *     recon_i(c) = vmin[i] + vdiff[i] * (c + 0.5) / 63
//  *                = rmin_i + a_i * c,  with a_i = vdiff[i] / 63,
//  *                  rmin_i = vmin[i] + 0.5 * a_i
//  * L2 contribution per dim: (q_i - recon_i)^2 = (e_i - a_i * c_i)^2
//  * where e_i = q_i - rmin_i is precomputed once per query in set_query.
//  *
//  * Codec6bit packs 4 dimensions into 3 bytes (b0,b1,b2):
//  *     c0 = b0 & 0x3F
//  *     c1 = (b0 >> 6) | ((b1 & 0x0F) << 2)
//  *     c2 = (b1 >> 4) | ((b2 & 0x03) << 4)
//  *     c3 = b2 >> 2
//  * Unlike 4-bit there is no lo/hi nibble symmetry: the kernel deinterleaves
//  * by (i & 3) into FOUR streams. The constructor splits a_i / rmin_i into
//  * s0..s3 streams once; set_query does the same for e_i.
//  *
//  * Hot loop per vl=VLMAX(e8m1) groups (3*vl code bytes, 4*vl dims):
//  * vlseg3e8 deinterleaves the 3-byte groups into b0/b1/b2 registers, the
//  * 6-bit fields are extracted with 8 u8 ALU ops (field combines use
//  * vmacc.vx), then each value stream goes u8 -> u16m2 (vzext_vf2) ->
//  * f32m4 (vfwcvt) and accumulates t = e - a*c (vfnmsac), acc += t*t
//  * (vfmacc). vsetvl hoisted (0 explicit vsetvl inside the hot loop).
//  *
//  * ID3 (R3): software pipeline the vlseg3e8 segment load — the NEXT
//  * chunk's code block is preloaded at the top of THIS iteration
//  * (prologue + rotate), stretching the load-use distance across the
//  * full iteration body (~20+ ops). Proven on qt_4bit family.
//  * (R2/ID2 strided-vlse8 duel: tied within noise, vlseg3e8 kept as
//  * simpler 1-instruction form.)
//  *
//  * Tail: dims beyond 4*(d/4) are decoded scalar (exact Codec6bit
//  * semantics); for the benchmark d=768 there is no tail.
//  ************************************************************************

// Raw 6-bit field extract, scalar (exact Codec6bit<NONE> bit semantics).
static inline uint8_t sq6_decode_raw(const uint8_t* code, size_t i) {
    const uint8_t* p = code + (i >> 2) * 3;
    switch (i & 3) {
        case 0:
            return p[0] & 0x3F;
        case 1:
            return uint8_t((p[0] >> 6) | ((p[1] & 0x0F) << 2));
        case 2:
            return uint8_t((p[1] >> 4) | ((p[2] & 0x03) << 4));
        default:
            return uint8_t(p[2] >> 2);
    }
}

template <>
struct DCTemplate<
        QuantizerTemplate<
                Codec6bit<SIMDLevel::RISCV_RVV>,
                QuantizerTemplateScaling::NON_UNIFORM,
                SIMDLevel::RISCV_RVV>,
        SimilarityL2<SIMDLevel::RISCV_RVV>,
        SIMDLevel::RISCV_RVV> : SQDistanceComputer {
    using Sim = SimilarityL2<SIMDLevel::RISCV_RVV>;

    size_t d;
    size_t ng; // full 4-dim groups = d / 4
    // a_i = vdiff[i]/63 and e_i = q_i - rmin_i, deinterleaved by (i & 3)
    std::vector<float> a0, a1, a2, a3;
    std::vector<float> e0, e1, e2, e3;
    // Interleaved copies for the scalar tail / symmetric_dis
    std::vector<float> a_all, rmin_all;

    DCTemplate(size_t d_in, const std::vector<float>& trained)
            : d(d_in),
              ng(d_in / 4),
              a0(ng, 0.0f),
              a1(ng, 0.0f),
              a2(ng, 0.0f),
              a3(ng, 0.0f),
              e0(ng, 0.0f),
              e1(ng, 0.0f),
              e2(ng, 0.0f),
              e3(ng, 0.0f),
              a_all(d_in, 0.0f),
              rmin_all(d_in, 0.0f) {
        const float* vmin = trained.data();
        const float* vdiff = trained.data() + d_in;
        for (size_t i = 0; i < d_in; i++) {
            float a = vdiff[i] / 63.0f;
            float rmin = vmin[i] + 0.5f * a;
            a_all[i] = a;
            rmin_all[i] = rmin;
            if (i < 4 * ng) {
                size_t k = i >> 2;
                switch (i & 3) {
                    case 0:
                        a0[k] = a;
                        break;
                    case 1:
                        a1[k] = a;
                        break;
                    case 2:
                        a2[k] = a;
                        break;
                    default:
                        a3[k] = a;
                        break;
                }
            }
        }
    }

    void set_query(const float* x) final {
        q = x;
        // e_i = q_i - rmin_i, deinterleaved into the 4 group-position
        // streams (amortized: called once per query, 1/n weight).
        for (size_t k = 0; k < ng; k++) {
            const float* xi = x + 4 * k;
            const float* rm = rmin_all.data() + 4 * k;
            e0[k] = xi[0] - rm[0];
            e1[k] = xi[1] - rm[1];
            e2[k] = xi[2] - rm[2];
            e3[k] = xi[3] - rm[3];
        }
        // Tail dims (d % 4) are handled scalar in compute_l2 via q.
    }

    /// Full-precision vector L2 over the packed 6-bit code.
    float compute_l2(const uint8_t* code) const {
        const size_t ngf = ng;
        float total = 0.0f;
        size_t g = 0;

        if (ngf > 0) {
            // Hoist vsetvl: VLMAX for e8m1 (== f32 lanes per m4 group)
            const size_t vl = __riscv_vsetvl_e8m1(ngf);
            vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);

            const float* pa0 = a0.data();
            const float* pa1 = a1.data();
            const float* pa2 = a2.data();
            const float* pa3 = a3.data();
            const float* pe0 = e0.data();
            const float* pe1 = e1.data();
            const float* pe2 = e2.data();
            const float* pe3 = e3.data();

            // ID3: software-pipelined vlseg3e8 — preload the NEXT
            // chunk's code block at the top of THIS iteration, then
            // compute on the already-loaded current chunk. The load-use
            // distance stretches across the full iteration body (12
            // f32m4 FMA + 4 conversions + 8 u8 ALU ops), hiding L1
            // miss/latency. Prologue + rotate pattern proven on qt_4bit
            // (uniform_ip v5/v6, ip v6).
            if (ngf >= vl) {
                // Prologue: preload chunk 0.
                vuint8m1x3_t seg =
                        __riscv_vlseg3e8_v_u8m1x3(code + 3 * g, vl);
                g += vl;

                // Main loop: while a full next chunk exists, issue its
                // load first, then process the current one.
                for (; g + vl <= ngf; g += vl) {
                    vuint8m1x3_t seg_next =
                            __riscv_vlseg3e8_v_u8m1x3(
                                    code + 3 * g, vl);
                    vuint8m1_t b0 =
                            __riscv_vget_v_u8m1x3_u8m1(seg, 0);
                    vuint8m1_t b1 =
                            __riscv_vget_v_u8m1x3_u8m1(seg, 1);
                    vuint8m1_t b2 =
                            __riscv_vget_v_u8m1x3_u8m1(seg, 2);

                    // ID1: stream-order rescheduling — light
                    // extracts (c0: 1 op, c3: 1 op) computed
                    // first and their convert+FMA chains issued
                    // immediately; heavy extracts (c1/c2: 3 ops
                    // each) overlap with in-flight FMAs.
                    // Accumulation order: c0, c3, c1, c2.
                    vuint8m1_t c0 =
                            __riscv_vand_vx_u8m1(b0, 0x3F, vl);
                    vuint8m1_t c3 =
                            __riscv_vsrl_vx_u8m1(b2, 2, vl);

                    vfloat32m4_t f0 = __riscv_vfwcvt_f_xu_v_f32m4(
                            __riscv_vzext_vf2_u16m2(c0, vl), vl);
                    vfloat32m4_t t0 =
                            __riscv_vle32_v_f32m4(pe0 + g - vl, vl);
                    t0 = __riscv_vfnmsac_vv_f32m4(
                            t0,
                            __riscv_vle32_v_f32m4(
                                    pa0 + g - vl, vl),
                            f0,
                            vl);
                    acc = __riscv_vfmacc_vv_f32m4(
                            acc, t0, t0, vl);

                    vfloat32m4_t f3 = __riscv_vfwcvt_f_xu_v_f32m4(
                            __riscv_vzext_vf2_u16m2(c3, vl), vl);
                    vfloat32m4_t t3 =
                            __riscv_vle32_v_f32m4(pe3 + g - vl, vl);
                    t3 = __riscv_vfnmsac_vv_f32m4(
                            t3,
                            __riscv_vle32_v_f32m4(
                                    pa3 + g - vl, vl),
                            f3,
                            vl);
                    acc = __riscv_vfmacc_vv_f32m4(
                            acc, t3, t3, vl);

                    vuint8m1_t c1 = __riscv_vmacc_vx_u8m1(
                            __riscv_vsrl_vx_u8m1(b0, 6, vl),
                            4,
                            __riscv_vand_vx_u8m1(b1, 0x0F, vl),
                            vl);

                    vfloat32m4_t f1 = __riscv_vfwcvt_f_xu_v_f32m4(
                            __riscv_vzext_vf2_u16m2(c1, vl), vl);
                    vfloat32m4_t t1 =
                            __riscv_vle32_v_f32m4(pe1 + g - vl, vl);
                    t1 = __riscv_vfnmsac_vv_f32m4(
                            t1,
                            __riscv_vle32_v_f32m4(
                                    pa1 + g - vl, vl),
                            f1,
                            vl);
                    acc = __riscv_vfmacc_vv_f32m4(
                            acc, t1, t1, vl);

                    vuint8m1_t c2 = __riscv_vmacc_vx_u8m1(
                            __riscv_vsrl_vx_u8m1(b1, 4, vl),
                            16,
                            __riscv_vand_vx_u8m1(b2, 0x03, vl),
                            vl);

                    vfloat32m4_t f2 = __riscv_vfwcvt_f_xu_v_f32m4(
                            __riscv_vzext_vf2_u16m2(c2, vl), vl);
                    vfloat32m4_t t2 =
                            __riscv_vle32_v_f32m4(pe2 + g - vl, vl);
                    t2 = __riscv_vfnmsac_vv_f32m4(
                            t2,
                            __riscv_vle32_v_f32m4(
                                    pa2 + g - vl, vl),
                            f2,
                            vl);
                    acc = __riscv_vfmacc_vv_f32m4(
                            acc, t2, t2, vl);

                    seg = seg_next; // rotate
                }

                // Epilogue: process the last full chunk (already loaded).
                // ID1: stream-order rescheduling c0,c3,c1,c2
                // (same pattern as the main loop).
                {
                    vuint8m1_t b0 =
                            __riscv_vget_v_u8m1x3_u8m1(seg, 0);
                    vuint8m1_t b1 =
                            __riscv_vget_v_u8m1x3_u8m1(seg, 1);
                    vuint8m1_t b2 =
                            __riscv_vget_v_u8m1x3_u8m1(seg, 2);

                    vuint8m1_t c0 =
                            __riscv_vand_vx_u8m1(b0, 0x3F, vl);
                    vuint8m1_t c3 =
                            __riscv_vsrl_vx_u8m1(b2, 2, vl);

                    vfloat32m4_t f0 = __riscv_vfwcvt_f_xu_v_f32m4(
                            __riscv_vzext_vf2_u16m2(c0, vl), vl);
                    vfloat32m4_t t0 = __riscv_vle32_v_f32m4(
                            pe0 + g - vl, vl);
                    t0 = __riscv_vfnmsac_vv_f32m4(
                            t0,
                            __riscv_vle32_v_f32m4(
                                    pa0 + g - vl, vl),
                            f0,
                            vl);
                    acc = __riscv_vfmacc_vv_f32m4(
                            acc, t0, t0, vl);

                    vfloat32m4_t f3 = __riscv_vfwcvt_f_xu_v_f32m4(
                            __riscv_vzext_vf2_u16m2(c3, vl), vl);
                    vfloat32m4_t t3 = __riscv_vle32_v_f32m4(
                            pe3 + g - vl, vl);
                    t3 = __riscv_vfnmsac_vv_f32m4(
                            t3,
                            __riscv_vle32_v_f32m4(
                                    pa3 + g - vl, vl),
                            f3,
                            vl);
                    acc = __riscv_vfmacc_vv_f32m4(
                            acc, t3, t3, vl);

                    vuint8m1_t c1 = __riscv_vmacc_vx_u8m1(
                            __riscv_vsrl_vx_u8m1(b0, 6, vl),
                            4,
                            __riscv_vand_vx_u8m1(b1, 0x0F, vl),
                            vl);

                    vfloat32m4_t f1 = __riscv_vfwcvt_f_xu_v_f32m4(
                            __riscv_vzext_vf2_u16m2(c1, vl), vl);
                    vfloat32m4_t t1 = __riscv_vle32_v_f32m4(
                            pe1 + g - vl, vl);
                    t1 = __riscv_vfnmsac_vv_f32m4(
                            t1,
                            __riscv_vle32_v_f32m4(
                                    pa1 + g - vl, vl),
                            f1,
                            vl);
                    acc = __riscv_vfmacc_vv_f32m4(
                            acc, t1, t1, vl);

                    vuint8m1_t c2 = __riscv_vmacc_vx_u8m1(
                            __riscv_vsrl_vx_u8m1(b1, 4, vl),
                            16,
                            __riscv_vand_vx_u8m1(b2, 0x03, vl),
                            vl);

                    vfloat32m4_t f2 = __riscv_vfwcvt_f_xu_v_f32m4(
                            __riscv_vzext_vf2_u16m2(c2, vl), vl);
                    vfloat32m4_t t2 = __riscv_vle32_v_f32m4(
                            pe2 + g - vl, vl);
                    t2 = __riscv_vfnmsac_vv_f32m4(
                            t2,
                            __riscv_vle32_v_f32m4(
                                    pa2 + g - vl, vl),
                            f2,
                            vl);
                    acc = __riscv_vfmacc_vv_f32m4(
                            acc, t2, t2, vl);
                }
            }

            // Tail groups: one shorter-vl pass into the same accumulator
            // (only the first vt lanes are touched; reduction below
            // covers all vl lanes).
            // ID1: stream-order rescheduling c0,c3,c1,c2 (same pattern).
            if (g < ngf) {
                const size_t vt = __riscv_vsetvl_e8m1(ngf - g);

                vuint8m1x3_t seg =
                        __riscv_vlseg3e8_v_u8m1x3(code + 3 * g, vt);
                vuint8m1_t b0 = __riscv_vget_v_u8m1x3_u8m1(seg, 0);
                vuint8m1_t b1 = __riscv_vget_v_u8m1x3_u8m1(seg, 1);
                vuint8m1_t b2 = __riscv_vget_v_u8m1x3_u8m1(seg, 2);

                vuint8m1_t c0 = __riscv_vand_vx_u8m1(b0, 0x3F, vt);
                vuint8m1_t c3 = __riscv_vsrl_vx_u8m1(b2, 2, vt);

                vfloat32m4_t f0 = __riscv_vfwcvt_f_xu_v_f32m4(
                        __riscv_vzext_vf2_u16m2(c0, vt), vt);
                vfloat32m4_t t0 = __riscv_vle32_v_f32m4(pe0 + g, vt);
                t0 = __riscv_vfnmsac_vv_f32m4(
                        t0, __riscv_vle32_v_f32m4(pa0 + g, vt), f0, vt);
                acc = __riscv_vfmacc_vv_f32m4(acc, t0, t0, vt);

                vfloat32m4_t f3 = __riscv_vfwcvt_f_xu_v_f32m4(
                        __riscv_vzext_vf2_u16m2(c3, vt), vt);
                vfloat32m4_t t3 = __riscv_vle32_v_f32m4(pe3 + g, vt);
                t3 = __riscv_vfnmsac_vv_f32m4(
                        t3, __riscv_vle32_v_f32m4(pa3 + g, vt), f3, vt);
                acc = __riscv_vfmacc_vv_f32m4(acc, t3, t3, vt);

                vuint8m1_t c1 = __riscv_vmacc_vx_u8m1(
                        __riscv_vsrl_vx_u8m1(b0, 6, vt),
                        4,
                        __riscv_vand_vx_u8m1(b1, 0x0F, vt),
                        vt);

                vfloat32m4_t f1 = __riscv_vfwcvt_f_xu_v_f32m4(
                        __riscv_vzext_vf2_u16m2(c1, vt), vt);
                vfloat32m4_t t1 = __riscv_vle32_v_f32m4(pe1 + g, vt);
                t1 = __riscv_vfnmsac_vv_f32m4(
                        t1, __riscv_vle32_v_f32m4(pa1 + g, vt), f1, vt);
                acc = __riscv_vfmacc_vv_f32m4(acc, t1, t1, vt);

                vuint8m1_t c2 = __riscv_vmacc_vx_u8m1(
                        __riscv_vsrl_vx_u8m1(b1, 4, vt),
                        16,
                        __riscv_vand_vx_u8m1(b2, 0x03, vt),
                        vt);

                vfloat32m4_t f2 = __riscv_vfwcvt_f_xu_v_f32m4(
                        __riscv_vzext_vf2_u16m2(c2, vt), vt);
                vfloat32m4_t t2 = __riscv_vle32_v_f32m4(pe2 + g, vt);
                t2 = __riscv_vfnmsac_vv_f32m4(
                        t2, __riscv_vle32_v_f32m4(pa2 + g, vt), f2, vt);
                acc = __riscv_vfmacc_vv_f32m4(acc, t2, t2, vt);
            }

            // Single horizontal reduction over all vl lanes.
            vfloat32m1_t red = __riscv_vfredusum_vs_f32m4_f32m1(
                    acc, __riscv_vfmv_v_f_f32m1(0.0f, 1), vl);
            total = __riscv_vfmv_f_s_f32m1_f32(red);
        }

        // Scalar tail: dims beyond the last full group (d % 4).
        for (size_t i = 4 * ngf; i < d; i++) {
            float r = rmin_all[i] +
                    a_all[i] * float(sq6_decode_raw(code, i));
            float diff = q[i] - r;
            total += diff * diff;
        }
        return total;
    }

    float query_to_code(const uint8_t* code) const final {
        return compute_l2(code);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        // Not on the benchmark-critical path; scalar per-dim evaluation.
        // recon1 - recon2 = a_k * (c1_k - c2_k)
        const uint8_t* c1 = codes + i * code_size;
        const uint8_t* c2 = codes + j * code_size;
        float acc = 0;
        for (size_t k = 0; k < d; k++) {
            int n1 = sq6_decode_raw(c1, k);
            int n2 = sq6_decode_raw(c2, k);
            float diff = a_all[k] * float(n1 - n2);
            acc += diff * diff;
        }
        return acc;
    }

    void query_to_codes_batch_4(
            const uint8_t* code_0,
            const uint8_t* code_1,
            const uint8_t* code_2,
            const uint8_t* code_3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) const final {
        dis0 = compute_l2(code_0);
        dis1 = compute_l2(code_1);
        dis2 = compute_l2(code_2);
        dis3 = compute_l2(code_3);
    }
};

//  * Fast path — QT_6bit (NON_UNIFORM) + IP
//  *
//  * Per-dimension affine reconstruction:
//  *     recon_i(c) = vmin[i] + vdiff[i] * (c + 0.5) / 63
//  *                = rmin_i + a_i * c,  with a_i = vdiff[i] / 63,
//  *                  rmin_i = vmin[i] + 0.5 * a_i
//  * The inner product against the query decomposes into a query-only
//  * constant plus a coefficient dot product over the integer codes:
//  *     IP(q, code) = sum_i q_i * recon_i(c_i)
//  *                 = sum_i q_i * rmin_i + sum_i (q_i * a_i) * c_i
//  *                 = K_q + sum_i b_i * c_i
//  * K_q and b_i = q_i * a_i depend only on the query, so set_query
//  * precomputes them once per query (amortized over all 2000 codes).
//  *
//  * Codec6bit packs 4 dimensions into 3 bytes (b0,b1,b2) — same layout
//  * as the L2 kernel above; the coefficient streams are deinterleaved by
//  * (i & 3) into FOUR b-streams in set_query.
//  *
//  * Hot loop per vl=VLMAX(e8m1) groups (3*vl code bytes, 4*vl dims):
//  * vlseg3e8 deinterleaves the 3-byte groups (software-pipelined depth 1:
//  * next chunk preloaded at the top of this iteration, prologue + rotate —
//  * board-proven on qt_6bit_l2 R3 / qt_4bit_ip R6), 6-bit fields extracted
//  * with 8 u8 ALU ops (field combines via vmacc.vx), then each value
//  * stream goes u8 -> u16m2 (vzext_vf2) -> f32m4 (vfwcvt, narrow widening
//  * chain proven on uniform_ip R2) and accumulates acc += b * c (vfmacc,
//  * single accumulator — 1 load + 1 FMA per stream, lighter than the L2
//  * kernel's 2 loads + FNMSAC + FMACC). vsetvl hoisted: 0 explicit vsetvl
//  * inside the hot loop, 2 in the whole function (hoist + tail).
//  *
//  * Tail: dims beyond 4*(d/4) are decoded scalar (exact Codec6bit
//  * semantics) and contribute q_i * a_i * c_i on the fly (K_q already
//  * covers ALL dims' rmin part); for the benchmark d=768 there is no tail.
//  ************************************************************************/

template <>
struct DCTemplate<
        QuantizerTemplate<
                Codec6bit<SIMDLevel::RISCV_RVV>,
                QuantizerTemplateScaling::NON_UNIFORM,
                SIMDLevel::RISCV_RVV>,
        SimilarityIP<SIMDLevel::RISCV_RVV>,
        SIMDLevel::RISCV_RVV> : SQDistanceComputer {
    using Sim = SimilarityIP<SIMDLevel::RISCV_RVV>;

    size_t d;
    size_t ng; // full 4-dim groups = d / 4
    // b_i = q_i * a_i, deinterleaved by (i & 3), rebuilt per query
    std::vector<float> b0, b1, b2, b3;
    // Interleaved constants for set_query / scalar tail / symmetric_dis
    std::vector<float> a_all, rmin_all;
    float k_q; // sum_i q_i * rmin_i, per query

    DCTemplate(size_t d_in, const std::vector<float>& trained)
            : d(d_in),
              ng(d_in / 4),
              b0(ng, 0.0f),
              b1(ng, 0.0f),
              b2(ng, 0.0f),
              b3(ng, 0.0f),
              a_all(d_in, 0.0f),
              rmin_all(d_in, 0.0f),
              k_q(0.0f) {
        const float* vmin = trained.data();
        const float* vdiff = trained.data() + d_in;
        for (size_t i = 0; i < d_in; i++) {
            float a = vdiff[i] / 63.0f;
            a_all[i] = a;
            rmin_all[i] = vmin[i] + 0.5f * a;
        }
    }

    void set_query(const float* x) final {
        q = x;
        // b_i = q_i * a_i deinterleaved into the 4 group-position streams;
        // K_q = sum over ALL dims of q_i * rmin_i (tail dims included, so
        // the scalar tail in compute_ip only adds the b_i * c_i part).
        float acc = 0;
        for (size_t k = 0; k < ng; k++) {
            const float* xi = x + 4 * k;
            const float* aa = a_all.data() + 4 * k;
            const float* rm = rmin_all.data() + 4 * k;
            b0[k] = xi[0] * aa[0];
            b1[k] = xi[1] * aa[1];
            b2[k] = xi[2] * aa[2];
            b3[k] = xi[3] * aa[3];
            acc += xi[0] * rm[0] + xi[1] * rm[1] + xi[2] * rm[2] +
                    xi[3] * rm[3];
        }
        for (size_t i = 4 * ng; i < d; i++) {
            acc += x[i] * rmin_all[i];
        }
        k_q = acc;
    }

    /// S = sum_i b_i * c_i over the packed 6-bit code; returns K_q + S.
    float compute_ip(const uint8_t* code) const {
        const size_t ngf = ng;
        float total = 0.0f;
        size_t g = 0;

        if (ngf > 0) {
            // Hoist vsetvl: VLMAX for e8m1 (== f32 lanes per m4 group)
            const size_t vl = __riscv_vsetvl_e8m1(ngf);
            vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);

            const float* pb0 = b0.data();
            const float* pb1 = b1.data();
            const float* pb2 = b2.data();
            const float* pb3 = b3.data();

            // Software-pipelined vlseg3e8 (depth 1): preload the NEXT
            // chunk's code block at the top of THIS iteration, then
            // compute on the already-loaded current chunk.
            if (ngf >= vl) {
                // Prologue: preload chunk 0.
                vuint8m1x3_t seg =
                        __riscv_vlseg3e8_v_u8m1x3(code + 3 * g, vl);
                g += vl;

                // Main loop: while a full next chunk exists, issue its
                // load first, then process the current one.
                for (; g + vl <= ngf; g += vl) {
                    vuint8m1x3_t seg_next =
                            __riscv_vlseg3e8_v_u8m1x3(
                                    code + 3 * g, vl);
                    vuint8m1_t r0 =
                            __riscv_vget_v_u8m1x3_u8m1(seg, 0);
                    vuint8m1_t r1 =
                            __riscv_vget_v_u8m1x3_u8m1(seg, 1);
                    vuint8m1_t r2 =
                            __riscv_vget_v_u8m1x3_u8m1(seg, 2);

                    // ID11: stream-order rescheduling — the two LIGHT
                    // extracts (c0: 1 op, c3: 1 op) are computed first
                    // and their convert+FMA chains issued immediately,
                    // so the first vfmacc starts ~4 ALU ops earlier;
                    // the two HEAVY extracts (c1/c2: 3 ops each) then
                    // overlap with the in-flight FMAs. Accumulation
                    // order becomes 0,3,1,2 (float reassociation, same
                    // semantics class as the other specializations).
                    // ID3 load-first order kept inside each stream.
                    vuint8m1_t c0 =
                            __riscv_vand_vx_u8m1(r0, 0x3F, vl);
                    vuint8m1_t c3 =
                            __riscv_vsrl_vx_u8m1(r2, 2, vl);

                    vfloat32m4_t vb0 = __riscv_vle32_v_f32m4(
                            pb0 + g - vl, vl);
                    vfloat32m4_t f0 = __riscv_vfwcvt_f_xu_v_f32m4(
                            __riscv_vzext_vf2_u16m2(c0, vl), vl);
                    acc = __riscv_vfmacc_vv_f32m4(acc, vb0, f0, vl);

                    vfloat32m4_t vb3 = __riscv_vle32_v_f32m4(
                            pb3 + g - vl, vl);
                    vfloat32m4_t f3 = __riscv_vfwcvt_f_xu_v_f32m4(
                            __riscv_vzext_vf2_u16m2(c3, vl), vl);
                    acc = __riscv_vfmacc_vv_f32m4(acc, vb3, f3, vl);

                    vuint8m1_t c1 = __riscv_vmacc_vx_u8m1(
                            __riscv_vsrl_vx_u8m1(r0, 6, vl),
                            4,
                            __riscv_vand_vx_u8m1(r1, 0x0F, vl),
                            vl);
                    vfloat32m4_t vb1 = __riscv_vle32_v_f32m4(
                            pb1 + g - vl, vl);
                    vfloat32m4_t f1 = __riscv_vfwcvt_f_xu_v_f32m4(
                            __riscv_vzext_vf2_u16m2(c1, vl), vl);
                    acc = __riscv_vfmacc_vv_f32m4(acc, vb1, f1, vl);

                    vuint8m1_t c2 = __riscv_vmacc_vx_u8m1(
                            __riscv_vsrl_vx_u8m1(r1, 4, vl),
                            16,
                            __riscv_vand_vx_u8m1(r2, 0x03, vl),
                            vl);
                    vfloat32m4_t vb2 = __riscv_vle32_v_f32m4(
                            pb2 + g - vl, vl);
                    vfloat32m4_t f2 = __riscv_vfwcvt_f_xu_v_f32m4(
                            __riscv_vzext_vf2_u16m2(c2, vl), vl);
                    acc = __riscv_vfmacc_vv_f32m4(acc, vb2, f2, vl);

                    seg = seg_next; // rotate
                }

                // Epilogue: process the last full chunk (already loaded).
                {
                    vuint8m1_t r0 =
                            __riscv_vget_v_u8m1x3_u8m1(seg, 0);
                    vuint8m1_t r1 =
                            __riscv_vget_v_u8m1x3_u8m1(seg, 1);
                    vuint8m1_t r2 =
                            __riscv_vget_v_u8m1x3_u8m1(seg, 2);

                    // ID11: stream-order rescheduling — the two LIGHT
                    // extracts (c0: 1 op, c3: 1 op) are computed first
                    // and their convert+FMA chains issued immediately,
                    // so the first vfmacc starts ~4 ALU ops earlier;
                    // the two HEAVY extracts (c1/c2: 3 ops each) then
                    // overlap with the in-flight FMAs. Accumulation
                    // order becomes 0,3,1,2 (float reassociation, same
                    // semantics class as the other specializations).
                    // ID3 load-first order kept inside each stream.
                    vuint8m1_t c0 =
                            __riscv_vand_vx_u8m1(r0, 0x3F, vl);
                    vuint8m1_t c3 =
                            __riscv_vsrl_vx_u8m1(r2, 2, vl);

                    vfloat32m4_t vb0 = __riscv_vle32_v_f32m4(
                            pb0 + g - vl, vl);
                    vfloat32m4_t f0 = __riscv_vfwcvt_f_xu_v_f32m4(
                            __riscv_vzext_vf2_u16m2(c0, vl), vl);
                    acc = __riscv_vfmacc_vv_f32m4(acc, vb0, f0, vl);

                    vfloat32m4_t vb3 = __riscv_vle32_v_f32m4(
                            pb3 + g - vl, vl);
                    vfloat32m4_t f3 = __riscv_vfwcvt_f_xu_v_f32m4(
                            __riscv_vzext_vf2_u16m2(c3, vl), vl);
                    acc = __riscv_vfmacc_vv_f32m4(acc, vb3, f3, vl);

                    vuint8m1_t c1 = __riscv_vmacc_vx_u8m1(
                            __riscv_vsrl_vx_u8m1(r0, 6, vl),
                            4,
                            __riscv_vand_vx_u8m1(r1, 0x0F, vl),
                            vl);
                    vfloat32m4_t vb1 = __riscv_vle32_v_f32m4(
                            pb1 + g - vl, vl);
                    vfloat32m4_t f1 = __riscv_vfwcvt_f_xu_v_f32m4(
                            __riscv_vzext_vf2_u16m2(c1, vl), vl);
                    acc = __riscv_vfmacc_vv_f32m4(acc, vb1, f1, vl);

                    vuint8m1_t c2 = __riscv_vmacc_vx_u8m1(
                            __riscv_vsrl_vx_u8m1(r1, 4, vl),
                            16,
                            __riscv_vand_vx_u8m1(r2, 0x03, vl),
                            vl);
                    vfloat32m4_t vb2 = __riscv_vle32_v_f32m4(
                            pb2 + g - vl, vl);
                    vfloat32m4_t f2 = __riscv_vfwcvt_f_xu_v_f32m4(
                            __riscv_vzext_vf2_u16m2(c2, vl), vl);
                    acc = __riscv_vfmacc_vv_f32m4(acc, vb2, f2, vl);
                }
            }

            // Tail groups: one shorter-vl pass into the same accumulator
            // (only the first vt lanes are touched; reduction below
            // covers all vl lanes).
            if (g < ngf) {
                const size_t vt = __riscv_vsetvl_e8m1(ngf - g);

                vuint8m1x3_t seg =
                        __riscv_vlseg3e8_v_u8m1x3(code + 3 * g, vt);
                vuint8m1_t r0 = __riscv_vget_v_u8m1x3_u8m1(seg, 0);
                vuint8m1_t r1 = __riscv_vget_v_u8m1x3_u8m1(seg, 1);
                vuint8m1_t r2 = __riscv_vget_v_u8m1x3_u8m1(seg, 2);

                vuint8m1_t c0 = __riscv_vand_vx_u8m1(r0, 0x3F, vt);
                vuint8m1_t c1 = __riscv_vmacc_vx_u8m1(
                        __riscv_vsrl_vx_u8m1(r0, 6, vt),
                        4,
                        __riscv_vand_vx_u8m1(r1, 0x0F, vt),
                        vt);
                vuint8m1_t c2 = __riscv_vmacc_vx_u8m1(
                        __riscv_vsrl_vx_u8m1(r1, 4, vt),
                        16,
                        __riscv_vand_vx_u8m1(r2, 0x03, vt),
                        vt);
                vuint8m1_t c3 = __riscv_vsrl_vx_u8m1(r2, 2, vt);

                vfloat32m4_t f0 = __riscv_vfwcvt_f_xu_v_f32m4(
                        __riscv_vzext_vf2_u16m2(c0, vt), vt);
                acc = __riscv_vfmacc_vv_f32m4(
                        acc, __riscv_vle32_v_f32m4(pb0 + g, vt), f0, vt);

                vfloat32m4_t f1 = __riscv_vfwcvt_f_xu_v_f32m4(
                        __riscv_vzext_vf2_u16m2(c1, vt), vt);
                acc = __riscv_vfmacc_vv_f32m4(
                        acc, __riscv_vle32_v_f32m4(pb1 + g, vt), f1, vt);

                vfloat32m4_t f2 = __riscv_vfwcvt_f_xu_v_f32m4(
                        __riscv_vzext_vf2_u16m2(c2, vt), vt);
                acc = __riscv_vfmacc_vv_f32m4(
                        acc, __riscv_vle32_v_f32m4(pb2 + g, vt), f2, vt);

                vfloat32m4_t f3 = __riscv_vfwcvt_f_xu_v_f32m4(
                        __riscv_vzext_vf2_u16m2(c3, vt), vt);
                acc = __riscv_vfmacc_vv_f32m4(
                        acc, __riscv_vle32_v_f32m4(pb3 + g, vt), f3, vt);
            }

            // Single horizontal reduction over all vl lanes.
            vfloat32m1_t red = __riscv_vfredusum_vs_f32m4_f32m1(
                    acc, __riscv_vfmv_v_f_f32m1(0.0f, 1), vl);
            total = __riscv_vfmv_f_s_f32m1_f32(red);
        }

        // Scalar tail: dims beyond the last full group (d % 4). K_q
        // already contains their q_i * rmin_i part; add b_i * c_i.
        for (size_t i = 4 * ngf; i < d; i++) {
            total += q[i] * a_all[i] * float(sq6_decode_raw(code, i));
        }
        return k_q + total;
    }

    float query_to_code(const uint8_t* code) const final {
        return compute_ip(code);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        // Not on the benchmark-critical path; scalar per-dim evaluation.
        // IP(recon(c1), recon(c2)) = sum_k recon1_k * recon2_k
        const uint8_t* c1 = codes + i * code_size;
        const uint8_t* c2 = codes + j * code_size;
        float acc = 0;
        for (size_t k = 0; k < d; k++) {
            float r1 = rmin_all[k] +
                    a_all[k] * float(sq6_decode_raw(c1, k));
            float r2 = rmin_all[k] +
                    a_all[k] * float(sq6_decode_raw(c2, k));
            acc += r1 * r2;
        }
        return acc;
    }

    void query_to_codes_batch_4(
            const uint8_t* code_0,
            const uint8_t* code_1,
            const uint8_t* code_2,
            const uint8_t* code_3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) const final {
        dis0 = compute_ip(code_0);
        dis1 = compute_ip(code_1);
        dis2 = compute_ip(code_2);
        dis3 = compute_ip(code_3);
    }
};

/**********************************************************
 * QT_8bit (NON_UNIFORM) + L2 — full RVV specialization
 *
 * Per-dimension affine reconstruction:
 *     recon_i(c) = vmin[i] + vdiff[i] * (c + 0.5) / 255
 *                = rmin_i + a_i * c,  with a_i = vdiff[i] / 255,
 *                  rmin_i = vmin[i] + 0.5 * a_i
 * L2 distance per dimension contributes (q_i - recon_i(c))²
 *     = (e_i - a_i * c_i)²  with  e_i = q_i - rmin_i.
 * The constructor precomputes a_i / rmin_i; set_query precomputes
 * e_i once per query (amortized over all codes of the scan).
 *
 * Unlike the 4/6-bit codecs there is no bit-field unpacking: each
 * code byte is one dimension, so the kernel is a single contiguous
 * stream. Hot loop per vl = VLMAX(e8m1) dims (16 at VLEN=128):
 * vle8 -> vzext_vf2 (u8->u16m2) -> vfwcvt (u16->f32m4), then
 * t = e - a*c (vfnmsac) and acc += t*t (vfmacc).
 * vsetvl is hoisted: 0 explicit vsetvl in the hot loop, 1 hoisted
 * + 1 for the tail (d % vl != 0; absent for the benchmark d=768).
 **********************************************************/

template <>
struct DCTemplate<
        QuantizerTemplate<
                Codec8bit<SIMDLevel::RISCV_RVV>,
                QuantizerTemplateScaling::NON_UNIFORM,
                SIMDLevel::RISCV_RVV>,
        SimilarityL2<SIMDLevel::RISCV_RVV>,
        SIMDLevel::RISCV_RVV> : SQDistanceComputer {
    using Sim = SimilarityL2<SIMDLevel::RISCV_RVV>;

    size_t d;
    std::vector<float> a_v;    // a_i = vdiff[i] / 255
    std::vector<float> rmin_v; // vmin[i] + 0.5 * a_i
    std::vector<float> e_v;    // q_i - rmin_i, per query

    DCTemplate(size_t d_in, const std::vector<float>& trained)
            : d(d_in), a_v(d_in, 0.0f), rmin_v(d_in, 0.0f), e_v(d_in, 0.0f) {
        const float* vmin = trained.data();
        const float* vdiff = trained.data() + d_in;
        for (size_t i = 0; i < d_in; i++) {
            float a = vdiff[i] / 255.0f;
            a_v[i] = a;
            rmin_v[i] = vmin[i] + 0.5f * a;
        }
    }

    void set_query(const float* x) final {
        q = x;
        // e_i = q_i - rmin_i (amortized: once per query, 1/n weight)
        for (size_t i = 0; i < d; i++) {
            e_v[i] = x[i] - rmin_v[i];
        }
    }

    /// Full-precision vector L2 over the 1-byte-per-dim code.
    float compute_l2(const uint8_t* code) const {
        const float* pa = a_v.data();
        const float* pe = e_v.data();

        // Hoist vsetvl: VLMAX for e8m1 (== f32 lanes per m4 group)
        const size_t vl = __riscv_vsetvl_e8m1(d);
        vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);

        size_t i = 0;
        if (i + vl <= d) {
            // ID2: software pipeline depth 1 — preload the NEXT chunk's
            // code bytes at the top of THIS iteration. ID6: u8->f32 via
            // the u32 intermediate (vzext_vf4 + vfcvt) instead of the
            // u16 widening chain (vzext_vf2 + vfwcvt).
            vuint8m1_t c8 = __riscv_vle8_v_u8m1(code + i, vl); // prologue
            i += vl;

            for (; i + vl <= d; i += vl) {
                vuint8m1_t c8_next = __riscv_vle8_v_u8m1(code + i, vl);
                vfloat32m4_t cf = __riscv_vfcvt_f_xu_v_f32m4(
                        __riscv_vzext_vf4_u32m4(c8, vl), vl);
                vfloat32m4_t t =
                        __riscv_vle32_v_f32m4(pe + i - vl, vl);
                t = __riscv_vfnmsac_vv_f32m4(
                        t,
                        __riscv_vle32_v_f32m4(pa + i - vl, vl),
                        cf,
                        vl);
                acc = __riscv_vfmacc_vv_f32m4(acc, t, t, vl);
                c8 = c8_next; // rotate
            }

            // Epilogue: process the last full chunk (already loaded).
            {
                vfloat32m4_t cf = __riscv_vfcvt_f_xu_v_f32m4(
                        __riscv_vzext_vf4_u32m4(c8, vl), vl);
                vfloat32m4_t t =
                        __riscv_vle32_v_f32m4(pe + i - vl, vl);
                t = __riscv_vfnmsac_vv_f32m4(
                        t,
                        __riscv_vle32_v_f32m4(pa + i - vl, vl),
                        cf,
                        vl);
                acc = __riscv_vfmacc_vv_f32m4(acc, t, t, vl);
            }
        }

        // Tail: fewer than vl dims left; accumulates into the first
        // lanes of acc (safe: reduction below covers all vl lanes).
        if (i < d) {
            size_t vt = __riscv_vsetvl_e8m1(d - i);
            vuint8m1_t c8 = __riscv_vle8_v_u8m1(code + i, vt);
            vfloat32m4_t cf = __riscv_vfcvt_f_xu_v_f32m4(
                    __riscv_vzext_vf4_u32m4(c8, vt), vt);
            vfloat32m4_t t = __riscv_vle32_v_f32m4(pe + i, vt);
            t = __riscv_vfnmsac_vv_f32m4(
                    t, __riscv_vle32_v_f32m4(pa + i, vt), cf, vt);
            acc = __riscv_vfmacc_vv_f32m4(acc, t, t, vt);
        }

        // Horizontal reduce over all vl lanes
        vfloat32m1_t red = __riscv_vfredusum_vs_f32m4_f32m1(
                acc, __riscv_vfmv_v_f_f32m1(0.0f, 1), vl);
        return __riscv_vfmv_f_s_f32m1_f32(red);
    }

    float query_to_code(const uint8_t* code) const final {
        return compute_l2(code);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        // Not on the benchmark-critical path; scalar per-dim evaluation.
        // recon1 - recon2 = a_k * (c1_k - c2_k)
        const uint8_t* c1 = codes + i * code_size;
        const uint8_t* c2 = codes + j * code_size;
        float acc = 0;
        for (size_t k = 0; k < d; k++) {
            float diff = a_v[k] * float(int(c1[k]) - int(c2[k]));
            acc += diff * diff;
        }
        return acc;
    }

    void query_to_codes_batch_4(
            const uint8_t* code_0,
            const uint8_t* code_1,
            const uint8_t* code_2,
            const uint8_t* code_3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) const final {
        dis0 = compute_l2(code_0);
        dis1 = compute_l2(code_1);
        dis2 = compute_l2(code_2);
        dis3 = compute_l2(code_3);
    }
};

/**********************************************************
 * QT_8bit (NON_UNIFORM) + IP — full RVV specialization
 *
 * Per-dimension affine reconstruction:
 *     recon_i(c) = vmin[i] + vdiff[i] * (c + 0.5) / 255
 *                = rmin_i + a_i * c,  with a_i = vdiff[i] / 255,
 *                  rmin_i = vmin[i] + 0.5 * a_i
 * The inner product against the query decomposes into a query-only
 * constant plus a coefficient dot product over the integer codes:
 *     IP(q, code) = sum_i q_i * recon_i(c_i)
 *                 = sum_i q_i * rmin_i + sum_i (q_i * a_i) * c_i
 *                 = K_q + sum_i b_i * c_i
 * K_q and b_i = q_i * a_i depend only on the query, so set_query
 * precomputes them once per query (amortized over all codes of the
 * scan).
 *
 * Unlike the 4/6-bit codecs there is no bit-field unpacking: each
 * code byte is one dimension, so the kernel is a single contiguous
 * stream — the lightest non-uniform kernel in this file (5 vector
 * ops per block): vle8 (software-pipelined depth 1, board-proven on
 * qt_8bit_l2 R2 / qt_6bit_ip R1) -> vzext_vf4 (u8->u32m4) -> vfcvt
 * (u32->f32m4) -> vle32 (b) -> vfmacc (acc += b * c, single
 * accumulator). vsetvl is hoisted: 0 explicit vsetvl in the hot
 * loop, 1 hoisted + 1 for the tail (d % vl != 0; absent for the
 * benchmark d=768).
 **********************************************************/

template <>
struct DCTemplate<
        QuantizerTemplate<
                Codec8bit<SIMDLevel::RISCV_RVV>,
                QuantizerTemplateScaling::NON_UNIFORM,
                SIMDLevel::RISCV_RVV>,
        SimilarityIP<SIMDLevel::RISCV_RVV>,
        SIMDLevel::RISCV_RVV> : SQDistanceComputer {
    using Sim = SimilarityIP<SIMDLevel::RISCV_RVV>;

    size_t d;
    std::vector<float> a_v;    // a_i = vdiff[i] / 255
    std::vector<float> rmin_v; // vmin[i] + 0.5 * a_i
    std::vector<float> b_v;    // b_i = q_i * a_i, per query
    float k_q;                 // sum_i q_i * rmin_i, per query

    DCTemplate(size_t d_in, const std::vector<float>& trained)
            : d(d_in),
              a_v(d_in, 0.0f),
              rmin_v(d_in, 0.0f),
              // ID4: b_v padded by 2*VLMAX zeros so the software-
              // pipelined next-iteration vle32 loads are always in
              // bounds (padding lanes are loaded but never used).
              b_v(d_in + 2 * __riscv_vsetvlmax_e8m1(), 0.0f),
              k_q(0.0f) {
        const float* vmin = trained.data();
        const float* vdiff = trained.data() + d_in;
        for (size_t i = 0; i < d_in; i++) {
            float a = vdiff[i] / 255.0f;
            a_v[i] = a;
            rmin_v[i] = vmin[i] + 0.5f * a;
        }
    }

    void set_query(const float* x) final {
        q = x;
        // b_i = q_i * a_i and K_q = sum_i q_i * rmin_i, once per
        // query (amortized 1/n over the codes of the scan).
        float acc = 0.0f;
        for (size_t i = 0; i < d; i++) {
            b_v[i] = x[i] * a_v[i];
            acc += x[i] * rmin_v[i];
        }
        k_q = acc;
    }

    /// S = sum_i b_i * c_i over the 1-byte-per-dim code; returns
    /// K_q + S.
    float compute_ip(const uint8_t* code) const {
        const float* pb = b_v.data();

        // Hoist vsetvl: VLMAX for e8m1 (== f32 lanes per m4 group)
        const size_t vl = __riscv_vsetvl_e8m1(d);
        vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        // ID3: second accumulator — 2x unrolled main loop feeds
        // acc/acc1 alternately, halving the serial vfmacc chain.
        // (ID11 4x/4acc board-rejected: +112.96% vs v3, register
        // spill — ILP ceiling for this kernel is 2x.)
        vfloat32m4_t acc1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);

        size_t i = 0;
        if (i + vl <= d) {
            // Software pipeline depth 1 — preload the NEXT chunk's
            // code bytes at the top of THIS iteration.
            vuint8m1_t c8 = __riscv_vle8_v_u8m1(code + i, vl); // prologue
            i += vl;

            // ID3: 2x unroll + dual accumulators (blocks k -> acc,
            // k+1 -> acc1). ID4: the b-coefficient stream is ALSO
            // software-pipelined depth 1 — both halves' vle32 for the
            // NEXT iteration are issued in THIS iteration (rotate
            // vb0/vb1); loads past d land in the zero padding of b_v
            // and are discarded on loop exit. Branch-free hot loop.
            if (i + 2 * vl <= d) {
                // b-prologue: preload both halves of iteration 0.
                vfloat32m4_t vb0 = __riscv_vle32_v_f32m4(pb + i - vl, vl);
                vfloat32m4_t vb1 = __riscv_vle32_v_f32m4(pb + i, vl);
                for (; i + 2 * vl <= d; i += 2 * vl) {
                    vuint8m1_t c8_n1 =
                            __riscv_vle8_v_u8m1(code + i, vl);
                    vfloat32m4_t vb0_next =
                            __riscv_vle32_v_f32m4(pb + i + vl, vl);
                    vfloat32m4_t cf0 = __riscv_vfcvt_f_xu_v_f32m4(
                            __riscv_vzext_vf4_u32m4(c8, vl), vl);
                    acc = __riscv_vfmacc_vv_f32m4(acc, vb0, cf0, vl);

                    vuint8m1_t c8_n2 =
                            __riscv_vle8_v_u8m1(code + i + vl, vl);
                    vfloat32m4_t vb1_next = __riscv_vle32_v_f32m4(
                            pb + i + 2 * vl, vl);
                    vfloat32m4_t cf1 = __riscv_vfcvt_f_xu_v_f32m4(
                            __riscv_vzext_vf4_u32m4(c8_n1, vl), vl);
                    acc1 = __riscv_vfmacc_vv_f32m4(acc1, vb1, cf1, vl);

                    c8 = c8_n2;     // rotate code stream
                    vb0 = vb0_next; // rotate b stream
                    vb1 = vb1_next;
                }
            }

            // Odd full chunk left beyond the current one: process the
            // current chunk and advance the pipeline by one.
            if (i + vl <= d) {
                vuint8m1_t c8_n1 = __riscv_vle8_v_u8m1(code + i, vl);
                vfloat32m4_t vb = __riscv_vle32_v_f32m4(pb + i - vl, vl);
                vfloat32m4_t cf = __riscv_vfcvt_f_xu_v_f32m4(
                        __riscv_vzext_vf4_u32m4(c8, vl), vl);
                acc = __riscv_vfmacc_vv_f32m4(acc, vb, cf, vl);
                c8 = c8_n1;
                i += vl;
            }

            // Epilogue: process the last full chunk (already loaded).
            {
                vfloat32m4_t vb = __riscv_vle32_v_f32m4(pb + i - vl, vl);
                vfloat32m4_t cf = __riscv_vfcvt_f_xu_v_f32m4(
                        __riscv_vzext_vf4_u32m4(c8, vl), vl);
                acc1 = __riscv_vfmacc_vv_f32m4(acc1, vb, cf, vl);
            }
        }

        // Tail: fewer than vl dims left; accumulates into the first
        // lanes of acc (safe: reduction below covers all vl lanes).
        if (i < d) {
            size_t vt = __riscv_vsetvl_e8m1(d - i);
            vuint8m1_t c8 = __riscv_vle8_v_u8m1(code + i, vt);
            vfloat32m4_t cf = __riscv_vfcvt_f_xu_v_f32m4(
                    __riscv_vzext_vf4_u32m4(c8, vt), vt);
            acc = __riscv_vfmacc_vv_f32m4(
                    acc, __riscv_vle32_v_f32m4(pb + i, vt), cf, vt);
        }

        // Merge the two accumulators, then horizontal reduce.
        acc = __riscv_vfadd_vv_f32m4(acc, acc1, vl);
        // Horizontal reduce over all vl lanes
        vfloat32m1_t red = __riscv_vfredusum_vs_f32m4_f32m1(
                acc, __riscv_vfmv_v_f_f32m1(0.0f, 1), vl);
        return k_q + __riscv_vfmv_f_s_f32m1_f32(red);
    }

    float query_to_code(const uint8_t* code) const final {
        return compute_ip(code);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        // Not on the benchmark-critical path; scalar per-dim
        // evaluation: IP(recon(c1), recon(c2)) = sum_k r1_k * r2_k
        const uint8_t* c1 = codes + i * code_size;
        const uint8_t* c2 = codes + j * code_size;
        float acc = 0;
        for (size_t k = 0; k < d; k++) {
            float r1 = rmin_v[k] + a_v[k] * float(c1[k]);
            float r2 = rmin_v[k] + a_v[k] * float(c2[k]);
            acc += r1 * r2;
        }
        return acc;
    }

    void query_to_codes_batch_4(
            const uint8_t* code_0,
            const uint8_t* code_1,
            const uint8_t* code_2,
            const uint8_t* code_3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) const final {
        dis0 = compute_ip(code_0);
        dis1 = compute_ip(code_1);
        dis2 = compute_ip(code_2);
        dis3 = compute_ip(code_3);
    }
};

/**********************************************************
 * QT_8bit_uniform + L2 — integer-domain RVV specialization
 *
 * 8-bit UNIFORM scaling: every component reconstructs as
 *     recon(c) = vmin + vdiff * (c + 0.5) / 255
 *              = final_scale * c + bias
 * where final_scale = vdiff / 255.
 *
 * L2 distance reduces to final_scale² * Σ(q_byte - code_byte)²,
 * so we quantize the query to uint8_t once at set_query and then
 * compute integer-domain squared differences via RVV vector ops
 * without ever decoding to float.
 *
 * Pattern mirrors QT_4bit_uniform+L2 (above), but simpler: each
 * code byte is an independent value in [0,255] — no nibble split.
 **********************************************************/

template <>
struct DCTemplate<
        QuantizerTemplate<
                Codec8bit<SIMDLevel::RISCV_RVV>,
                QuantizerTemplateScaling::UNIFORM,
                SIMDLevel::RISCV_RVV>,
        SimilarityL2<SIMDLevel::RISCV_RVV>,
        SIMDLevel::RISCV_RVV> : SQDistanceComputer {
    using Sim = SimilarityL2<SIMDLevel::RISCV_RVV>;

    size_t d;
    float vmin;
    float vdiff;
    float final_scale_sq;
    std::vector<uint8_t> query_bytes;

    DCTemplate(size_t d_in, const std::vector<float>& trained)
            : d(d_in),
              vmin(trained[0]),
              vdiff(trained[1]),
              query_bytes(d_in, 0) {
        const float final_scale = vdiff / 255.0f;
        final_scale_sq = final_scale * final_scale;
    }

    void set_query(const float* x) final {
        this->q = x;
        if (vdiff == 0.0f) {
            memset(query_bytes.data(), 0, d);
            return;
        }
        const float inv_scale = 255.0f / vdiff;
        uint8_t* out = query_bytes.data();

        // ID4: branch-free RVV quantization replicating the scalar
        // semantics exactly: code = trunc((x-vmin)*inv_scale + 0.5)
        // clamped to [0, 255]. Clamping the FLOAT to [0, 255] before
        // the RTZ convert is equivalent to clamping the int after
        // (trunc is monotonic and 0/255 are exact in f32).
        // 16 floats per chunk at e32m4 (VLEN=128); narrowing via two
        // plain vnsrl-by-0 steps (values already <= 255, no clip
        // needed). vsetvl hoisted; one shorter-vl tail pass.
        const size_t vl = __riscv_vsetvl_e32m4(d > 0 ? d : 1);
        size_t i = 0;
        for (; i + vl <= d; i += vl) {
            vfloat32m4_t vx = __riscv_vle32_v_f32m4(x + i, vl);
            vfloat32m4_t val = __riscv_vfmul_vf_f32m4(
                    __riscv_vfsub_vf_f32m4(vx, vmin, vl), inv_scale, vl);
            val = __riscv_vfadd_vf_f32m4(val, 0.5f, vl);
            val = __riscv_vfmax_vf_f32m4(val, 0.0f, vl);
            val = __riscv_vfmin_vf_f32m4(val, 255.0f, vl);
            vuint32m4_t u32 = __riscv_vfcvt_rtz_xu_f_v_u32m4(val, vl);
            vuint16m2_t u16 = __riscv_vnsrl_wx_u16m2(u32, 0, vl);
            vuint8m1_t u8 = __riscv_vnsrl_wx_u8m1(u16, 0, vl);
            __riscv_vse8_v_u8m1(out + i, u8, vl);
        }
        if (i < d) {
            const size_t vt = __riscv_vsetvl_e32m4(d - i);
            vfloat32m4_t vx = __riscv_vle32_v_f32m4(x + i, vt);
            vfloat32m4_t val = __riscv_vfmul_vf_f32m4(
                    __riscv_vfsub_vf_f32m4(vx, vmin, vt), inv_scale, vt);
            val = __riscv_vfadd_vf_f32m4(val, 0.5f, vt);
            val = __riscv_vfmax_vf_f32m4(val, 0.0f, vt);
            val = __riscv_vfmin_vf_f32m4(val, 255.0f, vt);
            vuint32m4_t u32 = __riscv_vfcvt_rtz_xu_f_v_u32m4(val, vt);
            vuint16m2_t u16 = __riscv_vnsrl_wx_u16m2(u32, 0, vt);
            vuint8m1_t u8 = __riscv_vnsrl_wx_u8m1(u16, 0, vt);
            __riscv_vse8_v_u8m1(out + i, u8, vt);
        }
    }

    /// Integer-domain squared L2 via RVV vector ops, returning the
    /// final scaled float distance.
    /// ID1 (kept as the best form): vsetvl hoisted out of the hot loop,
    /// cross-iteration i32m8 vector accumulator, one reduction per call.
    /// Signed-difference kernel: vwsubu(q, c) widens to u16 whose
    /// 2's-complement bit pattern reinterprets as the exact signed
    /// difference in [-255, 255]; vwmacc squares and accumulates in a
    /// single op — 4 vector ops per chunk. Measured alternatives:
    /// absdiff+vwmulu+vwaddu 7-op (+47%), LMUL e8m1 variant (+5.1%),
    /// 2x unroll (+122.9%) — this machine is issue-limited and prefers
    /// the minimal e8m2 full-lane loop.
    /// ID8 variant under test: pointer-bump addressing in the hot loop
    /// and a fused reduction tail — reduce to i32m1, convert + scale in
    /// the vector unit (vfcvt + vfmul.vf + vfmv.f.s), avoiding the
    /// vmv.x.s GPR round-trip and the scalar int->float convert. The
    /// i32 total is exact while d <= 33025 (sum <= d * 65025 < 2^31);
    /// larger d takes the exact widening i64 path (cold branch).
    float l2_scaled(const uint8_t* code) const {
        const uint8_t* pq = query_bytes.data();
        const uint8_t* pc = code;

        // Hoist vsetvl: full VLMAX for e8m2, reused across the hot loop.
        const size_t vl = __riscv_vsetvl_e8m2(d > 0 ? d : 1);
        vint32m8_t acc32 = __riscv_vmv_v_x_i32m8(0, vl);

        size_t i = 0;
        for (; i + vl <= d; i += vl) {
            vuint8m2_t vq = __riscv_vle8_v_u8m2(pq, vl);
            vuint8m2_t vc = __riscv_vle8_v_u8m2(pc, vl);
            vint16m4_t d16 = __riscv_vreinterpret_v_u16m4_i16m4(
                    __riscv_vwsubu_vv_u16m4(vq, vc, vl));
            acc32 = __riscv_vwmacc_vv_i32m8(acc32, d16, d16, vl);
            pq += vl;
            pc += vl;
        }

        // Tail: fewer than vl dims left — one shorter-vl pass into the
        // same accumulator (only the first vt lanes are touched).
        if (i < d) {
            const size_t vt = __riscv_vsetvl_e8m2(d - i);
            vuint8m2_t vq = __riscv_vle8_v_u8m2(pq, vt);
            vuint8m2_t vc = __riscv_vle8_v_u8m2(pc, vt);
            vint16m4_t d16 = __riscv_vreinterpret_v_u16m4_i16m4(
                    __riscv_vwsubu_vv_u16m4(vq, vc, vt));
            acc32 = __riscv_vwmacc_vv_i32m8(acc32, d16, d16, vt);
        }

        if (d > 33025) {
            // Exact wide path for huge d (not reachable for realistic
            // dims): widening reduction i32m8 -> i64, scale in scalar.
            vint64m1_t z64 = __riscv_vmv_v_x_i64m1(0, 1);
            vint64m1_t r64 =
                    __riscv_vwredsum_vs_i32m8_i64m1(acc32, z64, vl);
            return static_cast<float>(__riscv_vmv_x_s_i64m1_i64(r64)) *
                    final_scale_sq;
        }

        // Single horizontal reduction over all vl lanes, then convert
        // and scale inside the vector unit — no GPR round-trip.
        vint32m1_t z32 = __riscv_vmv_v_x_i32m1(0, 1);
        vint32m1_t red = __riscv_vredsum_vs_i32m8_i32m1(acc32, z32, vl);
        vfloat32m1_t f = __riscv_vfcvt_f_x_v_f32m1(red, 1);
        f = __riscv_vfmul_vf_f32m1(f, final_scale_sq, 1);
        return __riscv_vfmv_f_s_f32m1_f32(f);
    }

    float query_to_code(const uint8_t* code) const final {
        return l2_scaled(code);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        const uint8_t* c1 = codes + i * code_size;
        const uint8_t* c2 = codes + j * code_size;
        int64_t acc = 0;
        for (size_t k = 0; k < d; k++) {
            int diff = int(c1[k]) - int(c2[k]);
            acc += diff * diff;
        }
        return static_cast<float>(acc) * final_scale_sq;
    }

    void query_to_codes_batch_4(
            const uint8_t* code_0,
            const uint8_t* code_1,
            const uint8_t* code_2,
            const uint8_t* code_3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) const final {
        dis0 = l2_scaled(code_0);
        dis1 = l2_scaled(code_1);
        dis2 = l2_scaled(code_2);
        dis3 = l2_scaled(code_3);
    }
};

/**********************************************************
 * QT_8bit_uniform + IP — RVV specialization
 *
 * 8-bit UNIFORM scaling: every component reconstructs as
 *     recon(c) = vmin + vdiff * (c + 0.5) / 255 = c0 + s * c
 * with s = vdiff / 255 and c0 = vmin + 0.5 * s (SHARED scalar
 * constants). The inner product folds into a query-only constant
 * plus one uniformly-scaled integer-code dot product:
 *     IP(q, code) = sum_i q_i * (c0 + s * c_i)
 *                 = c0 * sum_i q_i + s * sum_i q_i * c_i
 *                 = K_q + s * S
 * K_q = c0 * sum(q) is precomputed once per query in set_query
 * (amortized 1/n over the codes of the scan), so the hot loop only
 * evaluates S = sum_i q_i * c_i.
 *
 * ID2: integer-domain fixed-point rewrite (board-proven 4-op int
 * kernel shape from QT_8bit_uniform+L2 v6, 457ms vs the f32 5-op
 * kernel family's 911ms). set_query re-expresses the query in
 * per-query fixed point: s_q = max|q| / B, q16_i = round(q_i /
 * s_q) in [-B, B] (B = 8191 for d <= 1028; shrunk for larger d so
 * that d * B * 255 < 2^31 keeps the i32 accumulator exact). Then
 *     S ~= s_q * sum_i q16_i * c_i
 * and the hot loop is pure integer, 4 vector ops per vl = 32 dims
 * (e8m2 at VLEN=128): vle8 (c, u8m2) + vle16 (q16, i16m4) +
 * vzext_vf2 (c -> u16m4, reinterpret i16m4) + vwmacc (i16 x i16
 * -> i32m8 cross-iteration accumulator). Query representation
 * error <= s_q/2 per component (~6e-5 relative) — 3 orders of
 * magnitude below the code's own 8-bit quantization error, and
 * far tighter than the u8-grid query quantization precedent of
 * the QT_8bit_uniform+L2 specialization above.
 *
 * vsetvl hoisted: 0 in the hot loop, 1 hoisted + 1 tail. Reduction
 * tail fused in the vector domain (uniform_l2 R6 pattern): vredsum
 * i32m8 -> i32m1, vfcvt, then a single vfmacc.vf folds K_q +
 * factor * sum (factor = s * s_q) with no vector->GPR round-trip.
 * Pointer-bump addressing in the hot loop. Huge-d degradation
 * (B < 64, i.e. d > ~129k) takes an exact scalar cold path.
 **********************************************************/

template <>
struct DCTemplate<
        QuantizerTemplate<
                Codec8bit<SIMDLevel::RISCV_RVV>,
                QuantizerTemplateScaling::UNIFORM,
                SIMDLevel::RISCV_RVV>,
        SimilarityIP<SIMDLevel::RISCV_RVV>,
        SIMDLevel::RISCV_RVV> : SQDistanceComputer {
    using Sim = SimilarityIP<SIMDLevel::RISCV_RVV>;

    size_t d;
    float vmin;
    float vdiff;
    float scale;  // vdiff / 255
    float c0;     // vmin + 0.5 * scale
    float k_q;    // c0 * sum_i q_i, per query
    float factor; // scale * s_q, per query
    int32_t qbound; // fixed-point bound B (i32 overflow guard)
    // Fixed-point query stream; padded by 2*VLMAX(e16m4) zeros so a
    // future software-pipelined next-chunk vle16 stays in bounds.
    std::vector<int16_t> q16;

    DCTemplate(size_t d_in, const std::vector<float>& trained)
            : d(d_in),
              vmin(trained[0]),
              vdiff(trained[1]),
              k_q(0.0f),
              factor(0.0f),
              q16(d_in + 2 * __riscv_vsetvlmax_e16m4(), 0) {
        scale = vdiff / 255.0f;
        c0 = vmin + 0.5f * scale;
        // Largest B with d * B * 255 < 2^31 (exact i32 total), capped
        // at 8191 (13 bits: query rel. error <= 6.1e-5).
        int64_t cap = d_in > 0
                ? (int64_t(1) << 31) / (int64_t(255) * int64_t(d_in)) - 1
                : 8191;
        qbound = cap < 8191 ? int32_t(cap) : 8191;
    }

    void set_query(const float* x) final {
        q = x;
        // ID3: RVV-vectorized set_query (uniform_l2 R4 precedent).
        // Pass 1 fuses sum(q) and max|q| in one sweep: e32m4 chunks,
        // vfadd into a vector sum accumulator + vfabs/vfmax into a
        // vector max accumulator; single vfredusum/vfredmax at the
        // end. Pass 2 converts to fixed point: vfmul.vf(inv) ->
        // vfcvt_x_f_v (i32, dynamic rounding mode = RNE by default,
        // matching lrintf under the default FP environment) ->
        // vncvt narrowing i32->i16 (values already in [-B, B], no
        // clip needed) -> vse16.
        const size_t vlmax = __riscv_vsetvl_e32m4(d > 0 ? d : 1);
        vfloat32m4_t vsum = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
        vfloat32m4_t vmax = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
        size_t i = 0;
        for (; i + vlmax <= d; i += vlmax) {
            vfloat32m4_t vx = __riscv_vle32_v_f32m4(x + i, vlmax);
            vsum = __riscv_vfadd_vv_f32m4(vsum, vx, vlmax);
            vmax = __riscv_vfmax_vv_f32m4(
                    vmax, __riscv_vfabs_v_f32m4(vx, vlmax), vlmax);
        }
        if (i < d) {
            const size_t vt = __riscv_vsetvl_e32m4(d - i);
            vfloat32m4_t vx = __riscv_vle32_v_f32m4(x + i, vt);
            // Tail lanes merge into the first vt lanes of the
            // accumulators; reductions below cover all vlmax lanes.
            vsum = __riscv_vfadd_vv_f32m4_tu(vsum, vsum, vx, vt);
            vmax = __riscv_vfmax_vv_f32m4_tu(
                    vmax, vmax, __riscv_vfabs_v_f32m4(vx, vt), vt);
        }
        vfloat32m1_t z = __riscv_vfmv_v_f_f32m1(0.0f, 1);
        float sum = __riscv_vfmv_f_s_f32m1_f32(
                __riscv_vfredusum_vs_f32m4_f32m1(vsum, z, vlmax));
        float m = __riscv_vfmv_f_s_f32m1_f32(
                __riscv_vfredmax_vs_f32m4_f32m1(vmax, z, vlmax));
        k_q = c0 * sum;

        if (m == 0.0f || qbound < 64) {
            // All-zero query, or huge-d cold path: integer kernel
            // unused (factor = 0 makes the vector result K_q).
            factor = 0.0f;
            if (qbound >= 64) {
                memset(q16.data(), 0, d * sizeof(int16_t));
            }
            return;
        }
        // Pass 2: fixed-point conversion q16_i = round(q_i * inv).
        const float inv = float(qbound) / m;
        int16_t* out = q16.data();
        i = 0;
        for (; i + vlmax <= d; i += vlmax) {
            vfloat32m4_t vx = __riscv_vle32_v_f32m4(x + i, vlmax);
            vint32m4_t vi = __riscv_vfcvt_x_f_v_i32m4(
                    __riscv_vfmul_vf_f32m4(vx, inv, vlmax), vlmax);
            __riscv_vse16_v_i16m2(
                    out + i, __riscv_vncvt_x_x_w_i16m2(vi, vlmax), vlmax);
        }
        if (i < d) {
            const size_t vt = __riscv_vsetvl_e32m4(d - i);
            vfloat32m4_t vx = __riscv_vle32_v_f32m4(x + i, vt);
            vint32m4_t vi = __riscv_vfcvt_x_f_v_i32m4(
                    __riscv_vfmul_vf_f32m4(vx, inv, vt), vt);
            __riscv_vse16_v_i16m2(
                    out + i, __riscv_vncvt_x_x_w_i16m2(vi, vt), vt);
        }
        factor = scale * (m / float(qbound));
    }

    /// Integer-domain dot product; returns K_q + factor * Σ q16_i*c_i
    /// with the reduction tail fused in the vector unit.
    /// ID5 (under test): software pipeline depth 1 on the q16 stream
    /// — the NEXT chunk's vle16 is issued at the top of THIS
    /// iteration (qt_8bit_ip R5 second-stream precedent, -2.25%);
    /// loads past d land in the 2*VLMAX zero padding of q16 and are
    /// discarded on loop exit. Branch-free hot loop, +1 in-flight
    /// m4 group (22/32 registers).
    float ip_fixed(const uint8_t* code) const {
        const int16_t* pq = q16.data();
        const uint8_t* pc = code;

        // Hoist vsetvl: full VLMAX for e8m2, reused across the loop.
        const size_t vl = __riscv_vsetvl_e8m2(d > 0 ? d : 1);
        vint32m8_t acc = __riscv_vmv_v_x_i32m8(0, vl);

        size_t i = 0;
        if (i + vl <= d) {
            vint16m4_t vq = __riscv_vle16_v_i16m4(pq, vl); // prologue
            for (; i + vl <= d; i += vl) {
                vint16m4_t vq_next =
                        __riscv_vle16_v_i16m4(pq + vl, vl);
                vuint8m2_t c8 = __riscv_vle8_v_u8m2(pc, vl);
                vint16m4_t c16 = __riscv_vreinterpret_v_u16m4_i16m4(
                        __riscv_vzext_vf2_u16m4(c8, vl));
                acc = __riscv_vwmacc_vv_i32m8(acc, vq, c16, vl);
                vq = vq_next; // rotate
                pc += vl;
                pq += vl;
            }
        }

        // Tail: fewer than vl dims left — one shorter-vl pass into
        // the same accumulator (only the first vt lanes touched).
        if (i < d) {
            const size_t vt = __riscv_vsetvl_e8m2(d - i);
            vuint8m2_t c8 = __riscv_vle8_v_u8m2(pc, vt);
            vint16m4_t vq = __riscv_vle16_v_i16m4(pq, vt);
            vint16m4_t c16 = __riscv_vreinterpret_v_u16m4_i16m4(
                    __riscv_vzext_vf2_u16m4(c8, vt));
            acc = __riscv_vwmacc_vv_i32m8(acc, vq, c16, vt);
        }

        // Single horizontal reduction, then convert + fold K_q and
        // factor inside the vector unit — no GPR round-trip.
        vint32m1_t z32 = __riscv_vmv_v_x_i32m1(0, 1);
        vint32m1_t red = __riscv_vredsum_vs_i32m8_i32m1(acc, z32, vl);
        vfloat32m1_t f = __riscv_vfcvt_f_x_v_f32m1(red, 1);
        vfloat32m1_t vk = __riscv_vfmv_v_f_f32m1(k_q, 1);
        vk = __riscv_vfmacc_vf_f32m1(vk, factor, f, 1);
        return __riscv_vfmv_f_s_f32m1_f32(vk);
    }

    float query_to_code(const uint8_t* code) const final {
        if (qbound < 64) {
            // Exact scalar cold path for absurdly large d (i32
            // fixed-point budget too small; unreachable for
            // realistic dims).
            float S = 0.0f;
            for (size_t k = 0; k < d; k++) {
                S += q[k] * float(code[k]);
            }
            return k_q + scale * S;
        }
        return ip_fixed(code);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        // Not on the benchmark-critical path; scalar per-dim
        // evaluation: IP(recon(c1), recon(c2)) = sum_k r1_k * r2_k
        const uint8_t* code1 = codes + i * code_size;
        const uint8_t* code2 = codes + j * code_size;
        float acc = 0;
        for (size_t k = 0; k < d; k++) {
            float r1 = c0 + scale * float(code1[k]);
            float r2 = c0 + scale * float(code2[k]);
            acc += r1 * r2;
        }
        return acc;
    }

    void query_to_codes_batch_4(
            const uint8_t* code_0,
            const uint8_t* code_1,
            const uint8_t* code_2,
            const uint8_t* code_3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) const final {
        dis0 = query_to_code(code_0);
        dis1 = query_to_code(code_1);
        dis2 = query_to_code(code_2);
        dis3 = query_to_code(code_3);
    }
};

/**********************************************************
 * QT_8bit_direct + L2 — full RVV specialization
 *
 * Direct 8-bit storage: encode is code[i] = (uint8_t)x[i] and
 * reconstruction is simply recon_i(c) = c_i (no vmin/vdiff affine,
 * no trained parameters). The distance is therefore
 *     L2(q, code) = sum_i (q_i - c_i)^2
 * with the raw byte value participating directly — the lightest
 * possible L2 kernel in this file: there are no a/e coefficient
 * streams at all, the query float stream is used as-is.
 *
 * ID1: hot loop per vl = VLMAX(e8m1) dims (16 at VLEN=128):
 * vle8 -> vzext_vf4 (u8->u32m4) -> vfcvt (u32->f32m4), then
 * t = q - c (vfsub) and acc += t*t (vfmacc). 6 vector ops per
 * block, single f32m4 accumulator, single vfredusum after the
 * loop. vsetvl hoisted: 0 explicit vsetvl in the hot loop, 1
 * hoisted + 1 for the tail (d % vl != 0; absent for d=768).
 *
 * ID2: integer-domain rewrite. set_query truncates the query to
 * bytes once (q8[i] = clamp(int(x[i]), 0, 255)) — the exact
 * semantics of x86 DistanceComputerByte::set_query (tmp[i] =
 * int(x[i])), which IS the kernel faiss runs for this qtype on
 * AVX2/AVX512 whenever d%16==0 / d%32==0 (both hit at d=768).
 * For in-contract data (components are byte values, per the
 * encoder code[i] = (uint8_t)x[i]) the result is bit-exact.
 * Hot loop becomes pure integer, 4 vector ops per block:
 * vle8(c) + vle8(q8) -> vwsubu (u8-u8 -> u16m2, reinterpret
 * i16m2: signed diff in [-255,255]) -> vwmacc (i16*i16 ->
 * i32m4 accumulator; max d*255^2 means no overflow for any
 * d < 33k). One vredsum + one int->float convert at the end.
 * Register-width traffic per block drops from ~21 beats
 * (m1+m4+m4+m4+m4+m4) to ~8 (m1+m1+m2+m4).
 *
 * (x86 note: AVX2/AVX512 route this qtype to DistanceComputerByte
 * — an integer-domain kernel with the query truncated to bytes —
 * whenever d%16==0 / d%32==0. The RVV dispatch keeps the
 * DCTemplate entry, so this specialization is the RVV analogue.)
 **********************************************************/

template <>
struct DCTemplate<
        Quantizer8bitDirect<SIMDLevel::RISCV_RVV>,
        SimilarityL2<SIMDLevel::RISCV_RVV>,
        SIMDLevel::RISCV_RVV> : SQDistanceComputer {
    using Sim = SimilarityL2<SIMDLevel::RISCV_RVV>;

    size_t d;
    std::vector<uint8_t> q8; // query truncated to bytes, per query

    DCTemplate(size_t d_in, const std::vector<float>& /* unused */)
            : d(d_in), q8(d_in, 0) {}

    void set_query(const float* x) final {
        q = x;
        // Truncate the query into the byte domain once per query
        // (amortized 1/n over the codes of the scan). Same semantics
        // as x86 DistanceComputerByte::set_query (tmp[i] = int(x[i]))
        // plus clamping so out-of-contract values cannot wrap.
        for (size_t i = 0; i < d; i++) {
            int v = static_cast<int>(x[i]);
            if (v < 0) {
                v = 0;
            }
            if (v > 255) {
                v = 255;
            }
            q8[i] = static_cast<uint8_t>(v);
        }
    }

    /// Integer-domain L2: sum_i (q8_i - c_i)^2 into an i32 vector
    /// accumulator; single reduction + single int->float at the end.
    /// ID5: LMUL m2 variant — e8m2 loads, vwsubu -> i16m4, vwmacc ->
    /// i32m8 single accumulator. Per 32 dims this is 4 instructions
    /// (2 loads m2 + m4-write + m8-write, 16 data beats) vs the v4
    /// 2x-unrolled m1 form's 8 instructions (same 16 beats): halves
    /// the front-end pressure at equal datapath traffic. Register
    /// budget: acc(8) + vc(2) + vq(2) + df(4) = 16/32, no spill.
    /// (Sibling LMUL-up refutations were all on f32-heavy kernels;
    /// this integer form is the premise-changed retest.)
    int64_t accumulate_int_l2(const uint8_t* code) const {
        const uint8_t* pq = q8.data();

        // Hoist vsetvl: VLMAX for e8m2, reused across the hot loop.
        const size_t vl = __riscv_vsetvl_e8m2(d);
        vint32m8_t acc = __riscv_vmv_v_x_i32m8(0, vl);

        size_t i = 0;
        // Hot loop: 2 loads, one widening subtract, one widening
        // square-accumulate. No vsetvl, no reduction inside.
        for (; i + vl <= d; i += vl) {
            vuint8m2_t vc = __riscv_vle8_v_u8m2(code + i, vl);
            vuint8m2_t vq = __riscv_vle8_v_u8m2(pq + i, vl);
            vint16m4_t df = __riscv_vreinterpret_v_u16m4_i16m4(
                    __riscv_vwsubu_vv_u16m4(vq, vc, vl));
            acc = __riscv_vwmacc_vv_i32m8(acc, df, df, vl);
        }

        // Tail: fewer than vl dims left — one shorter-vl pass into
        // the same accumulator (only the first vt lanes touched).
        if (i < d) {
            const size_t vt = __riscv_vsetvl_e8m2(d - i);
            vuint8m2_t vc = __riscv_vle8_v_u8m2(code + i, vt);
            vuint8m2_t vq = __riscv_vle8_v_u8m2(pq + i, vt);
            vint16m4_t df = __riscv_vreinterpret_v_u16m4_i16m4(
                    __riscv_vwsubu_vv_u16m4(vq, vc, vt));
            acc = __riscv_vwmacc_vv_i32m8(acc, df, df, vt);
        }

        // Single horizontal reduction over all vl lanes.
        vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
        vint32m1_t red = __riscv_vredsum_vs_i32m8_i32m1(acc, zero, vl);
        return __riscv_vmv_x_s_i32m1_i32(red);
    }

    float query_to_code(const uint8_t* code) const final {
        return static_cast<float>(accumulate_int_l2(code));
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        // Not on the benchmark-critical path; exact integer-domain
        // scalar evaluation (codes are raw bytes).
        const uint8_t* c1 = codes + i * code_size;
        const uint8_t* c2 = codes + j * code_size;
        int64_t acc = 0;
        for (size_t k = 0; k < d; k++) {
            int diff = int(c1[k]) - int(c2[k]);
            acc += diff * diff;
        }
        return static_cast<float>(acc);
    }

    void query_to_codes_batch_4(
            const uint8_t* code_0,
            const uint8_t* code_1,
            const uint8_t* code_2,
            const uint8_t* code_3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) const final {
        dis0 = static_cast<float>(accumulate_int_l2(code_0));
        dis1 = static_cast<float>(accumulate_int_l2(code_1));
        dis2 = static_cast<float>(accumulate_int_l2(code_2));
        dis3 = static_cast<float>(accumulate_int_l2(code_3));
    }
};

/**********************************************************
 * QT_8bit_direct + IP — full RVV specialization
 *
 * Direct 8-bit storage: reconstruction is recon_i(c) = c_i (no
 * vmin/vdiff affine, no trained parameters), so
 *     IP(q, code) = sum_i q_i * c_i
 * with the raw byte value participating directly — even lighter
 * than the direct+L2 kernel (no difference, no square): there are
 * no coefficient streams at all.
 *
 * ID1: float-domain hot loop per vl = VLMAX(e8m1) dims (16 at
 * VLEN=128): vle8(c) -> vzext_vf4 -> vfcvt -> vle32(q) -> vfmacc.
 * 5 vector ops per block, single f32m4 accumulator. Board:
 * -83.70% vs scalar baseline.
 *
 * ID2: integer-domain rewrite. set_query truncates the query to
 * bytes once (q8[i] = clamp(int(x[i]), 0, 255)) — the exact
 * semantics of x86 DistanceComputerByte::set_query (tmp[i] =
 * int(x[i])), which IS the kernel faiss runs for this qtype on
 * AVX2/AVX512 whenever d%16==0 / d%32==0 (both hit at d=768).
 * For in-contract data (components are byte values, per the
 * encoder code[i] = (uint8_t)x[i]) the result is bit-exact.
 * Hot loop becomes pure integer, 4 vector ops per block (e8m2,
 * vl=32 at VLEN=128): vle8(c) + vle8(q8) -> vwmulu (u8*u8 ->
 * u16m4; a single product <= 255^2 = 65025 just fits u16, so
 * the product CANNOT be accumulated in u16 — it is immediately
 * widened) -> vwaddu.wv (u32m8 accumulator += u16m4; max
 * d*65025 means no overflow for any d < 66k). One vredsum +
 * one uint->float convert at the end. Zero FP / zero convert
 * instructions in the hot loop. (Note: unlike the L2 kernel's
 * signed i16 diff in [-255,255], the IP product does NOT fit
 * i16 — the unsigned vwmulu/vwaddu path is mandatory.)
 *
 * (x86 note: AVX2/AVX512 route this qtype to DistanceComputerByte
 * — an integer-domain kernel with the query truncated to bytes —
 * whenever d%16==0 / d%32==0. The RVV dispatch keeps the
 * DCTemplate entry, so this specialization is the RVV analogue.)
 **********************************************************/

template <>
struct DCTemplate<
        Quantizer8bitDirect<SIMDLevel::RISCV_RVV>,
        SimilarityIP<SIMDLevel::RISCV_RVV>,
        SIMDLevel::RISCV_RVV> : SQDistanceComputer {
    using Sim = SimilarityIP<SIMDLevel::RISCV_RVV>;

    size_t d;
    // ID5 probe: query truncated to the byte range but stored
    // pre-widened as u16 — the hot loop loads it directly at
    // SEW=16 and uses a single fused vwmaccu instead of
    // vwmulu + vwaddu.wv (query stream bytes double: still L1).
    std::vector<uint16_t> q16; // query truncated to [0,255], u16

    DCTemplate(size_t d_in, const std::vector<float>& /* unused */)
            : d(d_in), q16(d_in, 0) {}

    void set_query(const float* x) final {
        q = x;
        // Truncate the query into the byte domain once per query
        // (amortized 1/n over the codes of the scan). Same semantics
        // as x86 DistanceComputerByte::set_query (tmp[i] = int(x[i]))
        // plus clamping so out-of-contract values cannot wrap.
        for (size_t i = 0; i < d; i++) {
            int v = static_cast<int>(x[i]);
            if (v < 0) {
                v = 0;
            }
            if (v > 255) {
                v = 255;
            }
            q16[i] = static_cast<uint16_t>(v);
        }
    }

    /// Integer-domain IP: sum_i q_i * c_i into a u32 vector
    /// accumulator; single reduction + single uint->float at the
    /// end. Base form ID5 (board: -90.70% vs baseline, -14.05%
    /// vs the vwmulu+vwaddu chain): fused-MAC — q stream
    /// pre-widened to u16 in set_query; per 32 dims: vle8(c,e8m2)
    /// -> vzext_vf2 + vle16(q16,u16m4) -> vwmaccu (u32m8 acc).
    /// ID7 (round 6 probe): bump-pointer addressing — replace the
    /// indexed `code + i` / `pq + i` forms and the i counter with
    /// two stepped pointers and a remaining-count, trimming the
    /// per-iteration scalar address arithmetic on this
    /// front-end-bound machine. Vector op sequence is unchanged.
    uint32_t accumulate_int_ip(const uint8_t* code) const {
        const uint16_t* pq = q16.data();
        const uint8_t* pc = code;

        // Hoist vsetvl: VLMAX for e8m2, reused across the hot loop.
        const size_t vl = __riscv_vsetvl_e8m2(d);
        vuint32m8_t acc = __riscv_vmv_v_x_u32m8(0, vl);

        size_t remaining = d;
        // Hot loop: 2 loads, one zero-extend, one fused widening
        // multiply-accumulate. No vsetvl, no reduction inside;
        // bump-pointer form, no indexed addressing.
        while (remaining >= vl) {
            vuint8m2_t vc = __riscv_vle8_v_u8m2(pc, vl);
            vuint16m4_t vq = __riscv_vle16_v_u16m4(pq, vl);
            vuint16m4_t c16 = __riscv_vzext_vf2_u16m4(vc, vl);
            acc = __riscv_vwmaccu_vv_u32m8(acc, vq, c16, vl);
            pc += vl;
            pq += vl;
            remaining -= vl;
        }

        // Tail: fewer than vl dims left — one shorter-vl pass into
        // the same accumulator (only the first vt lanes touched).
        if (remaining > 0) {
            const size_t vt = __riscv_vsetvl_e8m2(remaining);
            vuint8m2_t vc = __riscv_vle8_v_u8m2(pc, vt);
            vuint16m4_t vq = __riscv_vle16_v_u16m4(pq, vt);
            vuint16m4_t c16 = __riscv_vzext_vf2_u16m4(vc, vt);
            acc = __riscv_vwmaccu_vv_u32m8(acc, vq, c16, vt);
        }

        // Single horizontal reduction over all vl lanes.
        vuint32m1_t zero = __riscv_vmv_v_x_u32m1(0, 1);
        vuint32m1_t red = __riscv_vredsum_vs_u32m8_u32m1(acc, zero, vl);
        return __riscv_vmv_x_s_u32m1_u32(red);
    }

    float query_to_code(const uint8_t* code) const final {
        return static_cast<float>(accumulate_int_ip(code));
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        // Not on the benchmark-critical path; exact integer-domain
        // scalar evaluation (codes are raw bytes).
        const uint8_t* c1 = codes + i * code_size;
        const uint8_t* c2 = codes + j * code_size;
        int64_t acc = 0;
        for (size_t k = 0; k < d; k++) {
            acc += int(c1[k]) * int(c2[k]);
        }
        return static_cast<float>(acc);
    }

    void query_to_codes_batch_4(
            const uint8_t* code_0,
            const uint8_t* code_1,
            const uint8_t* code_2,
            const uint8_t* code_3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) const final {
        dis0 = static_cast<float>(accumulate_int_ip(code_0));
        dis1 = static_cast<float>(accumulate_int_ip(code_1));
        dis2 = static_cast<float>(accumulate_int_ip(code_2));
        dis3 = static_cast<float>(accumulate_int_ip(code_3));
    }
};

/**********************************************************
 * QT_8bit_direct_signed + L2 — full RVV specialization
 *
 * Signed direct 8-bit storage (see Quantizer8bitDirectSigned):
 *     stored_byte = value + 128,  i.e.  value = stored_byte - 128
 * Reconstruction is recon_i(c) = c_i - 128, so
 *     L2(q, code) = sum_i (q_i - (c_i - 128))^2
 *                 = sum_i ((q_i + 128) - c_i)^2
 * — the +128 storage bias cancels inside the difference (same
 * identity x86 DistanceComputerByteSigned<AVX512_SPR> relies on:
 * (s_a - 128) - (s_b - 128) == s_a - s_b), so the kernel is
 * structurally identical to the unsigned QT_8bit_direct+L2 one
 * with the query re-biased into the storage domain once per
 * query in set_query.
 *
 * ID1: float-domain hot loop per vl = VLMAX(e8m1) dims (16 at
 * VLEN=128): set_query precomputes qb[i] = x[i] + 128.0f (bias
 * folding, zero extra hot-loop cost), then vle8(c) ->
 * vzext_vf4 (u8->u32m4) -> vfcvt (u32->f32m4) -> vle32(qb) ->
 * vfsub -> vfmacc. 6 vector ops per block, single f32m4
 * accumulator, single vfredusum after the loop. Board: -79.76%
 * vs the scalar baseline.
 *
 * ID2: integer-domain rewrite (port of the direct_l2 final v5
 * kernel — bias cancellation makes the two kernels isomorphic).
 * set_query re-biases the query to bytes once (q8[i] =
 * clamp(int(x[i]) + 128, 0, 255)) — the exact semantics of x86
 * DistanceComputerByteSigned::set_query (tmp[i] =
 * uint8(int(x[i]) + 128)), which IS the kernel faiss runs for
 * this qtype on AVX512_SPR whenever d%64==0 (hit at d=768).
 * For in-contract data (components are integers in [-128,127],
 * per the encoder code[i] = (uint8_t)(x[i] + 128)) the result
 * is bit-exact. Hot loop becomes pure integer, 4 vector ops per
 * block (e8m2, vl=32 at VLEN=128): vle8(c) + vle8(q8) ->
 * vwsubu (u8-u8 -> u16m4, reinterpret i16m4: signed diff in
 * [-255,255]) -> vwmacc (i16*i16 -> i32m8 single accumulator;
 * max d*255^2 means no overflow for any d < 33k). One vredsum +
 * one int->float convert at the end; zero FP / zero convert
 * instructions in the hot loop.
 *
 * (x86 note: AVX512_SPR routes this qtype to
 * DistanceComputerByteSigned — an integer-domain kernel with the
 * query re-biased by +128 into bytes — whenever d%64==0 (hit at
 * d=768). The RVV dispatch keeps the DCTemplate entry, so this
 * specialization is the RVV analogue.)
 **********************************************************/

template <>
struct DCTemplate<
        Quantizer8bitDirectSigned<SIMDLevel::RISCV_RVV>,
        SimilarityL2<SIMDLevel::RISCV_RVV>,
        SIMDLevel::RISCV_RVV> : SQDistanceComputer {
    using Sim = SimilarityL2<SIMDLevel::RISCV_RVV>;

    size_t d;
    std::vector<uint8_t> q8; // query re-biased by +128 (storage domain)

    DCTemplate(size_t d_in, const std::vector<float>& /* unused */)
            : d(d_in), q8(d_in, 0) {}

    void set_query(const float* x) final {
        q = x;
        // Re-bias the query into the byte storage domain once per
        // query (amortized 1/n over the codes of the scan):
        // (x_i - (c_i - 128))^2 == ((x_i + 128) - c_i)^2. Same
        // semantics as x86 DistanceComputerByteSigned::set_query
        // (tmp[i] = uint8(int(x[i]) + 128)) plus clamping so
        // out-of-contract values cannot wrap.
        for (size_t i = 0; i < d; i++) {
            int v = static_cast<int>(x[i]) + 128;
            if (v < 0) {
                v = 0;
            }
            if (v > 255) {
                v = 255;
            }
            q8[i] = static_cast<uint8_t>(v);
        }
    }

    /// Integer-domain L2: sum_i (q8_i - c_i)^2 into an i32 vector
    /// accumulator; single reduction + single int->float at the end.
    /// LMUL m2 flat form (direct_l2 board-proven final shape):
    /// e8m2 loads, vwsubu -> i16m4, vwmacc -> i32m8 single
    /// accumulator. 4 instructions / 32 dims; register budget
    /// acc(8) + vc(2) + vq(2) + df(4) = 16/32, no spill.
    /// ID3 (round 3 probe): bump-pointer addressing — replace the
    /// indexed `code + i` / `pq + i` forms and the i counter with
    /// two stepped pointers and a remaining-count, trimming the
    /// per-iteration scalar address arithmetic on this
    /// front-end-bound machine (direct_ip round-6 board-proven
    /// -2.07%). Vector op sequence is unchanged.
    int64_t accumulate_int_l2(const uint8_t* code) const {
        const uint8_t* pq = q8.data();
        const uint8_t* pc = code;

        // Hoist vsetvl: VLMAX for e8m2, reused across the hot loop.
        const size_t vl = __riscv_vsetvl_e8m2(d);
        vint32m8_t acc = __riscv_vmv_v_x_i32m8(0, vl);

        size_t remaining = d;
        // Hot loop: 2 loads, one widening subtract, one widening
        // square-accumulate. No vsetvl, no reduction inside;
        // bump-pointer form, no indexed addressing.
        while (remaining >= vl) {
            vuint8m2_t vc = __riscv_vle8_v_u8m2(pc, vl);
            vuint8m2_t vq = __riscv_vle8_v_u8m2(pq, vl);
            vint16m4_t df = __riscv_vreinterpret_v_u16m4_i16m4(
                    __riscv_vwsubu_vv_u16m4(vq, vc, vl));
            acc = __riscv_vwmacc_vv_i32m8(acc, df, df, vl);
            pc += vl;
            pq += vl;
            remaining -= vl;
        }

        // Tail: fewer than vl dims left — one shorter-vl pass into
        // the same accumulator (only the first vt lanes touched).
        if (remaining > 0) {
            const size_t vt = __riscv_vsetvl_e8m2(remaining);
            vuint8m2_t vc = __riscv_vle8_v_u8m2(pc, vt);
            vuint8m2_t vq = __riscv_vle8_v_u8m2(pq, vt);
            vint16m4_t df = __riscv_vreinterpret_v_u16m4_i16m4(
                    __riscv_vwsubu_vv_u16m4(vq, vc, vt));
            acc = __riscv_vwmacc_vv_i32m8(acc, df, df, vt);
        }

        // Single horizontal reduction over all vl lanes.
        vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
        vint32m1_t red = __riscv_vredsum_vs_i32m8_i32m1(acc, zero, vl);
        return __riscv_vmv_x_s_i32m1_i32(red);
    }

    float query_to_code(const uint8_t* code) const final {
        return static_cast<float>(accumulate_int_l2(code));
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        // Not on the benchmark-critical path; exact integer-domain
        // scalar evaluation ((c1-128)-(c2-128) == c1-c2).
        const uint8_t* c1 = codes + i * code_size;
        const uint8_t* c2 = codes + j * code_size;
        int64_t acc = 0;
        for (size_t k = 0; k < d; k++) {
            int diff = int(c1[k]) - int(c2[k]);
            acc += diff * diff;
        }
        return static_cast<float>(acc);
    }

    void query_to_codes_batch_4(
            const uint8_t* code_0,
            const uint8_t* code_1,
            const uint8_t* code_2,
            const uint8_t* code_3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) const final {
        dis0 = static_cast<float>(accumulate_int_l2(code_0));
        dis1 = static_cast<float>(accumulate_int_l2(code_1));
        dis2 = static_cast<float>(accumulate_int_l2(code_2));
        dis3 = static_cast<float>(accumulate_int_l2(code_3));
    }
};

/**********************************************************
 * QT_8bit_direct_signed + IP — full RVV specialization
 *
 * Signed direct 8-bit storage (see Quantizer8bitDirectSigned):
 *     stored_byte = value + 128,  i.e.  value = stored_byte - 128
 * Reconstruction is recon_i(c) = c_i - 128, so
 *     IP(q, code) = sum_i q_i * (c_i - 128)
 * Unlike L2, the +128 storage bias does NOT cancel inside a
 * difference: expanding q_i * (c_i - 128) leaves a
 * -128 * sum_i q_i term. That term depends on the QUERY only,
 * so it can be hoisted out of the per-code kernel entirely
 * (precomputed once per query in set_query) — x86
 * DistanceComputerByteSigned<AVX512_SPR> uses the same algebra,
 * with its code-side sums; here the whole bias is query-side.
 *
 * ID1: float-domain hot loop per vl = VLMAX(e8m1) dims (16 at
 * VLEN=128): vle8(c) -> vzext_vf4 (u8->u32m4) -> vfcvt
 * (u32->f32m4) -> vfsub(128.0f) -> vle32(q) -> vfmacc. 6 vector
 * ops per block, single f32m4 accumulator, single vfredusum
 * after the loop. Board: -79.75% vs the scalar baseline.
 *
 * ID2: integer-domain rewrite. set_query truncates the query
 * once into the signed byte range, stored pre-widened as i16
 * (qs[i] = clamp(int(x[i]), -128, 127)) and precomputes the
 * query-only bias 128 * sum_i qs_i. This is the exact semantics
 * of x86 DistanceComputerByteSigned::set_query (tmp[i] =
 * uint8(int(x[i]) + 128)) — which IS the kernel faiss runs for
 * this qtype on AVX512_SPR whenever d%64==0 (hit at d=768) —
 * with both (tmp_i - 128) and (c_i - 128) expanded so that the
 * only non-query term left in
 *     IP = sum_i qs_i * (c_i - 128)
 *        = sum_i qs_i * c_i  -  128 * sum_i qs_i
 * is a plain mixed-sign dot product. For in-contract data
 * (components are integers in [-128,127], per the encoder
 * code[i] = (uint8_t)(x[i] + 128)) the result is bit-exact.
 * Hot loop per vl=32 dims (e8m2, VLEN=128): vle8(c) ->
 * vzext_vf2 (u8->u16m4) + vle16(qs,i16m4) -> vwmaccsu
 * (i16 x u16 -> i32m8 single accumulator) — 4 vector ops, the
 * direct_ip final-kernel shape (fused MAC on the acc chain,
 * off-chain only the zext). Products lie in [-32640, 32385], so
 * the i32 accumulator is safe for any d < 65k. The -128*sum(qs)
 * bias is injected as the vredsum SEED (vmv.v.x of -bias):
 * zero extra instructions in both the hot loop and the tail.
 * One vredsum + one int->float at the end; zero FP / zero
 * convert instructions in the hot loop.
 **********************************************************/

template <>
struct DCTemplate<
        Quantizer8bitDirectSigned<SIMDLevel::RISCV_RVV>,
        SimilarityIP<SIMDLevel::RISCV_RVV>,
        SIMDLevel::RISCV_RVV> : SQDistanceComputer {
    using Sim = SimilarityIP<SIMDLevel::RISCV_RVV>;

    size_t d;
    // ID2 probe: query truncated to the signed byte range but
    // stored pre-widened as i16 — the hot loop loads it directly
    // at SEW=16 and uses a single fused vwmaccsu (signed q x
    // unsigned zext'ed code) on the accumulator chain.
    std::vector<int16_t> q16; // query truncated to [-128,127], i16
    int32_t qbias = 0;        // 128 * sum_i q16[i] (query-only term)

    DCTemplate(size_t d_in, const std::vector<float>& /* unused */)
            : d(d_in), q16(d_in, 0) {}

    void set_query(const float* x) final {
        q = x;
        // Truncate the query into the signed byte domain once per
        // query (amortized 1/n over the codes of the scan). Same
        // semantics as x86 DistanceComputerByteSigned::set_query
        // (tmp[i] = uint8(int(x[i]) + 128), used there as
        // tmp_i - 128) plus clamping so out-of-contract values
        // cannot wrap. The query-only bias 128*sum(qs) is
        // precomputed here so the per-code kernel never sees it.
        int32_t s = 0;
        for (size_t i = 0; i < d; i++) {
            int v = static_cast<int>(x[i]);
            if (v < -128) {
                v = -128;
            }
            if (v > 127) {
                v = 127;
            }
            q16[i] = static_cast<int16_t>(v);
            s += v;
        }
        qbias = 128 * s;
    }

    /// Integer-domain signed IP:
    ///     sum_i qs_i * (c_i - 128)
    ///   = sum_i qs_i * c_i - 128 * sum_i qs_i
    /// The mixed-sign dot product runs in an i32 vector
    /// accumulator (fused vwmaccsu, direct_ip final-kernel
    /// shape); the query-only bias is injected as the vredsum
    /// seed — zero extra instructions anywhere on the hot path.
    /// Per 32 dims (e8m2, VLEN=128): vle8(c) + vle16(q16) ->
    /// vzext_vf2 (c -> u16m4) -> vwmaccsu (i32m8 acc). 4 vector
    /// ops per block, single accumulator.
    /// ID3 (round 3 probe): bump-pointer addressing — replace the
    /// indexed `pc + i` / `pq + i` forms and the i counter with
    /// two stepped pointers and a remaining-count, trimming the
    /// per-iteration scalar address arithmetic (the q16 stream's
    /// x2-scale indexed form blocks GCC's own strength
    /// reduction; direct_ip board evidence -2.07%). Vector op
    /// sequence is unchanged.
    int32_t accumulate_int_ip(const uint8_t* code) const {
        const int16_t* pq = q16.data();
        const uint8_t* pc = code;

        // Hoist vsetvl: VLMAX for e8m2, reused across the hot loop.
        const size_t vl = __riscv_vsetvl_e8m2(d);
        vint32m8_t acc = __riscv_vmv_v_x_i32m8(0, vl);

        size_t remaining = d;
        // Hot loop: 2 loads, one zero-extend, one fused widening
        // multiply-accumulate. No vsetvl, no reduction inside;
        // bump-pointer form, no indexed addressing.
        while (remaining >= vl) {
            vuint8m2_t vc = __riscv_vle8_v_u8m2(pc, vl);
            vint16m4_t vq = __riscv_vle16_v_i16m4(pq, vl);
            vuint16m4_t c16 = __riscv_vzext_vf2_u16m4(vc, vl);
            acc = __riscv_vwmaccsu_vv_i32m8(acc, vq, c16, vl);
            pc += vl;
            pq += vl;
            remaining -= vl;
        }

        // Tail: fewer than vl dims left — one shorter-vl pass into
        // the same accumulator (only the first vt lanes touched).
        if (remaining > 0) {
            const size_t vt = __riscv_vsetvl_e8m2(remaining);
            vuint8m2_t vc = __riscv_vle8_v_u8m2(pc, vt);
            vint16m4_t vq = __riscv_vle16_v_i16m4(pq, vt);
            vuint16m4_t c16 = __riscv_vzext_vf2_u16m4(vc, vt);
            acc = __riscv_vwmaccsu_vv_i32m8(acc, vq, c16, vt);
        }

        // Single horizontal reduction over all vl lanes, seeded
        // with -qbias: the bias subtraction rides the reduction.
        vint32m1_t seed = __riscv_vmv_v_x_i32m1(-qbias, 1);
        vint32m1_t red = __riscv_vredsum_vs_i32m8_i32m1(acc, seed, vl);
        return __riscv_vmv_x_s_i32m1_i32(red);
    }

    float query_to_code(const uint8_t* code) const final {
        return static_cast<float>(accumulate_int_ip(code));
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        // Not on the benchmark-critical path; exact integer-domain
        // scalar evaluation ((c1-128)*(c2-128)).
        const uint8_t* c1 = codes + i * code_size;
        const uint8_t* c2 = codes + j * code_size;
        int64_t acc = 0;
        for (size_t k = 0; k < d; k++) {
            acc += (int(c1[k]) - 128) * (int(c2[k]) - 128);
        }
        return static_cast<float>(acc);
    }

    void query_to_codes_batch_4(
            const uint8_t* code_0,
            const uint8_t* code_1,
            const uint8_t* code_2,
            const uint8_t* code_3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) const final {
        dis0 = static_cast<float>(accumulate_int_ip(code_0));
        dis1 = static_cast<float>(accumulate_int_ip(code_1));
        dis2 = static_cast<float>(accumulate_int_ip(code_2));
        dis3 = static_cast<float>(accumulate_int_ip(code_3));
    }
};

//  * Fast path — QT_bf16 + L2
//  *
//  * bf16 code: each dimension is the high 16 bits of an f32, stored as
//  * uint16. decode_bf16(v) = reinterpret_f32(u32(v) << 16).
//  *
//  * ID1 (direct form): the query stays in full f32 precision — exactly the
//  * scalar NONE fallback's numeric model:
//  *     L2 = sum_i (q_i - decode_bf16(c_i))^2
//  *
//  * march has no zvfbfmin/zvfbfwma (and no dpbf16), so bf16→f32 widening
//  * is manual: vle16 (u16m2) → vzext_vf2 (u32m4) → vsll 16 → reinterpret
//  * f32m4. Hot loop per vl=16 dims (VLEN=128): 1 code load + 2-op widen +
//  * 1 q load + vfsub + vfmacc — 6 vector ops, single f32m4 accumulator,
//  * vsetvl hoisted (0 inside the hot loop), one horizontal reduction at
//  * the end. Tail (d % vl) takes one shorter-vl pass into the same
//  * accumulator (d=768 → no tail).
//  ************************************************************************/

/// Widen a bf16 (u16) vector chunk to f32: (u32(v) << 16) reinterpreted.
static inline vfloat32m4_t bf16_widen_f32m4(vuint16m2_t v, size_t vl) {
    vuint32m4_t w = __riscv_vzext_vf2_u32m4(v, vl);
    w = __riscv_vsll_vx_u32m4(w, 16, vl);
    return __riscv_vreinterpret_v_u32m4_f32m4(w);
}

template <>
struct DCTemplate<
        QuantizerBF16<SIMDLevel::RISCV_RVV>,
        SimilarityL2<SIMDLevel::RISCV_RVV>,
        SIMDLevel::RISCV_RVV> : SQDistanceComputer {
    using Sim = SimilarityL2<SIMDLevel::RISCV_RVV>;

    size_t d;

    DCTemplate(size_t d_in, const std::vector<float>&) : d(d_in) {}

    void set_query(const float* x) final {
        q = x;
    }

    /// Direct-form L2 between the f32 query and a bf16 code.
    float compute_l2(const float* qf, const uint8_t* code8) const {
        const uint16_t* code = (const uint16_t*)code8;
        size_t i = 0;
        // Hoist vsetvl: VLMAX for e16m2 (== f32m4 lanes), reused across
        // the hot loop; 0 vsetvl inside.
        const size_t vl = __riscv_vsetvl_e16m2(d > 0 ? d : 1);
        vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);

        for (; i + vl <= d; i += vl) {
            vuint16m2_t vc = __riscv_vle16_v_u16m2(code + i, vl);
            vfloat32m4_t fc = bf16_widen_f32m4(vc, vl);
            vfloat32m4_t fq = __riscv_vle32_v_f32m4(qf + i, vl);
            vfloat32m4_t t = __riscv_vfsub_vv_f32m4(fq, fc, vl);
            acc = __riscv_vfmacc_vv_f32m4(acc, t, t, vl);
        }

        // Tail: fewer than vl dims left — one shorter-vl pass into the
        // same accumulator (only the first vt lanes are touched).
        if (i < d) {
            const size_t vt = __riscv_vsetvl_e16m2(d - i);
            vuint16m2_t vc = __riscv_vle16_v_u16m2(code + i, vt);
            vfloat32m4_t fc = bf16_widen_f32m4(vc, vt);
            vfloat32m4_t fq = __riscv_vle32_v_f32m4(qf + i, vt);
            vfloat32m4_t t = __riscv_vfsub_vv_f32m4(fq, fc, vt);
            acc = __riscv_vfmacc_vv_f32m4(acc, t, t, vt);
        }

        // Single horizontal reduction over all vl lanes.
        vfloat32m1_t red = __riscv_vfredusum_vs_f32m4_f32m1(
                acc, __riscv_vfmv_v_f_f32m1(0.0f, 1), vl);
        return __riscv_vfmv_f_s_f32m1_f32(red);
    }

    float query_to_code(const uint8_t* code) const final {
        return compute_l2(q, code);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        // Not on the benchmark-critical path; scalar per-dim evaluation.
        const uint16_t* a = (const uint16_t*)(codes + i * code_size);
        const uint16_t* b = (const uint16_t*)(codes + j * code_size);
        float accu = 0;
        for (size_t k = 0; k < d; k++) {
            float diff = decode_bf16(a[k]) - decode_bf16(b[k]);
            accu += diff * diff;
        }
        return accu;
    }

    void query_to_codes_batch_4(
            const uint8_t* code_0,
            const uint8_t* code_1,
            const uint8_t* code_2,
            const uint8_t* code_3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) const final {
        dis0 = compute_l2(q, code_0);
        dis1 = compute_l2(q, code_1);
        dis2 = compute_l2(q, code_2);
        dis3 = compute_l2(q, code_3);
    }
};

//  * Fast path — QT_bf16 + IP
//  *
//  * Same bf16 code layout as the L2 block above; the query stays in full
//  * f32 precision — exactly the scalar NONE fallback's numeric model:
//  *     IP = sum_i q_i * decode_bf16(c_i)
//  *
//  * ID1 (direct form): hot loop per vl=16 dims (VLEN=128, e16m2→f32m4):
//  * 1 code load + 2-op widen + 1 q load + vfmacc — 5 vector ops, single
//  * f32m4 accumulator, vsetvl hoisted (0 inside the hot loop), one
//  * horizontal reduction at the end. Tail (d % vl) takes one shorter-vl
//  * pass into the same accumulator (d=768 → no tail).
//  ************************************************************************/

template <>
struct DCTemplate<
        QuantizerBF16<SIMDLevel::RISCV_RVV>,
        SimilarityIP<SIMDLevel::RISCV_RVV>,
        SIMDLevel::RISCV_RVV> : SQDistanceComputer {
    using Sim = SimilarityIP<SIMDLevel::RISCV_RVV>;

    size_t d;

    DCTemplate(size_t d_in, const std::vector<float>&) : d(d_in) {}

    void set_query(const float* x) final {
        q = x;
    }

    /// Direct-form inner product between the f32 query and a bf16 code.
    float compute_ip(const float* qf, const uint8_t* code8) const {
        const uint16_t* code = (const uint16_t*)code8;
        size_t i = 0;
        // Hoist vsetvl: VLMAX for e16m2 (== f32m4 lanes), reused across
        // the hot loop; 0 vsetvl inside.
        const size_t vl = __riscv_vsetvl_e16m2(d > 0 ? d : 1);
        vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);

        for (; i + vl <= d; i += vl) {
            vuint16m2_t vc = __riscv_vle16_v_u16m2(code + i, vl);
            vfloat32m4_t fc = bf16_widen_f32m4(vc, vl);
            vfloat32m4_t fq = __riscv_vle32_v_f32m4(qf + i, vl);
            acc = __riscv_vfmacc_vv_f32m4(acc, fq, fc, vl);
        }

        // Tail: fewer than vl dims left — one shorter-vl pass into the
        // same accumulator (only the first vt lanes are touched).
        if (i < d) {
            const size_t vt = __riscv_vsetvl_e16m2(d - i);
            vuint16m2_t vc = __riscv_vle16_v_u16m2(code + i, vt);
            vfloat32m4_t fc = bf16_widen_f32m4(vc, vt);
            vfloat32m4_t fq = __riscv_vle32_v_f32m4(qf + i, vt);
            acc = __riscv_vfmacc_vv_f32m4(acc, fq, fc, vt);
        }

        // Single horizontal reduction over all vl lanes.
        vfloat32m1_t red = __riscv_vfredusum_vs_f32m4_f32m1(
                acc, __riscv_vfmv_v_f_f32m1(0.0f, 1), vl);
        return __riscv_vfmv_f_s_f32m1_f32(red);
    }

    float query_to_code(const uint8_t* code) const final {
        return compute_ip(q, code);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        // Not on the benchmark-critical path; scalar per-dim evaluation.
        const uint16_t* a = (const uint16_t*)(codes + i * code_size);
        const uint16_t* b = (const uint16_t*)(codes + j * code_size);
        float accu = 0;
        for (size_t k = 0; k < d; k++) {
            accu += decode_bf16(a[k]) * decode_bf16(b[k]);
        }
        return accu;
    }

    void query_to_codes_batch_4(
            const uint8_t* code_0,
            const uint8_t* code_1,
            const uint8_t* code_2,
            const uint8_t* code_3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) const final {
        dis0 = compute_ip(q, code_0);
        dis1 = compute_ip(q, code_1);
        dis2 = compute_ip(q, code_2);
        dis3 = compute_ip(q, code_3);
    }
};

//  * Fast path — QT_fp16 + L2
//  *
//  * fp16 code: each dimension is an IEEE-754 binary16 stored as uint16.
//  * The scalar NONE fallback decodes with the software bit-twiddling
//  * decode_fp16 (fp16-inl.h; RISC-V has no F16C/NEON path), which costs
//  * ~20 scalar ops per dimension — so the scalar gap is even more
//  * expensive here than for bf16.
//  *
//  * ID6 (LMUL sweep, down): ID1 direct form (the best so far: q in full
//  * f32, subtract-first — exactly the scalar NONE numeric model at
//  * f32-reassociation level) lowered from e16m2->f32m4 to e16m1->f32m2:
//  * 96 iterations at d=768 (2x the instruction count of the m2/m4
//  * sweet spot, ~6/32 registers). Family evidence (S9, 5x refuted for
//  * m4/m8 in R4) says the m2/m4 point is the sweet spot; this round is
//  * the designated lower-boundary verification.
//  *
//  * Toolchain note: march is rv64gcv_zvfhmin — NO full zvfh, so f16
//  * vector arithmetic is unavailable (GCC ICEs, R2 evidence); only
//  * vle16/vse16, vfwcvt.f.f.v, vfncvt.f.f.w are legal on f16.
//  *
//  * Hot loop per vl=8 dims (VLEN=128, e16m1 -> f32m2): vle16 c +
//  * vfwcvt + vle32 q + vfsub + vfmacc — 5 vector ops, single f32m2
//  * accumulator, vsetvl hoisted (0 inside the hot loop), one horizontal
//  * reduction at the end. Tail (d % vl) takes one shorter-vl pass into
//  * the same accumulator (d=768 -> no tail).
//  ************************************************************************/

template <>
struct DCTemplate<
        QuantizerFP16<SIMDLevel::RISCV_RVV>,
        SimilarityL2<SIMDLevel::RISCV_RVV>,
        SIMDLevel::RISCV_RVV> : SQDistanceComputer {
    using Sim = SimilarityL2<SIMDLevel::RISCV_RVV>;

    size_t d;

    DCTemplate(size_t d_in, const std::vector<float>&) : d(d_in) {}

    void set_query(const float* x) final {
        q = x;
    }

    /// Direct-form L2 between the f32 query and an fp16 code (m1/m2).
    float compute_l2(const float* qf, const uint8_t* code8) const {
        const _Float16* code = (const _Float16*)code8;
        size_t i = 0;
        // Hoist vsetvl: VLMAX for e16m1 (== f32m2 lanes), reused across
        // the hot loop; 0 vsetvl inside.
        const size_t vl = __riscv_vsetvl_e16m1(d > 0 ? d : 1);
        vfloat32m2_t acc = __riscv_vfmv_v_f_f32m2(0.0f, vl);

        for (; i + vl <= d; i += vl) {
            vfloat16m1_t vc = __riscv_vle16_v_f16m1(code + i, vl);
            vfloat32m2_t fc = __riscv_vfwcvt_f_f_v_f32m2(vc, vl);
            vfloat32m2_t fq = __riscv_vle32_v_f32m2(qf + i, vl);
            vfloat32m2_t t = __riscv_vfsub_vv_f32m2(fq, fc, vl);
            acc = __riscv_vfmacc_vv_f32m2(acc, t, t, vl);
        }

        // Tail: fewer than vl dims left — one shorter-vl pass into the
        // same accumulator (only the first vt lanes are touched).
        if (i < d) {
            const size_t vt = __riscv_vsetvl_e16m1(d - i);
            vfloat16m1_t vc = __riscv_vle16_v_f16m1(code + i, vt);
            vfloat32m2_t fc = __riscv_vfwcvt_f_f_v_f32m2(vc, vt);
            vfloat32m2_t fq = __riscv_vle32_v_f32m2(qf + i, vt);
            vfloat32m2_t t = __riscv_vfsub_vv_f32m2(fq, fc, vt);
            acc = __riscv_vfmacc_vv_f32m2(acc, t, t, vt);
        }

        // Single horizontal reduction over all vl lanes.
        vfloat32m1_t red = __riscv_vfredusum_vs_f32m2_f32m1(
                acc, __riscv_vfmv_v_f_f32m1(0.0f, 1), vl);
        return __riscv_vfmv_f_s_f32m1_f32(red);
    }

    float query_to_code(const uint8_t* code) const final {
        return compute_l2(q, code);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        // Not on the benchmark-critical path; scalar per-dim evaluation.
        const uint16_t* a = (const uint16_t*)(codes + i * code_size);
        const uint16_t* b = (const uint16_t*)(codes + j * code_size);
        float accu = 0;
        for (size_t k = 0; k < d; k++) {
            float diff = decode_fp16(a[k]) - decode_fp16(b[k]);
            accu += diff * diff;
        }
        return accu;
    }

    void query_to_codes_batch_4(
            const uint8_t* code_0,
            const uint8_t* code_1,
            const uint8_t* code_2,
            const uint8_t* code_3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) const final {
        dis0 = compute_l2(q, code_0);
        dis1 = compute_l2(q, code_1);
        dis2 = compute_l2(q, code_2);
        dis3 = compute_l2(q, code_3);
    }
};

//  * Fast path — QT_fp16 + IP
//  *
//  * fp16 code: each dimension is an IEEE-754 binary16 stored as uint16.
//  * The scalar NONE fallback decodes with the software bit-twiddling
//  * decode_fp16 (fp16-inl.h; RISC-V has no F16C/NEON path), which costs
//  * ~20 scalar ops per dimension.
//  *
//  * ID6 (software pipelining): IP kernel with prologue+rotate pattern.
//  * Hot loop per vl=8 dims (VLEN=128, e16m1->f32m2): preload next chunk's
//  * f16 code at loop top, process current chunk, rotate. Load-use distance
//  * stretched from 1 op to ~4 ops across full iteration body.
//  * 96 iterations at d=768, ~8/32 registers (+1 m1 for next_vc).
//  * q stays in f32, numerical model matches scalar NONE fallback.
//  *
//  * Toolchain: march=rv64gcv_zvfhmin — NO full zvfh.
//  * d=768 = 96 * vl=8 -> no tail.
//  ************************************************************************/

template <>
struct DCTemplate<
        QuantizerFP16<SIMDLevel::RISCV_RVV>,
        SimilarityIP<SIMDLevel::RISCV_RVV>,
        SIMDLevel::RISCV_RVV> : SQDistanceComputer {
    using Sim = SimilarityIP<SIMDLevel::RISCV_RVV>;

    size_t d;

    DCTemplate(size_t d_in, const std::vector<float>&) : d(d_in) {}

    void set_query(const float* x) final {
        q = x;
    }

    /// Direct-form IP with dual independent accumulators (m1/m2, ID4).
    /// Direct-form IP with software pipelining (m1/m2, ID6).
    /// Prologue+rotate pattern: preload next chunk's f16 code at loop top,
    /// then compute on current chunk, stretching load-use distance from 1 op
    /// to full iteration body (~4 ops).
    float compute_ip(const float* qf, const uint8_t* code8) const {
        const _Float16* code = (const _Float16*)code8;
        size_t i = 0;
        const size_t vl = __riscv_vsetvl_e16m1(d > 0 ? d : 1);
        vfloat32m2_t acc = __riscv_vfmv_v_f_f32m2(0.0f, vl);

        if (i + vl <= d) {
            // Prologue: preload chunk 0.
            vfloat16m1_t vc = __riscv_vle16_v_f16m1(code, vl);
            i += vl;

            // Main loop: while a full next chunk exists, issue its load
            // first, then process the current one.
            for (; i + vl <= d; i += vl) {
                vfloat16m1_t vc_next =
                        __riscv_vle16_v_f16m1(code + i, vl);

                vfloat32m2_t fc =
                        __riscv_vfwcvt_f_f_v_f32m2(vc, vl);
                vfloat32m2_t fq =
                        __riscv_vle32_v_f32m2(qf + i - vl, vl);
                acc = __riscv_vfmacc_vv_f32m2(acc, fq, fc, vl);

                vc = vc_next; // rotate
            }

            // Epilogue: process the last chunk (already loaded).
            {
                vfloat32m2_t fc =
                        __riscv_vfwcvt_f_f_v_f32m2(vc, vl);
                vfloat32m2_t fq =
                        __riscv_vle32_v_f32m2(qf + i - vl, vl);
                acc = __riscv_vfmacc_vv_f32m2(acc, fq, fc, vl);
            }
        }

        // Tail: fewer than vl dims left.
        if (i < d) {
            const size_t vt = __riscv_vsetvl_e16m1(d - i);
            vfloat16m1_t vc = __riscv_vle16_v_f16m1(code + i, vt);
            vfloat32m2_t fc = __riscv_vfwcvt_f_f_v_f32m2(vc, vt);
            vfloat32m2_t fq = __riscv_vle32_v_f32m2(qf + i, vt);
            acc = __riscv_vfmacc_vv_f32m2(acc, fq, fc, vt);
        }

        vfloat32m1_t red = __riscv_vfredusum_vs_f32m2_f32m1(
                acc, __riscv_vfmv_v_f_f32m1(0.0f, 1), vl);
        return __riscv_vfmv_f_s_f32m1_f32(red);
    }

    float query_to_code(const uint8_t* code) const final {
        return compute_ip(q, code);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        // Not on the benchmark-critical path; scalar per-dim evaluation.
        const uint16_t* a = (const uint16_t*)(codes + i * code_size);
        const uint16_t* b = (const uint16_t*)(codes + j * code_size);
        float accu = 0;
        for (size_t k = 0; k < d; k++) {
            accu += decode_fp16(a[k]) * decode_fp16(b[k]);
        }
        return accu;
    }

    void query_to_codes_batch_4(
            const uint8_t* code_0,
            const uint8_t* code_1,
            const uint8_t* code_2,
            const uint8_t* code_3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) const final {
        dis0 = compute_ip(q, code_0);
        dis1 = compute_ip(q, code_1);
        dis2 = compute_ip(q, code_2);
        dis3 = compute_ip(q, code_3);
    }
};

/**********************************************************
 * TurboQuant masked_sum RVV specialization (scalar fallback)
 **********************************************************/

template <SIMDLevel SL0>
float turboq_masked_sum(const float* arr, const uint8_t* bits, size_t d);

template <>
float turboq_masked_sum<SIMDLevel::RISCV_RVV>(
        const float* arr,
        const uint8_t* bits,
        size_t d) {
    float result = 0;
    for (size_t byte_idx = 0; byte_idx < (d + 7) / 8; byte_idx++) {
        uint8_t b = bits[byte_idx];
        size_t base = byte_idx * 8;
        size_t end = std::min(base + 8, d);
        for (size_t j = base; j < end; j++) {
            if (b & (1 << (j - base))) {
                result += arr[j];
            }
        }
    }
    return result;
}

} // namespace scalar_quantizer
} // namespace faiss

#define THE_LEVEL_TO_DISPATCH SIMDLevel::RISCV_RVV
#include <faiss/impl/scalar_quantizer/sq-dispatch.h>

#endif // COMPILE_SIMD_RISCV_RVV
