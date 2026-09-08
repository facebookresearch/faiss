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
#include <algorithm>
#include <cmath>
#include <cstring>

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

//  * Fast path — QT_4bit_uniform + L2 (float domain)
//  *
//  * 4-bit UNIFORM scaling: every component reconstructs as
//  *     recon(c) = vmin + vdiff * (c + 0.5) / 15 = c0 + a * c
//  * with a = vdiff / 15 and c0 = vmin + 0.5 * a.
//  *
//  * L2 distance is evaluated in the exact affine float domain,
//  *     L2(q, code) = sum_i (q_i - recon(c_i))^2 = sum_i (e_i - a*c_i)^2
//  * with e_i = q_i - c0, so the ORIGINAL float query is retained (no
//  * integer-grid pre-quantization) and the result matches the scalar
//  * NONE reference to float-reassociation precision. vdiff == 0
//  * degenerates naturally (a == 0 -> L2 = sum_i (q_i - vmin)^2).
//  *
//  * The packed 4-bit code interleaves dimensions (even -> low nibble,
//  * odd -> high nibble), so set_query deinterleaves e_i into e_lo/e_hi
//  * once per query. The hot loop unpacks nibbles, converts to f32 and
//  * accumulates (e - a*c)^2 via the scalar-operand vfnmsac.vf + vfmacc.
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
    size_t half; // bytes per code = ceil(d/2)
    float a;     // vdiff / 15
    float c0;    // vmin + 0.5 * a
    std::vector<float> e_lo, e_hi; // q_i - c0, deinterleaved per query

    DCTemplate(size_t d_in, const std::vector<float>& trained)
            : d(d_in),
              half((d_in + 1) / 2),
              e_lo(half, 0.0f),
              e_hi(half, 0.0f) {
        a = trained[1] / 15.0f;
        c0 = trained[0] + 0.5f * a;
    }

    void set_query(const float* x) final {
        q = x;
        // e_i = q_i - c0, deinterleaved. Padding slot (odd d) keeps 0.
        for (size_t i = 0; i < d; i++) {
            if (i % 2 == 0) {
                e_lo[i / 2] = x[i] - c0;
            } else {
                e_hi[i / 2] = x[i] - c0;
            }
        }
    }

    /// Exact affine float-domain L2 over the packed 4-bit code.
    /// Per byte chunk: unpack nibbles (e8), widen+convert to f32, then
    /// t = e - a*c (vfnmsac.vf, scalar a) and acc += t*t (vfmacc).
    /// vsetvl hoisted; _tu on the tail keeps lanes past vt intact.
    /// Odd d: the last byte's high nibble is padding, so it is excluded
    /// from the vector domain (nbv = nb - 1) and the low nibble is
    /// processed in scalar to stay exact for any producer.
    float compute_l2(const uint8_t* code) const {
        const size_t nb = half; // bytes per code
        const size_t nbv = (d & 1) ? (nb - 1) : nb; // vector-domain bytes
        float total = 0.0f;
        size_t b = 0;

        // Hoist vsetvl: VLMAX for e8m1 (== f32m4 lanes). d==0 guard.
        const size_t vlb = __riscv_vsetvl_e8m1(nbv > 0 ? nbv : 1);
        vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vlb);

        for (; b + vlb <= nbv; b += vlb) {
            vuint8m1_t packed = __riscv_vle8_v_u8m1(code + b, vlb);
            vuint8m1_t lo = __riscv_vand_vx_u8m1(packed, 0x0F, vlb);
            vuint8m1_t hi = __riscv_vsrl_vx_u8m1(packed, 4, vlb);

            vfloat32m4_t clo = __riscv_vfcvt_f_xu_v_f32m4(
                    __riscv_vzext_vf4_u32m4(lo, vlb), vlb);
            vfloat32m4_t t_lo = __riscv_vle32_v_f32m4(e_lo.data() + b, vlb);
            t_lo = __riscv_vfnmsac_vf_f32m4(t_lo, a, clo, vlb);
            acc = __riscv_vfmacc_vv_f32m4(acc, t_lo, t_lo, vlb);

            vfloat32m4_t chi = __riscv_vfcvt_f_xu_v_f32m4(
                    __riscv_vzext_vf4_u32m4(hi, vlb), vlb);
            vfloat32m4_t t_hi = __riscv_vle32_v_f32m4(e_hi.data() + b, vlb);
            t_hi = __riscv_vfnmsac_vf_f32m4(t_hi, a, chi, vlb);
            acc = __riscv_vfmacc_vv_f32m4(acc, t_hi, t_hi, vlb);
        }

        // Tail: fewer than vlb bytes left — one shorter-vl pass into the
        // same accumulator; _tu keeps lanes past vt intact.
        if (b < nbv) {
            const size_t vt = __riscv_vsetvl_e8m1(nbv - b);
            vuint8m1_t packed = __riscv_vle8_v_u8m1(code + b, vt);
            vuint8m1_t lo = __riscv_vand_vx_u8m1(packed, 0x0F, vt);
            vuint8m1_t hi = __riscv_vsrl_vx_u8m1(packed, 4, vt);

            vfloat32m4_t clo = __riscv_vfcvt_f_xu_v_f32m4(
                    __riscv_vzext_vf4_u32m4(lo, vt), vt);
            vfloat32m4_t t_lo = __riscv_vle32_v_f32m4(e_lo.data() + b, vt);
            t_lo = __riscv_vfnmsac_vf_f32m4(t_lo, a, clo, vt);
            acc = __riscv_vfmacc_vv_f32m4_tu(acc, t_lo, t_lo, vt);

            vfloat32m4_t chi = __riscv_vfcvt_f_xu_v_f32m4(
                    __riscv_vzext_vf4_u32m4(hi, vt), vt);
            vfloat32m4_t t_hi = __riscv_vle32_v_f32m4(e_hi.data() + b, vt);
            t_hi = __riscv_vfnmsac_vf_f32m4(t_hi, a, chi, vt);
            acc = __riscv_vfmacc_vv_f32m4_tu(acc, t_hi, t_hi, vt);
        }

        // Horizontal reduce over all vlb lanes.
        vfloat32m1_t red = __riscv_vfredusum_vs_f32m4_f32m1(
                acc, __riscv_vfmv_v_f_f32m1(0.0f, 1), vlb);
        total = __riscv_vfmv_f_s_f32m1_f32(red);

        // Odd d: last dimension lives in the low nibble of the last byte.
        if (d & 1) {
            float diff = e_lo[nb - 1] - a * float(code[nb - 1] & 0x0F);
            total += diff * diff;
        }
        return total;
    }

    float query_to_code(const uint8_t* code) const final {
        return compute_l2(code);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        // Not on the critical path for most workloads; reconstruct both
        // codes into nibbles scalar-style and compute squared distance.
        // recon1 - recon2 = a * (c1 - c2).
        const uint8_t* c1 = codes + i * code_size;
        const uint8_t* c2 = codes + j * code_size;
        float acc = 0.0f;
        for (size_t k = 0; k < d; k++) {
            uint8_t n1 = (k % 2 == 0) ? (c1[k / 2] & 0x0F) : (c1[k / 2] >> 4);
            uint8_t n2 = (k % 2 == 0) ? (c2[k / 2] & 0x0F) : (c2[k / 2] >> 4);
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

        // Hoist vsetvl: VLMAX for e8m1 (== f32 lanes per m4 group chunk).
        // d==0 => nb==0: guard so vsetvl(0) doesn't return 0 and the
        // chunk loop below (b + vlb <= nb with vlb==0) never stalls.
        size_t vlb = __riscv_vsetvl_e8m1(nb > 0 ? nb : 1);
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
            acc = __riscv_vfmacc_vv_f32m4_tu(acc, t_lo, t_lo, vt);

            vfloat32m4_t chi = __riscv_vfcvt_f_xu_v_f32m4(
                    __riscv_vzext_vf4_u32m4(hi, vt), vt);
            vfloat32m4_t t_hi = __riscv_vle32_v_f32m4(e_hi.data() + b, vt);
            t_hi = __riscv_vfnmsac_vv_f32m4(
                    t_hi,
                    __riscv_vle32_v_f32m4(a_hi.data() + b, vt),
                    chi,
                    vt);
            acc = __riscv_vfmacc_vv_f32m4_tu(acc, t_hi, t_hi, vt);
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
    /// Hot loop per byte chunk: unpack nibbles, widen to f32, acc += b*c.
    /// Software-pipelined: the next chunk's load is issued at the top of
    /// this iteration (prologue + rotate) to stretch the load-use distance.
    float compute_ip(const uint8_t* code) const {
        const size_t nb = half; // total bytes to process
        size_t bpos = 0;

        // Hoist vsetvl: VLMAX for e8m1 (== f32 lanes per m4 group chunk).
        // d==0 => nb==0: guard so the `if (nb >= vlb)` prologue + chunk
        // loop (bpos + 2*vlb <= nb) don't stall on vlb==0.
        size_t vlb = __riscv_vsetvl_e8m1(nb > 0 ? nb : 1);
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
            acc = __riscv_vfmacc_vv_f32m4_tu(
                    acc,
                    __riscv_vle32_v_f32m4(b_lo.data() + bpos, vt),
                    clo,
                    vt);

            vfloat32m4_t chi = __riscv_vfcvt_f_xu_v_f32m4(
                    __riscv_vzext_vf4_u32m4(hi, vt), vt);
            acc = __riscv_vfmacc_vv_f32m4_tu(
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
//  * Uniform scaling: recon(c) = vmin + vdiff*(c + 0.5)/15 = c0 + a*c,
//  * with a = vdiff/15, c0 = vmin + 0.5*a. The inner product folds into a
//  * query-only constant plus one uniformly-scaled integer-code dot product:
//  *     IP(q, code) = c0 * sum_i q_i + a * sum_i q_i*c_i = K_q + a*S
//  * K_q is precomputed in set_query; the hot loop only evaluates S.
//  *
//  * The packed code interleaves dims (even -> low nibble, odd -> high), so
//  * set_query deinterleaves q into q_lo/q_hi. Odd-d padding nibble keeps
//  * q_hi = 0. vdiff == 0 degenerates naturally (a = 0 -> IP = K_q).
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

    /// S = sum_i q_i * c_i over the packed code. Software-pipelined hot
    /// loop (prologue + rotate) with dual accumulators per nibble stream.
    float compute_qc_dot(const uint8_t* code) const {
        const size_t nb = half; // total bytes to process
        size_t bpos = 0;

        // Hoist vsetvl; d==0 => nb==0, guard so the chunk loop can't
        // stall on vlb==0.
        size_t vlb = __riscv_vsetvl_e8m1(nb > 0 ? nb : 1);
        // Dual accumulators (one per nibble stream) break the serial
        // chain of two dependent FMAs into two independent chains.
        vfloat32m4_t acc_lo = __riscv_vfmv_v_f_f32m4(0.0f, vlb);
        vfloat32m4_t acc_hi = __riscv_vfmv_v_f_f32m4(0.0f, vlb);

        if (nb >= vlb) {
            // Prologue: preload chunk 0.
            vuint8m1_t packed = __riscv_vle8_v_u8m1(code, vlb);

            // Main loop: while a full NEXT chunk exists, issue its load
            // first, then process the current one.
            for (; bpos + 2 * vlb <= nb; bpos += vlb) {
                __builtin_prefetch(code + bpos + 512, 0, 0);
                vuint8m1_t packed_next =
                        __riscv_vle8_v_u8m1(code + bpos + vlb, vlb);

                vuint8m1_t lo = __riscv_vand_vx_u8m1(packed, 0x0F, vlb);
                vuint8m1_t hi = __riscv_vsrl_vx_u8m1(packed, 4, vlb);

                // Issue both q-stream loads early to hide L1 load-use
                // latency ahead of the consuming FMAs.
                vfloat32m4_t vq_lo =
                        __riscv_vle32_v_f32m4(q_lo.data() + bpos, vlb);
                vfloat32m4_t vq_hi =
                        __riscv_vle32_v_f32m4(q_hi.data() + bpos, vlb);

                // Low-nibble stream (even dims): acc_lo += q_lo * c_lo
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
            acc_lo = __riscv_vfmacc_vv_f32m4_tu(
                    acc_lo,
                    __riscv_vle32_v_f32m4(q_lo.data() + bpos, vt),
                    clo,
                    vt);

            vfloat32m4_t chi = __riscv_vfwcvt_f_xu_v_f32m4(
                    __riscv_vzext_vf2_u16m2(hi, vt), vt);
            acc_hi = __riscv_vfmacc_vv_f32m4_tu(
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
//  * recon_i(c) = vmin[i] + vdiff[i]*(c + 0.5)/63 = rmin_i + a_i*c_i,
//  * with a_i = vdiff[i]/63, rmin_i = vmin[i] + 0.5*a_i. L2 contribution
//  * per dim: (q_i - recon_i)^2 = (e_i - a_i*c_i)^2, e_i = q_i - rmin_i.
//  *
//  * Codec6bit packs 4 dims into 3 bytes (b0,b1,b2); no lo/hi symmetry, so
//  * the kernel deinterleaves by (i & 3) into FOUR streams (a0..a3 / e0..e3).
//  * Hot loop: vlseg3e8 deinterleaves the 3-byte groups, 6-bit fields are
//  * extracted with u8 ALU ops, each stream widens to f32 and accumulates
//  * t = e - a*c (vfnmsac), acc += t*t (vfmacc). Software-pipelined
//  * (prologue + rotate). Tail dims (d % 4) are decoded scalar.
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

            // Software-pipelined vlseg3e8: preload the next chunk's code
            // block, then compute on the already-loaded current chunk.
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

                    // Light extracts (c0, c3) first so their FMA chains
                    // issue early; heavy extracts (c1, c2) overlap.
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

            // Tail groups: one shorter-vl pass into the same accumulator.
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
                acc = __riscv_vfmacc_vv_f32m4_tu(acc, t0, t0, vt);

                vfloat32m4_t f3 = __riscv_vfwcvt_f_xu_v_f32m4(
                        __riscv_vzext_vf2_u16m2(c3, vt), vt);
                vfloat32m4_t t3 = __riscv_vle32_v_f32m4(pe3 + g, vt);
                t3 = __riscv_vfnmsac_vv_f32m4(
                        t3, __riscv_vle32_v_f32m4(pa3 + g, vt), f3, vt);
                acc = __riscv_vfmacc_vv_f32m4_tu(acc, t3, t3, vt);

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
                acc = __riscv_vfmacc_vv_f32m4_tu(acc, t1, t1, vt);

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
                acc = __riscv_vfmacc_vv_f32m4_tu(acc, t2, t2, vt);
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
//  * recon_i(c) = rmin_i + a_i*c_i, a_i = vdiff[i]/63, rmin_i = vmin[i] +
//  * 0.5*a_i. IP decomposes into a query-only constant plus a coefficient
//  * dot product over the integer codes:
//  *     IP(q, code) = sum_i q_i*rmin_i + sum_i (q_i*a_i)*c_i = K_q + sum_i b_i*c_i
//  * K_q and b_i = q_i*a_i are precomputed in set_query. Codec6bit packs 4
//  * dims into 3 bytes, so b_i is deinterleaved by (i & 3) into four streams.
//  *
//  * Hot loop: software-pipelined vlseg3e8, 6-bit fields extracted with u8
//  * ALU ops, each stream widens to f32 and accumulates acc += b*c (vfmacc).
//  * Tail dims (d % 4) are decoded scalar.
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

                    // Light extracts (c0, c3) first so their FMA chains
                    // issue early; heavy extracts (c1, c2) overlap.
                    // Accumulation order: 0, 3, 1, 2.
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

                    // Light extracts (c0, c3) first so their FMA chains
                    // issue early; heavy extracts (c1, c2) overlap.
                    // Accumulation order: 0, 3, 1, 2.
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
                acc = __riscv_vfmacc_vv_f32m4_tu(
                        acc, __riscv_vle32_v_f32m4(pb0 + g, vt), f0, vt);

                vfloat32m4_t f1 = __riscv_vfwcvt_f_xu_v_f32m4(
                        __riscv_vzext_vf2_u16m2(c1, vt), vt);
                acc = __riscv_vfmacc_vv_f32m4_tu(
                        acc, __riscv_vle32_v_f32m4(pb1 + g, vt), f1, vt);

                vfloat32m4_t f2 = __riscv_vfwcvt_f_xu_v_f32m4(
                        __riscv_vzext_vf2_u16m2(c2, vt), vt);
                acc = __riscv_vfmacc_vv_f32m4_tu(
                        acc, __riscv_vle32_v_f32m4(pb2 + g, vt), f2, vt);

                vfloat32m4_t f3 = __riscv_vfwcvt_f_xu_v_f32m4(
                        __riscv_vzext_vf2_u16m2(c3, vt), vt);
                acc = __riscv_vfmacc_vv_f32m4_tu(
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

        // Hoist vsetvl: VLMAX for e8m1 (== f32 lanes per m4 group).
        // d==0: guard so vsetvl(0) doesn't return 0 and the prologue
        // `if (i + vl <= d)` + chunk loop don't stall.
        const size_t vl = __riscv_vsetvl_e8m1(d > 0 ? d : 1);
        vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);

        size_t i = 0;
        if (i + vl <= d) {
            // Software pipeline depth 1: preload the next chunk's code
            // bytes at the top of this iteration.
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
            acc = __riscv_vfmacc_vv_f32m4_tu(acc, t, t, vt);
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
 * recon_i(c) = rmin_i + a_i*c_i, a_i = vdiff[i]/255, rmin_i = vmin[i] +
 * 0.5*a_i. IP decomposes into a query-only constant plus a coefficient
 * dot product over the integer codes:
 *     IP(q, code) = sum_i q_i*rmin_i + sum_i (q_i*a_i)*c_i = K_q + sum_i b_i*c_i
 * K_q and b_i = q_i*a_i are precomputed in set_query. Each code byte is
 * one dim (no bit unpacking): hot loop vle8 -> vzext -> vfcvt -> vle32 ->
 * vfmacc, software-pipelined.
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
              // Pad b_v by 2*VLMAX so the pipelined next-iteration
              // vle32 loads stay in bounds (padding lanes never used).
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

        // Hoist vsetvl: VLMAX for e8m1 (== f32 lanes per m4 group).
        // d==0: guard so vsetvl(0) doesn't return 0 and the prologue
        // `if (i + vl <= d)` + chunk loop don't stall.
        const size_t vl = __riscv_vsetvl_e8m1(d > 0 ? d : 1);
        vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        // Second accumulator: the 2x-unrolled loop feeds acc/acc1
        // alternately, halving the serial FMA chain.
        vfloat32m4_t acc1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);

        size_t i = 0;
        if (i + vl <= d) {
            // Software pipeline depth 1: preload the next chunk's code
            // bytes at the top of this iteration.
            vuint8m1_t c8 = __riscv_vle8_v_u8m1(code + i, vl); // prologue
            i += vl;

            // 2x unroll + dual accumulators; the b stream is also
            // software-pipelined (loads past d land in b_v's zero
            // padding and are discarded on exit).
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
            acc = __riscv_vfmacc_vv_f32m4_tu(
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
 * QT_8bit_uniform + L2 — float-domain RVV specialization
 *
 * 8-bit UNIFORM scaling: every component reconstructs as
 *     recon(c) = vmin + vdiff * (c + 0.5) / 255 = c0 + a * c
 * with a = vdiff / 255 and c0 = vmin + 0.5 * a.
 *
 * L2 distance is evaluated in the exact affine float domain,
 *     L2(q, code) = sum_i (q_i - recon(c_i))^2
 *                 = sum_i (e_i - a * c_i)^2   with  e_i = q_i - c0,
 * so the ORIGINAL float query is retained — no integer-grid
 * pre-quantization — and the result matches the scalar NONE
 * reference to float-reassociation precision. vdiff == 0
 * degenerates naturally (a == 0 -> L2 = sum_i (q_i - vmin)^2).
 *
 * Hot loop per vl = VLMAX(e8m1) dims (16 at VLEN=128): vle8 ->
 * vzext_vf4 (u8 -> u32m4) -> vfcvt (u32 -> f32m4), then
 * t = e - a*c via the scalar-operand vfnmsac.vf (a is a SHARED
 * scalar constant — no per-dimension coefficient stream) and
 * acc += t*t (vfmacc). vsetvl hoisted: 1 hoisted + 1 for the tail.
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
    float a;  // vdiff / 255
    float c0; // vmin + 0.5 * a
    std::vector<float> e_v; // q_i - c0, per query

    DCTemplate(size_t d_in, const std::vector<float>& trained)
            : d(d_in), e_v(d_in, 0.0f) {
        a = trained[1] / 255.0f;
        c0 = trained[0] + 0.5f * a;
    }

    void set_query(const float* x) final {
        q = x;
        // e_i = q_i - c0, once per query (amortized over all codes).
        for (size_t i = 0; i < d; i++) {
            e_v[i] = x[i] - c0;
        }
    }

    /// Exact affine float-domain L2: sum_i (e_i - a*c_i)^2.
    /// Hot loop per chunk: vle8 (code byte) -> vzext_vf4 -> vfcvt to
    /// f32, then t = e - a*c (vfnmsac.vf, a shared scalar) and
    /// acc += t*t (vfmacc). 5 vector ops per chunk, single f32m4
    /// accumulator, single vfredusum at the end. vsetvl hoisted.
    float compute_l2(const uint8_t* code) const {
        const float* pe = e_v.data();

        // Hoist vsetvl: VLMAX for e8m1 (== f32m4 lanes).
        // d==0 guard: vsetvl(0) would return 0 and stall the prologue.
        const size_t vl = __riscv_vsetvl_e8m1(d > 0 ? d : 1);
        vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);

        size_t i = 0;
        if (i + vl <= d) {
            vuint8m1_t c8 = __riscv_vle8_v_u8m1(code + i, vl); // prologue
            i += vl;
            for (; i + vl <= d; i += vl) {
                vuint8m1_t c8_next = __riscv_vle8_v_u8m1(code + i, vl);
                vfloat32m4_t cf = __riscv_vfcvt_f_xu_v_f32m4(
                        __riscv_vzext_vf4_u32m4(c8, vl), vl);
                vfloat32m4_t t = __riscv_vle32_v_f32m4(pe + i - vl, vl);
                t = __riscv_vfnmsac_vf_f32m4(t, a, cf, vl);
                acc = __riscv_vfmacc_vv_f32m4(acc, t, t, vl);
                c8 = c8_next;
            }
            // Epilogue: process the last full chunk (already loaded).
            {
                vfloat32m4_t cf = __riscv_vfcvt_f_xu_v_f32m4(
                        __riscv_vzext_vf4_u32m4(c8, vl), vl);
                vfloat32m4_t t = __riscv_vle32_v_f32m4(pe + i - vl, vl);
                t = __riscv_vfnmsac_vf_f32m4(t, a, cf, vl);
                acc = __riscv_vfmacc_vv_f32m4(acc, t, t, vl);
            }
        }

        // Tail: fewer than vl dims left — one shorter-vl pass into the
        // same accumulator; _tu keeps lanes past vt (accumulated by
        // prior chunks) intact for the full-vl reduction below.
        if (i < d) {
            const size_t vt = __riscv_vsetvl_e8m1(d - i);
            vuint8m1_t c8 = __riscv_vle8_v_u8m1(code + i, vt);
            vfloat32m4_t cf = __riscv_vfcvt_f_xu_v_f32m4(
                    __riscv_vzext_vf4_u32m4(c8, vt), vt);
            vfloat32m4_t t = __riscv_vle32_v_f32m4(pe + i, vt);
            t = __riscv_vfnmsac_vf_f32m4(t, a, cf, vt);
            acc = __riscv_vfmacc_vv_f32m4_tu(acc, t, t, vt);
        }

        // Horizontal reduce over all vl lanes.
        vfloat32m1_t red = __riscv_vfredusum_vs_f32m4_f32m1(
                acc, __riscv_vfmv_v_f_f32m1(0.0f, 1), vl);
        return __riscv_vfmv_f_s_f32m1_f32(red);
    }

    float query_to_code(const uint8_t* code) const final {
        return compute_l2(code);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        const uint8_t* c1 = codes + i * code_size;
        const uint8_t* c2 = codes + j * code_size;
        int64_t acc = 0;
        for (size_t k = 0; k < d; k++) {
            int diff = int(c1[k]) - int(c2[k]);
            acc += diff * diff;
        }
        return static_cast<float>(acc) * (a * a);
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
 * QT_8bit_uniform + IP — float-domain RVV specialization
 *
 * 8-bit UNIFORM scaling: every component reconstructs as
 *     recon(c) = vmin + vdiff * (c + 0.5) / 255 = c0 + scale * c
 * with scale = vdiff / 255 and c0 = vmin + 0.5 * scale (SHARED
 * scalar constants). The inner product folds into a query-only
 * constant plus one uniformly-scaled integer-code dot product:
 *     IP(q, code) = sum_i q_i * (c0 + scale * c_i)
 *                 = c0 * sum_i q_i + scale * sum_i q_i * c_i
 *                 = K_q + scale * S
 * K_q = c0 * sum(q) is precomputed once per query in set_query
 * (amortized 1/n over the codes of the scan), so the hot loop only
 * evaluates S = sum_i q_i * c_i.
 *
 * The ORIGINAL float query is retained — no fixed-point / integer
 * pre-quantization of the query — so scores and rankings match the
 * scalar NONE reference to float-reassociation precision (same
 * semantics class as the QT_8bit_uniform+L2 and QT_8bit_nonuniform
 * IP float-domain kernels). scale == 0 degenerates naturally
 * (IP == K_q == c0 * sum(q)), no special-casing.
 *
 * Hot loop per vl = VLMAX(e8m1) dims (16 at VLEN=128): vle8 ->
 * vzext_vf4 (u8 -> u32m4) -> vfcvt (u32 -> f32m4), then
 * acc += q * c (vfmacc). 4 vector ops per chunk, single f32m4
 * accumulator, single vfredusum at the end. vsetvl hoisted:
 * 1 hoisted + 1 for the tail.
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
    float scale; // vdiff / 255
    float c0;    // vmin + 0.5 * scale
    float k_q;   // c0 * sum_i q_i, per query

    DCTemplate(size_t d_in, const std::vector<float>& trained)
            : d(d_in), k_q(0.0f) {
        scale = trained[1] / 255.0f;
        c0 = trained[0] + 0.5f * scale;
    }

    void set_query(const float* x) final {
        q = x;
        // K_q = c0 * sum_i q_i, once per query (amortized over codes).
        float sum = 0.0f;
        for (size_t i = 0; i < d; i++) {
            sum += x[i];
        }
        k_q = c0 * sum;
    }

    /// S = sum_i q_i * c_i over the 1-byte-per-dim code; returns
    /// K_q + scale * S.
    float compute_ip(const uint8_t* code) const {
        const float* pq = q;

        // Hoist vsetvl: VLMAX for e8m1 (== f32 lanes per m4 group).
        // d==0: guard so vsetvl(0) doesn't return 0 and the prologue
        // `if (i + vl <= d)` + chunk loop don't stall.
        const size_t vl = __riscv_vsetvl_e8m1(d > 0 ? d : 1);
        vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);

        size_t i = 0;
        if (i + vl <= d) {
            // Software pipeline depth 1 — preload the NEXT chunk's
            // code bytes at the top of THIS iteration.
            vuint8m1_t c8 = __riscv_vle8_v_u8m1(code + i, vl); // prologue
            i += vl;

            for (; i + vl <= d; i += vl) {
                vuint8m1_t c8_next = __riscv_vle8_v_u8m1(code + i, vl);
                vfloat32m4_t cf = __riscv_vfcvt_f_xu_v_f32m4(
                        __riscv_vzext_vf4_u32m4(c8, vl), vl);
                vfloat32m4_t vq = __riscv_vle32_v_f32m4(pq + i - vl, vl);
                acc = __riscv_vfmacc_vv_f32m4(acc, vq, cf, vl);
                c8 = c8_next; // rotate
            }

            // Epilogue: process the last full chunk (already loaded).
            {
                vfloat32m4_t cf = __riscv_vfcvt_f_xu_v_f32m4(
                        __riscv_vzext_vf4_u32m4(c8, vl), vl);
                vfloat32m4_t vq = __riscv_vle32_v_f32m4(pq + i - vl, vl);
                acc = __riscv_vfmacc_vv_f32m4(acc, vq, cf, vl);
            }
        }

        // Tail: fewer than vl dims left — one shorter-vl pass into the
        // same accumulator; _tu keeps lanes past vt (accumulated by
        // prior chunks) intact for the full-vl reduction below.
        if (i < d) {
            const size_t vt = __riscv_vsetvl_e8m1(d - i);
            vuint8m1_t c8 = __riscv_vle8_v_u8m1(code + i, vt);
            vfloat32m4_t cf = __riscv_vfcvt_f_xu_v_f32m4(
                    __riscv_vzext_vf4_u32m4(c8, vt), vt);
            vfloat32m4_t vq = __riscv_vle32_v_f32m4(pq + i, vt);
            acc = __riscv_vfmacc_vv_f32m4_tu(acc, vq, cf, vt);
        }

        // Horizontal reduce over all vl lanes.
        vfloat32m1_t red = __riscv_vfredusum_vs_f32m4_f32m1(
                acc, __riscv_vfmv_v_f_f32m1(0.0f, 1), vl);
        return k_q + scale * __riscv_vfmv_f_s_f32m1_f32(red);
    }

    float query_to_code(const uint8_t* code) const final {
        return compute_ip(code);
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
 * Direct 8-bit storage: recon_i(c) = c_i (no affine, no trained
 * params), so L2(q, code) = sum_i (q_i - c_i)^2.
 *
 * Integer-domain kernel: set_query truncates the query to bytes
 * (q8[i] = clamp(int(x[i]), 0, 255)) — matching x86
 * DistanceComputerByte::set_query (the kernel faiss runs for this
 * qtype on AVX2/AVX512). In-contract data is bit-exact. Hot loop:
 * vle8(c) + vle8(q8) -> vwsubu -> vwmacc (i32m8 accumulator), one
 * vredsum + one int->float at the end.
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

    /// Integer-domain L2: sum_i (q8_i - c_i)^2 into an i32m8 vector
    /// accumulator; single reduction + single int->float at the end.
    int64_t accumulate_int_l2(const uint8_t* code) const {
        const uint8_t* pq = q8.data();

        // Hoist vsetvl: VLMAX for e8m2, reused across the hot loop.
        // d==0: guard so vsetvl(0) doesn't return 0 and the chunk loop
        // `i + vl <= d` doesn't stall.
        const size_t vl = __riscv_vsetvl_e8m2(d > 0 ? d : 1);
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
            acc = __riscv_vwmacc_vv_i32m8_tu(acc, df, df, vt);
        }

        // Single horizontal reduction over all vl lanes, widened to i64
        // so the total d*65025 cannot overflow i32 for large d. Per-lane
        // i32 accumulation is safe while (d/vl)*65025 < 2^31 (~1e6 dims).
        vint64m1_t z64 = __riscv_vmv_v_x_i64m1(0, 1);
        vint64m1_t red = __riscv_vwredsum_vs_i32m8_i64m1(acc, z64, vl);
        return __riscv_vmv_x_s_i64m1_i64(red);
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
 * Direct 8-bit storage: recon_i(c) = c_i (no affine, no trained
 * params), so IP(q, code) = sum_i q_i * c_i.
 *
 * Integer-domain kernel: set_query truncates the query to bytes
 * (matching x86 DistanceComputerByte::set_query); the query is
 * pre-widened to u16 and the hot loop uses a fused vwmaccu into a
 * u32m8 accumulator (the u8*u8 product does not fit i16, so the
 * unsigned vwmulu/vwaddu path is mandatory). In-contract data is
 * bit-exact.
 **********************************************************/

template <>
struct DCTemplate<
        Quantizer8bitDirect<SIMDLevel::RISCV_RVV>,
        SimilarityIP<SIMDLevel::RISCV_RVV>,
        SIMDLevel::RISCV_RVV> : SQDistanceComputer {
    using Sim = SimilarityIP<SIMDLevel::RISCV_RVV>;

    size_t d;
    // Query stored pre-widened as u16 so the hot loop loads it at
    // SEW=16 and uses a single fused vwmaccu.
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

    /// Integer-domain IP: sum_i q_i * c_i into a u32m8 accumulator;
    /// single reduction + single uint->float at the end.
    uint64_t accumulate_int_ip(const uint8_t* code) const {
        const uint16_t* pq = q16.data();
        const uint8_t* pc = code;

        // Hoist vsetvl: VLMAX for e8m2, reused across the hot loop.
        // d==0: guard so vsetvl(0) doesn't return 0 and the
        // `while (remaining >= vl)` loop doesn't stall.
        const size_t vl = __riscv_vsetvl_e8m2(d > 0 ? d : 1);
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
            acc = __riscv_vwmaccu_vv_u32m8_tu(acc, vq, c16, vt);
        }

        // Single horizontal reduction over all vl lanes, widened to u64
        // so the total d*65025 cannot overflow u32 for large d. Per-lane
        // u32 accumulation is safe while (d/vl)*65025 < 2^32 (~1.3e6 dims).
        vuint64m1_t z64 = __riscv_vmv_v_x_u64m1(0, 1);
        vuint64m1_t red = __riscv_vwredsumu_vs_u32m8_u64m1(acc, z64, vl);
        return __riscv_vmv_x_s_u64m1_u64(red);
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
 * Signed direct storage (Quantizer8bitDirectSigned): recon_i(c) =
 * c_i - 128, so L2(q, code) = sum_i ((q_i + 128) - c_i)^2 — the +128
 * bias cancels inside the difference, making this kernel isomorphic to
 * the unsigned QT_8bit_direct+L2 one with the query re-biased by +128
 * in set_query (matching x86 DistanceComputerByteSigned::set_query).
 *
 * Integer-domain: set_query re-biases the query to bytes (q8[i] =
 * clamp(int(x[i]) + 128, 0, 255)); hot loop vle8(c)+vle8(q8) -> vwsubu
 * -> vwmacc (i32m8 accumulator), one vredsum + int->float. In-contract
 * data (integers in [-128,127]) is bit-exact.
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

    /// Integer-domain L2: sum_i (q8_i - c_i)^2 into an i32m8 vector
    /// accumulator; single reduction + single int->float at the end.
    int64_t accumulate_int_l2(const uint8_t* code) const {
        const uint8_t* pq = q8.data();
        const uint8_t* pc = code;

        // Hoist vsetvl: VLMAX for e8m2, reused across the hot loop.
        // d==0: guard so vsetvl(0) doesn't return 0 and the
        // `while (remaining >= vl)` loop doesn't stall.
        const size_t vl = __riscv_vsetvl_e8m2(d > 0 ? d : 1);
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
            acc = __riscv_vwmacc_vv_i32m8_tu(acc, df, df, vt);
        }

        // Single horizontal reduction over all vl lanes, widened to i64
        // so the total d*65025 cannot overflow i32 for large d. Per-lane
        // i32 accumulation is safe while (d/vl)*65025 < 2^31 (~1e6 dims).
        vint64m1_t z64 = __riscv_vmv_v_x_i64m1(0, 1);
        vint64m1_t red = __riscv_vwredsum_vs_i32m8_i64m1(acc, z64, vl);
        return __riscv_vmv_x_s_i64m1_i64(red);
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
 * Signed direct storage: recon_i(c) = c_i - 128, so IP(q, code) =
 * sum_i q_i*(c_i - 128). Unlike L2 the +128 bias does not cancel:
 * expanding leaves a query-only term -128*sum_i q_i, which set_query
 * hoists out (matching x86 DistanceComputerByteSigned::set_query).
 *
 * Integer-domain: set_query truncates the query to i16 (qs[i] =
 * clamp(int(x[i]), -128, 127)) and precomputes the bias 128*sum(qs).
 * Hot loop vle8(c) + vle16(qs) -> vwmaccsu (i32m8 accumulator); the
 * bias is injected as the vredsum seed. In-contract data is bit-exact.
 **********************************************************/

template <>
struct DCTemplate<
        Quantizer8bitDirectSigned<SIMDLevel::RISCV_RVV>,
        SimilarityIP<SIMDLevel::RISCV_RVV>,
        SIMDLevel::RISCV_RVV> : SQDistanceComputer {
    using Sim = SimilarityIP<SIMDLevel::RISCV_RVV>;

    size_t d;
    // Query stored pre-widened as i16 so the hot loop loads it at
    // SEW=16 and uses a single fused vwmaccsu.
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
    ///     sum_i qs_i * (c_i - 128) = sum_i qs_i*c_i - 128*sum_i qs_i
    /// The mixed-sign dot product runs in an i32m8 accumulator
    /// (fused vwmaccsu); the query-only bias is injected as the
    /// vredsum seed.
    int64_t accumulate_int_ip(const uint8_t* code) const {
        const int16_t* pq = q16.data();
        const uint8_t* pc = code;

        // Hoist vsetvl: VLMAX for e8m2, reused across the hot loop.
        // d==0: guard so vsetvl(0) doesn't return 0 and the
        // `while (remaining >= vl)` loop doesn't stall.
        const size_t vl = __riscv_vsetvl_e8m2(d > 0 ? d : 1);
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
            acc = __riscv_vwmaccsu_vv_i32m8_tu(acc, vq, c16, vt);
        }

        // Single horizontal reduction over all vl lanes, seeded with
        // -qbias (the bias subtraction rides the reduction), widened to
        // i64 so the total |sum| <= d*32640 cannot overflow i32 for large
        // d. Per-lane i32 accumulation is safe while (d/vl)*32640 < 2^31.
        vint64m1_t seed = __riscv_vmv_v_x_i64m1(-int64_t(qbias), 1);
        vint64m1_t red = __riscv_vwredsum_vs_i32m8_i64m1(acc, seed, vl);
        return __riscv_vmv_x_s_i64m1_i64(red);
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
//  * bf16 code: each dim is the high 16 bits of an f32, stored as uint16;
//  * decode_bf16(v) = reinterpret_f32(u32(v) << 16). The query stays in full
//  * f32 precision: L2 = sum_i (q_i - decode_bf16(c_i))^2.
//  *
//  * march has no zvfbfmin/zvfbfwma, so bf16→f32 widening is manual: vle16 →
//  * vzext_vf2 → vsll 16 → reinterpret. Hot loop: code load + 2-op widen + q
//  * load + vfsub + vfmacc.
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
            acc = __riscv_vfmacc_vv_f32m4_tu(acc, t, t, vt);
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
//  * Same bf16 layout as L2 above; query stays in full f32 precision:
//  *     IP = sum_i q_i * decode_bf16(c_i)
//  * Hot loop: code load + 2-op widen + q load + vfmacc.
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
            acc = __riscv_vfmacc_vv_f32m4_tu(acc, fq, fc, vt);
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
//  * fp16 code: each dim is an IEEE-754 binary16 stored as uint16. The
//  * query stays in full f32 precision: L2 = sum_i (q_i - decode_fp16(c_i))^2.
//  *
//  * Toolchain note: march is rv64gcv_zvfhmin — NO full zvfh, so f16
//  * vector arithmetic is unavailable; only vle16/vse16, vfwcvt.f.f.v,
//  * vfncvt.f.f.w are legal on f16.
//  *
//  * Hot loop per vl=8 dims (e16m1 -> f32m2): vle16 c + vfwcvt + vle32 q
//  * + vfsub + vfmacc, single f32m2 accumulator.
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
            acc = __riscv_vfmacc_vv_f32m2_tu(acc, t, t, vt);
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
//  * fp16 code: each dim is an IEEE-754 binary16 stored as uint16. The
//  * query stays in full f32 precision: IP = sum_i q_i * decode_fp16(c_i).
//  * Software-pipelined (prologue + rotate): preload the next chunk's f16
//  * code at loop top, then compute on the current chunk.
//  *
//  * Toolchain: march=rv64gcv_zvfhmin — NO full zvfh.
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

    /// Direct-form IP, software-pipelined (prologue + rotate): preload the
    /// next chunk's f16 code, then compute on the current chunk.
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
            acc = __riscv_vfmacc_vv_f32m2_tu(acc, fq, fc, vt);
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
