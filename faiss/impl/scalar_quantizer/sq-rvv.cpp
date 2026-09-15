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

/*************************************************************************
 * Fast path — QT_8bit_direct_signed (IP and L2)
 *
 * Reconstruct signed bytes in float, accumulate in vector registers, and
 * reduce once after the loop. Tail-undisturbed accumulation preserves
 * earlier lanes during short final iterations.
 ************************************************************************/

template <class Similarity>
struct DCTemplate<
        Quantizer8bitDirectSigned<SIMDLevel::RISCV_RVV>,
        Similarity,
        SIMDLevel::RISCV_RVV> : SQDistanceComputer {
    using Sim = Similarity;

    size_t d;

    DCTemplate(size_t d_in, const std::vector<float>& /* trained */)
            : d(d_in) {}

    void set_query(const float* x) final {
        this->q = x;
    }

    /// Vectorized reconstruction of code bytes (stored as value+128) into
    /// the float domain (code[i] - 128), one AVL-sized chunk per iteration.
    template <bool IS_L2>
    float distance_to_xf(const float* x, const uint8_t* code) const {
        const size_t vlmax = __riscv_vsetvlmax_e32m4();
        vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
        size_t i = 0;
        while (i < d) {
            size_t vl = __riscv_vsetvl_e8m1(d - i);
            // byte = value + 128, so reconstruct as signed (byte - 128).
            vuint8m1_t cb = __riscv_vle8_v_u8m1(code + i, vl);
            vuint32m4_t cu = __riscv_vzext_vf4_u32m4(cb, vl);
            vint32m4_t cs = __riscv_vreinterpret_v_u32m4_i32m4(cu);
            cs = __riscv_vsub_vx_i32m4(cs, 128, vl);
            vfloat32m4_t cf = __riscv_vfcvt_f_x_v_f32m4(cs, vl);
            vfloat32m4_t vx = __riscv_vle32_v_f32m4(x + i, vl);
            if constexpr (IS_L2) {
                vfloat32m4_t diff = __riscv_vfsub_vv_f32m4(vx, cf, vl);
                acc = __riscv_vfmacc_vv_f32m4_tu(acc, diff, diff, vl);
            } else {
                acc = __riscv_vfmacc_vv_f32m4_tu(acc, vx, cf, vl);
            }
            i += vl;
        }
        // Single final reduction over the whole loop range.
        vfloat32m1_t init = __riscv_vfmv_s_f_f32m1(0.0f, 1);
        vfloat32m1_t red = __riscv_vfredusum_vs_f32m4_f32m1(acc, init, vlmax);
        return __riscv_vfmv_f_s_f32m1_f32(red);
    }

    template <bool IS_L2>
    float code_to_code_distance(const uint8_t* code1, const uint8_t* code2)
            const {
        const size_t vlmax = __riscv_vsetvlmax_e32m4();
        vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
        size_t i = 0;
        while (i < d) {
            size_t vl = __riscv_vsetvl_e8m1(d - i);
            vuint8m1_t a8 = __riscv_vle8_v_u8m1(code1 + i, vl);
            vuint8m1_t b8 = __riscv_vle8_v_u8m1(code2 + i, vl);
            vuint32m4_t au = __riscv_vzext_vf4_u32m4(a8, vl);
            vuint32m4_t bu = __riscv_vzext_vf4_u32m4(b8, vl);
            vint32m4_t as = __riscv_vreinterpret_v_u32m4_i32m4(au);
            vint32m4_t bs = __riscv_vreinterpret_v_u32m4_i32m4(bu);
            as = __riscv_vsub_vx_i32m4(as, 128, vl);
            bs = __riscv_vsub_vx_i32m4(bs, 128, vl);
            vfloat32m4_t af = __riscv_vfcvt_f_x_v_f32m4(as, vl);
            vfloat32m4_t bf = __riscv_vfcvt_f_x_v_f32m4(bs, vl);
            if constexpr (IS_L2) {
                // (a - 128) - (b - 128) == a - b
                vfloat32m4_t diff = __riscv_vfsub_vv_f32m4(af, bf, vl);
                acc = __riscv_vfmacc_vv_f32m4_tu(acc, diff, diff, vl);
            } else {
                acc = __riscv_vfmacc_vv_f32m4_tu(acc, af, bf, vl);
            }
            i += vl;
        }
        // Single final reduction over the whole loop range.
        vfloat32m1_t init = __riscv_vfmv_s_f_f32m1(0.0f, 1);
        vfloat32m1_t red = __riscv_vfredusum_vs_f32m4_f32m1(acc, init, vlmax);
        return __riscv_vfmv_f_s_f32m1_f32(red);
    }

    float query_to_code(const uint8_t* code) const final {
        if constexpr (Sim::metric_type == METRIC_L2) {
            return distance_to_xf<true>(this->q, code);
        } else {
            return distance_to_xf<false>(this->q, code);
        }
    }

    float compute_distance(const float* x, const uint8_t* code) const {
        if constexpr (Sim::metric_type == METRIC_L2) {
            return distance_to_xf<true>(x, code);
        } else {
            return distance_to_xf<false>(x, code);
        }
    }

    float compute_code_distance(const uint8_t* code1, const uint8_t* code2)
            const {
        if constexpr (Sim::metric_type == METRIC_L2) {
            return code_to_code_distance<true>(code1, code2);
        } else {
            return code_to_code_distance<false>(code1, code2);
        }
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return compute_code_distance(
                codes + i * code_size, codes + j * code_size);
    }
};

/*************************************************************************
 * Fast path — QT_4bit_uniform + L2
 *
 * 4-bit UNIFORM scaling: every component reconstructs as an affine function
 * of the 4-bit code,
 *     recon(c) = vmin + vdiff * (c + 0.5) / 15 = final_scale * c + bias
 * where final_scale = vdiff / 15. L2 distance between two reconstructions
 * therefore reduces to final_scale^2 * (q_c - c_c)^2 over integer codes,
 * so we can stay in the int domain and pay one float multiply at the end.
 *
 * The RVV path pre-nibbles the query into q_lo / q_hi (even / odd lanes)
 * once at set_query time and then processes native-VL-sized chunks of code
 * without ever decoding to float.
 ************************************************************************/

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
    /// Uses RVV's native VL; no fixed width assumptions. Returns the raw
    /// integer sum — caller multiplies by final_scale_sq.
    int64_t accumulate_int_l2(const uint8_t* code) const {
        int64_t acc = 0;
        size_t i = 0;
        while (i < d) {
            // Process up to vl codes per iteration. Each code byte packs two
            // 4-bit codes, so we load (vl + 1) / 2 bytes; keep vl even to
            // keep the nibble split aligned with the i % 2 split we used at
            // set_query time.
            size_t remaining = d - i;
            size_t vl = __riscv_vsetvl_e8m1(remaining);
            if (vl & 1) {
                vl -= 1; // keep even; tail handled on next iter or scalar
            }
            if (vl == 0) {
                break;
            }
            const size_t byte_vl = vl / 2;

            vuint8m1_t packed = __riscv_vle8_v_u8m1(code + i / 2, byte_vl);
            vuint8m1_t ql = __riscv_vle8_v_u8m1(q_lo.data() + i / 2, byte_vl);
            vuint8m1_t qh = __riscv_vle8_v_u8m1(q_hi.data() + i / 2, byte_vl);

            vuint8m1_t lo_nib = __riscv_vand_vx_u8m1(packed, 0x0F, byte_vl);
            vuint8m1_t hi_nib = __riscv_vsrl_vx_u8m1(packed, 4, byte_vl);

            // |ql - lo| and |qh - hi| fit in u8 (values are in [0, 15]).
            vuint8m1_t d_lo = __riscv_vsub_vv_u8m1(
                    __riscv_vmaxu_vv_u8m1(ql, lo_nib, byte_vl),
                    __riscv_vminu_vv_u8m1(ql, lo_nib, byte_vl),
                    byte_vl);
            vuint8m1_t d_hi = __riscv_vsub_vv_u8m1(
                    __riscv_vmaxu_vv_u8m1(qh, hi_nib, byte_vl),
                    __riscv_vminu_vv_u8m1(qh, hi_nib, byte_vl),
                    byte_vl);

            // Square via widening multiply (each byte squared fits in u16,
            // since max byte value is 15 -> 225).
            vuint16m2_t sq_lo = __riscv_vwmulu_vv_u16m2(d_lo, d_lo, byte_vl);
            vuint16m2_t sq_hi = __riscv_vwmulu_vv_u16m2(d_hi, d_hi, byte_vl);
            vuint16m2_t sq_sum = __riscv_vadd_vv_u16m2(sq_lo, sq_hi, byte_vl);

            // Reduce to a scalar u32 (safe: byte_vl * 450 fits in u32 for
            // any realistic d).
            vuint32m1_t zero = __riscv_vmv_v_x_u32m1(0, 1);
            vuint32m1_t red =
                    __riscv_vwredsumu_vs_u16m2_u32m1(sq_sum, zero, byte_vl);
            acc += __riscv_vmv_x_s_u32m1_u32(red);

            i += vl;
        }
        // Scalar tail: cover any leftover odd lane (at most one).
        for (; i < d; i++) {
            uint8_t c_code =
                    (i % 2 == 0) ? (code[i / 2] & 0x0F) : (code[i / 2] >> 4);
            uint8_t q_code = (i % 2 == 0) ? q_lo[i / 2] : q_hi[i / 2];
            int diff = int(q_code) - int(c_code);
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
