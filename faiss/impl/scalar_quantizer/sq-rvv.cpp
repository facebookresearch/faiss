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

namespace faiss {

namespace scalar_quantizer {

template <>
struct Codec8bit<SIMDLevel::RISCV_RVV> : Codec8bit<SIMDLevel::NONE> {
    static FAISS_ALWAYS_INLINE vfloat32m8_t
    decode_m8_components(const uint8_t* code, size_t i, size_t vl) {
        vuint8m2_t vu8 = __riscv_vle8_v_u8m2(code + i, vl);
        vuint32m8_t vu32 = __riscv_vzext_vf4_u32m8(vu8, vl);
        vfloat32m8_t vf32 = __riscv_vfcvt_f_xu_v_f32m8(vu32, vl);
        vf32 = __riscv_vfadd_vf_f32m8(vf32, 0.5f, vl);
        vf32 = __riscv_vfdiv_vf_f32m8(vf32, 255.0f, vl);
        return vf32;
    }
};

template <>
struct Codec4bit<SIMDLevel::RISCV_RVV> : Codec4bit<SIMDLevel::NONE> {
    static FAISS_ALWAYS_INLINE vfloat32m8_t
    decode_m8_components(const uint8_t* code, size_t i, size_t vl) {
        // Packed positions derive from the absolute component index:
        // implementations may hand out a non-maximal, non-pack-aligned vl
        // when AVL is between VLMAX and 2*VLMAX, so a chunk can start on
        // an odd component (mid-byte).
        const size_t byte_base = i >> 1;
        const size_t byte_vl = ((i + vl - 1) >> 1) - byte_base + 1;
        vuint8m2_t packed = __riscv_vle8_v_u8m2(code + byte_base, byte_vl);
        // Widen to 32-bit lanes before gathering: with more than 256 active
        // lanes (VLEN > 1024 for e32m8) 8-bit lane indices would wrap around
        // and decode the wrong bytes.
        vuint32m8_t packed32 = __riscv_vzext_vf4_u32m8(packed, byte_vl);
        vuint32m8_t comp =
                __riscv_vadd_vx_u32m8(__riscv_vid_v_u32m8(vl), i, vl);
        vuint32m8_t rel = __riscv_vsub_vx_u32m8(
                __riscv_vsrl_vx_u32m8(comp, 1, vl), byte_base, vl);
        vuint32m8_t bytes = __riscv_vrgather_vv_u32m8(packed32, rel, vl);
        vuint32m8_t lo = __riscv_vand_vx_u32m8(bytes, 0xf, vl);
        vuint32m8_t hi = __riscv_vsrl_vx_u32m8(bytes, 4, vl);
        vuint32m8_t parity = __riscv_vand_vx_u32m8(comp, 1, vl);
        vbool4_t odd = __riscv_vmsne_vx_u32m8_b4(parity, 0, vl);
        vuint32m8_t q = __riscv_vmerge_vvm_u32m8(lo, hi, odd, vl);
        vfloat32m8_t result = __riscv_vfcvt_f_xu_v_f32m8(q, vl);
        result = __riscv_vfadd_vf_f32m8(result, 0.5f, vl);
        result = __riscv_vfdiv_vf_f32m8(result, 15.0f, vl);
        return result;
    }
};

template <>
struct Codec6bit<SIMDLevel::RISCV_RVV> : Codec6bit<SIMDLevel::NONE> {
    static FAISS_ALWAYS_INLINE vfloat32m8_t
    decode_m8_components(const uint8_t* code, size_t i, size_t vl) {
        size_t last = i + vl - 1;
        size_t last_used = 3 * (last >> 2) + ((last & 3) < 2 ? (last & 3) : 2);
        vuint32m8_t idx = __riscv_vid_v_u32m8(vl);
        idx = __riscv_vadd_vx_u32m8(idx, i, vl);
        vuint32m8_t block = __riscv_vsrl_vx_u32m8(idx, 2, vl);
        vuint32m8_t lane = __riscv_vand_vx_u32m8(idx, 3, vl);
        vuint32m8_t base = __riscv_vadd_vv_u32m8(block, block, vl);
        base = __riscv_vadd_vv_u32m8(base, block, vl);
        vuint8m2_t b0 = __riscv_vluxei32_v_u8m2(code, base, vl);
        vuint8m2_t b1 = __riscv_vluxei32_v_u8m2(
                code,
                __riscv_vminu_vx_u32m8(
                        __riscv_vadd_vx_u32m8(base, 1, vl), last_used, vl),
                vl);
        vuint8m2_t b2 = __riscv_vluxei32_v_u8m2(
                code,
                __riscv_vminu_vx_u32m8(
                        __riscv_vadd_vx_u32m8(base, 2, vl), last_used, vl),
                vl);
        vuint32m8_t wb0 = __riscv_vzext_vf4_u32m8(b0, vl);
        vuint32m8_t wb1 = __riscv_vzext_vf4_u32m8(b1, vl);
        vuint32m8_t wb2 = __riscv_vzext_vf4_u32m8(b2, vl);
        vuint32m8_t x0 = __riscv_vand_vx_u32m8(wb0, 0x3f, vl);
        vuint32m8_t x1 = __riscv_vor_vv_u32m8(
                __riscv_vsrl_vx_u32m8(wb0, 6, vl),
                __riscv_vsll_vx_u32m8(
                        __riscv_vand_vx_u32m8(wb1, 0xf, vl), 2, vl),
                vl);
        vuint32m8_t x2 = __riscv_vor_vv_u32m8(
                __riscv_vsrl_vx_u32m8(wb1, 4, vl),
                __riscv_vsll_vx_u32m8(__riscv_vand_vx_u32m8(wb2, 3, vl), 4, vl),
                vl);
        vuint32m8_t x3 = __riscv_vsrl_vx_u32m8(wb2, 2, vl);
        vbool4_t m0 = __riscv_vmseq_vx_u32m8_b4(lane, 0, vl);
        vbool4_t m1 = __riscv_vmseq_vx_u32m8_b4(lane, 1, vl);
        vbool4_t m2 = __riscv_vmseq_vx_u32m8_b4(lane, 2, vl);
        vuint32m8_t bits = x3;
        bits = __riscv_vmerge_vvm_u32m8(bits, x2, m2, vl);
        bits = __riscv_vmerge_vvm_u32m8(bits, x1, m1, vl);
        bits = __riscv_vmerge_vvm_u32m8(bits, x0, m0, vl);
        vfloat32m8_t out = __riscv_vfcvt_f_xu_v_f32m8(bits, vl);
        out = __riscv_vfadd_vf_f32m8(out, 0.5f, vl);
        out = __riscv_vfdiv_vf_f32m8(out, 63.0f, vl);
        return out;
    }
};

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

    FAISS_ALWAYS_INLINE vfloat32m8_t
    reconstruct_m8_components(const uint8_t* code, size_t i, size_t vl) const {
        vfloat32m8_t xi = Codec::decode_m8_components(code, i, vl);
        return __riscv_vfadd_vf_f32m8(
                __riscv_vfmul_vf_f32m8(xi, this->vdiff, vl), this->vmin, vl);
    }

    void decode_vector(const uint8_t* code, float* x) const final {
        size_t i = 0;
        const size_t d = this->d;
        while (i < d) {
            const size_t vl = __riscv_vsetvl_e32m8(d - i);
            vfloat32m8_t xi = Codec::decode_m8_components(code, i, vl);
            xi = __riscv_vfmul_vf_f32m8(xi, this->vdiff, vl);
            xi = __riscv_vfadd_vf_f32m8(xi, this->vmin, vl);
            __riscv_vse32_v_f32m8(x + i, xi, vl);
            i += vl;
        }
    }
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

    FAISS_ALWAYS_INLINE vfloat32m8_t
    reconstruct_m8_components(const uint8_t* code, size_t i, size_t vl) const {
        vfloat32m8_t xi = Codec::decode_m8_components(code, i, vl);
        vfloat32m8_t vminv = __riscv_vle32_v_f32m8(this->vmin + i, vl);
        vfloat32m8_t vdiffv = __riscv_vle32_v_f32m8(this->vdiff + i, vl);
        return __riscv_vfmadd_vv_f32m8(xi, vdiffv, vminv, vl);
    }

    void decode_vector(const uint8_t* code, float* x) const final {
        size_t i = 0;
        const size_t d = this->d;
        while (i < d) {
            const size_t vl = __riscv_vsetvl_e32m8(d - i);
            vfloat32m8_t xi = Codec::decode_m8_components(code, i, vl);
            vfloat32m8_t vminv = __riscv_vle32_v_f32m8(this->vmin + i, vl);
            vfloat32m8_t vdiffv = __riscv_vle32_v_f32m8(this->vdiff + i, vl);
            xi = __riscv_vfmul_vv_f32m8(xi, vdiffv, vl);
            xi = __riscv_vfadd_vv_f32m8(xi, vminv, vl);
            __riscv_vse32_v_f32m8(x + i, xi, vl);
            i += vl;
        }
    }
};

template <>
struct QuantizerFP16<SIMDLevel::RISCV_RVV> : QuantizerFP16<SIMDLevel::NONE> {
    QuantizerFP16(size_t d, const std::vector<float>& trained)
            : QuantizerFP16<SIMDLevel::NONE>(d, trained) {}

// Zvfhmin is part of the RISCV_RVV level's minimum ISA (see
// faiss/utils/simd_levels.h); this translation unit is compiled with
// -march=rv64gcv_zvfhmin, so the RVV FP16 reconstruction below is always
// emitted in faiss builds. The preprocessor fallback keeps the scalar FP16
// path for out-of-tree builds that compile this file without the extension;
// has_reconstruct_m8_v then resolves to false and QT_fp16 stays scalar.
#if defined(__riscv_zvfhmin) || defined(__riscv_zvfh)
    FAISS_ALWAYS_INLINE vfloat32m8_t
    reconstruct_m8_components(const uint8_t* code, size_t i, size_t vl) const {
        vfloat16m4_t vh =
                __riscv_vle16_v_f16m4((const _Float16*)(code + 2 * i), vl);
        return __riscv_vfwcvt_f_f_v_f32m8(vh, vl);
    }

    void decode_vector(const uint8_t* code, float* x) const final {
        const auto* code16 = reinterpret_cast<const uint16_t*>(code);
        size_t i = 0;
        const size_t d = this->d;
        while (i < d) {
            const size_t vl = __riscv_vsetvl_e16m4(d - i);
            vuint16m4_t vh = __riscv_vle16_v_u16m4(code16 + i, vl);
            vbool4_t is_nan = __riscv_vmsgtu_vx_u16m4_b4(
                    __riscv_vand_vx_u16m4(vh, 0x7FFF, vl), 0x7C00, vl);
            vfloat32m8_t vf = __riscv_vfwcvt_f_f_v_f32m8_m(
                    __riscv_vmnot_m_b4(is_nan, vl),
                    __riscv_vreinterpret_v_u16m4_f16m4(vh),
                    vl);
            vf = __riscv_vfsub_vf_f32m8_m(
                    __riscv_vmnot_m_b4(is_nan, vl), vf, 0.0f, vl);
            if (__riscv_vfirst_m_b4(is_nan, vl) >= 0) {
                vint32m8_t bits = __riscv_vsext_vf2_i32m8(
                        __riscv_vreinterpret_v_u16m4_i16m4(vh), vl);
                bits = __riscv_vsll_vx_i32m8(bits, 13, vl);
                bits = __riscv_vand_vx_i32m8(bits, -0x70002000, vl);
                bits = __riscv_vor_vx_i32m8(bits, 0x70000000, vl);
                vf = __riscv_vmerge_vvm_f32m8(
                        vf,
                        __riscv_vreinterpret_v_i32m8_f32m8(bits),
                        is_nan,
                        vl);
            }
            __riscv_vse32_v_f32m8(x + i, vf, vl);
            i += vl;
        }
    }
#endif
};

template <>
struct QuantizerBF16<SIMDLevel::RISCV_RVV> : QuantizerBF16<SIMDLevel::NONE> {
    QuantizerBF16(size_t d, const std::vector<float>& trained)
            : QuantizerBF16<SIMDLevel::NONE>(d, trained) {}

    FAISS_ALWAYS_INLINE vfloat32m8_t
    reconstruct_m8_components(const uint8_t* code, size_t i, size_t vl) const {
        vuint16m4_t v16 =
                __riscv_vle16_v_u16m4((const uint16_t*)(code + 2 * i), vl);
        vuint32m8_t v32 = __riscv_vzext_vf2_u32m8(v16, vl);
        v32 = __riscv_vsll_vx_u32m8(v32, 16, vl);
        return __riscv_vreinterpret_v_u32m8_f32m8(v32);
    }

    void decode_vector(const uint8_t* code, float* x) const final {
        const auto* code16 = reinterpret_cast<const uint16_t*>(code);
        size_t i = 0;
        const size_t d = this->d;
        while (i < d) {
            const size_t vl = __riscv_vsetvl_e16m4(d - i);
            vuint16m4_t v = __riscv_vle16_v_u16m4(code16 + i, vl);
            vuint32m8_t w = __riscv_vzext_vf2_u32m8(v, vl);
            w = __riscv_vsll_vx_u32m8(w, 16, vl);
            __riscv_vse32_v_f32m8(
                    x + i,
                    __riscv_vreinterpret_v_u32m8_f32m8(w),
                    vl);
            i += vl;
        }
    }
};

template <>
struct Quantizer8bitDirect<SIMDLevel::RISCV_RVV>
        : Quantizer8bitDirect<SIMDLevel::NONE> {
    Quantizer8bitDirect(size_t d, const std::vector<float>& trained)
            : Quantizer8bitDirect<SIMDLevel::NONE>(d, trained) {}

    FAISS_ALWAYS_INLINE vfloat32m8_t
    reconstruct_m8_components(const uint8_t* code, size_t i, size_t vl) const {
        vuint8m2_t vu8 = __riscv_vle8_v_u8m2(code + i, vl);
        vuint32m8_t vu32 = __riscv_vzext_vf4_u32m8(vu8, vl);
        return __riscv_vfcvt_f_xu_v_f32m8(vu32, vl);
    }

    void decode_vector(const uint8_t* code, float* x) const final {
        size_t i = 0;
        const size_t d = this->d;
        while (i < d) {
            const size_t vl = __riscv_vsetvl_e8m2(d - i);
            vuint8m2_t b = __riscv_vle8_v_u8m2(code + i, vl);
            vuint16m4_t u = __riscv_vzext_vf2_u16m4(b, vl);
            vfloat32m8_t f = __riscv_vfwcvt_f_xu_v_f32m8(u, vl);
            __riscv_vse32_v_f32m8(x + i, f, vl);
            i += vl;
        }
    }
};

template <>
struct Quantizer8bitDirectSigned<SIMDLevel::RISCV_RVV>
        : Quantizer8bitDirectSigned<SIMDLevel::NONE> {
    Quantizer8bitDirectSigned(size_t d, const std::vector<float>& trained)
            : Quantizer8bitDirectSigned<SIMDLevel::NONE>(d, trained) {}

    FAISS_ALWAYS_INLINE vfloat32m8_t
    reconstruct_m8_components(const uint8_t* code, size_t i, size_t vl) const {
        vuint8m2_t vu8 = __riscv_vle8_v_u8m2(code + i, vl);
        vuint32m8_t vu32 = __riscv_vzext_vf4_u32m8(vu8, vl);
        vfloat32m8_t vf = __riscv_vfcvt_f_xu_v_f32m8(vu32, vl);
        return __riscv_vfsub_vf_f32m8(vf, 128.0f, vl);
    }

    void decode_vector(const uint8_t* code, float* x) const final {
        size_t i = 0;
        const size_t d = this->d;
        while (i < d) {
            const size_t vl = __riscv_vsetvl_e8m2(d - i);
            vint8m2_t v = __riscv_vle8_v_i8m2(
                    reinterpret_cast<const int8_t*>(code) + i, vl);
            v = __riscv_vxor_vx_i8m2(v, -128, vl);
            vint16m4_t v16 = __riscv_vsext_vf2_i16m4(v, vl);
            vfloat32m8_t f = __riscv_vfwcvt_f_x_v_f32m8(v16, vl);
            __riscv_vse32_v_f32m8(x + i, f, vl);
            i += vl;
        }
    }
};

template <>
struct QuantizerLloydMax<1, SIMDLevel::RISCV_RVV>
        : QuantizerLloydMax<1, SIMDLevel::NONE> {
    using Base = QuantizerLloydMax<1, SIMDLevel::NONE>;

    QuantizerLloydMax(size_t d, const std::vector<float>& trained)
            : Base(d, trained) {}

    FAISS_ALWAYS_INLINE vfloat32m8_t
    reconstruct_m8_components(const uint8_t* code, size_t i, size_t vl) const {
        // Absolute component indexing: a non-maximal vl can leave a chunk
        // starting mid-byte (see Codec4bit above).
        const size_t byte_base = i >> 3;
        const size_t byte_vl = ((i + vl - 1) >> 3) - byte_base + 1;
        vuint8m2_t packed = __riscv_vle8_v_u8m2(code + byte_base, byte_vl);
        // 32-bit lane indices: 8-bit indices wrap above 256 active lanes
        vuint32m8_t packed32 = __riscv_vzext_vf4_u32m8(packed, byte_vl);
        vuint32m8_t comp =
                __riscv_vadd_vx_u32m8(__riscv_vid_v_u32m8(vl), i, vl);
        vuint32m8_t rel = __riscv_vsub_vx_u32m8(
                __riscv_vsrl_vx_u32m8(comp, 3, vl), byte_base, vl);
        vuint32m8_t bytes = __riscv_vrgather_vv_u32m8(packed32, rel, vl);
        vuint32m8_t shift = __riscv_vand_vx_u32m8(comp, 7, vl);
        vuint32m8_t idx = __riscv_vand_vx_u32m8(
                __riscv_vsrl_vv_u32m8(bytes, shift, vl), 1, vl);
        vuint32m8_t off = __riscv_vsll_vx_u32m8(idx, 2, vl);
        return __riscv_vluxei32_v_f32m8(this->centroids, off, vl);
    }

    void decode_vector(const uint8_t* code, float* x) const final {
        size_t i = 0;
        while (i < this->d) {
            size_t vl = __riscv_vsetvl_e32m8(this->d - i);
            vfloat32m8_t v = reconstruct_m8_components(code, i, vl);
            __riscv_vse32_v_f32m8(x + i, v, vl);
            i += vl;
        }
    }
};

template <>
struct QuantizerLloydMax<2, SIMDLevel::RISCV_RVV>
        : QuantizerLloydMax<2, SIMDLevel::NONE> {
    using Base = QuantizerLloydMax<2, SIMDLevel::NONE>;

    QuantizerLloydMax(size_t d, const std::vector<float>& trained)
            : Base(d, trained) {}

    FAISS_ALWAYS_INLINE vfloat32m8_t
    reconstruct_m8_components(const uint8_t* code, size_t i, size_t vl) const {
        // Absolute component indexing: a non-maximal vl can leave a chunk
        // starting mid-byte (see Codec4bit above).
        const size_t byte_base = i >> 2;
        const size_t byte_vl = ((i + vl - 1) >> 2) - byte_base + 1;
        vuint8m2_t packed = __riscv_vle8_v_u8m2(code + byte_base, byte_vl);
        // 32-bit lane indices: 8-bit indices wrap above 256 active lanes
        vuint32m8_t packed32 = __riscv_vzext_vf4_u32m8(packed, byte_vl);
        vuint32m8_t comp =
                __riscv_vadd_vx_u32m8(__riscv_vid_v_u32m8(vl), i, vl);
        vuint32m8_t rel = __riscv_vsub_vx_u32m8(
                __riscv_vsrl_vx_u32m8(comp, 2, vl), byte_base, vl);
        vuint32m8_t bytes = __riscv_vrgather_vv_u32m8(packed32, rel, vl);
        vuint32m8_t shift = __riscv_vsll_vx_u32m8(
                __riscv_vand_vx_u32m8(comp, 3, vl), 1, vl);
        vuint32m8_t idx = __riscv_vand_vx_u32m8(
                __riscv_vsrl_vv_u32m8(bytes, shift, vl), 3, vl);
        vuint32m8_t off = __riscv_vsll_vx_u32m8(idx, 2, vl);
        return __riscv_vluxei32_v_f32m8(this->centroids, off, vl);
    }

    void decode_vector(const uint8_t* code, float* x) const final {
        size_t i = 0;
        while (i < this->d) {
            size_t vl = __riscv_vsetvl_e32m8(this->d - i);
            vfloat32m8_t v = reconstruct_m8_components(code, i, vl);
            __riscv_vse32_v_f32m8(x + i, v, vl);
            i += vl;
        }
    }
};

template <>
struct QuantizerLloydMax<3, SIMDLevel::RISCV_RVV>
        : QuantizerLloydMax<3, SIMDLevel::NONE> {
    using Base = QuantizerLloydMax<3, SIMDLevel::NONE>;

    QuantizerLloydMax(size_t d, const std::vector<float>& trained)
            : Base(d, trained) {}

    FAISS_ALWAYS_INLINE vfloat32m8_t
    reconstruct_m8_components(const uint8_t* code, size_t i, size_t vl) const {
        vuint32m8_t idx0 =
                __riscv_vadd_vx_u32m8(__riscv_vid_v_u32m8(vl), i, vl);
        vuint32m8_t bitpos = __riscv_vmul_vx_u32m8(idx0, 3, vl);
        vuint32m8_t byteoff = __riscv_vsrl_vx_u32m8(bitpos, 3, vl);
        vuint32m8_t shift = __riscv_vand_vx_u32m8(bitpos, 7, vl);
        size_t last = i + vl - 1;
        size_t last_used = (3 * last + 2) >> 3;
        vuint8m2_t lo = __riscv_vluxei32_v_u8m2(code, byteoff, vl);
        vuint8m2_t hi = __riscv_vluxei32_v_u8m2(
                code,
                __riscv_vminu_vx_u32m8(
                        __riscv_vadd_vx_u32m8(byteoff, 1, vl), last_used, vl),
                vl);
        vuint32m8_t w = __riscv_vor_vv_u32m8(
                __riscv_vzext_vf4_u32m8(lo, vl),
                __riscv_vsll_vx_u32m8(__riscv_vzext_vf4_u32m8(hi, vl), 8, vl),
                vl);
        vuint32m8_t idx = __riscv_vand_vx_u32m8(
                __riscv_vsrl_vv_u32m8(w, shift, vl), 7, vl);
        vuint32m8_t off = __riscv_vsll_vx_u32m8(idx, 2, vl);
        return __riscv_vluxei32_v_f32m8(this->centroids, off, vl);
    }

    void decode_vector(const uint8_t* code, float* x) const final {
        size_t i = 0;
        while (i < this->d) {
            size_t vl = __riscv_vsetvl_e32m8(this->d - i);
            vfloat32m8_t v = reconstruct_m8_components(code, i, vl);
            __riscv_vse32_v_f32m8(x + i, v, vl);
            i += vl;
        }
    }
};

template <>
struct QuantizerLloydMax<4, SIMDLevel::RISCV_RVV>
        : QuantizerLloydMax<4, SIMDLevel::NONE> {
    using Base = QuantizerLloydMax<4, SIMDLevel::NONE>;

    QuantizerLloydMax(size_t d, const std::vector<float>& trained)
            : Base(d, trained) {}

    FAISS_ALWAYS_INLINE vfloat32m8_t
    reconstruct_m8_components(const uint8_t* code, size_t i, size_t vl) const {
        // Absolute component indexing: a non-maximal vl can leave a chunk
        // starting mid-byte (see Codec4bit above).
        const size_t byte_base = i >> 1;
        const size_t byte_vl = ((i + vl - 1) >> 1) - byte_base + 1;
        vuint8m2_t packed = __riscv_vle8_v_u8m2(code + byte_base, byte_vl);
        // 32-bit lane indices: 8-bit indices wrap above 256 active lanes
        vuint32m8_t packed32 = __riscv_vzext_vf4_u32m8(packed, byte_vl);
        vuint32m8_t comp =
                __riscv_vadd_vx_u32m8(__riscv_vid_v_u32m8(vl), i, vl);
        vuint32m8_t rel = __riscv_vsub_vx_u32m8(
                __riscv_vsrl_vx_u32m8(comp, 1, vl), byte_base, vl);
        vuint32m8_t bytes = __riscv_vrgather_vv_u32m8(packed32, rel, vl);
        vuint32m8_t lo = __riscv_vand_vx_u32m8(bytes, 0xf, vl);
        vuint32m8_t hi = __riscv_vsrl_vx_u32m8(bytes, 4, vl);
        vbool4_t odd = __riscv_vmsne_vx_u32m8_b4(
                __riscv_vand_vx_u32m8(comp, 1, vl), 0, vl);
        vuint32m8_t idx = __riscv_vmerge_vvm_u32m8(lo, hi, odd, vl);
        vuint32m8_t off = __riscv_vsll_vx_u32m8(idx, 2, vl);
        return __riscv_vluxei32_v_f32m8(this->centroids, off, vl);
    }

    void decode_vector(const uint8_t* code, float* x) const final {
        size_t i = 0;
        while (i < this->d) {
            size_t vl = __riscv_vsetvl_e32m8(this->d - i);
            vfloat32m8_t v = reconstruct_m8_components(code, i, vl);
            __riscv_vse32_v_f32m8(x + i, v, vl);
            i += vl;
        }
    }
};

template <>
struct QuantizerLloydMax<8, SIMDLevel::RISCV_RVV>
        : QuantizerLloydMax<8, SIMDLevel::NONE> {
    using Base = QuantizerLloydMax<8, SIMDLevel::NONE>;

    QuantizerLloydMax(size_t d, const std::vector<float>& trained)
            : Base(d, trained) {}

    FAISS_ALWAYS_INLINE vfloat32m8_t
    reconstruct_m8_components(const uint8_t* code, size_t i, size_t vl) const {
        vuint8m2_t vb = __riscv_vle8_v_u8m2(code + i, vl);
        vuint32m8_t off =
                __riscv_vsll_vx_u32m8(__riscv_vzext_vf4_u32m8(vb, vl), 2, vl);
        return __riscv_vluxei32_v_f32m8(this->centroids, off, vl);
    }

    void decode_vector(const uint8_t* code, float* x) const final {
        size_t i = 0;
        while (i < this->d) {
            size_t vl = __riscv_vsetvl_e32m8(this->d - i);
            vfloat32m8_t v = reconstruct_m8_components(code, i, vl);
            __riscv_vse32_v_f32m8(x + i, v, vl);
            i += vl;
        }
    }
};

template <>
struct SimilarityL2<SIMDLevel::RISCV_RVV> : SimilarityL2<SIMDLevel::NONE> {
    using SimilarityL2<SIMDLevel::NONE>::SimilarityL2;

    static constexpr SIMDLevel simd_level = SIMDLevel::RISCV_RVV;

    FAISS_ALWAYS_INLINE void begin_m8() {
        yi = y;
    }

    static FAISS_ALWAYS_INLINE vfloat32m8_t zero_m8(size_t vl) {
        return __riscv_vfmv_v_f_f32m8(0.0f, vl);
    }

    FAISS_ALWAYS_INLINE vfloat32m8_t
    add_m8_components(vfloat32m8_t accu, vfloat32m8_t x, size_t vl) {
        vfloat32m8_t yiv = __riscv_vle32_v_f32m8(yi, vl);
        yi += vl;
        vfloat32m8_t tmp = __riscv_vfsub_vv_f32m8(yiv, x, vl);
        // tail-undisturbed: the final short iteration must not clobber
        // accumulator lanes above vl that were filled by earlier iterations
        return __riscv_vfmacc_vv_f32m8_tu(accu, tmp, tmp, vl);
    }

    static FAISS_ALWAYS_INLINE vfloat32m8_t add_m8_components_2(
            vfloat32m8_t accu,
            vfloat32m8_t x,
            vfloat32m8_t y_2,
            size_t vl) {
        vfloat32m8_t tmp = __riscv_vfsub_vv_f32m8(y_2, x, vl);
        return __riscv_vfmacc_vv_f32m8_tu(accu, tmp, tmp, vl);
    }

    static FAISS_ALWAYS_INLINE float result_m8(vfloat32m8_t accu, size_t vl) {
        vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
        vfloat32m1_t sum = __riscv_vfredusum_vs_f32m8_f32m1(accu, zero, vl);
        return __riscv_vfmv_f_s_f32m1_f32(sum);
    }
};

template <>
struct SimilarityIP<SIMDLevel::RISCV_RVV> : SimilarityIP<SIMDLevel::NONE> {
    using SimilarityIP<SIMDLevel::NONE>::SimilarityIP;

    static constexpr SIMDLevel simd_level = SIMDLevel::RISCV_RVV;

    FAISS_ALWAYS_INLINE void begin_m8() {
        yi = y;
    }

    static FAISS_ALWAYS_INLINE vfloat32m8_t zero_m8(size_t vl) {
        return __riscv_vfmv_v_f_f32m8(0.0f, vl);
    }

    FAISS_ALWAYS_INLINE vfloat32m8_t
    add_m8_components(vfloat32m8_t accu, vfloat32m8_t x, size_t vl) {
        vfloat32m8_t yiv = __riscv_vle32_v_f32m8(yi, vl);
        yi += vl;
        return __riscv_vfmacc_vv_f32m8_tu(accu, yiv, x, vl);
    }

    static FAISS_ALWAYS_INLINE vfloat32m8_t add_m8_components_2(
            vfloat32m8_t accu,
            vfloat32m8_t x1,
            vfloat32m8_t x2,
            size_t vl) {
        return __riscv_vfmacc_vv_f32m8_tu(accu, x1, x2, vl);
    }

    static FAISS_ALWAYS_INLINE float result_m8(vfloat32m8_t accu, size_t vl) {
        vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
        vfloat32m1_t sum = __riscv_vfredusum_vs_f32m8_f32m1(accu, zero, vl);
        return __riscv_vfmv_f_s_f32m1_f32(sum);
    }
};

template <class Quantizer>
inline constexpr bool has_reconstruct_m8_v =
        requires(const Quantizer& q, const uint8_t* code, size_t i, size_t vl) {
            q.reconstruct_m8_components(code, i, vl);
        };

template <class Quantizer, class Similarity>
    requires(!has_reconstruct_m8_v<Quantizer>)
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

template <class Quantizer, class Similarity>
    requires(has_reconstruct_m8_v<Quantizer>)
struct DCTemplate<Quantizer, Similarity, SIMDLevel::RISCV_RVV>
        : SQDistanceComputer {
    using Sim = Similarity;

    Quantizer quant;

    DCTemplate(size_t d, const std::vector<float>& trained)
            : quant(d, trained) {}

    float compute_distance(const float* x, const uint8_t* code) const {
        if (quant.d == 0) {
            return 0.0f;
        }
        Similarity sim(x);
        sim.begin_m8();
        const size_t first_vl = __riscv_vsetvl_e32m8(quant.d);
        vfloat32m8_t accu = Sim::zero_m8(first_vl);
        size_t i = 0;
        while (i < quant.d) {
            size_t vl = __riscv_vsetvl_e32m8(quant.d - i);
            vfloat32m8_t xi = quant.reconstruct_m8_components(code, i, vl);
            accu = sim.add_m8_components(accu, xi, vl);
            i += vl;
        }
        return Sim::result_m8(accu, first_vl);
    }

    float compute_code_distance(const uint8_t* code1, const uint8_t* code2)
            const {
        if (quant.d == 0) {
            return 0.0f;
        }
        Similarity sim(nullptr);
        sim.begin_m8();
        const size_t first_vl = __riscv_vsetvl_e32m8(quant.d);
        vfloat32m8_t accu = Sim::zero_m8(first_vl);
        size_t i = 0;
        while (i < quant.d) {
            size_t vl = __riscv_vsetvl_e32m8(quant.d - i);
            vfloat32m8_t x1 = quant.reconstruct_m8_components(code1, i, vl);
            vfloat32m8_t x2 = quant.reconstruct_m8_components(code2, i, vl);
            accu = Sim::add_m8_components_2(accu, x1, x2, vl);
            i += vl;
        }
        return Sim::result_m8(accu, first_vl);
    }

    void set_query(const float* x) final {
        q = x;
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return compute_code_distance(
                codes + i * code_size, codes + j * code_size);
    }

    float query_to_code(const uint8_t* code) const final {
        return compute_distance(q, code);
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
        dis0 = compute_distance(q, code_0);
        dis1 = compute_distance(q, code_1);
        dis2 = compute_distance(q, code_2);
        dis3 = compute_distance(q, code_3);
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
