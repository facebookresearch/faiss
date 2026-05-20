/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/EDENQuantizerDistance.h>
#include <faiss/impl/simdlib/simdlib.h>

#include <cstdint>

namespace faiss {

namespace eden_distance {

enum class CodeDotLUTKind {
    None,
    Byte,
    HalfByte,
};

inline bool supports_byte_lut(size_t nb_bits) {
    return nb_bits == 1 || nb_bits == 2 || nb_bits == 4;
}

inline bool use_half_byte_lut(size_t nb_bits, size_t num_bytes) {
    if (nb_bits == 1) {
        return num_bytes >= 32;
    }
    if (nb_bits == 2) {
        return num_bytes >= 16;
    }
    if (nb_bits == 4) {
        return num_bytes >= 128;
    }
    return false;
}

inline void build_code_dot_lut(
        const float* query,
        size_t d,
        size_t nb_bits,
        std::vector<float>& lut,
        CodeDotLUTKind& lut_kind) {
    if (!supports_byte_lut(nb_bits)) {
        lut.clear();
        lut_kind = CodeDotLUTKind::None;
        return;
    }

    const size_t values_per_byte = 8 / nb_bits;
    const size_t num_bytes = (d + values_per_byte - 1) / values_per_byte;
    const uint8_t mask = static_cast<uint8_t>((1u << nb_bits) - 1);
    const float* centroids = codebook(nb_bits).data();

    if (use_half_byte_lut(nb_bits, num_bytes)) {
        lut_kind = CodeDotLUTKind::HalfByte;
        lut.resize(num_bytes * 32);

        if (nb_bits == 1) {
            for (size_t byte_no = 0; byte_no < num_bytes; byte_no++) {
                const size_t dim0 = byte_no * 8;
                float* table = lut.data() + byte_no * 32;
                const float q0 = dim0 < d ? query[dim0] : 0.0f;
                const float q1 = dim0 + 1 < d ? query[dim0 + 1] : 0.0f;
                const float q2 = dim0 + 2 < d ? query[dim0 + 2] : 0.0f;
                const float q3 = dim0 + 3 < d ? query[dim0 + 3] : 0.0f;
                const float q4 = dim0 + 4 < d ? query[dim0 + 4] : 0.0f;
                const float q5 = dim0 + 5 < d ? query[dim0 + 5] : 0.0f;
                const float q6 = dim0 + 6 < d ? query[dim0 + 6] : 0.0f;
                const float q7 = dim0 + 7 < d ? query[dim0 + 7] : 0.0f;
                for (size_t half_byte_value = 0; half_byte_value < 16;
                     half_byte_value++) {
                    const float c0 = centroids[half_byte_value & 0x01];
                    const float c1 = centroids[(half_byte_value >> 1) & 0x01];
                    const float c2 = centroids[(half_byte_value >> 2) & 0x01];
                    const float c3 = centroids[(half_byte_value >> 3) & 0x01];
                    table[half_byte_value] =
                            ((q0 * c0 + q1 * c1) + q2 * c2) + q3 * c3;
                    table[16 + half_byte_value] =
                            ((q4 * c0 + q5 * c1) + q6 * c2) + q7 * c3;
                }
            }
            return;
        }

        if (nb_bits == 4) {
            for (size_t byte_no = 0; byte_no < num_bytes; byte_no++) {
                const size_t dim0 = byte_no * 2;
                float* table = lut.data() + byte_no * 32;
                const float q0 = dim0 < d ? query[dim0] : 0.0f;
                const float q1 = dim0 + 1 < d ? query[dim0 + 1] : 0.0f;
                for (size_t assignment = 0; assignment < 16; assignment++) {
                    const float centroid = centroids[assignment];
                    table[assignment] = q0 * centroid;
                    table[16 + assignment] = q1 * centroid;
                }
            }
            return;
        }

        if (nb_bits == 2) {
            for (size_t byte_no = 0; byte_no < num_bytes; byte_no++) {
                const size_t dim0 = byte_no * 4;
                float* table = lut.data() + byte_no * 32;
                const float q0 = dim0 < d ? query[dim0] : 0.0f;
                const float q1 = dim0 + 1 < d ? query[dim0 + 1] : 0.0f;
                const float q2 = dim0 + 2 < d ? query[dim0 + 2] : 0.0f;
                const float q3 = dim0 + 3 < d ? query[dim0 + 3] : 0.0f;
                for (size_t half_byte_value = 0; half_byte_value < 16;
                     half_byte_value++) {
                    const float c0 = centroids[half_byte_value & 0x03];
                    const float c1 = centroids[(half_byte_value >> 2) & 0x03];
                    table[half_byte_value] = q0 * c0 + q1 * c1;
                    table[16 + half_byte_value] = q2 * c0 + q3 * c1;
                }
            }
            return;
        }

        const size_t values_per_half_byte = values_per_byte / 2;
        for (size_t byte_no = 0; byte_no < num_bytes; byte_no++) {
            const size_t dim0 = byte_no * values_per_byte;
            float* table = lut.data() + byte_no * 32;
            for (size_t half_byte_value = 0; half_byte_value < 16;
                 half_byte_value++) {
                float low_dot = 0.0f;
                float high_dot = 0.0f;
                for (size_t j = 0; j < values_per_half_byte; j++) {
                    uint8_t assignment = static_cast<uint8_t>(
                            (half_byte_value >> (j * nb_bits)) & mask);
                    size_t dim = dim0 + j;
                    if (dim < d) {
                        low_dot += query[dim] * centroids[assignment];
                    }
                    dim += values_per_half_byte;
                    if (dim < d) {
                        high_dot += query[dim] * centroids[assignment];
                    }
                }
                table[half_byte_value] = low_dot;
                table[16 + half_byte_value] = high_dot;
            }
        }
        return;
    }

    lut_kind = CodeDotLUTKind::Byte;
    lut.resize(num_bytes * 256);
    for (size_t byte_no = 0; byte_no < num_bytes; byte_no++) {
        const size_t dim0 = byte_no * values_per_byte;
        float* table = lut.data() + byte_no * 256;
        for (size_t byte_value = 0; byte_value < 256; byte_value++) {
            float dot = 0.0f;
            for (size_t j = 0; j < values_per_byte; j++) {
                const size_t dim = dim0 + j;
                if (dim >= d) {
                    break;
                }
                const uint8_t assignment = static_cast<uint8_t>(
                        (byte_value >> (j * nb_bits)) & mask);
                dot += query[dim] * centroids[assignment];
            }
            table[byte_value] = dot;
        }
    }
}

inline float compute_code_dot_lut(
        const uint8_t* __restrict code,
        const std::vector<float>& lut,
        CodeDotLUTKind lut_kind,
        size_t packed_size) {
    const float* __restrict table = lut.data();
    size_t i = 0;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    if (lut_kind == CodeDotLUTKind::HalfByte) {
        for (; i + 4 <= packed_size; i += 4) {
            uint8_t byte0 = code[i];
            uint8_t byte1 = code[i + 1];
            uint8_t byte2 = code[i + 2];
            uint8_t byte3 = code[i + 3];
            acc0 += table[byte0 & 0x0f] + table[16 + (byte0 >> 4)];
            acc1 += table[32 + (byte1 & 0x0f)] +
                    table[48 + (byte1 >> 4)];
            acc2 += table[64 + (byte2 & 0x0f)] +
                    table[80 + (byte2 >> 4)];
            acc3 += table[96 + (byte3 & 0x0f)] +
                    table[112 + (byte3 >> 4)];
            table += 128;
        }
        float dot = (acc0 + acc1) + (acc2 + acc3);
        for (; i < packed_size; i++) {
            const uint8_t byte = code[i];
            dot += table[byte & 0x0f] + table[16 + (byte >> 4)];
            table += 32;
        }
        return dot;
    }

    for (; i + 4 <= packed_size; i += 4) {
        acc0 += table[code[i]];
        acc1 += table[256 + code[i + 1]];
        acc2 += table[512 + code[i + 2]];
        acc3 += table[768 + code[i + 3]];
        table += 1024;
    }
    float dot = (acc0 + acc1) + (acc2 + acc3);
    for (; i < packed_size; i++) {
        dot += table[code[i]];
        table += 256;
    }
    return dot;
}

inline void compute_code_dot_half_byte_lut_batch_4(
        const uint8_t* __restrict code0,
        const uint8_t* __restrict code1,
        const uint8_t* __restrict code2,
        const uint8_t* __restrict code3,
        const std::vector<float>& lut,
        size_t packed_size,
        float& dot0,
        float& dot1,
        float& dot2,
        float& dot3) {
    const float* __restrict table = lut.data();
    dot0 = 0.0f;
    dot1 = 0.0f;
    dot2 = 0.0f;
    dot3 = 0.0f;

    float acc00 = 0.0f;
    float acc01 = 0.0f;
    float acc02 = 0.0f;
    float acc03 = 0.0f;
    float acc10 = 0.0f;
    float acc11 = 0.0f;
    float acc12 = 0.0f;
    float acc13 = 0.0f;
    float acc20 = 0.0f;
    float acc21 = 0.0f;
    float acc22 = 0.0f;
    float acc23 = 0.0f;
    float acc30 = 0.0f;
    float acc31 = 0.0f;
    float acc32 = 0.0f;
    float acc33 = 0.0f;
    size_t i = 0;
    for (; i + 4 <= packed_size; i += 4) {
        const float* table0 = table;
        const float* table1 = table + 32;
        const float* table2 = table + 64;
        const float* table3 = table + 96;
        uint8_t byte0 = code0[i];
        uint8_t byte1 = code1[i];
        uint8_t byte2 = code2[i];
        uint8_t byte3 = code3[i];
        acc00 += table0[byte0 & 0x0f] + table0[16 + (byte0 >> 4)];
        acc10 += table0[byte1 & 0x0f] + table0[16 + (byte1 >> 4)];
        acc20 += table0[byte2 & 0x0f] + table0[16 + (byte2 >> 4)];
        acc30 += table0[byte3 & 0x0f] + table0[16 + (byte3 >> 4)];

        byte0 = code0[i + 1];
        byte1 = code1[i + 1];
        byte2 = code2[i + 1];
        byte3 = code3[i + 1];
        acc01 += table1[byte0 & 0x0f] + table1[16 + (byte0 >> 4)];
        acc11 += table1[byte1 & 0x0f] + table1[16 + (byte1 >> 4)];
        acc21 += table1[byte2 & 0x0f] + table1[16 + (byte2 >> 4)];
        acc31 += table1[byte3 & 0x0f] + table1[16 + (byte3 >> 4)];

        byte0 = code0[i + 2];
        byte1 = code1[i + 2];
        byte2 = code2[i + 2];
        byte3 = code3[i + 2];
        acc02 += table2[byte0 & 0x0f] + table2[16 + (byte0 >> 4)];
        acc12 += table2[byte1 & 0x0f] + table2[16 + (byte1 >> 4)];
        acc22 += table2[byte2 & 0x0f] + table2[16 + (byte2 >> 4)];
        acc32 += table2[byte3 & 0x0f] + table2[16 + (byte3 >> 4)];

        byte0 = code0[i + 3];
        byte1 = code1[i + 3];
        byte2 = code2[i + 3];
        byte3 = code3[i + 3];
        acc03 += table3[byte0 & 0x0f] + table3[16 + (byte0 >> 4)];
        acc13 += table3[byte1 & 0x0f] + table3[16 + (byte1 >> 4)];
        acc23 += table3[byte2 & 0x0f] + table3[16 + (byte2 >> 4)];
        acc33 += table3[byte3 & 0x0f] + table3[16 + (byte3 >> 4)];
        table += 128;
    }
    dot0 = (acc00 + acc01) + (acc02 + acc03);
    dot1 = (acc10 + acc11) + (acc12 + acc13);
    dot2 = (acc20 + acc21) + (acc22 + acc23);
    dot3 = (acc30 + acc31) + (acc32 + acc33);
    for (; i < packed_size; i++) {
        const uint8_t byte0 = code0[i];
        const uint8_t byte1 = code1[i];
        const uint8_t byte2 = code2[i];
        const uint8_t byte3 = code3[i];
        dot0 += table[byte0 & 0x0f] + table[16 + (byte0 >> 4)];
        dot1 += table[byte1 & 0x0f] + table[16 + (byte1 >> 4)];
        dot2 += table[byte2 & 0x0f] + table[16 + (byte2 >> 4)];
        dot3 += table[byte3 & 0x0f] + table[16 + (byte3 >> 4)];
        table += 32;
    }
}

inline void compute_code_dot_half_byte_lut_batch_8(
        const uint8_t* __restrict code0,
        const uint8_t* __restrict code1,
        const uint8_t* __restrict code2,
        const uint8_t* __restrict code3,
        const uint8_t* __restrict code4,
        const uint8_t* __restrict code5,
        const uint8_t* __restrict code6,
        const uint8_t* __restrict code7,
        const std::vector<float>& lut,
        size_t packed_size,
        float& dot0,
        float& dot1,
        float& dot2,
        float& dot3,
        float& dot4,
        float& dot5,
        float& dot6,
        float& dot7) {
    const float* __restrict table = lut.data();
    dot0 = 0.0f;
    dot1 = 0.0f;
    dot2 = 0.0f;
    dot3 = 0.0f;
    dot4 = 0.0f;
    dot5 = 0.0f;
    dot6 = 0.0f;
    dot7 = 0.0f;

    for (size_t i = 0; i < packed_size; i++) {
        const float* table_hi = table + 16;
        uint8_t byte = code0[i];
        dot0 += table[byte & 0x0f] + table_hi[byte >> 4];
        byte = code1[i];
        dot1 += table[byte & 0x0f] + table_hi[byte >> 4];
        byte = code2[i];
        dot2 += table[byte & 0x0f] + table_hi[byte >> 4];
        byte = code3[i];
        dot3 += table[byte & 0x0f] + table_hi[byte >> 4];
        byte = code4[i];
        dot4 += table[byte & 0x0f] + table_hi[byte >> 4];
        byte = code5[i];
        dot5 += table[byte & 0x0f] + table_hi[byte >> 4];
        byte = code6[i];
        dot6 += table[byte & 0x0f] + table_hi[byte >> 4];
        byte = code7[i];
        dot7 += table[byte & 0x0f] + table_hi[byte >> 4];
        table += 32;
    }
}

inline void store_distance_batch_8_scalar(
        MetricType metric_type,
        float query_base,
        const EDENCodeFactors* factors[8],
        const float dots[8],
        float* distances) {
    for (size_t i = 0; i < 8; i++) {
        if (metric_type == MetricType::METRIC_L2) {
            distances[i] = query_base + factors[i]->l2_norm_term -
                    2.0f * factors[i]->scale * dots[i];
        } else {
            distances[i] = query_base + factors[i]->scale * dots[i];
        }
    }
}

template <SIMDLevel SL>
inline void store_distance_batch_8(
        MetricType metric_type,
        float query_base,
        const EDENCodeFactors* factors[8],
        const float dots[8],
        float* distances) {
    constexpr SIMDLevel SL8 = simd256_level_selector<SL>::value;
    if constexpr (SL8 == SIMDLevel::NONE) {
        store_distance_batch_8_scalar(
                metric_type, query_base, factors, dots, distances);
    } else {
        float scales[8];
        float l2_norm_terms[8];
        for (size_t i = 0; i < 8; i++) {
            scales[i] = factors[i]->scale;
            l2_norm_terms[i] = factors[i]->l2_norm_term;
        }

        const simd8float32_tpl<SL8> query_base_v(query_base);
        const simd8float32_tpl<SL8> scales_v(scales);
        const simd8float32_tpl<SL8> dots_v(dots);
        if (metric_type == MetricType::METRIC_L2) {
            const simd8float32_tpl<SL8> l2_norm_terms_v(l2_norm_terms);
            const simd8float32_tpl<SL8> two(2.0f);
            const simd8float32_tpl<SL8> result =
                    query_base_v + l2_norm_terms_v - two * scales_v * dots_v;
            result.storeu(distances);
        } else {
            const simd8float32_tpl<SL8> result =
                    query_base_v + scales_v * dots_v;
            result.storeu(distances);
        }
    }
}

template <SIMDLevel SL>
struct EDENOptimizedDistanceComputer : EDENDistanceComputerBase {
    std::vector<float> code_dot_lut;
    CodeDotLUTKind code_dot_lut_kind = CodeDotLUTKind::None;

    void set_query(const float* x) override {
        set_query_common(x);
        build_code_dot_lut(
                dot_query.data(),
                d,
                nb_bits,
                code_dot_lut,
                code_dot_lut_kind);
    }

    float distance_to_code(const uint8_t* code) final {
        const EDENCodeFactors* factors =
                reinterpret_cast<const EDENCodeFactors*>(code + packed_size);
        const float code_dot_query = code_dot_lut.empty()
                ? compute_code_dot_reference(
                          code, dot_query.data(), d, nb_bits)
                : compute_code_dot_lut(
                          code,
                          code_dot_lut,
                          code_dot_lut_kind,
                          packed_size);
        return distance_from_code_dot(factors, code_dot_query);
    }

    void distances_batch_4(
            idx_t idx0,
            idx_t idx1,
            idx_t idx2,
            idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) final {
        const uint8_t* code0 = codes + idx0 * code_size;
        const uint8_t* code1 = codes + idx1 * code_size;
        const uint8_t* code2 = codes + idx2 * code_size;
        const uint8_t* code3 = codes + idx3 * code_size;

        if (code_dot_lut.empty() ||
            code_dot_lut_kind != CodeDotLUTKind::HalfByte) {
            dis0 = distance_to_code(code0);
            dis1 = distance_to_code(code1);
            dis2 = distance_to_code(code2);
            dis3 = distance_to_code(code3);
            return;
        }

        float dot0;
        float dot1;
        float dot2;
        float dot3;
        compute_code_dot_half_byte_lut_batch_4(
                code0,
                code1,
                code2,
                code3,
                code_dot_lut,
                packed_size,
                dot0,
                dot1,
                dot2,
                dot3);

        dis0 = distance_from_code_dot(
                reinterpret_cast<const EDENCodeFactors*>(code0 + packed_size),
                dot0);
        dis1 = distance_from_code_dot(
                reinterpret_cast<const EDENCodeFactors*>(code1 + packed_size),
                dot1);
        dis2 = distance_from_code_dot(
                reinterpret_cast<const EDENCodeFactors*>(code2 + packed_size),
                dot2);
        dis3 = distance_from_code_dot(
                reinterpret_cast<const EDENCodeFactors*>(code3 + packed_size),
                dot3);
    }

    void consecutive_distances_batch_8(idx_t first, float* distances) final {
        const uint8_t* code[8];
        for (size_t i = 0; i < 8; i++) {
            code[i] = codes + (first + idx_t(i)) * code_size;
        }

        if (code_dot_lut.empty() ||
            code_dot_lut_kind != CodeDotLUTKind::HalfByte) {
            distances_batch_4(
                    first,
                    first + 1,
                    first + 2,
                    first + 3,
                    distances[0],
                    distances[1],
                    distances[2],
                    distances[3]);
            distances_batch_4(
                    first + 4,
                    first + 5,
                    first + 6,
                    first + 7,
                    distances[4],
                    distances[5],
                    distances[6],
                    distances[7]);
            return;
        }

        float dots[8];
        compute_code_dot_half_byte_lut_batch_8(
                code[0],
                code[1],
                code[2],
                code[3],
                code[4],
                code[5],
                code[6],
                code[7],
                code_dot_lut,
                packed_size,
                dots[0],
                dots[1],
                dots[2],
                dots[3],
                dots[4],
                dots[5],
                dots[6],
                dots[7]);

        const EDENCodeFactors* factors[8];
        for (size_t i = 0; i < 8; i++) {
            factors[i] =
                    reinterpret_cast<const EDENCodeFactors*>(
                            code[i] + packed_size);
        }
        store_distance_batch_8<SL>(
                metric_type, query_base, factors, dots, distances);
    }
};

template <SIMDLevel SL>
EDENFlatCodesDistanceComputer* make_optimized_distance_computer(
        MetricType metric_type,
        size_t d,
        size_t nb_bits,
        const float* centroid) {
    auto dc = std::make_unique<EDENOptimizedDistanceComputer<SL>>();
    dc->metric_type = metric_type;
    dc->d = d;
    dc->nb_bits = nb_bits;
    dc->centroid = centroid;
    dc->packed_size = eden_utils::packed_code_size(d, nb_bits);
    return dc.release();
}

} // namespace eden_distance

} // namespace faiss
