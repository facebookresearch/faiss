/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/EDENQuantizer.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/simd_dispatch.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/hamming.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

namespace faiss {

namespace eden_distance {

namespace {

enum class CodeDotLUTKind {
    None,
    Byte,
    HalfByte,
};

using CodeDotLUTBatch8Fn = void (*)(
        const uint8_t* const code[8],
        const float* lut,
        int lut_kind,
        size_t packed_size,
        float dots[8]);

using CodeDotLUTBatch16Fn = void (*)(
        const uint8_t* const code[16],
        const float* lut,
        int lut_kind,
        size_t packed_size,
        float dots[16]);

float compute_code_dot_reference(
        const uint8_t* code,
        const float* query,
        size_t d,
        size_t nb_bits,
        const float* scalar_centroids) {
    BitstringReader reader(code, eden_utils::packed_code_size(d, nb_bits));
    float dot = 0.0f;
    for (size_t i = 0; i < d; i++) {
        dot += query[i] * scalar_centroids[reader.read(nb_bits)];
    }
    return dot;
}

bool supports_byte_lut(size_t nb_bits) {
    return nb_bits == 1 || nb_bits == 2 || nb_bits == 4;
}

bool use_half_byte_lut(size_t nb_bits, size_t num_bytes) {
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

void build_code_dot_lut(
        const float* query,
        size_t d,
        size_t nb_bits,
        const float* scalar_centroids,
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
                    const float c0 =
                            scalar_centroids[half_byte_value & 0x01];
                    const float c1 =
                            scalar_centroids[(half_byte_value >> 1) & 0x01];
                    const float c2 =
                            scalar_centroids[(half_byte_value >> 2) & 0x01];
                    const float c3 =
                            scalar_centroids[(half_byte_value >> 3) & 0x01];
                    table[half_byte_value] =
                            ((q0 * c0 + q1 * c1) + q2 * c2) + q3 * c3;
                    table[16 + half_byte_value] =
                            ((q4 * c0 + q5 * c1) + q6 * c2) + q7 * c3;
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
                    const float c0 =
                            scalar_centroids[half_byte_value & 0x03];
                    const float c1 =
                            scalar_centroids[(half_byte_value >> 2) & 0x03];
                    table[half_byte_value] = q0 * c0 + q1 * c1;
                    table[16 + half_byte_value] = q2 * c0 + q3 * c1;
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
                    const float centroid = scalar_centroids[assignment];
                    table[assignment] = q0 * centroid;
                    table[16 + assignment] = q1 * centroid;
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
                        low_dot +=
                                query[dim] * scalar_centroids[assignment];
                    }
                    dim += values_per_half_byte;
                    if (dim < d) {
                        high_dot +=
                                query[dim] * scalar_centroids[assignment];
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
                dot += query[dim] * scalar_centroids[assignment];
            }
            table[byte_value] = dot;
        }
    }
}

float compute_code_dot_lut(
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

void compute_code_dot_lut_batch_8_scalar(
        const uint8_t* const code[8],
        const float* __restrict table,
        int lut_kind,
        size_t packed_size,
        float dots[8]) {
    std::fill(dots, dots + 8, 0.0f);
    if (static_cast<CodeDotLUTKind>(lut_kind) == CodeDotLUTKind::HalfByte) {
        for (size_t i = 0; i < packed_size; i++) {
            for (size_t j = 0; j < 8; j++) {
                const uint8_t byte = code[j][i];
                dots[j] += table[byte & 0x0f] + table[16 + (byte >> 4)];
            }
            table += 32;
        }
    } else {
        for (size_t i = 0; i < packed_size; i++) {
            for (size_t j = 0; j < 8; j++) {
                dots[j] += table[code[j][i]];
            }
            table += 256;
        }
    }
}

struct EDENDistanceComputerBase : EDENFlatCodesDistanceComputer {
    size_t d = 0;
    size_t nb_bits = 1;
    const float* scalar_centroids = nullptr;
    const float* centroid = nullptr;
    MetricType metric_type = MetricType::METRIC_L2;

    size_t packed_size = 0;
    std::vector<float> dot_query;
    float query_base = 0.0f;

    float symmetric_dis(idx_t /*i*/, idx_t /*j*/) override {
        FAISS_THROW_MSG("Not implemented");
    }

    void set_query_common(const float* x) {
        q = x;
        FAISS_ASSERT(x != nullptr);

        dot_query.resize(d);
        if (metric_type == MetricType::METRIC_L2) {
            query_base = centroid ? fvec_L2sqr(x, centroid, d)
                                  : fvec_norm_L2sqr(x, d);
            for (size_t i = 0; i < d; i++) {
                dot_query[i] = x[i] - (centroid ? centroid[i] : 0.0f);
            }
        } else if (metric_type == MetricType::METRIC_INNER_PRODUCT) {
            query_base = centroid ? fvec_inner_product(x, centroid, d) : 0.0f;
            memcpy(dot_query.data(), x, d * sizeof(float));
        } else {
            FAISS_THROW_MSG("EDEN supports only L2 and inner-product metrics");
        }
    }

    float distance_from_code_dot(
            const EDENCodeFactors* factors,
            float code_dot_query) const {
        if (metric_type == MetricType::METRIC_L2) {
            return query_base + factors->l2_norm_term -
                    2.0f * factors->scale * code_dot_query;
        }
        return query_base + factors->scale * code_dot_query;
    }
};

struct EDENReferenceDistanceComputer : EDENDistanceComputerBase {
    void set_query(const float* x) override {
        set_query_common(x);
    }

    float distance_to_code(const uint8_t* code) final {
        const EDENCodeFactors* factors =
                reinterpret_cast<const EDENCodeFactors*>(code + packed_size);
        const float code_dot_query = compute_code_dot_reference(
                code, dot_query.data(), d, nb_bits, scalar_centroids);
        return distance_from_code_dot(factors, code_dot_query);
    }
};

struct EDENOptimizedDistanceComputer : EDENDistanceComputerBase {
    std::vector<float> code_dot_lut;
    CodeDotLUTKind code_dot_lut_kind = CodeDotLUTKind::None;
    CodeDotLUTBatch8Fn code_dot_batch_8 = compute_code_dot_lut_batch_8_scalar;
    CodeDotLUTBatch16Fn code_dot_batch_16 = nullptr;

    void set_query(const float* x) override {
        set_query_common(x);
        build_code_dot_lut(
                dot_query.data(),
                d,
                nb_bits,
                scalar_centroids,
                code_dot_lut,
                code_dot_lut_kind);
    }

    float distance_to_code(const uint8_t* code) final {
        const EDENCodeFactors* factors =
                reinterpret_cast<const EDENCodeFactors*>(code + packed_size);
        const float code_dot_query = code_dot_lut.empty()
                ? compute_code_dot_reference(
                          code,
                          dot_query.data(),
                          d,
                          nb_bits,
                          scalar_centroids)
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

        if (code_dot_lut.empty()) {
            dis0 = distance_to_code(code0);
            dis1 = distance_to_code(code1);
            dis2 = distance_to_code(code2);
            dis3 = distance_to_code(code3);
            return;
        }

        const uint8_t* code_batch[8] = {
                code0,
                code1,
                code2,
                code3,
                code0,
                code1,
                code2,
                code3};
        float dots[8];
        code_dot_batch_8(
                code_batch,
                code_dot_lut.data(),
                static_cast<int>(code_dot_lut_kind),
                packed_size,
                dots);

        dis0 = distance_from_code_dot(
                reinterpret_cast<const EDENCodeFactors*>(code0 + packed_size),
                dots[0]);
        dis1 = distance_from_code_dot(
                reinterpret_cast<const EDENCodeFactors*>(code1 + packed_size),
                dots[1]);
        dis2 = distance_from_code_dot(
                reinterpret_cast<const EDENCodeFactors*>(code2 + packed_size),
                dots[2]);
        dis3 = distance_from_code_dot(
                reinterpret_cast<const EDENCodeFactors*>(code3 + packed_size),
                dots[3]);
    }

    void consecutive_distances_batch_8(idx_t first, float* distances) final {
        const uint8_t* code[8];
        for (size_t i = 0; i < 8; i++) {
            code[i] = codes + (first + idx_t(i)) * code_size;
        }

        if (code_dot_lut.empty()) {
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
        code_dot_batch_8(
                code,
                code_dot_lut.data(),
                static_cast<int>(code_dot_lut_kind),
                packed_size,
                dots);

        for (size_t i = 0; i < 8; i++) {
            const EDENCodeFactors* factors =
                    reinterpret_cast<const EDENCodeFactors*>(
                            code[i] + packed_size);
            distances[i] = distance_from_code_dot(factors, dots[i]);
        }
    }

    void consecutive_distances_batch_16(idx_t first, float* distances) final {
        if (code_dot_lut.empty() || code_dot_batch_16 == nullptr) {
            consecutive_distances_batch_8(first, distances);
            consecutive_distances_batch_8(first + 8, distances + 8);
            return;
        }

        const uint8_t* code[16];
        for (size_t i = 0; i < 16; i++) {
            code[i] = codes + (first + idx_t(i)) * code_size;
        }

        float dots[16];
        code_dot_batch_16(
                code,
                code_dot_lut.data(),
                static_cast<int>(code_dot_lut_kind),
                packed_size,
                dots);

        for (size_t i = 0; i < 16; i++) {
            const EDENCodeFactors* factors =
                    reinterpret_cast<const EDENCodeFactors*>(
                            code[i] + packed_size);
            distances[i] = distance_from_code_dot(factors, dots[i]);
        }
    }
};

EDENFlatCodesDistanceComputer* make_reference_distance_computer(
        MetricType metric_type,
        size_t d,
        size_t nb_bits,
        const float* scalar_centroids,
        const float* centroid) {
    auto dc = std::make_unique<EDENReferenceDistanceComputer>();
    dc->metric_type = metric_type;
    dc->d = d;
    dc->nb_bits = nb_bits;
    dc->scalar_centroids = scalar_centroids;
    dc->centroid = centroid;
    dc->packed_size = eden_utils::packed_code_size(d, nb_bits);
    return dc.release();
}

EDENFlatCodesDistanceComputer* make_optimized_distance_computer(
        MetricType metric_type,
        size_t d,
        size_t nb_bits,
        const float* scalar_centroids,
        const float* centroid,
        CodeDotLUTBatch8Fn code_dot_batch_8,
        CodeDotLUTBatch16Fn code_dot_batch_16 = nullptr) {
    auto dc = std::make_unique<EDENOptimizedDistanceComputer>();
    dc->metric_type = metric_type;
    dc->d = d;
    dc->nb_bits = nb_bits;
    dc->scalar_centroids = scalar_centroids;
    dc->centroid = centroid;
    dc->packed_size = eden_utils::packed_code_size(d, nb_bits);
    dc->code_dot_batch_8 = code_dot_batch_8;
    dc->code_dot_batch_16 = code_dot_batch_16;
    return dc.release();
}

} // namespace

#ifdef COMPILE_SIMD_AVX2
void compute_code_dot_lut_batch_8_avx2(
        const uint8_t* const code[8],
        const float* lut,
        int lut_kind,
        size_t packed_size,
        float dots[8]);
#endif

#ifdef COMPILE_SIMD_AVX512
void compute_code_dot_lut_batch_8_avx512(
        const uint8_t* const code[8],
        const float* lut,
        int lut_kind,
        size_t packed_size,
        float dots[8]);

void compute_code_dot_lut_batch_16_avx512(
        const uint8_t* const code[16],
        const float* lut,
        int lut_kind,
        size_t packed_size,
        float dots[16]);
#endif

template <SIMDLevel SL>
EDENFlatCodesDistanceComputer* make_distance_computer_for_level(
        MetricType metric_type,
        size_t d,
        size_t nb_bits,
        const float* scalar_centroids,
        const float* centroid) {
    return make_reference_distance_computer(
            metric_type, d, nb_bits, scalar_centroids, centroid);
}

#ifdef COMPILE_SIMD_AVX2
template <>
EDENFlatCodesDistanceComputer* make_distance_computer_for_level<
        SIMDLevel::AVX2>(
        MetricType metric_type,
        size_t d,
        size_t nb_bits,
        const float* scalar_centroids,
        const float* centroid) {
    return make_optimized_distance_computer(
            metric_type,
            d,
            nb_bits,
            scalar_centroids,
            centroid,
            compute_code_dot_lut_batch_8_avx2);
}
#endif

#ifdef COMPILE_SIMD_AVX512
template <>
EDENFlatCodesDistanceComputer* make_distance_computer_for_level<
        SIMDLevel::AVX512>(
        MetricType metric_type,
        size_t d,
        size_t nb_bits,
        const float* scalar_centroids,
        const float* centroid) {
    return make_optimized_distance_computer(
            metric_type,
            d,
            nb_bits,
            scalar_centroids,
            centroid,
            compute_code_dot_lut_batch_8_avx512,
            compute_code_dot_lut_batch_16_avx512);
}
#endif

#ifdef COMPILE_SIMD_ARM_NEON
template <>
EDENFlatCodesDistanceComputer* make_distance_computer_for_level<
        SIMDLevel::ARM_NEON>(
        MetricType metric_type,
        size_t d,
        size_t nb_bits,
        const float* scalar_centroids,
        const float* centroid) {
    return make_optimized_distance_computer(
            metric_type,
            d,
            nb_bits,
            scalar_centroids,
            centroid,
            compute_code_dot_lut_batch_8_scalar);
}
#endif

#ifdef COMPILE_SIMD_RISCV_RVV
template <>
EDENFlatCodesDistanceComputer* make_distance_computer_for_level<
        SIMDLevel::RISCV_RVV>(
        MetricType metric_type,
        size_t d,
        size_t nb_bits,
        const float* scalar_centroids,
        const float* centroid) {
    return make_optimized_distance_computer(
            metric_type,
            d,
            nb_bits,
            scalar_centroids,
            centroid,
            compute_code_dot_lut_batch_8_scalar);
}
#endif

} // namespace eden_distance

namespace eden_utils {

ScalarQuantizer::QuantizerType quantizer_type_for_bits(size_t nb_bits) {
    switch (nb_bits) {
        case 1:
            return ScalarQuantizer::QT_1bit_eden;
        case 2:
            return ScalarQuantizer::QT_2bit_eden;
        case 3:
            return ScalarQuantizer::QT_3bit_eden;
        case 4:
            return ScalarQuantizer::QT_4bit_eden;
        case 5:
            return ScalarQuantizer::QT_5bit_eden;
        case 6:
            return ScalarQuantizer::QT_6bit_eden;
        case 7:
            return ScalarQuantizer::QT_7bit_eden;
        case 8:
            return ScalarQuantizer::QT_8bit_eden;
        default:
            FAISS_THROW_MSG("EDEN nb_bits must be in [1, 8]");
    }
}

bool is_eden_quantizer_type(ScalarQuantizer::QuantizerType qtype) {
    return qtype >= ScalarQuantizer::QT_1bit_eden &&
            qtype <= ScalarQuantizer::QT_8bit_eden;
}

size_t nb_bits_for_qtype(ScalarQuantizer::QuantizerType qtype) {
    switch (qtype) {
        case ScalarQuantizer::QT_1bit_eden:
            return 1;
        case ScalarQuantizer::QT_2bit_eden:
            return 2;
        case ScalarQuantizer::QT_3bit_eden:
            return 3;
        case ScalarQuantizer::QT_4bit_eden:
            return 4;
        case ScalarQuantizer::QT_5bit_eden:
            return 5;
        case ScalarQuantizer::QT_6bit_eden:
            return 6;
        case ScalarQuantizer::QT_7bit_eden:
            return 7;
        case ScalarQuantizer::QT_8bit_eden:
            return 8;
        default:
            FAISS_THROW_MSG("expected an EDEN ScalarQuantizer qtype");
    }
}

size_t packed_code_size(size_t d, size_t nb_bits) {
    FAISS_THROW_IF_NOT_MSG(
            nb_bits >= 1 && nb_bits <= 8, "EDEN nb_bits must be in [1, 8]");
    return (d * nb_bits + 7) / 8;
}

size_t code_size(size_t d, size_t nb_bits) {
    return packed_code_size(d, nb_bits) + sizeof(EDENCodeFactors);
}

uint8_t extract_code(const uint8_t* codes, size_t index, size_t nb_bits) {
    if (nb_bits == 8) {
        return codes[index];
    }

    const size_t bit_pos = index * nb_bits;
    BitstringReader reader(codes, (bit_pos + nb_bits + 7) / 8);
    reader.i = bit_pos;
    return static_cast<uint8_t>(reader.read(nb_bits));
}

void compute_codes(
        const ScalarQuantizer& sq,
        MetricType metric_type,
        EDENScaleType scale_type,
        const float* x,
        uint8_t* codes,
        size_t n,
        const float* centroid) {
    FAISS_ASSERT(x != nullptr);
    FAISS_ASSERT(codes != nullptr);
    FAISS_THROW_IF_NOT_MSG(
            is_eden_quantizer_type(sq.qtype),
            "expected an EDEN ScalarQuantizer qtype");
    FAISS_THROW_IF_NOT_MSG(
            metric_type == MetricType::METRIC_L2 ||
                    metric_type == MetricType::METRIC_INNER_PRODUCT,
            "EDEN supports only L2 and inner-product metrics");
    FAISS_THROW_IF_NOT_MSG(
            scale_type == EDENScaleType_UNBIASED ||
                    scale_type == EDENScaleType_BIASED,
            "invalid EDEN scale type");

    if (n == 0) {
        return;
    }

    const size_t d = sq.d;
    const size_t nb_bits = sq.bits;
    const size_t packed_size = sq.code_size;
    const size_t full_code_size = code_size(d, nb_bits);
    const float sqrt_d = std::sqrt(static_cast<float>(d));

#pragma omp parallel if (n > 1000)
    {
        std::unique_ptr<ScalarQuantizer::SQuantizer> squant(
                sq.select_quantizer());
        std::vector<float> normalized(d);
        std::vector<float> decoded(d);

#pragma omp for
        for (int64_t i = 0; i < static_cast<int64_t>(n); i++) {
            const float* xi = x + i * d;
            uint8_t* code = codes + i * full_code_size;
            memset(code, 0, full_code_size);

            float norm_sqr = 0.0f;
            for (size_t j = 0; j < d; j++) {
                const float c = centroid ? centroid[j] : 0.0f;
                const float r = xi[j] - c;
                norm_sqr += r * r;
            }

            EDENCodeFactors* factors =
                    reinterpret_cast<EDENCodeFactors*>(code + packed_size);
            if (norm_sqr <= std::numeric_limits<float>::epsilon()) {
                factors->l2_norm_term = 0.0f;
                factors->scale = 0.0f;
                continue;
            }

            const float norm = std::sqrt(norm_sqr);
            const float inv_norm = 1.0f / norm;
            for (size_t j = 0; j < d; j++) {
                const float c = centroid ? centroid[j] : 0.0f;
                normalized[j] = (xi[j] - c) * sqrt_d * inv_norm;
            }

            squant->encode_vector(normalized.data(), code);
            squant->decode_vector(code, decoded.data());

            double code_norm_sqr = 0.0;
            double code_residual_ip = 0.0;
            for (size_t j = 0; j < d; j++) {
                const float c = centroid ? centroid[j] : 0.0f;
                const float r = xi[j] - c;
                const float q = decoded[j];
                code_norm_sqr += double(q) * q;
                code_residual_ip += double(q) * r;
            }

            float scale = 0.0f;
            float l2_norm_term = 0.0f;
            // Unbiased EDEN uses ||r||^2 / <q, r>. The biased scale follows
            // DRIVE (NeurIPS 2021): <q, r> / ||q||^2.
            if (scale_type == EDENScaleType_BIASED) {
                scale = static_cast<float>(code_residual_ip / code_norm_sqr);
                l2_norm_term = static_cast<float>(
                        double(scale) * scale * code_norm_sqr);
            } else {
                scale = static_cast<float>(double(norm_sqr) / code_residual_ip);
                l2_norm_term = norm_sqr;
            }
            if (!std::isfinite(scale)) {
                scale = 0.0f;
                l2_norm_term = 0.0f;
            }

            factors->scale = scale;
            factors->l2_norm_term = l2_norm_term;
        }
    }
}

void decode(
        const ScalarQuantizer& sq,
        const uint8_t* codes,
        float* x,
        size_t n,
        const float* centroid) {
    FAISS_ASSERT(codes != nullptr);
    FAISS_ASSERT(x != nullptr);
    FAISS_THROW_IF_NOT_MSG(
            is_eden_quantizer_type(sq.qtype),
            "expected an EDEN ScalarQuantizer qtype");

    const size_t d = sq.d;
    const size_t packed_size = sq.code_size;
    const size_t full_code_size = code_size(d, sq.bits);

#pragma omp parallel if (n > 1000)
    {
        std::unique_ptr<ScalarQuantizer::SQuantizer> squant(
                sq.select_quantizer());
        std::vector<float> decoded(d);

#pragma omp for
        for (int64_t i = 0; i < static_cast<int64_t>(n); i++) {
            const uint8_t* code = codes + i * full_code_size;
            const EDENCodeFactors* factors = reinterpret_cast<
                    const EDENCodeFactors*>(code + packed_size);
            float* xi = x + i * d;

            squant->decode_vector(code, decoded.data());
            for (size_t j = 0; j < d; j++) {
                const float c = centroid ? centroid[j] : 0.0f;
                xi[j] = c + factors->scale * decoded[j];
            }
        }
    }
}

} // namespace eden_utils

namespace eden_utils {

EDENFlatCodesDistanceComputer* get_distance_computer(
        const ScalarQuantizer& sq,
        MetricType metric_type,
        const float* centroid) {
    FAISS_THROW_IF_NOT_MSG(
            is_eden_quantizer_type(sq.qtype),
            "expected an EDEN ScalarQuantizer qtype");
    const size_t centroid_count = size_t{1} << sq.bits;
    FAISS_THROW_IF_NOT(sq.trained.size() >= centroid_count);
    const float* scalar_centroids = sq.trained.data();
    return with_simd_level(
            [&]<SIMDLevel SL>() -> EDENFlatCodesDistanceComputer* {
                return eden_distance::make_distance_computer_for_level<SL>(
                        metric_type,
                        sq.d,
                        sq.bits,
                        scalar_centroids,
                        centroid);
            });
}

} // namespace eden_utils

} // namespace faiss
