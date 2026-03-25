/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) Huawei Technologies Co., Ltd.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_ARM_NEON

#include <arm_neon.h>

#include <faiss/impl/pq_code_distance/pq_code_distance-inl.h>

namespace {

inline float distance_single_code_neon(
        const size_t M,
        const float* sim_table,
        const uint8_t* code) {
    const float* tab = sim_table;
    float result = 0;

    for (size_t m = 0; m < M; m++) {
        result += tab[*(code + m)];
        tab += 256;
    }

    return result;
}

inline void distance_four_codes_neon(
        const size_t M,
        const float* sim_table,
        const uint8_t* __restrict code0,
        const uint8_t* __restrict code1,
        const uint8_t* __restrict code2,
        const uint8_t* __restrict code3,
        float& result0,
        float& result1,
        float& result2,
        float& result3) {
    const float* tab = sim_table;
    float32x4_t res = vdupq_n_f32(0.0f);

    for (size_t m = 0; m < M; m++) {
        const float32x4_t neon_result = {
            tab[*(code0 + m)],
            tab[*(code1 + m)],
            tab[*(code2 + m)],
            tab[*(code3 + m)]};
        tab += 256;
        res = vaddq_f32(res, neon_result);
    }

    float results[4];
    vst1q_f32(results, res);
    result0 = results[0];
    result1 = results[1];
    result2 = results[2];
    result3 = results[3];
}

inline void distance_two_codes_n8(
        const size_t M,
        const float* sim_table,
        const uint8_t* __restrict codes,
        float* result,
        const float dis) {
    const float* tab = sim_table;
    float result0 = dis;
    float result1 = dis;

    for (size_t m = 0; m < M; m++) {
        result0 += tab[*(codes + m)];
        result1 += tab[*(codes + M + m)];
        tab += 256;
    }
    result[0] = result0;
    result[1] = result1;
}

inline void distance_four_codes_n8(
        const size_t M,
        const float* sim_table,
        const uint8_t* __restrict codes,
        float* result,
        const float dis) {
    const float* tab = sim_table;
    float32x4_t res = vdupq_n_f32(dis);

    for (size_t m = 0; m < M; m++) {
        const float32x4_t neon_result = {
            tab[*(codes + m)],
            tab[*(codes + M + m)],
            tab[*(codes + 2 * M + m)],
            tab[*(codes + 3 * M + m)]};
        tab += 256;
        res = vaddq_f32(res, neon_result);
    }
    vst1q_f32(result, res);
}

inline void distance_codes_n8_simd8(
        const size_t M,
        const float* sim_table,
        const uint8_t* __restrict codes,
        float* result,
        const float dis) {
    const float* tab = sim_table;
    float32x4_t neon_result0 = vdupq_n_f32(dis);
    float32x4_t neon_result1 = vdupq_n_f32(dis);
    for (size_t m = 0; m < M; m++) {
        const float32x4_t neon_single_dim0 = {
            tab[*(codes + m)],
            tab[*(codes + M + m)],
            tab[*(codes + 2 * M + m)],
            tab[*(codes + 3 * M + m)]};
        const float32x4_t neon_single_dim1 = {
            tab[*(codes + 4 * M + m)],
            tab[*(codes + 5 * M + m)],
            tab[*(codes + 6 * M + m)],
            tab[*(codes + 7 * M + m)]};
        tab += 256;
        neon_result0 = vaddq_f32(neon_result0, neon_single_dim0);
        neon_result1 = vaddq_f32(neon_result1, neon_single_dim1);
    }
    vst1q_f32(result, neon_result0);
    vst1q_f32(result + 4, neon_result1);
}

} // namespace

namespace faiss {
namespace pq_code_distance {

template <>
float pq_code_distance_single_impl<SIMDLevel::ARM_NEON>(
        size_t M,
        size_t nbits,
        const float* sim_table,
        const uint8_t* code) {
    if (nbits == 8) {
        return distance_single_code_neon(M, sim_table, code);
    }

    // Fallback for non-8bit
    const size_t ksub = 1 << nbits;
    const float* tab = sim_table;
    float result = 0;
    for (size_t m = 0; m < M; m++) {
        result += tab[code[m]];
        tab += ksub;
    }
    return result;
}

template <>
void pq_code_distance_four_impl<SIMDLevel::ARM_NEON>(
        size_t M,
        size_t nbits,
        const float* sim_table,
        const uint8_t* __restrict code0,
        const uint8_t* __restrict code1,
        const uint8_t* __restrict code2,
        const uint8_t* __restrict code3,
        float& result0,
        float& result1,
        float& result2,
        float& result3) {
    if (nbits == 8) {
        distance_four_codes_neon(
                M, sim_table, code0, code1, code2, code3,
                result0, result1, result2, result3);
        return;
    }

    // Fallback for non-8bit
    const size_t ksub = 1 << nbits;
    const float* tab = sim_table;
    result0 = result1 = result2 = result3 = 0;
    for (size_t m = 0; m < M; m++) {
        result0 += tab[code0[m]];
        result1 += tab[code1[m]];
        result2 += tab[code2[m]];
        result3 += tab[code3[m]];
        tab += ksub;
    }
}

void pq_code_distance_batch_neon(
        const size_t M,
        const size_t ncode,
        const uint8_t* codes,
        const float* sim_table,
        float* dis,
        const float dis0) {
    size_t j = 0;
    for (; j + 8 <= ncode; j += 8) {
        distance_codes_n8_simd8(M, sim_table, codes + j * M, dis + j, dis0);
    }
    if (ncode & 4) {
        distance_four_codes_n8(M, sim_table, codes + j * M, dis + j, dis0);
        j += 4;
    }
    if (ncode & 2) {
        distance_two_codes_n8(M, sim_table, codes + j * M, dis + j, dis0);
        j += 2;
    }
    if (ncode & 1) {
        dis[ncode - 1] = distance_single_code_neon(M, sim_table, codes + (ncode - 1) * M) + dis0;
    }
}

template <>
void pq_code_distance_batch_impl<SIMDLevel::ARM_NEON>(
        size_t M,
        size_t nbits,
        size_t ncode,
        const uint8_t* codes,
        const float* sim_table,
        float* dis,
        float dis0) {
    if (nbits == 8) {
        pq_code_distance_batch_neon(M, ncode, codes, sim_table, dis, dis0);
        return;
    }

    // Fallback for non-8bit
    for (size_t i = 0; i < ncode; i++) {
        dis[i] = pq_code_distance_single_impl<SIMDLevel::ARM_NEON>(
                M, nbits, sim_table, codes + i * M) + dis0;
    }
}

void pq_code_distance_batch_by_idx_neon(
        const size_t M,
        const size_t ncode,
        const uint8_t* codes,
        const float* sim_table,
        float* dis,
        const float dis0,
        const size_t* idx) {
    for (size_t j = 0; j < ncode; j++) {
        dis[j] = distance_single_code_neon(M, sim_table, codes + idx[j] * M) + dis0;
    }
}

} // namespace pq_code_distance
} // namespace faiss

#endif // COMPILE_SIMD_ARM_NEON
