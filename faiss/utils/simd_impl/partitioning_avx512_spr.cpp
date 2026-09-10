/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/utils/simd_impl/partitioning_avx512_spr.h>

#include <algorithm>
#include <cassert>
#include <cstdint>

#include <immintrin.h>

namespace faiss {
namespace partitioning_avx512_spr {

namespace {

template <bool is_max>
void count_lt_and_eq_impl(
        const uint16_t* vals,
        int n,
        uint16_t thresh,
        size_t& n_lt,
        size_t& n_eq) {
    n_lt = 0;
    n_eq = 0;

    size_t local_n_lt = 0;
    size_t local_n_eq = 0;

    int i = 0;
    constexpr int VEC_SIZE = 32;
    constexpr int cmp_op = is_max ? _MM_CMPINT_LT : _MM_CMPINT_GT;

    __m512i v_thresh = _mm512_set1_epi16(thresh);

    for (; i + VEC_SIZE <= n; i += VEC_SIZE) {
        __m512i v_vals = _mm512_loadu_si512(vals + i);

        __mmask32 k_lt = _mm512_cmp_epu16_mask(v_vals, v_thresh, cmp_op);
        __mmask32 k_eq = _mm512_cmp_epu16_mask(v_vals, v_thresh, _MM_CMPINT_EQ);
        __mmask32 k_eq_only = k_eq & ~k_lt;

        local_n_lt += _mm_popcnt_u32(k_lt);
        local_n_eq += _mm_popcnt_u32(k_eq_only);
    }

    for (; i < n; i++) {
        uint16_t v = vals[i];
        if (is_max ? (v < thresh) : (v > thresh)) {
            local_n_lt++;
        } else if (v == thresh) {
            local_n_eq++;
        }
    }

    n_lt = local_n_lt;
    n_eq = local_n_eq;
}

template <bool is_max>
int simd_compress_array_int_impl(
        uint16_t* vals,
        int* ids,
        size_t n,
        uint16_t thresh,
        int n_eq) {
    constexpr int cmp_op = is_max ? _MM_CMPINT_LT : _MM_CMPINT_GT;

    int wp = 0;
    size_t i = 0;

    constexpr int VEC_SIZE = 16;
    __m512i v_thresh = _mm512_set1_epi32(static_cast<int32_t>(thresh));

    for (; i + VEC_SIZE <= n; i += VEC_SIZE) {
        __m256i v_vals_u16 = _mm256_loadu_si256((__m256i*)(vals + i));
        __m512i v_ids_s32 = _mm512_loadu_si512(ids + i);
        __m512i v_vals_s32 = _mm512_cvtepu16_epi32(v_vals_u16);

        __mmask16 k_primary =
                _mm512_cmp_epi32_mask(v_vals_s32, v_thresh, cmp_op);
        __mmask16 k_equal_to_add = 0;

        int num_to_take = 0;

        if (n_eq > 0) {
            __mmask16 k_equal =
                    _mm512_cmp_epi32_mask(v_vals_s32, v_thresh, _MM_CMPINT_EQ);
            __mmask16 k_equal_only = k_equal & ~k_primary;

            int num_eq_found = _mm_popcnt_u32(k_equal_only);
            num_to_take = std::min(n_eq, num_eq_found);

            k_equal_to_add =
                    _pdep_u32((uint32_t(1) << num_to_take) - 1, k_equal_only);
        }

        __mmask16 k_final = k_primary | k_equal_to_add;
        _mm256_mask_compressstoreu_epi16(vals + wp, k_final, v_vals_u16);
        _mm512_mask_compressstoreu_epi32(ids + wp, k_final, v_ids_s32);

        wp += _mm_popcnt_u32(k_final);
        n_eq -= num_to_take;
    }

    for (; i < n; i++) {
        if (is_max ? (thresh > vals[i]) : (thresh < vals[i])) {
            vals[wp] = vals[i];
            ids[wp] = ids[i];
            wp++;
        } else if (n_eq > 0 && vals[i] == thresh) {
            vals[wp] = vals[i];
            ids[wp] = ids[i];
            wp++;
            n_eq--;
        }
    }

    assert(n_eq == 0);
    return wp;
}

template <bool is_max>
int simd_compress_array_int64_impl(
        uint16_t* vals,
        int64_t* ids,
        size_t n,
        uint16_t thresh,
        int n_eq) {
    constexpr int cmp_op = is_max ? _MM_CMPINT_LT : _MM_CMPINT_GT;

    int wp = 0;
    size_t i = 0;

    constexpr int VEC_SIZE = 8;
    __m512i v_thresh = _mm512_set1_epi64(static_cast<int64_t>(thresh));

    for (; i + VEC_SIZE <= n; i += VEC_SIZE) {
        __m128i v_vals_u16 = _mm_loadu_si128((__m128i*)(vals + i));
        __m512i v_ids_s64 = _mm512_loadu_si512(ids + i);
        __m512i v_vals_s64 = _mm512_cvtepu16_epi64(v_vals_u16);

        __mmask8 k_primary =
                _mm512_cmp_epi64_mask(v_vals_s64, v_thresh, cmp_op);
        __mmask8 k_equal_to_add = 0;

        int num_to_take = 0;

        if (n_eq > 0) {
            __mmask8 k_equal =
                    _mm512_cmp_epi64_mask(v_vals_s64, v_thresh, _MM_CMPINT_EQ);
            __mmask8 k_equal_only = k_equal & ~k_primary;

            int num_eq_found = _mm_popcnt_u32(k_equal_only);
            num_to_take = std::min(n_eq, num_eq_found);

            k_equal_to_add =
                    _pdep_u32((uint32_t(1) << num_to_take) - 1, k_equal_only);
        }

        __mmask8 k_final = k_primary | k_equal_to_add;
        _mm_mask_compressstoreu_epi16(vals + wp, k_final, v_vals_u16);
        _mm512_mask_compressstoreu_epi64(ids + wp, k_final, v_ids_s64);

        wp += _mm_popcnt_u32(k_final);
        n_eq -= num_to_take;
    }

    for (; i < n; i++) {
        if (is_max ? (thresh > vals[i]) : (thresh < vals[i])) {
            vals[wp] = vals[i];
            ids[wp] = ids[i];
            wp++;
        } else if (n_eq > 0 && vals[i] == thresh) {
            vals[wp] = vals[i];
            ids[wp] = ids[i];
            wp++;
            n_eq--;
        }
    }

    assert(n_eq == 0);
    return wp;
}

} // anonymous namespace

// Explicit instantiations for CMax (is_max = true) and CMin (is_max = false)

void count_lt_and_eq_max_avx512(
        const uint16_t* vals,
        int n,
        uint16_t thresh,
        size_t& n_lt,
        size_t& n_eq) {
    count_lt_and_eq_impl<true>(vals, n, thresh, n_lt, n_eq);
}

void count_lt_and_eq_min_avx512(
        const uint16_t* vals,
        int n,
        uint16_t thresh,
        size_t& n_lt,
        size_t& n_eq) {
    count_lt_and_eq_impl<false>(vals, n, thresh, n_lt, n_eq);
}

int simd_compress_array_max_int_avx512(
        uint16_t* vals,
        int* ids,
        size_t n,
        uint16_t thresh,
        int n_eq) {
    return simd_compress_array_int_impl<true>(vals, ids, n, thresh, n_eq);
}

int simd_compress_array_min_int_avx512(
        uint16_t* vals,
        int* ids,
        size_t n,
        uint16_t thresh,
        int n_eq) {
    return simd_compress_array_int_impl<false>(vals, ids, n, thresh, n_eq);
}

int simd_compress_array_max_int64_avx512(
        uint16_t* vals,
        int64_t* ids,
        size_t n,
        uint16_t thresh,
        int n_eq) {
    return simd_compress_array_int64_impl<true>(vals, ids, n, thresh, n_eq);
}

int simd_compress_array_min_int64_avx512(
        uint16_t* vals,
        int64_t* ids,
        size_t n,
        uint16_t thresh,
        int n_eq) {
    return simd_compress_array_int64_impl<false>(vals, ids, n, thresh, n_eq);
}

} // namespace partitioning_avx512_spr
} // namespace faiss
