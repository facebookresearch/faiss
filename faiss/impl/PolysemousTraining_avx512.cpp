/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/impl/PolysemousTraining.h>
#include <faiss/impl/PolysemousTraining_avx512.h>

#include <immintrin.h>

namespace faiss {

static inline int hamming_dis(uint64_t a, uint64_t b) {
    return __builtin_popcountl(a ^ b);
}

static inline double sqr(double x) {
    return x * x;
}

static inline __m512i popcnt_512(__m512i v) {
#ifdef __AVX512VPOPCNTDQ__
    return _mm512_popcnt_epi64(v);
#else
    const __m128i nibble_popcount =
            _mm_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    const __m512i lookup = _mm512_broadcast_i32x4(nibble_popcount);

    const __m512i low_mask = _mm512_set1_epi8(0x0f);
    const __m512i lo = _mm512_and_si512(v, low_mask);
    const __m512i hi = _mm512_and_si512(_mm512_srli_epi16(v, 4), low_mask);

    const __m512i popcnt_lo = _mm512_shuffle_epi8(lookup, lo);
    const __m512i popcnt_hi = _mm512_shuffle_epi8(lookup, hi);
    const __m512i popcnt_bytes = _mm512_add_epi8(popcnt_lo, popcnt_hi);

    return _mm512_sad_epu8(popcnt_bytes, _mm512_setzero_si512());
#endif
}

namespace polysemous_avx512 {

double hamming_compute_cost_avx512(
        int n,
        const int* perm,
        const double* target_dis,
        const double* weights) {
    double total_cost = 0.0;
    for (int i = 0; i < n; i++) {
        __m512d cost_vec = _mm512_setzero_pd();
        const __m512i perm_i_vec = _mm512_set1_epi64(perm[i]);
        const int bro = i * n;
        int j = 0;
        for (; j <= n - 8; j += 8) {
            const __m512d wanted_vec = _mm512_loadu_pd(&target_dis[bro + j]);
            const __m512d w_vec = _mm512_loadu_pd(&weights[bro + j]);
            const __m256i pj32 = _mm256_loadu_si256((__m256i const*)&perm[j]);
            const __m512i pj64 = _mm512_cvtepi32_epi64(pj32);
            const __m512i xor_res = _mm512_xor_si512(perm_i_vec, pj64);
            const __m512d actual_vec = _mm512_cvtepi64_pd(popcnt_512(xor_res));
            const __m512d diff = _mm512_sub_pd(wanted_vec, actual_vec);
            cost_vec =
                    _mm512_fmadd_pd(w_vec, _mm512_mul_pd(diff, diff), cost_vec);
        }
        total_cost += _mm512_reduce_add_pd(cost_vec);
        for (; j < n; j++) {
            double wanted = target_dis[bro + j];
            double w = weights[bro + j];
            double actual = hamming_dis(perm[i], perm[j]);
            total_cost += w * sqr(wanted - actual);
        }
    }
    return total_cost;
}

double hamming_cost_update_avx512(
        int n,
        const int* perm,
        int iw,
        int jw,
        const double* target_dis,
        const double* weights) {
    double delta_cost_scalar = 0;
    const __m512i v_idx_base = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
    __m512d delta_cost_vec = _mm512_setzero_pd();

    auto process_row = [&](int row, int old_pi, int new_pi) {
        const int bro = row * n;
        const __m512i v_old = _mm512_set1_epi64(old_pi);
        const __m512i v_new = _mm512_set1_epi64(new_pi);
        int j = 0;
        for (; j <= n - 8; j += 8) {
            __m512d wv = _mm512_loadu_pd(&target_dis[bro + j]);
            __m512d ww = _mm512_loadu_pd(&weights[bro + j]);
            __m256i pj32 = _mm256_loadu_si256((__m256i const*)&perm[j]);
            __m512i pjv = _mm512_cvtepi32_epi64(pj32);
            __m512d av = _mm512_cvtepi64_pd(
                    popcnt_512(_mm512_xor_si512(v_old, pjv)));
            __m512d to = _mm512_sub_pd(wv, av);
            to = _mm512_mul_pd(to, to);
            delta_cost_vec = _mm512_fnmadd_pd(ww, to, delta_cost_vec);

            __m512i ji = _mm512_add_epi64(_mm512_set1_epi64(j), v_idx_base);
            __mmask8 miw = _mm512_cmpeq_epi64_mask(ji, _mm512_set1_epi64(iw));
            __mmask8 mjw = _mm512_cmpeq_epi64_mask(ji, _mm512_set1_epi64(jw));
            __m512i pnj = _mm512_mask_blend_epi64(
                    mjw, pjv, _mm512_set1_epi64(perm[iw]));
            pnj = _mm512_mask_blend_epi64(
                    miw, pnj, _mm512_set1_epi64(perm[jw]));
            __m512d nav = _mm512_cvtepi64_pd(
                    popcnt_512(_mm512_xor_si512(v_new, pnj)));
            __m512d tn = _mm512_sub_pd(wv, nav);
            tn = _mm512_mul_pd(tn, tn);
            delta_cost_vec = _mm512_fmadd_pd(ww, tn, delta_cost_vec);
        }
        for (; j < n; j++) {
            double wanted = target_dis[bro + j];
            double w = weights[bro + j];
            double actual = hamming_dis(old_pi, perm[j]);
            delta_cost_scalar -= w * sqr(wanted - actual);
            double new_actual = hamming_dis(
                    new_pi,
                    perm[j == iw           ? jw
                                 : j == jw ? iw
                                           : j]);
            delta_cost_scalar += w * sqr(wanted - new_actual);
        }
    };
    process_row(iw, perm[iw], perm[jw]);
    process_row(jw, perm[jw], perm[iw]);

    for (int i = 0; i < n; ++i) {
        if (i == iw || i == jw)
            continue;
        int j = iw;
        {
            double wanted = target_dis[i * n + j];
            double w = weights[i * n + j];
            delta_cost_scalar -=
                    w * sqr(wanted - hamming_dis(perm[i], perm[j]));
            delta_cost_scalar +=
                    w * sqr(wanted - hamming_dis(perm[i], perm[jw]));
        }
        j = jw;
        {
            double wanted = target_dis[i * n + j];
            double w = weights[i * n + j];
            delta_cost_scalar -=
                    w * sqr(wanted - hamming_dis(perm[i], perm[j]));
            delta_cost_scalar +=
                    w * sqr(wanted - hamming_dis(perm[i], perm[iw]));
        }
    }
    return _mm512_reduce_add_pd(delta_cost_vec) + delta_cost_scalar;
}

double distances_compute_cost_avx512(
        const ReproduceDistancesObjective& obj,
        const int* perm) {
    const int n = obj.n;
    double total_cost = 0.0;
    for (int i = 0; i < n; ++i) {
        const int pi = perm[i];
        const int bro_t = i * n, bro_s = pi * n;
        __m512d sum = _mm512_setzero_pd();
        int j = 0;
        for (; j <= n - 8; j += 8) {
            __m512d wv = _mm512_loadu_pd(&obj.target_dis[bro_t + j]);
            __m512d ww = _mm512_loadu_pd(&obj.weights[bro_t + j]);
            __m256i pj = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(&perm[j]));
            __m256i idx = _mm256_add_epi32(_mm256_set1_epi32(bro_s), pj);
            __m512d av = _mm512_i32gather_pd(idx, obj.source_dis.data(), 8);
            __m512d d = _mm512_sub_pd(wv, av);
            sum = _mm512_fmadd_pd(_mm512_mul_pd(d, d), ww, sum);
        }
        total_cost += _mm512_reduce_add_pd(sum);
        for (; j < n; ++j) {
            double wanted = obj.target_dis[bro_t + j];
            double w = obj.weights[bro_t + j];
            double actual = obj.get_source_dis(pi, perm[j]);
            total_cost += w * sqr(wanted - actual);
        }
    }
    return total_cost;
}

double distances_cost_update_avx512(
        const ReproduceDistancesObjective& obj,
        const int* perm,
        int iw,
        int jw) {
    const int n = obj.n;
    double delta_cost = 0.0;
    const int p_iw = perm[iw], p_jw = perm[jw];
    const __m256i v_joff = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i vi = _mm256_set1_epi32(iw);
    const __m256i vj = _mm256_set1_epi32(jw);
    const __m256i vpi = _mm256_set1_epi32(p_iw);
    const __m256i vpj = _mm256_set1_epi32(p_jw);
    const __m256i vpin = _mm256_set1_epi32(p_iw * n);
    const __m256i vpjn = _mm256_set1_epi32(p_jw * n);

    auto process_row =
            [&](int row, int old_p, int new_p, __m256i old_pn, __m256i new_pn) {
                const int bro = row * n;
                __m512d dv = _mm512_setzero_pd();
                int j = 0;
                for (; j <= n - 8; j += 8) {
                    __m512d wv = _mm512_loadu_pd(&obj.target_dis[bro + j]);
                    __m512d ww = _mm512_loadu_pd(&obj.weights[bro + j]);
                    __m256i pjv = _mm256_loadu_si256(
                            reinterpret_cast<const __m256i*>(&perm[j]));
                    __m256i ia = _mm256_add_epi32(old_pn, pjv);
                    __m512d av =
                            _mm512_i32gather_pd(ia, obj.source_dis.data(), 8);
                    __m512d da = _mm512_sub_pd(wv, av);
                    dv = _mm512_fnmadd_pd(ww, _mm512_mul_pd(da, da), dv);
                    __m256i vk = _mm256_add_epi32(_mm256_set1_epi32(j), v_joff);
                    __mmask8 mi = _mm256_cmpeq_epi32_mask(vk, vi);
                    __mmask8 mj = _mm256_cmpeq_epi32_mask(vk, vj);
                    __m256i pnj = _mm256_mask_blend_epi32(mi, pjv, vpj);
                    pnj = _mm256_mask_blend_epi32(mj, pnj, vpi);
                    __m256i in2 = _mm256_add_epi32(new_pn, pnj);
                    __m512d nav =
                            _mm512_i32gather_pd(in2, obj.source_dis.data(), 8);
                    __m512d dn = _mm512_sub_pd(wv, nav);
                    dv = _mm512_fmadd_pd(ww, _mm512_mul_pd(dn, dn), dv);
                }
                delta_cost += _mm512_reduce_add_pd(dv);
                for (; j < n; ++j) {
                    double wanted = obj.target_dis[bro + j];
                    double w = obj.weights[bro + j];
                    double actual = obj.get_source_dis(old_p, perm[j]);
                    delta_cost -= w * sqr(wanted - actual);
                    int pnj = (j == iw) ? p_jw : ((j == jw) ? p_iw : perm[j]);
                    double na = obj.get_source_dis(new_p, pnj);
                    delta_cost += w * sqr(wanted - na);
                }
            };
    process_row(iw, p_iw, p_jw, vpin, vpjn);
    process_row(jw, p_jw, p_iw, vpjn, vpin);

    for (int i = 0; i < n; ++i) {
        if (i == iw || i == jw)
            continue;
        double wanted = obj.target_dis[i * n + iw], w = obj.weights[i * n + iw];
        double actual = obj.get_source_dis(perm[i], p_iw);
        delta_cost -= w * sqr(wanted - actual);
        double na = obj.get_source_dis(perm[i], p_jw);
        delta_cost += w * sqr(wanted - na);
        wanted = obj.target_dis[i * n + jw];
        w = obj.weights[i * n + jw];
        actual = obj.get_source_dis(perm[i], p_jw);
        delta_cost -= w * sqr(wanted - actual);
        na = obj.get_source_dis(perm[i], p_iw);
        delta_cost += w * sqr(wanted - na);
    }
    return delta_cost;
}

} // namespace polysemous_avx512
} // namespace faiss
