/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/impl/PolysemousTraining.h>

#include <omp.h>
#include <stdint.h>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>

#include <faiss/utils/distances.h>
#include <faiss/utils/hamming.h>
#include <faiss/utils/random.h>
#include <faiss/utils/utils.h>

#include <faiss/impl/FaissAssert.h>

/*****************************************
 * Mixed PQ / Hamming
 ******************************************/

namespace faiss {

/****************************************************
 * Optimization code
 ****************************************************/

// what would the cost update be if iw and jw were swapped?
// default implementation just computes both and computes the difference
double PermutationObjective::cost_update(const int* perm, int iw, int jw)
        const {
    double orig_cost = compute_cost(perm);

    std::vector<int> perm2(n);
    for (int i = 0; i < n; i++) {
        perm2[i] = perm[i];
    }
    perm2[iw] = perm[jw];
    perm2[jw] = perm[iw];

    double new_cost = compute_cost(perm2.data());
    return new_cost - orig_cost;
}

SimulatedAnnealingOptimizer::SimulatedAnnealingOptimizer(
        PermutationObjective* obj,
        const SimulatedAnnealingParameters& p)
        : SimulatedAnnealingParameters(p),
          obj(obj),
          n(obj->n),
          logfile(nullptr) {
    rnd = new RandomGenerator(p.seed);
    FAISS_THROW_IF_NOT(n < 100000 && n >= 0);
}

SimulatedAnnealingOptimizer::~SimulatedAnnealingOptimizer() {
    delete rnd;
}

// run the optimization and return the best result in best_perm
double SimulatedAnnealingOptimizer::run_optimization(int* best_perm) {
    double min_cost = 1e30;

    // just do a few runs of the annealing and keep the lowest output cost
    for (int it = 0; it < n_redo; it++) {
        std::vector<int> perm(n);
        for (int i = 0; i < n; i++) {
            perm[i] = i;
        }
        if (init_random) {
            for (int i = 0; i < n; i++) {
                int j = i + rnd->rand_int(n - i);
                std::swap(perm[i], perm[j]);
            }
        }
        float cost = optimize(perm.data());
        if (logfile) {
            fprintf(logfile, "\n");
        }
        if (verbose > 1) {
            printf("    optimization run %d: cost=%g %s\n",
                   it,
                   cost,
                   cost < min_cost ? "keep" : "");
        }
        if (cost < min_cost) {
            memcpy(best_perm, perm.data(), sizeof(perm[0]) * n);
            min_cost = cost;
        }
    }
    return min_cost;
}

// perform the optimization loop, starting from and modifying
// permutation in-place
double SimulatedAnnealingOptimizer::optimize(int* perm) {
    double cost = init_cost = obj->compute_cost(perm);
    int log2n = 0;
    while (!(n <= (1 << log2n))) {
        log2n++;
    }
    double temperature = init_temperature;
    int n_swap = 0, n_hot = 0;
    for (int it = 0; it < n_iter; it++) {
        temperature = temperature * temperature_decay;
        int iw, jw;
        if (only_bit_flips) {
            iw = rnd->rand_int(n);
            jw = iw ^ (1 << rnd->rand_int(log2n));
        } else {
            iw = rnd->rand_int(n);
            jw = rnd->rand_int(n - 1);
            if (jw == iw) {
                jw++;
            }
        }
        double delta_cost = obj->cost_update(perm, iw, jw);
        if (delta_cost < 0 || rnd->rand_float() < temperature) {
            std::swap(perm[iw], perm[jw]);
            cost += delta_cost;
            n_swap++;
            if (delta_cost >= 0) {
                n_hot++;
            }
        }
        if (verbose > 2 || (verbose > 1 && it % 10000 == 0)) {
            printf("      iteration %d cost %g temp %g n_swap %d "
                   "(%d hot)     \r",
                   it,
                   cost,
                   temperature,
                   n_swap,
                   n_hot);
            fflush(stdout);
        }
        if (logfile) {
            fprintf(logfile,
                    "%d %g %g %d %d\n",
                    it,
                    cost,
                    temperature,
                    n_swap,
                    n_hot);
        }
    }
    if (verbose > 1) {
        printf("\n");
    }
    return cost;
}

/****************************************************
 * Cost functions: ReproduceDistanceTable
 ****************************************************/

static inline int hamming_dis(uint64_t a, uint64_t b) {
    return __builtin_popcountl(a ^ b);
}

namespace {

/// optimize permutation to reproduce a distance table with Hamming distances
struct ReproduceWithHammingObjective : PermutationObjective {
    int nbits;
    double dis_weight_factor;

    static double sqr(double x) {
        return x * x;
    }

    // weihgting of distances: it is more important to reproduce small
    // distances well
    double dis_weight(double x) const {
        return exp(-dis_weight_factor * x);
    }

    std::vector<double> target_dis; // wanted distances (size n^2)
    std::vector<double> weights;    // weights for each distance (size n^2)

#if defined(__AVX512F__) && defined(__AVX512DQ__)

    static inline __m512i popcnt_u64(__m512i xor_v) {
        uint64_t t_xor[8];
        _mm512_storeu_si512(t_xor, xor_v);

        t_xor[0] = _mm_popcnt_u64(t_xor[0]);
        t_xor[1] = _mm_popcnt_u64(t_xor[1]);
        t_xor[2] = _mm_popcnt_u64(t_xor[2]);
        t_xor[3] = _mm_popcnt_u64(t_xor[3]);
        t_xor[4] = _mm_popcnt_u64(t_xor[4]);
        t_xor[5] = _mm_popcnt_u64(t_xor[5]);
        t_xor[6] = _mm_popcnt_u64(t_xor[6]);
        t_xor[7] = _mm_popcnt_u64(t_xor[7]);

        return _mm512_loadu_si512(t_xor);
    }

    double compute_cost(const int* perm) const override {
        double total_cost = 0.0;

        for (int i = 0; i < n; i++) {
            __m512d cost_vec = _mm512_setzero_pd();
            const int perm_i_scalar = perm[i];
            const __m512i perm_i_vec = _mm512_set1_epi64(perm_i_scalar);

            const int base_row_offset = i * n;

            int j = 0;
            for (; j <= n - 8; j += 8) {
                const __m512d wanted_vec =
                        _mm512_loadu_pd(&target_dis[base_row_offset + j]);
                const __m512d w_vec =
                        _mm512_loadu_pd(&weights[base_row_offset + j]);

                const __m256i perm_j_vec_i32 =
                        _mm256_loadu_si256((__m256i const*)&perm[j]);
                const __m512i perm_j_vec_i64 =
                        _mm512_cvtepi32_epi64(perm_j_vec_i32);

                const __m512i xor_res =
                        _mm512_xor_si512(perm_i_vec, perm_j_vec_i64);
#ifdef __AVX512VPOPCNTDQ__
                const __m512i popcnt_res = _mm512_popcnt_epi64(xor_res);
#else
                const __m512i popcnt_res = popcnt_u64(xor_res);
#endif
                const __m512d actual_vec = _mm512_cvtepi64_pd(popcnt_res);

                const __m512d diff = _mm512_sub_pd(wanted_vec, actual_vec);
                const __m512d diff_sq = _mm512_mul_pd(diff, diff);

                cost_vec = _mm512_fmadd_pd(w_vec, diff_sq, cost_vec);
            }

            total_cost += _mm512_reduce_add_pd(cost_vec);

            for (; j < n; j++) {
                double wanted = target_dis[base_row_offset + j];
                double w = weights[base_row_offset + j];
                double actual = hamming_dis(perm[i], perm[j]);
                total_cost += w * sqr(wanted - actual);
            }
        }
        return total_cost;
    }

    double cost_update(const int* perm, int iw, int jw) const override {
        double delta_cost_scalar = 0;
        __m512d delta_cost_vec = _mm512_setzero_pd();

        // Process row iw
        {
            const int base_row_offset = iw * n;
            __m512i v_perm_i_old = _mm512_set1_epi64(perm[iw]);
            __m512i v_perm_i_new = _mm512_set1_epi64(perm[jw]);
            int j = 0;
            for (; j <= n - 8; j += 8) {
                __m512d wanted_vec =
                        _mm512_loadu_pd(&target_dis[base_row_offset + j]);
                __m512d w_vec = _mm512_loadu_pd(&weights[base_row_offset + j]);

                __m256i perm_j_vec_i32 =
                        _mm256_loadu_si256((__m256i const*)&perm[j]);
                __m512i perm_j_vec = _mm512_cvtepi32_epi64(perm_j_vec_i32);
                __m512i xor_res = _mm512_xor_si512(v_perm_i_old, perm_j_vec);
#ifdef __AVX512VPOPCNTDQ__
                __m512i popcnt_res = _mm512_popcnt_epi64(xor_res);
#else
                __m512i popcnt_res = popcnt_u64(xor_res);
#endif
                __m512d actual_vec = _mm512_cvtepi64_pd(popcnt_res);
                __m512d term_old = _mm512_sub_pd(wanted_vec, actual_vec);
                term_old = _mm512_mul_pd(term_old, term_old);
                delta_cost_vec =
                        _mm512_fnmadd_pd(w_vec, term_old, delta_cost_vec);

                const __m512i v_indices_base =
                        _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
                __m512i j_indices =
                        _mm512_add_epi64(_mm512_set1_epi64(j), v_indices_base);
                __mmask8 mask_iw = _mm512_cmpeq_epi64_mask(
                        j_indices, _mm512_set1_epi64(iw));
                __mmask8 mask_jw = _mm512_cmpeq_epi64_mask(
                        j_indices, _mm512_set1_epi64(jw));
                __m512i perm_new_j_vec = _mm512_mask_blend_epi64(
                        mask_jw, perm_j_vec, _mm512_set1_epi64(perm[iw]));
                perm_new_j_vec = _mm512_mask_blend_epi64(
                        mask_iw, perm_new_j_vec, _mm512_set1_epi64(perm[jw]));

                xor_res = _mm512_xor_si512(v_perm_i_new, perm_new_j_vec);
#ifdef __AVX512VPOPCNTDQ__
                popcnt_res = _mm512_popcnt_epi64(xor_res);
#else
                popcnt_res = popcnt_u64(xor_res);
#endif
                __m512d new_actual_vec = _mm512_cvtepi64_pd(popcnt_res);
                __m512d term_new = _mm512_sub_pd(wanted_vec, new_actual_vec);
                term_new = _mm512_mul_pd(term_new, term_new);
                delta_cost_vec =
                        _mm512_fmadd_pd(w_vec, term_new, delta_cost_vec);
            }
            for (; j < n; j++) {
                double wanted = target_dis[base_row_offset + j];
                double w = weights[base_row_offset + j];
                double actual = hamming_dis(perm[iw], perm[j]);
                delta_cost_scalar -= w * sqr(wanted - actual);
                double new_actual = hamming_dis(
                        perm[jw],
                        perm[j == iw           ? jw
                                     : j == jw ? iw
                                               : j]);
                delta_cost_scalar += w * sqr(wanted - new_actual);
            }
        }

        // Process row jw
        {
            const int base_row_offset = jw * n;
            __m512i v_perm_i_old = _mm512_set1_epi64(perm[jw]);
            __m512i v_perm_i_new = _mm512_set1_epi64(perm[iw]);
            int j = 0;
            for (; j <= n - 8; j += 8) {
                __m512d wanted_vec =
                        _mm512_loadu_pd(&target_dis[base_row_offset + j]);
                __m512d w_vec = _mm512_loadu_pd(&weights[base_row_offset + j]);

                __m256i perm_j_vec_i32 =
                        _mm256_loadu_si256((__m256i const*)&perm[j]);
                __m512i perm_j_vec = _mm512_cvtepi32_epi64(perm_j_vec_i32);

                __m512i xor_res = _mm512_xor_si512(v_perm_i_old, perm_j_vec);
#ifdef __AVX512VPOPCNTDQ__
                __m512i popcnt_res = _mm512_popcnt_epi64(xor_res);
#else
                __m512i popcnt_res = popcnt_u64(xor_res);
#endif
                __m512d actual_vec = _mm512_cvtepi64_pd(popcnt_res);
                __m512d term_old = _mm512_sub_pd(wanted_vec, actual_vec);
                term_old = _mm512_mul_pd(term_old, term_old);
                delta_cost_vec =
                        _mm512_fnmadd_pd(w_vec, term_old, delta_cost_vec);

                const __m512i v_indices_base =
                        _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
                __m512i j_indices =
                        _mm512_add_epi64(_mm512_set1_epi64(j), v_indices_base);
                __mmask8 mask_iw = _mm512_cmpeq_epi64_mask(
                        j_indices, _mm512_set1_epi64(iw));
                __mmask8 mask_jw = _mm512_cmpeq_epi64_mask(
                        j_indices, _mm512_set1_epi64(jw));
                __m512i perm_new_j_vec = _mm512_mask_blend_epi64(
                        mask_jw, perm_j_vec, _mm512_set1_epi64(perm[iw]));
                perm_new_j_vec = _mm512_mask_blend_epi64(
                        mask_iw, perm_new_j_vec, _mm512_set1_epi64(perm[jw]));

                xor_res = _mm512_xor_si512(v_perm_i_new, perm_new_j_vec);
#ifdef __AVX512VPOPCNTDQ__
                popcnt_res = _mm512_popcnt_epi64(xor_res);
#else
                popcnt_res = popcnt_u64(xor_res);
#endif
                __m512d new_actual_vec = _mm512_cvtepi64_pd(popcnt_res);
                __m512d term_new = _mm512_sub_pd(wanted_vec, new_actual_vec);
                term_new = _mm512_mul_pd(term_new, term_new);
                delta_cost_vec =
                        _mm512_fmadd_pd(w_vec, term_new, delta_cost_vec);
            }
            for (; j < n; j++) {
                double wanted = target_dis[base_row_offset + j];
                double w = weights[base_row_offset + j];
                double actual = hamming_dis(perm[jw], perm[j]);
                delta_cost_scalar -= w * sqr(wanted - actual);
                double new_actual = hamming_dis(
                        perm[iw],
                        perm[j == iw           ? jw
                                     : j == jw ? iw
                                               : j]);
                delta_cost_scalar += w * sqr(wanted - new_actual);
            }
        }

        // Process other rows
        for (int i = 0; i < n; ++i) {
            if (i == iw || i == jw)
                continue;
            int j = iw;
            {
                double wanted = target_dis[i * n + j];
                double w = weights[i * n + j];
                double actual = hamming_dis(perm[i], perm[j]);
                delta_cost_scalar -= w * sqr(wanted - actual);
                double new_actual = hamming_dis(perm[i], perm[jw]);
                delta_cost_scalar += w * sqr(wanted - new_actual);
            }
            j = jw;
            {
                double wanted = target_dis[i * n + j];
                double w = weights[i * n + j];
                double actual = hamming_dis(perm[i], perm[j]);
                delta_cost_scalar -= w * sqr(wanted - actual);
                double new_actual = hamming_dis(perm[i], perm[iw]);
                delta_cost_scalar += w * sqr(wanted - new_actual);
            }
        }

        return _mm512_reduce_add_pd(delta_cost_vec) + delta_cost_scalar;
    }

#else

    // cost = quadratic difference between actual distance and Hamming distance
    double compute_cost(const int* perm) const override {
        double cost = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double wanted = target_dis[i * n + j];
                double w = weights[i * n + j];
                double actual = hamming_dis(perm[i], perm[j]);
                cost += w * sqr(wanted - actual);
            }
        }
        return cost;
    }

    // what would the cost update be if iw and jw were swapped?
    // computed in O(n) instead of O(n^2) for the full re-computation
    double cost_update(const int* perm, int iw, int jw) const override {
        double delta_cost = 0;

        for (int i = 0; i < n; i++) {
            if (i == iw) {
                for (int j = 0; j < n; j++) {
                    double wanted = target_dis[i * n + j],
                           w = weights[i * n + j];
                    double actual = hamming_dis(perm[i], perm[j]);
                    delta_cost -= w * sqr(wanted - actual);
                    double new_actual = hamming_dis(
                            perm[jw],
                            perm[j == iw           ? jw
                                         : j == jw ? iw
                                                   : j]);
                    delta_cost += w * sqr(wanted - new_actual);
                }
            } else if (i == jw) {
                for (int j = 0; j < n; j++) {
                    double wanted = target_dis[i * n + j],
                           w = weights[i * n + j];
                    double actual = hamming_dis(perm[i], perm[j]);
                    delta_cost -= w * sqr(wanted - actual);
                    double new_actual = hamming_dis(
                            perm[iw],
                            perm[j == iw           ? jw
                                         : j == jw ? iw
                                                   : j]);
                    delta_cost += w * sqr(wanted - new_actual);
                }
            } else {
                int j = iw;
                {
                    double wanted = target_dis[i * n + j],
                           w = weights[i * n + j];
                    double actual = hamming_dis(perm[i], perm[j]);
                    delta_cost -= w * sqr(wanted - actual);
                    double new_actual = hamming_dis(perm[i], perm[jw]);
                    delta_cost += w * sqr(wanted - new_actual);
                }
                j = jw;
                {
                    double wanted = target_dis[i * n + j],
                           w = weights[i * n + j];
                    double actual = hamming_dis(perm[i], perm[j]);
                    delta_cost -= w * sqr(wanted - actual);
                    double new_actual = hamming_dis(perm[i], perm[iw]);
                    delta_cost += w * sqr(wanted - new_actual);
                }
            }
        }

        return delta_cost;
    }

#endif

    ReproduceWithHammingObjective(
            int nbits,
            const std::vector<double>& dis_table,
            double dis_weight_factor)
            : nbits(nbits), dis_weight_factor(dis_weight_factor) {
        n = 1 << nbits;
        FAISS_THROW_IF_NOT(dis_table.size() == n * n);
        set_affine_target_dis(dis_table);
    }

    void set_affine_target_dis(const std::vector<double>& dis_table) {
        double sum = 0, sum2 = 0;
        int n2 = n * n;
        for (int i = 0; i < n2; i++) {
            sum += dis_table[i];
            sum2 += dis_table[i] * dis_table[i];
        }
        double mean = sum / n2;
        double stddev = sqrt(sum2 / n2 - (sum / n2) * (sum / n2));

        target_dis.resize(n2);

        for (int i = 0; i < n2; i++) {
            // the mapping function
            double td = (dis_table[i] - mean) / stddev * sqrt(nbits / 4) +
                    nbits / 2;
            target_dis[i] = td;
            // compute a weight
            weights.push_back(dis_weight(td));
        }
    }

    ~ReproduceWithHammingObjective() override {}
};

} // anonymous namespace

// weihgting of distances: it is more important to reproduce small
// distances well
double ReproduceDistancesObjective::dis_weight(double x) const {
    return exp(-dis_weight_factor * x);
}

double ReproduceDistancesObjective::get_source_dis(int i, int j) const {
    return source_dis[i * n + j];
}

#if defined(__AVX512F__) && defined(__AVX512VL__)

double ReproduceDistancesObjective::compute_cost(const int* perm) const {
    double total_cost = 0.0;

    for (int i = 0; i < n; ++i) {
        const int pi = perm[i];
        const int base_row_offset_target = i * n;
        const int base_row_offset_source = pi * n;
        __m512d cost_vec_sum = _mm512_setzero_pd();

        int j = 0;
        for (; j <= n - 8; j += 8) {
            __m512d wanted_vec =
                    _mm512_loadu_pd(&target_dis[base_row_offset_target + j]);
            __m512d weights_vec =
                    _mm512_loadu_pd(&weights[base_row_offset_target + j]);

            __m256i perm_j_ivec = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(&perm[j]));
            __m256i indices_ivec = _mm256_add_epi32(
                    _mm256_set1_epi32(base_row_offset_source), perm_j_ivec);

            __m512d actual_vec =
                    _mm512_i32gather_pd(indices_ivec, source_dis.data(), 8);
            __m512d diff_vec = _mm512_sub_pd(wanted_vec, actual_vec);

            cost_vec_sum = _mm512_fmadd_pd(
                    _mm512_mul_pd(diff_vec, diff_vec),
                    weights_vec,
                    cost_vec_sum);
        }

        total_cost += _mm512_reduce_add_pd(cost_vec_sum);

        for (; j < n; ++j) {
            double wanted = target_dis[base_row_offset_target + j];
            double w = weights[base_row_offset_target + j];
            double actual = get_source_dis(pi, perm[j]);
            total_cost += w * sqr(wanted - actual);
        }
    }

    return total_cost;
}

double ReproduceDistancesObjective::cost_update(const int* perm, int iw, int jw)
        const {
    double delta_cost = 0.0;

    const int p_iw = perm[iw];
    const int p_jw = perm[jw];

    const __m256i v_j_offsets = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i v_iw = _mm256_set1_epi32(iw);
    const __m256i v_jw = _mm256_set1_epi32(jw);
    const __m256i v_p_iw = _mm256_set1_epi32(p_iw);
    const __m256i v_p_jw = _mm256_set1_epi32(p_jw);
    const __m256i v_p_iw_n = _mm256_set1_epi32(p_iw * n);
    const __m256i v_p_jw_n = _mm256_set1_epi32(p_jw * n);

    // Process row iw
    {
        const int base_row_offset_target = iw * n;
        __m512d delta_vec = _mm512_setzero_pd();

        int j = 0;
        for (; j <= n - 8; j += 8) {
            __m512d wanted_vec =
                    _mm512_loadu_pd(&target_dis[base_row_offset_target + j]);
            __m512d weights_vec =
                    _mm512_loadu_pd(&weights[base_row_offset_target + j]);

            __m256i perm_j_ivec = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(&perm[j]));
            __m256i indices_actual_ivec =
                    _mm256_add_epi32(v_p_iw_n, perm_j_ivec);
            __m512d actual_vec = _mm512_i32gather_pd(
                    indices_actual_ivec, source_dis.data(), 8);
            __m512d diff_actual_vec = _mm512_sub_pd(wanted_vec, actual_vec);
            delta_vec = _mm512_fnmadd_pd(
                    weights_vec,
                    _mm512_mul_pd(diff_actual_vec, diff_actual_vec),
                    delta_vec);

            __m256i v_j = _mm256_add_epi32(_mm256_set1_epi32(j), v_j_offsets);
            __mmask8 mask_is_iw = _mm256_cmpeq_epi32_mask(v_j, v_iw);
            __mmask8 mask_is_jw = _mm256_cmpeq_epi32_mask(v_j, v_jw);
            __m256i perm_new_j_ivec =
                    _mm256_mask_blend_epi32(mask_is_iw, perm_j_ivec, v_p_jw);
            perm_new_j_ivec = _mm256_mask_blend_epi32(
                    mask_is_jw, perm_new_j_ivec, v_p_iw);

            __m256i indices_new_ivec =
                    _mm256_add_epi32(v_p_jw_n, perm_new_j_ivec);
            __m512d new_actual_vec =
                    _mm512_i32gather_pd(indices_new_ivec, source_dis.data(), 8);
            __m512d diff_new_vec = _mm512_sub_pd(wanted_vec, new_actual_vec);
            delta_vec = _mm512_fmadd_pd(
                    weights_vec,
                    _mm512_mul_pd(diff_new_vec, diff_new_vec),
                    delta_vec);
        }
        delta_cost += _mm512_reduce_add_pd(delta_vec);

        for (; j < n; ++j) {
            double wanted = target_dis[base_row_offset_target + j];
            double w = weights[base_row_offset_target + j];
            double actual = get_source_dis(p_iw, perm[j]);
            delta_cost -= w * sqr(wanted - actual);
            int perm_new_at_j = (j == iw) ? p_jw : ((j == jw) ? p_iw : perm[j]);
            double new_actual = get_source_dis(p_jw, perm_new_at_j);
            delta_cost += w * sqr(wanted - new_actual);
        }
    }

    // Process row jw
    {
        const int base_row_offset_target = jw * n;
        __m512d delta_vec = _mm512_setzero_pd();

        int j = 0;
        for (; j <= n - 8; j += 8) {
            __m512d wanted_vec =
                    _mm512_loadu_pd(&target_dis[base_row_offset_target + j]);
            __m512d weights_vec =
                    _mm512_loadu_pd(&weights[base_row_offset_target + j]);

            __m256i perm_j_ivec = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(&perm[j]));
            __m256i indices_actual_ivec =
                    _mm256_add_epi32(v_p_jw_n, perm_j_ivec);
            __m512d actual_vec = _mm512_i32gather_pd(
                    indices_actual_ivec, source_dis.data(), 8);
            __m512d diff_actual_vec = _mm512_sub_pd(wanted_vec, actual_vec);
            delta_vec = _mm512_fnmadd_pd(
                    weights_vec,
                    _mm512_mul_pd(diff_actual_vec, diff_actual_vec),
                    delta_vec);

            __m256i v_j = _mm256_add_epi32(_mm256_set1_epi32(j), v_j_offsets);
            __mmask8 mask_is_iw = _mm256_cmpeq_epi32_mask(v_j, v_iw);
            __mmask8 mask_is_jw = _mm256_cmpeq_epi32_mask(v_j, v_jw);
            __m256i perm_new_j_ivec =
                    _mm256_mask_blend_epi32(mask_is_iw, perm_j_ivec, v_p_jw);
            perm_new_j_ivec = _mm256_mask_blend_epi32(
                    mask_is_jw, perm_new_j_ivec, v_p_iw);

            __m256i indices_new_ivec =
                    _mm256_add_epi32(v_p_iw_n, perm_new_j_ivec);
            __m512d new_actual_vec =
                    _mm512_i32gather_pd(indices_new_ivec, source_dis.data(), 8);
            __m512d diff_new_vec = _mm512_sub_pd(wanted_vec, new_actual_vec);
            delta_vec = _mm512_fmadd_pd(
                    weights_vec,
                    _mm512_mul_pd(diff_new_vec, diff_new_vec),
                    delta_vec);
        }
        delta_cost += _mm512_reduce_add_pd(delta_vec);

        for (; j < n; ++j) {
            double wanted = target_dis[base_row_offset_target + j];
            double w = weights[base_row_offset_target + j];
            double actual = get_source_dis(p_jw, perm[j]);
            delta_cost -= w * sqr(wanted - actual);
            int perm_new_at_j = (j == iw) ? p_jw : ((j == jw) ? p_iw : perm[j]);
            double new_actual = get_source_dis(p_iw, perm_new_at_j);
            delta_cost += w * sqr(wanted - new_actual);
        }
    }

    for (int i = 0; i < n; ++i) {
        if (i == iw || i == jw)
            continue;

        double wanted = target_dis[i * n + iw], w = weights[i * n + iw];
        double actual = get_source_dis(perm[i], p_iw);
        delta_cost -= w * sqr(wanted - actual);
        double new_actual = get_source_dis(perm[i], p_jw);
        delta_cost += w * sqr(wanted - new_actual);

        wanted = target_dis[i * n + jw], w = weights[i * n + jw];
        actual = get_source_dis(perm[i], p_jw);
        delta_cost -= w * sqr(wanted - actual);
        new_actual = get_source_dis(perm[i], p_iw);
        delta_cost += w * sqr(wanted - new_actual);
    }

    return delta_cost;
}

#else

// cost = quadratic difference between actual distance and Hamming distance
double ReproduceDistancesObjective::compute_cost(const int* perm) const {
    double cost = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double wanted = target_dis[i * n + j];
            double w = weights[i * n + j];
            double actual = get_source_dis(perm[i], perm[j]);
            cost += w * sqr(wanted - actual);
        }
    }
    return cost;
}

// what would the cost update be if iw and jw were swapped?
// computed in O(n) instead of O(n^2) for the full re-computation
double ReproduceDistancesObjective::cost_update(const int* perm, int iw, int jw)
        const {
    double delta_cost = 0;
    for (int i = 0; i < n; i++) {
        if (i == iw) {
            for (int j = 0; j < n; j++) {
                double wanted = target_dis[i * n + j], w = weights[i * n + j];
                double actual = get_source_dis(perm[i], perm[j]);
                delta_cost -= w * sqr(wanted - actual);
                double new_actual = get_source_dis(
                        perm[jw],
                        perm[j == iw           ? jw
                                     : j == jw ? iw
                                               : j]);
                delta_cost += w * sqr(wanted - new_actual);
            }
        } else if (i == jw) {
            for (int j = 0; j < n; j++) {
                double wanted = target_dis[i * n + j], w = weights[i * n + j];
                double actual = get_source_dis(perm[i], perm[j]);
                delta_cost -= w * sqr(wanted - actual);
                double new_actual = get_source_dis(
                        perm[iw],
                        perm[j == iw           ? jw
                                     : j == jw ? iw
                                               : j]);
                delta_cost += w * sqr(wanted - new_actual);
            }
        } else {
            int j = iw;
            {
                double wanted = target_dis[i * n + j], w = weights[i * n + j];
                double actual = get_source_dis(perm[i], perm[j]);
                delta_cost -= w * sqr(wanted - actual);
                double new_actual = get_source_dis(perm[i], perm[jw]);
                delta_cost += w * sqr(wanted - new_actual);
            }
            j = jw;
            {
                double wanted = target_dis[i * n + j], w = weights[i * n + j];
                double actual = get_source_dis(perm[i], perm[j]);
                delta_cost -= w * sqr(wanted - actual);
                double new_actual = get_source_dis(perm[i], perm[iw]);
                delta_cost += w * sqr(wanted - new_actual);
            }
        }
    }
    return delta_cost;
}

#endif

ReproduceDistancesObjective::ReproduceDistancesObjective(
        int n,
        const double* source_dis_in,
        const double* target_dis_in,
        double dis_weight_factor)
        : dis_weight_factor(dis_weight_factor), target_dis(target_dis_in) {
    this->n = n;
    set_affine_target_dis(source_dis_in);
}

void ReproduceDistancesObjective::compute_mean_stdev(
        const double* tab,
        size_t n2,
        double* mean_out,
        double* stddev_out) {
    double sum = 0, sum2 = 0;
    for (int i = 0; i < n2; i++) {
        sum += tab[i];
        sum2 += tab[i] * tab[i];
    }
    double mean = sum / n2;
    double stddev = sqrt(sum2 / n2 - (sum / n2) * (sum / n2));
    *mean_out = mean;
    *stddev_out = stddev;
}

void ReproduceDistancesObjective::set_affine_target_dis(
        const double* source_dis_in) {
    int n2 = n * n;

    double mean_src, stddev_src;
    compute_mean_stdev(source_dis_in, n2, &mean_src, &stddev_src);

    double mean_target, stddev_target;
    compute_mean_stdev(target_dis, n2, &mean_target, &stddev_target);

    printf("map mean %g std %g -> mean %g std %g\n",
           mean_src,
           stddev_src,
           mean_target,
           stddev_target);

    source_dis.resize(n2);
    weights.resize(n2);

    for (int i = 0; i < n2; i++) {
        // the mapping function
        source_dis[i] =
                (source_dis_in[i] - mean_src) / stddev_src * stddev_target +
                mean_target;

        // compute a weight
        weights[i] = dis_weight(target_dis[i]);
    }
}

/****************************************************
 * Cost functions: RankingScore
 ****************************************************/

/// Maintains a 3D table of elementary costs.
/// Accumulates elements based on Hamming distance comparisons
template <typename Ttab, typename Taccu>
struct Score3Computer : PermutationObjective {
    int nc;

    // cost matrix of size nc * nc *nc
    // n_gt (i,j,k) = count of d_gt(x, y-) < d_gt(x, y+)
    // where x has PQ code i, y- PQ code j and y+ PQ code k
    std::vector<Ttab> n_gt;

    /// the cost is a triple loop on the nc * nc * nc matrix of entries.
    ///
    Taccu compute(const int* perm) const {
        Taccu accu = 0;
        const Ttab* p = n_gt.data();
        for (int i = 0; i < nc; i++) {
            int ip = perm[i];
            for (int j = 0; j < nc; j++) {
                int jp = perm[j];
                for (int k = 0; k < nc; k++) {
                    int kp = perm[k];
                    if (hamming_dis(ip, jp) < hamming_dis(ip, kp)) {
                        accu += *p; // n_gt [ ( i * nc + j) * nc + k];
                    }
                    p++;
                }
            }
        }
        return accu;
    }

    /** cost update if entries iw and jw of the permutation would be
     * swapped.
     *
     * The computation is optimized by avoiding elements in the
     * nc*nc*nc cube that are known not to change. For nc=256, this
     * reduces the nb of cells to visit to about 6/256 th of the
     * cells. Practical speedup is about 8x, and the code is quite
     * complex :-/
     */
    Taccu compute_update(const int* perm, int iw, int jw) const {
        assert(iw != jw);
        if (iw > jw) {
            std::swap(iw, jw);
        }

        Taccu accu = 0;
        const Ttab* n_gt_i = n_gt.data();
        for (int i = 0; i < nc; i++) {
            int ip0 = perm[i];
            int ip = perm[i == iw ? jw : i == jw ? iw : i];

            // accu += update_i (perm, iw, jw, ip0, ip, n_gt_i);

            accu += update_i_cross(perm, iw, jw, ip0, ip, n_gt_i);

            if (ip != ip0) {
                accu += update_i_plane(perm, iw, jw, ip0, ip, n_gt_i);
            }

            n_gt_i += nc * nc;
        }

        return accu;
    }

    Taccu update_i(
            const int* perm,
            int iw,
            int jw,
            int ip0,
            int ip,
            const Ttab* n_gt_i) const {
        Taccu accu = 0;
        const Ttab* n_gt_ij = n_gt_i;
        for (int j = 0; j < nc; j++) {
            int jp0 = perm[j];
            int jp = perm[j == iw ? jw : j == jw ? iw : j];
            for (int k = 0; k < nc; k++) {
                int kp0 = perm[k];
                int kp = perm[k == iw ? jw : k == jw ? iw : k];
                int ng = n_gt_ij[k];
                if (hamming_dis(ip, jp) < hamming_dis(ip, kp)) {
                    accu += ng;
                }
                if (hamming_dis(ip0, jp0) < hamming_dis(ip0, kp0)) {
                    accu -= ng;
                }
            }
            n_gt_ij += nc;
        }
        return accu;
    }

    // 2 inner loops for the case ip0 != ip
    Taccu update_i_plane(
            const int* perm,
            int iw,
            int jw,
            int ip0,
            int ip,
            const Ttab* n_gt_i) const {
        Taccu accu = 0;
        const Ttab* n_gt_ij = n_gt_i;

        for (int j = 0; j < nc; j++) {
            if (j != iw && j != jw) {
                int jp = perm[j];
                for (int k = 0; k < nc; k++) {
                    if (k != iw && k != jw) {
                        int kp = perm[k];
                        Ttab ng = n_gt_ij[k];
                        if (hamming_dis(ip, jp) < hamming_dis(ip, kp)) {
                            accu += ng;
                        }
                        if (hamming_dis(ip0, jp) < hamming_dis(ip0, kp)) {
                            accu -= ng;
                        }
                    }
                }
            }
            n_gt_ij += nc;
        }
        return accu;
    }

    /// used for the 8 cells were the 3 indices are swapped
    inline Taccu update_k(
            const int* perm,
            int iw,
            int jw,
            int ip0,
            int ip,
            int jp0,
            int jp,
            int k,
            const Ttab* n_gt_ij) const {
        Taccu accu = 0;
        int kp0 = perm[k];
        int kp = perm[k == iw ? jw : k == jw ? iw : k];
        Ttab ng = n_gt_ij[k];
        if (hamming_dis(ip, jp) < hamming_dis(ip, kp)) {
            accu += ng;
        }
        if (hamming_dis(ip0, jp0) < hamming_dis(ip0, kp0)) {
            accu -= ng;
        }
        return accu;
    }

    /// compute update on a line of k's, where i and j are swapped
    Taccu update_j_line(
            const int* perm,
            int iw,
            int jw,
            int ip0,
            int ip,
            int jp0,
            int jp,
            const Ttab* n_gt_ij) const {
        Taccu accu = 0;
        for (int k = 0; k < nc; k++) {
            if (k == iw || k == jw) {
                continue;
            }
            int kp = perm[k];
            Ttab ng = n_gt_ij[k];
            if (hamming_dis(ip, jp) < hamming_dis(ip, kp)) {
                accu += ng;
            }
            if (hamming_dis(ip0, jp0) < hamming_dis(ip0, kp)) {
                accu -= ng;
            }
        }
        return accu;
    }

    /// considers the 2 pairs of crossing lines j=iw or jw and k = iw or kw
    Taccu update_i_cross(
            const int* perm,
            int iw,
            int jw,
            int ip0,
            int ip,
            const Ttab* n_gt_i) const {
        Taccu accu = 0;
        const Ttab* n_gt_ij = n_gt_i;

        for (int j = 0; j < nc; j++) {
            int jp0 = perm[j];
            int jp = perm[j == iw ? jw : j == jw ? iw : j];

            accu += update_k(perm, iw, jw, ip0, ip, jp0, jp, iw, n_gt_ij);
            accu += update_k(perm, iw, jw, ip0, ip, jp0, jp, jw, n_gt_ij);

            if (jp != jp0) {
                accu += update_j_line(perm, iw, jw, ip0, ip, jp0, jp, n_gt_ij);
            }

            n_gt_ij += nc;
        }
        return accu;
    }

    /// PermutationObjective implementeation (just negates the scores
    /// for minimization)

    double compute_cost(const int* perm) const override {
        return -compute(perm);
    }

    double cost_update(const int* perm, int iw, int jw) const override {
        double ret = -compute_update(perm, iw, jw);
        return ret;
    }

    ~Score3Computer() override {}
};

struct IndirectSort {
    const float* tab;
    bool operator()(int a, int b) {
        return tab[a] < tab[b];
    }
};

struct RankingScore2 : Score3Computer<float, double> {
    int nbits;
    int nq, nb;
    const uint32_t *qcodes, *bcodes;
    const float* gt_distances;

    RankingScore2(
            int nbits,
            int nq,
            int nb,
            const uint32_t* qcodes,
            const uint32_t* bcodes,
            const float* gt_distances)
            : nbits(nbits),
              nq(nq),
              nb(nb),
              qcodes(qcodes),
              bcodes(bcodes),
              gt_distances(gt_distances) {
        n = nc = 1 << nbits;
        n_gt.resize(nc * nc * nc);
        init_n_gt();
    }

    double rank_weight(int r) {
        return 1.0 / (r + 1);
    }

    /// count nb of i, j in a x b st. i < j
    /// a and b should be sorted on input
    /// they are the ranks of j and k respectively.
    /// specific version for diff-of-rank weighting, cannot optimized
    /// with a cumulative table
    double accum_gt_weight_diff(
            const std::vector<int>& a,
            const std::vector<int>& b) {
        const auto nb_2 = b.size();
        const auto na = a.size();

        double accu = 0;
        size_t j = 0;
        for (size_t i = 0; i < na; i++) {
            const auto ai = a[i];
            while (j < nb_2 && ai >= b[j]) {
                j++;
            }

            double accu_i = 0;
            for (auto k = j; k < b.size(); k++) {
                accu_i += rank_weight(b[k] - ai);
            }

            accu += rank_weight(ai) * accu_i;
        }
        return accu;
    }

    void init_n_gt() {
        for (int q = 0; q < nq; q++) {
            const float* gtd = gt_distances + q * nb;
            const uint32_t* cb = bcodes; // all same codes
            float* n_gt_q = &n_gt[qcodes[q] * nc * nc];

            printf("init gt for q=%d/%d    \r", q, nq);
            fflush(stdout);

            std::vector<int> rankv(nb);
            int* ranks = rankv.data();

            // elements in each code bin, ordered by rank within each bin
            std::vector<std::vector<int>> tab(nc);

            { // build rank table
                IndirectSort s = {gtd};
                for (int j = 0; j < nb; j++) {
                    ranks[j] = j;
                }
                std::sort(ranks, ranks + nb, s);
            }

            for (int rank = 0; rank < nb; rank++) {
                int i = ranks[rank];
                tab[cb[i]].push_back(rank);
            }

            // this is very expensive. Any suggestion for improvement
            // welcome.
            for (int i = 0; i < nc; i++) {
                std::vector<int>& di = tab[i];
                for (int j = 0; j < nc; j++) {
                    std::vector<int>& dj = tab[j];
                    n_gt_q[i * nc + j] += accum_gt_weight_diff(di, dj);
                }
            }
        }
    }
};

/*****************************************
 * PolysemousTraining
 ******************************************/

PolysemousTraining::PolysemousTraining() {
    optimization_type = OT_ReproduceDistances_affine;
    ntrain_permutation = 0;
    dis_weight_factor = log(2);
    // max 20 G RAM
    max_memory = (size_t)(20) * 1024 * 1024 * 1024;
}

void PolysemousTraining::optimize_reproduce_distances(
        ProductQuantizer& pq) const {
    int dsub = pq.dsub;

    int n = pq.ksub;
    int nbits = pq.nbits;

    size_t mem1 = memory_usage_per_thread(pq);
    int nt = std::min(omp_get_max_threads(), int(pq.M));
    FAISS_THROW_IF_NOT_FMT(
            mem1 < max_memory,
            "Polysemous training will use %zd bytes per thread, while the max is set to %zd",
            mem1,
            max_memory);

    if (mem1 * nt > max_memory) {
        nt = max_memory / mem1;
        fprintf(stderr,
                "Polysemous training: WARN, reducing number of threads to %d to save memory",
                nt);
    }

#pragma omp parallel for num_threads(nt)
    for (int m = 0; m < pq.M; m++) {
        std::vector<double> dis_table;

        // printf ("Optimizing quantizer %d\n", m);

        float* centroids = pq.get_centroids(m, 0);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                dis_table.push_back(fvec_L2sqr(
                        centroids + i * dsub, centroids + j * dsub, dsub));
            }
        }

        std::vector<int> perm(n);
        ReproduceWithHammingObjective obj(nbits, dis_table, dis_weight_factor);

        SimulatedAnnealingOptimizer optim(&obj, *this);

        if (log_pattern.size()) {
            char fname[256];
            snprintf(fname, 256, log_pattern.c_str(), m);
            printf("opening log file %s\n", fname);
            optim.logfile = fopen(fname, "w");
            FAISS_THROW_IF_NOT_MSG(optim.logfile, "could not open logfile");
        }
        double final_cost = optim.run_optimization(perm.data());

        if (verbose > 0) {
            printf("SimulatedAnnealingOptimizer for m=%d: %g -> %g\n",
                   m,
                   optim.init_cost,
                   final_cost);
        }

        if (log_pattern.size()) {
            fclose(optim.logfile);
        }

        std::vector<float> centroids_copy;
        for (int i = 0; i < dsub * n; i++) {
            centroids_copy.push_back(centroids[i]);
        }

        for (int i = 0; i < n; i++) {
            memcpy(centroids + perm[i] * dsub,
                   centroids_copy.data() + i * dsub,
                   dsub * sizeof(centroids[0]));
        }
    }
}

void PolysemousTraining::optimize_ranking(
        ProductQuantizer& pq,
        size_t n,
        const float* x) const {
    int dsub = pq.dsub;
    int nbits = pq.nbits;

    std::vector<uint8_t> all_codes(pq.code_size * n);

    pq.compute_codes(x, all_codes.data(), n);

    FAISS_THROW_IF_NOT(pq.nbits == 8);

    if (n == 0) {
        pq.compute_sdc_table();
    }

#pragma omp parallel for
    for (int m = 0; m < pq.M; m++) {
        size_t nq, nb;
        std::vector<uint32_t> codes;     // query codes, then db codes
        std::vector<float> gt_distances; // nq * nb matrix of distances

        if (n > 0) {
            std::vector<float> xtrain(n * dsub);
            for (int i = 0; i < n; i++) {
                memcpy(xtrain.data() + i * dsub,
                       x + i * pq.d + m * dsub,
                       sizeof(float) * dsub);
            }

            codes.resize(n);
            for (int i = 0; i < n; i++) {
                codes[i] = all_codes[i * pq.code_size + m];
            }

            nq = n / 4;
            nb = n - nq;
            const float* xq = xtrain.data();
            const float* xb = xq + nq * dsub;

            gt_distances.resize(nq * nb);

            pairwise_L2sqr(dsub, nq, xq, nb, xb, gt_distances.data());
        } else {
            nq = nb = pq.ksub;
            codes.resize(2 * nq);
            for (int i = 0; i < nq; i++) {
                codes[i] = codes[i + nq] = i;
            }

            gt_distances.resize(nq * nb);

            memcpy(gt_distances.data(),
                   pq.sdc_table.data() + m * nq * nb,
                   sizeof(float) * nq * nb);
        }

        double t0 = getmillisecs();

        std::unique_ptr<PermutationObjective> obj(new RankingScore2(
                nbits,
                nq,
                nb,
                codes.data(),
                codes.data() + nq,
                gt_distances.data()));

        if (verbose > 0) {
            printf("   m=%d, nq=%zd, nb=%zd, initialize RankingScore "
                   "in %.3f ms\n",
                   m,
                   nq,
                   nb,
                   getmillisecs() - t0);
        }

        SimulatedAnnealingOptimizer optim(obj.get(), *this);

        if (log_pattern.size()) {
            char fname[256];
            snprintf(fname, 256, log_pattern.c_str(), m);
            printf("opening log file %s\n", fname);
            optim.logfile = fopen(fname, "w");
            FAISS_THROW_IF_NOT_FMT(
                    optim.logfile, "could not open logfile %s", fname);
        }

        std::vector<int> perm(pq.ksub);

        double final_cost = optim.run_optimization(perm.data());
        printf("SimulatedAnnealingOptimizer for m=%d: %g -> %g\n",
               m,
               optim.init_cost,
               final_cost);

        if (log_pattern.size()) {
            fclose(optim.logfile);
        }

        float* centroids = pq.get_centroids(m, 0);

        std::vector<float> centroids_copy;
        for (int i = 0; i < dsub * pq.ksub; i++) {
            centroids_copy.push_back(centroids[i]);
        }

        for (int i = 0; i < pq.ksub; i++) {
            memcpy(centroids + perm[i] * dsub,
                   centroids_copy.data() + i * dsub,
                   dsub * sizeof(centroids[0]));
        }
    }
}

void PolysemousTraining::optimize_pq_for_hamming(
        ProductQuantizer& pq,
        size_t n,
        const float* x) const {
    if (optimization_type == OT_None) {
    } else if (optimization_type == OT_ReproduceDistances_affine) {
        optimize_reproduce_distances(pq);
    } else {
        optimize_ranking(pq, n, x);
    }

    pq.compute_sdc_table();
}

size_t PolysemousTraining::memory_usage_per_thread(
        const ProductQuantizer& pq) const {
    size_t n = pq.ksub;

    switch (optimization_type) {
        case OT_None:
            return 0;
        case OT_ReproduceDistances_affine:
            return n * n * sizeof(double) * 3;
        case OT_Ranking_weighted_diff:
            return n * n * n * sizeof(float);
    }

    FAISS_THROW_MSG("Invalid optmization type");
    return 0;
}

} // namespace faiss
