/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <vector>

#include <faiss/Index.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ResidualQuantizer.h>
#include <faiss/utils/approx_topk/mode.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/simdlib.h>
#include <faiss/utils/utils.h>

#include <faiss/utils/approx_topk/approx_topk.h>

namespace faiss {

// This file contains the SIMD templatized implementations of various routines.

/********************************************************************
 * Basic routines
 ********************************************************************/

template <size_t M, size_t NK, SIMDLevel SL>
void accum_and_store_tab(
        const size_t m_offset,
        const float* const __restrict codebook_cross_norms,
        const uint64_t* const __restrict codebook_offsets,
        const int32_t* const __restrict codes_i,
        const size_t b,
        const size_t ldc,
        const size_t K,
        float* const __restrict output) {
    // load pointers into registers
    const float* cbs[M];
    for (size_t ij = 0; ij < M; ij++) {
        const size_t code = static_cast<size_t>(codes_i[b * m_offset + ij]);
        cbs[ij] = &codebook_cross_norms[(codebook_offsets[ij] + code) * ldc];
    }

// do accumulation in registers using SIMD.
// It is possible that compiler may be smart enough so that
//   this manual SIMD unrolling might be unneeded.
#if defined(__AVX2__) || defined(__aarch64__)
#if defined(__AVX2__)
    using simd8float32 = simd8float32<SIMDLevel::AVX2>;
#else
    using simd8float32 = simd8float32<SIMDLevel::ARM_NEON>;
#endif
    const size_t K8 = (K / (8 * NK)) * (8 * NK);

    // process in chunks of size (8 * NK) floats
    for (size_t kk = 0; kk < K8; kk += 8 * NK) {
        simd8float32 regs[NK];
        for (size_t ik = 0; ik < NK; ik++) {
            regs[ik].loadu(cbs[0] + kk + ik * 8);
        }

        for (size_t ij = 1; ij < M; ij++) {
            for (size_t ik = 0; ik < NK; ik++) {
                regs[ik] += simd8float32(cbs[ij] + kk + ik * 8);
            }
        }

        // write the result
        for (size_t ik = 0; ik < NK; ik++) {
            regs[ik].storeu(output + kk + ik * 8);
        }
    }
#else
    const size_t K8 = 0;
#endif

    // process leftovers
    for (size_t kk = K8; kk < K; kk++) {
        float reg = cbs[0][kk];
        for (size_t ij = 1; ij < M; ij++) {
            reg += cbs[ij][kk];
        }
        output[kk] = reg;
    }
}

template <size_t M, size_t NK, SIMDLevel SL>
void accum_and_add_tab(
        const size_t m_offset,
        const float* const __restrict codebook_cross_norms,
        const uint64_t* const __restrict codebook_offsets,
        const int32_t* const __restrict codes_i,
        const size_t b,
        const size_t ldc,
        const size_t K,
        float* const __restrict output) {
    // load pointers into registers
    const float* cbs[M];
    for (size_t ij = 0; ij < M; ij++) {
        const size_t code = static_cast<size_t>(codes_i[b * m_offset + ij]);
        cbs[ij] = &codebook_cross_norms[(codebook_offsets[ij] + code) * ldc];
    }

// do accumulation in registers using SIMD.
// It is possible that compiler may be smart enough so that
//   this manual SIMD unrolling might be unneeded
#if defined(__AVX2__) || defined(__aarch64__)
#if defined(__AVX2__)
    using simd8float32 = simd8float32<SIMDLevel::AVX2>;
#else
    using simd8float32 = simd8float32<SIMDLevel::ARM_NEON>;
#endif
    const size_t K8 = (K / (8 * NK)) * (8 * NK);

    // process in chunks of size (8 * NK) floats
    for (size_t kk = 0; kk < K8; kk += 8 * NK) {
        simd8float32 regs[NK];
        for (size_t ik = 0; ik < NK; ik++) {
            regs[ik].loadu(cbs[0] + kk + ik * 8);
        }

        for (size_t ij = 1; ij < M; ij++) {
            for (size_t ik = 0; ik < NK; ik++) {
                regs[ik] += simd8float32(cbs[ij] + kk + ik * 8);
            }
        }

        // write the result
        for (size_t ik = 0; ik < NK; ik++) {
            simd8float32 existing(output + kk + ik * 8);
            existing += regs[ik];
            existing.storeu(output + kk + ik * 8);
        }
    }
#else
    const size_t K8 = 0;
#endif

    // process leftovers
    for (size_t kk = K8; kk < K; kk++) {
        float reg = cbs[0][kk];
        for (size_t ij = 1; ij < M; ij++) {
            reg += cbs[ij][kk];
        }
        output[kk] += reg;
    }
}

template <size_t M, size_t NK, SIMDLevel SL>
void accum_and_finalize_tab(
        const float* const __restrict codebook_cross_norms,
        const uint64_t* const __restrict codebook_offsets,
        const int32_t* const __restrict codes_i,
        const size_t b,
        const size_t ldc,
        const size_t K,
        const float* const __restrict distances_i,
        const float* const __restrict cd_common,
        float* const __restrict output) {
    // load pointers into registers
    const float* cbs[M];
    for (size_t ij = 0; ij < M; ij++) {
        const size_t code = static_cast<size_t>(codes_i[b * M + ij]);
        cbs[ij] = &codebook_cross_norms[(codebook_offsets[ij] + code) * ldc];
    }

// do accumulation in registers using SIMD.
// It is possible that compiler may be smart enough so that
//   this manual SIMD unrolling might be unneeded.
#if defined(__AVX2__) || defined(__aarch64__)
#if defined(__AVX2__)
    using simd8float32 = simd8float32<SIMDLevel::AVX2>;
#else
    using simd8float32 = simd8float32<SIMDLevel::ARM_NEON>;
#endif

    const size_t K8 = (K / (8 * NK)) * (8 * NK);

    // process in chunks of size (8 * NK) floats
    for (size_t kk = 0; kk < K8; kk += 8 * NK) {
        simd8float32 regs[NK];
        for (size_t ik = 0; ik < NK; ik++) {
            regs[ik].loadu(cbs[0] + kk + ik * 8);
        }

        for (size_t ij = 1; ij < M; ij++) {
            for (size_t ik = 0; ik < NK; ik++) {
                regs[ik] += simd8float32(cbs[ij] + kk + ik * 8);
            }
        }

        simd8float32 two(2.0f);
        for (size_t ik = 0; ik < NK; ik++) {
            // cent_distances[b * K + k] = distances_i[b] + cd_common[k]
            //     + 2 * dp[k];

            simd8float32 common_v(cd_common + kk + ik * 8);
            common_v = fmadd(two, regs[ik], common_v);

            common_v += simd8float32(distances_i[b]);
            common_v.storeu(output + b * K + kk + ik * 8);
        }
    }
#else
    const size_t K8 = 0;
#endif

    // process leftovers
    for (size_t kk = K8; kk < K; kk++) {
        float reg = cbs[0][kk];
        for (size_t ij = 1; ij < M; ij++) {
            reg += cbs[ij][kk];
        }

        output[b * K + kk] = distances_i[b] + cd_common[kk] + 2 * reg;
    }
}

/********************************************************************
 * Single step of encoding
 ********************************************************************/

/** Encode a set of vectors using their dot products with the codebooks
 *
 * @param K           number of vectors in the codebook
 * @param n           nb of vectors to encode
 * @param beam_size   input beam size
 * @param codebook_cross_norms inner product of this codebook with the m
 *                             previously encoded codebooks
 * @param codebook_offsets     offsets into codebook_cross_norms for each
 *                             previous codebook
 * @param query_cp    dot products of query vectors with ???
 * @param cent_norms_i  norms of centroids
 */
template <SIMDLevel SL>
void beam_search_encode_step_tab(
        size_t K,
        size_t n,
        size_t beam_size,                  // input sizes
        const float* codebook_cross_norms, // size K * ldc
        size_t ldc,                        // >= K
        const uint64_t* codebook_offsets,  // m
        const float* query_cp,             // size n * ldqc
        size_t ldqc,                       // >= K
        const float* cent_norms_i,         // size K
        size_t m,
        const int32_t* codes,   // n * beam_size * m
        const float* distances, // n * beam_size
        size_t new_beam_size,
        int32_t* new_codes,   // n * new_beam_size * (m + 1)
        float* new_distances, // n * new_beam_size
        ApproxTopK_mode_t approx_topk_mode = ApproxTopK_mode_t::EXACT_TOPK) {
    FAISS_THROW_IF_NOT(ldc >= K);

#pragma omp parallel for if (n > 100) schedule(dynamic)
    for (int64_t i = 0; i < n; i++) {
        std::vector<float> cent_distances(beam_size * K);
        std::vector<float> cd_common(K);

        const int32_t* codes_i = codes + i * m * beam_size;
        const float* query_cp_i = query_cp + i * ldqc;
        const float* distances_i = distances + i * beam_size;

        for (size_t k = 0; k < K; k++) {
            cd_common[k] = cent_norms_i[k] - 2 * query_cp_i[k];
        }

        bool use_baseline_implementation = false;

        // This is the baseline implementation. Its primary flaw
        //   that it writes way too many info to the temporary buffer
        //   called dp.
        //
        // This baseline code is kept intentionally because it is easy to
        // understand what an optimized version optimizes exactly.
        //
        if (use_baseline_implementation) {
            for (size_t b = 0; b < beam_size; b++) {
                std::vector<float> dp(K);

                for (size_t m1 = 0; m1 < m; m1++) {
                    size_t c = codes_i[b * m + m1];
                    const float* cb =
                            &codebook_cross_norms
                                    [(codebook_offsets[m1] + c) * ldc];
                    fvec_add(K, cb, dp.data(), dp.data());
                }

                for (size_t k = 0; k < K; k++) {
                    cent_distances[b * K + k] =
                            distances_i[b] + cd_common[k] + 2 * dp[k];
                }
            }

        } else {
            // An optimized implementation that avoids using a temporary buffer
            // and does the accumulation in registers.

            // Compute a sum of NK AQ codes.
#define ACCUM_AND_FINALIZE_TAB(NK)               \
    case NK:                                     \
        for (size_t b = 0; b < beam_size; b++) { \
            accum_and_finalize_tab<NK, 4, SL>(   \
                    codebook_cross_norms,        \
                    codebook_offsets,            \
                    codes_i,                     \
                    b,                           \
                    ldc,                         \
                    K,                           \
                    distances_i,                 \
                    cd_common.data(),            \
                    cent_distances.data());      \
        }                                        \
        break;

            // this version contains many switch-case scenarios, but
            // they won't affect branch predictor.
            switch (m) {
                case 0:
                    // trivial case
                    for (size_t b = 0; b < beam_size; b++) {
                        for (size_t k = 0; k < K; k++) {
                            cent_distances[b * K + k] =
                                    distances_i[b] + cd_common[k];
                        }
                    }
                    break;

                    ACCUM_AND_FINALIZE_TAB(1)
                    ACCUM_AND_FINALIZE_TAB(2)
                    ACCUM_AND_FINALIZE_TAB(3)
                    ACCUM_AND_FINALIZE_TAB(4)
                    ACCUM_AND_FINALIZE_TAB(5)
                    ACCUM_AND_FINALIZE_TAB(6)
                    ACCUM_AND_FINALIZE_TAB(7)

                default: {
                    // m >= 8 case.

                    // A temporary buffer has to be used due to the lack of
                    // registers. But we'll try to accumulate up to 8 AQ codes
                    // in registers and issue a single write operation to the
                    // buffer, while the baseline does no accumulation. So, the
                    // number of write operations to the temporary buffer is
                    // reduced 8x.

                    // allocate a temporary buffer
                    std::vector<float> dp(K);

                    for (size_t b = 0; b < beam_size; b++) {
                        // Initialize it. Compute a sum of first 8 AQ codes
                        // because m >= 8 .
                        accum_and_store_tab<8, 4, SL>(
                                m,
                                codebook_cross_norms,
                                codebook_offsets,
                                codes_i,
                                b,
                                ldc,
                                K,
                                dp.data());

#define ACCUM_AND_ADD_TAB(NK)          \
    case NK:                           \
        accum_and_add_tab<NK, 4, SL>(  \
                m,                     \
                codebook_cross_norms,  \
                codebook_offsets + im, \
                codes_i + im,          \
                b,                     \
                ldc,                   \
                K,                     \
                dp.data());            \
        break;

                        // accumulate up to 8 additional AQ codes into
                        // a temporary buffer
                        for (size_t im = 8; im < ((m + 7) / 8) * 8; im += 8) {
                            size_t m_left = m - im;
                            if (m_left > 8) {
                                m_left = 8;
                            }

                            switch (m_left) {
                                ACCUM_AND_ADD_TAB(1)
                                ACCUM_AND_ADD_TAB(2)
                                ACCUM_AND_ADD_TAB(3)
                                ACCUM_AND_ADD_TAB(4)
                                ACCUM_AND_ADD_TAB(5)
                                ACCUM_AND_ADD_TAB(6)
                                ACCUM_AND_ADD_TAB(7)
                                ACCUM_AND_ADD_TAB(8)
                            }
                        }

                        // done. finalize the result
                        for (size_t k = 0; k < K; k++) {
                            cent_distances[b * K + k] =
                                    distances_i[b] + cd_common[k] + 2 * dp[k];
                        }
                    }
                }
            }

            // the optimized implementation ends here
        }
        using C = CMax<float, int>;
        int32_t* new_codes_i = new_codes + i * (m + 1) * new_beam_size;
        float* new_distances_i = new_distances + i * new_beam_size;

        const float* cent_distances_i = cent_distances.data();

        // then we have to select the best results
        for (int i_2 = 0; i_2 < new_beam_size; i_2++) {
            new_distances_i[i_2] = C::neutral();
        }
        std::vector<int> perm(new_beam_size, -1);

#define HANDLE_APPROX(NB, BD)                                  \
    case ApproxTopK_mode_t::APPROX_TOPK_BUCKETS_B##NB##_D##BD: \
        HeapWithBuckets<C, NB, BD>::bs_addn(                   \
                beam_size,                                     \
                K,                                             \
                cent_distances_i,                              \
                new_beam_size,                                 \
                new_distances_i,                               \
                perm.data());                                  \
        break;

        switch (approx_topk_mode) {
            HANDLE_APPROX(8, 3)
            HANDLE_APPROX(8, 2)
            HANDLE_APPROX(16, 2)
            HANDLE_APPROX(32, 2)
            default:
                heap_addn<C>(
                        new_beam_size,
                        new_distances_i,
                        perm.data(),
                        cent_distances_i,
                        nullptr,
                        beam_size * K);
                break;
        }

        heap_reorder<C>(new_beam_size, new_distances_i, perm.data());

#undef HANDLE_APPROX

        for (int j = 0; j < new_beam_size; j++) {
            int js = perm[j] / K;
            int ls = perm[j] % K;
            if (m > 0) {
                memcpy(new_codes_i, codes_i + js * m, sizeof(*codes) * m);
            }
            new_codes_i[m] = ls;
            new_codes_i += m + 1;
        }
    }
}

namespace {
extern "C" {

#ifndef FINTEGER
#define FINTEGER int
#endif

// general matrix multiplication
int sgemm_(
        const char* transa,
        const char* transb,
        FINTEGER* m,
        FINTEGER* n,
        FINTEGER* k,
        const float* alpha,
        const float* a,
        FINTEGER* lda,
        const float* b,
        FINTEGER* ldb,
        float* beta,
        float* c,
        FINTEGER* ldc);
}
} // namespace

/********************************************************************
 * Multiple encoding steps
 ********************************************************************/

namespace rq_encode_steps {

// Preallocated memory chunk for refine_beam_LUT_mp() call
struct RefineBeamLUTMemoryPool {
    std::vector<int32_t> new_codes;
    std::vector<float> new_distances;

    std::vector<int32_t> codes;
    std::vector<float> distances;
};

// this is for use_beam_LUT == 1 in compute_codes_add_centroids_mp_lut1() call
struct ComputeCodesAddCentroidsLUT1MemoryPool {
    std::vector<int32_t> codes;
    std::vector<float> distances;
    std::vector<float> query_norms;
    std::vector<float> query_cp;
    std::vector<float> residuals;
    RefineBeamLUTMemoryPool refine_beam_lut_pool;
};

template <SIMDLevel SL>
void refine_beam_LUT_mp(
        const ResidualQuantizer& rq,
        size_t n,
        const float* query_norms, // size n
        const float* query_cp,    //
        int out_beam_size,
        int32_t* out_codes,
        float* out_distances,
        RefineBeamLUTMemoryPool& pool) {
    {
        int beam_size = 1;

        double t0 = getmillisecs();

        // find the max_beam_size
        int max_beam_size = 0;
        {
            int tmp_beam_size = beam_size;
            for (int m = 0; m < rq.M; m++) {
                int K = 1 << rq.nbits[m];
                int new_beam_size = std::min(tmp_beam_size * K, out_beam_size);
                tmp_beam_size = new_beam_size;

                if (max_beam_size < new_beam_size) {
                    max_beam_size = new_beam_size;
                }
            }
        }

        // preallocate buffers
        pool.new_codes.resize(n * max_beam_size * (rq.M + 1));
        pool.new_distances.resize(n * max_beam_size);

        pool.codes.resize(n * max_beam_size * (rq.M + 1));
        pool.distances.resize(n * max_beam_size);

        for (size_t i = 0; i < n; i++) {
            pool.distances[i] = query_norms[i];
        }

        // set up pointers to buffers
        int32_t* __restrict new_codes_ptr = pool.new_codes.data();
        float* __restrict new_distances_ptr = pool.new_distances.data();

        int32_t* __restrict codes_ptr = pool.codes.data();
        float* __restrict distances_ptr = pool.distances.data();

        // main loop
        size_t codes_size = 0;
        size_t distances_size = 0;
        size_t cross_ofs = 0;
        for (int m = 0; m < rq.M; m++) {
            int K = 1 << rq.nbits[m];

            // it is guaranteed that (new_beam_size <= max_beam_size)
            int new_beam_size = std::min(beam_size * K, out_beam_size);

            codes_size = n * new_beam_size * (m + 1);
            distances_size = n * new_beam_size;
            FAISS_THROW_IF_NOT(
                    cross_ofs + rq.codebook_offsets[m] * K <=
                    rq.codebook_cross_products.size());
            beam_search_encode_step_tab<SL>(
                    K,
                    n,
                    beam_size,
                    rq.codebook_cross_products.data() + cross_ofs,
                    K,
                    rq.codebook_offsets.data(),
                    query_cp + rq.codebook_offsets[m],
                    rq.total_codebook_size,
                    rq.centroid_norms.data() + rq.codebook_offsets[m],
                    m,
                    codes_ptr,
                    distances_ptr,
                    new_beam_size,
                    new_codes_ptr,
                    new_distances_ptr,
                    rq.approx_topk_mode);
            cross_ofs += rq.codebook_offsets[m] * K;
            std::swap(codes_ptr, new_codes_ptr);
            std::swap(distances_ptr, new_distances_ptr);

            beam_size = new_beam_size;

            if (rq.verbose) {
                float sum_distances = 0;
                for (int j = 0; j < distances_size; j++) {
                    sum_distances += distances_ptr[j];
                }
                printf("[%.3f s] encode stage %d, %d bits, "
                       "total error %g, beam_size %d\n",
                       (getmillisecs() - t0) / 1000,
                       m,
                       int(rq.nbits[m]),
                       sum_distances,
                       beam_size);
            }
        }
        if (out_codes) {
            memcpy(out_codes, codes_ptr, codes_size * sizeof(*codes_ptr));
        }
        if (out_distances) {
            memcpy(out_distances,
                   distances_ptr,
                   distances_size * sizeof(*distances_ptr));
        }
    }
}

template <SIMDLevel SL>
void compute_codes_add_centroids_mp_lut1(
        const ResidualQuantizer& rq,
        const float* x,
        uint8_t* codes_out,
        size_t n,
        const float* centroids,
        ComputeCodesAddCentroidsLUT1MemoryPool& pool) {
    pool.codes.resize(rq.max_beam_size * rq.M * n);
    pool.distances.resize(rq.max_beam_size * n);

    FAISS_THROW_IF_NOT_MSG(
            rq.M == 1 || rq.codebook_cross_products.size() > 0,
            "call compute_codebook_tables first");

    pool.query_norms.resize(n);
    fvec_norms_L2sqr(pool.query_norms.data(), x, rq.d, n);

    pool.query_cp.resize(n * rq.total_codebook_size);
    {
        FINTEGER ti = rq.total_codebook_size, di = rq.d, ni = n;
        float zero = 0, one = 1;
        sgemm_("Transposed",
               "Not transposed",
               &ti,
               &ni,
               &di,
               &one,
               rq.codebooks.data(),
               &di,
               x,
               &di,
               &zero,
               pool.query_cp.data(),
               &ti);
    }

    refine_beam_LUT_mp<SIMDLevel::NONE>(
            rq,
            n,
            pool.query_norms.data(),
            pool.query_cp.data(),
            rq.max_beam_size,
            pool.codes.data(),
            pool.distances.data(),
            pool.refine_beam_lut_pool);

    // pack only the first code of the beam
    //   (hence the ld_codes=M * max_beam_size)
    rq.pack_codes(
            n,
            pool.codes.data(),
            codes_out,
            rq.M * rq.max_beam_size,
            nullptr,
            centroids);
}

} // namespace rq_encode_steps

} // namespace faiss
