/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/residual_quantizer_encode_steps/residual_quantizer_encode_steps.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ResidualQuantizer.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/utils.h>

#include <faiss/utils/approx_topk/approx_topk.h>

extern "C" {

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

namespace faiss {

/********************************************************************
 * Single encoding step
 ********************************************************************/

void beam_search_encode_step(
        size_t d,
        size_t K,
        const float* cent, /// size (K, d)
        size_t n,
        size_t beam_size,
        const float* residuals, /// size (n, beam_size, d)
        size_t m,
        const int32_t* codes, /// size (n, beam_size, m)
        size_t new_beam_size,
        int32_t* new_codes,   /// size (n, new_beam_size, m + 1)
        float* new_residuals, /// size (n, new_beam_size, d)
        float* new_distances, /// size (n, new_beam_size)
        Index* assign_index,
        ApproxTopK_mode_t approx_topk_mode) {
    // we have to fill in the whole output matrix
    FAISS_THROW_IF_NOT(new_beam_size <= beam_size * K);

    std::vector<float> cent_distances;
    std::vector<idx_t> cent_ids;

    if (assign_index) {
        // search beam_size distances per query
        FAISS_THROW_IF_NOT(assign_index->d == d);
        cent_distances.resize(n * beam_size * new_beam_size);
        cent_ids.resize(n * beam_size * new_beam_size);
        if (assign_index->ntotal != 0) {
            // then we assume the codebooks are already added to the index
            FAISS_THROW_IF_NOT(assign_index->ntotal == K);
        } else {
            assign_index->add(K, cent);
        }

        // printf("beam_search_encode_step -- mem usage %zd\n",
        // get_mem_usage_kb());
        assign_index->search(
                n * beam_size,
                residuals,
                new_beam_size,
                cent_distances.data(),
                cent_ids.data());
    } else {
        // do one big distance computation
        cent_distances.resize(n * beam_size * K);
        pairwise_L2sqr(
                d, n * beam_size, residuals, K, cent, cent_distances.data());
    }
    InterruptCallback::check();

#pragma omp parallel for if (n > 100)
    for (int64_t i = 0; i < n; i++) {
        const int32_t* codes_i = codes + i * m * beam_size;
        int32_t* new_codes_i = new_codes + i * (m + 1) * new_beam_size;
        const float* residuals_i = residuals + i * d * beam_size;
        float* new_residuals_i = new_residuals + i * d * new_beam_size;

        float* new_distances_i = new_distances + i * new_beam_size;
        using C = CMax<float, int>;

        if (assign_index) {
            const float* cent_distances_i =
                    cent_distances.data() + i * beam_size * new_beam_size;
            const idx_t* cent_ids_i =
                    cent_ids.data() + i * beam_size * new_beam_size;

            // here we could be a tad more efficient by merging sorted arrays
            for (int i_2 = 0; i_2 < new_beam_size; i_2++) {
                new_distances_i[i_2] = C::neutral();
            }
            std::vector<int> perm(new_beam_size, -1);
            heap_addn<C>(
                    new_beam_size,
                    new_distances_i,
                    perm.data(),
                    cent_distances_i,
                    nullptr,
                    beam_size * new_beam_size);
            heap_reorder<C>(new_beam_size, new_distances_i, perm.data());

            for (int j = 0; j < new_beam_size; j++) {
                int js = perm[j] / new_beam_size;
                int ls = cent_ids_i[perm[j]];
                if (m > 0) {
                    memcpy(new_codes_i, codes_i + js * m, sizeof(*codes) * m);
                }
                new_codes_i[m] = ls;
                new_codes_i += m + 1;
                fvec_sub(
                        d,
                        residuals_i + js * d,
                        cent + ls * d,
                        new_residuals_i);
                new_residuals_i += d;
            }

        } else {
            const float* cent_distances_i =
                    cent_distances.data() + i * beam_size * K;
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
                fvec_sub(
                        d,
                        residuals_i + js * d,
                        cent + ls * d,
                        new_residuals_i);
                new_residuals_i += d;
            }
        }
    }
}

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
        ApproxTopK_mode_t approx_topk_mode) {
    beam_search_encode_step_tab<SIMDLevel::NONE>(
            K,
            n,
            beam_size,
            codebook_cross_norms,
            ldc,
            codebook_offsets,
            query_cp,
            ldqc,
            cent_norms_i,
            m,
            codes,
            distances,
            new_beam_size,
            new_codes,
            new_distances,
            approx_topk_mode);
}

/********************************************************************
 * Multiple encoding steps
 ********************************************************************/

namespace rq_encode_steps {

void refine_beam_mp(
        const ResidualQuantizer& rq,
        size_t n,
        size_t beam_size,
        const float* x,
        int out_beam_size,
        int32_t* out_codes,
        float* out_residuals,
        float* out_distances,
        RefineBeamMemoryPool& pool) {
    int cur_beam_size = beam_size;

    double t0 = getmillisecs();

    // find the max_beam_size
    int max_beam_size = 0;
    {
        int tmp_beam_size = cur_beam_size;
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
    pool.new_residuals.resize(n * max_beam_size * rq.d);

    pool.codes.resize(n * max_beam_size * (rq.M + 1));
    pool.distances.resize(n * max_beam_size);
    pool.residuals.resize(n * rq.d * max_beam_size);

    for (size_t i = 0; i < n * rq.d * beam_size; i++) {
        pool.residuals[i] = x[i];
    }

    // set up pointers to buffers
    int32_t* __restrict codes_ptr = pool.codes.data();
    float* __restrict residuals_ptr = pool.residuals.data();

    int32_t* __restrict new_codes_ptr = pool.new_codes.data();
    float* __restrict new_residuals_ptr = pool.new_residuals.data();

    // index
    std::unique_ptr<Index> assign_index;
    if (rq.assign_index_factory) {
        assign_index.reset((*rq.assign_index_factory)(rq.d));
    }

    // main loop
    size_t codes_size = 0;
    size_t distances_size = 0;
    size_t residuals_size = 0;

    for (int m = 0; m < rq.M; m++) {
        int K = 1 << rq.nbits[m];

        const float* __restrict codebooks_m =
                rq.codebooks.data() + rq.codebook_offsets[m] * rq.d;

        const int new_beam_size = std::min(cur_beam_size * K, out_beam_size);

        codes_size = n * new_beam_size * (m + 1);
        residuals_size = n * new_beam_size * rq.d;
        distances_size = n * new_beam_size;

        beam_search_encode_step(
                rq.d,
                K,
                codebooks_m,
                n,
                cur_beam_size,
                residuals_ptr,
                m,
                codes_ptr,
                new_beam_size,
                new_codes_ptr,
                new_residuals_ptr,
                pool.distances.data(),
                assign_index.get(),
                rq.approx_topk_mode);

        if (assign_index != nullptr) {
            assign_index->reset();
        }

        std::swap(codes_ptr, new_codes_ptr);
        std::swap(residuals_ptr, new_residuals_ptr);

        cur_beam_size = new_beam_size;

        if (rq.verbose) {
            float sum_distances = 0;
            for (int j = 0; j < distances_size; j++) {
                sum_distances += pool.distances[j];
            }

            printf("[%.3f s] encode stage %d, %d bits, "
                   "total error %g, beam_size %d\n",
                   (getmillisecs() - t0) / 1000,
                   m,
                   int(rq.nbits[m]),
                   sum_distances,
                   cur_beam_size);
        }
    }

    if (out_codes) {
        memcpy(out_codes, codes_ptr, codes_size * sizeof(*codes_ptr));
    }
    if (out_residuals) {
        memcpy(out_residuals,
               residuals_ptr,
               residuals_size * sizeof(*residuals_ptr));
    }
    if (out_distances) {
        memcpy(out_distances,
               pool.distances.data(),
               distances_size * sizeof(pool.distances[0]));
    }
}

// this is for use_beam_LUT == 0
void compute_codes_add_centroids_mp_lut0(
        const ResidualQuantizer& rq,
        const float* x,
        uint8_t* codes_out,
        size_t n,
        const float* centroids,
        ComputeCodesAddCentroidsLUT0MemoryPool& pool) {
    pool.codes.resize(rq.max_beam_size * rq.M * n);
    pool.distances.resize(rq.max_beam_size * n);

    pool.residuals.resize(rq.max_beam_size * n * rq.d);

    refine_beam_mp(
            rq,
            n,
            1,
            x,
            rq.max_beam_size,
            pool.codes.data(),
            pool.residuals.data(),
            pool.distances.data(),
            pool.refine_beam_pool);

    if (rq.search_type == ResidualQuantizer::ST_norm_float ||
        rq.search_type == ResidualQuantizer::ST_norm_qint8 ||
        rq.search_type == ResidualQuantizer::ST_norm_qint4) {
        pool.norms.resize(n);
        // recover the norms of reconstruction as
        // || original_vector - residual ||^2
        for (size_t i = 0; i < n; i++) {
            pool.norms[i] = fvec_L2sqr(
                    x + i * rq.d,
                    pool.residuals.data() + i * rq.max_beam_size * rq.d,
                    rq.d);
        }
    }

    // pack only the first code of the beam
    //   (hence the ld_codes=M * max_beam_size)
    rq.pack_codes(
            n,
            pool.codes.data(),
            codes_out,
            rq.M * rq.max_beam_size,
            (pool.norms.size() > 0) ? pool.norms.data() : nullptr,
            centroids);
}

// use_beam_LUT == 1
void compute_codes_add_centroids_mp_lut1(
        const ResidualQuantizer& rq,
        const float* x,
        uint8_t* codes_out,
        size_t n,
        const float* centroids,
        ComputeCodesAddCentroidsLUT1MemoryPool& pool) {
    compute_codes_add_centroids_mp_lut1<SIMDLevel::NONE>(
            rq, x, codes_out, n, centroids, pool);
}

} // namespace rq_encode_steps

} // namespace faiss
