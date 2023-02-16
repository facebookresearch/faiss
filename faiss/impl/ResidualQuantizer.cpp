/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/ResidualQuantizer.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <memory>

#include <faiss/IndexFlat.h>
#include <faiss/VectorTransform.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/hamming.h>
#include <faiss/utils/utils.h>

#include <faiss/utils/simdlib.h>

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

// http://www.netlib.org/clapack/old/single/sgels.c
// solve least squares

int sgelsd_(
        FINTEGER* m,
        FINTEGER* n,
        FINTEGER* nrhs,
        float* a,
        FINTEGER* lda,
        float* b,
        FINTEGER* ldb,
        float* s,
        float* rcond,
        FINTEGER* rank,
        float* work,
        FINTEGER* lwork,
        FINTEGER* iwork,
        FINTEGER* info);
}

namespace faiss {

ResidualQuantizer::ResidualQuantizer() {
    d = 0;
    M = 0;
    verbose = false;
}

ResidualQuantizer::ResidualQuantizer(
        size_t d,
        const std::vector<size_t>& nbits,
        Search_type_t search_type)
        : ResidualQuantizer() {
    this->search_type = search_type;
    this->d = d;
    M = nbits.size();
    this->nbits = nbits;
    set_derived_values();
}

ResidualQuantizer::ResidualQuantizer(
        size_t d,
        size_t M,
        size_t nbits,
        Search_type_t search_type)
        : ResidualQuantizer(d, std::vector<size_t>(M, nbits), search_type) {}

void ResidualQuantizer::initialize_from(
        const ResidualQuantizer& other,
        int skip_M) {
    FAISS_THROW_IF_NOT(M + skip_M <= other.M);
    FAISS_THROW_IF_NOT(skip_M >= 0);

    Search_type_t this_search_type = search_type;
    int this_M = M;

    // a first good approximation: override everything
    *this = other;

    // adjust derived values
    M = this_M;
    search_type = this_search_type;
    nbits.resize(M);
    memcpy(nbits.data(),
           other.nbits.data() + skip_M,
           nbits.size() * sizeof(nbits[0]));

    set_derived_values();

    // resize codebooks if trained
    if (codebooks.size() > 0) {
        FAISS_THROW_IF_NOT(codebooks.size() == other.total_codebook_size * d);
        codebooks.resize(total_codebook_size * d);
        memcpy(codebooks.data(),
               other.codebooks.data() + other.codebook_offsets[skip_M] * d,
               codebooks.size() * sizeof(codebooks[0]));
        // TODO: norm_tabs?
    }
}

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
            for (int i = 0; i < new_beam_size; i++) {
                new_distances_i[i] = C::neutral();
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
            for (int i = 0; i < new_beam_size; i++) {
                new_distances_i[i] = C::neutral();
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

void ResidualQuantizer::train(size_t n, const float* x) {
    codebooks.resize(d * codebook_offsets.back());

    if (verbose) {
        printf("Training ResidualQuantizer, with %zd steps on %zd %zdD vectors\n",
               M,
               n,
               size_t(d));
    }

    int cur_beam_size = 1;
    std::vector<float> residuals(x, x + n * d);
    std::vector<int32_t> codes;
    std::vector<float> distances;
    double t0 = getmillisecs();
    double clustering_time = 0;

    for (int m = 0; m < M; m++) {
        int K = 1 << nbits[m];

        // on which residuals to train
        std::vector<float>& train_residuals = residuals;
        std::vector<float> residuals1;
        if (train_type & Train_top_beam) {
            residuals1.resize(n * d);
            for (size_t j = 0; j < n; j++) {
                memcpy(residuals1.data() + j * d,
                       residuals.data() + j * d * cur_beam_size,
                       sizeof(residuals[0]) * d);
            }
            train_residuals = residuals1;
        }
        std::vector<float> codebooks;
        float obj = 0;

        std::unique_ptr<Index> assign_index;
        if (assign_index_factory) {
            assign_index.reset((*assign_index_factory)(d));
        } else {
            assign_index.reset(new IndexFlatL2(d));
        }

        double t1 = getmillisecs();

        if (!(train_type & Train_progressive_dim)) { // regular kmeans
            Clustering clus(d, K, cp);
            clus.train(
                    train_residuals.size() / d,
                    train_residuals.data(),
                    *assign_index.get());
            codebooks.swap(clus.centroids);
            assign_index->reset();
            obj = clus.iteration_stats.back().obj;
        } else { // progressive dim clustering
            ProgressiveDimClustering clus(d, K, cp);
            ProgressiveDimIndexFactory default_fac;
            clus.train(
                    train_residuals.size() / d,
                    train_residuals.data(),
                    assign_index_factory ? *assign_index_factory : default_fac);
            codebooks.swap(clus.centroids);
            obj = clus.iteration_stats.back().obj;
        }
        clustering_time += (getmillisecs() - t1) / 1000;

        memcpy(this->codebooks.data() + codebook_offsets[m] * d,
               codebooks.data(),
               codebooks.size() * sizeof(codebooks[0]));

        // quantize using the new codebooks

        int new_beam_size = std::min(cur_beam_size * K, max_beam_size);
        std::vector<int32_t> new_codes(n * new_beam_size * (m + 1));
        std::vector<float> new_residuals(n * new_beam_size * d);
        std::vector<float> new_distances(n * new_beam_size);

        size_t bs;
        { // determine batch size
            size_t mem = memory_per_point();
            if (n > 1 && mem * n > max_mem_distances) {
                // then split queries to reduce temp memory
                bs = std::max(max_mem_distances / mem, size_t(1));
            } else {
                bs = n;
            }
        }

        for (size_t i0 = 0; i0 < n; i0 += bs) {
            size_t i1 = std::min(i0 + bs, n);

            /* printf("i0: %ld i1: %ld K %d ntotal assign index %ld\n",
                i0, i1, K, assign_index->ntotal); */

            beam_search_encode_step(
                    d,
                    K,
                    codebooks.data(),
                    i1 - i0,
                    cur_beam_size,
                    residuals.data() + i0 * cur_beam_size * d,
                    m,
                    codes.data() + i0 * cur_beam_size * m,
                    new_beam_size,
                    new_codes.data() + i0 * new_beam_size * (m + 1),
                    new_residuals.data() + i0 * new_beam_size * d,
                    new_distances.data() + i0 * new_beam_size,
                    assign_index.get(),
                    approx_topk_mode);
        }
        codes.swap(new_codes);
        residuals.swap(new_residuals);
        distances.swap(new_distances);

        float sum_distances = 0;
        for (int j = 0; j < distances.size(); j++) {
            sum_distances += distances[j];
        }

        if (verbose) {
            printf("[%.3f s, %.3f s clustering] train stage %d, %d bits, kmeans objective %g, "
                   "total distance %g, beam_size %d->%d (batch size %zd)\n",
                   (getmillisecs() - t0) / 1000,
                   clustering_time,
                   m,
                   int(nbits[m]),
                   obj,
                   sum_distances,
                   cur_beam_size,
                   new_beam_size,
                   bs);
        }
        cur_beam_size = new_beam_size;
    }

    is_trained = true;

    if (train_type & Train_refine_codebook) {
        for (int iter = 0; iter < niter_codebook_refine; iter++) {
            if (verbose) {
                printf("re-estimating the codebooks to minimize "
                       "quantization errors (iter %d).\n",
                       iter);
            }
            retrain_AQ_codebook(n, x);
        }
    }

    // find min and max norms
    std::vector<float> norms(n);

    for (size_t i = 0; i < n; i++) {
        norms[i] = fvec_L2sqr(
                x + i * d, residuals.data() + i * cur_beam_size * d, d);
    }

    // fvec_norms_L2sqr(norms.data(), x, d, n);
    train_norm(n, norms.data());

    if (!(train_type & Skip_codebook_tables)) {
        compute_codebook_tables();
    }
}

float ResidualQuantizer::retrain_AQ_codebook(size_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(n >= total_codebook_size, "too few training points");

    if (verbose) {
        printf("  encoding %zd training vectors\n", n);
    }
    std::vector<uint8_t> codes(n * code_size);
    compute_codes(x, codes.data(), n);

    // compute reconstruction error
    float input_recons_error;
    {
        std::vector<float> x_recons(n * d);
        decode(codes.data(), x_recons.data(), n);
        input_recons_error = fvec_L2sqr(x, x_recons.data(), n * d);
        if (verbose) {
            printf("  input quantization error %g\n", input_recons_error);
        }
    }

    // build matrix of the linear system
    std::vector<float> C(n * total_codebook_size);
    for (size_t i = 0; i < n; i++) {
        BitstringReader bsr(codes.data() + i * code_size, code_size);
        for (int m = 0; m < M; m++) {
            int idx = bsr.read(nbits[m]);
            C[i + (codebook_offsets[m] + idx) * n] = 1;
        }
    }

    // transpose training vectors
    std::vector<float> xt(n * d);

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < d; j++) {
            xt[j * n + i] = x[i * d + j];
        }
    }

    { // solve least squares
        FINTEGER lwork = -1;
        FINTEGER di = d, ni = n, tcsi = total_codebook_size;
        FINTEGER info = -1, rank = -1;

        float rcond = 1e-4; // this is an important parameter because the code
                            // matrix can be rank deficient for small problems,
                            // the default rcond=-1 does not work
        float worksize;
        std::vector<float> sing_vals(total_codebook_size);
        FINTEGER nlvl = 1000; // formula is a bit convoluted so let's take an
                              // upper bound
        std::vector<FINTEGER> iwork(
                3 * total_codebook_size * nlvl + 11 * total_codebook_size);

        // worksize query
        sgelsd_(&ni,
                &tcsi,
                &di,
                C.data(),
                &ni,
                xt.data(),
                &ni,
                sing_vals.data(),
                &rcond,
                &rank,
                &worksize,
                &lwork,
                iwork.data(),
                &info);
        FAISS_THROW_IF_NOT(info == 0);

        lwork = worksize;
        std::vector<float> work(lwork);
        // actual call
        sgelsd_(&ni,
                &tcsi,
                &di,
                C.data(),
                &ni,
                xt.data(),
                &ni,
                sing_vals.data(),
                &rcond,
                &rank,
                work.data(),
                &lwork,
                iwork.data(),
                &info);
        FAISS_THROW_IF_NOT_FMT(info == 0, "SGELS returned info=%d", int(info));
        if (verbose) {
            printf("   sgelsd rank=%d/%d\n",
                   int(rank),
                   int(total_codebook_size));
        }
    }

    // result is in xt, re-transpose to codebook

    for (size_t i = 0; i < total_codebook_size; i++) {
        for (size_t j = 0; j < d; j++) {
            codebooks[i * d + j] = xt[j * n + i];
            FAISS_THROW_IF_NOT(std::isfinite(codebooks[i * d + j]));
        }
    }

    float output_recons_error = 0;
    for (size_t j = 0; j < d; j++) {
        output_recons_error += fvec_norm_L2sqr(
                xt.data() + total_codebook_size + n * j,
                n - total_codebook_size);
    }
    if (verbose) {
        printf("  output quantization error %g\n", output_recons_error);
    }
    return output_recons_error;
}

size_t ResidualQuantizer::memory_per_point(int beam_size) const {
    if (beam_size < 0) {
        beam_size = max_beam_size;
    }
    size_t mem;
    mem = beam_size * d * 2 * sizeof(float); // size for 2 beams at a time
    mem += beam_size * beam_size *
            (sizeof(float) + sizeof(idx_t)); // size for 1 beam search result
    return mem;
}

// a namespace full of preallocated buffers
namespace {

// Preallocated memory chunk for refine_beam_mp() call
struct RefineBeamMemoryPool {
    std::vector<int32_t> new_codes;
    std::vector<float> new_residuals;

    std::vector<float> residuals;
    std::vector<int32_t> codes;
    std::vector<float> distances;
};

// Preallocated memory chunk for refine_beam_LUT_mp() call
struct RefineBeamLUTMemoryPool {
    std::vector<int32_t> new_codes;
    std::vector<float> new_distances;

    std::vector<int32_t> codes;
    std::vector<float> distances;
};

// this is for use_beam_LUT == 0 in compute_codes_add_centroids_mp_lut0() call
struct ComputeCodesAddCentroidsLUT0MemoryPool {
    std::vector<int32_t> codes;
    std::vector<float> norms;
    std::vector<float> distances;
    std::vector<float> residuals;
    RefineBeamMemoryPool refine_beam_pool;
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

} // namespace

// forward declaration
void refine_beam_mp(
        const ResidualQuantizer& rq,
        size_t n,
        size_t beam_size,
        const float* x,
        int out_beam_size,
        int32_t* out_codes,
        float* out_residuals,
        float* out_distances,
        RefineBeamMemoryPool& pool);

// forward declaration
void refine_beam_LUT_mp(
        const ResidualQuantizer& rq,
        size_t n,
        const float* query_norms, // size n
        const float* query_cp,    //
        int out_beam_size,
        int32_t* out_codes,
        float* out_distances,
        RefineBeamLUTMemoryPool& pool);

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
    //
    pool.codes.resize(rq.max_beam_size * rq.M * n);
    pool.distances.resize(rq.max_beam_size * n);

    FAISS_THROW_IF_NOT_MSG(
            rq.codebook_cross_products.size() ==
                    rq.total_codebook_size * rq.total_codebook_size,
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

    refine_beam_LUT_mp(
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

void ResidualQuantizer::compute_codes_add_centroids(
        const float* x,
        uint8_t* codes_out,
        size_t n,
        const float* centroids) const {
    FAISS_THROW_IF_NOT_MSG(is_trained, "RQ is not trained yet.");

    //
    size_t mem = memory_per_point();

    size_t bs = max_mem_distances / mem;
    if (bs == 0) {
        bs = 1; // otherwise we can't do much
    }

    // prepare memory pools
    ComputeCodesAddCentroidsLUT0MemoryPool pool0;
    ComputeCodesAddCentroidsLUT1MemoryPool pool1;

    for (size_t i0 = 0; i0 < n; i0 += bs) {
        size_t i1 = std::min(n, i0 + bs);
        const float* cent = nullptr;
        if (centroids != nullptr) {
            cent = centroids + i0 * d;
        }

        // compute_codes_add_centroids(
        //   x + i0 * d,
        //   codes_out + i0 * code_size,
        //   i1 - i0,
        //   cent);
        if (use_beam_LUT == 0) {
            compute_codes_add_centroids_mp_lut0(
                    *this,
                    x + i0 * d,
                    codes_out + i0 * code_size,
                    i1 - i0,
                    cent,
                    pool0);
        } else if (use_beam_LUT == 1) {
            compute_codes_add_centroids_mp_lut1(
                    *this,
                    x + i0 * d,
                    codes_out + i0 * code_size,
                    i1 - i0,
                    cent,
                    pool1);
        }
    }
}

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
    } else {
        assign_index.reset(new IndexFlatL2(rq.d));
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
                // residuals.data(),
                residuals_ptr,
                m,
                // codes.data(),
                codes_ptr,
                new_beam_size,
                // new_codes.data(),
                new_codes_ptr,
                // new_residuals.data(),
                new_residuals_ptr,
                pool.distances.data(),
                assign_index.get(),
                rq.approx_topk_mode);

        assign_index->reset();

        std::swap(codes_ptr, new_codes_ptr);
        std::swap(residuals_ptr, new_residuals_ptr);

        cur_beam_size = new_beam_size;

        if (rq.verbose) {
            float sum_distances = 0;
            // for (int j = 0; j < distances.size(); j++) {
            //     sum_distances += distances[j];
            // }
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
        // memcpy(out_codes, codes.data(), codes.size() * sizeof(codes[0]));
        memcpy(out_codes, codes_ptr, codes_size * sizeof(*codes_ptr));
    }
    if (out_residuals) {
        // memcpy(out_residuals,
        //        residuals.data(),
        //        residuals.size() * sizeof(residuals[0]));
        memcpy(out_residuals,
               residuals_ptr,
               residuals_size * sizeof(*residuals_ptr));
    }
    if (out_distances) {
        // memcpy(out_distances,
        //        distances.data(),
        //        distances.size() * sizeof(distances[0]));
        memcpy(out_distances,
               pool.distances.data(),
               distances_size * sizeof(pool.distances[0]));
    }
}

void ResidualQuantizer::refine_beam(
        size_t n,
        size_t beam_size,
        const float* x,
        int out_beam_size,
        int32_t* out_codes,
        float* out_residuals,
        float* out_distances) const {
    RefineBeamMemoryPool pool;
    refine_beam_mp(
            *this,
            n,
            beam_size,
            x,
            out_beam_size,
            out_codes,
            out_residuals,
            out_distances,
            pool);
}

/*******************************************************************
 * Functions using the dot products between codebook entries
 *******************************************************************/

void ResidualQuantizer::compute_codebook_tables() {
    codebook_cross_products.resize(total_codebook_size * total_codebook_size);
    cent_norms.resize(total_codebook_size);
    // stricly speaking we could use ssyrk
    {
        FINTEGER ni = total_codebook_size;
        FINTEGER di = d;
        float zero = 0, one = 1;
        sgemm_("Transposed",
               "Not transposed",
               &ni,
               &ni,
               &di,
               &one,
               codebooks.data(),
               &di,
               codebooks.data(),
               &di,
               &zero,
               codebook_cross_products.data(),
               &ni);
    }
    for (size_t i = 0; i < total_codebook_size; i++) {
        cent_norms[i] = codebook_cross_products[i + i * total_codebook_size];
    }
}

namespace {

template <size_t M, size_t NK>
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
        output[b * K + kk] = reg;
    }
}

template <size_t M, size_t NK>
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
    //   this manual SIMD unrolling might be unneeded.
#if defined(__AVX2__) || defined(__aarch64__)
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
        output[b * K + kk] += reg;
    }
}

template <size_t M, size_t NK>
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

} // namespace

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
        int32_t* new_codes,                 // n * new_beam_size * (m + 1)
        float* new_distances,               // n * new_beam_size
        ApproxTopK_mode_t approx_topk_mode) //
{
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

        /*
        // This is the baseline implementation. Its primary flaw
        //   that it writes way too many info to the temporary buffer
        //   called dp.
        //
        // This baseline code is kept intentionally because it is easy to
        // understand what an optimized version optimizes exactly.
        //
        for (size_t b = 0; b < beam_size; b++) {
            std::vector<float> dp(K);

            for (size_t m1 = 0; m1 < m; m1++) {
                size_t c = codes_i[b * m + m1];
                const float* cb =
                        &codebook_cross_norms[(codebook_offsets[m1] + c) * ldc];
                fvec_add(K, cb, dp.data(), dp.data());
            }

            for (size_t k = 0; k < K; k++) {
                cent_distances[b * K + k] =
                        distances_i[b] + cd_common[k] + 2 * dp[k];
            }
        }
        */

        // An optimized implementation that avoids using a temporary buffer
        // and does the accumulation in registers.

        // Compute a sum of NK AQ codes.
#define ACCUM_AND_FINALIZE_TAB(NK)               \
    case NK:                                     \
        for (size_t b = 0; b < beam_size; b++) { \
            accum_and_finalize_tab<NK, 4>(       \
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
                // registers. But we'll try to accumulate up to 8 AQ codes in
                // registers and issue a single write operation to the buffer,
                // while the baseline does no accumulation. So, the number of
                // write operations to the temporary buffer is reduced 8x.

                // allocate a temporary buffer
                std::vector<float> dp(K);

                for (size_t b = 0; b < beam_size; b++) {
                    // Initialize it. Compute a sum of first 8 AQ codes
                    // because m >= 8 .
                    accum_and_store_tab<8, 4>(
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
        accum_and_add_tab<NK, 4>(      \
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

        using C = CMax<float, int>;
        int32_t* new_codes_i = new_codes + i * (m + 1) * new_beam_size;
        float* new_distances_i = new_distances + i * new_beam_size;

        const float* cent_distances_i = cent_distances.data();

        // then we have to select the best results
        for (int i = 0; i < new_beam_size; i++) {
            new_distances_i[i] = C::neutral();
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

//
void refine_beam_LUT_mp(
        const ResidualQuantizer& rq,
        size_t n,
        const float* query_norms, // size n
        const float* query_cp,    //
        int out_beam_size,
        int32_t* out_codes,
        float* out_distances,
        RefineBeamLUTMemoryPool& pool) {
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
    for (int m = 0; m < rq.M; m++) {
        int K = 1 << rq.nbits[m];

        // it is guaranteed that (new_beam_size <= than max_beam_size) ==
        // true
        int new_beam_size = std::min(beam_size * K, out_beam_size);

        // std::vector<int32_t> new_codes(n * new_beam_size * (m + 1));
        // std::vector<float> new_distances(n * new_beam_size);

        codes_size = n * new_beam_size * (m + 1);
        distances_size = n * new_beam_size;

        beam_search_encode_step_tab(
                K,
                n,
                beam_size,
                rq.codebook_cross_products.data() + rq.codebook_offsets[m],
                rq.total_codebook_size,
                rq.codebook_offsets.data(),
                query_cp + rq.codebook_offsets[m],
                rq.total_codebook_size,
                rq.cent_norms.data() + rq.codebook_offsets[m],
                m,
                // codes.data(),
                codes_ptr,
                // distances.data(),
                distances_ptr,
                new_beam_size,
                // new_codes.data(),
                new_codes_ptr,
                // new_distances.data()
                new_distances_ptr,
                rq.approx_topk_mode);

        // codes.swap(new_codes);
        std::swap(codes_ptr, new_codes_ptr);
        // distances.swap(new_distances);
        std::swap(distances_ptr, new_distances_ptr);

        beam_size = new_beam_size;

        if (rq.verbose) {
            float sum_distances = 0;
            // for (int j = 0; j < distances.size(); j++) {
            //     sum_distances += distances[j];
            // }
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
        // memcpy(out_codes, codes.data(), codes.size() * sizeof(codes[0]));
        memcpy(out_codes, codes_ptr, codes_size * sizeof(*codes_ptr));
    }
    if (out_distances) {
        // memcpy(out_distances,
        //        distances.data(),
        //        distances.size() * sizeof(distances[0]));
        memcpy(out_distances,
               distances_ptr,
               distances_size * sizeof(*distances_ptr));
    }
}

void ResidualQuantizer::refine_beam_LUT(
        size_t n,
        const float* query_norms, // size n
        const float* query_cp,    //
        int out_beam_size,
        int32_t* out_codes,
        float* out_distances) const {
    RefineBeamLUTMemoryPool pool;
    refine_beam_LUT_mp(
            *this,
            n,
            query_norms,
            query_cp,
            out_beam_size,
            out_codes,
            out_distances,
            pool);
}

} // namespace faiss
