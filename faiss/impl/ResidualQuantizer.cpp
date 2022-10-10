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

ResidualQuantizer::ResidualQuantizer()
        : train_type(Train_progressive_dim),
          niter_codebook_refine(5),
          max_beam_size(5),
          use_beam_LUT(0),
          assign_index_factory(nullptr) {
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
        Index* assign_index) {
    // we have to fill in the whole output matrix
    FAISS_THROW_IF_NOT(new_beam_size <= beam_size * K);

    using idx_t = Index::idx_t;

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
            heap_addn<C>(
                    new_beam_size,
                    new_distances_i,
                    perm.data(),
                    cent_distances_i,
                    nullptr,
                    beam_size * K);
            heap_reorder<C>(new_beam_size, new_distances_i, perm.data());

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
                    assign_index.get());
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
            (sizeof(float) +
             sizeof(Index::idx_t)); // size for 1 beam search result
    return mem;
}

void ResidualQuantizer::compute_codes_add_centroids(
        const float* x,
        uint8_t* codes_out,
        size_t n,
        const float* centroids) const {
    FAISS_THROW_IF_NOT_MSG(is_trained, "RQ is not trained yet.");

    size_t mem = memory_per_point();
    if (n > 1 && mem * n > max_mem_distances) {
        // then split queries to reduce temp memory
        size_t bs = max_mem_distances / mem;
        if (bs == 0) {
            bs = 1; // otherwise we can't do much
        }
        for (size_t i0 = 0; i0 < n; i0 += bs) {
            size_t i1 = std::min(n, i0 + bs);
            const float* cent = nullptr;
            if (centroids != nullptr) {
                cent = centroids + i0 * d;
            }
            compute_codes_add_centroids(
                    x + i0 * d, codes_out + i0 * code_size, i1 - i0, cent);
        }
        return;
    }

    std::vector<int32_t> codes(max_beam_size * M * n);
    std::vector<float> norms;
    std::vector<float> distances(max_beam_size * n);

    if (use_beam_LUT == 0) {
        std::vector<float> residuals(max_beam_size * n * d);

        refine_beam(
                n,
                1,
                x,
                max_beam_size,
                codes.data(),
                residuals.data(),
                distances.data());

        if (search_type == ST_norm_float || search_type == ST_norm_qint8 ||
            search_type == ST_norm_qint4) {
            norms.resize(n);
            // recover the norms of reconstruction as
            // || original_vector - residual ||^2
            for (size_t i = 0; i < n; i++) {
                norms[i] = fvec_L2sqr(
                        x + i * d, residuals.data() + i * max_beam_size * d, d);
            }
        }
    } else if (use_beam_LUT == 1) {
        FAISS_THROW_IF_NOT_MSG(
                codebook_cross_products.size() ==
                        total_codebook_size * total_codebook_size,
                "call compute_codebook_tables first");

        std::vector<float> query_norms(n);
        fvec_norms_L2sqr(query_norms.data(), x, d, n);

        std::vector<float> query_cp(n * total_codebook_size);
        {
            FINTEGER ti = total_codebook_size, di = d, ni = n;
            float zero = 0, one = 1;
            sgemm_("Transposed",
                   "Not transposed",
                   &ti,
                   &ni,
                   &di,
                   &one,
                   codebooks.data(),
                   &di,
                   x,
                   &di,
                   &zero,
                   query_cp.data(),
                   &ti);
        }

        refine_beam_LUT(
                n,
                query_norms.data(),
                query_cp.data(),
                max_beam_size,
                codes.data(),
                distances.data());
    }
    // pack only the first code of the beam (hence the ld_codes=M *
    // max_beam_size)
    pack_codes(
            n,
            codes.data(),
            codes_out,
            M * max_beam_size,
            norms.size() > 0 ? norms.data() : nullptr,
            centroids);
}

void ResidualQuantizer::refine_beam(
        size_t n,
        size_t beam_size,
        const float* x,
        int out_beam_size,
        int32_t* out_codes,
        float* out_residuals,
        float* out_distances) const {
    int cur_beam_size = beam_size;

    std::vector<float> residuals(x, x + n * d * beam_size);
    std::vector<int32_t> codes;
    std::vector<float> distances;
    double t0 = getmillisecs();

    std::unique_ptr<Index> assign_index;
    if (assign_index_factory) {
        assign_index.reset((*assign_index_factory)(d));
    } else {
        assign_index.reset(new IndexFlatL2(d));
    }

    for (int m = 0; m < M; m++) {
        int K = 1 << nbits[m];

        const float* codebooks_m =
                this->codebooks.data() + codebook_offsets[m] * d;

        int new_beam_size = std::min(cur_beam_size * K, out_beam_size);

        std::vector<int32_t> new_codes(n * new_beam_size * (m + 1));
        std::vector<float> new_residuals(n * new_beam_size * d);
        distances.resize(n * new_beam_size);

        beam_search_encode_step(
                d,
                K,
                codebooks_m,
                n,
                cur_beam_size,
                residuals.data(),
                m,
                codes.data(),
                new_beam_size,
                new_codes.data(),
                new_residuals.data(),
                distances.data(),
                assign_index.get());

        assign_index->reset();

        codes.swap(new_codes);
        residuals.swap(new_residuals);

        cur_beam_size = new_beam_size;

        if (verbose) {
            float sum_distances = 0;
            for (int j = 0; j < distances.size(); j++) {
                sum_distances += distances[j];
            }
            printf("[%.3f s] encode stage %d, %d bits, "
                   "total error %g, beam_size %d\n",
                   (getmillisecs() - t0) / 1000,
                   m,
                   int(nbits[m]),
                   sum_distances,
                   cur_beam_size);
        }
    }

    if (out_codes) {
        memcpy(out_codes, codes.data(), codes.size() * sizeof(codes[0]));
    }
    if (out_residuals) {
        memcpy(out_residuals,
               residuals.data(),
               residuals.size() * sizeof(residuals[0]));
    }
    if (out_distances) {
        memcpy(out_distances,
               distances.data(),
               distances.size() * sizeof(distances[0]));
    }
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
        float* new_distances) // n * new_beam_size
{
    FAISS_THROW_IF_NOT(ldc >= K);

#pragma omp parallel for if (n > 100)
    for (int64_t i = 0; i < n; i++) {
        std::vector<float> cent_distances(beam_size * K);
        std::vector<float> cd_common(K);

        const int32_t* codes_i = codes + i * m * beam_size;
        const float* query_cp_i = query_cp + i * ldqc;
        const float* distances_i = distances + i * beam_size;

        for (size_t k = 0; k < K; k++) {
            cd_common[k] = cent_norms_i[k] - 2 * query_cp_i[k];
        }

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

        using C = CMax<float, int>;
        int32_t* new_codes_i = new_codes + i * (m + 1) * new_beam_size;
        float* new_distances_i = new_distances + i * new_beam_size;

        const float* cent_distances_i = cent_distances.data();

        // then we have to select the best results
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
                beam_size * K);
        heap_reorder<C>(new_beam_size, new_distances_i, perm.data());

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

void ResidualQuantizer::refine_beam_LUT(
        size_t n,
        const float* query_norms, // size n
        const float* query_cp,    //
        int out_beam_size,
        int32_t* out_codes,
        float* out_distances) const {
    int beam_size = 1;

    std::vector<int32_t> codes;
    std::vector<float> distances(query_norms, query_norms + n);
    double t0 = getmillisecs();

    for (int m = 0; m < M; m++) {
        int K = 1 << nbits[m];

        int new_beam_size = std::min(beam_size * K, out_beam_size);
        std::vector<int32_t> new_codes(n * new_beam_size * (m + 1));
        std::vector<float> new_distances(n * new_beam_size);

        beam_search_encode_step_tab(
                K,
                n,
                beam_size,
                codebook_cross_products.data() + codebook_offsets[m],
                total_codebook_size,
                codebook_offsets.data(),
                query_cp + codebook_offsets[m],
                total_codebook_size,
                cent_norms.data() + codebook_offsets[m],
                m,
                codes.data(),
                distances.data(),
                new_beam_size,
                new_codes.data(),
                new_distances.data());

        codes.swap(new_codes);
        distances.swap(new_distances);
        beam_size = new_beam_size;

        if (verbose) {
            float sum_distances = 0;
            for (int j = 0; j < distances.size(); j++) {
                sum_distances += distances[j];
            }
            printf("[%.3f s] encode stage %d, %d bits, "
                   "total error %g, beam_size %d\n",
                   (getmillisecs() - t0) / 1000,
                   m,
                   int(nbits[m]),
                   sum_distances,
                   beam_size);
        }
    }

    if (out_codes) {
        memcpy(out_codes, codes.data(), codes.size() * sizeof(codes[0]));
    }
    if (out_distances) {
        memcpy(out_distances,
               distances.data(),
               distances.size() * sizeof(distances[0]));
    }
}

} // namespace faiss
