/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include "faiss/impl/ResidualQuantizer.h"
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ResidualQuantizer.h>
#include "faiss/utils/utils.h"

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <memory>

#include <algorithm>

#include <faiss/IndexFlat.h>
#include <faiss/VectorTransform.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/hamming.h>
#include <faiss/utils/utils.h>

namespace faiss {

ResidualQuantizer::ResidualQuantizer()
        : train_type(Train_progressive_dim),
          max_beam_size(30),
          max_mem_distances(5 * (size_t(1) << 30)), // 5 GiB
          assign_index_factory(nullptr) {
    d = 0;
    M = 0;
    verbose = false;
}

ResidualQuantizer::ResidualQuantizer(size_t d, const std::vector<size_t>& nbits)
        : ResidualQuantizer() {
    this->d = d;
    M = nbits.size();
    this->nbits = nbits;
    set_derived_values();
}

ResidualQuantizer::ResidualQuantizer(size_t d, size_t M, size_t nbits)
        : ResidualQuantizer(d, std::vector<size_t>(M, nbits)) {}

namespace {

void fvec_sub(size_t d, const float* a, const float* b, float* c) {
    for (size_t i = 0; i < d; i++) {
        c[i] = a[i] - b[i];
    }
}

} // anonymous namespace

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
            FAISS_THROW_IF_NOT(assign_index->ntotal != K);
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
        train_type_t tt = train_type_t(train_type & ~Train_top_beam);

        std::vector<float> codebooks;
        float obj = 0;

        std::unique_ptr<Index> assign_index;
        if (assign_index_factory) {
            assign_index.reset((*assign_index_factory)(d));
        } else {
            assign_index.reset(new IndexFlatL2(d));
        }
        if (tt == Train_default) {
            Clustering clus(d, K, cp);
            clus.train(
                    train_residuals.size() / d,
                    train_residuals.data(),
                    *assign_index.get());
            codebooks.swap(clus.centroids);
            assign_index->reset();
            obj = clus.iteration_stats.back().obj;
        } else if (tt == Train_progressive_dim) {
            ProgressiveDimClustering clus(d, K, cp);
            ProgressiveDimIndexFactory default_fac;
            clus.train(
                    train_residuals.size() / d,
                    train_residuals.data(),
                    assign_index_factory ? *assign_index_factory : default_fac);
            codebooks.swap(clus.centroids);
            obj = clus.iteration_stats.back().obj;
        } else {
            FAISS_THROW_MSG("train type not supported");
        }

        memcpy(this->codebooks.data() + codebook_offsets[m] * d,
               codebooks.data(),
               codebooks.size() * sizeof(codebooks[0]));

        // quantize using the new codebooks

        int new_beam_size = std::min(cur_beam_size * K, max_beam_size);
        std::vector<int32_t> new_codes(n * new_beam_size * (m + 1));
        std::vector<float> new_residuals(n * new_beam_size * d);
        std::vector<float> new_distances(n * new_beam_size);

        beam_search_encode_step(
                d,
                K,
                codebooks.data(),
                n,
                cur_beam_size,
                residuals.data(),
                m,
                codes.data(),
                new_beam_size,
                new_codes.data(),
                new_residuals.data(),
                new_distances.data(),
                assign_index.get());

        codes.swap(new_codes);
        residuals.swap(new_residuals);
        distances.swap(new_distances);

        float sum_distances = 0;
        for (int j = 0; j < distances.size(); j++) {
            sum_distances += distances[j];
        }

        if (verbose) {
            printf("[%.3f s] train stage %d, %d bits, kmeans objective %g, "
                   "total distance %g, beam_size %d->%d\n",
                   (getmillisecs() - t0) / 1000,
                   m,
                   int(nbits[m]),
                   obj,
                   sum_distances,
                   cur_beam_size,
                   new_beam_size);
        }
        cur_beam_size = new_beam_size;
    }

    is_trained = true;
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

void ResidualQuantizer::compute_codes(
        const float* x,
        uint8_t* codes_out,
        size_t n) const {
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
            compute_codes(x + i0 * d, codes_out + i0 * code_size, i1 - i0);
        }
        return;
    }

    std::vector<float> residuals(max_beam_size * n * d);
    std::vector<int32_t> codes(max_beam_size * M * n);
    std::vector<float> distances(max_beam_size * n);

    refine_beam(
            n,
            1,
            x,
            max_beam_size,
            codes.data(),
            residuals.data(),
            distances.data());

    // pack only the first code of the beam (hence the ld_codes=M *
    // max_beam_size)
    pack_codes(n, codes.data(), codes_out, M * max_beam_size);
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

} // namespace faiss
