/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ResidualQuantizer.h>

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <memory>

#include <algorithm>

#include <faiss/IndexFlat.h>
#include <faiss/VectorTransform.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/hamming.h>
#include <faiss/utils/utils.h>

namespace faiss {

ResidualQuantizer::ResidualQuantizer()
        : d(0),
          M(0),
          verbose(false),
          train_type(Train_progressive_dim),
          max_beam_size(30),
          max_mem_distances(5 * (size_t(1) << 30)), // 5 GiB
          assign_index_factory(nullptr) {}

ResidualQuantizer::ResidualQuantizer(size_t d, const std::vector<size_t>& nbits)
        : ResidualQuantizer() {
    this->d = d;
    M = nbits.size();
    this->nbits = nbits;
    set_derived_values();
}

ResidualQuantizer::ResidualQuantizer(size_t d, size_t M, size_t nbits)
        : ResidualQuantizer(d, std::vector<size_t>(M, nbits)) {}

void ResidualQuantizer::set_derived_values() {
    code_size = 0;
    is_byte_aligned = true;
    centroid_offsets.resize(M + 1, 0);
    for (int i = 0; i < M; i++) {
        int nbit = nbits[i];
        size_t k = 1 << nbit;
        centroid_offsets[i + 1] = centroid_offsets[i] + k;
        code_size += nbit;
        if (nbit % 8 != 0) {
            is_byte_aligned = false;
        }
    }
    // convert bits to bytes
    code_size = (code_size + 7) / 8;
}

namespace {

void fvec_sub(size_t d, const float* a, const float* b, float* c) {
    for (size_t i = 0; i < d; i++) {
        c[i] = a[i] - b[i];
    }
}

// c and a and b can overlap
void fvec_add(size_t d, const float* a, const float* b, float* c) {
    for (size_t i = 0; i < d; i++) {
        c[i] = a[i] + b[i];
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
        if (assign_index->ntotal != 0) {
            // then we assume the centroids are already added to the index
            FAISS_THROW_IF_NOT(assign_index->ntotal != K);
        } else {
            assign_index->add(K, cent);
        }
        cent_ids.resize(n * beam_size * new_beam_size);
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
    centroids.resize(d * centroid_offsets.back());

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

        std::vector<float> centroids;
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
            centroids.swap(clus.centroids);
            assign_index->reset();
            obj = clus.iteration_stats.back().obj;
        } else if (tt == Train_progressive_dim) {
            ProgressiveDimClustering clus(d, K, cp);
            ProgressiveDimIndexFactory default_fac;
            clus.train(
                    train_residuals.size() / d,
                    train_residuals.data(),
                    assign_index_factory ? *assign_index_factory : default_fac);
            centroids.swap(clus.centroids);
            obj = clus.iteration_stats.back().obj;
        } else {
            FAISS_THROW_MSG("train type not supported");
        }

        memcpy(this->centroids.data() + centroid_offsets[m] * d,
               centroids.data(),
               centroids.size() * sizeof(centroids[0]));

        // quantize using the new centroids

        int new_beam_size = std::min(cur_beam_size * K, max_beam_size);
        std::vector<int32_t> new_codes(n * new_beam_size * (m + 1));
        std::vector<float> new_residuals(n * new_beam_size * d);
        std::vector<float> new_distances(n * new_beam_size);

        beam_search_encode_step(
                d,
                K,
                centroids.data(),
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
}

void ResidualQuantizer::compute_codes(
        const float* x,
        uint8_t* codes_out,
        size_t n) const {
    int cur_beam_size = 1;

    std::vector<float> residuals(x, x + n * d);
    std::vector<int32_t> codes;
    double t0 = getmillisecs();

    std::unique_ptr<Index> assign_index;
    if (assign_index_factory) {
        assign_index.reset((*assign_index_factory)(d));
    } else {
        assign_index.reset(new IndexFlatL2(d));
    }

    for (int m = 0; m < M; m++) {
        int K = 1 << nbits[m];

        const float* centroids_m =
                this->centroids.data() + centroid_offsets[m] * d;

        int new_beam_size = std::min(cur_beam_size * K, max_beam_size);

        std::vector<int32_t> new_codes(n * new_beam_size * (m + 1));
        std::vector<float> new_residuals(n * new_beam_size * d);
        std::vector<float> distances(n * new_beam_size);

        beam_search_encode_step(
                d,
                K,
                centroids_m,
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

    // pack only the first code of the beam (hence the ld_codes=M *
    // cur_beam_size)
    pack_codes(n, codes.data(), codes_out, M * cur_beam_size);
}

void ResidualQuantizer::pack_codes(
        size_t n,
        const int32_t* codes,
        uint8_t* packed_codes,
        int64_t ld_codes) const {
    if (ld_codes == -1) {
        ld_codes = M;
    }
#pragma omp parallel for if (n > 1000)
    for (int64_t i = 0; i < n; i++) {
        const int32_t* codes1 = codes + i * ld_codes;
        BitstringWriter bsw(packed_codes + i * code_size, code_size);
        for (int m = 0; m < M; m++) {
            bsw.write(codes1[m], nbits[m]);
        }
    }
}

void ResidualQuantizer::decode(const uint8_t* code, float* x, size_t n) const {
    // standard additive quantizer decoding
#pragma omp parallel for if (n > 1000)
    for (int64_t i = 0; i < n; i++) {
        BitstringReader bsr(code + i * code_size, code_size);
        float* xi = x + i * d;
        for (int m = 0; m < M; m++) {
            int idx = bsr.read(nbits[m]);
            const float* c = centroids.data() + d * (centroid_offsets[m] + idx);
            if (m == 0) {
                memcpy(xi, c, sizeof(*x) * d);
            } else {
                fvec_add(d, xi, c, xi);
            }
        }
    }
}

} // namespace faiss
