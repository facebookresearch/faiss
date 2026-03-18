/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexIVFPQPanorama.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/panorama_kernels/panorama_kernels.h>
#include <faiss/utils/Heap.h>

namespace faiss {

/*****************************************
 * Constructor
 ******************************************/

IndexIVFPQPanorama::IndexIVFPQPanorama(
        Index* quantizer,
        size_t d,
        size_t nlist,
        size_t M,
        size_t nbits_per_idx,
        int n_levels,
        size_t batch_size,
        MetricType metric,
        bool own_invlists)
        : IndexIVFPQ(
                  quantizer,
                  d,
                  nlist,
                  M,
                  nbits_per_idx,
                  metric,
                  own_invlists),
          n_levels(n_levels),
          batch_size(batch_size),
          chunk_size(code_size / n_levels),
          levels_size(d / n_levels),
          m_level_width(M / n_levels) {
    FAISS_THROW_IF_NOT_MSG(M % n_levels == 0, "M must be divisible by n_levels");
    FAISS_THROW_IF_NOT_MSG(batch_size % 64 == 0, "batch_size must be multiple of 64");
    FAISS_THROW_IF_NOT_MSG(nbits_per_idx == 8, "only 8-bit PQ codes supported");
    FAISS_THROW_IF_NOT_MSG(M == code_size, "M must equal code_size for 8-bit PQ");
    FAISS_THROW_IF_NOT_MSG(metric == METRIC_L2, "only L2 metric supported");
}

/*****************************************
 * add — transpose codes into column-major layout and precompute norms
 ******************************************/

void IndexIVFPQPanorama::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(!added, "IndexIVFPQPanorama only supports a single add() call");
    added = true;
    num_points = n;

    IndexIVFPQ::add(n, x);

    // Compute column offsets (each list rounded up to batch_size).
    size_t total_column_bytes = 0;
    column_offsets = new size_t[nlist];
    for (size_t i = 0; i < nlist; i++) {
        column_offsets[i] = total_column_bytes;
        size_t n_batches =
                (invlists->list_size(i) + batch_size - 1) / batch_size;
        total_column_bytes += n_batches * batch_size * code_size;
    }

    // Transpose codes from row-major [point0_code, point1_code, ...] into
    // column-major within each batch: M columns of batch_size bytes each.
    column_storage = new uint8_t[total_column_bytes]();
    for (size_t list_no = 0; list_no < nlist; list_no++) {
        size_t col_offset = column_offsets[list_no];
        size_t list_size = invlists->list_size(list_no);
        size_t n_batches = (list_size + batch_size - 1) / batch_size;
        const uint8_t* row_codes = invlists->get_codes(list_no);

        for (size_t batch_no = 0; batch_no < n_batches; batch_no++) {
            size_t batch_offset = batch_no * batch_size * code_size;
            size_t curr_batch_size =
                    std::min(list_size - batch_no * batch_size, batch_size);
            for (size_t m = 0; m < pq.M; m++) {
                for (size_t p = 0; p < curr_batch_size; p++) {
                    column_storage[col_offset + batch_offset +
                                   m * batch_size + p] =
                            row_codes[batch_no * batch_size * code_size +
                                      p * code_size + m];
                }
            }
        }
    }

    // Precompute cumulative residual norms and initial exact distances.
    cum_sum_offsets = new size_t[nlist];
    init_exact_distances_offsets = new size_t[nlist];

    size_t cum_size = 0;
    size_t init_size = 0;
    for (size_t list_no = 0; list_no < nlist; list_no++) {
        cum_sum_offsets[list_no] = cum_size;
        cum_size += invlists->list_size(list_no) * (n_levels + 1);
        init_exact_distances_offsets[list_no] = init_size;
        init_size += invlists->list_size(list_no);
    }

    cum_sums = new float[cum_size];
    init_exact_distances = new float[init_size];

    for (size_t list_no = 0; list_no < nlist; list_no++) {
        size_t list_size = invlists->list_size(list_no);

        std::vector<float> centroid(d);
        quantizer->reconstruct(list_no, centroid.data());

        size_t n_batches = (list_size + batch_size - 1) / batch_size;

        for (size_t batch_no = 0; batch_no < n_batches; batch_no++) {
            size_t b_offset = batch_no * batch_size;
            size_t curr_batch_size =
                    std::min(list_size - b_offset, batch_size);

            for (size_t p = 0; p < curr_batch_size; p++) {
                std::vector<float> vec(d);
                const uint8_t* code =
                        invlists->get_single_code(list_no, b_offset + p);
                pq.decode(code, vec.data());

                float init_dist = 0.0f;
                std::vector<float> suffix(d + 1, 0.0f);
                for (int j = d - 1; j >= 0; j--) {
                    init_dist += vec[j] * vec[j] + 2 * vec[j] * centroid[j];
                    suffix[j] = suffix[j + 1] + vec[j] * vec[j];
                }

                for (int level = 0; level < n_levels; level++) {
                    int start_idx = level * levels_size;
                    size_t offset = cum_sum_offsets[list_no] +
                            b_offset * (n_levels + 1) +
                            level * curr_batch_size + p;
                    cum_sums[offset] = start_idx < (int)d
                            ? std::sqrt(suffix[start_idx])
                            : 0.0f;
                }

                size_t last_offset = cum_sum_offsets[list_no] +
                        b_offset * (n_levels + 1) +
                        n_levels * curr_batch_size + p;
                cum_sums[last_offset] = 0.0f;

                init_exact_distances
                        [init_exact_distances_offsets[list_no] + b_offset + p] =
                                init_dist;
            }
        }
    }
}

/*****************************************
 * Panorama scanner — overrides scan_codes with batch processing
 ******************************************/

namespace {

using idx_t = faiss::idx_t;

template <class C, bool use_sel>
struct IVFPQScannerPanorama : InvertedListScanner {
    const IndexIVFPQPanorama& index;
    const ProductQuantizer& pq;

    // Query state
    const float* qi = nullptr;
    std::vector<float> query_cum_norms;
    std::vector<float> sim_table_2;

    // Per-list state
    float coarse_dis = 0;

    IVFPQScannerPanorama(
            const IndexIVFPQPanorama& index,
            bool store_pairs,
            const IDSelector* sel)
            : InvertedListScanner(store_pairs, sel),
              index(index),
              pq(index.pq) {
        this->keep_max = is_similarity_metric(index.metric_type);
        this->code_size = pq.code_size;
        query_cum_norms.resize(index.n_levels + 1);
        sim_table_2.resize(pq.M * pq.ksub);
    }

    void set_query(const float* query) override {
        this->qi = query;

        FAISS_ASSERT(index.by_residual);
        FAISS_ASSERT(index.use_precomputed_table == 1);

        pq.compute_inner_prod_table(qi, sim_table_2.data());

        // Compute query suffix sums → cum norms per level.
        std::vector<float> suffix(index.d + 1, 0.0f);
        for (int j = index.d - 1; j >= 0; j--) {
            suffix[j] = suffix[j + 1] + qi[j] * qi[j];
        }
        for (int level = 0; level < index.n_levels; level++) {
            int start = level * index.levels_size;
            query_cum_norms[level] =
                    start < (int)index.d ? std::sqrt(suffix[start]) : 0.0f;
        }
        query_cum_norms[index.n_levels] = 0.0f;
    }

    void set_list(idx_t list_no, float coarse_dis) override {
        this->list_no = list_no;
        this->coarse_dis = coarse_dis;
    }

    float distance_to_code(const uint8_t* code) const override {
        FAISS_THROW_MSG(
                "IndexIVFPQPanorama does not support distance_to_code");
    }

    size_t scan_codes(
            size_t list_size,
            const uint8_t* /* codes (row-major, unused) */,
            const idx_t* ids,
            float* distances,
            idx_t* labels,
            size_t k) const override {
        size_t nup = 0;

        const size_t bs = index.batch_size;
        const size_t cs = index.chunk_size;
        const int n_levels = index.n_levels;

        const size_t n_batches = (list_size + bs - 1) / bs;
        const size_t sim_table_size = pq.ksub * pq.M;

        // Panorama column-major codes for this list.
        const uint8_t* col_codes =
                index.column_storage + index.column_offsets[list_no];
        const float* list_cum_sums =
                index.cum_sums + index.cum_sum_offsets[list_no];
        const float* list_init_dists =
                index.init_exact_distances +
                index.init_exact_distances_offsets[list_no];

        // Scratch buffers.
        std::vector<float> exact_distances(bs);
        std::vector<uint8_t> bitset(bs);
        std::vector<uint32_t> active_indices(bs);
        std::vector<uint8_t> compressed_codes(bs * cs);
        std::vector<float> sim_table_cache(sim_table_size);
        float dis0_cache = 0;

        for (size_t batch_no = 0; batch_no < n_batches; batch_no++) {
            size_t curr_batch_size =
                    std::min(list_size - batch_no * bs, bs);
            size_t b_offset = batch_no * bs;

            // Initialize active set.
            std::iota(
                    active_indices.begin(),
                    active_indices.begin() + curr_batch_size,
                    b_offset);
            std::fill(bitset.begin(), bitset.begin() + curr_batch_size, 1);
            std::fill(bitset.begin() + curr_batch_size, bitset.end(), 0);

            for (size_t idx = 0; idx < curr_batch_size; idx++) {
                exact_distances[idx] = list_init_dists[b_offset + idx];
            }

            const uint8_t* batch_codes = col_codes + b_offset * code_size;
            const float* batch_cums =
                    list_cum_sums + b_offset * (n_levels + 1);

            size_t next_num_active = curr_batch_size;
            float dis0 = 0;
            size_t batch_offset = batch_no * bs;

            for (int level = 0;
                 level < n_levels && next_num_active > 0;
                 level++) {
                // Compute sim table for this level (cached across batches
                // within same list, only for first batch).
                size_t level_sim_offset = level * pq.ksub * cs;

                if (level == 0 && batch_no == 0) {
                    // Precompute LUT: sim_table = -2 * sim_table_2
                    // (the precomputed_table term is added via dis0).
                    dis0_cache = coarse_dis;
                    const size_t n = pq.M * pq.ksub;
                    for (size_t i = 0; i < n; i++) {
                        sim_table_cache[i] = -2.0f * sim_table_2[i];
                    }
                }
                dis0 = dis0_cache;

                float query_cum_norm =
                        2 * query_cum_norms[level + 1];
                float heap_max = distances[0];

                const float* cum_sums_level =
                        batch_cums + curr_batch_size * level;
                const uint8_t* codes_level =
                        batch_codes + bs * cs * level;

                float* sim_table_level =
                        sim_table_cache.data() + level_sim_offset;

                bool is_sparse = next_num_active < bs / 16;

                size_t num_active_for_filtering = 0;
                if (is_sparse) {
                    // Sparse path: use active_indices for indirection.
                    for (size_t ci = 0; ci < cs; ci++) {
                        size_t chunk_off = ci * bs;
                        float* chunk_sim = sim_table_level + ci * pq.ksub;
                        for (size_t i = 0; i < next_num_active; i++) {
                            size_t real_idx =
                                    active_indices[i] - batch_offset;
                            exact_distances[i] +=
                                    chunk_sim[codes_level[chunk_off + real_idx]];
                        }
                    }
                    num_active_for_filtering = next_num_active;
                } else {
                    auto [cc, na] =
                            panorama_kernels::process_code_compression(
                                    next_num_active,
                                    bs,
                                    cs,
                                    compressed_codes.data(),
                                    bitset.data(),
                                    codes_level);

                    panorama_kernels::process_chunks(
                            cs, bs, na, sim_table_level, cc,
                            exact_distances.data());
                    num_active_for_filtering = na;
                }

                next_num_active = panorama_kernels::process_filtering(
                        num_active_for_filtering,
                        exact_distances.data(),
                        active_indices.data(),
                        const_cast<float*>(cum_sums_level),
                        bitset.data(),
                        batch_offset,
                        dis0,
                        query_cum_norm,
                        heap_max);
            }

            // Insert surviving candidates into heap.
            for (size_t i = 0; i < next_num_active; i++) {
                float dis = dis0 + exact_distances[i];
                if (C::cmp(distances[0], dis)) {
                    idx_t id = store_pairs
                            ? lo_build(list_no, active_indices[i])
                            : ids[active_indices[i]];
                    heap_replace_top<C>(k, distances, labels, dis, id);
                    nup++;
                }
            }
        }

        return nup;
    }

    size_t scan_codes(
            size_t n,
            const uint8_t* codes,
            const idx_t* ids,
            ResultHandler& handler) const override {
        FAISS_THROW_MSG(
                "IndexIVFPQPanorama: ResultHandler scan_codes not supported");
    }
};

} // anonymous namespace

/*****************************************
 * get_InvertedListScanner
 ******************************************/

InvertedListScanner* IndexIVFPQPanorama::get_InvertedListScanner(
        bool store_pairs,
        const IDSelector* sel,
        const IVFSearchParameters*) const {
    FAISS_THROW_IF_NOT_MSG(
            metric_type == METRIC_L2, "only L2 metric supported");
    FAISS_THROW_IF_NOT_MSG(
            use_precomputed_table == 1,
            "Panorama PQ requires use_precomputed_table == 1");
    FAISS_THROW_IF_NOT_MSG(
            pq.nbits == 8, "only 8-bit PQ codes supported");
    FAISS_THROW_IF_NOT_MSG(
            by_residual, "Panorama PQ requires by_residual");
    FAISS_THROW_IF_NOT_MSG(
            polysemous_ht == 0,
            "Panorama PQ does not support polysemous");

    if (sel) {
        return new IVFPQScannerPanorama<CMax<float, idx_t>, true>(
                *this, store_pairs, sel);
    } else {
        return new IVFPQScannerPanorama<CMax<float, idx_t>, false>(
                *this, store_pairs, sel);
    }
}

} // namespace faiss
