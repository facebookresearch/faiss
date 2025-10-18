/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexIVFFlatPanorama.h>

#include <cstdio>

#include <faiss/IndexFlat.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/IDSelector.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/extra_distances.h>
#include <faiss/utils/utils.h>

namespace faiss {

IndexIVFFlatPanorama::IndexIVFFlatPanorama(
        Index* quantizer,
        size_t d,
        size_t nlist,
        int n_levels,
        MetricType metric,
        bool own_invlists)
        : IndexIVFFlat(quantizer, d, nlist, metric, false), n_levels(n_levels) {
    // For now, we only support L2 distance.
    // Supporting dot product and cosine distance is a trivial addition
    // left for future work.
    FAISS_THROW_IF_NOT(metric == METRIC_L2);

    // We construct the inverted lists here so that we can use the
    // level-oriented storage. This does not cause a leak as we constructed
    // IndexIVF first, with own_invlists set to false.
    this->invlists = new ArrayInvertedListsPanorama(nlist, code_size, n_levels);
    this->own_invlists = own_invlists;
}

namespace {

template <typename VectorDistance, bool use_sel>
struct IVFFlatScannerPanorama : InvertedListScanner {
    VectorDistance vd;
    const ArrayInvertedListsPanorama* storage;
    using C = typename VectorDistance::C;

    IVFFlatScannerPanorama(
            const VectorDistance& vd,
            const ArrayInvertedListsPanorama* storage,
            bool store_pairs,
            const IDSelector* sel)
            : InvertedListScanner(store_pairs, sel), vd(vd), storage(storage) {
        keep_max = vd.is_similarity;
        code_size = vd.d * sizeof(float);
        cum_sums.resize(storage->n_levels + 1);
    }

    const float* xi = nullptr;
    std::vector<float> cum_sums;
    float q_norm = 0.0f;
    void set_query(const float* query) override {
        this->xi = query;

        const size_t d = vd.d;
        const size_t level_width_floats = storage->level_width / sizeof(float);

        std::vector<float> suffix_sums(d + 1);
        suffix_sums[d] = 0.0f;

        for (int j = d - 1; j >= 0; j--) {
            float squared_val = query[j] * query[j];
            suffix_sums[j] = suffix_sums[j + 1] + squared_val;
        }

        for (size_t level = 0; level < storage->n_levels; level++) {
            size_t start_idx = level * level_width_floats;
            if (start_idx < d) {
                cum_sums[level] = sqrt(suffix_sums[start_idx]);
            } else {
                cum_sums[level] = 0.0f;
            }
        }

        cum_sums[storage->n_levels] = 0.0f;
        q_norm = suffix_sums[0];
    }

    void set_list(idx_t list_no, float /* coarse_dis */) override {
        this->list_no = list_no;
    }

    /// This function is unreachable as `IndexIVF` only calls this within
    /// iterators, which are not supported by `IndexIVFFlatPanorama`.
    /// To avoid undefined behavior, we throw an error here.
    float distance_to_code(const uint8_t* /* code */) const override {
        FAISS_THROW_MSG(
                "IndexIVFFlatPanorama does not support distance_to_code");
    }

    /// Helper function for progressive filtering that both scan_codes and
    /// scan_codes_range use. Processes a batch of vectors through all levels,
    /// computing exact distances and pruning based on a threshold.
    /// Returns the number of active survivors after all levels.
    size_t progressive_filter_batch(
            size_t batch_no,
            size_t list_size,
            const uint8_t* codes_base,
            const float* cum_sums_data,
            float threshold,
            std::vector<float>& exact_distances,
            std::vector<uint32_t>& active_indices,
            const idx_t* ids) const {
        const size_t d = vd.d;
        const size_t level_width_floats = storage->level_width / sizeof(float);

        size_t batch_start = batch_no * storage->kBatchSize;
        size_t curr_batch_size =
                std::min(list_size - batch_start, storage->kBatchSize);

        size_t cumsum_batch_offset =
                batch_no * storage->kBatchSize * (storage->n_levels + 1);
        const float* batch_cum_sums = cum_sums_data + cumsum_batch_offset;

        size_t batch_offset = batch_no * storage->kBatchSize * code_size;
        const uint8_t* storage_base = codes_base + batch_offset;

        // Initialize active set with ID-filtered vectors.
        size_t num_active = 0;
        for (size_t i = 0; i < curr_batch_size; i++) {
            size_t global_idx = batch_start + i;
            bool include = !use_sel || sel->is_member(ids[global_idx]);

            active_indices[num_active] = i;
            float cum_sum = batch_cum_sums[i];
            exact_distances[i] = cum_sum * cum_sum + q_norm;

            num_active += include;
        }

        if (num_active == 0) {
            return 0;
        }

        const float* level_cum_sums = batch_cum_sums + storage->kBatchSize;

        // Progressive filtering through levels.
        for (size_t level = 0; level < storage->n_levels; level++) {
            float query_cum_norm = cum_sums[level + 1];

            size_t level_offset =
                    level * storage->level_width * storage->kBatchSize;
            const float* level_storage =
                    (const float*)(storage_base + level_offset);

            size_t next_active = 0;
            for (size_t i = 0; i < num_active; i++) {
                uint32_t idx = active_indices[i];
                const float* yj = level_storage + idx * level_width_floats;
                const float* query_level = xi + level * level_width_floats;

                size_t actual_level_width = std::min(
                        level_width_floats, d - level * level_width_floats);
                float dot_product =
                        fvec_inner_product(query_level, yj, actual_level_width);

                exact_distances[idx] -= 2.0f * dot_product;

                float cum_sum = level_cum_sums[idx];
                float cauchy_schwarz_bound = 2.0f * cum_sum * query_cum_norm;
                float lower_bound = exact_distances[idx] - cauchy_schwarz_bound;

                active_indices[next_active] = idx;
                next_active += C::cmp(threshold, lower_bound) ? 1 : 0;
            }

            num_active = next_active;
            level_cum_sums += storage->kBatchSize;
        }

        return num_active;
    }

    size_t scan_codes(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            float* simi,
            idx_t* idxi,
            size_t k) const override {
        size_t nup = 0;

        const size_t n_batches =
                (list_size + storage->kBatchSize - 1) / storage->kBatchSize;

        const uint8_t* codes_base = codes;
        const float* cum_sums_data = storage->get_cum_sums(list_no);

        std::vector<float> exact_distances(storage->kBatchSize);
        std::vector<uint32_t> active_indices(storage->kBatchSize);

        // Panorama's IVFFlat core progressive filtering algorithm:
        // Process vectors in batches for cache efficiency. For each batch:
        // 1. Apply ID selection filter and initialize distances
        // (||y||^2 + ||x||^2).
        // 2. Maintain an "active set" of candidate indices that haven't been
        //    pruned yet.
        // 3. For each level, refine distances incrementally and compact the
        //    active set:
        //    - Compute dot product for current level: exact_dist -= 2*<x,y>.
        //    - Use Cauchy-Schwarz bound on remaining levels to get lower bound
        //    - Prune candidates whose lower bound exceeds k-th best distance.
        //    - Compact active_indices to remove pruned candidates (branchless)
        // 4. After all levels, survivors are exact distances; update heap.
        // This achieves early termination while maintaining SIMD-friendly
        // sequential access patterns in the level-oriented storage layout.
        for (size_t batch_no = 0; batch_no < n_batches; batch_no++) {
            size_t batch_start = batch_no * storage->kBatchSize;

            size_t num_active = progressive_filter_batch(
                    batch_no,
                    list_size,
                    codes_base,
                    cum_sums_data,
                    simi[0],
                    exact_distances,
                    active_indices,
                    ids);

            // Add batch survivors to heap.
            for (size_t i = 0; i < num_active; i++) {
                uint32_t idx = active_indices[i];
                size_t global_idx = batch_start + idx;
                float dis = exact_distances[idx];

                if (C::cmp(simi[0], dis)) {
                    int64_t id = store_pairs ? lo_build(list_no, global_idx)
                                             : ids[global_idx];
                    heap_replace_top<C>(k, simi, idxi, dis, id);
                    nup++;
                }
            }
        }

        return nup;
    }

    void scan_codes_range(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            float radius,
            RangeQueryResult& res) const override {
        const size_t n_batches =
                (list_size + storage->kBatchSize - 1) / storage->kBatchSize;

        const uint8_t* codes_base = codes;
        const float* cum_sums_data = storage->get_cum_sums(list_no);

        std::vector<float> exact_distances(storage->kBatchSize);
        std::vector<uint32_t> active_indices(storage->kBatchSize);

        // Same progressive filtering as scan_codes, but with fixed radius
        // threshold instead of dynamic heap threshold.
        for (size_t batch_no = 0; batch_no < n_batches; batch_no++) {
            size_t batch_start = batch_no * storage->kBatchSize;

            size_t num_active = progressive_filter_batch(
                    batch_no,
                    list_size,
                    codes_base,
                    cum_sums_data,
                    radius,
                    exact_distances,
                    active_indices,
                    ids);

            // Add batch survivors to range result.
            for (size_t i = 0; i < num_active; i++) {
                uint32_t idx = active_indices[i];
                size_t global_idx = batch_start + idx;
                float dis = exact_distances[idx];

                if (C::cmp(radius, dis)) {
                    int64_t id = store_pairs ? lo_build(list_no, global_idx)
                                             : ids[global_idx];
                    res.add(dis, id);
                }
            }
        }
    }
};

struct Run_get_InvertedListScanner {
    using T = InvertedListScanner*;

    template <class VD>
    InvertedListScanner* f(
            VD& vd,
            const IndexIVFFlatPanorama* ivf,
            bool store_pairs,
            const IDSelector* sel) {
        // Safely cast to ArrayInvertedListsPanorama to access cumulative sums.
        const ArrayInvertedListsPanorama* storage =
                dynamic_cast<const ArrayInvertedListsPanorama*>(ivf->invlists);
        FAISS_THROW_IF_NOT_MSG(
                storage,
                "IndexIVFFlatPanorama requires ArrayInvertedListsPanorama");

        if (sel) {
            return new IVFFlatScannerPanorama<VD, true>(
                    vd, storage, store_pairs, sel);
        } else {
            return new IVFFlatScannerPanorama<VD, false>(
                    vd, storage, store_pairs, sel);
        }
    }
};

} // anonymous namespace

InvertedListScanner* IndexIVFFlatPanorama::get_InvertedListScanner(
        bool store_pairs,
        const IDSelector* sel,
        const IVFSearchParameters*) const {
    Run_get_InvertedListScanner run;
    return dispatch_VectorDistance(
            d, metric_type, metric_arg, run, this, store_pairs, sel);
}

void IndexIVFFlatPanorama::reconstruct_from_offset(
        int64_t list_no,
        int64_t offset,
        float* recons) const {
    const uint8_t* code = invlists->get_single_code(list_no, offset);
    memcpy(recons, code, code_size);
    invlists->release_codes(list_no, code);
}

} // namespace faiss
