/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexRaBitQ.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ResultHandler.h>
#include <memory>

namespace faiss {

// Forward declaration from RaBitQuantizer.cpp
struct RaBitQDistanceComputer;

IndexRaBitQ::IndexRaBitQ() = default;

IndexRaBitQ::IndexRaBitQ(idx_t d, MetricType metric, uint8_t nb_bits_in)
        : IndexFlatCodes(0, d, metric), rabitq(d, metric, nb_bits_in) {
    // Update code size based on nb_bits
    code_size = rabitq.code_size;

    is_trained = false;
}

void IndexRaBitQ::train(idx_t n, const float* x) {
    // compute a centroid
    std::vector<float> centroid(d, 0);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < d; j++) {
            centroid[j] += x[i * d + j];
        }
    }

    if (n != 0) {
        for (size_t j = 0; j < d; j++) {
            centroid[j] /= (float)n;
        }
    }

    center = std::move(centroid);

    //
    rabitq.train(n, x);
    is_trained = true;
}

void IndexRaBitQ::sa_encode(idx_t n, const float* x, uint8_t* bytes) const {
    FAISS_THROW_IF_NOT(is_trained);
    rabitq.compute_codes_core(x, bytes, n, center.data());
}

void IndexRaBitQ::sa_decode(idx_t n, const uint8_t* bytes, float* x) const {
    FAISS_THROW_IF_NOT(is_trained);
    rabitq.decode_core(bytes, x, n, center.data());
}

FlatCodesDistanceComputer* IndexRaBitQ::get_FlatCodesDistanceComputer() const {
    FlatCodesDistanceComputer* dc =
            rabitq.get_distance_computer(qb, center.data(), centered);
    dc->code_size = rabitq.code_size;
    dc->codes = codes.data();
    return dc;
}

FlatCodesDistanceComputer* IndexRaBitQ::get_quantized_distance_computer(
        const uint8_t qb,
        bool centered) const {
    FlatCodesDistanceComputer* dc =
            rabitq.get_distance_computer(qb, center.data(), centered);
    dc->code_size = rabitq.code_size;
    dc->codes = codes.data();
    return dc;
}

namespace {

struct Run_search_with_dc_res {
    using T = void;

    uint8_t qb = 0;
    bool centered = false;
    uint8_t nb_bits = 1; // Number of bits per dimension

    template <class BlockResultHandler>
    void f(BlockResultHandler& res, const IndexRaBitQ* index, const float* xq) {
        size_t ntotal = index->ntotal;
        using SingleResultHandler =
                typename BlockResultHandler::SingleResultHandler;
        const int d = index->d;
        size_t ex_bits = nb_bits - 1;

#pragma omp parallel
        {
            std::unique_ptr<FlatCodesDistanceComputer> dc_base(
                    index->get_quantized_distance_computer(qb, centered));
            SingleResultHandler resi(res);
#pragma omp for
            for (int64_t q = 0; q < res.nq; q++) {
                resi.begin(q);
                dc_base->set_query(xq + d * q);

                // Stats tracking for multi-bit two-stage search only
                // n_1bit_evaluations: candidates evaluated using 1-bit lower
                // bound n_multibit_evaluations: candidates requiring full
                // multi-bit distance
                size_t local_1bit_evaluations = 0;
                size_t local_multibit_evaluations = 0;

                if (ex_bits == 0) {
                    // 1-bit: Standard single-stage search (no stats tracking)
                    for (size_t i = 0; i < ntotal; i++) {
                        if (res.is_in_selection(i)) {
                            float dis = (*dc_base)(i);
                            resi.add_result(dis, i);
                        }
                    }
                } else {
                    // Multi-bit: Two-stage search with adaptive filtering
                    // Note: Even with query quantization (qb > 0), ex-bits
                    // distance computation uses the float query to maintain
                    // consistency with encoding-time factor computation. See
                    // RaBitQuantizer.cpp for details.
                    auto* dc = dynamic_cast<RaBitQDistanceComputer*>(
                            dc_base.get());
                    FAISS_THROW_IF_NOT_MSG(
                            dc != nullptr,
                            "Failed to cast to RaBitQDistanceComputer for two-stage search");

                    // Use appropriate comparison based on metric type
                    bool is_similarity =
                            is_similarity_metric(index->metric_type);

                    for (size_t i = 0; i < ntotal; i++) {
                        if (res.is_in_selection(i)) {
                            const uint8_t* code =
                                    index->codes.data() + i * index->code_size;

                            local_1bit_evaluations++;

                            // Stage 1: Compute 1-bit lower bound
                            float lower_bound = dc->lower_bound_distance(code);

                            // Stage 2: Adaptive filtering using threshold
                            // For L2 (min-heap): filter if lower_bound <
                            // resi.threshold For IP (max-heap): filter if
                            // lower_bound > resi.threshold Note: Using
                            // resi.threshold directly (not cached) enables more
                            // aggressive filtering as the heap is updated
                            bool should_refine = is_similarity
                                    ? (lower_bound > resi.threshold)
                                    : (lower_bound < resi.threshold);

                            if (should_refine) {
                                local_multibit_evaluations++;
                                // Compute full multi-bit distance
                                float dist_full =
                                        dc->distance_to_code_full(code);
                                resi.add_result(dist_full, i);
                            }
                        }
                    }
                }

                // Update global stats atomically
#pragma omp atomic
                rabitq_stats.n_1bit_evaluations += local_1bit_evaluations;
#pragma omp atomic
                rabitq_stats.n_multibit_evaluations +=
                        local_multibit_evaluations;

                resi.end();
            }
        }
    }
};

} // namespace

void IndexRaBitQ::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params_in) const {
    FAISS_THROW_IF_NOT(is_trained);

    // Extract search parameters
    uint8_t used_qb = qb;
    bool used_centered = centered;
    if (auto params = dynamic_cast<const RaBitQSearchParameters*>(params_in)) {
        used_qb = params->qb;
        used_centered = params->centered;
    }

    const IDSelector* sel = (params_in != nullptr) ? params_in->sel : nullptr;

    // Set up functor with all necessary parameters
    Run_search_with_dc_res r;
    r.qb = used_qb;
    r.centered = used_centered;
    r.nb_bits = rabitq.nb_bits; // Pass multi-bit info to functor

    // Use Faiss framework for all cases (single-stage and two-stage)
    dispatch_knn_ResultHandler(
            n, distances, labels, k, metric_type, sel, r, this, x);
}

void IndexRaBitQ::range_search(
        idx_t n,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params_in) const {
    uint8_t used_qb = qb;
    if (auto params = dynamic_cast<const RaBitQSearchParameters*>(params_in)) {
        used_qb = params->qb;
    }

    const IDSelector* sel = (params_in != nullptr) ? params_in->sel : nullptr;
    Run_search_with_dc_res r;
    r.qb = used_qb;

    dispatch_range_ResultHandler(result, radius, metric_type, sel, r, this, x);
}

} // namespace faiss
