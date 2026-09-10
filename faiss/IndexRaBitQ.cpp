/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexRaBitQ.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/RaBitQUtils.h>
#include <faiss/impl/ResultHandler.h>
#include <memory>

namespace faiss {

// Forward declaration from RaBitQuantizer.cpp
struct RaBitQDistanceComputer;

using rabitq_utils::SignBitFactorsWithError;

IndexRaBitQ::IndexRaBitQ() = default;

IndexRaBitQ::IndexRaBitQ(idx_t d_in, MetricType metric, uint8_t nb_bits_in)
        : IndexFlatCodes(0, d_in, metric), rabitq(d_in, metric, nb_bits_in) {
    // Update code size based on nb_bits
    code_size = rabitq.code_size;

    is_trained = false;
}

void IndexRaBitQ::train(idx_t n, const float* x) {
    // compute a centroid
    std::vector<float> centroid(d, 0);
    for (idx_t i = 0; i < n; i++) {
        for (size_t j = 0; j < static_cast<size_t>(d); j++) {
            centroid[j] += x[i * d + j];
        }
    }

    if (n != 0) {
        for (size_t j = 0; j < static_cast<size_t>(d); j++) {
            centroid[j] /= (float)n;
        }
    }

    center = std::move(centroid);

    //
    rabitq.train(n, x);
    is_trained = true;
}

void IndexRaBitQ::add(idx_t n, const float* x) {
    IndexFlatCodes::add(n, x);
    if (full_code_mode != RABITQ_FULL_CODE_PACKED) {
        rebuild_expanded_codes();
    }
}

void IndexRaBitQ::add_sa_codes(idx_t n, const uint8_t* x, const idx_t* xids) {
    IndexFlatCodes::add_sa_codes(n, x, xids);
    if (full_code_mode != RABITQ_FULL_CODE_PACKED) {
        rebuild_expanded_codes();
    }
}

void IndexRaBitQ::reset() {
    IndexFlatCodes::reset();
    expanded_codes.clear();
}

size_t IndexRaBitQ::remove_ids(const IDSelector& sel) {
    const size_t removed = IndexFlatCodes::remove_ids(sel);
    if (removed > 0 && full_code_mode != RABITQ_FULL_CODE_PACKED) {
        rebuild_expanded_codes();
    }
    return removed;
}

void IndexRaBitQ::merge_from(Index& other_index, idx_t add_id) {
    IndexFlatCodes::merge_from(other_index, add_id);
    if (full_code_mode != RABITQ_FULL_CODE_PACKED) {
        rebuild_expanded_codes();
    }
}

void IndexRaBitQ::permute_entries(const idx_t* perm) {
    IndexFlatCodes::permute_entries(perm);
    if (full_code_mode != RABITQ_FULL_CODE_PACKED) {
        rebuild_expanded_codes();
    }
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
    if (full_code_mode != RABITQ_FULL_CODE_PACKED) {
        FAISS_THROW_IF_NOT_MSG(
                expanded_codes.size() ==
                        static_cast<size_t>(ntotal) * expanded_code_size(),
                "stale expanded RaBitQ cache; call set_full_code_mode again");
        return rabitq.get_expanded_distance_computer(
                expanded_codes.data(),
                center.data(),
                full_code_mode == RABITQ_FULL_CODE_INT8);
    }
    FlatCodesDistanceComputer* dc =
            rabitq.get_distance_computer(qb, center.data(), centered);
    dc->code_size = rabitq.code_size;
    dc->codes = codes.data();
    return dc;
}

size_t IndexRaBitQ::expanded_code_size() const {
    return static_cast<size_t>(d) + sizeof(rabitq_utils::ExtraBitsFactors);
}

void IndexRaBitQ::rebuild_expanded_codes() {
    if (full_code_mode == RABITQ_FULL_CODE_PACKED) {
        expanded_codes.clear();
        return;
    }
    expanded_codes.resize(static_cast<size_t>(ntotal) * expanded_code_size());
    rabitq.expand_codes(codes.data(), ntotal, expanded_codes.data());
}

void IndexRaBitQ::set_full_code_mode(uint8_t mode) {
    FAISS_THROW_IF_NOT_MSG(
            mode == RABITQ_FULL_CODE_PACKED ||
                    mode == RABITQ_FULL_CODE_EXPANDED ||
                    mode == RABITQ_FULL_CODE_INT8,
            "invalid RaBitQ full-code mode");
    if (mode != RABITQ_FULL_CODE_PACKED) {
        FAISS_THROW_IF_NOT_MSG(
                metric_type == METRIC_L2,
                "expanded RaBitQ ADC supports only L2");
        FAISS_THROW_IF_NOT_MSG(
                rabitq.nb_bits >= 2 && rabitq.nb_bits <= 8,
                "expanded RaBitQ ADC requires 2..8 total bits");
    }
    full_code_mode = static_cast<RaBitQFullCodeMode>(mode);
    rebuild_expanded_codes();
}

FlatCodesDistanceComputer* IndexRaBitQ::get_quantized_distance_computer(
        const uint8_t qb_in,
        bool centered_in) const {
    if (full_code_mode != RABITQ_FULL_CODE_PACKED) {
        return get_FlatCodesDistanceComputer();
    }
    FlatCodesDistanceComputer* dc =
            rabitq.get_distance_computer(qb_in, center.data(), centered_in);
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
    bool full_only = false;

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
            for (int64_t q = 0; q < static_cast<int64_t>(res.nq); q++) {
                resi.begin(q);
                dc_base->set_query(xq + d * q);

                if (ex_bits == 0 || full_only) {
                    // 1-bit: Standard single-stage search
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
                    FAISS_THROW_IF_MSG(
                            dc == nullptr,
                            "Failed to cast to RaBitQDistanceComputer for two-stage search");

                    bool is_similarity =
                            is_similarity_metric(index->metric_type);

                    for (size_t i = 0; i < ntotal; i++) {
                        if (res.is_in_selection(i)) {
                            const uint8_t* code =
                                    index->codes.data() + i * index->code_size;

                            float est_distance =
                                    dc->distance_to_code_1bit(code);

                            size_t code_size_base = (index->d + 7) / 8;
                            const rabitq_utils::SignBitFactorsWithError*
                                    base_fac = reinterpret_cast<
                                            const rabitq_utils::
                                                    SignBitFactorsWithError*>(
                                            code + code_size_base);

                            bool should_refine =
                                    rabitq_utils::should_refine_candidate(
                                            est_distance,
                                            base_fac->f_error,
                                            dc->g_error,
                                            resi.threshold,
                                            is_similarity);
                            if (should_refine) {
                                float dist_full =
                                        dc->distance_to_code_full(code);
                                resi.add_result(dist_full, i);
                            }
                        }
                    }
                }

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
    r.full_only = full_code_mode != RABITQ_FULL_CODE_PACKED;

    // Use Faiss framework for all cases (single-stage and two-stage)
    dispatch_knn_ResultHandler(
            n, distances, labels, k, metric_type, sel, r, this, x);
}

void IndexRaBitQ::range_search(
        idx_t /*n*/,
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
    r.centered = centered;
    r.nb_bits = rabitq.nb_bits;
    r.full_only = full_code_mode != RABITQ_FULL_CODE_PACKED;

    dispatch_range_ResultHandler(result, radius, metric_type, sel, r, this, x);
}

} // namespace faiss
