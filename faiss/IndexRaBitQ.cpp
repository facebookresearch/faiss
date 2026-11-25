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
        const uint8_t qb_in,
        bool centered_in) const {
    FlatCodesDistanceComputer* dc =
            rabitq.get_distance_computer(qb_in, center.data(), centered_in);
    dc->code_size = rabitq.code_size;
    dc->codes = codes.data();
    return dc;
}

namespace {

// Whether resi.add_result would keep dis
template <class C>
bool beats_threshold(const ResultHandlerT<C>& resi, float dis) {
    return C::cmp(resi.threshold, dis);
}

// Single-stage search: the estimate is the distance. It is the full distance
// for the multi-bit codes that range_search scans this way.
template <class ResultHandler>
struct OneStage {
    RaBitQDistanceComputer& dc;
    ResultHandler& resi;
    const uint8_t* codes;
    size_t code_size;

    const uint8_t* code(size_t i) const {
        return codes + i * code_size;
    }

    float estimate(size_t i) {
        return dc.nb_bits == 1 ? dc.distance_to_code_1bit(code(i))
                               : dc.distance_to_code_full(code(i));
    }

    void estimate_batch_4(size_t i, float* dis) {
        if (dc.nb_bits > 1) {
            for (size_t j = 0; j < 4; j++) {
                dis[j] = estimate(i + j);
            }
            return;
        }
        const uint8_t* batch[4] = {
                code(i), code(i + 1), code(i + 2), code(i + 3)};
        dc.distance_to_code_1bit_batch_4(batch, dis);
    }

    bool passes(size_t, float dis) const {
        return beats_threshold(resi, dis);
    }

    void add(size_t i, float dis) {
        resi.add_result(dis, i);
    }
};

// Two-stage search of multi-bit codes: the full distance is computed only
// when the error bound of the 1-bit estimate reaches past the threshold.
template <class ResultHandler>
struct TwoStage {
    RaBitQDistanceComputer& dc;
    ResultHandler& resi;
    const uint8_t* codes;
    size_t code_size;
    bool is_similarity;

    const uint8_t* code(size_t i) const {
        return codes + i * code_size;
    }

    float estimate(size_t i) {
        return dc.distance_to_code_1bit(code(i));
    }

    void estimate_batch_4(size_t i, float* dis) {
        const uint8_t* batch[4] = {
                code(i), code(i + 1), code(i + 2), code(i + 3)};
        dc.distance_to_code_1bit_batch_4(batch, dis);
    }

    bool passes(size_t i, float est_distance) const {
        return dc.should_refine(
                code(i), est_distance, resi.threshold, is_similarity);
    }

    void add(size_t i, float) {
        resi.add_result(dc.distance_to_code_full(code(i)), i);
    }
};

// Search without a selector, estimating four codes at a time
template <class Stage>
void scan_all(size_t ntotal, Stage& stage) {
    size_t i = 0;
    for (; i + 4 <= ntotal; i += 4) {
        float dis[4];
        stage.estimate_batch_4(i, dis);
        for (size_t j = 0; j < 4; j++) {
            if (stage.passes(i + j, dis[j])) {
                stage.add(i + j, dis[j]);
            }
        }
    }
    for (; i < ntotal; i++) {
        float dis = stage.estimate(i);
        if (stage.passes(i, dis)) {
            stage.add(i, dis);
        }
    }
}

// Search that tests the selector before each estimate
template <class InSelection, class Stage>
void scan_sel_first(
        size_t ntotal,
        const InSelection& in_selection,
        Stage& stage) {
    for (size_t i = 0; i < ntotal; i++) {
        if (in_selection(i)) {
            float dis = stage.estimate(i);
            if (stage.passes(i, dis)) {
                stage.add(i, dis);
            }
        }
    }
}

// Search that tests the selector after the estimate while at most a third of
// the estimates in the last window passed, and before it otherwise. Testing
// after skips the selector for the codes that fail, but estimates codes that
// the selector would reject. The results are the same either way. Until the
// results fill up, every estimate passes, so the selector starts first.
// Testing after, codes are estimated four at a time; the window is a multiple
// of four, so the batches end on window boundaries.
template <class InSelection, class Stage>
void scan_sel_after_distance(
        size_t ntotal,
        const InSelection& in_selection,
        Stage& stage) {
    constexpr size_t window = 64;
    bool sel_after = false;
    size_t estimated = 0;
    size_t passed = 0;
    auto take = [&](size_t i, float dis) {
        estimated++;
        if (stage.passes(i, dis)) {
            passed++;
            if (!sel_after || in_selection(i)) {
                stage.add(i, dis);
            }
        }
    };
    for (size_t i = 0; i < ntotal;) {
        if (sel_after && i + 4 <= ntotal) {
            float dis[4];
            stage.estimate_batch_4(i, dis);
            for (size_t j = 0; j < 4; j++, i++) {
                take(i, dis[j]);
            }
        } else {
            if (sel_after || in_selection(i)) {
                take(i, stage.estimate(i));
            }
            i++;
        }
        if (estimated >= window) {
            sel_after = 3 * passed <= window;
            estimated = 0;
            passed = 0;
        }
    }
}

struct Run_search_with_dc_res {
    using T = void;

    uint8_t qb = 0;
    bool centered = false;
    uint8_t nb_bits = 1; // Number of bits per dimension
    bool sel_after_distance = false;

    template <class BlockResultHandler>
    void f(BlockResultHandler& res, const IndexRaBitQ* index, const float* xq) {
        size_t ntotal = index->ntotal;
        using SingleResultHandler =
                typename BlockResultHandler::SingleResultHandler;
        const int d = index->d;
        size_t ex_bits = nb_bits - 1;
        auto in_selection = [&res](size_t i) { return res.is_in_selection(i); };
        auto scan = [&](auto& stage) {
            if (res.sel == nullptr) {
                scan_all(ntotal, stage);
            } else if (sel_after_distance) {
                scan_sel_after_distance(ntotal, in_selection, stage);
            } else {
                scan_sel_first(ntotal, in_selection, stage);
            }
        };

#pragma omp parallel
        {
            std::unique_ptr<FlatCodesDistanceComputer> dc_base(
                    index->get_quantized_distance_computer(qb, centered));
            auto* dc = dynamic_cast<RaBitQDistanceComputer*>(dc_base.get());
            FAISS_ASSERT_MSG(
                    dc, "Failed to cast to RaBitQDistanceComputer for search");
            SingleResultHandler resi(res);
#pragma omp for
            for (int64_t q = 0; q < static_cast<int64_t>(res.nq); q++) {
                resi.begin(q);
                dc_base->set_query(xq + d * q);

                if (ex_bits == 0) {
                    // 1-bit: Standard single-stage search
                    OneStage<SingleResultHandler> stage{
                            *dc, resi, index->codes.data(), index->code_size};
                    scan(stage);
                } else {
                    // Multi-bit: Two-stage search with adaptive filtering
                    // Note: Even with query quantization (qb > 0), ex-bits
                    // distance computation uses the float query to maintain
                    // consistency with encoding-time factor computation. See
                    // RaBitQuantizer.cpp for details.
                    bool is_similarity =
                            is_similarity_metric(index->metric_type);

                    TwoStage<SingleResultHandler> stage{
                            *dc,
                            resi,
                            index->codes.data(),
                            index->code_size,
                            is_similarity};
                    scan(stage);
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
    bool sel_after_distance = false;
    if (auto params = dynamic_cast<const RaBitQSearchParameters*>(params_in)) {
        used_qb = params->qb;
        used_centered = params->centered;
        sel_after_distance = params->sel_after_distance;
    }

    const IDSelector* sel = (params_in != nullptr) ? params_in->sel : nullptr;

    // Set up functor with all necessary parameters
    Run_search_with_dc_res r;
    r.qb = used_qb;
    r.centered = used_centered;
    r.nb_bits = rabitq.nb_bits; // Pass multi-bit info to functor
    r.sel_after_distance = sel_after_distance;

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

    dispatch_range_ResultHandler(result, radius, metric_type, sel, r, this, x);
}

} // namespace faiss
