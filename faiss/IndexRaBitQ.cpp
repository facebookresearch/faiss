/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexRaBitQ.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ResultHandler.h>

namespace faiss {

IndexRaBitQ::IndexRaBitQ() = default;

IndexRaBitQ::IndexRaBitQ(idx_t d, MetricType metric)
        : IndexFlatCodes(0, d, metric), rabitq(d, metric) {
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
            rabitq.get_distance_computer(qb, center.data());
    dc->code_size = rabitq.code_size;
    dc->codes = codes.data();
    return dc;
}

FlatCodesDistanceComputer* IndexRaBitQ::get_quantized_distance_computer(
        const uint8_t qb) const {
    FlatCodesDistanceComputer* dc =
            rabitq.get_distance_computer(qb, center.data());
    dc->code_size = rabitq.code_size;
    dc->codes = codes.data();
    return dc;
}

namespace {

struct Run_search_with_dc_res {
    using T = void;

    uint8_t qb = 0;

    template <class BlockResultHandler>
    void f(BlockResultHandler& res, const IndexRaBitQ* index, const float* xq) {
        size_t ntotal = index->ntotal;
        using SingleResultHandler =
                typename BlockResultHandler::SingleResultHandler;
        const int d = index->d;

#pragma omp parallel // if (res.nq > 100)
        {
            std::unique_ptr<FlatCodesDistanceComputer> dc(
                    index->get_quantized_distance_computer(qb));
            SingleResultHandler resi(res);
#pragma omp for
            for (int64_t q = 0; q < res.nq; q++) {
                resi.begin(q);
                dc->set_query(xq + d * q);
                for (size_t i = 0; i < ntotal; i++) {
                    if (res.is_in_selection(i)) {
                        float dis = (*dc)(i);
                        resi.add_result(dis, i);
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
    uint8_t used_qb = qb;
    if (auto params = dynamic_cast<const RaBitQSearchParameters*>(params_in)) {
        used_qb = params->qb;
    }

    const IDSelector* sel = (params_in != nullptr) ? params_in->sel : nullptr;
    Run_search_with_dc_res r;
    r.qb = used_qb;

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
