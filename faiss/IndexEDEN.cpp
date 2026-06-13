/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexEDEN.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ResultHandler.h>
#include <memory>

namespace faiss {

IndexEDEN::IndexEDEN() = default;

IndexEDEN::IndexEDEN(
        idx_t d_in,
        MetricType metric,
        uint8_t nb_bits_in,
        EDENScaleType scale_type)
        : IndexFlatCodes(0, d_in, metric),
          eden(d_in, metric, nb_bits_in, scale_type) {
    code_size = eden.code_size;
    is_trained = false;
}

void IndexEDEN::train(idx_t n, const float* x) {
    std::vector<float> centroid(d, 0.0f);
    for (idx_t i = 0; i < n; i++) {
        for (size_t j = 0; j < static_cast<size_t>(d); j++) {
            centroid[j] += x[i * d + j];
        }
    }

    if (n != 0) {
        for (size_t j = 0; j < static_cast<size_t>(d); j++) {
            centroid[j] /= static_cast<float>(n);
        }
    }

    center = std::move(centroid);
    eden.train(n, x);
    is_trained = true;
}

void IndexEDEN::sa_encode(idx_t n, const float* x, uint8_t* bytes) const {
    FAISS_THROW_IF_NOT(is_trained);
    eden.compute_codes_core(x, bytes, n, center.data());
}

void IndexEDEN::sa_decode(idx_t n, const uint8_t* bytes, float* x) const {
    FAISS_THROW_IF_NOT(is_trained);
    eden.decode_core(bytes, x, n, center.data());
}

FlatCodesDistanceComputer* IndexEDEN::get_FlatCodesDistanceComputer() const {
    FlatCodesDistanceComputer* dc = eden.get_distance_computer(center.data());
    dc->code_size = eden.code_size;
    dc->codes = codes.data();
    return dc;
}

namespace {

bool use_eden_batch_scan(size_t d, size_t nb_bits) {
    const size_t values_per_byte = 8 / nb_bits;
    const size_t num_bytes = (d + values_per_byte - 1) / values_per_byte;
    if (nb_bits == 1) {
        return num_bytes >= 32;
    }
    if (nb_bits == 2) {
        return num_bytes >= 16;
    }
    if (nb_bits == 4) {
        return num_bytes >= 128;
    }
    return false;
}

struct Run_search_with_eden_dc {
    using T = void;

    template <class BlockResultHandler>
    void f(BlockResultHandler& res, const IndexEDEN* index, const float* xq) {
        const size_t ntotal = index->ntotal;
        using SingleResultHandler =
                typename BlockResultHandler::SingleResultHandler;
        const int d = index->d;
        const bool use_batch_scan =
                use_eden_batch_scan(d, index->eden.nb_bits);

#pragma omp parallel
        {
            std::unique_ptr<EDENFlatCodesDistanceComputer> dc(
                    index->eden.get_distance_computer(index->center.data()));
            dc->code_size = index->eden.code_size;
            dc->codes = index->codes.data();
            SingleResultHandler resi(res);

#pragma omp for
            for (int64_t q = 0; q < static_cast<int64_t>(res.nq); q++) {
                resi.begin(q);
                dc->set_query(xq + d * q);

                if (use_batch_scan) {
                    size_t i = 0;
                    for (; i + 8 <= ntotal; i += 8) {
                        const bool keep0 = res.is_in_selection(i);
                        const bool keep1 = res.is_in_selection(i + 1);
                        const bool keep2 = res.is_in_selection(i + 2);
                        const bool keep3 = res.is_in_selection(i + 3);
                        const bool keep4 = res.is_in_selection(i + 4);
                        const bool keep5 = res.is_in_selection(i + 5);
                        const bool keep6 = res.is_in_selection(i + 6);
                        const bool keep7 = res.is_in_selection(i + 7);
                        if (keep0 && keep1 && keep2 && keep3 && keep4 &&
                            keep5 && keep6 && keep7) {
                            float dis[8];
                            dc->consecutive_distances_batch_8(i, dis);
                            resi.add_result(dis[0], i);
                            resi.add_result(dis[1], i + 1);
                            resi.add_result(dis[2], i + 2);
                            resi.add_result(dis[3], i + 3);
                            resi.add_result(dis[4], i + 4);
                            resi.add_result(dis[5], i + 5);
                            resi.add_result(dis[6], i + 6);
                            resi.add_result(dis[7], i + 7);
                        } else {
                            if (keep0) {
                                resi.add_result((*dc)(i), i);
                            }
                            if (keep1) {
                                resi.add_result((*dc)(i + 1), i + 1);
                            }
                            if (keep2) {
                                resi.add_result((*dc)(i + 2), i + 2);
                            }
                            if (keep3) {
                                resi.add_result((*dc)(i + 3), i + 3);
                            }
                            if (keep4) {
                                resi.add_result((*dc)(i + 4), i + 4);
                            }
                            if (keep5) {
                                resi.add_result((*dc)(i + 5), i + 5);
                            }
                            if (keep6) {
                                resi.add_result((*dc)(i + 6), i + 6);
                            }
                            if (keep7) {
                                resi.add_result((*dc)(i + 7), i + 7);
                            }
                        }
                    }
                    for (; i + 4 <= ntotal; i += 4) {
                        const bool keep0 = res.is_in_selection(i);
                        const bool keep1 = res.is_in_selection(i + 1);
                        const bool keep2 = res.is_in_selection(i + 2);
                        const bool keep3 = res.is_in_selection(i + 3);
                        if (keep0 && keep1 && keep2 && keep3) {
                            float dis0;
                            float dis1;
                            float dis2;
                            float dis3;
                            dc->distances_batch_4(
                                    i,
                                    i + 1,
                                    i + 2,
                                    i + 3,
                                    dis0,
                                    dis1,
                                    dis2,
                                    dis3);
                            resi.add_result(dis0, i);
                            resi.add_result(dis1, i + 1);
                            resi.add_result(dis2, i + 2);
                            resi.add_result(dis3, i + 3);
                        } else {
                            if (keep0) {
                                resi.add_result((*dc)(i), i);
                            }
                            if (keep1) {
                                resi.add_result((*dc)(i + 1), i + 1);
                            }
                            if (keep2) {
                                resi.add_result((*dc)(i + 2), i + 2);
                            }
                            if (keep3) {
                                resi.add_result((*dc)(i + 3), i + 3);
                            }
                        }
                    }
                    for (; i < ntotal; i++) {
                        if (res.is_in_selection(i)) {
                            resi.add_result((*dc)(i), i);
                        }
                    }
                } else {
                    for (size_t i = 0; i < ntotal; i++) {
                        if (res.is_in_selection(i)) {
                            resi.add_result((*dc)(i), i);
                        }
                    }
                }

                resi.end();
            }
        }
    }
};

} // namespace

void IndexEDEN::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(is_trained);
    const IDSelector* sel = params != nullptr ? params->sel : nullptr;
    Run_search_with_eden_dc r;
    dispatch_knn_ResultHandler(
            n, distances, labels, k, metric_type, sel, r, this, x);
}

void IndexEDEN::range_search(
        idx_t /*n*/,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params) const {
    const IDSelector* sel = params != nullptr ? params->sel : nullptr;
    Run_search_with_eden_dc r;
    dispatch_range_ResultHandler(result, radius, metric_type, sel, r, this, x);
}

} // namespace faiss
