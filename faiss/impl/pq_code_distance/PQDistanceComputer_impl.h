/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifndef THE_SIMD_LEVEL
#error "THE_SIMD_LEVEL must be defined before including PQDistanceComputer_impl.h"
#endif

#include <faiss/IndexPQ.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/pq_code_distance/pq_code_distance-inl.h>

namespace faiss {
namespace pq_code_distance {

template <class PQCodeDist>
struct PQDistanceComputer : FlatCodesDistanceComputer {
    using PQDecoder = typename PQCodeDist::PQDecoder;
    size_t d;
    MetricType metric;
    idx_t nb;
    const ProductQuantizer& pq;
    const float* sdc;
    std::vector<float> precomputed_table;
    size_t ndis;
    const float* q;

    float distance_to_code(const uint8_t* code) final {
        ndis++;

        float dis = PQCodeDist::distance_single_code(
                pq.M, pq.nbits, precomputed_table.data(), code);
        return dis;
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        FAISS_THROW_IF_NOT(sdc);
        const float* sdci = sdc;
        float accu = 0;
        PQDecoder codei(codes + i * code_size, pq.nbits);
        PQDecoder codej(codes + j * code_size, pq.nbits);

        for (size_t l = 0; l < pq.M; l++) {
            accu += sdci[codei.decode() + (codej.decode() << codei.nbits)];
            sdci += uint64_t(1) << (2 * codei.nbits);
        }
        ndis++;
        return accu;
    }

    explicit PQDistanceComputer(const IndexPQ& storage)
            : FlatCodesDistanceComputer(
                      storage.codes.data(),
                      storage.code_size),
              pq(storage.pq),
              q(nullptr) {
        precomputed_table.resize(pq.M * pq.ksub);
        nb = storage.ntotal;
        d = storage.d;
        metric = storage.metric_type;
        if (pq.sdc_table.size() == pq.ksub * pq.ksub * pq.M) {
            sdc = pq.sdc_table.data();
        } else {
            sdc = nullptr;
        }
        ndis = 0;
    }

    void set_query(const float* x) override {
        q = x;
        if (metric == METRIC_L2) {
            pq.compute_distance_table(x, precomputed_table.data());
        } else {
            pq.compute_inner_prod_table(x, precomputed_table.data());
        }
    }
};

template <SIMDLevel SL>
FlatCodesDistanceComputer* get_PQFlatCodesDistanceComputer(
        const IndexPQ& index);

// NOLINTNEXTLINE(facebook-hte-MisplacedTemplateSpecialization)
template <>
FlatCodesDistanceComputer* get_PQFlatCodesDistanceComputer<THE_SIMD_LEVEL>(
        const IndexPQ& index) {
    if (index.pq.nbits == 8) {
        return new PQDistanceComputer<
                PQCodeDistance<PQDecoder8, THE_SIMD_LEVEL>>(index);
    } else if (index.pq.nbits == 16) {
        return new PQDistanceComputer<
                PQCodeDistance<PQDecoder16, THE_SIMD_LEVEL>>(index);
    } else {
        return new PQDistanceComputer<
                PQCodeDistance<PQDecoderGeneric, THE_SIMD_LEVEL>>(index);
    }
}

} // namespace pq_code_distance
} // namespace faiss
