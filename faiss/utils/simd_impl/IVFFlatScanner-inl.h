/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/IndexIVFFlat.h>
#include <faiss/impl/expanded_scanners.h>
#include <faiss/utils/distances.h>

#ifndef THE_SIMD_LEVEL
#error "THE_SIMD_LEVEL not defined"
#endif

namespace faiss {

constexpr faiss::SIMDLevel THE_SL = THE_SIMD_LEVEL;

template <MetricType metric, class ScannerType, class C, bool store_pairs>
size_t run_ivfflat_scan_codes_batch_4(
        const ScannerType& scanner,
        size_t list_size,
        const uint8_t* codes,
        const idx_t* ids,
        ResultHandler& handler) {
    static_assert(metric == METRIC_L2 || metric == METRIC_INNER_PRODUCT);
    size_t nup = 0;
    float threshold = handler.threshold;
    const size_t d = scanner.vd.d;
    size_t j = 0;

    auto add_result = [&](size_t offset, float dis) {
        if (C::cmp(threshold, dis)) {
            const idx_t id = store_pairs ? lo_build(scanner.list_no, offset)
                                         : ids[offset];
            if (handler.add_result(dis, id)) {
                handler.stats.nheap_updates++;
                nup++;
                threshold = handler.threshold;
            }
        }
    };

    for (; list_size - j >= 4; j += 4) {
        const float* y0 = reinterpret_cast<const float*>(codes);
        const float* y1 = y0 + d;
        const float* y2 = y1 + d;
        const float* y3 = y2 + d;
        float dis0, dis1, dis2, dis3;
        if constexpr (metric == METRIC_L2) {
            fvec_L2sqr_batch_4<THE_SL>(
                    scanner.xi, y0, y1, y2, y3, d, dis0, dis1, dis2, dis3);
        } else {
            fvec_inner_product_batch_4<THE_SL>(
                    scanner.xi, y0, y1, y2, y3, d, dis0, dis1, dis2, dis3);
        }
        handler.stats.scan_cnt += 4;
        add_result(j + 0, dis0);
        add_result(j + 1, dis1);
        add_result(j + 2, dis2);
        add_result(j + 3, dis3);
        codes += 4 * scanner.code_size;
    }

    for (; j < list_size; ++j) {
        handler.stats.scan_cnt++;
        add_result(j, scanner.distance_to_code(codes));
        codes += scanner.code_size;
    }
    return nup;
}

template <MetricType metric, class ScannerType, class C>
size_t run_ivfflat_scan_codes_batch_4(
        const ScannerType& scanner,
        size_t list_size,
        const uint8_t* codes,
        const idx_t* ids,
        ResultHandler& handler) {
    if (scanner.store_pairs) {
        return run_ivfflat_scan_codes_batch_4<metric, ScannerType, C, true>(
                scanner, list_size, codes, ids, handler);
    }
    return run_ivfflat_scan_codes_batch_4<metric, ScannerType, C, false>(
            scanner, list_size, codes, ids, handler);
}

#define DEFINE_IVFFLAT_SCANNER_METHODS(mt)                                     \
    template <>                                                                \
    float IVFFlatScanner<VectorDistance<mt, THE_SL>>::distance_to_code(        \
            const uint8_t* code) const {                                       \
        const float* yj = (float*)code;                                        \
        return vd(xi, yj);                                                     \
    }                                                                          \
    template <>                                                                \
    size_t IVFFlatScanner<VectorDistance<mt, THE_SL>>::scan_codes(             \
            size_t list_size,                                                  \
            const uint8_t* codes,                                              \
            const idx_t* ids,                                                  \
            ResultHandler& handler) const {                                    \
        if constexpr (mt == METRIC_L2 || mt == METRIC_INNER_PRODUCT) {         \
            if (this->sel == nullptr && list_size >= 4) {                      \
                return run_ivfflat_scan_codes_batch_4<                         \
                        mt,                                                    \
                        IVFFlatScanner<VectorDistance<mt, THE_SL>>,            \
                        C>(*this, list_size, codes, ids, handler);             \
            }                                                                  \
        }                                                                      \
        return run_scan_codes_fix_C<C>(*this, list_size, codes, ids, handler); \
    }

DEFINE_IVFFLAT_SCANNER_METHODS(METRIC_L2)
DEFINE_IVFFLAT_SCANNER_METHODS(METRIC_INNER_PRODUCT)
DEFINE_IVFFLAT_SCANNER_METHODS(METRIC_L1)
DEFINE_IVFFLAT_SCANNER_METHODS(METRIC_Linf)
DEFINE_IVFFLAT_SCANNER_METHODS(METRIC_Lp)
DEFINE_IVFFLAT_SCANNER_METHODS(METRIC_Canberra)
DEFINE_IVFFLAT_SCANNER_METHODS(METRIC_BrayCurtis)
DEFINE_IVFFLAT_SCANNER_METHODS(METRIC_JensenShannon)
DEFINE_IVFFLAT_SCANNER_METHODS(METRIC_Jaccard)
DEFINE_IVFFLAT_SCANNER_METHODS(METRIC_NaNEuclidean)
DEFINE_IVFFLAT_SCANNER_METHODS(METRIC_GOWER)

#undef DEFINE_IVFFLAT_SCANNER_METHODS

} // namespace faiss
