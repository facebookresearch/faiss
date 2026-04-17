/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/IndexIVFFlat.h>
#include <faiss/impl/expanded_scanners.h>

#ifndef THE_SIMD_LEVEL
#error "THE_SIMD_LEVEL not defined"
#endif

namespace faiss {

constexpr faiss::SIMDLevel THE_SL = THE_SIMD_LEVEL;

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
