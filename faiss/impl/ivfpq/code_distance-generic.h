/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/ProductQuantizer.h>

namespace faiss {

/// Returns the distance to a single code.
template <typename PQDecoderT>
inline float distance_single_code_generic(
        // the product quantizer
        const ProductQuantizer& pq,
        // precomputed distances, layout (M, ksub)
        const float* sim_table,
        // the code
        const uint8_t* code) {
    PQDecoderT decoder(code, pq.nbits);

    const float* tab = sim_table;
    float result = 0;

    for (size_t m = 0; m < pq.M; m++) {
        result += tab[decoder.decode()];
        tab += pq.ksub;
    }

    return result;
}

/// Combines 4 operations of distance_single_code()
/// General-purpose version.
template <typename PQDecoderT>
inline void distance_four_codes_generic(
        // the product quantizer
        const ProductQuantizer& pq,
        // precomputed distances, layout (M, ksub)
        const float* sim_table,
        // codes
        const uint8_t* __restrict code0,
        const uint8_t* __restrict code1,
        const uint8_t* __restrict code2,
        const uint8_t* __restrict code3,
        // computed distances
        float& result0,
        float& result1,
        float& result2,
        float& result3) {
    PQDecoderT decoder0(code0, pq.nbits);
    PQDecoderT decoder1(code1, pq.nbits);
    PQDecoderT decoder2(code2, pq.nbits);
    PQDecoderT decoder3(code3, pq.nbits);

    const float* tab = sim_table;
    result0 = 0;
    result1 = 0;
    result2 = 0;
    result3 = 0;

    for (size_t m = 0; m < pq.M; m++) {
        result0 += tab[decoder0.decode()];
        result1 += tab[decoder1.decode()];
        result2 += tab[decoder2.decode()];
        result3 += tab[decoder3.decode()];
        tab += pq.ksub;
    }
}

} // namespace faiss
