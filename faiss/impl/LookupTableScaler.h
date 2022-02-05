/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <cstdlib>

#include <faiss/utils/simdlib.h>

namespace faiss {

struct DummyScaler {
    inline simd16uint16 scale(int sq, const simd16uint16& x) const {
        return x;
    }

    template <class dist_t>
    inline dist_t scale_one(int sq, const dist_t& x) const {
        return x;
    }
};

struct NormTableScaler {
    size_t M_scale;
    int scale_int;
    simd16uint16 scale_simd;

    NormTableScaler(int scale, size_t M_scale)
            : scale_int(scale), scale_simd(scale), M_scale(M_scale) {}

    inline simd16uint16 scale(int sq, const simd16uint16& x) const {
        if (sq < M_scale) {
            return x;
        }
        return x * scale_simd;
    }

    template <class dist_t>
    inline dist_t scale_one(int sq, const dist_t& x) const {
        if (sq < M_scale) {
            return x;
        }
        return x * scale_int;
    }
};

} // namespace faiss
