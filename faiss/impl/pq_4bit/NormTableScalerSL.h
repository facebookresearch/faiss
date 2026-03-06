/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/utils/simdlib.h>

namespace faiss {

/// Private SL-typed NormTableScaler. Mirrors the public NormTableScaler
/// but uses SL-parameterized SIMD types. Created from NormTableScaler's
/// scale_int inside ScannerMixIn.
template <SIMDLevel SL>
struct NormTableScalerSL {
    static constexpr int nscale = 2;
    static constexpr SIMDLevel SL256 = simd256_level_selector<SL>::value;

    int scale_int;
    simd16uint16<SL256> scale_simd;

    explicit NormTableScalerSL(int scale)
            : scale_int(scale), scale_simd(scale) {}

    inline simd32uint8<SL256> lookup(
            const simd32uint8<SL256>& lut,
            const simd32uint8<SL256>& c) const {
        return lut.lookup_2_lanes(c);
    }

    inline simd16uint16<SL256> scale_lo(const simd32uint8<SL256>& res) const {
        return simd16uint16<SL256>(res) * scale_simd;
    }

    inline simd16uint16<SL256> scale_hi(const simd32uint8<SL256>& res) const {
        return (simd16uint16<SL256>(res) >> 8) * scale_simd;
    }

    inline simd64uint8<SL> lookup(
            const simd64uint8<SL>& lut,
            const simd64uint8<SL>& c) const {
        return lut.lookup_4_lanes(c);
    }

    inline simd32uint16<SL> scale_lo(const simd64uint8<SL>& res) const {
        auto scale_simd_wide = simd32uint16<SL>(scale_simd, scale_simd);
        return simd32uint16<SL>(res) * scale_simd_wide;
    }

    inline simd32uint16<SL> scale_hi(const simd64uint8<SL>& res) const {
        auto scale_simd_wide = simd32uint16<SL>(scale_simd, scale_simd);
        return (simd32uint16<SL>(res) >> 8) * scale_simd_wide;
    }

    template <class dist_t>
    inline dist_t scale_one(const dist_t& x) const {
        return x * scale_int;
    }
};

} // namespace faiss
