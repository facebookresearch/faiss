/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <cstdlib>

#include <faiss/utils/simdlib.h>

/*******************************************
 * The Scaler objects are used to specialize the handling of the
 * norm components in Additive quantizer fast-scan.
 ********************************************/

namespace faiss {

/// no-op handler
template <SIMDLevel SL = SINGLE_SIMD_LEVEL_256>
struct DummyScaler {
    static constexpr int nscale = 0;
    static constexpr SIMDLevel SL256 = simd256_level_selector<SL>::value;

    inline simd32uint8<SL256> lookup(
            const simd32uint8<SL256>&,
            const simd32uint8<SL256>&) const {
        FAISS_THROW_MSG("DummyScaler::lookup should not be called.");
        return simd32uint8<SL256>(0);
    }

    inline simd16uint16<SL256> scale_lo(const simd32uint8<SL256>&) const {
        FAISS_THROW_MSG("DummyScaler::scale_lo should not be called.");
        return simd16uint16<SL256>(0);
    }

    inline simd16uint16<SL256> scale_hi(const simd32uint8<SL256>&) const {
        FAISS_THROW_MSG("DummyScaler::scale_hi should not be called.");
        return simd16uint16<SL256>(0);
    }

    inline simd64uint8<SL> lookup(
            const simd64uint8<SL>&,
            const simd64uint8<SL>&) const {
        FAISS_THROW_MSG("DummyScaler::lookup should not be called.");
        return simd64uint8<SL>(0);
    }

    inline simd32uint16<SL> scale_lo(const simd64uint8<SL>&) const {
        FAISS_THROW_MSG("DummyScaler::scale_lo should not be called.");
        return simd32uint16<SL>(0);
    }

    inline simd32uint16<SL> scale_hi(const simd64uint8<SL>&) const {
        FAISS_THROW_MSG("DummyScaler::scale_hi should not be called.");
        return simd32uint16<SL>(0);
    }

    template <class dist_t>
    inline dist_t scale_one(const dist_t&) const {
        FAISS_THROW_MSG("DummyScaler::scale_one should not be called.");
        return 0;
    }
};

/// consumes 2x4 bits to encode a norm as a scalar additive quantizer
/// the norm is scaled because its range is larger than other components
struct NormTableScaler {
    static constexpr int nscale = 2;
    int scale_int;
    simd16uint16<SINGLE_SIMD_LEVEL_256> scale_simd;

    explicit NormTableScaler(int scale) : scale_int(scale), scale_simd(scale) {}

    inline simd32uint8<SINGLE_SIMD_LEVEL_256> lookup(
            const simd32uint8<SINGLE_SIMD_LEVEL_256>& lut,
            const simd32uint8<SINGLE_SIMD_LEVEL_256>& c) const {
        return lut.lookup_2_lanes(c);
    }

    inline simd16uint16<SINGLE_SIMD_LEVEL_256> scale_lo(
            const simd32uint8<SINGLE_SIMD_LEVEL_256>& res) const {
        return simd16uint16<SINGLE_SIMD_LEVEL_256>(res) * scale_simd;
    }

    inline simd16uint16<SINGLE_SIMD_LEVEL_256> scale_hi(
            const simd32uint8<SINGLE_SIMD_LEVEL_256>& res) const {
        return (simd16uint16<SINGLE_SIMD_LEVEL_256>(res) >> 8) * scale_simd;
    }

#ifdef __AVX512F__
    inline simd64uint8<SINGLE_SIMD_LEVEL> lookup(
            const simd64uint8<SINGLE_SIMD_LEVEL>& lut,
            const simd64uint8<SINGLE_SIMD_LEVEL>& c) const {
        return lut.lookup_4_lanes(c);
    }

    inline simd32uint16<SINGLE_SIMD_LEVEL> scale_lo(
            const simd64uint8<SINGLE_SIMD_LEVEL>& res) const {
        auto scale_simd_wide =
                simd32uint16<SINGLE_SIMD_LEVEL>(scale_simd, scale_simd);
        return simd32uint16<SINGLE_SIMD_LEVEL>(res) * scale_simd_wide;
    }

    inline simd32uint16<SINGLE_SIMD_LEVEL> scale_hi(
            const simd64uint8<SINGLE_SIMD_LEVEL>& res) const {
        auto scale_simd_wide =
                simd32uint16<SINGLE_SIMD_LEVEL>(scale_simd, scale_simd);
        return (simd32uint16<SINGLE_SIMD_LEVEL>(res) >> 8) * scale_simd_wide;
    }
#endif

    // for non-SIMD implem 2, 3, 4
    template <class dist_t>
    inline dist_t scale_one(const dist_t& x) const {
        return x * scale_int;
    }
};

} // namespace faiss
