/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <cstdlib>

#include <faiss/impl/simdlib/simdlib_dispatch.h>

/*******************************************
 * The Scaler objects are used to specialize the handling of the
 * norm components in Additive quantizer fast-scan.
 ********************************************/

namespace faiss {

/// no-op handler
template <SIMDLevel SL = SINGLE_SIMD_LEVEL>
struct DummyScaler {
    static constexpr int nscale = 0;
    static constexpr SIMDLevel SL256 = simd256_level_selector<SL>::value;
    using simd32uint8 = simd32uint8_tpl<SL256>;
    using simd16uint16 = simd16uint16_tpl<SL256>;
    using simd64uint8 = simd64uint8_tpl<SL>;
    using simd32uint16 = simd32uint16_tpl<SL>;

    inline simd32uint8 lookup(const simd32uint8&, const simd32uint8&) const {
        FAISS_THROW_MSG("DummyScaler::lookup should not be called.");
        return simd32uint8(0);
    }

    inline simd16uint16 scale_lo(const simd32uint8&) const {
        FAISS_THROW_MSG("DummyScaler::scale_lo should not be called.");
        return simd16uint16(0);
    }

    inline simd16uint16 scale_hi(const simd32uint8&) const {
        FAISS_THROW_MSG("DummyScaler::scale_hi should not be called.");
        return simd16uint16(0);
    }

    inline simd64uint8 lookup(const simd64uint8&, const simd64uint8&) const {
        FAISS_THROW_MSG("DummyScaler::lookup should not be called.");
        return simd64uint8(0);
    }

    inline simd32uint16 scale_lo(const simd64uint8&) const {
        FAISS_THROW_MSG("DummyScaler::scale_lo should not be called.");
        return simd32uint16(0);
    }

    inline simd32uint16 scale_hi(const simd64uint8&) const {
        FAISS_THROW_MSG("DummyScaler::scale_hi should not be called.");
        return simd32uint16(0);
    }

    template <class dist_t>
    inline dist_t scale_one(const dist_t&) const {
        FAISS_THROW_MSG("DummyScaler::scale_one should not be called.");
        return 0;
    }
};

/// consumes 2x4 bits to encode a norm as a scalar additive quantizer
/// the norm is scaled because its range is larger than other components
template <SIMDLevel SL = SINGLE_SIMD_LEVEL>
struct NormTableScaler {
    static constexpr int nscale = 2;
    static constexpr SIMDLevel SL256 = simd256_level_selector<SL>::value;
    using simd32uint8 = simd32uint8_tpl<SL256>;
    using simd16uint16 = simd16uint16_tpl<SL256>;
    using simd64uint8 = simd64uint8_tpl<SL>;
    using simd32uint16 = simd32uint16_tpl<SL>;

    int scale_int;
    simd16uint16 scale_simd;

    explicit NormTableScaler(int scale) : scale_int(scale), scale_simd(scale) {}

    inline simd32uint8 lookup(const simd32uint8& lut, const simd32uint8& c)
            const {
        return lut.lookup_2_lanes(c);
    }

    inline simd16uint16 scale_lo(const simd32uint8& res) const {
        return simd16uint16(res) * scale_simd;
    }

    inline simd16uint16 scale_hi(const simd32uint8& res) const {
        return (simd16uint16(res) >> 8) * scale_simd;
    }

    inline simd64uint8 lookup(const simd64uint8& lut, const simd64uint8& c)
            const {
        return lut.lookup_4_lanes(c);
    }

    inline simd32uint16 scale_lo(const simd64uint8& res) const {
        auto scale_simd_wide = simd32uint16(scale_simd, scale_simd);
        return simd32uint16(res) * scale_simd_wide;
    }

    inline simd32uint16 scale_hi(const simd64uint8& res) const {
        auto scale_simd_wide = simd32uint16(scale_simd, scale_simd);
        return (simd32uint16(res) >> 8) * scale_simd_wide;
    }

    // for non-SIMD implem 2, 3, 4
    template <class dist_t>
    inline dist_t scale_one(const dist_t& x) const {
        return x * scale_int;
    }
};

} // namespace faiss
