/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cstring>

#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/utils/hamming.h>

/** Helpers shared by IndexSQFastScan and IndexIVFSQFastScan.
 *
 * Both re-quantise a higher-precision scalar quantizer down to 4-bit nibbles
 * for the SIMD scan and keep the original codes to rerank with, so the
 * classification of quantizer types and the nibble packing have to agree
 * between them.
 */

namespace faiss {
namespace sq_fastscan {

inline size_t roundup(size_t a, size_t b) {
    return (a + b - 1) / b * b;
}

inline bool is_native_4bit(ScalarQuantizer::QuantizerType qtype) {
    return qtype == ScalarQuantizer::QT_4bit ||
            qtype == ScalarQuantizer::QT_4bit_uniform;
}

inline bool needs_rerank(ScalarQuantizer::QuantizerType qtype) {
    return qtype == ScalarQuantizer::QT_6bit ||
            qtype == ScalarQuantizer::QT_8bit ||
            qtype == ScalarQuantizer::QT_8bit_uniform ||
            qtype == ScalarQuantizer::QT_8bit_direct ||
            qtype == ScalarQuantizer::QT_8bit_direct_signed;
}

inline bool is_fallback(ScalarQuantizer::QuantizerType qtype) {
    return !is_native_4bit(qtype) && !needs_rerank(qtype);
}

inline bool is_uniform_range(ScalarQuantizer::QuantizerType qtype) {
    return qtype == ScalarQuantizer::QT_8bit_uniform ||
            qtype == ScalarQuantizer::QT_8bit_direct ||
            qtype == ScalarQuantizer::QT_8bit_direct_signed;
}

inline void get_uniform_range(
        const ScalarQuantizer& sq,
        float& vmin,
        float& vdiff) {
    if (sq.qtype == ScalarQuantizer::QT_8bit_direct) {
        vmin = 0;
        vdiff = 255;
    } else if (sq.qtype == ScalarQuantizer::QT_8bit_direct_signed) {
        vmin = -128;
        vdiff = 255;
    } else {
        vmin = sq.trained[0];
        vdiff = sq.trained[1];
    }
}

/// Bits per dimension the quantizer stores.
inline int sq_bits(ScalarQuantizer::QuantizerType qtype) {
    if (qtype == ScalarQuantizer::QT_6bit) {
        return 6;
    }
    return is_native_4bit(qtype) ? 4 : 8;
}

/// Bits left over once the scan has taken its 4. Zero for a native 4-bit
/// code, which needs no rerank and therefore stores nothing extra.
inline int sq_rerank_bits(ScalarQuantizer::QuantizerType qtype) {
    return sq_bits(qtype) - 4;
}

/// Bytes per vector needed to hold the leftover bits of every dimension.
inline size_t sq_rerank_size(int d, ScalarQuantizer::QuantizerType qtype) {
    const int bits = sq_rerank_bits(qtype);
    return bits > 0 ? (size_t(d) * bits + 7) / 8 : 0;
}

/// The code for one value at the quantizer's own width.
inline int sq_code_of(float x, float vmin, float vdiff, int bits) {
    const int maxv = (1 << bits) - 1;
    const float f = (vdiff > 0) ? (x - vmin) / vdiff : 0.0f;
    return std::min(maxv, std::max(0, int(f * float(maxv))));
}

/// Per-dimension range of the trained quantizer, resolved once for a batch.
struct SQRanges {
    SQRanges(const ScalarQuantizer& sq, int d)
            : uniform(is_uniform_range(sq.qtype) ||
                      sq.qtype == ScalarQuantizer::QT_4bit_uniform) {
        if (uniform) {
            get_uniform_range(sq, vmin_s, vdiff_s);
        } else {
            vmin_arr = sq.trained.data();
            vdiff_arr = sq.trained.data() + d;
        }
    }

    float vmin(int m) const {
        return uniform ? vmin_s : vmin_arr[m];
    }
    float vdiff(int m) const {
        return uniform ? vdiff_s : vdiff_arr[m];
    }

    const bool uniform;
    const float* vmin_arr = nullptr;
    const float* vdiff_arr = nullptr;
    float vmin_s = 0, vdiff_s = 0;
};

/** Split each code into the top 4 bits the SIMD kernel scans and the rest,
 *  which rerank keeps. `hi` is packed two dimensions per byte as
 *  pq4_pack_codes expects. `lo` is bit-packed at `bits - 4` per dimension.
 *  Total width is the quantizer's own, not that width plus 4.
 */
inline void float_to_split_codes(
        const float* x,
        uint8_t* hi,
        uint8_t* lo,
        idx_t n,
        int d,
        size_t M2,
        const ScalarQuantizer& sq) {
    const int bits = sq_bits(sq.qtype);
    const int lo_bits = bits - 4;
    const size_t hi_stride = M2 / 2;
    const size_t lo_stride = sq_rerank_size(d, sq.qtype);
    const SQRanges ranges(sq, d);
    for (idx_t i = 0; i < n; i++) {
        const float* xi = x + i * d;
        uint8_t* hdst = hi + i * hi_stride;
        memset(hdst, 0, hi_stride);
        uint8_t* ldst = lo + i * lo_stride;
        memset(ldst, 0, lo_stride);
        BitstringWriter bsw(ldst, lo_stride);
        for (int m = 0; m < d; m++) {
            const int code =
                    sq_code_of(xi[m], ranges.vmin(m), ranges.vdiff(m), bits);
            const int h = code >> lo_bits;
            hdst[m / 2] |= (m & 1) ? uint8_t(h << 4) : uint8_t(h);
            bsw.write(code & ((1 << lo_bits) - 1), lo_bits);
        }
    }
}

/// Rebuild the code values of one vector from its two halves.
inline void split_to_codes(
        const uint8_t* hi,
        const uint8_t* lo,
        int d,
        ScalarQuantizer::QuantizerType qtype,
        int* codes_out) {
    const int lo_bits = sq_rerank_bits(qtype);
    BitstringReader bsr(lo, sq_rerank_size(d, qtype));
    for (int m = 0; m < d; m++) {
        const int h = (m & 1) ? (hi[m / 2] >> 4) : (hi[m / 2] & 0xF);
        codes_out[m] = (h << lo_bits) | int(bsr.read(lo_bits));
    }
}

inline void float_to_4bit_nibbles(
        const float* x,
        uint8_t* nibbles,
        idx_t n,
        int d,
        int M2,
        const ScalarQuantizer& sq) {
    const bool is_uniform = is_uniform_range(sq.qtype) ||
            sq.qtype == ScalarQuantizer::QT_4bit_uniform;
    const float* vmin_arr = nullptr;
    const float* vdiff_arr = nullptr;
    float vmin_s = 0, inv_vdiff_s = 0;

    if (is_uniform) {
        float vdiff;
        get_uniform_range(sq, vmin_s, vdiff);
        inv_vdiff_s = (vdiff > 0) ? (1.0f / vdiff) : 0;
    } else {
        vmin_arr = sq.trained.data();
        vdiff_arr = sq.trained.data() + d;
    }

    const int half = M2 / 2;
    for (idx_t i = 0; i < n; i++) {
        const float* xi = x + i * d;
        uint8_t* dst = nibbles + i * half;
        memset(dst, 0, half);
        for (int m = 0; m + 1 < d; m += 2) {
            float f0, f1;
            if (is_uniform) {
                f0 = (xi[m] - vmin_s) * inv_vdiff_s;
                f1 = (xi[m + 1] - vmin_s) * inv_vdiff_s;
            } else {
                f0 = (vdiff_arr[m] > 0) ? (xi[m] - vmin_arr[m]) / vdiff_arr[m]
                                        : 0;
                f1 = (vdiff_arr[m + 1] > 0)
                        ? (xi[m + 1] - vmin_arr[m + 1]) / vdiff_arr[m + 1]
                        : 0;
            }
            uint8_t lo = (uint8_t)std::min(15, std::max(0, (int)(f0 * 15.0f)));
            uint8_t hi = (uint8_t)std::min(15, std::max(0, (int)(f1 * 15.0f)));
            dst[m / 2] = lo | (hi << 4);
        }
        if (d & 1) {
            float f;
            if (is_uniform) {
                f = (xi[d - 1] - vmin_s) * inv_vdiff_s;
            } else {
                f = (vdiff_arr[d - 1] > 0)
                        ? (xi[d - 1] - vmin_arr[d - 1]) / vdiff_arr[d - 1]
                        : 0;
            }
            dst[(d - 1) / 2] =
                    (uint8_t)std::min(15, std::max(0, (int)(f * 15.0f)));
        }
    }
}

} // namespace sq_fastscan
} // namespace faiss
