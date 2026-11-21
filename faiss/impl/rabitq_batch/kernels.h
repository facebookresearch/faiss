/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/** Integer kernels of the RaBitQ batch build.
 *
 * Internal header, included only by batch_build-avx512-spr.cpp, which is
 * compiled with AVX-512 BW/VBMI/VNNI and AMX-TILE/AMX-INT8. Callers check the
 * CPU before calling into it.
 *
 * An 8-bit RaBitQ code is decoded to w in [0, 255] per dimension (sign bit as
 * the high bit, the 7 extra bits as the low bits). Rows are stored with a
 * stride rounded up to 64 bytes and zero padded; zero padding does not
 * change sum_k(w_a[k] * w_b[k]).
 *
 * The candidate side is packed by groups of 16 columns, per 64-dimension
 * block, in the layout the AMX B tile expects (and that the VNNI kernel
 * reads as well):
 *
 *   bp[((j * kc + c) * 16 + r) * 64 + col * 4 + t] =
 *           w[j * 16 + col][c * 64 + r * 4 + t]
 */

#include <immintrin.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace faiss {
namespace rabitq_batch {

/// d: dimension; stride: bytes per decoded row (multiple of 64); kc: number
/// of 64-dimension blocks; block: bytes of one packed group of 16 columns.
struct Dims {
    size_t d = 0, stride = 0, kc = 0, block = 0;
    Dims() = default;
    explicit Dims(size_t d)
            : d(d),
              stride((d + 63) / 64 * 64),
              kc(stride / 64),
              block(kc * 1024) {}
};

/// Decodes the w of dimensions [64 * chunk, 64 * chunk + 64) of an 8-bit
/// code; dimensions >= d are set to 0. Does not read past the extra bits.
inline __m512i decode64(const uint8_t* code, size_t d, size_t chunk) {
    const size_t k = chunk * 64;
    const size_t r = d - k < 64 ? d - k : 64;
    const uint8_t* extra = code + (d + 7) / 8 + 12 + chunk * 56;
    const size_t extra_bytes = (7 * r + 7) / 8;
    const __m512i packed =
            _mm512_maskz_loadu_epi8((__mmask64(1) << extra_bytes) - 1, extra);
    // each 64-bit lane receives the 7 source bytes of 8 dimensions
    alignas(64) static constexpr uint8_t spread[64] = {
            0,  1,  2,  3,  4,  5,  6,  6,  7,  8,  9,  10, 11, 12, 13, 13,
            14, 15, 16, 17, 18, 19, 20, 20, 21, 22, 23, 24, 25, 26, 27, 27,
            28, 29, 30, 31, 32, 33, 34, 34, 35, 36, 37, 38, 39, 40, 41, 41,
            42, 43, 44, 45, 46, 47, 48, 48, 49, 50, 51, 52, 53, 54, 55, 55};
    const __m512i lanes =
            _mm512_permutexvar_epi8(_mm512_load_si512(spread), packed);
    // dimension t of a lane starts at bit 7 t; keep the low 7 bits
    const __m512i shifts = _mm512_set1_epi64(0x312A231C150E0700LL);
    const __m512i e = _mm512_and_si512(
            _mm512_multishift_epi64_epi8(shifts, lanes),
            _mm512_set1_epi8(0x7F));
    uint64_t signs = 0;
    std::memcpy(&signs, code + k / 8, (r + 7) / 8);
    const __mmask64 valid = r == 64 ? ~__mmask64(0) : (__mmask64(1) << r) - 1;
    return _mm512_maskz_mov_epi8(
            valid,
            _mm512_mask_blend_epi8(
                    signs & valid,
                    e,
                    _mm512_or_si512(e, _mm512_set1_epi8(char(0x80)))));
}

struct alignas(64) TileConfig {
    uint8_t palette = 1, start = 0, reserved0[14]{};
    uint16_t colsb[16]{};
    uint8_t rows[16]{};
};
static_assert(sizeof(TileConfig) == 64, "AMX tile configuration is 64 bytes");

inline void amx_configure() {
    TileConfig cfg;
    for (int t = 0; t < 8; ++t) {
        cfg.rows[t] = 16;
        cfg.colsb[t] = 64;
    }
    _tile_loadconfig(&cfg);
}

inline void amx_release() {
    _tile_release();
}

/** 32 rows x 32 columns of sum_k(w_a[k] * w_b[k]), over the 64-dimension
 * blocks [c0, c1). If first is false, adds to the partial sums already in
 * out. Tiles 0-1 hold two groups of 16 rows, tiles 2-3 two groups of 16
 * packed columns, tiles 4-7 the 2 x 2 accumulators. */
inline void amx_block_part(
        const Dims& m,
        const uint8_t* rows,
        const uint8_t* bp,
        int32_t* out,
        size_t c0,
        size_t c1,
        bool first) {
    if (first) {
        _tile_zero(4);
        _tile_zero(5);
        _tile_zero(6);
        _tile_zero(7);
    } else {
        _tile_loadd(4, out, 128);
        _tile_loadd(5, out + 16, 128);
        _tile_loadd(6, out + 16 * 32, 128);
        _tile_loadd(7, out + 16 * 32 + 16, 128);
    }
    for (size_t c = c0; c < c1; ++c) {
        _tile_loadd(0, rows + c * 64, m.stride);
        _tile_loadd(1, rows + 16 * m.stride + c * 64, m.stride);
        _tile_loadd(2, bp + c * 1024, 64);
        _tile_loadd(3, bp + m.block + c * 1024, 64);
        _tile_dpbuud(4, 0, 2);
        _tile_dpbuud(6, 1, 2);
        _tile_dpbuud(5, 0, 3);
        _tile_dpbuud(7, 1, 3);
    }
    _tile_stored(4, out, 128);
    _tile_stored(5, out + 16, 128);
    _tile_stored(6, out + 16 * 32, 128);
    _tile_stored(7, out + 16 * 32 + 16, 128);
}

/** Same as amx_block_part() with AVX512-VNNI. vpdpbusd multiplies unsigned
 * by signed bytes, so the row side is stored as w - 128 (signed_rows), which
 * gives sum_k(w_b[k] * (w_a[k] - 128)); with first, the accumulators start
 * from 128 * W_b (col_sum = W_b = sum_k w_b[k]) to give sum_k(w_a w_b). */
inline void vnni_block_part(
        const Dims& m,
        const uint8_t* signed_rows,
        const uint8_t* bp,
        const int32_t* col_sum,
        int32_t* out,
        size_t k0,
        size_t k1,
        bool first) {
    const __m512i fix0 = _mm512_slli_epi32(_mm512_loadu_si512(col_sum), 7);
    const __m512i fix1 = _mm512_slli_epi32(_mm512_loadu_si512(col_sum + 16), 7);
    for (size_t rr = 0; rr < 32; rr += 8) {
        __m512i acc0[8], acc1[8];
        for (int i = 0; i < 8; ++i) {
            acc0[i] = first ? fix0 : _mm512_loadu_si512(out + (rr + i) * 32);
            acc1[i] =
                    first ? fix1 : _mm512_loadu_si512(out + (rr + i) * 32 + 16);
        }
        for (size_t c = k0; c < k1; ++c) {
            for (size_t r = 0; r < 16; ++r) {
                const uint8_t* b = bp + c * 1024 + r * 64;
                const __m512i b0 = _mm512_load_si512(b);
                const __m512i b1 = _mm512_load_si512(b + m.block);
                const uint8_t* a = signed_rows + rr * m.stride + c * 64 + r * 4;
                for (int i = 0; i < 8; ++i) {
                    int32_t word;
                    std::memcpy(&word, a + i * m.stride, 4);
                    const __m512i q = _mm512_set1_epi32(word);
                    acc0[i] = _mm512_dpbusd_epi32(acc0[i], b0, q);
                    acc1[i] = _mm512_dpbusd_epi32(acc1[i], b1, q);
                }
            }
        }
        for (int i = 0; i < 8; ++i) {
            _mm512_storeu_si512(out + (rr + i) * 32, acc0[i]);
            _mm512_storeu_si512(out + (rr + i) * 32 + 16, acc1[i]);
        }
    }
}

} // namespace rabitq_batch
} // namespace faiss
