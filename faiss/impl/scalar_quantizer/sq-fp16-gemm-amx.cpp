/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// This file is compiled with its own -mamx-tile -mamx-fp16 flags (see
// CMakeLists.txt). Capability detection and the thread's tile-data
// permission request live in sq-fp16-gemm-common.cpp; an explicit request
// for the AMX kernel never silently falls back to another one.
#include <faiss/impl/scalar_quantizer/sq-fp16-gemm.h>

#include <immintrin.h>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/fp16.h>

namespace faiss {
namespace fp16_gemm {
namespace {

struct alignas(64) TileConfig {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved[14];
    uint16_t colsb[16];
    uint8_t rows[16];
};
static_assert(sizeof(TileConfig) == 64);

void configure_tiles() {
    TileConfig config{};
    config.palette_id = 1;
    for (size_t tile = 0; tile < 8; ++tile) {
        config.rows[tile] = 16;
        config.colsb[tile] = 64;
    }
    // Without this, the compiler can conclude that `config` is otherwise
    // unused and dead-code-eliminate the writes above; _tile_loadconfig
    // takes a raw pointer, so it doesn't see that it reads them.
    asm volatile("" : : "m"(config) : "memory");
    _tile_loadconfig(&config);
}

bool is_subnormal(uint16_t value) {
    return (value & 0x7c00) == 0 && (value & 0x03ff) != 0;
}

// Runs the actual instruction and checks each input side's smallest
// subnormal separately. Only called from an AMX entry point that has
// already verified tile permission; not part of CPUID capability
// detection.
AMXSubnormalBehavior probe_subnormals() {
    alignas(64) uint16_t a[16 * 32]{};
    alignas(64) uint16_t b[16 * 32]{};
    alignas(64) float c[16 * 16]{};
    a[0] = 1;
    a[32 + 2] = 0x3c00;
    b[0] = 0x3c00;
    b[32 + 2] = 1;
    configure_tiles();
    _tile_zero(0);
    _tile_loadd(4, a, 64);
    _tile_loadd(6, b, 64);
    _tile_dpfp16ps(0, 4, 6);
    _tile_stored(0, c, 64);
    _tile_release();
    const float expected = faiss::decode_fp16(1);
    FAISS_THROW_IF_NOT_MSG(
            (c[0] == 0 || c[0] == expected) &&
                    (c[17] == 0 || c[17] == expected) && c[0] == c[17],
            "AMX-FP16 subnormal probe behaved inconsistently or unexpectedly "
            "between the two input sides");
    return {c[0], c[17], c[0] == expected};
}

} // namespace

AMXSubnormalBehavior amx_subnormal_behavior_impl() {
    static const AMXSubnormalBehavior behavior = probe_subnormals();
    return behavior;
}

void gemm_amx_impl(
        const uint16_t* a,
        size_t m,
        size_t a_stride,
        const uint16_t* packed_b,
        const uint16_t* b_rows,
        size_t b_stride,
        size_t n_padded,
        size_t n_real,
        size_t d,
        float* out,
        size_t out_stride) {
    const size_t full = (d / 32) * 32;
    const size_t kblocks = full / 32;
    const bool preserves = amx_subnormal_behavior_impl().preserves;
    configure_tiles();
    // Every dimension of a 16-centroid group is stored contiguously (see
    // sq-fp16-gemm.h). Scoring 32 centroids reads tile 6 and tile 7 each
    // start to end through one group, 1 KiB apart.
    size_t i = 0;
    for (; i + kFP16BlockM <= m; i += kFP16BlockM) {
        const uint16_t* a0 = a + i * a_stride;
        const uint16_t* a1 = a0 + 16 * a_stride;
        float* c0 = out + i * out_stride;
        float* c1 = c0 + 16 * out_stride;
        for (size_t j = 0; j < n_padded; j += 32) {
            const uint16_t* group0 = packed_b + (j / 16) * kblocks * 512;
            const uint16_t* group1 = group0 + kblocks * 512;
            _tile_zero(0);
            _tile_zero(1);
            _tile_zero(2);
            _tile_zero(3);
            for (size_t kb = 0; kb < kblocks; ++kb) {
                _tile_loadd(4, a0 + kb * 32, a_stride * 2);
                _tile_loadd(6, group0 + kb * 512, 64);
                _tile_dpfp16ps(0, 4, 6);
                _tile_loadd(7, group1 + kb * 512, 64);
                _tile_dpfp16ps(1, 4, 7);
                _tile_loadd(5, a1 + kb * 32, a_stride * 2);
                _tile_dpfp16ps(2, 5, 6);
                _tile_dpfp16ps(3, 5, 7);
            }
            _tile_stored(0, c0 + j, out_stride * 4);
            _tile_stored(1, c0 + j + 16, out_stride * 4);
            _tile_stored(2, c1 + j, out_stride * 4);
            _tile_stored(3, c1 + j + 16, out_stride * 4);
        }
    }
    if (i < m) {
        // Only 16 rows left (always true once a single query has been
        // padded up): load only one A tile and compute only the top half,
        // instead of doing half the multiplies against padding rows. Each
        // result's accumulation order matches the full-block case above.
        const uint16_t* a0 = a + i * a_stride;
        float* c0 = out + i * out_stride;
        for (size_t j = 0; j < n_padded; j += 32) {
            const uint16_t* group0 = packed_b + (j / 16) * kblocks * 512;
            const uint16_t* group1 = group0 + kblocks * 512;
            _tile_zero(0);
            _tile_zero(1);
            for (size_t kb = 0; kb < kblocks; ++kb) {
                _tile_loadd(4, a0 + kb * 32, a_stride * 2);
                _tile_loadd(6, group0 + kb * 512, 64);
                _tile_dpfp16ps(0, 4, 6);
                _tile_loadd(7, group1 + kb * 512, 64);
                _tile_dpfp16ps(1, 4, 7);
            }
            _tile_stored(0, c0 + j, out_stride * 4);
            _tile_stored(1, c0 + j + 16, out_stride * 4);
        }
    }
    _tile_release();

    if (!preserves) {
        // On CPUs that flush subnormal fp16 inputs to zero, patch in just
        // the contribution of those dimensions; everything else is still
        // computed by AMX. When both sides are subnormal at the same
        // dimension, add it back only once.
        for (size_t i = 0; i < m; ++i) {
            for (size_t kk = 0; kk < full; ++kk) {
                const uint16_t av = a[i * a_stride + kk];
                if (!is_subnormal(av)) {
                    continue;
                }
                const float af = faiss::decode_fp16(av);
                for (size_t j = 0; j < n_real; ++j) {
                    out[i * out_stride + j] = std::fma(
                            af,
                            faiss::decode_fp16(b_rows[j * b_stride + kk]),
                            out[i * out_stride + j]);
                }
            }
        }
        for (size_t j = 0; j < n_real; ++j) {
            for (size_t kk = 0; kk < full; ++kk) {
                const uint16_t bv = b_rows[j * b_stride + kk];
                if (!is_subnormal(bv)) {
                    continue;
                }
                const float bf = faiss::decode_fp16(bv);
                for (size_t i = 0; i < m; ++i) {
                    const uint16_t av = a[i * a_stride + kk];
                    if (is_subnormal(av)) {
                        continue; // already added above
                    }
                    out[i * out_stride + j] = std::fma(
                            faiss::decode_fp16(av),
                            bf,
                            out[i * out_stride + j]);
                }
            }
        }
    }
    for (size_t i = 0; i < m && full < d; ++i) {
        for (size_t j = 0; j < n_real; ++j) {
            float acc = out[i * out_stride + j];
            for (size_t kk = full; kk < d; ++kk) {
                acc = std::fma(
                        faiss::decode_fp16(a[i * a_stride + kk]),
                        faiss::decode_fp16(b_rows[j * b_stride + kk]),
                        acc);
            }
            out[i * out_stride + j] = acc;
        }
    }
}

} // namespace fp16_gemm
} // namespace faiss
