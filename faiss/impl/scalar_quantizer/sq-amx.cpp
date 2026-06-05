/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// AMX-BF16 tile distance computer for the bf16 scalar quantizer.
//
// QuantizerBF16 already stores each component as bf16, so the SQ code storage
// is exactly the bf16 data the AMX tile engine consumes -- no re-encoding.
// distances_batch_16() computes 16 bf16 inner products in one AMX tile pass.
// Only the inner-product metric uses this path: L2 (which needs a per-vector
// ||c||^2 that offsets the tile gain) and all other quantizer types delegate
// to the AVX512 scalar-quantizer distance computers.

#ifdef COMPILE_SIMD_AMX

#include <immintrin.h>

#include <cstring>
#include <vector>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/scalar_quantizer/codecs.h>
#include <faiss/impl/scalar_quantizer/distance_computers.h>
#include <faiss/impl/scalar_quantizer/quantizers.h>
#include <faiss/impl/scalar_quantizer/scanners.h>
#include <faiss/impl/scalar_quantizer/similarities.h>
#include <faiss/impl/simdlib/simdlib_avx512.h>

#include <faiss/impl/scalar_quantizer/sq-avx512-impl.h>

namespace faiss {
namespace scalar_quantizer {

namespace {

// AMX tile configuration for a batch-16 bf16 dot product. Swapped-tile
// layout (candidates in tile 0, query in tile 1, fp32 accumulators in
// tile 2) avoids VNNI repacking of the query. Ported from hnswlib-amx.
struct alignas(64) AmxTileConfig {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved[14];
    uint16_t colsb[16];
    uint8_t rows[16];
};

inline void amx_tile_config_batch16() {
    AmxTileConfig cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.palette_id = 1;
    cfg.rows[0] = 16; // tile 0: 16 candidates ...
    cfg.colsb[0] = 64; //         ... x 64 bytes (32 bf16) per row
    cfg.rows[1] = 16; // tile 1: query, 16 rows ...
    cfg.colsb[1] = 4; //         ... x 4 bytes
    cfg.rows[2] = 16; // tile 2: 16 fp32 accumulators ...
    cfg.colsb[2] = 4;
    // GCC does not model _tile_loadconfig as reading the 64-byte cfg, so it
    // dead-store-eliminates the rows/colsb writes above and loads an all-zero
    // (unconfigured) tile shape -> #UD on the first tile op. Force the stores.
    asm volatile("" : : "m"(cfg) : "memory");
    _tile_loadconfig(&cfg);
}

inline float bf16_to_float(uint16_t v) {
    uint32_t bits = static_cast<uint32_t>(v) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

inline uint16_t float_to_bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    // round-to-nearest-even
    uint32_t rounding_bias = ((bits >> 16) & 1) + 0x7FFF;
    return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

// Raw inner products of one bf16 query against 16 bf16 candidates.
// AMX tiles must already be configured by the caller.
inline void amx_bf16_dot16(
        const uint16_t* query_bf16,
        const uint16_t* const candidates[16],
        float* dots,
        size_t dim) {
    _tile_zero(2);
    alignas(64) uint8_t tile_a[16 * 64];

    const size_t dim32 = dim & ~size_t(31);
    for (size_t d = 0; d < dim32; d += 32) {
        for (int n = 0; n < 16; n++) {
            std::memcpy(tile_a + n * 64, candidates[n] + d, 64);
        }
        _tile_loadd(0, tile_a, 64);
        _tile_loadd(1, query_bf16 + d, 4);
        _tile_dpbf16ps(2, 0, 1);
    }

    alignas(64) float tile_c[16];
    _tile_stored(2, tile_c, 4);
    for (int n = 0; n < 16; n++) {
        dots[n] = tile_c[n];
    }

    // Tail dimensions (dim % 32) in scalar.
    if (dim32 < dim) {
        for (int n = 0; n < 16; n++) {
            float t = 0;
            for (size_t d = dim32; d < dim; d++) {
                t += bf16_to_float(query_bf16[d]) *
                        bf16_to_float(candidates[n][d]);
            }
            dots[n] += t;
        }
    }
}

} // namespace

// BF16 distance computer that fills a full AMX tile (16 candidates) per
// distances_batch_16 call. Inherits the AVX512 bf16 DC so that set_query's
// fp32 query pointer and the scalar operator()/query_to_code path (used for
// the <16 remainder of a neighbor list) keep working unchanged.
template <class Similarity>
struct DCBF16Amx : SQDistanceComputer {
    // AVX512 bf16 distance computer, composed (not inherited) because its
    // set_query / query_to_code are final. Used for the scalar operator() /
    // query_to_code path (the <16 remainder of a neighbor list) and for
    // symmetric_dis.
    DCTemplate<QuantizerBF16<SIMDLevel::AVX512>, Similarity, SIMDLevel::AVX512>
            scalar_dc;

    std::vector<uint16_t> query_bf16;

    DCBF16Amx(size_t d, const std::vector<float>& trained)
            : scalar_dc(d, trained) {}

    void set_query(const float* x) override {
        this->q = x;
        scalar_dc.set_query(x);
        const size_t d = scalar_dc.quant.d;
        query_bf16.resize(d);
        for (size_t i = 0; i < d; i++) {
            query_bf16[i] = float_to_bf16(x[i]);
        }
    }

    float query_to_code(const uint8_t* code) const override {
        return scalar_dc.query_to_code(code);
    }

    void distances_batch_16(const idx_t* idx, float* dis) override {
        // AMX TILECFG is caller-saved: the HNSW loop makes function calls
        // between batches that clobber it, so (re)configure every call.
        amx_tile_config_batch16();
        const size_t d = scalar_dc.quant.d;
        const uint16_t* cand[16];
        for (int j = 0; j < 16; j++) {
            cand[j] = reinterpret_cast<const uint16_t*>(
                    this->codes + idx[j] * this->code_size);
        }
        // Raw inner product written straight into dis; the IP metric wraps the
        // storage DC in a NegativeDistanceComputer at the HNSW layer.
        amx_bf16_dot16(query_bf16.data(), cand, dis, d);
        _tile_release();
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return scalar_dc.compute_code_distance(
                this->codes + i * this->code_size,
                this->codes + j * this->code_size);
    }
};

// AMX distance-computer selection: only the bf16 quantizer has a dedicated
// AMX tile kernel; all other quantizer types fall back to AVX512.
template <>
SQDistanceComputer* sq_select_distance_computer<SIMDLevel::AMX>(
        MetricType metric,
        ScalarQuantizer::QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained) {
    // AMX accelerates only the bf16 inner-product distance. L2 (which needs a
    // per-vector ||c||^2 that erodes the tile speedup) and every other
    // quantizer type fall back to the AVX512 scalar-quantizer path.
    if (qtype == ScalarQuantizer::QT_bf16 && metric == METRIC_INNER_PRODUCT) {
        return new DCBF16Amx<SimilarityIP<SIMDLevel::AVX512>>(d, trained);
    }
    return sq_select_distance_computer<SIMDLevel::AVX512>(
            metric, qtype, d, trained);
}

} // namespace scalar_quantizer
} // namespace faiss

#endif // COMPILE_SIMD_AMX
