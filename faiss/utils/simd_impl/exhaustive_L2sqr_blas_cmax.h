/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>

#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/simd_levels.h>
#include <faiss/utils/utils.h>

namespace faiss {

/** fp32 query tiles for the BLAS exhaustive-search kernels.
 *
 * A kernel calls prepare_norms() once, then tile(i0, i1, x_norms) for each
 * query tile; the returned rows stay valid until the next tile() call.
 */
struct Fp32QueryTiles {
    const float* x;
    size_t d;

    /// squared L2 norms of all nx queries
    void prepare_norms(size_t nx, float* x_norms) const {
        fvec_norms_L2sqr(x_norms, x, d, nx);
    }

    const float* tile(size_t i0, size_t /*i1*/, float* /*x_norms*/) const {
        return x + i0 * d;
    }
};

/** Packed IEEE binary16 query tiles for the BLAS exhaustive-search kernels.
 *
 * Queries are widened in chunks of whole tiles (at most chunk_bytes of fp32,
 * at least one tile) into a buffer reused across chunks: the fp32 copy stays
 * bounded and each widened tile serves all database tiles. Norms are computed
 * from the widened rows, so results match the fp32 kernels on the widened
 * queries.
 */
struct Fp16QueryTiles {
    static constexpr size_t chunk_bytes = size_t(16) << 20;

    const uint16_t* x;
    size_t d;
    size_t nx;
    std::unique_ptr<float[]> buffer;
    size_t chunk_rows = 0;
    size_t chunk_begin = 0;
    size_t chunk_end = 0;

    Fp16QueryTiles(const uint16_t* x_in, size_t d_in, size_t nx_in)
            : x(x_in), d(d_in), nx(nx_in) {}

    void prepare_norms(size_t /*nx*/, float* /*x_norms*/) const {}

    /** widened rows [i0, i1), and their norms if x_norms is not null
     *
     * Tiles are requested in order, all of the same size except the last.
     */
    const float* tile(size_t i0, size_t i1, float* x_norms) {
        if (i1 > chunk_end) {
            if (!buffer) {
                const size_t tile_rows = i1 - i0;
                chunk_rows =
                        tile_rows *
                        std::max<size_t>(
                                1,
                                chunk_bytes / (tile_rows * d * sizeof(float)));
                chunk_rows = std::min(chunk_rows, nx - i0);
                buffer.reset(new float[chunk_rows * d]);
            }
            chunk_begin = i0;
            chunk_end = std::min(nx, i0 + chunk_rows);
            const int64_t n = static_cast<int64_t>(chunk_end - chunk_begin);
#pragma omp parallel for if (n * static_cast<int64_t>(d) > (1 << 16))
            for (int64_t i = 0; i < n; i++) {
                float* row = buffer.get() + i * d;
                fp16_to_fp32(d, x + (chunk_begin + i) * d, row);
                if (x_norms) {
                    x_norms[chunk_begin + i] = fvec_norm_L2sqr(row, d);
                }
            }
        }
        return buffer.get() + (i0 - chunk_begin) * d;
    }
};

/// BLAS-accelerated exhaustive L2 search for the k=1 (top-1) case.
/// Specializations live in the per-SIMD translation units under simd_impl/.
template <SIMDLevel>
void exhaustive_L2sqr_blas_cmax(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        Top1BlockResultHandler<CMax<float, int64_t>>& res,
        const float* y_norms);

/// Same as exhaustive_L2sqr_blas_cmax for packed IEEE binary16 queries.
template <SIMDLevel>
void exhaustive_L2sqr_blas_cmax_fp16(
        const uint16_t* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        Top1BlockResultHandler<CMax<float, int64_t>>& res,
        const float* y_norms);

} // namespace faiss
