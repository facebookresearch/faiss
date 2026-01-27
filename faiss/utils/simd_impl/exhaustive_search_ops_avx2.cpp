/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <omp.h>

#include <faiss/utils/distances_fused/exhaustive_l2sqr_fused_cmax_256bit.h>
#include <faiss/utils/exhaustive_search_ops.h>
#include <faiss/utils/simd_impl/exhaustive_search_ops_avx2.h>

namespace faiss {

template <>
bool exhaustive_L2sqr_fused_cmax_simdlib<SIMDLevel::AVX2>(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        Top1BlockResultHandler<CMax<float, int64_t>>& res,
        const float* y_norms) {
    // Process only cases with certain dimensionalities.
    // An acceptable dimensionality value is limited by the number of
    // available registers.

    // faiss/benchs/bench_quantizer.py was used for benchmarking
    // and tuning 2nd and 3rd parameters values.
    // Basically, the larger the values for 2nd and 3rd parameters are,
    // the faster the execution is, but the more SIMD registers are needed.
    // This can be compensated with L1 cache, this is why this
    // code might operate with more registers than available
    // because of concurrent ports operations for ALU and LOAD/STORE.

    // It was possible to tweak these parameters on x64 machine.
    switch (d) {
        // Dimensions 1-3 with NX=6
        DISPATCH_L2SQR_FUSED_CMAX(
                exhaustive_L2sqr_fused_cmax_core, 1, 6, 1, SIMDLevel::AVX2)
        DISPATCH_L2SQR_FUSED_CMAX(
                exhaustive_L2sqr_fused_cmax_core, 2, 6, 1, SIMDLevel::AVX2)
        DISPATCH_L2SQR_FUSED_CMAX(
                exhaustive_L2sqr_fused_cmax_core, 3, 6, 1, SIMDLevel::AVX2)

        // Dimensions 4-12 with NX=8
        DISPATCH_L2SQR_FUSED_CMAX(
                exhaustive_L2sqr_fused_cmax_core, 4, 8, 1, SIMDLevel::AVX2)
        DISPATCH_L2SQR_FUSED_CMAX(
                exhaustive_L2sqr_fused_cmax_core, 5, 8, 1, SIMDLevel::AVX2)
        DISPATCH_L2SQR_FUSED_CMAX(
                exhaustive_L2sqr_fused_cmax_core, 6, 8, 1, SIMDLevel::AVX2)
        DISPATCH_L2SQR_FUSED_CMAX(
                exhaustive_L2sqr_fused_cmax_core, 7, 8, 1, SIMDLevel::AVX2)
        DISPATCH_L2SQR_FUSED_CMAX(
                exhaustive_L2sqr_fused_cmax_core, 8, 8, 1, SIMDLevel::AVX2)
        DISPATCH_L2SQR_FUSED_CMAX(
                exhaustive_L2sqr_fused_cmax_core, 9, 8, 1, SIMDLevel::AVX2)
        DISPATCH_L2SQR_FUSED_CMAX(
                exhaustive_L2sqr_fused_cmax_core, 10, 8, 1, SIMDLevel::AVX2)
        DISPATCH_L2SQR_FUSED_CMAX(
                exhaustive_L2sqr_fused_cmax_core, 11, 8, 1, SIMDLevel::AVX2)
        DISPATCH_L2SQR_FUSED_CMAX(
                exhaustive_L2sqr_fused_cmax_core, 12, 8, 1, SIMDLevel::AVX2)

        // Dimensions 13-16 with NX=6
        DISPATCH_L2SQR_FUSED_CMAX(
                exhaustive_L2sqr_fused_cmax_core, 13, 6, 1, SIMDLevel::AVX2)
        DISPATCH_L2SQR_FUSED_CMAX(
                exhaustive_L2sqr_fused_cmax_core, 14, 6, 1, SIMDLevel::AVX2)
        DISPATCH_L2SQR_FUSED_CMAX(
                exhaustive_L2sqr_fused_cmax_core, 15, 6, 1, SIMDLevel::AVX2)
        DISPATCH_L2SQR_FUSED_CMAX(
                exhaustive_L2sqr_fused_cmax_core, 16, 6, 1, SIMDLevel::AVX2)
    }

    return false;
}

template <>
void exhaustive_L2sqr_blas_simd<SIMDLevel::AVX2>(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        Top1BlockResultHandler<CMax<float, int64_t>>& res,
        const float* y_norms) {
    if (nx == 0 || ny == 0) {
        return;
    }

    if (exhaustive_L2sqr_fused_cmax_simdlib<SIMDLevel::AVX2>(
                x, y, d, nx, ny, res, y_norms)) {
        return;
    }

    exhaustive_L2sqr_blas_cmax_avx2(x, y, d, nx, ny, res, y_norms);
}

void exhaustive_L2sqr_blas_cmax_avx2(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        Top1BlockResultHandler<CMax<float, int64_t>>& res,
        const float* y_norms) {
    // BLAS does not like empty matrices
    if (nx == 0 || ny == 0)
        return;

    /* block sizes */
    const size_t bs_x = distance_compute_blas_query_bs;
    const size_t bs_y = distance_compute_blas_database_bs;
    // const size_t bs_x = 16, bs_y = 16;
    std::unique_ptr<float[]> ip_block(new float[bs_x * bs_y]);
    std::unique_ptr<float[]> x_norms(new float[nx]);
    std::unique_ptr<float[]> del2;

    fvec_norms_L2sqr(x_norms.get(), x, d, nx);

    if (!y_norms) {
        float* y_norms2 = new float[ny];
        del2.reset(y_norms2);
        fvec_norms_L2sqr(y_norms2, y, d, ny);
        y_norms = y_norms2;
    }

    for (size_t i0 = 0; i0 < nx; i0 += bs_x) {
        size_t i1 = i0 + bs_x;
        if (i1 > nx)
            i1 = nx;

        res.begin_multiple(i0, i1);

        for (size_t j0 = 0; j0 < ny; j0 += bs_y) {
            size_t j1 = j0 + bs_y;
            if (j1 > ny)
                j1 = ny;
            /* compute the actual dot products */
            {
                float one = 1, zero = 0;
                FINTEGER nyi = j1 - j0, nxi = i1 - i0, di = d;
                sgemm_("Transpose",
                       "Not transpose",
                       &nyi,
                       &nxi,
                       &di,
                       &one,
                       y + j0 * d,
                       &di,
                       x + i0 * d,
                       &di,
                       &zero,
                       ip_block.get(),
                       &nyi);
            }
#pragma omp parallel for
            for (int64_t i = i0; i < i1; i++) {
                float* ip_line = ip_block.get() + (i - i0) * (j1 - j0);

                _mm_prefetch((const char*)ip_line, _MM_HINT_NTA);
                _mm_prefetch((const char*)(ip_line + 16), _MM_HINT_NTA);

                // constant
                const __m256 mul_minus2 = _mm256_set1_ps(-2);

                // Track 8 min distances + 8 min indices.
                // All the distances tracked do not take x_norms[i]
                //   into account in order to get rid of extra
                //   _mm256_add_ps(x_norms[i], ...) instructions
                //   is distance computations.
                __m256 min_distances =
                        _mm256_set1_ps(res.dis_tab[i] - x_norms[i]);

                // these indices are local and are relative to j0.
                // so, value 0 means j0.
                __m256i min_indices = _mm256_set1_epi32(0);

                __m256i current_indices =
                        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                const __m256i indices_delta = _mm256_set1_epi32(8);

                // current j index
                size_t idx_j = 0;
                size_t count = j1 - j0;

                // process 16 elements per loop
                for (; idx_j < (count / 16) * 16; idx_j += 16, ip_line += 16) {
                    _mm_prefetch((const char*)(ip_line + 32), _MM_HINT_NTA);
                    _mm_prefetch((const char*)(ip_line + 48), _MM_HINT_NTA);

                    // load values for norms
                    const __m256 y_norm_0 =
                            _mm256_loadu_ps(y_norms + idx_j + j0 + 0);
                    const __m256 y_norm_1 =
                            _mm256_loadu_ps(y_norms + idx_j + j0 + 8);

                    // load values for dot products
                    const __m256 ip_0 = _mm256_loadu_ps(ip_line + 0);
                    const __m256 ip_1 = _mm256_loadu_ps(ip_line + 8);

                    // compute dis = y_norm[j] - 2 * dot(x_norm[i], y_norm[j]).
                    // x_norm[i] was dropped off because it is a constant for a
                    // given i. We'll deal with it later.
                    __m256 distances_0 =
                            _mm256_fmadd_ps(ip_0, mul_minus2, y_norm_0);
                    __m256 distances_1 =
                            _mm256_fmadd_ps(ip_1, mul_minus2, y_norm_1);

                    // compare the new distances to the min distances
                    // for each of the first group of 8 AVX2 components.
                    const __m256 comparison_0 = _mm256_cmp_ps(
                            min_distances, distances_0, _CMP_LE_OS);

                    // update min distances and indices with closest vectors if
                    // needed.
                    min_distances = _mm256_blendv_ps(
                            distances_0, min_distances, comparison_0);
                    min_indices = _mm256_castps_si256(_mm256_blendv_ps(
                            _mm256_castsi256_ps(current_indices),
                            _mm256_castsi256_ps(min_indices),
                            comparison_0));
                    current_indices =
                            _mm256_add_epi32(current_indices, indices_delta);

                    // compare the new distances to the min distances
                    // for each of the second group of 8 AVX2 components.
                    const __m256 comparison_1 = _mm256_cmp_ps(
                            min_distances, distances_1, _CMP_LE_OS);

                    // update min distances and indices with closest vectors if
                    // needed.
                    min_distances = _mm256_blendv_ps(
                            distances_1, min_distances, comparison_1);
                    min_indices = _mm256_castps_si256(_mm256_blendv_ps(
                            _mm256_castsi256_ps(current_indices),
                            _mm256_castsi256_ps(min_indices),
                            comparison_1));
                    current_indices =
                            _mm256_add_epi32(current_indices, indices_delta);
                }

                // dump values and find the minimum distance / minimum index
                float min_distances_scalar[8];
                uint32_t min_indices_scalar[8];
                _mm256_storeu_ps(min_distances_scalar, min_distances);
                _mm256_storeu_si256(
                        (__m256i*)(min_indices_scalar), min_indices);

                float current_min_distance = res.dis_tab[i];
                uint32_t current_min_index = res.ids_tab[i];

                // This unusual comparison is needed to maintain the behavior
                // of the original implementation: if two indices are
                // represented with equal distance values, then
                // the index with the min value is returned.
                for (size_t jv = 0; jv < 8; jv++) {
                    // add missing x_norms[i]
                    float distance_candidate =
                            min_distances_scalar[jv] + x_norms[i];

                    // negative values can occur for identical vectors
                    //    due to roundoff errors.
                    if (distance_candidate < 0)
                        distance_candidate = 0;

                    int64_t index_candidate = min_indices_scalar[jv] + j0;

                    if (current_min_distance > distance_candidate) {
                        current_min_distance = distance_candidate;
                        current_min_index = index_candidate;
                    } else if (
                            current_min_distance == distance_candidate &&
                            current_min_index > index_candidate) {
                        current_min_index = index_candidate;
                    }
                }

                // process leftovers
                for (; idx_j < count; idx_j++, ip_line++) {
                    float ip = *ip_line;
                    float dis = x_norms[i] + y_norms[idx_j + j0] - 2 * ip;
                    // negative values can occur for identical vectors
                    //    due to roundoff errors.
                    if (dis < 0)
                        dis = 0;

                    if (current_min_distance > dis) {
                        current_min_distance = dis;
                        current_min_index = idx_j + j0;
                    }
                }

                //
                res.add_result(i, current_min_distance, current_min_index);
            }
        }
        // Does nothing for SingleBestResultHandler, but
        // keeping the call for the consistency.
        res.end_multiple();
        InterruptCallback::check();
    }
}

} // namespace faiss
