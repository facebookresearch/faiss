/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/utils/distances_fused/simdlib_based.h>

#if defined(__AVX2__) || defined(__aarch64__)

#include <faiss/utils/simdlib.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace faiss {

namespace {

// It makes sense to like to overload certain cases because the further
// kernels are in need of registers. So, let's tell compiler
// not to waste registers on a bit faster code, if needed.
template <size_t DIM>
float l2_sqr(const float* const x) {
    // compiler should be smart enough to handle that
    float output = x[0] * x[0];
    for (size_t i = 1; i < DIM; i++) {
        output += x[i] * x[i];
    }

    return output;
}

template <size_t DIM>
float dot_product(
        const float* const __restrict x,
        const float* const __restrict y) {
    // compiler should be smart enough to handle that
    float output = x[0] * y[0];
    for (size_t i = 1; i < DIM; i++) {
        output += x[i] * y[i];
    }

    return output;
}

// The kernel for low dimensionality vectors.
// Finds the closest one from y for every given NX_POINTS_PER_LOOP points from x
//
// DIM is the dimensionality of the data
// NX_POINTS_PER_LOOP is the number of x points that get processed
//   simultaneously.
// NY_POINTS_PER_LOOP is the number of y points that get processed
//   simultaneously.
template <size_t DIM, size_t NX_POINTS_PER_LOOP, size_t NY_POINTS_PER_LOOP>
void kernel(
        const float* const __restrict x,
        const float* const __restrict y,
        const float* const __restrict y_transposed,
        const size_t ny,
        SingleBestResultHandler<CMax<float, int64_t>>& res,
        const float* __restrict y_norms,
        const size_t i) {
    const size_t ny_p =
            (ny / (8 * NY_POINTS_PER_LOOP)) * (8 * NY_POINTS_PER_LOOP);

    // compute
    const float* const __restrict xd_0 = x + i * DIM;

    // prefetch the next point
#if defined(__AVX2__)
    _mm_prefetch(xd_0 + DIM * sizeof(float), _MM_HINT_NTA);
#endif

    // load a single point from x
    // load -2 * value
    simd8float32 x_i[NX_POINTS_PER_LOOP][DIM];
    for (size_t nx_k = 0; nx_k < NX_POINTS_PER_LOOP; nx_k++) {
        for (size_t dd = 0; dd < DIM; dd++) {
            x_i[nx_k][dd] = simd8float32(-2 * *(xd_0 + nx_k * DIM + dd));
        }
    }

    // compute x_norm
    float x_norm_i[NX_POINTS_PER_LOOP];
    for (size_t nx_k = 0; nx_k < NX_POINTS_PER_LOOP; nx_k++) {
        x_norm_i[nx_k] = l2_sqr<DIM>(xd_0 + nx_k * DIM);
    }

    // distances and indices
    simd8float32 min_distances_i[NX_POINTS_PER_LOOP];
    for (size_t nx_k = 0; nx_k < NX_POINTS_PER_LOOP; nx_k++) {
        min_distances_i[nx_k] =
                simd8float32(res.dis_tab[i + nx_k] - x_norm_i[nx_k]);
    }

    simd8uint32 min_indices_i[NX_POINTS_PER_LOOP];
    for (size_t nx_k = 0; nx_k < NX_POINTS_PER_LOOP; nx_k++) {
        min_indices_i[nx_k] = simd8uint32((uint32_t)0);
    }

    //
    simd8uint32 current_indices = simd8uint32(0, 1, 2, 3, 4, 5, 6, 7);
    const simd8uint32 indices_delta = simd8uint32(8);

    // main loop
    size_t j = 0;
    for (; j < ny_p; j += NY_POINTS_PER_LOOP * 8) {
        // compute dot products for NX_POINTS from x and NY_POINTS from y
        // technically, we're multiplying -2x and y
        simd8float32 dp_i[NX_POINTS_PER_LOOP][NY_POINTS_PER_LOOP];

        // DIM 0 that uses MUL
        for (size_t ny_k = 0; ny_k < NY_POINTS_PER_LOOP; ny_k++) {
            simd8float32 y_i =
                    simd8float32(y_transposed + j + ny_k * 8 + ny * 0);
            for (size_t nx_k = 0; nx_k < NX_POINTS_PER_LOOP; nx_k++) {
                dp_i[nx_k][ny_k] = x_i[nx_k][0] * y_i;
            }
        }

        // other DIMs that use FMA
        for (size_t dd = 1; dd < DIM; dd++) {
            for (size_t ny_k = 0; ny_k < NY_POINTS_PER_LOOP; ny_k++) {
                simd8float32 y_i =
                        simd8float32(y_transposed + j + ny_k * 8 + ny * dd);

                for (size_t nx_k = 0; nx_k < NX_POINTS_PER_LOOP; nx_k++) {
                    dp_i[nx_k][ny_k] =
                            fmadd(x_i[nx_k][dd], y_i, dp_i[nx_k][ny_k]);
                }
            }
        }

        // compute y^2 + (-2x,y)
        for (size_t ny_k = 0; ny_k < NY_POINTS_PER_LOOP; ny_k++) {
            simd8float32 y_l2_sqr = simd8float32(y_norms + j + ny_k * 8);

            for (size_t nx_k = 0; nx_k < NX_POINTS_PER_LOOP; nx_k++) {
                dp_i[nx_k][ny_k] = dp_i[nx_k][ny_k] + y_l2_sqr;
            }
        }

        // do the comparisons and alter the min indices
        for (size_t ny_k = 0; ny_k < NY_POINTS_PER_LOOP; ny_k++) {
            for (size_t nx_k = 0; nx_k < NX_POINTS_PER_LOOP; nx_k++) {
                // cmpps
                cmplt_and_blend_inplace(
                        dp_i[nx_k][ny_k],
                        current_indices,
                        min_distances_i[nx_k],
                        min_indices_i[nx_k]);
            }

            current_indices = current_indices + indices_delta;
        }
    }

    // dump values and find the minimum distance / minimum index
    for (size_t nx_k = 0; nx_k < NX_POINTS_PER_LOOP; nx_k++) {
        float min_distances_scalar[8];
        uint32_t min_indices_scalar[8];

        min_distances_i[nx_k].storeu(min_distances_scalar);
        min_indices_i[nx_k].storeu(min_indices_scalar);

        float current_min_distance = res.dis_tab[i + nx_k];
        uint32_t current_min_index = res.ids_tab[i + nx_k];

        // This unusual comparison is needed to maintain the behavior
        // of the original implementation: if two indices are
        // represented with equal distance values, then
        // the index with the min value is returned.
        for (size_t jv = 0; jv < 8; jv++) {
            // add missing x_norms[i]
            float distance_candidate =
                    min_distances_scalar[jv] + x_norm_i[nx_k];

            // negative values can occur for identical vectors
            //    due to roundoff errors.
            if (distance_candidate < 0) {
                distance_candidate = 0;
            }

            const int64_t index_candidate = min_indices_scalar[jv];

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
        for (size_t j0 = j; j0 < ny; j0++) {
            const float dp =
                    dot_product<DIM>(x + (i + nx_k) * DIM, y + j0 * DIM);
            float dis = x_norm_i[nx_k] + y_norms[j0] - 2 * dp;
            // negative values can occur for identical vectors
            //    due to roundoff errors.
            if (dis < 0) {
                dis = 0;
            }

            if (current_min_distance > dis) {
                current_min_distance = dis;
                current_min_index = j0;
            }
        }

        // done
        res.add_result(i + nx_k, current_min_distance, current_min_index);
    }
}

template <size_t DIM, size_t NX_POINTS_PER_LOOP, size_t NY_POINTS_PER_LOOP>
void exhaustive_L2sqr_fused_cmax(
        const float* const __restrict x,
        const float* const __restrict y,
        size_t nx,
        size_t ny,
        SingleBestResultHandler<CMax<float, int64_t>>& res,
        const float* __restrict y_norms) {
    // BLAS does not like empty matrices
    if (nx == 0 || ny == 0) {
        return;
    }

    // compute norms for y
    std::unique_ptr<float[]> del2;
    if (!y_norms) {
        float* y_norms2 = new float[ny];
        del2.reset(y_norms2);

        for (size_t i = 0; i < ny; i++) {
            y_norms2[i] = l2_sqr<DIM>(y + i * DIM);
        }

        y_norms = y_norms2;
    }

    // initialize res
    res.begin_multiple(0, nx);

    // transpose y
    std::vector<float> y_transposed(DIM * ny);
    for (size_t j = 0; j < DIM; j++) {
        for (size_t i = 0; i < ny; i++) {
            y_transposed[j * ny + i] = y[j + i * DIM];
        }
    }

    const size_t nx_p = (nx / NX_POINTS_PER_LOOP) * NX_POINTS_PER_LOOP;
    // the main loop.
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < nx_p; i += NX_POINTS_PER_LOOP) {
        kernel<DIM, NX_POINTS_PER_LOOP, NY_POINTS_PER_LOOP>(
                x, y, y_transposed.data(), ny, res, y_norms, i);
    }

    for (size_t i = nx_p; i < nx; i++) {
        kernel<DIM, 1, NY_POINTS_PER_LOOP>(
                x, y, y_transposed.data(), ny, res, y_norms, i);
    }

    // Does nothing for SingleBestResultHandler, but
    // keeping the call for the consistency.
    res.end_multiple();
    InterruptCallback::check();
}

} // namespace

bool exhaustive_L2sqr_fused_cmax_simdlib(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        SingleBestResultHandler<CMax<float, int64_t>>& res,
        const float* y_norms) {
    // Process only cases with certain dimensionalities.
    // An acceptable dimensionality value is limited by the number of
    // available registers.

#define DISPATCH(DIM, NX_POINTS_PER_LOOP, NY_POINTS_PER_LOOP)    \
    case DIM: {                                                  \
        exhaustive_L2sqr_fused_cmax<                             \
                DIM,                                             \
                NX_POINTS_PER_LOOP,                              \
                NY_POINTS_PER_LOOP>(x, y, nx, ny, res, y_norms); \
        return true;                                             \
    }

    switch (d) {
        DISPATCH(2, 2, 2)
        DISPATCH(3, 2, 2)
        DISPATCH(4, 2, 1)
        DISPATCH(6, 1, 1)
        DISPATCH(8, 1, 1)
    }

    return false;
#undef DISPATCH
}

} // namespace faiss

#endif
