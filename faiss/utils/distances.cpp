/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/utils/distances.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>

#include <omp.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/ResultHandler.h>

#include <faiss/utils/distances_fused/distances_fused.h>

#ifndef FINTEGER
#define FINTEGER long
#endif

extern "C" {

/* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */

int sgemm_(
        const char* transa,
        const char* transb,
        FINTEGER* m,
        FINTEGER* n,
        FINTEGER* k,
        const float* alpha,
        const float* a,
        FINTEGER* lda,
        const float* b,
        FINTEGER* ldb,
        float* beta,
        float* c,
        FINTEGER* ldc);
}

namespace faiss {

/***************************************************************************
 * Matrix/vector ops
 ***************************************************************************/

/* Compute the L2 norm of a set of nx vectors */
void fvec_norms_L2(
        float* __restrict nr,
        const float* __restrict x,
        size_t d,
        size_t nx) {
#pragma omp parallel for
    for (int64_t i = 0; i < nx; i++) {
        nr[i] = sqrtf(fvec_norm_L2sqr(x + i * d, d));
    }
}

void fvec_norms_L2sqr(
        float* __restrict nr,
        const float* __restrict x,
        size_t d,
        size_t nx) {
#pragma omp parallel for
    for (int64_t i = 0; i < nx; i++)
        nr[i] = fvec_norm_L2sqr(x + i * d, d);
}

void fvec_renorm_L2(size_t d, size_t nx, float* __restrict x) {
#pragma omp parallel for
    for (int64_t i = 0; i < nx; i++) {
        float* __restrict xi = x + i * d;

        float nr = fvec_norm_L2sqr(xi, d);

        if (nr > 0) {
            size_t j;
            const float inv_nr = 1.0 / sqrtf(nr);
            for (j = 0; j < d; j++)
                xi[j] *= inv_nr;
        }
    }
}

/***************************************************************************
 * KNN functions
 ***************************************************************************/

namespace {

/* Find the nearest neighbors for nx queries in a set of ny vectors */
template <class ResultHandler, bool use_sel = false>
void exhaustive_inner_product_seq(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        ResultHandler& res,
        const IDSelector* sel = nullptr) {
    using SingleResultHandler = typename ResultHandler::SingleResultHandler;
    int nt = std::min(int(nx), omp_get_max_threads());

    FAISS_ASSERT(use_sel == (sel != nullptr));

#pragma omp parallel num_threads(nt)
    {
        SingleResultHandler resi(res);
#pragma omp for
        for (int64_t i = 0; i < nx; i++) {
            const float* x_i = x + i * d;
            const float* y_j = y;

            resi.begin(i);

            for (size_t j = 0; j < ny; j++, y_j += d) {
                if (use_sel && !sel->is_member(j)) {
                    continue;
                }
                float ip = fvec_inner_product(x_i, y_j, d);
                resi.add_result(ip, j);
            }
            resi.end();
        }
    }
}

template <class ResultHandler, bool use_sel = false>
void exhaustive_L2sqr_seq(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        ResultHandler& res,
        const IDSelector* sel = nullptr) {
    using SingleResultHandler = typename ResultHandler::SingleResultHandler;
    int nt = std::min(int(nx), omp_get_max_threads());

    FAISS_ASSERT(use_sel == (sel != nullptr));

#pragma omp parallel num_threads(nt)
    {
        SingleResultHandler resi(res);
#pragma omp for
        for (int64_t i = 0; i < nx; i++) {
            const float* x_i = x + i * d;
            const float* y_j = y;
            resi.begin(i);
            for (size_t j = 0; j < ny; j++, y_j += d) {
                if (use_sel && !sel->is_member(j)) {
                    continue;
                }
                float disij = fvec_L2sqr(x_i, y_j, d);
                resi.add_result(disij, j);
            }
            resi.end();
        }
    }
}

/** Find the nearest neighbors for nx queries in a set of ny vectors */
template <class ResultHandler>
void exhaustive_inner_product_blas(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        ResultHandler& res) {
    // BLAS does not like empty matrices
    if (nx == 0 || ny == 0)
        return;

    /* block sizes */
    const size_t bs_x = distance_compute_blas_query_bs;
    const size_t bs_y = distance_compute_blas_database_bs;
    std::unique_ptr<float[]> ip_block(new float[bs_x * bs_y]);

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

            res.add_results(j0, j1, ip_block.get());
        }
        res.end_multiple();
        InterruptCallback::check();
    }
}

// distance correction is an operator that can be applied to transform
// the distances
template <class ResultHandler>
void exhaustive_L2sqr_blas_default_impl(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        ResultHandler& res,
        const float* y_norms = nullptr) {
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

                for (size_t j = j0; j < j1; j++) {
                    float ip = *ip_line;
                    float dis = x_norms[i] + y_norms[j] - 2 * ip;

                    // negative values can occur for identical vectors
                    // due to roundoff errors
                    if (dis < 0)
                        dis = 0;

                    *ip_line = dis;
                    ip_line++;
                }
            }
            res.add_results(j0, j1, ip_block.get());
        }
        res.end_multiple();
        InterruptCallback::check();
    }
}

template <class ResultHandler>
void exhaustive_L2sqr_blas(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        ResultHandler& res,
        const float* y_norms = nullptr) {
    exhaustive_L2sqr_blas_default_impl(x, y, d, nx, ny, res);
}

#ifdef __AVX2__
void exhaustive_L2sqr_blas_cmax_avx2(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        SingleBestResultHandler<CMax<float, int64_t>>& res,
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

                _mm_prefetch(ip_line, _MM_HINT_NTA);
                _mm_prefetch(ip_line + 16, _MM_HINT_NTA);

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
                    _mm_prefetch(ip_line + 32, _MM_HINT_NTA);
                    _mm_prefetch(ip_line + 48, _MM_HINT_NTA);

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
#endif

// an override if only a single closest point is needed
template <>
void exhaustive_L2sqr_blas<SingleBestResultHandler<CMax<float, int64_t>>>(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        SingleBestResultHandler<CMax<float, int64_t>>& res,
        const float* y_norms) {
#if defined(__AVX2__)
    // use a faster fused kernel if available
    if (exhaustive_L2sqr_fused_cmax(x, y, d, nx, ny, res, y_norms)) {
        // the kernel is available and it is complete, we're done.
        return;
    }

    // run the specialized AVX2 implementation
    exhaustive_L2sqr_blas_cmax_avx2(x, y, d, nx, ny, res, y_norms);

#elif defined(__aarch64__)
    // use a faster fused kernel if available
    if (exhaustive_L2sqr_fused_cmax(x, y, d, nx, ny, res, y_norms)) {
        // the kernel is available and it is complete, we're done.
        return;
    }

    // run the default implementation
    exhaustive_L2sqr_blas_default_impl<
            SingleBestResultHandler<CMax<float, int64_t>>>(
            x, y, d, nx, ny, res, y_norms);
#else
    // run the default implementation
    exhaustive_L2sqr_blas_default_impl<
            SingleBestResultHandler<CMax<float, int64_t>>>(
            x, y, d, nx, ny, res, y_norms);
#endif
}

template <class ResultHandler>
void knn_L2sqr_select(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        ResultHandler& res,
        const float* y_norm2,
        const IDSelector* sel) {
    if (sel) {
        exhaustive_L2sqr_seq<ResultHandler, true>(x, y, d, nx, ny, res, sel);
    } else if (nx < distance_compute_blas_threshold) {
        exhaustive_L2sqr_seq(x, y, d, nx, ny, res);
    } else {
        exhaustive_L2sqr_blas(x, y, d, nx, ny, res, y_norm2);
    }
}

} // anonymous namespace

/*******************************************************
 * KNN driver functions
 *******************************************************/

int distance_compute_blas_threshold = 20;
int distance_compute_blas_query_bs = 4096;
int distance_compute_blas_database_bs = 1024;
int distance_compute_min_k_reservoir = 100;

void knn_inner_product(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        size_t k,
        float* val,
        int64_t* ids,
        const IDSelector* sel) {
    int64_t imin = 0;
    if (auto selr = dynamic_cast<const IDSelectorRange*>(sel)) {
        imin = std::max(selr->imin, int64_t(0));
        int64_t imax = std::min(selr->imax, int64_t(ny));
        ny = imax - imin;
        y += d * imin;
        sel = nullptr;
    }
    if (auto sela = dynamic_cast<const IDSelectorArray*>(sel)) {
        knn_inner_products_by_idx(
                x, y, sela->ids, d, nx, sela->n, k, val, ids, 0);
        return;
    }
    if (k < distance_compute_min_k_reservoir) {
        using RH = HeapResultHandler<CMin<float, int64_t>>;
        RH res(nx, val, ids, k);
        if (sel) {
            exhaustive_inner_product_seq<RH, true>(x, y, d, nx, ny, res, sel);
        } else if (nx < distance_compute_blas_threshold) {
            exhaustive_inner_product_seq(x, y, d, nx, ny, res);
        } else {
            exhaustive_inner_product_blas(x, y, d, nx, ny, res);
        }
    } else {
        using RH = ReservoirResultHandler<CMin<float, int64_t>>;
        RH res(nx, val, ids, k);
        if (sel) {
            exhaustive_inner_product_seq<RH, true>(x, y, d, nx, ny, res, sel);
        } else if (nx < distance_compute_blas_threshold) {
            exhaustive_inner_product_seq(x, y, d, nx, ny, res, nullptr);
        } else {
            exhaustive_inner_product_blas(x, y, d, nx, ny, res);
        }
    }
    if (imin != 0) {
        for (size_t i = 0; i < nx * k; i++) {
            if (ids[i] >= 0) {
                ids[i] += imin;
            }
        }
    }
}

void knn_inner_product(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        float_minheap_array_t* res,
        const IDSelector* sel) {
    FAISS_THROW_IF_NOT(nx == res->nh);
    knn_inner_product(x, y, d, nx, ny, res->k, res->val, res->ids, sel);
}

void knn_L2sqr(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        size_t k,
        float* vals,
        int64_t* ids,
        const float* y_norm2,
        const IDSelector* sel) {
    int64_t imin = 0;
    if (auto selr = dynamic_cast<const IDSelectorRange*>(sel)) {
        imin = std::max(selr->imin, int64_t(0));
        int64_t imax = std::min(selr->imax, int64_t(ny));
        ny = imax - imin;
        y += d * imin;
        sel = nullptr;
    }
    if (auto sela = dynamic_cast<const IDSelectorArray*>(sel)) {
        knn_L2sqr_by_idx(x, y, sela->ids, d, nx, sela->n, k, vals, ids, 0);
        return;
    }
    if (k == 1) {
        SingleBestResultHandler<CMax<float, int64_t>> res(nx, vals, ids);
        knn_L2sqr_select(x, y, d, nx, ny, res, y_norm2, sel);
    } else if (k < distance_compute_min_k_reservoir) {
        HeapResultHandler<CMax<float, int64_t>> res(nx, vals, ids, k);
        knn_L2sqr_select(x, y, d, nx, ny, res, y_norm2, sel);
    } else {
        ReservoirResultHandler<CMax<float, int64_t>> res(nx, vals, ids, k);
        knn_L2sqr_select(x, y, d, nx, ny, res, y_norm2, sel);
    }
    if (imin != 0) {
        for (size_t i = 0; i < nx * k; i++) {
            if (ids[i] >= 0) {
                ids[i] += imin;
            }
        }
    }
}

void knn_L2sqr(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        float_maxheap_array_t* res,
        const float* y_norm2,
        const IDSelector* sel) {
    FAISS_THROW_IF_NOT(res->nh == nx);
    knn_L2sqr(x, y, d, nx, ny, res->k, res->val, res->ids, y_norm2, sel);
}

/***************************************************************************
 * Range search
 ***************************************************************************/

void range_search_L2sqr(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        float radius,
        RangeSearchResult* res,
        const IDSelector* sel) {
    using RH = RangeSearchResultHandler<CMax<float, int64_t>>;
    RH resh(res, radius);
    if (sel) {
        exhaustive_L2sqr_seq<RH, true>(x, y, d, nx, ny, resh, sel);
    } else if (nx < distance_compute_blas_threshold) {
        exhaustive_L2sqr_seq(x, y, d, nx, ny, resh, sel);
    } else {
        exhaustive_L2sqr_blas(x, y, d, nx, ny, resh);
    }
}

void range_search_inner_product(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        float radius,
        RangeSearchResult* res,
        const IDSelector* sel) {
    using RH = RangeSearchResultHandler<CMin<float, int64_t>>;
    RH resh(res, radius);
    if (sel) {
        exhaustive_inner_product_seq<RH, true>(x, y, d, nx, ny, resh, sel);
    } else if (nx < distance_compute_blas_threshold) {
        exhaustive_inner_product_seq(x, y, d, nx, ny, resh);
    } else {
        exhaustive_inner_product_blas(x, y, d, nx, ny, resh);
    }
}

/***************************************************************************
 * compute a subset of  distances
 ***************************************************************************/

/* compute the inner product between x and a subset y of ny vectors,
   whose indices are given by idy.  */
void fvec_inner_products_by_idx(
        float* __restrict ip,
        const float* x,
        const float* y,
        const int64_t* __restrict ids, /* for y vecs */
        size_t d,
        size_t nx,
        size_t ny) {
#pragma omp parallel for
    for (int64_t j = 0; j < nx; j++) {
        const int64_t* __restrict idsj = ids + j * ny;
        const float* xj = x + j * d;
        float* __restrict ipj = ip + j * ny;
        for (size_t i = 0; i < ny; i++) {
            if (idsj[i] < 0)
                continue;
            ipj[i] = fvec_inner_product(xj, y + d * idsj[i], d);
        }
    }
}

/* compute the inner product between x and a subset y of ny vectors,
   whose indices are given by idy.  */
void fvec_L2sqr_by_idx(
        float* __restrict dis,
        const float* x,
        const float* y,
        const int64_t* __restrict ids, /* ids of y vecs */
        size_t d,
        size_t nx,
        size_t ny) {
#pragma omp parallel for
    for (int64_t j = 0; j < nx; j++) {
        const int64_t* __restrict idsj = ids + j * ny;
        const float* xj = x + j * d;
        float* __restrict disj = dis + j * ny;
        for (size_t i = 0; i < ny; i++) {
            if (idsj[i] < 0)
                continue;
            disj[i] = fvec_L2sqr(xj, y + d * idsj[i], d);
        }
    }
}

void pairwise_indexed_L2sqr(
        size_t d,
        size_t n,
        const float* x,
        const int64_t* ix,
        const float* y,
        const int64_t* iy,
        float* dis) {
#pragma omp parallel for
    for (int64_t j = 0; j < n; j++) {
        if (ix[j] >= 0 && iy[j] >= 0) {
            dis[j] = fvec_L2sqr(x + d * ix[j], y + d * iy[j], d);
        }
    }
}

void pairwise_indexed_inner_product(
        size_t d,
        size_t n,
        const float* x,
        const int64_t* ix,
        const float* y,
        const int64_t* iy,
        float* dis) {
#pragma omp parallel for
    for (int64_t j = 0; j < n; j++) {
        if (ix[j] >= 0 && iy[j] >= 0) {
            dis[j] = fvec_inner_product(x + d * ix[j], y + d * iy[j], d);
        }
    }
}

/* Find the nearest neighbors for nx queries in a set of ny vectors
   indexed by ids. May be useful for re-ranking a pre-selected vector list */
void knn_inner_products_by_idx(
        const float* x,
        const float* y,
        const int64_t* ids,
        size_t d,
        size_t nx,
        size_t ny,
        size_t k,
        float* res_vals,
        int64_t* res_ids,
        int64_t ld_ids) {
    if (ld_ids < 0) {
        ld_ids = ny;
    }

#pragma omp parallel for if (nx > 100)
    for (int64_t i = 0; i < nx; i++) {
        const float* x_ = x + i * d;
        const int64_t* idsi = ids + i * ld_ids;
        size_t j;
        float* __restrict simi = res_vals + i * k;
        int64_t* __restrict idxi = res_ids + i * k;
        minheap_heapify(k, simi, idxi);

        for (j = 0; j < ny; j++) {
            if (idsi[j] < 0)
                break;
            float ip = fvec_inner_product(x_, y + d * idsi[j], d);

            if (ip > simi[0]) {
                minheap_replace_top(k, simi, idxi, ip, idsi[j]);
            }
        }
        minheap_reorder(k, simi, idxi);
    }
}

void knn_L2sqr_by_idx(
        const float* x,
        const float* y,
        const int64_t* __restrict ids,
        size_t d,
        size_t nx,
        size_t ny,
        size_t k,
        float* res_vals,
        int64_t* res_ids,
        int64_t ld_ids) {
    if (ld_ids < 0) {
        ld_ids = ny;
    }
#pragma omp parallel for if (nx > 100)
    for (int64_t i = 0; i < nx; i++) {
        const float* x_ = x + i * d;
        const int64_t* __restrict idsi = ids + i * ld_ids;
        float* __restrict simi = res_vals + i * k;
        int64_t* __restrict idxi = res_ids + i * k;
        maxheap_heapify(k, simi, idxi);
        for (size_t j = 0; j < ny; j++) {
            float disij = fvec_L2sqr(x_, y + d * idsi[j], d);

            if (disij < simi[0]) {
                maxheap_replace_top(k, simi, idxi, disij, idsi[j]);
            }
        }
        maxheap_reorder(k, simi, idxi);
    }
}

void pairwise_L2sqr(
        int64_t d,
        int64_t nq,
        const float* xq,
        int64_t nb,
        const float* xb,
        float* dis,
        int64_t ldq,
        int64_t ldb,
        int64_t ldd) {
    if (nq == 0 || nb == 0)
        return;
    if (ldq == -1)
        ldq = d;
    if (ldb == -1)
        ldb = d;
    if (ldd == -1)
        ldd = nb;

    // store in beginning of distance matrix to avoid malloc
    float* b_norms = dis;

#pragma omp parallel for
    for (int64_t i = 0; i < nb; i++)
        b_norms[i] = fvec_norm_L2sqr(xb + i * ldb, d);

#pragma omp parallel for
    for (int64_t i = 1; i < nq; i++) {
        float q_norm = fvec_norm_L2sqr(xq + i * ldq, d);
        for (int64_t j = 0; j < nb; j++)
            dis[i * ldd + j] = q_norm + b_norms[j];
    }

    {
        float q_norm = fvec_norm_L2sqr(xq, d);
        for (int64_t j = 0; j < nb; j++)
            dis[j] += q_norm;
    }

    {
        FINTEGER nbi = nb, nqi = nq, di = d, ldqi = ldq, ldbi = ldb, lddi = ldd;
        float one = 1.0, minus_2 = -2.0;

        sgemm_("Transposed",
               "Not transposed",
               &nbi,
               &nqi,
               &di,
               &minus_2,
               xb,
               &ldbi,
               xq,
               &ldqi,
               &one,
               dis,
               &lddi);
    }
}

void inner_product_to_L2sqr(
        float* __restrict dis,
        const float* nr1,
        const float* nr2,
        size_t n1,
        size_t n2) {
#pragma omp parallel for
    for (int64_t j = 0; j < n1; j++) {
        float* disj = dis + j * n2;
        for (size_t i = 0; i < n2; i++)
            disj[i] = nr1[j] + nr2[i] - 2 * disj[i];
    }
}

} // namespace faiss
