/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/distances.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <vector>

#include <omp.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/ResultHandler.h>

#include <faiss/impl/simd_dispatch.h>
#include <faiss/utils/distances_dispatch.h>
#include <faiss/utils/distances_fused/distances_fused.h>
#include <faiss/utils/simd_impl/exhaustive_L2sqr_blas_cmax.h>

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
 * Public API dispatch wrappers
 ***************************************************************************/

float fvec_L1(const float* x, const float* y, size_t d) {
    return fvec_L1_dispatch(x, y, d);
}

float fvec_Linf(const float* x, const float* y, size_t d) {
    return fvec_Linf_dispatch(x, y, d);
}

float fvec_norm_L2sqr(const float* x, size_t d) {
    return fvec_norm_L2sqr_dispatch(x, d);
}

float fvec_L2sqr(const float* x, const float* y, size_t d) {
    return fvec_L2sqr_dispatch(x, y, d);
}

float fvec_inner_product(const float* x, const float* y, size_t d) {
    return fvec_inner_product_dispatch(x, y, d);
}

void fvec_inner_product_batch_4(
        const float* x,
        const float* y0,
        const float* y1,
        const float* y2,
        const float* y3,
        const size_t d,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3) {
    fvec_inner_product_batch_4_dispatch(
            x, y0, y1, y2, y3, d, dis0, dis1, dis2, dis3);
}

void fvec_L2sqr_batch_4(
        const float* x,
        const float* y0,
        const float* y1,
        const float* y2,
        const float* y3,
        const size_t d,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3) {
    fvec_L2sqr_batch_4_dispatch(x, y0, y1, y2, y3, d, dis0, dis1, dis2, dis3);
}

void fvec_L2sqr_ny_transposed(
        float* dis,
        const float* x,
        const float* y,
        const float* y_sqlen,
        size_t d,
        size_t d_offset,
        size_t ny) {
    fvec_L2sqr_ny_transposed_dispatch(dis, x, y, y_sqlen, d, d_offset, ny);
}

void fvec_inner_products_ny(
        float* ip,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    fvec_inner_products_ny_dispatch(ip, x, y, d, ny);
}

void fvec_L2sqr_ny(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    fvec_L2sqr_ny_dispatch(dis, x, y, d, ny);
}

size_t fvec_L2sqr_ny_nearest(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    return fvec_L2sqr_ny_nearest_dispatch(distances_tmp_buffer, x, y, d, ny);
}

size_t fvec_L2sqr_ny_nearest_y_transposed(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        const float* y_sqlen,
        size_t d,
        size_t d_offset,
        size_t ny) {
    return fvec_L2sqr_ny_nearest_y_transposed_dispatch(
            distances_tmp_buffer, x, y, y_sqlen, d, d_offset, ny);
}

void fvec_madd(size_t n, const float* a, float bf, const float* b, float* c) {
    fvec_madd_dispatch(n, a, bf, b, c);
}

int fvec_madd_and_argmin(
        size_t n,
        const float* a,
        float bf,
        const float* b,
        float* c) {
    return fvec_madd_and_argmin_dispatch(n, a, bf, b, c);
}

void fvec_sub(size_t d, const float* a, const float* b, float* c) {
    fvec_sub_dispatch(d, a, b, c);
}

void fvec_add(size_t d, const float* a, const float* b, float* c) {
    fvec_add_dispatch(d, a, b, c);
}

void fvec_add(size_t d, const float* a, float b, float* c) {
    fvec_add_scalar_dispatch(d, a, b, c);
}

void compute_PQ_dis_tables_dsub2(
        size_t d,
        size_t ksub,
        const float* all_centroids,
        size_t nx,
        const float* x,
        bool is_inner_product,
        float* dis_tables) {
    compute_PQ_dis_tables_dsub2_dispatch(
            d, ksub, all_centroids, nx, x, is_inner_product, dis_tables);
}

/***************************************************************************
 * Matrix/vector ops
 ***************************************************************************/

/* Compute the L2 norm of a set of nx vectors */
void fvec_norms_L2(
        float* __restrict nr,
        const float* __restrict x,
        size_t d,
        size_t nx) {
    with_simd_level([&]<SIMDLevel SL>() {
#pragma omp parallel for if (nx > 10000)
        for (int64_t i = 0; i < static_cast<int64_t>(nx); i++) {
            nr[i] = sqrtf(fvec_norm_L2sqr<SL>(x + i * d, d));
        }
    });
}

void fvec_norms_L2sqr(
        float* __restrict nr,
        const float* __restrict x,
        size_t d,
        size_t nx) {
    with_simd_level([&]<SIMDLevel SL>() {
#pragma omp parallel for if (nx > 10000)
        for (int64_t i = 0; i < static_cast<int64_t>(nx); i++) {
            nr[i] = fvec_norm_L2sqr<SL>(x + i * d, d);
        }
    });
}

// The following is a workaround to a problem
// in OpenMP in fbcode. The crash occurs
// inside OMP when IndexIVFSpectralHash::set_query()
// calls fvec_renorm_L2. set_query() is always
// calling this function with nx == 1, so even
// the omp version should run single threaded,
// as per the if condition of the omp pragma.
// Instead, the omp version crashes inside OMP.
// The workaround below is explicitly branching
// off to a codepath without omp.

void fvec_renorm_L2_noomp(size_t d, size_t nx, float* __restrict x) {
    with_simd_level([&]<SIMDLevel SL>() {
        for (int64_t i = 0; i < static_cast<int64_t>(nx); i++) {
            float* __restrict xi = x + i * d;
            float nr = fvec_norm_L2sqr<SL>(xi, d);
            if (nr > 0) {
                const float inv_nr = 1.0 / sqrtf(nr);
                for (size_t j = 0; j < d; j++) {
                    xi[j] *= inv_nr;
                }
            }
        }
    });
}

void fvec_renorm_L2_omp(size_t d, size_t nx, float* __restrict x) {
    with_simd_level([&]<SIMDLevel SL>() {
#pragma omp parallel for if (nx > 10000)
        for (int64_t i = 0; i < static_cast<int64_t>(nx); i++) {
            float* __restrict xi = x + i * d;
            float nr = fvec_norm_L2sqr<SL>(xi, d);
            if (nr > 0) {
                const float inv_nr = 1.0 / sqrtf(nr);
                for (size_t j = 0; j < d; j++) {
                    xi[j] *= inv_nr;
                }
            }
        }
    });
}

void fvec_renorm_L2(size_t d, size_t nx, float* __restrict x) {
    if (nx <= 10000) {
        fvec_renorm_L2_noomp(d, nx, x);
    } else {
        fvec_renorm_L2_omp(d, nx, x);
    }
}

/***************************************************************************
 * KNN functions
 ***************************************************************************/

namespace {

/* Find the nearest neighbors for nx queries in a set of ny vectors */
template <class BlockResultHandler>
void exhaustive_inner_product_seq(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        BlockResultHandler& res) {
    using SingleResultHandler =
            typename BlockResultHandler::SingleResultHandler;
    [[maybe_unused]] int nt = std::min(int(nx), omp_get_max_threads());

#pragma omp parallel num_threads(nt)
    {
        SingleResultHandler resi(res);
        with_simd_level([&]<SIMDLevel SL>() {
#pragma omp for
            for (int64_t i = 0; i < static_cast<int64_t>(nx); i++) {
                const float* x_i = x + i * d;
                const float* y_j = y;

                resi.begin(i);

                for (size_t j = 0; j < ny; j++, y_j += d) {
                    if (!res.is_in_selection(j)) {
                        continue;
                    }
                    float ip = fvec_inner_product<SL>(x_i, y_j, d);
                    resi.add_result(ip, j);
                }
                resi.end();
            }
        });
    }
}

template <class BlockResultHandler>
void exhaustive_L2sqr_seq(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        BlockResultHandler& res) {
    using SingleResultHandler =
            typename BlockResultHandler::SingleResultHandler;
    [[maybe_unused]] int nt = std::min(int(nx), omp_get_max_threads());

#pragma omp parallel num_threads(nt)
    {
        SingleResultHandler resi(res);
        with_simd_level([&]<SIMDLevel SL>() {
#pragma omp for
            for (int64_t i = 0; i < static_cast<int64_t>(nx); i++) {
                const float* x_i = x + i * d;
                const float* y_j = y;
                resi.begin(i);
                for (size_t j = 0; j < ny; j++, y_j += d) {
                    if (!res.is_in_selection(j)) {
                        continue;
                    }
                    float disij = fvec_L2sqr<SL>(x_i, y_j, d);
                    resi.add_result(disij, j);
                }
                resi.end();
            }
        });
    }
}

/** Find the nearest neighbors for nx queries in a set of ny vectors */
template <class BlockResultHandler>
void exhaustive_inner_product_blas(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        BlockResultHandler& res) {
    // BLAS does not like empty matrices
    if (nx == 0 || ny == 0) {
        return;
    }

    /* block sizes */
    const size_t bs_x = distance_compute_blas_query_bs;
    const size_t bs_y = distance_compute_blas_database_bs;
    std::unique_ptr<float[]> ip_block(new float[bs_x * bs_y]);

    for (size_t i0 = 0; i0 < nx; i0 += bs_x) {
        size_t i1 = i0 + bs_x;
        if (i1 > nx) {
            i1 = nx;
        }

        res.begin_multiple(i0, i1);

        for (size_t j0 = 0; j0 < ny; j0 += bs_y) {
            size_t j1 = j0 + bs_y;
            if (j1 > ny) {
                j1 = ny;
            }
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
template <class BlockResultHandler>
void exhaustive_L2sqr_blas_default_impl(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        BlockResultHandler& res,
        const float* y_norms = nullptr) {
    // BLAS does not like empty matrices
    if (nx == 0 || ny == 0) {
        return;
    }

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
        if (i1 > nx) {
            i1 = nx;
        }

        res.begin_multiple(i0, i1);

        for (size_t j0 = 0; j0 < ny; j0 += bs_y) {
            size_t j1 = j0 + bs_y;
            if (j1 > ny) {
                j1 = ny;
            }
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
            for (size_t i = i0; i < i1; i++) {
                float* ip_line = ip_block.get() + (i - i0) * (j1 - j0);

                for (size_t j = j0; j < j1; j++) {
                    float ip = *ip_line;
                    float dis = x_norms[i] + y_norms[j] - 2 * ip;

                    if (!res.is_in_selection(j)) {
                        dis = HUGE_VALF;
                    }
                    // negative values can occur for identical vectors
                    // due to roundoff errors
                    if (dis < 0) {
                        dis = 0;
                    }

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

template <class BlockResultHandler>
void exhaustive_L2sqr_blas(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        BlockResultHandler& res,
        const float* y_norms = nullptr) {
    exhaustive_L2sqr_blas_default_impl(x, y, d, nx, ny, res, y_norms);
}

} // anonymous namespace

namespace {

// an override if only a single closest point is needed
template <>
void exhaustive_L2sqr_blas<Top1BlockResultHandler<CMax<float, int64_t>>>(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        Top1BlockResultHandler<CMax<float, int64_t>>& res,
        const float* y_norms) {
    // use a faster fused kernel if available
    if (exhaustive_L2sqr_fused_cmax(x, y, d, nx, ny, res, y_norms)) {
        return;
    }

    with_selected_simd_levels<AVAILABLE_SIMD_LEVELS_A2>([&]<SIMDLevel SL>() {
        if constexpr (SL == SIMDLevel::AVX2 || SL == SIMDLevel::ARM_SVE) {
            exhaustive_L2sqr_blas_cmax<SL>(x, y, d, nx, ny, res, y_norms);
        } else {
            exhaustive_L2sqr_blas_default_impl<
                    Top1BlockResultHandler<CMax<float, int64_t>>>(
                    x, y, d, nx, ny, res, y_norms);
        }
    });
}

struct Run_search_inner_product {
    using T = void;
    template <class BlockResultHandler>
    void f(BlockResultHandler& res,
           const float* x,
           const float* y,
           size_t d,
           size_t nx,
           size_t ny) {
        if (res.sel ||
            nx * d < static_cast<size_t>(distance_compute_blas_threshold)) {
            exhaustive_inner_product_seq(x, y, d, nx, ny, res);
        } else {
            exhaustive_inner_product_blas(x, y, d, nx, ny, res);
        }
    }
};

struct Run_search_L2sqr {
    using T = void;
    template <class BlockResultHandler>
    void f(BlockResultHandler& res,
           const float* x,
           const float* y,
           size_t d,
           size_t nx,
           size_t ny,
           const float* y_norm2) {
        if (res.sel ||
            nx * d < static_cast<size_t>(distance_compute_blas_threshold)) {
            exhaustive_L2sqr_seq(x, y, d, nx, ny, res);
        } else {
            exhaustive_L2sqr_blas(x, y, d, nx, ny, res, y_norm2);
        }
    }
};

} // anonymous namespace

/*******************************************************
 * KNN driver functions
 *******************************************************/

int distance_compute_blas_threshold = 128000;
int distance_compute_blas_query_bs = 4096;
int distance_compute_blas_database_bs = 1024;
int distance_compute_min_k_reservoir = 100;

// Database-parallel KNN: parallelizes over database segments instead of
// queries, for the case where nx < nthreads and the database is large.
static constexpr size_t kDbParallelMinVectors = 10000;

template <class C>
static void knn_db_parallel_impl(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        size_t k,
        float* vals,
        int64_t* ids,
        const float* y_norms) {
    using T = typename C::T;
    using TI = typename C::TI;

    int nt = omp_get_max_threads();
    const size_t bs_y = distance_compute_blas_database_bs;

    // Per-thread result heaps: nt threads x nx queries x k results
    std::vector<T> all_dis(static_cast<size_t>(nt) * nx * k);
    std::vector<TI> all_ids(static_cast<size_t>(nt) * nx * k);

    std::unique_ptr<float[]> x_norms_storage;
    std::unique_ptr<float[]> y_norms_storage;
    const float* x_norms = nullptr;
    // C::is_max corresponds to L2 (CMax), not IP (CMin)
    if constexpr (C::is_max) {
        x_norms_storage.reset(new float[nx]);
        fvec_norms_L2sqr(x_norms_storage.get(), x, d, nx);
        x_norms = x_norms_storage.get();

        if (!y_norms) {
            y_norms_storage.reset(new float[ny]);
            y_norms = y_norms_storage.get();
        }
    }

#pragma omp parallel num_threads(nt)
    {
        int tid = omp_get_thread_num();
        size_t j_begin = static_cast<size_t>(tid) * ny / nt;
        size_t j_end = static_cast<size_t>(tid + 1) * ny / nt;
        size_t local_ny = j_end - j_begin;

        // Compute y_norms for this thread's segment (cache locality)
        if constexpr (C::is_max) {
            if (y_norms_storage && local_ny > 0) {
                fvec_norms_L2sqr(
                        y_norms_storage.get() + j_begin,
                        y + j_begin * d,
                        d,
                        local_ny);
            }
        }

        T* my_dis = all_dis.data() + tid * nx * k;
        TI* my_ids = all_ids.data() + tid * nx * k;

        // Each thread initializes its own heaps
        for (size_t i = 0; i < nx; i++) {
            heap_heapify<C>(k, my_dis + i * k, my_ids + i * k);
        }

        if (local_ny > 0) {
            size_t max_block = std::min(bs_y, local_ny);
            std::unique_ptr<float[]> ip_block(new float[nx * max_block]);

            for (size_t jj0 = 0; jj0 < local_ny; jj0 += bs_y) {
                size_t jj1 = std::min(jj0 + bs_y, local_ny);
                size_t block_ny = jj1 - jj0;

                {
                    float one = 1, zero = 0;
                    FINTEGER nyi = static_cast<FINTEGER>(block_ny);
                    FINTEGER nxi = static_cast<FINTEGER>(nx);
                    FINTEGER di = static_cast<FINTEGER>(d);
                    sgemm_("Transpose",
                           "Not transpose",
                           &nyi,
                           &nxi,
                           &di,
                           &one,
                           y + (j_begin + jj0) * d,
                           &di,
                           x,
                           &di,
                           &zero,
                           ip_block.get(),
                           &nyi);
                }

                for (size_t i = 0; i < nx; i++) {
                    T* heap_dis = my_dis + i * k;
                    TI* heap_ids = my_ids + i * k;
                    const float* ip_line = ip_block.get() + i * block_ny;
                    T thresh = heap_dis[0];

                    for (size_t jj = 0; jj < block_ny; jj++) {
                        size_t global_j = j_begin + jj0 + jj;
                        float ip = ip_line[jj];
                        T dis;

                        if constexpr (C::is_max) {
                            dis = x_norms[i] + y_norms[global_j] - 2 * ip;
                            if (dis < 0) {
                                dis = 0;
                            }
                        } else {
                            dis = ip;
                        }

                        if (C::cmp(thresh, dis)) {
                            heap_replace_top<C>(
                                    k, heap_dis, heap_ids, dis, global_j);
                            thresh = heap_dis[0];
                        }
                    }
                }
            }
        }
    }

    // Merge per-thread heaps into output, parallelized over queries
#pragma omp parallel for
    for (int64_t i = 0; i < static_cast<int64_t>(nx); i++) {
        heap_heapify<C>(k, vals + i * k, ids + i * k);

        for (int t = 0; t < nt; t++) {
            T* t_dis = all_dis.data() + (t * nx + i) * k;
            TI* t_ids = all_ids.data() + (t * nx + i) * k;
            T* out_dis = vals + i * k;
            TI* out_ids = ids + i * k;

            for (size_t j = 0; j < k; j++) {
                if (t_ids[j] >= 0 && C::cmp(out_dis[0], t_dis[j])) {
                    heap_replace_top<C>(
                            k, out_dis, out_ids, t_dis[j], t_ids[j]);
                }
            }
        }

        heap_reorder<C>(k, vals + i * k, ids + i * k);
    }
}

static bool should_use_db_parallel(
        size_t nx,
        size_t ny,
        const IDSelector* sel) {
    if (sel) {
        return false;
    }
    int nt = omp_get_max_threads();
    size_t min_ny = std::max(
            kDbParallelMinVectors,
            static_cast<size_t>(nt) *
                    static_cast<size_t>(distance_compute_blas_database_bs));
    return nt > 1 && nx < static_cast<size_t>(nt) && ny >= min_ny;
}

void knn_inner_product(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        size_t k,
        float* vals,
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
                x, y, sela->ids, d, nx, ny, sela->n, k, vals, ids, 0);
        return;
    }

    if (should_use_db_parallel(nx, ny, sel)) {
        knn_db_parallel_impl<CMin<float, int64_t>>(
                x, y, d, nx, ny, k, vals, ids, nullptr);
    } else {
        Run_search_inner_product r;
        // @lint-ignore CLANGTIDY facebook-hte-NullableDereference
        dispatch_knn_ResultHandler(
                nx,
                vals,
                ids,
                k,
                METRIC_INNER_PRODUCT,
                sel,
                r,
                x,
                y,
                d,
                nx,
                ny);
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
        knn_L2sqr_by_idx(x, y, sela->ids, d, nx, ny, sela->n, k, vals, ids, 0);
        return;
    }

    if (should_use_db_parallel(nx, ny, sel)) {
        knn_db_parallel_impl<CMax<float, int64_t>>(
                x, y, d, nx, ny, k, vals, ids, y_norm2);
    } else {
        Run_search_L2sqr r;
        // @lint-ignore CLANGTIDY facebook-hte-NullableDereference
        dispatch_knn_ResultHandler(
                nx, vals, ids, k, METRIC_L2, sel, r, x, y, d, nx, ny, y_norm2);
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

// TODO accept a y_norm2 as well
void range_search_L2sqr(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        float radius,
        RangeSearchResult* res,
        const IDSelector* sel) {
    Run_search_L2sqr r;
    dispatch_range_ResultHandler(
            res, radius, METRIC_L2, sel, r, x, y, d, nx, ny, nullptr);
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
    Run_search_inner_product r;
    dispatch_range_ResultHandler(
            res, radius, METRIC_INNER_PRODUCT, sel, r, x, y, d, nx, ny);
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
    with_simd_level([&]<SIMDLevel SL>() {
#pragma omp parallel for
        for (int64_t j = 0; j < static_cast<int64_t>(nx); j++) {
            const int64_t* __restrict idsj = ids + j * ny;
            const float* xj = x + j * d;
            float* __restrict ipj = ip + j * ny;
            for (size_t i = 0; i < ny; i++) {
                if (idsj[i] < 0) {
                    ipj[i] = -INFINITY;
                } else {
                    ipj[i] = fvec_inner_product<SL>(xj, y + d * idsj[i], d);
                }
            }
        }
    });
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
    with_simd_level([&]<SIMDLevel SL>() {
#pragma omp parallel for
        for (int64_t j = 0; j < static_cast<int64_t>(nx); j++) {
            const int64_t* __restrict idsj = ids + j * ny;
            const float* xj = x + j * d;
            float* __restrict disj = dis + j * ny;
            for (size_t i = 0; i < ny; i++) {
                if (idsj[i] < 0) {
                    disj[i] = INFINITY;
                } else {
                    disj[i] = fvec_L2sqr<SL>(xj, y + d * idsj[i], d);
                }
            }
        }
    });
}

void pairwise_indexed_L2sqr(
        size_t d,
        size_t n,
        const float* x,
        const int64_t* ix,
        const float* y,
        const int64_t* iy,
        float* dis) {
    with_simd_level([&]<SIMDLevel SL>() {
#pragma omp parallel for if (n > 1)
        for (int64_t j = 0; j < static_cast<int64_t>(n); j++) {
            if (ix[j] >= 0 && iy[j] >= 0) {
                dis[j] = fvec_L2sqr<SL>(x + d * ix[j], y + d * iy[j], d);
            } else {
                dis[j] = INFINITY;
            }
        }
    });
}

void pairwise_indexed_inner_product(
        size_t d,
        size_t n,
        const float* x,
        const int64_t* ix,
        const float* y,
        const int64_t* iy,
        float* dis) {
    with_simd_level([&]<SIMDLevel SL>() {
#pragma omp parallel for if (n > 1)
        for (int64_t j = 0; j < static_cast<int64_t>(n); j++) {
            if (ix[j] >= 0 && iy[j] >= 0) {
                dis[j] =
                        fvec_inner_product<SL>(x + d * ix[j], y + d * iy[j], d);
            } else {
                dis[j] = -INFINITY;
            }
        }
    });
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
        size_t nsubset,
        size_t k,
        float* res_vals,
        int64_t* res_ids,
        int64_t ld_ids) {
    if (ld_ids < 0) {
        ld_ids = ny;
    }

    with_simd_level([&]<SIMDLevel SL>() {
#pragma omp parallel for if (nx > 100)
        for (int64_t i = 0; i < static_cast<int64_t>(nx); i++) {
            const float* x_ = x + i * d;
            const int64_t* idsi = ids + i * ld_ids;
            size_t j;
            float* __restrict simi = res_vals + i * k;
            int64_t* __restrict idxi = res_ids + i * k;
            minheap_heapify(k, simi, idxi);

            for (j = 0; j < nsubset; j++) {
                if (idsi[j] < 0 || static_cast<size_t>(idsi[j]) >= ny) {
                    break;
                }
                float ip = fvec_inner_product<SL>(x_, y + d * idsi[j], d);

                if (ip > simi[0]) {
                    minheap_replace_top(k, simi, idxi, ip, idsi[j]);
                }
            }
            minheap_reorder(k, simi, idxi);
        }
    });
}

void knn_L2sqr_by_idx(
        const float* x,
        const float* y,
        const int64_t* __restrict ids,
        size_t d,
        size_t nx,
        size_t ny,
        size_t nsubset,
        size_t k,
        float* res_vals,
        int64_t* res_ids,
        int64_t ld_ids) {
    if (ld_ids < 0) {
        ld_ids = ny;
    }
    with_simd_level([&]<SIMDLevel SL>() {
#pragma omp parallel for if (nx > 100)
        for (int64_t i = 0; i < static_cast<int64_t>(nx); i++) {
            const float* x_ = x + i * d;
            const int64_t* __restrict idsi = ids + i * ld_ids;
            float* __restrict simi = res_vals + i * k;
            int64_t* __restrict idxi = res_ids + i * k;
            maxheap_heapify(k, simi, idxi);
            for (size_t j = 0; j < nsubset; j++) {
                if (idsi[j] < 0 || static_cast<size_t>(idsi[j]) >= ny) {
                    break;
                }
                float disij = fvec_L2sqr<SL>(x_, y + d * idsi[j], d);

                if (disij < simi[0]) {
                    maxheap_replace_top(k, simi, idxi, disij, idsi[j]);
                }
            }
            maxheap_reorder(k, simi, idxi);
        }
    });
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
    if (nq == 0 || nb == 0) {
        return;
    }
    if (ldq == -1) {
        ldq = d;
    }
    if (ldb == -1) {
        ldb = d;
    }
    if (ldd == -1) {
        ldd = nb;
    }

    // store in beginning of distance matrix to avoid malloc
    float* b_norms = dis;

    with_simd_level([&]<SIMDLevel SL>() {
#pragma omp parallel for if (nb > 1)
        for (int64_t i = 0; i < nb; i++) {
            b_norms[i] = fvec_norm_L2sqr<SL>(xb + i * ldb, d);
        }

#pragma omp parallel for
        for (int64_t i = 1; i < nq; i++) {
            float q_norm = fvec_norm_L2sqr<SL>(xq + i * ldq, d);
            for (int64_t j = 0; j < nb; j++) {
                dis[i * ldd + j] = q_norm + b_norms[j];
            }
        }

        {
            float q_norm = fvec_norm_L2sqr<SL>(xq, d);
            for (int64_t j = 0; j < nb; j++) {
                dis[j] += q_norm;
            }
        }
    });

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
    for (int64_t j = 0; j < static_cast<int64_t>(n1); j++) {
        float* disj = dis + j * n2;
        for (size_t i = 0; i < n2; i++) {
            disj[i] = nr1[j] + nr2[i] - 2 * disj[i];
        }
    }
}

} // namespace faiss
