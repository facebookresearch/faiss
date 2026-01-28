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

#include <omp.h>

#ifdef __AVX2__
#include <immintrin.h>
#elif defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/exhaustive_search_ops.h>

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
#pragma omp parallel for if (nx > 10000)
    for (int64_t i = 0; i < nx; i++) {
        nr[i] = sqrtf(fvec_norm_L2sqr(x + i * d, d));
    }
}

void fvec_norms_L2sqr(
        float* __restrict nr,
        const float* __restrict x,
        size_t d,
        size_t nx) {
#pragma omp parallel for if (nx > 10000)
    for (int64_t i = 0; i < nx; i++) {
        nr[i] = fvec_norm_L2sqr(x + i * d, d);
    }
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

#define FVEC_RENORM_L2_IMPL                   \
    float* __restrict xi = x + i * d;         \
                                              \
    float nr = fvec_norm_L2sqr(xi, d);        \
                                              \
    if (nr > 0) {                             \
        size_t j;                             \
        const float inv_nr = 1.0 / sqrtf(nr); \
        for (j = 0; j < d; j++)               \
            xi[j] *= inv_nr;                  \
    }

void fvec_renorm_L2_noomp(size_t d, size_t nx, float* __restrict x) {
    for (int64_t i = 0; i < nx; i++) {
        FVEC_RENORM_L2_IMPL
    }
}

void fvec_renorm_L2_omp(size_t d, size_t nx, float* __restrict x) {
#pragma omp parallel for if (nx > 10000)
    for (int64_t i = 0; i < nx; i++) {
        FVEC_RENORM_L2_IMPL
    }
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

namespace {} // anonymous namespace

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

    Run_search_inner_product r;
    dispatch_knn_ResultHandler(
            nx, vals, ids, k, METRIC_INNER_PRODUCT, sel, r, x, y, d, nx, ny);

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

    Run_search_L2sqr r;
    dispatch_knn_ResultHandler(
            nx, vals, ids, k, METRIC_L2, sel, r, x, y, d, nx, ny, y_norm2);

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
#pragma omp parallel for
    for (int64_t j = 0; j < nx; j++) {
        const int64_t* __restrict idsj = ids + j * ny;
        const float* xj = x + j * d;
        float* __restrict ipj = ip + j * ny;
        for (size_t i = 0; i < ny; i++) {
            if (idsj[i] < 0) {
                ipj[i] = -INFINITY;
            } else {
                ipj[i] = fvec_inner_product(xj, y + d * idsj[i], d);
            }
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
            if (idsj[i] < 0) {
                disj[i] = INFINITY;
            } else {
                disj[i] = fvec_L2sqr(xj, y + d * idsj[i], d);
            }
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
#pragma omp parallel for if (n > 1)
    for (int64_t j = 0; j < n; j++) {
        if (ix[j] >= 0 && iy[j] >= 0) {
            dis[j] = fvec_L2sqr(x + d * ix[j], y + d * iy[j], d);
        } else {
            dis[j] = INFINITY;
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
#pragma omp parallel for if (n > 1)
    for (int64_t j = 0; j < n; j++) {
        if (ix[j] >= 0 && iy[j] >= 0) {
            dis[j] = fvec_inner_product(x + d * ix[j], y + d * iy[j], d);
        } else {
            dis[j] = -INFINITY;
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
        size_t nsubset,
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

        for (j = 0; j < nsubset; j++) {
            if (idsi[j] < 0 || idsi[j] >= ny) {
                break;
            }
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
        size_t nsubset,
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
        for (size_t j = 0; j < nsubset; j++) {
            if (idsi[j] < 0 || idsi[j] >= ny) {
                break;
            }
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

#pragma omp parallel for if (nb > 1)
    for (int64_t i = 0; i < nb; i++) {
        b_norms[i] = fvec_norm_L2sqr(xb + i * ldb, d);
    }

#pragma omp parallel for
    for (int64_t i = 1; i < nq; i++) {
        float q_norm = fvec_norm_L2sqr(xq + i * ldq, d);
        for (int64_t j = 0; j < nb; j++) {
            dis[i * ldd + j] = q_norm + b_norms[j];
        }
    }

    {
        float q_norm = fvec_norm_L2sqr(xq, d);
        for (int64_t j = 0; j < nb; j++) {
            dis[j] += q_norm;
        }
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
        for (size_t i = 0; i < n2; i++) {
            disj[i] = nr1[j] + nr2[i] - 2 * disj[i];
        }
    }
}

} // namespace faiss
