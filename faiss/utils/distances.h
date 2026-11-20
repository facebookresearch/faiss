/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* All distance functions for L2 and IP distances.
 * The actual functions are implemented in distances.cpp and distances_simd.cpp
 */

#pragma once

#include <stdint.h>

#include <faiss/impl/platform_macros.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/simd_levels.h>

namespace faiss {

struct IDSelector;

/*********************************************************
 * Optimized distance/norm/inner prod computations
 *********************************************************/

/// Squared L2 distance between two vectors
float fvec_L2sqr(const float* x, const float* y, size_t d);

template <SIMDLevel>
float fvec_L2sqr(const float* x, const float* y, size_t d);

/// inner product
float fvec_inner_product(const float* x, const float* y, size_t d);

template <SIMDLevel>
float fvec_inner_product(const float* x, const float* y, size_t d);

/// L1 distance
float fvec_L1(const float* x, const float* y, size_t d);

template <SIMDLevel>
float fvec_L1(const float* x, const float* y, size_t d);

/// infinity distance
float fvec_Linf(const float* x, const float* y, size_t d);

template <SIMDLevel>
float fvec_Linf(const float* x, const float* y, size_t d);

/// Special version of inner product that computes 4 distances
/// between x and yi, which is performance oriented.
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
        float& dis3);

template <SIMDLevel>
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
        float& dis3);

/// Special version of L2sqr that computes 4 distances
/// between x and yi, which is performance oriented.
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
        float& dis3);

template <SIMDLevel>
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
        float& dis3);

/** Compute pairwise distances between sets of vectors
 *
 * @param d     dimension of the vectors
 * @param nq    nb of query vectors
 * @param nb    nb of database vectors
 * @param xq    query vectors (size nq * d)
 * @param xb    database vectors (size nb * d)
 * @param dis   output distances (size nq * nb)
 * @param ldq,ldb, ldd strides for the matrices
 */
void pairwise_L2sqr(
        int64_t d,
        int64_t nq,
        const float* xq,
        int64_t nb,
        const float* xb,
        float* dis,
        int64_t ldq = -1,
        int64_t ldb = -1,
        int64_t ldd = -1);

/* compute the inner product between nx vectors x and one y */
void fvec_inner_products_ny(
        float* ip, /* output inner product */
        const float* x,
        const float* y,
        size_t d,
        size_t ny);

template <SIMDLevel>
void fvec_inner_products_ny(
        float* ip, /* output inner product */
        const float* x,
        const float* y,
        size_t d,
        size_t ny);

/* compute ny square L2 distance between x and a set of contiguous y vectors */
void fvec_L2sqr_ny(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny);

template <SIMDLevel>
void fvec_L2sqr_ny(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny);

/* compute ny square L2 distance between x and a set of transposed contiguous
   y vectors. squared lengths of y should be provided as well */
void fvec_L2sqr_ny_transposed(
        float* dis,
        const float* x,
        const float* y,
        const float* y_sqlen,
        size_t d,
        size_t d_offset,
        size_t ny);

template <SIMDLevel>
void fvec_L2sqr_ny_transposed(
        float* dis,
        const float* x,
        const float* y,
        const float* y_sqlen,
        size_t d,
        size_t d_offset,
        size_t ny);

/* compute ny square L2 distance between x and a set of contiguous y vectors
   and return the index of the nearest vector.
   return 0 if ny == 0. */
size_t fvec_L2sqr_ny_nearest(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        size_t d,
        size_t ny);

template <SIMDLevel>
size_t fvec_L2sqr_ny_nearest(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        size_t d,
        size_t ny);

/* compute ny square L2 distance between x and a set of transposed contiguous
   y vectors and return the index of the nearest vector.
   squared lengths of y should be provided as well
   return 0 if ny == 0. */
size_t fvec_L2sqr_ny_nearest_y_transposed(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        const float* y_sqlen,
        size_t d,
        size_t d_offset,
        size_t ny);

template <SIMDLevel>
size_t fvec_L2sqr_ny_nearest_y_transposed(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        const float* y_sqlen,
        size_t d,
        size_t d_offset,
        size_t ny);

/** squared norm of a vector */
float fvec_norm_L2sqr(const float* x, size_t d);

template <SIMDLevel>
float fvec_norm_L2sqr(const float* x, size_t d);

/** compute the L2 norms for a set of vectors
 *
 * @param  norms    output norms, size nx
 * @param  x        set of vectors, size nx * d
 */
void fvec_norms_L2(float* norms, const float* x, size_t d, size_t nx);

/// same as fvec_norms_L2, but computes squared norms
void fvec_norms_L2sqr(float* norms, const float* x, size_t d, size_t nx);

/* L2-renormalize a set of vector. Nothing done if the vector is 0-normed */
void fvec_renorm_L2(size_t d, size_t nx, float* x);

/* This function exists because the Torch counterpart is extremely slow
   (not multi-threaded + unexpected overhead even in single thread).
   It is here to implement the usual property |x-y|^2=|x|^2+|y|^2-2<x|y>  */
void inner_product_to_L2sqr(
        float* dis,
        const float* nr1,
        const float* nr2,
        size_t n1,
        size_t n2);

/*********************************************************
 * Vector to vector functions
 *********************************************************/

/** compute c := a + b for vectors
 *
 * c and a can overlap, c and b can overlap
 *
 * @param a size d
 * @param b size d
 * @param c size d
 */
void fvec_add(size_t d, const float* a, const float* b, float* c);

template <SIMDLevel>
void fvec_add(size_t d, const float* a, const float* b, float* c);

/** compute c := a + b for a, c vectors and b a scalar
 *
 * c and a can overlap
 *
 * @param a size d
 * @param c size d
 */
void fvec_add(size_t d, const float* a, float b, float* c);

template <SIMDLevel>
void fvec_add(size_t d, const float* a, float b, float* c);

/** compute c := a - b for vectors
 *
 * c and a can overlap, c and b can overlap
 *
 * @param a size d
 * @param b size d
 * @param c size d
 */
void fvec_sub(size_t d, const float* a, const float* b, float* c);

template <SIMDLevel>
void fvec_sub(size_t d, const float* a, const float* b, float* c);

/***************************************************************************
 * Compute a subset of  distances
 ***************************************************************************/

/** compute the inner product between x and a subset y of ny vectors defined by
 * ids
 *
 * ip(i, j) = inner_product(x(i, :), y(ids(i, j), :))
 *
 * @param ip    output array, size nx * ny
 * @param x     first-term vector, size nx * d
 * @param y     second-term vector, size (max(ids) + 1) * d
 * @param ids   ids to sample from y, size nx * ny
 */
void fvec_inner_products_by_idx(
        float* ip,
        const float* x,
        const float* y,
        const int64_t* ids,
        size_t d,
        size_t nx,
        size_t ny);

/** compute the squared L2 distances between x and a subset y of ny vectors
 * defined by ids
 *
 * dis(i, j) = inner_product(x(i, :), y(ids(i, j), :))
 *
 * @param dis   output array, size nx * ny
 * @param x     first-term vector, size nx * d
 * @param y     second-term vector, size (max(ids) + 1) * d
 * @param ids   ids to sample from y, size nx * ny
 */
void fvec_L2sqr_by_idx(
        float* dis,
        const float* x,
        const float* y,
        const int64_t* ids, /* ids of y vecs */
        size_t d,
        size_t nx,
        size_t ny);

/** compute dis[j] = L2sqr(x[ix[j]], y[iy[j]]) forall j=0..n-1
 *
 * @param x  size (max(ix) + 1, d)
 * @param y  size (max(iy) + 1, d)
 * @param ix size n
 * @param iy size n
 * @param dis size n
 */
void pairwise_indexed_L2sqr(
        size_t d,
        size_t n,
        const float* x,
        const int64_t* ix,
        const float* y,
        const int64_t* iy,
        float* dis);

/** compute dis[j] = inner_product(x[ix[j]], y[iy[j]]) forall j=0..n-1
 *
 * @param x  size (max(ix) + 1, d)
 * @param y  size (max(iy) + 1, d)
 * @param ix size n
 * @param iy size n
 * @param dis size n
 */
void pairwise_indexed_inner_product(
        size_t d,
        size_t n,
        const float* x,
        const int64_t* ix,
        const float* y,
        const int64_t* iy,
        float* dis);

/***************************************************************************
 * KNN functions
 ***************************************************************************/

// threshold on nx * d above which we switch to BLAS to compute distances
FAISS_API extern int distance_compute_blas_threshold;

// block sizes for BLAS distance computations
FAISS_API extern int distance_compute_blas_query_bs;
FAISS_API extern int distance_compute_blas_database_bs;

// above this number of results we switch to a reservoir to collect results
// rather than a heap
FAISS_API extern int distance_compute_min_k_reservoir;

/** Return the k nearest neighbors of each of the nx vectors x among the ny
 *  vector y, w.r.t to max inner product.
 *
 * @param x    query vectors, size nx * d
 * @param y    database vectors, size ny * d
 * @param res  result heap structure, which also provides k. Sorted on output
 */
void knn_inner_product(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        float_minheap_array_t* res,
        const IDSelector* sel = nullptr);

/**  Return the k nearest neighbors of each of the nx vectors x among the ny
 *  vector y, for the inner product metric.
 *
 * @param x    query vectors, size nx * d
 * @param y    database vectors, size ny * d
 * @param distances  output distances, size nq * k
 * @param indexes    output vector ids, size nq * k
 */
void knn_inner_product(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        size_t k,
        float* distances,
        int64_t* indexes,
        const IDSelector* sel = nullptr);

/** Return the k nearest neighbors of each of the nx vectors x among the ny
 *  vector y, for the L2 distance
 * @param x    query vectors, size nx * d
 * @param y    database vectors, size ny * d
 * @param res  result heap structure, which also provides k. Sorted on output
 * @param y_norm2    (optional) norms for the y vectors (nullptr or size ny)
 * @param sel  search in this subset of vectors
 */
void knn_L2sqr(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        float_maxheap_array_t* res,
        const float* y_norm2 = nullptr,
        const IDSelector* sel = nullptr);

/**  Return the k nearest neighbors of each of the nx vectors x among the ny
 *  vector y, for the L2 distance
 *
 * @param x    query vectors, size nx * d
 * @param y    database vectors, size ny * d
 * @param distances  output distances, size nq * k
 * @param indexes    output vector ids, size nq * k
 * @param y_norm2    (optional) norms for the y vectors (nullptr or size ny)
 * @param sel  search in this subset of vectors
 */
void knn_L2sqr(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        size_t k,
        float* distances,
        int64_t* indexes,
        const float* y_norm2 = nullptr,
        const IDSelector* sel = nullptr);

/** knn_inner_product for packed IEEE binary16 queries (x: nx * d uint16_t).
 *
 * The BLAS kernels widen each query tile once and reuse it for all database
 * tiles, so there is no fp32 copy of the queries. Results are the same as
 * knn_inner_product on the widened queries.
 */
void knn_inner_product_fp16(
        const uint16_t* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        size_t k,
        float* distances,
        int64_t* indexes,
        const IDSelector* sel = nullptr);

/// knn_L2sqr for packed IEEE binary16 queries, see knn_inner_product_fp16
void knn_L2sqr_fp16(
        const uint16_t* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        size_t k,
        float* distances,
        int64_t* indexes,
        const float* y_norm2 = nullptr,
        const IDSelector* sel = nullptr);

/** Find the max inner product neighbors for nx queries in a set of ny vectors
 * indexed by ids. May be useful for re-ranking a pre-selected vector list
 *
 * @param x    query vectors, size nx * d
 * @param y    database vectors, size (max(ids) + 1) * d
 * @param ids  subset of database vectors to consider, size (nx, nsubset)
 * @param res  result structure
 * @param ld_ids stride for the ids array. -1: use nsubset, 0: all queries
 * process the same subset
 */
void knn_inner_products_by_idx(
        const float* x,
        const float* y,
        const int64_t* subset,
        size_t d,
        size_t nx,
        size_t ny,
        size_t nsubset,
        size_t k,
        float* vals,
        int64_t* ids,
        int64_t ld_ids = -1);

/** Find the nearest neighbors for nx queries in a set of ny vectors
 * indexed by ids. May be useful for re-ranking a pre-selected vector list
 *
 * @param x    query vectors, size nx * d
 * @param y    database vectors, size (max(ids) + 1) * d
 * @param subset subset of database vectors to consider, size (nx, nsubset)
 * @param res  result structure
 * @param ld_subset stride for the subset array. -1: use nsubset, 0: all queries
 * process the same subset
 */
void knn_L2sqr_by_idx(
        const float* x,
        const float* y,
        const int64_t* subset,
        size_t d,
        size_t nx,
        size_t ny,
        size_t nsubset,
        size_t k,
        float* vals,
        int64_t* ids,
        int64_t ld_subset = -1);

/***************************************************************************
 * Range search
 ***************************************************************************/

/// Forward declaration, see AuxIndexStructures.h
struct RangeSearchResult;

/** Return the k nearest neighbors of each of the nx vectors x among the ny
 *  vector y, w.r.t to max inner product
 *
 * @param x      query vectors, size nx * d
 * @param y      database vectors, size ny * d
 * @param radius search radius around the x vectors
 * @param result result structure
 */
void range_search_L2sqr(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        float radius,
        RangeSearchResult* result,
        const IDSelector* sel = nullptr);

/// same as range_search_L2sqr for the inner product similarity
void range_search_inner_product(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        float radius,
        RangeSearchResult* result,
        const IDSelector* sel = nullptr);

/***************************************************************************
 * PQ tables computations
 ***************************************************************************/

/// specialized function for PQ2
void compute_PQ_dis_tables_dsub2(
        size_t d,
        size_t ksub,
        const float* centroids,
        size_t nx,
        const float* x,
        bool is_inner_product,
        float* dis_tables);

template <SIMDLevel>
void compute_PQ_dis_tables_dsub2(
        size_t d,
        size_t ksub,
        const float* centroids,
        size_t nx,
        const float* x,
        bool is_inner_product,
        float* dis_tables);

/***************************************************************************
 * Templatized versions of distance functions
 ***************************************************************************/

/***************************************************************************
 * Misc  matrix and vector manipulation functions
 ***************************************************************************/

/** Per-cluster sums and counts of labeled vectors.
 *
 * For each i < n, adds x[i] to sums[labels[i], 0:d] and 1 to
 * sums[labels[i], d]. The result equals numpy.add.at(sums[:, :d], labels, x)
 * and numpy.add.at(sums[:, d], labels, 1) bit for bit: the rows of each
 * cluster are added in increasing i, whatever the number of threads.
 * Unlike numpy.add.at, which is single-threaded, it runs with OpenMP when
 * n * d > 65536: each thread owns a contiguous range of labels and scans all
 * rows, so no two threads write the same row of sums and no reduction is
 * needed.
 *
 * All arrays are contiguous and row-major; x and sums are float32, labels
 * int64; sums must not overlap x or labels.
 *
 * @param d       dimension of the vectors
 * @param n       number of vectors
 * @param k       number of clusters
 * @param x       input vectors, size n * d
 * @param labels  input cluster of each vector, size n, in [0, k)
 * @param sums    in/out sums and counts, size k * (d + 1): row a holds the
 *                d components of the sum and then the count; results are
 *                added to its contents
 */
void accumulate_cluster_sums(
        size_t d,
        size_t n,
        size_t k,
        const float* x,
        const int64_t* labels,
        float* sums);

/** Per-cluster removal costs and Oja long-axis terms of an assignment.
 *
 * Used by e-means' cost-based revival. For each i < n, with a = labels[i],
 * y = x[i] - centroids[a] and p = <y, axes[a]>, adds p * y to
 * stats[a, 0:d] and the removal cost d2[i] - d1[i] to stats[a, d] (zero is
 * added when the computed difference is not finite). The rows of each
 * cluster are added in increasing i, so the result does not depend on the
 * number of threads. OpenMP as in accumulate_cluster_sums: each thread owns
 * a range of labels.
 *
 * All arrays are contiguous and row-major, float32 except the int64
 * labels; stats must not overlap any input.
 *
 * @param d          dimension of the vectors
 * @param n          number of vectors
 * @param k          number of clusters
 * @param x          input vectors, size n * d
 * @param labels     input nearest centroid of each vector, size n, in [0, k)
 * @param d1         input squared distance to the nearest centroid, size n
 * @param d2         input squared distance to the second-nearest centroid,
 *                   size n
 * @param centroids  input centroids used for the assignment, size k * d
 * @param axes       input unit long axis of each cluster, size k * d
 * @param stats      in/out per-cluster sums, size k * (d + 1): row a holds
 *                   the sum of p * y and then the sum of d2 - d1; results
 *                   are added to its contents
 */
void accumulate_removal_stats(
        size_t d,
        size_t n,
        size_t k,
        const float* x,
        const int64_t* labels,
        const float* d1,
        const float* d2,
        const float* centroids,
        const float* axes,
        float* stats);

/** Oja update of the cluster axes from accumulate_removal_stats sums.
 *
 * For each cluster a = ids[i], i < n: u = axis[a, 0:d] += weight *
 * batch[a, 0:d]; if |u| > 0 (computed in double), axes[a] = u / |u|;
 * batch[a] is then zeroed. Clusters are independent; OpenMP over ids when
 * n * d > 65536. Rows not in ids keep their contents, and so does axes[a]
 * when |u| = 0.
 *
 * All arrays are contiguous and row-major, float32 except the int64 ids;
 * batch, axis and axes have k rows (k is the number of clusters they were
 * allocated for) and must not overlap each other or ids.
 *
 * @param d       dimension of the vectors
 * @param n       number of clusters to update
 * @param ids     input clusters to update, size n, distinct, each in [0, k)
 * @param weight  input weight of the batch sums
 * @param batch   in/out batch sums, k rows of size d + 1 (as stats in
 *                accumulate_removal_stats); rows ids are zeroed
 * @param axis    in/out running sums u, k rows of size d + 1 (component d
 *                is not used)
 * @param axes    out unit axes, k rows of size d; rows ids with |u| > 0 are
 *                overwritten
 */
void update_removal_axes(
        size_t d,
        size_t n,
        const int64_t* ids,
        float weight,
        float* batch,
        float* axis,
        float* axes);

/** compute c := a + bf * b for a, b and c tables
 *
 * @param n   size of the tables
 * @param a   size n
 * @param b   size n
 * @param c   result table, size n
 */
void fvec_madd(size_t n, const float* a, float bf, const float* b, float* c);

/* same statically */
template <SIMDLevel>
void fvec_madd(size_t n, const float* a, float bf, const float* b, float* c);

/** same as fvec_madd, also return index of the min of the result table
 * @return    index of the min of table c
 */
int fvec_madd_and_argmin(
        size_t n,
        const float* a,
        float bf,
        const float* b,
        float* c);

template <SIMDLevel>
int fvec_madd_and_argmin(
        size_t n,
        const float* a,
        float bf,
        const float* b,
        float* c);

/// sum of x[j] for every j < d whose bit is set in bits (LSB-first per byte)
template <SIMDLevel>
float fvec_masked_sum(const float* x, const uint8_t* bits, size_t d);

} // namespace faiss
