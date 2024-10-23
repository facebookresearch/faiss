/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c -*-

#ifndef FAISS_DISTANCES_C_H
#define FAISS_DISTANCES_C_H

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/*********************************************************
 * Optimized distance/norm/inner prod computations
 *********************************************************/

/// Compute pairwise distances between sets of vectors
void faiss_pairwise_L2sqr(
        int64_t d,
        int64_t nq,
        const float* xq,
        int64_t nb,
        const float* xb,
        float* dis,
        int64_t ldq,
        int64_t ldb,
        int64_t ldd);

/// Compute pairwise distances between sets of vectors
/// arguments from "faiss_pairwise_L2sqr"
/// ldq equal -1 by default
/// ldb equal -1 by default
/// ldd equal -1 by default
void faiss_pairwise_L2sqr_with_defaults(
        int64_t d,
        int64_t nq,
        const float* xq,
        int64_t nb,
        const float* xb,
        float* dis);

/// compute the inner product between nx vectors x and one y
void faiss_fvec_inner_products_ny(
        float* ip, /* output inner product */
        const float* x,
        const float* y,
        size_t d,
        size_t ny);

/// compute ny square L2 distance between x and a set of contiguous y vectors
void faiss_fvec_L2sqr_ny(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny);

/// squared norm of a vector
float faiss_fvec_norm_L2sqr(const float* x, size_t d);

/// compute the L2 norms for a set of vectors
void faiss_fvec_norms_L2(float* norms, const float* x, size_t d, size_t nx);

/// same as fvec_norms_L2, but computes squared norms
void faiss_fvec_norms_L2sqr(float* norms, const float* x, size_t d, size_t nx);

/// L2-renormalize a set of vector. Nothing done if the vector is 0-normed
void faiss_fvec_renorm_L2(size_t d, size_t nx, float* x);

/// Setter of threshold value on nx above which we switch to BLAS to compute
/// distances
void faiss_set_distance_compute_blas_threshold(int value);

/// Getter of threshold value on nx above which we switch to BLAS to compute
/// distances
int faiss_get_distance_compute_blas_threshold();

/// Setter of block sizes value for BLAS distance computations
void faiss_set_distance_compute_blas_query_bs(int value);

/// Getter of block sizes value for BLAS distance computations
int faiss_get_distance_compute_blas_query_bs();

/// Setter of block sizes value for BLAS distance computations
void faiss_set_distance_compute_blas_database_bs(int value);

/// Getter of block sizes value for BLAS distance computations
int faiss_get_distance_compute_blas_database_bs();

/// Setter of number of results we switch to a reservoir to collect results
/// rather than a heap
void faiss_set_distance_compute_min_k_reservoir(int value);

/// Getter of number of results we switch to a reservoir to collect results
/// rather than a heap
int faiss_get_distance_compute_min_k_reservoir();

#ifdef __cplusplus
}
#endif

#endif
