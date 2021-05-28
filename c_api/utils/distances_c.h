/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.
// -*- c -*-

#ifndef FAISS_DISTANCES_C_H
#define FAISS_DISTANCES_C_H

#ifdef __cplusplus
extern "C" {
#endif

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
