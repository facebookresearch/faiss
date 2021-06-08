/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.
// -*- c++ -*-

#include "distances_c.h"
#include <faiss/utils/distances.h>

void faiss_set_distance_compute_blas_threshold(int value) {
    faiss::distance_compute_blas_threshold = value;
}

int faiss_get_distance_compute_blas_threshold() {
    return faiss::distance_compute_blas_threshold;
}

void faiss_set_distance_compute_blas_query_bs(int value) {
    faiss::distance_compute_blas_query_bs = value;
}

int faiss_get_distance_compute_blas_query_bs() {
    return faiss::distance_compute_blas_query_bs;
}

void faiss_set_distance_compute_blas_database_bs(int value) {
    faiss::distance_compute_blas_database_bs = value;
}

int faiss_get_distance_compute_blas_database_bs() {
    return faiss::distance_compute_blas_database_bs;
}

void faiss_set_distance_compute_min_k_reservoir(int value) {
    faiss::distance_compute_min_k_reservoir = value;
}

int faiss_get_distance_compute_min_k_reservoir() {
    return faiss::distance_compute_min_k_reservoir;
}
