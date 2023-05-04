/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.
// -*- c++ -*-

#include "Index_c.h"
#include <faiss/Index.h>
#include <faiss/impl/IDSelector.h>
#include "macros_impl.h"

extern "C" {

DEFINE_DESTRUCTOR(SearchParameters)

int faiss_SearchParameters_new(
        FaissSearchParameters** p_sp,
        FaissIDSelector* sel) {
    try {
        faiss::SearchParameters* params = new faiss::SearchParameters;
        params->sel = reinterpret_cast<faiss::IDSelector*>(sel);
        *p_sp = reinterpret_cast<FaissSearchParameters*>(params);
        return 0;
    }
    CATCH_AND_HANDLE
}

DEFINE_DESTRUCTOR(Index)

DEFINE_GETTER(Index, int, d)

DEFINE_GETTER(Index, int, is_trained)

DEFINE_GETTER(Index, idx_t, ntotal)

DEFINE_GETTER(Index, FaissMetricType, metric_type)

DEFINE_GETTER(Index, int, verbose);
DEFINE_SETTER(Index, int, verbose);

int faiss_Index_train(FaissIndex* index, idx_t n, const float* x) {
    try {
        reinterpret_cast<faiss::Index*>(index)->train(n, x);
    }
    CATCH_AND_HANDLE
}

int faiss_Index_add(FaissIndex* index, idx_t n, const float* x) {
    try {
        reinterpret_cast<faiss::Index*>(index)->add(n, x);
    }
    CATCH_AND_HANDLE
}

int faiss_Index_add_with_ids(
        FaissIndex* index,
        idx_t n,
        const float* x,
        const idx_t* xids) {
    try {
        reinterpret_cast<faiss::Index*>(index)->add_with_ids(n, x, xids);
    }
    CATCH_AND_HANDLE
}

int faiss_Index_search(
        const FaissIndex* index,
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) {
    try {
        reinterpret_cast<const faiss::Index*>(index)->search(
                n, x, k, distances, labels);
    }
    CATCH_AND_HANDLE
}

int faiss_Index_search_with_params(
        const FaissIndex* index,
        idx_t n,
        const float* x,
        idx_t k,
        const FaissSearchParameters* params,
        float* distances,
        idx_t* labels) {
    try {
        reinterpret_cast<const faiss::Index*>(index)->search(
                n,
                x,
                k,
                distances,
                labels,
                reinterpret_cast<const faiss::SearchParameters*>(params));
    }
    CATCH_AND_HANDLE
}

int faiss_Index_range_search(
        const FaissIndex* index,
        idx_t n,
        const float* x,
        float radius,
        FaissRangeSearchResult* result) {
    try {
        reinterpret_cast<const faiss::Index*>(index)->range_search(
                n,
                x,
                radius,
                reinterpret_cast<faiss::RangeSearchResult*>(result));
    }
    CATCH_AND_HANDLE
}

int faiss_Index_assign(
        FaissIndex* index,
        idx_t n,
        const float* x,
        idx_t* labels,
        idx_t k) {
    try {
        reinterpret_cast<faiss::Index*>(index)->assign(n, x, labels, k);
    }
    CATCH_AND_HANDLE
}

int faiss_Index_reset(FaissIndex* index) {
    try {
        reinterpret_cast<faiss::Index*>(index)->reset();
    }
    CATCH_AND_HANDLE
}

int faiss_Index_remove_ids(
        FaissIndex* index,
        const FaissIDSelector* sel,
        size_t* n_removed) {
    try {
        size_t n{reinterpret_cast<faiss::Index*>(index)->remove_ids(
                *reinterpret_cast<const faiss::IDSelector*>(sel))};
        if (n_removed) {
            *n_removed = n;
        }
    }
    CATCH_AND_HANDLE
}

int faiss_Index_reconstruct(const FaissIndex* index, idx_t key, float* recons) {
    try {
        reinterpret_cast<const faiss::Index*>(index)->reconstruct(key, recons);
    }
    CATCH_AND_HANDLE
}

int faiss_Index_reconstruct_n(
        const FaissIndex* index,
        idx_t i0,
        idx_t ni,
        float* recons) {
    try {
        reinterpret_cast<const faiss::Index*>(index)->reconstruct_n(
                i0, ni, recons);
    }
    CATCH_AND_HANDLE
}

int faiss_Index_compute_residual(
        const FaissIndex* index,
        const float* x,
        float* residual,
        idx_t key) {
    try {
        reinterpret_cast<const faiss::Index*>(index)->compute_residual(
                x, residual, key);
    }
    CATCH_AND_HANDLE
}

int faiss_Index_compute_residual_n(
        const FaissIndex* index,
        idx_t n,
        const float* x,
        float* residuals,
        const idx_t* keys) {
    try {
        reinterpret_cast<const faiss::Index*>(index)->compute_residual_n(
                n, x, residuals, keys);
    }
    CATCH_AND_HANDLE
}

int faiss_Index_sa_code_size(const FaissIndex* index, size_t* size) {
    try {
        reinterpret_cast<const faiss::Index*>(index)->sa_code_size();
    }
    CATCH_AND_HANDLE
}

int faiss_Index_sa_encode(
        const FaissIndex* index,
        idx_t n,
        const float* x,
        uint8_t* bytes) {
    try {
        reinterpret_cast<const faiss::Index*>(index)->sa_encode(n, x, bytes);
    }
    CATCH_AND_HANDLE
}

int faiss_Index_sa_decode(
        const FaissIndex* index,
        idx_t n,
        const uint8_t* bytes,
        float* x) {
    try {
        reinterpret_cast<const faiss::Index*>(index)->sa_decode(n, bytes, x);
    }
    CATCH_AND_HANDLE
}
}
