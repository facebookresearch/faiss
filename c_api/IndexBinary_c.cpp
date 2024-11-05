/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include "IndexBinary_c.h"
#include <faiss/IndexBinary.h>
#include "macros_impl.h"

extern "C" {

DEFINE_DESTRUCTOR(IndexBinary)

DEFINE_GETTER(IndexBinary, int, d)

DEFINE_GETTER(IndexBinary, int, is_trained)

DEFINE_GETTER(IndexBinary, idx_t, ntotal)

DEFINE_GETTER(IndexBinary, FaissMetricType, metric_type)

DEFINE_GETTER(IndexBinary, int, verbose);
DEFINE_SETTER(IndexBinary, int, verbose);

int faiss_IndexBinary_train(
        FaissIndexBinary* index,
        idx_t n,
        const uint8_t* x) {
    try {
        reinterpret_cast<faiss::IndexBinary*>(index)->train(n, x);
    }
    CATCH_AND_HANDLE
}

int faiss_IndexBinary_add(FaissIndexBinary* index, idx_t n, const uint8_t* x) {
    try {
        reinterpret_cast<faiss::IndexBinary*>(index)->add(n, x);
    }
    CATCH_AND_HANDLE
}

int faiss_IndexBinary_add_with_ids(
        FaissIndexBinary* index,
        idx_t n,
        const uint8_t* x,
        const idx_t* xids) {
    try {
        reinterpret_cast<faiss::IndexBinary*>(index)->add_with_ids(n, x, xids);
    }
    CATCH_AND_HANDLE
}

int faiss_IndexBinary_search(
        const FaissIndexBinary* index,
        idx_t n,
        const uint8_t* x,
        idx_t k,
        int32_t* distances,
        idx_t* labels) {
    try {
        reinterpret_cast<const faiss::IndexBinary*>(index)->search(
                n, x, k, distances, labels);
    }
    CATCH_AND_HANDLE
}

int faiss_IndexBinary_range_search(
        const FaissIndexBinary* index,
        idx_t n,
        const uint8_t* x,
        int radius,
        FaissRangeSearchResult* result) {
    try {
        reinterpret_cast<const faiss::IndexBinary*>(index)->range_search(
                n,
                x,
                radius,
                reinterpret_cast<faiss::RangeSearchResult*>(result));
    }
    CATCH_AND_HANDLE
}

int faiss_IndexBinary_assign(
        FaissIndexBinary* index,
        idx_t n,
        const uint8_t* x,
        idx_t* labels,
        idx_t k) {
    try {
        reinterpret_cast<faiss::IndexBinary*>(index)->assign(n, x, labels, k);
    }
    CATCH_AND_HANDLE
}

int faiss_IndexBinary_reset(FaissIndexBinary* index) {
    try {
        reinterpret_cast<faiss::IndexBinary*>(index)->reset();
    }
    CATCH_AND_HANDLE
}

int faiss_IndexBinary_remove_ids(
        FaissIndexBinary* index,
        const FaissIDSelector* sel,
        size_t* n_removed) {
    try {
        size_t n{reinterpret_cast<faiss::IndexBinary*>(index)->remove_ids(
                *reinterpret_cast<const faiss::IDSelector*>(sel))};
        if (n_removed) {
            *n_removed = n;
        }
    }
    CATCH_AND_HANDLE
}

int faiss_IndexBinary_reconstruct(
        const FaissIndexBinary* index,
        idx_t key,
        uint8_t* recons) {
    try {
        reinterpret_cast<const faiss::IndexBinary*>(index)->reconstruct(
                key, recons);
    }
    CATCH_AND_HANDLE
}

int faiss_IndexBinary_reconstruct_n(
        const FaissIndexBinary* index,
        idx_t i0,
        idx_t ni,
        uint8_t* recons) {
    try {
        reinterpret_cast<const faiss::IndexBinary*>(index)->reconstruct_n(
                i0, ni, recons);
    }
    CATCH_AND_HANDLE
}
}
