/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.
// -*- c++ -*-

#include "IndexFlat_c.h"
#include <faiss/IndexFlat.h>
#include <faiss/IndexRefine.h>
#include "macros_impl.h"

extern "C" {

using faiss::Index;
using faiss::IndexFlat;
using faiss::IndexFlat1D;
using faiss::IndexFlatIP;
using faiss::IndexFlatL2;
using faiss::IndexRefineFlat;

DEFINE_DESTRUCTOR(IndexFlat)
DEFINE_INDEX_DOWNCAST(IndexFlat)

int faiss_IndexFlat_new(FaissIndexFlat** p_index) {
    try {
        *p_index = reinterpret_cast<FaissIndexFlat*>(new IndexFlat());
        return 0;
    }
    CATCH_AND_HANDLE
}

int faiss_IndexFlat_new_with(
        FaissIndexFlat** p_index,
        idx_t d,
        FaissMetricType metric) {
    try {
        IndexFlat* index =
                new IndexFlat(d, static_cast<faiss::MetricType>(metric));
        *p_index = reinterpret_cast<FaissIndexFlat*>(index);
        return 0;
    }
    CATCH_AND_HANDLE
}

void faiss_IndexFlat_xb(FaissIndexFlat* index, float** p_xb, size_t* p_size) {
    IndexFlat* indexf = reinterpret_cast<IndexFlat*>(index);
    *p_xb = indexf->get_xb();
    if (p_size) {
        *p_size = indexf->codes.size() / sizeof(float);
    }
}

int faiss_IndexFlat_compute_distance_subset(
        FaissIndex* index,
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        const idx_t* labels) {
    try {
        reinterpret_cast<IndexFlat*>(index)->compute_distance_subset(
                n, x, k, distances, labels);
        return 0;
    }
    CATCH_AND_HANDLE
}

DEFINE_DESTRUCTOR(IndexFlatIP)
DEFINE_INDEX_DOWNCAST(IndexFlatIP)

int faiss_IndexFlatIP_new(FaissIndexFlatIP** p_index) {
    try {
        IndexFlatIP* index = new IndexFlatIP();
        *p_index = reinterpret_cast<FaissIndexFlatIP*>(index);
        return 0;
    }
    CATCH_AND_HANDLE
}

int faiss_IndexFlatIP_new_with(FaissIndexFlatIP** p_index, idx_t d) {
    try {
        IndexFlatIP* index = new IndexFlatIP(d);
        *p_index = reinterpret_cast<FaissIndexFlatIP*>(index);
        return 0;
    }
    CATCH_AND_HANDLE
}

DEFINE_DESTRUCTOR(IndexFlatL2)
DEFINE_INDEX_DOWNCAST(IndexFlatL2)

int faiss_IndexFlatL2_new(FaissIndexFlatL2** p_index) {
    try {
        IndexFlatL2* index = new IndexFlatL2();
        *p_index = reinterpret_cast<FaissIndexFlatL2*>(index);
        return 0;
    }
    CATCH_AND_HANDLE
}

int faiss_IndexFlatL2_new_with(FaissIndexFlatL2** p_index, idx_t d) {
    try {
        IndexFlatL2* index = new IndexFlatL2(d);
        *p_index = reinterpret_cast<FaissIndexFlatL2*>(index);
        return 0;
    }
    CATCH_AND_HANDLE
}

int faiss_IndexRefineFlat_new(
        FaissIndexRefineFlat** p_index,
        FaissIndex* base_index) {
    try {
        IndexRefineFlat* index = new IndexRefineFlat(
                reinterpret_cast<faiss::Index*>(base_index));
        *p_index = reinterpret_cast<FaissIndexRefineFlat*>(index);
        return 0;
    }
    CATCH_AND_HANDLE
}

DEFINE_DESTRUCTOR(IndexRefineFlat)
DEFINE_INDEX_DOWNCAST(IndexRefineFlat)

DEFINE_GETTER(IndexRefineFlat, int, own_fields)
DEFINE_SETTER(IndexRefineFlat, int, own_fields)

DEFINE_GETTER(IndexRefineFlat, float, k_factor)
DEFINE_SETTER(IndexRefineFlat, float, k_factor)

DEFINE_DESTRUCTOR(IndexFlat1D)
DEFINE_INDEX_DOWNCAST(IndexFlat1D)

int faiss_IndexFlat1D_new(FaissIndexFlat1D** p_index) {
    try {
        IndexFlat1D* index = new IndexFlat1D();
        *p_index = reinterpret_cast<FaissIndexFlat1D*>(index);
        return 0;
    }
    CATCH_AND_HANDLE
}

int faiss_IndexFlat1D_new_with(
        FaissIndexFlat1D** p_index,
        int continuous_update) {
    try {
        IndexFlat1D* index =
                new IndexFlat1D(static_cast<bool>(continuous_update));
        *p_index = reinterpret_cast<FaissIndexFlat1D*>(index);
        return 0;
    }
    CATCH_AND_HANDLE
}

int faiss_IndexFlat1D_update_permutation(FaissIndexFlat1D* index) {
    try {
        reinterpret_cast<IndexFlat1D*>(index)->update_permutation();
        return 0;
    }
    CATCH_AND_HANDLE
}
}
