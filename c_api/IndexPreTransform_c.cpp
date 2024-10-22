/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include "IndexPreTransform_c.h"
#include <faiss/IndexPreTransform.h>
#include <faiss/VectorTransform.h>
#include "macros_impl.h"

using faiss::Index;
using faiss::IndexPreTransform;
using faiss::VectorTransform;

extern "C" {

DEFINE_DESTRUCTOR(IndexPreTransform)
DEFINE_INDEX_DOWNCAST(IndexPreTransform)

DEFINE_GETTER_PERMISSIVE(IndexPreTransform, FaissIndex*, index)

DEFINE_GETTER(IndexPreTransform, int, own_fields)
DEFINE_SETTER(IndexPreTransform, int, own_fields)

int faiss_IndexPreTransform_new(FaissIndexPreTransform** p_index) {
    try {
        *p_index = reinterpret_cast<FaissIndexPreTransform*>(
                new IndexPreTransform());
    }
    CATCH_AND_HANDLE
}

int faiss_IndexPreTransform_new_with(
        FaissIndexPreTransform** p_index,
        FaissIndex* index) {
    try {
        auto ind = reinterpret_cast<Index*>(index);
        *p_index = reinterpret_cast<FaissIndexPreTransform*>(
                new IndexPreTransform(ind));
    }
    CATCH_AND_HANDLE
}

int faiss_IndexPreTransform_new_with_transform(
        FaissIndexPreTransform** p_index,
        FaissVectorTransform* ltrans,
        FaissIndex* index) {
    try {
        auto lt = reinterpret_cast<VectorTransform*>(ltrans);
        auto ind = reinterpret_cast<Index*>(index);
        *p_index = reinterpret_cast<FaissIndexPreTransform*>(
                new IndexPreTransform(lt, ind));
    }
    CATCH_AND_HANDLE
}

int faiss_IndexPreTransform_prepend_transform(
        FaissIndexPreTransform* index,
        FaissVectorTransform* ltrans) {
    try {
        auto lt = reinterpret_cast<VectorTransform*>(ltrans);
        reinterpret_cast<IndexPreTransform*>(index)->prepend_transform(lt);
    }
    CATCH_AND_HANDLE
}
}
