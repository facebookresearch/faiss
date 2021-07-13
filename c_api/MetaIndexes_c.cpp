/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.
// -*- c++ -*-

#include "MetaIndexes_c.h"
#include <faiss/MetaIndexes.h>
#include "macros_impl.h"

using faiss::Index;
using faiss::IndexIDMap;
using faiss::IndexIDMap2;

DEFINE_GETTER(IndexIDMap, int, own_fields)
DEFINE_SETTER(IndexIDMap, int, own_fields)

DEFINE_INDEX_DOWNCAST(IndexIDMap)

DEFINE_GETTER(IndexIDMap2, int, own_fields)
DEFINE_SETTER(IndexIDMap2, int, own_fields)

DEFINE_INDEX_DOWNCAST(IndexIDMap2)

int faiss_IndexIDMap_new(FaissIndexIDMap** p_index, FaissIndex* index) {
    try {
        auto out = new IndexIDMap(reinterpret_cast<Index*>(index));
        *p_index = reinterpret_cast<FaissIndexIDMap*>(out);
    }
    CATCH_AND_HANDLE
}

void faiss_IndexIDMap_id_map(
        FaissIndexIDMap* index,
        idx_t** p_id_map,
        size_t* p_size) {
    auto idx = reinterpret_cast<IndexIDMap*>(index);
    if (p_id_map)
        *p_id_map = idx->id_map.data();
    if (p_size)
        *p_size = idx->id_map.size();
}

FaissIndex* faiss_IndexIDMap_sub_index(FaissIndexIDMap* index) {
    auto idx = reinterpret_cast<IndexIDMap*>(index);
    return (FaissIndex*)reinterpret_cast<Index*>(idx->index);
}

int faiss_IndexIDMap2_new(FaissIndexIDMap2** p_index, FaissIndex* index) {
    try {
        auto out = new IndexIDMap2(reinterpret_cast<Index*>(index));
        *p_index = reinterpret_cast<FaissIndexIDMap2*>(out);
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIDMap2_construct_rev_map(FaissIndexIDMap2* index) {
    try {
        reinterpret_cast<IndexIDMap2*>(index)->construct_rev_map();
    }
    CATCH_AND_HANDLE
}

void faiss_IndexIDMap2_id_map(
        FaissIndexIDMap2* index,
        idx_t** p_id_map,
        size_t* p_size) {
    auto idx = reinterpret_cast<IndexIDMap2*>(index);
    if (p_id_map)
        *p_id_map = idx->id_map.data();
    if (p_size)
        *p_size = idx->id_map.size();
}

FaissIndex* faiss_IndexIDMap2_sub_index(FaissIndexIDMap2* index) {
    auto idx = reinterpret_cast<IndexIDMap2*>(index);
    return (FaissIndex*)reinterpret_cast<Index*>(idx->index);
}
