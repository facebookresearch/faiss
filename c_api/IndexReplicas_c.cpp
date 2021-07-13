/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "IndexReplicas_c.h"
#include <faiss/IndexReplicas.h>
#include "macros_impl.h"

using faiss::Index;
using faiss::IndexReplicas;

DEFINE_DESTRUCTOR(IndexReplicas)

DEFINE_GETTER(IndexReplicas, int, own_fields)
DEFINE_SETTER(IndexReplicas, int, own_fields)

int faiss_IndexReplicas_new(FaissIndexReplicas** p_index, idx_t d) {
    try {
        auto out = new IndexReplicas(d);
        *p_index = reinterpret_cast<FaissIndexReplicas*>(out);
    }
    CATCH_AND_HANDLE
}

int faiss_IndexReplicas_new_with_options(
        FaissIndexReplicas** p_index,
        idx_t d,
        int threaded) {
    try {
        auto out = new IndexReplicas(d, static_cast<bool>(threaded));
        *p_index = reinterpret_cast<FaissIndexReplicas*>(out);
    }
    CATCH_AND_HANDLE
}

int faiss_IndexReplicas_add_replica(
        FaissIndexReplicas* index,
        FaissIndex* replica) {
    try {
        reinterpret_cast<IndexReplicas*>(index)->add_replica(
                reinterpret_cast<Index*>(replica));
    }
    CATCH_AND_HANDLE
}

int faiss_IndexReplicas_remove_replica(
        FaissIndexReplicas* index,
        FaissIndex* replica) {
    try {
        reinterpret_cast<IndexReplicas*>(index)->remove_replica(
                reinterpret_cast<Index*>(replica));
    }
    CATCH_AND_HANDLE
}

FaissIndex* faiss_IndexReplicas_at(FaissIndexReplicas* index, int i) {
    auto replica = reinterpret_cast<IndexReplicas*>(index)->at(i);
    return reinterpret_cast<FaissIndex*>(replica);
}
