/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include "IndexHNSW_c.h"
#include <faiss/impl/HNSW.h>
#include "macros_impl.h"

using faiss::SearchParametersHNSW;

/// SearchParametersHNSW definitions

DEFINE_DESTRUCTOR(SearchParametersHNSW)
DEFINE_SEARCH_PARAMETERS_DOWNCAST(SearchParametersHNSW)

int faiss_SearchParametersHNSW_new(FaissSearchParametersHNSW** p_sp) {
    try {
        SearchParametersHNSW* sp = new SearchParametersHNSW;
        *p_sp = reinterpret_cast<FaissSearchParametersHNSW*>(sp);
        return 0;
    }
    CATCH_AND_HANDLE
}

int faiss_SearchParametersHNSW_new_with(
        FaissSearchParametersHNSW** p_sp,
        FaissIDSelector* sel,
        int efSearch) {
    try {
        SearchParametersHNSW* sp = new SearchParametersHNSW;
        sp->sel = reinterpret_cast<faiss::IDSelector*>(sel);
        sp->efSearch = efSearch;
        *p_sp = reinterpret_cast<FaissSearchParametersHNSW*>(sp);
        return 0;
    }
    CATCH_AND_HANDLE
}

DEFINE_GETTER_PERMISSIVE(SearchParametersHNSW, const FaissIDSelector*, sel)

DEFINE_GETTER(SearchParametersHNSW, int, efSearch)
DEFINE_SETTER(SearchParametersHNSW, int, efSearch)

DEFINE_GETTER(SearchParametersHNSW, int, check_relative_distance)
DEFINE_SETTER(SearchParametersHNSW, int, check_relative_distance)

DEFINE_GETTER(SearchParametersHNSW, int, bounded_queue)
DEFINE_SETTER(SearchParametersHNSW, int, bounded_queue)
