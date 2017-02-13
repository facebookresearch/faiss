
/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

// Copyright 2004-present Facebook. All Rights Reserved.
// -*- c++ -*-

#ifndef INDEX_NESTED_H
#define INDEX_NESTED_H

#include "IndexIVF.h"
#include "IndexIVFPQ.h"

namespace faiss {


/** Nested Index construction: construct a hierarchy of index.
 * The function returns an object IndexIVFFlat *
 *
 * @param d              dimensionality of the indexed vectors
 * @param nlevels        number of levels in the hierarchy (1 = Flat, no IVF)
 * @param branch_factor  ratio between #list on levels l+1 and l
 * @param nprobe         number of lists visited at each level during the search
 * @param metric         METRIC_INNER_PRODUCT | METRIC_L2
 */
IndexIVFFlat * nested_ivf_raw (
        size_t d,
        int nlevels,
        size_t branch_factor,
        size_t nprobe,
        MetricType metric = METRIC_INNER_PRODUCT);

/** Nested Index construction with specification of number of lists.
 * The function returns an object IndexIVFFlat *
 *
 * @param d           dimensionality of the indexed vectors
 * @param nlevels     number of levels in the hierarchy (1 = Flat, no IVF)
 * @param nlists      (size: nlevels-1) #lists for IVF indexes (all but flat)
 * @param nprobes     (size: nlevels-1) #probes at each level
 * @param metric      METRIC_INNER_PRODUCT | METRIC_L2
 */
 IndexIVFFlat * nested_ivf_raw (
        size_t d,
        int nlevels,
        size_t * nlists,
        size_t * nprobes,
        MetricType metric = METRIC_INNER_PRODUCT);

IndexIVFPQ * nested_ivf_pq (
        size_t d,
        int nlevels,
        size_t * nlists,
        size_t * nprobes,
        size_t M);

IndexIVFPQR * nested_ivf_pqr (
        size_t d,
        int nlevels,
        size_t * nlists,
        size_t * nprobes,
        size_t M,
        size_t M_refine);

} // namespace faiss

#endif
