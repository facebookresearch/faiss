
/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

// Copyright 2004-present Facebook. All Rights Reserved
// -*- c++ -*-

#include "IndexNested.h"

#include <cstdlib>
#include <cstdio>

#include "IndexFlat.h"
#include "FaissAssert.h"

namespace faiss {

/* Construct a nested Index */
IndexIVFFlat * nested_ivf_raw (
        size_t d,
        int nlevels,
        size_t branch_factor,
        size_t nprobe,
        MetricType metric)
{
    Index * core_index = NULL;
    IndexIVFFlat * ivf_index = NULL;
    FAISS_ASSERT (nlevels > 1);

    switch (metric) {
        case METRIC_INNER_PRODUCT:  core_index = new IndexFlatIP (d); break;
        case METRIC_L2:             core_index = new IndexFlatL2 (d); break;
        default: fprintf (stderr, "Invalid metric type"); abort();
    }

    // Construct iteratively
    int nlist = branch_factor;
    for (int l = 1; l < nlevels; l++) {
        ivf_index = new IndexIVFFlat (core_index, d, nlist, metric);
        ivf_index->own_fields = true;
        ivf_index->nprobe = nprobe > nlist ? nlist : nprobe;

        // The index IVF becomes the core index of the next layer
        core_index = ivf_index;
        fprintf (stderr, "level %d -> nlist = %d\n", l, nlist);
        nlist *= branch_factor;
    }
    return ivf_index;
}


IndexIVFFlat * nested_ivf_raw (
        size_t d,
        int nlevels,
        size_t * nlists,
        size_t * nprobes,
        MetricType metric)
{
    Index * core_index = nullptr;
    IndexIVFFlat * ivf_index = nullptr;
    FAISS_ASSERT (nlevels > 1);

    switch (metric) {
        case METRIC_INNER_PRODUCT:  core_index = new IndexFlatIP (d); break;
        case METRIC_L2:             core_index = new IndexFlatL2 (d); break;
        default: fprintf (stderr, "Invalid metric type"); abort();
    }

    // Construct iteratively
    for (int l = 1; l < nlevels; l++) {
        size_t nlist = nlists[l-1];
        ivf_index = new IndexIVFFlat (core_index, d, nlist, metric);
        ivf_index->own_fields = true;
        ivf_index->nprobe = nprobes[l-1] > nlist ? nlist : nprobes[l-1];
        ivf_index->maintain_direct_map = true;

        // The index IVF becomes the core index of the next layer
        core_index = ivf_index;
        fprintf (stdout, "level %d -> nlist = %ld\n", l, nlist);
    }
    return ivf_index;
}


IndexIVFPQ * nested_ivf_pq (
        size_t d,
        int nlevels,
        size_t * nlists,
        size_t * nprobes,
        size_t M)
{
    Index * core_index = new IndexFlatL2 (d);
    IndexIVFFlat * ivf_index = nullptr;
    FAISS_ASSERT (nlevels > 1);

    // Construct iteratively
    for (int l = 1; l < nlevels-1; l++) {
        size_t nlist = nlists[l-1];
        ivf_index = new IndexIVFFlat (core_index, d, nlist, METRIC_L2);
        ivf_index->own_fields = true;
        ivf_index->nprobe = nprobes[l-1] > nlist ? nlist : nprobes[l-1];
        ivf_index->maintain_direct_map = true;

        // The index IVF becomes the core index of the next layer
        core_index = ivf_index;
        fprintf (stdout, "level %d -> nlist = %ld\n", l, nlist);
    }
    size_t nclust = nlists[nlevels-2];
    IndexIVFPQ * ivf_pq = new faiss::IndexIVFPQ (core_index, d, nclust, M, 8);
    ivf_pq->own_fields = true;
    ivf_pq->use_precomputed_table = true;
    ivf_pq->nprobe = nprobes[nlevels-2] > nclust ? nclust : nprobes[nlevels-2];
    fprintf (stdout, "level %d (PQ) -> nlist = %ld\n", nlevels-1, nclust);

    return ivf_pq;
}


IndexIVFPQR * nested_ivf_pqr (
        size_t d,
        int nlevels,
        size_t * nlists,
        size_t * nprobes,
        size_t M,
        size_t M_refine)
{
    Index * core_index = new IndexFlatL2 (d);
    IndexIVFFlat * ivf_index = nullptr;
    FAISS_ASSERT (nlevels > 1);

    // Construct iteratively
    for (int l = 1; l < nlevels-1; l++) {
        size_t nlist = nlists[l-1];
        ivf_index = new IndexIVFFlat (core_index, d, nlist, METRIC_L2);
        ivf_index->own_fields = true;
        ivf_index->nprobe = nprobes[l-1] > nlist ? nlist : nprobes[l-1];
        ivf_index->maintain_direct_map = true;

        // The index IVF becomes the core index of the next layer
        core_index = ivf_index;
        fprintf (stdout, "level %d -> nlist = %ld\n", l, nlist);
    }
    size_t nclust = nlists[nlevels-2];
    IndexIVFPQR * ivf_pqr = new faiss::IndexIVFPQR (core_index, d, nclust, M, 8,
                                                    M_refine, 8);
    ivf_pqr->own_fields = true;
    ivf_pqr->use_precomputed_table = true;
    ivf_pqr->nprobe = nprobes[nlevels-2] > nclust ? nclust : nprobes[nlevels-2];
    fprintf (stdout, "level %d (PQ) -> nlist = %ld\n", nlevels-1, nclust);

    return ivf_pqr;
}


} // namespace faiss
