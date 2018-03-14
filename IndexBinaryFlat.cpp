/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved

#include "IndexBinaryFlat.h"

#include <cstring>
#include "hamming.h"
#include "utils.h"
#include "Heap.h"
#include "FaissAssert.h"
#include "AuxIndexStructures.h"

namespace faiss {

IndexBinaryFlat::IndexBinaryFlat(idx_t d, MetricType metric)
    : IndexBinary(d, metric) {}

void IndexBinaryFlat::add(idx_t n, const uint8_t *x) {
  xb.insert(xb.end(), x, x + n * (d / 8));
  ntotal += n;
}

void IndexBinaryFlat::reset() {
  xb.clear();
  ntotal = 0;
}

void IndexBinaryFlat::search(idx_t n, const uint8_t *x, idx_t k,
                             float *distances, idx_t *labels) const {
  // We see the distances and labels as heaps.
  if (metric_type == METRIC_INNER_PRODUCT) {
    FAISS_THROW_MSG("search not implemented for inner product on binary index");
  } else if (metric_type == METRIC_L2) {
    int * idistances = new int [n * k];
    ScopeDeleter<int> del (idistances);

    int_maxheap_array_t res = {
      size_t(n), size_t(k), labels, idistances
    };

    hammings_knn(&res, x, xb.data(), ntotal, d / 8, /* ordered = */ true);

    // Convert distances to floats.
    for (int i = 0; i < k * n; i++)
      distances[i] = idistances[i];
  }
}

long IndexBinaryFlat::remove_ids(const IDSelector& sel) {
  idx_t j = 0;
  for (idx_t i = 0; i < ntotal; i++) {
    if (sel.is_member(i)) {
      // should be removed
    } else {
      if (i > j) {
        memmove(&xb[(d / 8) * j], &xb[(d / 8) * i], sizeof(xb[0]) * (d / 8));
      }
      j++;
    }
  }
  long nremove = ntotal - j;
  if (nremove > 0) {
    ntotal = j;
    xb.resize(ntotal * (d / 8));
  }
  return nremove;
}

void IndexBinaryFlat::reconstruct(idx_t key, uint8_t *recons) const {
  memcpy(recons, &(xb[(d / 8) * key]), sizeof(*recons) * (d / 8));
}


}  // namespace faiss
