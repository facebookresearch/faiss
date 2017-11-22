/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved

#include "IndexFlat.h"
#include "FaissAssert.h"

namespace faiss {


void Index::range_search (idx_t , const float *, float,
                          RangeSearchResult *) const
{
  FAISS_THROW_MSG ("range search not implemented");
}

void Index::assign (idx_t n, const float * x, idx_t * labels, idx_t k)
{
  float * distances = new float[n * k];
  ScopeDeleter<float> del(distances);
  search (n, x, k, distances, labels);
}

void Index::add_with_ids(
    idx_t /*n*/,
    const float* /*x*/,
    const long* /*xids*/) {
  FAISS_THROW_MSG ("add_with_ids not implemented for this type of index");
}

long Index::remove_ids(const IDSelector& /*sel*/) {
  FAISS_THROW_MSG ("remove_ids not implemented for this type of index");
  return -1;
}


void Index::reconstruct (idx_t, float * ) const {
  FAISS_THROW_MSG ("reconstruct not implemented for this type of index");
}


void Index::reconstruct_n (idx_t i0, idx_t ni, float *recons) const {
  for (idx_t i = 0; i < ni; i++) {
    reconstruct (i0 + i, recons + i * d);
  }
}



void Index::compute_residual (const float * x,
                              float * residual, idx_t key) const {
  reconstruct (key, residual);
  for (size_t i = 0; i < d; i++)
    residual[i] = x[i] - residual[i];
}


void Index::display () const {
  printf ("Index: %s  -> %ld elements\n", typeid (*this).name(), ntotal);
}

}
