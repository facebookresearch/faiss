/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved
// -*- c++ -*-

#ifndef INDEX_BINARY_FLAT_H
#define INDEX_BINARY_FLAT_H

#include <vector>

#include "IndexBinary.h"

namespace faiss {


/** Index that stores the full vectors and performs exhaustive search. */
struct IndexBinaryFlat : IndexBinary {
  /// database vectors, size ntotal * d / 8
  std::vector<uint8_t> xb;

  explicit IndexBinaryFlat(idx_t d, MetricType metric = METRIC_INNER_PRODUCT);

  void add(idx_t n, const uint8_t *x) override;

  void reset() override;

  void search(idx_t n, const uint8_t *x, idx_t k,
              float *distances, idx_t *labels) const override;

  void range_search(idx_t n, const uint8_t *x, float radius,
                    RangeSearchResult* result) const override;

  void reconstruct(idx_t key, uint8_t *recons) const override;

  /** Remove some ids. Note that because of the indexing structure,
   * the semantics of this operation are different from the usual ones:
   * the new ids are shifted. */
  long remove_ids(const IDSelector& sel) override;

  IndexBinaryFlat() {}
};


struct IndexBinaryFlatIP : IndexBinaryFlat {
  explicit IndexBinaryFlatIP(idx_t d)
    : IndexBinaryFlat(d, METRIC_INNER_PRODUCT) {}
  IndexBinaryFlatIP() {}
};


struct IndexBinaryFlatL2 : IndexBinaryFlat {
  explicit IndexBinaryFlatL2(idx_t d)
    : IndexBinaryFlat(d, METRIC_L2) {}
  IndexBinaryFlatL2() {}
};


}  // namespace faiss

#endif  // INDEX_BINARY_FLAT_H
