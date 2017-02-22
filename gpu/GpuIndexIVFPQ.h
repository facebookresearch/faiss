
/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "GpuIndexIVF.h"
#include <vector>

namespace faiss { struct IndexIVFPQ; }

namespace faiss { namespace gpu {

class GpuIndexFlat;
class IVFPQ;

/// IVFPQ index for the GPU
class GpuIndexIVFPQ : public GpuIndexIVF {
 public:
  /// Construct from a pre-existing faiss::IndexIVFPQ instance, copying
  /// data over to the given GPU, if the input index is trained.
  GpuIndexIVFPQ(GpuResources* resources,
                int device,
                IndicesOptions indicesOptions,
                bool useFloat16LookupTables,
                const faiss::IndexIVFPQ* index);

  /// Construct an empty index
  GpuIndexIVFPQ(GpuResources* resources,
                int device,
                int dims,
                int nlist,
                int subQuantizers,
                int bitsPerCode,
                bool usePrecomputed,
                IndicesOptions indicesOptions,
                bool useFloat16LookupTables,
                faiss::MetricType metric);

  ~GpuIndexIVFPQ() override;

  /// Reserve space on the GPU for the inverted lists for `num`
  /// vectors, assumed equally distributed among

  /// Initialize ourselves from the given CPU index; will overwrite
  /// all data in ourselves
  void copyFrom(const faiss::IndexIVFPQ* index);

  /// Copy ourselves to the given CPU index; will overwrite all data
  /// in the index instance
  void copyTo(faiss::IndexIVFPQ* index) const;

  /// Reserve GPU memory in our inverted lists for this number of vectors
  void reserveMemory(size_t numVecs);

  /// Enable or disable pre-computed codes
  void setPrecomputedCodes(bool enable);

  /// Are pre-computed codes enabled?
  bool getPrecomputedCodes() const;

  /// Are float16 residual distance lookup tables enabled?
  bool getFloat16LookupTables() const;

  /// Return the number of sub-quantizers we are using
  int getNumSubQuantizers() const;

  /// Return the number of bits per PQ code
  int getBitsPerCode() const;

  /// Return the number of centroids per PQ code (2^bits per code)
  int getCentroidsPerSubQuantizer() const;

  /// After adding vectors, one can call this to reclaim device memory
  /// to exactly the amount needed. Returns space reclaimed in bytes
  size_t reclaimMemory();

  /// Clears out all inverted lists, but retains the coarse and
  /// product centroid information
  void reset() override;

  void train(Index::idx_t n, const float* x) override;

  /// `x` and `xids` can be resident on the CPU or any GPU; the proper
  /// copies are performed
  void add_with_ids(Index::idx_t n,
                    const float* x,
                    const Index::idx_t* xids) override;

  /// `x`, `distances` and `labels` can be resident on the CPU or any
  /// GPU; copies are performed as needed
  void search(faiss::Index::idx_t n,
              const float* x,
              faiss::Index::idx_t k,
              float* distances,
              faiss::Index::idx_t* labels) const override;

  void set_typename() override;

  /// For debugging purposes, return the list length of a particular
  /// list
  int getListLength(int listId) const;

  /// For debugging purposes, return the list codes of a particular
  /// list
  std::vector<unsigned char> getListCodes(int listId) const;

  /// For debugging purposes, return the list indices of a particular
  /// list
  std::vector<long> getListIndices(int listId) const;

 private:
  void assertSettings_() const;

  void trainResidualQuantizer_(Index::idx_t n, const float* x);

 private:
  /// Do we use float16 residual distance lookup tables for query?
  const bool useFloat16LookupTables_;

  /// Number of sub-quantizers per encoded vector
  int subQuantizers_;

  /// Bits per sub-quantizer code
  int bitsPerCode_;

  /// Should we or should we not use precomputed codes?
  bool usePrecomputed_;

  /// Desired inverted list memory reservation
  size_t reserveMemoryVecs_;

  /// The product quantizer instance that we own; contains the
  /// inverted lists
  IVFPQ* index_;
};

} } // namespace
