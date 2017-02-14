
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

namespace faiss { struct IndexIVFFlat; }

namespace faiss { namespace gpu {

class IVFFlat;
class GpuIndexFlat;

/// Wrapper around the GPU implementation that looks like
/// faiss::IndexIVFFlat
class GpuIndexIVFFlat : public GpuIndexIVF {
 public:
  /// Constructs a new instance with an empty flat quantizer; the user
  /// provides the number of lists desired.
  GpuIndexIVFFlat(GpuResources* resources,
                  int device,
                  // Does the coarse quantizer use float16?
                  bool useFloat16CoarseQuantizer,
                  // Is our IVF storage of vectors in float16?
                  bool useFloat16IVFStorage,
                  int dims,
                  int nlist,
                  IndicesOptions indicesOptions,
                  faiss::MetricType metric);

  /// Call to initialize ourselves from a GpuIndexFlat instance. The
  /// quantizer must match the dimension parameters specified; if
  /// populated, it must also match the number of list elements
  /// available.
  /// The index must also be present on the same device as ourselves.
  /// We do not own this quantizer instance.
  GpuIndexIVFFlat(GpuResources* resources,
                  int device,
                  GpuIndexFlat* quantizer,
                  bool useFloat16,
                  int dims,
                  int nlist,
                  IndicesOptions indicesOptions,
                  faiss::MetricType metric);

  ~GpuIndexIVFFlat() override;

  /// Reserve GPU memory in our inverted lists for this number of vectors
  void reserveMemory(size_t numVecs);

  /// Initialize ourselves from the given CPU index; will overwrite
  /// all data in ourselves
  void copyFrom(const faiss::IndexIVFFlat* index);

  /// Copy ourselves to the given CPU index; will overwrite all data
  /// in the index instance
  void copyTo(faiss::IndexIVFFlat* index) const;

  /// After adding vectors, one can call this to reclaim device memory
  /// to exactly the amount needed. Returns space reclaimed in bytes
  size_t reclaimMemory();

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

 private:
  /// Is float16 encoding enabled for our IVF data?
  bool useFloat16IVFStorage_;

  /// Desired inverted list memory reservation
  size_t reserveMemoryVecs_;

  /// Instance that we own; contains the inverted list
  IVFFlat* index_;
};

} } // namespace
