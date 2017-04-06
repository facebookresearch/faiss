
/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "GpuIndex.h"

namespace faiss {

struct IndexFlat;
struct IndexFlatL2;
struct IndexFlatIP;

}

namespace faiss { namespace gpu {

struct FlatIndex;

struct GpuIndexFlatConfig {
  inline GpuIndexFlatConfig()
      : device(0),
        useFloat16(false),
        storeTransposed(false) {
  }

  int device;
  bool useFloat16;
  bool storeTransposed;
};

/// Wrapper around the GPU implementation that looks like
/// faiss::IndexFlat; copies over centroid data from a given
/// faiss::IndexFlat
class GpuIndexFlat : public GpuIndex {
 public:
  /// Construct from a pre-existing faiss::IndexFlat instance, copying
  /// data over to the given GPU
  GpuIndexFlat(GpuResources* resources,
               const faiss::IndexFlat* index,
               GpuIndexFlatConfig config = GpuIndexFlatConfig());

  /// Construct an empty instance that can be added to
  GpuIndexFlat(GpuResources* resources,
               int dims,
               faiss::MetricType metric,
               GpuIndexFlatConfig config = GpuIndexFlatConfig());

  virtual ~GpuIndexFlat();

  /// Set the minimum data size for searches (in MiB) for which we use
  /// CPU -> GPU paging
  void setMinPagingSize(size_t size);

  /// Returns the current minimum data size for paged searches
  size_t getMinPagingSize() const;

  /// Do we store vectors and perform math in float16?
  bool getUseFloat16() const;

  /// Initialize ourselves from the given CPU index; will overwrite
  /// all data in ourselves
  void copyFrom(const faiss::IndexFlat* index);

  /// Copy ourselves to the given CPU index; will overwrite all data
  /// in the index instance
  void copyTo(faiss::IndexFlat* index) const;

  /// Returns the number of vectors we contain
  size_t getNumVecs() const;

  /// Clears all vectors from this index
  virtual void reset();

  /// This index is not trained, so this does nothing
  virtual void train(Index::idx_t n, const float* x);

  /// `x`, `distances` and `labels` can be resident on the CPU or any
  /// GPU; copies are performed as needed
  /// We have our own implementation here which handles CPU async
  /// copies; searchImpl_ is not called
  /// FIXME: move paged impl into GpuIndex
  virtual void search(faiss::Index::idx_t n,
                      const float* x,
                      faiss::Index::idx_t k,
                      float* distances,
                      faiss::Index::idx_t* labels) const;

  /// Reconstruction methods; prefer the batch reconstruct as it will
  /// be more efficient
  virtual void reconstruct(faiss::Index::idx_t key, float* out) const;

  /// Batch reconstruction method
  virtual void reconstruct_n(faiss::Index::idx_t i0,
                             faiss::Index::idx_t num,
                             float* out) const;

  virtual void set_typename();

  /// For internal access
  inline FlatIndex* getGpuData() { return data_; }

 protected:
  /// Called from GpuIndex for add
  virtual void addImpl_(faiss::Index::idx_t n,
                        const float* x,
                        const faiss::Index::idx_t* ids);

  /// Should not be called (we have our own implementation)
  virtual void searchImpl_(faiss::Index::idx_t n,
                           const float* x,
                           faiss::Index::idx_t k,
                           float* distances,
                           faiss::Index::idx_t* labels) const;

  /// Called from search when the input data is on the CPU;
  /// potentially allows for pinned memory usage
  void searchFromCpuPaged_(int n,
                           const float* x,
                           int k,
                           float* outDistancesData,
                           int* outIndicesData) const;

  void searchNonPaged_(int n,
                       const float* x,
                       int k,
                       float* outDistancesData,
                       int* outIndicesData) const;

 protected:
  /// Size above which we page copies from the CPU to GPU
  size_t minPagedSize_;

  const GpuIndexFlatConfig config_;

  /// Holds our GPU data containing the list of vectors
  FlatIndex* data_;
};

/// Wrapper around the GPU implementation that looks like
/// faiss::IndexFlatL2; copies over centroid data from a given
/// faiss::IndexFlat
class GpuIndexFlatL2 : public GpuIndexFlat {
 public:
  /// Construct from a pre-existing faiss::IndexFlatL2 instance, copying
  /// data over to the given GPU
  GpuIndexFlatL2(GpuResources* resources,
                 faiss::IndexFlatL2* index,
                 GpuIndexFlatConfig config = GpuIndexFlatConfig());

  /// Construct an empty instance that can be added to
  GpuIndexFlatL2(GpuResources* resources,
                 int dims,
                 GpuIndexFlatConfig config = GpuIndexFlatConfig());

  /// Initialize ourselves from the given CPU index; will overwrite
  /// all data in ourselves
  void copyFrom(faiss::IndexFlatL2* index);

  /// Copy ourselves to the given CPU index; will overwrite all data
  /// in the index instance
  void copyTo(faiss::IndexFlatL2* index);
};

/// Wrapper around the GPU implementation that looks like
/// faiss::IndexFlatIP; copies over centroid data from a given
/// faiss::IndexFlat
class GpuIndexFlatIP : public GpuIndexFlat {
 public:
  /// Construct from a pre-existing faiss::IndexFlatIP instance, copying
  /// data over to the given GPU
  GpuIndexFlatIP(GpuResources* resources,
                 faiss::IndexFlatIP* index,
                 GpuIndexFlatConfig config = GpuIndexFlatConfig());

  /// Construct an empty instance that can be added to
  GpuIndexFlatIP(GpuResources* resources,
                 int dims,
                 GpuIndexFlatConfig config = GpuIndexFlatConfig());

  /// Initialize ourselves from the given CPU index; will overwrite
  /// all data in ourselves
  void copyFrom(faiss::IndexFlatIP* index);

  /// Copy ourselves to the given CPU index; will overwrite all data
  /// in the index instance
  void copyTo(faiss::IndexFlatIP* index);
};

} } // namespace
