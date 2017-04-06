
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
#include "GpuIndicesOptions.h"
#include "../Clustering.h"

namespace faiss { struct IndexIVF; }

namespace faiss { namespace gpu {

class GpuIndexFlat;
class GpuResources;

class GpuIndexIVF : public GpuIndex {
 public:
  GpuIndexIVF(GpuResources* resources,
              int device,
              IndicesOptions indicesOptions,
              bool useFloat16CoarseQuantizer,
              int dims,
              faiss::MetricType metric,
              int nlist);

  GpuIndexIVF(GpuResources* resources,
              int device,
              IndicesOptions indicesOptions,
              int dims,
              faiss::MetricType metric,
              int nlist,
              GpuIndexFlat* quantizer);

  ~GpuIndexIVF() override;

 private:
  /// Shared initialization functions
  void init_();

 public:
  /// What indices storage options are we using?
  IndicesOptions getIndicesOptions() const;

  /// Is our coarse quantizer storing and performing math in float16?
  bool getUseFloat16CoarseQuantizer() const;

  /// Copy what we need from the CPU equivalent
  void copyFrom(const faiss::IndexIVF* index);

  /// Copy what we have to the CPU equivalent
  void copyTo(faiss::IndexIVF* index) const;

  /// Returns the number of inverted lists we're managing
  int getNumLists() const;

  /// Sets the number of list probes per query
  void setNumProbes(int nprobe);

  /// Returns our current number of list probes per query
  int getNumProbes() const;

  /// `x` can be resident on the CPU or any GPU; the proper copies are
  /// performed
  /// Forwards to add_with_ids; assigns IDs as needed
  /// FIXME: remove override for C++03 compatibility
  void add(Index::idx_t n, const float* x) override;

 protected:
  void trainQuantizer_(faiss::Index::idx_t n, const float* x);

 protected:
  /// How should indices be stored on the GPU?
  const IndicesOptions indicesOptions_;

  /// Do we want to use float16 storage and math in our coarse
  /// quantizer?
  const bool useFloat16CoarseQuantizer_;

  /// Number of inverted lists that we manage
  int nlist_;

  /// Number of inverted list probes per query
  int nprobe_;

  /// Ability to override default clustering parameters
  ClusteringParameters cp_;

  /// Quantizer for inverted lists
  GpuIndexFlat* quantizer_;

  /// Do we own the above quantizer instance?
  bool ownsQuantizer_;
};

} } // namespace
