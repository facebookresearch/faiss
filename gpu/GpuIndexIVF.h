/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "GpuIndex.h"
#include "GpuIndexFlat.h"
#include "GpuIndicesOptions.h"
#include "../Clustering.h"

namespace faiss { struct IndexIVF; }

namespace faiss { namespace gpu {

class GpuIndexFlat;
class GpuResources;

struct GpuIndexIVFConfig : public GpuIndexConfig {
  inline GpuIndexIVFConfig()
      : indicesOptions(INDICES_64_BIT) {
  }

  /// Index storage options for the GPU
  IndicesOptions indicesOptions;

  /// Configuration for the coarse quantizer object
  GpuIndexFlatConfig flatConfig;
};

class GpuIndexIVF : public GpuIndex {
 public:
  GpuIndexIVF(GpuResources* resources,
              int dims,
              faiss::MetricType metric,
              int nlist,
              GpuIndexIVFConfig config = GpuIndexIVFConfig());

  ~GpuIndexIVF() override;

 private:
  /// Shared initialization functions
  void init_();

 public:
  /// Copy what we need from the CPU equivalent
  void copyFrom(const faiss::IndexIVF* index);

  /// Copy what we have to the CPU equivalent
  void copyTo(faiss::IndexIVF* index) const;

  /// Returns the number of inverted lists we're managing
  int getNumLists() const;

  /// Return the quantizer we're using
  GpuIndexFlat* getQuantizer();

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

 public:
  /// Exposed as IndexIVF does to allow overriding clustering
  /// parameters
  ClusteringParameters cp;

 protected:
  GpuIndexIVFConfig ivfConfig_;

  /// Number of inverted lists that we manage
  int nlist_;

  /// Number of inverted list probes per query
  int nprobe_;

  /// Quantizer for inverted lists
  GpuIndexFlat* quantizer_;
};

} } // namespace
