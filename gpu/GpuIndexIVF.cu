
/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#include "GpuIndexIVF.h"
#include "../FaissAssert.h"
#include "../IndexFlat.h"
#include "../IndexIVF.h"
#include "GpuIndexFlat.h"
#include "utils/DeviceUtils.h"
#include "utils/Float16.cuh"

namespace faiss { namespace gpu {

GpuIndexIVF::GpuIndexIVF(GpuResources* resources,
                         int device,
                         IndicesOptions indicesOptions,
                         bool useFloat16CoarseQuantizer,
                         int dims,
                         faiss::MetricType metric,
                         int nlist) :
    GpuIndex(resources, device, dims, metric),
    indicesOptions_(indicesOptions),
    useFloat16CoarseQuantizer_(useFloat16CoarseQuantizer),
    nlist_(nlist),
    nprobe_(1),
    quantizer_(nullptr),
    ownsQuantizer_(true) {
#ifndef FAISS_USE_FLOAT16
  FAISS_ASSERT(!useFloat16CoarseQuantizer_,
               "float16 unsupported; need CUDA SDK >= 7.5");
#endif

  init_();
}

GpuIndexIVF::GpuIndexIVF(GpuResources* resources,
                         int device,
                         IndicesOptions indicesOptions,
                         int dims,
                         faiss::MetricType metric,
                         int nlist,
                         GpuIndexFlat* quantizer) :
    GpuIndex(resources, device, dims, metric),
    indicesOptions_(indicesOptions),
    useFloat16CoarseQuantizer_(quantizer->getUseFloat16()),
    nlist_(nlist),
    nprobe_(1),
    quantizer_(quantizer),
    ownsQuantizer_(false) {
#ifndef FAISS_USE_FLOAT16
  FAISS_ASSERT(!useFloat16CoarseQuantizer_,
               "float16 unsupported; need CUDA SDK >= 7.5");
#endif

  FAISS_ASSERT(quantizer_->d == this->d);
  FAISS_ASSERT(quantizer_->metric_type == this->metric_type);

  init_();
}

void
GpuIndexIVF::init_() {
  FAISS_ASSERT(nlist_ > 0);

  // Spherical by default if the metric is inner_product
  if (this->metric_type == faiss::METRIC_INNER_PRODUCT) {
    cp_.spherical = true;
  }

  // here we set a low # iterations because this is typically used
  // for large clusterings
  cp_.niter = 10;
  cp_.verbose = this->verbose;

  if (!quantizer_) {
    // Construct an empty quantizer
    if (this->metric_type == faiss::METRIC_L2) {
      // FIXME: 2 different float16 options?
      quantizer_ = new GpuIndexFlatL2(resources_, device_, this->d,
                                      useFloat16CoarseQuantizer_);
    } else if (this->metric_type == faiss::METRIC_INNER_PRODUCT) {
      // FIXME: 2 different float16 options?
      quantizer_ = new GpuIndexFlatIP(resources_, device_, this->d,
                                      useFloat16CoarseQuantizer_);
    } else {
      // unknown metric type
      FAISS_ASSERT(false);
    }
  }
}

GpuIndexIVF::~GpuIndexIVF() {
  if (ownsQuantizer_) {
    delete quantizer_;
  }
}

IndicesOptions
GpuIndexIVF::getIndicesOptions() const {
  return indicesOptions_;
}

bool
GpuIndexIVF::getUseFloat16CoarseQuantizer() const {
  return useFloat16CoarseQuantizer_;
}

void
GpuIndexIVF::copyFrom(const faiss::IndexIVF* index) {
  DeviceScope scope(device_);

  this->d = index->d;
  this->metric_type = index->metric_type;

  FAISS_ASSERT(index->nlist > 0);
  FAISS_ASSERT(index->nlist <=
               (faiss::Index::idx_t) std::numeric_limits<int>::max());
  nlist_ = index->nlist;
  nprobe_ = index->nprobe;

  // The metric type may have changed as well, so we might have to
  // change our quantizer
  delete quantizer_;
  quantizer_ = nullptr;

  if (index->metric_type == faiss::METRIC_L2) {
    // FIXME: 2 different float16 options?
    quantizer_ = new GpuIndexFlatL2(resources_, device_, this->d,
                                    useFloat16CoarseQuantizer_);
  } else if (index->metric_type == faiss::METRIC_INNER_PRODUCT) {
    // FIXME: 2 different float16 options?
    quantizer_ = new GpuIndexFlatIP(resources_, device_, this->d,
                                    useFloat16CoarseQuantizer_);
  } else {
    // unknown metric type
    FAISS_ASSERT(false);
  }

  if (!index->is_trained) {
    this->is_trained = false;
    this->ntotal = 0;
    return;
  }

  // Otherwise, we can populate ourselves from the other index
  this->is_trained = true;

  // Only use `int` on GPU
  FAISS_ASSERT(index->ntotal <=
               (faiss::Index::idx_t) std::numeric_limits<int>::max());
  this->ntotal = index->ntotal;

  // Since we're trained, the quantizer must have data
  FAISS_ASSERT(index->quantizer->ntotal > 0);

  if (index->metric_type == faiss::METRIC_L2) {
    auto q = dynamic_cast<faiss::IndexFlatL2*>(index->quantizer);
    FAISS_ASSERT(q);

    quantizer_->copyFrom(q);
  } else if (index->metric_type == faiss::METRIC_INNER_PRODUCT) {
    auto q = dynamic_cast<faiss::IndexFlatIP*>(index->quantizer);
    FAISS_ASSERT(q);

    quantizer_->copyFrom(q);
  } else {
    // unknown metric type
    FAISS_ASSERT(false);
  }
}

void
GpuIndexIVF::copyTo(faiss::IndexIVF* index) const {
  DeviceScope scope(device_);

  //
  // Index information
  //
  index->ntotal = this->ntotal;
  index->d = this->d;
  index->metric_type = this->metric_type;
  index->is_trained = this->is_trained;

  //
  // IndexIVF information
  //
  index->nlist = nlist_;
  index->nprobe = nprobe_;

  // Construct and copy the appropriate quantizer
  faiss::IndexFlat* q = nullptr;

  if (this->metric_type == faiss::METRIC_L2) {
    q = new faiss::IndexFlatL2(this->d);

  } else if (this->metric_type == faiss::METRIC_INNER_PRODUCT) {
    q = new faiss::IndexFlatIP(this->d);

  } else {
    // unknown metric type
    FAISS_ASSERT(false);
  }

  FAISS_ASSERT(quantizer_);
  quantizer_->copyTo(q);

  if (index->own_fields) {
    delete index->quantizer;
  }

  index->quantizer = q;
  index->quantizer_trains_alone = false;
  index->own_fields = true;
  index->cp = cp_;
  index->ids.clear();
  index->ids.resize(nlist_);
  index->maintain_direct_map = false;
  index->direct_map.clear();
}

int
GpuIndexIVF::getNumLists() const {
  return nlist_;
}

void
GpuIndexIVF::setNumProbes(int nprobe) {
  FAISS_ASSERT(nprobe > 0);
  nprobe_ = nprobe;
}

int
GpuIndexIVF::getNumProbes() const {
  return nprobe_;
}


void
GpuIndexIVF::add(Index::idx_t n, const float* x) {
  // FIXME: GPU-ize
  std::vector<Index::idx_t> ids(n);
  for (Index::idx_t i = 0; i < n; ++i) {
    ids[i] = this->ntotal + i;
  }

  add_with_ids(n, x, ids.data());
}

void
GpuIndexIVF::trainQuantizer_(faiss::Index::idx_t n, const float* x) {
  if (n == 0) {
    // nothing to do
    return;
  }

  if (quantizer_->is_trained && (quantizer_->ntotal == nlist_)) {
    if (this->verbose) {
      printf ("IVF quantizer does not need training.\n");
    }

    return;
  }

  if (this->verbose) {
    printf ("Training IVF quantizer on %ld vectors in %dD\n", n, d);
  }

  DeviceScope scope(device_);

  // leverage the CPU-side k-means code, which works for the GPU
  // flat index as well
  quantizer_->reset();
  Clustering clus(this->d, nlist_, cp_);
  clus.verbose = verbose;
  clus.train(n, x, *quantizer_);
  quantizer_->is_trained = true;

  FAISS_ASSERT(quantizer_->ntotal == nlist_);
}


} } // namespace
