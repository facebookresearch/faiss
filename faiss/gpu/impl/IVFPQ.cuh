/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


#pragma once

#include <faiss/MetricType.h>
#include <faiss/gpu/impl/IVFBase.cuh>
#include <faiss/gpu/utils/Float16.cuh>

namespace faiss { namespace gpu {

/// Implementing class for IVFPQ on the GPU
class IVFPQ : public IVFBase {
 public:
  IVFPQ(GpuResources* resources,
        faiss::MetricType metric,
        float metricArg,
        /// We do not own this reference
        FlatIndex* quantizer,
        int numSubQuantizers,
        int bitsPerSubQuantizer,
        bool useFloat16LookupTables,
        bool useMMCodeDistance,
        bool alternativeLayout,
        float* pqCentroidData,
        IndicesOptions indicesOptions,
        MemorySpace space);

  /// Returns true if we support PQ in this size
  static bool isSupportedPQCodeLength(int size);

  ~IVFPQ() override;

  /// Enable or disable pre-computed codes
  void setPrecomputedCodes(bool enable);

  /// Find the approximate k nearest neigbors for `queries` against
  /// our database
  void query(Tensor<float, 2, true>& queries,
             int nprobe,
             int k,
             Tensor<float, 2, true>& outDistances,
             Tensor<long, 2, true>& outIndices);

  /// Return the list codes of a particular list back to the CPU
  std::vector<unsigned char> getListCodes(int listId) const;

  /// Returns our set of sub-quantizers of the form
  /// (sub q)(code id)(sub dim)
  Tensor<float, 3, true> getPQCentroids();

 protected:
  /// Returns the encoding size for a PQ-encoded IVF list
  size_t getGpuVectorsEncodingSize_(int numVecs) const override;
  size_t getCpuVectorsEncodingSize_(int numVecs) const override;

  /// Translate to our preferred GPU encoding
  std::vector<unsigned char> translateCodesToGpu_(
    std::vector<unsigned char> codes,
    size_t numVecs) const override;

  /// Translate from our preferred GPU encoding
  std::vector<unsigned char> translateCodesFromGpu_(
    std::vector<unsigned char> codes,
    size_t numVecs) const override;

  /// Encode the vectors that we're adding and append to our IVF lists
  void appendVectors_(Tensor<float, 2, true>& vecs,
                      Tensor<long, 1, true>& indices,
                      Tensor<int, 1, true>& listIds,
                      Tensor<int, 1, true>& listOffset,
                      cudaStream_t stream) override;

  /// Sets the current product quantizer centroids; the data can be
  /// resident on either the host or the device. It will be transposed
  /// into our preferred data layout
  /// Data must be a row-major, 3-d array of size
  /// (numSubQuantizers, numSubQuantizerCodes, dim / numSubQuantizers)
  void setPQCentroids_(float* data);

  /// Calculate precomputed residual distance information
  void precomputeCodes_();

  /// Calculate precomputed residual distance information (for different coarse
  /// centroid type)
  template <typename CentroidT>
  void precomputeCodesT_();

  /// Runs kernels for scanning inverted lists with precomputed codes
  void runPQPrecomputedCodes_(Tensor<float, 2, true>& queries,
                              DeviceTensor<float, 2, true>& coarseDistances,
                              DeviceTensor<int, 2, true>& coarseIndices,
                              int k,
                              Tensor<float, 2, true>& outDistances,
                              Tensor<long, 2, true>& outIndices);

  /// Runs kernels for scanning inverted lists without precomputed codes
  void runPQNoPrecomputedCodes_(Tensor<float, 2, true>& queries,
                                DeviceTensor<float, 2, true>& coarseDistances,
                                DeviceTensor<int, 2, true>& coarseIndices,
                                int k,
                                Tensor<float, 2, true>& outDistances,
                                Tensor<long, 2, true>& outIndices);

  /// Runs kernels for scanning inverted lists without precomputed codes (for
  /// different coarse centroid type)
  template <typename CentroidT>
  void runPQNoPrecomputedCodesT_(Tensor<float, 2, true>& queries,
                                 DeviceTensor<float, 2, true>& coarseDistances,
                                 DeviceTensor<int, 2, true>& coarseIndices,
                                 int k,
                                 Tensor<float, 2, true>& outDistances,
                                 Tensor<long, 2, true>& outIndices);

 private:
  /// Number of sub-quantizers per vector
  const int numSubQuantizers_;

  /// Number of bits per sub-quantizer
  const int bitsPerSubQuantizer_;

  /// Number of per sub-quantizer codes (2^bits)
  const int numSubQuantizerCodes_;

  /// Number of dimensions per each sub-quantizer
  const int dimPerSubQuantizer_;

  /// Do we maintain precomputed terms and lookup tables in float16
  /// form?
  const bool useFloat16LookupTables_;

  /// For usage without precomputed codes, do we force usage of the
  /// general-purpose MM code distance computation? This is for testing
  /// purposes.
  const bool useMMCodeDistance_;

  /// The default memory layout is [vector][PQ component]:
  /// (v0 d0) (v0 d1) ... (v0 dD-1) (v1 d0) (v1 d1) ...
  ///
  /// An alternative memory layout (layoutBy32) is
  /// [vector / 32][PQ component][vector % 32] with padding:
  /// (v0 d0) (v1 d0) ... (v31 d0) (v0 d1) (v1 d1) ... (v31 dD-1) (v32 d0) (v33
  /// d0) ...
  /// so the list length is always a multiple of numSubQuantizers * 32
  const bool alternativeLayout_;

  /// On the GPU, we prefer different PQ centroid data layouts for
  /// different purposes.
  ///
  /// (sub q)(sub dim)(code id)
  DeviceTensor<float, 3, true> pqCentroidsInnermostCode_;

  /// (sub q)(code id)(sub dim)
  DeviceTensor<float, 3, true> pqCentroidsMiddleCode_;

  /// Are precomputed codes enabled? (additional factoring and
  /// precomputation of the residual distance, to reduce query-time work)
  bool precomputedCodes_;

  /// Precomputed term 2 in float form
  /// (centroid id)(sub q)(code id)
  DeviceTensor<float, 3, true> precomputedCode_;

  /// Precomputed term 2 in half form
  DeviceTensor<half, 3, true> precomputedCodeHalf_;
};

} } // namespace
