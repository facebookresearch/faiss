
/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "Tensor.cuh"
#include "DeviceMemory.h"

namespace faiss { namespace gpu {

template <typename T,
          int Dim,
          bool Contig = false,
          typename IndexT = int,
          template <typename U> class PtrTraits = traits::DefaultPtrTraits>
class DeviceTensor : public Tensor<T, Dim, Contig, IndexT, PtrTraits> {
 public:
  typedef IndexT IndexType;
  typedef typename PtrTraits<T>::PtrType DataPtrType;

  /// Default constructor
  __host__ DeviceTensor();

  /// Destructor
  __host__ ~DeviceTensor();

  /// Move constructor
  __host__ DeviceTensor(DeviceTensor<T, Dim, Contig, IndexT, PtrTraits>&& t);

  /// Move assignment
  __host__ DeviceTensor<T, Dim, Contig, IndexT, PtrTraits>&
  operator=(DeviceTensor<T, Dim, Contig, IndexT, PtrTraits>&& t);

  /// Constructs a tensor of the given size, allocating memory for it
  /// locally
  __host__ DeviceTensor(const IndexT sizes[Dim]);
  __host__ DeviceTensor(std::initializer_list<IndexT> sizes);

  /// Constructs a tensor of the given size, reserving a temporary
  /// memory reservation via a memory manager.
  /// The memory reservation should be ordered with respect to the
  /// given stream.
  __host__ DeviceTensor(DeviceMemory& m,
                        const IndexT sizes[Dim],
                        cudaStream_t stream);
  __host__ DeviceTensor(DeviceMemory& m,
                        std::initializer_list<IndexT> sizes,
                        cudaStream_t stream);

  /// Constructs a tensor of the given size and stride, referencing a
  /// memory region we do not own
  __host__ DeviceTensor(DataPtrType data,
                        const IndexT sizes[Dim]);
  __host__ DeviceTensor(DataPtrType data,
                        std::initializer_list<IndexT> sizes);

  /// Constructs a tensor of the given size and stride, referencing a
  /// memory region we do not own
  __host__ DeviceTensor(DataPtrType data,
                        const IndexT sizes[Dim],
                        const IndexT strides[Dim]);

  /// Copies a tensor into ourselves, allocating memory for it locally
  __host__ DeviceTensor(Tensor<T, Dim, Contig, IndexT, PtrTraits>& t,
                        cudaStream_t stream);

  /// Copies a tensor into ourselves, reserving a temporary
  /// memory reservation via a memory manager.
  __host__ DeviceTensor(DeviceMemory& m,
                        Tensor<T, Dim, Contig, IndexT, PtrTraits>& t,
                        cudaStream_t stream);

  /// Call to zero out memory
  __host__ DeviceTensor<T, Dim, Contig, IndexT, PtrTraits>&
  zero(cudaStream_t stream);

 private:
  enum AllocState {
    /// This tensor itself owns the memory, which must be freed via
    /// cudaFree
    Owner,

    /// This tensor itself is not an owner of the memory; there is
    /// nothing to free
    NotOwner,

    /// This tensor has the memory via a temporary memory reservation
    Reservation
  };

  AllocState state_;
  DeviceMemoryReservation reservation_;
};

} } // namespace

#include "DeviceTensor-inl.cuh"
