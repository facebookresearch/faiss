/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h> // idx_t
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/Tensor.cuh>

namespace faiss {
namespace gpu {

template <
        typename T,
        int Dim,
        bool InnerContig = false,
        typename IndexT = idx_t,
        template <typename U> class PtrTraits = traits::DefaultPtrTraits>
class DeviceTensor : public Tensor<T, Dim, InnerContig, IndexT, PtrTraits> {
   public:
    typedef IndexT IndexType;
    typedef typename PtrTraits<T>::PtrType DataPtrType;

    /// Default constructor
    __host__ DeviceTensor();

    /// Destructor
    __host__ ~DeviceTensor();

    /// Move constructor
    __host__ DeviceTensor(
            DeviceTensor<T, Dim, InnerContig, IndexT, PtrTraits>&& t);

    /// Move assignment
    __host__ DeviceTensor<T, Dim, InnerContig, IndexT, PtrTraits>& operator=(
            DeviceTensor<T, Dim, InnerContig, IndexT, PtrTraits>&& t);

    /// Constructs a tensor of the given size, allocating memory for it
    /// via temporary or other allocation.
    /// `stream` specifies the stream on which the memory will be used
    __host__ DeviceTensor(
            GpuResources* res,
            const AllocInfo& info,
            const IndexT sizes[Dim]);

    /// Constructs a tensor of the given size, allocating memory for it
    /// via temporary or other allocation.
    /// `stream` specifies the stream on which the memory will be used
    __host__ DeviceTensor(
            GpuResources* res,
            const AllocInfo& info,
            std::initializer_list<IndexT> sizes);

    /// Constructs a tensor of the given size and stride, referencing a
    /// memory region we do not own
    __host__ DeviceTensor(DataPtrType data, const IndexT sizes[Dim]);

    /// Constructs a tensor of the given size and stride, referencing a
    /// memory region we do not own
    __host__ DeviceTensor(
            DataPtrType data,
            std::initializer_list<IndexT> sizes);

    /// Constructs a tensor of the given size and stride, referencing a
    /// memory region we do not own
    __host__ DeviceTensor(
            DataPtrType data,
            const IndexT sizes[Dim],
            const IndexT strides[Dim]);

    /// Copies a tensor into ourselves, allocating memory for it.
    /// `stream` specifies the stream of the copy and thus the stream on which
    /// the memory will initially be used.
    __host__ DeviceTensor(
            GpuResources* res,
            const AllocInfo& info,
            Tensor<T, Dim, InnerContig, IndexT, PtrTraits>& t);

    /// Call to zero out memory
    __host__ DeviceTensor<T, Dim, InnerContig, IndexT, PtrTraits>& zero(
            cudaStream_t stream);

   private:
    /// If we own the memory (temporary or non-temporary memory reservation),
    /// this holds the memory and will release it when we are destroyed
    GpuMemoryReservation reservation_;
};

} // namespace gpu
} // namespace faiss

#include <faiss/gpu/utils/DeviceTensor-inl.cuh>
