/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <utility> // std::move

namespace faiss {
namespace gpu {

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT,
        template <typename U>
        class PtrTraits>
__host__ DeviceTensor<T, Dim, InnerContig, IndexT, PtrTraits>::DeviceTensor()
        : Tensor<T, Dim, InnerContig, IndexT, PtrTraits>() {}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT,
        template <typename U>
        class PtrTraits>
__host__ DeviceTensor<T, Dim, InnerContig, IndexT, PtrTraits>::DeviceTensor(
        DeviceTensor<T, Dim, InnerContig, IndexT, PtrTraits>&& t)
        : Tensor<T, Dim, InnerContig, IndexT, PtrTraits>() {
    this->operator=(std::move(t));
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT,
        template <typename U>
        class PtrTraits>
__host__ DeviceTensor<T, Dim, InnerContig, IndexT, PtrTraits>& DeviceTensor<
        T,
        Dim,
        InnerContig,
        IndexT,
        PtrTraits>::
operator=(DeviceTensor<T, Dim, InnerContig, IndexT, PtrTraits>&& t) {
    this->Tensor<T, Dim, InnerContig, IndexT, PtrTraits>::operator=(
            std::move(t));

    this->reservation_ = std::move(t.reservation_);
    return *this;
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT,
        template <typename U>
        class PtrTraits>
__host__ DeviceTensor<T, Dim, InnerContig, IndexT, PtrTraits>::~DeviceTensor() {
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT,
        template <typename U>
        class PtrTraits>
__host__ DeviceTensor<T, Dim, InnerContig, IndexT, PtrTraits>::DeviceTensor(
        GpuResources* res,
        const AllocInfo& info,
        const IndexT sizes[Dim])
        : Tensor<T, Dim, InnerContig, IndexT, PtrTraits>(nullptr, sizes) {
    this->reservation_ = std::move(
            res->allocMemoryHandle(AllocRequest(info, this->getSizeInBytes())));
    this->data_ = (T*)reservation_.get();

    FAISS_ASSERT(this->data_ || (this->getSizeInBytes() == 0));
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT,
        template <typename U>
        class PtrTraits>
__host__ DeviceTensor<T, Dim, InnerContig, IndexT, PtrTraits>::DeviceTensor(
        GpuResources* res,
        const AllocInfo& info,
        std::initializer_list<IndexT> sizes)
        : Tensor<T, Dim, InnerContig, IndexT, PtrTraits>(nullptr, sizes) {
    this->reservation_ = std::move(
            res->allocMemoryHandle(AllocRequest(info, this->getSizeInBytes())));
    this->data_ = (T*)reservation_.get();

    FAISS_ASSERT(this->data_ || (this->getSizeInBytes() == 0));
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT,
        template <typename U>
        class PtrTraits>
__host__ DeviceTensor<T, Dim, InnerContig, IndexT, PtrTraits>::DeviceTensor(
        DataPtrType data,
        const IndexT sizes[Dim])
        : Tensor<T, Dim, InnerContig, IndexT, PtrTraits>(data, sizes) {}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT,
        template <typename U>
        class PtrTraits>
__host__ DeviceTensor<T, Dim, InnerContig, IndexT, PtrTraits>::DeviceTensor(
        DataPtrType data,
        std::initializer_list<IndexT> sizes)
        : Tensor<T, Dim, InnerContig, IndexT, PtrTraits>(data, sizes) {}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT,
        template <typename U>
        class PtrTraits>
__host__ DeviceTensor<T, Dim, InnerContig, IndexT, PtrTraits>::DeviceTensor(
        DataPtrType data,
        const IndexT sizes[Dim],
        const IndexT strides[Dim])
        : Tensor<T, Dim, InnerContig, IndexT, PtrTraits>(data, sizes, strides) {
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT,
        template <typename U>
        class PtrTraits>
__host__ DeviceTensor<T, Dim, InnerContig, IndexT, PtrTraits>::DeviceTensor(
        GpuResources* res,
        const AllocInfo& info,
        Tensor<T, Dim, InnerContig, IndexT, PtrTraits>& t)
        : Tensor<T, Dim, InnerContig, IndexT, PtrTraits>(
                  nullptr,
                  t.sizes(),
                  t.strides()) {
    this->reservation_ = std::move(
            res->allocMemoryHandle(AllocRequest(info, this->getSizeInBytes())));
    this->data_ = (T*)reservation_.get();

    FAISS_ASSERT(this->data_ || (this->getSizeInBytes() == 0));

    this->copyFrom(t, info.stream);
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT,
        template <typename U>
        class PtrTraits>
__host__ DeviceTensor<T, Dim, InnerContig, IndexT, PtrTraits>& DeviceTensor<
        T,
        Dim,
        InnerContig,
        IndexT,
        PtrTraits>::zero(cudaStream_t stream) {
    if (this->data_) {
        // Region must be contiguous
        FAISS_ASSERT(this->isContiguous());

        CUDA_VERIFY(cudaMemsetAsync(
                this->data_, 0, this->getSizeInBytes(), stream));
    }

    return *this;
}

} // namespace gpu
} // namespace faiss
