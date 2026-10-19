/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>

#include <memory>

namespace faiss {
struct IndexHNSWCagra;
struct IndexBinaryHNSWCagra;

namespace gpu {
class GpuResourcesProvider;

/** Caller-controlled dense vector storage for cuVS CAGRA (C++ only).
 *
 * Dimensions and row strides are in elements, with contiguous columns. Binary
 * vectors use UInt8, and their dimension is the number of bytes, not bits.
 * Views never own or copy their allocation. An owning Dataset controls the
 * allocation's lifetime; CAGRA always borrows it. Keep the backing allocation
 * alive and unchanged until all indexes using it and queued GPU work finish.
 * Moving an owning Dataset preserves its allocation and existing borrowed
 * views.
 */
class Dataset {
   public:
    enum class Layout { Standard, Padded };

    ~Dataset();
    Dataset(Dataset&&) noexcept;
    Dataset& operator=(Dataset&&) noexcept;
    Dataset(const Dataset&) = delete;
    Dataset& operator=(const Dataset&) = delete;

    /// Borrow host storage. rowStride must be at least dim.
    static Dataset viewHost(
            const void* data,
            idx_t numRows,
            int dim,
            NumericType numericType,
            idx_t rowStride);

    /// Borrow storage accessible on device. A Standard view can also be
    /// CAGRA-compatible when its physical stride and alignment satisfy padding.
    /// When used by CAGRA, elements between dim and rowStride must be zero.
    static Dataset viewDevice(
            const void* data,
            idx_t numRows,
            int dim,
            NumericType numericType,
            idx_t rowStride,
            int device);

    /// Borrow CAGRA-compatible device storage without copying. The base address
    /// must be 16-byte aligned and rowStride must equal dim rounded up to a
    /// 16-byte row width. The caller must zero elements between dim and
    /// rowStride, including for compatible Standard views used by CAGRA.
    static Dataset viewDevicePadded(
            const void* data,
            idx_t numRows,
            int dim,
            NumericType numericType,
            idx_t rowStride,
            int device);

    /// Create padded device storage using cuVS make_device_padded_dataset.
    /// Device/managed input whose stride already matches the required padding
    /// is rejected; use a view for that storage. Copies logical columns and
    /// zeroes padding asynchronously on the resource's stream. Source data must
    /// be ready on that stream and stay alive and unchanged until copying
    /// finishes. Order subsequent use on that stream. Retains the resources
    /// needed to release the allocation.
    static Dataset makeDevicePadded(
            GpuResourcesProvider* resources,
            const Dataset& source,
            int device);

    /// Explicitly prepare independent GPU storage from CPU HNSW vectors. The
    /// floating-point overload preserves dtype and decodes signed Int8 codes.
    /// Copying completes before return; the Dataset does not depend on the
    /// CPU index's lifetime.
    static Dataset makeDevicePaddedFromIndex(
            GpuResourcesProvider* resources,
            const IndexHNSWCagra& source,
            int device);
    static Dataset makeDevicePaddedFromIndex(
            GpuResourcesProvider* resources,
            const IndexBinaryHNSWCagra& source,
            int device);

    const void* getData() const;
    idx_t getNumRows() const;
    int getDim() const;
    idx_t getRowStride() const;
    NumericType getNumericType() const;
    int getDevice() const; // -1 for host storage
    Layout getLayout() const;
    bool isDevice() const;
    bool ownsData() const;

    /// Reports device residency, base alignment, and exact required row stride.
    /// Does not inspect padding values; zero padding is a caller contract.
    bool isCagraCompatible() const;

    /// Create a non-owning view of the same storage by copying its metadata.
    /// Does not retain ownership of the backing allocation.
    Dataset view() const&;
    Dataset view() && = delete;
    Dataset view() const&& = delete;

    /// Copy logical columns into packed host storage. destination must have
    /// numRows * dim * sizeof(element) bytes. Completes all device work using
    /// the source device before copying and returns with the host data ready.
    void copyToHost(void* destination) const;

   private:
    struct Impl;
    explicit Dataset(std::unique_ptr<Impl> impl);
    const Impl& getImpl() const;
    std::unique_ptr<Impl> impl_;
};

} // namespace gpu
} // namespace faiss
