/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>

#include <cstdint>
#include <memory>

namespace faiss {
namespace gpu {
class GpuResourcesProvider;

enum class BBQCodeLayout {
    Packed1Bit,
    Transposed2Bit,
    Transposed4Bit,
    Packed4Bit,
    Packed7Bit,
    Packed8Bit
};

/** Precomputed BBQ codes and corrections in caller-owned device buffers.
 *
 * Codes are packed rows in codeLayout with a 4-byte aligned base pointer. Each
 * correction array has one entry per row; centroid has dim entries of the
 * Dataset numeric type. Corrections must have been generated for metric. All
 * buffers must outlive uses of the Dataset.
 */
struct BBQQuantizer {
    const uint8_t* codes = nullptr;
    const float* lowerIntervals = nullptr;
    const float* upperIntervals = nullptr;
    const float* additionalCorrections = nullptr;
    const int32_t* quantizedComponentSums = nullptr;
    const void* centroid = nullptr;
    const float* dequantDelta = nullptr;
    const float* dequantSumDelta = nullptr;
    const float* rowNorm = nullptr;
    BBQCodeLayout codeLayout = BBQCodeLayout::Packed1Bit;
    MetricType metric = METRIC_L2;
    float centroidNormSquared = 0;
};

/** Caller-controlled vector storage for cuVS CAGRA.
 *
 * Dense dimensions and row strides are in elements, with contiguous columns.
 * Binary vectors use UInt8, and their dimension is the number of bytes, not
 * bits. Views never own or copy their allocation. An owning Dataset controls
 * the allocation's lifetime; CAGRA always borrows it. Keep the backing
 * allocation alive and unchanged until all indexes using it and queued GPU work
 * finish. Moving an owning Dataset preserves its allocation and existing
 * borrowed views.
 */
class Dataset {
   public:
    enum class Layout { Standard, Padded, BBQ };

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

    /// Borrow one symmetric or two asymmetric BBQ quantizers without copying
    /// their buffers. Both quantizers describe the same vectors, numeric type,
    /// and metric. BBQ supports NN-descent graph construction; attach dense
    /// padded storage with CAGRA updateDataset before searching.
    static Dataset viewDeviceBBQ(
            const BBQQuantizer& source,
            idx_t numRows,
            int dim,
            NumericType numericType,
            int device,
            const BBQQuantizer* secondary = nullptr);

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

    /// Dense storage only; BBQ has separate code and correction buffers.
    const void* getData() const;
    idx_t getNumRows() const;
    int getDim() const;
    /// Dense storage only, in elements.
    idx_t getRowStride() const;
    NumericType getNumericType() const;
    int getDevice() const; // -1 for host storage
    Layout getLayout() const;
    bool isDevice() const;
    bool ownsData() const;

    /// Reports dense device residency, alignment, and exact required row
    /// stride. Returns false for BBQ. Does not inspect padding values; zero
    /// padding is a caller contract.
    bool isCagraCompatible() const;

    /// Create a non-owning view of the same storage by copying its metadata.
    /// Does not retain ownership of the backing allocation.
    Dataset view() const;

    /// Copy dense logical columns into packed host storage. destination must
    /// have numRows * dim * sizeof(element) bytes. Completes all device work
    /// using the source device before copying and returns with the host data
    /// ready.
    void copyToHost(void* destination) const;

   private:
    template <typename>
    friend class CuvsCagra;

    struct Impl;
    explicit Dataset(std::unique_ptr<Impl> impl);
    const Impl& getImpl() const;
    const BBQQuantizer* getBBQQuantizer(bool secondary) const;
    std::unique_ptr<Impl> impl_;
};

} // namespace gpu
} // namespace faiss
