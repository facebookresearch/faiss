/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/Dataset.h>

#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/impl/FaissAssert.h>

#include <cuvs/neighbors/common.hpp>
#include <cuvs/preprocessing/quantize/bbq.hpp>

#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

namespace faiss {
namespace gpu {
namespace {

void checkCuda(cudaError_t error) {
    FAISS_THROW_IF_NOT_FMT(
            error == cudaSuccess,
            "Dataset CUDA operation failed: %s",
            cudaGetErrorString(error));
}

size_t elementSize(NumericType numericType) {
    switch (numericType) {
        case NumericType::Float32:
            return sizeof(float);
        case NumericType::Float16:
            return sizeof(half);
        case NumericType::Int8:
        case NumericType::UInt8:
            return 1;
        default:
            FAISS_THROW_MSG("Unsupported Dataset numeric type");
    }
}

void validateMetadata(
        const void* data,
        idx_t numRows,
        int dim,
        NumericType numericType,
        idx_t rowStride) {
    const size_t size = elementSize(numericType);
    FAISS_THROW_IF_NOT_MSG(
            numRows >= 0, "Dataset row count must be nonnegative");
    FAISS_THROW_IF_NOT_MSG(dim > 0, "Dataset dimension must be positive");
    FAISS_THROW_IF_NOT_MSG(
            rowStride >= dim &&
                    rowStride <= std::numeric_limits<uint32_t>::max(),
            "Dataset row stride must be at least dim and fit in uint32_t");
    FAISS_THROW_IF_NOT_MSG(
            numRows == 0 || data != nullptr,
            "A nonempty Dataset requires a data pointer");
    FAISS_THROW_IF_NOT_MSG(
            reinterpret_cast<uintptr_t>(data) % size == 0,
            "Dataset pointer must be aligned for its numeric type");
    FAISS_THROW_IF_NOT_MSG(
            static_cast<uint64_t>(numRows) <=
                    static_cast<uint64_t>(
                            std::numeric_limits<ptrdiff_t>::max()) /
                            static_cast<uint64_t>(rowStride) / size,
            "Dataset storage size overflows the address space");
}

void validateHostPointer(const void* data, idx_t numRows) {
    if (numRows == 0) {
        return;
    }
    cudaPointerAttributes attributes{};
    checkCuda(cudaPointerGetAttributes(&attributes, data));
    FAISS_THROW_IF_NOT_MSG(
            attributes.type != cudaMemoryTypeDevice,
            "Dataset::viewHost requires host-accessible memory");
}

void validateDevicePointer(const void* data, idx_t numRows, int device) {
    FAISS_THROW_IF_NOT_MSG(
            device >= 0 && device < getNumDevices(),
            "Dataset requires a valid CUDA device");
    if (numRows == 0) {
        return;
    }
    DeviceScope scope(device);
    cudaPointerAttributes attributes{};
    checkCuda(cudaPointerGetAttributes(&attributes, data));
    FAISS_THROW_IF_NOT_MSG(
            attributes.type == cudaMemoryTypeDevice ||
                    attributes.type == cudaMemoryTypeManaged ||
                    attributes.devicePointer == data,
            "Dataset::viewDevice requires device-accessible memory");
    FAISS_THROW_IF_NOT_MSG(
            attributes.type != cudaMemoryTypeDevice ||
                    attributes.device == device,
            "Dataset pointer belongs to a different CUDA device");
}

void validateBBQQuantizer(
        const BBQQuantizer& quantizer,
        idx_t numRows,
        int dim,
        NumericType numericType,
        int device) {
    namespace bbq = cuvs::preprocessing::quantize::bbq;
    validateMetadata(quantizer.centroid, 1, dim, numericType, dim);
    validateDevicePointer(quantizer.centroid, 1, device);
    FAISS_THROW_IF_NOT_MSG(
            quantizer.codeLayout >= BBQCodeLayout::Packed1Bit &&
                    quantizer.codeLayout <= BBQCodeLayout::Packed8Bit,
            "Invalid BBQ code layout");
    const auto rowLength = bbq::get_encoded_row_length(
            dim, static_cast<bbq::bbq_code_layout>(quantizer.codeLayout));
    validateMetadata(
            quantizer.codes, numRows, rowLength, NumericType::UInt8, rowLength);
    FAISS_THROW_IF_NOT_MSG(
            reinterpret_cast<uintptr_t>(quantizer.codes) % alignof(uint32_t) ==
                    0,
            "BBQ codes require a 4-byte aligned device pointer");
    validateDevicePointer(quantizer.codes, numRows, device);
    for (const float* values :
         {quantizer.lowerIntervals,
          quantizer.upperIntervals,
          quantizer.additionalCorrections,
          quantizer.dequantDelta,
          quantizer.dequantSumDelta,
          quantizer.rowNorm}) {
        validateMetadata(values, numRows, 1, NumericType::Float32, 1);
        validateDevicePointer(values, numRows, device);
    }
    const auto sumsAddress =
            reinterpret_cast<uintptr_t>(quantizer.quantizedComponentSums);
    FAISS_THROW_IF_NOT_MSG(
            (numRows == 0 || quantizer.quantizedComponentSums != nullptr) &&
                    sumsAddress % alignof(int32_t) == 0,
            "BBQ component sums require an aligned int32 device pointer");
    validateDevicePointer(quantizer.quantizedComponentSums, numRows, device);
}

struct DatasetStorage {
    virtual ~DatasetStorage() = default;
    virtual const void* getData() const = 0;
};

template <typename Storage>
struct TypedDatasetStorage : DatasetStorage {
    explicit TypedDatasetStorage(Storage value) : value_(std::move(value)) {}

    const void* getData() const override {
        return value_.view().data_handle();
    }

    Storage value_;
};

template <typename T>
std::unique_ptr<DatasetStorage> makeOwnedStorage(
        GpuResources* resources,
        const Dataset& source,
        int device) {
    auto& handle = resources->getRaftHandle(device);
    const auto* data = static_cast<const T*>(source.getData());
    const auto rows = source.getNumRows();
    const auto dim = static_cast<int64_t>(source.getDim());
    const auto stride = source.getRowStride();
    auto owner = source.isDevice()
            ? cuvs::neighbors::make_device_padded_dataset(
                      handle,
                      raft::make_device_strided_matrix_view(
                              data, rows, dim, stride))
            : cuvs::neighbors::make_device_padded_dataset(
                      handle,
                      raft::make_host_strided_matrix_view(
                              data, rows, dim, stride));
    using Owner = cuvs::neighbors::device_padded_dataset<T, int64_t>;
    return std::make_unique<TypedDatasetStorage<Owner>>(std::move(*owner));
}

} // namespace

struct Dataset::Impl {
    const void* data;
    idx_t numRows;
    int dim;
    idx_t rowStride;
    NumericType numericType;
    int device;
    Layout layout;
    std::optional<BBQQuantizer> bbq;
    std::optional<BBQQuantizer> secondaryBbq;
    // Declared before storage so resources outlive the allocation.
    std::shared_ptr<GpuResources> resources;
    std::unique_ptr<DatasetStorage> storage;

    Impl(const void* ptr,
         idx_t rows,
         int columns,
         idx_t stride,
         NumericType type,
         int gpu,
         Layout l)
            : data(ptr),
              numRows(rows),
              dim(columns),
              rowStride(stride),
              numericType(type),
              device(gpu),
              layout(l) {}

    ~Impl() {
        if (storage) {
            DeviceScope scope(device);
            storage.reset();
        }
    }
};

Dataset::Dataset(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Dataset::~Dataset() = default;
Dataset::Dataset(Dataset&&) noexcept = default;
Dataset& Dataset::operator=(Dataset&&) noexcept = default;

const Dataset::Impl& Dataset::getImpl() const {
    FAISS_THROW_IF_NOT_MSG(impl_, "Cannot access a moved-from Dataset");
    return *impl_;
}

Dataset Dataset::viewHost(
        const void* data,
        idx_t numRows,
        int dim,
        NumericType numericType,
        idx_t rowStride) {
    validateMetadata(data, numRows, dim, numericType, rowStride);
    validateHostPointer(data, numRows);
    auto impl = std::make_unique<Impl>(
            data, numRows, dim, rowStride, numericType, -1, Layout::Standard);
    return Dataset(std::move(impl));
}

Dataset Dataset::viewDevice(
        const void* data,
        idx_t numRows,
        int dim,
        NumericType numericType,
        idx_t rowStride,
        int device) {
    validateMetadata(data, numRows, dim, numericType, rowStride);
    validateDevicePointer(data, numRows, device);
    auto impl = std::make_unique<Impl>(
            data,
            numRows,
            dim,
            rowStride,
            numericType,
            device,
            Layout::Standard);
    return Dataset(std::move(impl));
}

Dataset Dataset::viewDevicePadded(
        const void* data,
        idx_t numRows,
        int dim,
        NumericType numericType,
        idx_t rowStride,
        int device) {
    auto result =
            viewDevice(data, numRows, dim, numericType, rowStride, device);
    FAISS_THROW_IF_NOT_MSG(
            result.isCagraCompatible(),
            "A padded Dataset requires 16-byte base alignment and the exact "
            "CAGRA row stride");
    result.impl_->layout = Layout::Padded;
    return result;
}

Dataset Dataset::viewDeviceBBQ(
        const BBQQuantizer& source,
        idx_t numRows,
        int dim,
        NumericType numericType,
        int device,
        const BBQQuantizer* secondary) {
    validateBBQQuantizer(source, numRows, dim, numericType, device);
    if (secondary) {
        validateBBQQuantizer(*secondary, numRows, dim, numericType, device);
    }
    auto impl = std::make_unique<Impl>(
            nullptr, numRows, dim, 0, numericType, device, Layout::BBQ);
    impl->bbq = source;
    if (secondary) {
        impl->secondaryBbq = *secondary;
    }
    return Dataset(std::move(impl));
}

Dataset Dataset::makeDevicePadded(
        GpuResourcesProvider* resources,
        const Dataset& source,
        int device) {
    FAISS_THROW_IF_NOT_MSG(resources, "Dataset requires GPU resources");
    FAISS_THROW_IF_NOT_MSG(
            device >= 0 && device < getNumDevices(),
            "Dataset requires a valid CUDA device");
    const auto& src = source.getImpl();
    FAISS_THROW_IF_NOT_MSG(
            src.layout != Layout::BBQ,
            "Dataset::makeDevicePadded requires a dense Dataset");
    const auto stride = cuvs::neighbors::cagra_required_row_width(
            src.dim, elementSize(src.numericType));
    validateMetadata(
            source.getData(), src.numRows, src.dim, src.numericType, stride);
    auto impl = std::make_unique<Impl>(
            nullptr,
            src.numRows,
            src.dim,
            stride,
            src.numericType,
            device,
            Layout::Padded);
    impl->resources = resources->getResources();
    FAISS_THROW_IF_NOT_MSG(impl->resources, "Dataset requires GPU resources");

    DeviceScope scope(device);
    switch (src.numericType) {
        case NumericType::Float32:
            impl->storage = makeOwnedStorage<float>(
                    impl->resources.get(), source, device);
            break;
        case NumericType::Float16:
            impl->storage = makeOwnedStorage<half>(
                    impl->resources.get(), source, device);
            break;
        case NumericType::Int8:
            impl->storage = makeOwnedStorage<int8_t>(
                    impl->resources.get(), source, device);
            break;
        case NumericType::UInt8:
            impl->storage = makeOwnedStorage<uint8_t>(
                    impl->resources.get(), source, device);
            break;
        default:
            FAISS_THROW_MSG("Unsupported Dataset numeric type");
    }
    return Dataset(std::move(impl));
}

const void* Dataset::getData() const {
    const auto& impl = getImpl();
    FAISS_THROW_IF_NOT_MSG(
            impl.layout != Layout::BBQ,
            "A BBQ Dataset has no dense data pointer");
    return impl.storage ? impl.storage->getData() : impl.data;
}
idx_t Dataset::getNumRows() const {
    return getImpl().numRows;
}
int Dataset::getDim() const {
    return getImpl().dim;
}
idx_t Dataset::getRowStride() const {
    const auto& impl = getImpl();
    FAISS_THROW_IF_NOT_MSG(
            impl.layout != Layout::BBQ,
            "A BBQ Dataset has no dense row stride");
    return impl.rowStride;
}
NumericType Dataset::getNumericType() const {
    return getImpl().numericType;
}
int Dataset::getDevice() const {
    return getImpl().device;
}
Dataset::Layout Dataset::getLayout() const {
    return getImpl().layout;
}
bool Dataset::isDevice() const {
    return getImpl().device >= 0;
}
bool Dataset::ownsData() const {
    return bool(getImpl().storage);
}
bool Dataset::isCagraCompatible() const {
    const auto& impl = getImpl();
    return impl.layout != Layout::BBQ && isDevice() &&
            reinterpret_cast<uintptr_t>(getData()) % 16 == 0 &&
            impl.rowStride ==
            cuvs::neighbors::cagra_required_row_width(
                    impl.dim, elementSize(impl.numericType));
}

Dataset Dataset::view() const {
    const auto& src = getImpl();
    auto impl = std::make_unique<Impl>(
            src.layout == Layout::BBQ ? nullptr : getData(),
            src.numRows,
            src.dim,
            src.rowStride,
            src.numericType,
            src.device,
            src.layout);
    impl->bbq = src.bbq;
    impl->secondaryBbq = src.secondaryBbq;
    return Dataset(std::move(impl));
}

const BBQQuantizer* Dataset::getBBQQuantizer(bool secondary) const {
    const auto& impl = getImpl();
    const auto& quantizer = secondary ? impl.secondaryBbq : impl.bbq;
    return quantizer ? &*quantizer : nullptr;
}

void Dataset::copyToHost(void* destination) const {
    const auto& impl = getImpl();
    FAISS_THROW_IF_NOT_MSG(
            impl.layout != Layout::BBQ,
            "Dataset::copyToHost requires a dense Dataset");
    if (impl.numRows == 0) {
        return;
    }
    FAISS_THROW_IF_NOT_MSG(destination, "Dataset copy requires a destination");
    const size_t size = elementSize(impl.numericType);
    const size_t width = impl.dim * size;
    const size_t pitch = impl.rowStride * size;
    if (isDevice()) {
        DeviceScope scope(impl.device);
        checkCuda(cudaDeviceSynchronize());
        checkCuda(cudaMemcpy2D(
                destination,
                width,
                getData(),
                pitch,
                width,
                impl.numRows,
                cudaMemcpyDefault));
    } else {
        const auto* input = static_cast<const uint8_t*>(getData());
        auto* output = static_cast<uint8_t*>(destination);
        for (idx_t row = 0; row < impl.numRows; ++row) {
            std::memcpy(output + row * width, input + row * pitch, width);
        }
    }
}

} // namespace gpu
} // namespace faiss
