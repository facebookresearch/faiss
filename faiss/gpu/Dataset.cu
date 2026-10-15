/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/Dataset.h>

#include <faiss/IndexBinaryHNSW.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/impl/FaissAssert.h>

#include <cuvs/neighbors/common.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

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
std::unique_ptr<DatasetStorage> makeViewStorage(
        const void* data,
        idx_t numRows,
        int dim,
        idx_t rowStride,
        int device,
        Dataset::Layout layout) {
    if (device < 0) {
        using View = cuvs::neighbors::host_standard_dataset_view<T, int64_t>;
        return std::make_unique<TypedDatasetStorage<View>>(
                View(raft::make_host_matrix_view<const T, int64_t>(
                             static_cast<const T*>(data), numRows, rowStride),
                     static_cast<uint32_t>(dim)));
    }
    const auto matrix = raft::make_device_matrix_view<const T, int64_t>(
            static_cast<const T*>(data), numRows, rowStride);
    if (layout == Dataset::Layout::Padded) {
        using View = cuvs::neighbors::device_padded_dataset_view<T, int64_t>;
        return std::make_unique<TypedDatasetStorage<View>>(
                View(matrix, static_cast<uint32_t>(dim)));
    }
    using View = cuvs::neighbors::device_standard_dataset_view<T, int64_t>;
    return std::make_unique<TypedDatasetStorage<View>>(
            View(matrix, static_cast<uint32_t>(dim)));
}

template <typename T>
std::unique_ptr<DatasetStorage> makeOwnedStorage(
        GpuResources* resources,
        const void* sourceData,
        idx_t numRows,
        int dim,
        idx_t sourceStride,
        idx_t targetStride,
        int device) {
    auto& handle = resources->getRaftHandle(device);
    const auto stream =
            cuda::stream_ref{raft::resource::get_cuda_stream(handle)}.get();
    auto matrix =
            raft::make_device_matrix<T, int64_t>(handle, numRows, targetStride);
    if (numRows != 0) {
        checkCuda(cudaMemsetAsync(
                matrix.data_handle(), 0, matrix.size() * sizeof(T), stream));
        checkCuda(cudaMemcpy2DAsync(
                matrix.data_handle(),
                targetStride * sizeof(T),
                sourceData,
                sourceStride * sizeof(T),
                dim * sizeof(T),
                numRows,
                cudaMemcpyDefault,
                stream));
        checkCuda(cudaStreamSynchronize(stream));
    }
    using Owner = cuvs::neighbors::device_padded_dataset<T, int64_t>;
    return std::make_unique<TypedDatasetStorage<Owner>>(
            Owner(std::move(matrix), static_cast<uint32_t>(dim)));
}

} // namespace

struct Dataset::Impl {
    idx_t numRows;
    int dim;
    idx_t rowStride;
    NumericType numericType;
    int device;
    Layout layout;
    bool ownsData = false;
    // Declared before storage so resources outlive the allocation.
    std::shared_ptr<GpuResources> resources;
    std::unique_ptr<DatasetStorage> storage;

    Impl(idx_t rows,
         int columns,
         idx_t stride,
         NumericType type,
         int gpu,
         Layout l)
            : numRows(rows),
              dim(columns),
              rowStride(stride),
              numericType(type),
              device(gpu),
              layout(l) {}

    ~Impl() {
        if (ownsData) {
            DeviceScope scope(device);
            storage.reset();
        }
    }

    void createView(const void* data) {
        switch (numericType) {
            case NumericType::Float32:
                storage = makeViewStorage<float>(
                        data, numRows, dim, rowStride, device, layout);
                break;
            case NumericType::Float16:
                storage = makeViewStorage<half>(
                        data, numRows, dim, rowStride, device, layout);
                break;
            case NumericType::Int8:
                storage = makeViewStorage<int8_t>(
                        data, numRows, dim, rowStride, device, layout);
                break;
            case NumericType::UInt8:
                storage = makeViewStorage<uint8_t>(
                        data, numRows, dim, rowStride, device, layout);
                break;
            default:
                FAISS_THROW_MSG("Unsupported Dataset numeric type");
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
            numRows, dim, rowStride, numericType, -1, Layout::Standard);
    impl->createView(data);
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
            numRows, dim, rowStride, numericType, device, Layout::Standard);
    impl->createView(data);
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
    result.impl_->createView(data);
    return result;
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
    const auto stride = cuvs::neighbors::cagra_required_row_width(
            src.dim, elementSize(src.numericType));
    validateMetadata(
            source.getData(), src.numRows, src.dim, src.numericType, stride);
    auto impl = std::make_unique<Impl>(
            src.numRows,
            src.dim,
            stride,
            src.numericType,
            device,
            Layout::Padded);
    impl->resources = resources->getResources();
    FAISS_THROW_IF_NOT_MSG(impl->resources, "Dataset requires GPU resources");

    // Explicit cross-device preparation also works without peer access.
    std::vector<uint8_t> staging;
    const void* sourceData = source.getData();
    idx_t sourceStride = src.rowStride;
    if (source.isDevice() && src.device != device) {
        staging.resize(src.numRows * src.dim * elementSize(src.numericType));
        source.copyToHost(staging.data());
        sourceData = staging.data();
        sourceStride = src.dim;
    }
    DeviceScope scope(device);
    switch (src.numericType) {
        case NumericType::Float32:
            impl->storage = makeOwnedStorage<float>(
                    impl->resources.get(),
                    sourceData,
                    src.numRows,
                    src.dim,
                    sourceStride,
                    stride,
                    device);
            break;
        case NumericType::Float16:
            impl->storage = makeOwnedStorage<half>(
                    impl->resources.get(),
                    sourceData,
                    src.numRows,
                    src.dim,
                    sourceStride,
                    stride,
                    device);
            break;
        case NumericType::Int8:
            impl->storage = makeOwnedStorage<int8_t>(
                    impl->resources.get(),
                    sourceData,
                    src.numRows,
                    src.dim,
                    sourceStride,
                    stride,
                    device);
            break;
        case NumericType::UInt8:
            impl->storage = makeOwnedStorage<uint8_t>(
                    impl->resources.get(),
                    sourceData,
                    src.numRows,
                    src.dim,
                    sourceStride,
                    stride,
                    device);
            break;
        default:
            FAISS_THROW_MSG("Unsupported Dataset numeric type");
    }
    impl->ownsData = true;
    return Dataset(std::move(impl));
}

Dataset Dataset::makeDevicePaddedFromIndex(
        GpuResourcesProvider* resources,
        const IndexHNSWCagra& source,
        int device) {
    FAISS_THROW_IF_NOT_MSG(
            source.storage, "CPU HNSW index has no vector storage");
    FAISS_THROW_IF_NOT_MSG(
            source.storage->ntotal == source.ntotal &&
                    source.storage->d == source.d,
            "CPU HNSW vector storage does not match index metadata");
    const NumericType type = source.get_numeric_type();
    const size_t size = elementSize(type);
    const void* data = nullptr;
    std::vector<int8_t> decoded;
    if (type == NumericType::Float32) {
        const auto* flat = dynamic_cast<const IndexFlat*>(source.storage);
        FAISS_THROW_IF_NOT_MSG(
                flat, "Float32 HNSW import requires flat storage");
        data = flat->get_xb();
        validateMetadata(data, source.ntotal, source.d, type, source.d);
        FAISS_THROW_IF_NOT_MSG(
                flat->codes.size() >=
                        static_cast<size_t>(source.ntotal) * source.d * size,
                "CPU HNSW flat storage is truncated");
    } else {
        const auto* sq =
                dynamic_cast<const IndexScalarQuantizer*>(source.storage);
        FAISS_THROW_IF_NOT_MSG(
                sq, "HNSW import requires scalar quantizer storage");
        const auto expectedType = type == NumericType::Float16
                ? ScalarQuantizer::QT_fp16
                : type == NumericType::Int8
                ? ScalarQuantizer::QT_8bit_direct_signed
                : ScalarQuantizer::QT_8bit_direct;
        FAISS_THROW_IF_NOT_MSG(
                sq->sq.qtype == expectedType,
                "CPU HNSW scalar encoding does not match its numeric type");
        validateMetadata(
                sq->codes.data(), source.ntotal, source.d, type, source.d);
        FAISS_THROW_IF_NOT_MSG(
                sq->codes.size() >=
                        static_cast<size_t>(source.ntotal) * source.d * size,
                "CPU HNSW scalar storage is truncated");
        if (type == NumericType::Int8) {
            decoded.resize(source.ntotal * source.d);
            for (size_t i = 0; i < decoded.size(); ++i) {
                decoded[i] = static_cast<int8_t>(
                        static_cast<int>(sq->codes[i]) - 128);
            }
            data = decoded.data();
        } else {
            data = sq->codes.data();
        }
    }
    auto view = viewHost(data, source.ntotal, source.d, type, source.d);
    return makeDevicePadded(resources, view, device);
}

Dataset Dataset::makeDevicePaddedFromIndex(
        GpuResourcesProvider* resources,
        const IndexBinaryHNSWCagra& source,
        int device) {
    const auto* flat = dynamic_cast<const IndexBinaryFlat*>(source.storage);
    FAISS_THROW_IF_NOT_MSG(flat, "Binary HNSW import requires flat storage");
    FAISS_THROW_IF_NOT_MSG(
            source.d > 0 && source.d % 8 == 0 &&
                    flat->ntotal == source.ntotal && flat->d == source.d,
            "CPU binary HNSW vector storage does not match index metadata");
    validateMetadata(
            flat->xb.data(),
            source.ntotal,
            source.d / 8,
            NumericType::UInt8,
            source.d / 8);
    FAISS_THROW_IF_NOT_MSG(
            flat->xb.size() >=
                    static_cast<size_t>(source.ntotal) * (source.d / 8),
            "CPU binary HNSW storage is truncated");
    auto view = viewHost(
            flat->xb.data(),
            source.ntotal,
            source.d / 8,
            NumericType::UInt8,
            source.d / 8);
    return makeDevicePadded(resources, view, device);
}

const void* Dataset::getData() const {
    return getImpl().storage->getData();
}
idx_t Dataset::getNumRows() const {
    return getImpl().numRows;
}
int Dataset::getDim() const {
    return getImpl().dim;
}
idx_t Dataset::getRowStride() const {
    return getImpl().rowStride;
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
    return getImpl().ownsData;
}
bool Dataset::isCagraCompatible() const {
    const auto& impl = getImpl();
    return isDevice() && reinterpret_cast<uintptr_t>(getData()) % 16 == 0 &&
            impl.rowStride ==
            cuvs::neighbors::cagra_required_row_width(
                    impl.dim, elementSize(impl.numericType));
}

Dataset Dataset::borrow() const& {
    const auto& src = getImpl();
    auto impl = std::make_unique<Impl>(
            src.numRows,
            src.dim,
            src.rowStride,
            src.numericType,
            src.device,
            src.layout);
    impl->createView(getData());
    return Dataset(std::move(impl));
}

void Dataset::copyToHost(void* destination) const {
    const auto& impl = getImpl();
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
