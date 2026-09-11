// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <faiss/MetricType.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/utils/Tensor.cuh>

#include <cuvs/core/c_api.h>
#include <cuvs/core/dataset.h>
#include <cuvs/distance/distance.h>
#include <dlpack/dlpack.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>

#pragma GCC visibility push(default)
namespace faiss {
namespace gpu {

inline cuvsDistanceType metricFaissToCuvs(
        MetricType metric,
        bool exactDistance) {
    (void)exactDistance;
    switch (metric) {
        case MetricType::METRIC_INNER_PRODUCT:
            return InnerProduct;
        case MetricType::METRIC_L2:
            return L2Expanded;
        case MetricType::METRIC_L1:
            return L1;
        case MetricType::METRIC_Linf:
            return Linf;
        case MetricType::METRIC_Lp:
            return LpUnexpanded;
        case MetricType::METRIC_Canberra:
            return Canberra;
        case MetricType::METRIC_BrayCurtis:
            return BrayCurtis;
        case MetricType::METRIC_JensenShannon:
            return JensenShannon;
        default:
            RAFT_FAIL("Distance type not supported");
    }
}

inline void cuvsCheck(cuvsError_t status, const char* operation) {
    FAISS_THROW_IF_NOT_FMT(
            status == CUVS_SUCCESS,
            "%s failed: %s",
            operation,
            cuvsGetLastErrorText());
}

inline cuvsResources_t cuvsResourcesFromGpuResources(GpuResources* resources) {
    return reinterpret_cast<cuvsResources_t>(
            &resources->getRaftHandleCurrentDevice());
}

template <typename T>
constexpr DLDataType cuvsDtype() {
    using Value = std::remove_cv_t<T>;
    if constexpr (std::is_same_v<Value, float>) {
        return DLDataType{kDLFloat, 32, 1};
    } else if constexpr (std::is_same_v<Value, half>) {
        return DLDataType{kDLFloat, 16, 1};
    } else if constexpr (std::is_same_v<Value, int64_t>) {
        return DLDataType{kDLInt, 64, 1};
    } else if constexpr (std::is_same_v<Value, uint64_t>) {
        return DLDataType{kDLUInt, 64, 1};
    } else if constexpr (std::is_same_v<Value, int32_t>) {
        return DLDataType{kDLInt, 32, 1};
    } else if constexpr (std::is_same_v<Value, uint32_t>) {
        return DLDataType{kDLUInt, 32, 1};
    } else if constexpr (std::is_same_v<Value, int8_t>) {
        return DLDataType{kDLInt, 8, 1};
    } else if constexpr (std::is_same_v<Value, uint8_t>) {
        return DLDataType{kDLUInt, 8, 1};
    } else {
        static_assert(!sizeof(T), "unsupported DLPack data type");
    }
}

enum class CuvsTensorLayout { RowMajor, ColumnMajor };

/// A non-owning DLPack view whose metadata remains valid for this object's
/// lifetime. The data pointer can refer to host or device memory.
template <typename T, size_t Rank>
class CuvsTensor {
   public:
    template <typename... Extents>
    explicit CuvsTensor(T* data, Extents... extents)
            : CuvsTensor(
                      data,
                      CuvsTensorLayout::RowMajor,
                      static_cast<int64_t>(extents)...) {}

    template <typename... Extents>
    CuvsTensor(T* data, CuvsTensorLayout layout, Extents... extents)
            : shape_{static_cast<int64_t>(extents)...} {
        static_assert(sizeof...(Extents) == Rank);
        static_assert(Rank > 0);

        if (layout == CuvsTensorLayout::RowMajor) {
            strides_[Rank - 1] = 1;
            for (size_t i = Rank - 1; i > 0; --i) {
                strides_[i - 1] = strides_[i] * shape_[i];
            }
        } else {
            strides_[0] = 1;
            for (size_t i = 1; i < Rank; ++i) {
                strides_[i] = strides_[i - 1] * shape_[i - 1];
            }
        }

        auto device = getDeviceForAddress(data);
        tensor_.dl_tensor = DLTensor{
                const_cast<std::remove_const_t<T>*>(data),
                DLDevice{
                        device >= 0 ? kDLCUDA : kDLCPU,
                        device >= 0 ? device : 0},
                static_cast<int32_t>(Rank),
                cuvsDtype<T>(),
                shape_.data(),
                layout == CuvsTensorLayout::RowMajor ? nullptr
                                                     : strides_.data(),
                0};
        tensor_.manager_ctx = nullptr;
        tensor_.deleter = nullptr;
    }

    CuvsTensor(const CuvsTensor&) = delete;
    CuvsTensor& operator=(const CuvsTensor&) = delete;

    DLManagedTensor* get() {
        return &tensor_;
    }

   private:
    std::array<int64_t, Rank> shape_{};
    std::array<int64_t, Rank> strides_{};
    DLManagedTensor tensor_{};
};

/// Owns only the metadata populated by cuVS C API getter functions. The
/// underlying data remains owned by the cuVS index.
class CuvsOutputTensor {
   public:
    CuvsOutputTensor() = default;
    CuvsOutputTensor(const CuvsOutputTensor&) = delete;
    CuvsOutputTensor& operator=(const CuvsOutputTensor&) = delete;

    ~CuvsOutputTensor() {
        if (tensor_.deleter) {
            tensor_.deleter(&tensor_);
        }
    }

    DLManagedTensor* get() {
        return &tensor_;
    }

    template <typename T>
    T* data() const {
        return reinterpret_cast<T*>(
                static_cast<uint8_t*>(tensor_.dl_tensor.data) +
                tensor_.dl_tensor.byte_offset);
    }

    int64_t extent(size_t dimension) const {
        FAISS_ASSERT(dimension < static_cast<size_t>(tensor_.dl_tensor.ndim));
        return tensor_.dl_tensor.shape[dimension];
    }

   private:
    DLManagedTensor tensor_{};
};

template <typename T, typename... Extents>
auto makeCuvsTensor(T* data, Extents... extents) {
    return CuvsTensor<T, sizeof...(Extents)>(data, extents...);
}

/// release/26.10 rejects an owning padded copy when a device tensor is
/// already 16-byte row aligned; that case must use a padded view.
template <typename T>
bool cuvsPaddedDatasetNeedsView(const T* data, int64_t columns) {
    return getDeviceForAddress(data) >= 0 &&
            (columns * static_cast<int64_t>(sizeof(T))) % 16 == 0;
}

/// Construct the C dataset form required by CAGRA from a raw pointer. The
/// returned handle owns padded storage unless release/26.10 requires a view.
template <typename T>
cuvsDataset_t makeCuvsPaddedDataset(
        GpuResources* resources,
        const T* data,
        int64_t rows,
        int64_t columns) {
    auto sourceTensor = makeCuvsTensor(const_cast<T*>(data), rows, columns);
    cuvsDataset_t dataset = nullptr;
    if (cuvsPaddedDatasetNeedsView(data, columns)) {
        cuvsCheck(
                cuvsDatasetMakePaddedView(
                        cuvsResourcesFromGpuResources(resources),
                        sourceTensor.get(),
                        &dataset),
                "cuvsDatasetMakePaddedView");
    } else {
        cuvsCheck(
                cuvsDatasetMakePadded(
                        cuvsResourcesFromGpuResources(resources),
                        sourceTensor.get(),
                        CUVS_DATASET_MEM_TYPE_DEVICE,
                        &dataset),
                "cuvsDatasetMakePadded");
    }
    return dataset;
}

template <typename Handle, cuvsError_t (*Destroy)(Handle*)>
struct CuvsDeleter {
    void operator()(Handle* handle) const noexcept {
        if (handle) {
            Destroy(handle);
        }
    }
};

template <typename Handle, cuvsError_t (*Destroy)(Handle*)>
using CuvsUniquePtr = std::unique_ptr<Handle, CuvsDeleter<Handle, Destroy>>;

/// Identify matrix rows containing non NaN values. validRows[i] is false if row
/// i contains a NaN value and true otherwise.
void validRowIndices(
        GpuResources* res,
        Tensor<float, 2, true>& vecs,
        bool* validRows);

/// Filter out matrix rows containing NaN values. The vectors and indices are
/// updated in-place.
idx_t inplaceGatherFilteredRows(
        GpuResources* res,
        Tensor<float, 2, true>& vecs,
        Tensor<idx_t, 1, true>& indices);

/// Copy uint32_t indices to idx_t, replacing any index >= n with -1.
/// cuVS CAGRA returns sentinel values (e.g. INT32_MAX) for result slots
/// where filtered search couldn't find a valid neighbor.
void sanitizeCuvsIndices(
        GpuResources* res,
        uint32_t* src,
        idx_t* dst,
        size_t count,
        idx_t n);

} // namespace gpu
} // namespace faiss
#pragma GCC visibility pop
