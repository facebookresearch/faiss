/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <sstream>

namespace faiss {
namespace gpu {

std::string allocTypeToString(AllocType t) {
    switch (t) {
        case AllocType::Other:
            return "Other";
        case AllocType::FlatData:
            return "FlatData";
        case AllocType::IVFLists:
            return "IVFLists";
        case AllocType::Quantizer:
            return "Quantizer";
        case AllocType::QuantizerPrecomputedCodes:
            return "QuantizerPrecomputedCodes";
        case AllocType::TemporaryMemoryBuffer:
            return "TemporaryMemoryBuffer";
        case AllocType::TemporaryMemoryOverflow:
            return "TemporaryMemoryOverflow";
        default:
            return "Unknown";
    }
}

std::string memorySpaceToString(MemorySpace s) {
    switch (s) {
        case MemorySpace::Temporary:
            return "Temporary";
        case MemorySpace::Device:
            return "Device";
        case MemorySpace::Unified:
            return "Unified";
        default:
            return "Unknown";
    }
}

std::string AllocInfo::toString() const {
    std::stringstream ss;
    ss << "type " << allocTypeToString(type) << " dev " << device << " space "
       << memorySpaceToString(space) << " stream " << (void*)stream;

    return ss.str();
}

std::string AllocRequest::toString() const {
    std::stringstream ss;
    ss << AllocInfo::toString() << " size " << size << " bytes";

    return ss.str();
}

AllocInfo makeDevAlloc(AllocType at, cudaStream_t st) {
    return AllocInfo(at, getCurrentDevice(), MemorySpace::Device, st);
}

AllocInfo makeTempAlloc(AllocType at, cudaStream_t st) {
    return AllocInfo(at, getCurrentDevice(), MemorySpace::Temporary, st);
}

AllocInfo makeSpaceAlloc(AllocType at, MemorySpace sp, cudaStream_t st) {
    return AllocInfo(at, getCurrentDevice(), sp, st);
}

//
// GpuMemoryReservation
//

GpuMemoryReservation::GpuMemoryReservation()
        : res(nullptr), device(0), stream(nullptr), data(nullptr), size(0) {}

GpuMemoryReservation::GpuMemoryReservation(
        GpuResources* r,
        int dev,
        cudaStream_t str,
        void* p,
        size_t sz)
        : res(r), device(dev), stream(str), data(p), size(sz) {}

GpuMemoryReservation::GpuMemoryReservation(GpuMemoryReservation&& m) noexcept {
    res = m.res;
    m.res = nullptr;
    device = m.device;
    m.device = 0;
    stream = m.stream;
    m.stream = nullptr;
    data = m.data;
    m.data = nullptr;
    size = m.size;
    m.size = 0;
}

GpuMemoryReservation& GpuMemoryReservation::operator=(
        GpuMemoryReservation&& m) {
    // Can't be both a valid allocation and the same allocation
    FAISS_ASSERT(
            !(res && res == m.res && device == m.device && data == m.data));

    release();
    res = m.res;
    m.res = nullptr;
    device = m.device;
    m.device = 0;
    stream = m.stream;
    m.stream = nullptr;
    data = m.data;
    m.data = nullptr;
    size = m.size;
    m.size = 0;

    return *this;
}

void GpuMemoryReservation::release() {
    if (res) {
        res->deallocMemory(device, data);
        res = nullptr;
        device = 0;
        stream = nullptr;
        data = nullptr;
        size = 0;
    }
}

GpuMemoryReservation::~GpuMemoryReservation() {
    if (res) {
        res->deallocMemory(device, data);
    }
}

//
// GpuResources
//

GpuResources::~GpuResources() {}

cublasHandle_t GpuResources::getBlasHandleCurrentDevice() {
    return getBlasHandle(getCurrentDevice());
}

cudaStream_t GpuResources::getDefaultStreamCurrentDevice() {
    return getDefaultStream(getCurrentDevice());
}

std::vector<cudaStream_t> GpuResources::getAlternateStreamsCurrentDevice() {
    return getAlternateStreams(getCurrentDevice());
}

cudaStream_t GpuResources::getAsyncCopyStreamCurrentDevice() {
    return getAsyncCopyStream(getCurrentDevice());
}

void GpuResources::syncDefaultStream(int device) {
    CUDA_VERIFY(cudaStreamSynchronize(getDefaultStream(device)));
}

void GpuResources::syncDefaultStreamCurrentDevice() {
    syncDefaultStream(getCurrentDevice());
}

GpuMemoryReservation GpuResources::allocMemoryHandle(const AllocRequest& req) {
    return GpuMemoryReservation(
            this, req.device, req.stream, allocMemory(req), req.size);
}

size_t GpuResources::getTempMemoryAvailableCurrentDevice() const {
    return getTempMemoryAvailable(getCurrentDevice());
}

//
// GpuResourcesProvider
//

GpuResourcesProvider::~GpuResourcesProvider() {}

//
// GpuResourcesProviderFromResourceInstance
//

GpuResourcesProviderFromInstance::GpuResourcesProviderFromInstance(
        std::shared_ptr<GpuResources> p)
        : res_(p) {}

GpuResourcesProviderFromInstance::~GpuResourcesProviderFromInstance() {}

std::shared_ptr<GpuResources> GpuResourcesProviderFromInstance::getResources() {
    return res_;
}

} // namespace gpu
} // namespace faiss
