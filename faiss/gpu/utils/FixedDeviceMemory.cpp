/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/FixedDeviceMemory.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <iostream>
#include <sstream>

namespace faiss {
namespace gpu {

FixedDeviceMemory::FixedDeviceMemory(
        GpuResources* res,
        int device,
        size_t allocSize,
        MemorySpace space)
        : res_(res),
          device_(device),
          alloc_(nullptr),
          allocSize_(allocSize),
          offset_(0) {
    if (allocSize_ == 0) {
        return;
    }

    // ensures this memory block is aligned to CUDA pre-requisites for
    // read/write
    allocSize_ = utils::roundUp(allocSize_, (size_t)256);

    allocSize_ += 256;

    // the first address is the address of the entire block, so it can't be the
    // address of a sub-block within the block
    offset_ += 256;

    DeviceScope s(device_);
    auto defaultStream = res_->getDefaultStream(device_);
    auto req = AllocRequest(
            AllocType::Other, device_, space, defaultStream, allocSize_);

    alloc_ = (char*)res_->allocMemory(req);
    FAISS_ASSERT_FMT(
            alloc_,
            "could not reserve fixed memory region of size %zu",
            allocSize_);
}

FixedDeviceMemory::~FixedDeviceMemory() {
    DeviceScope s(device_);
    if (alloc_) {
        res_->deallocMemory(device_, alloc_);
    }
}

int FixedDeviceMemory::getDevice() const {
    return device_;
}

void* FixedDeviceMemory::allocMemory(size_t size) {
    if (getSizeAvailable() < size) {
        return nullptr;
    }

    char* freeAlloc = alloc_ + offset_;
    offset_ += size;
    return freeAlloc;
}

size_t FixedDeviceMemory::getSizeAvailable() const {
    return allocSize_ - offset_;
}

std::string FixedDeviceMemory::toString() const {
    std::stringstream s;
    s << "DM device " << device_ << ": Total memory " << allocSize_ << "\n";
    s << "Begin " << alloc_ << "\n";
    s << "End " << alloc_ + allocSize_ << "\n";
    s << "Available memory " << alloc_ + allocSize_ - offset_ << "\n";
    return s.str();
}

} // namespace gpu
} // namespace faiss
