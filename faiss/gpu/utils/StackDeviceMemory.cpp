/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StackDeviceMemory.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <algorithm>
#include <sstream>

namespace faiss {
namespace gpu {

namespace {

size_t adjustStackSize(size_t sz) {
    if (sz == 0) {
        return 0;
    } else {
        // ensure that we have at least 16 bytes, as all allocations are bumped
        // up to 16
        return utils::roundUp(sz, (size_t)16);
    }
}

} // namespace

StackDeviceMemory::Stack::Stack(GpuResources* res, int d, size_t sz)
        : res_(res),
          device_(d),
          alloc_(nullptr),
          allocSize_(adjustStackSize(sz)),
          start_(nullptr),
          end_(nullptr),
          head_(nullptr),
          highWaterMemoryUsed_(0) {
    if (allocSize_ == 0) {
        return;
    }

    DeviceScope s(device_);
    auto req = AllocRequest(
            AllocType::TemporaryMemoryBuffer,
            device_,
            MemorySpace::Device,
            res_->getDefaultStream(device_),
            allocSize_);

    alloc_ = (char*)res_->allocMemory(req);
    FAISS_ASSERT_FMT(
            alloc_,
            "could not reserve temporary memory region of size %zu",
            allocSize_);

    // In order to disambiguate between our entire region of temporary memory
    // versus the first allocation in the temporary memory region, ensure that
    // the first address returned is +16 bytes from the beginning
    start_ = alloc_ + 16;
    head_ = start_;
    end_ = alloc_ + allocSize_;
}

StackDeviceMemory::Stack::~Stack() {
    DeviceScope s(device_);

    // FIXME: make sure there are no outstanding memory allocations?
    if (alloc_) {
        res_->deallocMemory(device_, alloc_);
    }
}

size_t StackDeviceMemory::Stack::getSizeAvailable() const {
    return (end_ - head_);
}

char* StackDeviceMemory::Stack::getAlloc(size_t size, cudaStream_t stream) {
    // The user must check to see that the allocation fit within us
    auto sizeRemaining = getSizeAvailable();

    FAISS_ASSERT(size <= sizeRemaining);

    // We can make the allocation out of our stack
    // Find all the ranges that we overlap that may have been
    // previously allocated; our allocation will be [head, endAlloc)
    char* startAlloc = head_;
    char* endAlloc = head_ + size;

    while (lastUsers_.size() > 0) {
        auto& prevUser = lastUsers_.back();

        // Because there is a previous user, we must overlap it
        FAISS_ASSERT(
                prevUser.start_ <= endAlloc && prevUser.end_ >= startAlloc);

        if (stream != prevUser.stream_) {
            // Synchronization required
            streamWait({stream}, {prevUser.stream_});
        }

        if (endAlloc < prevUser.end_) {
            // Update the previous user info
            prevUser.start_ = endAlloc;

            break;
        }

        // If we're the exact size of the previous request, then we
        // don't need to continue
        bool done = (prevUser.end_ == endAlloc);

        lastUsers_.pop_back();

        if (done) {
            break;
        }
    }

    head_ = endAlloc;
    FAISS_ASSERT(head_ <= end_);

    highWaterMemoryUsed_ =
            std::max(highWaterMemoryUsed_, (size_t)(head_ - start_));
    FAISS_ASSERT(startAlloc);
    return startAlloc;
}

void StackDeviceMemory::Stack::returnAlloc(
        char* p,
        size_t size,
        cudaStream_t stream) {
    // This allocation should be within ourselves
    FAISS_ASSERT(p >= start_ && p < end_);

    // All allocations should have been adjusted to a multiple of 16 bytes
    FAISS_ASSERT(size % 16 == 0);

    // This is on our stack
    // Allocations should be freed in the reverse order they are made
    if (p + size != head_) {
        FAISS_ASSERT(p + size == head_);
    }

    head_ = p;
    lastUsers_.push_back(Range(p, p + size, stream));
}

std::string StackDeviceMemory::Stack::toString() const {
    std::stringstream s;

    s << "SDM device " << device_ << ": Total memory " << allocSize_ << " ["
      << (void*)start_ << ", " << (void*)end_ << ")\n";
    s << "     Available memory " << (size_t)(end_ - head_) << " ["
      << (void*)head_ << ", " << (void*)end_ << ")\n";
    s << "     High water temp alloc " << highWaterMemoryUsed_ << "\n";

    int i = lastUsers_.size();
    for (auto it = lastUsers_.rbegin(); it != lastUsers_.rend(); ++it) {
        s << i-- << ": size " << (size_t)(it->end_ - it->start_) << " stream "
          << it->stream_ << " [" << (void*)it->start_ << ", " << (void*)it->end_
          << ")\n";
    }

    return s.str();
}

StackDeviceMemory::StackDeviceMemory(
        GpuResources* res,
        int device,
        size_t allocPerDevice)
        : device_(device), stack_(res, device, allocPerDevice) {}

StackDeviceMemory::~StackDeviceMemory() {}

int StackDeviceMemory::getDevice() const {
    return device_;
}

size_t StackDeviceMemory::getSizeAvailable() const {
    return stack_.getSizeAvailable();
}

std::string StackDeviceMemory::toString() const {
    return stack_.toString();
}

void* StackDeviceMemory::allocMemory(cudaStream_t stream, size_t size) {
    // All allocations should have been adjusted to a multiple of 16 bytes
    FAISS_ASSERT(size % 16 == 0);
    return stack_.getAlloc(size, stream);
}

void StackDeviceMemory::deallocMemory(
        int device,
        cudaStream_t stream,
        size_t size,
        void* p) {
    FAISS_ASSERT(p);
    FAISS_ASSERT(device == device_);

    stack_.returnAlloc((char*)p, size, stream);
}

} // namespace gpu
} // namespace faiss
