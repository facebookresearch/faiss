#include "MetalMemoryPool.h"

namespace faiss {
namespace metal {

MetalMemoryPool::MetalMemoryPool(std::shared_ptr<MetalResources> resources)
    : resources_(resources) {}

MetalMemoryPool::~MetalMemoryPool() {
    for (auto& buffer : pool_) {
        [buffer release];
    }
}

id<MTLBuffer> MetalMemoryPool::alloc(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto it = pool_.begin(); it != pool_.end(); ++it) {
        if ([*it length] >= size) {
            id<MTLBuffer> buffer = *it;
            pool_.erase(it);
            return buffer;
        }
    }

    id<MTLDevice> device = resources_->getDevice(0);
    return [device newBufferWithLength:size options:MTLResourceStorageModeShared];
}

void MetalMemoryPool::release(id<MTLBuffer> buffer) {
    std::lock_guard<std::mutex> lock(mutex_);
    pool_.push_back(buffer);
}

} // namespace metal
} // namespace faiss
