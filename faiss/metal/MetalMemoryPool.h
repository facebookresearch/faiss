#pragma once

#include "MetalResources.h"
#include <list>
#include <mutex>

namespace faiss {
namespace metal {

class MetalMemoryPool {
   public:
    MetalMemoryPool(std::shared_ptr<MetalResources> resources);
    ~MetalMemoryPool();

    id<MTLBuffer> alloc(size_t size);

    void release(id<MTLBuffer> buffer);

   private:
    std::shared_ptr<MetalResources> resources_;
    std::list<id<MTLBuffer>> pool_;
    std::mutex mutex_;
};

} // namespace metal
} // namespace faiss
