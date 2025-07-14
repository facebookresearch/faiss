#pragma once

#include <Metal/Metal.h>
#include <queue>
#include <mutex>
#include <memory>

namespace faiss {
namespace metal {

// Buffer pool to reduce allocation overhead
class MetalBufferPool {
public:
    MetalBufferPool(id<MTLDevice> device, size_t buffer_size, size_t initial_count = 10)
        : device_(device), buffer_size_(buffer_size) {
        // Pre-allocate buffers
        for (size_t i = 0; i < initial_count; ++i) {
            id<MTLBuffer> buffer = [device_ newBufferWithLength:buffer_size_
                                                        options:MTLResourceStorageModeShared];
            available_buffers_.push(buffer);
        }
    }
    
    id<MTLBuffer> getBuffer() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (available_buffers_.empty()) {
            // Allocate new buffer on demand
            return [device_ newBufferWithLength:buffer_size_
                                       options:MTLResourceStorageModeShared];
        }
        
        id<MTLBuffer> buffer = available_buffers_.front();
        available_buffers_.pop();
        return buffer;
    }
    
    void returnBuffer(id<MTLBuffer> buffer) {
        std::lock_guard<std::mutex> lock(mutex_);
        available_buffers_.push(buffer);
    }
    
    size_t getBufferSize() const { return buffer_size_; }
    
private:
    id<MTLDevice> device_;
    size_t buffer_size_;
    std::queue<id<MTLBuffer>> available_buffers_;
    std::mutex mutex_;
};

// RAII wrapper for automatic buffer return
class PooledBuffer {
public:
    PooledBuffer(std::shared_ptr<MetalBufferPool> pool)
        : pool_(pool), buffer_(pool->getBuffer()) {}
    
    ~PooledBuffer() {
        if (buffer_ && pool_) {
            pool_->returnBuffer(buffer_);
        }
    }
    
    // Move constructor
    PooledBuffer(PooledBuffer&& other) noexcept
        : pool_(std::move(other.pool_)), buffer_(other.buffer_) {
        other.buffer_ = nil;
    }
    
    // Delete copy operations
    PooledBuffer(const PooledBuffer&) = delete;
    PooledBuffer& operator=(const PooledBuffer&) = delete;
    
    id<MTLBuffer> get() const { return buffer_; }
    operator id<MTLBuffer>() const { return buffer_; }
    
private:
    std::shared_ptr<MetalBufferPool> pool_;
    id<MTLBuffer> buffer_;
};

} // namespace metal
} // namespace faiss