#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <faiss/metal/MetalIndexFlat.h>
#include <faiss/metal/MetalResources.h>
#include <faiss/IndexFlat.h>
#include <faiss/utils/random.h>

// Concrete implementation of MetalResources for testing
class StandardMetalResources : public faiss::metal::MetalResources {
public:
    StandardMetalResources() {
        device_ = MTLCreateSystemDefaultDevice();
        if (device_) {
            commandQueue_ = [device_ newCommandQueue];
        }
    }
    
    ~StandardMetalResources() override = default;
    
    void initializeForDevice(int device) override {
        // For now, we only support the default device
    }
    
    id<MTLDevice> getDevice(int device) override {
        return device_;
    }
    
    id<MTLCommandQueue> getCommandQueue(int device) override {
        return commandQueue_;
    }
    
    void* allocMemory(size_t size) override {
        id<MTLBuffer> buffer = [device_ newBufferWithLength:size
                                                     options:MTLResourceStorageModeShared];
        return (__bridge_retained void*)buffer;
    }
    
    void deallocMemory(void* ptr) override {
        id<MTLBuffer> buffer = (__bridge_transfer id<MTLBuffer>)ptr;
        buffer = nil;
    }
    
private:
    id<MTLDevice> device_;
    id<MTLCommandQueue> commandQueue_;
};

int main() {
    std::cout << "=== Faiss Metal IndexFlat Test ===" << std::endl;
    
    // Check if Metal is available
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        std::cout << "Metal is not available on this system" << std::endl;
        return 1;
    }
    std::cout << "Metal device: " << [[device name] UTF8String] << std::endl;
    
    try {
        int d = 128;   // dimension
        int nb = 1000; // database size
        int nq = 5;    // nb of queries
        int k = 4;     // nb of neighbors
        
        // Create resources
        auto resources = std::make_shared<StandardMetalResources>();
        
        // Generate random vectors
        std::vector<float> xb(nb * d);
        std::vector<float> xq(nq * d);
        
        faiss::float_rand(xb.data(), nb * d, 1234);
        faiss::float_rand(xq.data(), nq * d, 5678);
        
        // Test L2 distance
        std::cout << "\n--- Testing L2 Distance ---" << std::endl;
        {
            // CPU version
            faiss::IndexFlatL2 cpu_index(d);
            cpu_index.add(nb, xb.data());
            
            std::vector<float> cpu_distances(nq * k);
            std::vector<faiss::idx_t> cpu_labels(nq * k);
            cpu_index.search(nq, xq.data(), k, cpu_distances.data(), cpu_labels.data());
            
            // Metal version
            faiss::metal::MetalIndexFlat metal_index(resources, d, faiss::METRIC_L2);
            metal_index.add(nb, xb.data());
            
            std::vector<float> metal_distances(nq * k);
            std::vector<faiss::idx_t> metal_labels(nq * k);
            metal_index.search(nq, xq.data(), k, metal_distances.data(), metal_labels.data());
            
            // Compare results
            std::cout << "First query results:" << std::endl;
            std::cout << "CPU:   ";
            for (int i = 0; i < k; i++) {
                std::cout << cpu_labels[i] << "(" << cpu_distances[i] << ") ";
            }
            std::cout << std::endl;
            
            std::cout << "Metal: ";
            for (int i = 0; i < k; i++) {
                std::cout << metal_labels[i] << "(" << metal_distances[i] << ") ";
            }
            std::cout << std::endl;
            
            // Check accuracy
            int matches = 0;
            for (int q = 0; q < nq; q++) {
                for (int i = 0; i < k; i++) {
                    for (int j = 0; j < k; j++) {
                        if (cpu_labels[q * k + i] == metal_labels[q * k + j]) {
                            matches++;
                            break;
                        }
                    }
                }
            }
            float accuracy = (float)matches / (nq * k);
            std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;
            
            if (accuracy > 0.9) {
                std::cout << "✓ L2 distance test PASSED" << std::endl;
            } else {
                std::cout << "✗ L2 distance test FAILED" << std::endl;
            }
        }
        
        // Test Inner Product
        std::cout << "\n--- Testing Inner Product Distance ---" << std::endl;
        {
            // Normalize vectors for inner product
            for (int i = 0; i < nb; i++) {
                float norm = 0;
                for (int j = 0; j < d; j++) {
                    norm += xb[i * d + j] * xb[i * d + j];
                }
                norm = std::sqrt(norm);
                for (int j = 0; j < d; j++) {
                    xb[i * d + j] /= norm;
                }
            }
            
            // CPU version
            faiss::IndexFlatIP cpu_index(d);
            cpu_index.add(nb, xb.data());
            
            std::vector<float> cpu_distances(nq * k);
            std::vector<faiss::idx_t> cpu_labels(nq * k);
            cpu_index.search(nq, xq.data(), k, cpu_distances.data(), cpu_labels.data());
            
            // Metal version
            faiss::metal::MetalIndexFlat metal_index(resources, d, faiss::METRIC_INNER_PRODUCT);
            metal_index.add(nb, xb.data());
            
            std::vector<float> metal_distances(nq * k);
            std::vector<faiss::idx_t> metal_labels(nq * k);
            metal_index.search(nq, xq.data(), k, metal_distances.data(), metal_labels.data());
            
            // Compare results
            std::cout << "First query results:" << std::endl;
            std::cout << "CPU:   ";
            for (int i = 0; i < k; i++) {
                std::cout << cpu_labels[i] << "(" << cpu_distances[i] << ") ";
            }
            std::cout << std::endl;
            
            std::cout << "Metal: ";
            for (int i = 0; i < k; i++) {
                std::cout << metal_labels[i] << "(" << metal_distances[i] << ") ";
            }
            std::cout << std::endl;
            
            // Check accuracy
            int matches = 0;
            for (int q = 0; q < nq; q++) {
                for (int i = 0; i < k; i++) {
                    for (int j = 0; j < k; j++) {
                        if (cpu_labels[q * k + i] == metal_labels[q * k + j]) {
                            matches++;
                            break;
                        }
                    }
                }
            }
            float accuracy = (float)matches / (nq * k);
            std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;
            
            if (accuracy > 0.9) {
                std::cout << "✓ Inner product test PASSED" << std::endl;
            } else {
                std::cout << "✗ Inner product test FAILED" << std::endl;
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}