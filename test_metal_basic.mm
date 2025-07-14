#include <iostream>
#include <vector>
#include <faiss/metal/MetalResources.h>
#include <faiss/metal/MetalIndexFlat.h>
#include <faiss/IndexFlat.h>

int main() {
    std::cout << "Testing basic Metal functionality..." << std::endl;
    
    try {
        // Get Metal resources
        auto resources = faiss::metal::get_default_metal_resources();
        std::cout << "✓ Metal resources created" << std::endl;
        
        // Create MetalKernels to test library loading
        faiss::metal::MetalKernels kernels(resources);
        std::cout << "✓ Metal kernels loaded" << std::endl;
        
        // Test with MetalIndexFlat which we know works
        const int d = 64;
        const int nb = 100;
        const int nq = 5;
        const int k = 3;
        
        std::vector<float> database(nb * d);
        std::vector<float> queries(nq * d);
        
        // Generate simple test data
        for (int i = 0; i < nb * d; ++i) {
            database[i] = (float)(i % 100) / 100.0f;
        }
        for (int i = 0; i < nq * d; ++i) {
            queries[i] = (float)(i % 100) / 100.0f;
        }
        
        // Test MetalIndexFlat
        faiss::metal::MetalIndexFlat metal_index(resources, d, faiss::METRIC_L2);
        metal_index.add(nb, database.data());
        std::cout << "✓ Added " << nb << " vectors to Metal index" << std::endl;
        
        std::vector<float> distances(nq * k);
        std::vector<faiss::idx_t> labels(nq * k);
        
        metal_index.search(nq, queries.data(), k, distances.data(), labels.data());
        std::cout << "✓ Search completed" << std::endl;
        
        // Show results
        std::cout << "\nFirst query results:" << std::endl;
        for (int i = 0; i < k; i++) {
            std::cout << "  " << labels[i] << " (distance: " << distances[i] << ")" << std::endl;
        }
        
        std::cout << "\n✓ All tests passed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}