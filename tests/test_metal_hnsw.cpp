#include <gtest/gtest.h>
#include <faiss/metal/MetalIndexHNSW.h>
#include <faiss/metal/MetalResources.h>
#include <faiss/IndexFlat.h>
#include <faiss/utils/random.h>
#include <memory>
#include <vector>

TEST(MetalIndexHNSW, BasicAddSearch) {
    int d = 128;  // dimension
    int nb = 1000; // database size
    int nq = 10;   // nb of queries
    int k = 5;     // nb of neighbors
    int M = 32;    // HNSW parameter

    // Create a standard resources provider
    class StandardMetalResourcesProvider : public faiss::metal::MetalResourcesProvider {
    public:
        std::shared_ptr<faiss::metal::MetalResources> getResources() override {
            // Return a concrete implementation of MetalResources
            // This is a placeholder - the actual implementation would be in MetalResources.mm
            return nullptr;
        }
    };

    auto provider = std::make_unique<StandardMetalResourcesProvider>();
    auto resources = provider->getResources();
    
    if (!resources) {
        GTEST_SKIP() << "Metal resources not available on this platform";
    }

    // Create Metal HNSW index
    faiss::metal::MetalIndexHNSW index(resources, d, M);

    // Generate random vectors
    std::vector<float> xb(nb * d);
    std::vector<float> xq(nq * d);
    
    faiss::float_rand(xb.data(), nb * d, 1234);
    faiss::float_rand(xq.data(), nq * d, 5678);

    // Add vectors to index
    index.add(nb, xb.data());

    // Search
    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    // Basic sanity checks
    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < k; j++) {
            // Check that labels are valid indices
            ASSERT_GE(labels[i * k + j], 0);
            ASSERT_LT(labels[i * k + j], nb);
            
            // Check that distances are non-negative (for L2)
            ASSERT_GE(distances[i * k + j], 0);
        }
    }
}

TEST(MetalIndexHNSW, CompareWithCPU) {
    int d = 64;
    int nb = 500;
    int nq = 5;
    int k = 10;
    int M = 16;

    // Create a standard resources provider
    class StandardMetalResourcesProvider : public faiss::metal::MetalResourcesProvider {
    public:
        std::shared_ptr<faiss::metal::MetalResources> getResources() override {
            return nullptr;
        }
    };

    auto provider = std::make_unique<StandardMetalResourcesProvider>();
    auto resources = provider->getResources();
    
    if (!resources) {
        GTEST_SKIP() << "Metal resources not available on this platform";
    }

    // Create both CPU and Metal indices
    faiss::IndexHNSW cpu_index(d, M);
    faiss::metal::MetalIndexHNSW metal_index(resources, d, M);

    // Generate random vectors
    std::vector<float> xb(nb * d);
    std::vector<float> xq(nq * d);
    
    faiss::float_rand(xb.data(), nb * d, 1234);
    faiss::float_rand(xq.data(), nq * d, 5678);

    // Add to both indices
    cpu_index.add(nb, xb.data());
    metal_index.add(nb, xb.data());

    // Search on both
    std::vector<float> cpu_distances(nq * k);
    std::vector<faiss::idx_t> cpu_labels(nq * k);
    std::vector<float> metal_distances(nq * k);
    std::vector<faiss::idx_t> metal_labels(nq * k);
    
    cpu_index.search(nq, xq.data(), k, cpu_distances.data(), cpu_labels.data());
    metal_index.search(nq, xq.data(), k, metal_distances.data(), metal_labels.data());

    // Compare results - they should be similar but might not be identical
    // due to different traversal orders in parallel execution
    for (int i = 0; i < nq; i++) {
        // Check that at least 50% of the neighbors are the same
        int common_neighbors = 0;
        for (int j = 0; j < k; j++) {
            for (int l = 0; l < k; l++) {
                if (cpu_labels[i * k + j] == metal_labels[i * k + l]) {
                    common_neighbors++;
                    break;
                }
            }
        }
        ASSERT_GE(common_neighbors, k / 2);
    }
}