#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <memory>
#include <faiss/MetricType.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexFlat.h>
#include <faiss/metal/MetalIndexHNSW.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

using namespace std;

void test_hnsw_implementation() {
    const int d = 128;  // dimension
    const int nb = 100;   // database size (small for debugging)
    const int nq = 5;     // number of queries (small for debugging)
    const int k = 10;     // k nearest neighbors
    const int M = 32;     // HNSW parameter
    
    // Generate random data
    std::mt19937 rng(42);
    std::normal_distribution<float> distrib(0.0, 1.0);
    
    std::vector<float> database(nb * d);
    std::vector<float> queries(nq * d);
    
    for (int i = 0; i < nb * d; ++i) {
        database[i] = distrib(rng);
    }
    
    for (int i = 0; i < nq * d; ++i) {
        queries[i] = distrib(rng);
    }
    
    // Create indices
    auto resources = faiss::metal::get_default_metal_resources();
    
    // CPU HNSW
    faiss::IndexFlatL2 quantizer(d);
    faiss::IndexHNSW cpu_index(&quantizer, M);
    cpu_index.own_fields = false;  // Don't delete the quantizer
    cpu_index.hnsw.efConstruction = 200;
    cpu_index.hnsw.efSearch = 64;
    
    // Metal HNSW
    faiss::metal::MetalIndexHNSW metal_index(resources, d, M, faiss::METRIC_L2);
    metal_index.hnsw.efConstruction = 200;
    metal_index.hnsw.efSearch = 64;
    
    cout << "Building HNSW indices..." << endl;
    
    // Add vectors
    auto start = chrono::high_resolution_clock::now();
    cpu_index.add(nb, database.data());
    auto cpu_build_time = chrono::high_resolution_clock::now() - start;
    cout << "CPU build time: " << chrono::duration_cast<chrono::milliseconds>(cpu_build_time).count() << " ms" << endl;
    
    start = chrono::high_resolution_clock::now();
    metal_index.add(nb, database.data());
    auto metal_build_time = chrono::high_resolution_clock::now() - start;
    cout << "Metal build time: " << chrono::duration_cast<chrono::milliseconds>(metal_build_time).count() << " ms" << endl;
    
    // Search
    std::vector<float> cpu_distances(nq * k);
    std::vector<faiss::idx_t> cpu_labels(nq * k);
    std::vector<float> metal_distances(nq * k);
    std::vector<faiss::idx_t> metal_labels(nq * k);
    
    cout << "\nSearching..." << endl;
    
    // CPU search
    start = chrono::high_resolution_clock::now();
    cpu_index.search(nq, queries.data(), k, cpu_distances.data(), cpu_labels.data());
    auto cpu_search_time = chrono::high_resolution_clock::now() - start;
    cout << "CPU search time: " << chrono::duration_cast<chrono::milliseconds>(cpu_search_time).count() << " ms" << endl;
    
    // Metal search
    start = chrono::high_resolution_clock::now();
    metal_index.search(nq, queries.data(), k, metal_distances.data(), metal_labels.data());
    auto metal_search_time = chrono::high_resolution_clock::now() - start;
    cout << "Metal search time: " << chrono::duration_cast<chrono::milliseconds>(metal_search_time).count() << " ms" << endl;
    
    // Verify results
    cout << "\nVerifying results..." << endl;
    int correct = 0;
    for (int q = 0; q < nq; q++) {
        for (int i = 0; i < k; i++) {
            bool found = false;
            for (int j = 0; j < k; j++) {
                if (cpu_labels[q * k + i] == metal_labels[q * k + j]) {
                    found = true;
                    break;
                }
            }
            if (found) correct++;
        }
    }
    
    float accuracy = 100.0f * correct / (nq * k);
    cout << "Accuracy: " << accuracy << "%" << endl;
    
    // Show some results
    cout << "\nFirst query results comparison:" << endl;
    cout << "CPU:   ";
    for (int i = 0; i < min(5, k); i++) {
        cout << cpu_labels[i] << "(" << cpu_distances[i] << ") ";
    }
    cout << endl;
    
    cout << "Metal: ";
    for (int i = 0; i < min(5, k); i++) {
        cout << metal_labels[i] << "(" << metal_distances[i] << ") ";
    }
    cout << endl;
    
    if (accuracy > 90) {
        cout << "\n✓ HNSW Metal test PASSED" << endl;
    } else {
        cout << "\n✗ HNSW Metal test FAILED" << endl;
    }
}

int main() {
    cout << "Testing Metal HNSW implementation..." << endl;
    test_hnsw_implementation();
    return 0;
}