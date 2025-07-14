#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/metal/MetalIndexIVFFlat.h>
#include <faiss/metal/MetalIndexIVFPQ.h>
#include <faiss/metal/MetalResources.h>

using namespace std;
using namespace std::chrono;

void test_ivfflat() {
    cout << "\n=== Testing MetalIndexIVFFlat ===" << endl;
    
    const int d = 128;       // dimension
    const int nb = 10000;    // database size
    const int nq = 100;      // number of queries
    const int k = 10;        // k nearest neighbors
    const int nlist = 100;   // number of inverted lists
    
    // Generate random data
    std::mt19937 rng(42);
    std::normal_distribution<float> distrib(0.0, 1.0);
    
    vector<float> database(nb * d);
    vector<float> queries(nq * d);
    
    for (int i = 0; i < nb * d; ++i) {
        database[i] = distrib(rng);
    }
    
    for (int i = 0; i < nq * d; ++i) {
        queries[i] = distrib(rng);
    }
    
    // Create quantizer
    faiss::IndexFlatL2 quantizer(d);
    
    // CPU IVFFlat
    cout << "Testing CPU IVFFlat..." << endl;
    faiss::IndexIVFFlat cpu_index(&quantizer, d, nlist);
    cpu_index.train(nb, database.data());
    cpu_index.add(nb, database.data());
    cpu_index.nprobe = 10;
    
    vector<float> cpu_distances(nq * k);
    vector<faiss::idx_t> cpu_labels(nq * k);
    
    auto start = high_resolution_clock::now();
    cpu_index.search(nq, queries.data(), k, cpu_distances.data(), cpu_labels.data());
    auto cpu_time = duration_cast<microseconds>(high_resolution_clock::now() - start);
    
    cout << "CPU search time: " << cpu_time.count() / 1000.0 << " ms" << endl;
    
    // Metal IVFFlat
    cout << "\nTesting Metal IVFFlat..." << endl;
    auto resources = faiss::metal::get_default_metal_resources();
    
    // Reset quantizer
    faiss::IndexFlatL2 metal_quantizer(d);
    cout << "Creating Metal IVF index..." << endl;
    faiss::metal::MetalIndexIVFFlat metal_index(resources, &metal_quantizer, d, nlist);
    cout << "Training..." << endl;
    metal_index.train(nb, database.data());
    cout << "Adding vectors..." << endl;
    metal_index.add(nb, database.data());
    cout << "Setting nprobe..." << endl;
    metal_index.nprobe = 10;
    
    vector<float> metal_distances(nq * k);
    vector<faiss::idx_t> metal_labels(nq * k);
    
    start = high_resolution_clock::now();
    metal_index.search(nq, queries.data(), k, metal_distances.data(), metal_labels.data());
    auto metal_time = duration_cast<microseconds>(high_resolution_clock::now() - start);
    
    cout << "Metal search time: " << metal_time.count() / 1000.0 << " ms" << endl;
    cout << "Speedup: " << (float)cpu_time.count() / metal_time.count() << "x" << endl;
    
    // Verify results
    int matches = 0;
    for (int i = 0; i < nq; ++i) {
        bool query_match = false;
        for (int j = 0; j < k; ++j) {
            for (int l = 0; l < k; ++l) {
                if (cpu_labels[i * k + j] == metal_labels[i * k + l]) {
                    query_match = true;
                    break;
                }
            }
            if (query_match) break;
        }
        if (query_match) matches++;
    }
    
    float accuracy = 100.0f * matches / nq;
    cout << "Accuracy: " << accuracy << "%" << endl;
    
    if (accuracy < 90.0f) {
        cout << "WARNING: Accuracy is lower than expected!" << endl;
    }
}

void test_ivfpq() {
    cout << "\n=== Testing MetalIndexIVFPQ ===" << endl;
    
    const int d = 128;       // dimension
    const int nb = 10000;    // database size
    const int nq = 100;      // number of queries
    const int k = 10;        // k nearest neighbors
    const int nlist = 100;   // number of inverted lists
    const int M = 8;         // number of subquantizers
    const int nbits = 8;     // bits per subquantizer
    
    // Generate random data
    std::mt19937 rng(42);
    std::normal_distribution<float> distrib(0.0, 1.0);
    
    vector<float> database(nb * d);
    vector<float> queries(nq * d);
    
    for (int i = 0; i < nb * d; ++i) {
        database[i] = distrib(rng);
    }
    
    for (int i = 0; i < nq * d; ++i) {
        queries[i] = distrib(rng);
    }
    
    // Create quantizer
    faiss::IndexFlatL2 quantizer(d);
    
    // CPU IVFPQ
    cout << "Testing CPU IVFPQ..." << endl;
    faiss::IndexIVFPQ cpu_index(&quantizer, d, nlist, M, nbits);
    cpu_index.train(nb, database.data());
    cpu_index.add(nb, database.data());
    cpu_index.nprobe = 10;
    
    vector<float> cpu_distances(nq * k);
    vector<faiss::idx_t> cpu_labels(nq * k);
    
    auto start = high_resolution_clock::now();
    cpu_index.search(nq, queries.data(), k, cpu_distances.data(), cpu_labels.data());
    auto cpu_time = duration_cast<microseconds>(high_resolution_clock::now() - start);
    
    cout << "CPU search time: " << cpu_time.count() / 1000.0 << " ms" << endl;
    
    // Metal IVFPQ
    cout << "\nTesting Metal IVFPQ..." << endl;
    auto resources = faiss::metal::get_default_metal_resources();
    
    // Reset quantizer
    faiss::IndexFlatL2 metal_quantizer(d);
    faiss::metal::MetalIndexIVFPQ metal_index(resources, &metal_quantizer, d, nlist, M, nbits);
    metal_index.train(nb, database.data());
    metal_index.add(nb, database.data());
    metal_index.nprobe = 10;
    
    vector<float> metal_distances(nq * k);
    vector<faiss::idx_t> metal_labels(nq * k);
    
    start = high_resolution_clock::now();
    metal_index.search(nq, queries.data(), k, metal_distances.data(), metal_labels.data());
    auto metal_time = duration_cast<microseconds>(high_resolution_clock::now() - start);
    
    cout << "Metal search time: " << metal_time.count() / 1000.0 << " ms" << endl;
    cout << "Speedup: " << (float)cpu_time.count() / metal_time.count() << "x" << endl;
    
    // Verify results
    int matches = 0;
    for (int i = 0; i < nq; ++i) {
        bool query_match = false;
        for (int j = 0; j < k; ++j) {
            for (int l = 0; l < k; ++l) {
                if (cpu_labels[i * k + j] == metal_labels[i * k + l]) {
                    query_match = true;
                    break;
                }
            }
            if (query_match) break;
        }
        if (query_match) matches++;
    }
    
    float accuracy = 100.0f * matches / nq;
    cout << "Accuracy: " << accuracy << "%" << endl;
    
    if (accuracy < 80.0f) {  // Lower threshold for PQ due to quantization
        cout << "WARNING: Accuracy is lower than expected!" << endl;
    }
}

int main() {
    cout << "=== Metal IVF Index Test ===" << endl;
    
    try {
        test_ivfflat();
        test_ivfpq();
        
        cout << "\nAll tests completed!" << endl;
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}