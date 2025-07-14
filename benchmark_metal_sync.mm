#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <memory>
#include <faiss/MetricType.h>
#include <faiss/IndexFlat.h>
#include <faiss/metal/MetalIndexFlat.h>
#include <faiss/metal/MetalBufferPool.h>

using namespace std;
using namespace std::chrono;

void benchmark_synchronization() {
    const int d = 128;     // dimension
    const int nb = 10000;  // database size
    const int nq = 1000;   // number of queries
    const int k = 10;      // k nearest neighbors
    
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
    
    // CPU baseline
    faiss::IndexFlatL2 cpu_index(d);
    cpu_index.add(nb, database.data());
    
    // Metal index
    faiss::metal::MetalIndexFlat metal_index(resources, d, faiss::METRIC_L2);
    metal_index.add(nb, database.data());
    
    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    
    // Benchmark CPU
    cout << "Benchmarking CPU IndexFlat..." << endl;
    auto start = high_resolution_clock::now();
    cpu_index.search(nq, queries.data(), k, distances.data(), labels.data());
    auto cpu_time = high_resolution_clock::now() - start;
    cout << "CPU time: " << duration_cast<milliseconds>(cpu_time).count() << " ms" << endl;
    
    // Benchmark Metal (current synchronous version)
    cout << "\nBenchmarking Metal IndexFlat (synchronous)..." << endl;
    start = high_resolution_clock::now();
    metal_index.search(nq, queries.data(), k, distances.data(), labels.data());
    auto metal_sync_time = high_resolution_clock::now() - start;
    cout << "Metal sync time: " << duration_cast<milliseconds>(metal_sync_time).count() << " ms" << endl;
    
    // Calculate speedup
    double speedup = (double)duration_cast<microseconds>(cpu_time).count() / 
                     duration_cast<microseconds>(metal_sync_time).count();
    cout << "Speedup: " << speedup << "x" << endl;
    
    // Test buffer pool performance
    cout << "\nTesting buffer pool..." << endl;
    id<MTLDevice> device = resources->getDevice(0);
    auto pool = std::make_shared<faiss::metal::MetalBufferPool>(device, d * sizeof(float), 50);
    
    // Benchmark allocations without pool
    start = high_resolution_clock::now();
    std::vector<id<MTLBuffer>> buffers1;
    for (int i = 0; i < 1000; ++i) {
        buffers1.push_back([device newBufferWithLength:d * sizeof(float)
                                               options:MTLResourceStorageModeShared]);
    }
    auto alloc_time = high_resolution_clock::now() - start;
    cout << "Direct allocation time (1000 buffers): " 
         << duration_cast<microseconds>(alloc_time).count() << " µs" << endl;
    
    // Benchmark allocations with pool
    start = high_resolution_clock::now();
    std::vector<faiss::metal::PooledBuffer> buffers2;
    for (int i = 0; i < 1000; ++i) {
        buffers2.emplace_back(pool);
    }
    auto pool_time = high_resolution_clock::now() - start;
    cout << "Pool allocation time (1000 buffers): " 
         << duration_cast<microseconds>(pool_time).count() << " µs" << endl;
    
    double pool_speedup = (double)duration_cast<microseconds>(alloc_time).count() / 
                          duration_cast<microseconds>(pool_time).count();
    cout << "Pool speedup: " << pool_speedup << "x" << endl;
}

int main() {
    cout << "Metal Synchronization Benchmark" << endl;
    cout << "===============================" << endl;
    
    try {
        benchmark_synchronization();
    } catch (const std::exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}