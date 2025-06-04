/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c_api/faiss_c.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <random>

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIDMap.h>
#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/GpuIndexCagra.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/index_factory.h>
#include <faiss/gpu/utils/CopyUtils.cuh>

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/stats/neighborhood_recall.cuh>

using idx_t = faiss::idx_t;
std::string data_path = "/path/to/data/sift";

bool read_fvecs_file(
        const std::string& filepath,
        std::vector<float>& out_data,
        size_t& out_num_vecs,
        size_t& out_dim) {
    std::ifstream input(filepath, std::ios::binary);
    if (!input) {
        std::cerr << "Cannot open file: " << filepath << "\n";
        return false;
    }

    // Read dimension of first vector
    int dim = 0;
    input.read(reinterpret_cast<char*>(&dim), sizeof(int));

    // Calculate number of vectors
    input.seekg(0, std::ios::end);
    size_t file_size = input.tellg();
    size_t entry_size = sizeof(int) + dim * sizeof(float);
    if (file_size % entry_size != 0) {
        std::cerr << "File size not divisible by vector size in: " << filepath
                  << "\n";
        return false;
    }
    size_t num_vecs = file_size / entry_size;

    // Read all vectors
    input.seekg(0, std::ios::beg);
    out_data.resize(num_vecs * dim);

    for (size_t i = 0; i < num_vecs; ++i) {
        int cur_dim;
        input.read(reinterpret_cast<char*>(&cur_dim), sizeof(int));
        if (cur_dim != dim) {
            std::cerr << "Inconsistent vector dimension at index " << i
                      << " in file: " << filepath << "\n";
            return false;
        }
        input.read(
                reinterpret_cast<char*>(&out_data[i * dim]),
                dim * sizeof(float));
    }

    out_dim = static_cast<size_t>(dim);
    out_num_vecs = num_vecs;
    return true;
}

int main() {
    // === Params ===
    size_t graph_degree = 32;
    size_t intermediate_graph_degree = 64;
    auto metric = faiss::METRIC_L2;
    auto build_algo = faiss::gpu::graph_build_algo::NN_DESCENT;
    size_t k = 10;

    faiss::gpu::StandardGpuResources res;
    res.noTempMemory();
    auto gpuRes = res.getResources();

    // === Load train and query vectors for fp32 and also convert to fp16 ===
    std::vector<float> trainVecs, queryVecs;
    size_t n_train, n_query, n_dim;

    read_fvecs_file(data_path + "/sift_base.fvecs", trainVecs, n_train, n_dim);
    read_fvecs_file(data_path + "/sift_query.fvecs", queryVecs, n_query, n_dim);

    std::cout << "Train: " << n_train << " x " << n_dim << std::endl;
    std::cout << "Query: " << n_query << " x " << n_dim << std::endl;

    std::vector<__half> trainVecs_half(trainVecs.size());
    std::vector<__half> queryVecs_half(queryVecs.size());

    for (size_t i = 0; i < trainVecs.size(); ++i) {
        trainVecs_half[i] = __float2half(trainVecs[i]);
    }
    for (size_t i = 0; i < queryVecs.size(); ++i) {
        queryVecs_half[i] = __float2half(queryVecs[i]);
    }

    // === Train and search brute force (IndexFlat) for comparison ===
    std::cout << "Train and search brute force (IndexFlat) for comparison\n";
    faiss::IndexFlat bruteforce(n_dim, faiss::METRIC_L2);
    bruteforce.add(n_train, trainVecs.data());

    std::vector<float> bf_Distance(n_query * k, 0);
    std::vector<faiss::idx_t> bf_Indices(n_query * k, -1);
    bruteforce.search(
            n_query,
            queryVecs.data(),
            k,
            bf_Distance.data(),
            bf_Indices.data());

    auto bf_DistanceDev = faiss::gpu::toDeviceTemporary(
            gpuRes.get(), bf_Distance, gpuRes->getDefaultStreamCurrentDevice());
    auto bf_IndicesDev = faiss::gpu::toDeviceTemporary(
            gpuRes.get(), bf_Indices, gpuRes->getDefaultStreamCurrentDevice());

    // === Train and search cpu index for comparison ===
    std::cout << "Train and search cpu index (IndexHNSWFlat) for comparison\n";
    faiss::IndexHNSWFlat cpuIndex(n_dim, graph_degree / 2, metric);
    cpuIndex.hnsw.efConstruction = k * 2;
    cpuIndex.add(n_train, trainVecs.data());

    std::vector<float> refDistance(n_query * k, 0);
    std::vector<faiss::idx_t> refIndices(n_query * k, -1);
    faiss::SearchParametersHNSW cpuSearchParams;
    cpuSearchParams.efSearch = k * 2;
    cpuIndex.search(
            n_query,
            queryVecs.data(),
            k,
            refDistance.data(),
            refIndices.data(),
            &cpuSearchParams);

    auto refDistanceDev = faiss::gpu::toDeviceTemporary(
            gpuRes.get(), refDistance, gpuRes->getDefaultStreamCurrentDevice());
    auto refIndicesDev = faiss::gpu::toDeviceTemporary(
            gpuRes.get(), refIndices, gpuRes->getDefaultStreamCurrentDevice());

    // === Train and search gpu index FP16 ===
    std::cout << "Train and search gpu index for FP16\n";
    faiss::gpu::GpuIndexCagraConfig config;
    config.device = 0;
    config.graph_degree = graph_degree;
    config.intermediate_graph_degree = intermediate_graph_degree;
    config.build_algo = build_algo;

    faiss::gpu::GpuIndexCagra gpuIndex(&res, cpuIndex.d, metric, config);
    gpuIndex.train(
            n_train,
            static_cast<void*>(trainVecs_half.data()),
            faiss::NumericType::Float16);

    auto devAlloc = faiss::gpu::makeDevAlloc(
            faiss::gpu::AllocType::FlatData,
            gpuRes->getDefaultStreamCurrentDevice());
    faiss::gpu::DeviceTensor<float, 2, true> testDistance(
            gpuRes.get(),
            devAlloc,
            {static_cast<long>(n_query), static_cast<long>(k)});
    faiss::gpu::DeviceTensor<faiss::idx_t, 2, true> testIndices(
            gpuRes.get(),
            devAlloc,
            {static_cast<long>(n_query), static_cast<long>(k)});

    gpuIndex.search(
            n_query,
            static_cast<void*>(queryVecs_half.data()),
            faiss::NumericType::Float16,
            k,
            testDistance.data(),
            testIndices.data());

    // === Train and search gpu index FP32 ===
    std::cout << "Train and search gpu index for FP32\n";

    faiss::gpu::GpuIndexCagra gpuIndex_fp32(&res, cpuIndex.d, metric, config);
    gpuIndex_fp32.train(n_train, trainVecs.data());

    faiss::gpu::DeviceTensor<float, 2, true> testDistance_fp32(
            gpuRes.get(),
            devAlloc,
            {static_cast<long>(n_query), static_cast<long>(k)});
    faiss::gpu::DeviceTensor<faiss::idx_t, 2, true> testIndices_fp32(
            gpuRes.get(),
            devAlloc,
            {static_cast<long>(n_query), static_cast<long>(k)});
    std::cout << "done training\n";
    gpuIndex_fp32.search(
            n_query,
            queryVecs.data(),
            k,
            testDistance_fp32.data(),
            testIndices_fp32.data());

    // === Compare results ===
    double scalar_init_bf_vs_fp16 = 0;
    auto recall_score_bf_vs_fp16 =
            raft::make_host_scalar(scalar_init_bf_vs_fp16);
    raft::stats::neighborhood_recall(
            gpuRes->getRaftHandleCurrentDevice(),
            raft::make_device_matrix_view<const faiss::idx_t, int>(
                    testIndices.data(), n_query, k), // indices from gpu
            raft::make_device_matrix_view<const faiss::idx_t, int>(
                    bf_IndicesDev.data(), n_query, k), // indices from cpu
            recall_score_bf_vs_fp16.view(),
            std::make_optional(raft::make_device_matrix_view<const float, int>(
                    testDistance.data(), n_query, k)), // distances from gpu
            std::make_optional(raft::make_device_matrix_view<const float, int>(
                    bf_DistanceDev.data(), n_query, k))); // distances from cpu
    std::cout << "Final recall IndexFlat (brute force) vs GpuIndexCagra FP16 "
              << *recall_score_bf_vs_fp16.data_handle() << std::endl;

    double scalar_init_hnsw_vs_fp16 = 0;
    auto recall_score_hnsw_vs_fp16 =
            raft::make_host_scalar(scalar_init_hnsw_vs_fp16);
    raft::stats::neighborhood_recall(
            gpuRes->getRaftHandleCurrentDevice(),
            raft::make_device_matrix_view<const faiss::idx_t, int>(
                    testIndices.data(), n_query, k), // indices from gpu
            raft::make_device_matrix_view<const faiss::idx_t, int>(
                    refIndicesDev.data(), n_query, k), // indices from cpu
            recall_score_hnsw_vs_fp16.view(),
            std::make_optional(raft::make_device_matrix_view<const float, int>(
                    testDistance.data(), n_query, k)), // distances from gpu
            std::make_optional(raft::make_device_matrix_view<const float, int>(
                    refDistanceDev.data(), n_query, k))); // distances from cpu
    std::cout << "Final recall IndexHNSWFlat vs GpuIndexCagra FP16 "
              << *recall_score_hnsw_vs_fp16.data_handle() << std::endl;

    double scalar_init_hnsw_vs_fp32 = 0;
    auto recall_score_hnsw_vs_fp32 =
            raft::make_host_scalar(scalar_init_hnsw_vs_fp32);
    raft::stats::neighborhood_recall(
            gpuRes->getRaftHandleCurrentDevice(),
            raft::make_device_matrix_view<const faiss::idx_t, int>(
                    testIndices_fp32.data(), n_query, k), // indices from gpu
            raft::make_device_matrix_view<const faiss::idx_t, int>(
                    refIndicesDev.data(), n_query, k), // indices from cpu
            recall_score_hnsw_vs_fp32.view(),
            std::make_optional(raft::make_device_matrix_view<const float, int>(
                    testDistance_fp32.data(),
                    n_query,
                    k)), // distances from gpu
            std::make_optional(raft::make_device_matrix_view<const float, int>(
                    refDistanceDev.data(), n_query, k))); // distances from cpu
    std::cout << "Final recall IndexHNSWFlat vs GpuIndexCagra FP32 "
              << *recall_score_hnsw_vs_fp32.data_handle() << std::endl;

    double scalar_init_fp16_vs_fp32 = 0;
    auto recall_score_fp16_vs_fp32 =
            raft::make_host_scalar(scalar_init_fp16_vs_fp32);
    raft::stats::neighborhood_recall(
            gpuRes->getRaftHandleCurrentDevice(),
            raft::make_device_matrix_view<const faiss::idx_t, int>(
                    testIndices_fp32.data(),
                    n_query,
                    k), // indices from gpu fp32
            raft::make_device_matrix_view<const faiss::idx_t, int>(
                    testIndices.data(), n_query, k), // indices from gpu fp16
            recall_score_fp16_vs_fp32.view(),
            std::make_optional(raft::make_device_matrix_view<const float, int>(
                    testDistance_fp32.data(),
                    n_query,
                    k)), // distances from gpu
            std::make_optional(raft::make_device_matrix_view<const float, int>(
                    testDistance.data(), n_query, k))); // distances gpu fp16
    std::cout << "Final recall GpuIndexCagra FP16 vs GpuIndexCagra FP32 "
              << *recall_score_fp16_vs_fp32.data_handle() << std::endl;

    return 0;
}
