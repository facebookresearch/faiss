/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexBinaryHNSW.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/gpu/Dataset.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndexBinaryCagra.h>
#include <faiss/gpu/GpuIndexCagra.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissException.h>
#include <faiss/impl/io.h>
#include <faiss/index_io.h>
#include <faiss/gpu/utils/CopyUtils.cuh>

#include <algorithm>
#include <cstring>
#include <memory>
#include <optional>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
using faiss::idx_t;
using faiss::NumericType;
using faiss::gpu::Dataset;
using faiss::gpu::GpuIndexBinaryCagra;
using faiss::gpu::GpuIndexCagra;
using faiss::gpu::StandardGpuResources;

struct TrainDataset {
    template <typename Index, typename Data>
    auto operator()(Index& index, Data&& data) const
            -> decltype(index.train(std::forward<Data>(data)));
};

struct AddDataset {
    template <typename Index, typename Data>
    auto operator()(Index& index, Data&& data) const
            -> decltype(index.add(std::forward<Data>(data)));
};

struct UpdateDataset {
    template <typename Index, typename Data>
    auto operator()(Index& index, Data&& data) const
            -> decltype(index.updateDataset(std::forward<Data>(data)));
};

struct CopyDataset {
    template <typename Index, typename CpuIndex, typename Data>
    auto operator()(Index& index, const CpuIndex* source, Data&& data) const
            -> decltype(index.copyFrom(source, std::forward<Data>(data)));
};

struct CloneDataset {
    template <typename Data>
    auto operator()(Data&& data) const -> decltype(faiss::gpu::indexCpuToGpu(
            nullptr,
            0,
            nullptr,
            std::forward<Data>(data)));
};

struct CloneBinaryDataset {
    template <typename Data>
    auto operator()(Data&& data) const
            -> decltype(faiss::gpu::indexBinaryCpuToGpu(
                    nullptr,
                    0,
                    nullptr,
                    std::forward<Data>(data)));
};

template <typename Operation, typename... Args>
constexpr bool acceptsDatasetLvaluesOnly =
        std::is_invocable_v<Operation, Args..., Dataset&> &&
        std::is_invocable_v<Operation, Args..., const Dataset&> &&
        !std::is_invocable_v<Operation, Args..., Dataset&&> &&
        !std::is_invocable_v<Operation, Args..., const Dataset&&>;

static_assert(acceptsDatasetLvaluesOnly<TrainDataset, GpuIndexCagra&>);
static_assert(acceptsDatasetLvaluesOnly<AddDataset, GpuIndexCagra&>);
static_assert(acceptsDatasetLvaluesOnly<UpdateDataset, GpuIndexCagra&>);
static_assert(acceptsDatasetLvaluesOnly<
              CopyDataset,
              GpuIndexCagra&,
              const faiss::IndexHNSWCagra*>);
static_assert(acceptsDatasetLvaluesOnly<TrainDataset, GpuIndexBinaryCagra&>);
static_assert(acceptsDatasetLvaluesOnly<AddDataset, GpuIndexBinaryCagra&>);
static_assert(acceptsDatasetLvaluesOnly<UpdateDataset, GpuIndexBinaryCagra&>);
static_assert(acceptsDatasetLvaluesOnly<
              CopyDataset,
              GpuIndexBinaryCagra&,
              const faiss::IndexBinaryHNSWCagra*>);
static_assert(acceptsDatasetLvaluesOnly<CloneDataset>);
static_assert(acceptsDatasetLvaluesOnly<CloneBinaryDataset>);

constexpr idx_t kRows = 512;

faiss::gpu::GpuIndexCagraConfig config() {
    faiss::gpu::GpuIndexCagraConfig out;
    out.device = 0;
    out.graph_degree = 16;
    out.intermediate_graph_degree = 32;
    out.build_algo = faiss::gpu::graph_build_algo::NN_DESCENT;
    return out;
}

std::vector<float> floatData(int dim) {
    std::mt19937 random(1234);
    std::uniform_real_distribution<float> value(-1.0f, 1.0f);
    std::vector<float> data(kRows * dim);
    for (auto& x : data) {
        x = value(random);
    }
    return data;
}

std::vector<uint8_t> binaryData(int bytes) {
    std::mt19937 random(4321);
    std::vector<uint8_t> data(kRows * bytes);
    for (auto& x : data) {
        x = static_cast<uint8_t>(random());
    }
    return data;
}

void checkFloatSearch(faiss::Index& index, const float* queries) {
    std::vector<float> distances(8);
    std::vector<idx_t> labels(8);
    index.search(8, queries, 1, distances.data(), labels.data());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    int selfHits = 0;
    for (int i = 0; i < 8; ++i) {
        EXPECT_GE(labels[i], 0);
        EXPECT_LT(labels[i], kRows);
        selfHits += labels[i] == i;
    }
    EXPECT_GE(selfHits, 6);
}

void checkBinarySearch(faiss::IndexBinary& index, const uint8_t* queries) {
    std::vector<int> distances(8);
    std::vector<idx_t> labels(8);
    index.search(8, queries, 1, distances.data(), labels.data());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    int selfHits = 0;
    for (int i = 0; i < 8; ++i) {
        EXPECT_GE(labels[i], 0);
        EXPECT_LT(labels[i], kRows);
        selfHits += labels[i] == i;
    }
    EXPECT_GE(selfHits, 6);
}

TEST(CagraDataset, PointerConstructionRejectedThroughBaseInterfaces) {
    StandardGpuResources resources;
    auto floats = floatData(16);
    auto bytes = binaryData(16);
    auto floatView = Dataset::viewHost(
            floats.data(), kRows, 16, NumericType::Float32, 16);
    auto byteView =
            Dataset::viewHost(bytes.data(), kRows, 16, NumericType::UInt8, 16);
    GpuIndexCagra cagra(&resources, 16, faiss::METRIC_L2, config());
    GpuIndexBinaryCagra binary(&resources, 128, config());
    faiss::Index* base = &cagra;
    faiss::IndexBinary* binaryBase = &binary;
    EXPECT_THROW(base->train(kRows, floats.data()), faiss::FaissException);
    EXPECT_THROW(base->add(kRows, floats.data()), faiss::FaissException);
    EXPECT_THROW(
            base->train_ex(kRows, floats.data(), NumericType::Float32),
            faiss::FaissException);
    EXPECT_THROW(
            base->add_ex(kRows, floats.data(), NumericType::Float32),
            faiss::FaissException);
    EXPECT_THROW(binaryBase->train(kRows, bytes.data()), faiss::FaissException);
    EXPECT_THROW(binaryBase->add(kRows, bytes.data()), faiss::FaissException);
    EXPECT_THROW(
            binaryBase->train_ex(kRows, bytes.data(), NumericType::UInt8),
            faiss::FaissException);
    EXPECT_THROW(
            binaryBase->add_ex(kRows, bytes.data(), NumericType::UInt8),
            faiss::FaissException);
    EXPECT_FALSE(cagra.is_trained);
    EXPECT_FALSE(binary.is_trained);

    faiss::gpu::GpuIndexFlatL2 flat(&resources, 16);
    base = &flat;
    EXPECT_THROW(base->train(floatView), faiss::FaissException);
    EXPECT_THROW(base->add(floatView), faiss::FaissException);
    EXPECT_NO_THROW(base->add(kRows, floats.data()));
    EXPECT_EQ(flat.ntotal, kRows);

    faiss::IndexBinaryFlat binaryFlat(128);
    binaryBase = &binaryFlat;
    EXPECT_THROW(binaryBase->train(byteView), faiss::FaissException);
    EXPECT_THROW(binaryBase->add(byteView), faiss::FaissException);
    EXPECT_NO_THROW(binaryBase->add(kRows, bytes.data()));
    EXPECT_EQ(binaryFlat.ntotal, kRows);
}

TEST(CagraDataset, BruteForceRequiresMultipleDevices) {
    StandardGpuResources resources;
    auto values = floatData(20);
    auto source = Dataset::viewHost(
            values.data(), kRows, 20, NumericType::Float32, 20);
    auto options = config();
    options.build_algo = faiss::gpu::graph_build_algo::BRUTE_FORCE;
    GpuIndexCagra index(&resources, 20, faiss::METRIC_L2, options);
    try {
        index.train(source);
        FAIL() << "Single-device BRUTE_FORCE must be rejected";
    } catch (const faiss::FaissException& error) {
        EXPECT_NE(
                std::string(error.what()).find("BRUTE_FORCE"),
                std::string::npos);
    }
    EXPECT_FALSE(index.is_trained);
}

TEST(CagraDataset, CompatibleStandardDeviceBorrowsWithoutOwning) {
    // Twenty Float32 columns satisfy 16-byte rows without dim % 16 == 0.
    constexpr int dim = 20;
    StandardGpuResources resources;
    auto values = floatData(dim);
    auto source = Dataset::viewHost(
            values.data(), kRows, dim, NumericType::Float32, dim);
    auto storage = Dataset::makeDevicePadded(&resources, source, 0);
    GpuIndexCagra index(&resources, dim, faiss::METRIC_L2, config());
    {
        auto standard = Dataset::viewDevice(
                storage.getData(), kRows, dim, NumericType::Float32, dim, 0);
        EXPECT_EQ(standard.getLayout(), Dataset::Layout::Standard);
        EXPECT_EQ(standard.getData(), storage.getData());
        EXPECT_FALSE(standard.ownsData());
        EXPECT_TRUE(standard.isCagraCompatible());
        faiss::Index* base = &index;
        base->train(standard);
    }
    EXPECT_TRUE(index.isSearchReady());
    auto movedStorage = std::move(storage);
    checkFloatSearch(index, values.data());
    EXPECT_THROW(index.train(movedStorage), faiss::FaissException);
    index.reset();
    EXPECT_FALSE(index.isSearchReady());
    EXPECT_FALSE(index.is_trained);
    EXPECT_EQ(index.ntotal, 0);
    index.add(movedStorage);
    checkFloatSearch(index, values.data());
}

TEST(CagraDataset, GraphOnlyHostExportAndExplicitAttachment) {
    constexpr int dim = 19;
    constexpr int stride = 23;
    StandardGpuResources resources;
    auto values = floatData(dim);
    std::vector<float> strided(kRows * stride, 1234.0f);
    for (idx_t i = 0; i < kRows; ++i) {
        std::copy_n(values.data() + i * dim, dim, strided.data() + i * stride);
    }
    auto source = Dataset::viewHost(
            strided.data(), kRows, dim, NumericType::Float32, stride);
    auto storage = Dataset::makeDevicePadded(&resources, source, 0);
    GpuIndexCagra index(&resources, dim, faiss::METRIC_L2, config());
    faiss::Index* base = &index;
    base->train(source);
    EXPECT_TRUE(index.is_trained);
    EXPECT_FALSE(index.isSearchReady());
    float distance;
    idx_t label;
    EXPECT_THROW(
            index.search(1, values.data(), 1, &distance, &label),
            faiss::FaissException);

    faiss::IndexHNSWCagra cpu;
    cpu.base_level_only = true;
    index.copyTo(&cpu);
    std::vector<float> exported(values.size());
    cpu.reconstruct_n(0, kRows, exported.data());
    EXPECT_EQ(exported, values);
    EXPECT_EQ(strided[stride - 1], 1234.0f);
    faiss::RangeSearchResult range(8);
    cpu.range_search(8, values.data(), 1.0e9f, &range);
    for (int i = 0; i < 8; ++i) {
        EXPECT_GT(range.lims[i + 1], range.lims[i]);
    }

    index.updateDataset(storage);
    EXPECT_TRUE(index.isSearchReady());
    const auto graph = index.get_knngraph();
    auto wrongRows = Dataset::viewDevice(
            storage.getData(),
            kRows - 1,
            dim,
            NumericType::Float32,
            storage.getRowStride(),
            0);
    EXPECT_THROW(index.updateDataset(wrongRows), faiss::FaissException);
    auto wrongType = Dataset::viewDevice(
            storage.getData(),
            kRows,
            dim,
            NumericType::UInt8,
            storage.getRowStride() * sizeof(float),
            0);
    EXPECT_THROW(index.updateDataset(wrongType), faiss::FaissException);
    auto wrongDim = Dataset::viewDevice(
            storage.getData(),
            kRows,
            dim - 1,
            NumericType::Float32,
            storage.getRowStride(),
            0);
    EXPECT_THROW(index.updateDataset(wrongDim), faiss::FaissException);
    EXPECT_THROW(index.updateDataset(source), faiss::FaissException);
    if (faiss::gpu::getNumDevices() > 1) {
        auto wrongDevice = Dataset::makeDevicePadded(&resources, source, 1);
        EXPECT_THROW(index.updateDataset(wrongDevice), faiss::FaissException);
    }
    EXPECT_TRUE(index.isSearchReady());
    EXPECT_EQ(index.get_knngraph(), graph);
    checkFloatSearch(index, values.data());

    // CPU serialization and re-import retain their existing ownership behavior.
    faiss::VectorIOWriter writer;
    faiss::write_index(&cpu, &writer);
    faiss::VectorIOReader reader;
    reader.data = std::move(writer.data);
    std::unique_ptr<faiss::Index> restored(faiss::read_index(&reader));
    auto* restoredCagra = dynamic_cast<faiss::IndexHNSWCagra*>(restored.get());
    ASSERT_NE(restoredCagra, nullptr);
    auto importedStorage =
            Dataset::makeDevicePaddedFromIndex(&resources, *restoredCagra, 0);
    std::unique_ptr<faiss::Index> imported(
            faiss::gpu::indexCpuToGpu(
                    &resources, 0, restored.get(), importedStorage));
    EXPECT_THROW(
            faiss::gpu::index_cpu_to_gpu(&resources, 0, restored.get()),
            faiss::FaissException);
    restored.reset();
    index.reset();
    checkFloatSearch(*imported, values.data());
    checkFloatSearch(cpu, values.data());
}

template <bool Binary>
void checkAsyncDatasetLifetime() {
    constexpr int dim = 19;
    constexpr int nq = 64;
    using Value = std::conditional_t<Binary, uint8_t, float>;
    using Distance = std::conditional_t<Binary, int, float>;
    using Index =
            std::conditional_t<Binary, GpuIndexBinaryCagra, GpuIndexCagra>;
    std::vector<Value> values;
    if constexpr (Binary) {
        values = binaryData(dim);
    } else {
        values = floatData(dim);
    }
    faiss::gpu::DeviceScope scope(0);
    cudaStream_t alternateStream;
    ASSERT_EQ(
            cudaStreamCreateWithFlags(&alternateStream, cudaStreamNonBlocking),
            cudaSuccess);
    const auto destroyStream = [](cudaStream_t stream) {
        EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    };
    // cuVS descriptor caches retain their allocation stream in the resource
    // handle, so the external stream must also outlive those resources.
    std::unique_ptr<
            std::remove_pointer_t<cudaStream_t>,
            decltype(destroyStream)>
            streamOwner(alternateStream, destroyStream);
    StandardGpuResources resources;
    auto source = Dataset::viewHost(
            values.data(),
            kRows,
            dim,
            Binary ? NumericType::UInt8 : NumericType::Float32,
            dim);
    std::optional<Dataset> first(
            Dataset::makeDevicePadded(&resources, source, 0));
    std::optional<Dataset> second(
            Dataset::makeDevicePadded(&resources, source, 0));
    std::unique_ptr<Index> index;
    if constexpr (Binary) {
        index = std::make_unique<Index>(&resources, dim * 8, config());
    } else {
        index = std::make_unique<Index>(
                &resources, dim, faiss::METRIC_L2, config());
    }
    index->train(*first);

    auto gpuResources = resources.getResources();
    auto originalStream = gpuResources->getDefaultStream(0);
    auto queries = faiss::gpu::toDeviceTemporary(
            gpuResources.get(), values, originalStream);
    auto alloc = faiss::gpu::makeTempAlloc(
            faiss::gpu::AllocType::Other, originalStream);
    faiss::gpu::DeviceTensor<Distance, 2, true> distances(
            gpuResources.get(), alloc, {nq, 1});
    faiss::gpu::DeviceTensor<idx_t, 2, true> labels(
            gpuResources.get(), alloc, {nq, 1});
    resources.setDefaultStream(0, alternateStream);
    auto checkResults = [&](cudaStream_t stream) {
        std::vector<Distance> actualDistances(nq);
        std::vector<idx_t> actualLabels(nq);
        faiss::gpu::fromDevice(distances, actualDistances.data(), stream);
        faiss::gpu::fromDevice(labels, actualLabels.data(), stream);
        int selfHits = 0;
        for (int i = 0; i < nq; ++i) {
            EXPECT_GE(actualLabels[i], 0);
            EXPECT_LT(actualLabels[i], kRows);
            if (actualLabels[i] == i) {
                ++selfHits;
                EXPECT_NEAR(actualDistances[i], 0, 1e-5);
            }
        }
        EXPECT_GE(selfHits, nq * 3 / 4);
    };

    // These searches keep both inputs and outputs on device and queue work.
    // Rebind/reset must finish use of the old allocation before its release.
    index->search(nq, queries.data(), 1, distances.data(), labels.data());
    index->updateDataset(*second);
    first.reset();
    checkResults(alternateStream);
    index->search(nq, queries.data(), 1, distances.data(), labels.data());
    index->reset();
    second.reset();
    checkResults(alternateStream);

    // Build on B, then search on A and destroy without an explicit reset.
    // Keep the caller-owned data alive to isolate graph-storage destruction.
    std::optional<Dataset> third(
            Dataset::makeDevicePadded(&resources, source, 0));
    index->train(*third);
    resources.revertDefaultStream(0);
    index->search(nq, queries.data(), 1, distances.data(), labels.data());
    index.reset();
    checkResults(originalStream);
    third.reset();
}

TEST(CagraDataset, FloatAsyncRebindResetAndDestruction) {
    checkAsyncDatasetLifetime<false>();
}

TEST(CagraDataset, BinaryAsyncRebindResetAndDestruction) {
    checkAsyncDatasetLifetime<true>();
}

TEST(CagraDataset, MultiGpuDatasetBuildAndCpuExport) {
    int deviceCount = 0;
    ASSERT_EQ(cudaGetDeviceCount(&deviceCount), cudaSuccess);
    if (deviceCount < 2) {
        GTEST_SKIP() << "Requires at least two GPUs";
    }
    constexpr int dim = 32;
    constexpr idx_t rows = 4096;
    StandardGpuResources resources;
    std::mt19937 random(1234);
    std::uniform_real_distribution<float> value(-1.0f, 1.0f);
    std::vector<float> values(rows * dim);
    for (auto& x : values) {
        x = value(random);
    }
    auto source = Dataset::viewHost(
            values.data(), rows, dim, NumericType::Float32, dim);
    auto options = config();
    options.devices = {0, 1};
    // Exact cluster graphs bound the work while exercising the full
    // multi-GPU graph-build and CPU-export path.
    options.build_algo = faiss::gpu::graph_build_algo::BRUTE_FORCE;
    options.graph_degree = 32;
    options.intermediate_graph_degree = 64;
    options.all_neighbors_params.n_clusters = 4;
    options.all_neighbors_params.overlap_factor = 2;
    GpuIndexCagra index(&resources, dim, faiss::METRIC_L2, options);
    index.train(source);
    ASSERT_TRUE(index.is_trained);
    EXPECT_EQ(index.ntotal, rows);
    EXPECT_FALSE(index.isSearchReady());
    EXPECT_EQ(index.get_knngraph().size(), rows * options.graph_degree);
    EXPECT_THROW(index.updateDataset(source), faiss::FaissException);
    faiss::IndexHNSWCagra cpu;
    cpu.base_level_only = true;
    index.copyTo(&cpu);
    EXPECT_EQ(cpu.ntotal, rows);
    std::vector<float> exported(values.size());
    cpu.reconstruct_n(0, rows, exported.data());
    EXPECT_EQ(exported, values);
    index.reset();
    float distance;
    idx_t label;
    cpu.search(1, values.data(), 1, &distance, &label);
    EXPECT_EQ(label, 0);
    EXPECT_FLOAT_EQ(distance, 0.0f);
}

TEST(CagraDataset, BinaryGraphOnlyAttachmentAndCpuRoundTrip) {
    constexpr int bytes = 19;
    StandardGpuResources resources;
    auto values = binaryData(bytes);
    auto source = Dataset::viewHost(
            values.data(), kRows, bytes, NumericType::UInt8, bytes);
    auto storage = Dataset::makeDevicePadded(&resources, source, 0);
    GpuIndexBinaryCagra index(&resources, bytes * 8, config());
    faiss::IndexBinary* base = &index;
    base->add(source);
    EXPECT_FALSE(index.isSearchReady());
    int distance;
    idx_t label;
    EXPECT_THROW(
            index.search(1, values.data(), 1, &distance, &label),
            faiss::FaissException);
    faiss::IndexBinaryHNSWCagra cpu;
    cpu.base_level_only = true;
    index.copyTo(&cpu);
    std::vector<uint8_t> exported(values.size());
    for (idx_t i = 0; i < kRows; ++i) {
        cpu.reconstruct(i, exported.data() + i * bytes);
    }
    EXPECT_EQ(exported, values);
    index.updateDataset(storage);
    EXPECT_TRUE(index.isSearchReady());
    auto wrongRows = Dataset::viewDevice(
            storage.getData(),
            kRows - 1,
            bytes,
            NumericType::UInt8,
            storage.getRowStride(),
            0);
    EXPECT_THROW(index.updateDataset(wrongRows), faiss::FaissException);
    EXPECT_TRUE(index.isSearchReady());
    checkBinarySearch(index, values.data());

    faiss::VectorIOWriter writer;
    faiss::write_index_binary(&cpu, &writer);
    faiss::VectorIOReader reader;
    reader.data = std::move(writer.data);
    std::unique_ptr<faiss::IndexBinary> restored(
            faiss::read_index_binary(&reader));
    auto* restoredCagra =
            dynamic_cast<faiss::IndexBinaryHNSWCagra*>(restored.get());
    ASSERT_NE(restoredCagra, nullptr);
    auto importedStorage =
            Dataset::makeDevicePaddedFromIndex(&resources, *restoredCagra, 0);
    std::unique_ptr<faiss::IndexBinary> imported(
            faiss::gpu::indexBinaryCpuToGpu(
                    &resources, 0, restored.get(), importedStorage));
    EXPECT_THROW(
            faiss::gpu::index_binary_cpu_to_gpu(&resources, 0, restored.get()),
            faiss::FaissException);
    restored.reset();
    index.reset();
    EXPECT_FALSE(index.isSearchReady());
    index.train(storage);
    checkBinarySearch(index, values.data());
    checkBinarySearch(*imported, values.data());
}

TEST(CagraDataset, BinaryAlignedStandardDeviceSearch) {
    constexpr int bytes = 16;
    StandardGpuResources resources;
    auto values = binaryData(bytes);
    auto source = Dataset::viewHost(
            values.data(), kRows, bytes, NumericType::UInt8, bytes);
    auto storage = Dataset::makeDevicePadded(&resources, source, 0);
    auto standard = Dataset::viewDevice(
            storage.getData(), kRows, bytes, NumericType::UInt8, bytes, 0);
    GpuIndexBinaryCagra index(&resources, bytes * 8, config());
    index.train(standard);
    EXPECT_TRUE(index.isSearchReady());
    EXPECT_FALSE(standard.ownsData());
    checkBinarySearch(index, values.data());
}

template <typename T>
void checkTypedExport(NumericType type) {
    constexpr int dim = 19;
    StandardGpuResources resources;
    std::mt19937 random(9012);
    std::vector<T> values(kRows * dim);
    for (auto& x : values) {
        x = static_cast<T>(static_cast<int>(random() % 201) - 100);
    }
    auto original = values;
    auto source = Dataset::viewHost(values.data(), kRows, dim, type, dim);
    auto storage = Dataset::makeDevicePadded(&resources, source, 0);
    GpuIndexCagra index(&resources, dim, faiss::METRIC_L2, config());
    index.train(storage);
    faiss::IndexHNSWCagra cpu;
    cpu.base_level_only = true;
    index.copyTo(&cpu);
    EXPECT_EQ(cpu.get_numeric_type(), type);
    std::vector<T> afterExport(values.size());
    storage.copyToHost(afterExport.data());
    EXPECT_EQ(
            std::memcmp(
                    afterExport.data(),
                    original.data(),
                    original.size() * sizeof(T)),
            0);
    EXPECT_EQ(
            std::memcmp(
                    values.data(),
                    original.data(),
                    original.size() * sizeof(T)),
            0);

    auto importedStorage =
            Dataset::makeDevicePaddedFromIndex(&resources, cpu, 0);
    EXPECT_EQ(importedStorage.getNumericType(), type);
    std::vector<T> decoded(values.size());
    importedStorage.copyToHost(decoded.data());
    EXPECT_EQ(
            std::memcmp(
                    decoded.data(),
                    original.data(),
                    original.size() * sizeof(T)),
            0);
    GpuIndexCagra imported(&resources, dim, faiss::METRIC_L2, config());
    imported.copyFrom(&cpu, importedStorage);
    EXPECT_TRUE(imported.isSearchReady());
    EXPECT_EQ(imported.get_knngraph(), index.get_knngraph());
    // Exercise the unchanged typed query interface for both the original and
    // re-imported graph, including Int8 whose Python build path is unavailable.
    for (auto* searched : {&index, &imported}) {
        std::vector<float> distances(8);
        std::vector<idx_t> labels(8);
        searched->search_ex(
                8, original.data(), type, 1, distances.data(), labels.data());
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        int selfHits = 0;
        for (int i = 0; i < 8; ++i) {
            EXPECT_GE(labels[i], 0);
            EXPECT_LT(labels[i], kRows);
            if (labels[i] == i) {
                ++selfHits;
                EXPECT_NEAR(distances[i], 0.0f, 1.0f);
            }
        }
        EXPECT_GE(selfHits, 6);
    }
}

TEST(CagraDataset, HalfAndInt8ExportPreserveCallerData) {
    checkTypedExport<half>(NumericType::Float16);
    checkTypedExport<int8_t>(NumericType::Int8);
}
} // namespace

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
