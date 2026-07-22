/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndexHNSW.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/test/TestUtils.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/impl/FaissException.h>
#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

namespace {

// Recall@k of `test` (approximate) against `groundTruth` (exact top-k ids).
float recallAtK(
        const std::vector<faiss::idx_t>& groundTruth,
        const std::vector<faiss::idx_t>& test,
        int numQuery,
        int k) {
    size_t hits = 0;
    for (int q = 0; q < numQuery; q++) {
        for (int i = 0; i < k; i++) {
            faiss::idx_t want = groundTruth[q * k + i];
            for (int j = 0; j < k; j++) {
                if (test[q * k + j] == want) {
                    hits++;
                    break;
                }
            }
        }
    }
    return static_cast<float>(hits) / static_cast<float>(numQuery * k);
}

void normalizeRows(std::vector<float>& v, int n, int dim) {
    for (int i = 0; i < n; i++) {
        float* row = v.data() + static_cast<size_t>(i) * dim;
        float norm = 0.0f;
        for (int d = 0; d < dim; d++)
            norm += row[d] * row[d];
        if (norm > 0.0f) {
            float inv = 1.0f / std::sqrt(norm);
            for (int d = 0; d < dim; d++)
                row[d] *= inv;
        }
    }
}

// Map uniform [0,1) components to integer values in [-100, 100]. QT_8bit_-
// direct_signed is a fixed code=(x+128) map with no trained range, so it only
// represents integer-valued data in [-128,127]; feeding it small floats (as
// randVecs produces) collapses every code to ~0. INT8 vectors in production are
// genuine int8 values, so this is the representative input for the codec.
void toInt8Range(std::vector<float>& v) {
    for (float& x : v) {
        x = std::round((x - 0.5f) * 200.0f);
    }
}

// Build a CPU IndexHNSW, clone it to GPU via index_cpu_to_gpu, and assert the
// GPU search recall against exact (IndexFlat) ground truth is high.
void testHnswRecall(
        faiss::IndexHNSW& cpuIndex,
        faiss::MetricType metric,
        int numVecs,
        int dim,
        bool normalize,
        float minRecall,
        bool int8Range = false) {
    int numQuery = 64;
    int k = 10;

    std::vector<float> vecs = faiss::gpu::randVecs(numVecs, dim);
    std::vector<float> queries = faiss::gpu::randVecs(numQuery, dim);
    if (int8Range) {
        toInt8Range(vecs);
        toInt8Range(queries);
    }
    if (normalize) {
        normalizeRows(vecs, numVecs, dim);
        normalizeRows(queries, numQuery, dim);
    }

    cpuIndex.hnsw.efConstruction = 80;
    cpuIndex.add(numVecs, vecs.data());

    // Exact ground truth from a brute-force flat index with the same metric.
    faiss::IndexFlat flat(dim, metric);
    flat.add(numVecs, vecs.data());
    std::vector<float> gtDist(numQuery * k);
    std::vector<faiss::idx_t> gtInd(numQuery * k);
    flat.search(numQuery, queries.data(), k, gtDist.data(), gtInd.data());

    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuClonerOptions options;
    std::unique_ptr<faiss::Index> gpuIndex(
            faiss::gpu::index_cpu_to_gpu(&res, 0, &cpuIndex, &options));

    // The cloner must produce a GpuIndexHNSW.
    auto* gpuHnsw = dynamic_cast<faiss::gpu::GpuIndexHNSW*>(gpuIndex.get());
    ASSERT_NE(gpuHnsw, nullptr);
    EXPECT_EQ(gpuHnsw->d, dim);
    EXPECT_EQ(gpuHnsw->ntotal, numVecs);
    EXPECT_EQ(
            gpuHnsw->useInnerProduct(),
            metric == faiss::METRIC_INNER_PRODUCT);

    faiss::gpu::SearchParametersGpuHNSW params;
    params.ef = 256;

    std::vector<float> gpuDist(numQuery * k);
    std::vector<faiss::idx_t> gpuInd(numQuery * k);
    gpuIndex->search(
            numQuery,
            queries.data(),
            k,
            gpuDist.data(),
            gpuInd.data(),
            &params);

    // All returned ids must be valid (no sentinels leaking through).
    for (int i = 0; i < numQuery * k; i++) {
        EXPECT_GE(gpuInd[i], 0);
        EXPECT_LT(gpuInd[i], numVecs);
    }

    // Inner-product / cosine similarities are larger-is-better; L2 distances
    // are smaller-is-better. Verify the top-1 ordering sign is respected.
    for (int q = 0; q < numQuery; q++) {
        for (int i = 1; i < k; i++) {
            if (metric == faiss::METRIC_INNER_PRODUCT) {
                EXPECT_GE(
                        gpuDist[q * k + i - 1] + 1e-4f, gpuDist[q * k + i]);
            } else {
                EXPECT_LE(
                        gpuDist[q * k + i - 1] - 1e-4f, gpuDist[q * k + i]);
            }
        }
    }

    float recall = recallAtK(gtInd, gpuInd, numQuery, k);
    EXPECT_GE(recall, minRecall) << "recall@" << k << " = " << recall;
}

} // namespace

TEST(TestGpuIndexHNSW, Flat_L2) {
    int dim = 64;
    faiss::IndexHNSWFlat cpuIndex(dim, 16, faiss::METRIC_L2);
    testHnswRecall(cpuIndex, faiss::METRIC_L2, 4000, dim, false, 0.90f);
}

TEST(TestGpuIndexHNSW, Flat_IP) {
    int dim = 64;
    faiss::IndexHNSWFlat cpuIndex(dim, 16, faiss::METRIC_INNER_PRODUCT);
    testHnswRecall(
            cpuIndex, faiss::METRIC_INNER_PRODUCT, 4000, dim, false, 0.90f);
}

// Cosine is IP over L2-normalized vectors (no separate metric).
TEST(TestGpuIndexHNSW, Flat_Cosine) {
    int dim = 64;
    faiss::IndexHNSWFlat cpuIndex(dim, 16, faiss::METRIC_INNER_PRODUCT);
    testHnswRecall(
            cpuIndex, faiss::METRIC_INNER_PRODUCT, 4000, dim, true, 0.90f);
}

// INT8 storage via QT_8bit_direct_signed (native DP4A path). dim % 4 == 0 so
// the DP4A kernel is exercised; quantization lowers the recall bar. The input
// is int8-range integer data (int8Range=true) so the direct-signed codec is
// exercised on representable values instead of collapsing sub-1.0 floats to ~0.
TEST(TestGpuIndexHNSW, SQ_Int8_L2) {
    int dim = 64;
    faiss::IndexHNSWSQ cpuIndex(
            dim,
            faiss::ScalarQuantizer::QT_8bit_direct_signed,
            16,
            faiss::METRIC_L2);
    testHnswRecall(
            cpuIndex,
            faiss::METRIC_L2,
            4000,
            dim,
            false,
            0.70f,
            /*int8Range=*/true);
}

TEST(TestGpuIndexHNSW, SQ_Fp16_L2) {
    int dim = 64;
    faiss::IndexHNSWSQ cpuIndex(
            dim, faiss::ScalarQuantizer::QT_fp16, 16, faiss::METRIC_L2);
    testHnswRecall(cpuIndex, faiss::METRIC_L2, 4000, dim, false, 0.88f);
}

// Cosine over scalar-quantized storage = normalize + IP with the SQ codec.
// fp16 and bf16 represent normalized components (magnitude ~1/sqrt(d))
// accurately, so recall stays high. int8 cosine has NO gate here on purpose:
// QT_8bit_direct_signed is a fixed code=x+128 map, so normalized vectors
// collapse onto a few of the 256 levels and int8-cosine recall is poor at the
// faiss level. The Knowhere consumer therefore re-encodes int8 cosine as fp16
// (see faiss_hnsw.cc ToVanillaHnsw), which is exactly SQ_Fp16_Cosine below.
TEST(TestGpuIndexHNSW, SQ_Fp16_Cosine) {
    int dim = 64;
    faiss::IndexHNSWSQ cpuIndex(
            dim,
            faiss::ScalarQuantizer::QT_fp16,
            16,
            faiss::METRIC_INNER_PRODUCT);
    testHnswRecall(
            cpuIndex, faiss::METRIC_INNER_PRODUCT, 4000, dim, true, 0.85f);
}

TEST(TestGpuIndexHNSW, SQ_Bf16_Cosine) {
    int dim = 64;
    faiss::IndexHNSWSQ cpuIndex(
            dim,
            faiss::ScalarQuantizer::QT_bf16,
            16,
            faiss::METRIC_INNER_PRODUCT);
    testHnswRecall(
            cpuIndex, faiss::METRIC_INNER_PRODUCT, 4000, dim, true, 0.80f);
}

// copyTo / index_gpu_to_cpu is intentionally unsupported (search-only index).
TEST(TestGpuIndexHNSW, CopyToThrows) {
    int dim = 32;
    int numVecs = 1000;
    std::vector<float> vecs = faiss::gpu::randVecs(numVecs, dim);

    faiss::IndexHNSWFlat cpuIndex(dim, 16, faiss::METRIC_L2);
    cpuIndex.add(numVecs, vecs.data());

    faiss::gpu::StandardGpuResources res;
    std::unique_ptr<faiss::Index> gpuIndex(
            faiss::gpu::index_cpu_to_gpu(&res, 0, &cpuIndex, nullptr));
    auto* gpuHnsw = dynamic_cast<faiss::gpu::GpuIndexHNSW*>(gpuIndex.get());
    ASSERT_NE(gpuHnsw, nullptr);

    faiss::IndexHNSWFlat dst;
    EXPECT_THROW(gpuHnsw->copyTo(&dst), faiss::FaissException);
    EXPECT_THROW(
            {
                faiss::Index* cpu =
                        faiss::gpu::index_gpu_to_cpu(gpuIndex.get());
                delete cpu;
            },
            faiss::FaissException);
}

// Unsupported metrics must be rejected at copy time, not silently mishandled.
TEST(TestGpuIndexHNSW, RejectsUnsupportedMetric) {
    int dim = 32;
    faiss::gpu::StandardGpuResources res;
    EXPECT_THROW(
            faiss::gpu::GpuIndexHNSW(&res, dim, faiss::METRIC_L1),
            faiss::FaissException);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);

    // just run with a fixed test seed
    faiss::gpu::setTestSeed(100);

    return RUN_ALL_TESTS();
}
