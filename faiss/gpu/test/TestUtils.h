/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/invlists/InvertedLists.h>
#include <gtest/gtest.h>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

namespace faiss {
namespace gpu {

/// Generates and displays a new seed for the test
void newTestSeed();

/// Uses an explicit seed for the test
void setTestSeed(long seed);

/// Returns the relative error in difference between a and b
/// (|a - b| / (0.5 * (|a| + |b|))
float relativeError(float a, float b);

/// Generates a random integer in the range [a, b]
int randVal(int a, int b);

/// Generates a random bool
bool randBool();

/// Select a random value from the given list of values provided as an
/// initializer_list
template <typename T>
T randSelect(std::initializer_list<T> vals) {
    FAISS_ASSERT(vals.size() > 0);
    int sel = randVal(0, vals.size());

    int i = 0;
    for (auto v : vals) {
        if (i++ == sel) {
            return v;
        }
    }

    // should not get here
    return *vals.begin();
}

/// Generates a collection of random vectors in the range [0, 1]
std::vector<float> randVecs(size_t num, size_t dim);

/// Generates a collection of random bit vectors
std::vector<unsigned char> randBinaryVecs(size_t num, size_t dim);

// returns to_fp32(to_fp16(v)); useful in comparing fp16 results on CPU
std::vector<float> roundToHalf(const std::vector<float>& v);

/// Compare two indices via query for similarity, with a user-specified set of
/// query vectors
void compareIndices(
        const std::vector<float>& queryVecs,
        faiss::Index& refIndex,
        faiss::Index& testIndex,
        int numQuery,
        int dim,
        int k,
        const std::string& configMsg,
        float maxRelativeError = 6e-5f,
        float pctMaxDiff1 = 0.1f,
        float pctMaxDiffN = 0.005f);

/// Compare two indices via query for similarity, generating random query
/// vectors
void compareIndices(
        faiss::Index& refIndex,
        faiss::Index& testIndex,
        int numQuery,
        int dim,
        int k,
        const std::string& configMsg,
        float maxRelativeError = 6e-5f,
        float pctMaxDiff1 = 0.1f,
        float pctMaxDiffN = 0.005f);

/// Display specific differences in the two (distance, index) lists
void compareLists(
        const float* refDist,
        const faiss::idx_t* refInd,
        const float* testDist,
        const faiss::idx_t* testInd,
        int dim1,
        int dim2,
        const std::string& configMsg,
        bool printBasicStats,
        bool printDiffs,
        bool assertOnErr,
        float maxRelativeError = 6e-5f,
        float pctMaxDiff1 = 0.1f,
        float pctMaxDiffN = 0.005f);

/// Compare IVF lists between a CPU and GPU index
template <typename A, typename B>
void testIVFEquality(A& cpuIndex, B& gpuIndex) {
    // Ensure equality of the inverted lists
    EXPECT_EQ(cpuIndex.nlist, gpuIndex.nlist);

    for (int i = 0; i < cpuIndex.nlist; ++i) {
        auto cpuLists = cpuIndex.invlists;

        // Code equality
        EXPECT_EQ(cpuLists->list_size(i), gpuIndex.getListLength(i));
        std::vector<uint8_t> cpuCodes(
                cpuLists->list_size(i) * cpuLists->code_size);

        auto sc = faiss::InvertedLists::ScopedCodes(cpuLists, i);
        std::memcpy(
                cpuCodes.data(),
                sc.get(),
                cpuLists->list_size(i) * cpuLists->code_size);

        auto gpuCodes = gpuIndex.getListVectorData(i, false);
        EXPECT_EQ(cpuCodes, gpuCodes);

        // Index equality
        std::vector<idx_t> cpuIndices(cpuLists->list_size(i));

        auto si = faiss::InvertedLists::ScopedIds(cpuLists, i);
        std::memcpy(
                cpuIndices.data(),
                si.get(),
                cpuLists->list_size(i) * sizeof(faiss::idx_t));
        EXPECT_EQ(cpuIndices, gpuIndex.getListIndices(i));
    }
}

} // namespace gpu
} // namespace faiss
