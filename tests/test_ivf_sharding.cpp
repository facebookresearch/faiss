/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <faiss/IVFlib.h>
#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexBinaryIVF.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>

#include "test_util.h"

// Leak regression for handle_ivf / handle_binary_ivf (faiss/IVFlib.cpp), only
// visible under AddressSanitizer (@mode/dev-asan): the per-call clone, and with
// generate_ids=true the quantizer wrapped in IndexIDMap2. Each test runs both
// generate_ids paths.

namespace {

pthread_mutex_t temp_file_mutex = PTHREAD_MUTEX_INITIALIZER;

// Builds a shard filename template ("<unique-prefix>.%d.index") from a fresh
// temp path so parallel test shards don't collide.
std::string shardTemplate(Tempfilename& prefix) {
    return std::string(prefix.c_str()) + ".%d.index";
}

void removeShardFiles(const std::string& tmpl, int64_t shardCount) {
    char fname[256];
    for (int64_t i = 0; i < shardCount; i++) {
        snprintf(fname, sizeof(fname), tmpl.c_str(), i);
        std::remove(fname);
    }
}

TEST(TestIvfSharding, ShardIvfCentroidsNoLeak) {
    constexpr int d = 16;
    constexpr int nlist = 32;
    constexpr int64_t shardCount = 3;

    std::vector<float> centroids(nlist * d);
    for (size_t i = 0; i < centroids.size(); i++) {
        centroids[i] = static_cast<float>(i % 100) * 0.01f;
    }

    for (bool generate_ids : {false, true}) {
        faiss::IndexFlatL2 quantizer(d);
        faiss::IndexIVFFlat index(&quantizer, d, nlist);
        index.quantizer->add(nlist, centroids.data());

        Tempfilename prefix(&temp_file_mutex, "/tmp/faiss_ivf_shard_XXXXXX");
        const std::string tmpl = shardTemplate(prefix);

        faiss::ivflib::shard_ivf_index_centroids(
                &index,
                shardCount,
                tmpl,
                /*sharding_function=*/nullptr,
                generate_ids);

        char fname[256];
        for (int64_t i = 0; i < shardCount; i++) {
            snprintf(fname, sizeof(fname), tmpl.c_str(), i);
            faiss::Index* shard = faiss::read_index(fname);
            delete shard;
        }
        removeShardFiles(tmpl, shardCount);
    }
}

TEST(TestIvfSharding, ShardBinaryIvfCentroidsNoLeak) {
    constexpr int d = 64; // bits; d / 8 bytes per centroid
    constexpr int nlist = 32;
    constexpr int64_t shardCount = 3;

    std::vector<uint8_t> centroids(nlist * (d / 8));
    for (size_t i = 0; i < centroids.size(); i++) {
        centroids[i] = static_cast<uint8_t>(i);
    }

    for (bool generate_ids : {false, true}) {
        faiss::IndexBinaryFlat quantizer(d);
        faiss::IndexBinaryIVF index(&quantizer, d, nlist);
        index.quantizer->add(nlist, centroids.data());

        Tempfilename prefix(&temp_file_mutex, "/tmp/faiss_binivf_shard_XXXXXX");
        const std::string tmpl = shardTemplate(prefix);

        faiss::ivflib::shard_binary_ivf_index_centroids(
                &index,
                shardCount,
                tmpl,
                /*sharding_function=*/nullptr,
                generate_ids);

        char fname[256];
        for (int64_t i = 0; i < shardCount; i++) {
            snprintf(fname, sizeof(fname), tmpl.c_str(), i);
            faiss::IndexBinary* shard = faiss::read_index_binary(fname);
            delete shard;
        }
        removeShardFiles(tmpl, shardCount);
    }
}

} // namespace
